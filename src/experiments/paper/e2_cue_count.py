"""Controlled crossed cue-count series for E2."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.paper import PROTOCOL_VERSION
from experiments.paper.metrics import cosine_matrix
from experiments.paper.runner import (
    CONDITIONS,
    _array_digest,
    _balanced_ca3_weights,
    _git_metadata,
    _json_dump,
    _sha256,
    _simulate,
    _train_autoencoder,
    _unique_sparse_patterns,
)
from experiments.paper.seeds import (
    DEVELOPMENT_SEEDS,
    FINAL_SEEDS,
    SCHEMA_SEED,
    STREAM_IDS,
    SeedStreams,
    random_derangement,
    random_nonidentity_permutation,
)


METRICS = (
    "raw_cosine",
    "topk_overlap",
    "identity_correct",
    "chance_corrected_cosine",
    "mse",
)


def _factor_patterns(
    config: dict[str, Any], streams: SeedStreams
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    factor = config["factor_code"]
    positions, _ = _unique_sparse_patterns(
        factor["positions"],
        factor["position_dimension"],
        factor["position_active"],
        streams.numpy("position_codes"),
    )
    cues, _ = _unique_sparse_patterns(
        factor["maximum_cues"],
        factor["cue_dimension"],
        factor["cue_active"],
        streams.numpy("cue_identity_codes"),
    )
    rows = []
    cue_labels = []
    position_labels = []
    for cue_index in range(factor["maximum_cues"]):
        for position_index in range(factor["positions"]):
            rows.append(np.concatenate([positions[position_index], cues[cue_index]]))
            cue_labels.append(cue_index)
            position_labels.append(position_index)
    return (
        positions.astype(np.float32),
        cues.astype(np.float32),
        np.stack(rows).astype(np.float32),
        np.asarray(cue_labels, dtype=np.int64),
        np.asarray(position_labels, dtype=np.int64),
    )


def _paper_config(config: dict[str, Any], memory_count: int) -> dict[str, Any]:
    return {
        "data": {
            "dimension": config["factor_code"]["position_dimension"]
            + config["factor_code"]["cue_dimension"],
            "active": config["autoencoder"]["k"],
            "memory_count": memory_count,
        },
        "autoencoder": config["autoencoder"],
        "memory": config["memory"],
        "tolerances": config["tolerances"],
    }


def _count_rng(seed: int, stream_name: str, cue_count: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence(
            [SCHEMA_SEED, int(seed), STREAM_IDS[stream_name], int(cue_count)]
        )
    )


def _restart_seed(seed: int, restart: int, role: int) -> int:
    state = np.random.SeedSequence(
        [SCHEMA_SEED, int(seed), STREAM_IDS["cue_ae_restart"], int(restart), role]
    ).generate_state(2, dtype=np.uint32)
    return ((int(state[0]) << 32) | int(state[1])) % (2**63 - 1)


def _train_cue_autoencoder(
    config: dict[str, Any],
    streams: SeedStreams,
    inputs: np.ndarray,
) -> tuple[torch.nn.Module, dict[str, Any], np.ndarray]:
    candidates = []
    trained = []
    for restart in range(config["autoencoder"]["deterministic_restarts"]):
        model, quality, reconstructed, _ = _train_autoencoder(
            _paper_config(config, len(inputs)),
            streams,
            inputs,
            inputs,
            seed_overrides={
                "ae_initialization": _restart_seed(streams.root_seed, restart, 1),
                "ae_minibatch_order": _restart_seed(streams.root_seed, restart, 2),
            },
        )
        candidates.append(
            {
                "restart": restart,
                "mse": quality["mse"],
                "mean_cosine": quality["mean_cosine"],
                "pass": quality["pass"],
            }
        )
        trained.append((model, reconstructed))
    selected = min(range(len(candidates)), key=lambda index: candidates[index]["mse"])
    quality = {
        **candidates[selected],
        "selected_restart": selected,
        "candidates": candidates,
    }
    model, reconstructed = trained[selected]
    return model, quality, reconstructed


def _run_seed(
    config: dict[str, Any], seed: int
) -> tuple[dict[str, np.ndarray], dict[str, Any], torch.nn.Module]:
    streams = SeedStreams(seed)
    positions, cues, full_inputs, full_cue_labels, full_position_labels = _factor_patterns(
        config, streams
    )
    model, quality, reconstructed = _train_cue_autoencoder(
        config, streams, full_inputs
    )
    if not quality["pass"]:
        raise RuntimeError(f"Cue-task AE gate failed for seed {seed}: {quality}")
    dimension = full_inputs.shape[1]
    ca3_weights = _balanced_ca3_weights(
        dimension,
        config["memory"]["ca3_inputs_per_unit"],
        streams.numpy("ca3_wiring"),
    )
    permutation = random_nonidentity_permutation(
        config["autoencoder"]["latent_dimension"],
        streams.numpy("coordinate_permutation"),
    )
    full_order = streams.numpy("cue_storage_order").permutation(len(full_inputs))
    counts = config["cue_counts"]
    maximum_items = config["factor_code"]["maximum_cues"] * config["factor_code"]["positions"]
    scalar_shape = (len(counts), len(CONDITIONS), maximum_items)
    vector_shape = (*scalar_shape, dimension)
    outputs = np.full(vector_shape, np.nan, dtype=np.float32)
    recalled_ca1 = np.full(vector_shape, np.nan, dtype=np.float32)
    inputs = np.full((len(counts), maximum_items, dimension), np.nan, dtype=np.float32)
    cue_labels = np.full((len(counts), maximum_items), -1, dtype=np.int64)
    position_labels = np.full((len(counts), maximum_items), -1, dtype=np.int64)
    metrics = {
        name: np.full(scalar_shape, np.nan, dtype=np.float64) for name in METRICS
    }
    cue_correct = np.full(scalar_shape, np.nan, dtype=np.float64)
    position_correct = np.full(scalar_shape, np.nan, dtype=np.float64)
    final_weights = np.zeros(
        (len(counts), len(CONDITIONS), dimension, dimension), dtype=np.float32
    )
    diagnostics: dict[str, Any] = {}

    for count_index, cue_count in enumerate(counts):
        selected = np.flatnonzero(full_cue_labels < cue_count)
        selected_set = set(selected.tolist())
        ordered_indices = np.asarray(
            [index for index in full_order if int(index) in selected_set], dtype=np.int64
        )
        task_inputs = full_inputs[ordered_indices]
        task_cues = full_cue_labels[ordered_indices]
        task_positions = full_position_labels[ordered_indices]
        item_count = len(task_inputs)
        derangement = random_derangement(
            item_count, _count_rng(seed, "content_derangement", cue_count)
        )
        run_config = _paper_config(config, item_count)
        result, check = _simulate(
            run_config,
            model,
            task_inputs,
            ca3_weights,
            np.arange(item_count, dtype=np.int64),
            permutation,
            derangement,
        )
        if not check["all_checks_pass"]:
            raise RuntimeError(
                f"Cue-count invariant failure seed={seed}, cues={cue_count}: {check}"
            )
        inputs[count_index, :item_count] = task_inputs
        cue_labels[count_index, :item_count] = task_cues
        position_labels[count_index, :item_count] = task_positions
        outputs[count_index, :, :item_count] = result["outputs"]
        recalled_ca1[count_index, :, :item_count] = result["recalled_ca1"]
        final_weights[count_index] = result["final_weights"]
        for metric in METRICS:
            metrics[metric][count_index, :, :item_count] = result[f"metric_{metric}"]
        for condition_index in range(len(CONDITIONS)):
            condition_output = result["outputs"][condition_index]
            cue_prediction = np.argmax(
                cosine_matrix(
                    condition_output[:, config["factor_code"]["position_dimension"] :],
                    cues[:cue_count],
                ),
                axis=1,
            )
            position_prediction = np.argmax(
                cosine_matrix(
                    condition_output[:, : config["factor_code"]["position_dimension"]],
                    positions,
                ),
                axis=1,
            )
            cue_correct[count_index, condition_index, :item_count] = (
                cue_prediction == task_cues
            )
            position_correct[count_index, condition_index, :item_count] = (
                position_prediction == task_positions
            )
        diagnostics[str(cue_count)] = check

    arrays = {
        "position_codes": positions,
        "cue_codes": cues,
        "full_task_inputs": full_inputs,
        "full_task_reconstruction": reconstructed,
        "inputs": inputs,
        "cue_labels": cue_labels,
        "position_labels": position_labels,
        "outputs": outputs,
        "recalled_ca1": recalled_ca1,
        "final_weights": final_weights,
        "cue_identity_correct": cue_correct,
        "position_correct": position_correct,
        **{f"metric_{name}": value for name, value in metrics.items()},
    }
    return arrays, {"autoencoder_quality": quality, "diagnostics": diagnostics}, model


def _write_source(path: Path, arrays: dict[str, np.ndarray]) -> None:
    counts = arrays["cue_counts"]
    conditions = arrays["condition_names"].tolist()
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "root_seed", "cue_count", "condition", "item_index", "cue_identity",
            "position", *METRICS, "cue_identity_correct", "position_correct",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for seed_index, seed in enumerate(arrays["root_seeds"]):
            for count_index, count in enumerate(counts):
                item_count = int(count) * 2
                for condition_index, condition in enumerate(conditions):
                    for item_index in range(item_count):
                        row: dict[str, Any] = {
                            "root_seed": int(seed),
                            "cue_count": int(count),
                            "condition": condition,
                            "item_index": item_index,
                            "cue_identity": int(arrays["cue_labels"][seed_index, count_index, item_index]),
                            "position": int(arrays["position_labels"][seed_index, count_index, item_index]),
                            "cue_identity_correct": float(arrays["cue_identity_correct"][seed_index, count_index, condition_index, item_index]),
                            "position_correct": float(arrays["position_correct"][seed_index, count_index, condition_index, item_index]),
                        }
                        for metric in METRICS:
                            row[metric] = float(arrays[f"metric_{metric}"][seed_index, count_index, condition_index, item_index])
                        writer.writerow(row)


def run(config_path: Path, output_dir: Path, split: str) -> dict[str, Any]:
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = DEVELOPMENT_SEEDS if split == "development" else FINAL_SEEDS
    seeds = tuple(config[f"{split}_seeds"])
    if seeds != expected:
        raise ValueError(f"Cue-count {split} seeds do not match frozen set")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = output_dir / "autoencoders"
    checkpoints.mkdir()
    seed_results = []
    seed_reports = {}
    checkpoint_hashes = {}
    for position, seed in enumerate(seeds, start=1):
        print(f"cue-count {split} seed {seed} ({position}/{len(seeds)})", flush=True)
        result, seed_report, model = _run_seed(config, seed)
        checkpoint = checkpoints / f"seed_{seed}.pt"
        torch.save(model.state_dict(), checkpoint)
        checkpoint_hashes[str(seed)] = _sha256(checkpoint)
        seed_results.append(result)
        seed_reports[str(seed)] = seed_report
    arrays: dict[str, np.ndarray] = {
        "root_seeds": np.asarray(seeds, dtype=np.int64),
        "cue_counts": np.asarray(config["cue_counts"], dtype=np.int64),
        "condition_names": np.asarray(CONDITIONS),
    }
    for key in seed_results[0]:
        arrays[key] = np.stack([item[key] for item in seed_results])
    summaries = {}
    for count_index, cue_count in enumerate(config["cue_counts"]):
        item_count = cue_count * 2
        summaries[str(cue_count)] = {}
        for condition_index, condition in enumerate(CONDITIONS):
            summaries[str(cue_count)][condition] = {
                "raw_cosine": float(
                    arrays["metric_raw_cosine"][:, count_index, condition_index, :item_count].mean()
                ),
                "cue_identity_accuracy": float(
                    arrays["cue_identity_correct"][:, count_index, condition_index, :item_count].mean()
                ),
                "position_accuracy": float(
                    arrays["position_correct"][:, count_index, condition_index, :item_count].mean()
                ),
            }
    scientific_digest = _array_digest(arrays)
    arrays_path = output_dir / "arrays.npz"
    config_out = output_dir / "config.json"
    source_path = output_dir / "source_data.csv"
    report_path = output_dir / "report.json"
    manifest_path = output_dir / "manifest.json"
    np.savez_compressed(arrays_path, **arrays)
    _json_dump(config_out, {**config, "active_split": split})
    _write_source(source_path, arrays)
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": "e2_cue_count",
        "split": split,
        "scientific_digest": scientific_digest,
        "seed_reports": seed_reports,
        "summaries": summaries,
    }
    _json_dump(report_path, report)
    repo_root = Path(__file__).resolve().parents[3]
    source_files = sorted(Path(__file__).resolve().parent.glob("*.py"))
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": PROTOCOL_VERSION,
        "experiment": "e2_cue_count",
        "split": split,
        "root_seeds": list(seeds),
        "git": _git_metadata(repo_root),
        "source_files": {
            str(path.relative_to(repo_root)): _sha256(path) for path in source_files
        },
        "config_source": {"path": str(config_path), "sha256": _sha256(config_path)},
        "checkpoint_hashes": checkpoint_hashes,
        "artifacts": {
            path.name: _sha256(path)
            for path in (arrays_path, config_out, source_path, report_path)
        },
        "scientific_digest": scientific_digest,
    }
    _json_dump(manifest_path, manifest)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("development", "final"))
    args = parser.parse_args()
    report = run(args.config, args.output, args.split)
    print(json.dumps(report["summaries"], indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
