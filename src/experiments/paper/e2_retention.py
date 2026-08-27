"""Sequential storage, memory-age, and load experiment (E2)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kam-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "kam-cache"))

import numpy as np
import torch

from core.functions import sparsemoid
from core.models import Autoencoder
from experiments.paper import PROTOCOL_VERSION
from experiments.paper.final_e1 import _interval
from experiments.paper.metrics import (
    chance_corrected_cosine,
    identity_correct,
    row_cosine,
    topk_overlap,
)
from experiments.paper.runner import (
    CONDITIONS,
    _array_digest,
    _balanced_ca3_weights,
    _ca3_code,
    _git_metadata,
    _json_dump,
    _recall,
    _sha256,
    _tensor_sha256,
    _unique_sparse_patterns,
    _update_weights,
)
from experiments.paper.seeds import (
    FINAL_SEEDS,
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


def evaluate_gate_b1(
    endpoint_condition_means: np.ndarray,
    condition_names: list[str],
    inference_cfg: dict[str, float],
) -> dict[str, Any]:
    indices = {name: condition_names.index(name) for name in condition_names}
    aligned = endpoint_condition_means[:, indices["aligned"]]
    fixed = endpoint_condition_means[:, indices["fixed_permutation"]]
    rescue = endpoint_condition_means[:, indices["matched_decoder_rescue"]]
    critical = inference_cfg["superiority_t_critical_df19"]
    contrasts = {
        "aligned_minus_fixed": _interval(aligned - fixed, critical),
        "rescue_minus_fixed": _interval(rescue - fixed, critical),
        "rescue_minus_aligned": _interval(
            rescue - aligned, inference_cfg["equivalence_t_critical_df19"]
        ),
    }
    equivalence = contrasts["rescue_minus_aligned"]["interval"]
    margin = inference_cfg["rescue_equivalence_margin"]
    checks = {
        "aligned_minus_fixed_ci_above_zero": bool(
            contrasts["aligned_minus_fixed"]["interval"][0] > 0.0
        ),
        "rescue_minus_fixed_ci_above_zero": bool(
            contrasts["rescue_minus_fixed"]["interval"][0] > 0.0
        ),
        "rescue_equivalent_to_aligned": bool(
            equivalence[0] >= -margin and equivalence[1] <= margin
        ),
        "complete_final_sample": len(endpoint_condition_means) == len(FINAL_SEEDS),
    }
    return {
        "contrasts": contrasts,
        "checks": checks,
        "gate_b1_pass": bool(all(checks.values())),
    }


def _load_autoencoder(
    checkpoint: Path, config: dict[str, Any]
) -> Autoencoder:
    data = config["data"]
    ae = config["autoencoder"]
    model = Autoencoder(
        dim_ei=data["dimension"],
        dim_ca1=ae["latent_dimension"],
        K_ca1=data["active"],
        K_eo=data["active"],
        beta_ei=ae["beta_latent"],
        beta_eo=ae["beta_output"],
        use_bias=ae["use_bias"],
    ).cpu()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _metrics_for_prefix(outputs: np.ndarray, targets: np.ndarray, k: int) -> dict[str, np.ndarray]:
    result = {
        "raw_cosine": row_cosine(outputs, targets),
        "topk_overlap": topk_overlap(outputs, targets, k),
        "identity_correct": identity_correct(outputs, targets),
        "mse": np.mean((outputs - targets) ** 2, axis=1),
    }
    result["chance_corrected_cosine"] = (
        chance_corrected_cosine(outputs, targets)
        if len(outputs) > 1
        else np.full(1, np.nan, dtype=np.float64)
    )
    return result


def _run_seed(
    config: dict[str, Any], seed: int, model: Autoencoder
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    data = config["data"]
    memory = config["memory"]
    streams = SeedStreams(seed)
    _, training_seen = _unique_sparse_patterns(
        data["training_size"], data["dimension"], data["active"],
        streams.numpy("ae_training_patterns"),
    )
    _, validation_seen = _unique_sparse_patterns(
        data["validation_size"], data["dimension"], data["active"],
        streams.numpy("ae_validation_patterns"), forbidden=training_seen,
    )
    bank, _ = _unique_sparse_patterns(
        data["maximum_load"], data["dimension"], data["active"],
        streams.numpy("memory_bank"), forbidden=validation_seen,
    )
    order = streams.numpy("storage_order").permutation(data["maximum_load"])
    inputs_np = bank[order]
    inputs = torch.tensor(inputs_np, dtype=torch.float32)
    ca3_np = _balanced_ca3_weights(
        data["dimension"], memory["ca3_inputs_per_unit"], streams.numpy("ca3_wiring")
    )
    ca3_weights = torch.tensor(ca3_np, dtype=torch.float32)
    permutation = random_nonidentity_permutation(
        config["autoencoder"]["latent_dimension"],
        streams.numpy("coordinate_permutation"),
    )
    derangement = random_derangement(
        data["maximum_load"], streams.numpy("content_derangement")
    )
    p = torch.tensor(permutation, dtype=torch.long)
    q = torch.tensor(derangement, dtype=torch.long)
    encoder = model.encoder[0].weight.detach().cpu()
    decoder = model.decoder[0].weight.detach().cpu()
    with torch.no_grad():
        codes = sparsemoid(
            inputs @ encoder.T,
            K=memory["k_ca1"],
            beta=config["autoencoder"]["beta_latent"],
        )
    instructions = {
        "aligned": codes,
        "fixed_permutation": codes[:, p],
        "matched_decoder_rescue": codes[:, p],
        "random_content_matched": codes[q],
        "no_plasticity": codes,
    }
    decoders = {name: decoder for name in CONDITIONS}
    decoders["matched_decoder_rescue"] = decoder[:, p]
    learned = {
        name: torch.zeros((data["dimension"], data["dimension"]), dtype=torch.float32)
        for name in ("aligned", "fixed_permutation", "random_content_matched")
    }
    loads = data["evaluation_loads"]
    maximum = data["maximum_load"]
    shape_scalar = (len(CONDITIONS), len(loads), maximum)
    shape_vector = (*shape_scalar, data["dimension"])
    outputs_all = np.full(shape_vector, np.nan, dtype=np.float32)
    ca1_all = np.full(shape_vector, np.nan, dtype=np.float32)
    metric_arrays = {
        name: np.full(shape_scalar, np.nan, dtype=np.float64) for name in METRICS
    }
    weight_snapshots = np.zeros(
        (len(CONDITIONS), len(loads), data["dimension"], data["dimension"]),
        dtype=np.float32,
    )
    recall_frozen = True
    load_to_index = {int(load): index for index, load in enumerate(loads)}

    for store_index in range(maximum):
        for condition in ("aligned", "fixed_permutation", "random_content_matched"):
            ca3 = _ca3_code(
                inputs[store_index], ca3_weights, memory["k_ca3"], memory["beta_ca3"]
            )
            learned[condition] = _update_weights(
                learned[condition], ca3, instructions[condition][store_index],
                memory["plasticity_rule"], memory["alpha"],
                memory["k_ca1"], memory["beta_ca1"],
            )
        current_load = store_index + 1
        if current_load not in load_to_index:
            continue
        checkpoint_index = load_to_index[current_load]
        condition_weights = {
            "aligned": learned["aligned"],
            "fixed_permutation": learned["fixed_permutation"],
            "matched_decoder_rescue": learned["fixed_permutation"],
            "random_content_matched": learned["random_content_matched"],
            "no_plasticity": torch.zeros_like(learned["aligned"]),
        }
        for condition_index, condition in enumerate(CONDITIONS):
            before = _tensor_sha256(condition_weights[condition])
            with torch.no_grad():
                output, recalled = _recall(
                    inputs[:current_load], condition_weights[condition], ca3_weights,
                    decoders[condition], memory,
                )
            after = _tensor_sha256(condition_weights[condition])
            recall_frozen = recall_frozen and before == after
            output_np = output.numpy()
            recalled_np = recalled.numpy()
            outputs_all[condition_index, checkpoint_index, :current_load] = output_np
            ca1_all[condition_index, checkpoint_index, :current_load] = recalled_np
            for metric_name, values in _metrics_for_prefix(
                output_np, inputs_np[:current_load], data["active"]
            ).items():
                metric_arrays[metric_name][
                    condition_index, checkpoint_index, :current_load
                ] = values
            weight_snapshots[condition_index, checkpoint_index] = condition_weights[
                condition
            ].numpy()

    fixed = CONDITIONS.index("fixed_permutation")
    rescue = CONDITIONS.index("matched_decoder_rescue")
    aligned = CONDITIONS.index("aligned")
    checks = {
        "recall_frozen": bool(recall_frozen),
        "fixed_rescue_weight_identity": bool(
            np.array_equal(weight_snapshots[fixed], weight_snapshots[rescue])
        ),
        "fixed_rescue_ca1_identity": bool(
            np.array_equal(ca1_all[fixed], ca1_all[rescue], equal_nan=True)
        ),
        "coordinate_equivariance": bool(
            np.max(
                np.abs(
                    weight_snapshots[fixed]
                    - weight_snapshots[aligned][:, permutation]
                )
            )
            <= config["tolerances"]["coordinate_equivariance"]
        ),
        "signal_statistics_match": bool(
            np.max(
                np.abs(
                    np.sort(instructions["aligned"].numpy(), axis=1)
                    - np.sort(instructions["fixed_permutation"].numpy(), axis=1)
                )
            )
            <= config["tolerances"]["signal_match"]
        ),
    }
    arrays = {
        "inputs": inputs_np.astype(np.float32),
        "original_memory_indices": order.astype(np.int64),
        "coordinate_permutation": permutation.astype(np.int64),
        "content_derangement": derangement.astype(np.int64),
        "instruction_codes": codes.numpy().astype(np.float32),
        "outputs": outputs_all,
        "recalled_ca1": ca1_all,
        "weight_snapshots": weight_snapshots,
        **{f"metric_{name}": value for name, value in metric_arrays.items()},
    }
    return arrays, {"checks": checks, "all_checks_pass": bool(all(checks.values()))}


def _write_source(path: Path, arrays: dict[str, np.ndarray]) -> None:
    conditions = arrays["condition_names"].tolist()
    loads = arrays["evaluation_loads"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["root_seed", "condition", "evaluation_load", "memory_age", *METRICS]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for seed_index, seed in enumerate(arrays["root_seeds"]):
            for condition_index, condition in enumerate(conditions):
                for load_index, load in enumerate(loads):
                    for memory_index in range(int(load)):
                        row = {
                            "root_seed": int(seed),
                            "condition": condition,
                            "evaluation_load": int(load),
                            "memory_age": int(load) - 1 - memory_index,
                        }
                        for metric in METRICS:
                            row[metric] = float(
                                arrays[f"metric_{metric}"][
                                    seed_index, condition_index, load_index, memory_index
                                ]
                            )
                        writer.writerow(row)


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seeds = tuple(int(seed) for seed in config["root_seeds"])
    if seeds != FINAL_SEEDS:
        raise ValueError("E2 must use the frozen final seed set")
    if config["conditions"] != list(CONDITIONS):
        raise ValueError("E2 conditions do not match the protocol")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[3]
    e1_root = (repo_root / config["e1_artifact"]).resolve()
    seed_results = []
    diagnostics = {}
    checkpoint_hashes = {}
    for position, seed in enumerate(seeds, start=1):
        print(f"E2 seed {seed} ({position}/{len(seeds)})", flush=True)
        checkpoint = e1_root / "seeds" / str(seed) / "autoencoder.pt"
        model = _load_autoencoder(checkpoint, config)
        checkpoint_hashes[str(seed)] = _sha256(checkpoint)
        arrays, seed_diagnostics = _run_seed(config, seed, model)
        if not seed_diagnostics["all_checks_pass"]:
            raise RuntimeError(f"E2 invariant failure for seed {seed}: {seed_diagnostics}")
        seed_results.append(arrays)
        diagnostics[str(seed)] = seed_diagnostics

    aggregate: dict[str, np.ndarray] = {
        "root_seeds": np.asarray(seeds, dtype=np.int64),
        "condition_names": np.asarray(CONDITIONS),
        "evaluation_loads": np.asarray(config["data"]["evaluation_loads"], dtype=np.int64),
    }
    for key in seed_results[0]:
        aggregate[key] = np.stack([item[key] for item in seed_results])
    endpoint_load_index = config["data"]["evaluation_loads"].index(config["endpoint"]["load"])
    oldest = int(config["endpoint"]["oldest_count"])
    endpoint = aggregate["metric_raw_cosine"][:, :, endpoint_load_index, :oldest].mean(axis=-1)
    gate_b1 = evaluate_gate_b1(
        endpoint, list(CONDITIONS), config["inference"]
    )
    final_load_index = len(config["data"]["evaluation_loads"]) - 1
    threshold = config["endpoint"]["capacity_cosine_threshold"]
    capacity = np.sum(
        (aggregate["metric_raw_cosine"][:, :, final_load_index] >= threshold)
        & (aggregate["metric_identity_correct"][:, :, final_load_index] == 1.0),
        axis=-1,
    )
    scientific_digest = _array_digest(aggregate)
    arrays_path = output_dir / "arrays.npz"
    config_out = output_dir / "config.json"
    source_path = output_dir / "source_data.csv"
    report_path = output_dir / "report.json"
    manifest_path = output_dir / "manifest.json"
    np.savez_compressed(arrays_path, **aggregate)
    _json_dump(config_out, config)
    _write_source(source_path, aggregate)
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": "e2_retention",
        "scientific_digest": scientific_digest,
        "all_invariants_pass": True,
        "diagnostics": diagnostics,
        "gate_b1": gate_b1,
        "endpoint_condition_summaries": {
            condition: _interval(
                endpoint[:, index], config["inference"]["superiority_t_critical_df19"]
            )
            for index, condition in enumerate(CONDITIONS)
        },
        "capacity_at_load_40": {
            condition: _interval(
                capacity[:, index], config["inference"]["superiority_t_critical_df19"]
            )
            for index, condition in enumerate(CONDITIONS)
        },
    }
    _json_dump(report_path, report)
    source_files = sorted(Path(__file__).resolve().parent.glob("*.py"))
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": PROTOCOL_VERSION,
        "experiment": "e2_retention",
        "root_seeds": list(seeds),
        "git": _git_metadata(repo_root),
        "source_files": {
            str(path.relative_to(repo_root)): _sha256(path) for path in source_files
        },
        "config_source": {"path": str(config_path), "sha256": _sha256(config_path)},
        "reused_e1_checkpoint_hashes": checkpoint_hashes,
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
    args = parser.parse_args()
    report = run(args.config, args.output)
    print(json.dumps(report["gate_b1"], indent=2, sort_keys=True, allow_nan=False))
    if not report["gate_b1"]["gate_b1_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
