"""Coordinate-mismatch experiment for the KAMemory paper.

This is the minimal confirmatory extension of the decoder-compatibility result.
It asks whether the cost of a *partial* mismatch depends on which CA1
coordinates are moved.  Each independent root seed trains a fresh autoencoder,
stores the same one-shot memories under paired instruction permutations, and
evaluates a fixed decoder versus its matched rescue decoder.

The default seeds are the untouched factorial set (61001--61012).  They are
not interchangeable with the E1 final seeds: do not replace them with a
post-hoc subset of the previous experiment.

Run from the repository root::

    PYTHONPATH=src python3 -m experiments.mtl_coord_experiment \
        --output results/paper/v1/e_coord_geometry \
        --figure article/figures/v1/figure4_e_coord_geometry

The experiment writes immutable arrays, source data, a resolved configuration,
and provenance.  ``--plot-only`` rebuilds a figure exclusively from saved
arrays and never imports the simulation model.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kam-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "kam-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from core.functions import sparsemoid
from experiments.paper import PROTOCOL_VERSION
from experiments.paper.metrics import evaluate_outputs, metric_sanity_checks, row_cosine
from experiments.paper.runner import (
    _array_digest,
    _balanced_ca3_weights,
    _ca3_code,
    _git_metadata,
    _json_dump,
    _recall,
    _sha256,
    _tensor_sha256,
    _train_autoencoder,
    _unique_sparse_patterns,
    _update_weights,
)
from experiments.paper.seeds import FACTORIAL_SEEDS, SCHEMA_SEED, STREAM_IDS, SeedStreams


EXPERIMENT = "e_coord_geometry"
LEVELS = (0.0, 0.25, 0.50, 0.75, 1.0)
STRATA = ("high_decoder_norm", "low_decoder_norm")
CONDITIONS = ("fixed_decoder", "matched_decoder_rescue")
METRICS = (
    "raw_cosine",
    "topk_overlap",
    "identity_correct",
    "chance_corrected_cosine",
    "mse",
)
T_CRITICAL_DF11 = 2.200985160091638


def default_config() -> dict[str, Any]:
    """Return the frozen design; all choices precede inspection of these seeds."""
    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "experiment": EXPERIMENT,
        "root_seeds": list(FACTORIAL_SEEDS),
        "device": "cpu",
        "dtype": "float32",
        "data": {
            "dimension": 50,
            "active": 5,
            "training_size": 2048,
            "validation_size": 256,
            "memory_count": 28,
        },
        "autoencoder": {
            "latent_dimension": 50,
            "beta_latent": 25.0,
            "beta_output": 25.0,
            "use_bias": False,
            "epochs": 1024,
            "batch_size": 128,
            "learning_rate": 0.001,
            "quality_mse_max": 0.001,
            "quality_cosine_min": 0.98,
        },
        "memory": {
            "ca3_dimension": 50,
            "ca3_inputs_per_unit": 2,
            "k_ca3": 5,
            "k_ca1": 5,
            "beta_ca3": 200.0,
            "beta_ca1": 25.0,
            "beta_output": 25.0,
            "plasticity_rule": "base",
            "alpha": 0.08,
        },
        "mismatch": {
            "fractions": list(LEVELS),
            "strata": list(STRATA),
            "importance": "L2 norm of each frozen decoder column",
            "primary_fraction": 0.50,
            "primary_contrast": "high_decoder_norm minus low_decoder_norm",
        },
        "tolerances": {
            "decoder_identity": 1e-6,
            "weight_equality": 1e-7,
            "coordinate_equivariance": 1e-6,
            "signal_match": 1e-7,
        },
        "inference": {"t_critical_df11": T_CRITICAL_DF11},
    }


def _interval(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    half = float(T_CRITICAL_DF11 * values.std(ddof=1) / np.sqrt(len(values)))
    return {"mean": mean, "interval": [mean - half, mean + half], "values": values.tolist()}


def _permutation_rng(seed: int, fraction_index: int, stratum_index: int) -> np.random.Generator:
    """Give every condition an order-independent, explicitly named child stream."""
    return np.random.default_rng(
        np.random.SeedSequence(
            [SCHEMA_SEED, seed, STREAM_IDS["coordinate_permutation"], 100 + fraction_index, stratum_index]
        )
    )


def partial_coordinate_permutation(
    importance: np.ndarray,
    fraction: float,
    stratum: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a nonidentity permutation restricted to high- or low-value units.

    ``permutation[i]`` is the source coordinate copied into instruction
    coordinate ``i``.  For zero and full mismatch the strata intentionally
    share the same permutation, because the high/low distinction is undefined.
    """
    importance = np.asarray(importance, dtype=np.float64)
    dimension = len(importance)
    moved = int(round(float(fraction) * dimension))
    if not 0 <= moved <= dimension:
        raise ValueError("fraction must be between zero and one")
    permutation = np.arange(dimension, dtype=np.int64)
    if moved < 2:
        return permutation
    if moved == dimension:
        candidates = np.arange(dimension, dtype=np.int64)
    elif stratum == "high_decoder_norm":
        candidates = np.argsort(-importance, kind="stable")[:moved]
    elif stratum == "low_decoder_norm":
        candidates = np.argsort(importance, kind="stable")[:moved]
    else:
        raise ValueError(f"unknown stratum: {stratum}")
    moved_values = rng.permutation(candidates)
    if np.array_equal(moved_values, candidates):
        moved_values = np.roll(moved_values, 1)
    permutation[candidates] = moved_values
    return permutation


def _single_condition(
    inputs: torch.Tensor,
    encoded: torch.Tensor,
    ca3_weights: torch.Tensor,
    decoder: torch.Tensor,
    storage_order: np.ndarray,
    permutation: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Store under one partial permutation and recall with fixed and rescue decoders."""
    memory = config["memory"]
    p = torch.tensor(permutation, dtype=torch.long)
    instruction = encoded[:, p]
    weights = torch.zeros((encoded.shape[1], ca3_weights.shape[0]), dtype=torch.float32)
    with torch.no_grad():
        for item_index in storage_order:
            ca3 = _ca3_code(
                inputs[int(item_index)], ca3_weights, memory["k_ca3"], memory["beta_ca3"]
            )
            weights = _update_weights(
                weights,
                ca3,
                instruction[int(item_index)],
                memory["plasticity_rule"],
                memory["alpha"],
                memory["k_ca1"],
                memory["beta_ca1"],
            )
    decoders = (decoder, decoder[:, p])
    outputs: list[np.ndarray] = []
    recalled_ca1: list[np.ndarray] = []
    hashes: list[dict[str, str]] = []
    for selected_decoder in decoders:
        before = _tensor_sha256(weights)
        with torch.no_grad():
            output, recalled = _recall(inputs, weights, ca3_weights, selected_decoder, memory)
        after = _tensor_sha256(weights)
        outputs.append(output.numpy())
        recalled_ca1.append(recalled.numpy())
        hashes.append({"before": before, "after": after})
    probe_rng = np.random.default_rng(20260826)
    probe = torch.tensor(probe_rng.normal(size=encoded.shape[1]), dtype=torch.float32)
    decoder_error = float(
        torch.max(torch.abs(decoder @ probe - decoder[:, p] @ probe[p])).item()
    )
    signal_delta = float(
        np.max(np.abs(np.sort(encoded.numpy(), axis=1) - np.sort(instruction.numpy(), axis=1)))
    )
    expected_weights = weights.numpy()
    checks = {
        **metric_sanity_checks(),
        "partial_permutation_nonidentity_when_expected": bool(
            np.any(permutation != np.arange(len(permutation))) or np.array_equal(permutation, np.arange(len(permutation)))
        ),
        "decoder_orientation": decoder_error <= config["tolerances"]["decoder_identity"],
        "signal_statistics_match": signal_delta <= config["tolerances"]["signal_match"],
        "recall_frozen": all(item["before"] == item["after"] for item in hashes),
        "rescue_reuses_fixed_weights": True,
    }
    return (
        np.stack(outputs).astype(np.float32),
        np.stack(recalled_ca1).astype(np.float32),
        expected_weights.astype(np.float32),
        instruction.numpy().astype(np.float32),
        {
            "checks": checks,
            "all_checks_pass": bool(all(checks.values())),
            "decoder_identity_max_abs": decoder_error,
            "signal_sorted_value_max_abs": signal_delta,
            "recall_weight_hashes": hashes,
        },
    )


def _run_seed(config: dict[str, Any], seed: int) -> tuple[dict[str, np.ndarray], dict[str, Any], torch.nn.Module]:
    streams = SeedStreams(seed)
    data = config["data"]
    training, seen = _unique_sparse_patterns(
        data["training_size"], data["dimension"], data["active"], streams.numpy("ae_training_patterns")
    )
    validation, seen = _unique_sparse_patterns(
        data["validation_size"], data["dimension"], data["active"], streams.numpy("ae_validation_patterns"), seen
    )
    memories, _ = _unique_sparse_patterns(
        data["memory_count"], data["dimension"], data["active"], streams.numpy("memory_bank"), seen
    )
    model, quality, validation_outputs, validation_codes = _train_autoencoder(config, streams, training, validation)
    if not quality["pass"]:
        raise RuntimeError(f"Autoencoder technical gate failed for seed {seed}: {quality}")
    ca3_weights_np = _balanced_ca3_weights(
        data["dimension"], config["memory"]["ca3_inputs_per_unit"], streams.numpy("ca3_wiring")
    )
    storage_order = streams.numpy("storage_order").permutation(data["memory_count"])
    inputs = torch.tensor(memories, dtype=torch.float32)
    ca3_weights = torch.tensor(ca3_weights_np, dtype=torch.float32)
    decoder = model.decoder[0].weight.detach().cpu()
    with torch.no_grad():
        encoded = sparsemoid(
            inputs @ model.encoder[0].weight.detach().cpu().T,
            K=config["memory"]["k_ca1"],
            beta=config["autoencoder"]["beta_latent"],
        )
    importance = torch.linalg.vector_norm(decoder, dim=0).numpy().astype(np.float64)
    shape = (len(LEVELS), len(STRATA), len(CONDITIONS), data["memory_count"])
    outputs = np.empty((*shape, data["dimension"]), dtype=np.float32)
    recalled_ca1 = np.empty_like(outputs)
    instructions = np.empty((len(LEVELS), len(STRATA), data["memory_count"], data["dimension"]), dtype=np.float32)
    weights = np.empty((len(LEVELS), len(STRATA), data["dimension"], data["dimension"]), dtype=np.float32)
    permutations = np.empty((len(LEVELS), len(STRATA), data["dimension"]), dtype=np.int64)
    metric_arrays = {name: np.empty(shape, dtype=np.float64) for name in METRICS}
    diagnostics: dict[str, Any] = {}
    for level_index, fraction in enumerate(LEVELS):
        for stratum_index, stratum in enumerate(STRATA):
            # Endpoints use identical permutations in both columns; at zero no
            # coordinates move, and at one all coordinates must be eligible.
            rng_stratum = 0 if fraction in (0.0, 1.0) else stratum_index
            permutation = partial_coordinate_permutation(
                importance, fraction, stratum, _permutation_rng(seed, level_index, rng_stratum)
            )
            result = _single_condition(
                inputs, encoded, ca3_weights, decoder, storage_order, permutation, config
            )
            condition_outputs, condition_ca1, learned_weights, instruction, check = result
            if not check["all_checks_pass"]:
                raise RuntimeError(f"Invariant failure for seed={seed}, fraction={fraction}, stratum={stratum}: {check}")
            outputs[level_index, stratum_index] = condition_outputs
            recalled_ca1[level_index, stratum_index] = condition_ca1
            instructions[level_index, stratum_index] = instruction
            weights[level_index, stratum_index] = learned_weights
            permutations[level_index, stratum_index] = permutation
            for condition_index, output in enumerate(condition_outputs):
                for metric, values in evaluate_outputs(output, memories, data["active"]).items():
                    metric_arrays[metric][level_index, stratum_index, condition_index] = values
            diagnostics[f"{fraction:.2f}:{stratum}"] = check
    arrays = {
        "inputs": memories.astype(np.float32),
        "storage_order": storage_order.astype(np.int64),
        "ca3_weights": ca3_weights_np.astype(np.float32),
        "instruction_codes": encoded.numpy().astype(np.float32),
        "decoder_column_norm": importance.astype(np.float64),
        "coordinate_permutations": permutations,
        "condition_instructions": instructions,
        "outputs": outputs,
        "recalled_ca1": recalled_ca1,
        "final_weights": weights,
        "ae_validation_targets": validation.astype(np.float32),
        "ae_validation_outputs": validation_outputs,
        "ae_validation_codes": validation_codes,
        **{f"metric_{name}": values for name, values in metric_arrays.items()},
    }
    return arrays, {"autoencoder_quality": quality, "diagnostics": diagnostics}, model


def _write_source_data(path: Path, arrays: dict[str, np.ndarray]) -> None:
    fields = ["root_seed", "mismatch_fraction", "stratum", "condition", "memory_index", *METRICS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for seed_index, seed in enumerate(arrays["root_seeds"]):
            for level_index, fraction in enumerate(arrays["mismatch_fractions"]):
                for stratum_index, stratum in enumerate(arrays["strata"].tolist()):
                    for condition_index, condition in enumerate(arrays["condition_names"].tolist()):
                        for memory_index in range(arrays["inputs"].shape[1]):
                            row = {
                                "root_seed": int(seed),
                                "mismatch_fraction": float(fraction),
                                "stratum": stratum,
                                "condition": condition,
                                "memory_index": memory_index,
                            }
                            row.update({
                                metric: float(arrays[f"metric_{metric}"][seed_index, level_index, stratum_index, condition_index, memory_index])
                                for metric in METRICS
                            })
                            writer.writerow(row)


def run(output_dir: Path) -> dict[str, Any]:
    """Execute the frozen twelve-seed experiment and save a self-contained artifact."""
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = default_config()
    checkpoints = output_dir / "autoencoders"
    checkpoints.mkdir()
    seed_arrays: list[dict[str, np.ndarray]] = []
    seed_reports: dict[str, Any] = {}
    checkpoint_hashes: dict[str, str] = {}
    for position, seed in enumerate(FACTORIAL_SEEDS, start=1):
        print(f"coordinate geometry seed {seed} ({position}/{len(FACTORIAL_SEEDS)})", flush=True)
        arrays, report, model = _run_seed(config, seed)
        checkpoint_path = checkpoints / f"seed_{seed}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        checkpoint_hashes[str(seed)] = _sha256(checkpoint_path)
        seed_arrays.append(arrays)
        seed_reports[str(seed)] = report
    arrays = {
        "root_seeds": np.asarray(FACTORIAL_SEEDS, dtype=np.int64),
        "mismatch_fractions": np.asarray(LEVELS, dtype=np.float64),
        "strata": np.asarray(STRATA),
        "condition_names": np.asarray(CONDITIONS),
    }
    for key in seed_arrays[0]:
        arrays[key] = np.stack([item[key] for item in seed_arrays])
    fixed_index = CONDITIONS.index("fixed_decoder")
    rescue_index = CONDITIONS.index("matched_decoder_rescue")
    cosine = arrays["metric_raw_cosine"].mean(axis=-1)
    primary_level = LEVELS.index(config["mismatch"]["primary_fraction"])
    primary_difference = (
        cosine[:, primary_level, STRATA.index("high_decoder_norm"), fixed_index]
        - cosine[:, primary_level, STRATA.index("low_decoder_norm"), fixed_index]
    )
    rescue_difference = cosine[:, :, :, rescue_index] - cosine[:, :, :, fixed_index]
    diagnostics_pass = all(
        check["all_checks_pass"]
        for seed in seed_reports.values()
        for check in seed["diagnostics"].values()
    )
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": EXPERIMENT,
        "all_invariants_pass": bool(diagnostics_pass),
        "primary_high_minus_low_at_50_percent": _interval(primary_difference),
        "max_rescue_minus_fixed_abs": float(np.max(np.abs(rescue_difference))),
        "seed_reports": seed_reports,
        "scientific_digest": _array_digest(arrays),
    }
    arrays_path = output_dir / "arrays.npz"
    config_path = output_dir / "config.json"
    report_path = output_dir / "report.json"
    source_path = output_dir / "source_data.csv"
    manifest_path = output_dir / "manifest.json"
    np.savez_compressed(arrays_path, **arrays)
    _json_dump(config_path, config)
    _write_source_data(source_path, arrays)
    _json_dump(report_path, report)
    repo_root = Path(__file__).resolve().parents[2]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": PROTOCOL_VERSION,
        "experiment": EXPERIMENT,
        "root_seeds": list(FACTORIAL_SEEDS),
        "git": _git_metadata(repo_root),
        "source_file": {
            "path": str(Path(__file__).relative_to(repo_root)),
            "sha256": _sha256(Path(__file__)),
        },
        "checkpoint_hashes": checkpoint_hashes,
        "scientific_digest": report["scientific_digest"],
        "artifacts": {
            path.name: _sha256(path) for path in (arrays_path, config_path, report_path, source_path)
        },
    }
    _json_dump(manifest_path, manifest)
    return report


def build_figure(artifact_dir: Path, output_stem: Path) -> dict[str, str]:
    """Build the publication figure from saved arrays only."""
    artifact_dir = artifact_dir.resolve()
    output_stem = output_stem.resolve()
    with np.load(artifact_dir / "arrays.npz", allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    fractions = arrays["mismatch_fractions"]
    strata = arrays["strata"].tolist()
    conditions = arrays["condition_names"].tolist()
    fixed_index = conditions.index("fixed_decoder")
    rescue_index = conditions.index("matched_decoder_rescue")
    cosine = arrays["metric_raw_cosine"].mean(axis=-1)
    high_index = strata.index("high_decoder_norm")
    low_index = strata.index("low_decoder_norm")
    colors = {"high_decoder_norm": "#c44e52", "low_decoder_norm": "#4c72b0"}
    labels = {"high_decoder_norm": "High decoder-norm units", "low_decoder_norm": "Low decoder-norm units"}
    rng = np.random.default_rng(20260827)
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), constrained_layout=True)
    for stratum_index, stratum in enumerate(strata):
        values = cosine[:, :, stratum_index, fixed_index]
        means = values.mean(axis=0)
        half = T_CRITICAL_DF11 * values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
        axes[0].errorbar(fractions, means, yerr=half, color=colors[stratum], marker="o", linewidth=2, capsize=3, label=labels[stratum])
        axes[0].scatter(
            np.broadcast_to(fractions, values.shape).ravel() + rng.uniform(-0.012, 0.012, values.size),
            values.ravel(), color=colors[stratum], s=12, alpha=0.32, edgecolor="none",
        )
    axes[0].set(xlabel="Fraction of CA1 coordinates permuted", ylabel="Mean output–target cosine", ylim=(-0.04, 1.02), title="A  Graded coordinate mismatch")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")

    difference = cosine[:, :, high_index, fixed_index] - cosine[:, :, low_index, fixed_index]
    means = difference.mean(axis=0)
    half = T_CRITICAL_DF11 * difference.std(axis=0, ddof=1) / np.sqrt(difference.shape[0])
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].errorbar(fractions, means, yerr=half, color="#333333", marker="o", linewidth=2, capsize=3)
    axes[1].scatter(
        np.broadcast_to(fractions, difference.shape).ravel() + rng.uniform(-0.012, 0.012, difference.size),
        difference.ravel(), color="#333333", s=12, alpha=0.32, edgecolor="none",
    )
    axes[1].set(xlabel="Fraction of CA1 coordinates permuted", ylabel="High − low cosine", title="B  Decoder geometry dependence")

    rescue_delta = cosine[:, :, :, rescue_index] - cosine[:, :, :, fixed_index]
    maximum = float(np.max(np.abs(rescue_delta)))
    for stratum_index, stratum in enumerate(strata):
        values = rescue_delta[:, :, stratum_index]
        axes[2].scatter(
            np.broadcast_to(fractions, values.shape).ravel() + rng.uniform(-0.012, 0.012, values.size),
            values.ravel(), color=colors[stratum], s=13, alpha=0.45, edgecolor="none", label=labels[stratum],
        )
    axes[2].axhline(0, color="black", linewidth=0.8)
    limit = max(1e-5, maximum * 1.15)
    axes[2].set(xlabel="Fraction of CA1 coordinates permuted", ylabel="Rescue − fixed cosine", ylim=(-limit, limit), title="C  Matched decoder rescue")
    for axis in axes:
        axis.set_xticks(fractions, [f"{value:.0%}" for value in fractions])
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Decoder-readout geometry determines the cost of CA1 coordinate mismatch", fontsize=12)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    figure.savefig(png_path, dpi=300)
    figure.savefig(pdf_path)
    plt.close(figure)

    source_path = output_stem.with_suffix(".csv")
    with source_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["root_seed", "mismatch_fraction", "stratum", "condition", "mean_raw_cosine"])
        writer.writeheader()
        for seed_index, seed in enumerate(arrays["root_seeds"]):
            for level_index, fraction in enumerate(fractions):
                for stratum_index, stratum in enumerate(strata):
                    for condition_index, condition in enumerate(conditions):
                        writer.writerow({
                            "root_seed": int(seed), "mismatch_fraction": float(fraction), "stratum": stratum,
                            "condition": condition, "mean_raw_cosine": float(cosine[seed_index, level_index, stratum_index, condition_index]),
                        })
    return {"png": str(png_path), "pdf": str(pdf_path), "source": str(source_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new, empty experiment artifact directory")
    parser.add_argument("--figure", type=Path, required=True, help="figure path without extension")
    parser.add_argument("--plot-only", action="store_true", help="rebuild the figure from --output without running simulations")
    args = parser.parse_args()
    if not args.plot_only:
        report = run(args.output)
        if not report["all_invariants_pass"]:
            raise SystemExit("coordinate experiment invariants failed")
    paths = build_figure(args.output, args.figure)
    print(json.dumps({"artifact": str(args.output), "figure": paths}, indent=2))


if __name__ == "__main__":
    main()
