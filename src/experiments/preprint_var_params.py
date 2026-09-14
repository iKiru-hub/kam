"""One-at-a-time sensitivity analysis around the evolved MTL parameters.

Each MTL parameter is multiplied by a fixed factor while all other parameters
remain at that plasticity rule's evolved optimum.  Candidates are evaluated
with the same clean/MEC-drop/LEC-drop objective used by
``preprint_mlt_evolution.py``.

The independent parameter points are distributed across worker processes.  A
worker receives the prepared data and frozen autoencoders once at startup,
rather than retraining the autoencoder or repeatedly transferring it for every
grid point.

Full run::

    python src/experiments/preprint_var_params.py --workers 8

Fast pipeline check::

    python src/experiments/preprint_var_params.py --quick --workers 2
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiments.preprint_mlt_evolution import (  # noqa: E402
    PARAMETER_NAMES,
    RULE_LABELS,
    evaluate_candidate_detailed,
    prepare_seeds,
)


DEFAULT_MTL_RESULTS = ROOT_DIR / "results/preprint/mtl_evolution"
DEFAULT_OUTPUT = ROOT_DIR / "results/preprint/parameter_sensitivity"
DEFAULT_MULTIPLIERS = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 7.0, 9.0]

# Independent evaluation seeds, disjoint from the three CMA-ES development
# seeds. Perturbed and unmodified candidates are paired on every seed.
N_EVALUATION_RUNS = 10
EVALUATION_SEEDS = list(range(54001, 54001 + N_EVALUATION_RUNS))

# Sensitivity bounds should express model validity, not the narrower domain
# chosen for CMA-ES.  Reusing the search bounds previously mapped every
# beta_CA1 value below 5 back to 5 when the evolved optimum sat on that bound.
# This made genuinely different requested multipliers appear as duplicate 0s.
SENSITIVITY_LOWER = np.array([1.0, 1.0, 0.10, 0.10, 1e-5])
SENSITIVITY_UPPER = np.array([49.0, 49.0, 5000.0, 5000.0, 1.0])

# These globals are initialized once inside each process.  This avoids sending
# three pretrained autoencoders and all track laps with every individual job.
_WORKER_PREPARED: list[dict[str, Any]] | None = None
_WORKER_SETTINGS: dict[str, Any] | None = None


def initialize_worker(prepared: list[dict[str, Any]], settings: dict[str, Any]) -> None:
    """Install shared read-only evaluation data in a worker process."""

    global _WORKER_PREPARED, _WORKER_SETTINGS
    _WORKER_PREPARED = prepared
    _WORKER_SETTINGS = settings

    # A process already supplies parallelism.  Restricting numerical libraries
    # to one thread per process prevents severe CPU oversubscription.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def evaluate_job(job: tuple[str, tuple[float, ...]]) -> tuple[str, tuple[float, ...], dict[str, Any]]:
    """Evaluate one unique (rule, candidate) pair inside a worker."""

    if _WORKER_PREPARED is None or _WORKER_SETTINGS is None:
        raise RuntimeError("worker was not initialized")
    rule, candidate_tuple = job
    candidate = np.asarray(candidate_tuple, dtype=float)
    details = evaluate_candidate_detailed(
        candidate,
        _WORKER_PREPARED,
        _WORKER_SETTINGS,
        rule,
    )
    return rule, candidate_tuple, details


def load_inputs(results: Path, rules: list[str],
                degradation_fraction: float | None) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, float]]:
    """Load evolved optima and place both rules on one evaluation protocol.

    The current saved searches used different degradation fractions.  Parameter
    sensitivity must compare rules under the same probe, so this analysis
    explicitly replaces that one setting with ``degradation_fraction`` while
    requiring every other evaluation setting to match.
    """

    settings = None
    best = {}
    source_fractions = {}
    loaded_settings = {}
    for rule in rules:
        rule_directory = results / rule
        config_path = rule_directory / "config.json"
        best_path = rule_directory / "best_parameters.json"
        if not config_path.exists() or not best_path.exists():
            raise FileNotFoundError(
                f"missing evolved results for {rule}: expected {config_path} and {best_path}"
            )

        rule_config = json.loads(config_path.read_text())
        rule_settings = rule_config["settings"]
        source_fractions[rule] = float(rule_settings["degradation_fraction"])
        loaded_settings[rule] = rule_settings

        values = json.loads(best_path.read_text())
        best[rule] = np.asarray([values[name] for name in PARAMETER_NAMES], dtype=float)

    # Normally inherit the fraction actually optimized.  If historical search
    # files disagree, require an explicit common value instead of guessing.
    if degradation_fraction is None:
        unique_fractions = set(source_fractions.values())
        if len(unique_fractions) != 1:
            raise ValueError(
                "MTL searches used different degradation fractions; pass "
                "--degradation-fraction to choose a common sensitivity probe"
            )
        degradation_fraction = unique_fractions.pop()

    for rule in rules:
        rule_settings = loaded_settings[rule]
        rule_settings["degradation_fraction"] = float(degradation_fraction)
        if settings is None:
            settings = rule_settings
        elif rule_settings != settings:
            raise ValueError(
                "plasticity searches used different evaluation settings; "
                "a matched sensitivity comparison requires identical settings"
            )

    assert settings is not None
    return settings, best, source_fractions


def varied_candidate(best: np.ndarray, parameter_index: int, multiplier: float) -> np.ndarray:
    """Vary one value, enforce model-valid bounds, and round counts."""

    candidate = np.asarray(best, dtype=float).copy()
    candidate[parameter_index] *= multiplier
    candidate = np.clip(candidate, SENSITIVITY_LOWER, SENSITIVITY_UPPER)
    candidate[:2] = np.rint(candidate[:2])
    return candidate


def make_jobs(best: dict[str, np.ndarray], rules: list[str],
              multipliers: list[float]) -> tuple[list[tuple[str, tuple[float, ...]]], dict]:
    """Create unique jobs plus a lookup from heatmap cell to candidate.

    Integer rounding means, for example, 0.8 x K=3 and 0.9 x K=3 may both
    produce K=3.  Deduplicating these candidates avoids needless simulations.
    """

    jobs = set()
    cells = {}
    for rule in rules:
        for parameter_index in range(len(PARAMETER_NAMES)):
            for multiplier_index, multiplier in enumerate(multipliers):
                candidate = varied_candidate(best[rule], parameter_index, multiplier)
                key = (rule, tuple(float(value) for value in candidate))
                jobs.add(key)
                cells[(rule, parameter_index, multiplier_index)] = key
    return sorted(jobs), cells


def run_jobs(jobs: list[tuple[str, tuple[float, ...]]], prepared: list[dict[str, Any]],
             settings: dict[str, Any], workers: int) -> dict:
    """Evaluate all unique candidates serially or with a spawn process pool."""

    if workers == 1:
        initialize_worker(prepared, settings)
        completed = [evaluate_job(job) for job in jobs]
    else:
        # Spawn is safe with PyTorch and matches the evolution runner.  Thread
        # limits are inherited before native numerical libraries initialize.
        variables = (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
        )
        previous = {name: os.environ.get(name) for name in variables}
        try:
            for name in variables:
                os.environ[name] = "1"
            with mp.get_context("spawn").Pool(
                processes=workers,
                initializer=initialize_worker,
                initargs=(prepared, settings),
            ) as pool:
                completed = pool.map(evaluate_job, jobs, chunksize=1)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    return {(rule, candidate): details for rule, candidate, details in completed}


def organize_results(rules: list[str], multipliers: list[float], best: dict[str, np.ndarray],
                     cells: dict, evaluations: dict) -> tuple[np.ndarray, np.ndarray, list[dict], list[dict]]:
    """Convert worker results into arrays and tidy source-data rows."""

    shape = (len(rules), len(PARAMETER_NAMES), len(multipliers))
    components = ("fitness", "clean_score", "mec_degraded_score", "lec_degraded_score")
    scores = np.empty(shape + (len(components),), dtype=float)
    actual_values = np.empty(shape, dtype=float)
    rows, seed_rows = [], []

    for rule_index, rule in enumerate(rules):
        for parameter_index, parameter in enumerate(PARAMETER_NAMES):
            for multiplier_index, multiplier in enumerate(multipliers):
                key = cells[(rule, parameter_index, multiplier_index)]
                candidate = np.asarray(key[1])
                details = evaluations[key]
                actual = candidate[parameter_index]
                actual_values[rule_index, parameter_index, multiplier_index] = actual
                scores[rule_index, parameter_index, multiplier_index] = [
                    details[name] for name in components
                ]
                rows.append({
                    "plasticity_rule": rule,
                    "plasticity_label": RULE_LABELS[rule],
                    "varied_parameter": parameter,
                    "multiplier": multiplier,
                    "evolved_value": best[rule][parameter_index],
                    "actual_value": actual,
                    **{name: details[name] for name in components},
                })
                for per_seed in details["per_seed"]:
                    seed_rows.append({
                        "plasticity_rule": rule,
                        "varied_parameter": parameter,
                        "multiplier": multiplier,
                        "actual_value": actual,
                        **per_seed,
                    })
    return scores, actual_values, rows, seed_rows


def paired_delta_summary(rules: list[str], multipliers: list[float],
                         seed_rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return mean and SD of paired per-seed changes from the 1x candidate."""

    shape = (len(rules), len(PARAMETER_NAMES), len(multipliers))
    means = np.empty(shape, dtype=float)
    deviations = np.empty(shape, dtype=float)

    for rule_index, rule in enumerate(rules):
        for parameter_index, parameter in enumerate(PARAMETER_NAMES):
            selected = [row for row in seed_rows
                        if row["plasticity_rule"] == rule
                        and row["varied_parameter"] == parameter]
            baseline = {
                int(row["seed"]): float(row["composite_score"])
                for row in selected
                if np.isclose(float(row["multiplier"]), 1.0)
            }
            for multiplier_index, multiplier in enumerate(multipliers):
                deltas = np.asarray([
                    float(row["composite_score"]) - baseline[int(row["seed"])]
                    for row in selected
                    if np.isclose(float(row["multiplier"]), multiplier)
                ])
                if len(deltas) != len(baseline):
                    raise RuntimeError("missing paired sensitivity evaluation")
                means[rule_index, parameter_index, multiplier_index] = deltas.mean()
                deviations[rule_index, parameter_index, multiplier_index] = (
                    deltas.std(ddof=1) if len(deltas) > 1 else 0.0
                )
    return means, deviations


def make_heatmap(output: Path, rules: list[str], multipliers: list[float],
                 delta_mean: np.ndarray, delta_std: np.ndarray) -> None:
    """Plot paired mean score change and across-run standard deviation."""

    baseline_index = int(np.argmin(np.abs(np.asarray(multipliers) - 1.0)))
    if not np.isclose(multipliers[baseline_index], 1.0):
        raise ValueError("multipliers must contain 1.0 for the evolved baseline")

    color_limit = max(float(np.max(np.abs(delta_mean))), 0.01)

    figure, axes = plt.subplots(
        1, len(rules),
        figsize=(6.2 * len(rules), 3.9),
        constrained_layout=True,
        squeeze=False,
    )
    images = []
    parameter_labels = (
        "CA3 fan-in", r"$K_{CA3}$", r"$\beta_{CA3}$",
        r"$\beta_{CA1}$", r"$\alpha$",
    )
    for rule_index, rule in enumerate(rules):
        axis = axes[0, rule_index]
        image = axis.imshow(
            delta_mean[rule_index],
            aspect="auto",
            cmap="RdBu_r",
            vmin=-color_limit,
            vmax=color_limit,
        )
        images.append(image)
        axis.set(
            xticks=np.arange(len(multipliers)),
            xticklabels=[f"{value:g}x" for value in multipliers],
            yticks=np.arange(len(PARAMETER_NAMES)),
            yticklabels=parameter_labels,
            xlabel="Multiplier of evolved value",
            title=RULE_LABELS[rule],
        )
        axis.tick_params(axis="x", rotation=45)

        # Numeric annotations keep modest effects visible in print and make the
        # plot interpretable without guessing exact colors.
        for row in range(len(PARAMETER_NAMES)):
            for column in range(len(multipliers)):
                value = delta_mean[rule_index, row, column]
                uncertainty = delta_std[rule_index, row, column]
                color = "white" if abs(value) > 0.55 * color_limit else "black"
                label = "0.000" if abs(value) < 0.0005 else f"{value:+.3f}"
                axis.text(column, row, f"{label}\n±{uncertainty:.3f}",
                          ha="center", va="center", fontsize=6.2,
                          linespacing=0.9, color=color)

    colorbar = figure.colorbar(images[0], ax=axes.ravel().tolist(), shrink=0.85)
    colorbar.set_label(r"Mean paired $\Delta$ from selected configuration")
    figure.savefig(output / "parameter_sensitivity_heatmap.svg", bbox_inches="tight")
    figure.savefig(output / "parameter_sensitivity_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_results(output: Path, rules: list[str], multipliers: list[float],
                 settings: dict[str, Any], best: dict[str, np.ndarray],
                 scores: np.ndarray, actual_values: np.ndarray,
                 rows: list[dict], seed_rows: list[dict], workers: int,
                 source: Path, source_fractions: dict[str, float],
                 optimization_seeds: list[int]) -> None:
    """Save plot-ready arrays, tidy tables, and full analysis provenance."""

    output.mkdir(parents=True, exist_ok=True)
    delta_mean, delta_std = paired_delta_summary(rules, multipliers, seed_rows)
    np.savez_compressed(
        output / "arrays.npz",
        rules=np.asarray(rules),
        parameter_names=np.asarray(PARAMETER_NAMES),
        multipliers=np.asarray(multipliers),
        component_names=np.asarray([
            "fitness", "clean_score", "mec_degraded_score", "lec_degraded_score"
        ]),
        scores=scores,
        actual_values=actual_values,
        best_parameters=np.stack([best[rule] for rule in rules]),
        paired_delta_mean=delta_mean,
        paired_delta_std=delta_std,
    )
    config = {
        "analysis": "one-at-a-time sensitivity around each rule's evolved optimum",
        "source_mtl_results": str(source.resolve()),
        "source_search_degradation_fractions": source_fractions,
        "optimization_seeds": optimization_seeds,
        "independent_sensitivity_seeds": settings["seeds"],
        "common_sensitivity_degradation_fraction": settings["degradation_fraction"],
        "rules": rules,
        "parameter_names": list(PARAMETER_NAMES),
        "multipliers": multipliers,
        "sensitivity_lower_bounds": SENSITIVITY_LOWER.tolist(),
        "sensitivity_upper_bounds": SENSITIVITY_UPPER.tolist(),
        "workers": workers,
        "settings": settings,
        "note": "cells report paired mean change plus across-seed SD; integer parameters are rounded; sensitivity uses model-valid bounds, not narrower CMA search bounds",
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    for filename, table in (("source_data.csv", rows), ("per_seed.csv", seed_rows)):
        with (output / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    make_heatmap(output, rules, multipliers, delta_mean, delta_std)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mtl-results", type=Path, default=DEFAULT_MTL_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rules", nargs="+", choices=tuple(RULE_LABELS),
                        default=list(RULE_LABELS))
    parser.add_argument("--multipliers", nargs="+", type=float,
                        default=DEFAULT_MULTIPLIERS)
    parser.add_argument("--workers", type=int,
                        default=min(8, os.cpu_count() or 1))
    parser.add_argument("--degradation-fraction", type=float, default=None,
                        help="common dropped fraction; defaults to the value shared by the searches")
    parser.add_argument("--evaluation-seeds", nargs="+", type=int,
                        default=EVALUATION_SEEDS,
                        help="independent seeds used for paired sensitivity evaluation")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if any(multiplier <= 0 for multiplier in args.multipliers):
        raise ValueError("all multipliers must be positive")
    if args.degradation_fraction is not None and not 0 <= args.degradation_fraction <= 1:
        raise ValueError("degradation fraction must lie between zero and one")

    settings, best, source_fractions = load_inputs(
        args.mtl_results, args.rules, args.degradation_fraction
    )
    optimization_seeds = list(settings["seeds"])
    settings["seeds"] = list(args.evaluation_seeds)
    multipliers = list(args.multipliers)
    output = args.output

    if args.quick:
        # The quick run validates multiprocessing and output generation; it is
        # not a scientific sensitivity result.
        settings["seeds"] = settings["seeds"][:2]
        settings["autoencoder"]["epochs"] = 3
        settings["track"]["training_laps"] = 4
        settings["track"]["validation_laps"] = 2
        settings["masks_per_modality"] = 2
        multipliers = [0.75, 1.0, 1.25]
        if args.output == DEFAULT_OUTPUT:
            output = ROOT_DIR / "results/preprint/parameter_sensitivity_quick"

    print("Preparing matched inputs and frozen autoencoders...")
    prepared = prepare_seeds(settings)
    jobs, cells = make_jobs(best, args.rules, multipliers)
    workers = min(args.workers, len(jobs), os.cpu_count() or 1)
    print(f"Evaluating {len(jobs)} unique parameter points with {workers} workers")
    evaluations = run_jobs(jobs, prepared, settings, workers)
    scores, actual_values, rows, seed_rows = organize_results(
        args.rules, multipliers, best, cells, evaluations
    )
    save_results(
        output, args.rules, multipliers, settings, best,
        scores, actual_values, rows, seed_rows, workers, args.mtl_results,
        source_fractions, optimization_seeds,
    )
    print(f"Saved sensitivity data and heatmap to {output}")


if __name__ == "__main__":
    main()
