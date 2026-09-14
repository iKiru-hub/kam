"""Choose MTL memory parameters for the preprint with CMA-ES.

The filename keeps the requested ``mlt`` spelling; the model is called MTL in
the codebase and manuscript.

This script is intentionally small.  It reuses the existing autoencoder,
MEC+LEC stimulus generator, MTL model, plasticity rules, degradation helpers,
and CMA-ES runner.  CMA-ES searches five parameters that affect both update
rules:

    CA3 fan-in, CA3 sparsity, beta_CA3, beta_CA1, and write rate alpha.

For each candidate we generate MEC+LEC vectors from cue-track laps, randomly
permute the vectors within every lap to remove neighboring-position order, and
calculate three scores:

1. clean recall of the stored lap,
2. recall after dropping 50% of MEC units,
3. recall after dropping 50% of LEC units.

Their equally weighted mean is maximized.  The instructive-driven (``base``)
and error-driven (``err2``) rules are optimized in separate matched searches,
because one rule's best parameters need not be fair to the other.

Examples
--------
Smoke test::

    python src/experiments/preprint_mlt_evolution.py --quick

Full searches, optionally using parameters selected by the AE search::

    python src/experiments/preprint_mlt_evolution.py --workers 4 \
        --ae-parameters results/preprint/ae_evolution/best_parameters.json
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
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

from core import datagen, models  # noqa: E402
from experiments.evolution import _lib  # noqa: E402
from experiments.preprint_common import (  # noqa: E402
    build_mtl,
    cue_track,
    row_cosine,
    run_mtl,
    train_autoencoder,
)
from experiments.preprint_figure_3 import (  # noqa: E402
    cue_accuracy,
    degrade,
    position_accuracy,
)


RULE_LABELS = {
    "base": "Instructive-driven",
    "err2": "Error-driven",
}

# ``shuffled`` preserves the MEC+LEC sample distribution but independently
# destroys track order in every lap before MTL storage. ``track`` retains the
# original chronological traversal and is kept as an explicit comparison.
STIMULUS_ORDER = "shuffled"
STIMULUS_ORDERS = ("shuffled", "track")

# The centers are the current Figure 2/3 memory parameters.  A zero CMA genome
# therefore starts from the manuscript configuration.
PARAMETER_NAMES = (
    "ca3_inputs_per_unit",
    "k_ca3",
    "beta_ca3",
    "beta_ca1",
    "alpha",
)
PARAMETER_CENTERS = np.array([29.0, 3.0, 170.6, 41.8, 0.092])
PARAMETER_SCALES = np.array([8.0, 2.0, 60.0, 20.0, 0.05])
PARAMETER_LOWER = np.array([1.0, 1.0, 5.0, 5.0, 0.005])
PARAMETER_UPPER = np.array([49.0, 15.0, 300.0, 120.0, 0.40])


DEFAULT_SETTINGS = {
    "seeds": [42001, 42002, 42003],
    "dimension": 50,
    "active": 5,
    "stimulus_order": STIMULUS_ORDER,
    "autoencoder": {
        "latent_dimension": 50,
        "beta_latent": 25.0,
        "beta_output": 25.0,
        "epochs": 256,
        "batch_size": 64,
        "learning_rate": 0.001,
    },
    "memory": {
        "ca3_dimension": 50,
        "k_ca1": 5,
    },
    "track": {
        "training_laps": 20,
        "validation_laps": 5,
        "lap_length": 50,
        "size": 50,
        "cue_positions": [10, 30],
        "cue_sigma": 4.0,
        "cue_beta": 40.0,
        "cue_alpha": 0.1,
        "mec_binarized": True,
        "mec_sigma": 5.0,
        "lec_sigma": 5.0,
    },
    "degradation_fraction": 0.85,
    "masks_per_modality": 6,
    "score_weights": {
        "clean": 1.0 / 3.0,
        "mec_degraded": 1.0 / 3.0,
        "lec_degraded": 1.0 / 3.0,
    },
}


DEFAULT_SEARCH = {
    "generations": 32,
    "population_size": 4 + int(3 * np.log(len(PARAMETER_NAMES))),
    "workers": 1,
    "boundary_penalty": 0.05,
}


def seed_value(seed: int, stream: int) -> int:
    """Produce stable independent random streams from a root seed."""

    return int(np.random.SeedSequence([seed, stream]).generate_state(1)[0])


def sanitize_candidate(genome: np.ndarray) -> np.ndarray:
    """Decode standardized CMA coordinates into valid MTL parameters."""

    genome = np.asarray(genome, dtype=float)
    if genome.shape != PARAMETER_CENTERS.shape:
        raise ValueError(f"expected {PARAMETER_CENTERS.shape}, got {genome.shape}")
    candidate = np.clip(
        PARAMETER_CENTERS + PARAMETER_SCALES * genome,
        PARAMETER_LOWER,
        PARAMETER_UPPER,
    )
    # Fan-in and top-K sparsity are counts.
    candidate[:2] = np.rint(candidate[:2])
    return candidate


def load_ae_parameters(settings: dict[str, Any], path: Path | None) -> None:
    """Optionally replace the preprint AE defaults with evolved parameters."""

    if path is None:
        return
    values = json.loads(path.read_text())
    settings["active"] = int(values["K_ca1"])
    settings["memory"]["k_ca1"] = int(values["K_ca1"])
    settings["autoencoder"]["beta_latent"] = float(values["beta_latent"])
    settings["autoencoder"]["beta_output"] = float(values["beta_output"])


def pack_autoencoder(model: models.Autoencoder) -> dict[str, Any]:
    """Turn a pretrained AE into a small process-safe dictionary."""

    return {
        "state": {
            name: value.detach().cpu().numpy()
            for name, value in model.state_dict().items()
        },
        "dim_ei": model._dim_ei,
        "dim_ca1": model._dim_ca1,
        "K_ca1": model._K_ca1,
        "K_eo": model._K_eo,
        "beta_ei": model._beta_ei,
        "beta_eo": model._beta_eo,
        "use_bias": model._use_bias,
    }


def unpack_autoencoder(bundle: dict[str, Any]) -> models.Autoencoder:
    """Reconstruct the frozen AE inside a candidate-evaluation process."""

    model = models.Autoencoder(
        dim_ei=bundle["dim_ei"],
        dim_ca1=bundle["dim_ca1"],
        K_ca1=bundle["K_ca1"],
        K_eo=bundle["K_eo"],
        beta_ei=bundle["beta_ei"],
        beta_eo=bundle["beta_eo"],
        use_bias=bundle["use_bias"],
    )
    model.load_state_dict({
        name: torch.as_tensor(value, dtype=torch.float32)
        for name, value in bundle["state"].items()
    })
    model.eval()
    return model


def order_lap_samples(
    laps: np.ndarray,
    cue_positions: list[int],
    seed: int,
    stimulus_order: str,
) -> tuple[np.ndarray, list[list[int]]]:
    """Optionally destroy track order while preserving every MEC+LEC vector.

    Cue locations are transformed by the same permutation so cue decoding is
    evaluated at the correct rows after shuffling. Position decoding requires
    no additional labels: every recalled MEC row is compared with the matching
    row and all alternative MEC templates in that shuffled target set.
    """

    if stimulus_order == "track":
        return laps.copy(), [list(cue_positions) for _ in range(len(laps))]
    if stimulus_order != "shuffled":
        raise ValueError(
            f"Unknown stimulus_order {stimulus_order!r}; choose one of {STIMULUS_ORDERS}"
        )

    rng = np.random.default_rng(seed)
    ordered = np.empty_like(laps)
    relocated_cues = []
    for lap_index, lap in enumerate(laps):
        permutation = rng.permutation(len(lap))
        ordered[lap_index] = lap[permutation]
        relocated_cues.append([
            int(np.flatnonzero(permutation == cue_position)[0])
            for cue_position in cue_positions
        ])
    return ordered, relocated_cues


def prepare_seeds(settings: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate inputs and pretrain one shared autoencoder per seed.

    This happens once before evolution.  Every candidate and both plasticity
    rules consequently see identical inputs, encoders, and degradation masks.
    """

    prepared = []
    track = settings["track"]
    for seed in settings["seeds"]:
        training = cue_track(
            track["training_laps"],
            track,
            [[0, 1]] * track["training_laps"],
            seed_value(seed, 1),
        )
        validation = cue_track(
            track["validation_laps"],
            track,
            [[0, 1]] * track["validation_laps"],
            seed_value(seed, 2),
        )
        training, training_cue_positions = order_lap_samples(
            training, track["cue_positions"], seed_value(seed, 4),
            settings["stimulus_order"],
        )
        validation, _ = order_lap_samples(
            validation, track["cue_positions"], seed_value(seed, 5),
            settings["stimulus_order"],
        )
        autoencoder, ae_mse = train_autoencoder(
            training.reshape(-1, settings["dimension"]),
            validation.reshape(-1, settings["dimension"]),
            settings,
            seed_value(seed, 3),
        )
        prepared.append({
            "seed": seed,
            "autoencoder": pack_autoencoder(autoencoder),
            "training_laps": training,
            "target": training[-1],
            "target_cue_positions": training_cue_positions[-1],
            "cue_patterns": datagen.make_cues(2, settings["dimension"] // 2, fixed=True),
            "ae_validation_mse": ae_mse,
        })
    return prepared


def candidate_memory(candidate: np.ndarray, rule: str, settings: dict[str, Any]) -> dict[str, Any]:
    """Map the five decoded values onto ``build_mtl``'s memory settings."""

    values = dict(zip(PARAMETER_NAMES, map(float, candidate)))
    return {
        "ca3_dimension": settings["memory"]["ca3_dimension"],
        "ca3_inputs_per_unit": int(values["ca3_inputs_per_unit"]),
        "k_ca3": int(values["k_ca3"]),
        "k_ca1": settings["memory"]["k_ca1"],
        "beta_ca3": values["beta_ca3"],
        "beta_ca1": values["beta_ca1"],
        "alpha": values["alpha"],
        "plasticity_rule": rule,
    }


def recall_scores(recall: np.ndarray, target: np.ndarray,
                  cues: np.ndarray, positions: list[int]) -> tuple[float, float, float]:
    """Return full fidelity, spatial decoding, and cue decoding scores."""

    return (
        float(row_cosine(recall, target).mean()),
        position_accuracy(recall, target),
        cue_accuracy(recall, cues, positions),
    )


def evaluate_candidate_detailed(
    candidate: np.ndarray,
    prepared: list[dict[str, Any]],
    settings: dict[str, Any],
    rule: str,
) -> dict[str, Any]:
    """Train and score one MTL candidate over all fixed development seeds."""

    memory = candidate_memory(candidate, rule, settings)
    track = settings["track"]
    fraction = settings["degradation_fraction"]
    per_seed = []

    for item in prepared:
        seed = item["seed"]
        model = build_mtl(
            unpack_autoencoder(item["autoencoder"]),
            memory,
            seed_value(seed, 10),
        )

        # Store all clean laps, then freeze learning during every probe.
        for lap in item["training_laps"]:
            run_mtl(model, lap, learn=True)

        clean_recall, _, _ = run_mtl(model, item["target"], learn=False)
        clean_fidelity, clean_position, clean_cue = recall_scores(
            clean_recall, item["target"], item["cue_patterns"],
            item["target_cue_positions"],
        )
        clean_score = np.mean([clean_fidelity, clean_position, clean_cue])

        mec_scores, lec_scores = [], []
        for mask_index in range(settings["masks_per_modality"]):
            # Masks depend only on seed/modality/repetition, never candidate.
            mec_probe, _ = degrade(
                item["target"], fraction, "mec",
                np.random.default_rng(np.random.SeedSequence([seed, 20, mask_index])),
            )
            mec_recall, _, _ = run_mtl(model, mec_probe, learn=False)
            _, mec_position, mec_cue = recall_scores(
                mec_recall, item["target"], item["cue_patterns"],
                item["target_cue_positions"],
            )
            # Figure 3 asks whether spatial recall degrades gracefully while
            # intact cue identity remains decodable.
            mec_scores.append(np.mean([mec_position, mec_cue]))

            lec_probe, _ = degrade(
                item["target"], fraction, "lec",
                np.random.default_rng(np.random.SeedSequence([seed, 30, mask_index])),
            )
            lec_recall, _, _ = run_mtl(model, lec_probe, learn=False)
            _, _, lec_cue = recall_scores(
                lec_recall, item["target"], item["cue_patterns"],
                item["target_cue_positions"],
            )
            half = settings["dimension"] // 2
            lec_mec_fidelity = float(
                row_cosine(lec_recall[:, :half], item["target"][:, :half]).mean()
            )
            # This mirrors the complementary LEC-degradation panels: cue
            # classification plus preservation of the intact spatial output.
            lec_scores.append(np.mean([lec_cue, lec_mec_fidelity]))

        row = {
            "seed": seed,
            "clean_score": float(clean_score),
            "clean_fidelity": clean_fidelity,
            "clean_position_accuracy": clean_position,
            "clean_cue_accuracy": clean_cue,
            "mec_degraded_score": float(np.mean(mec_scores)),
            "lec_degraded_score": float(np.mean(lec_scores)),
            "ae_validation_mse": item["ae_validation_mse"],
        }
        weights = settings["score_weights"]
        row["composite_score"] = float(
            weights["clean"] * row["clean_score"]
            + weights["mec_degraded"] * row["mec_degraded_score"]
            + weights["lec_degraded"] * row["lec_degraded_score"]
        )
        per_seed.append(row)

    return {
        "fitness": float(np.mean([row["composite_score"] for row in per_seed])),
        "clean_score": float(np.mean([row["clean_score"] for row in per_seed])),
        "mec_degraded_score": float(np.mean([row["mec_degraded_score"] for row in per_seed])),
        "lec_degraded_score": float(np.mean([row["lec_degraded_score"] for row in per_seed])),
        "per_seed": per_seed,
    }


def evaluate_candidate(candidate: np.ndarray, prepared: list[dict[str, Any]],
                       settings: dict[str, Any], rule: str) -> float:
    """Picklable scalar objective used by CMA-ES."""

    score = evaluate_candidate_detailed(candidate, prepared, settings, rule)["fitness"]
    return float(score) if np.isfinite(score) else 0.0


def evaluate_population(population: np.ndarray, prepared: list[dict[str, Any]],
                        settings: dict[str, Any], rule: str) -> list[float]:
    """Serial fallback for population evaluation."""

    return [evaluate_candidate(candidate, prepared, settings, rule) for candidate in population]


def save_rule_results(output: Path, rule: str, record: dict[str, Any],
                      details: dict[str, Any], settings: dict[str, Any],
                      search: dict[str, Any]) -> None:
    """Save one rule's source data, best parameters, and summary figure."""

    rule_output = output / rule
    rule_output.mkdir(parents=True, exist_ok=True)
    fields = (
        "generations", "populations", "fitness", "optimizer_means",
        "best_candidates", "generation_best", "best_fitness",
        "population_mean_fitness", "sigma", "raw_populations",
        "raw_optimizer_means", "raw_best_candidates",
    )
    arrays = {field: np.asarray(record[field]) for field in fields}
    np.savez_compressed(rule_output / "history.npz", **arrays)

    parameters = dict(zip(PARAMETER_NAMES, map(float, arrays["best_candidates"][-1])))
    parameters["ca3_inputs_per_unit"] = int(round(parameters["ca3_inputs_per_unit"]))
    parameters["k_ca3"] = int(round(parameters["k_ca3"]))
    parameters.update({key: details[key] for key in (
        "fitness", "clean_score", "mec_degraded_score", "lec_degraded_score"
    )})
    (rule_output / "best_parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")

    config = {
        "plasticity_rule": rule,
        "plasticity_label": RULE_LABELS[rule],
        "parameter_names": list(PARAMETER_NAMES),
        "parameter_centers": PARAMETER_CENTERS.tolist(),
        "parameter_scales": PARAMETER_SCALES.tolist(),
        "parameter_lower": PARAMETER_LOWER.tolist(),
        "parameter_upper": PARAMETER_UPPER.tolist(),
        "settings": settings,
        "search": search,
        "actual_workers": record["workers"],
    }
    (rule_output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    with (rule_output / "best_validation_by_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details["per_seed"][0]))
        writer.writeheader()
        writer.writerows(details["per_seed"])

    with (rule_output / "candidate_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["generation", "candidate", "composite_score", *PARAMETER_NAMES],
        )
        writer.writeheader()
        for generation_index, generation in enumerate(arrays["generations"]):
            for candidate_index, candidate in enumerate(arrays["populations"][generation_index]):
                writer.writerow({
                    "generation": int(generation),
                    "candidate": candidate_index,
                    "composite_score": arrays["fitness"][generation_index, candidate_index],
                    **dict(zip(PARAMETER_NAMES, candidate)),
                })

    # One compact diagnostic plot is enough here; all numerical source data are
    # available for a manuscript-specific plot later.
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.1), constrained_layout=True)
    generations = arrays["generations"]
    axes[0].plot(generations, arrays["population_mean_fitness"], color="0.55", label="Population mean")
    axes[0].plot(generations, arrays["generation_best"], color="#4c78a8", label="Generation best")
    axes[0].plot(generations, arrays["best_fitness"], color="#e45756", linewidth=2, label="Best seen")
    axes[0].set(xlabel="Generation", ylabel="Composite score", title="A  Evolution search", ylim=(0, 1.02))
    axes[0].legend(frameon=False, fontsize=7)

    component_names = ["Clean", "MEC dropped", "LEC dropped"]
    component_keys = ["clean_score", "mec_degraded_score", "lec_degraded_score"]
    for row in details["per_seed"]:
        axes[1].plot(range(3), [row[key] for key in component_keys], color="0.75", linewidth=1)
    means = [details[key] for key in component_keys]
    axes[1].scatter(range(3), means, color=["#59a14f", "#4c78a8", "#f28e2b"], zorder=3)
    axes[1].set(xticks=range(3), xticklabels=component_names,
                ylabel="Best-candidate score", title="B  Objective components", ylim=(0, 1.02))
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.18)
    figure.suptitle(RULE_LABELS[rule])
    figure.savefig(rule_output / "mtl_evolution_summary.svg", bbox_inches="tight")
    figure.savefig(rule_output / "mtl_evolution_summary.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def run_rule(rule: str, prepared: list[dict[str, Any]], settings: dict[str, Any],
             search: dict[str, Any], output: Path) -> dict[str, Any]:
    """Run and save one plasticity rule's independent CMA-ES search."""

    latent_lower = (PARAMETER_LOWER - PARAMETER_CENTERS) / PARAMETER_SCALES
    latent_upper = (PARAMETER_UPPER - PARAMETER_CENTERS) / PARAMETER_SCALES
    cma_settings = {
        "num_parameters": len(PARAMETER_NAMES),
        "generations": search["generations"],
        "population_size": search["population_size"],
        "direction": "maximize",
        "metric_name": "composite_recall_score",
        "workers": search["workers"],
        "boundary_penalty": search["boundary_penalty"],
        "latent_lower": latent_lower,
        "latent_upper": latent_upper,
        "verbose": True,
    }
    individual = functools.partial(
        evaluate_candidate, prepared=prepared, settings=settings, rule=rule
    )
    population = functools.partial(
        evaluate_population, prepared=prepared, settings=settings, rule=rule
    )
    record = _lib.evolution_run(
        settings=cma_settings,
        evaluate=population,
        evaluate_individual=individual,
        sanitizer=sanitize_candidate,
        live_plot=False,
    )
    best = np.asarray(record["best_candidates"][-1])
    details = evaluate_candidate_detailed(best, prepared, settings, rule)
    save_rule_results(output, rule, record, details, settings, search)
    return {"candidate": best, **details}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT_DIR / "results/preprint/mtl_evolution")
    parser.add_argument("--rules", nargs="+", choices=tuple(RULE_LABELS),
                        default=list(RULE_LABELS))
    parser.add_argument("--generations", type=int, default=DEFAULT_SEARCH["generations"])
    parser.add_argument("--workers", type=int, default=DEFAULT_SEARCH["workers"])
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SETTINGS["seeds"])
    parser.add_argument("--stimulus-order", choices=STIMULUS_ORDERS,
                        default=STIMULUS_ORDER,
                        help="shuffle MEC+LEC samples within each lap or retain track order")
    parser.add_argument("--ae-parameters", type=Path, default=None,
                        help="best_parameters.json produced by preprint_ae_evolution.py")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.generations < 1 or args.workers < 1:
        raise ValueError("generations and workers must be positive")

    # Copy the few nested dictionaries before command-line changes.
    settings = dict(DEFAULT_SETTINGS)
    settings["autoencoder"] = dict(DEFAULT_SETTINGS["autoencoder"])
    settings["memory"] = dict(DEFAULT_SETTINGS["memory"])
    settings["track"] = dict(DEFAULT_SETTINGS["track"])
    settings["score_weights"] = dict(DEFAULT_SETTINGS["score_weights"])
    settings["seeds"] = list(args.seeds)
    settings["stimulus_order"] = args.stimulus_order
    load_ae_parameters(settings, args.ae_parameters)

    search = dict(DEFAULT_SEARCH)
    search["generations"] = args.generations
    search["workers"] = args.workers
    output = args.output
    if args.quick:
        settings["seeds"] = [42001]
        settings["autoencoder"]["epochs"] = 3
        settings["track"]["training_laps"] = 4
        settings["track"]["validation_laps"] = 2
        settings["masks_per_modality"] = 2
        search["generations"] = 2
        search["workers"] = 1
        if args.output == ROOT_DIR / "results/preprint/mtl_evolution":
            output = ROOT_DIR / "results/preprint/mtl_evolution_quick"

    print("Preparing matched data and frozen autoencoders...")
    prepared = prepare_seeds(settings)
    print(f"Output: {output}")
    print(f"Rules: {', '.join(args.rules)}")
    print(f"Seeds: {settings['seeds']}")
    print(f"Stimulus order: {settings['stimulus_order']}")
    print(f"Generations: {search['generations']}  population: {search['population_size']}")

    for rule in args.rules:
        print(f"\nSearching {RULE_LABELS[rule]} ({rule})")
        result = run_rule(rule, prepared, settings, search, output)
        print("Best parameters:")
        for name, value in zip(PARAMETER_NAMES, result["candidate"]):
            if name in {"ca3_inputs_per_unit", "k_ca3"}:
                print(f"  {name}: {int(round(value))}")
            else:
                print(f"  {name}: {value:.5g}")
        print(f"  composite:    {result['fitness']:.4f}")
        print(f"  clean:        {result['clean_score']:.4f}")
        print(f"  MEC degraded: {result['mec_degraded_score']:.4f}")
        print(f"  LEC degraded: {result['lec_degraded_score']:.4f}")

    print(f"\nSaved search artifacts to {output}")


if __name__ == "__main__":
    main()
