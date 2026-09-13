"""Select autoencoder hyperparameters for the preprint with CMA-ES.

This experiment answers a deliberately limited question:

    Which *effective* autoencoder hyperparameters give a useful frozen
    EC--CA1--EC encoding basis for the two input distributions used in the
    preprint?

The two distributions are:

1. unique random sparse EC patterns (used for the decoder-compatibility test),
2. MEC+LEC activity generated on the cue-rich circular track.

For every candidate, a fresh autoencoder is trained and evaluated on held-out
data in each regime.  The CMA-ES objective is the equally weighted mean of the
two validation MSE values.  The final record retains the two components, so it
is always possible to check whether an apparently good average conceals poor
performance on one of the regimes.

The script intentionally reuses the project's scientific backbone:

* ``core.models.Autoencoder`` defines the network,
* ``core.ae_tools.train_autoencoder`` performs gradient-based training,
* ``experiments.preprint_common`` generates sparse and MEC+LEC stimuli,
* ``experiments.evolution._lib`` runs the existing CMA-ES implementation.

No model weights are evolved. CMA-ES selects three hyperparameters; Adam still
learns the encoder and decoder weights in the normal way for every candidate.

Examples
--------
Run a tiny end-to-end check before launching the real search::

    python src/experiments/preprint_ae_evolution.py --quick

Run the default search using four independent evaluation workers::

    python src/experiments/preprint_ae_evolution.py --workers 4

The output directory contains JSON/CSV/NPZ source data and both individual and
combined PNG/SVG plots.  These files are intended to document parameter choice;
the best values can subsequently be copied into the Figure 2/3 settings.
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import os
import sys
from pathlib import Path
from typing import Any

# Use a non-interactive backend: this is a batch experiment and should also run
# correctly over SSH or from a terminal without a graphical display.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import MSELoss


# -----------------------------------------------------------------------------
# Imports from the existing repository
# -----------------------------------------------------------------------------

# Running ``python src/experiments/preprint_ae_evolution.py`` places
# ``src/experiments`` (not ``src``) on sys.path.  Adding src explicitly makes
# the script work from either the repository root or the experiments folder.
SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core import ae_tools, models  # noqa: E402
from experiments.evolution import _lib  # noqa: E402
from experiments.preprint_common import cue_track, sparse_patterns  # noqa: E402


# -----------------------------------------------------------------------------
# Parameters exposed to CMA-ES
# -----------------------------------------------------------------------------

# The current Autoencoder.forward applies:
#
#   CA1 = sparsemoid(W_encoder x; K_ca1, beta_ei)
#   EC' = sigmoid(beta_eo W_decoder CA1)
#
# Therefore these are the three parameters that directly affect its forward
# computation.  K_eo is intentionally absent: output sparsemoid is commented
# out in core.models.Autoencoder, so evolving K_eo would introduce a neutral
# (scientifically uninterpretable) search dimension.
PARAMETER_NAMES = (
    "K_ca1",
    "beta_latent",
    "beta_output",
)

# CMA-ES operates in an unbounded standardized coordinate system.  A latent
# genome of [0, 0, 0] decodes to the current preprint values [5, 25, 25].
# Scales say how much a one-unit CMA displacement changes each model parameter.
PARAMETER_CENTERS = np.array([5.0, 25.0, 25.0])
PARAMETER_SCALES = np.array([3.0, 20.0, 20.0])

# Broad but plausible bounds prevent degenerate sparse codes or extremely hard
# nonlinearities.  They are methodological choices and are saved in config.json.
PARAMETER_LOWER = np.array([1.0, 1.0, 1.0])
PARAMETER_UPPER = np.array([20.0, 160.0, 160.0])


# -----------------------------------------------------------------------------
# Default experiment specification
# -----------------------------------------------------------------------------

# These values mirror the dimensionality and cue-track geometry of the current
# preprint experiments.  Search-time sample counts and epochs are kept moderate:
# CMA-ES trains many networks, and its purpose here is hyperparameter selection,
# not producing the final Figure 2/3 realizations.
DEFAULT_EVALUATION = {
    "dimension": 50,
    "latent_dimension": 50,
    "active_input": 5,
    "use_bias": False,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "epochs": 64,
    "sparse_training_patterns": 1024,
    "sparse_validation_patterns": 128,
    "track_training_laps": 20,
    "track_validation_laps": 5,
    # Fixed development seeds make every candidate see the same data and the
    # same initial weights.  This reduces nuisance variance in candidate ranks.
    "development_seeds": [41001, 41002, 41003],
    # Equal weights prevent the random-pattern test or the track test from
    # silently dominating merely because it contains more individual rows.
    "sparse_weight": 0.5,
    "track_weight": 0.5,
    "track": {
        "size": 50,
        "lap_length": 50,
        "cue_positions": [10, 30],
        "cue_sigma": 4,
        "cue_beta": 40,
        "cue_alpha": 0.1,
        "mec_binarized": True,
        "mec_sigma": 5.0,
        "lec_sigma": 5.0,
        "swap_every_laps": 10,
    },
}


DEFAULT_SEARCH = {
    "generations": 32,
    # Three dimensions imply the population size used by the existing C++
    # implementation: 4 + floor(3 log(3)) = 7 candidates per generation.
    "population_size": 4 + int(3 * np.log(len(PARAMETER_NAMES))),
    # Serial execution is the safest transparent default.  --workers can make
    # candidate evaluation parallel without changing the objective.
    "workers": 1,
    "boundary_penalty": 0.1,
}


# -----------------------------------------------------------------------------
# Small deterministic utilities
# -----------------------------------------------------------------------------

def derived_seed(seed: int, stream: int) -> int:
    """Derive reproducible, separate RNG streams from one development seed.

    Data generation, model initialization, and DataLoader shuffling should not
    accidentally consume one shared random stream.  Distinct integer streams
    make that separation explicit while keeping the whole experiment repeatable.
    """

    return int((int(seed) * 1009 + int(stream) * 9176) % (2**31 - 1))


def cue_assignments(laps: int, swap_every: int) -> list[list[int]]:
    """Return the same blockwise cue-order schedule used in the preprint.

    Context 0 presents cue identities [0, 1] at the two cue locations; context
    1 swaps them to [1, 0].  Alternating blocks expose the autoencoder to both
    arrangements without giving either one more training laps.
    """

    if laps < 1:
        raise ValueError("laps must be positive")
    if swap_every < 1:
        raise ValueError("swap_every must be positive")
    return [
        [0, 1] if (lap // swap_every) % 2 == 0 else [1, 0]
        for lap in range(laps)
    ]


def sanitize_candidate(genome: np.ndarray) -> np.ndarray:
    """Decode a standardized CMA genome into valid model hyperparameters."""

    genome = np.asarray(genome, dtype=float)
    if genome.shape != PARAMETER_CENTERS.shape:
        raise ValueError(
            f"expected genome shape {PARAMETER_CENTERS.shape}, got {genome.shape}"
        )

    candidate = PARAMETER_CENTERS + PARAMETER_SCALES * genome
    candidate = np.clip(candidate, PARAMETER_LOWER, PARAMETER_UPPER)

    # K_ca1 is a count, whereas both beta values remain continuous.
    candidate[0] = np.rint(candidate[0])
    return candidate


def validate_settings(settings: dict[str, Any]) -> None:
    """Fail early when an experiment specification is internally inconsistent."""

    if settings["dimension"] != settings["track"]["size"]:
        raise ValueError("track size must equal autoencoder input dimension")
    if not 1 <= settings["active_input"] < settings["dimension"]:
        raise ValueError("active_input must be between 1 and dimension - 1")
    if settings["latent_dimension"] < 2:
        raise ValueError("latent_dimension must be at least 2")
    if settings["epochs"] < 1 or settings["batch_size"] < 1:
        raise ValueError("epochs and batch_size must be positive")
    if not settings["development_seeds"]:
        raise ValueError("at least one development seed is required")

    weight_sum = settings["sparse_weight"] + settings["track_weight"]
    if not np.isclose(weight_sum, 1.0):
        raise ValueError("sparse_weight and track_weight must sum to one")
    if settings["sparse_weight"] < 0 or settings["track_weight"] < 0:
        raise ValueError("objective weights cannot be negative")


# -----------------------------------------------------------------------------
# Data generation
# -----------------------------------------------------------------------------

def make_sparse_data(settings: dict[str, Any], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate disjoint train/validation sets of unique sparse EC patterns."""

    rng = np.random.default_rng(derived_seed(seed, 1))
    training, seen = sparse_patterns(
        count=settings["sparse_training_patterns"],
        dimension=settings["dimension"],
        active=settings["active_input"],
        rng=rng,
    )
    # Passing ``seen`` guarantees that validation contains no training pattern.
    validation, _ = sparse_patterns(
        count=settings["sparse_validation_patterns"],
        dimension=settings["dimension"],
        active=settings["active_input"],
        rng=rng,
        seen=seen,
    )
    return training, validation


def make_track_data(settings: dict[str, Any], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate held-out MEC+LEC activity from the project's track generator."""

    track = settings["track"]
    training_laps = settings["track_training_laps"]
    validation_laps = settings["track_validation_laps"]

    training = cue_track(
        laps=training_laps,
        track=track,
        assignments=cue_assignments(training_laps, track["swap_every_laps"]),
        seed=derived_seed(seed, 2),
    )

    # Validation is generated from a separate cue/MEC random stream.  A shorter
    # alternation period ensures both cue arrangements appear even when only a
    # few validation laps are requested.
    validation_swap = max(1, validation_laps // 2)
    validation = cue_track(
        laps=validation_laps,
        track=track,
        assignments=cue_assignments(validation_laps, validation_swap),
        seed=derived_seed(seed, 3),
    )

    # cue_track preserves the natural (lap, position, EC unit) organization
    # needed by the memory experiments.  Autoencoder pretraining treats every
    # position as one sample, exactly as preprint_figure_2/3 do, so flatten only
    # the first two axes here and retain the EC feature axis.
    return (
        training.reshape(-1, settings["dimension"]),
        validation.reshape(-1, settings["dimension"]),
    )


# -----------------------------------------------------------------------------
# Autoencoder training and scoring
# -----------------------------------------------------------------------------

def train_and_measure(candidate: np.ndarray, training: np.ndarray,
                      validation: np.ndarray, settings: dict[str, Any],
                      seed: int,) -> dict[str, float]:
    """ Train one candidate and return correctly aggregated validation metrics.

    We use the established project trainer but calculate final validation
    metrics in one vectorized pass.  This is both faster and avoids relying on
    the legacy per-sample ``ae_tools.testing`` aggregation.
    """

    K_ca1, beta_latent, beta_output = candidate

    # Resetting PyTorch before construction and DataLoader creation gives every
    # candidate the same initialization/shuffle stream for this development
    # seed.  Candidate comparisons therefore focus on hyperparameters.
    torch.manual_seed(derived_seed(seed, 4))
    model = models.Autoencoder(
        dim_ei=settings["dimension"],
        dim_ca1=settings["latent_dimension"],
        K_ca1=int(K_ca1),
        # K_eo is stored by the class but is inactive in forward().  Fixing it
        # to input sparsity documents intent and avoids pretending it was fit.
        K_eo=settings["active_input"],
        beta_ei=float(beta_latent),
        beta_eo=float(beta_output),
        use_bias=settings["use_bias"],
    ).cpu()

    ae_tools.train_autoencoder(
        training_data=training,
        test_data=validation,
        autoencoder=model,
        epochs=settings["epochs"],
        batch_size=settings["batch_size"],
        learning_rate=settings["learning_rate"],
        criterion=MSELoss(),
        disable=True,
        # The objective only needs the final held-out score.  Skipping periodic
        # tests avoids substantial overhead inside a population search.
        test_every=None,
        device="cpu",
    )

    model.eval()
    target = np.asarray(validation, dtype=np.float32)
    with torch.no_grad():
        target_tensor = torch.as_tensor(target, dtype=torch.float32)
        reconstruction_tensor, latent_tensor = model(target_tensor, ca1=True)
        reconstruction = reconstruction_tensor.cpu().numpy()
        latent = latent_tensor.cpu().numpy()

    mse = float(np.mean((reconstruction - target) ** 2))

    # Cosine similarity is not optimized; it is retained as an interpretable
    # secondary reconstruction measure for the report and plots.
    numerator = np.sum(reconstruction * target, axis=1)
    denominator = np.linalg.norm(reconstruction, axis=1) * np.linalg.norm(target, axis=1)
    cosine = float(np.mean(numerator / np.maximum(denominator, 1e-12)))

    # This descriptive statistic verifies that the nominal K produces a sparse
    # CA1 basis.  It is also not part of the CMA objective.
    latent_active = float(np.mean(np.sum(latent > 0.5, axis=1)))
    return {
        "mse": mse,
        "cosine": cosine,
        "latent_active_above_half": latent_active,
    }


def evaluate_candidate_detailed(candidate: np.ndarray,
                                settings: dict[str, Any],) -> dict[str, Any]:
    """ Evaluate one decoded candidate across regimes and development seeds """

    candidate = np.asarray(candidate, dtype=float)
    per_seed: list[dict[str, float]] = []

    for seed in settings["development_seeds"]:
        # The two regimes train separate autoencoders.  This matches the
        # preprint experiments, in which the frozen basis is trained for the
        # distribution needed by that simulation rather than sharing weights.
        sparse_training, sparse_validation = make_sparse_data(settings, seed)
        sparse = train_and_measure(
            candidate,
            sparse_training,
            sparse_validation,
            settings,
            seed=derived_seed(seed, 10),
        )

        track_training, track_validation = make_track_data(settings, seed)
        track = train_and_measure(
            candidate,
            track_training,
            track_validation,
            settings,
            seed=derived_seed(seed, 20),
        )

        composite = (
            settings["sparse_weight"] * sparse["mse"]
            + settings["track_weight"] * track["mse"]
        )
        per_seed.append(
            {
                "seed": int(seed),
                "sparse_mse": sparse["mse"],
                "track_mse": track["mse"],
                "composite_mse": float(composite),
                "sparse_cosine": sparse["cosine"],
                "track_cosine": track["cosine"],
                "sparse_latent_active": sparse["latent_active_above_half"],
                "track_latent_active": track["latent_active_above_half"],
            }
        )

    # CMA-ES receives only this scalar.  The complete components are recomputed
    # and saved for the winning candidate after the search.
    return {
        "fitness": float(np.mean([row["composite_mse"] for row in per_seed])),
        "sparse_mse": float(np.mean([row["sparse_mse"] for row in per_seed])),
        "track_mse": float(np.mean([row["track_mse"] for row in per_seed])),
        "per_seed": per_seed,
    }


def evaluate_candidate(candidate: np.ndarray, settings: dict[str, Any]) -> float:
    """ Picklable scalar objective used by CMA-ES worker processes """

    value = evaluate_candidate_detailed(candidate, settings)["fitness"]
    # A finite fallback gives the optimizer a poor but valid value if numerical
    # instability ever appears; ordinary runs should never take this branch.
    return float(value) if np.isfinite(value) else 1.0


def evaluate_population(population: np.ndarray, settings: dict[str, Any]) -> list[float]:
    """ Sequential population evaluator used when ``--workers 1`` is selected """

    return [evaluate_candidate(candidate, settings) for candidate in population]


# -----------------------------------------------------------------------------
# Source-data and figure output
# -----------------------------------------------------------------------------

def record_arrays(record: dict[str, Any]) -> dict[str, np.ndarray]:
    """ Convert the CMA record's lists to portable numerical NPZ arrays """

    fields = (
        "generations",
        "populations",
        "fitness",
        "optimizer_means",
        "best_candidates",
        "generation_best",
        "best_fitness",
        "population_mean_fitness",
        "sigma",
        "raw_populations",
        "raw_optimizer_means",
        "raw_best_candidates",
    )
    return {field: np.asarray(record[field]) for field in fields}


def save_csv_tables(output: Path, arrays: dict[str, np.ndarray],
                    best_details: dict[str, Any],) -> None:
    """ Write tidy tables that can be inspected without loading Python objects """

    with (output / "generation_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "generation",
                "generation_best_mse",
                "best_seen_mse",
                "population_mean_mse",
                "sigma",
                *PARAMETER_NAMES,
            ],
        )
        writer.writeheader()
        for index, generation in enumerate(arrays["generations"]):
            best = arrays["best_candidates"][index]
            writer.writerow(
                {
                    "generation": int(generation),
                    "generation_best_mse": arrays["generation_best"][index],
                    "best_seen_mse": arrays["best_fitness"][index],
                    "population_mean_mse": arrays["population_mean_fitness"][index],
                    "sigma": arrays["sigma"][index],
                    **dict(zip(PARAMETER_NAMES, best)),
                }
            )

    with (output / "candidate_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["generation", "candidate", "composite_mse", *PARAMETER_NAMES],
        )
        writer.writeheader()
        for generation_index, generation in enumerate(arrays["generations"]):
            for candidate_index, candidate in enumerate(arrays["populations"][generation_index]):
                writer.writerow(
                    {
                        "generation": int(generation),
                        "candidate": candidate_index,
                        "composite_mse": arrays["fitness"][generation_index, candidate_index],
                        **dict(zip(PARAMETER_NAMES, candidate)),
                    }
                )

    with (output / "best_validation_by_seed.csv").open("w", newline="") as handle:
        rows = best_details["per_seed"]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def style_axis(axis: plt.Axes) -> None:
    """ Apply the restrained plotting style used by the preprint scripts """

    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.18, linewidth=0.7)


def save_figure(figure: plt.Figure, output: Path, name: str) -> None:
    """ Save one panel as both editable vector and convenient raster output """

    figure.savefig(output / f"{name}.svg", bbox_inches="tight")
    figure.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def make_plots(output: Path, arrays: dict[str, np.ndarray],
               best_details: dict[str, Any],) -> None:
    """ Create three self-contained panels and one compact combined summary """

    generations = arrays["generations"]

    # Panel A: did optimization improve, and how variable was each population?
    figure_a, axis = plt.subplots(figsize=(4.4, 3.2), constrained_layout=True)
    axis.plot(generations, arrays["population_mean_fitness"], color="0.55", label="Population mean")
    axis.plot(generations, arrays["generation_best"], color="#4c78a8", label="Generation best")
    axis.plot(generations, arrays["best_fitness"], color="#e45756", linewidth=2.2, label="Best seen")
    axis.set(xlabel="CMA-ES generation", ylabel="Composite validation MSE", title="Evolution search")
    axis.legend(frameon=False, fontsize=8)
    style_axis(axis)
    save_figure(figure_a, output, "plot_ae_evolution_a")

    # Panel B: the best-so-far decoded parameters through the search.  Separate
    # axes are necessary because K is an integer count and beta values are much
    # larger continuous quantities.
    figure_b, axes = plt.subplots(1, 3, figsize=(8.4, 2.8), constrained_layout=True)
    colors = ("#59a14f", "#f28e2b", "#b07aa1")
    labels = (r"$K_{CA1}$", r"$\beta_{latent}$", r"$\beta_{output}$")
    for index, (axis, color, label) in enumerate(zip(axes, colors, labels)):
        axis.step(generations, arrays["best_candidates"][:, index], where="post", color=color, linewidth=2)
        axis.axhline(PARAMETER_CENTERS[index], color="0.6", linestyle="--", linewidth=1, label="Starting value")
        axis.set(xlabel="Generation", ylabel=label)
        style_axis(axis)
    axes[0].legend(frameon=False, fontsize=7)
    figure_b.suptitle("Best parameter values seen")
    save_figure(figure_b, output, "plot_ae_evolution_b")

    # Panel C: decompose the winning candidate across seeds and regimes.  Paired
    # dots expose whether its composite score is robust or seed-dependent.
    rows = best_details["per_seed"]
    sparse = np.asarray([row["sparse_mse"] for row in rows])
    track = np.asarray([row["track_mse"] for row in rows])
    figure_c, axis = plt.subplots(figsize=(4.0, 3.2), constrained_layout=True)
    for left, right in zip(sparse, track):
        axis.plot([0, 1], [left, right], color="0.75", linewidth=1, zorder=1)
    axis.scatter(np.zeros(len(sparse)), sparse, color="#4c78a8", s=28, zorder=2)
    axis.scatter(np.ones(len(track)), track, color="#f28e2b", s=28, zorder=2)
    axis.scatter([0, 1], [np.mean(sparse), np.mean(track)], color="black", marker="_", s=220, linewidth=2.5, zorder=3)
    axis.set(xticks=[0, 1], xticklabels=["Sparse patterns", "MEC+LEC track"],
             ylabel="Validation MSE", title="Best candidate by data regime")
    style_axis(axis)
    save_figure(figure_c, output, "plot_ae_evolution_c")

    # The combined image is convenient for lab discussion.  The individual SVG
    # panels above remain preferable when composing a manuscript in Inkscape.
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.3), constrained_layout=True)
    axes[0].plot(generations, arrays["population_mean_fitness"], color="0.55",
                 label="Population mean")
    axes[0].plot(generations, arrays["generation_best"], color="#4c78a8",
                 label="Generation best")
    axes[0].plot(generations, arrays["best_fitness"], color="#e45756",
                 linewidth=2.2, label="Best seen")
    axes[0].set(xlabel="Generation", ylabel="Composite validation MSE",
                title="A  Evolution search")
    axes[0].legend(frameon=False, fontsize=7)

    for index, (color, label) in enumerate(zip(colors, labels)):
        # Normalize to each parameter's starting value only in this combined
        # panel so all three trajectories can share one readable axis.
        axes[1].step(
            generations,
            arrays["best_candidates"][:, index] / PARAMETER_CENTERS[index],
            where="post",
            color=color,
            linewidth=1.8,
            label=label,
        )
    axes[1].axhline(1, color="0.6", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Generation", ylabel="Value / starting value",
                title="B  Selected parameters")
    axes[1].legend(frameon=False, fontsize=7)

    for left, right in zip(sparse, track):
        axes[2].plot([0, 1], [left, right], color="0.75", linewidth=1)
    axes[2].scatter(np.zeros(len(sparse)), sparse, color="#4c78a8", s=25)
    axes[2].scatter(np.ones(len(track)), track, color="#f28e2b", s=25)
    axes[2].scatter([0, 1], [np.mean(sparse), np.mean(track)], color="black",
                    marker="_", s=180, linewidth=2.3)
    axes[2].set(xticks=[0, 1], xticklabels=["Sparse", "Track"],
                ylabel="Validation MSE", title="C  Best-candidate validation")

    for axis in axes:
        style_axis(axis)
    save_figure(figure, output, "ae_evolution_summary")


def save_results(
    output: Path,
    evaluation_settings: dict[str, Any],
    search_settings: dict[str, Any],
    record: dict[str, Any],
    best_details: dict[str, Any],
) -> None:
    """Persist enough provenance to audit or redraw every result."""

    output.mkdir(parents=True, exist_ok=True)
    arrays = record_arrays(record)
    np.savez_compressed(output / "history.npz", **arrays)

    best_candidate = arrays["best_candidates"][-1]
    best_parameters = dict(zip(PARAMETER_NAMES, map(float, best_candidate)))
    best_parameters["K_ca1"] = int(round(best_parameters["K_ca1"]))
    best_parameters.update(
        {
            "composite_validation_mse": best_details["fitness"],
            "sparse_validation_mse": best_details["sparse_mse"],
            "track_validation_mse": best_details["track_mse"],
        }
    )
    (output / "best_parameters.json").write_text(
        json.dumps(best_parameters, indent=2) + "\n"
    )

    config = {
        "purpose": "Select an autoencoder encoding basis for preprint simulations",
        "objective": "0.5 * sparse validation MSE + 0.5 * MEC+LEC track validation MSE",
        "parameter_names": list(PARAMETER_NAMES),
        "parameter_centers": PARAMETER_CENTERS.tolist(),
        "parameter_scales": PARAMETER_SCALES.tolist(),
        "parameter_lower": PARAMETER_LOWER.tolist(),
        "parameter_upper": PARAMETER_UPPER.tolist(),
        "evaluation": evaluation_settings,
        "search": search_settings,
        "actual_workers": int(record["workers"]),
        "important_note": "K_eo is fixed because output sparsemoid is inactive in Autoencoder.forward",
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    save_csv_tables(output, arrays, best_details)
    make_plots(output, arrays, best_details)


# -----------------------------------------------------------------------------
# Experiment orchestration and command line
# -----------------------------------------------------------------------------

def run_search(
    evaluation_settings: dict[str, Any],
    search_settings: dict[str, Any],
    output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run CMA-ES, re-evaluate the winner, and write all artifacts."""

    validate_settings(evaluation_settings)
    number_parameters = len(PARAMETER_NAMES)

    # Bounds in latent CMA coordinates are needed only for the smooth boundary
    # penalty.  The sanitizer itself always clips decoded model parameters.
    latent_lower = (PARAMETER_LOWER - PARAMETER_CENTERS) / PARAMETER_SCALES
    latent_upper = (PARAMETER_UPPER - PARAMETER_CENTERS) / PARAMETER_SCALES

    cma_settings = {
        "num_parameters": number_parameters,
        "generations": search_settings["generations"],
        "population_size": search_settings["population_size"],
        "direction": "minimize",
        "metric_name": "composite_validation_mse",
        "workers": search_settings["workers"],
        "boundary_penalty": search_settings["boundary_penalty"],
        "latent_lower": latent_lower,
        "latent_upper": latent_upper,
        "verbose": True,
    }

    # functools.partial is picklable, so exactly the same evaluator works in a
    # serial run and in the spawn-based worker pool used by _lib.evolution_run.
    individual_evaluator = functools.partial(
        evaluate_candidate,
        settings=evaluation_settings,
    )
    population_evaluator = functools.partial(
        evaluate_population,
        settings=evaluation_settings,
    )

    record = _lib.evolution_run(
        settings=cma_settings,
        evaluate=population_evaluator,
        evaluate_individual=individual_evaluator,
        sanitizer=sanitize_candidate,
        live_plot=False,
    )

    best_candidate = np.asarray(record["best_candidates"][-1], dtype=float)
    # Re-evaluation is intentional: it produces the per-seed/per-regime table
    # needed to interpret the composite objective.  Because all seeds are fixed,
    # its scalar score should match the best score recorded by CMA-ES.
    best_details = evaluate_candidate_detailed(best_candidate, evaluation_settings)
    save_results(output, evaluation_settings, search_settings, record, best_details)
    return record, best_details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evolve effective autoencoder hyperparameters for the preprint."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "results/preprint/ae_evolution",
        help="directory for source data and PNG/SVG plots",
    )
    parser.add_argument("--generations", type=int, default=DEFAULT_SEARCH["generations"])
    parser.add_argument("--workers", type=int, default=DEFAULT_SEARCH["workers"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_EVALUATION["epochs"])
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_EVALUATION["development_seeds"],
        help="fixed development seeds averaged within every candidate score",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="tiny smoke test (2 generations, 1 seed, 3 epochs, small datasets)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.generations < 1:
        raise ValueError("--generations must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")

    # Copy nested dictionaries before applying command-line overrides.  This
    # keeps module-level defaults immutable for imports and unit-style checks.
    evaluation_settings = dict(DEFAULT_EVALUATION)
    evaluation_settings["track"] = dict(DEFAULT_EVALUATION["track"])
    evaluation_settings["epochs"] = args.epochs
    evaluation_settings["development_seeds"] = list(args.seeds)

    search_settings = dict(DEFAULT_SEARCH)
    search_settings["generations"] = args.generations
    search_settings["workers"] = args.workers

    output = args.output
    if args.quick:
        search_settings["generations"] = 2
        search_settings["workers"] = 1
        evaluation_settings.update(
            {
                "epochs": 3,
                "development_seeds": [41001],
                "sparse_training_patterns": 64,
                "sparse_validation_patterns": 24,
                "track_training_laps": 4,
                "track_validation_laps": 2,
            }
        )
        # Avoid silently overwriting a full search when --quick is used with
        # the default output path.  An explicit --output remains respected.
        if args.output == ROOT_DIR / "results/preprint/ae_evolution":
            output = ROOT_DIR / "results/preprint/ae_evolution_quick"

    print("Autoencoder preprint evolution")
    print(f"  output:      {output}")
    print(f"  generations: {search_settings['generations']}")
    print(f"  population:  {search_settings['population_size']}")
    print(f"  workers:     {search_settings['workers']}")
    print(f"  seeds:       {evaluation_settings['development_seeds']}")
    print(f"  epochs:      {evaluation_settings['epochs']}")

    record, best = run_search(evaluation_settings, search_settings, output)
    candidate = record["best_candidates"][-1]
    print("\nBest decoded parameters")
    for name, value in zip(PARAMETER_NAMES, candidate):
        formatted = str(int(round(value))) if name == "K_ca1" else f"{value:.4g}"
        print(f"  {name}: {formatted}")
    print(f"  composite validation MSE: {best['fitness']:.6f}")
    print(f"  sparse validation MSE:    {best['sparse_mse']:.6f}")
    print(f"  track validation MSE:     {best['track_mse']:.6f}")
    print(f"\nSaved source data and plots to {output}")


if __name__ == "__main__":
    main()
