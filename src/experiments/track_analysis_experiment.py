"""Analyse EI-to-CA1 receptive fields after a cue-swap track experiment.

The model is exposed once to each position of 20 circular laps.  Two sensory
cues exchange positions after lap 10.  The frozen EI-to-CA1 encoder is then
analysed through its spatial and cue-driven tuning and CA1 units are assigned
to interpretable spatial, cue, mixed, or silent classes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core import functions  # noqa: E402
from experiments import track_experiment as track  # noqa: E402


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / \
    "track_analysis_experiment.json"
CLASS_NAMES = ("spatial", "cue", "mixed", "silent")
CLASS_COLORS = {
    "spatial": "tab:blue",
    "cue": "tab:orange",
    "mixed": "tab:green",
    "silent": "0.55",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train on a cue-swap track and analyse EI-to-CA1 tuning."
    )
    parser.add_argument("--ae-name", default="ae_cue_nb_7")
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--silent-fraction",
        type=float,
        default=0.05,
        help=(
            "units below this fraction of the largest combined tuning "
            "strength are labelled silent"
        ),
    )
    parser.add_argument(
        "--dominance",
        type=float,
        default=0.67,
        help="modality fraction required for a spatial or cue label",
    )
    parser.add_argument("--k-ca3", type=int, default=track.DEFAULT_MTL_SETTINGS["K_ca3"])
    parser.add_argument(
        "--beta-ca3", type=float, default=track.DEFAULT_MTL_SETTINGS["beta_ca3"]
    )
    parser.add_argument(
        "--beta-ca1", type=float, default=track.DEFAULT_MTL_SETTINGS["beta_ca1"]
    )
    parser.add_argument("--alpha", type=float, default=track.DEFAULT_MTL_SETTINGS["alpha"])
    parser.add_argument(
        "--nb-ei-ca3", type=int, default=track.DEFAULT_MTL_SETTINGS["nb_ei_ca3"]
    )
    parser.add_argument(
        "--num-swaps-ca3",
        type=int,
        default=track.DEFAULT_MTL_SETTINGS["num_swaps_ca3"],
    )
    parser.add_argument(
        "--num-swaps-ca1",
        type=int,
        default=track.DEFAULT_MTL_SETTINGS["num_swaps_ca1"],
    )
    parser.add_argument(
        "--plasticity",
        choices=("base", "nois", "isout", "err1", "err2"),
        default=track.DEFAULT_MTL_SETTINGS["plasticity"],
    )
    return parser.parse_args()


def train_track_model(ae_name: str, seed: int, mtl_settings: dict,
                      quiet: bool = False):
    """Train a fresh MTL model on the prescribed 20-lap cue track."""

    data_settings = {
        **track.DEFAULT_DATA_SETTINGS,
        "num_laps": 20,
        "lap_length": 50,
        "swap_every": 10,
        "cue_positions": [10, 30],
    }
    np.random.seed(seed)
    torch.manual_seed(seed)
    stimuli, assignments = track.make_swapping_cue_track(data_settings)
    model, ae_session = track.build_mtl(ae_name, mtl_settings)
    initial_ei_ca1 = model.W_ei_ca1.detach().cpu().numpy().copy()

    for lap in tqdm(
            stimuli,
            desc="training track laps",
            disable=quiet,
            leave=False):
        track._train_lap(model, lap)

    final_ei_ca1 = model.W_ei_ca1.detach().cpu().numpy().copy()
    weight_change = float(np.max(np.abs(final_ei_ca1 - initial_ei_ca1)))
    return {
        "model": model,
        "stimuli": stimuli,
        "cue_assignments": assignments,
        "data_settings": data_settings,
        "autoencoder_session": ae_session,
        "ei_ca1_change_max_abs": weight_change,
    }


def _instructive_activity(model, stimuli: np.ndarray) -> np.ndarray:
    """Calculate CA1 instructive activity for every lap and position."""

    flat_stimuli = torch.as_tensor(
        stimuli.reshape(-1, stimuli.shape[-1]), dtype=torch.float32
    )
    with torch.no_grad():
        preactivation = flat_stimuli @ model.W_ei_ca1.detach().T
        preactivation = preactivation + model.B_ei_ca1.detach().reshape(1, -1)
        activity = functions.sparsemoid(
            preactivation,
            K=model._K_ca1,
            beta=model._beta_is,
        )
    return activity.cpu().numpy().reshape(
        stimuli.shape[0], stimuli.shape[1], -1
    )


def analyse_ei_ca1(model, stimuli: np.ndarray,
                   silent_fraction: float = 0.05,
                   dominance: float = 0.67) -> dict:
    """Extract tuning fields and assign functional EI modality classes."""

    if not 0.0 <= silent_fraction < 1.0:
        raise ValueError("silent_fraction must be in [0, 1)")
    if not 0.5 < dominance < 1.0:
        raise ValueError("dominance must be in (0.5, 1)")

    weights = model.W_ei_ca1.detach().cpu().numpy().copy()
    input_size = weights.shape[1]
    if input_size % 2:
        raise ValueError("EI input must split evenly into MEC and LEC")
    mec_size = input_size // 2
    lec_size = input_size - mec_size

    activity = _instructive_activity(model, stimuli)
    # Average repetitions of each position while retaining the two cue-swap
    # conditions in the raw activity saved to JSON.
    place_tuning = activity.mean(axis=0).T
    preferred_position = np.argmax(place_tuning, axis=1)
    peak_activity = place_tuning.max(axis=1)

    # Estimate the modality-specific linear drive supported by W_ei_ca1.
    # MEC profiles are averaged over laps to remove binomial observation noise.
    mean_mec_by_position = stimuli[:, :, :mec_size].mean(axis=0)
    spatial_drive = weights[:, :mec_size] @ mean_mec_by_position.T
    cue_patterns = track.datagen.make_cues(
        n=2, size=lec_size, fixed=True
    )
    cue_drive = weights[:, mec_size:] @ cue_patterns.T

    spatial_strength = np.ptp(spatial_drive, axis=1)
    cue_strength = np.ptp(cue_drive, axis=1)
    combined_strength = np.hypot(spatial_strength, cue_strength)
    population_peak = max(float(combined_strength.max()), 1e-12)
    silent = combined_strength <= silent_fraction * population_peak
    spatial_fraction = spatial_strength / np.maximum(
        spatial_strength + cue_strength, 1e-12
    )

    class_index = np.full(len(weights), CLASS_NAMES.index("mixed"), dtype=int)
    class_index[spatial_fraction >= dominance] = CLASS_NAMES.index("spatial")
    class_index[spatial_fraction <= 1.0 - dominance] = CLASS_NAMES.index("cue")
    class_index[silent] = CLASS_NAMES.index("silent")
    classes = np.asarray(CLASS_NAMES, dtype=object)[class_index]

    preferred_spatial_input = np.argmax(spatial_drive, axis=1)
    preferred_cue = np.argmax(cue_drive, axis=1)
    place_sort = np.lexsort((preferred_position, silent.astype(int)))
    preferred_ei_input = np.argmax(np.abs(weights), axis=1)
    class_sort = np.lexsort((preferred_ei_input, class_index))
    class_counts = {
        name: int(np.sum(classes == name)) for name in CLASS_NAMES
    }

    return {
        "ei_ca1_weights": weights,
        "instructive_activity": activity,
        "place_tuning": place_tuning,
        "spatial_drive": spatial_drive,
        "cue_drive": cue_drive,
        "spatial_strength": spatial_strength,
        "cue_strength": cue_strength,
        "combined_strength": combined_strength,
        "spatial_fraction": spatial_fraction,
        "peak_activity": peak_activity,
        "preferred_position": preferred_position,
        "preferred_spatial_input": preferred_spatial_input,
        "preferred_cue": preferred_cue,
        "preferred_ei_input": preferred_ei_input,
        "classes": classes,
        "class_index": class_index,
        "class_counts": class_counts,
        "place_sort": place_sort,
        "class_sort": class_sort,
        "mec_size": mec_size,
        "lec_size": lec_size,
        "silent_fraction": silent_fraction,
        "dominance": dominance,
    }


def analyse_cue_remapping(instructive_activity: np.ndarray,
                          swap_lap: int = 10,
                          correlation_threshold: float = 0.8,
                          error_threshold: float = 0.25) -> dict:
    """Compare CA1 tuning before and after the cue identities exchange."""

    if not 0 < swap_lap < len(instructive_activity):
        raise ValueError("swap_lap must split the recorded laps")
    before = instructive_activity[:swap_lap].mean(axis=0).T
    after = instructive_activity[swap_lap:].mean(axis=0).T
    difference = after - before

    before_centered = before - before.mean(axis=1, keepdims=True)
    after_centered = after - after.mean(axis=1, keepdims=True)
    numerator = np.sum(before_centered * after_centered, axis=1)
    denominator = np.linalg.norm(before_centered, axis=1) * np.linalg.norm(
        after_centered, axis=1
    )
    correlation = numerator / np.maximum(denominator, 1e-8)
    rmse = np.sqrt(np.mean(difference ** 2, axis=1))
    activity_range = np.maximum(
        np.maximum(np.ptp(before, axis=1), np.ptp(after, axis=1)),
        1e-8,
    )
    normalized_rmse = rmse / activity_range
    stable = (
        (correlation >= correlation_threshold)
        & (normalized_rmse <= error_threshold)
    )
    preferred_before = np.argmax(before, axis=1)
    preferred_after = np.argmax(after, axis=1)
    track_length = before.shape[1]
    absolute_shift = np.abs(preferred_after - preferred_before)
    preferred_position_shift = np.minimum(
        absolute_shift, track_length - absolute_shift
    )

    stable_candidates = np.flatnonzero(stable)
    adapting_candidates = np.flatnonzero(~stable)
    if len(stable_candidates):
        stable_examples = stable_candidates[
            np.argsort(rmse[stable_candidates])[:3]
        ]
    else:
        stable_examples = np.argsort(rmse)[:3]
    if len(adapting_candidates):
        adapting_examples = adapting_candidates[
            np.argsort(rmse[adapting_candidates])[-3:][::-1]
        ]
    else:
        adapting_examples = np.argsort(rmse)[-3:][::-1]

    return {
        "tuning_before_swap": before,
        "tuning_after_swap": after,
        "tuning_difference": difference,
        "tuning_correlation": correlation,
        "tuning_rmse": rmse,
        "normalized_rmse": normalized_rmse,
        "preferred_position_before": preferred_before,
        "preferred_position_after": preferred_after,
        "preferred_position_shift": preferred_position_shift,
        "stable": stable,
        "stable_count": int(stable.sum()),
        "adapting_count": int((~stable).sum()),
        "stable_examples": stable_examples,
        "adapting_examples": adapting_examples,
        "sort_before": np.argsort(preferred_before),
        "swap_lap": swap_lap,
        "correlation_threshold": correlation_threshold,
        "error_threshold": error_threshold,
    }


def _row_cosine(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    numerator = np.sum(first * second, axis=1)
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    return numerator / np.maximum(denominator, 1e-8)


def analyse_key_value_coupling(model, stimuli: np.ndarray,
                               lap_indices=(9, 19)) -> dict:
    """Record key-value activity and reconstruction for two cue orders."""

    if len(lap_indices) != 2:
        raise ValueError("lap_indices must contain exactly two laps")
    conditions = []
    for lap_index in lap_indices:
        if not 0 <= lap_index < len(stimuli):
            raise ValueError(f"lap index {lap_index} is outside the stimulus set")
        _, _, activity = track._recall_lap(
            model, stimuli[lap_index], return_activity=True
        )
        conditions.append(activity)

    stage_keys = ("x_ei", "ca3", "ca1", "IS", "eo")
    cross_condition_similarity = {
        key: _row_cosine(conditions[0][key], conditions[1][key])
        for key in stage_keys
    }
    ca1_target_similarity = [
        _row_cosine(condition["ca1"], condition["IS"])
        for condition in conditions
    ]
    eo_input_similarity = [
        _row_cosine(condition["eo"], condition["x_ei"])
        for condition in conditions
    ]
    return {
        "lap_indices": list(lap_indices),
        "conditions": conditions,
        "cross_condition_similarity": cross_condition_similarity,
        "ca1_target_similarity": np.asarray(ca1_target_similarity),
        "eo_input_similarity": np.asarray(eo_input_similarity),
    }


def aggregate_receptive_fields(analyses: list[dict]) -> dict:
    """Average receptive fields and retain their repetition-wise spread."""

    if not analyses:
        raise ValueError("at least one receptive-field analysis is required")
    aggregate = dict(analyses[0])
    continuous_keys = (
        "ei_ca1_weights",
        "instructive_activity",
        "place_tuning",
        "spatial_drive",
        "cue_drive",
        "spatial_strength",
        "cue_strength",
        "combined_strength",
        "spatial_fraction",
        "peak_activity",
    )
    for key in continuous_keys:
        values = np.stack([analysis[key] for analysis in analyses])
        aggregate[key] = values.mean(axis=0)
        aggregate[f"{key}_std"] = values.std(axis=0)

    spatial_strength = aggregate["spatial_strength"]
    cue_strength = aggregate["cue_strength"]
    combined_strength = np.hypot(spatial_strength, cue_strength)
    population_peak = max(float(combined_strength.max()), 1e-12)
    silent = (
        combined_strength
        <= float(aggregate["silent_fraction"]) * population_peak
    )
    spatial_fraction = spatial_strength / np.maximum(
        spatial_strength + cue_strength, 1e-12
    )
    dominance = float(aggregate["dominance"])
    class_index = np.full(
        len(spatial_strength), CLASS_NAMES.index("mixed"), dtype=int
    )
    class_index[spatial_fraction >= dominance] = CLASS_NAMES.index("spatial")
    class_index[spatial_fraction <= 1.0 - dominance] = CLASS_NAMES.index("cue")
    class_index[silent] = CLASS_NAMES.index("silent")
    classes = np.asarray(CLASS_NAMES, dtype=object)[class_index]

    aggregate["combined_strength"] = combined_strength
    aggregate["spatial_fraction"] = spatial_fraction
    aggregate["class_index"] = class_index
    aggregate["classes"] = classes
    aggregate["class_counts"] = {
        name: int(np.sum(classes == name)) for name in CLASS_NAMES
    }
    aggregate["preferred_position"] = np.argmax(
        aggregate["place_tuning"], axis=1
    )
    aggregate["preferred_spatial_input"] = np.argmax(
        aggregate["spatial_drive"], axis=1
    )
    aggregate["preferred_cue"] = np.argmax(aggregate["cue_drive"], axis=1)
    aggregate["preferred_ei_input"] = np.argmax(
        np.abs(aggregate["ei_ca1_weights"]), axis=1
    )
    aggregate["place_sort"] = np.lexsort((
        aggregate["preferred_position"], silent.astype(int)
    ))
    aggregate["class_sort"] = np.lexsort((
        aggregate["preferred_ei_input"], class_index
    ))
    aggregate["num_repetitions"] = len(analyses)
    return aggregate


def aggregate_cue_remapping(remappings: list[dict],
                            mean_activity: np.ndarray) -> dict:
    """Classify mean remapping and attach uncertainty across repetitions."""

    if not remappings:
        raise ValueError("at least one cue-remapping analysis is required")
    first = remappings[0]
    aggregate = analyse_cue_remapping(
        mean_activity,
        swap_lap=int(first["swap_lap"]),
        correlation_threshold=float(first["correlation_threshold"]),
        error_threshold=float(first["error_threshold"]),
    )
    continuous_keys = (
        "tuning_before_swap",
        "tuning_after_swap",
        "tuning_difference",
        "tuning_correlation",
        "tuning_rmse",
        "normalized_rmse",
        "preferred_position_shift",
    )
    for key in continuous_keys:
        values = np.stack([remapping[key] for remapping in remappings])
        aggregate[f"{key}_repetition_mean"] = values.mean(axis=0)
        aggregate[f"{key}_std"] = values.std(axis=0)
    aggregate["num_repetitions"] = len(remappings)
    return aggregate


def aggregate_key_value_coupling(couplings: list[dict]) -> dict:
    """Average aligned pathway activity and similarities across models."""

    if not couplings:
        raise ValueError("at least one key-value coupling analysis is required")
    stage_keys = ("x_ei", "ca3", "ca1", "IS", "eo")
    conditions = []
    conditions_std = []
    for condition_index in range(2):
        condition_mean = {}
        condition_std = {}
        for key in stage_keys:
            values = np.stack([
                coupling["conditions"][condition_index][key]
                for coupling in couplings
            ])
            condition_mean[key] = values.mean(axis=0)
            condition_std[key] = values.std(axis=0)
        conditions.append(condition_mean)
        conditions_std.append(condition_std)

    cross_condition_similarity = {}
    cross_condition_similarity_std = {}
    for key in stage_keys:
        values = np.stack([
            coupling["cross_condition_similarity"][key]
            for coupling in couplings
        ])
        cross_condition_similarity[key] = values.mean(axis=0)
        cross_condition_similarity_std[key] = values.std(axis=0)

    ca1_target = np.stack([
        coupling["ca1_target_similarity"] for coupling in couplings
    ])
    eo_input = np.stack([
        coupling["eo_input_similarity"] for coupling in couplings
    ])
    return {
        "lap_indices": list(couplings[0]["lap_indices"]),
        "conditions": conditions,
        "conditions_std": conditions_std,
        "cross_condition_similarity": cross_condition_similarity,
        "cross_condition_similarity_std": cross_condition_similarity_std,
        "ca1_target_similarity": ca1_target.mean(axis=0),
        "ca1_target_similarity_std": ca1_target.std(axis=0),
        "eo_input_similarity": eo_input.mean(axis=0),
        "eo_input_similarity_std": eo_input.std(axis=0),
        "num_repetitions": len(couplings),
    }


def _class_boundaries(sorted_class_index: np.ndarray) -> list[int]:
    return (
        np.flatnonzero(np.diff(sorted_class_index) != 0) + 1
    ).tolist()


def plot_analysis(analysis: dict, cue_positions=(10, 30)):
    """Plot sorted place tuning, sorted weights, and modality clusters."""

    figure, axes = plt.subplots(
        1, 4, figsize=(21, 5.5), constrained_layout=True
    )

    place_sort = analysis["place_sort"]
    tuning_image = axes[0].imshow(
        analysis["place_tuning"][place_sort],
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    for position in cue_positions:
        axes[0].axvline(
            position, color="white", linestyle=":", linewidth=1.2
        )
    axes[0].set(
        xlabel="position around circular track",
        ylabel="CA1 unit, sorted by preferred position",
        title="Sorted CA1 place-field tuning",
    )
    figure.colorbar(
        tuning_image, ax=axes[0], label="mean instructive activity", shrink=0.9
    )

    tuning_std_image = axes[1].imshow(
        analysis["place_tuning_std"][place_sort],
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        vmin=0.0,
        cmap="magma",
    )
    for position in cue_positions:
        axes[1].axvline(
            position, color="white", linestyle=":", linewidth=1.2
        )
    axes[1].set(
        xlabel="position around circular track",
        ylabel="same sorted CA1 unit order",
        title="Across-repetition tuning variability",
    )
    figure.colorbar(
        tuning_std_image,
        ax=axes[1],
        label="activity standard deviation",
        shrink=0.9,
    )

    class_sort = analysis["class_sort"]
    sorted_weights = analysis["ei_ca1_weights"][class_sort]
    weight_limit = max(
        float(np.percentile(np.abs(sorted_weights), 99)), 1e-9
    )
    weight_image = axes[2].imshow(
        sorted_weights,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        vmin=-weight_limit,
        vmax=weight_limit,
        cmap="coolwarm",
    )
    axes[2].axvline(
        analysis["mec_size"] - 0.5,
        color="black",
        linestyle="--",
        linewidth=1.2,
    )
    sorted_classes = analysis["class_index"][class_sort]
    for boundary in _class_boundaries(sorted_classes):
        axes[2].axhline(boundary - 0.5, color="black", linewidth=1.0)
    axes[2].set(
        xlabel="EI input unit (MEC left, LEC right)",
        ylabel="CA1 unit, grouped by functional class",
        title="EI→CA1 receptive-field weights",
    )
    figure.colorbar(
        weight_image, ax=axes[2], label="weight", shrink=0.9
    )

    for name in CLASS_NAMES:
        selected = analysis["classes"] == name
        axes[3].errorbar(
            analysis["spatial_strength"][selected],
            analysis["cue_strength"][selected],
            xerr=analysis["spatial_strength_std"][selected],
            yerr=analysis["cue_strength_std"][selected],
            fmt="none",
            ecolor=CLASS_COLORS[name],
            elinewidth=0.8,
            alpha=0.22,
        )
        axes[3].scatter(
            analysis["spatial_strength"][selected],
            analysis["cue_strength"][selected],
            s=45,
            alpha=0.8,
            color=CLASS_COLORS[name],
            edgecolor="white",
            linewidth=0.4,
            label=f"{name} (n={selected.sum()})",
        )
    maximum_strength = max(
        float(analysis["spatial_strength"].max()),
        float(analysis["cue_strength"].max()),
        1e-9,
    )
    boundary_axis = np.linspace(0.0, maximum_strength * 1.1, 200)
    dominance = float(analysis["dominance"])
    axes[3].plot(
        boundary_axis,
        boundary_axis * (1.0 - dominance) / dominance,
        color="tab:blue",
        linestyle=":",
        linewidth=1.1,
        alpha=0.7,
    )
    axes[3].plot(
        boundary_axis,
        boundary_axis * dominance / (1.0 - dominance),
        color="tab:orange",
        linestyle=":",
        linewidth=1.1,
        alpha=0.7,
    )
    silent_radius = (
        float(analysis["silent_fraction"])
        * float(analysis["combined_strength"].max())
    )
    angle = np.linspace(0.0, np.pi / 2.0, 100)
    axes[3].plot(
        silent_radius * np.cos(angle),
        silent_radius * np.sin(angle),
        color="0.4",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
    )
    axes[3].set(
        xlabel="spatial tuning strength (MEC drive range)",
        ylabel="cue tuning strength (LEC drive range)",
        title="Functional receptive-field classes",
    )
    axes[3].grid(alpha=0.2)
    axes[3].legend(loc="best")
    figure.suptitle(
        "EI→CA1 receptive fields after 20 cue-track laps | "
        f"mean ± variability across {analysis['num_repetitions']} repetitions"
    )
    return figure


def plot_cue_remapping(remapping: dict, cue_positions=(10, 30)):
    """Visualize cue-dependent CA1 remapping and invariant tuning fields."""

    figure = plt.figure(figsize=(16, 10), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=(1.1, 0.9))
    axes = np.asarray([
        [figure.add_subplot(grid[0, column]) for column in range(3)],
        [figure.add_subplot(grid[1, column]) for column in range(3)],
    ])
    sorting = remapping["sort_before"]
    before = remapping["tuning_before_swap"]
    after = remapping["tuning_after_swap"]
    difference = remapping["tuning_difference"]

    for axis, values, title in (
            (axes[0, 0], before, "Cue order A: laps 1–10"),
            (axes[0, 1], after, "Cue order B: laps 11–20")):
        image = axis.imshow(
            values[sorting],
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        for position in cue_positions:
            axis.axvline(position, color="white", linestyle=":", linewidth=1.0)
        axis.set(
            xlabel="track position",
            ylabel="same CA1 unit order",
            title=title,
        )
    figure.colorbar(
        image, ax=[axes[0, 0], axes[0, 1]], label="mean activity", shrink=0.85
    )

    difference_limit = max(
        float(np.percentile(np.abs(difference), 99)), 1e-9
    )
    difference_image = axes[0, 2].imshow(
        difference[sorting],
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        vmin=-difference_limit,
        vmax=difference_limit,
        cmap="coolwarm",
    )
    for position in cue_positions:
        axes[0, 2].axvline(
            position, color="black", linestyle=":", linewidth=1.0
        )
    axes[0, 2].set(
        xlabel="track position",
        ylabel="same CA1 unit order",
        title="Tuning change: B − A",
    )
    figure.colorbar(
        difference_image, ax=axes[0, 2], label="activity change", shrink=0.85
    )

    stable = remapping["stable"]
    for selected, color in (
            (stable, "tab:blue"), (~stable, "tab:red")):
        axes[1, 0].errorbar(
            remapping["tuning_correlation"][selected],
            remapping["normalized_rmse"][selected],
            xerr=remapping["tuning_correlation_std"][selected],
            yerr=remapping["normalized_rmse_std"][selected],
            fmt="none",
            ecolor=color,
            elinewidth=0.8,
            alpha=0.2,
        )
    axes[1, 0].scatter(
        remapping["tuning_correlation"][stable],
        remapping["normalized_rmse"][stable],
        color="tab:blue",
        alpha=0.8,
        label=f"stable (n={remapping['stable_count']})",
    )
    axes[1, 0].scatter(
        remapping["tuning_correlation"][~stable],
        remapping["normalized_rmse"][~stable],
        color="tab:red",
        alpha=0.8,
        label=f"adapting (n={remapping['adapting_count']})",
    )
    axes[1, 0].axvline(
        remapping["correlation_threshold"], color="0.4", linestyle=":"
    )
    axes[1, 0].axhline(
        remapping["error_threshold"], color="0.4", linestyle=":"
    )
    axes[1, 0].set(
        xlabel="A–B tuning correlation",
        ylabel="normalized tuning RMSE",
        title="Stable versus cue-adapting fields",
    )
    axes[1, 0].grid(alpha=0.2)
    axes[1, 0].legend(loc="best")

    positions = np.arange(before.shape[1])
    for axis, example_key, title in (
            (axes[1, 1], "stable_examples", "Example invariant neurons"),
            (axes[1, 2], "adapting_examples", "Example adapting neurons")):
        for color_index, neuron in enumerate(remapping[example_key]):
            color = f"C{color_index}"
            before_std = remapping["tuning_before_swap_std"][neuron]
            after_std = remapping["tuning_after_swap_std"][neuron]
            axis.fill_between(
                positions,
                np.clip(before[neuron] - before_std, 0.0, 1.0),
                np.clip(before[neuron] + before_std, 0.0, 1.0),
                color=color,
                alpha=0.10,
            )
            axis.fill_between(
                positions,
                np.clip(after[neuron] - after_std, 0.0, 1.0),
                np.clip(after[neuron] + after_std, 0.0, 1.0),
                color=color,
                alpha=0.07,
            )
            axis.plot(
                positions,
                before[neuron],
                color=color,
                linewidth=1.8,
                label=f"CA1 {neuron} A",
            )
            axis.plot(
                positions,
                after[neuron],
                color=color,
                linestyle="--",
                linewidth=1.5,
                label=f"CA1 {neuron} B",
            )
        for position in cue_positions:
            axis.axvline(position, color="0.5", linestyle=":", linewidth=0.9)
        axis.set(
            xlabel="track position",
            ylabel="mean CA1 activity",
            ylim=(0.0, 1.02),
            title=title,
        )
        axis.grid(alpha=0.15)
        axis.legend(fontsize=8, ncol=2, loc="best")

    figure.suptitle(
        "CA1 tuning stability and remapping when cue identities exchange "
        f"positions | {remapping['num_repetitions']} repetitions"
    )
    return figure


def plot_key_value_coupling(coupling: dict, cue_assignments: list[list[int]]):
    """Plot aligned activity through the EI→CA3→CA1 key-value pathway."""

    figure = plt.figure(figsize=(22, 9), constrained_layout=True)
    grid = figure.add_gridspec(3, 5, height_ratios=(1.0, 1.0, 0.65))
    axes = np.asarray([
        [figure.add_subplot(grid[row, column]) for column in range(5)]
        for row in range(2)
    ])
    similarity_axis = figure.add_subplot(grid[2, :])
    stages = (
        ("x_ei", "EI input pattern"),
        ("ca3", "CA3 key"),
        ("ca1", "Recalled CA1 value"),
        ("IS", "Target CA1 / instructive value"),
        ("eo", "EO reconstruction"),
    )

    image = None
    for row, (lap_index, condition) in enumerate(zip(
            coupling["lap_indices"], coupling["conditions"])):
        assignment = cue_assignments[lap_index]
        for column, (key, title) in enumerate(stages):
            values = condition[key]
            image = axes[row, column].imshow(
                values.T,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                vmin=0.0,
                vmax=1.0,
                cmap="magma",
            )
            axes[row, column].set(
                xlabel="track position",
                ylabel=f"{key} unit",
                title=(
                    f"{title}\nlap {lap_index + 1}, cues {assignment[0]}→10 / "
                    f"{assignment[1]}→30"
                ),
            )
            if key == "x_ei":
                axes[row, column].axhline(
                    values.shape[1] / 2 - 0.5,
                    color="white",
                    linestyle="--",
                    linewidth=0.9,
                )
    figure.colorbar(
        image, ax=axes.ravel().tolist(), label="activity", shrink=0.8
    )

    similarity_labels = {
        "x_ei": "EI input",
        "ca3": "CA3 key",
        "ca1": "recalled CA1",
        "IS": "target CA1 / IS",
        "eo": "EO reconstruction",
    }
    similarity_colors = {
        key: f"C{index}" for index, key in enumerate(similarity_labels)
    }
    for key, values in coupling["cross_condition_similarity"].items():
        standard_deviation = coupling[
            "cross_condition_similarity_std"
        ][key]
        positions = np.arange(len(values))
        similarity_axis.fill_between(
            positions,
            np.clip(values - standard_deviation, -1.0, 1.0),
            np.clip(values + standard_deviation, -1.0, 1.0),
            color=similarity_colors[key],
            alpha=0.10,
        )
        similarity_axis.plot(
            values,
            color=similarity_colors[key],
            linewidth=1.8,
            label=similarity_labels[key],
        )
    for condition_index, values in enumerate(
            coupling["ca1_target_similarity"]):
        similarity_axis.plot(
            values,
            color="0.15" if condition_index == 0 else "0.5",
            linestyle="--",
            linewidth=1.2,
            label=(
                "CA1↔target fidelity, lap "
                f"{coupling['lap_indices'][condition_index] + 1}"
            ),
        )
        target_std = coupling["ca1_target_similarity_std"][condition_index]
        similarity_axis.fill_between(
            np.arange(len(values)),
            np.clip(values - target_std, -1.0, 1.0),
            np.clip(values + target_std, -1.0, 1.0),
            color="0.3",
            alpha=0.06,
        )
    for condition_index, values in enumerate(coupling["eo_input_similarity"]):
        output_std = coupling["eo_input_similarity_std"][condition_index]
        color = "tab:purple" if condition_index == 0 else "tab:brown"
        similarity_axis.plot(
            values,
            color=color,
            linestyle="-.",
            linewidth=1.3,
            label=(
                "EO↔input fidelity, lap "
                f"{coupling['lap_indices'][condition_index] + 1}"
            ),
        )
        similarity_axis.fill_between(
            np.arange(len(values)),
            np.clip(values - output_std, -1.0, 1.0),
            np.clip(values + output_std, -1.0, 1.0),
            color=color,
            alpha=0.07,
        )
    similarity_axis.set(
        xlabel="track position",
        ylabel="cosine similarity between cue orders",
        ylim=(-0.05, 1.05),
        title="Where the cue-sequence change propagates through the pathway",
    )
    similarity_axis.grid(alpha=0.2)
    similarity_axis.legend(loc="lower right", ncol=4, fontsize=8)
    figure.suptitle(
        "Activity coupling: EI pattern → CA3 key → CA1 value → EO "
        f"reconstruction | mean across {coupling['num_repetitions']} repetitions"
    )
    return figure


def _json_ready(value):
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return value.tolist()
        if not np.all(np.isfinite(value)):
            raise ValueError("cannot serialize arrays containing NaN or infinity")
        if np.issubdtype(value.dtype, np.floating):
            value = np.round(value.astype(np.float64), 6)
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def save_analysis(result: dict, output: Path) -> Path:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(
            _json_ready(result),
            file,
            allow_nan=False,
            separators=(",", ":"),
        )
        file.write("\n")
    return output


def run_analysis(ae_name: str = "ae_cue_nb_5", seed: int = 3980,
                 reps: int = 5,
                 mtl_settings: dict | None = None,
                 silent_fraction: float = 0.05,
                 dominance: float = 0.67,
                 quiet: bool = False):
    if reps < 1:
        raise ValueError("reps must be at least 1")
    mtl_settings = {
        **track.DEFAULT_MTL_SETTINGS,
        **({} if mtl_settings is None else mtl_settings),
    }
    repetitions = []
    receptive_fields = []
    remappings = []
    couplings = []
    first_trained = None
    for repetition in tqdm(
            range(reps), desc="analysis repetitions", disable=quiet):
        repetition_seed = seed + repetition
        trained = train_track_model(
            ae_name=ae_name,
            seed=repetition_seed,
            mtl_settings=mtl_settings,
            quiet=True,
        )
        analysis = analyse_ei_ca1(
            model=trained["model"],
            stimuli=trained["stimuli"],
            silent_fraction=silent_fraction,
            dominance=dominance,
        )
        cue_remapping = analyse_cue_remapping(
            analysis["instructive_activity"], swap_lap=10
        )
        coupling = analyse_key_value_coupling(
            trained["model"], trained["stimuli"], lap_indices=(9, 19)
        )
        receptive_fields.append(analysis)
        remappings.append(cue_remapping)
        couplings.append(coupling)
        repetitions.append({
            "repetition": repetition,
            "seed": repetition_seed,
            "ei_ca1_change_max_abs": trained["ei_ca1_change_max_abs"],
            "analysis": analysis,
            "cue_remapping": cue_remapping,
            "key_value_coupling": coupling,
        })
        if first_trained is None:
            first_trained = trained

    aggregate_analysis = aggregate_receptive_fields(receptive_fields)
    aggregate_remapping = aggregate_cue_remapping(
        remappings, aggregate_analysis["instructive_activity"]
    )
    aggregate_coupling = aggregate_key_value_coupling(couplings)
    maximum_weight_change = max(
        repetition["ei_ca1_change_max_abs"] for repetition in repetitions
    )
    return {
        "schema_version": 2,
        "description": (
            "Repeated functional EI-to-CA1 receptive-field, cue-remapping, "
            "and key-value coupling analysis."
        ),
        "settings": {
            "ae_name": ae_name,
            "seed": seed,
            "reps": reps,
            "data": first_trained["data_settings"],
            "mtl": mtl_settings,
            "classification": {
                "silent_fraction": silent_fraction,
                "dominance": dominance,
                "class_order": list(CLASS_NAMES),
            },
        },
        "autoencoder_session": first_trained["autoencoder_session"],
        "cue_assignments": first_trained["cue_assignments"],
        "ei_ca1_change_max_abs": maximum_weight_change,
        "analysis": aggregate_analysis,
        "cue_remapping": aggregate_remapping,
        "key_value_coupling": aggregate_coupling,
        "repetitions": repetitions,
    }


def main():
    args = parse_args()
    mtl_settings = {
        **track.DEFAULT_MTL_SETTINGS,
        "K_ca3": args.k_ca3,
        "beta_ca3": args.beta_ca3,
        "beta_ca1": args.beta_ca1,
        "alpha": args.alpha,
        "nb_ei_ca3": args.nb_ei_ca3,
        "num_swaps_ca3": args.num_swaps_ca3,
        "num_swaps_ca1": args.num_swaps_ca1,
        "plasticity": args.plasticity,
    }
    result = run_analysis(
        ae_name=args.ae_name,
        seed=args.seed,
        reps=args.reps,
        mtl_settings=mtl_settings,
        silent_fraction=args.silent_fraction,
        dominance=args.dominance,
        quiet=args.quiet,
    )
    receptive_field_figure = plot_analysis(
        result["analysis"],
        cue_positions=result["settings"]["data"]["cue_positions"],
    )
    remapping_figure = plot_cue_remapping(
        result["cue_remapping"],
        cue_positions=result["settings"]["data"]["cue_positions"],
    )
    coupling_figure = plot_key_value_coupling(
        result["key_value_coupling"], result["cue_assignments"]
    )
    output = save_analysis(result, args.output)
    figure_path = (
        args.figure.expanduser().resolve()
        if args.figure is not None else output.with_suffix(".png")
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    remapping_path = figure_path.with_name(
        f"{figure_path.stem}_cue_remapping{figure_path.suffix}"
    )
    coupling_path = figure_path.with_name(
        f"{figure_path.stem}_key_value_coupling{figure_path.suffix}"
    )
    receptive_field_figure.savefig(figure_path, dpi=180)
    remapping_figure.savefig(remapping_path, dpi=180)
    coupling_figure.savefig(coupling_path, dpi=180)
    print(f"class counts: {result['analysis']['class_counts']}")
    print(
        "cue-remapping counts: "
        f"stable={result['cue_remapping']['stable_count']}, "
        f"adapting={result['cue_remapping']['adapting_count']}"
    )
    print(
        "maximum EI→CA1 weight change during MTL training: "
        f"{result['ei_ca1_change_max_abs']:.3g}"
    )
    print(f"analysis saved to {output}")
    print(f"receptive-field figure saved to {figure_path}")
    print(f"cue-remapping figure saved to {remapping_path}")
    print(f"key-value coupling figure saved to {coupling_path}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(receptive_field_figure)
        plt.close(remapping_figure)
        plt.close(coupling_figure)


if __name__ == "__main__":
    main()
