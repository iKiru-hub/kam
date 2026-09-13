"""Analyse EI-to-CA1 receptive fields after a cue-swap track experiment.

The model is exposed once to each position of 20 circular laps.  Two sensory
cues exchange positions after lap 10.  The frozen EI-to-CA1 encoder is then
analysed through its spatial and cue-driven tuning and CA1 units are assigned
to interpretable spatial, cue, mixed, or silent classes.
"""

from __future__ import annotations

import argparse
import json
import secrets
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
DYNAMIC_CLASS_NAMES = ("invariant", "cue_locked", "lap_variable")
DYNAMIC_CLASS_LABELS = {
    "invariant": "invariant",
    "cue_locked": "cue-locked",
    "lap_variable": "lap-variable",
}
DYNAMIC_CLASS_COLORS = {
    "invariant": "tab:blue",
    "cue_locked": "tab:purple",
    "lap_variable": "tab:red",
}


def parse_seed(value: str) -> int | None:
    """Accept an integer seed or a request for fresh random variation."""

    if str(value).strip().lower() in {"random", "rand", "auto", "none"}:
        return None
    try:
        seed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "seed must be an integer or 'random'"
        ) from error
    if not 0 <= seed < 2 ** 32:
        raise argparse.ArgumentTypeError("seed must be between 0 and 2**32 - 1")
    return seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train on a cue-swap track and analyse EI-to-CA1 tuning."
    )
    parser.add_argument("--ae-name", default="ae_cue_nb_7")
    parser.add_argument(
        "--seed",
        type=parse_seed,
        default=40,
        help="integer seed, or 'random' to generate and report a fresh seed",
    )
    parser.add_argument(
        "--random-seed",
        action="store_true",
        help="shorthand overriding --seed with a fresh random seed",
    )
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
    parser.add_argument(
        "--change-threshold",
        type=float,
        default=0.20,
        help=(
            "cosine-distance threshold separating stable from remapping "
            "CA1 fields (default: 0.20, equivalent to similarity 0.80)"
        ),
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


def analyse_remapping_dynamics(instructive_activity: np.ndarray,
                               swap_lap: int = 10) -> dict:
    """Resolve CA1 field similarity and remapping separately for every lap."""

    if not 0 < swap_lap < len(instructive_activity):
        raise ValueError("swap_lap must split the recorded laps")
    # [lap, neuron, position]
    fields = np.transpose(instructive_activity, (0, 2, 1))
    num_laps, num_neurons, track_length = fields.shape
    flattened = fields.reshape(num_laps, -1)
    flat_norms = np.linalg.norm(flattened, axis=1)
    lap_similarity = flattened @ flattened.T / np.maximum(
        flat_norms[:, None] * flat_norms[None, :], 1e-8
    )

    template_a = fields[:swap_lap].mean(axis=0)
    template_b = fields[swap_lap:].mean(axis=0)
    template_similarity = np.stack([
        np.asarray([
            _row_cosine(lap_fields, template).mean()
            for lap_fields in fields
        ])
        for template in (template_a, template_b)
    ])
    neuron_template_similarity_a = np.stack([
        _row_cosine(lap_fields, template_a) for lap_fields in fields
    ])
    neuron_template_similarity_b = np.stack([
        _row_cosine(lap_fields, template_b) for lap_fields in fields
    ])
    context_index = (
        neuron_template_similarity_b - neuron_template_similarity_a
    )

    consecutive_neuron_change = np.stack([
        1.0 - _row_cosine(fields[lap], fields[lap - 1])
        for lap in range(1, num_laps)
    ])
    consecutive_population_change = np.asarray([
        1.0 - _row_cosine(
            flattened[lap:lap + 1], flattened[lap - 1:lap]
        )[0]
        for lap in range(1, num_laps)
    ])
    context_shift = context_index[-1] - context_index[0]

    return {
        "lap_similarity": lap_similarity,
        "template_similarity": template_similarity,
        "neuron_template_similarity_a": neuron_template_similarity_a,
        "neuron_template_similarity_b": neuron_template_similarity_b,
        "context_index": context_index,
        "consecutive_neuron_change": consecutive_neuron_change,
        "consecutive_population_change": consecutive_population_change,
        "context_sort": np.argsort(context_shift),
        "swap_lap": swap_lap,
        "num_laps": num_laps,
        "num_neurons": num_neurons,
        "track_length": track_length,
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


def _decode_circular_population(activity: np.ndarray,
                                track_length: int) -> np.ndarray:
    """Decode position from uniformly spaced MEC output units."""

    num_units = activity.shape[-1]
    unit_angles = 2.0 * np.pi * np.arange(num_units) / num_units
    population_vector = np.sum(
        np.maximum(activity, 0.0) * np.exp(1j * unit_angles), axis=-1
    )
    angles = np.mod(np.angle(population_vector), 2.0 * np.pi)
    return angles * track_length / (2.0 * np.pi)


def analyse_decoded_track(model, stimuli: np.ndarray,
                          assignments: list[list[int]],
                          cue_positions=(10, 30),
                          lap_indices=(9, 19)) -> dict:
    """Decode circular position and cue identity from recalled CA1 activity."""

    track_length = stimuli.shape[1]
    input_size = stimuli.shape[2]
    mec_size = input_size // 2
    if input_size % 2:
        raise ValueError("decoded-track analysis expects equal MEC/LEC halves")

    cue_samples = {0: [], 1: []}
    for lap_index, lap_assignment in enumerate(assignments):
        for position_index, identity in enumerate(lap_assignment):
            cue_samples[int(identity)].append(
                stimuli[lap_index, cue_positions[position_index], mec_size:]
            )
    cue_templates = np.stack([
        np.mean(cue_samples[identity], axis=0) for identity in (0, 1)
    ])
    template_norm = np.linalg.norm(cue_templates, axis=1)

    conditions = []
    true_position = np.arange(track_length, dtype=float)
    for lap_index in lap_indices:
        _, _, activity = track._recall_lap(
            model, stimuli[lap_index], return_activity=True
        )
        eo = activity["eo"]
        decoded_position = _decode_circular_population(
            eo[:, :mec_size], track_length
        )
        absolute_error = np.abs(decoded_position - true_position)
        position_error = np.minimum(
            absolute_error, track_length - absolute_error
        )
        reconstructed_lec = eo[:, mec_size:]
        cue_similarity = (
            reconstructed_lec @ cue_templates.T
        ) / np.maximum(
            np.linalg.norm(reconstructed_lec, axis=1, keepdims=True)
            * template_norm[None, :],
            1e-8,
        )
        cue_evidence = (
            cue_similarity[:, 1] - cue_similarity[:, 0]
        ) / np.maximum(np.sum(np.abs(cue_similarity), axis=1), 1e-8)
        cue_predictions = np.argmax(cue_similarity, axis=1)
        lap_assignment = np.asarray(assignments[lap_index], dtype=int)
        predicted_at_cues = cue_predictions[np.asarray(cue_positions)]
        conditions.append({
            "lap_index": int(lap_index),
            "assignment": lap_assignment,
            "x_ei": activity["x_ei"],
            "ca1": activity["ca1"],
            "eo": eo,
            "decoded_position": decoded_position,
            "position_error": position_error,
            "cue_similarity": cue_similarity,
            "cue_evidence": cue_evidence,
            "predicted_identity_at_cues": predicted_at_cues,
            "cue_identity_accuracy": float(np.mean(
                predicted_at_cues == lap_assignment
            )),
        })
    return {
        "lap_indices": list(lap_indices),
        "conditions": conditions,
        "cue_templates": cue_templates,
        "cue_positions": list(cue_positions),
        "true_position": true_position,
        "track_length": track_length,
        "mec_size": mec_size,
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


def aggregate_remapping_dynamics(dynamics: list[dict],
                                 stable_mask: np.ndarray) -> dict:
    """Average lap-resolved remapping and summarize stable/adapting groups."""

    if not dynamics:
        raise ValueError("at least one remapping-dynamics analysis is required")
    aggregate = {
        key: dynamics[0][key]
        for key in ("swap_lap", "num_laps", "num_neurons", "track_length")
    }
    continuous_keys = (
        "lap_similarity",
        "template_similarity",
        "neuron_template_similarity_a",
        "neuron_template_similarity_b",
        "context_index",
        "consecutive_neuron_change",
        "consecutive_population_change",
    )
    for key in continuous_keys:
        values = np.stack([item[key] for item in dynamics])
        aggregate[key] = values.mean(axis=0)
        aggregate[f"{key}_std"] = values.std(axis=0)

    context_shift = (
        aggregate["context_index"][-1] - aggregate["context_index"][0]
    )
    aggregate["context_sort"] = np.argsort(context_shift)
    group_change = {}
    group_change_std = {}
    for name, selected in (
            ("stable", stable_mask), ("adapting", ~stable_mask)):
        if np.any(selected):
            repetition_values = np.stack([
                item["consecutive_neuron_change"][:, selected].mean(axis=1)
                for item in dynamics
            ])
            group_change[name] = repetition_values.mean(axis=0)
            group_change_std[name] = repetition_values.std(axis=0)
        else:
            length = int(aggregate["num_laps"]) - 1
            group_change[name] = np.zeros(length)
            group_change_std[name] = np.zeros(length)
    aggregate["group_consecutive_change"] = group_change
    aggregate["group_consecutive_change_std"] = group_change_std
    aggregate["num_repetitions"] = len(dynamics)
    return aggregate


def analyse_cue_locked_partition(dynamics: list[dict], analysis: dict,
                                 change_threshold: float = 0.20) -> dict:
    """Partition neurons by within-context stability and swap remapping.

    A cue-locked neuron is stable across consecutive laps on both sides of the
    cue swap, but crosses the change threshold at the swap itself.  Invariant
    neurons remain stable at that transition too; lap-variable neurons are not
    stable within at least one of the two same-cue blocks.
    """

    if not dynamics:
        raise ValueError("at least one remapping-dynamics analysis is required")
    if not 0.0 < change_threshold < 2.0:
        raise ValueError("change_threshold must be between 0 and 2")
    swap_lap = int(dynamics[0]["swap_lap"])
    swap_transition = swap_lap - 1

    def metrics(item: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        changes = np.asarray(item["consecutive_neuron_change"])
        before = np.median(changes[:swap_transition], axis=0)
        after = np.median(changes[swap_transition + 1:], axis=0)
        within = np.maximum(before, after)
        return before, after, changes[swap_transition]

    repetition_metrics = [metrics(item) for item in dynamics]
    before_values = np.stack([item[0] for item in repetition_metrics])
    after_values = np.stack([item[1] for item in repetition_metrics])
    swap_values = np.stack([item[2] for item in repetition_metrics])
    before_change = before_values.mean(axis=0)
    after_change = after_values.mean(axis=0)
    within_change = np.maximum(before_change, after_change)
    swap_change = swap_values.mean(axis=0)

    stable_within = within_change <= change_threshold
    cue_locked = stable_within & (swap_change > change_threshold)
    invariant = stable_within & ~cue_locked
    lap_variable = ~stable_within
    class_index = np.full(len(within_change), 2, dtype=int)
    class_index[invariant] = 0
    class_index[cue_locked] = 1
    classes = np.asarray(DYNAMIC_CLASS_NAMES, dtype=object)[class_index]

    repetition_class_index = []
    for before, after, swap in repetition_metrics:
        stable = np.maximum(before, after) <= change_threshold
        indices = np.full(len(stable), 2, dtype=int)
        indices[stable & (swap <= change_threshold)] = 0
        indices[stable & (swap > change_threshold)] = 1
        repetition_class_index.append(indices)
    repetition_class_index = np.stack(repetition_class_index)
    class_consistency = np.mean(
        repetition_class_index == class_index[None, :], axis=0
    )

    sorting = np.lexsort((within_change, swap_change, class_index))
    rf_class_index = np.asarray(analysis["class_index"], dtype=int)
    contingency = np.zeros(
        (len(DYNAMIC_CLASS_NAMES), len(CLASS_NAMES)), dtype=int
    )
    for dynamic_index in range(len(DYNAMIC_CLASS_NAMES)):
        for rf_index in range(len(CLASS_NAMES)):
            contingency[dynamic_index, rf_index] = np.sum(
                (class_index == dynamic_index) & (rf_class_index == rf_index)
            )
    contingency_fraction = contingency / np.maximum(
        contingency.sum(axis=1, keepdims=True), 1
    )

    return {
        "classes": classes,
        "class_index": class_index,
        "class_counts": {
            name: int(np.sum(classes == name))
            for name in DYNAMIC_CLASS_NAMES
        },
        "class_consistency": class_consistency,
        "same_cue_change_before": before_change,
        "same_cue_change_before_std": before_values.std(axis=0),
        "same_cue_change_after": after_change,
        "same_cue_change_after_std": after_values.std(axis=0),
        "within_context_change": within_change,
        "swap_change": swap_change,
        "swap_change_std": swap_values.std(axis=0),
        "cue_locking_index": swap_change - within_change,
        "change_threshold": change_threshold,
        "neuron_sort": sorting,
        "consecutive_neuron_change": np.mean([
            item["consecutive_neuron_change"] for item in dynamics
        ], axis=0),
        "consecutive_neuron_change_std": np.std([
            item["consecutive_neuron_change"] for item in dynamics
        ], axis=0),
        "spatial_strength": analysis["spatial_strength"],
        "cue_strength": analysis["cue_strength"],
        "spatial_fraction": analysis["spatial_fraction"],
        "rf_classes": analysis["classes"],
        "rf_class_index": rf_class_index,
        "class_by_rf_class": contingency,
        "class_by_rf_class_fraction": contingency_fraction,
        "swap_lap": swap_lap,
        "num_laps": int(dynamics[0]["num_laps"]),
        "num_neurons": int(dynamics[0]["num_neurons"]),
        "num_repetitions": len(dynamics),
    }


def analyse_ca1_activity_trajectories(analysis: dict, dynamics: dict,
                                      stable_mask: np.ndarray) -> dict:
    """Arrange mean CA1 activity as lap-resolved field trajectories."""

    # Convert [lap, position, neuron] to [lap, neuron, position].
    fields = np.transpose(analysis["instructive_activity"], (0, 2, 1))
    fields_std = np.transpose(
        analysis["instructive_activity_std"], (0, 2, 1)
    )
    neuron_sort = np.asarray(dynamics["context_sort"], dtype=int)
    preferred_position = np.argmax(fields, axis=2)
    peak_activity = fields.max(axis=2)
    mean_activity = fields.mean(axis=2)
    num_laps, num_neurons, track_length = fields.shape
    return {
        "fields": fields,
        "fields_std": fields_std,
        "preferred_position": preferred_position,
        "peak_activity": peak_activity,
        "mean_activity": mean_activity,
        "neuron_sort": neuron_sort,
        "stable": np.asarray(stable_mask, dtype=bool),
        "activity_carpet": fields[:, neuron_sort, :].reshape(
            num_laps * num_neurons, track_length
        ),
        "activity_carpet_std": fields_std[:, neuron_sort, :].reshape(
            num_laps * num_neurons, track_length
        ),
        "num_laps": num_laps,
        "num_neurons": num_neurons,
        "track_length": track_length,
        "swap_lap": int(dynamics["swap_lap"]),
        "num_repetitions": int(analysis["num_repetitions"]),
    }


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


def aggregate_decoded_tracks(decoded_tracks: list[dict]) -> dict:
    """Aggregate decoded tracks, respecting circular decoded positions."""

    if not decoded_tracks:
        raise ValueError("at least one decoded-track analysis is required")
    first = decoded_tracks[0]
    track_length = int(first["track_length"])
    conditions = []
    for condition_index in range(len(first["conditions"])):
        items = [
            decoded["conditions"][condition_index]
            for decoded in decoded_tracks
        ]
        positions = np.stack([item["decoded_position"] for item in items])
        angles = positions * (2.0 * np.pi / track_length)
        mean_vector = np.mean(np.exp(1j * angles), axis=0)
        decoded_position = (
            np.mod(np.angle(mean_vector), 2.0 * np.pi)
            * track_length / (2.0 * np.pi)
        )
        conditions.append({
            "lap_index": int(items[0]["lap_index"]),
            "assignment": items[0]["assignment"],
            "x_ei": np.mean([item["x_ei"] for item in items], axis=0),
            "x_ei_std": np.std([item["x_ei"] for item in items], axis=0),
            "ca1": np.mean([item["ca1"] for item in items], axis=0),
            "ca1_std": np.std([item["ca1"] for item in items], axis=0),
            "eo": np.mean([item["eo"] for item in items], axis=0),
            "eo_std": np.std([item["eo"] for item in items], axis=0),
            "decoded_position": decoded_position,
            "decoded_position_resultant": np.abs(mean_vector),
            "position_error": np.mean(
                [item["position_error"] for item in items], axis=0
            ),
            "position_error_std": np.std(
                [item["position_error"] for item in items], axis=0
            ),
            "cue_similarity": np.mean(
                [item["cue_similarity"] for item in items], axis=0
            ),
            "cue_similarity_std": np.std(
                [item["cue_similarity"] for item in items], axis=0
            ),
            "cue_evidence": np.mean(
                [item["cue_evidence"] for item in items], axis=0
            ),
            "cue_evidence_std": np.std(
                [item["cue_evidence"] for item in items], axis=0
            ),
            "cue_identity_accuracy": float(np.mean([
                item["cue_identity_accuracy"] for item in items
            ])),
        })
    return {
        "lap_indices": list(first["lap_indices"]),
        "conditions": conditions,
        "cue_templates": np.mean(
            [item["cue_templates"] for item in decoded_tracks], axis=0
        ),
        "cue_positions": list(first["cue_positions"]),
        "true_position": first["true_position"],
        "track_length": track_length,
        "mec_size": int(first["mec_size"]),
        "num_repetitions": len(decoded_tracks),
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
    invariant_axis = axes[1, 1]
    stable_indices = np.flatnonzero(stable)
    invariant_colors = plt.cm.viridis(
        np.linspace(0.05, 0.95, max(len(stable_indices), 1))
    )
    for neuron, color in zip(stable_indices, invariant_colors):
        invariant_axis.plot(
            positions, before[neuron], color=color, linewidth=0.9, alpha=0.55
        )
        invariant_axis.plot(
            positions,
            after[neuron],
            color=color,
            linestyle="--",
            linewidth=0.8,
            alpha=0.55,
        )
    if len(stable_indices):
        invariant_axis.plot(
            positions,
            before[stable_indices].mean(axis=0),
            color="black",
            linewidth=2.2,
            label="invariant mean A",
        )
        invariant_axis.plot(
            positions,
            after[stable_indices].mean(axis=0),
            color="black",
            linestyle="--",
            linewidth=2.0,
            label="invariant mean B",
        )
    invariant_axis.set_title(f"All invariant neurons (n={len(stable_indices)})")
    invariant_axis.legend(fontsize=8, loc="best")

    adapting_axis = axes[1, 2]
    for color_index, neuron in enumerate(remapping["adapting_examples"]):
        color = f"C{color_index}"
        before_std = remapping["tuning_before_swap_std"][neuron]
        after_std = remapping["tuning_after_swap_std"][neuron]
        adapting_axis.fill_between(
            positions,
            np.clip(before[neuron] - before_std, 0.0, 1.0),
            np.clip(before[neuron] + before_std, 0.0, 1.0),
            color=color,
            alpha=0.10,
        )
        adapting_axis.fill_between(
            positions,
            np.clip(after[neuron] - after_std, 0.0, 1.0),
            np.clip(after[neuron] + after_std, 0.0, 1.0),
            color=color,
            alpha=0.07,
        )
        adapting_axis.plot(
            positions,
            before[neuron],
            color=color,
            linewidth=1.8,
            label=f"CA1 {neuron} A",
        )
        adapting_axis.plot(
            positions,
            after[neuron],
            color=color,
            linestyle="--",
            linewidth=1.5,
            label=f"CA1 {neuron} B",
        )
    adapting_axis.set_title("Example adapting neurons")
    adapting_axis.legend(fontsize=8, ncol=2, loc="best")

    for axis in (invariant_axis, adapting_axis):
        for position in cue_positions:
            axis.axvline(position, color="0.5", linestyle=":", linewidth=0.9)
        axis.set(
            xlabel="track position",
            ylabel="mean CA1 activity",
            ylim=(0.0, 1.02),
        )
        axis.grid(alpha=0.15)

    figure.suptitle(
        "CA1 tuning stability and remapping when cue identities exchange "
        f"positions | {remapping['num_repetitions']} repetitions"
    )
    return figure


def plot_remapping_dynamics(dynamics: dict):
    """Plot the lap-by-lap emergence of opposite-cue CA1 tuning states."""

    figure, axes = plt.subplots(
        2, 2, figsize=(14, 10), constrained_layout=True
    )
    num_laps = int(dynamics["num_laps"])
    swap_lap = int(dynamics["swap_lap"])
    lap_numbers = np.arange(1, num_laps + 1)
    transition_laps = np.arange(2, num_laps + 1)

    similarity_image = axes[0, 0].imshow(
        dynamics["lap_similarity"],
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
        extent=(0.5, num_laps + 0.5, 0.5, num_laps + 0.5),
    )
    axes[0, 0].axvline(swap_lap + 0.5, color="white", linestyle="--")
    axes[0, 0].axhline(swap_lap + 0.5, color="white", linestyle="--")
    axes[0, 0].set(
        xlabel="lap",
        ylabel="lap",
        title="Population tuning similarity between laps",
    )
    figure.colorbar(
        similarity_image, ax=axes[0, 0], label="cosine similarity", shrink=0.9
    )

    template_colors = ("tab:blue", "tab:orange")
    template_labels = ("cue-order A template", "cue-order B template")
    for template_index, (color, label) in enumerate(zip(
            template_colors, template_labels)):
        values = dynamics["template_similarity"][template_index]
        deviation = dynamics["template_similarity_std"][template_index]
        axes[0, 1].plot(
            lap_numbers, values, color=color, linewidth=2.0, label=label
        )
        axes[0, 1].fill_between(
            lap_numbers,
            np.clip(values - deviation, -1.0, 1.0),
            np.clip(values + deviation, -1.0, 1.0),
            color=color,
            alpha=0.16,
        )
    axes[0, 1].axvline(
        swap_lap + 0.5, color="0.35", linestyle="--", label="cue swap"
    )
    axes[0, 1].set(
        xlabel="lap",
        ylabel="similarity to context template",
        ylim=(-0.05, 1.05),
        title="Transition between opposite-cue tuning states",
    )
    axes[0, 1].grid(alpha=0.2)
    axes[0, 1].legend(loc="best")

    sorting = dynamics["context_sort"]
    context_values = dynamics["context_index"][:, sorting].T
    context_limit = max(
        float(np.percentile(np.abs(context_values), 99)), 1e-9
    )
    context_image = axes[1, 0].imshow(
        context_values,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        vmin=-context_limit,
        vmax=context_limit,
        cmap="coolwarm",
        extent=(0.5, num_laps + 0.5, 0, context_values.shape[0]),
    )
    axes[1, 0].axvline(swap_lap + 0.5, color="black", linestyle="--")
    axes[1, 0].set(
        xlabel="lap",
        ylabel="CA1 unit, sorted by context shift",
        title="Neuron-wise cue-context index (B similarity − A similarity)",
    )
    figure.colorbar(
        context_image, ax=axes[1, 0], label="context preference", shrink=0.9
    )

    population = dynamics["consecutive_population_change"]
    population_std = dynamics["consecutive_population_change_std"]
    axes[1, 1].plot(
        transition_laps,
        population,
        color="black",
        linewidth=2.1,
        label="whole population",
    )
    axes[1, 1].fill_between(
        transition_laps,
        np.clip(population - population_std, 0.0, None),
        population + population_std,
        color="black",
        alpha=0.10,
    )
    for name, color in (("stable", "tab:blue"), ("adapting", "tab:red")):
        values = dynamics["group_consecutive_change"][name]
        deviation = dynamics["group_consecutive_change_std"][name]
        axes[1, 1].plot(
            transition_laps,
            values,
            color=color,
            linewidth=1.8,
            label=f"{name} neurons",
        )
        axes[1, 1].fill_between(
            transition_laps,
            np.clip(values - deviation, 0.0, None),
            values + deviation,
            color=color,
            alpha=0.13,
        )
    axes[1, 1].axvline(
        swap_lap + 1,
        color="0.35",
        linestyle="--",
        label="first opposite-cue lap",
    )
    axes[1, 1].set(
        xlabel="new lap",
        ylabel="change from preceding lap (1 − cosine)",
        title="Remapping introduced at each lap transition",
    )
    axes[1, 1].grid(alpha=0.2)
    axes[1, 1].legend(loc="best")

    figure.suptitle(
        "Lap-resolved CA1 tuning-field dynamics | "
        f"mean ± variability across {dynamics['num_repetitions']} repetitions"
    )
    return figure


def plot_cue_locked_partition(partition: dict):
    """Show cue-correlated remapping classes and their EI sensitivity."""

    figure, axes = plt.subplots(
        2, 3, figsize=(18, 10), constrained_layout=True
    )
    threshold = float(partition["change_threshold"])
    class_index = np.asarray(partition["class_index"])
    num_laps = int(partition["num_laps"])
    swap_lap = int(partition["swap_lap"])
    transition_laps = np.arange(2, num_laps + 1)

    for index, name in enumerate(DYNAMIC_CLASS_NAMES):
        selected = class_index == index
        if not np.any(selected):
            continue
        axes[0, 0].scatter(
            partition["within_context_change"][selected],
            partition["swap_change"][selected],
            s=34,
            alpha=0.75,
            color=DYNAMIC_CLASS_COLORS[name],
            label=(
                f"{DYNAMIC_CLASS_LABELS[name]} "
                f"(n={partition['class_counts'][name]})"
            ),
        )
    axes[0, 0].axvline(threshold, color="0.3", linestyle="--")
    axes[0, 0].axhline(threshold, color="0.3", linestyle="--")
    limit = max(
        threshold * 1.25,
        float(np.max(partition["within_context_change"])),
        float(np.max(partition["swap_change"])),
    ) * 1.08
    axes[0, 0].plot([0, limit], [0, limit], color="0.7", linestyle=":")
    axes[0, 0].set(
        xlabel="same-cue lap variability (worst context)",
        ylabel="change at cue swap",
        xlim=(-0.02, limit),
        ylim=(-0.02, limit),
        title="Cue-locking partition",
    )
    axes[0, 0].grid(alpha=0.15)
    axes[0, 0].legend(loc="best", fontsize=8)

    sorting = np.asarray(partition["neuron_sort"])
    changes = partition["consecutive_neuron_change"][:, sorting].T
    change_limit = max(float(np.percentile(changes, 99)), threshold, 1e-8)
    change_image = axes[0, 1].imshow(
        changes,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        vmin=0.0,
        vmax=change_limit,
        cmap="magma",
        extent=(1.5, num_laps + 0.5, 0, len(sorting)),
    )
    axes[0, 1].axvline(swap_lap + 0.5, color="cyan", linestyle="--")
    sorted_classes = class_index[sorting]
    for boundary in _class_boundaries(sorted_classes):
        axes[0, 1].axhline(boundary, color="white", linewidth=1.0)
    axes[0, 1].set(
        xlabel="new lap",
        ylabel="CA1 neuron, grouped by dynamic class",
        title="When each neuron remaps",
    )
    figure.colorbar(
        change_image, ax=axes[0, 1], label="1 − field cosine similarity"
    )

    for index, name in enumerate(DYNAMIC_CLASS_NAMES):
        selected = class_index == index
        if not np.any(selected):
            continue
        values = partition["consecutive_neuron_change"][:, selected].mean(axis=1)
        deviation = partition["consecutive_neuron_change"][:, selected].std(axis=1)
        color = DYNAMIC_CLASS_COLORS[name]
        axes[0, 2].plot(
            transition_laps, values, color=color, linewidth=2,
            label=DYNAMIC_CLASS_LABELS[name],
        )
        axes[0, 2].fill_between(
            transition_laps,
            np.clip(values - deviation, 0, None),
            values + deviation,
            color=color,
            alpha=0.14,
        )
    axes[0, 2].axvline(
        swap_lap + 1, color="0.3", linestyle="--", label="cue swap"
    )
    axes[0, 2].set(
        xlabel="new lap",
        ylabel="change from preceding lap",
        title="Class-average remapping trajectory",
    )
    axes[0, 2].grid(alpha=0.2)
    axes[0, 2].legend(loc="best", fontsize=8)

    for index, name in enumerate(DYNAMIC_CLASS_NAMES):
        selected = class_index == index
        axes[1, 0].scatter(
            partition["spatial_strength"][selected],
            partition["cue_strength"][selected],
            color=DYNAMIC_CLASS_COLORS[name],
            alpha=0.72,
            s=36,
            label=DYNAMIC_CLASS_LABELS[name],
        )
    axes[1, 0].set(
        xlabel="spatial sensitivity (MEC drive range)",
        ylabel="cue sensitivity (LEC drive range)",
        title="Dynamic class versus EI modality sensitivity",
    )
    axes[1, 0].grid(alpha=0.15)
    axes[1, 0].legend(loc="best", fontsize=8)

    fraction_groups = [
        partition["spatial_fraction"][class_index == index]
        for index in range(len(DYNAMIC_CLASS_NAMES))
    ]
    nonempty = [(index, values) for index, values in enumerate(fraction_groups)
                if len(values)]
    if nonempty:
        positions = [index + 1 for index, _ in nonempty]
        boxes = axes[1, 1].boxplot(
            [values for _, values in nonempty],
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
        )
        for box, (index, _) in zip(boxes["boxes"], nonempty):
            box.set_facecolor(DYNAMIC_CLASS_COLORS[DYNAMIC_CLASS_NAMES[index]])
            box.set_alpha(0.45)
    for index, values in enumerate(fraction_groups):
        if len(values):
            offsets = np.linspace(-0.16, 0.16, len(values))
            axes[1, 1].scatter(
                index + 1 + offsets,
                values,
                color=DYNAMIC_CLASS_COLORS[DYNAMIC_CLASS_NAMES[index]],
                s=17,
                alpha=0.55,
            )
    axes[1, 1].axhline(0.5, color="0.5", linestyle=":")
    axes[1, 1].set_xticks(
        np.arange(1, len(DYNAMIC_CLASS_NAMES) + 1),
        [DYNAMIC_CLASS_LABELS[name] for name in DYNAMIC_CLASS_NAMES],
        rotation=12,
    )
    axes[1, 1].set(
        ylabel="spatial fraction (spatial / [spatial + cue])",
        ylim=(-0.03, 1.03),
        title="Spatial–cue balance by dynamic class",
    )
    axes[1, 1].grid(axis="y", alpha=0.15)

    contingency = partition["class_by_rf_class_fraction"]
    contingency_image = axes[1, 2].imshow(
        contingency,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
        cmap="Blues",
    )
    counts = partition["class_by_rf_class"]
    for row in range(contingency.shape[0]):
        for column in range(contingency.shape[1]):
            axes[1, 2].text(
                column,
                row,
                f"{counts[row, column]}\n{contingency[row, column]:.0%}",
                ha="center",
                va="center",
                color="white" if contingency[row, column] > 0.55 else "black",
                fontsize=9,
            )
    axes[1, 2].set_xticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES)
    axes[1, 2].set_yticks(
        np.arange(len(DYNAMIC_CLASS_NAMES)),
        [DYNAMIC_CLASS_LABELS[name] for name in DYNAMIC_CLASS_NAMES],
    )
    axes[1, 2].set(
        xlabel="spatial/cue receptive-field class",
        ylabel="dynamic class",
        title="Cross-classification (row-normalized)",
    )
    figure.colorbar(
        contingency_image, ax=axes[1, 2], label="fraction of dynamic class"
    )

    figure.suptitle(
        "CA1 remapping correlated with cue identity | "
        f"mean across {partition['num_repetitions']} repetitions"
    )
    return figure


def plot_ca1_activity_trajectories(trajectories: dict, cue_positions=(10, 30)):
    """Plot raw CA1 fields and preferred-position trajectories over laps."""

    figure, axes = plt.subplots(
        2, 2, figsize=(15, 10), constrained_layout=True
    )
    num_laps = int(trajectories["num_laps"])
    num_neurons = int(trajectories["num_neurons"])
    track_length = int(trajectories["track_length"])
    swap_lap = int(trajectories["swap_lap"])
    neuron_sort = trajectories["neuron_sort"]
    laps = np.arange(1, num_laps + 1)

    carpet_image = axes[0, 0].imshow(
        trajectories["activity_carpet"],
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
        extent=(0, track_length, 0, num_laps * num_neurons),
    )
    axes[0, 0].axhline(
        swap_lap * num_neurons,
        color="white",
        linestyle="--",
        linewidth=1.3,
    )
    lap_ticks = np.arange(0, num_laps, 2)
    axes[0, 0].set_yticks((lap_ticks + 0.5) * num_neurons)
    axes[0, 0].set_yticklabels(lap_ticks + 1)
    for position in cue_positions:
        axes[0, 0].axvline(position, color="white", linestyle=":", linewidth=0.8)
    axes[0, 0].set(
        xlabel="track position",
        ylabel="lap block (each block contains every CA1 neuron)",
        title="CA1 tuning-field activity through time",
    )
    figure.colorbar(
        carpet_image, ax=axes[0, 0], label="mean CA1 activity", shrink=0.9
    )

    preferred = trajectories["preferred_position"][:, neuron_sort].T
    preferred_image = axes[0, 1].imshow(
        preferred,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=track_length,
        cmap="twilight",
        extent=(0.5, num_laps + 0.5, 0, num_neurons),
    )
    axes[0, 1].axvline(swap_lap + 0.5, color="white", linestyle="--")
    axes[0, 1].set(
        xlabel="lap",
        ylabel="CA1 neuron, sorted by cue-context shift",
        title="Preferred-position matrix",
    )
    figure.colorbar(
        preferred_image, ax=axes[0, 1], label="preferred track position", shrink=0.9
    )

    peak_activity = trajectories["peak_activity"][:, neuron_sort].T
    peak_image = axes[1, 0].imshow(
        peak_activity,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
        cmap="magma",
        extent=(0.5, num_laps + 0.5, 0, num_neurons),
    )
    axes[1, 0].axvline(swap_lap + 0.5, color="white", linestyle="--")
    axes[1, 0].set(
        xlabel="lap",
        ylabel="same sorted CA1 neuron order",
        title="Peak field activity over laps",
    )
    figure.colorbar(
        peak_image, ax=axes[1, 0], label="peak activity", shrink=0.9
    )

    stable = trajectories["stable"]
    preferred_unsorted = trajectories["preferred_position"]
    for neuron in range(num_neurons):
        color = "tab:blue" if stable[neuron] else "tab:red"
        axes[1, 1].plot(
            laps,
            preferred_unsorted[:, neuron],
            color=color,
            linewidth=0.8,
            alpha=0.35,
        )
    axes[1, 1].plot([], [], color="tab:blue", label="invariant neurons")
    axes[1, 1].plot([], [], color="tab:red", label="adapting neurons")
    axes[1, 1].axvline(
        swap_lap + 0.5, color="0.25", linestyle="--", label="cue swap"
    )
    for position in cue_positions:
        axes[1, 1].axhline(position, color="0.5", linestyle=":", linewidth=0.9)
    axes[1, 1].set(
        xlabel="lap",
        ylabel="preferred track position",
        xlim=(1, num_laps),
        ylim=(0, track_length - 1),
        title="Individual CA1 field trajectories",
    )
    axes[1, 1].grid(alpha=0.15)
    axes[1, 1].legend(loc="best")

    figure.suptitle(
        "CA1 activity and receptive-field trajectories across laps | "
        f"mean across {trajectories['num_repetitions']} repetitions"
    )
    return figure


def plot_decoded_track(decoded: dict):
    """Plot CA1→EO reconstruction as circular position plus cue identity."""

    figure = plt.figure(figsize=(16, 9), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.25, 1.0))
    axis_3d = figure.add_subplot(grid[:, 0], projection="3d")
    position_axis = figure.add_subplot(grid[0, 1])
    cue_axis = figure.add_subplot(grid[1, 1])
    track_length = int(decoded["track_length"])
    true_position = np.asarray(decoded["true_position"])
    true_angle = 2.0 * np.pi * true_position / track_length
    cue_positions = np.asarray(decoded["cue_positions"], dtype=int)
    identity_colors = ("tab:cyan", "tab:pink")
    condition_colors = ("tab:blue", "tab:orange")

    ring_angle = np.linspace(0, 2.0 * np.pi, 400)
    axis_3d.plot(
        np.cos(ring_angle), np.sin(ring_angle), np.zeros_like(ring_angle),
        color="0.65", linestyle=":", linewidth=1.4, label="physical track",
    )
    for condition_index, condition in enumerate(decoded["conditions"]):
        color = condition_colors[condition_index % len(condition_colors)]
        decoded_angle = (
            2.0 * np.pi * condition["decoded_position"] / track_length
        )
        x = np.cos(decoded_angle)
        y = np.sin(decoded_angle)
        z = condition["cue_evidence"]
        label = (
            f"lap {condition['lap_index'] + 1}, cues "
            f"{np.asarray(condition['assignment'], dtype=int).tolist()}"
        )
        axis_3d.plot(x, y, z, color=color, linewidth=2.0, label=label)
        axis_3d.scatter(x, y, z, color=color, s=11, alpha=0.55)
        for slot, cue_position in enumerate(cue_positions):
            identity = int(condition["assignment"][slot])
            axis_3d.scatter(
                x[cue_position], y[cue_position], z[cue_position],
                color=identity_colors[identity], edgecolor="black",
                marker="*", s=180, depthshade=False,
            )
            axis_3d.text(
                x[cue_position], y[cue_position], z[cue_position] + 0.05,
                f"cue {identity}", fontsize=8,
            )
    axis_3d.set(
        xlabel="decoded track x",
        ylabel="decoded track y",
        zlabel="decoded cue evidence (cue 1 − cue 0)",
        title="CA1-decoded track and cue identity",
        xlim=(-1.15, 1.15),
        ylim=(-1.15, 1.15),
    )
    axis_3d.view_init(elev=25, azim=-55)
    axis_3d.legend(loc="upper left", fontsize=8)

    for condition_index, condition in enumerate(decoded["conditions"]):
        color = condition_colors[condition_index % len(condition_colors)]
        mae = float(np.mean(condition["position_error"]))
        position_axis.scatter(
            true_position,
            condition["decoded_position"],
            color=color,
            s=22,
            alpha=0.70,
            label=f"lap {condition['lap_index'] + 1} (circular MAE={mae:.2f})",
        )
    position_axis.plot(
        [0, track_length], [0, track_length], color="0.35", linestyle="--",
        label="exact position",
    )
    for cue_position in cue_positions:
        position_axis.axvline(cue_position, color="0.65", linestyle=":")
    position_axis.set(
        xlabel="physical track position",
        ylabel="position decoded from reconstructed MEC",
        xlim=(-1, track_length),
        ylim=(-1, track_length),
        title="Spatial reconstruction from decoded CA1",
    )
    position_axis.grid(alpha=0.15)
    position_axis.legend(loc="best", fontsize=8)

    for condition_index, condition in enumerate(decoded["conditions"]):
        color = condition_colors[condition_index % len(condition_colors)]
        evidence = condition["cue_evidence"]
        deviation = condition["cue_evidence_std"]
        cue_axis.plot(
            true_position, evidence, color=color, linewidth=2.0,
            label=(
                f"lap {condition['lap_index'] + 1}; cue accuracy "
                f"{condition['cue_identity_accuracy']:.0%}"
            ),
        )
        cue_axis.fill_between(
            true_position, evidence - deviation, evidence + deviation,
            color=color, alpha=0.14,
        )
        for slot, cue_position in enumerate(cue_positions):
            identity = int(condition["assignment"][slot])
            cue_axis.scatter(
                cue_position, evidence[cue_position],
                color=identity_colors[identity], edgecolor="black",
                marker="*", s=145, zorder=5,
            )
            cue_axis.annotate(
                f"cue {identity}",
                (cue_position, evidence[cue_position]),
                xytext=(4, 7), textcoords="offset points", fontsize=8,
            )
    cue_axis.axhline(0.0, color="0.4", linestyle="--", linewidth=1.0)
    for cue_position in cue_positions:
        cue_axis.axvline(cue_position, color="0.65", linestyle=":")
    cue_axis.set(
        xlabel="physical track position",
        ylabel="LEC cue evidence (negative: cue 0; positive: cue 1)",
        xlim=(0, track_length - 1),
        title="Reconstructed cue identity along the track",
    )
    cue_axis.grid(alpha=0.15)
    cue_axis.legend(loc="best", fontsize=8)

    figure.suptitle(
        "Circular-track reconstruction decoded from CA1 activity | "
        f"mean ± variability across {decoded['num_repetitions']} repetitions"
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


def run_analysis(ae_name: str = "ae_cue_nb_5", seed: int | None = 3980,
                 reps: int = 5,
                 mtl_settings: dict | None = None,
                 silent_fraction: float = 0.05,
                 dominance: float = 0.67,
                 change_threshold: float = 0.20,
                 quiet: bool = False):
    if reps < 1:
        raise ValueError("reps must be at least 1")
    resolved_seed = secrets.randbits(32) if seed is None else int(seed)
    mtl_settings = {
        **track.DEFAULT_MTL_SETTINGS,
        **({} if mtl_settings is None else mtl_settings),
    }
    repetitions = []
    receptive_fields = []
    remappings = []
    remapping_dynamics = []
    couplings = []
    decoded_tracks = []
    first_trained = None
    for repetition in tqdm(
            range(reps), desc="analysis repetitions", disable=quiet):
        repetition_seed = (resolved_seed + repetition) % (2 ** 32)
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
        dynamics = analyse_remapping_dynamics(
            analysis["instructive_activity"], swap_lap=10
        )
        coupling = analyse_key_value_coupling(
            trained["model"], trained["stimuli"], lap_indices=(9, 19)
        )
        decoded_track = analyse_decoded_track(
            trained["model"],
            trained["stimuli"],
            trained["cue_assignments"],
            cue_positions=trained["data_settings"]["cue_positions"],
            lap_indices=(9, 19),
        )
        cue_locked_partition = analyse_cue_locked_partition(
            [dynamics], analysis, change_threshold=change_threshold
        )
        receptive_fields.append(analysis)
        remappings.append(cue_remapping)
        remapping_dynamics.append(dynamics)
        couplings.append(coupling)
        decoded_tracks.append(decoded_track)
        repetitions.append({
            "repetition": repetition,
            "seed": repetition_seed,
            "ei_ca1_change_max_abs": trained["ei_ca1_change_max_abs"],
            "analysis": analysis,
            "cue_remapping": cue_remapping,
            "remapping_dynamics": dynamics,
            "cue_locked_partition": cue_locked_partition,
            "key_value_coupling": coupling,
            "decoded_track": decoded_track,
        })
        if first_trained is None:
            first_trained = trained

    aggregate_analysis = aggregate_receptive_fields(receptive_fields)
    aggregate_remapping = aggregate_cue_remapping(
        remappings, aggregate_analysis["instructive_activity"]
    )
    aggregate_dynamics = aggregate_remapping_dynamics(
        remapping_dynamics, aggregate_remapping["stable"]
    )
    cue_locked_partition = analyse_cue_locked_partition(
        remapping_dynamics,
        aggregate_analysis,
        change_threshold=change_threshold,
    )
    activity_trajectories = analyse_ca1_activity_trajectories(
        aggregate_analysis,
        aggregate_dynamics,
        aggregate_remapping["stable"],
    )
    aggregate_coupling = aggregate_key_value_coupling(couplings)
    aggregate_decoded_track = aggregate_decoded_tracks(decoded_tracks)
    maximum_weight_change = max(
        repetition["ei_ca1_change_max_abs"] for repetition in repetitions
    )
    return {
        "schema_version": 4,
        "description": (
            "Repeated functional EI-to-CA1 receptive-field, cue-remapping, "
            "key-value coupling, and decoded track reconstruction analysis."
        ),
        "settings": {
            "ae_name": ae_name,
            "seed": resolved_seed,
            "seed_mode": "random" if seed is None else "fixed",
            "reps": reps,
            "data": first_trained["data_settings"],
            "mtl": mtl_settings,
            "classification": {
                "silent_fraction": silent_fraction,
                "dominance": dominance,
                "class_order": list(CLASS_NAMES),
                "change_threshold": change_threshold,
                "dynamic_class_order": list(DYNAMIC_CLASS_NAMES),
            },
        },
        "autoencoder_session": first_trained["autoencoder_session"],
        "cue_assignments": first_trained["cue_assignments"],
        "ei_ca1_change_max_abs": maximum_weight_change,
        "analysis": aggregate_analysis,
        "cue_remapping": aggregate_remapping,
        "remapping_dynamics": aggregate_dynamics,
        "cue_locked_partition": cue_locked_partition,
        "ca1_activity_trajectories": activity_trajectories,
        "key_value_coupling": aggregate_coupling,
        "decoded_track": aggregate_decoded_track,
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
        seed=None if args.random_seed else args.seed,
        reps=args.reps,
        mtl_settings=mtl_settings,
        silent_fraction=args.silent_fraction,
        dominance=args.dominance,
        change_threshold=args.change_threshold,
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
    dynamics_figure = plot_remapping_dynamics(
        result["remapping_dynamics"]
    )
    cue_locked_figure = plot_cue_locked_partition(
        result["cue_locked_partition"]
    )
    trajectory_figure = plot_ca1_activity_trajectories(
        result["ca1_activity_trajectories"],
        cue_positions=result["settings"]["data"]["cue_positions"],
    )
    coupling_figure = plot_key_value_coupling(
        result["key_value_coupling"], result["cue_assignments"]
    )
    decoded_track_figure = plot_decoded_track(result["decoded_track"])
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
    dynamics_path = figure_path.with_name(
        f"{figure_path.stem}_remapping_dynamics{figure_path.suffix}"
    )
    trajectory_path = figure_path.with_name(
        f"{figure_path.stem}_ca1_activity_trajectories{figure_path.suffix}"
    )
    cue_locked_path = figure_path.with_name(
        f"{figure_path.stem}_cue_locked_partition{figure_path.suffix}"
    )
    decoded_track_path = figure_path.with_name(
        f"{figure_path.stem}_decoded_track{figure_path.suffix}"
    )
    receptive_field_figure.savefig(figure_path, dpi=180)
    remapping_figure.savefig(remapping_path, dpi=180)
    dynamics_figure.savefig(dynamics_path, dpi=180)
    cue_locked_figure.savefig(cue_locked_path, dpi=180)
    trajectory_figure.savefig(trajectory_path, dpi=180)
    coupling_figure.savefig(coupling_path, dpi=180)
    decoded_track_figure.savefig(decoded_track_path, dpi=180)
    print(f"class counts: {result['analysis']['class_counts']}")
    print(
        f"random seed: {result['settings']['seed']} "
        f"({result['settings']['seed_mode']})"
    )
    print(
        "cue-remapping counts: "
        f"stable={result['cue_remapping']['stable_count']}, "
        f"adapting={result['cue_remapping']['adapting_count']}"
    )
    print(
        "cue-correlated dynamic classes: "
        f"{result['cue_locked_partition']['class_counts']}"
    )
    print(
        "maximum EI→CA1 weight change during MTL training: "
        f"{result['ei_ca1_change_max_abs']:.3g}"
    )
    print(f"analysis saved to {output}")
    print(f"receptive-field figure saved to {figure_path}")
    print(f"cue-remapping figure saved to {remapping_path}")
    print(f"remapping-dynamics figure saved to {dynamics_path}")
    print(f"cue-locked partition figure saved to {cue_locked_path}")
    print(f"CA1 activity-trajectory figure saved to {trajectory_path}")
    print(f"key-value coupling figure saved to {coupling_path}")
    print(f"decoded-track figure saved to {decoded_track_path}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(receptive_field_figure)
        plt.close(remapping_figure)
        plt.close(dynamics_figure)
        plt.close(cue_locked_figure)
        plt.close(trajectory_figure)
        plt.close(coupling_figure)
        plt.close(decoded_track_figure)


if __name__ == "__main__":
    main()
