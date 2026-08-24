"""Three-cue version of :mod:`track_analysis_experiment`.

Three distinct LEC cue patterns occupy three evenly spaced locations on a
20-lap circular track.  Their location assignment rotates after lap 10 from
``[0, 1, 2]`` to ``[1, 2, 0]``.  The full receptive-field, remapping,
cue-locking, trajectory, key-value, and decoded-track analyses are retained.
"""

from __future__ import annotations

import argparse
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

from core import datagen  # noqa: E402
from experiments import track_analysis_experiment as analysis_1  # noqa: E402
from experiments import track_experiment as track  # noqa: E402


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / \
    "track_analysis_experiment_2.json"
NUM_CUES = 3
CUE_POSITIONS = (8, 25, 42)
CUE_COLORS = ("tab:cyan", "tab:pink", "tab:olive")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train on a three-cue circular track and run the complete CA1 "
            "receptive-field/remapping analysis."
        )
    )
    parser.add_argument("--ae-name", default="ae_cue_nb_7")
    parser.add_argument(
        "--seed",
        type=analysis_1.parse_seed,
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
    parser.add_argument("--silent-fraction", type=float, default=0.05)
    parser.add_argument("--dominance", type=float, default=0.67)
    parser.add_argument(
        "--change-threshold",
        type=float,
        default=0.20,
        help="cosine-distance threshold for stable versus remapping fields",
    )
    parser.add_argument(
        "--k-ca3", type=int, default=track.DEFAULT_MTL_SETTINGS["K_ca3"]
    )
    parser.add_argument(
        "--beta-ca3", type=float,
        default=track.DEFAULT_MTL_SETTINGS["beta_ca3"],
    )
    parser.add_argument(
        "--beta-ca1", type=float,
        default=track.DEFAULT_MTL_SETTINGS["beta_ca1"],
    )
    parser.add_argument(
        "--alpha", type=float, default=track.DEFAULT_MTL_SETTINGS["alpha"]
    )
    parser.add_argument(
        "--nb-ei-ca3", type=int,
        default=track.DEFAULT_MTL_SETTINGS["nb_ei_ca3"],
    )
    parser.add_argument(
        "--num-swaps-ca3", type=int,
        default=track.DEFAULT_MTL_SETTINGS["num_swaps_ca3"],
    )
    parser.add_argument(
        "--num-swaps-ca1", type=int,
        default=track.DEFAULT_MTL_SETTINGS["num_swaps_ca1"],
    )
    parser.add_argument(
        "--plasticity",
        choices=("base", "nois", "isout", "err1", "err2"),
        default=track.DEFAULT_MTL_SETTINGS["plasticity"],
    )
    return parser.parse_args()


def three_cue_assignments(num_laps: int, swap_lap: int = 10) -> list[list[int]]:
    """Rotate three identities across three positions at the context switch."""

    if not 0 < swap_lap < num_laps:
        raise ValueError("swap_lap must split the recorded laps")
    return [
        [0, 1, 2] if lap < swap_lap else [1, 2, 0]
        for lap in range(num_laps)
    ]


def make_three_cue_track(settings: dict) -> tuple[np.ndarray, list[list[int]]]:
    """Generate a circular track containing three distinct cue patterns."""

    size = int(settings["size"])
    if size < 2 or size % 2:
        raise ValueError("size must be an even integer of at least 2")
    cue_positions = list(settings["cue_positions"])
    if len(cue_positions) != NUM_CUES:
        raise ValueError("the three-cue experiment requires three positions")
    assignments = three_cue_assignments(
        int(settings["num_laps"]), int(settings["swap_every"])
    )
    cue_patterns = datagen.make_cues(
        n=NUM_CUES, size=size // 2, fixed=True
    )
    laps = {
        "n": int(settings["num_laps"]),
        "length": int(settings["lap_length"]),
        "cues_positions": cue_positions,
        "cues_patterns": cue_patterns,
        "cues_sequence": assignments,
        "cue_sigma": float(settings["cue_sigma"]),
        "cue_beta": float(settings["cue_beta"]),
        "cue_alpha": float(settings["cue_alpha"]),
        "mec_binarized": bool(settings["mec_binarized"]),
    }
    stimuli, _ = datagen.sparse_stimulus_generator_sensory(
        laps=laps,
        mec_size=size // 2,
        mec_sigma=float(settings["mec_sigma"]),
        lec_sigma=float(settings["lec_sigma"]),
    )
    return stimuli.astype(np.float32), assignments


def train_track_model(ae_name: str, seed: int, mtl_settings: dict,
                      quiet: bool = False):
    data_settings = {
        **track.DEFAULT_DATA_SETTINGS,
        "num_laps": 20,
        "lap_length": 50,
        "swap_every": 10,
        "cue_positions": list(CUE_POSITIONS),
        "num_cue_patterns": NUM_CUES,
    }
    np.random.seed(seed)
    torch.manual_seed(seed)
    stimuli, assignments = make_three_cue_track(data_settings)
    model, ae_session = track.build_mtl(ae_name, mtl_settings)
    initial_ei_ca1 = model.W_ei_ca1.detach().cpu().numpy().copy()
    for lap in tqdm(
            stimuli, desc="training three-cue track laps",
            disable=quiet, leave=False):
        track._train_lap(model, lap)
    final_ei_ca1 = model.W_ei_ca1.detach().cpu().numpy().copy()
    return {
        "model": model,
        "stimuli": stimuli,
        "cue_assignments": assignments,
        "data_settings": data_settings,
        "autoencoder_session": ae_session,
        "ei_ca1_change_max_abs": float(np.max(np.abs(
            final_ei_ca1 - initial_ei_ca1
        ))),
    }


def analyse_decoded_track(model, stimuli: np.ndarray,
                          assignments: list[list[int]],
                          cue_positions=CUE_POSITIONS,
                          lap_indices=(9, 19)) -> dict:
    """Decode position and three-way cue identity from recalled CA1."""

    track_length = stimuli.shape[1]
    mec_size = stimuli.shape[2] // 2
    cue_samples = {identity: [] for identity in range(NUM_CUES)}
    for lap_index, assignment in enumerate(assignments):
        for slot, identity in enumerate(assignment):
            cue_samples[int(identity)].append(
                stimuli[lap_index, cue_positions[slot], mec_size:]
            )
    cue_templates = np.stack([
        np.mean(cue_samples[identity], axis=0)
        for identity in range(NUM_CUES)
    ])
    template_norm = np.linalg.norm(cue_templates, axis=1)
    true_position = np.arange(track_length, dtype=float)
    conditions = []
    for lap_index in lap_indices:
        _, _, activity = track._recall_lap(
            model, stimuli[lap_index], return_activity=True
        )
        eo = activity["eo"]
        decoded_position = analysis_1._decode_circular_population(
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
        logits = 8.0 * (
            cue_similarity - cue_similarity.max(axis=1, keepdims=True)
        )
        cue_probability = np.exp(logits)
        cue_probability /= np.maximum(
            cue_probability.sum(axis=1, keepdims=True), 1e-8
        )
        identity_vectors = np.exp(
            2j * np.pi * np.arange(NUM_CUES) / NUM_CUES
        )
        cue_vector = cue_probability @ identity_vectors
        cue_angle = np.mod(np.angle(cue_vector), 2.0 * np.pi)
        cue_confidence = np.abs(cue_vector)
        cue_predictions = np.argmax(cue_probability, axis=1)
        assignment = np.asarray(assignments[lap_index], dtype=int)
        predicted_at_cues = cue_predictions[np.asarray(cue_positions)]
        conditions.append({
            "lap_index": int(lap_index),
            "assignment": assignment,
            "x_ei": activity["x_ei"],
            "ca1": activity["ca1"],
            "eo": eo,
            "decoded_position": decoded_position,
            "position_error": position_error,
            "cue_similarity": cue_similarity,
            "cue_probability": cue_probability,
            "cue_angle": cue_angle,
            # Retain the base aggregator's generic scalar evidence field.
            "cue_evidence": cue_confidence,
            "cue_confidence": cue_confidence,
            "predicted_identity": cue_predictions,
            "predicted_identity_at_cues": predicted_at_cues,
            "cue_identity_accuracy": float(np.mean(
                predicted_at_cues == assignment
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
        "num_cues": NUM_CUES,
    }


def aggregate_decoded_tracks(decoded_tracks: list[dict]) -> dict:
    aggregate = analysis_1.aggregate_decoded_tracks(decoded_tracks)
    aggregate["num_cues"] = NUM_CUES
    for condition_index, condition in enumerate(aggregate["conditions"]):
        items = [
            result["conditions"][condition_index]
            for result in decoded_tracks
        ]
        for key in ("cue_probability", "cue_confidence"):
            values = np.stack([item[key] for item in items])
            condition[key] = values.mean(axis=0)
            condition[f"{key}_std"] = values.std(axis=0)
        cue_vectors = np.stack([
            np.exp(1j * item["cue_angle"]) * item["cue_confidence"]
            for item in items
        ])
        mean_vector = cue_vectors.mean(axis=0)
        condition["cue_angle"] = np.mod(
            np.angle(mean_vector), 2.0 * np.pi
        )
        condition["predicted_identity"] = np.argmax(
            condition["cue_probability"], axis=1
        )
    return aggregate


def plot_decoded_track(decoded: dict):
    """Visualize the three-class CA1→EO circular-track reconstruction."""

    figure = plt.figure(figsize=(16, 9), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.25, 1.0))
    axis_3d = figure.add_subplot(grid[:, 0], projection="3d")
    position_axis = figure.add_subplot(grid[0, 1])
    probability_axis = figure.add_subplot(grid[1, 1])
    track_length = int(decoded["track_length"])
    positions = np.asarray(decoded["true_position"])
    cue_positions = np.asarray(decoded["cue_positions"], dtype=int)
    condition_colors = ("tab:blue", "tab:orange")

    circle = np.linspace(0, 2.0 * np.pi, 400)
    axis_3d.plot(
        np.cos(circle), np.sin(circle), np.zeros_like(circle),
        color="0.65", linestyle=":", label="physical track",
    )
    for condition_index, condition in enumerate(decoded["conditions"]):
        decoded_angle = (
            2.0 * np.pi * condition["decoded_position"] / track_length
        )
        x = np.cos(decoded_angle)
        y = np.sin(decoded_angle)
        confidence = condition["cue_confidence"]
        color = condition_colors[condition_index]
        axis_3d.plot(
            x, y, confidence, color=color, linewidth=1.7,
            label=(
                f"lap {condition['lap_index'] + 1}, cues "
                f"{np.asarray(condition['assignment'], dtype=int).tolist()}"
            ),
        )
        axis_3d.scatter(
            x, y, confidence,
            c=condition["cue_angle"], cmap="hsv", vmin=0,
            vmax=2.0 * np.pi, s=16, alpha=0.7,
        )
        for slot, cue_position in enumerate(cue_positions):
            identity = int(condition["assignment"][slot])
            axis_3d.scatter(
                x[cue_position], y[cue_position], confidence[cue_position],
                color=CUE_COLORS[identity], edgecolor="black",
                marker="*", s=190, depthshade=False,
            )
            axis_3d.text(
                x[cue_position], y[cue_position],
                confidence[cue_position] + 0.04,
                f"cue {identity}", fontsize=8,
            )
    axis_3d.set(
        xlabel="decoded track x",
        ylabel="decoded track y",
        zlabel="three-way cue confidence",
        zlim=(0, 1.05),
        title="CA1-decoded track and three cue identities",
    )
    axis_3d.view_init(elev=25, azim=-55)
    axis_3d.legend(loc="upper left", fontsize=8)

    for condition_index, condition in enumerate(decoded["conditions"]):
        mae = float(np.mean(condition["position_error"]))
        position_axis.scatter(
            positions, condition["decoded_position"],
            color=condition_colors[condition_index], s=22, alpha=0.7,
            label=f"lap {condition['lap_index'] + 1} (MAE={mae:.2f})",
        )
    position_axis.plot(
        [0, track_length], [0, track_length],
        color="0.35", linestyle="--", label="exact position",
    )
    for cue_position in cue_positions:
        position_axis.axvline(cue_position, color="0.65", linestyle=":")
    position_axis.set(
        xlabel="physical track position",
        ylabel="position decoded from reconstructed MEC",
        xlim=(-1, track_length), ylim=(-1, track_length),
        title="Spatial reconstruction from decoded CA1",
    )
    position_axis.grid(alpha=0.15)
    position_axis.legend(loc="best", fontsize=8)

    row_labels = []
    probability_rows = []
    for condition in decoded["conditions"]:
        for identity in range(NUM_CUES):
            probability_rows.append(condition["cue_probability"][:, identity])
            row_labels.append(
                f"lap {condition['lap_index'] + 1}: cue {identity}"
            )
    probability_image = probability_axis.imshow(
        np.asarray(probability_rows),
        origin="upper", aspect="auto", interpolation="nearest",
        vmin=0, vmax=1, cmap="viridis",
        extent=(0, track_length, len(probability_rows) - 0.5, -0.5),
    )
    probability_axis.set_yticks(np.arange(len(row_labels)), row_labels)
    for cue_position in cue_positions:
        probability_axis.axvline(cue_position, color="white", linestyle=":")
    probability_axis.set(
        xlabel="physical track position",
        ylabel="decoded cue channel",
        title="Three-way reconstructed cue probability",
    )
    figure.colorbar(
        probability_image, ax=probability_axis, label="cue probability"
    )
    figure.suptitle(
        "Three-cue circular-track reconstruction decoded from CA1 | "
        f"{decoded['num_repetitions']} repetitions"
    )
    return figure


def run_analysis(ae_name: str = "ae_cue_nb_7", seed: int | None = 40,
                 reps: int = 5, mtl_settings: dict | None = None,
                 silent_fraction: float = 0.05,
                 dominance: float = 0.67,
                 change_threshold: float = 0.20,
                 quiet: bool = False) -> dict:
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
    dynamics_items = []
    couplings = []
    decoded_tracks = []
    first_trained = None
    for repetition in tqdm(
            range(reps), desc="three-cue analysis repetitions",
            disable=quiet):
        repetition_seed = (resolved_seed + repetition) % (2 ** 32)
        trained = train_track_model(
            ae_name, repetition_seed, mtl_settings, quiet=True
        )
        analysis = analysis_1.analyse_ei_ca1(
            trained["model"], trained["stimuli"],
            silent_fraction=silent_fraction, dominance=dominance,
        )
        remapping = analysis_1.analyse_cue_remapping(
            analysis["instructive_activity"], swap_lap=10
        )
        dynamics = analysis_1.analyse_remapping_dynamics(
            analysis["instructive_activity"], swap_lap=10
        )
        coupling = analysis_1.analyse_key_value_coupling(
            trained["model"], trained["stimuli"], lap_indices=(9, 19)
        )
        decoded = analyse_decoded_track(
            trained["model"], trained["stimuli"],
            trained["cue_assignments"], lap_indices=(9, 19)
        )
        partition = analysis_1.analyse_cue_locked_partition(
            [dynamics], analysis, change_threshold
        )
        receptive_fields.append(analysis)
        remappings.append(remapping)
        dynamics_items.append(dynamics)
        couplings.append(coupling)
        decoded_tracks.append(decoded)
        repetitions.append({
            "repetition": repetition,
            "seed": repetition_seed,
            "ei_ca1_change_max_abs": trained["ei_ca1_change_max_abs"],
            "analysis": analysis,
            "cue_remapping": remapping,
            "remapping_dynamics": dynamics,
            "cue_locked_partition": partition,
            "key_value_coupling": coupling,
            "decoded_track": decoded,
        })
        if first_trained is None:
            first_trained = trained

    aggregate_analysis = analysis_1.aggregate_receptive_fields(
        receptive_fields
    )
    aggregate_remapping = analysis_1.aggregate_cue_remapping(
        remappings, aggregate_analysis["instructive_activity"]
    )
    aggregate_dynamics = analysis_1.aggregate_remapping_dynamics(
        dynamics_items, aggregate_remapping["stable"]
    )
    partition = analysis_1.analyse_cue_locked_partition(
        dynamics_items, aggregate_analysis, change_threshold
    )
    trajectories = analysis_1.analyse_ca1_activity_trajectories(
        aggregate_analysis, aggregate_dynamics, aggregate_remapping["stable"]
    )
    return {
        "schema_version": 1,
        "description": (
            "Complete CA1 receptive-field and remapping analysis for three "
            "cue identities rotating across three circular-track positions."
        ),
        "settings": {
            "ae_name": ae_name,
            "seed": resolved_seed,
            "seed_mode": "random" if seed is None else "fixed",
            "reps": reps,
            "num_cues": NUM_CUES,
            "data": first_trained["data_settings"],
            "mtl": mtl_settings,
            "classification": {
                "silent_fraction": silent_fraction,
                "dominance": dominance,
                "class_order": list(analysis_1.CLASS_NAMES),
                "change_threshold": change_threshold,
                "dynamic_class_order": list(
                    analysis_1.DYNAMIC_CLASS_NAMES
                ),
            },
        },
        "autoencoder_session": first_trained["autoencoder_session"],
        "cue_assignments": first_trained["cue_assignments"],
        "ei_ca1_change_max_abs": max(
            item["ei_ca1_change_max_abs"] for item in repetitions
        ),
        "analysis": aggregate_analysis,
        "cue_remapping": aggregate_remapping,
        "remapping_dynamics": aggregate_dynamics,
        "cue_locked_partition": partition,
        "ca1_activity_trajectories": trajectories,
        "key_value_coupling": analysis_1.aggregate_key_value_coupling(
            couplings
        ),
        "decoded_track": aggregate_decoded_tracks(decoded_tracks),
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
    cue_positions = result["settings"]["data"]["cue_positions"]
    figures = {
        "": analysis_1.plot_analysis(result["analysis"], cue_positions),
        "_cue_remapping": analysis_1.plot_cue_remapping(
            result["cue_remapping"], cue_positions
        ),
        "_remapping_dynamics": analysis_1.plot_remapping_dynamics(
            result["remapping_dynamics"]
        ),
        "_cue_locked_partition": analysis_1.plot_cue_locked_partition(
            result["cue_locked_partition"]
        ),
        "_ca1_activity_trajectories": (
            analysis_1.plot_ca1_activity_trajectories(
                result["ca1_activity_trajectories"], cue_positions
            )
        ),
        "_key_value_coupling": analysis_1.plot_key_value_coupling(
            result["key_value_coupling"], result["cue_assignments"]
        ),
        "_decoded_track": plot_decoded_track(result["decoded_track"]),
    }
    output = analysis_1.save_analysis(result, args.output)
    figure_path = (
        args.figure.expanduser().resolve()
        if args.figure is not None else output.with_suffix(".png")
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    saved_paths = {}
    for suffix, figure in figures.items():
        path = figure_path.with_name(
            f"{figure_path.stem}{suffix}{figure_path.suffix}"
        )
        figure.savefig(path, dpi=180)
        saved_paths[suffix or "_receptive_fields"] = path

    print(f"class counts: {result['analysis']['class_counts']}")
    print(
        f"random seed: {result['settings']['seed']} "
        f"({result['settings']['seed_mode']})"
    )
    print(f"dynamic classes: {result['cue_locked_partition']['class_counts']}")
    print(f"analysis saved to {output}")
    for name, path in saved_paths.items():
        print(f"{name.lstrip('_').replace('_', '-')} figure saved to {path}")
    if not args.no_show:
        plt.show()
    else:
        for figure in figures.values():
            plt.close(figure)


if __name__ == "__main__":
    main()
