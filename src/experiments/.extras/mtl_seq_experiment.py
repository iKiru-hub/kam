"""Study prompted and unprompted cue recall across short cue sequences.

Two independent MTL models experience five laps with cue pair [1, 2], then
five laps after a context change.  In simulation 1 only one cue changes
([1, 2] -> [1, 3]); in simulation 2 both positions change identity
([1, 2] -> [2, 3]).  Plasticity is paused for prompted and MEC-only test laps
after each phase.
"""

from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core import datagen  # noqa: E402
from experiments import track_analysis_experiment as analysis  # noqa: E402
from experiments import track_experiment as track  # noqa: E402


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / \
    "mtl_seq_experiment.json"
DEFAULT_MTL_SETTINGS = {
    "K_ca3": 4,
    "dim_ca3": 50,
    "beta_ca3": 98,
    "beta_ca1": 117,
    "alpha": 0.036,
    "nb_ei_ca3": 17,
    "num_swaps_ca1": 0,
    "num_swaps_ca3": 0,
    "random_IS": False,
    "plasticity": "err2",
}
DEFAULT_DATA_SETTINGS = {
    "size": 50,
    "lap_length": 50,
    "cue_positions": [10, 30],
    "cue_sigma": 3.0,
    "cue_beta": 40.0,
    "cue_alpha": 0.2,
    "mec_binarized": True,
    "mec_sigma": 4.0,
    "lec_sigma": 5.0,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure cue memory across one- and two-cue changes."
    )
    parser.add_argument("--ae-name", default="ae_3cues_0")
    parser.add_argument(
        "--seed", type=analysis.parse_seed, default=3980,
        help="integer seed, or 'random' for a fresh reported seed",
    )
    parser.add_argument("--random-seed", action="store_true")
    parser.add_argument("--phase-laps", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument(
        "--k-ca3", type=int, default=DEFAULT_MTL_SETTINGS["K_ca3"]
    )
    parser.add_argument(
        "--beta-ca3", type=float, default=DEFAULT_MTL_SETTINGS["beta_ca3"]
    )
    parser.add_argument(
        "--beta-ca1", type=float, default=DEFAULT_MTL_SETTINGS["beta_ca1"]
    )
    parser.add_argument(
        "--alpha", type=float, default=DEFAULT_MTL_SETTINGS["alpha"]
    )
    parser.add_argument(
        "--nb-ei-ca3", type=int,
        default=DEFAULT_MTL_SETTINGS["nb_ei_ca3"],
    )
    parser.add_argument(
        "--num-swaps-ca3", type=int,
        default=DEFAULT_MTL_SETTINGS["num_swaps_ca3"],
    )
    parser.add_argument(
        "--num-swaps-ca1", type=int,
        default=DEFAULT_MTL_SETTINGS["num_swaps_ca1"],
    )
    parser.add_argument(
        "--plasticity",
        choices=("base", "nois", "isout", "err1", "err2", "err3"),
        default=DEFAULT_MTL_SETTINGS["plasticity"],
    )
    return parser.parse_args()


def make_protocol_laps(first_pair: tuple[int, int],
                       second_pair: tuple[int, int],
                       phase_laps: int,
                       settings: dict) -> tuple[np.ndarray, np.ndarray]:
    """Generate training laps plus one held-out test lap for each phase."""

    if phase_laps < 1:
        raise ValueError("phase_laps must be at least 1")
    # Public cue labels are one-based; the generator indexes templates from 0.
    pairs = np.asarray((first_pair, second_pair), dtype=int)
    if np.any((pairs < 1) | (pairs > 3)):
        raise ValueError("cue identities must be in {1, 2, 3}")
    internal_pairs = pairs - 1
    phase_size = phase_laps + 1
    sequence = (
        [internal_pairs[0].tolist()] * phase_size
        + [internal_pairs[1].tolist()] * phase_size
    )
    cue_patterns = datagen.make_cues(
        n=3, size=int(settings["size"]) // 2, fixed=True
    )
    laps = {
        "n": len(sequence),
        "length": int(settings["lap_length"]),
        "cues_positions": list(settings["cue_positions"]),
        "cues_patterns": cue_patterns,
        "cues_sequence": sequence,
        "cue_sigma": float(settings["cue_sigma"]),
        "cue_beta": float(settings["cue_beta"]),
        "cue_alpha": float(settings["cue_alpha"]),
        "mec_binarized": bool(settings["mec_binarized"]),
    }
    stimuli, _ = datagen.sparse_stimulus_generator_sensory(
        laps=laps,
        mec_size=int(settings["size"]) // 2,
        mec_sigma=float(settings["mec_sigma"]),
        lec_sigma=float(settings["lec_sigma"]),
    )
    return stimuli.astype(np.float32), cue_patterns.astype(np.float32)


def recall_test_lap(model, target_lap: np.ndarray,
                    cue_patterns: np.ndarray,
                    cue_positions: list[int],
                    mec_only: bool) -> dict:
    """Recall a held-out lap without modifying weights."""

    target = np.asarray(target_lap, dtype=np.float32)
    presented = target.copy()
    mec_size = presented.shape[1] // 2
    if mec_only:
        presented[:, mec_size:] = 0.0

    model.reset()
    model.pause_lr()
    reconstructed = []
    ca3_activity = []
    ca1_activity = []
    with torch.no_grad():
        for sample in presented:
            x = torch.as_tensor(sample, dtype=torch.float32).reshape(-1, 1)
            model(x)
            reconstructed.append(track._vector(model._eo))
            ca3_activity.append(track._vector(model._ca3))
            ca1_activity.append(track._vector(model._ca1))
    reconstructed = np.asarray(reconstructed)

    def cosine_rows(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        numerator = np.sum(first * second, axis=1)
        denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(
            second, axis=1
        )
        both_zero = (
            (np.linalg.norm(first, axis=1) <= 1e-8)
            & (np.linalg.norm(second, axis=1) <= 1e-8)
        )
        values = numerator / np.maximum(denominator, 1e-8)
        values[both_zero] = 1.0
        return values

    target_lec = target[:, mec_size:]
    reconstructed_lec = reconstructed[:, mec_size:]
    cue_indices = np.asarray(cue_positions, dtype=int)
    cue_outputs = reconstructed_lec[cue_indices]
    cue_similarity = (
        cue_outputs @ cue_patterns.T
    ) / np.maximum(
        np.linalg.norm(cue_outputs, axis=1, keepdims=True)
        * np.linalg.norm(cue_patterns, axis=1)[None, :],
        1e-8,
    )
    target_identity = np.argmax(
        target_lec[cue_indices] @ cue_patterns.T, axis=1
    )
    predicted_identity = np.argmax(cue_similarity, axis=1)
    return {
        "mode": "mec_only" if mec_only else "prompted",
        "target": target,
        "presented": presented,
        "reconstructed": reconstructed,
        "ca3_activity": np.asarray(ca3_activity),
        "ca1_activity": np.asarray(ca1_activity),
        "overall_mse": float(np.mean((target - reconstructed) ** 2)),
        "mec_mse": float(np.mean(
            (target[:, :mec_size] - reconstructed[:, :mec_size]) ** 2
        )),
        "lec_mse": float(np.mean((target_lec - reconstructed_lec) ** 2)),
        "overall_cosine": float(np.mean(cosine_rows(
            target, reconstructed
        ))),
        "mec_cosine": float(np.mean(cosine_rows(
            target[:, :mec_size], reconstructed[:, :mec_size]
        ))),
        "lec_cosine": float(np.mean(cosine_rows(
            target_lec, reconstructed_lec
        ))),
        "cue_similarity": cue_similarity,
        "target_cue_identity": target_identity + 1,
        "predicted_cue_identity": predicted_identity + 1,
        "cue_identity_accuracy": float(np.mean(
            predicted_identity == target_identity
        )),
    }


def run_sequence_simulation(name: str,
                            first_pair: tuple[int, int],
                            second_pair: tuple[int, int],
                            ae_name: str,
                            seed: int,
                            phase_laps: int,
                            data_settings: dict,
                            mtl_settings: dict) -> dict:
    """Train and test one two-phase cue sequence on a fresh MTL model."""

    # Reusing the same seed across the two simulations gives matched MEC noise
    # and CA3 connectivity, isolating the difference in cue protocol.
    np.random.seed(seed)
    torch.manual_seed(seed)
    laps, cue_patterns = make_protocol_laps(
        first_pair, second_pair, phase_laps, data_settings
    )
    model, ae_session = track.build_mtl(ae_name, mtl_settings)
    phase_size = phase_laps + 1

    for lap in laps[:phase_laps]:
        track._train_lap(model, lap)
    tests = [
        recall_test_lap(
            model, laps[phase_laps], cue_patterns,
            data_settings["cue_positions"], mec_only=False,
        ),
        recall_test_lap(
            model, laps[phase_laps], cue_patterns,
            data_settings["cue_positions"], mec_only=True,
        ),
    ]

    phase_b_start = phase_size
    for lap in laps[phase_b_start:phase_b_start + phase_laps]:
        track._train_lap(model, lap)
    phase_b_test = laps[phase_b_start + phase_laps]
    tests.extend([
        recall_test_lap(
            model, phase_b_test, cue_patterns,
            data_settings["cue_positions"], mec_only=False,
        ),
        recall_test_lap(
            model, phase_b_test, cue_patterns,
            data_settings["cue_positions"], mec_only=True,
        ),
    ])
    phase_labels = (
        f"after {phase_laps} laps {list(first_pair)}",
        f"after {phase_laps} laps {list(second_pair)}",
    )
    for index, test in enumerate(tests):
        test["phase"] = "A" if index < 2 else "B"
        test["cue_pair"] = list(first_pair if index < 2 else second_pair)
        test["label"] = (
            phase_labels[0 if index < 2 else 1]
            + (" | MEC only" if test["mode"] == "mec_only" else " | prompted")
        )
    return {
        "name": name,
        "first_pair": list(first_pair),
        "second_pair": list(second_pair),
        "tests": tests,
        "final_W_ca3_ca1": model.W_ca3_ca1.detach().cpu().numpy().copy(),
        "autoencoder_session": ae_session,
    }


def plot_sequence_tests(simulation: dict, cue_positions: list[int]):
    """Plot target, presented input, and reconstruction for all four tests."""

    tests = simulation["tests"]
    figure, axes = plt.subplots(
        3, 4, figsize=(19, 11), sharex=True, sharey=True,
        constrained_layout=True,
    )
    row_names = ("Full target", "Presented input", "CA1→EO reconstruction")
    value_keys = ("target", "presented", "reconstructed")
    image = None
    for column, test in enumerate(tests):
        for row, (row_name, key) in enumerate(zip(row_names, value_keys)):
            values = test[key]
            image = axes[row, column].imshow(
                values.T,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                vmin=0.0,
                vmax=1.0,
                cmap="viridis",
                extent=(0, len(values), 0, values.shape[1]),
            )
            axes[row, column].axhline(
                values.shape[1] / 2, color="white", linestyle="--",
                linewidth=0.9,
            )
            for position in cue_positions:
                axes[row, column].axvline(
                    position, color="white", linestyle=":", linewidth=0.75
                )
            if column == 0:
                axes[row, column].set_ylabel(
                    f"{row_name}\nEC unit (MEC below, LEC above)"
                )
        tests_title = (
            f"{test['label']}\n"
            f"MSE={test['overall_mse']:.3f}, "
            f"LEC MSE={test['lec_mse']:.3f}, "
            f"cue ID={test['cue_identity_accuracy']:.0%}"
        )
        axes[0, column].set_title(tests_title, fontsize=10)
        axes[-1, column].set_xlabel("track position (cm)")
    figure.colorbar(image, ax=axes, label="activity", shrink=0.82)
    figure.suptitle(
        f"{simulation['name']}: prompted versus MEC-only cue recall"
    )
    return figure


def run_experiment(ae_name: str = "ae_3cues_0",
                   seed: int | None = 3982,
                   phase_laps: int = 15,
                   data_settings: dict | None = None,
                   mtl_settings: dict | None = None) -> dict:
    resolved_seed = secrets.randbits(32) if seed is None else int(seed)
    data_settings = {
        **DEFAULT_DATA_SETTINGS,
        **({} if data_settings is None else data_settings),
    }
    mtl_settings = {
        **DEFAULT_MTL_SETTINGS,
        **({} if mtl_settings is None else mtl_settings),
    }
    simulations = [
        run_sequence_simulation(
            "Simulation 1 — one cue changes",
            (1, 2), (1, 3), ae_name, resolved_seed, phase_laps,
            data_settings, mtl_settings,
        ),
        run_sequence_simulation(
            "Simulation 2 — both cue positions change",
            (1, 2), (2, 3), ae_name, resolved_seed, phase_laps,
            data_settings, mtl_settings,
        ),
    ]
    return {
        "schema_version": 1,
        "description": (
            "Prompted and MEC-only recall after two phases of continual "
            "learning with one or both cue identities changing."
        ),
        "settings": {
            "ae_name": ae_name,
            "seed": resolved_seed,
            "seed_mode": "random" if seed is None else "fixed",
            "phase_laps": phase_laps,
            "data": data_settings,
            "mtl": mtl_settings,
        },
        "simulations": simulations,
    }


def main():
    args = parse_args()
    mtl_settings = {
        **DEFAULT_MTL_SETTINGS,
        "K_ca3": args.k_ca3,
        "beta_ca3": args.beta_ca3,
        "beta_ca1": args.beta_ca1,
        "alpha": args.alpha,
        "nb_ei_ca3": args.nb_ei_ca3,
        "num_swaps_ca3": args.num_swaps_ca3,
        "num_swaps_ca1": args.num_swaps_ca1,
        "plasticity": args.plasticity,
    }
    result = run_experiment(
        ae_name=args.ae_name,
        seed=None if args.random_seed else args.seed,
        phase_laps=args.phase_laps,
        mtl_settings=mtl_settings,
    )
    output = analysis.save_analysis(result, args.output)
    base_figure = (
        args.figure.expanduser().resolve()
        if args.figure is not None else output.with_suffix(".png")
    )
    base_figure.parent.mkdir(parents=True, exist_ok=True)
    figures = []
    figure_paths = []
    suffixes = ("one_cue_change", "two_cue_change")
    for simulation, suffix in zip(result["simulations"], suffixes):
        figure = plot_sequence_tests(
            simulation, result["settings"]["data"]["cue_positions"]
        )
        path = base_figure.with_name(
            f"{base_figure.stem}_{suffix}{base_figure.suffix}"
        )
        figure.savefig(path, dpi=180)
        figures.append(figure)
        figure_paths.append(path)

    print(
        f"random seed: {result['settings']['seed']} "
        f"({result['settings']['seed_mode']})"
    )
    for simulation in result["simulations"]:
        print(simulation["name"])
        for test in simulation["tests"]:
            print(
                f"  {test['label']}: MSE={test['overall_mse']:.4f}, "
                f"LEC MSE={test['lec_mse']:.4f}, "
                f"cue ID={test['cue_identity_accuracy']:.0%}"
            )
    print(f"results saved to {output}")
    for path in figure_paths:
        print(f"figure saved to {path}")
    if not args.no_show:
        plt.show()
    else:
        for figure in figures:
            plt.close(figure)


if __name__ == "__main__":
    main()
