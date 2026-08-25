"""Pattern-completion experiments with progressive EC input degradation.

After clean training on cue sequence [1, 2], plasticity is paused and the
model is probed at five dropout fractions.  Simulation 1 drops neurons across
all entorhinal input; simulation 2 drops only LEC neurons to isolate cue-trace
completion.  A neuron mask is fixed across each lap and resampled per probe.
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
from experiments import mtl_seq_experiment as seq  # noqa: E402
from experiments import track_analysis_experiment as analysis  # noqa: E402
from experiments import track_experiment as track  # noqa: E402


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / \
    "mtl_pcom_experiment.json"
DEFAULT_FRACTIONS = (0.0, 0.25, 0.50, 0.75, 0.90)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe MTL pattern completion under progressive dropout."
    )
    parser.add_argument("--ae-name", default="ae_3cues_0")
    parser.add_argument(
        "--seed", type=analysis.parse_seed, default=3980,
        help="integer seed, or 'random' for a fresh reported seed",
    )
    parser.add_argument("--random-seed", action="store_true")
    parser.add_argument(
        "--laps-per-level", "--n", dest="laps_per_level",
        type=int, default=5,
    )
    parser.add_argument(
        "--fractions", type=float, nargs="+", default=DEFAULT_FRACTIONS,
        help="ordered neuron-dropout fractions (default: 0 .25 .5 .75 .9)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument(
        "--k-ca3", type=int, default=seq.DEFAULT_MTL_SETTINGS["K_ca3"]
    )
    parser.add_argument(
        "--beta-ca3", type=float,
        default=seq.DEFAULT_MTL_SETTINGS["beta_ca3"],
    )
    parser.add_argument(
        "--beta-ca1", type=float,
        default=seq.DEFAULT_MTL_SETTINGS["beta_ca1"],
    )
    parser.add_argument(
        "--alpha", type=float, default=seq.DEFAULT_MTL_SETTINGS["alpha"]
    )
    parser.add_argument(
        "--nb-ei-ca3", type=int,
        default=seq.DEFAULT_MTL_SETTINGS["nb_ei_ca3"],
    )
    parser.add_argument(
        "--num-swaps-ca3", type=int,
        default=seq.DEFAULT_MTL_SETTINGS["num_swaps_ca3"],
    )
    parser.add_argument(
        "--num-swaps-ca1", type=int,
        default=seq.DEFAULT_MTL_SETTINGS["num_swaps_ca1"],
    )
    parser.add_argument(
        "--plasticity",
        choices=("base", "nois", "isout", "err1", "err2", "err3"),
        default=seq.DEFAULT_MTL_SETTINGS["plasticity"],
    )
    return parser.parse_args()


def validate_fractions(fractions) -> np.ndarray:
    values = np.asarray(fractions, dtype=float)
    if values.ndim != 1 or len(values) < 1:
        raise ValueError("fractions must be a non-empty one-dimensional list")
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("all dropout fractions must lie between 0 and 1")
    if np.any(np.diff(values) < 0):
        raise ValueError("dropout fractions must be in increasing order")
    if not np.isclose(values[0], 0.0):
        raise ValueError("the first dropout fraction must be 0")
    return values


def make_clean_laps(num_laps: int, settings: dict):
    """Generate independent clean laps with fixed one-based cue pair [1, 2]."""

    cue_patterns = datagen.make_cues(
        n=3, size=int(settings["size"]) // 2, fixed=True
    )
    laps = {
        "n": int(num_laps),
        "length": int(settings["lap_length"]),
        "cues_positions": list(settings["cue_positions"]),
        "cues_patterns": cue_patterns,
        "cues_sequence": [[0, 1] for _ in range(num_laps)],
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


def apply_neuron_dropout(lap: np.ndarray, fraction: float,
                         lec_only: bool) -> tuple[np.ndarray, np.ndarray]:
    """Zero a fixed random subset of eligible EC neurons for a whole lap."""

    corrupted = np.asarray(lap, dtype=np.float32).copy()
    input_size = corrupted.shape[1]
    mec_size = input_size // 2
    eligible = (
        np.arange(mec_size, input_size)
        if lec_only else np.arange(input_size)
    )
    num_dropped = int(np.floor(float(fraction) * len(eligible) + 0.5))
    if num_dropped:
        dropped = np.sort(np.random.choice(
            eligible, size=num_dropped, replace=False
        ))
        corrupted[:, dropped] = 0.0
    else:
        dropped = np.asarray([], dtype=int)
    return corrupted, dropped


def _row_cosine(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    numerator = np.sum(first * second, axis=1)
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(
        second, axis=1
    )
    result = numerator / np.maximum(denominator, 1e-8)
    both_zero = (
        (np.linalg.norm(first, axis=1) <= 1e-8)
        & (np.linalg.norm(second, axis=1) <= 1e-8)
    )
    result[both_zero] = 1.0
    return result


def recall_corrupted_lap(model, target: np.ndarray, presented: np.ndarray,
                         cue_patterns: np.ndarray,
                         cue_positions: list[int]) -> dict:
    """Recall one corrupted probe while plasticity remains disabled."""

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
    input_size = target.shape[1]
    mec_size = input_size // 2
    cue_indices = np.asarray(cue_positions, dtype=int)
    cue_output = reconstructed[cue_indices, mec_size:]
    cue_similarity = (
        cue_output @ cue_patterns.T
    ) / np.maximum(
        np.linalg.norm(cue_output, axis=1, keepdims=True)
        * np.linalg.norm(cue_patterns, axis=1)[None, :], 1e-8
    )
    target_cue = np.argmax(
        target[cue_indices, mec_size:] @ cue_patterns.T, axis=1
    )
    predicted_cue = np.argmax(cue_similarity, axis=1)
    return {
        "target": target,
        "presented": presented,
        "reconstructed": reconstructed,
        "ca3_activity": np.asarray(ca3_activity),
        "ca1_activity": np.asarray(ca1_activity),
        "input_mse": float(np.mean((target - presented) ** 2)),
        "reconstruction_mse": float(np.mean((target - reconstructed) ** 2)),
        "mec_mse": float(np.mean(
            (target[:, :mec_size] - reconstructed[:, :mec_size]) ** 2
        )),
        "lec_mse": float(np.mean(
            (target[:, mec_size:] - reconstructed[:, mec_size:]) ** 2
        )),
        "reconstruction_cosine": float(np.mean(_row_cosine(
            target, reconstructed
        ))),
        "cue_similarity": cue_similarity,
        "target_cue_identity": target_cue + 1,
        "predicted_cue_identity": predicted_cue + 1,
        "cue_identity_accuracy": float(np.mean(
            target_cue == predicted_cue
        )),
    }


def aggregate_level(probes: list[dict], fraction: float,
                    masks: list[np.ndarray]) -> dict:
    """Retain every probe and summarize reconstruction at one dropout level."""

    continuous_metrics = (
        "input_mse", "reconstruction_mse", "mec_mse", "lec_mse",
        "reconstruction_cosine", "cue_identity_accuracy",
    )
    result = {
        "fraction": float(fraction),
        "num_probes": len(probes),
        "dropped_neuron_indices": masks,
        "probes": probes,
        "mean_target": np.mean([probe["target"] for probe in probes], axis=0),
        "mean_presented": np.mean(
            [probe["presented"] for probe in probes], axis=0
        ),
        "mean_reconstruction": np.mean(
            [probe["reconstructed"] for probe in probes], axis=0
        ),
        "std_reconstruction": np.std(
            [probe["reconstructed"] for probe in probes], axis=0
        ),
    }
    for metric in continuous_metrics:
        values = np.asarray([probe[metric] for probe in probes])
        result[metric] = float(values.mean())
        result[f"{metric}_std"] = float(values.std())
    result["completion_gain"] = (
        result["input_mse"] - result["reconstruction_mse"]
    )
    return result


def run_completion_simulation(name: str, lec_only: bool,
                              ae_name: str, seed: int,
                              laps_per_level: int,
                              fractions: np.ndarray,
                              data_settings: dict,
                              mtl_settings: dict) -> dict:
    """Train clean once, then probe all dropout levels without learning."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    total_laps = laps_per_level * (1 + len(fractions))
    clean_laps, cue_patterns = make_clean_laps(total_laps, data_settings)
    model, ae_session = track.build_mtl(ae_name, mtl_settings)
    for lap in clean_laps[:laps_per_level]:
        track._train_lap(model, lap)

    levels = []
    probe_start = laps_per_level
    for level_index, fraction in enumerate(fractions):
        start = probe_start + level_index * laps_per_level
        targets = clean_laps[start:start + laps_per_level]
        probes = []
        masks = []
        for target in targets:
            presented, dropped = apply_neuron_dropout(
                target, fraction, lec_only=lec_only
            )
            probes.append(recall_corrupted_lap(
                model, target, presented, cue_patterns,
                data_settings["cue_positions"],
            ))
            masks.append(dropped)
        levels.append(aggregate_level(probes, fraction, masks))
    return {
        "name": name,
        "dropout_scope": "lec_only" if lec_only else "all_ei",
        "cue_pair": [1, 2],
        "levels": levels,
        "final_W_ca3_ca1": model.W_ca3_ca1.detach().cpu().numpy().copy(),
        "autoencoder_session": ae_session,
    }


def plot_completion_grid(simulation: dict, cue_positions: list[int]):
    """Plot target, degraded probe, and reconstruction at every fraction."""

    levels = simulation["levels"]
    figure, axes = plt.subplots(
        3, len(levels), figsize=(4.0 * len(levels), 11),
        sharex=True, sharey=True, constrained_layout=True,
    )
    if len(levels) == 1:
        axes = axes[:, None]
    row_specs = (
        ("mean_target", "Full target"),
        ("mean_presented", "Degraded input"),
        ("mean_reconstruction", "CA1→EO reconstruction"),
    )
    image = None
    for column, level in enumerate(levels):
        for row, (key, label) in enumerate(row_specs):
            values = level[key]
            image = axes[row, column].imshow(
                values.T, origin="lower", aspect="auto",
                interpolation="nearest", vmin=0, vmax=1, cmap="viridis",
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
                    f"{label}\nEC unit (MEC below, LEC above)"
                )
        axes[0, column].set_title(
            f"p={level['fraction']:.2f}\n"
            f"recon MSE={level['reconstruction_mse']:.3f}, "
            f"gain={level['completion_gain']:+.3f}\n"
            f"cue ID={level['cue_identity_accuracy']:.0%}",
            fontsize=10,
        )
        axes[-1, column].set_xlabel("track position (cm)")
    figure.colorbar(image, ax=axes, label="mean activity", shrink=0.82)
    figure.suptitle(
        f"{simulation['name']} | mean across "
        f"{levels[0]['num_probes']} independently masked laps per p"
    )
    return figure


def plot_completion_summary(simulations: list[dict]):
    """Compare reconstruction and cue completion as dropout increases."""

    figure, axes = plt.subplots(
        1, 3, figsize=(15, 4.5), constrained_layout=True
    )
    colors = ("tab:blue", "tab:orange")
    for simulation, color in zip(simulations, colors):
        levels = simulation["levels"]
        fractions = np.asarray([level["fraction"] for level in levels])
        label = simulation["dropout_scope"].replace("_", " ")
        for axis, metric, ylabel in (
            (axes[0], "reconstruction_mse", "full reconstruction MSE"),
            (axes[1], "lec_mse", "LEC reconstruction MSE"),
            (axes[2], "cue_identity_accuracy", "cue identity accuracy"),
        ):
            values = np.asarray([level[metric] for level in levels])
            deviation = np.asarray([
                level[f"{metric}_std"] for level in levels
            ])
            axis.plot(
                fractions, values, marker="o", color=color,
                linewidth=2, label=label,
            )
            axis.fill_between(
                fractions, values - deviation, values + deviation,
                color=color, alpha=0.15,
            )
            axis.set(xlabel="fraction of eligible neurons zeroed", ylabel=ylabel)
            axis.grid(alpha=0.2)
    axes[2].set_ylim(-0.05, 1.05)
    axes[0].legend(loc="best")
    figure.suptitle("MTL pattern completion under progressive EC degradation")
    return figure


def run_experiment(ae_name: str = "ae_3cues_0",
                   seed: int | None = 3980,
                   laps_per_level: int = 5,
                   fractions=DEFAULT_FRACTIONS,
                   data_settings: dict | None = None,
                   mtl_settings: dict | None = None) -> dict:
    if laps_per_level < 1:
        raise ValueError("laps_per_level must be at least 1")
    fractions = validate_fractions(fractions)
    resolved_seed = secrets.randbits(32) if seed is None else int(seed)
    data_settings = {
        **seq.DEFAULT_DATA_SETTINGS,
        **({} if data_settings is None else data_settings),
    }
    mtl_settings = {
        **seq.DEFAULT_MTL_SETTINGS,
        **({} if mtl_settings is None else mtl_settings),
    }
    simulations = [
        run_completion_simulation(
            "Simulation 1 — dropout across MEC and LEC",
            False, ae_name, resolved_seed, laps_per_level, fractions,
            data_settings, mtl_settings,
        ),
        run_completion_simulation(
            "Simulation 2 — dropout restricted to LEC",
            True, ae_name, resolved_seed, laps_per_level, fractions,
            data_settings, mtl_settings,
        ),
    ]
    return {
        "schema_version": 1,
        "description": (
            "Pattern completion after clean [1, 2] cue training, using "
            "progressive full-EI or LEC-only neuron dropout probes."
        ),
        "settings": {
            "ae_name": ae_name,
            "seed": resolved_seed,
            "seed_mode": "random" if seed is None else "fixed",
            "laps_per_level": laps_per_level,
            "fractions": fractions,
            "plasticity_during_degraded_probes": False,
            "mask_scope": "fixed within lap, resampled between laps",
            "data": data_settings,
            "mtl": mtl_settings,
        },
        "simulations": simulations,
    }


def main():
    args = parse_args()
    mtl_settings = {
        **seq.DEFAULT_MTL_SETTINGS,
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
        laps_per_level=args.laps_per_level,
        fractions=args.fractions,
        mtl_settings=mtl_settings,
    )
    output = analysis.save_analysis(result, args.output)
    base_figure = (
        args.figure.expanduser().resolve()
        if args.figure is not None else output.with_suffix(".png")
    )
    base_figure.parent.mkdir(parents=True, exist_ok=True)
    figures = []
    paths = []
    for simulation, suffix in zip(
            result["simulations"], ("all_ei", "lec_only")):
        figure = plot_completion_grid(
            simulation, result["settings"]["data"]["cue_positions"]
        )
        path = base_figure.with_name(
            f"{base_figure.stem}_{suffix}{base_figure.suffix}"
        )
        figure.savefig(path, dpi=180)
        figures.append(figure)
        paths.append(path)
    summary = plot_completion_summary(result["simulations"])
    summary_path = base_figure.with_name(
        f"{base_figure.stem}_summary{base_figure.suffix}"
    )
    summary.savefig(summary_path, dpi=180)
    figures.append(summary)
    paths.append(summary_path)

    print(
        f"random seed: {result['settings']['seed']} "
        f"({result['settings']['seed_mode']})"
    )
    for simulation in result["simulations"]:
        print(simulation["name"])
        for level in simulation["levels"]:
            print(
                f"  p={level['fraction']:.2f}: "
                f"MSE={level['reconstruction_mse']:.4f}, "
                f"LEC MSE={level['lec_mse']:.4f}, "
                f"gain={level['completion_gain']:+.4f}, "
                f"cue ID={level['cue_identity_accuracy']:.0%}"
            )
    print(f"results saved to {output}")
    for path in paths:
        print(f"figure saved to {path}")
    if not args.no_show:
        plt.show()
    else:
        for figure in figures:
            plt.close(figure)


if __name__ == "__main__":
    main()
