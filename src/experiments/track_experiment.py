"""Continual MTL learning on a circular track with periodically swapped cues.

The default experiment presents 50 laps of 50 positions.  Two fixed sensory
cues exchange their track positions every 10 laps.  Each stimulus is learned
once, after which all laps encountered so far are recalled without learning.
Results from every repetition are saved as JSON for downstream analysis.
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

from core import ae_tools, datagen, models  # noqa: E402


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / \
    "track_experiment.json"


DEFAULT_DATA_SETTINGS = {
    "num_laps": 50,
    "lap_length": 50,
    "size": 50,
    "cue_positions": [10, 30],
    "swap_every": 10,
    "cue_sigma": 4.0,
    "cue_beta": 40.0,
    "cue_alpha": 0.1,
    "mec_binarized": True,
    "mec_sigma": 5.0,
    "lec_sigma": 5.0,
}


# Defaults are initialized from the latest saved cue-evolution result.  The
# command-line options make every value replaceable without editing this file.
DEFAULT_MTL_SETTINGS = {
    "K_ca3": 2,
    "dim_ca3": 50,
    "beta_ca3": 100.61739206314087,
    "beta_ca1": 11.801433563232422,
    "alpha": 0.09247108435630799,
    "nb_ei_ca3": 19,
    "num_swaps_ca1": 0,
    "num_swaps_ca3": 0,
    "random_IS": False,
    "plasticity": "base",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train MTL continually on a circular track whose two cue "
            "identities swap positions periodically."
        )
    )
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--laps", type=int, default=50)
    parser.add_argument("--lap-length", type=int, default=50)
    parser.add_argument("--swap-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=384)
    parser.add_argument("--ae-name", default="ae_cue_nb_5")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--figure",
        type=Path,
        default=None,
        help="figure path; defaults to the output JSON path with .png suffix",
    )
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--k-ca3", type=int, default=DEFAULT_MTL_SETTINGS["K_ca3"])
    parser.add_argument(
        "--beta-ca3", type=float, default=DEFAULT_MTL_SETTINGS["beta_ca3"]
    )
    parser.add_argument(
        "--beta-ca1", type=float, default=DEFAULT_MTL_SETTINGS["beta_ca1"]
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_MTL_SETTINGS["alpha"])
    parser.add_argument(
        "--nb-ei-ca3", type=int, default=DEFAULT_MTL_SETTINGS["nb_ei_ca3"]
    )
    parser.add_argument(
        "--num-swaps-ca3",
        type=int,
        default=DEFAULT_MTL_SETTINGS["num_swaps_ca3"],
    )
    parser.add_argument(
        "--num-swaps-ca1",
        type=int,
        default=DEFAULT_MTL_SETTINGS["num_swaps_ca1"],
    )
    parser.add_argument(
        "--plasticity",
        choices=("base", "nois", "isout", "err1", "err2"),
        default=DEFAULT_MTL_SETTINGS["plasticity"],
    )
    return parser.parse_args()


def cue_assignments(num_laps: int, swap_every: int) -> list[list[int]]:
    """Return cue identities at the two positions for every lap."""

    if num_laps < 1:
        raise ValueError("num_laps must be at least 1")
    if swap_every < 1:
        raise ValueError("swap_every must be at least 1")
    return [
        [0, 1] if (lap // swap_every) % 2 == 0 else [1, 0]
        for lap in range(num_laps)
    ]


def make_swapping_cue_track(settings: dict) -> tuple[np.ndarray, list[list[int]]]:
    """Generate complete laps with two cues swapping identity by block."""

    size = int(settings["size"])
    if size < 2 or size % 2:
        raise ValueError("size must be an even integer of at least 2")
    cue_positions = list(settings["cue_positions"])
    if len(cue_positions) != 2:
        raise ValueError("this experiment requires exactly two cue positions")

    assignments = cue_assignments(
        int(settings["num_laps"]), int(settings["swap_every"])
    )
    cue_patterns = datagen.make_cues(n=2, size=size // 2, fixed=True)
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


def build_mtl(ae_name: str, settings: dict) -> tuple[models.MTL, dict]:
    """Construct a fresh MTL model around a frozen autoencoder checkpoint."""

    autoencoder, ae_session = ae_tools.load_autoencoder(name=ae_name)
    params = autoencoder.get_weights(bias=False)
    model = models.MTL(
        W_ei_ca1=params[0],
        W_ca1_eo=params[1],
        K_ca1=autoencoder._K_ca1,
        K_eo=autoencoder._K_eo,
        K_ca3=int(settings["K_ca3"]),
        dim_ca3=int(settings["dim_ca3"]),
        beta_is=autoencoder._beta_ei,
        beta_ca3=float(settings["beta_ca3"]),
        beta_ca1=float(settings["beta_ca1"]),
        beta_eo=autoencoder._beta_eo,
        alpha=float(settings["alpha"]),
        nb_ei_ca3=int(settings["nb_ei_ca3"]),
        num_swaps_ca3=int(settings["num_swaps_ca3"]),
        num_swaps_ca1=int(settings["num_swaps_ca1"]),
        B_ei_ca1=params[2],
        B_ca1_eo=params[3],
        random_IS=bool(settings["random_IS"]),
        plasticity=str(settings["plasticity"]),
    )
    return model, ae_session


def _vector(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().reshape(-1).cpu().numpy().copy()


def _train_lap(model: models.MTL, lap: np.ndarray) -> dict[str, np.ndarray]:
    """Present one lap once and return activity from each training pass."""

    model.reset()
    model.resume_lr()
    activity = {key: [] for key in ("x_ei", "ca3", "IS", "ca1", "eo")}
    with torch.no_grad():
        for sample in lap:
            x = torch.as_tensor(sample, dtype=torch.float32).reshape(-1, 1)
            model(x)
            activity["x_ei"].append(_vector(x))
            activity["ca3"].append(_vector(model._ca3))
            activity["IS"].append(_vector(model.recordings["IS"][-1]))
            activity["ca1"].append(_vector(model._ca1))
            activity["eo"].append(_vector(model._eo))
    return {key: np.asarray(values) for key, values in activity.items()}


def _recall_lap(model: models.MTL, lap: np.ndarray,
                return_activity: bool = False):
    """Recall one complete lap without modifying the learned weights."""

    model.reset()
    model.pause_lr()
    activity = {key: [] for key in ("x_ei", "ca3", "IS", "ca1", "eo")}
    with torch.no_grad():
        for sample in lap:
            x = torch.as_tensor(sample, dtype=torch.float32).reshape(-1, 1)
            model(x)
            if return_activity:
                activity["x_ei"].append(_vector(x))
                activity["ca3"].append(_vector(model._ca3))
                activity["IS"].append(_vector(model.recordings["IS"][-1]))
                activity["ca1"].append(_vector(model._ca1))
                activity["eo"].append(_vector(model._eo))

    reconstructed = np.stack([
        _vector(value) for value in model.recordings["eo"]
    ])
    sample_mse = np.mean((lap - reconstructed) ** 2, axis=1)
    accuracy = float(np.mean(np.clip(1.0 - sample_mse, 0.0, 1.0)))
    numerator = np.sum(lap * reconstructed, axis=1)
    denominator = np.linalg.norm(lap, axis=1) * np.linalg.norm(
        reconstructed, axis=1
    )
    cosine = float(np.mean(numerator / np.maximum(denominator, 1e-8)))

    if not return_activity:
        model.reset()
        return accuracy, cosine

    arrays = {key: np.asarray(values) for key, values in activity.items()}
    model.reset()
    return accuracy, cosine, arrays


def _model_weights(model: models.MTL) -> dict[str, np.ndarray]:
    names = (
        "W_ei_ca3",
        "W_ei_ca1",
        "W_ca3_ca1",
        "W_ca1_eo",
        "B_ei_ca1",
        "B_ca1_eo",
    )
    return {
        name: getattr(model, name).detach().cpu().numpy().copy()
        for name in names
    }


def run_repetition(rep: int, seed: int, ae_name: str,
                   data_settings: dict, mtl_settings: dict,
                   quiet: bool = False) -> tuple[dict, dict]:
    """Run one independent continual-learning simulation."""

    repetition_seed = int(seed + rep)
    np.random.seed(repetition_seed)
    torch.manual_seed(repetition_seed)

    stimuli, assignments = make_swapping_cue_track(data_settings)
    model, ae_session = build_mtl(ae_name, mtl_settings)
    num_laps = len(stimuli)
    recall_accuracy = np.zeros((num_laps, num_laps), dtype=np.float32)
    recall_cosine = np.zeros((num_laps, num_laps), dtype=np.float32)
    training_activity = {
        key: [] for key in ("x_ei", "ca3", "IS", "ca1", "eo")
    }
    weight_trajectory = []

    lap_iterator = tqdm(
        range(num_laps),
        desc=f"repetition {rep + 1}",
        leave=False,
        disable=quiet,
    )
    for learned_lap in lap_iterator:
        lap_activity = _train_lap(model, stimuli[learned_lap])
        for key, values in lap_activity.items():
            training_activity[key].append(values)
        weight_trajectory.append(
            model.W_ca3_ca1.detach().cpu().numpy().copy()
        )

        # Recall swaps are stochastic.  Preserve the RNG state so diagnostic
        # evaluations do not alter the noise subsequently seen during learning.
        training_rng_state = torch.random.get_rng_state()
        for recalled_lap in range(learned_lap + 1):
            accuracy, cosine = _recall_lap(model, stimuli[recalled_lap])
            recall_accuracy[learned_lap, recalled_lap] = accuracy
            recall_cosine[learned_lap, recalled_lap] = cosine
        torch.random.set_rng_state(training_rng_state)

    final_recall_activity = {
        key: [] for key in ("x_ei", "ca3", "IS", "ca1", "eo")
    }
    for lap in stimuli:
        _, _, lap_activity = _recall_lap(
            model, lap, return_activity=True
        )
        for key, values in lap_activity.items():
            final_recall_activity[key].append(values)

    repetition = {
        "repetition": rep,
        "seed": repetition_seed,
        "cue_assignments": assignments,
        "recall_accuracy": recall_accuracy,
        "recall_cosine": recall_cosine,
        "training_activity": {
            key: np.asarray(values) for key, values in training_activity.items()
        },
        "weight_trajectory": {
            "W_ca3_ca1_after_lap": np.asarray(weight_trajectory)
        },
        "final_recall_activity": {
            key: np.asarray(values)
            for key, values in final_recall_activity.items()
        },
        "final_weights": _model_weights(model),
    }
    return repetition, ae_session


def aggregate_repetitions(repetitions: list[dict]) -> dict[str, np.ndarray]:
    """Compute cross-repetition means and standard deviations."""

    accuracy = np.stack([rep["recall_accuracy"] for rep in repetitions])
    cosine = np.stack([rep["recall_cosine"] for rep in repetitions])
    valid = np.tri(accuracy.shape[1], dtype=bool)
    return {
        "evaluated_mask": valid,
        "recall_accuracy_mean": accuracy.mean(axis=0),
        "recall_accuracy_std": accuracy.std(axis=0),
        "recall_cosine_mean": cosine.mean(axis=0),
        "recall_cosine_std": cosine.std(axis=0),
    }


def _trajectory(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return immediate, all-seen, and oldest-lap recall trajectories."""

    immediate = np.diag(matrix)
    seen_mean = np.asarray([
        matrix[lap, :lap + 1].mean() for lap in range(len(matrix))
    ])
    oldest = matrix[:, 0]
    return immediate, seen_mean, oldest


def plot_results(repetitions: list[dict], aggregate: dict,
                 data_settings: dict):
    """Create a representative retention and reconstruction summary."""

    accuracy_stack = np.stack([
        rep["recall_accuracy"] for rep in repetitions
    ])
    trajectories = np.stack([
        np.stack(_trajectory(matrix)) for matrix in accuracy_stack
    ])
    mean_trajectory = trajectories.mean(axis=0)
    std_trajectory = trajectories.std(axis=0)
    laps = np.arange(1, accuracy_stack.shape[1] + 1)

    final_seen = trajectories[:, 1, -1]
    representative_index = int(np.argmin(np.abs(
        final_seen - final_seen.mean()
    )))
    representative = repetitions[representative_index]
    oldest_original = representative["training_activity"]["x_ei"][0]
    oldest_reconstruction = representative["final_recall_activity"]["eo"][0]

    figure, axes = plt.subplots(
        2, 2, figsize=(14, 10), constrained_layout=True
    )
    labels = ("newly learned lap", "all encountered laps", "oldest lap")
    colors = ("tab:green", "tab:blue", "tab:red")
    for index, (label, color) in enumerate(zip(labels, colors)):
        axes[0, 0].plot(
            laps, mean_trajectory[index], color=color, label=label
        )
        axes[0, 0].fill_between(
            laps,
            mean_trajectory[index] - std_trajectory[index],
            mean_trajectory[index] + std_trajectory[index],
            color=color,
            alpha=0.16,
        )
    for swap in range(
            int(data_settings["swap_every"]),
            int(data_settings["num_laps"]),
            int(data_settings["swap_every"])):
        axes[0, 0].axvline(swap + 0.5, color="0.5", linestyle=":", alpha=0.6)
    axes[0, 0].set(
        xlabel="lap learned",
        ylabel="reconstruction accuracy (1 - MSE)",
        ylim=(0.0, 1.01),
        title="Continual-learning retention",
    )
    axes[0, 0].grid(alpha=0.2)
    axes[0, 0].legend(loc="best")

    mean_matrix = aggregate["recall_accuracy_mean"]
    masked_matrix = np.ma.masked_where(
        ~aggregate["evaluated_mask"], mean_matrix
    )
    accuracy_image = axes[0, 1].imshow(
        masked_matrix,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    axes[0, 1].set(
        xlabel="recalled lap",
        ylabel="lap learned so far",
        title="Mean recall across repetitions",
    )
    figure.colorbar(
        accuracy_image, ax=axes[0, 1], label="accuracy", shrink=0.9
    )

    extent = (0, len(oldest_original), 0, oldest_original.shape[1])
    for axis, values, title in (
            (axes[1, 0], oldest_original, "Oldest lap: original stimuli"),
            (
                axes[1, 1],
                oldest_reconstruction,
                "Oldest lap: recall after final lap",
            )):
        image = axis.imshow(
            values.T,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=extent,
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        axis.axhline(
            values.shape[1] / 2,
            color="white",
            linestyle="--",
            linewidth=1,
        )
        axis.set(
            xlabel="position around circular track",
            ylabel="EC unit (MEC below, LEC above)",
            title=title,
        )
    figure.colorbar(
        image, ax=axes[1, :], label="activity", shrink=0.9
    )
    figure.suptitle(
        "MTL cue-swap track experiment | dotted lines mark cue swaps | "
        f"representative repetition {representative_index + 1}"
    )
    return figure


def _json_ready(value):
    """Convert NumPy-heavy results into finite, compact JSON values."""

    if isinstance(value, np.ndarray):
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


def save_results(results: dict, output: Path) -> Path:
    """Save the complete experiment in notebook-friendly JSON."""

    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(
            _json_ready(results),
            file,
            allow_nan=False,
        )
        file.write("\n")
    return output


def run_experiment(reps: int = 5, seed: int = 3980,
                   ae_name: str = "ae_cue_nb_5",
                   data_settings: dict | None = None,
                   mtl_settings: dict | None = None,
                   quiet: bool = False):
    """Run and aggregate all independent repetitions."""

    if reps < 1:
        raise ValueError("reps must be at least 1")
    data_settings = {
        **DEFAULT_DATA_SETTINGS,
        **({} if data_settings is None else data_settings),
    }
    mtl_settings = {
        **DEFAULT_MTL_SETTINGS,
        **({} if mtl_settings is None else mtl_settings),
    }

    repetitions = []
    ae_session = None
    for rep in range(reps):
        repetition, loaded_session = run_repetition(
            rep=rep,
            seed=seed,
            ae_name=ae_name,
            data_settings=data_settings,
            mtl_settings=mtl_settings,
            quiet=quiet,
        )
        repetitions.append(repetition)
        if ae_session is None:
            ae_session = loaded_session

    aggregate = aggregate_repetitions(repetitions)
    results = {
        "schema_version": 1,
        "description": (
            "Continual MTL learning on circular laps with two cue identities "
            "swapped periodically between two positions."
        ),
        "metric_definition": {
            "recall_accuracy": "mean over positions of clip(1 - sample MSE, 0, 1)",
            "recall_cosine": "mean cosine similarity over positions",
            "matrix_axes": "[lap learned so far, recalled lap]",
        },
        "settings": {
            "reps": reps,
            "seed": seed,
            "ae_name": ae_name,
            "data": data_settings,
            "mtl": mtl_settings,
        },
        "autoencoder_session": ae_session,
        "cue_assignments": cue_assignments(
            int(data_settings["num_laps"]),
            int(data_settings["swap_every"]),
        ),
        "aggregate": aggregate,
        "repetitions": repetitions,
    }
    return results


def main():
    args = parse_args()
    data_settings = {
        **DEFAULT_DATA_SETTINGS,
        "num_laps": args.laps,
        "lap_length": args.lap_length,
        "swap_every": args.swap_every,
    }
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
    results = run_experiment(
        reps=args.reps,
        seed=args.seed,
        ae_name=args.ae_name,
        data_settings=data_settings,
        mtl_settings=mtl_settings,
        quiet=args.quiet,
    )
    figure = plot_results(
        results["repetitions"], results["aggregate"], data_settings
    )
    output = save_results(results, args.output)
    figure_path = (
        args.figure.expanduser().resolve()
        if args.figure is not None else output.with_suffix(".png")
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=180)
    print(f"results saved to {output}")
    print(f"figure saved to {figure_path}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
