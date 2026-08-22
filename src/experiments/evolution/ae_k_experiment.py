"""Measure AE evolution sensitivity to the sparsity of the training data."""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(__file__).split("src")[0] + "src/experiments")

import ae_evolution_experiment as aee
from core.logger import logger
from core.utils import tqdm_enumerate


GENERATIONS = 96
REPS = 5
NUM_K_VALUES = 10

AE_SETTINGS_SIM = {
    "data_training_size": 728,
    "data_test_size": 96,
    "epochs": 96,
    "disable": True,
    "batch_size": 16,
    "learning_rate": 1e-3,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Average AE hyperparameter evolution over training-data K."
    )
    parser.add_argument("--generations", type=int, default=GENERATIONS)
    parser.add_argument("--repetitions", type=int, default=REPS)
    parser.add_argument("--pause", type=float, default=0.01)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="workers per search; defaults to min(population size, CPU count)",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def make_k_values(num_values=NUM_K_VALUES) -> np.ndarray:
    """Return unique integer sparsity values valid for 50-dimensional data."""
    num_values = int(num_values)
    if num_values < 1:
        raise ValueError("num_values must be at least 1")
    return np.unique(np.rint(np.linspace(2, 49, num_values)).astype(int))


def average_evolution_runs(runs: list[dict], data_k: int) -> dict:
    """Average aligned AE evolution trajectories and final optima."""
    if not runs:
        raise ValueError("at least one AE evolution run is required")

    generations = np.asarray(runs[0].get("generations", []), dtype=int)
    if generations.size == 0:
        raise ValueError("AE evolution run contains no completed generations")

    trajectories = []
    final_candidates = []
    for repetition, run in enumerate(runs, start=1):
        run_generations = np.asarray(run.get("generations", []), dtype=int)
        if not np.array_equal(run_generations, generations):
            raise ValueError(
                "cannot average runs with different generations; "
                f"repetition {repetition} is not aligned"
            )
        fitness = np.asarray(run.get("best_fitness", []), dtype=float)
        if fitness.shape != generations.shape:
            raise ValueError(
                "best_fitness must contain one value per completed generation"
            )
        trajectories.append(fitness)
        final_candidates.append(
            np.asarray(run["best_candidates"][-1], dtype=float)
        )

    trajectories = np.stack(trajectories, axis=0)
    final_candidates = np.stack(final_candidates, axis=0)
    final_best = trajectories[:, -1]
    return {
        "data_k": int(data_k),
        "generations": generations.tolist(),
        "repetitions": len(runs),
        "best_fitness": trajectories.mean(axis=0).tolist(),
        "best_fitness_std": trajectories.std(axis=0).tolist(),
        "final_best_per_run": final_best.tolist(),
        "final_best_mean": float(final_best.mean()),
        "final_best_std": float(final_best.std()),
        "best_candidate_mean": final_candidates.mean(axis=0).tolist(),
        "best_candidate_std": final_candidates.std(axis=0).tolist(),
        "runs": runs,
    }


def plot_k_results(records: dict, show=True):
    """Plot mean final loss and evolved parameters with ±1 SD error bars."""
    if not records:
        raise ValueError("cannot plot empty AE K results")

    k_values = np.asarray(sorted(records), dtype=int)
    ordered = [records[int(value)] for value in k_values]
    parameter_count = len(aee.AE_PARAMETER_NAMES)
    fig, axes = plt.subplots(
        parameter_count + 1,
        1,
        figsize=(9, 2.7 * (parameter_count + 1)),
        sharex=True,
    )
    fig.suptitle("AE evolution sensitivity to training-data sparsity")

    final_mean = np.asarray([record["final_best_mean"] for record in ordered])
    final_std = np.asarray([record["final_best_std"] for record in ordered])
    axes[0].errorbar(
        k_values,
        final_mean,
        yerr=final_std,
        marker="o",
        linewidth=2,
        capsize=3,
    )
    axes[0].set_ylabel("validation MSE\n(lower is better)")

    candidate_mean = np.asarray(
        [record["best_candidate_mean"] for record in ordered],
        dtype=float,
    )
    candidate_std = np.asarray(
        [record["best_candidate_std"] for record in ordered],
        dtype=float,
    )
    for index, name in enumerate(aee.AE_PARAMETER_NAMES):
        axes[index + 1].errorbar(
            k_values,
            candidate_mean[:, index],
            yerr=candidate_std[:, index],
            marker="o",
            capsize=3,
        )
        axes[index + 1].set_ylabel(name)

    for axis in axes:
        axis.grid(alpha=0.3)
    axes[-1].set_xticks(k_values)
    axes[-1].set_xlabel("number of active inputs in training data (K)")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


def run_k_experiment(generations=GENERATIONS, repetitions=REPS, pause=0.01,
                     workers=None, live_plot=False, comparison_plot=True,
                     k_values=None, settings_sim=None):
    """Run repeated AE searches for each requested training-data K value."""
    repetitions = int(repetitions)
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    if k_values is None:
        k_values = make_k_values()
    k_values = np.asarray(k_values, dtype=int)
    if k_values.ndim != 1 or k_values.size == 0:
        raise ValueError("k_values must be a non-empty one-dimensional sequence")
    if np.any((k_values < 1) | (k_values > 49)):
        raise ValueError("every training-data K must be between 1 and 49")

    simulation = dict(AE_SETTINGS_SIM if settings_sim is None else settings_sim)
    records = {}
    logger("AE evolution sensitivity to the training-data K parameter")
    for _, data_k in tqdm_enumerate(k_values):
        logger(f"starting data K={data_k}")
        runs = []
        for repetition in range(repetitions):
            logger(f"K={data_k} repetition {repetition + 1}/{repetitions}")
            runs.append(aee.aesearch(
                generations=generations,
                pause=pause,
                live_plot=live_plot,
                workers=workers,
                settings_sim=simulation,
                settings_data={"size": 50, "K": int(data_k)},
                verbose=False,
                save=False,
            ))

        records[int(data_k)] = average_evolution_runs(runs, int(data_k))
        aggregate = records[int(data_k)]
        logger(
            f"[K={data_k}] best validation MSE="
            f"{aggregate['final_best_mean']:.4f} ± "
            f"{aggregate['final_best_std']:.4f}"
        )

    if comparison_plot:
        plot_k_results(records)
    return records


def main():
    args = parse_args()
    run_k_experiment(
        generations=args.generations,
        repetitions=args.repetitions,
        pause=args.pause,
        workers=args.workers,
        live_plot=False,
        comparison_plot=not args.no_plot,
    )
    logger("[done]")


if __name__ == "__main__":
    main()
