"""Run independent CMA-ES searches for multiple MTL plasticity variants."""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_SRC = os.path.abspath(__file__).split("src")[0] + "src"
sys.path.append(PROJECT_SRC)
sys.path.append(PROJECT_SRC + "/experiments")

import mtl_evolution_experiment as mee
from core.logger import logger


AVAILABLE_VARIANTS = mee.PLASTICITY_VARIANTS
DEFAULT_VARIANTS = ("base", "nois", "err1", "err2")
REPS = 1
AVERAGED_TRAJECTORY_KEYS = (
    "best_fitness",
    "generation_best",
    "population_mean_fitness",
    "sigma",
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evolve MTL parameters independently for plasticity variants."
    )
    parser.add_argument("--generations", type=int, default=128)
    parser.add_argument(
        "--repetitions",
        type=int,
        default=REPS,
        help="independent evolution runs to average per plasticity variant",
    )
    parser.add_argument("--pause", type=float, default=0.1)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="workers per search; defaults to min(population size, CPU count)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        type=str.lower,
        choices=AVAILABLE_VARIANTS,
        default=list(DEFAULT_VARIANTS),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="plot best-fitness trajectories after all searches finish",
    )
    parser.add_argument(
        "--live-plot",
        action="store_true",
        help="show the detailed live plot for each individual search",
    )
    return parser.parse_args()


def average_evolution_records(rep_results: list[dict]) -> dict:
    """Average aligned generation-level metrics from independent CMA runs."""
    if not rep_results:
        raise ValueError("at least one evolution result is required")

    reference_generations = np.asarray(
        rep_results[0].get("generations", []),
        dtype=int,
    )
    if reference_generations.size == 0:
        raise ValueError("evolution results contain no completed generations")

    for repetition, record in enumerate(rep_results[1:], start=2):
        generations = np.asarray(record.get("generations", []), dtype=int)
        if not np.array_equal(generations, reference_generations):
            raise ValueError(
                "cannot average evolution runs with different generations; "
                f"repetition 1 has {reference_generations.tolist()}, "
                f"repetition {repetition} has {generations.tolist()}"
            )

    averaged = {
        "direction": rep_results[0].get("direction"),
        "metric_name": rep_results[0].get("metric_name", "fitness"),
        "generations": reference_generations.tolist(),
        "workers": rep_results[0].get("workers", 1),
        "repetitions": len(rep_results),
        # Preserve full runs for candidate/population-level analysis. Those
        # arrays are deliberately not averaged because individual indices are
        # not aligned between independent CMA populations.
        "runs": rep_results,
    }

    for key in AVERAGED_TRAJECTORY_KEYS:
        trajectories = [np.asarray(record.get(key, []), dtype=float)
                        for record in rep_results]
        expected_shape = trajectories[0].shape
        if expected_shape != reference_generations.shape:
            raise ValueError(
                f"{key!r} must contain one value per completed generation"
            )
        if any(values.shape != expected_shape for values in trajectories[1:]):
            raise ValueError(
                f"cannot average evolution runs with different {key!r} shapes"
            )
        stacked = np.stack(trajectories, axis=0)
        averaged[key] = stacked.mean(axis=0).tolist()
        averaged[f"{key}_std"] = stacked.std(axis=0).tolist()

    final_best = np.asarray(
        [record["best_fitness"][-1] for record in rep_results],
        dtype=float,
    )
    averaged["final_best_per_run"] = final_best.tolist()
    return averaged


def plot_fitness_trajectories(records: dict, show: bool=True):
    """Plot the best weighted-accuracy history for every completed variant."""
    if not records:
        raise ValueError("cannot plot an empty collection of variant records")

    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = 0
    for plasticity, record in records.items():
        fitness = np.asarray(record.get("best_fitness", []), dtype=float)
        generations = np.asarray(
            record.get("generations", np.arange(len(fitness))),
            dtype=int,
        )
        length = min(len(generations), len(fitness))
        if length == 0:
            continue
        line, = ax.plot(
            generations[:length],
            fitness[:length],
            linewidth=2,
            label=str(plasticity).upper(),
        )
        fitness_std = np.asarray(
            record.get("best_fitness_std", []),
            dtype=float,
        )
        if len(fitness_std) >= length and record.get("repetitions", 1) > 1:
            ax.fill_between(
                generations[:length],
                np.clip(fitness[:length] - fitness_std[:length], 0.0, 1.0),
                np.clip(fitness[:length] + fitness_std[:length], 0.0, 1.0),
                color=line.get_color(),
                alpha=0.18,
            )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        raise ValueError("variant records contain no completed generations")

    ax.set_xlabel("generation")
    ax.set_ylabel("best weighted accuracy")
    ax.set_title("MTL plasticity-variant evolution (mean ± 1 SD)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(title="plasticity")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def run_variants(generations=128, pause=0.1, live_plot=False,
                 workers=None, variants=DEFAULT_VARIANTS,
                 comparison_plot=False, repetitions=REPS):
    """Run and average independent searches for each plasticity rule."""
    repetitions = int(repetitions)
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")

    records = {}
    for plasticity in variants:
        plasticity = str(plasticity).lower()
        logger(f"starting plasticity variant {plasticity.upper()}")

        rep_results = []
        for repetition in range(repetitions):
            logger(
                f"{plasticity.upper()} repetition "
                f"{repetition + 1}/{repetitions}"
            )
            res = mee.mtlsearch(
                generations=generations,
                pause=pause,
                live_plot=live_plot,
                workers=workers,
                plasticity=plasticity,
                verbose=False,
                disable=False,
                save_name=f"mtl_evolution_{plasticity}",
            )
            rep_results.append(res)

        records[plasticity] = average_evolution_records(rep_results)
        final_values = np.asarray(
            records[plasticity]["final_best_per_run"],
            dtype=float,
        )
        logger(
            f"best={final_values.mean():.4f} ± "
            f"{final_values.std():.4f}"
        )
    if comparison_plot:
        plot_fitness_trajectories(records)
    return records


def main():
    args = parse_args()
    run_variants(
        generations=args.generations,
        pause=args.pause,
        live_plot=args.live_plot,
        workers=args.workers,
        variants=args.variants,
        comparison_plot=args.plot,
        repetitions=args.repetitions,
    )
    logger("all requested plasticity variants completed")


if __name__ == "__main__":
    main()
