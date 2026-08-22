"""Reusable live logging for a black-box CMA-ES optimization."""

import argparse
import functools
import sys, os
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(__file__).split("src")[0] + "src/experiments")
PROJECT_SRC = os.path.abspath(__file__).split("src")[0] + "src"

import ae_experiments
import ae_evolution_experiment as aee
import _lib
from core.logger import logger
from core.utils import tqdm_enumerate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evolve sparse-autoencoder hyperparameters with CMA-ES."
    )
    parser.add_argument("--generations", type=int, default=64)
    parser.add_argument("--pause", type=float, default=0.01)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="parallel workers; defaults to min(population size, CPU count)",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


AE_SETTINGS_SIM = {
    "data_training_size": 1024,
    "data_test_size": 128,
    "epochs": 128,
    "disable": True,
    "batch_size": 16,
    "learning_rate": 1e-3,
}

GENERATIONS = 128


def _main():

    num = 16
    variable = np.linspace(2, 49, num).astype(int)

    logs = {"f": [], "p": []}
    for i, v in tqdm_enumerate(variable):
        settings_data = {"size": 50, "K": v}

        results = aee.aesearch(
            generations=GENERATIONS,
            live_plot=False,
            settings_sim=aee.AE_SETTINGS_SIM,
            settings_data=settings_data,
            verbose=False,
            save_name="ae_vars_0"
        )

        logger(f"[K={v}] best={results['best_fitness'][-1]:.4f}")
        logs["f"] += [results['best_fitness'][-1]]
        logs["p"] += [results["best_candidates"][-1].tolist()]

    # plot
    logsp = np.array(logs["p"])
    fig, axs = plt.subplots(len(logsp[0])+1, 1, sharex=True)
    fig.suptitle(f"each search over {GENERATIONS} generations")
    for i, ax in enumerate(axs):
        print(i)
        if i == 0:
            ax.plot(range(num), logs["f"], lw=2)
            ax.set_ylabel("[fitness] loss")
        else:
            ax.plot(range(num), logsp[:, i-1])
            ax.set_ylabel(f"{aee.AE_PARAMETER_NAMES[i-1]}")
            # ax.set_ylim((0, aee.AE_PARAMETER_UPPER[i-1]))

        ax.set_xticks(range(num))
        ax.set_xticklabels(variable)
        ax.set_xlabel("K")
        ax.grid()

    plt.show()




if __name__ == "__main__":
    _main()
    print("[done]")
