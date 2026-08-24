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
import ae_experiments
import _lib



AE_PARAMETER_NAMES = (
    # "encoding_dim",
    "K_ca1",
    "K_eo",
    "beta_ei",
    "beta_eo",
)
# The zero genome represents a known sensible starting configuration. CMA-ES
# searches standardized coordinates around it, while the decoder enforces the
# parameter-specific domains below.
AE_PARAMETER_CENTERS = np.array([5., 5., 25., 25.])
AE_PARAMETER_SCALES = np.array([3., 3., 20., 20.])
AE_PARAMETER_LOWER = np.array([1., 1., 1., 1.])
AE_PARAMETER_UPPER = np.array([49., 49., 256., 256.])
AE_EVALUATION_SEED = 1701



AE_SETTINGS_DATA = {
    "size": 50,
    "K": 5,

    "num_cue_patterns": 5,
    "size": 50,
    "lap_length": 50,
    "cue_positions": [10, 30],
    "cue_sigma": 5,
    "cue_beta": 20,
    "cue_alpha": 0.2,
    "mec_binarized": True,
    "mec_sigma": 5.,
    "lec_sigma": 5.,
}
DATA_LABEL = "cue"
if DATA_LABEL == "cue":
    AE_SETTINGS_SIM = {
        "data_training_size": 16,
        "data_test_size": 4,
        "epochs": 64,
        "disable": True,
        "batch_size": 32,
        "learning_rate": 1e-3,
    }
else:
    AE_SETTINGS_SIM = {
        "data_training_size": 1024,
        "data_test_size": 96,
        "epochs": 64,
        "disable": True,
        "batch_size": 64,
        "learning_rate": 1e-3,
    }


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
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()



def sanitizer(genome):
    """Decode a standardized CMA genome into valid AE hyperparameters."""
    genome = np.asarray(genome, dtype=float)
    if genome.shape != AE_PARAMETER_CENTERS.shape:
        raise ValueError(
            f"expected a genome of shape {AE_PARAMETER_CENTERS.shape}, "
            f"got {genome.shape}"
        )

    parameters = AE_PARAMETER_CENTERS + AE_PARAMETER_SCALES * genome
    parameters = np.clip(
        parameters,
        AE_PARAMETER_LOWER,
        AE_PARAMETER_UPPER,
    )
    # Both sparsity parameters index the two values surrounding the top-K
    # threshold, so they must be integral and strictly smaller than dim=50.
    parameters[[0, 1]] = np.rint(parameters[[0, 1]])
    return parameters


def evaluate_ae_individual(ind,
                           settings_sim: dict|None=None,
                           settings_data: dict|None=None):
    """Train and score one decoded autoencoder candidate."""
    settings_sim = dict(AE_SETTINGS_SIM if settings_sim is None else settings_sim)
    settings_data = dict(AE_SETTINGS_DATA if settings_data is None else settings_data)

    np.random.seed(AE_EVALUATION_SEED)
    torch.manual_seed(AE_EVALUATION_SEED)

    settings_ae = {
        "dim_ei": settings_data["size"],
        "dim_ca1": 50,
        "K_ca1": int(ind[0]),
        "K_eo": int(ind[1]),
        "beta_ei": float(ind[2]),
        "beta_eo": float(ind[3]),
        "use_bias": False,
    }
    if DATA_LABEL == "random":
        result = ae_experiments.train_random_data(
            settings_sim=settings_sim,
            settings_data=settings_data,
            settings_ae=settings_ae,
            save=False,
            plot=False,
        )
    elif DATA_LABEL == "cue":
        result = ae_experiments.train_cue_data(
            settings_sim=settings_sim,
            settings_data=settings_data,
            settings_ae=settings_ae,
            save=False,
            plot=False,
        )
    else:
        raise NameError("wrong data label")

    result = float(result)
    return result if np.isfinite(result) else 1.0

def evaluate_ae_random(population: list,
                       settings_sim: dict|None=None,
                       settings_data: dict|None=None):
    """Return final validation MSE for every decoded candidate."""

    settings_sim = dict(AE_SETTINGS_SIM if settings_sim is None else settings_sim)
    settings_data = dict(AE_SETTINGS_DATA if settings_data is None else settings_data)

    return [
        evaluate_ae_individual(ind, settings_sim, settings_data)
        for ind in tqdm(population, disable=settings_sim["disable"])
    ]

""" autoencoder search """

def aesearch(generations=64, pause=0.01, live_plot=True, workers=None, save: bool=False,
             settings_sim: dict|None=None, settings_data: dict|None=None,
             save_name: str="ae_evolution_1", verbose: bool=True):

    settings_sim = dict(AE_SETTINGS_SIM if settings_sim is None else settings_sim)
    settings_data = dict(AE_SETTINGS_DATA if settings_data is None else settings_data)

    parameter_spec_lengths = {
        "names": len(AE_PARAMETER_NAMES),
        "centers": len(AE_PARAMETER_CENTERS),
        "scales": len(AE_PARAMETER_SCALES),
        "lower bounds": len(AE_PARAMETER_LOWER),
        "upper bounds": len(AE_PARAMETER_UPPER),
    }
    if len(set(parameter_spec_lengths.values())) != 1:
        raise ValueError(
            "inconsistent autoencoder parameter specification lengths: "
            + ", ".join(
                f"{name}={length}"
                for name, length in parameter_spec_lengths.items()
            )
        )
    num_parameters = len(AE_PARAMETER_NAMES)

    # Match the population size used internally by the C++ CMAES instance.
    population_size = 4 + int(3 * np.log(num_parameters))
    settings = {
        "num_parameters": num_parameters,
        "generations": generations,
        "population_size": population_size,
        "pause_time": pause,
        "direction": "minimize",
        "verbose": verbose,
        "metric_name": "validation_mse",
        "workers": workers,
    }
    evaluation_settings = dict(settings_sim)
    if workers is None or workers > 1:
        # A process per CPU core should not make every process compete for the
        # same CUDA/MPS device. A serial run may still use the best accelerator.
        evaluation_settings["device"] = "cpu"
    individual_evaluator = functools.partial(
        evaluate_ae_individual,
        settings_sim=evaluation_settings,
        settings_data=settings_data,
    )
    batch_evaluator = functools.partial(
        evaluate_ae_random,
        settings_sim=evaluation_settings,
        settings_data=settings_data,
    )
    record = _lib.evolution_run(settings=settings,
                                evaluate=batch_evaluator,
                                evaluate_individual=individual_evaluator,
                                sanitizer=sanitizer,
                                live_plot=live_plot)

    if save:
        import time

        logs = {
            "date": f"{time.localtime().tm_mday}.{time.localtime().tm_mon}.{time.localtime().tm_year}",
            "settings": settings,
            "best": record["best_candidates"][-1].tolist(),
            "best_parameters": dict(zip(
                AE_PARAMETER_NAMES,
                record["best_candidates"][-1].tolist(),
            )),
            "validation_mse": record["best_fitness"][-1],
            "workers": record["workers"],
        }

        _lib.save_genome(info=logs, name=save_name)
        print(f"saved as {save_name}")

    return record


if __name__ == "__main__":
    args = parse_args()
    aesearch(
        generations=args.generations,
        pause=args.pause,
        live_plot=not args.no_plot,
        save=args.save,
        workers=args.workers,
        save_name=f"ae_{DATA_LABEL}_x"
    )
    print("[done]")
