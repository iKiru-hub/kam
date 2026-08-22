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
import time

sys.path.append(os.path.abspath(__file__).split("src")[0] + "src/experiments")

import mtl_experiments
import _lib

""" example """

class CurveObjective:
    """Example black-box objective used by ``mocksim``."""

    def __init__(self, num_parameters: int):
        x = np.linspace(0.0, 31.4, num_parameters)
        self.target = x
        self.m = np.random.randn(num_parameters, num_parameters)

    def __call__(self, population):
        population = np.asarray(population, dtype=float)
        fitness = []
        for ind in population:
            fitness += [np.mean((ind - self.target)**2)]

        return fitness
        # return np.mean((population - self.values) ** 2, axis=1)


def mocksim(num_parameters=64, generations=300, pause=0.01, live_plot=True):
    population_size = 4 + int(3 * np.log(num_parameters))
    settings = {
        "num_parameters": num_parameters,
        "generations": generations,
        "population_size": population_size,
        "pause_time": pause,
        "direction": "minimize",
    }
    return _lib.evolution_run(
        settings=settings,
        evaluate=CurveObjective(num_parameters),
        live_plot=live_plot,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evolve MTL hyperparameters with CMA-ES."
    )
    parser.add_argument("--generations", type=int, default=196)
    parser.add_argument("--pause", type=float, default=0.01)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="parallel workers; defaults to min(population size, CPU count)",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


""" mtl """

MTL_PARAMETER_NAMES = (
    "K_ca3",
    "K_lat",
    "K_out",
    "beta_is",
    "beta_ca3",
    "beta_ca1",
    "alpha",
    "nb_ei_ca3",
)
MTL_PARAMETER_CENTERS = np.array([10., 10., 10., 10., 10., 10., 0.166, 10.])
MTL_PARAMETER_SCALES = np.array([4., 5., 5., 15., 50., 10., 0.10, 10.])
MTL_PARAMETER_LOWER = np.array([2., 2., 2., 2., 2., 2., 1e-4, 2.])
MTL_PARAMETER_UPPER = np.array([49., 49., 49., 196., 196., 196., 1., 49])
MTL_EVALUATION_SEED = 3980

HORIZON = 64
PLASTICITY_VARIANTS = ("base", "nois", "isout", "err1", "err2")


def sanitizer(genome):
    """Decode a standardized CMA genome into bounded MTL parameters."""
    genome = np.asarray(genome, dtype=float)
    if genome.shape != MTL_PARAMETER_CENTERS.shape:
        raise ValueError(
            f"expected a genome of shape {MTL_PARAMETER_CENTERS.shape}, "
            f"got {genome.shape}"
        )

    parameters = MTL_PARAMETER_CENTERS + MTL_PARAMETER_SCALES * genome
    parameters = np.clip(
        parameters,
        MTL_PARAMETER_LOWER,
        MTL_PARAMETER_UPPER,
    )
    # Top-k values and the number of EI-to-CA3 connections are discrete.
    parameters[[0, 1, 2, 7]] = np.rint(parameters[[0, 1, 2, 7]])
    return parameters




def evaluate_mtl_individual(ind: list, plasticity: str="base"):
    """Train and score one decoded MTL candidate."""

    settings_sim = {
        "data_training_size": 96,
        "criterion": mtl_experiments.mtlct.cosine_criterion,
        "reps": 1,
        "use_bias": False,
        "ae_name": "ae_random_nb_0"
    }

    settings_data = {
        "size": 50,
        "K": 5,
    }

    np.random.seed(MTL_EVALUATION_SEED)
    torch.manual_seed(MTL_EVALUATION_SEED)

    for i in range(len(ind)):
        if np.isnan(ind[i]) and i != 6:
            ind[i] = 2


    settings_mtl = {
        "K_ca3": ind[0],
        "K_lat": ind[1],
        "K_out": ind[2],
        "dim_ca3": 50,
        "beta_is": ind[3],
        "beta_ca3": ind[4],
        "beta_ca1": ind[5],
        "beta_eo": 20,
        "alpha": ind[6],
        "nb_ei_ca3": int(ind[7]),
        "num_swaps_ca1": 1,
        "num_swaps_ca3": 1,
        "random_IS": False,
        "plasticity": plasticity
    }

    results = mtl_experiments.train_mtl_random_data(settings_sim=settings_sim,
                                                    settings_data=settings_data,
                                                    settings_mtl=settings_mtl,
                                                    plot=False,
                                                    disable=True)
    score = float(_lib.exp_eval(results, sigma=HORIZON).mean())
    if not np.isfinite(score):
        score = 0.
    return float(np.clip(score, 0., 1.))

def evaluate_mtl_random(population: list, plasticity: str="base"):
    return [evaluate_mtl_individual(ind=ind, plasticity=plasticity) for ind in tqdm(population)]


def mtlsearch(num_parameters: int|None=None, generations: int=96, pause: float=0.01,
              live_plot: bool=False, workers=None, plasticity: str="base",
              save_name: str="mtl_evolution_1", verbose: bool=True):
    plasticity = str(plasticity).lower()
    if plasticity not in PLASTICITY_VARIANTS:
        raise ValueError(
            f"plasticity must be one of {PLASTICITY_VARIANTS}, "
            f"got {plasticity!r}"
        )

    expected_parameters = len(MTL_PARAMETER_NAMES)
    if num_parameters is None:
        num_parameters = expected_parameters
    elif num_parameters != expected_parameters:
        raise ValueError(
            f"MTL search requires {expected_parameters} parameters, "
            f"got {num_parameters}"
        )

    # Match the population size used internally by the C++ CMAES instance.
    population_size = 4 + int(3 * np.log(num_parameters))
    settings = {
        "num_parameters": num_parameters,
        "generations": generations,
        "population_size": population_size,
        "pause_time": pause,
        "direction": "maximize",
        "verbose": verbose,
        "metric_name": "weighted_accuracy",
        "workers": workers,
        "plasticity": plasticity,
    }

    batch_evaluator = functools.partial(
        evaluate_mtl_random,
        plasticity=plasticity,
    )
    individual_evaluator = functools.partial(
        evaluate_mtl_individual,
        plasticity=plasticity,
    )
    record = _lib.evolution_run(settings=settings,
                                evaluate=batch_evaluator,
                                evaluate_individual=individual_evaluator,
                                sanitizer=sanitizer,
                                live_plot=live_plot)

    logs = {
        "date": f"{time.localtime().tm_mday}.{time.localtime().tm_mon}.{time.localtime().tm_year}",
        "settings": settings,
        "best": record["best_candidates"][-1].tolist(),
        "best_parameters": dict(zip(
            MTL_PARAMETER_NAMES,
            record["best_candidates"][-1].tolist(),
        )),
        "fitness": record["best_fitness"][-1],
        "workers": record["workers"],
    }

    if save_name is not None:
        _lib.save_genome(info=logs, name=save_name)
    return record


if __name__ == "__main__":
    args = parse_args()
    mtlsearch(
        generations=args.generations,
        pause=args.pause,
        live_plot=not args.no_plot,
        workers=args.workers,
    )
    print("[done]")
