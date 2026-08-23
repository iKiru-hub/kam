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
import mtl_cue_experiments
import _lib


""" settings """

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

HORIZON = 128
DATA_TRAINING_SIZE = 4
# DATA_LABEL = "random"
DATA_LABEL = "cue"
# EVAL_FUNC = _lib.exp_eval
EVAL_FUNC = _lib.id_eval
# EVAL_FUNC = _lib.mean_eval
PLASTICITY_VARIANTS = ("base", "nois", "isout", "err1", "err2")
_PLASTICITY = "err2"

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




def evaluate_mtl_individual(ind: list, plasticity: str="base",
                            horizon: float=HORIZON):
    """Train and score one decoded MTL candidate."""

    if DATA_LABEL == "random":
        ae_name = "ae_random_nb_0"
    elif DATA_LABEL == "cue":
        ae_name = "ae_cue_nb_0"
    else:
        raise NameError("wrong data label")

    settings_sim = {
        "data_training_size": DATA_TRAINING_SIZE,
        "criterion": mtl_experiments.mtlct.cosine_criterion,
        "reps": 3,
        "use_bias": False,
        "disable": True,
        "plot": False,
        "ae_name": ae_name
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

    if DATA_LABEL == "random":
        results = mtl_experiments.train_mtl_random_data(settings_sim=settings_sim,
                                                        settings_data=settings_data,
                                                        settings_mtl=settings_mtl)
    elif DATA_LABEL == "cue":
        results = mtl_cue_experiments.train_mtl_cue_data(settings_sim=settings_sim,
                                                         settings_data=settings_data,
                                                         settings_mtl=settings_mtl)
    else:
        raise NameError("wrong data label")


    score = float(EVAL_FUNC(results, sigma=horizon).mean())
    if not np.isfinite(score):
        score = 0.
    return float(np.clip(score, 0., 1.))

def evaluate_mtl_random(population: list, plasticity: str="base",
                        horizon: float=HORIZON):
    return [evaluate_mtl_individual(ind=ind, plasticity=plasticity,
                                    horizon=horizon) for ind in tqdm(population)]


def mtlsearch(generations: int=96, pause: float=0.01, live_plot: bool=False,
              workers=None, plasticity: str="base", save_name: str="mtl_evolution_1",
              verbose: bool=True, disable: bool=False, horizon: int=HORIZON,
              save: bool=False):

    num_parameters = len(MTL_PARAMETER_NAMES)

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
        "disable": disable,
        "metric_name": "weighted_accuracy",
        "workers": workers,
        "plasticity": plasticity,
    }

    batch_evaluator = functools.partial(
        evaluate_mtl_random,
        plasticity=plasticity,
        horizon=horizon,
    )
    individual_evaluator = functools.partial(
        evaluate_mtl_individual,
        plasticity=plasticity,
        horizon=horizon,
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
        "plasticity": plasticity
    }

    if save_name is not None and save:
        _lib.save_genome(info=logs, name=save_name)
    return record


if __name__ == "__main__":
    args = parse_args()
    mtlsearch(
        generations=args.generations,
        pause=args.pause,
        live_plot=not args.no_plot,
        workers=args.workers,
        plasticity=_PLASTICITY,
        save=True
    )
    print("[done]")
