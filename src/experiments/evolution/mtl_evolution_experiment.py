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

# sys.path.append(os.path.abspath(__file__).split("src")[0] + "src/experiments")
sys.path.append(os.path.abspath(__file__).split("src")[0] + "src")

# from core.main import K_ca3
import experiments.mtl_experiments as mtl_experiments
import experiments.mtl_cue_experiments as mtl_cue_experiments
import core.functions as functions
from core.logger import logger
from core.constants import MTL_PATH
import _lib


CUE_SPACING = 1
NUM_CUE_PATTERNS = 5
NOISE_LEVEL = 0.1
DIM_CA1 = 50
BIT_KIND = 2
N = 2

DATA_TRAINING_SIZE = 50*CUE_SPACING*NUM_CUE_PATTERNS*N
DATA_LABEL = "cue" # | "random"

PLASTICITY_VARIANTS = ("base", "nois", "isout", "err1", "err2", "err3", "xbtsp", "btsp")
_PLASTICITY = "err2"


""" settings """

MTL_PARAMETER_NAMES = (
    "K_ca3",
    "beta_ca3",
    "beta_ca1",
    "alpha",
    "alpha_plus",
    "alpha_minus",
    "a_plus",
    "b_plus",
    "a_minus",
    "b_minus",
    "nb_ei_ca3",
)
MTL_PARAMETER_CENTERS = np.array([10., 30., 30., 0.166, 0.166, 0.166, 10., 0.1, 10., 0.1, 10.])
MTL_PARAMETER_SCALES = np.array([5., 15., 15., 0.10, 0.10, 0.10, 15, 0.1, 15., 0.1, 5.])
MTL_PARAMETER_LOWER = np.array([2., 2., 2., 1e-3, 1e-3, 1e-3, 2., 1e-3, 2., 1e-3, 2.])
MTL_PARAMETER_UPPER = np.array([49., 512., 512., 1., 1., 1., 512, 1, 512, 1., 49])
MTL_LATENT_LOWER = (
    MTL_PARAMETER_LOWER - MTL_PARAMETER_CENTERS
) / MTL_PARAMETER_SCALES
MTL_LATENT_UPPER = (
    MTL_PARAMETER_UPPER - MTL_PARAMETER_CENTERS
) / MTL_PARAMETER_SCALES
MTL_EVALUATION_SEED = int(np.random.uniform()*10000)

MTL_CRITERIA = {
    "cosine": mtl_experiments.mtlct.cosine_criterion,
    "mse": functions.mse,
    "modified-mse": functions.modified_mse_score,
    "gaussian": functions.gaussian,
}
DEFAULT_CRITERION = "cosine" # "modified-mse"

HORIZON = 10
EVAL_FUNC = _lib.id_eval # _lib.exp_eval # "_lib.id_eval" "_lib.mean_eval"


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
    # K_ca3 and the number of EI-to-CA3 connections are discrete.
    parameters[[0, 10]] = np.rint(parameters[[0, 10]])
    return parameters




def evaluate_mtl_individual(ind: list, plasticity: str="base",
                            horizon: float=HORIZON,
                            criterion_name: str=DEFAULT_CRITERION,
                            noise_level: float=NOISE_LEVEL,
                            return_diagnostics: bool=False):
    """Train and score one decoded MTL candidate."""

    if DATA_LABEL == "random":
        ae_name = "ae_random_nb_0"
    elif DATA_LABEL == "cue":
        ae_name = f"ae_{NUM_CUE_PATTERNS}cues_100ca10" # 6: 4 cues
    else:
        raise NameError("wrong data label")

    try:
        criterion = MTL_CRITERIA[criterion_name]
    except KeyError as error:
        raise ValueError(
            f"criterion_name must be one of {tuple(MTL_CRITERIA)}, "
            f"got {criterion_name!r}"
        ) from error

    settings_sim = {
        "data_training_size": DATA_TRAINING_SIZE,
        "criterion": criterion,
        "reps": 1,
        "use_bias": False,
        "disable": True,
        "plot": False,
        "ae_name": ae_name
    }

    settings_data = {
        "size": 50,
        "K": 5,
        "num_cue_patterns": NUM_CUE_PATTERNS,
        "lap_length": 50,
        "cue_positions": [10., 30.],
        "cue_sigma": 3.,
        "cue_beta": 40.,
        "cue_alpha": 0.2,
        "mec_binarized": True,
        "mec_sigma": 4,
        "cue_spacing": CUE_SPACING,
        "noise_level": float(noise_level),
        "bit_kind": BIT_KIND
    }

    np.random.seed(MTL_EVALUATION_SEED)
    torch.manual_seed(MTL_EVALUATION_SEED)

    for i in range(len(ind)):
        if np.isnan(ind[i]) and i != 6:
            ind[i] = 2


    settings_mtl = {
        "K_ca3": ind[0],
        "dim_ca3": 50, # evolve?
        "dim_ca1": DIM_CA1,
        "beta_ca3": ind[1],
        "beta_ca1": ind[2],
        "alpha": ind[3],
        "alpha_plus": ind[4],
        "alpha_minus": ind[5],
        "a_plus": ind[6],
        "b_plus": ind[7],
        "a_minus": ind[8],
        "b_minus": ind[9],
        "nb_ei_ca3": int(ind[10]),
        "num_swaps_ca1": 0,
        "num_swaps_ca3": 0,
        "random_IS": False,
        "plasticity": plasticity
    }

    if DATA_LABEL == "random":
        if return_diagnostics:
            raise ValueError(
                "live reconstruction diagnostics currently require cue data"
            )
        results = mtl_experiments.train_mtl_random_data(settings_sim=settings_sim,
                                                        settings_data=settings_data,
                                                        settings_mtl=settings_mtl)
    elif DATA_LABEL == "cue":
        evaluation = mtl_cue_experiments.train_mtl_cue_data(
            settings_sim=settings_sim,
            settings_data=settings_data,
            settings_mtl=settings_mtl,
            return_diagnostics=return_diagnostics,
        )
        results = evaluation["results"] if return_diagnostics else evaluation
    else:
        raise NameError("wrong data label")


    # score = float(EVAL_FUNC(results, sigma=horizon).mean())
    score = np.mean(results)
    if not np.isfinite(score):
        score = 0.
    score = float(np.clip(score, 0., 1.))
    if return_diagnostics:
        return {
            "original_stimuli": evaluation["original_stimuli"],
            "reconstructed_stimuli": evaluation["reconstructed_stimuli"],
            "ca3_activity": evaluation["ca3_activity"],
            "ca1_activity": evaluation["ca1_activity"],
            "reconstruction_fidelity": score,
        }
    return score

def evaluate_mtl_random(population: list, plasticity: str="base",
                        horizon: float=HORIZON,
                        criterion_name: str=DEFAULT_CRITERION,
                        noise_level: float=NOISE_LEVEL):
    return [evaluate_mtl_individual(ind=ind, plasticity=plasticity,
                                    horizon=horizon,
                                    criterion_name=criterion_name,
                                    noise_level=noise_level)
            for ind in tqdm(population)]


def mtlsearch(generations: int=96, pause: float=0.01, live_plot: bool=False,
              workers=None, plasticity: str="base",
              verbose: bool=True, disable: bool=False, horizon: int=HORIZON,
              save: bool=False,
              criterion_name: str=DEFAULT_CRITERION,
              noise_level: float=NOISE_LEVEL,
              bit_kind: int=BIT_KIND):

    logger("-- MTL evolution --")
    logger(f"{NUM_CUE_PATTERNS=}")
    logger(f"{DIM_CA1=}")
    logger(f"{CUE_SPACING=}")
    noise_level = float(noise_level)
    logger(f"{noise_level=}")
    logger(f"{bit_kind=}")
    logger(f"{plasticity=}")
    save_name = "mtl_"
    save_name += f"{time.localtime().tm_mday}:{time.localtime().tm_mon}:" + \
            f"{time.localtime().tm_hour}:{time.localtime().tm_min}:" + \
            f"{time.localtime().tm_sec}"
    logger(f"{save_name=}")

    parameter_spec_lengths = {
        "names": len(MTL_PARAMETER_NAMES),
        "centers": len(MTL_PARAMETER_CENTERS),
        "scales": len(MTL_PARAMETER_SCALES),
        "lower bounds": len(MTL_PARAMETER_LOWER),
        "upper bounds": len(MTL_PARAMETER_UPPER),
    }
    if len(set(parameter_spec_lengths.values())) != 1:
        raise ValueError(
            "inconsistent MTL parameter specification lengths: "
            + ", ".join(
                f"{name}={length}"
                for name, length in parameter_spec_lengths.items()
            )
        )
    num_parameters = len(MTL_PARAMETER_NAMES)

    plasticity = str(plasticity).lower()
    if plasticity not in PLASTICITY_VARIANTS:
        raise ValueError(
            f"plasticity must be one of {PLASTICITY_VARIANTS}, "
            f"got {plasticity!r}"
        )
    if criterion_name not in MTL_CRITERIA:
        raise ValueError(
            f"criterion_name must be one of {tuple(MTL_CRITERIA)}, "
            f"got {criterion_name!r}"
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
        "criterion": criterion_name,
        # Keep CMA-ES's raw mean/covariance near the region that decodes to
        # valid parameters. This avoids sigma growth on flat clipped plateaus.
        "latent_lower": MTL_LATENT_LOWER.tolist(),
        "latent_upper": MTL_LATENT_UPPER.tolist(),
        "boundary_penalty": 0.05,
        "dim_ca1": DIM_CA1,
        "num_cue_patterns": NUM_CUE_PATTERNS,
        "noise_level": noise_level,
        "bit_kind": bit_kind,
    }

    batch_evaluator = functools.partial(
        evaluate_mtl_random,
        plasticity=plasticity,
        horizon=horizon,
        criterion_name=criterion_name,
        noise_level=noise_level,
    )
    individual_evaluator = functools.partial(
        evaluate_mtl_individual,
        plasticity=plasticity,
        horizon=horizon,
        criterion_name=criterion_name,
        noise_level=noise_level,
    )
    diagnostic_evaluator = None
    if live_plot and DATA_LABEL == "cue":
        uncached_diagnostic_evaluator = functools.partial(
            evaluate_mtl_individual,
            plasticity=plasticity,
            horizon=horizon,
            criterion_name=criterion_name,
            noise_level=noise_level,
            return_diagnostics=True,
        )
        cached_candidate = None
        cached_diagnostic = None

        def diagnostic_evaluator(candidate):
            nonlocal cached_candidate, cached_diagnostic
            candidate = np.asarray(candidate, dtype=float)
            if (cached_candidate is None
                    or not np.array_equal(candidate, cached_candidate)):
                cached_candidate = candidate.copy()
                cached_diagnostic = uncached_diagnostic_evaluator(
                    candidate.copy()
                )
            return cached_diagnostic

    record = _lib.evolution_run(settings=settings,
                                evaluate=batch_evaluator,
                                evaluate_individual=individual_evaluator,
                                generation_diagnostics=diagnostic_evaluator,
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
    logger(f"best score: {record["best_fitness"][-1]:.5f}")

    if save_name is not None and save:
        _lib.save_genome(info=logs, name=f"{save_name}",
                         path=MTL_PATH)
    return record


""" args """


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evolve MTL hyperparameters with CMA-ES."
    )
    parser.add_argument("--generations", type=int, default=196)
    parser.add_argument("--pause", type=float, default=0.01)
    parser.add_argument("--noise", type=float, default=NOISE_LEVEL)
    parser.add_argument("--plasticity", type=str, default="base")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="parallel workers; defaults to min(population size, CPU count)",
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument(
        "--criterion",
        choices=tuple(MTL_CRITERIA),
        default=DEFAULT_CRITERION,
        help="higher-is-better reconstruction criterion used for fitness",
    )
    return parser.parse_args()




if __name__ == "__main__":
    args = parse_args()
    mtlsearch(
        generations=args.generations,
        pause=args.pause,
        live_plot=not args.no_plot,
        workers=args.workers,
        plasticity=args.plasticity,
        criterion_name=args.criterion,
        noise_level=args.noise,
        verbose=False,
        save=args.save
    )
    print("[done]")
