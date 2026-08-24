import numpy as np
import matplotlib.pyplot as plt
import torch
import warnings

from tqdm import tqdm
import os, sys
import json
sys.path.append(os.path.abspath(__file__).split("src")[0] + "src")

import core.models as models
import core.datagen as dg
import core.training as ct
import core.utils as utils
import core.ae_tools as aect
import core.mtl_tools as mtlct
from core.logger import logger
import evolution._lib as _lib

import mtl_experiments as mtlexp


SIZE = 15  # gets split | total number of samples
SETUP = {"K_ca3": (2, 49), "K_lat": (2, 49), "beta_is": (1, 256), "beta_ca3": (1, 256),
         "beta_ca1": (1, 256), "alpha": (0.0001, 0.2), "nb_ei_ca3": (1, 48)}
NAMES = list(SETUP.keys())
REPS = 10
HORIZON = 64
BASE_SEED = 3980
DISCRETE_PARAMETERS = {"K_ca3", "K_lat", "K_out", "nb_ei_ca3"}
_PLASTICITY = "base"

# setup
settings_sim = {
    "data_training_size": 96,
    "criterion": mtlct.cosine_criterion,
    # Repetition is controlled by the outer parameter sweep.
    "reps": 1,
    "use_bias": False,
    "disable": True,
    "plot": False,
    "ae_name": "ae_random_nb_0"
}

settings_data = {
    "size": 50,
    "K": 5,
}

settings_mtl_base = {
    "K_ca3": 5,
    "K_lat": 15,
    "K_out": 5,
    "dim_ca3": 50,
    "beta_is": 48,
    "beta_ca3": 196,
    "beta_ca1": 24,
    "beta_eo": 20,
    "alpha": 0.018,
    "nb_ei_ca3": 2,
    "num_swaps_ca1": 1,
    "num_swaps_ca3": 1,
    "random_IS": False,
    "plasticity": "base",
}

settings_mtl_err2 = {
    "K_ca3": 2,
    "K_lat": 19,
    "K_out": 5,
    "dim_ca3": 50,
    "beta_is": 59,
    "beta_ca3": 196,
    "beta_ca1": 42,
    "beta_eo": 20,
    "alpha": 0.052,
    "nb_ei_ca3": 2,
    "num_swaps_ca1": 1,
    "num_swaps_ca3": 1,
    "random_IS": False,
    "plasticity": "err2",
}

if _PLASTICITY == "base":
    settings_mtl = settings_mtl_base
elif _PLASTICITY == "err2":
    settings_mtl = settings_mtl_err2
else:
    raise NameError("plasticity name mistake")


def _plot(results: np.ndarray):

    # plot
    fig, ax = plt.subplots()
    ax.imshow(results, cmap="magma")
    ax.set_yticks(range(len(NAMES)))
    ax.set_yticklabels(NAMES)
    ax.set_xticks(range(SIZE))
    ax.set_xticklabels(
        [f"{100 * index / (SIZE - 1):.0f}%" for index in range(SIZE)],
        rotation=45,
        ha="right",
    )
    ax.set_xlabel("position within each parameter range")
    ax.set_ylabel("parameters")
    ax.set_title("MTL weighted accuracy (higher is better)")

    avg = results.mean()
    # Robust color version
    for i in range(len(results)):
       for j in range(len(results[0])):
           val = results[i, j]
           # white for dark cells, black for light cells
           text_color = "white" if val < avg else "black"
           ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=text_color)

    plt.show()


def _parameter_values(name: str) -> np.ndarray:
    values = np.linspace(SETUP[name][0], SETUP[name][1], SIZE)
    if name in DISCRETE_PARAMETERS:
        values = np.rint(values).astype(int)
    return values


def _run(_key: str, _value: float, seed: int|None=None):
    if _key not in settings_mtl:
        raise KeyError(f"unknown MTL parameter: {_key}")
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    _settings_mtl = settings_mtl.copy()
    _settings_mtl[_key] = int(_value) \
        if _key in DISCRETE_PARAMETERS else float(_value)
    results = mtlexp.train_mtl_random_data(settings_sim=settings_sim,
                                           settings_data=settings_data,
                                           settings_mtl=_settings_mtl)

    score = float(_lib.exp_eval(results, sigma=HORIZON).mean())
    if not np.isfinite(score):
        score = 0.
    return float(np.clip(score, 0., 1.))


def _main():

    results = np.zeros((len(NAMES), SIZE, REPS))

    for i, name in enumerate(NAMES):
        logger(f"running `{name}`")
        values = _parameter_values(name)
        logger(f"{np.around(values, 1)} [{settings_mtl[name]}]")

        for j, value in utils.tqdm_enumerate(values):
            for r in range(REPS):
                # Reuse the same random realization for repetition r across
                # all parameter values, isolating the parameter's effect.
                results[i, j, r] = _run(
                    _key=name,
                    _value=value,
                    seed=BASE_SEED + r,
                )

    mean_results = results.mean(axis=2)

    _plot(results=mean_results)
    return mean_results



if __name__ == "__main__":

    _main()
    logger("[done]")
