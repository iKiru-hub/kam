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
from core.logger import logger

import ae_experiments as aexp


SIZE = 20  # gets split | total number of samples
SETUP = {"K": (2, 49), "beta": (1, 128), "gain_out": (1, 128)}
NAMES = list(SETUP.keys())
REPS = 5
BASE_SEED = 3980
DISCRETE_PARAMETERS = {"K"}

# setup
settings_sim = {
        "data_training_size": 512,
        "data_test_size": 96,
        "epochs": 64,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "disable": True
}

settings_data = {
        "size": 50,
        "K": 5,
}

settings_ae = {
    "input_dim": settings_data["size"],
    "encoding_dim": 50,
    "K": 5,
    "beta": 25.,
    "gain_out": 20.,
    "offset_out": 0.,
    "use_bias": False,
}


def _plot(results: np.ndarray):

    # plot
    fig, ax = plt.subplots()
    ax.imshow(results, cmap="magma_r")
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
    ax.set_title("Autoencoder validation MSE (lower is better)")

    avg = results.mean()
    # Robust color version
    for i in range(len(results)):
       for j in range(len(results[0])):
           val = results[i, j]
           # white for dark cells, black for light cells
           text_color = "white" if val > avg else "black"
           ax.text(j, i, f'{val:.4f}', ha='center', va='center', color=text_color)

    plt.show()


def _parameter_values(name: str) -> np.ndarray:
    values = np.linspace(SETUP[name][0], SETUP[name][1], SIZE)
    if name in DISCRETE_PARAMETERS:
        values = np.rint(values).astype(int)
    return values


def _run(_key: str, _value: float, seed: int|None=None):
    if _key not in settings_ae:
        raise KeyError(f"unknown autoencoder parameter: {_key}")
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    _settings_ae = settings_ae.copy()
    _settings_ae[_key] = int(_value) \
        if _key in DISCRETE_PARAMETERS else float(_value)
    return aexp.train_random_data(settings_sim=settings_sim,
                                  settings_data=settings_data,
                                  settings_ae=_settings_ae,
                                  save=False,
                                  plot=False)


def _main():

    results = np.zeros((len(NAMES), SIZE, REPS))

    for i, name in enumerate(NAMES):
        logger(f"running `{name}`")
        values = _parameter_values(name)
        logger(f"{np.around(values, 1)} [{settings_ae[name]}]")

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
