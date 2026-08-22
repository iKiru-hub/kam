import numpy as np
import matplotlib.pyplot as plt
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


DVAR = 30  # percentage to vary up/down the value
SIZE = 3  # gets split | total number of samples
NAMES = ["K", "beta", "gain_out"]
REPS = 4

# setup
settings_sim = {
        "data_training_size": 512,
        "data_test_size": 96,
        "epochs": 128,
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



def _run(_key: str, _value: float):
    _settings_ae = settings_ae.copy()
    _settings_ae[_key] = _value
    return aexp.train_random_data(settings_sim=settings_sim,
                                  settings_data=settings_data,
                                  settings_ae=settings_ae)


def _main():

    results = np.zeros((len(NAMES), SIZE, REPS))

    for i, name in utils.tqdm_enumerate(NAMES):
        values = np.linspace(settings_ae[name]*(100-DVAR)/100,
                             settings_ae[name]*(100+DVAR)/100,
                             SIZE)

        for j, value in enumerate(values):
            for r in range(REPS):
                results[i, j, r] = _run(_key=name, _value=value)

    results = results.mean(axis=2)

    # plot
    fig, ax = plt.subplots()
    ax.imshow(results, cmap="magma_r")
    ax.set_yticks(range(len(NAMES)))
    ax.set_yticklabels(NAMES)
    ax.set_xlabel(f"$\\pm${DVAR}% variation")
    ax.set_ylabel("parameters")
    plt.show()



if __name__ == "__main__":

    _main()
    logger("[done]")
