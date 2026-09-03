import numpy as np
import matplotlib.pyplot as plt
import warnings

from tqdm import tqdm
import os, sys
import json
sys.path.append(os.path.abspath(__file__).split("src")[0] + "src")

import core.models as models
import experiments.mtl_experiments as mtl_experiments
import core.datagen as dg
import experiments.mtl_cue_experiments as mtl_cue_experiments
import core.training as ct
import core.utils as utils
import core.functions as functions
import core.ae_tools as aect
import core.mtl_tools as mtlct
from core.logger import logger
from experiments.evolution._lib import id_eval

TOT_NUMS = 20

SIZE = 50
REPS = 1
TQDM_REPS = False
CUE_SPACING = 1
N = 5

DIM_CA1 = 50
NUM_CUE_PATTERNS = 5
MAX_NUM_PATTERNS = None
NOISE_LEVEL = 0.
BIT_KIND = 2

PLASTICITY = "err2"


def _main():

    logger("-- MTL cue experiment --")
    logger(f"{NUM_CUE_PATTERNS=}")
    logger(f"{DIM_CA1=}")
    logger(f"{CUE_SPACING=}")
    logger(f"{NOISE_LEVEL=}")
    logger(f"{PLASTICITY=}")

    # setup
    settings_sim = {
        "data_training_size": 50*CUE_SPACING*NUM_CUE_PATTERNS*N,
        "criterion": mtl_experiments.mtlct.cosine_criterion,
        "reps": 1,
        "use_bias": False,
        "disable": False,
        "plot": False,
        "ae_name": f"ae_{NUM_CUE_PATTERNS}cues_100ca10"
    }

    settings_data = {
        "size": 50,
        "K": 5,
        "num_cue_patterns": NUM_CUE_PATTERNS,
        "max_num_patterns": MAX_NUM_PATTERNS,
        "lap_length": 50,
        "cue_positions": [10., 30.],
        "cue_sigma": 3.,
        "cue_beta": 40.,
        "cue_alpha": 0.2,
        "mec_binarized": True,
        "mec_sigma": 4,
        "cue_spacing": CUE_SPACING,
        "noise_level": NOISE_LEVEL,
        "bit_kind": BIT_KIND
    }

    found = mtlct.find_mtl(dim_ca1=DIM_CA1, noise_level=NOISE_LEVEL,
                           num_cue_patterns=NUM_CUE_PATTERNS,
                           plasticity=PLASTICITY,
                           bit_kind=BIT_KIND)
    if len(found) <= 0: sys.exit("no evolved MTL matches the current settings")
    logger(f"saved MTL found: {[f[0] for f in found]}")

    _settings_mtl = found[-1][1]["best_parameters"]
    _settings_mtl["dim_ca3"] = 50
    _settings_mtl["num_swaps_ca1"] = 0
    _settings_mtl["num_swaps_ca3"] = 0
    _settings_mtl["random_IS"] = False
    _settings_mtl["plasticity"] = PLASTICITY

    names = ("base", "err2", "btsp", "xbtsp")
    nums = range(2, TOT_NUMS)

    # --
    logs = {nm: [] for nm in names}
    for name in names:
        for n in tqdm(nums):
            _settings_mtl["plasticity"] = name

            settings_data["num_cue_patterns"] = n
            settings_data["data_training_size"] = 50*CUE_SPACING*n*N,

            results = mtl_cue_experiments.train_mtl_cue_data(
                            settings_sim=settings_sim,
                            settings_data=settings_data,
                            settings_mtl=_settings_mtl,
                            return_diagnostics=False)

            score = np.mean(results)
            if not np.isfinite(score):
                score = 0.
            score = float(np.clip(score, 0., 1.))

            logs[name] += [score]

    fig, ax = plt.subplots()
    for name in names:
        ax.plot(nums, logs[name], 'o-', label=name)

    ax.set_xlabel("#patterns")
    ax.set_ylabel("accuracy")
    ax.grid()
    ax.legend()
    plt.show()


if __name__ == "__main__":
    _main()
