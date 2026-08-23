import numpy as np
import matplotlib.pyplot as plt
import warnings
from torch.nn import MSELoss

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
from experiments.evolution._lib import id_eval



NUM_CUE_PATTERNS = 2
SIZE = 50
LEC_SIZE = SIZE // 2

NUM = 1
NUM_LAPS = NUM
LAP_LENGTH = 50
CUE_POSITIONS = [10, 30]
REPS = 128


def make_cue_data():

    cue_sequence = []
    for l in range(NUM_LAPS):
       cue_sequence += [np.random.choice(list(range(NUM_CUE_PATTERNS)),
                                         replace=False,
                                         size=len(CUE_POSITIONS)).tolist()] 

    laps = {
        "n": NUM_LAPS,
        "length": LAP_LENGTH,
        "cues_positions": CUE_POSITIONS,
        "cues_patterns": dg.make_cues(n=NUM_CUE_PATTERNS, size=LEC_SIZE, fixed=True, p=0.2),
        "cues_sequence": cue_sequence
    }

    return dg.sparse_stimulus_generator_sensory(laps=laps, mec_sigma=5, lec_sigma=5)[0].reshape(-1, SIZE)



def train_mtl_cue_data(settings_sim: dict,
                       settings_data: dict,
                       settings_mtl: dict):

    """ train the autoencoder with cue samples """

    num_samples = NUM * LAP_LENGTH
    ae_name = settings_sim.get("ae_name", None)
    use_bias = settings_sim.get("use_bias", False)
    disable = settings_sim.get("disable", False)
    plot = settings_sim.get("plot", False)

    autoencoder, info = aect.load_autoencoder(name=ae_name)
    params = autoencoder.get_weights(bias=use_bias)

    model = models.MTL(W_ei_ca1=params[0], W_ca1_eo=params[1],
                       K_lat=settings_mtl["K_lat"],
                       K_ca3=settings_mtl["K_ca3"],
                       K_out=settings_mtl["K_out"],
                       dim_ca3=settings_mtl["dim_ca3"],
                       beta_is=settings_mtl["beta_is"],
                       beta_ca3=settings_mtl["beta_ca3"],
                       beta_ca1=settings_mtl["beta_ca1"],
                       beta_eo=settings_mtl["beta_eo"],
                       alpha=settings_mtl["alpha"],
                       nb_ei_ca3=int(settings_mtl.get("nb_ei_ca3", 10)),
                       num_swaps_ca3=settings_mtl["num_swaps_ca3"],
                       num_swaps_ca1=settings_mtl["num_swaps_ca1"],
                       B_ei_ca1=params[2],
                       B_ca1_eo=params[3],
                       random_IS=settings_mtl["random_IS"],
                       plasticity=settings_mtl.get("plasticity", "base"))

    reps = settings_sim.get("reps", 32)
    criterion = settings_sim.get("criterion", mtlct.cosine_criterion)

    results = np.zeros((reps, num_samples, num_samples))
    rdisable = not(reps > 2)
    for r in tqdm(range(reps), disable=not(rdisable and (not disable))):

        training_data = make_cue_data()

        logs, model = mtlct.train_for_accuracy(data=training_data,
                                               model=model,
                                               criterion=criterion,
                                               disable=not((not rdisable) and (not disable)))
        results[r] = logs["rec_loss"]

    results = results.mean(axis=0)
    score = results.sum()/(len(results)**2/2)
    scoreid = id_eval(results).mean()

    if plot:
        logger(f"MTL results={np.around(results, 2)}")
        logger(f"accuracy={score:.3f}")
        logger(f"score 'id_eval'={scoreid:.3f}")

        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        im = plt.imshow(results, aspect="auto")
        plt.colorbar(im)
        plt.grid()
        plt.title(f"mtl, best={score:.3f}")
        plt.show()

    return results


""" main functions """


def main_cue(plot: bool):

    # setup
    settings_sim = {
        "data_training_size": 1024,
        "criterion": mtlct.cosine_criterion,
        "reps": REPS,
        "use_bias": False,
        "disable": True,
        "plot": True,
        "ae_name": "ae_cue_nb_0"
    }

    settings_data = {
        "size": 50,
        "K": 5,
    }

    settings_mtl = {
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

    train_mtl_cue_data(settings_sim=settings_sim,
                       settings_data=settings_data,
                       settings_mtl=settings_mtl)

if __name__ == "__main__":

    main_cue(plot=False)
