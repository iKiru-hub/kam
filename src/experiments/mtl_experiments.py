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



def train_mtl_random_data(settings_sim: dict,
                          settings_data: dict,
                          settings_mtl: dict):

    """ train the autoencoder with uniform random samples """ 

    num_samples = settings_sim["data_training_size"]
    ae_name = settings_sim.get("ae_name", "ae_random_0")
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

    reps = settings_sim.get("reps", 1)
    criterion = settings_sim.get("criterion", mtlct.cosine_criterion)

    results = np.zeros((reps, num_samples, num_samples))
    rdisable = not(reps > 2)
    for r in tqdm(range(reps), disable=not(rdisable and (not disable))):

        training_data = dg.sparse_stimulus_generator(N=num_samples,
                                                     K=settings_data["K"],
                                                     size=settings_data["size"],
                                                     plot=False)

        logs, model = mtlct.train_for_accuracy(data=training_data,
                                               model=model,
                                               criterion=criterion,
                                               disable=not((not rdisable) and (not disable)))
        results[r] = logs["rec_loss"]

    results = results.mean(axis=0)

    if plot:
        logger(f"MTL results={np.around(results, 2)}")
        logger(f"max accuracy={results.max():.3f}")

        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        im = plt.imshow(results, aspect="auto")
        plt.colorbar(im)
        plt.grid()
        plt.title(f"mtl, best={results.max():.3f}")
        plt.show()

    return results


""" main functions """


def main_random(plot: bool):

    # setup
    settings_sim = {
        "data_training_size": 96,
        "criterion": mtlct.cosine_criterion,
        "reps": 10,
        "use_bias": False,
        "disable": False,
        "plot": True,
        "ae_name": "ae_random_nb_0"
    }

    settings_data = {
        "size": 50,
        "K": 5,
    }

    settings_mtl = {
        "K_ca3": 8,
        "K_lat": 10,
        "K_out": 10,
        "dim_ca3": 50,
        "beta_is": 25,
        "beta_ca3": 105,
        "beta_ca1": 10,
        "beta_eo": 20,
        "alpha": 0.166,
        "num_swaps_ca1": 1,
        "num_swaps_ca3": 1,
        "random_IS": False,
    }

    train_mtl_random_data(settings_sim=settings_sim,
                          settings_data=settings_data,
                          settings_mtl=settings_mtl)

if __name__ == "__main__":

    main_random(plot=False)
