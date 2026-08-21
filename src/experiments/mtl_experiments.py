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
                          settings_mtl: dict,
                          ae_name: str,
                          plot: bool=False):

    """ train the autoencoder with uniform random samples """ 

    training_data = dg.sparse_stimulus_generator(N=settings_sim["data_training_size"],
                                                 K=settings_data["K"],
                                                 size=settings_data["size"],
                                                 plot=False)

    autoencoder, info = aect.load_autoencoder(name=ae_name)
    params = autoencoder.get_weights()

    model = models.MTL(W_ei_ca1=params[0], W_ca1_eo=params[1],
                       K_lat=settings_mtl["K_lat"],
                       K_out=settings_mtl["K_out"],
                       dim_ca3=settings_mtl["dim_ca3"],
                       beta=settings_mtl["beta"],
                       alpha=settings_mtl["alpha"],
                       num_swaps=settings_mtl["num_swaps"],
                       B_ei_ca1=params[2],
                       B_ca1_eo=params[3],
                       random_IS=settings_mtl["random_IS"])

    results, model = mtlct.train_for_accuracy(data=training_data,
                                              model=model,
                                              criterion=mtlct.cosine_criterion,
                                              # criterion=MSELoss(),
                                              disable=False)

    if plot:
        print(f"MTL results={np.around(results, 2)}")

        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        im = plt.imshow(results, aspect="auto")
        plt.colorbar(im)
        plt.grid()
        plt.title("mtl")
        plt.show()


""" main functions """


def main_random(plot: bool):

    # setup
    settings_sim = {
        "data_training_size": 200,
    }

    settings_data = {
        "size": 50,
        "K": 5,
    }

    settings_mtl = {
        "K_lat": 20,
        "K_out": 20,
        "dim_ca3": 50,
        "beta": 10,
        "alpha": 0.3,
        "num_swaps": 8,
        "random_IS": False,
    }

    train_mtl_random_data(settings_sim=settings_sim,
                          settings_data=settings_data,
                          settings_mtl=settings_mtl,
                          ae_name="ae_random_1",
                          plot=True)

if __name__ == "__main__":

    main_random(plot=True)

