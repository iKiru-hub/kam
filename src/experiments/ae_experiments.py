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

""" train functions """

def train_random_data(settings_sim: dict,
                      settings_data: dict,
                      settings_ae: dict,
                      save: bool=False,
                      plot: bool=False):

    """ train the autoencoder with uniform random samples """ 

    training_data = dg.sparse_stimulus_generator(N=settings_sim["data_training_size"],
                                                 K=settings_data["K"],
                                                 size=settings_data["size"],
                                                 plot=False)

    test_data = dg.sparse_stimulus_generator(N=settings_sim["data_test_size"],
                                             K=settings_data["K"],
                                             size=settings_data["size"],
                                             plot=False)

    autoencoder = models.Autoencoder(input_dim=settings_ae["input_dim"],
                                     encoding_dim=settings_ae["encoding_dim"],
                                     K=settings_ae["K"],
                                     beta=settings_ae["beta"],
                                     gain_out=settings_ae["gain_out"],
                                     offset_out=settings_ae["offset_out"],
                                     use_bias=settings_ae["use_bias"])

    info, autoencoder = aect.train_autoencoder(training_data=training_data,
                                             test_data=test_data,
                                             autoencoder=autoencoder,
                                             epochs=settings_sim["epochs"],
                                             batch_size=settings_sim["batch_size"],
                                         learning_rate=settings_sim["learning_rate"],
                                               disable=False)

    if plot:
        ltrain, ltest = info["loss"], info["test"]
        plt.plot(range(len(ltrain)), ltrain, '-b', label="train")
        plt.plot(range(len(ltest)), ltest, '-r', label="test")
        plt.legend()
        plt.grid()

        results = aect.reconstruct_data(data=test_data, model=autoencoder,
                                        num=5, column=False, show=True, plot=True)

    if save:
        session = {"settings_sim": settings_sim,
                   "settings_data": settings_data,
                   "settings_ae": settings_ae,
                   "results": info}
        name = "ae_random_"
        nb = len([f for f in os.listdir(aect.AE_PATH) if name in f])
        aect.save_autoencoder(autoencoder=autoencoder, session=session,
                              name=name + str(nb))

    return info["test"][-1]

""" main search """

def search_a():

    # setup
    settings_sim = {
            "data_training_size": 1000,
            "data_test_size": 100,
            "epochs": 252,
            "batch_size": 50,
            "learning_rate": 1e-3
    }

    settings_data = {
            "size": 50,
            "K": 5,
    }

    settings_ae = {
        "input_dim": settings_data["size"],
        "encoding_dim": 30,
        "K": 5,
        "beta": 70.,
        "gain_out": 20.,
        "offset_out": 0.,
        "use_bias": True,
    }

    # parameters to iterate over
    beta_size = 4
    gain_size = 4
    encoding_dim_size = 4
    variables = {
        "beta": np.linspace(1, 100, beta_size).tolist(),
        "gain_out": np.linspace(1, 100, gain_size).tolist(),
        "encoding_dim": np.linspace(5, 100, encoding_dim_size).astype(int).tolist(),
    }

    # init
    results = np.zeros((beta_size, gain_size, encoding_dim_size))

    for i, v1 in utils.tqdm_enumerate(variables["beta"]):
        logger(f"=== beta={v1:.2f} =====")
        for j, v2 in utils.tqdm_enumerate(variables["gain_out"]):
            logger(f"=== gain_out={v2:.2f} ===")
            for k, v3 in utils.tqdm_enumerate(variables["encoding_dim"]):
                logger(f"=== encoding_dim={v3:.2f} =")
                run_settings_ae = {
                    **settings_ae,
                    "beta": v1,
                    "gain_out": v2,
                    "encoding_dim": v3,
                }
                results[i, j, k] = train_random_data(settings_sim=settings_sim,
                                                     settings_data=settings_data,
                                                     settings_ae=run_settings_ae,
                                                     save=False)

    results_path = aect.save_search_results(
        results=results,
        variables=variables,
        settings_sim=settings_sim,
        settings_data=settings_data,
        settings_ae=settings_ae,
        name="search_a",
    )

    fig, axs = plt.subplots(1, beta_size)
    for i, ax in enumerate(axs):
        ax.imshow(results[i], aspect="auto")
        ax.set_title(f"beta={variables['beta'][i]:.1f}")
        ax.set_xticks(range(encoding_dim_size))
        ax.set_yticks(range(gain_size))
        ax.set_xticklabels(np.around(variables["encoding_dim"], 1))
        ax.set_yticklabels(np.around(variables["gain_out"], 1))
    plt.show()

    return results, results_path




""" main """

def main(save: bool=False, plot: bool=False):

    # setup
    settings_sim = {
            "data_training_size": 2000,
            "data_test_size": 200,
            "epochs": 5000,
            "batch_size": 50,
            "learning_rate": 1e-3
    }

    settings_data = {
            "size": 50,
            "K": 5,
    }

    settings_ae = {
        "input_dim": settings_data["size"],
        "encoding_dim": 30,
        "K": 5,
        "beta": 30.,
        "gain_out": 20.,
        "offset_out": 0.,
        "use_bias": True,
    }

    train_random_data(settings_sim=settings_sim,
                      settings_data=settings_data,
                      settings_ae=settings_ae,
                      save=save,
                      plot=plot)




if __name__ == "__main__":

    main(save=True, plot=True)
    # search_a()
