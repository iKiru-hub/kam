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
import core.functions as functions
import core.utils as utils
import core.ae_tools as aect
from core.logger import logger



""" train functions """


def train_random_data(settings_sim: dict,
                      settings_data: dict,
                      settings_ae: dict,
                      save: bool=False,
                      name: str="ae_random_",
                      plot: bool=False):

    """ train the autoencoder with uniform random samples """ 

    disable = settings_sim.get("disable", False)

    training_data = dg.sparse_stimulus_generator(N=settings_sim["data_training_size"],
                                                 K=settings_data["K"],
                                                 size=settings_data["size"],
                                                 plot=False)

    test_data = dg.sparse_stimulus_generator(N=settings_sim["data_test_size"],
                                             K=settings_data["K"],
                                             size=settings_data["size"],
                                             plot=False)

    autoencoder = models.Autoencoder(dim_ei=settings_ae["dim_ei"],
                                     dim_ca1=settings_ae["dim_ca1"],
                                     K_ca1=settings_ae["K_ca1"],
                                     K_eo=settings_ae["K_eo"],
                                     beta_ei=settings_ae["beta_ei"],
                                     beta_eo=settings_ae["beta_eo"],
                                     use_bias=settings_ae["use_bias"])

    info, autoencoder = aect.train_autoencoder(training_data=training_data,
                                               test_data=test_data,
                                               autoencoder=autoencoder,
                                               epochs=settings_sim["epochs"],
                                               batch_size=settings_sim["batch_size"],
                                               learning_rate=settings_sim["learning_rate"],
                                               disable=disable,
                                               device=settings_sim.get("device"))

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
        nb = len([f for f in os.listdir(aect.AE_PATH) if name in f])
        aect.save_autoencoder(autoencoder=autoencoder, session=session,
                              name=name + str(nb))

    return info["test"][-1]


def make_cue_data(num: int, settings_sim: dict={}, settings_data: dict={}):
    """Generate ``num`` complete laps using the shared cue-task settings."""

    return dg.make_cue_track_data(
        num_samples=num * dg.DEFAULT_CUE_LAP_LENGTH,
        size=settings_data.get("size", 10),
        num_cue_patterns=settings_sim.get("num_cue_patterns", dg.DEFAULT_CUE_NUM_PATTERNS),
        lap_length=settings_data.get("lap_length", dg.DEFAULT_CUE_LAP_LENGTH),
        cue_positions=settings_data.get("cue_positions", dg.DEFAULT_CUE_POSITIONS),
        cue_sigma=settings_data.get("cue_sigma", dg.DEFAULT_CUE_SIGMA),
        cue_beta=settings_data.get("cue_beta", dg.DEFAULT_CUE_BETA),
        cue_alpha=settings_data.get("cue_alpha", dg.DEFAULT_CUE_ALPHA),
        mec_binarized=settings_data.get("mec_binarized", dg.DEFAULT_MEC_BINARIZED),
        mec_sigma=settings_data.get("mec_sigma", dg.DEFAULT_CUE_MEC_SIGMA),
        lec_sigma=settings_data.get("lec_sigma", dg.DEFAULT_CUE_LEC_SIGMA),
        cue_spacing=settings_data.get("cue_spacing", dg.DEFAULT_CUE_SPACING),
    )


def train_cue_data(settings_sim: dict,
                   settings_data: dict,
                   settings_ae: dict,
                   save: bool=False,
                   name: str="ae_cue_",
                   plot: bool=False):

    """Train the autoencoder with samples from the shared cue task."""

    training_data = make_cue_data(num=settings_sim["data_training_size"],
                                  settings_sim=settings_sim,
                                  settings_data=settings_data)

    test_data = make_cue_data(num=settings_sim["data_test_size"],
                              settings_sim=settings_sim,
                              settings_data=settings_data)

    autoencoder = models.Autoencoder(dim_ei=settings_ae["dim_ei"],
                                     dim_ca1=settings_ae["dim_ca1"],
                                     K_ca1=settings_ae["K_ca1"],
                                     K_eo=settings_ae["K_eo"],
                                     beta_ei=settings_ae["beta_ei"],
                                     beta_eo=settings_ae["beta_eo"],
                                     use_bias=settings_ae["use_bias"])

    info, autoencoder = aect.train_autoencoder(training_data=training_data,
                                             test_data=test_data,
                                             autoencoder=autoencoder,
                                             criterion=MSELoss(),
                                             epochs=settings_sim["epochs"],
                                             batch_size=settings_sim["batch_size"],
                                         learning_rate=settings_sim["learning_rate"],
                                               disable=settings_sim.get("disable", True))

    if plot:
        ltrain, ltest = info["loss"], info["test"]
        plt.plot(range(len(ltrain)), ltrain, '-b', label="train")
        plt.plot(range(len(ltest)), ltest, '-r', label="test")
        plt.legend()
        plt.grid()

        results = aect.reconstruct_data(data=test_data, model=autoencoder,
                                        num=64,
                                        column=False, show=True, plot=True)

    if save:
        session = {"settings_sim": settings_sim,
                   "settings_data": settings_data,
                   "settings_ae": settings_ae,
                   "results": info}
        nb = len([f for f in os.listdir(aect.AE_PATH) if name in f])
        aect.save_autoencoder(autoencoder=autoencoder, session=session,
                              name=name + str(nb))

    return info["test"][-1]

""" main search """

def search_a():

    # setup
    settings_sim = {
            "data_training_size": 1024,
            "data_test_size": 96,
            "epochs": 252,
            "batch_size": 32,
            "learning_rate": 1e-3
    }

    settings_data = {
            "size": 50,
            "K": 5,
            "num_cue_patterns": 2,
            "lap_length": 50,
            "cue_positions": [10., 30.],
            "cue_sigma": 5.,
            "cue_beta": 40.,
            "cue_alpha": 0.2,
            "mec_binarized": True,
            "mec_sigma": 5,
    }

    settings_ae = {
        "input_dim": settings_data["size"],
        "encoding_dim": 50,
        "K": 5,
        "beta": 70.,
        "gain_out": 20.,
        "offset_out": 0.,
        "use_bias": True,
    }

    # parameters to iterate over
    beta_size = 5
    gain_size = 5
    encoding_dim_size = 5
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
            "data_training_size": 1448,
            "data_test_size": 256,
            "epochs": 1048,
            "batch_size": 100,
            "learning_rate": 1e-3
    }

    settings_data = {
            "size": 50,
            "K": 5,
    }

    settings_ae = {
        "dim_ei": settings_data["size"],
        "dim_ca1": 50,
        "K_ca1": 5,
        "K_eo": 5,
        "beta_ei": 25.,
        "beta_eo": 25.,
        "use_bias": False,
    }

    train_random_data(settings_sim=settings_sim,
                      settings_data=settings_data,
                      settings_ae=settings_ae,
                      save=save,
                      name="ae_random_nb_",
                      plot=plot)


def main_cue(save: bool=False, plot: bool=False):

    # setup
    settings_sim = {
            "data_training_size": 50*10,
            "data_test_size": 8,
            "epochs": 196,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "disable": False
    }

    settings_data = {
            "size": 50,
            "K": 5,
            "num_cue_patterns": 2,
            "lap_length": 50,
            "cue_positions": [10., 30.],
            "cue_sigma": 3.,
            "cue_beta": 40.,
            "cue_alpha": 0.2,
            "mec_binarized": True,
            "mec_sigma": 4,
            "cue_spacing": 5,
    }

    settings_ae = {
            "dim_ei": settings_data["size"],
            "dim_ca1": 50,
            "K_ca1": 17,
            "K_eo": 5,
            "beta_ei": 96.,
            "beta_eo": 25.,
            "use_bias": False,
    }

    train_cue_data(settings_sim=settings_sim,
                   settings_data=settings_data,
                   settings_ae=settings_ae,
                   save=save,
                   name="ae_cue_nb_",
                   plot=plot)




if __name__ == "__main__":

    # main(save=True, plot=True)
    main_cue(save=True, plot=True)
    # search_a()
