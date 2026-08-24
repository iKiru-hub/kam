import numpy as np
import matplotlib.pyplot as plt
import warnings
import torch
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


def reconstruct_inputs(model: models.MTL,
                       data: np.ndarray,
                       num_inputs: int=64):
    """Recall an age-spanning subset of inputs with learning disabled."""

    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2 or len(data) == 0:
        raise ValueError("data must be a non-empty (samples, units) array")

    num_inputs = min(max(int(num_inputs), 1), len(data))
    indices = np.linspace(0, len(data) - 1, num_inputs, dtype=int)
    original = data[indices]

    model.pause_lr()
    model.eval()
    reconstructed = []
    with torch.no_grad():
        for sample in original:
            x = torch.as_tensor(sample, dtype=torch.float32).reshape(-1, 1)
            reconstructed.append(model(x).reshape(-1).cpu().numpy())

    return indices, original, np.asarray(reconstructed)


def plot_input_reconstructions(model: models.MTL,
                               data: np.ndarray,
                               num_inputs: int=64):
    """Plot many stored inputs and their final MTL reconstructions."""

    indices, original, reconstructed = reconstruct_inputs(
        model=model,
        data=data,
        num_inputs=num_inputs,
    )
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    extent = (0, len(original), 0, original.shape[1])
    for axis, values, title in zip(
            axes,
            (original, reconstructed),
            ("Original random stimuli", "Final MTL reconstructions")):
        image = axis.imshow(
            values.T,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=extent,
            vmin=0.,
            vmax=1.,
            cmap="viridis",
        )
        axis.set_title(title)
        axis.set_ylabel("EC unit")

    tick_locations = np.linspace(0, len(original) - 1, min(6, len(original)))
    tick_indices = np.rint(tick_locations).astype(int)
    axes[-1].set_xticks(tick_indices + 0.5)
    axes[-1].set_xticklabels(indices[tick_indices])
    axes[-1].set_xlabel(
        "Training-stream index (oldest to newest sampled memories)"
    )
    figure.colorbar(image, ax=axes, label="activity", shrink=0.9)
    figure.suptitle(
        f"MTL recall after ongoing learning ({len(original)} inputs)"
    )
    return figure, axes



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
                       K_ca1=autoencoder._K_ca1,
                       K_eo=autoencoder._K_eo,
                       K_ca3=settings_mtl["K_ca3"],
                       dim_ca3=settings_mtl["dim_ca3"],
                       beta_is=autoencoder._beta_ei,
                       beta_ca3=settings_mtl["beta_ca3"],
                       beta_ca1=settings_mtl["beta_ca1"],
                       beta_eo=autoencoder._beta_eo,
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
    score = results.sum()/(len(results)**2/2)
    scoreid = id_eval(results).mean()

    if plot:
        logger(f"MTL results={np.around(results, 2)}")
        logger(f"accuracy={score:.3f}")
        logger(f"score 'id_eval'={scoreid:.3f}")

        figure, axis = plt.subplots(1, 1, figsize=(10, 10))
        image = axis.imshow(results, aspect="auto")
        figure.colorbar(image, ax=axis)
        axis.grid()
        axis.set_title(f"MTL recall accuracy, best={score:.3f}")

        plot_input_reconstructions(
            model=model,
            data=training_data,
            num_inputs=settings_sim.get("reconstruction_samples", 64),
        )
        plt.show()

    return results


""" main functions """


def main_random(plot: bool):

    # setup
    settings_sim = {
        "data_training_size": 1024,
        "criterion": mtlct.cosine_criterion,
        "reps": 5,
        "use_bias": False,
        "disable": False,
        "plot": True,
        "reconstruction_samples": 64,
        "ae_name": "ae_random_nb_1"
    }

    settings_data = {
        "size": 50,
        "K": 5,
    }

    settings_mtl = {
        "K_ca3": 5,
        "dim_ca3": 50,
        "beta_ca3": 196,
        "beta_ca1": 24,
        "alpha": 0.018,
        "nb_ei_ca3": 2,
        "num_swaps_ca1": 1,
        "num_swaps_ca3": 1,
        "random_IS": False,
        "plasticity": "base",
    }

    settings_mtl_err2 = {
        "K_ca3": 2,
        "dim_ca3": 50,
        "beta_ca3": 196,
        "beta_ca1": 42,
        "alpha": 0.052,
        "nb_ei_ca3": 2,
        "num_swaps_ca1": 1,
        "num_swaps_ca3": 1,
        "random_IS": False,
        "plasticity": "err2",
    }

    train_mtl_random_data(settings_sim=settings_sim,
                          settings_data=settings_data,
                          settings_mtl=settings_mtl_err2)
                          # settings_mtl=settings_mtl)

if __name__ == "__main__":

    main_random(plot=False)
