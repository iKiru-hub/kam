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



SIZE = 50
REPS = 2
TQDM_REPS = False


def make_cue_data(num_samples: int, settings_data: dict|None=None):
    """Generate exactly ``num_samples`` from the configured cue task."""

    settings_data = {} if settings_data is None else settings_data

    return dg.make_cue_track_data(
        num_samples=num_samples,
        size=settings_data.get("size", 10),
        num_cue_patterns=settings_data.get(
            "num_cue_patterns", dg.DEFAULT_CUE_NUM_PATTERNS
        ),
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


def reconstruct_track(model: models.MTL,
                      data: np.ndarray,
                      lap_length: int=dg.DEFAULT_CUE_LAP_LENGTH,
                      return_internal_activity: bool=False):
    """Return recall diagnostics for one track lap.

    When ``return_internal_activity`` is true, CA3 and CA1 activity from the
    same forward passes used to make the reconstructions are returned too.
    """

    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2 or len(data) == 0:
        raise ValueError("data must be a non-empty (samples, units) array")

    lap_length = int(lap_length)
    if lap_length < 1:
        raise ValueError("lap_length must be at least 1")
    displayed_length = min(lap_length, len(data))
    start = max(0, (len(data) // lap_length - 1) * lap_length)
    track_data = data[start:start + displayed_length]

    model.pause_lr()
    model.eval()
    reconstructed = []
    ca3_activity = []
    ca1_activity = []
    with torch.no_grad():
        for sample in track_data:
            x = torch.as_tensor(sample, dtype=torch.float32).reshape(-1, 1)
            reconstructed.append(model(x).reshape(-1).cpu().numpy())
            if return_internal_activity:
                ca3_activity.append(model._ca3.reshape(-1).cpu().numpy())
                ca1_activity.append(model._ca1.reshape(-1).cpu().numpy())
    reconstructed = np.asarray(reconstructed)
    if return_internal_activity:
        return (
            track_data,
            reconstructed,
            np.asarray(ca3_activity),
            np.asarray(ca1_activity),
        )
    return track_data, reconstructed


def plot_track_reconstruction(model: models.MTL,
                              data: np.ndarray,
                              lap_length: int=dg.DEFAULT_CUE_LAP_LENGTH):
    """Plot input and MTL reconstruction over one circular-track lap."""

    track_data, reconstructed = reconstruct_track(
        model=model,
        data=data,
        lap_length=lap_length,
    )
    displayed_length = len(track_data)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    extent = (0, displayed_length, 0, track_data.shape[1])
    images = []
    for axis, values, title in zip(
            axes,
            (track_data, reconstructed),
            ("Original stimuli", "MTL reconstruction")):
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
        images.append(image)
        axis.axhline(
            track_data.shape[1] / 2,
            color="white",
            linestyle="--",
            linewidth=1,
            alpha=0.8,
        )
        axis.set_title(title)
        axis.set_ylabel("EC unit (MEC below, LEC above)")

    tick_positions = np.linspace(0, displayed_length, 5)
    axes[-1].set_xticks(tick_positions)
    axes[-1].set_xticklabels([
        f"{position:g}" if index < 4 else "0 / lap end"
        for index, position in enumerate(tick_positions)
    ])
    axes[-1].set_xlabel("Position around circular track")
    figure.colorbar(images[-1], ax=axes, label="activity", shrink=0.9)
    figure.suptitle("Stimulus reconstruction around one circular-track lap")
    return figure, axes


def train_mtl_cue_data(settings_sim: dict,
                       settings_data: dict,
                       settings_mtl: dict,
                       return_diagnostics: bool=False):

    """Train and evaluate an MTL model on shared cue-task samples."""

    num_samples = settings_sim["data_training_size"]
    ae_name = settings_sim.get("ae_name", "ae_cue_nb_0")
    use_bias = settings_sim.get("use_bias", False)
    plot = settings_sim.get("plot", False)
    disable = settings_sim.get("disable", False)

    autoencoder, info = aect.load_autoencoder(name=ae_name)
    params = autoencoder.get_weights(bias=use_bias)

    model = models.MTL(W_ei_ca1=params[0],
                       W_ca1_eo=params[1],
                       K_ca1=autoencoder._K_ca1,
                       K_eo=autoencoder._K_eo,
                       K_ca3=settings_mtl["K_ca3"],
                       dim_ca3=settings_mtl["dim_ca3"],
                       beta_is=autoencoder._beta_ei,
                       beta_ca3=settings_mtl["beta_ca3"],
                       beta_ca1=settings_mtl["beta_ca1"],
                       beta_eo=autoencoder._beta_eo,
                       alpha=settings_mtl["alpha"],
                       alpha_plus=settings_mtl.get("alpha_plus"),
                       alpha_minus=settings_mtl.get("alpha_minus"),
                       a_plus=settings_mtl.get("a_plus", 0.),
                       b_plus=settings_mtl.get("b_plus", 1.),
                       a_minus=settings_mtl.get("a_minus", 0.),
                       b_minus=settings_mtl.get("b_minus", 1.),
                       nb_ei_ca3=int(settings_mtl.get("nb_ei_ca3", 10)),
                       num_swaps_ca3=settings_mtl["num_swaps_ca3"],
                       num_swaps_ca1=settings_mtl["num_swaps_ca1"],
                       B_ei_ca1=params[2],
                       B_ca1_eo=params[3],
                       random_IS=settings_mtl["random_IS"],
                       plasticity=settings_mtl.get("plasticity", "base"))

    reps = settings_sim.get("reps", 32)
    criterion = settings_sim.get("criterion", mtlct.cosine_criterion)

    # results = np.zeros((reps, num_samples, num_samples))
    results = np.zeros((reps, num_samples))
    rdisable = not(reps > 2)
    # for r in tqdm(range(reps), disable=not(rdisable and (not disable))):
    for r in tqdm(range(reps), disable=not TQDM_REPS):

        # training_data = make_cue_data(
        #     num_samples=num_samples,
        #     size=settings_data["size"],
        # )

        training_data = make_cue_data(
            num_samples=num_samples,
            settings_data=settings_data,
        )

        # logs, model = mtlct.train_for_accuracy(data=training_data,
        #                                        model=model,
        #                                        criterion=criterion,
        #                                        disable=disable)
        logs, model = mtlct.train_for_accuracy_single(data=training_data,
                                                      model=model,
                                                      criterion=criterion,
                                                      test_last=True,
                                                      disable=True)
        results[r] = logs["rec_loss"][-1]
        # results[r] = logs["rec_loss"]

    # results = results.mean(axis=0)
    # score = results.sum()/(len(results)**2/2)
    score = 1-results.mean()
    scoreid = 1.-results.mean()
    # print(f"{score=}")
    # scoreid = id_eval(results).mean()

    diagnostic_data = None
    if plot or return_diagnostics:
        lap_length = int(settings_data.get(
            "lap_length", dg.DEFAULT_CUE_LAP_LENGTH
        ))
        diagnostic_data = training_data
        if len(diagnostic_data) < lap_length:
            # Fitness experiments may deliberately use a short prefix.  The
            # live display still evaluates the trained model around one full
            # track lap so generations remain visually comparable.
            diagnostic_data = make_cue_data(
                num_samples=lap_length,
                settings_data=settings_data,
            )

    if plot:
        logger(f"MTL results={np.around(results, 2)}")
        logger(f"accuracy={score:.3f}")
        logger(f"score 'id_eval'={scoreid:.3f}")
        logger(f"num={len(training_data)}")

        figure, axis = plt.subplots(1, 1, figsize=(10, 10))
        image = axis.imshow(results.reshape(1, -1), aspect="auto")
        figure.colorbar(image, ax=axis)
        axis.grid()
        axis.set_title(f"MTL recall accuracy, best={score:.3f}")

        plot_track_reconstruction(
            model=model,
            data=diagnostic_data,
            lap_length=settings_data.get(
                "lap_length", dg.DEFAULT_CUE_LAP_LENGTH
            ),
        )

        # fig, axs = plt.subplots(1, 750//50)
        # print(f"{len(axs)=}")
        _data = []
        _tdata = []
        print(f"{len(logs['reconstructions'][0])=}")
        for k in range(0, len(logs["reconstructions"][0]), 2):
            _data += [logs["reconstructions"][0][k]]
            _tdata += [logs["target"][0][k]]
        fig, axs = plt.subplots(2, 1)
        axs[0].imshow(np.stack(_tdata).T.reshape(50, -1), aspect="auto")
        axs[0].set_title("target")
        axs[1].imshow(np.stack(_data).T.reshape(50, -1), aspect="auto")
        axs[1].set_title("recall")
        # for i, ax in enumerate(axs.flatten()):
        #     if i > (len(_data)-1):
        #         ax.axis("off")
        #         continue
        #     ax.imshow(_data[i], aspect="auto")
        #     ax.set_title(f"{i}")

        plt.show()

    if return_diagnostics:
        original, reconstructed, ca3_activity, ca1_activity = reconstruct_track(
            model=model,
            data=diagnostic_data,
            lap_length=settings_data.get(
                "lap_length", dg.DEFAULT_CUE_LAP_LENGTH
            ),
            return_internal_activity=True,
        )
        return {
            "results": results,
            "original_stimuli": original,
            "reconstructed_stimuli": reconstructed,
            "ca3_activity": ca3_activity,
            "ca1_activity": ca1_activity,
        }

    return results


""" main functions """


def main_cue(plot: bool):

    # setup
    plasticity = "err2"
    cue_spacing = 1
    num_cue_patterns = 15
    settings_sim = {
        "data_training_size": 50*cue_spacing*15,
        "criterion": mtlct.cosine_criterion,
        "reps": 5,
        "use_bias": False,
        "disable": False,
        "plot": plot,
        "ae_name": f"ae_{num_cue_patterns}cues_0"
    }

    settings_data = {
        "size": 50,
        "K": 5,
        "num_cue_patterns": num_cue_patterns,
        "lap_length": 50,
        "cue_positions": [10., 30.],
        "cue_sigma": 3.,
        "cue_beta": 40.,
        "cue_alpha": 0.2,
        "mec_binarized": True,
        "mec_sigma": 4,
        "cue_spacing": cue_spacing,
    }

    _rpath = os.path.abspath(__file__).split("src")[0] + \
            "src/experiments/evolution/data"
    name = f"{_rpath}/{plasticity}_mtl_{num_cue_patterns}cue.json"
    print(f"{name=}")

    settings_mtl = {
        "K_ca3": 3,
        "dim_ca3": 50,
        "beta_ca3": 86,
        "beta_ca1": 37,
        "alpha": 0.024,
        "nb_ei_ca3": 18,
        "num_swaps_ca1": 0,
        "num_swaps_ca3": 0,
        "random_IS": False,
        "plasticity": "base",
    }

    settings_mtl_err2 = {
        "K_ca3": 3,
        "dim_ca3": 50,
        "beta_ca3": 114,
        "beta_ca1": 53,
        "alpha": 0.073,
        "nb_ei_ca3": 11,
        "num_swaps_ca1": 0,
        "num_swaps_ca3": 0,
        "random_IS": False,
        "plasticity": "err2",
    }

    settings_mtl_xbtsp = {
        "K_ca3": 5,
        "dim_ca3": 50,
        "beta_ca3": 161,
        "beta_ca1": 34,
        "alpha": 0.55,
        "alpha_plus": 0.084,
        "alpha_minus": 0.088,
        "a_plus": 43.,
        "b_plus": 0.001,
        "a_minus": 87.,
        "b_minus": 0.36,
        "nb_ei_ca3": 37,
        "num_swaps_ca1": 0,
        "num_swaps_ca3": 0,
        "random_IS": False,
        "plasticity": "xbtsp",
    }

    settings_mtl_btsp = {
        "K_ca3": 13,
        "dim_ca3": 50,
        "beta_ca3": 101,
        "beta_ca1": 91,
        "alpha": 0.21,
        "alpha_plus": 0.079,
        "alpha_minus": 0.164,
        "a_plus": 113.,
        "b_plus": 0.74,
        "a_minus": 85.,
        "b_minus": 0.099,
        "nb_ei_ca3": 25,
        "num_swaps_ca1": 0,
        "num_swaps_ca3": 0,
        "random_IS": False,
        "plasticity": "err2",
    }

    settings_by_plasticity = {
        "base": settings_mtl,
        "err2": settings_mtl_err2,
        "xbtsp": settings_mtl_xbtsp,
        "btsp": settings_mtl_btsp,
    }
    _settings_mtl = settings_by_plasticity[plasticity].copy()
    try:
        with open(name, "r") as file:
            evolved_settings = json.load(file)["best_parameters"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as error:
        warnings.warn(
            f"Could not load evolved MTL parameters from {name}: {error}. "
            "Using the built-in settings instead.",
            RuntimeWarning,
        )
    else:
        _settings_mtl.update(evolved_settings)

    train_mtl_cue_data(settings_sim=settings_sim,
                       settings_data=settings_data,
                       settings_mtl=_settings_mtl)

if __name__ == "__main__":

    TQDM_REPS = True
    main_cue(plot=True)
