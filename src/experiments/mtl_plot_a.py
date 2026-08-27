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
import core.functions as functions
import core.ae_tools as aect
import core.mtl_tools as mtlct
from core.logger import logger
from experiments.evolution._lib import id_eval


SIZE = 50
REPS = 10
TQDM_REPS = True
CUE_SPACING = 1
N = 3
ITERATIONS = 10

DIM_CA1 = 50
NUM_CUE_PATTERNS = 5
NOISE_LEVEL = 0.0
BIT_KIND = 0

PLASTICITY = "base"



""" functions """


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
                      noise_level: float=NOISE_LEVEL,
                      bit_kind: float=BIT_KIND,
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
    noisy_samples = []
    ca3_activity = []
    ca1_activity = []
    with torch.no_grad():
        for sample in track_data:
            if bit_kind == 0:
                z = dg.bitflip(x=sample, fraction=noise_level)
            else:
                z = dg.bitkill(x=sample, fraction=noise_level)
            x = torch.as_tensor(z, dtype=torch.float32).reshape(-1, 1)
            reconstructed.append(model(x).reshape(-1).cpu().numpy())
            noisy_samples.append(z)
            if return_internal_activity:
                ca3_activity.append(model._ca3.reshape(-1).cpu().numpy())
                ca1_activity.append(model._ca1.reshape(-1).cpu().numpy())
    reconstructed = np.asarray(reconstructed)
    if return_internal_activity:
        return (
            track_data,
            reconstructed,
            np.asarray(noisy_samples),
            np.asarray(ca3_activity),
            np.asarray(ca1_activity),
        )
    return track_data, reconstructed, noisy_samples


def plot_track_reconstruction(model: models.MTL,
                              data: np.ndarray,
                              noise_level: float=NOISE_LEVEL,
                              bit_kind: float=BIT_KIND,
                              lap_length: int=dg.DEFAULT_CUE_LAP_LENGTH):
    """Plot input and MTL reconstruction over one circular-track lap."""

    track_data, reconstructed, noisy = reconstruct_track(
        model=model,
        data=data,
        lap_length=lap_length,
        bit_kind=bit_kind,
        noise_level=noise_level
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
            # (track_data, reconstructed),
            (np.array(noisy), reconstructed),
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
    dim_ca1 = settings_mtl.get("dim_ca1", DIM_CA1)
    num_cue_patterns = settings_data.get("num_cue_patterns", NUM_CUE_PATTERNS)
    noise_level = settings_data.get("noise_level", NOISE_LEVEL)
    storage_noise_level = settings_data.get(
        "storage_noise_level", noise_level
    )
    bit_kind = settings_data.get("bit_kind", BIT_KIND)
    ae_train_noise_level = settings_sim.get(
        "ae_train_noise_level", noise_level
    )
    ae_bit_kind = settings_sim.get("ae_bit_kind", bit_kind)

    # logger(f"matching ae: {DIM_CA1=} {NOISE_LEVEL=} {NUM_CUE_PATTERNS=}")
    # autoencoder, info = aect.load_autoencoder(name=ae_name)
    out = aect.find_ae(dim_ca1=dim_ca1, num_cue_patterns=num_cue_patterns,
                       noise_level=ae_train_noise_level,
                       bit_kind=ae_bit_kind)
    if len(out) <= 0 and ae_bit_kind == 0:
        # Older sessions predate the bit_kind metadata field; bit-flip was
        # the only implementation at that time, so missing means legacy 0.
        legacy = aect.find_ae(
            dim_ca1=dim_ca1,
            num_cue_patterns=num_cue_patterns,
            noise_level=ae_train_noise_level,
        )
        out = [item for item in legacy
               if "bit_kind" not in item[2].get("settings_data", {})]
        if out:
            warnings.warn(
                "using a legacy AE session without bit_kind metadata as "
                "bit_kind=0",
                stacklevel=2,
            )
    if len(out) <= 0:
        sys.exit(
            "ERROR: no autoencoder found with "
            f"{dim_ca1=} {num_cue_patterns=} "
            f"train_noise={ae_train_noise_level} {ae_bit_kind=}"
        )
    def validation_loss(item):
        values = item[2].get("results", {}).get("test", [])
        return float(np.mean(values[-10:])) if values else np.inf

    name, autoencoder, info = min(out, key=validation_loss)
    # logger(f"loaded autoencoder {name}")

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
    for r in tqdm(range(reps), disable=True):

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
                                                      noise_level=storage_noise_level,
                                                      bit_kind=bit_kind,
                                                      disable=True)
        results[r] = logs["rec_loss"][-1]
        # results[r] = logs["rec_loss"]

    # results = results.mean(axis=0)
    # score = results.sum()/(len(results)**2/2)
    score = 1-results.mean()
    scoreid = 1.-results.mean()

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

    original, reconstructed, noisy, ca3_activity, ca1_activity = reconstruct_track(
        model=model,
        data=diagnostic_data,
        lap_length=settings_data.get(
            "lap_length", dg.DEFAULT_CUE_LAP_LENGTH
        ),
        return_internal_activity=True,
        bit_kind=bit_kind,
        noise_level=noise_level
    )
    autoencoder.eval()
    with torch.no_grad():
        ae_reconstruction = autoencoder(
            torch.as_tensor(noisy, dtype=torch.float32)
        ).cpu().numpy()
    return {
        "results": results,
        "original_stimuli": original,
        "reconstructed_stimuli": reconstructed,
        "ca3_activity": ca3_activity,
        "ca1_activity": ca1_activity,
        "score": score,
        "ae_score": float(np.mean((ae_reconstruction - original) ** 2)),
        "mtl_score": float(np.mean((reconstructed - original) ** 2)),
    }



""" main functions """


def main_cue(plot: bool):

    logger("-- MTL cue experiment --")
    logger(f"{NUM_CUE_PATTERNS=}")
    logger(f"{DIM_CA1=}")
    logger(f"{CUE_SPACING=}")
    logger(f"{BIT_KIND=}")

    # setup
    settings_sim = {
        "data_training_size": 50*CUE_SPACING*NUM_CUE_PATTERNS*N,
        "criterion": functions.mse,
        "reps": 1,
        "use_bias": False,
        "disable": False,
        "plot": plot,
        "ae_name": f"ae_{NUM_CUE_PATTERNS}cues_100ca10"
    }

    settings_data = {
        "size": 50,
        "K": 5,
        "num_cue_patterns": NUM_CUE_PATTERNS,
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

    noise_levels = np.around(np.linspace(0., 0.1, ITERATIONS), 2).astype(float)
    logger(f"{noise_levels=}")

    # plot
    fig, axs = plt.subplots(4, 2, sharex=True)
    fig.suptitle(f"{BIT_KIND=}")

    name = ("base", "err2", "btsp", "xbtsp")
    for k, (ax, ax2) in enumerate(axs):
        logs = np.zeros((ITERATIONS, ITERATIONS))
        aelogs = np.zeros((ITERATIONS, ITERATIONS))
        for i in tqdm(range(ITERATIONS)):
            for j in range(ITERATIONS):
                # print(f"{noise_levels[i]=}")
                found = mtlct.find_mtl(dim_ca1=DIM_CA1, noise_level=noise_levels[i],
                                       num_cue_patterns=NUM_CUE_PATTERNS,
                                       plasticity=name[k],
                                       bit_kind=BIT_KIND)
                if len(found) <= 0 and BIT_KIND == 0:
                    legacy = mtlct.find_mtl(
                        dim_ca1=DIM_CA1,
                        noise_level=noise_levels[i],
                        num_cue_patterns=NUM_CUE_PATTERNS,
                        plasticity=name[k],
                    )
                    found = [item for item in legacy
                             if "bit_kind" not in item[1].get("settings", {})]
                    if found:
                        warnings.warn(
                            "using a legacy MTL record without bit_kind "
                            "metadata as bit_kind=0",
                            stacklevel=2,
                        )
                if len(found) <= 0: sys.exit("no evolved MTL matches the current settings")
                # logger(f"saved MTL found: {[f[0] for f in found]}")

                selected = max(found, key=lambda item: float(item[1]["fitness"]))
                _settings_mtl = dict(selected[1]["best_parameters"])
                _settings_mtl["dim_ca3"] = 50
                _settings_mtl["num_swaps_ca1"] = 0
                _settings_mtl["num_swaps_ca3"] = 0
                _settings_mtl["random_IS"] = False
                _settings_mtl["plasticity"] = name[k]

                settings_data["noise_level"] = noise_levels[j]
                settings_data["storage_noise_level"] = noise_levels[i]
                settings_sim["ae_train_noise_level"] = noise_levels[i]
                settings_sim["ae_bit_kind"] = BIT_KIND

                # --
                info = train_mtl_cue_data(settings_sim=settings_sim,
                                          settings_data=settings_data,
                                          settings_mtl=_settings_mtl)
                logs[i, j] = info["mtl_score"]
                aelogs[i, j] = info["ae_score"]


        ax.imshow(logs, aspect="auto", cmap="Greens", vmin=0, vmax=0.1)
        ax.set_xticks(range(len(noise_levels)))
        ax.set_yticklabels(noise_levels)
        ax.set_ylabel("train noise")
        ax.set_title(f"MTL {name[k]}")

        im = ax2.imshow(aelogs, aspect="auto", cmap="Greens", vmin=0., vmax=0.1)
        ax2.set_xticks(range(len(noise_levels)))
        ax2.set_yticklabels(noise_levels)
        ax2.set_title("autoencoder")

        if k == 3:
            ax.set_xlabel("test noise")
            ax.set_xticks(range(len(noise_levels)))
            ax.set_xticklabels(noise_levels)
            ax2.set_xlabel("test noise")
            ax2.set_xticks(range(len(noise_levels)))
            ax2.set_xticklabels(noise_levels)
            plt.colorbar(im)

    plt.show()



if __name__ == "__main__":

    TQDM_REPS = True
    main_cue(plot=True)
