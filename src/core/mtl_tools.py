import numpy as np
import matplotlib.pyplot as plt
import warnings

import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

from tqdm import tqdm
import os, sys
import json
from pathlib import Path
from collections.abc import Callable

sys.path.append(os.path.abspath(__file__).split("src")[0] + "src")
import core.functions as functions
import core.models as models
import core.datagen as dg
import core.ae_tools as aect
from core.constants import AE_PATH, MTL_PATH
from core.utils import tqdm_enumerate
from core.logger import logger



def cosine_criterion(x: torch.Tensor, y: torch.Tensor,
                     p1: float=0.2, p2: float=0.8):
    z = functions.cosine_similarity_vec(x, y)
    return (z.item() - p1) / p2


def train_for_accuracy(data: np.ndarray,
                       model: models.MTL | models.Autoencoder,
                       criterion: Callable=cosine_criterion,
                       test_last: bool=False,
                       disable: bool=True) -> tuple:

    """
    trainings for a given alpha (already set in the model)

    Parameters
    ----------
    alpha : float
        learning rate
    num_rep : int
        number of repetitions
    num_samples : int
        number of samples
    disable : bool
        disable tqdm bar

    Returns
    -------
    np.ndarray
        outputs
    """

    # data
    num_samples = len(data)
    datasets = []
    for k in range(num_samples):
        tensor_data = torch.tensor(data[:k+1], dtype=torch.float32)
        dataloader = DataLoader(TensorDataset(tensor_data),
                                batch_size=1,
                                shuffle=False)
        datasets += [dataloader]

    logs = {
        "train_loss": np.zeros((num_samples, num_samples)),
        "rec_loss": np.zeros((num_samples, num_samples))
    }

    if isinstance(model, models.Autoencoder):
        device = aect._resolve_device("mps")
        model.to(device)

    # --- run new repetition
    for i in tqdm(range(num_samples), disable=disable):

        # reset the model
        if isinstance(model, models.MTL):
            model.reset()
            model.resume_lr()

        # train a dataset with pattern index 0.. i
        model.eval()
        with torch.no_grad():

            # forward one pattern at a time
            # for k, batch in enumerate(dataloader):
            for j, batch in enumerate(datasets[i]):
                # if k > i: break
                # if isinstance(model, models.Autoencoder):
                #     x = batch[0].to(device, non_blocking=device.type == "cuda")
                # else:
                #     x = batch[-1].reshape(-1, 1)
                x = batch[-1].reshape(-1, 1)
                y = model(x)

                logs["train_loss"][i, j] = criterion(x, y)

        # --- test a dataset with pattern index 0.. i
        if isinstance(model, models.MTL):
            model.pause_lr()
        model.eval()
        with torch.no_grad():
            # forward one pattern at a time
            # for j, batch in enumerate(dataloader):
            #     if j > i: break
                # if isinstance(model, models.Autoencoder):
                #     x = batch[0].to(device, non_blocking=device.type == "cuda")
                # else:
                #     x = batch[-1].reshape(-1, 1)
            # for j, test_x in enumerate(data):
            if test_last and i < (num_samples-1):
                continue
            for j, batch in enumerate(datasets[i]):
                x = batch[-1].reshape(-1, 1)

                y = model(x)
                logs["rec_loss"][i, j] = criterion(x, y)

    return logs, model



def train_for_accuracy_single(data: np.ndarray,
                              model: models.MTL | models.Autoencoder,
                              criterion: Callable=cosine_criterion,
                              test_last = False,
                              noise_level: float=0.0,
                              bit_kind: int=0,
                              disable: bool=True) -> tuple:

    """
    trainings for a given alpha (already set in the model)

    Parameters
    ----------
    alpha : float
        learning rate
    num_rep : int
        number of repetitions
    num_samples : int
        number of samples
    disable : bool
        disable tqdm bar

    Returns
    -------
    np.ndarray
        outputs
    """

    # data
    num_samples = len(data)

    dataset = torch.tensor(data, dtype=torch.float32)
    # dataset = DataLoader(TensorDataset(tensor_data), batch_size=1, shuffle=False)

    logs = {
        "train_loss": np.zeros((num_samples, num_samples)),
        "rec_loss": np.zeros((num_samples, num_samples)),
        "target": [],
        "reconstructions": [],
    }

    if isinstance(model, models.Autoencoder):
        device = aect._resolve_device("mps")
        model.to(device)

    # --- run new repetition
    for i in tqdm(range(num_samples), disable=disable):

        # reset the model
        if isinstance(model, models.MTL):
            model.reset()
            model.resume_lr()

        # train a dataset with pattern index 0.. i
        model.eval()
        with torch.no_grad():

            # x = dataset[i].reshape(-1, 1)
            z = dataset[i].reshape(-1, 1)
            x = torch.tensor(dg.bitflip(x=z, fraction=noise_level))

            if bit_kind == 0:
                x = torch.tensor(dg.bitflip(x=z, fraction=noise_level))
            elif bit_kind == 1:
                x = torch.tensor(dg.bitkill(x=z, fraction=noise_level))
            else:
                x = torch.tensor(dg.bitkill(x=z, fraction=noise_level))
            y = model(x)

            logs["train_loss"][i] = criterion(y, z)

        if test_last and i < (num_samples-1):
            continue

        # --- test a dataset with pattern index 0.. i
        if isinstance(model, models.MTL):
            model.pause_lr()
        model.eval()
        _rsamples = []
        # _tsamples = []
        with torch.no_grad():
            # forward one pattern at a time
            _tsamples = []
            _rsamples = []
            for j, batch in enumerate(dataset[:i]):


                # x = torch.tensor(dg.bitflip(x=batch.reshape(-1, 1),
                #                             fraction=0.9))
                # y = model(x)

                # x = dataset[i].reshape(-1, 1)
                # x = torch.tensor(dg.bitflip(x=batch, fraction=noise_level).reshape(-1, 1))

                if bit_kind == 0:
                    x = torch.tensor(dg.bitflip(x=batch, fraction=noise_level).reshape(-1, 1))
                elif bit_kind == 1:
                    x = torch.tensor(dg.bitkill(x=batch, fraction=noise_level).reshape(-1, 1))
                else:
                    x = dg.bitnoise(x=batch, fraction=noise_level).reshape(-1, 1)
                y = model(x)

                _tsamples += [x.detach().numpy()]
                _rsamples += [y.detach().numpy()]
                logs["rec_loss"][i, j] = criterion(x, y)

            logs["target"] += [_tsamples]
            logs["reconstructions"] += [_rsamples]

    return logs, model



def _load_mtl(name: str):

    try:
        with open(f"{MTL_PATH}/{name}", "r") as f:
            file = json.load(f)
        return file
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as error:
        return None



def _match_mtl(name: str, dim_ca1: int|None=None, noise_level: float|None=None,
               num_cue_patterns: int|None=None, plasticity: str|None=None,
               bit_kind: int|None=None) -> float:

    session = _load_mtl(name)
    if session is None: return 0

    score = 0
    tot = 0
    # print(session.keys())
    if dim_ca1 is not None:
        if "dim_ca1" not in session["settings"].keys():
            return 0.
        score += int(dim_ca1 == session["settings"]["dim_ca1"])
        tot += 1
    if noise_level is not None:
        if "noise_level" not in session["settings"].keys():
            return 0.
        score += int((noise_level-session["settings"]["noise_level"])**2 < 0.0001)
        tot += 1
    if num_cue_patterns is not None:
        if "num_cue_patterns" not in session["settings"].keys():
            return 0.
        score += int(num_cue_patterns == session["settings"]["num_cue_patterns"])
        tot += 1
    if plasticity is not None:
        if "plasticity" not in session["settings"].keys():
            return 0.
        score += int(plasticity == session["settings"]["plasticity"])
        tot += 1
    if bit_kind is not None:
        if "bit_kind" not in session["settings"].keys():
            return 0.
        score += int(bit_kind == session["settings"]["bit_kind"])
        tot += 1

    return score / tot


def find_mtl(dim_ca1: int|None=None, noise_level: float|None=None,
             num_cue_patterns: int|None=None, plasticity: str|None=None,
             bit_kind: int|None=None):

    """
    attempt to retrieve the saved MTL evolution records that satisfy all provided
    conditions in their training setup.
    It returns a list of all matches in finding order.
    """

    out = []
    for _mtl in sorted(os.listdir(MTL_PATH)):
        score = _match_mtl(name=_mtl, dim_ca1=dim_ca1, noise_level=noise_level,
                           num_cue_patterns=num_cue_patterns,
                           plasticity=plasticity, bit_kind=bit_kind)
        if score == 1:
            _loaded = _load_mtl(_mtl)
            if _loaded is not None:
                out += [[_mtl, _loaded]]
                # print(f"retrieved: {_mtl} with {dim_ca1=}, {noise_level=}, {num_cue_patterns=}, {score=}")

    return out


