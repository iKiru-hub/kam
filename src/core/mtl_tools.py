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
import core.ae_tools as aect
from core.utils import tqdm_enumerate
from core.logger import logger

AE_PATH = os.path.abspath(__file__).split("src")[0] + "src/data"



def cosine_criterion(x: torch.Tensor, y: torch.Tensor,
                     p1: float=0.2, p2: float=0.8):
    z = functions.cosine_similarity_vec(x, y)
    return (z.item() - p1) / p2



def train_for_accuracy(data: np.ndarray,
                       model: models.MTL | models.Autoencoder,
                       criterion: Callable=cosine_criterion,
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
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    results = np.zeros((num_samples, num_samples))

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
            for k, batch in enumerate(dataloader):
                if k > i: break
                if isinstance(model, models.Autoencoder):
                    x = batch[0].to(device, non_blocking=device.type == "cuda")
                else:
                    x = batch[-1].reshape(-1, 1)

                _ = model(x)

        # --- test a dataset with pattern index 0.. i
        if isinstance(model, models.MTL):
            model.pause_lr()
        model.eval()
        with torch.no_grad():
            # forward one pattern at a time
            for j, batch in enumerate(dataloader):
                if j > i: break
                if isinstance(model, models.Autoencoder):
                    x = batch[0].to(device, non_blocking=device.type == "cuda")
                else:
                    x = batch[-1].reshape(-1, 1)

                y = model(x)
                results[i, j] = criterion(x, y)

    return results, model



