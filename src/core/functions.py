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

"""
=============================================================================
ACTIVATION FUNCTIONS
=============================================================================
"""


class Identity(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def sparsemoid(z: torch.Tensor, K: int,
               beta: float, flag=False) -> torch.Tensor:

    if K > 0:
        z_sorted = torch.sort(z, descending=True, dim=1).values

        alpha = z_sorted[:, K-1: K+1]
        alpha = alpha.mean(axis=1).reshape(-1, 1)
    else:
        alpha = 0.

    # apply
    z = beta * (z - alpha)
    return torch.sigmoid(z)

def generalized_sigmoid(x: float|np.ndarray, beta: float, alpha: float, top: float=1., offset: float=0.):
    return np.clip(top / (1 + np.exp(-beta * (x - alpha))) - offset, 0., 1.)


def normalized_sigmoid(x: torch.Tensor, beta: float,
                       alpha: float) -> torch.Tensor:
    """BTSP sigmoid normalized to map zero to 0 and one to 1.

    The calculation is performed in float64 to retain the ratio between the
    two small sigmoid values when the threshold is far above the [0, 1]
    overlap range, then converted back to the input dtype.
    """

    if not torch.is_tensor(x):
        raise TypeError("normalized_sigmoid expects a torch.Tensor")

    work = x.to(dtype=torch.float64)
    beta_tensor = work.new_tensor(abs(float(beta)))
    alpha_tensor = work.new_tensor(float(alpha))
    lower = torch.sigmoid(-beta_tensor * alpha_tensor)
    upper = torch.sigmoid(beta_tensor * (1.0 - alpha_tensor))
    denominator = upper - lower

    if denominator.abs() <= torch.finfo(work.dtype).tiny:
        # beta == 0 (or an extreme, numerically unresolved threshold) has no
        # sigmoid contrast. The continuous useful fallback is a linear gate.
        result = work.clamp(0.0, 1.0)
    else:
        raw = torch.sigmoid(beta_tensor * (work - alpha_tensor))
        result = ((raw - lower) / denominator).clamp(0.0, 1.0)
    return result.to(dtype=x.dtype)


def cross_entropy(x: torch.Tensor, y: torch.Tensor, eps=1e-8):
    return F.binary_cross_entropy(x, y)


def cosine_similarity_vec(x: torch.Tensor, y: torch.Tensor):
    denominator = (torch.norm(x) * torch.norm(y)).clamp_min(1e-8)
    return (y.T @ x) / denominator

def mse(x, y):
    # return torch.mean((x - y)**2)
    return -1. * F.mse_loss(x, y)*0.7 + 0.3*cosine_similarity_vec(x, y)

def gaussian(x, y):
    return torch.exp(-0.5*(torch.sum((x-y)**2)))

def modified_mse_loss(x: torch.Tensor, y: torch.Tensor,
                      cosine_weight: float=0.1) -> torch.Tensor:
    """Combine magnitude-sensitive MSE and cosine distance.

    The loss is zero for a perfect reconstruction. Unlike cosine distance on
    its own, it penalizes outputs that have the right direction but the wrong
    activity magnitude.
    """

    cosine_weight = float(cosine_weight)
    if not 0. <= cosine_weight <= 1.:
        raise ValueError("cosine_weight must be between 0 and 1")
    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have the same shape, got {x.shape} and {y.shape}"
        )

    x_vector = x.reshape(-1, 1)
    y_vector = y.reshape(-1, 1)
    cosine = cosine_similarity_vec(
        x_vector, y_vector
    ).squeeze().clamp(-1., 1.)
    both_zero = (torch.norm(x_vector) <= 1e-8) & (
        torch.norm(y_vector) <= 1e-8
    )
    cosine = torch.where(both_zero, torch.ones_like(cosine), cosine)
    pixel_mse = F.mse_loss(y, x)
    return (
        cosine_weight * (1. - cosine)
        + (1. - cosine_weight) * pixel_mse
    ).clamp_min(0.)


def modified_mse_score(x: torch.Tensor, y: torch.Tensor,
                       cosine_weight: float=0.5) -> float:
    """Return a higher-is-better score for evolutionary optimization."""

    loss = modified_mse_loss(
        x=x,
        y=y,
        cosine_weight=cosine_weight,
    )
    return float((1. - loss).clamp(0., 1.).item())

def cosine_similarity_mat(matrix1: np.ndarray, matrix2: np.ndarray):
    """
    Compute the normalized dot product (cosine similarity) between two matrices.

    Parameters
    ---------
    matrix1 : numpy.ndarray
        first matrix with shape (m, n)
    matrix2 : numpy.ndarray
        second matrix with shape (m, p)

    Returns:
    numpy.ndarray:
        cosine similarity matrix with shape (n, p)
    """

    # Compute the dot product
    dot_product = matrix1.T @ matrix2  # Shape: (n, p)

    # Compute the norms (Frobenius norm) for each column
    norm1 = np.linalg.norm(matrix1, axis=0).reshape(-1, 1)  # Shape: (n, 1)
    norm2 = np.linalg.norm(matrix2, axis=0).reshape(1, -1)  # Shape: (1, p)

    # Compute the outer product of norms
    norm_product = norm1 @ norm2  # Shape: (n, p)

    # Normalize the dot product by dividing by the norm product
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-8
    cosine_sim = dot_product / (norm_product + epsilon)

    return cosine_sim


def gaussian_kernel(x: np.ndarray, y: np.ndarray, sigma: float=1.0):
    """ standard formula: exp(-(x-y)^2 / (2 * sigma^2)) """
    return np.exp(-0.5 * ((x - y) / sigma)**2)


def circular_distance(x1: np.ndarray | float, x2: np.ndarray | float, N: int):
    """
    Computes the minimum circular distance in the
    x-direction (wraps around the boundaries).
    """
    return np.minimum(np.abs(x1 - x2), N - np.abs(x1 - x2))


def gaussian_circular_distance(x1: np.ndarray | float,
                               x2: np.ndarray | float,
                               N: int,
                               sigma: float=1.):
    """
    Computes the minimum circular distance in the
    x-direction (wraps around the boundaries).
    """
    dist = np.minimum(np.abs(x1 - x2), N - np.abs(x1 - x2))
    return np.exp( - 0.5 * (dist / sigma) ** 2)

if __name__ == "__main__":
    print(f"[{__file__.split("/")[-1]} done]")

    x = np.arange(10)
    y = 8
    print(circular_distance(x, y, 10))
    print()
    print(np.around(gaussian_circular_distance(x, y, 10, 1), 2))
