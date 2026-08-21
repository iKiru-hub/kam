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


def cross_entropy(x: torch.Tensor, y: torch.Tensor, eps=1e-8):
    return F.binary_cross_entropy(x, y)


def cosine_similarity_vec(x: torch.Tensor, y: torch.Tensor):
    return (y.T @ x) / (torch.norm(x) * torch.norm(y))


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
