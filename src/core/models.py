from __future__ import annotations

import torch
from torch import nn

import numpy as np
import argparse
import os, sys, json
from pprint import pprint

sys.path.append(os.path.abspath(__file__).split("src")[0] + "src")

# local
import core.utils as utils
import core.functions as functions


"""
parameters:
    [EI]
    K
    dim_ei
    W_ei_ca1
    W_ei_ca3
    B_ei_ca1
    B_ei_ca3
    beta_ei
    K_ca1

    [CA3]
    dim_ca3
    K_ca3
    W_ca3_ca1
    beta_ca3
    num_swaps_ca3

    [CA1]
    dim_ca1
    K_ca1
    W_ca1_eo
    B_ca1_eo
    beta_ca1
    alpha
    num_swaps_ca1

    [EO]
    dim_eo # = dim_ei
    K_eo
    gain_out # remove
    offset_out # remove
    beta_eo


"""


# =============================================================================
# Autoencoder
# =============================================================================

class Autoencoder(nn.Module):

    # def __init__(self, input_dim: int=10, encoding_dim=10,
    #              K: int=10, beta: float=20.,
    #              gain_out: float=10., offset_out: float=0.1,
    #              use_bias: bool=True):

    def __init__(self, dim_ei: int=10, dim_ca1=10,
                 K_ca1: int=10, K_eo: int=10, beta_ei: float=20.,
                 beta_eo: float=10., use_bias: bool=True):

        """
        Simple autoencoder with a single linear layer as encoder and decoder.

        Parameters
        ----------
        input_dim: int
            the size of the input data
        encoding_dim: int
            the size of the encoded data
        K: int
            the number of top values to select.
            Default is 10
        beta: float
            the beta value for the sparsemoid function.
            Default is 20.
        use_bias: bool
            use bias in the linear layers.
            Default is True.
        """

        super(Autoencoder, self).__init__()

        self._dim_ei = dim_ei
        self._dim_ca1 = dim_ca1
        self._K_ca1 = K_ca1
        self._K_eo = K_eo
        self._beta_ei = beta_ei
        self._beta_eo = beta_eo
        self._use_bias = use_bias

        # Encoder
        self.encoder = nn.Sequential(nn.Linear(dim_ei,
                                               dim_ca1,
                                               bias=use_bias),)

        # Decoder
        self.decoder = nn.Sequential(nn.Linear(dim_ca1,
                                               dim_ei,
                                               bias=use_bias),)

    def forward(self, x: torch.Tensor, ca1: bool=False):

        """
        Forward pass

        Parameters
        ----------
        x: torch.Tensor
            input data
        latent: bool
            return the data from CA1 latent layer. Default is False

        Returns
        -------
        torch.Tensor
            reconstructed data
        """

        x_ca1 = self.encoder(x)
        x_ca1 = functions.sparsemoid(z=x_ca1, K=self._K_ca1,
                                     beta=self._beta_ei)

        # ---
        x_eo = self.decoder(x_ca1)
        # x_eo = functions.sparsemoid(z=x_eo, K=self._K_eo,
        #                             beta=self._beta_eo)
        x_eo = torch.sigmoid(self._beta_eo * x_eo)

        if ca1: return x_eo, x_ca1
        return x_eo

    def get_weights(self, bias: bool=False):

        """
        Get the weights of the autoencoder model

        Returns
        -------
        tuple
            the weights of the encoder and decoder
        """

        W_ei_ca1 = self.encoder[0].weight.data.reshape(self._dim_ca1,
                                                       self._dim_ei)
        W_ca1_eo = self.decoder[0].weight.data.reshape(self._dim_ei,
                                                       self._dim_ca1)

        if bias and self._use_bias:
            B_ei_ca1 = self.encoder[0].bias.data.reshape(-1, 1)
            B_ca1_eo = self.decoder[0].bias.data.reshape(-1, 1)
            return W_ei_ca1, W_ca1_eo, B_ei_ca1, B_ca1_eo

        return W_ei_ca1, W_ca1_eo, None, None


# =============================================================================
# MTL MODEL
# =============================================================================


class MTL(nn.Module):

    def __init__(self,
                 W_ei_ca1: torch.Tensor,
                 W_ca1_eo: torch.Tensor,
                 K_ca1: int,
                 K_eo: int,
                 K_ca3: int,
                 dim_ca3: int,
                 beta_eo: float,
                 beta_is: float,
                 beta_ca1: float,
                 beta_ca3: float,
                 alpha: float=0.01,
                 nb_ei_ca3: int=10,
                 num_swaps_ca1: int=0,
                 num_swaps_ca3: int=0,
                 identity_IS : bool=False,
                 random_IS : bool=False,
                 plasticity: str="base",
                 B_ei_ca1: torch.Tensor|None=None,
                 B_ca1_eo: torch.Tensor|None=None):

        # make docstrings
        """
        Multi-target learning model with BTSP learning rule

        Parameters
        ----------
        W_ei_ca1: torch.Tensor
            the weight matrix from entorhinal cortex to CA1
        W_ca1_eo: torch.Tensor
            the weight matrix from CA1 to entorhinal cortex output
        B_ei_ca1: torch.Tensor
            the bias for the EC to CA1 layer.
            Default is None
        B_ca1_eo: torch.Tensor
            the bias for the CA1 to EC output layer.
            Default is None
        K_ca1: int
            the number of top values to select
        K_eo: int
            the number of top values to select for the output
        K_ca3: int
            the number of top values to select for the CA3 layer
        beta: float
            the beta value for the sparsemoid function
        alpha: float
            the learning rate for the weight update
        num_swaps: int
            number of swaps of neural activations
        dim_ca3: int
            the size of the CA3 layer
        """

        super(MTL, self).__init__()

        # infer dimensions of EC input and output and CA1
        self._dim_ei = W_ei_ca1.shape[1]
        self._dim_eo = W_ca1_eo.shape[0]
        self._dim_ca1 = W_ca1_eo.shape[1]
        self._dim_ca3 = dim_ca3

        # network parameters
        self._K_ca1 = abs(int(K_ca1))
        self._K_ca3 = abs(int(K_ca3))
        self._K_eo = abs(int(K_eo))
        self._beta_eo = abs(float(beta_eo))
        self._beta_is = abs(float(beta_is))
        self._beta_ca3 = abs(float(beta_ca3))
        self._beta_ca1 = abs(float(beta_ca1))
        self._alpha = abs(float(alpha))
        self._num_swaps_ca1 = abs(int(num_swaps_ca1))
        self._num_swaps_ca3 = abs(int(num_swaps_ca3))
        self._nb_ei_ca3 = int(nb_ei_ca3)
        self._plasticity = str(plasticity).lower()

        # Initialize weight matrices for each layer
        # self.W_ei_ca3 = nn.Parameter(torch.randn(dim_ca3,
        #                                          self._dim_ei) / dim_ca3)
        assert self._dim_ca3 == self._dim_ei, "unqual dimension ei-ca3"
        if not 1 <= self._nb_ei_ca3 <= self._dim_ei:
            raise ValueError(
                "nb_ei_ca3 must be between 1 and the EC input dimension "
                f"({self._dim_ei}), got {self._nb_ei_ca3}"
            )
        connection_indices = utils.make_equal_tuning(
            dim_ca3,
            self._nb_ei_ca3,
        )
        W_ei_ca3 = torch.zeros(dim_ca3, self._dim_ei)
        for ca3_index, ec_indices in enumerate(connection_indices):
            W_ei_ca3[ca3_index, ec_indices] = 1.0 / dim_ca3
        self.W_ei_ca3 = nn.Parameter(W_ei_ca3)
        self.W_ei_ca1 = nn.Parameter(W_ei_ca1)
        self.W_ca3_ca1 = nn.Parameter(torch.zeros(self._dim_ca1, dim_ca3))
        self.W_ca1_eo = nn.Parameter(W_ca1_eo)

        # if B_ei_ca1 is None: print(">>> no bias uses in MTL")

        self.B_ei_ca1 = nn.Parameter(torch.zeros(self._dim_ca1, 1) \
                                    if B_ei_ca1 is None else B_ei_ca1)
        self.B_ca1_eo = nn.Parameter(torch.zeros(self._dim_eo, 1) \
                                    if B_ca1_eo is None else B_ca1_eo)
        self.is_bias = B_ei_ca1 is not None and B_ca1_eo is not None

        self._ca1 = None
        self._ca3 = None
        self._eo = None

        self.identity_IS = identity_IS
        self.random_IS = random_IS

        # mode
        self.mode = "train"

        self.recordings = {}
        self.recordings["x_ei"] = []
        self.recordings["ca3"] = []
        self.recordings["IS"] = []
        self.recordings["ca1"] = []
        self.recordings["eo"] = []
        self.recordings["W_ca3_ca1"] = []

    def __repr__(self):
        return f"MTLev(dim_ei={self._dim_ei}, dim_ca1={self._dim_ca1}," + \
            f" dim_ca3={self.W_ei_ca3.shape[0]}, dim_eo={self._dim_eo}," + \
            f" bias={self.is_bias}, " + \
            f" beta_is={self._beta_is},  beta_ca3={self._beta_ca3}," + \
            f" beta_eo={self._beta_eo},  beta_ca1={self._beta_ca1}," + \
            f" alpha={self._alpha}, K_ca1={self._K_ca1}," + \
            f" K_eo={self._K_eo}, num_swaps_ca1={self._num_swaps_ca1}" + \
            f" num_swaps_ca3={self._num_swaps_ca3}"

    def forward(self, x_ei: torch.Tensor, ca1: bool=False, test: bool=False):

        """
        Forward pass

        Parameters
        ----------
        x_ei: torch.Tensor
            input data
        ca1: bool
            return the data from CA1. Default is False
        test: bool
            Default is False

        Returns
        -------
        torch.Tensor
            reconstructed data
        """

        # forward pass through the entorhinal cortex to CA3
        x_ca3 = self.W_ei_ca3 @ x_ei # 50, 1
        x_ca3 = functions.sparsemoid(x_ca3.reshape(1, -1),
                                     K=self._K_ca3,
                                     beta=self._beta_ca3).reshape(-1, 1)

        x_ca3 = utils.get_sample_from_num_swaps(x_0=x_ca3,
                                                num_swaps=self._num_swaps_ca3)

        # forward pass through CA3 to CA1
        x_ca1 = self.W_ca3_ca1 @ x_ca3 + self.B_ca1_eo # 50, 1
        x_ca1 = functions.sparsemoid(x_ca1.reshape(1, -1),
                                     K=self._K_ca1,
                                     beta=self._beta_ca1,
                                     flag=False).reshape(-1, 1)

        x_ca1 = utils.get_sample_from_num_swaps(x_0=x_ca1,
                                                num_swaps=self._num_swaps_ca1)

        # compute instructive signal
        if self.identity_IS:
            IS = x_ei
        else:
            IS = self.W_ei_ca1 @ x_ei + self.B_ei_ca1
            IS = functions.sparsemoid(IS.reshape(1, -1), K=self._K_ca1,
                                      beta=self._beta_is).reshape(-1, 1)
            if self.random_IS:
                # permute the IS
                IS = IS[torch.randperm(IS.size(0))]

        # weight update
        if self.mode == "train" and not test:
            # method 1
            if self._plasticity == "base":
                self.W_ca3_ca1 = nn.Parameter((1 - IS * self._alpha) * \
                    self.W_ca3_ca1 + self._alpha * (IS @ x_ca3.T))
            elif self._plasticity == "nois":
                # method 2
                self.W_ca3_ca1 = nn.Parameter((1 - self._alpha) * \
                    self.W_ca3_ca1 + self._alpha * (IS @ x_ca3.T))
            elif self._plasticity == "isout":
                # method 3
                self.W_ca3_ca1 = nn.Parameter((1 - self._alpha) * IS * \
                    self.W_ca3_ca1 + self._alpha * (IS @ x_ca3.T))
            elif self._plasticity == "err1":
                # Normalized error-correcting (LMS/delta) rule.
                error = IS - x_ca1
                normalizer = x_ca3.square().sum().clamp_min(1e-6)
                update = self._alpha * (error @ x_ca3.T) / normalizer
                self.W_ca3_ca1 = nn.Parameter(
                    (self.W_ca3_ca1 + update).clamp(0.0, 1.0)
                )
            elif self._plasticity == "err2":
                # Bounded error-driven potentiation and depression.
                positive_error = torch.relu(IS - x_ca1)
                negative_error = torch.relu(x_ca1 - IS)
                potentiation = self._alpha * \
                    (positive_error @ x_ca3.T) * (1.0 - self.W_ca3_ca1)
                depression = self._alpha * \
                    (negative_error @ x_ca3.T) * self.W_ca3_ca1
                self.W_ca3_ca1 = nn.Parameter(
                    (self.W_ca3_ca1 + potentiation - depression).clamp(0.0, 1.0)
                )
            else:
                raise NameError("wrong plasticity provided")

        # Forward pass through CA1 to entorhinal cortex output
        x_eo = self.W_ca1_eo @ x_ca1 + self.B_ca1_eo

        # activation function
        # x_eo = functions.sparsemoid(x_eo.reshape(1, -1),
        #                             K=self._K_eo,
        #                             beta=self._beta_eo).reshape(-1, 1)
        # activation function
        x_eo = torch.sigmoid(x_eo * self._beta_eo).reshape(-1, 1)

        self._ca1 = x_ca1
        self._ca3 = x_ca3
        self._eo = x_eo

        # --
        self.record(x_ei, IS)

        if ca1: return x_eo, x_ca1
        return x_eo

    def pause_lr(self):
        """ Pause learning rate """

        self.mode = "test"

    @property
    def testing_mode(self):
        self.mode = "test"

    def resume_lr(self):
        """ Resume learning rate """

        self.mode = "train"

    @property
    def training_mode(self):
        self.mode = "train"

    def set_alpha(self, alpha: float):
        """ Set the learning rate """

        self._alpha = alpha

    def record(self, x_ei, IS):
        self.recordings["x_ei"].append(x_ei.clone())
        self.recordings["ca3"].append(self._ca3.clone())
        self.recordings["ca1"].append(self._ca1.clone())
        self.recordings["eo"].append(self._eo.clone())
        self.recordings["W_ca3_ca1"].append(self.W_ca3_ca1.clone())
        self.recordings["IS"].append(IS.clone())

    def reset(self):

        self._ca1 = None
        self._ca3 = None
        self._eo = None

        # mode
        self.mode = "train"
        self.recordings = {}
        self.recordings["x_ei"] = []
        self.recordings["ca3"] = []
        self.recordings["IS"] = []
        self.recordings["ca1"] = []
        self.recordings["eo"] = []
        self.recordings["W_ca3_ca1"] = []




if __name__ == "__main__":
    print(f"[{__file__.split("/")[-1]} done]")
