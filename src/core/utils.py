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

try:
    from numba import jit
except ModuleNotFoundError:
    def jit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator

from tqdm import tqdm
import os, sys
import json
sys.path.append(os.path.abspath(__file__).split("src")[0] + "src")

import core.models as models
import core.visualization as visualization
from core.logger import logger


MEDIA_PATH = "".join((os.getcwd().split("KAMemory")[0], "KAMemory/media/"))
DATA_PATH = "".join((os.getcwd().split("KAMemory")[0], "KAMemory/src/data/"))
AE_PATH = "".join((os.getcwd().split("KAMemory")[0], "KAMemory/src/data/autoencoders/"))
CONFIGS_PATH = "".join((os.getcwd().split("KAMemory")[0], "KAMemory/src/configs/"))



"""
=============================================================================
MISCELLANOUS
=============================================================================
"""


def tqdm_enumerate(iter, **tqdm_kwargs):
    """ use 'enumerate' together with 'tqdm' progress bar """
    i = 0
    for y in tqdm(iter, **tqdm_kwargs):
        yield i, y
        i += 1


def calc_capacity(outputs: np.ndarray, threshold: float,
                  nsmooth: int=20, idx_pattern: int=None) -> int:

    """
    This function calculates the capacity of the network
    by finding the number of patterns that can be stored.

    Parameters
    ----------
    outputs : np.ndarray
        outputs of the network
    threshold : float
        threshold value
    nsmooth : int, optional
        smoothing factor, by default 20
    idx_pattern : int, optional
        index of the pattern, by default None

    Returns
    -------
    int or list
        capacity
    """

    assert outputs.ndim == 2, "outputs must be 2D"

    if idx_pattern is None:

        results = []
        for i in range(outputs.shape[0]):
            idx = calc_capacity(outputs=outputs,
                                threshold=threshold,
                                nsmooth=nsmooth,
                                idx_pattern=i)
            results.append(idx)

        return results

    # select the first pattern and pad it
    padded_out = np.pad(outputs[idx_pattern:, idx_pattern],
                        (nsmooth-1, 0), mode="edge")

    # smooth it
    outputs = np.convolve(padded_out,
                      np.ones(nsmooth)/nsmooth,
                      mode="valid")

    # find the highest index where the output
    # is below the threshold
    idx = np.argmin(np.where(outputs >= threshold,
                             outputs, -np.inf),
                    axis=0).item()

    return idx


def save_model(loss_ae: float, autoencoder: object,
               configs: dict):

    """
    save the session logs and the autoencoder parameters

    Parameters
    ----------
    loss_ae : float
        final test loss of the model
    autoencoder : Autoencoder
    """

    # -- import
    import os
    import datetime
    import json

    # --
    info = {
        # "dim_ei": CONFIGS['dim_ei'],
        # "dim_ca3": CONFIGS['dim_ca3'],
        # "dim_ca1": CONFIGS['dim_ca1'],
        # "dim_eo": CONFIGS['dim_eo'],
        # "K": CONFIGS['K'],
        # "K_lat": CONFIGS['K_lat'],
        # "beta": CONFIGS['beta'],
        # "learning_rate": CONFIGS['learning_rate'],
        "hyperparameters": configs,

        "num_samples": configs['num_samples'],
        "epochs": configs['epochs'],
        "batch_size": configs['batch_size'],
        "loss_ae": round(loss_ae, 5),
        "date": datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    }

    # -- make directory
    nb = len([name for name in os.listdir(utils.AE_PATH) \
        if os.path.isdir(f"{utils.AE_PATH}/{name}") and "ae_" in name])

    dir_name = f"{utils.AE_PATH}/ae_{nb}"
    os.makedirs(dir_name, exist_ok=True)

    # -- save model
    torch.save(autoencoder.state_dict(), f"{dir_name}/autoencoder.pt")

    # save info
    with open(f"{dir_name}/info.json", "w") as f:
        json.dump(info, f)

    logger(f"[Autoencoder saved in {dir_name}]")




"""
=============================================================================
STIMULUS GENERATOR
=============================================================================
"""


def circular_distance(x1: np.ndarray, x2: np.ndarray, N: int):
    """
    Computes the minimum circular distance in the x-direction (wraps around the boundaries).
    """
    return np.minimum(np.abs(x1 - x2), N - np.abs(x1 - x2))


def place_field_activity(N_x: int, N_y: int, sigma: float, xi: float, yi: float):
    """
    Computes place field activity for each cell on an NxN grid for a given location (xi, yi).
    """

    # Create a grid of size NxN with place cells at each position
    x = np.linspace(0, N_x-1, N_x)
    y = np.linspace(0, N_y-1, N_y)
    X, Y = np.meshgrid(x, y)
    # Calculate the squared Euclidean distance between (xi, yi) and each place cell location
    dist_squared = circular_distance(X, xi, N_x) ** 2 + (Y - yi) ** 2

    # Compute Gaussian activity for each place cell
    activity = np.exp(-dist_squared / (2 * sigma ** 2))
    return activity


def stimulus_generator(N: int, size: int=10, heads: int=2, variance: float=0.1,
                       higher_heads: int=None, higher_variance: float=None,
                       plot: bool=False, use_uniform: bool=True) -> np.ndarray:

    """
    This function generates random z patterns with a certain
    degree of structure

    Parameters
    ----------
    N : int
        Number of samples
    size : int, optional
        Size of the z patterns, by default 10
    # generate docstring
        heads : int, optional
        Number of heads, by default 2
    variance : float, optional
        Variance of the Gaussian used to generate the z patterns, by default 0.1
    higher_heads : int, optional
        Higher number of heads, by default None
    higher_variance : float, optional
        Higher variance of the Gaussian used to generate the z patterns, by default None
    plot : bool, optional
        Whether to plot the z patterns, by default False

    Returns
    -------
    samples : np.ndarray
        z patterns
    """

    # generate the position of the heads drawing from a distribution defined
    # by higher_heads and higher_variance
    if higher_heads is not None and higher_variance is not None:
        if higher_heads != heads:
            warnings.warn("higher_heads must be equal to heads, setting higher_heads = heads")
            higher_heads = heads

        high_mu = np.linspace(1/(higher_heads+1), 1 - 1/(higher_heads+1),
                              higher_heads, endpoint=True) * size
        high_variance = np.array([higher_variance]*heads) * size

        # generate the positions of the heads
        mu = np.zeros((N, heads))
        for i in range(N):
            if use_uniform:
                mu[i, :] = np.random.choice(range(size), replace=False, size=heads)
            else:
                for k, (hh, hv) in enumerate(zip(high_mu, high_variance)):
                    np.random.normal(hh, hv)
        variance = np.array([variance]*heads) * size

        # tile for the number of samples
        variance = np.tile(variance, (N, 1))

    # generate the position of the heads as equidistant points
    else:
        mu = np.linspace(1/(heads+1), 1 - 1/(heads+1), heads, endpoint=True) * size
        variance = np.array([variance]*heads) * size

        # tile for the number of samples
        mu = np.tile(mu, (N, 1))
        variance = np.tile(variance, (N, 1))

    # generate the z patterns
    samples = np.zeros((N, size))
    for i in range(N):
        for k in range(heads):
            for x in range(size):
                p = np.exp(-((x-mu[i, k])**2)/(2*variance[i, k]))
                samples[i, x] += np.random.binomial(1, p)

    if plot:
        visualizations.plot_stimuli(samples=samples)

    return samples


def sparse_stimulus_generator(N: int, K: int, size: int=10,
                              plot: bool=False) -> np.ndarray:

    """
    This function generates random z patterns with a certain
    degree of sparsity

    Parameters
    ----------
    N : int
        Number of samples
    K : int
        Number of active units
    size : int, optional
        Size of the z patterns, by default 10
    plot : bool, optional
        Whether to plot the z patterns.
        Default False

    Returns
    -------
    samples : np.ndarray
        z patterns
    """

    samples = np.zeros((N, size))
    for i in range(N):
        idx = np.random.choice(range(size), replace=False, size=K)
        samples[i, idx] = 1

    samples = samples.astype(np.float32)

    if plot:
        visualizations.plot_stimuli(samples=samples)

    return samples


def sparse_stimulus_generator_sensory(num_stimuli: int, K : int,
                                      mec_size: int,  lec_size: int,
                                      N_x : int, N_y : int,
                                      pf_sigma: int,
                                      num_laps: int,
                                      lap_length: int=None,
                                      num_cues: int=None,
                                      position_list=None,
                                      cue_positions=None,
                                      sen_list=None,
                                      plot: bool=False,
                                      sigma: float=5,
                                      verbose: bool=True,
                                      binarize: bool=False) -> np.ndarray:

    """
    This function generates random z patterns with a certain
    degree of sparsity

    Parameters
    ----------
    N : int
        Number of samples
    K : int
        Number of active units
    size : int, optional
        Size of the z patterns, by default 10
    plot : bool, optional
        Whether to plot the z patterns.
        Default False

    Returns
    -------
    samples : np.ndarray
        z patterns
    """


    # --- init
    IS_CUE = position_list is not None

    # --- make cue patterns
    #cue_pattern_size = lec_size//num_cues  #
    if lec_size > 0 and num_cues is not None:
        fixed_cue = np.zeros((num_cues, lec_size))
        for k in range(num_cues):
            cue_idx = np.random.choice(range(lec_size), replace=False, size=K)
            #fixed_cue[cue_idx] = 1
            fixed_cue[k, K*k: K*(k+1)] = 1  # | 1, 1, .0s  , 1, 1, ..0s.. |

    # 0, 0, 1, 0, 1, 1...
    #lap_cues = np.random.choice(range(num_cues), size=num_laps) if num_laps is not None else None
    lap_cues = np.zeros(num_laps) if position_list is not None else None
    position_pool = np.arange(N_x)  # Possible positions in 1D track
    samples = np.zeros((num_stimuli, mec_size + lec_size))
    alpha_samples = np.zeros(num_stimuli)

    lap_i = 0
    cue_duration = 1

    # --- loop over each time step in each lap
    for i in range(num_stimuli): # laps x length

        # determine lap-cue association
        if IS_CUE:

            # count laps
            if i % lap_length == 0:
                lap_idx = (lap_i // cue_duration) % num_cues
                lap_cues[lap_i] = lap_idx

                if verbose:
                    print(f"-------------\nlap {lap_i}, cue {lap_idx}")
                lap_i += 1

        # SPATIAL input
        if mec_size > 0:
            # Reset the pool every 50 samples
            if position_list is None:
              if i % mec_size == 0:
                  np.random.shuffle(position_pool)  # Shuffle the pool to get random order
              pos_idx = i % 50  # Get position index within the shuffled pool
              x_i = position_pool[pos_idx]  # Choose the x position
              y_i = np.random.randint(0, N_y)
            else:
              x_i, y_i = position_list[i]
            activity_grid = place_field_activity(N_x, N_y, pf_sigma, x_i, y_i)
            if binarize:
                activity_grid = (activity_grid > 0.5).astype(float)
            samples[i, :mec_size] = activity_grid.flatten()

        # SENSORY input
        if lec_size > 0:

            if IS_CUE:
                #p = samples[i, cue_positions[lap_cues[lap_idx]]] / \
                p = samples[i, cue_positions[lap_idx]] / \
                    samples[i, :mec_size].max()
                alpha_samples[i] = p

                if np.random.binomial(1, p):
                    activity_lec = fixed_cue[lap_cues[lap_idx].astype(int)]
                else:
                    #activity_lec = np.zeros((lec_size))
                    #lec_idx = np.random.choice(range(lec_size),
                    #                           replace=False, size=K)
                    #activity_lec[lec_idx] = 1

                    activity_lec = np.zeros((lec_size))
                    lec_idx = np.random.choice(range(lec_size),
                                               replace=False, size=K)
                    activity_lec[lec_idx] = 1
            else:
                if IS_CUE:
                    lec_idx = np.random.choice(range(lec_size),
                                               replace=False, size=K)
                    activity_lec[lec_idx] = 1

                else:
                    activity_lec = np.zeros((lec_size))
                    lec_idx = np.random.choice(range(lec_size),
                                           replace=False, size=K)
                    activity_lec[lec_idx] = 1


            samples[i, mec_size:] = sen_list[i] if sen_list is not None else activity_lec

    samples = samples.astype(np.float32) # [ 00000 |      ]

    return samples, lap_cues, alpha_samples


def make_equal_tuning(n: int, nj: int):

    i1 = np.arange(n).tolist()
    u1 = np.ones(n) * nj
    w12 = []
    all_indices = np.arange(n)

    for i in range(n):
        if len(i1) >= nj:
            j2 = np.random.choice(i1, replace=False, size=nj)
        else:
            j2 = np.array(i1, dtype=int)
            n_missing = nj - len(j2)
            if n_missing > 0:
                fill = np.random.choice(
                    all_indices,
                    replace=n_missing > len(all_indices),
                    size=n_missing,
                )
                j2 = np.concatenate((j2, fill))
        w12 += [j2.tolist()]

        # update
        to_del = []
        for _j in j2:
            if u1[_j] <= 0:
                continue
            u1[_j] -= 1
            if u1[_j] == 0:
                to_del += [_j]
                for k, _i in enumerate(i1):
                    if _i == _j:
                        del i1[k]
                        break

    return w12


def test_equal_tuning(n: int=5, nj: int=2):

    c = np.arange(n)
    w = make_equal_tuning(n, nj)

    # results
    print(f"{n=} {nj=}\n{w=}")
    z = np.zeros(n)
    for wi in w: z[wi] += 1
    print(z)

    # plot
    fig, ax = plt.subplots()
    for i in range(n):
        plt.scatter(i, 1, s=200, color="black")
        plt.scatter(i, 0, s=200, color="black")
        for wj in w[i]:
            plt.plot([i, wj], [1, 0], color="grey")

    plt.axis('off')
    plt.show()


def get_sample_from_num_swaps(x_0, num_swaps: int, regions=None):
    """Generate a perturbed binary tensor by randomly swapping 1s and 0s.

    Args:
        x_0: Input binary tensor.
        num_swaps: Number of swaps to perform (each swap moves one 1 to 0 and one 0 to 1).
        regions: Optional list of index tensors. If provided, swaps are distributed
                 proportionally across each region based on its size.

    Returns:
        A new binary tensor with num_swaps positions flipped from 1 to 0 and
        num_swaps positions flipped from 0 to 1.
    """
    if regions == None:
      x = x_0.clone().detach()
      #get on and off index
      on_index = x_0.nonzero().squeeze(1)
      off_index = (x_0 ==0).nonzero().squeeze(1)
      #choose at random num_flips indices
      flip_off = on_index[torch.randperm(len(on_index))[:int(num_swaps)]]
      flip_on = off_index[torch.randperm(len(off_index))[:int(num_swaps)]]
      #flip on to off and off to on
      x[flip_off] = 0
      x[flip_on] = 1
      return x
    
    else:
      x = x_0.clone().detach()
      total_size = sum([len(region) for region in regions])  # Total size of all regions

      for region in regions:
          # Get the size of the region
          region_size = len(region)

          # Determine the number of swaps for this region
          num_swaps_region = round(num_swaps * region_size / total_size)

          # Get on and off indices for this region
          on_index = region[x_0[region] == 1]
          off_index = region[x_0[region] == 0]

          # Choose at random num_swaps_region indices
          flip_off = on_index[torch.randperm(len(on_index))[:num_swaps_region]]
          flip_on = off_index[torch.randperm(len(off_index))[:num_swaps_region]]

          # Flip on to off and off to on within this region
          x[flip_off] = 0
          x[flip_on] = 1

      return x


if __name__ == "__main__":



    # # generate the z patterns
    # N = 10
    # size = 50

    # heads = 3
    # variance = 0.01

    # higher_heads = heads
    # higher_variance = 0.5

    # samples = stimulus_generator(N, size, heads, variance,
    #                              higher_heads=higher_heads,
    #                              higher_variance=higher_variance,
    #                              plot=True)


    """ test activation function """

    # z = torch.randn(6)
    # print(f"z: {z}")
    # z_soft = torch.nn.functional.softmax(z, dim=-1)
    # print(f"z_soft: {z_soft}")

    # plt.subplot(311)
    # plt.imshow(z.numpy().reshape(1, -1),
    #            aspect="auto", vmin=-2, vmax=2)
    # plt.title("input")
    # plt.axhline(0, color="black", alpha=0.2)
    # plt.plot(z.numpy(), label="z", alpha=0.4)

    # softsigmoid = SoftSigmoid(gamma=2.,
    #                           beta=1.,
    #                           alpha=0.5)
    # z_sigmoid = softsigmoid(z)

    # print(f"[1] z_sigmoid: {z_sigmoid}")
    # plt.subplot(312)
    # plt.imshow(z_sigmoid.numpy().reshape(1, -1),
    #            aspect="auto", vmin=-2, vmax=2)
    # plt.plot(z_sigmoid.numpy(), label="1")

    # softsigmoid = SoftSigmoid(gamma=2.,
    #                           beta=100.,
    #                           alpha=0.2)
    # z_sigmoid = softsigmoid(z)

    # print(f"[2] z_sigmoid: {z_sigmoid}")
    # plt.subplot(313)
    # plt.imshow(z_sigmoid.numpy().reshape(1, -1),
    #            aspect="auto", vmin=-2, vmax=2)
    # plt.plot(z_sigmoid.numpy(), label="2")

    # plt.ylim(-2, 2)
    # plt.legend()
    # plt.grid()
    # plt.show()

    # --- test spars stimulus generator ---
    # N = 2
    # data = sparse_stimulus_generator(N, K=5, size=50, plot=True)

    test_equal_tuning(n=10, nj=3)




