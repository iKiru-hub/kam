import numpy as np
import matplotlib.pyplot as plt
import warnings

from tqdm import tqdm
import os, sys
import json
sys.path.append(os.path.abspath(__file__).split("src")[0] + "src")

import core.visualization as visualization
import core.functions as functions
from core.logger import logger

"""
=============================================================================
STIMULUS GENERATOR
=============================================================================
"""



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
        variance_vec = np.array([variance]*heads) * size

        # tile for the number of samples
        variance_vec = np.tile(variance, (N, 1))

    # generate the position of the heads as equidistant points
    else:
        mu = np.linspace(1/(heads+1), 1 - 1/(heads+1), heads, endpoint=True) * size
        variance_vec = np.array([variance]*heads) * size

        # tile for the number of samples
        mu = np.tile(mu, (N, 1))
        variance_vec = np.tile(variance_vec, (N, 1))

    # generate the z patterns
    samples = np.zeros((N, size))
    for i in range(N):
        for k in range(heads):
            for x in range(size):
                p = np.exp(-((x-mu[i, k])**2)/(2*variance_vec[i, k]))
                samples[i, x] += np.random.binomial(1, p)

    if plot:
        visualization.plot_stimuli(samples=samples)

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
        visualization.plot_stimuli(samples=samples)

    return samples



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

def gaussian_activity(gridsize: tuple, center: tuple, sigma: float):
    """
    Computes a Gaussian receptive field over a grid
    """
    assert len(gridsize) == 2, "grid len should be a 2-tuple"
    assert len(center) == 2, "center len should be a 2-tuple"

    N_x, N_y = gridsize
    xc, yc = center

    x = np.linspace(0, N_x-1, N_x)
    y = np.linspace(0, N_y-1, N_y)
    X, Y = np.meshgrid(x, y)

    z = np.empty((N_x, N_y))
    for i, (xi, yi) in enumerate(zip(X, Y)):
        for j, (xii, yii) in enumerate(zip(xi, yi)):
            z[i, j] = np.exp(-((xii-xc)**2 + (yii-yc)**2) / sigma)

    return z


def make_cues(n: int, size: int, fixed: bool=True, p: float=-1.):
    """ generate `n` cues of length `size` """

    cuesize = size // n
    p = p if p > 0. else 1 / size

    patterns = np.empty((n, size))
    for k in range(n):
        if fixed:
            patterns[k, cuesize * k: cuesize * (k+1)] = 1
        else:
            patterns[k] = np.random.binomial(1, p, size)

    return patterns


def sparse_stimulus_generator_sensory(laps: dict, mec_size: int=-1,
                                      mec_sigma: float=1.,
                                      lec_sigma: float=1.) -> tuple:

    """
    This function generates random z patterns with a certain
    degree of sparsity

    Parameters
    ----------
    laps : dict
        expected keys: n: int, length: int, cues_positions: [], cues_patterns: [],
            cues_sequence: [[]*#cue_positions]
    mec_size : int
        size of the spatial segment of the stimulus, if -1 is set to lec_size
    mec_sigma : int
        width of the gaussian circular distance function
    lec_sigma : int
        width of the gaussian circular distance function

    Returns
    -------
    samples : np.ndarray
        (n, length, mec_size, cue_pattern length)
    laps_cues : list
        [[cue_position, cue_pattern, cue_index]]
    """

    lec_size = len(laps["cues_patterns"][0])
    mec_size = mec_size if mec_size > 0 else lec_size

    # make a list of (cue_position, cue_pattern, cue_index) for each lap
    laps_cues = []
    for lseq in laps["cues_sequence"]:
        _cues = []
        for xi, si in enumerate(lseq):
            _cues += [(laps["cues_positions"][xi], laps["cues_patterns"][si], si)]
        laps_cues += [_cues]

    # setup
    length = laps["length"]
    num_cues_per_laps = len(laps["cues_positions"])
    xbase = np.arange(mec_size)
    for z in xbase:
        z = length / z if z > 0 else 0.
    samples = np.zeros((laps["n"], length, mec_size + lec_size))

    # --- loop over each time step in each lap
    for l in range(laps["n"]):
        for x in range(length):

            # -- add spatial MEC
            mec_obs = np.around(functions.gaussian_circular_distance(xbase, x, length,
                                                           mec_sigma), 3)
            samples[l, x, :mec_size] = mec_obs

            # -- add sensory LEC
            lec_obs = np.zeros(lec_size)
            for cue in laps_cues[l]:

                # probability of adding pattern k now
                p = functions.gaussian_circular_distance(cue[0], x, length, lec_sigma)
                if np.random.binomial(1, p):
                    samples[l, x, mec_size:] = cue[1]

    return samples, laps_cues


def make_equal_tuning(n: int, nj: int):

    """
    function that assigns to n target notes nj input notes attempting to
    ensure an equal number of projections and evenly distributed

    Parameters
    ----------
    n : int
    nj : int
    """

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

    """ visualize make_equal_tuning """

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
    """Generate a perturbed binary array by randomly swapping 1s and 0s.

    Parameters
    ----------
        x_0: Input binary NumPy array.
        num_swaps: Number of swaps to perform (each swap moves one
            1 to 0 and one 0 to 1).
        regions: Optional list of index arrays. If provided, swaps are distributed
                 proportionally across each region based on its size.

    Returns
    -------
        A new binary array with num_swaps positions flipped from 1 to 0 and
        num_swaps positions flipped from 0 to 1.
    """
    x = np.copy(x_0)

    if regions is None:
        # Get indices of 1s and 0s (handling multi-dimensional or 1D arrays cleanly)
        on_index = np.argwhere(x_0 == 1).squeeze(axis=-1)
        off_index = np.argwhere(x_0 == 0).squeeze(axis=-1)

        # Randomly select indices without replacement
        k = int(num_swaps)
        flip_off = np.random.choice(on_index, size=k, replace=False)
        flip_on = np.random.choice(off_index, size=k, replace=False)

        # Apply flips
        x[flip_off] = 0
        x[flip_on] = 1
        return x

    else:
        total_size = sum(len(region) for region in regions)

        for region in regions:
            region_size = len(region)
            num_swaps_region = int(round(num_swaps * region_size / total_size))

            # Extract indices of 1s and 0s within the region
            region_values = x_0[region]
            on_index = region[region_values == 1]
            off_index = region[region_values == 0]

            # Randomly select indices without replacement
            flip_off = np.random.choice(
                on_index, size=num_swaps_region, replace=False
            )
            flip_on = np.random.choice(
                off_index, size=num_swaps_region, replace=False
            )

            # Apply flips within the region
            x[flip_off] = 0
            x[flip_on] = 1

        return x

if __name__ == "__main__":

    x_0 = np.array([0, 0, 1, 1, 1, 1, 0, 0])
    data = get_sample_from_num_swaps(x_0, num_swaps=4)
    print(f"{x_0=}")
    print(f"{data=}")

    # .
    print(f"[{__file__.split("/")[-1]} done]")
