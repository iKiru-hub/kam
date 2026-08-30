import numpy as np
import matplotlib.pyplot as plt
import warnings

from tqdm import tqdm
import torch
import os, sys
import json
sys.path.append(os.path.abspath(__file__).split("src")[0] + "src")

import core.visualization as visualization
import core.functions as functions
from core.logger import logger


DEFAULT_CUE_NUM_PATTERNS = 2
DEFAULT_CUE_LAP_LENGTH = 50
DEFAULT_CUE_POSITIONS = [10, 30]
DEFAULT_CUE_MEC_SIGMA = 5.0
DEFAULT_CUE_LEC_SIGMA = 5.0
DEFAULT_CUE_SIGMA = 10.0
DEFAULT_CUE_BETA = 10.0
DEFAULT_CUE_ALPHA = 0.0
DEFAULT_MEC_BINARIZED = True
DEFAULT_CUE_SPACING = 10


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
    dist_squared = functions.circular_distance(X, xi, N_x) ** 2 + (Y - yi) ** 2

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

    if n < 1:
        raise ValueError("n must be at least 1")
    if size < 1:
        raise ValueError("size must be at least 1")
    if fixed and n > size:
        raise ValueError("fixed cues require n <= size")

    cuesize = size // n
    p = p if p > 0. else 1 / size

    # Fixed cues only write their active block below, so this must be
    # zero-initialized.  ``np.empty`` left every inactive element undefined.
    patterns = np.zeros((n, size), dtype=np.float32)
    for k in range(n):
        if fixed:
            patterns[k, cuesize * k: cuesize * (k+1)] = 1
        else:
            patterns[k] = np.random.binomial(1, p, size)

    return patterns

def make_random_cue_sequence(num_laps: int, cue_positions: list, num_cue_patterns):
    return [
        np.random.choice(
            num_cue_patterns,
            replace=False,
            size=len(cue_positions),
        ).tolist()
        for _ in range(num_laps)
    ]

def make_spaced_cue_sequence(num_laps: int, cue_positions: list, num_cue_patterns,
                             spacing: int):
    sequence = []
    head = 0
    for lap in range(num_laps):
        if lap % spacing == 0:
            head += 1
        sequence += [[head % num_cue_patterns, (head +1) % num_cue_patterns]]
    return sequence

def make_cue_track_data(
        num_samples: int,
        size: int = 50,
        num_cue_patterns: int = DEFAULT_CUE_NUM_PATTERNS,
        lap_length: int = DEFAULT_CUE_LAP_LENGTH,
        cue_positions=DEFAULT_CUE_POSITIONS,
        cue_sigma=DEFAULT_CUE_SIGMA,
        cue_beta=DEFAULT_CUE_BETA,
        cue_alpha=DEFAULT_CUE_ALPHA,
        mec_binarized=DEFAULT_MEC_BINARIZED,
        mec_sigma: float = DEFAULT_CUE_MEC_SIGMA,
        lec_sigma: float = DEFAULT_CUE_LEC_SIGMA,
        max_num_patterns: int|None=None,
        cue_spacing: int = DEFAULT_CUE_SPACING) -> np.ndarray:
    """Generate exactly ``num_samples`` from the shared MEC+LEC cue task.


    Autoencoder and MTL experiments use this function so their cue identities,
    widths, positions, and track geometry cannot silently diverge.
    """

    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    if size < 2 or size % 2:
        raise ValueError("size must be an even integer of at least 2")
    if lap_length < 1:
        raise ValueError("lap_length must be at least 1")


    cue_positions = tuple(int(position) for position in cue_positions)
    if not cue_positions:
        raise ValueError("cue_positions must not be empty")
    if any(position < 0 or position >= lap_length for position in cue_positions):
        raise ValueError("cue positions must lie within the lap")
    if num_cue_patterns < len(cue_positions):
        raise ValueError(
            "num_cue_patterns must be at least the number of cue positions"
        )

    lec_size = size // 2
    # max_num = pattern_size if pattern_size is not None else lec_size // num_cue_patterns
    max_num_patterns = max_num_patterns if max_num_patterns is not None else num_cue_patterns
    num_laps = int(np.ceil(num_samples / lap_length))
    cue_patterns = make_cues(
        n=max_num_patterns,
        size=lec_size,
        fixed=True,
        p=0.2,
        )[:num_cue_patterns]
    # make cue sequence
    # cue_sequence = [
    #     np.random.choice(
    #         num_cue_patterns,
    #         replace=False,
    #         size=len(cue_positions),
    #     ).tolist()
    #     for _ in range(num_laps)
    # ]
    cue_sequence = make_spaced_cue_sequence(num_laps=num_laps,
                                            cue_positions=cue_positions,
                                            num_cue_patterns=num_cue_patterns,
                                            spacing=cue_spacing)
    laps = {
        "n": num_laps,
        "length": lap_length,
        "cues_positions": list(cue_positions),
        "cues_patterns": cue_patterns,
        "cues_sequence": cue_sequence,
        "cue_sigma": cue_sigma,
        "cue_beta": cue_beta,
        "cue_alpha": cue_alpha,
        "mec_binarized": mec_binarized
    }
    samples, _ = sparse_stimulus_generator_sensory(
        laps=laps,
        mec_sigma=mec_sigma,
        lec_sigma=lec_sigma,
    )
    return samples.reshape(-1, size)[:num_samples]


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
    if length < 1:
        raise ValueError("lap length must be at least 1")
    # MEC units are place cells whose field centers tile the *whole* circular
    # track.  Using np.arange(mec_size) placed every center in [0, mec_size),
    # which left an uncovered arc whenever track length exceeded mec_size.
    mec_centers = np.linspace(0., float(length), mec_size, endpoint=False)
    samples = np.zeros((laps["n"], length, mec_size + lec_size))

    cue_sigma = laps.get("cue_sigma", 10)
    cue_alpha = laps.get("cue_alpha", 0.)
    cue_beta = laps.get("cue_beta", 1)
    mec_binarized = laps.get("mec_binarized", True)

    # --- loop over each time step in each lap
    for l in range(laps["n"]):
        for x in range(length):

            # -- add spatial MEC
            mec_obs = np.around(
                functions.gaussian_circular_distance(
                    mec_centers, x, length, mec_sigma
                ), 3, )
            if mec_binarized:
                mec_obs = list(map(lambda x: np.random.binomial(1, np.clip(x, 0, 1)), mec_obs))
            samples[l, x, :mec_size] = mec_obs

            # -- add sensory LEC
            lec_obs = np.zeros(lec_size)
            for cue in laps_cues[l]:

                # probability of adding pattern k now
                cue_distance = functions.gaussian_circular_distance(cue[0], x, length, cue_sigma)
                cue_probability = functions.generalized_sigmoid(x=cue_distance,
                                                                beta=cue_beta,
                                                                alpha=cue_alpha,
                                                                top=2., offset=1.)
                if np.random.binomial(1, cue_probability):
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


def _bitkill(x, fraction: float, regions=None):
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
    z = np.copy(x)
    num_swaps = int(fraction * np.asarray(x).size)

    if regions is None:
        if num_swaps <= 0:
            return z

        # Get indices of 1s and 0s (handling multi-dimensional or 1D arrays cleanly)
        on_index = np.argwhere(x == 1).squeeze(axis=-1)

        # Randomly select indices without replacement
        k = min(int(num_swaps), len(on_index))
        if k <= 0:
            return z
        flip_off = np.random.choice(on_index, size=k, replace=False)

        # Apply flips
        z[flip_off] = 0
        return z

    else:
        total_size = sum(len(region) for region in regions)

        for region in regions:
            region_size = len(region)
            num_swaps_region = int(round(num_swaps * region_size / total_size))
            if num_swaps_region <= 0:
                continue

            # Extract indices of 1s and 0s within the region
            region_values = x[region]
            on_index = region[region_values == 1]

            # Randomly select indices without replacement
            num_swaps_region = min(num_swaps_region, len(on_index))
            if num_swaps_region <= 0:
                continue
            flip_off = np.random.choice(
                on_index, size=num_swaps_region, replace=False
            )

            # Apply flips within the region
            z[flip_off] = 0

        return z

def bitkill(x, fraction: float, regions=None):
    x = np.asarray(x)
    if x.ndim == 1:
        return _bitkill(x=x, fraction=fraction, regions=regions)
    return np.stack([
        _bitkill(x=xi, fraction=fraction, regions=regions)
        for xi in x
    ], axis=0)



def _bitflip(x, fraction: float, regions=None):
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
    z = np.copy(x)
    num_swaps = int(fraction * np.asarray(x).size)

    if regions is None:
        if num_swaps <= 0:
            return z

        # Get indices of 1s and 0s (handling multi-dimensional or 1D arrays cleanly)
        on_index = np.argwhere(x == 1).squeeze(axis=-1)
        off_index = np.argwhere(x == 0).squeeze(axis=-1)

        # Randomly select indices without replacement
        k = min(int(num_swaps), len(on_index), len(off_index))
        if k <= 0:
            return z
        flip_off = np.random.choice(on_index, size=k, replace=False)
        flip_on = np.random.choice(off_index, size=k, replace=False)

        # Apply flips
        z[flip_off] = 0
        z[flip_on] = 1
        return z

    else:
        total_size = sum(len(region) for region in regions)

        for region in regions:
            region_size = len(region)
            num_swaps_region = int(round(num_swaps * region_size / total_size))
            if num_swaps_region <= 0:
                continue

            # Extract indices of 1s and 0s within the region
            region_values = x[region]
            on_index = region[region_values == 1]
            off_index = region[region_values == 0]

            # Randomly select indices without replacement
            num_swaps_region = min(num_swaps_region, len(on_index), len(off_index))
            if num_swaps_region <= 0:
                continue
            flip_off = np.random.choice(
                on_index, size=num_swaps_region, replace=False
            )
            flip_on = np.random.choice(
                off_index, size=num_swaps_region, replace=False
            )

            # Apply flips within the region
            z[flip_off] = 0
            z[flip_on] = 1

        return z

def bitflip(x, fraction: float, regions=None):
    x = np.asarray(x)
    if x.ndim == 1:
        return _bitflip(x=x, fraction=fraction, regions=regions)
    return np.stack([
        _bitflip(x=xi, fraction=fraction, regions=regions)
        for xi in x
    ], axis=0)

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


def _numpy_stochastic_count(expected: float, maximum: int,
                            rng: np.random.Generator | None) -> int:
    base = int(np.floor(expected))
    fractional = expected - base
    draw = rng.random() if rng is not None else np.random.random()
    return min(maximum, base + int(draw < fractional))


def _torch_stochastic_count(expected: float, maximum: int, device: torch.device,
                            generator: torch.Generator | None) -> int:
    base = int(np.floor(expected))
    fractional = expected - base
    draw = float(torch.rand((), device=device, generator=generator).item())
    return min(maximum, base + int(draw < fractional))



def _validate_bitnoise_rate(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie between zero and one")
    return value

def bitnoise(
    x: np.ndarray | torch.Tensor,
    fraction: float,
    false_positive_rate: float = 0.0,
    *,
    rng: np.random.Generator | None = None,
    generator: torch.Generator | None = None,
) -> np.ndarray | torch.Tensor:
    """Corrupt binary vectors or row-wise binary matrices for denoising.

    ``noise_level`` is the expected fraction of *active* bits to delete in
    each row.  It is therefore independent of the ambient dimensionality: for
    a five-active-bit stimulus, ``noise_level=0.25`` removes one active bit on
    average, rather than trying to alter 25% of all 50 coordinates.

    ``false_positive_rate`` optionally adds inactive bits, expressed relative
    to the original number of active bits.  It defaults to zero, so the usual
    corruption is dropout/occlusion rather than an identity-changing swap.
    Inputs must be binary and have shape ``(features,)`` or
    ``(samples, features)``.  The returned array/tensor has the same type,
    shape, dtype, and device as the input.  Pass ``rng`` or ``generator`` to
    make NumPy or PyTorch calls reproducible, respectively.
    """
    dropout_rate = _validate_bitnoise_rate("noise_level", fraction)
    add_rate = _validate_bitnoise_rate("false_positive_rate", false_positive_rate)
    if isinstance(x, torch.Tensor):
        if rng is not None:
            raise TypeError("rng is only valid for NumPy inputs; use generator for tensors")
        if x.ndim not in (1, 2):
            raise ValueError("bitnoise expects a binary vector or a row-wise binary matrix")
        if not bool(torch.all((x == 0) | (x == 1)).item()):
            raise ValueError("bitnoise expects binary values containing only zero and one")
        rows = x.unsqueeze(0) if x.ndim == 1 else x
        noisy = rows.clone()
        for row in noisy:
            active = torch.nonzero(row == 1, as_tuple=False).flatten()
            delete_count = _torch_stochastic_count(
                dropout_rate * len(active), len(active), row.device, generator
            )
            if delete_count:
                deleted = active[torch.randperm(len(active), device=row.device, generator=generator)[:delete_count]]
                row[deleted] = 0
            inactive = torch.nonzero(row == 0, as_tuple=False).flatten()
            add_count = _torch_stochastic_count(
                add_rate * len(active), len(inactive), row.device, generator
            )
            if add_count:
                added = inactive[torch.randperm(len(inactive), device=row.device, generator=generator)[:add_count]]
                row[added] = 1
        return noisy.squeeze(0) if x.ndim == 1 else noisy

    if not isinstance(x, np.ndarray):
        raise TypeError("bitnoise expects a NumPy array or torch.Tensor")
    if generator is not None:
        raise TypeError("generator is only valid for torch inputs; use rng for NumPy arrays")
    if x.ndim not in (1, 2):
        raise ValueError("bitnoise expects a binary vector or a row-wise binary matrix")
    if not np.all((x == 0) | (x == 1)):
        raise ValueError("bitnoise expects binary values containing only zero and one")
    rows = x[None, :] if x.ndim == 1 else x
    noisy = rows.copy()
    for row in noisy:
        active = np.flatnonzero(row == 1)
        delete_count = _numpy_stochastic_count(dropout_rate * len(active), len(active), rng)
        if delete_count:
            chooser = rng if rng is not None else np.random
            row[chooser.choice(active, size=delete_count, replace=False)] = 0
        inactive = np.flatnonzero(row == 0)
        add_count = _numpy_stochastic_count(add_rate * len(active), len(inactive), rng)
        if add_count:
            chooser = rng if rng is not None else np.random
            row[chooser.choice(inactive, size=add_count, replace=False)] = 1
    return noisy[0] if x.ndim == 1 else noisy

def makebitfunction(kind: int):
    if kind == 0: return bitflip
    elif kind == 1: return bitkill
    elif kind == 2: return bitnoise
    else: raise NameError(f"wrong {kind=}")


if __name__ == "__main__":

    x_0 = np.array([0, 0, 1, 1, 1, 1, 0, 0])
    data = get_sample_from_num_swaps(x_0, num_swaps=4)
    print(f"{x_0=}")
    print(f"{data=}")

    # .
    print(f"[{__file__.split("/")[-1]} done]")
