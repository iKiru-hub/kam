import numpy as np
import matplotlib.pyplot as plt
import warnings
import os, sys

sys.path.append(os.path.abspath(__file__).split("src")[0] + "src")

# ==============================================================================

def plot_squashed_data(data: np.ndarray, title: str="",
                       ax: plt.Axes=None, squash: bool=False,
                       proper_title: bool=False):

    """
    This function plots the squashed data

    Parameters
    ----------
    data : np.ndarray
        squashed data
    title : str, optional
        title of the plot, by default ""
    ax : plt.Axes, optional
        axis of the plot, by default None
    """

    if squash:
        data = data.sum(axis=0).reshape(1, -1)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.imshow(data, aspect="auto", cmap="gray_r", vmin=0, vmax=1)
    if proper_title:
        ax.set_title(title, fontsize=15)
    else:
        ax.set_ylabel(title, fontsize=15)
    ax.set_yticks(range(len(data)), range(1, 1+len(data)))
    ax.set_xticks([])

    if ax is None:
        plt.show()


# ==============================================================================


def plot_input(input: np.ndarray, network_params: dict):

    """ plot """

    mec = input[:network_params["dim_mec"]]
    lec = input[network_params["dim_mec"]:]

    fig, axs = plt.subplots(1, 2, figsize=(12, 12))
    axs[0].imshow(mec.reshape(network_params["mec_N_y"], network_params["mec_N_x"]), cmap='gray_r')
    axs[0].set_title('space')
    lec_width = int(np.sqrt(network_params["dim_lec"]))
    axs[1].imshow(lec.reshape((-1, network_params["dim_lec"])), cmap='gray_r')
    axs[1].set_title('sensory')
    axs[0].set_yticks([])
    axs[1].set_yticks([])
    axs[0].set_xticks([])
    axs[1].set_xticks([])
    plt.show()


# ==============================================================================


def plot_stimuli(samples: np.ndarray):

    """
    This function plots the z patterns

    Parameters
    ----------
    samples : np.ndarray
        z patterns
    """

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 5))
    ax.imshow(samples, aspect="auto", vmin=0, vmax=1, cmap="gray_r")
    ax.set_xlabel("size")
    ax.set_ylabel("samples")
    if len(samples) < 20:
        ax.set_yticks(range(samples.shape[0]))
        ax.set_yticklabels(range(1, 1+samples.shape[0]))
    ax.set_title("z patterns")

    ax2.imshow(samples.sum(axis=0).reshape(1, -1),
               aspect="auto", cmap="gray_r")
    ax2.set_xlabel("size")
    ax2.set_title("Average z pattern")
    plt.show()


if __name__ == "__main__":
    print(f"[{__file__.split("/")[-1]} done]")
