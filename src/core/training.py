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
from core.logger import logger


def tqdm_enumerate(iterable, **tqdm_kwargs):
    return enumerate(tqdm(iterable, **tqdm_kwargs))



""" training """


def testing(data: np.ndarray, autoencoder: object,
            criterion: object=MSELoss(),
            column: bool=False,
            use_tensor: bool=False,
            progressive_test: bool=False):

    """
    Test the model

    Parameters
    ----------
    data: np.ndarray
        z data
    model: nn.Module
        the model
    """

    if not isinstance(data, DataLoader):
        # Convert numpy array to torch tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)

        # Create a dataset and data loader
        dataset = TensorDataset(data_tensor)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    else:
        dataloader = data

    if use_tensor:
        try:
            data_tensor
        except NameError:
            raise ValueError("data_tensor is not defined")
        dataloader = data_tensor.unsqueeze(1)

    # Set the model to evaluation mode
    autoencoder.eval()
    loss = 0.
    acc_matrix = torch.zeros(len(dataloader), len(dataloader))

    record = []

    with torch.no_grad():

        for i, batch in enumerate(dataloader):
            x = batch[0] if not column else batch[0].reshape(-1, 1)

            # Forward pass
            outputs = autoencoder(x)  # MTL training BTSP
            loss += criterion(outputs, x)

    autoencoder.train()

    return loss / len(dataloader), autoencoder

def train_autoencoder(training_data: np.ndarray,
                      test_data: np.ndarray,
                      autoencoder: models.Autoencoder,
                      epochs: int=20, batch_size: int=64,
                      learning_rate: float=1e-3):

    """
    Train the autoencoder model

    Parameters
    ----------
    training_data: np.ndarray
        z training data
    test_data: np.ndarray
        z test data
    autoencoder: nn.Module
        the autoencoder model
    epochs: int
        the number of epochs
    batch_size: int
        the batch size
    learning_rate: float
        the learning rate
    """

    # Convert numpy array to torch tensor
    data_tensor = torch.tensor(training_data, dtype=torch.float32)

    # Create a dataset and data loader
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # test data
    test_data_tensor = torch.tensor(test_data, dtype=torch.float32)
    test_dataset = TensorDataset(test_data_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Loss function and optimizer
    criterion = MSELoss()
    optimizer = Adam(autoencoder.parameters(), lr=learning_rate)

    # Set the model to training mode
    autoencoder.train()

    # Training loop
    epoch = 0
    epoch_log = 100
    results = {"loss": [], "test": []}
    for epoch in (pbar := tqdm(range(epochs), desc = f"{epoch}")):
        total_loss = 0
        for batch in dataloader:
            zs = batch[0]

            # Forward pass
            outputs = autoencoder(zs)
            loss = criterion(outputs, zs)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # test loss
        test_loss, _ = testing(data=test_dataloader,
                               autoencoder=autoencoder,
                               criterion=criterion)

        results["loss"] += [total_loss / len(dataloader)]
        results["test"] += [test_loss]

        if (epoch+1) % epoch_log == 0:
            pbar.set_description(f"Epoch [{epoch+1}], " + \
                f"Loss: {total_loss / len(dataloader):.4f}, " + \
                                 f"Test: {test_loss:.4f}")

    return results, autoencoder


def reconstruct_data(data: np.ndarray, model: object, num: int=5,
                     column: bool=False, show: bool=True, plot: bool=True):

    """
    Reconstruct data using the autoencoder model

    Parameters
    ----------
    data: np.ndarray
        z data
    num: int
        the number of samples to reconstruct
    model: nn.Module
        the autoencoder model

    Returns
    -------
    np.ndarray
        reconstructed data
    """

    # Convert numpy array to torch tensor
    if not isinstance(data, torch.Tensor):
        data_tensor = torch.tensor(data[:num],
                                   dtype=torch.float32)
    else:
        data_tensor = data[:num].clone().detach()

    # Create a dataset and data loader
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Set the model to evaluation mode
    model.eval()
    criterion = MSELoss()

    # Reconstruct data
    reconstructed_data = []
    latent_data = []
    loss = 0.
    with torch.no_grad():

        for batch in tqdm(dataloader):

            zs = batch[0] if not column else batch[0].reshape(-1, 1)

            # Forward pass
            outputs, latent = model(zs, ca1=True)
            reconstructed_data.append(outputs.numpy().flatten())
            latent_data.append(latent.numpy().flatten())

            # evaluate the output
            loss += criterion(outputs, zs)

    # Convert list to numpy array
    reconstructed_data = np.array(reconstructed_data)

    # difference between original and reconstructed data
    diff_data = (data[:num] - reconstructed_data)

    loss /= len(dataloader)

    # plot
    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        ax1.imshow(data_tensor, aspect="auto", vmin=0, vmax=1, cmap="gray_r")
        ax1.set_title("Original data")
        ax1.set_axis_off()

        ax2.imshow(reconstructed_data, aspect="auto", vmin=0, vmax=1, cmap="gray_r")
        ax2.set_title("Reconstructed data")
        ax2.set_axis_off()

        ax3.imshow(diff_data, aspect="auto", cmap="seismic", vmin=-1, vmax=1)
        ax3.set_title(f"Difference [loss={loss:.4f}]")
        ax3.set_axis_off()

        if show:
            plt.show()

    return reconstructed_data, latent_data



def testing_mod(data: np.ndarray, model: object,
                   alpha_samples: np.ndarray,
                   alpha_baseline: float=0.1,
                   criterion: object=MSELoss(),
                   column: bool=False,
                   use_tensor: bool=False,
                   progressive_test: bool=False):

    """
    Test the model

    Parameters
    ----------
    data: np.ndarray
        z data
    model: nn.Module
        the model
    """

    if not isinstance(data, DataLoader):
        # Convert numpy array to torch tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)

        # Create a dataset and data loader
        dataset = TensorDataset(data_tensor)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    else:
        dataloader = data

    if use_tensor:
        try:
            data_tensor
        except NameError:
            raise ValueError("data_tensor is not defined")
        dataloader = data_tensor.unsqueeze(1)

    # Set the model to evaluation mode
    model.eval()
    loss = 0.
    acc_matrix = torch.zeros(len(dataloader), len(dataloader))

    alpha = model._alpha
    logger("testing...")

    with torch.no_grad():

        for i, batch in tqdm_enumerate(dataloader):
            x = batch[0] if not column else batch[0].reshape(-1, 1)

            new_alpha = np.maximum(alpha_baseline, alpha * alpha_samples[i])


            # Forward pass
            model.set_alpha(alpha=new_alpha)
            outputs = model(x)  # MTL training BTSP
            loss += criterion(outputs, x)

    model.train()

    return loss / len(dataloader), model


def progressive_testing(data: np.ndarray, model: object):

    """
    Test the model

    Parameters
    ----------
    data: np.ndarray
        z data
    model: nn.Module
        the model
    """

    datasets = []
    for k in range(len(data)):
        data = torch.tensor(data[:k+1], dtype=torch.float32, requires_grad=False)
        dataloader = DataLoader(TensorDataset(data),
                                batch_size=1,
                                shuffle=False)
        datasets += [dataloader]

    # Set the model to evaluation mode
    model.eval()
    acc_matrix = torch.zeros(len(data),
                             len(data))

    with torch.no_grad():

        # loop over all patterns
        for i, x in enumerate(datasets[-1]):

            logger.debug(f"dataset {i}, {x}")
            x = x[0].reshape(-1, 1)

            # Forward pass
            outputs = model(x)

            # test all previous patterns
            model.pause_lr()

            # testing over all previous patterns 0.. i
            for j, x in enumerate(datasets[i]):
                x = x[0].reshape(-1, 1)

                # forward
                y = model(x)

                # record : cosine similarity
                value = (y.T @ x) / \
                    (torch.norm(x) * torch.norm(y))
                acc_matrix[i, j] = (value.item() - 0.2) / 0.8

                # logger(f"pattern {j}")

            model.resume_lr()

    model.train()

    return acc_matrix, model


def reconstruction_loss(y: np.ndarray, y_pred: np.ndarray) -> float:

    """
    Calculate the reconstruction loss

    Parameters
    ----------
    y: np.ndarray
        the original data
    y_pred: np.ndarray
        the predicted data

    Returns
    -------
    float
        the reconstruction loss
    """

    return np.mean((y - y_pred)**2).item()


def train_for_accuracy(alpha: float,
                       num_rep: int,
                       num_samples: int,
                       complete_dataset: object=None,
                       **kwargs) -> np.ndarray:

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
    complete_dataset : object
        data. Default None
    **kwargs
        idx : int
            index of the autoencoder to load.
            Default 0.
        use_bias : bool
            use bias. Default True.
        shuffled : bool
            shuffled the IS. Default False.
        verbose : bool
            verbose. Default False.

    Returns
    -------
    np.ndarray
        outputs
    """

    verbose = kwargs.get("verbose", False)

    # --- load autoencoder
    logger.debug(f"{kwargs.get('idx', 0)=}")
    info, autoencoder = models.load_session(
        idx=kwargs.get("idx", 0))


    # information
    dim_ei = info["dim_ei"]
    dim_ca3 = info["dim_ca3"]
    dim_ca1 = info["dim_ca1"]
    dim_eo = info["dim_eo"]
    K_lat = info["K_lat"]
    beta = info["beta"]
    K = info["K"]

    # number training samples used for the AE
    # num_samples = info["num_samples"]

    # get parameters
    W_ei_ca1, W_ca1_eo, B_ei_ca1, B_ca1_eo = autoencoder.get_weights(
                                bias=kwargs.get("use_bias", True))

    if verbose:
        logger(f"{autoencoder=}")
        logger("<<< Loaded session >>>")

    # --- make model

    # get weights from the autoencoder
    if kwargs.get("use_bias", True):
        W_ei_ca1, W_ca1_eo, B_ei_ca1, B_ca1_eo = autoencoder.get_weights(
                                                        bias=True)
    else:
        W_ei_ca1, W_ca1_eo = autoencoder.get_weights(bias=False)
        B_ei_ca1 = None
        B_ca1_eo = None

    # make model
    model = models.MTL(W_ei_ca1=W_ei_ca1,
                W_ca1_eo=W_ca1_eo,
                B_ei_ca1=B_ei_ca1,
                B_ca1_eo=B_ca1_eo,
                dim_ca3=dim_ca3,
                K_lat=K_lat,
                K_out=K,
                alpha=alpha,
                beta=beta,
                random_IS=kwargs.get("shuffled", False))

    if verbose:
        logger(f"%MTL: {model}")

    #
    outputs = np.zeros((num_rep, num_samples, num_samples))

    if complete_dataset is None:
        complete_dataset = []
        is_dataset = False
    else:
        is_dataset = True

    for l in tqdm(range(num_rep)):

        # --- make new data
        if is_dataset:
            datasets = complete_dataset[l]
        else:
            stimuli = sparse_stimulus_generator(N=num_samples,
                                                K=K,
                                                size=dim_ei,
                                                plot=False)
            datasets = []
            for k in range(num_samples):
                data = torch.tensor(stimuli[:k+1], dtype=torch.float32)
                dataloader = DataLoader(TensorDataset(data),
                                        batch_size=1,
                                        shuffle=False)
                datasets += [dataloader]
            complete_dataset += [datasets]

        # --- run new repetition
        for i in tqdm(range(num_samples), disable=True):

            # reset the model
            model.reset()

            # train a dataset with pattern index 0.. i
            model.eval()
            with torch.no_grad():

                # one pattern at a time
                for batch in datasets[i]:
                    # forward
                    _ = model(batch[-1].reshape(-1, 1))

            # --- test a dataset with pattern index 0.. i
            model.pause_lr()
            model.eval()
            with torch.no_grad():
                # one pattern at a time
                for j, batch in enumerate(datasets[i]):
                    x = batch[-1].reshape(-1, 1)

                    # forward
                    y = model(x)

                    # record : cosine similarity
                    value = (y.T @ x) / \
                        (torch.norm(x) * torch.norm(y))

                    outputs[l, i, j] = (value.item() - 0.2) / 0.8

    return outputs, model, complete_dataset


def train_for_accuracy_lec(num_rep: int,
                       num_samples: int,
                       complete_dataset: object=None,
                       alpha: float=None,
                       **kwargs) -> np.ndarray:

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
    complete_dataset : object
        data. Default None
    **kwargs
        idx : int
            index of the autoencoder to load.
            Default 0.
        use_bias : bool
            use bias. Default True.
        shuffled : bool
            shuffled the IS. Default False.
        verbose : bool
            verbose. Default False.
        binarize : bool
            binarize the activity. Default False.

    Returns
    -------
    np.ndarray
        outputs
    """

    verbose = kwargs.get("verbose", False)

    # --- load autoencoder
    info, autoencoder = models.load_session(
        idx=kwargs.get("idx", 0), verbose=verbose)

    # information
    if "network_params" not in info:
        raise ValueError("network_params not found in the info file")

    network_params = info["network_params"]

    # get parameters
    W_ei_ca1, W_ca1_eo, B_ei_ca1, B_ca1_eo = autoencoder.get_weights(
                                bias=kwargs.get("use_bias", True))

    if verbose:
        logger(f"{autoencoder=}")
        logger("<<< Loaded session >>>")

    # --- make model

    # get weights from the autoencoder
    if kwargs.get("use_bias", True):
        W_ei_ca1, W_ca1_eo, B_ei_ca1, B_ca1_eo = autoencoder.get_weights(bias=True)
    else:
        W_ei_ca1, W_ca1_eo = autoencoder.get_weights(bias=False)
        B_ei_ca1 = None
        B_ca1_eo = None

    # make model
    model = models.MTL(W_ei_ca1=W_ei_ca1,
            W_ca1_eo=W_ca1_eo,
            B_ei_ca1=B_ei_ca1,
            B_ca1_eo=B_ca1_eo,
            dim_ca3=network_params["dim_ca3"],
            K_lat=network_params["K_ca1"],
            K_out=network_params["K_eo"],
            K_ca3=network_params["K_ca3"],
            beta=network_params["beta_ca1"],
            alpha=network_params["alpha"] if alpha is None else alpha,
            identity_IS=False,
            random_IS=kwargs.get("shuffled", False))

    if verbose:
        logger(f"%MTL: {model}")

    #
    outputs = np.zeros((num_rep, num_samples, num_samples))

    if complete_dataset is None:
        complete_dataset = []
        is_dataset = False
    else:
        is_dataset = True

    for l in tqdm(range(num_rep)):

        # --- make new data
        if is_dataset:
            datasets = complete_dataset[l]
        else:
            stimuli, _, _ = sparse_stimulus_generator_sensory(
                                num_stimuli=num_samples,
                                K = network_params["K_lec"],
                                mec_size=network_params["dim_mec"],
                                lec_size=network_params["dim_lec"],
                                N_x=network_params["mec_N_x"],
                                N_y=network_params["mec_N_y"],
                                pf_sigma=network_params["mec_sigma"],
                                lap_length=network_params["mec_N_x"],
                                num_laps=None,
                                num_cues=None,
                                position_list=None,
                                cue_positions=None,
                                sen_list=None,
                                plot=False,
                                binarize=kwargs.get("binarize", False))
            datasets = []
            for k in range(num_samples):
                data = torch.tensor(stimuli[:k+1], dtype=torch.float32)
                dataloader = DataLoader(TensorDataset(data),
                                        batch_size=1,
                                        shuffle=False)
                datasets += [dataloader]
            complete_dataset += [datasets]

        # --- run new repetition
        for i in tqdm(range(num_samples), disable=True):

            # reset the model
            model.reset()

            # train a dataset with pattern index 0.. i
            model.eval()
            with torch.no_grad():

                # one pattern at a time
                for batch in datasets[i]:
                    # forward
                    _ = model(batch[-1].reshape(-1, 1))

            # --- test a dataset with pattern index 0.. i
            model.pause_lr()
            model.eval()
            with torch.no_grad():
                # one pattern at a time
                for j, batch in enumerate(datasets[i]):
                    x = batch[-1].reshape(-1, 1)

                    # forward
                    y = model(x)

                    # record : cosine similarity
                    value = (y.T @ x) / \
                        (torch.norm(x) * torch.norm(y))

                    outputs[l, i, j] = (value.item() - 0.2) / 0.8

    return outputs, model, complete_dataset


def train_for_accuracy_v2(alpha: float,
                       num_rep: int,
                       num_samples: int,
                       complete_dataset: object=None,
                       **kwargs) -> np.ndarray:

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
    complete_dataset : object
        data. Default None
    **kwargs
        idx : int
            index of the autoencoder to load.
            Default 0.
        use_bias : bool
            use bias. Default True.
        shuffled : bool
            shuffled the IS. Default False.
        verbose : bool
            verbose. Default False.

    Returns
    -------
    np.ndarray
        outputs
    """

    verbose = kwargs.get("verbose", False)

    # --- load autoencoder
    logger.debug(f"{kwargs.get('idx', 0)=}")
    info, autoencoder = models.load_session(
        idx=kwargs.get("idx", 0))


    # information
    dim_ei = info["dim_ei"]
    dim_ca3 = info["dim_ca3"]
    dim_ca1 = info["dim_ca1"]
    dim_eo = info["dim_eo"]
    K_lat = info["K_lat"]
    beta = info["beta"]
    K = info["K"]

    # number training samples used for the AE
    # num_samples = info["num_samples"]

    # get parameters
    W_ei_ca1, W_ca1_eo, B_ei_ca1, B_ca1_eo = autoencoder.get_weights(
                                bias=kwargs.get("use_bias", True))

    if verbose:
        logger(f"{autoencoder=}")
        logger("<<< Loaded session >>>")

    # --- make model

    # get weights from the autoencoder
    if kwargs.get("use_bias", True):
        W_ei_ca1, W_ca1_eo, B_ei_ca1, B_ca1_eo = autoencoder.get_weights(
                                                        bias=True)
    else:
        W_ei_ca1, W_ca1_eo = autoencoder.get_weights(bias=False)
        B_ei_ca1 = None
        B_ca1_eo = None

    # make model
    model = models.MTL(W_ei_ca1=W_ei_ca1,
                W_ca1_eo=W_ca1_eo,
                B_ei_ca1=B_ei_ca1,
                B_ca1_eo=B_ca1_eo,
                dim_ca3=dim_ca3,
                K_lat=K_lat,
                K_out=K,
                alpha=alpha,
                beta=beta,
                random_IS=kwargs.get("shuffled", False))

    if verbose:
        logger(f"%MTL: {model}")

    #
    outputs = np.zeros((num_rep, num_samples, num_samples))
    activity = []

    if complete_dataset is None:
        complete_dataset = []
        is_dataset = False
    else:
        is_dataset = True

    for l in tqdm(range(num_rep)):

        # --- make new data
        if is_dataset:
            datasets = complete_dataset[l]
        else:
            stimuli = sparse_stimulus_generator(N=num_samples,
                                                K=K,
                                                size=dim_ei,
                                                plot=False)
            datasets = []
            for k in range(num_samples):
                data = torch.tensor(stimuli[:k+1], dtype=torch.float32)
                dataloader = DataLoader(TensorDataset(data),
                                        batch_size=1,
                                        shuffle=False)
                datasets += [dataloader]
            complete_dataset += [datasets]

        # --- run new repetition
        for i in tqdm(range(num_samples), disable=True):

            episode_logs = []

            # reset the model
            model.reset()

            # train a dataset with pattern index 0.. i
            model.eval()
            with torch.no_grad():

                # one pattern at a time
                for batch in datasets[i]:
                    # forward
                    x = batch[-1].reshape(-1, 1)
                    y = model(x)
                    loss = cross_entropy(x.reshape(-1), y.reshape(-1))

                    logs = [x.reshape(-1).detach().numpy(), y.detach().numpy(), loss.item()]

                    y = model(x, test=True)
                    loss = cross_entropy(x.reshape(-1), y.reshape(-1))
                    logs += [y.detach().numpy(), loss.item()]

                    episode_logs += [logs]

            # --- test a dataset with pattern index 0.. i
            model.pause_lr()
            model.eval()
            with torch.no_grad():
                # one pattern at a time
                for j, batch in enumerate(datasets[i]):
                    x = batch[-1].reshape(-1, 1)

                    # forward
                    y = model(x)

                    # record : cosine similarity
                    loss = cross_entropy(x.reshape(-1), y.reshape(-1))
                    norm_ = cosine_similarity_vec(x, y)

                    outputs[l, i, j] = (norm_.item() - 0.2) / 0.8

                    episode_logs[j] += [x.reshape(-1).detach().numpy(), y.detach().numpy(), loss.item()]
                    activity += [episode_logs[j]]

    return outputs, model, complete_dataset, activity


def train_for_reconstruction(alpha: float,
                             num_samples: int,
                             use_lec: bool=False,
                             **kwargs) -> np.ndarray:

    """
    trainings for a given alpha (already set in the model)

    Parameters
    ----------
    alpha : float
        learning rate
    num_samples : int
        number of samples
    use_lec: bool
        use LEC. Default True.
    **kwargs
        idx : int
            index of the autoencoder to load.
            Default 0.
        use_bias : bool
            use bias. Default True.
        binarize : bool
            binarize the activity. Default False.

    Returns
    -------
    np.ndarray
        outputs
    """

    if use_lec:
        logger("using LEC data")
        _, model, complete_dataset = train_for_accuracy_lec(alpha=alpha,
                            num_rep=1,
                            num_samples=num_samples,
                            idx=kwargs.get("idx", 0),
                            use_bias=kwargs.get("use_bias", True),
                            binarize=kwargs.get("binarize", False))

        _, model_rnd, _ = train_for_accuracy_lec(alpha=alpha,
                            num_rep=1,
                            num_samples=num_samples,
                            idx=kwargs.get("idx", 0),
                            use_bias=kwargs.get("use_bias", True),
                            shuffled=True,
                            complete_dataset=complete_dataset,
                            binarize=kwargs.get("binarize", False))
    else:
        _, model, complete_dataset = train_for_accuracy(alpha=alpha,
                            num_rep=1,
                            num_samples=num_samples,
                            idx=kwargs.get("idx", 0),
                            use_bias=kwargs.get("use_bias", True))

        _, model_rnd, _ = train_for_accuracy(alpha=alpha,
                            num_rep=1,
                            num_samples=num_samples,
                            idx=kwargs.get("idx", 0),
                            use_bias=kwargs.get("use_bias", True),
                            shuffled=True,
                            complete_dataset=complete_dataset)

    data = complete_dataset[-1][-1].dataset.tensors[0]
    num = len(data)

    # --- reconstruct data
    #
    model.pause_lr()
    out_mtl, latent_mtl = reconstruct_data(
                     data=data,
                     num=num,
                     model=model,
                     column=True,
                     plot=False)
    rec_loss = np.mean((data.numpy() - out_mtl)**2).item()

    #
    model_rnd.pause_lr()
    out_mtl_rnd, latent_mtl_rnd = reconstruct_data(
                     data=data,
                     num=num,
                     model=model_rnd,
                     column=True,
                     plot=False)
    rec_loss_rnd = np.mean(
        (data.numpy() - out_mtl_rnd)**2).item()

    # --- load autoencoder
    _, autoencoder = models.load_session(
        idx=kwargs.get("idx", 0), verbose=False)

    logger.debug(f"{len(data)=}, {len(data[0])=}")
    logger.debug(f"{data.shape=}")
    out_mtl_ae, latent_mtl_ae = reconstruct_data(
                     data=data,
                     num=num,
                     model=autoencoder,
                     column=False,
                     plot=False)
    rec_loss_rnd = np.mean(
        (data.numpy() - out_mtl_rnd)**2).item()

    record = {
        "data": data.numpy(),
        "out_mtl": out_mtl,
        "out_mtl_rnd": out_mtl_rnd,
        "out_ae": out_mtl_ae,
        "rec_loss": rec_loss,
        "rec_loss_rnd": rec_loss_rnd
    }

    return record


def train_for_reconstruction_v2(alpha: float,
                                num_samples: int,
                                use_lec: bool=False,
                                **kwargs) -> np.ndarray:

    """
    trainings for a given alpha (already set in the model)

    Parameters
    ----------
    alpha : float
        learning rate
    num_samples : int
        number of samples
    use_lec: bool
        use LEC. Default True.
    **kwargs
        idx : int
            index of the autoencoder to load.
            Default 0.
        use_bias : bool
            use bias. Default True.
        binarize : bool
            binarize the activity. Default False.

    Returns
    -------
    np.ndarray
        outputs
    """

    if use_lec:
        logger("using LEC data")
        _, model, complete_dataset = train_for_accuracy_lec(alpha=alpha,
                            num_rep=1,
                            num_samples=num_samples,
                            idx=kwargs.get("idx", 0),
                            use_bias=kwargs.get("use_bias", True),
                            binarize=kwargs.get("binarize", False))

        _, model_rnd, _ = train_for_accuracy_lec(alpha=alpha,
                            num_rep=1,
                            num_samples=num_samples,
                            idx=kwargs.get("idx", 0),
                            use_bias=kwargs.get("use_bias", True),
                            shuffled=True,
                            complete_dataset=complete_dataset,
                            binarize=kwargs.get("binarize", False))
    else:
        _, model, complete_dataset, activity_mtl = train_for_accuracy_v2(alpha=alpha,
                            num_rep=1,
                            num_samples=num_samples,
                            idx=kwargs.get("idx", 0),
                            use_bias=kwargs.get("use_bias", True))

        _, model_rnd, _, activity_rnd = train_for_accuracy_v2(alpha=alpha,
                            num_rep=1,
                            num_samples=num_samples,
                            idx=kwargs.get("idx", 0),
                            use_bias=kwargs.get("use_bias", True),
                            shuffled=True,
                            complete_dataset=complete_dataset)

    data = complete_dataset[-1][-1].dataset.tensors[0]
    num = len(data)

    # --- reconstruct data
    #
    model.pause_lr()
    out_mtl, latent_mtl = reconstruct_data(
                     data=data,
                     num=num,
                     model=model,
                     column=True,
                     plot=False)
    rec_loss = np.mean((data.numpy() - out_mtl)**2).item()

    #
    model_rnd.pause_lr()
    out_mtl_rnd, latent_mtl_rnd = reconstruct_data(
                     data=data,
                     num=num,
                     model=model_rnd,
                     column=True,
                     plot=False)
    rec_loss_rnd = np.mean(
        (data.numpy() - out_mtl_rnd)**2).item()

    # --- load autoencoder
    _, autoencoder = models.load_session(
        idx=kwargs.get("idx", 0), verbose=False)

    logger.debug(f"{len(data)=}, {len(data[0])=}")
    logger.debug(f"{data.shape=}")
    out_mtl_ae, latent_mtl_ae = reconstruct_data(
                     data=data,
                     num=num,
                     model=autoencoder,
                     column=False,
                     plot=False)
    rec_loss_rnd = np.mean(
        (data.numpy() - out_mtl_rnd)**2).item()

    record = {
        "data": data.numpy(),
        "out_mtl": out_mtl,
        "out_mtl_rnd": out_mtl_rnd,
        "out_ae": out_mtl_ae,
        "rec_loss": rec_loss,
        "rec_loss_rnd": rec_loss_rnd,
        "activity_mtl": activity_mtl,
        "activity_rnd": activity_rnd
    }

    return record


def train_for_weight_plot(alpha: float,
                       num_rep: int,
                       num_samples: int,
                       complete_dataset: object=None,
                       **kwargs) -> np.ndarray:

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
    complete_dataset : object
        data. Default None
    **kwargs
        idx : int
            index of the autoencoder to load.
            Default 0.
        use_bias : bool
            use bias. Default True.
        shuffled : bool
            shuffled the IS. Default False.
        verbose : bool
            verbose. Default False.

    Returns
    -------
    np.ndarray
        outputs
    """

    verbose = kwargs.get("verbose", False)

    # --- load autoencoder
    info, autoencoder = models.load_session(
        idx=kwargs.get("idx", 0), verbose=verbose)

    # information
    dim_ei = info["dim_ei"]
    dim_ca3 = info["dim_ca3"]
    dim_ca1 = info["dim_ca1"]
    dim_eo = info["dim_eo"]
    K_lat = info["K_lat"]
    beta = info["beta"]
    K = info["K"]

    # number training samples used for the AE
    # num_samples = info["num_samples"]

    # get parameters
    W_ei_ca1, W_ca1_eo, B_ei_ca1, B_ca1_eo = autoencoder.get_weights(
                                bias=kwargs.get("use_bias", True))

    if verbose:
        logger(f"{autoencoder=}")
        logger("<<< Loaded session >>>")

    # --- make model

    # get weights from the autoencoder
    if kwargs.get("use_bias", True):
        W_ei_ca1, W_ca1_eo, B_ei_ca1, B_ca1_eo = autoencoder.get_weights(bias=True)
    else:
        W_ei_ca1, W_ca1_eo = autoencoder.get_weights(bias=False)
        B_ei_ca1 = None
        B_ca1_eo = None

    # make model
    model = models.MTL(W_ei_ca1=W_ei_ca1,
                W_ca1_eo=W_ca1_eo,
                B_ei_ca1=B_ei_ca1,
                B_ca1_eo=B_ca1_eo,
                dim_ca3=dim_ca3,
                K_lat=K_lat,
                K_out=K,
                alpha=alpha,
                beta=beta,
                shuffled_is=kwargs.get("shuffled", False))

    if verbose:
        logger(f"%MTL: {model}")

    #
    record = np.zeros((num_rep, num_samples, dim_ca3, dim_ca3))

    if complete_dataset is None:
        complete_dataset = []
        is_dataset = False
    else:
        is_dataset = True

    for l in tqdm(range(num_rep)):

        # --- make new data
        if is_dataset:
            datasets = complete_dataset[l]
        else:
            stimuli = sparse_stimulus_generator(N=num_samples,
                                                K=K,
                                                size=dim_ei,
                                                plot=False)
            datasets = []
            for k in range(num_samples):
                data = torch.tensor(stimuli[:k+1], dtype=torch.float32)
                dataloader = DataLoader(TensorDataset(data),
                                        batch_size=1,
                                        shuffle=False)
                datasets += [dataloader]
            complete_dataset += [datasets]

        # --- run new repetition
        for i in tqdm(range(num_samples), disable=True):

            # reset the model
            model.reset()

            # train a dataset with pattern index 0.. i
            model.eval()
            with torch.no_grad():

                # one pattern at a time
                for batch in datasets[i]:

                    # forward
                    _ = model(batch[0].reshape(-1, 1))


    return outputs, model, complete_dataset


def testing_mod(data: np.ndarray, model: object,
                   alpha_samples: np.ndarray,
                   alpha_baseline: float=0.1,
                   criterion: object=MSELoss(),
                   column: bool=False,
                   use_tensor: bool=False,
                   progressive_test: bool=False):

    """
    Test the model

    Parameters
    ----------
    data: np.ndarray
        z data
    model: nn.Module
        the model
    """

    if not isinstance(data, DataLoader):
        # Convert numpy array to torch tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)

        # Create a dataset and data loader
        dataset = TensorDataset(data_tensor)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    else:
        dataloader = data

    if use_tensor:
        try:
            data_tensor
        except NameError:
            raise ValueError("data_tensor is not defined")
        dataloader = data_tensor.unsqueeze(1)

    # Set the model to evaluation mode
    model.eval()
    loss = 0.
    acc_matrix = torch.zeros(len(dataloader), len(dataloader))

    alpha = model._alpha
    logger("testing...")

    with torch.no_grad():

        for i, batch in tqdm_enumerate(dataloader):
            x = batch[0] if not column else batch[0].reshape(-1, 1)

            new_alpha = np.maximum(alpha_baseline, alpha * alpha_samples[i])

            # Forward pass
            model.set_alpha(alpha=new_alpha)
            outputs = model(x)  # MTL training BTSP
            loss += criterion(outputs, x)

    model.train()

    return loss / len(dataloader), model


""" EXPERIMENTS """

def sparse_stimulus_generator_sensory(num_stimuli: int, K : int,
                                      mec_size: int,  lec_size: int,
                                      N_x : int, N_y : int,
                                      pf_sigma: int,
                                      num_laps: int,
                                      cue_duration: int=1,
                                      lap_length: int=None,
                                      num_cues: int=None,
                                      position_list=None,
                                      cue_positions=None,
                                      verbose: bool=False,
                                      sen_list=None,
                                      plot: bool=False,
                                      sigma: float=5) -> np.ndarray:

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

    def place_field_activity(N_x, N_y, sigma, xi, yi):
        """
        Computes place field activity for each cell on an NxN grid for a given location (xi, yi).
        """

        def circular_distance(x1, x2, N):
            """
            Computes the minimum circular distance in the x-direction (wraps around the boundaries).
            """
            return np.minimum(np.abs(x1 - x2), N - np.abs(x1 - x2))

        # Create a grid of size NxN with place cells at each position
        x = np.linspace(0, N_x-1, N_x)
        y = np.linspace(0, N_y-1, N_y)
        X, Y = np.meshgrid(x, y)
        # Calculate the squared Euclidean distance between (xi, yi) and each place cell location
        dist_squared = circular_distance(X, xi, N_x) ** 2 + (Y - yi) ** 2

        # Compute Gaussian activity for each place cell
        activity = np.exp(-dist_squared / (2 * sigma ** 2))
        return activity

    def pc_cue_probability(x, c, sigma):
        p = np.exp(-((x - c) ** 2) / (2 * sigma ** 2))
        return np.random.binomial(1, p)

    samples = np.zeros((num_stimuli, mec_size + lec_size))
    alpha_samples = np.zeros(num_stimuli)

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

    """
    [0, 1, 3, 4.. 10, 0, 1, 3, ...10]
    """
    lap_i = 0
    for i in range(num_stimuli): # laps x length

        if cue_positions is not None:

            # count laps
            if i % lap_length == 0:
                lap_idx = (lap_i // cue_duration) % num_cues
                lap_cues[lap_i] = lap_idx

                if verbose:
                    print("---------------------------")
                    print(f"lap {lap_i}, cue {lap_idx}")
                lap_i += 1

        """
        if mec_size > 0:
            x_i, y_i = (position_list[i] if position_list is not None \
                                else (np.random.randint(0, N_x), np.random.randint(0, N_y)))
            activity_grid = place_field_activity(N_x, N_y, pf_sigma, x_i, y_i)
            samples[i, :mec_size] = activity_grid.flatten()

        """
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
            samples[i, :mec_size] = activity_grid.flatten()

        if lec_size > 0:

            # cue positions provided
            if cue_positions is not None:
                p = samples[i, cue_positions[lap_idx]] / \
                    samples[i, :mec_size].max()
                alpha_samples[i] = p

                if np.random.binomial(1, p):
                    activity_lec = fixed_cue[lap_cues[lap_idx].astype(int)]
                else:
                    activity_lec = np.zeros((lec_size))
                    lec_idx = np.random.choice(range(lec_size),
                                               replace=False, size=K)
                    activity_lec[lec_idx] = 1
            else:

                if cue_positions is not None:
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


def get_track_input(track_params: dict, network_params: dict, cue_duration: int=1):

  position_list = [(x, 0) for lap in range(track_params["num_laps"]) for x in range(track_params["length"])]

  if track_params["reward"] == "random":
    reward_list = None
  if track_params["cue"] == "random":
    cue_list = None

  sen_list = None

  track_input, lap_cues, alpha_samples = sparse_stimulus_generator_sensory(num_stimuli=track_params["num_laps"]*track_params["length"],
                                                  K = network_params["K_lec"],
                                                  mec_size=network_params["dim_mec"],
                                                  lec_size=network_params["dim_lec"],
                                                  N_x=network_params["mec_N_x"],
                                                  N_y=network_params["mec_N_y"],
                                                  cue_duration=cue_duration,
                                                  pf_sigma=network_params["mec_sigma"],
                                                  lap_length=track_params["length"],
                                                  num_laps=track_params["num_laps"],
                                                  num_cues=network_params["num_cues"],
                                                  position_list=position_list,
                                                  cue_positions=track_params["cue_position"],
                                                  sen_list=None,
                                                  plot=False)
  return track_input, lap_cues, alpha_samples
