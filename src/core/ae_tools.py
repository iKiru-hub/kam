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
import core.models as models
from core.utils import tqdm_enumerate
from core.logger import logger

AE_PATH = os.path.abspath(__file__).split("src")[0] + "src/data"


def _resolve_device(device=None) -> torch.device:
    """Return an explicitly requested device or the best available accelerator."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _model_device(model: nn.Module) -> torch.device:
    """Return the device on which a model's parameters live."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")



""" training """


def testing(data: np.ndarray, autoencoder: models.Autoencoder,
            criterion: Callable=MSELoss(),
            column: bool=False,
            use_tensor: bool=False,
            progressive_test: bool=False,
            device=None):

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

    device = torch.device(device) if device is not None else _model_device(autoencoder)
    autoencoder.to(device)

    # Set the model to evaluation mode
    autoencoder.eval()
    loss = 0.
    acc_matrix = torch.zeros(len(dataloader), len(dataloader))

    record = []

    with torch.no_grad():

        for i, batch in enumerate(dataloader):
            x = batch[0] if not column else batch[0].reshape(-1, 1)
            x = x.to(device, non_blocking=device.type == "cuda")

            # Forward pass
            outputs = autoencoder(x)  # MTL training BTSP
            loss += criterion(outputs, x)

    autoencoder.train()

    return loss / len(dataloader), autoencoder

def train_autoencoder(training_data: np.ndarray,
                      test_data: np.ndarray,
                      autoencoder: models.Autoencoder,
                      epochs: int=20, batch_size: int=64,
                      learning_rate: float=1e-3,
                      criterion: Callable=MSELoss(),
                      disable: bool=True,
                      device=None) -> tuple:

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
    disable: bool
        disable tqdm bar
    device: str or torch.device, optional
        Training device. If omitted, CUDA is preferred, followed by Apple MPS
        and then CPU.
    """

    device = _resolve_device(device)
    autoencoder.to(device)
    # logger(f"Training autoencoder on {device}")

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
    # criterion = MSELoss()
    optimizer = Adam(autoencoder.parameters(), lr=learning_rate)

    # Set the model to training mode
    autoencoder.train()

    # Training loop
    epoch = 0
    epoch_log = 100
    results = {"loss": [], "test": []}
    for epoch in (pbar := tqdm(range(epochs), desc = f"{epoch}", disable=disable)):
    # for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            zs = batch[0].to(device, non_blocking=device.type == "cuda")

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
                               criterion=criterion,
                               device=device)
        # test_loss = test_loss.item()

        results["loss"] += [total_loss / len(dataloader)]
        results["test"] += [test_loss.item()]

        if (epoch+1) % epoch_log == 0:
            pbar.set_description(f"Epoch [{epoch+1}], " + \
                f"Loss: {total_loss / len(dataloader):.4f}, " + \
                                 f"Test: {test_loss:.4f}")

    return results, autoencoder


def reconstruct_data(data: np.ndarray, model: models.Autoencoder | models.MTL,
                     criterion: Callable=MSELoss, num: int=5, column: bool=False,
                     show: bool=True, plot: bool=True):

    """
    Reconstruct data using the autoencoder model

    Parameters
    ----------
    data: np.ndarray
        data
    model: object
        autoencoder or mtl
    num: int
        the number of samples to reconstruct
    model: nn.Module
        the autoencoder model

    Returns
    -------
    np.ndarray
        reconstructed data
    """

    device = _model_device(model)

    # Keep the dataset on CPU and transfer individual batches to the model.
    if not isinstance(data, torch.Tensor):
        data_tensor = torch.tensor(data[:num],
                                   dtype=torch.float32)
    else:
        assert not isinstance(data, np.ndarray), "not torch tensor"
        data_tensor = data[:num].clone().detach().cpu()

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
            zs = zs.to(device, non_blocking=device.type == "cuda")

            # Forward pass
            outputs, latent = model(zs, ca1=True)
            reconstructed_data.append(outputs.detach().cpu().numpy().flatten())
            latent_data.append(latent.detach().cpu().numpy().flatten())

            # evaluate the output
            loss += criterion(outputs, zs)

    # Convert list to numpy array
    reconstructed_data = np.array(reconstructed_data)

    # difference between original and reconstructed data
    original_data = data_tensor.numpy()
    diff_data = original_data - reconstructed_data

    loss = loss / len(dataloader)

    # plot
    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        ax1.imshow(original_data, aspect="auto", vmin=0, vmax=1, cmap="gray_r")
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


""" save & load """

def save_autoencoder(autoencoder: models.Autoencoder, session: dict, name: str):

    """Save an autoencoder checkpoint and its session metadata in ``AE_PATH``."""

    if not isinstance(autoencoder, models.Autoencoder):
        raise TypeError("autoencoder must be an instance of models.Autoencoder")
    if not isinstance(session, dict):
        raise TypeError("session must be a dictionary")

    # Validate the metadata before writing either checkpoint file.
    serialized_session = json.dumps(session, indent=2, sort_keys=True)
    session_path = _autoencoder_session_path(name)
    session_path.mkdir(parents=True, exist_ok=True)

    model_config = {
        "dim_ei": autoencoder._dim_ei,
        "dim_ca1": autoencoder._dim_ca1,
        "K_ca1": autoencoder._K_ca1,
        "K_eo": autoencoder._K_eo,
        "beta_ei": autoencoder._beta_ei,
        "beta_eo": autoencoder._beta_eo,
        "use_bias": autoencoder._use_bias,
    }
    checkpoint = {
        "model_config": model_config,
        "model_state_dict": autoencoder.state_dict(),
    }

    torch.save(checkpoint, session_path / "autoencoder.pt")
    with (session_path / "session.json").open("w", encoding="utf-8") as file:
        file.write(serialized_session)
        file.write("\n")

    logger(f"Autoencoder saved in {session_path}")
    return session_path


def load_autoencoder(name: str):

    """Load a named autoencoder on CPU and return ``(model, session)``."""

    session_path = _autoencoder_session_path(name)
    checkpoint_path = session_path / "autoencoder.pt"
    metadata_path = session_path / "session.json"

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Autoencoder checkpoint not found: {checkpoint_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Session metadata not found: {metadata_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    try:
        model_config = checkpoint["model_config"]
        state_dict = checkpoint["model_state_dict"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Invalid autoencoder checkpoint: {checkpoint_path}") from error

    autoencoder = models.Autoencoder(**model_config)
    autoencoder.load_state_dict(state_dict)

    with metadata_path.open("r", encoding="utf-8") as file:
        session = json.load(file)
    if not isinstance(session, dict):
        raise ValueError(f"Session metadata must contain a dictionary: {metadata_path}")

    # logger(f"Autoencoder loaded from {session_path}")
    return autoencoder, session


def _autoencoder_session_path(name: str) -> Path:
    """Resolve a session name without allowing it to escape ``AE_PATH``."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")

    name_path = Path(name)
    if name_path.is_absolute() or len(name_path.parts) != 1 or name_path.name in {".", ".."}:
        raise ValueError("name must be a single directory name")

    return Path(AE_PATH) / name_path


def save_search_results(results: np.ndarray,
                        variables: dict,
                        settings_sim: dict,
                        settings_data: dict,
                        settings_ae: dict,
                        name: str) -> Path:
    """Save a parameter-search result grid and its settings as JSON.

    The order of ``variables`` defines the axes of ``results``. For example,
    variables ordered as beta, gain_out, and encoding_dim require a result
    array with shape ``(len(beta), len(gain_out), len(encoding_dim))``.
    """

    if not isinstance(variables, dict) or not variables:
        raise ValueError("variables must be a non-empty dictionary")

    results_array = np.asarray(results)
    axes = list(variables)
    expected_shape = tuple(len(variables[axis]) for axis in axes)
    if results_array.shape != expected_shape:
        raise ValueError(
            f"results has shape {results_array.shape}, expected {expected_shape} "
            f"for axes {axes}"
        )

    name_path = Path(name)
    if name_path.suffix == ".json":
        name_path = name_path.with_suffix("")
    if (not name_path.name or name_path.is_absolute()
            or len(name_path.parts) != 1 or name_path.name in {".", ".."}):
        raise ValueError("name must be a single file name")

    search = {
        "settings_sim": settings_sim,
        "settings_data": settings_data,
        "settings_ae": settings_ae,
        "variables": variables,
        "result_axes": axes,
        "result_shape": list(results_array.shape),
        "results": results_array.tolist(),
    }

    # Serialization is performed before creating the output file so invalid
    # metadata cannot leave a partially written JSON document behind.
    serialized_search = json.dumps(
        search,
        indent=2,
        sort_keys=False,
        allow_nan=False,
    )
    search_path = Path(AE_PATH) / "searches" / f"{name_path.name}.json"
    search_path.parent.mkdir(parents=True, exist_ok=True)
    with search_path.open("w", encoding="utf-8") as file:
        file.write(serialized_search)
        file.write("\n")

    logger(f"Search results saved in {search_path}")
    return search_path
