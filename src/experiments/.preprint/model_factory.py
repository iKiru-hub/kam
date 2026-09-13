"""Small helpers around the existing Autoencoder and MTL implementations."""

from __future__ import annotations

import numpy as np
import torch

from core import functions, models

from experiments.preprint.metrics import row_cosine


def train_autoencoder(training: np.ndarray, validation: np.ndarray, config: dict, init_seed: int, batch_seed: int) -> tuple[models.Autoencoder, dict[str, float]]:
    """Train the existing one-layer autoencoder deterministically on CPU."""

    ae_cfg = config["autoencoder"]
    data_cfg = config["data"]
    torch.manual_seed(init_seed)
    model = models.Autoencoder(
        dim_ei=data_cfg["dimension"], dim_ca1=ae_cfg["latent_dimension"],
        K_ca1=data_cfg["active"], K_eo=data_cfg["active"],
        beta_ei=ae_cfg["beta_latent"], beta_eo=ae_cfg["beta_output"], use_bias=False,
    ).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=ae_cfg["learning_rate"])
    data = torch.as_tensor(training, dtype=torch.float32)
    generator = torch.Generator().manual_seed(batch_seed)
    for _ in range(int(ae_cfg["epochs"])):
        order = torch.randperm(len(data), generator=generator)
        for start in range(0, len(data), int(ae_cfg["batch_size"])):
            batch = data[order[start:start + int(ae_cfg["batch_size"])]]
            loss = torch.mean((model(batch) - batch) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        output = model(torch.as_tensor(validation, dtype=torch.float32)).numpy()
    quality = {"mse": float(np.mean((output - validation) ** 2)), "cosine": float(row_cosine(output, validation).mean())}
    return model, quality


def balanced_ca3_weights(dimension: int, inputs_per_unit: int, rng: np.random.Generator, mode: str = "balanced") -> torch.Tensor:
    """Build a sparse CA3 key map; alternatives are completion ablations."""

    if mode == "identity":
        return torch.eye(dimension, dtype=torch.float32) / dimension
    if mode == "dense":
        return torch.full((dimension, dimension), 1.0 / dimension, dtype=torch.float32)
    weights = np.zeros((dimension, dimension), dtype=np.float32)
    if mode == "balanced":
        for _ in range(inputs_per_unit):
            weights[np.arange(dimension), rng.permutation(dimension)] += 1.0 / dimension
    elif mode == "shuffled":
        for row in range(dimension):
            weights[row, rng.choice(dimension, size=inputs_per_unit, replace=False)] = 1.0 / dimension
    else:
        raise ValueError(f"Unknown CA3 mode: {mode}")
    return torch.as_tensor(weights)


def ca3_code(inputs: torch.Tensor, ca3_weights: torch.Tensor, k: int, beta: float) -> torch.Tensor:
    return functions.sparsemoid(inputs @ ca3_weights.T, K=k, beta=beta)


def update_weights(weights: torch.Tensor, keys: torch.Tensor, instructions: torch.Tensor, alpha: float) -> torch.Tensor:
    """The simple bounded write rule used for the compatibility experiment."""

    for key, instruction in zip(keys, instructions):
        weights = (1.0 - alpha * instruction[:, None]) * weights + alpha * instruction[:, None] @ key[None, :]
    return weights


def recall(keys: torch.Tensor, weights: torch.Tensor, decoder: torch.Tensor, memory: dict) -> tuple[np.ndarray, np.ndarray]:
    ca1 = functions.sparsemoid(keys @ weights.T, K=memory["k_ca1"], beta=memory["beta_ca1"])
    output = torch.sigmoid(memory["beta_output"] * (ca1 @ decoder.T))
    return output.detach().numpy().astype(np.float32), ca1.detach().numpy().astype(np.float32)


def build_mtl(autoencoder: models.Autoencoder, memory: dict, wiring_seed: int | None = None) -> models.MTL:
    weights = autoencoder.get_weights(bias=False)
    state = np.random.get_state()
    if wiring_seed is not None:
        np.random.seed(wiring_seed % (2**32 - 1))
    try:
        return models.MTL(
            W_ei_ca1=weights[0].detach().clone(), W_ca1_eo=weights[1].detach().clone(),
            K_ca1=autoencoder._K_ca1, K_eo=autoencoder._K_eo, K_ca3=int(memory["k_ca3"]),
            dim_ca3=int(memory["ca3_dimension"]), beta_is=autoencoder._beta_ei,
            beta_ca3=float(memory["beta_ca3"]), beta_ca1=float(memory["beta_ca1"]),
            beta_eo=autoencoder._beta_eo, alpha=float(memory["alpha"]),
            nb_ei_ca3=int(memory["ca3_inputs_per_unit"]), plasticity=memory.get("plasticity_rule", "base"),
        )
    finally:
        np.random.set_state(state)


def run_mtl(model: models.MTL, inputs: np.ndarray, learn: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run samples through MTL and return output, CA1, and CA3 activity."""

    if learn:
        model.resume_lr()
    else:
        model.pause_lr()
    outputs, ca1, ca3 = [], [], []
    with torch.no_grad():
        for sample in inputs:
            model(torch.as_tensor(sample, dtype=torch.float32).reshape(-1, 1))
            outputs.append(model._eo.reshape(-1).numpy().copy())
            ca1.append(model._ca1.reshape(-1).numpy().copy())
            ca3.append(model._ca3.reshape(-1).numpy().copy())
    return np.stack(outputs), np.stack(ca1), np.stack(ca3)
