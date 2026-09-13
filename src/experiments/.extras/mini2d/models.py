"""Small 2D→3D→2D models for visualizing associative-memory dynamics.

Inputs and reconstructions live in the unit square.  The autoencoder latent
layer and the MTL CA3/CA1 populations have three competing units, allowing a
two-dimensional sparse representation to be inspected directly in 3D.
"""

from __future__ import annotations

import torch
from torch import nn


INPUT_DIM = 2
LATENT_DIM = 3
CA3_DIM = 3
OUTPUT_DIM = 2
SPARSITY = 1
PLASTICITY_RULES = ("base", "nois", "isout", "err1", "err2")


def _validate_last_dimension(x: torch.Tensor, expected: int,
                             name: str) -> None:
    if x.ndim == 0 or x.shape[-1] != expected:
        raise ValueError(
            f"{name} must have final dimension {expected}, got {tuple(x.shape)}"
        )


def _sparsemoid(x: torch.Tensor, beta: float,
                k: int = SPARSITY) -> torch.Tensor:
    """Apply the same competitive activation used by the full model."""
    if x.ndim == 0:
        raise ValueError("sparsemoid input must have at least one dimension")
    if not 1 <= k < x.shape[-1]:
        raise ValueError(
            f"k must be between 1 and {x.shape[-1] - 1}, got {k}"
        )
    sorted_values = torch.sort(x, descending=True, dim=-1).values
    threshold = sorted_values[..., k - 1:k + 1].mean(
        dim=-1,
        keepdim=True,
    )
    return torch.sigmoid(float(beta) * (x - threshold))


def _matrix(value: torch.Tensor | None, shape: tuple[int, int],
            default: torch.Tensor, name: str) -> torch.Tensor:
    tensor = default if value is None else torch.as_tensor(value)
    tensor = tensor.detach().clone().to(dtype=torch.float32)
    if tensor.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    return tensor


def _column(value: torch.Tensor | None, size: int,
            name: str) -> torch.Tensor:
    tensor = torch.zeros(size, 1) if value is None else torch.as_tensor(value)
    tensor = tensor.detach().clone().to(dtype=torch.float32)
    if tensor.shape == (size,):
        tensor = tensor.reshape(size, 1)
    if tensor.shape != (size, 1):
        raise ValueError(
            f"{name} must have shape ({size},) or ({size}, 1), "
            f"got {tuple(tensor.shape)}"
        )
    return tensor


class MinAutoencoder(nn.Module):
    """A trainable 2D autoencoder with a three-unit sparse latent layer."""

    def __init__(self, beta: float = 8.0, gain_out: float = 10.0,
                 offset_out: float = 0.1, use_bias: bool = True):
        super().__init__()
        self.input_dim = INPUT_DIM
        self.encoding_dim = LATENT_DIM
        self.beta = float(beta)
        self.gain_out = float(gain_out)
        self.offset_out = float(offset_out)
        self.use_bias = bool(use_bias)

        self.encoder = nn.Linear(INPUT_DIM, LATENT_DIM, bias=self.use_bias)
        self.decoder = nn.Linear(LATENT_DIM, OUTPUT_DIM, bias=self.use_bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map points in the unit square into the 3D sparse representation."""
        _validate_last_dimension(x, INPUT_DIM, "autoencoder input")
        return _sparsemoid(self.encoder(x), self.beta)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode three-unit activity back into the two-dimensional square."""
        _validate_last_dimension(z, LATENT_DIM, "latent input")
        decoded = self.decoder(z)
        return torch.sigmoid(self.gain_out * (decoded - self.offset_out))

    def forward(self, x: torch.Tensor, return_latent: bool = False,
                ca1: bool | None = None):
        """Return reconstruction, optionally together with latent activity.

        ``ca1`` is accepted as a compatibility alias for ``return_latent``.
        """
        if ca1 is not None:
            return_latent = bool(ca1)
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        if return_latent:
            return reconstruction, latent
        return reconstruction

    def get_weights(self, bias: bool = False):
        """Return encoder/decoder matrices and, optionally, bias columns."""
        encoder_bias = None
        decoder_bias = None
        if bias and self.use_bias:
            encoder_bias = self.encoder.bias.reshape(LATENT_DIM, 1)
            decoder_bias = self.decoder.bias.reshape(OUTPUT_DIM, 1)
        return self.encoder.weight, self.decoder.weight, encoder_bias, decoder_bias


class MinMTL(nn.Module):
    """A small 2D-input associative memory with 3D CA3 and CA1 activity.

    The learned ``W_ca3_ca1`` matrix is the only plastic connection.  All
    recorded internal states are detached snapshots intended for plotting.
    """

    def __init__(self, W_ei_ca1: torch.Tensor | None = None,
                 W_ca1_eo: torch.Tensor | None = None,
                 W_ei_ca3: torch.Tensor | None = None,
                 B_ei_ca1: torch.Tensor | None = None,
                 B_ei_ca3: torch.Tensor | None = None,
                 B_ca1: torch.Tensor | None = None,
                 B_ca1_eo: torch.Tensor | None = None,
                 beta_is: float = 8.0, beta_ca3: float = 8.0,
                 beta_ca1: float = 8.0, gain_out: float = 10.0,
                 offset_out: float = 0.1, alpha: float = 0.01,
                 plasticity: str = "base", record_history: bool = True):
        super().__init__()
        plasticity = str(plasticity).lower()
        if plasticity not in PLASTICITY_RULES:
            raise ValueError(
                f"plasticity must be one of {PLASTICITY_RULES}, got {plasticity!r}"
            )

        # A deterministic three-direction projection makes runs reproducible
        # while keeping CA3 distinct from the learned autoencoder encoder.
        ca3_projection = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]],
        )
        ca3_bias = torch.tensor([[0.0], [0.0], [1.0]])
        self.register_buffer(
            "W_ei_ca1",
            _matrix(
                W_ei_ca1,
                (LATENT_DIM, INPUT_DIM),
                torch.zeros(LATENT_DIM, INPUT_DIM),
                "W_ei_ca1",
            ),
        )
        self.register_buffer(
            "W_ca1_eo",
            _matrix(
                W_ca1_eo,
                (OUTPUT_DIM, LATENT_DIM),
                torch.zeros(OUTPUT_DIM, LATENT_DIM),
                "W_ca1_eo",
            ),
        )
        self.register_buffer(
            "W_ei_ca3",
            _matrix(
                W_ei_ca3,
                (CA3_DIM, INPUT_DIM),
                ca3_projection,
                "W_ei_ca3",
            ),
        )
        self.register_buffer(
            "B_ei_ca1", _column(B_ei_ca1, LATENT_DIM, "B_ei_ca1")
        )
        self.register_buffer(
            "B_ei_ca3", _column(B_ei_ca3, CA3_DIM, "B_ei_ca3")
        )
        if B_ei_ca3 is None:
            self.B_ei_ca3.copy_(ca3_bias)
        self.register_buffer("B_ca1", _column(B_ca1, LATENT_DIM, "B_ca1"))
        self.register_buffer(
            "B_ca1_eo", _column(B_ca1_eo, OUTPUT_DIM, "B_ca1_eo")
        )
        self.register_buffer(
            "W_ca3_ca1", torch.zeros(LATENT_DIM, CA3_DIM)
        )

        self.beta_is = float(beta_is)
        self.beta_ca3 = float(beta_ca3)
        self.beta_ca1 = float(beta_ca1)
        self.gain_out = float(gain_out)
        self.offset_out = float(offset_out)
        self.alpha = abs(float(alpha))
        self.plasticity = plasticity
        self.learning = True
        self.record_history = bool(record_history)
        self.history: dict[str, list[torch.Tensor]] = {}
        self.clear_history()

    @classmethod
    def from_autoencoder(cls, autoencoder: MinAutoencoder, **kwargs) -> "MinMTL":
        """Initialize fixed input/CA1/output maps from a mini autoencoder."""
        if not isinstance(autoencoder, MinAutoencoder):
            raise TypeError("autoencoder must be a MinAutoencoder")
        encoder, decoder, encoder_bias, decoder_bias = autoencoder.get_weights(
            bias=True,
        )
        kwargs.setdefault("beta_is", autoencoder.beta)
        kwargs.setdefault("gain_out", autoencoder.gain_out)
        kwargs.setdefault("offset_out", autoencoder.offset_out)
        model = cls(
            W_ei_ca1=encoder,
            W_ca1_eo=decoder,
            B_ei_ca1=encoder_bias,
            B_ca1_eo=decoder_bias,
            **kwargs,
        )
        # Constructor defaults are CPU tensors; consolidate every buffer on
        # the same device and dtype as the trained autoencoder.
        return model.to(device=encoder.device, dtype=encoder.dtype)

    def _as_column(self, x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        x = torch.as_tensor(
            x,
            dtype=self.W_ei_ca3.dtype,
            device=self.W_ei_ca3.device,
        )
        was_vector = x.shape == (INPUT_DIM,)
        if was_vector:
            x = x.reshape(INPUT_DIM, 1)
        if x.shape != (INPUT_DIM, 1):
            raise ValueError(
                f"MTL input must have shape (2,) or (2, 1), got {tuple(x.shape)}"
            )
        return x, was_vector

    @staticmethod
    def _activate_column(x: torch.Tensor, beta: float) -> torch.Tensor:
        return _sparsemoid(x.T, beta).T

    def _updated_weights(self, instructive_signal: torch.Tensor,
                         ca3: torch.Tensor, ca1: torch.Tensor) -> torch.Tensor:
        weights = self.W_ca3_ca1
        coactivity = instructive_signal @ ca3.T
        if self.plasticity == "base":
            updated = (1.0 - self.alpha * instructive_signal) * weights
            updated = updated + self.alpha * coactivity
        elif self.plasticity == "nois":
            updated = (1.0 - self.alpha) * weights + self.alpha * coactivity
        elif self.plasticity == "isout":
            updated = (1.0 - self.alpha) * instructive_signal * weights
            updated = updated + self.alpha * coactivity
        elif self.plasticity == "err1":
            error = instructive_signal - ca1
            normalizer = ca3.square().sum().clamp_min(1e-6)
            updated = weights + self.alpha * (error @ ca3.T) / normalizer
            updated = updated.clamp(0.0, 1.0)
        else:  # err2
            positive_error = torch.relu(instructive_signal - ca1)
            negative_error = torch.relu(ca1 - instructive_signal)
            potentiation = self.alpha * (positive_error @ ca3.T) * (1.0 - weights)
            depression = self.alpha * (negative_error @ ca3.T) * weights
            updated = (weights + potentiation - depression).clamp(0.0, 1.0)
        return updated

    def forward(self, x_ei: torch.Tensor, learn: bool | None = None,
                return_state: bool = False, ca1: bool = False):
        """Process one 2D point and optionally perform one plasticity update."""
        x_ei, was_vector = self._as_column(x_ei)
        should_learn = self.learning if learn is None else bool(learn)

        x_ca3 = self._activate_column(
            self.W_ei_ca3 @ x_ei + self.B_ei_ca3,
            self.beta_ca3,
        )
        x_ca1 = self._activate_column(
            self.W_ca3_ca1 @ x_ca3 + self.B_ca1,
            self.beta_ca1,
        )
        instructive_signal = self._activate_column(
            self.W_ei_ca1 @ x_ei + self.B_ei_ca1,
            self.beta_is,
        )

        if should_learn:
            with torch.no_grad():
                self.W_ca3_ca1.copy_(
                    self._updated_weights(instructive_signal, x_ca3, x_ca1)
                )

        decoded = self.W_ca1_eo @ x_ca1 + self.B_ca1_eo
        x_eo = torch.sigmoid(self.gain_out * (decoded - self.offset_out))
        state = {
            "input": x_ei,
            "ca3": x_ca3,
            "ca1": x_ca1,
            "instructive_signal": instructive_signal,
            "output": x_eo,
            "W_ca3_ca1": self.W_ca3_ca1,
        }
        if self.record_history:
            for name, value in state.items():
                self.history[name].append(value.detach().clone())

        output = x_eo.reshape(OUTPUT_DIM) if was_vector else x_eo
        if return_state:
            returned_state = {
                name: value.detach().clone()
                for name, value in state.items()
            }
            return output, returned_state
        if ca1:
            returned_ca1 = x_ca1.reshape(LATENT_DIM) if was_vector else x_ca1
            return output, returned_ca1
        return output

    def pause_learning(self) -> None:
        self.learning = False

    def resume_learning(self) -> None:
        self.learning = True

    # Compatibility aliases used by the larger MTL model.
    pause_lr = pause_learning
    resume_lr = resume_learning

    def clear_history(self) -> None:
        self.history = {
            "input": [],
            "ca3": [],
            "ca1": [],
            "instructive_signal": [],
            "output": [],
            "W_ca3_ca1": [],
        }

    def reset(self, reset_weights: bool = False) -> None:
        """Clear recorded dynamics and optionally reset plastic weights."""
        self.clear_history()
        self.learning = True
        if reset_weights:
            self.W_ca3_ca1.zero_()
