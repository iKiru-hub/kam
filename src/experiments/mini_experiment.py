"""Train and visualize the minimal 2D→3D→2D AE→MTL system."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

PROJECT_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_SRC))

from models import MinAutoencoder, MinMTL, PLASTICITY_RULES


DATA_KINDS = ("uniform", "circle", "sine", "landscape")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a 2D autoencoder with visualizable 3D AE/MTL activity."
    )
    parser.add_argument("--training-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patterns", type=int, default=24)
    parser.add_argument(
        "--data-kind",
        choices=DATA_KINDS,
        default="circle",
        help="probability field used to draw fresh 2D samples",
    )
    parser.add_argument(
        "--field-width",
        type=float,
        default=0.04,
        help="standard deviation of the circle/curve probability band",
    )
    parser.add_argument("--field-resolution", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument(
        "--plasticity",
        type=str.lower,
        choices=PLASTICITY_RULES,
        default="base",
        help="online CA3-to-CA1 plasticity rule",
    )
    parser.add_argument("--seed", type=int, default=3980)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve an explicitly requested device or the best available backend."""
    if device != "auto":
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if resolved.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return resolved
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_uniform_data(training_size: int, test_size: int,
                      seed: int = 3980) -> tuple[torch.Tensor, torch.Tensor]:
    """Create reproducible uniform samples from the unit square."""
    if training_size < 1 or test_size < 1:
        raise ValueError("training_size and test_size must both be positive")
    generator = torch.Generator().manual_seed(seed)
    training_data = torch.rand(training_size, 2, generator=generator)
    test_data = torch.rand(test_size, 2, generator=generator)
    return training_data, test_data


def make_density_field(data_kind: str = "circle", field_width: float = 0.04,
                       resolution: int = 128) -> dict:
    """Create a normalized probability field over the unit square."""
    data_kind = str(data_kind).lower()
    if data_kind not in DATA_KINDS:
        raise ValueError(f"data_kind must be one of {DATA_KINDS}")
    if field_width <= 0:
        raise ValueError("field_width must be positive")
    if resolution < 8:
        raise ValueError("field_resolution must be at least 8")

    coordinates = (
        torch.arange(resolution, dtype=torch.float32) + 0.5
    ) / resolution
    grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    if data_kind == "uniform":
        density = torch.ones_like(grid_x)
    elif data_kind == "circle":
        radius = 0.30
        radial_distance = torch.sqrt(
            (grid_x - 0.5).square() + (grid_y - 0.5).square()
        )
        distance = (radial_distance - radius).abs()
        density = torch.exp(-0.5 * (distance / field_width).square())
    elif data_kind == "sine":
        amplitude = 0.28
        angular_frequency = 2.0 * torch.pi
        curve_y = 0.5 + amplitude * torch.sin(angular_frequency * grid_x)
        curve_slope = amplitude * angular_frequency * torch.cos(
            angular_frequency * grid_x
        )
        # Local normal distance gives the band a roughly constant thickness.
        distance = (grid_y - curve_y).abs() / torch.sqrt(
            1.0 + curve_slope.square()
        )
        density = torch.exp(-0.5 * (distance / field_width).square())
    else:  # landscape
        def gaussian_peak(center_x: float, center_y: float,
                          scale_x: float, scale_y: float,
                          angle: float) -> torch.Tensor:
            delta_x = grid_x - center_x
            delta_y = grid_y - center_y
            cosine = float(np.cos(angle))
            sine = float(np.sin(angle))
            rotated_x = cosine * delta_x + sine * delta_y
            rotated_y = -sine * delta_x + cosine * delta_y
            return torch.exp(-0.5 * (
                (rotated_x / scale_x).square()
                + (rotated_y / scale_y).square()
            ))

        # A winding ridge provides structure across the square.
        ridge_y = 0.34 + 0.13 * torch.sin(3.5 * torch.pi * grid_x + 0.35)
        ridge_window = torch.exp(-0.5 * ((grid_x - 0.47) / 0.33).square())
        ridge = torch.exp(
            -0.5 * ((grid_y - ridge_y) / field_width).square()
        ) * ridge_window

        # Two rotated hills add unequal, locally two-dimensional modes.
        peak_width = max(1.5 * field_width, 0.055)
        peak_1 = gaussian_peak(0.25, 0.76, 1.8 * peak_width,
                               peak_width, 0.65)
        peak_2 = gaussian_peak(0.76, 0.63, peak_width,
                               1.6 * peak_width, -0.45)

        # A gated ring creates an additional curved mode without closing it.
        ring_radius = torch.sqrt(
            (grid_x - 0.72).square() + (grid_y - 0.73).square()
        )
        ring = torch.exp(
            -0.5 * ((ring_radius - 0.18) / (0.8 * field_width)).square()
        )
        ring_gate = torch.sigmoid(25.0 * (grid_x + grid_y - 1.25))
        density = ridge + 0.85 * peak_1 + 0.70 * peak_2 \
            + 0.55 * ring * ring_gate

    density = density / density.sum()
    return {
        "kind": data_kind,
        "density": density,
        "resolution": resolution,
        "field_width": float(field_width),
    }


def sample_density_field(field: dict, num_samples: int,
                         generator: torch.Generator | None = None
                         ) -> torch.Tensor:
    """Draw continuous 2D points from a discretized probability field."""
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    density = torch.as_tensor(field["density"], dtype=torch.float32)
    if density.ndim != 2 or density.shape[0] != density.shape[1]:
        raise ValueError("density field must be a square matrix")
    resolution = density.shape[0]
    flat_indices = torch.multinomial(
        density.reshape(-1),
        num_samples,
        replacement=True,
        generator=generator,
    )
    rows = torch.div(flat_indices, resolution, rounding_mode="floor")
    columns = flat_indices.remainder(resolution)
    jitter = torch.rand(num_samples, 2, generator=generator)
    x = (columns.to(torch.float32) + jitter[:, 0]) / resolution
    y = (rows.to(torch.float32) + jitter[:, 1]) / resolution
    return torch.stack((x, y), dim=1).clamp(0.0, 1.0)


def make_field_data(training_size: int, test_size: int,
                    data_kind: str = "circle", field_width: float = 0.04,
                    resolution: int = 128, seed: int = 3980
                    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Create independent train/test samples from one probability field."""
    if training_size < 1 or test_size < 1:
        raise ValueError("training_size and test_size must both be positive")
    field = make_density_field(data_kind, field_width, resolution)
    generator = torch.Generator().manual_seed(seed)
    samples = sample_density_field(
        field,
        training_size + test_size,
        generator,
    )
    return samples[:training_size], samples[training_size:], field


def _mean_mse(model: MinAutoencoder, data: torch.Tensor,
              device: torch.device, criterion: nn.Module) -> float:
    model.eval()
    with torch.no_grad():
        prediction = model(data.to(device))
        loss = criterion(prediction, data.to(device))
    return float(loss.detach().cpu())


def train_min_autoencoder(training_data: torch.Tensor,
                          test_data: torch.Tensor,
                          epochs: int = 300,
                          batch_size: int = 64,
                          learning_rate: float = 1e-3,
                          device: str | torch.device = "cpu",
                          seed: int = 3980,
                          density_field: dict | None = None):
    """Fit ``MinAutoencoder``, optionally resampling a field every epoch."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    torch.manual_seed(seed)
    device = torch.device(device)
    model = MinAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loader_generator = torch.Generator().manual_seed(seed)
    field_generator = torch.Generator().manual_seed(seed + 1)
    history = {"train_loss": [], "test_loss": []}

    for _ in range(epochs):
        epoch_data = training_data
        if density_field is not None:
            epoch_data = sample_density_field(
                density_field,
                len(training_data),
                field_generator,
            )
        dataloader = DataLoader(
            TensorDataset(epoch_data),
            batch_size=batch_size,
            shuffle=True,
            generator=loader_generator,
        )
        model.train()
        total_loss = 0.0
        total_samples = 0
        for (batch,) in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item() * len(batch)
            total_samples += len(batch)

        history["train_loss"].append(total_loss / total_samples)
        history["test_loss"].append(
            _mean_mse(model, test_data, device, criterion)
        )

    model.eval()
    return model, history


def reconstruction_accuracy(target: torch.Tensor,
                            prediction: torch.Tensor) -> float:
    """Return ``1 - MSE`` in [0, 1] for unit-square reconstructions."""
    prediction = prediction.detach().reshape(-1)
    target = torch.as_tensor(
        target,
        dtype=prediction.dtype,
        device=prediction.device,
    ).reshape(-1)
    mse = torch.mean((target - prediction).square())
    return float((1.0 - mse).clamp(0.0, 1.0).cpu())


def train_min_mtl(autoencoder: MinAutoencoder, data: torch.Tensor,
                  alpha: float = 0.1, plasticity: str = "base"):
    """Train independent sequence prefixes and return their recall matrix."""
    if data.ndim != 2 or data.shape[1] != 2 or len(data) == 0:
        raise ValueError("MTL data must have shape (num_patterns, 2)")

    model = MinMTL.from_autoencoder(
        autoencoder,
        alpha=alpha,
        plasticity=plasticity,
        record_history=True,
    )
    accuracy = np.full((len(data), len(data)), np.nan, dtype=float)

    with torch.no_grad():
        for final_pattern in range(len(data)):
            # Every row represents a fresh memory trained on patterns 0..i.
            model.reset(reset_weights=True)
            model.resume_learning()
            for pattern in data[:final_pattern + 1]:
                model(pattern, learn=True)

            model.pause_learning()
            for pattern_index, pattern in enumerate(data[:final_pattern + 1]):
                reconstruction = model(pattern, learn=False)
                accuracy[final_pattern, pattern_index] = reconstruction_accuracy(
                    pattern,
                    reconstruction,
                )

    return model, accuracy


def _draw_simplex(ax) -> None:
    """Draw the three-unit simplex boundary on a 3D activity axis."""
    vertices = np.eye(3)
    for start, end in ((0, 1), (1, 2), (2, 0)):
        edge = vertices[[start, end]]
        ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], color="0.55", lw=1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(0.0, 1.0)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("unit 1")
    ax.set_ylabel("unit 2")
    ax.set_zlabel("unit 3")


def _draw_density_field(ax, density_field: dict | None) -> None:
    """Draw a probability field as a light background on a 2D axis."""
    if density_field is None:
        return
    field_image = torch.as_tensor(density_field["density"]).cpu().numpy()
    field_image = field_image / field_image.max()
    ax.imshow(
        field_image,
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        cmap="Greys",
        alpha=0.28,
        vmin=0.0,
        vmax=1.0,
    )


def plot_results(autoencoder: MinAutoencoder, mtl: MinMTL, history: dict,
                 test_data: torch.Tensor, accuracy: np.ndarray,
                 plasticity: str, density_field: dict | None = None,
                 show: bool = True):
    """Plot reconstruction, 3D internal dynamics, weights, and recall."""
    device = next(autoencoder.parameters()).device
    with torch.no_grad():
        reconstruction, latent = autoencoder(
            test_data.to(device),
            return_latent=True,
        )
    reconstruction = reconstruction.cpu().numpy()
    latent = latent.cpu().numpy()
    original = test_data.cpu().numpy()

    num_patterns = accuracy.shape[0]
    previous_record_history = mtl.record_history
    mtl.record_history = False
    try:
        with torch.no_grad():
            mtl_predictions = torch.stack([
                mtl(pattern, learn=False)
                for pattern in test_data[:num_patterns]
            ])
    finally:
        mtl.record_history = previous_record_history
    mtl_predictions = mtl_predictions.cpu().numpy()

    fig = plt.figure(figsize=(22, 10))
    axes = [
        fig.add_subplot(2, 4, 1),
        fig.add_subplot(2, 4, 2),
        fig.add_subplot(2, 4, 3, projection="3d"),
        fig.add_subplot(2, 4, 4, projection="3d"),
        fig.add_subplot(2, 4, 5),
        fig.add_subplot(2, 4, 6),
        fig.add_subplot(2, 4, 7),
        fig.add_subplot(2, 4, 8),
    ]

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["test_loss"], label="validation")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Minimal autoencoder training")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    _draw_density_field(axes[1], density_field)
    axes[1].scatter(
        original[:, 0], original[:, 1],
        s=14, alpha=0.45, label="sample",
    )
    axes[1].scatter(
        reconstruction[:, 0], reconstruction[:, 1],
        s=14, alpha=0.5, label="reconstruction",
    )
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlabel("dimension 1")
    axes[1].set_ylabel("dimension 2")
    field_name = "data" if density_field is None else density_field["kind"]
    axes[1].set_title(f"{str(field_name).title()} field and reconstruction")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].scatter(
        latent[:, 0], latent[:, 1], latent[:, 2],
        c=original[:, 0], cmap="viridis", s=18, alpha=0.65,
    )
    _draw_simplex(axes[2])
    axes[2].set_title("Autoencoder latent activity")

    ca3 = torch.stack(mtl.history["ca3"][:num_patterns]).squeeze(-1).cpu().numpy()
    ca1 = torch.stack(mtl.history["ca1"][:num_patterns]).squeeze(-1).cpu().numpy()
    pattern_index = np.arange(num_patterns)
    axes[3].plot(
        ca3[:, 0], ca3[:, 1], ca3[:, 2],
        color="tab:blue", alpha=0.55, label="CA3",
    )
    axes[3].scatter(
        ca3[:, 0], ca3[:, 1], ca3[:, 2],
        c=pattern_index, cmap="Blues", s=25,
    )
    axes[3].plot(
        ca1[:, 0], ca1[:, 1], ca1[:, 2],
        color="tab:orange", alpha=0.55, label="CA1",
    )
    axes[3].scatter(
        ca1[:, 0], ca1[:, 1], ca1[:, 2],
        c=pattern_index, cmap="Oranges", s=25,
    )
    _draw_simplex(axes[3])
    axes[3].set_title("MTL learning trajectory")
    axes[3].legend()

    recall_targets = original[:num_patterns]
    _draw_density_field(axes[4], density_field)
    recall_delta = mtl_predictions - recall_targets
    axes[4].quiver(
        recall_targets[:, 0], recall_targets[:, 1],
        recall_delta[:, 0], recall_delta[:, 1],
        angles="xy", scale_units="xy", scale=1.0,
        color="0.35", alpha=0.5, width=0.004,
    )
    axes[4].scatter(
        recall_targets[:, 0], recall_targets[:, 1],
        facecolors="none", edgecolors="tab:blue", s=38, label="target",
    )
    axes[4].scatter(
        mtl_predictions[:, 0], mtl_predictions[:, 1],
        color="tab:orange", marker="x", s=38, label="MTL prediction",
    )
    axes[4].set_xlim(0.0, 1.0)
    axes[4].set_ylim(0.0, 1.0)
    axes[4].set_aspect("equal", adjustable="box")
    axes[4].set_xlabel("dimension 1")
    axes[4].set_ylabel("dimension 2")
    axes[4].set_title(f"MTL reconstruction after {num_patterns} patterns")
    axes[4].grid(alpha=0.3)
    axes[4].legend()

    colormap = plt.get_cmap("viridis").copy()
    colormap.set_bad(color="white")
    image = axes[5].imshow(
        np.ma.masked_invalid(accuracy),
        origin="upper",
        aspect="auto",
        cmap=colormap,
        vmin=0.0,
        vmax=1.0,
    )
    axes[5].set_xlabel("recalled pattern")
    axes[5].set_ylabel("number of patterns learned")
    axes[5].set_title(f"MTL recall ({plasticity.upper()})")
    fig.colorbar(image, ax=axes[5], label="accuracy (1 - MSE)")

    weights = mtl.W_ca3_ca1.detach().cpu().numpy()
    weight_image = axes[6].imshow(weights, cmap="magma", aspect="equal")
    axes[6].set_xticks(range(3), labels=("CA3 1", "CA3 2", "CA3 3"))
    axes[6].set_yticks(range(3), labels=("CA1 1", "CA1 2", "CA1 3"))
    axes[6].set_title("Final plastic CA3→CA1 weights")
    fig.colorbar(weight_image, ax=axes[6], label="weight")

    final_accuracy = accuracy[-1, :num_patterns]
    pattern_numbers = np.arange(1, num_patterns + 1)
    axes[7].plot(pattern_numbers, final_accuracy, marker="o", ms=4)
    axes[7].axhline(
        np.mean(final_accuracy),
        color="tab:red", linestyle="--", alpha=0.7,
        label=f"mean={np.mean(final_accuracy):.3f}",
    )
    axes[7].set_ylim(0.0, 1.02)
    axes[7].set_xlabel("pattern")
    axes[7].set_ylabel("accuracy (1 - MSE)")
    axes[7].set_title("Recall after the complete sequence")
    axes[7].grid(alpha=0.3)
    axes[7].legend()

    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


def run_experiment(training_size: int = 1000, test_size: int = 200,
                   epochs: int = 300, batch_size: int = 64,
                   learning_rate: float = 1e-3, num_patterns: int = 24,
                   alpha: float = 0.1, plasticity: str = "base",
                   data_kind: str = "circle", field_width: float = 0.04,
                   field_resolution: int = 128,
                   seed: int = 3980, device: str = "auto",
                   plot: bool = True):
    """Run the complete minimal AE→MTL experiment."""
    if not 1 <= num_patterns <= test_size:
        raise ValueError("num_patterns must be between 1 and test_size")
    resolved_device = resolve_device(device)
    training_data, test_data, density_field = make_field_data(
        training_size,
        test_size,
        data_kind=data_kind,
        field_width=field_width,
        resolution=field_resolution,
        seed=seed,
    )
    autoencoder, history = train_min_autoencoder(
        training_data,
        test_data,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=resolved_device,
        seed=seed,
        density_field=density_field,
    )
    mtl, accuracy = train_min_mtl(
        autoencoder,
        test_data[:num_patterns],
        alpha=alpha,
        plasticity=plasticity,
    )
    figure = None
    if plot:
        figure = plot_results(
            autoencoder,
            mtl,
            history,
            test_data,
            accuracy,
            plasticity,
            density_field=density_field,
        )[0]
    return {
        "autoencoder": autoencoder,
        "mtl": mtl,
        "training_data": training_data,
        "test_data": test_data,
        "density_field": density_field,
        "data_kind": density_field["kind"],
        "plasticity": mtl.plasticity,
        "history": history,
        "accuracy": accuracy,
        "figure": figure,
        "device": str(resolved_device),
    }


def main():
    args = parse_args()
    results = run_experiment(
        training_size=args.training_size,
        test_size=args.test_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_patterns=args.patterns,
        data_kind=args.data_kind,
        field_width=args.field_width,
        field_resolution=args.field_resolution,
        alpha=args.alpha,
        plasticity=args.plasticity,
        seed=args.seed,
        device=args.device,
        plot=not args.no_plot,
    )
    print(
        f"data={results['data_kind']}  "
        f"plasticity={results['plasticity'].upper()}  "
        f"device={results['device']}  "
        f"validation_mse={results['history']['test_loss'][-1]:.6f}  "
        f"mean_mtl_accuracy={np.nanmean(results['accuracy']):.4f}"
    )


if __name__ == "__main__":
    main()
