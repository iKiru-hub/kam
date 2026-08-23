"""Evolve mini-MTL parameters for accurate two-dimensional reconstruction."""

from __future__ import annotations

import argparse
import functools
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


EVOLUTION_DIR = Path(__file__).resolve().parent
PROJECT_SRC = EVOLUTION_DIR.parents[1]
sys.path.insert(0, str(EVOLUTION_DIR))
sys.path.insert(0, str(PROJECT_SRC))

import _lib
from experiments.mini_experiment import (
    DATA_KINDS,
    make_field_data,
    resolve_device,
    sample_density_field,
    train_min_autoencoder,
)
from models import MinAutoencoder, MinMTL, PLASTICITY_RULES


MINI_PARAMETER_NAMES = (
    "alpha",
    "beta_is",
    "beta_ca3",
    "beta_ca1",
    "gain_out",
    "offset_out",
    "ca3_angle",
)
MINI_PARAMETER_CENTERS = np.array([
    0.10, 8.0, 8.0, 8.0, 10.0, 0.10, 0.0,
])
MINI_PARAMETER_SCALES = np.array([
    0.08, 6.0, 6.0, 6.0, 6.0, 0.08, np.pi / 2.0,
])
MINI_PARAMETER_LOWER = np.array([
    1e-4, 0.5, 0.5, 0.5, 1.0, -0.25, -np.pi,
])
MINI_PARAMETER_UPPER = np.array([
    0.80, 64.0, 64.0, 64.0, 40.0, 0.50, np.pi,
])

DEFAULT_GENERATIONS = 512
DEFAULT_PATTERNS = 256
DEFAULT_EVALUATION_REPETITIONS = 3
DEFAULT_AE_TRAINING_SIZE = 1000
DEFAULT_AE_TEST_SIZE = 200
DEFAULT_AE_EPOCHS = 300
DEFAULT_SEED = 3980


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evolve mini-MTL parameters to minimize reconstruction MSE."
    )
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--patterns", type=int, default=DEFAULT_PATTERNS)
    parser.add_argument(
        "--evaluation-repetitions",
        type=int,
        default=DEFAULT_EVALUATION_REPETITIONS,
    )
    parser.add_argument(
        "--data-kind",
        choices=DATA_KINDS,
        default="landscape",
    )
    parser.add_argument("--field-width", type=float, default=0.04)
    parser.add_argument(
        "--plasticity",
        type=str.lower,
        choices=PLASTICITY_RULES,
        default="err2",
    )
    parser.add_argument("--ae-training-size", type=int,
                        default=DEFAULT_AE_TRAINING_SIZE)
    parser.add_argument("--ae-test-size", type=int,
                        default=DEFAULT_AE_TEST_SIZE)
    parser.add_argument("--ae-epochs", type=int, default=DEFAULT_AE_EPOCHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--pause", type=float, default=0.01)
    parser.add_argument(
        "--tuning-resolution",
        type=int,
        default=31,
        help="grid resolution for the final CA1 and reconstruction fields",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="parallel workers; defaults to min(population size, CPU count)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="device used once for autoencoder training; evolution uses CPUs",
    )
    parser.add_argument("--no-live-plot", action="store_true")
    parser.add_argument("--no-success-plot", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save-name", default="mini_evolution")
    return parser.parse_args()


def sanitizer(genome) -> np.ndarray:
    """Decode standardized CMA coordinates into bounded MTL parameters."""
    genome = np.asarray(genome, dtype=float)
    if genome.shape != MINI_PARAMETER_CENTERS.shape:
        raise ValueError(
            f"expected genome shape {MINI_PARAMETER_CENTERS.shape}, "
            f"got {genome.shape}"
        )
    parameters = MINI_PARAMETER_CENTERS + MINI_PARAMETER_SCALES * genome
    parameters[:-1] = np.clip(
        parameters[:-1],
        MINI_PARAMETER_LOWER[:-1],
        MINI_PARAMETER_UPPER[:-1],
    )
    # CA3 orientation is circular: +pi and -pi describe the same projection.
    parameters[-1] = (parameters[-1] + np.pi) % (2.0 * np.pi) - np.pi
    return parameters


def _pack_autoencoder(autoencoder: MinAutoencoder) -> dict:
    """Convert a trained mini autoencoder into a process-safe bundle."""
    return {
        "state_dict": {
            name: value.detach().cpu().numpy()
            for name, value in autoencoder.state_dict().items()
        },
        "beta": autoencoder.beta,
        "gain_out": autoencoder.gain_out,
        "offset_out": autoencoder.offset_out,
        "use_bias": autoencoder.use_bias,
    }


def _unpack_autoencoder(bundle: dict) -> MinAutoencoder:
    """Recreate a CPU autoencoder inside an evaluation process."""
    autoencoder = MinAutoencoder(
        beta=bundle["beta"],
        gain_out=bundle["gain_out"],
        offset_out=bundle["offset_out"],
        use_bias=bundle["use_bias"],
    )
    state = {
        name: torch.as_tensor(value, dtype=torch.float32)
        for name, value in bundle["state_dict"].items()
    }
    autoencoder.load_state_dict(state)
    autoencoder.eval()
    return autoencoder


def _candidate_parameters(candidate) -> dict[str, float]:
    candidate = np.asarray(candidate, dtype=float)
    if candidate.shape != MINI_PARAMETER_CENTERS.shape:
        raise ValueError(
            f"expected candidate shape {MINI_PARAMETER_CENTERS.shape}, "
            f"got {candidate.shape}"
        )
    return {
        name: float(value)
        for name, value in zip(MINI_PARAMETER_NAMES, candidate)
    }


def _train_candidate_model(candidate, autoencoder_bundle: dict,
                           memory_data, plasticity: str
                           ) -> tuple[MinMTL, torch.Tensor]:
    """Build and train one candidate MTL on a memory sequence."""
    parameters = _candidate_parameters(candidate)
    autoencoder = _unpack_autoencoder(autoencoder_bundle)
    ca3_angles = parameters["ca3_angle"] + (
        2.0 * np.pi * torch.arange(3, dtype=torch.float32) / 3.0
    )
    ca3_projection = torch.stack(
        (torch.cos(ca3_angles), torch.sin(ca3_angles)),
        dim=1,
    )
    ca3_bias = -ca3_projection @ torch.full((2, 1), 0.5)
    model = MinMTL.from_autoencoder(
        autoencoder,
        W_ei_ca3=ca3_projection,
        B_ei_ca3=ca3_bias,
        alpha=parameters["alpha"],
        beta_is=parameters["beta_is"],
        beta_ca3=parameters["beta_ca3"],
        beta_ca1=parameters["beta_ca1"],
        gain_out=parameters["gain_out"],
        offset_out=parameters["offset_out"],
        plasticity=plasticity,
        record_history=False,
    )
    memory_data = torch.as_tensor(memory_data, dtype=torch.float32)
    with torch.no_grad():
        model.resume_learning()
        for pattern in memory_data:
            model(pattern, learn=True)
        model.pause_learning()
    return model, memory_data


def predict_memory_sequence(candidate, autoencoder_bundle: dict,
                            memory_data, plasticity: str) -> np.ndarray:
    """Train one candidate on a sequence and return final reconstructions."""
    model, memory_data = _train_candidate_model(
        candidate,
        autoencoder_bundle,
        memory_data,
        plasticity,
    )
    with torch.no_grad():
        predictions = torch.stack([
            model(pattern, learn=False)
            for pattern in memory_data
        ])
    return predictions.cpu().numpy()


def evaluate_tuning_fields(candidate, autoencoder_bundle: dict,
                           memory_data, plasticity: str,
                           resolution: int = 31) -> dict:
    """Evaluate frozen CA1 tuning and reconstruction over the input square."""
    resolution = int(resolution)
    if resolution < 5:
        raise ValueError("tuning_resolution must be at least 5")
    model, _ = _train_candidate_model(
        candidate,
        autoencoder_bundle,
        memory_data,
        plasticity,
    )
    coordinates = torch.linspace(0.0, 1.0, resolution)
    grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    grid_points = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)
    outputs = []
    ca1_activity = []
    with torch.no_grad():
        for point in grid_points:
            output, state = model(point, learn=False, return_state=True)
            outputs.append(output)
            ca1_activity.append(state["ca1"].reshape(-1))
    outputs = torch.stack(outputs).reshape(resolution, resolution, 2)
    ca1_activity = torch.stack(ca1_activity).reshape(
        resolution,
        resolution,
        3,
    )
    input_grid = torch.stack((grid_x, grid_y), dim=-1)
    vectors = outputs - input_grid
    error = vectors.square().mean(dim=-1)
    return {
        "x": grid_x.cpu().numpy(),
        "y": grid_y.cpu().numpy(),
        "ca1": ca1_activity.cpu().numpy(),
        "predictions": outputs.cpu().numpy(),
        "vectors": vectors.cpu().numpy(),
        "error": error.cpu().numpy(),
        "weights": model.W_ca3_ca1.detach().cpu().numpy(),
    }


def evaluate_mini_individual(candidate, autoencoder_bundle: dict,
                             memory_sets, plasticity: str = "base") -> float:
    """Return mean final recall MSE across deterministic memory sequences."""
    memory_sets = np.asarray(memory_sets, dtype=np.float32)
    losses = []
    for memory_data in memory_sets:
        predictions = predict_memory_sequence(
            candidate,
            autoencoder_bundle,
            memory_data,
            plasticity,
        )
        losses.append(float(np.mean((predictions - memory_data) ** 2)))
    loss = float(np.mean(losses))
    return loss if np.isfinite(loss) else 1.0


def evaluate_mini_population(population, autoencoder_bundle: dict,
                             memory_sets, plasticity: str = "base",
                             disable: bool = True) -> list[float]:
    """Sequential fallback evaluator used when multiprocessing is disabled."""
    return [
        evaluate_mini_individual(
            candidate,
            autoencoder_bundle,
            memory_sets,
            plasticity,
        )
        for candidate in tqdm(population, disable=disable)
    ]


def _sample_memory_sets(field: dict, repetitions: int, patterns: int,
                        seed: int) -> np.ndarray:
    generator = torch.Generator().manual_seed(seed)
    samples = sample_density_field(
        field,
        repetitions * patterns,
        generator,
    )
    return samples.reshape(repetitions, patterns, 2).numpy()


def _draw_field(ax, field: dict) -> None:
    density = torch.as_tensor(field["density"]).cpu().numpy()
    density = density / density.max()
    ax.imshow(
        density,
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        cmap="Greys",
        alpha=0.28,
        vmin=0.0,
        vmax=1.0,
    )


def _draw_density_contours(ax, field: dict, color: str = "white") -> None:
    density = torch.as_tensor(field["density"]).cpu().numpy()
    density = density / density.max()
    resolution = density.shape[0]
    coordinates = (np.arange(resolution) + 0.5) / resolution
    ax.contour(
        coordinates,
        coordinates,
        density,
        levels=(0.2, 0.5, 0.8),
        colors=color,
        linewidths=0.8,
        alpha=0.8,
    )


def _format_unit_square(ax) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("dimension 1")
    ax.set_ylabel("dimension 2")


def plot_search_success(record: dict, field: dict, holdout_data,
                        baseline_predictions, best_predictions,
                        tuning_fields: dict,
                        show: bool = True):
    """Plot CA1 tuning, learned fields, and reconstruction success."""
    holdout_data = np.asarray(holdout_data, dtype=float)
    baseline_predictions = np.asarray(baseline_predictions, dtype=float)
    best_predictions = np.asarray(best_predictions, dtype=float)
    baseline_errors = np.mean(
        (baseline_predictions - holdout_data) ** 2,
        axis=1,
    )
    best_errors = np.mean((best_predictions - holdout_data) ** 2, axis=1)
    baseline_mse = float(baseline_errors.mean())
    best_mse = float(best_errors.mean())
    improvement = 100.0 * (baseline_mse - best_mse) / max(baseline_mse, 1e-12)

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(18, 16),
        constrained_layout=True,
    )

    activity_min = float(np.min(tuning_fields["ca1"]))
    activity_max = float(np.max(tuning_fields["ca1"]))
    if activity_max - activity_min < 1e-8:
        activity_min -= 1e-4
        activity_max += 1e-4
    tuning_image = None
    for neuron in range(3):
        tuning_image = axes[0, neuron].imshow(
            tuning_fields["ca1"][..., neuron],
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            cmap="viridis",
            vmin=activity_min,
            vmax=activity_max,
        )
        _draw_density_contours(axes[0, neuron], field)
        _format_unit_square(axes[0, neuron])
        axes[0, neuron].set_title(f"CA1 neuron {neuron + 1} tuning")
    fig.colorbar(
        tuning_image,
        ax=axes[0, :].tolist(),
        label="CA1 activity (white contours: input density)",
        shrink=0.85,
    )

    grid_x = tuning_fields["x"]
    grid_y = tuning_fields["y"]
    vectors = tuning_fields["vectors"]
    vector_magnitude = np.linalg.norm(vectors, axis=-1)
    stride = max(1, int(np.ceil(grid_x.shape[0] / 12)))
    _draw_field(axes[1, 0], field)
    vector_plot = axes[1, 0].quiver(
        grid_x[::stride, ::stride],
        grid_y[::stride, ::stride],
        vectors[::stride, ::stride, 0],
        vectors[::stride, ::stride, 1],
        vector_magnitude[::stride, ::stride],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        cmap="plasma",
        width=0.004,
    )
    _draw_density_contours(axes[1, 0], field, color="0.25")
    _format_unit_square(axes[1, 0])
    axes[1, 0].set_title(r"Reconstruction flow $\hat{x}(x)-x$")
    fig.colorbar(vector_plot, ax=axes[1, 0], label="displacement magnitude")

    error_image = axes[1, 1].imshow(
        tuning_fields["error"],
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        cmap="magma",
        vmin=0.0,
    )
    _draw_density_contours(axes[1, 1], field)
    _format_unit_square(axes[1, 1])
    axes[1, 1].set_title("Reconstruction-error field")
    fig.colorbar(error_image, ax=axes[1, 1], label="MSE")

    _draw_field(axes[1, 2], field)
    delta = best_predictions - holdout_data
    axes[1, 2].quiver(
        holdout_data[:, 0], holdout_data[:, 1],
        delta[:, 0], delta[:, 1],
        angles="xy", scale_units="xy", scale=1.0,
        color="0.35", alpha=0.5, width=0.004,
    )
    axes[1, 2].scatter(
        holdout_data[:, 0], holdout_data[:, 1],
        facecolors="none", edgecolors="tab:blue", s=42, label="target",
    )
    axes[1, 2].scatter(
        baseline_predictions[:, 0], baseline_predictions[:, 1],
        color="0.45", marker="+", s=38, label="baseline",
    )
    axes[1, 2].scatter(
        best_predictions[:, 0], best_predictions[:, 1],
        color="tab:orange", marker="x", s=42, label="evolved",
    )
    _format_unit_square(axes[1, 2])
    axes[1, 2].set_title(
        f"Holdout recall: {baseline_mse:.4f} → {best_mse:.4f} MSE"
    )
    axes[1, 2].grid(alpha=0.3)
    axes[1, 2].legend()

    generations = np.asarray(record["generations"], dtype=int)
    axes[2, 0].plot(
        generations,
        record["population_mean_fitness"],
        color="tab:blue",
        alpha=0.75,
        label="population mean",
    )
    axes[2, 0].plot(
        generations,
        record["best_fitness"],
        color="tab:green",
        linewidth=2,
        label="best seen",
    )
    axes[2, 0].set_xlabel("generation")
    axes[2, 0].set_ylabel("reconstruction MSE")
    axes[2, 0].set_title("Mini-MTL evolution")
    axes[2, 0].grid(alpha=0.3)
    axes[2, 0].legend()

    pattern_numbers = np.arange(1, len(holdout_data) + 1)
    axes[2, 1].plot(
        pattern_numbers,
        baseline_errors,
        marker="o",
        ms=4,
        color="0.45",
        label="baseline",
    )
    axes[2, 1].plot(
        pattern_numbers,
        best_errors,
        marker="o",
        ms=4,
        color="tab:orange",
        label="evolved",
    )
    axes[2, 1].set_xlabel("stored pattern")
    axes[2, 1].set_ylabel("reconstruction MSE")
    axes[2, 1].set_title(f"Holdout improvement: {improvement:+.1f}%")
    axes[2, 1].grid(alpha=0.3)
    axes[2, 1].legend()

    weight_image = axes[2, 2].imshow(
        tuning_fields["weights"],
        cmap="magma",
        aspect="equal",
    )
    axes[2, 2].set_xticks(range(3), labels=("CA3 1", "CA3 2", "CA3 3"))
    axes[2, 2].set_yticks(range(3), labels=("CA1 1", "CA1 2", "CA1 3"))
    axes[2, 2].set_title("Evolved final CA3→CA1 weights")
    fig.colorbar(weight_image, ax=axes[2, 2], label="weight")

    fig.suptitle(
        f"Mini-MTL search success | {record['data_kind'].upper()} | "
        f"{record['plasticity'].upper()}"
    )
    if show:
        plt.show()
    return fig, axes


def mini_search(generations: int = DEFAULT_GENERATIONS,
                patterns: int = DEFAULT_PATTERNS,
                evaluation_repetitions: int = DEFAULT_EVALUATION_REPETITIONS,
                data_kind: str = "landscape", field_width: float = 0.04,
                plasticity: str = "base", ae_training_size: int = 1000,
                ae_test_size: int = 200, ae_epochs: int = 300,
                seed: int = DEFAULT_SEED, pause: float = 0.01,
                workers=None, device: str = "auto", live_plot: bool = True,
                success_plot: bool = True, tuning_resolution: int = 31,
                verbose: bool = True,
                save: bool = False, save_name: str = "mini_evolution") -> dict:
    """Train the mini AE, evolve MTL parameters, and validate the winner."""
    generations = int(generations)
    patterns = int(patterns)
    evaluation_repetitions = int(evaluation_repetitions)
    if generations < 1:
        raise ValueError("generations must be at least 1")
    if patterns < 1:
        raise ValueError("patterns must be at least 1")
    if evaluation_repetitions < 1:
        raise ValueError("evaluation_repetitions must be at least 1")
    if int(tuning_resolution) < 5:
        raise ValueError("tuning_resolution must be at least 5")
    plasticity = str(plasticity).lower()
    if plasticity not in PLASTICITY_RULES:
        raise ValueError(f"plasticity must be one of {PLASTICITY_RULES}")

    resolved_device = resolve_device(device)
    training_data, test_data, field = make_field_data(
        ae_training_size,
        ae_test_size,
        data_kind=data_kind,
        field_width=field_width,
        seed=seed,
    )
    autoencoder, ae_history = train_min_autoencoder(
        training_data,
        test_data,
        epochs=ae_epochs,
        batch_size=64,
        learning_rate=1e-3,
        device=resolved_device,
        seed=seed,
        density_field=field,
    )
    autoencoder_bundle = _pack_autoencoder(autoencoder)
    memory_sets = _sample_memory_sets(
        field,
        evaluation_repetitions,
        patterns,
        seed + 101,
    )
    holdout_data = _sample_memory_sets(field, 1, patterns, seed + 202)[0]

    num_parameters = len(MINI_PARAMETER_NAMES)
    population_size = 4 + int(3 * np.log(num_parameters))
    settings = {
        "num_parameters": num_parameters,
        "generations": generations,
        "population_size": population_size,
        "pause_time": pause,
        "direction": "minimize",
        "metric_name": "reconstruction_mse",
        "workers": workers,
        "verbose": verbose,
        "disable": False,
    }
    individual_evaluator = functools.partial(
        evaluate_mini_individual,
        autoencoder_bundle=autoencoder_bundle,
        memory_sets=memory_sets,
        plasticity=plasticity,
    )
    batch_evaluator = functools.partial(
        evaluate_mini_population,
        autoencoder_bundle=autoencoder_bundle,
        memory_sets=memory_sets,
        plasticity=plasticity,
        disable=True,
    )
    record = _lib.evolution_run(
        settings=settings,
        evaluate=batch_evaluator,
        evaluate_individual=individual_evaluator,
        sanitizer=sanitizer,
        live_plot=live_plot,
    )

    best_candidate = np.asarray(record["best_candidates"][-1], dtype=float)
    baseline_predictions = predict_memory_sequence(
        MINI_PARAMETER_CENTERS,
        autoencoder_bundle,
        holdout_data,
        plasticity,
    )
    best_predictions = predict_memory_sequence(
        best_candidate,
        autoencoder_bundle,
        holdout_data,
        plasticity,
    )
    baseline_mse = float(np.mean((baseline_predictions - holdout_data) ** 2))
    holdout_mse = float(np.mean((best_predictions - holdout_data) ** 2))
    record["parameter_names"] = list(MINI_PARAMETER_NAMES)
    record["best_parameters"] = dict(zip(
        MINI_PARAMETER_NAMES,
        best_candidate.tolist(),
    ))
    record["plasticity"] = plasticity
    record["data_kind"] = field["kind"]
    record["ae_validation_mse"] = ae_history["test_loss"][-1]
    record["baseline_holdout_mse"] = baseline_mse
    record["holdout_mse"] = holdout_mse

    if verbose:
        print(f"best_parameters={record['best_parameters']}")
        print(
            f"holdout_reconstruction_mse={holdout_mse:.6f}  "
            f"baseline={baseline_mse:.6f}"
        )

    if save:
        logs = {
            "date": time.strftime("%d.%m.%Y"),
            "settings": settings,
            "best": best_candidate.tolist(),
            "best_parameters": record["best_parameters"],
            "fitness": record["best_fitness"][-1],
            "baseline_holdout_mse": baseline_mse,
            "holdout_mse": holdout_mse,
            "plasticity": plasticity,
            "data_kind": field["kind"],
            "workers": record["workers"],
        }
        _lib.save_genome(logs, save_name)

    if success_plot:
        tuning_fields = evaluate_tuning_fields(
            best_candidate,
            autoencoder_bundle,
            holdout_data,
            plasticity,
            resolution=tuning_resolution,
        )
        plot_search_success(
            record,
            field,
            holdout_data,
            baseline_predictions,
            best_predictions,
            tuning_fields,
        )
    return record


def main():
    args = parse_args()
    mini_search(
        generations=args.generations,
        patterns=args.patterns,
        evaluation_repetitions=args.evaluation_repetitions,
        data_kind=args.data_kind,
        field_width=args.field_width,
        plasticity=args.plasticity,
        ae_training_size=args.ae_training_size,
        ae_test_size=args.ae_test_size,
        ae_epochs=args.ae_epochs,
        seed=args.seed,
        pause=args.pause,
        tuning_resolution=args.tuning_resolution,
        workers=args.workers,
        device=args.device,
        live_plot=not args.no_live_plot,
        success_plot=not args.no_success_plot,
        save=args.save,
        save_name=args.save_name,
    )
    print("[done]")


if __name__ == "__main__":
    main()
