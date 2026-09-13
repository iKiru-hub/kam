"""Test cue-conditioned recall after selective LEC input degradation.

The model is trained on clean cue track laps and plasticity is then frozen.
Each probe zeros a fixed random subset of LEC input units for the entire lap.
The experiment reports cue identity at the two cue positions, retained MEC
spatial reconstruction, and CA3 key stability for each configured key map.

Example:
    PYTHONPATH=src python3 -m experiments.mtl_cue_degr_experiment \
      --output results/mtl_cue_degradation/v1 --no-show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from core import datagen
from experiments.preprint.artifacts import create_artifact, load_arrays
from experiments.preprint.completion import corrupt
from experiments.preprint.config import read_config
from experiments.preprint.figures.common import mean_sem
from experiments.preprint.metrics import row_cosine
from experiments.preprint.model_factory import build_mtl, run_mtl, train_autoencoder
from experiments.preprint.seeds import SeedStreams
from experiments.preprint.stimuli import cue_track


DEFAULT_CONFIG = Path("src/experiments/preprint/configs/final_completion.json")
DEFAULT_OUTPUT = Path("results/mtl_cue_degradation/v1")


def set_key_mode(model, mode: str, rng: np.random.Generator) -> None:
    """Apply the same EC-to-CA3 key-map controls as the completion study."""

    dimension = model._dim_ei
    if mode == "normal":
        return
    if mode == "identity":
        weights = torch.eye(dimension) / dimension
    elif mode == "dense":
        weights = torch.full((dimension, dimension), 1.0 / dimension)
    elif mode == "shuffled":
        weights = torch.zeros((dimension, dimension))
        for row in range(dimension):
            indices = rng.choice(dimension, size=int(model._nb_ei_ca3), replace=False)
            weights[row, indices] = 1.0 / dimension
    else:
        raise ValueError(f"Unknown key mode: {mode}")
    model.W_ei_ca3 = torch.nn.Parameter(weights)


def cue_accuracy(output: np.ndarray, cue_patterns: np.ndarray, cue_positions: list[int]) -> float:
    """Classify the two recalled LEC cue traces against the known templates."""

    lec = output[np.asarray(cue_positions), output.shape[1] // 2:]
    denominator = np.maximum(np.linalg.norm(lec, axis=1, keepdims=True) * np.linalg.norm(cue_patterns, axis=1)[None, :], 1e-12)
    predicted = np.argmax((lec @ cue_patterns.T) / denominator, axis=1)
    return float(np.mean(predicted == np.arange(len(cue_positions))))


def run_seed(config: dict, root_seed: int) -> tuple[dict[str, np.ndarray], list[dict], dict]:
    streams = SeedStreams(root_seed)
    track = config["track"]
    completion = config["completion"]
    clean = cue_track(track["training_laps"], track, [[0, 1]] * track["training_laps"], streams.integer("track"))
    validation = cue_track(track["validation_laps"], track, [[0, 1]] * track["validation_laps"], streams.integer("ae_valid"))
    autoencoder, quality = train_autoencoder(clean.reshape(-1, track["size"]), validation.reshape(-1, track["size"]), config, streams.integer("ae_init"), streams.integer("ae_batches"))
    target = clean[-1]
    cue_patterns = datagen.make_cues(2, target.shape[1] // 2, fixed=True)
    modes = completion["key_modes"]
    fractions = completion["fractions"]
    masks_per_fraction = int(completion["masks_per_fraction"])
    shape = (len(modes), len(fractions), masks_per_fraction)
    result = {name: np.full(shape, np.nan, dtype=np.float64) for name in ("cue_accuracy", "mec_cosine", "output_cosine", "key_cosine")}
    rows = []
    example_index = int(np.argmin(np.abs(np.asarray(fractions, dtype=float) - 0.5)))
    examples = {"example_target": target, "example_probe": None, "example_output": []}
    for mode_index, mode in enumerate(modes):
        model = build_mtl(autoencoder, config["memory"], streams.integer("ca3_wiring") + mode_index)
        set_key_mode(model, mode, streams.numpy("ca3_wiring"))
        for lap in clean:
            run_mtl(model, lap, learn=True)
        _, _, clean_key = run_mtl(model, target, learn=False)
        for fraction_index, fraction in enumerate(fractions):
            for mask_index in range(masks_per_fraction):
                mask_rng = np.random.default_rng(np.random.SeedSequence([root_seed, mode_index, fraction_index, mask_index]))
                probe, mask = corrupt(target, float(fraction), mask_rng, lec_only=True)
                output, _, key = run_mtl(model, probe, learn=False)
                values = {
                    "cue_accuracy": cue_accuracy(output, cue_patterns, track["cue_positions"]),
                    "mec_cosine": float(row_cosine(output[:, :output.shape[1] // 2], target[:, :target.shape[1] // 2]).mean()),
                    "output_cosine": float(row_cosine(output, target).mean()),
                    "key_cosine": float(row_cosine(clean_key, key).mean()),
                }
                for name, value in values.items():
                    result[name][mode_index, fraction_index, mask_index] = value
                rows.append({"root_seed": root_seed, "key_mode": mode, "fraction": float(fraction), "mask_index": mask_index, "dropped_lec_units": int(len(mask)), **values})
                if fraction_index == example_index and mask_index == 0:
                    if examples["example_probe"] is None:
                        examples["example_probe"] = probe
                    examples["example_output"].append(output)
        if len(examples["example_output"]) <= mode_index:
            raise RuntimeError("Missing representative cue-degradation output")
    result.update({
        "example_target": examples["example_target"],
        "example_probe": examples["example_probe"],
        "example_output": np.stack(examples["example_output"]),
    })
    return result, rows, quality


def run(config: dict, output: Path) -> Path:
    results, rows, quality = [], [], []
    for position, root_seed in enumerate(config["root_seeds"], start=1):
        print(f"cue degradation seed {root_seed} ({position}/{len(config['root_seeds'])})", flush=True)
        result, seed_rows, seed_quality = run_seed(config, int(root_seed))
        results.append(result)
        rows.extend(seed_rows)
        quality.append(seed_quality)
    arrays = {name: np.stack([result[name] for result in results]) for name in results[0]}
    arrays["root_seeds"] = np.asarray(config["root_seeds"], dtype=np.int64)
    arrays["key_modes"] = np.asarray(config["completion"]["key_modes"])
    arrays["fractions"] = np.asarray(config["completion"]["fractions"], dtype=np.float64)
    report = {
        "experiment": "mtl_cue_degradation",
        "protocol": "clean training followed by frozen recall with LEC-only dropout",
        "autoencoder_quality": quality,
    }
    return create_artifact(output, config, arrays, rows, report)


def build_figure(artifact: Path, output: Path) -> Path:
    """Build summary curves and representative input/reconstruction panels."""

    arrays = load_arrays(artifact)
    modes = arrays["key_modes"].tolist()
    fractions = arrays["fractions"]
    figure = plt.figure(figsize=(13, 7), constrained_layout=True)
    grid = figure.add_gridspec(2, 3)
    curve_axes = [figure.add_subplot(grid[0, column]) for column in range(3)]
    for mode_index, mode in enumerate(modes):
        for axis, metric, label in zip(curve_axes, ("cue_accuracy", "mec_cosine", "key_cosine"), ("Cue identity accuracy", "MEC output-target cosine", "Clean-corrupted CA3 cosine")):
            values = arrays[metric][:, mode_index].mean(axis=-1)
            mean, sem = mean_sem(values)
            axis.plot(fractions, mean, marker="o", label=mode)
            axis.fill_between(fractions, mean - sem, mean + sem, alpha=0.15)
            axis.set(xlabel="Dropped LEC fraction", ylabel=label, ylim=(-0.04, 1.04))
    curve_axes[-1].legend(frameon=False, fontsize=8)
    for axis in curve_axes:
        axis.spines[["top", "right"]].set_visible(False)

    displayed = ["example_target", "example_probe"]
    displayed.extend(["example_output"] * min(1, len(modes)))
    images = []
    labels = ["Clean target", "LEC-degraded probe", f"Recall: {modes[0]} key"]
    for index, (key, label) in enumerate(zip(displayed, labels)):
        axis = figure.add_subplot(grid[1, index])
        values = arrays[key][0] if key != "example_output" else arrays[key][0, 0]
        image = axis.imshow(values.T, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="viridis")
        axis.axhline(values.shape[1] / 2 - 0.5, color="white", linestyle="--", linewidth=0.8)
        axis.set(title=label, xlabel="Track position", ylabel="EC output unit")
        images.append(image)
    figure.colorbar(images[-1], ax=figure.axes[-3:], label="activity", fraction=0.03)
    figure.suptitle("Cue-conditioned recall under selective LEC degradation")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300)
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=None)
    parser.add_argument("--limit-seeds", type=int, default=None, help="run only the first N configured seeds")
    parser.add_argument("--no-show", action="store_true", help="accepted for compatibility; figures are saved without opening a window")
    args = parser.parse_args()
    config = read_config(args.config)
    if args.limit_seeds is not None:
        if args.limit_seeds < 1:
            raise ValueError("--limit-seeds must be positive")
        config["root_seeds"] = config["root_seeds"][:args.limit_seeds]
    artifact = run(config, args.output)
    figure = args.figure if args.figure is not None else artifact / "summary.png"
    print(build_figure(artifact, figure))


if __name__ == "__main__":
    main()
