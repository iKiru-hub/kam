"""Frozen-recall pattern completion with CA3 key-map ablations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.preprint.artifacts import create_artifact
from experiments.preprint.config import read_config
from experiments.preprint.metrics import output_metrics, row_cosine
from experiments.preprint.model_factory import build_mtl, run_mtl, train_autoencoder
from experiments.preprint.seeds import SeedStreams
from experiments.preprint.stimuli import cue_track


def corrupt(values: np.ndarray, fraction: float, rng: np.random.Generator, lec_only: bool) -> tuple[np.ndarray, np.ndarray]:
    result = values.copy()
    start = values.shape[1] // 2 if lec_only else 0
    eligible = np.arange(start, values.shape[1])
    count = int(round(fraction * len(eligible)))
    mask = np.sort(rng.choice(eligible, size=count, replace=False)) if count else np.empty(0, dtype=np.int64)
    result[:, mask] = 0.0
    return result, mask


def _set_key_mode(model, mode: str, rng: np.random.Generator) -> None:
    """Replace the fixed EC-to-CA3 key map while keeping its dimensions fixed."""

    dimension = model._dim_ei
    if mode == "normal":
        return
    if mode == "identity":
        value = torch.eye(dimension) / dimension
    elif mode == "dense":
        value = torch.full((dimension, dimension), 1.0 / dimension)
    elif mode == "shuffled":
        value = torch.zeros((dimension, dimension))
        degree = int(model._nb_ei_ca3)
        for row in range(dimension):
            value[row, rng.choice(dimension, size=degree, replace=False)] = 1.0 / dimension
    else:
        raise ValueError(f"Unknown key mode: {mode}")
    model.W_ei_ca3 = torch.nn.Parameter(value)


def run_seed(config: dict, root_seed: int) -> tuple[dict[str, np.ndarray], list[dict], dict]:
    streams = SeedStreams(root_seed)
    track = config["track"]
    completion = config["completion"]
    clean = cue_track(track["training_laps"], track, [[0, 1]] * track["training_laps"], streams.integer("track"))
    validation = cue_track(track["validation_laps"], track, [[0, 1]] * track["validation_laps"], streams.integer("ae_valid"))
    autoencoder, quality = train_autoencoder(clean.reshape(-1, track["size"]), validation.reshape(-1, track["size"]), config, streams.integer("ae_init"), streams.integer("ae_batches"))
    target = clean[-1]
    modes = completion["key_modes"]
    fractions = completion["fractions"]
    masks = int(completion["masks_per_fraction"])
    shape = (len(modes), len(fractions), masks)
    results = {name: np.full(shape, np.nan, dtype=np.float64) for name in ("output_cosine", "identity", "key_cosine")}
    rows = []
    for mode_index, mode in enumerate(modes):
        model = build_mtl(autoencoder, config["memory"], streams.integer("ca3_wiring") + mode_index)
        _set_key_mode(model, mode, streams.numpy("ca3_wiring"))
        for lap in clean:
            run_mtl(model, lap, learn=True)
        _, clean_ca1, clean_key = run_mtl(model, target, learn=False)
        for fraction_index, fraction in enumerate(fractions):
            for mask_index in range(masks):
                mask_rng = np.random.default_rng(np.random.SeedSequence([root_seed, mode_index, fraction_index, mask_index]))
                probe, mask = corrupt(target, float(fraction), mask_rng, bool(completion["lec_only"]))
                output, _, key = run_mtl(model, probe, learn=False)
                metrics = output_metrics(output, target, config["data"]["active"])
                results["output_cosine"][mode_index, fraction_index, mask_index] = metrics["cosine"].mean()
                results["identity"][mode_index, fraction_index, mask_index] = metrics["identity"].mean()
                results["key_cosine"][mode_index, fraction_index, mask_index] = row_cosine(clean_key, key).mean()
                rows.append({"root_seed": root_seed, "key_mode": mode, "fraction": fraction, "mask_index": mask_index, "output_cosine": float(results["output_cosine"][mode_index, fraction_index, mask_index]), "identity": float(results["identity"][mode_index, fraction_index, mask_index]), "key_cosine": float(results["key_cosine"][mode_index, fraction_index, mask_index]), "dropped_units": int(len(mask))})
    return results, rows, quality


def run(config: dict, output: Path) -> Path:
    results, rows, quality = [], [], []
    for position, root_seed in enumerate(config["root_seeds"], start=1):
        print(f"completion seed {root_seed} ({position}/{len(config['root_seeds'])})", flush=True)
        result, seed_rows, seed_quality = run_seed(config, int(root_seed))
        results.append(result)
        rows.extend(seed_rows)
        quality.append(seed_quality)
    arrays = {name: np.stack([result[name] for result in results]) for name in results[0]}
    arrays["root_seeds"] = np.asarray(config["root_seeds"], dtype=np.int64)
    arrays["key_modes"] = np.asarray(config["completion"]["key_modes"])
    arrays["fractions"] = np.asarray(config["completion"]["fractions"], dtype=np.float64)
    report = {"experiment": "completion", "autoencoder_quality": quality}
    return create_artifact(output, config, arrays, rows, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(run(read_config(args.config), args.output))


if __name__ == "__main__":
    main()
