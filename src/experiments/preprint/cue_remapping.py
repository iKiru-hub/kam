"""CA1 tuning and context sensitivity during a predeclared cue-swap schedule."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.preprint.artifacts import create_artifact
from experiments.preprint.config import read_config
from experiments.preprint.model_factory import build_mtl, run_mtl, train_autoencoder
from experiments.preprint.seeds import SeedStreams
from experiments.preprint.stimuli import alternating_assignments, cue_track


def _unit_metrics(fields: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return spatial stability and cue modulation for every CA1 unit."""

    first, second = fields
    first_centered = first - first.mean(axis=0)
    second_centered = second - second.mean(axis=0)
    stability = np.sum(first_centered * second_centered, axis=0)
    stability /= np.maximum(np.linalg.norm(first_centered, axis=0) * np.linalg.norm(second_centered, axis=0), 1e-12)
    modulation = np.mean(np.abs(first - second), axis=0)
    return stability, modulation


def run_seed(config: dict, root_seed: int) -> tuple[dict[str, np.ndarray], list[dict], dict]:
    streams = SeedStreams(root_seed)
    track = config["track"]
    training_assignments = alternating_assignments(track["training_laps"], track["swap_every"])
    training_laps = cue_track(track["training_laps"], track, training_assignments, streams.integer("track"))
    validation_laps = cue_track(track["validation_laps"], track, [[0, 1]] * track["validation_laps"], streams.integer("ae_valid"))
    autoencoder, quality = train_autoencoder(
        training_laps.reshape(-1, track["size"]), validation_laps.reshape(-1, track["size"]),
        config, streams.integer("ae_init"), streams.integer("ae_batches"),
    )
    model = build_mtl(autoencoder, config["memory"], streams.integer("ca3_wiring"))
    history = []
    for lap in training_laps:
        _, ca1, _ = run_mtl(model, lap, learn=True)
        history.append(ca1)
    context_a = cue_track(1, track, [[0, 1]], streams.integer("ae_train"))[0]
    context_b = cue_track(1, track, [[1, 0]], streams.integer("ae_train") + 1)[0]
    output_a, ca1_a, _ = run_mtl(model, context_a, learn=False)
    output_b, ca1_b, _ = run_mtl(model, context_b, learn=False)
    fields = np.stack([ca1_a, ca1_b])
    stability, modulation = _unit_metrics(fields)
    rows = [{"root_seed": root_seed, "unit": unit, "spatial_stability": float(stability[unit]), "cue_modulation": float(modulation[unit])} for unit in range(fields.shape[-1])]
    return {
        "training_ca1": np.stack(history), "probe_ca1": fields, "probe_output": np.stack([output_a, output_b]),
        "spatial_stability": stability, "cue_modulation": modulation,
        "cue_assignments": np.asarray(training_assignments, dtype=np.int64),
    }, rows, quality


def run(config: dict, output: Path) -> Path:
    results, rows, quality = [], [], []
    for position, root_seed in enumerate(config["root_seeds"], start=1):
        print(f"cue remapping seed {root_seed} ({position}/{len(config['root_seeds'])})", flush=True)
        result, seed_rows, seed_quality = run_seed(config, int(root_seed))
        results.append(result)
        rows.extend(seed_rows)
        quality.append(seed_quality)
    arrays = {name: np.stack([result[name] for result in results]) for name in results[0]}
    arrays["root_seeds"] = np.asarray(config["root_seeds"], dtype=np.int64)
    report = {"experiment": "cue_remapping", "autoencoder_quality": quality, "mean_spatial_stability": float(arrays["spatial_stability"].mean()), "mean_cue_modulation": float(arrays["cue_modulation"].mean())}
    return create_artifact(output, config, arrays, rows, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(run(read_config(args.config), args.output))


if __name__ == "__main__":
    main()
