"""Paired fixed-parameter comparison of base and error-driven plasticity."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np

from experiments.preprint.artifacts import create_artifact
from experiments.preprint.completion import corrupt
from experiments.preprint.config import read_config
from experiments.preprint.metrics import output_metrics
from experiments.preprint.model_factory import build_mtl, run_mtl, train_autoencoder
from experiments.preprint.seeds import SeedStreams
from experiments.preprint.stimuli import cue_track


def run_seed(config: dict, root_seed: int) -> tuple[dict[str, np.ndarray], list[dict], dict]:
    streams = SeedStreams(root_seed)
    track = config["track"]
    ablation = config["plasticity_ablation"]
    clean = cue_track(track["training_laps"], track, [[0, 1]] * track["training_laps"], streams.integer("track"))
    validation = cue_track(track["validation_laps"], track, [[0, 1]] * track["validation_laps"], streams.integer("ae_valid"))
    autoencoder, quality = train_autoencoder(clean.reshape(-1, track["size"]), validation.reshape(-1, track["size"]), config, streams.integer("ae_init"), streams.integer("ae_batches"))
    rules = ablation["rules"]
    fractions = ablation["fractions"]
    masks = int(ablation["masks_per_fraction"])
    shape = (len(rules), len(fractions), masks)
    result = {name: np.full(shape, np.nan, dtype=np.float64) for name in ("output_cosine", "identity")}
    rows = []
    target = clean[-1]
    for rule_index, rule in enumerate(rules):
        memory = deepcopy(config["memory"])
        memory["plasticity_rule"] = rule
        model = build_mtl(autoencoder, memory, streams.integer("ca3_wiring"))
        for lap in clean:
            run_mtl(model, lap, learn=True)
        for fraction_index, fraction in enumerate(fractions):
            for mask_index in range(masks):
                mask_rng = np.random.default_rng(np.random.SeedSequence([root_seed, rule_index, fraction_index, mask_index]))
                probe, mask = corrupt(target, float(fraction), mask_rng, bool(ablation["lec_only"]))
                output, _, _ = run_mtl(model, probe, learn=False)
                metrics = output_metrics(output, target, config["data"]["active"])
                result["output_cosine"][rule_index, fraction_index, mask_index] = metrics["cosine"].mean()
                result["identity"][rule_index, fraction_index, mask_index] = metrics["identity"].mean()
                rows.append({"root_seed": root_seed, "rule": rule, "fraction": fraction, "mask_index": mask_index, "output_cosine": float(result["output_cosine"][rule_index, fraction_index, mask_index]), "identity": float(result["identity"][rule_index, fraction_index, mask_index]), "dropped_units": int(len(mask))})
    return result, rows, quality


def run(config: dict, output: Path) -> Path:
    results, rows, quality = [], [], []
    for position, root_seed in enumerate(config["root_seeds"], start=1):
        print(f"plasticity ablation seed {root_seed} ({position}/{len(config['root_seeds'])})", flush=True)
        result, seed_rows, seed_quality = run_seed(config, int(root_seed))
        results.append(result)
        rows.extend(seed_rows)
        quality.append(seed_quality)
    arrays = {name: np.stack([result[name] for result in results]) for name in results[0]}
    arrays["root_seeds"] = np.asarray(config["root_seeds"], dtype=np.int64)
    arrays["rules"] = np.asarray(config["plasticity_ablation"]["rules"])
    arrays["fractions"] = np.asarray(config["plasticity_ablation"]["fractions"], dtype=np.float64)
    report = {"experiment": "plasticity_ablation", "comparison": "all parameters fixed except plasticity_rule", "autoencoder_quality": quality}
    return create_artifact(output, config, arrays, rows, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(run(read_config(args.config), args.output))


if __name__ == "__main__":
    main()
