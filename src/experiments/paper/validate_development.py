"""Independently validate and recompute a development-sweep artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.paper.development import select_development_configuration
from experiments.paper.runner import _array_digest, _sha256
from experiments.paper.seeds import DEVELOPMENT_SEEDS, FINAL_SEEDS


def validate(artifact_dir: Path) -> dict[str, bool]:
    artifact_dir = artifact_dir.resolve()
    config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    report = json.loads((artifact_dir / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    with np.load(artifact_dir / "arrays.npz", allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}

    recomputed = select_development_configuration(
        arrays["metric_raw_cosine"],
        arrays["rule_names"].tolist(),
        arrays["alphas"],
        arrays["condition_names"].tolist(),
        config["development"],
    )
    names = arrays["condition_names"].tolist()
    fixed = names.index("fixed_permutation")
    rescue = names.index("matched_decoder_rescue")
    checkpoint_hashes = manifest["checkpoint_hashes"]
    checkpoints_match = all(
        _sha256(artifact_dir / "autoencoders" / f"seed_{seed}.pt")
        == checkpoint_hashes[str(seed)]
        for seed in DEVELOPMENT_SEEDS
    )
    with (artifact_dir / "source_data.csv").open(encoding="utf-8") as handle:
        source_rows = sum(1 for _ in handle) - 1
    expected_rows = (
        len(arrays["rule_names"])
        * len(arrays["alphas"])
        * len(arrays["root_seeds"])
        * len(arrays["condition_names"])
        * arrays["inputs"].shape[1]
    )
    return {
        "scientific_digest_matches_report": _array_digest(arrays)
        == report["scientific_digest"],
        "scientific_digest_matches_manifest": _array_digest(arrays)
        == manifest["scientific_digest"],
        "selection_recomputes_exactly": recomputed == report["selection"],
        "development_seeds_exact": tuple(arrays["root_seeds"].tolist())
        == DEVELOPMENT_SEEDS,
        "no_final_seed_access": not bool(set(arrays["root_seeds"].tolist()) & set(FINAL_SEEDS)),
        "all_invariants_pass": bool(report["all_invariants_pass"]),
        "all_ae_quality_gates_pass": all(
            item["mse"] <= config["autoencoder"]["quality_mse_max"]
            and item["mean_cosine"] >= config["autoencoder"]["quality_cosine_min"]
            for item in report["autoencoder_quality"].values()
        ),
        "fixed_rescue_weights_equal": bool(
            np.array_equal(
                arrays["final_weights"][:, :, :, fixed],
                arrays["final_weights"][:, :, :, rescue],
            )
        ),
        "checkpoint_hashes_match": checkpoints_match,
        "source_rows_complete": source_rows == expected_rows,
        "step3_pass": bool(recomputed["step3_pass"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    args = parser.parse_args()
    checks = validate(args.artifact)
    print(json.dumps(checks, indent=2, sort_keys=True))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

