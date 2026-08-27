"""Validate the frozen E2 retention artifact and Gate B1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.paper.e2_retention import evaluate_gate_b1
from experiments.paper.runner import _array_digest, _sha256
from experiments.paper.seeds import FINAL_SEEDS


def validate(artifact_dir: Path) -> dict[str, bool]:
    artifact_dir = artifact_dir.resolve()
    config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    report = json.loads((artifact_dir / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    with np.load(artifact_dir / "arrays.npz", allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    conditions = arrays["condition_names"].tolist()
    load_index = arrays["evaluation_loads"].tolist().index(config["endpoint"]["load"])
    oldest = config["endpoint"]["oldest_count"]
    endpoint = arrays["metric_raw_cosine"][:, :, load_index, :oldest].mean(axis=-1)
    recomputed = evaluate_gate_b1(endpoint, conditions, config["inference"])
    fixed = conditions.index("fixed_permutation")
    rescue = conditions.index("matched_decoder_rescue")
    repo_root = Path(__file__).resolve().parents[3]
    e1_root = (repo_root / config["e1_artifact"]).resolve()
    checkpoint_hashes_match = all(
        _sha256(e1_root / "seeds" / str(seed) / "autoencoder.pt")
        == manifest["reused_e1_checkpoint_hashes"][str(seed)]
        for seed in FINAL_SEEDS
    )
    with (artifact_dir / "source_data.csv").open(encoding="utf-8") as handle:
        source_rows = sum(1 for _ in handle) - 1
    expected_rows = (
        len(FINAL_SEEDS)
        * len(conditions)
        * sum(config["data"]["evaluation_loads"])
    )
    padding_valid = True
    raw = arrays["metric_raw_cosine"]
    for load_index, load in enumerate(arrays["evaluation_loads"]):
        padding_valid = padding_valid and bool(
            np.isfinite(raw[:, :, load_index, : int(load)]).all()
            and np.isnan(raw[:, :, load_index, int(load) :]).all()
        )
    return {
        "scientific_digest_matches_report": _array_digest(arrays)
        == report["scientific_digest"],
        "scientific_digest_matches_manifest": _array_digest(arrays)
        == manifest["scientific_digest"],
        "gate_b1_recomputes_exactly": recomputed == report["gate_b1"],
        "gate_b1_pass": bool(recomputed["gate_b1_pass"]),
        "final_seeds_exact": tuple(arrays["root_seeds"].tolist()) == FINAL_SEEDS,
        "evaluation_loads_exact": arrays["evaluation_loads"].tolist()
        == config["data"]["evaluation_loads"],
        "all_invariants_pass": bool(report["all_invariants_pass"]),
        "fixed_rescue_weights_equal": bool(
            np.array_equal(
                arrays["weight_snapshots"][:, fixed],
                arrays["weight_snapshots"][:, rescue],
            )
        ),
        "fixed_rescue_ca1_equal": bool(
            np.array_equal(
                arrays["recalled_ca1"][:, fixed],
                arrays["recalled_ca1"][:, rescue],
                equal_nan=True,
            )
        ),
        "reused_checkpoint_hashes_match": checkpoint_hashes_match,
        "triangular_padding_valid": padding_valid,
        "source_rows_complete": source_rows == expected_rows,
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

