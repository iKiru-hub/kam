"""Validate a crossed cue-count artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.paper.runner import _array_digest, _sha256
from experiments.paper.seeds import DEVELOPMENT_SEEDS, FINAL_SEEDS


def validate(artifact_dir: Path) -> dict[str, bool]:
    artifact_dir = artifact_dir.resolve()
    config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    report = json.loads((artifact_dir / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    with np.load(artifact_dir / "arrays.npz", allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    expected_seeds = DEVELOPMENT_SEEDS if report["split"] == "development" else FINAL_SEEDS
    conditions = arrays["condition_names"].tolist()
    fixed = conditions.index("fixed_permutation")
    rescue = conditions.index("matched_decoder_rescue")
    checkpoints_match = all(
        _sha256(artifact_dir / "autoencoders" / f"seed_{seed}.pt")
        == manifest["checkpoint_hashes"][str(seed)]
        for seed in expected_seeds
    )
    seed_quality_valid = True
    diagnostics_valid = True
    restart_selection_valid = True
    for seed_report in report["seed_reports"].values():
        quality = seed_report["autoencoder_quality"]
        seed_quality_valid = seed_quality_valid and bool(quality["pass"])
        best = min(quality["candidates"], key=lambda item: item["mse"])
        restart_selection_valid = restart_selection_valid and (
            best["restart"] == quality["selected_restart"]
        )
        diagnostics_valid = diagnostics_valid and all(
            item["all_checks_pass"] for item in seed_report["diagnostics"].values()
        )
    with (artifact_dir / "source_data.csv").open(encoding="utf-8") as handle:
        source_rows = sum(1 for _ in handle) - 1
    expected_rows = len(expected_seeds) * len(conditions) * sum(
        2 * count for count in arrays["cue_counts"]
    )
    padding_valid = True
    for count_index, count in enumerate(arrays["cue_counts"]):
        item_count = 2 * int(count)
        raw = arrays["metric_raw_cosine"][:, count_index]
        padding_valid = padding_valid and bool(
            np.isfinite(raw[:, :, :item_count]).all()
            and np.isnan(raw[:, :, item_count:]).all()
        )
    return {
        "scientific_digest_matches_report": bool(
            _array_digest(arrays) == report["scientific_digest"]
        ),
        "scientific_digest_matches_manifest": bool(
            _array_digest(arrays) == manifest["scientific_digest"]
        ),
        "seed_set_exact": bool(
            tuple(arrays["root_seeds"].tolist()) == expected_seeds
        ),
        "cue_counts_exact": bool(
            arrays["cue_counts"].tolist() == config["cue_counts"]
        ),
        "all_autoencoder_gates_pass": seed_quality_valid,
        "lowest_mse_restart_selected": restart_selection_valid,
        "all_causal_invariants_pass": diagnostics_valid,
        "fixed_rescue_weights_equal": bool(
            np.array_equal(
                arrays["final_weights"][:, :, fixed],
                arrays["final_weights"][:, :, rescue],
            )
        ),
        "checkpoint_hashes_match": bool(checkpoints_match),
        "padding_valid": bool(padding_valid),
        "source_rows_complete": bool(source_rows == expected_rows),
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
