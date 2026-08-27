"""Validate a saved paper artifact without rerunning its simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.paper.metrics import metric_sanity_checks
from experiments.paper.seeds import assert_seed_sets_disjoint


def _array_digest(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def validate(artifact_dir: Path) -> dict[str, bool]:
    artifact_dir = artifact_dir.resolve()
    report = json.loads((artifact_dir / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    with np.load(artifact_dir / "arrays.npz", allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    assert_seed_sets_disjoint()
    names = arrays["condition_names"].tolist()
    fixed = names.index("fixed_permutation")
    rescue = names.index("matched_decoder_rescue")
    with (artifact_dir / "source_data.csv").open(encoding="utf-8") as handle:
        source_row_count = sum(1 for _ in handle)
    checks = {
        "scientific_digest_matches_report": _array_digest(arrays)
        == report["scientific_digest"],
        "scientific_digest_matches_manifest": _array_digest(arrays)
        == manifest["scientific_digest"],
        "report_checks_pass": bool(report["all_checks_pass"]),
        "fixed_rescue_weights_equal": bool(
            np.array_equal(arrays["final_weights"][fixed], arrays["final_weights"][rescue])
        ),
        "fixed_rescue_ca1_equal": bool(
            np.array_equal(arrays["recalled_ca1"][fixed], arrays["recalled_ca1"][rescue])
        ),
        "source_rows_complete": source_row_count
        == len(names) * len(arrays["inputs"]) + 1,
        **metric_sanity_checks(),
    }
    return checks


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
