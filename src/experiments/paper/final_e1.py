"""Run frozen E1 across all final seeds and apply Gate A."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from experiments.paper import PROTOCOL_VERSION
from experiments.paper.runner import (
    _array_digest,
    _git_metadata,
    _json_dump,
    _sha256,
    _validate_config,
    run as run_seed,
)
from experiments.paper.seeds import FINAL_SEEDS, assert_seed_sets_disjoint


AGGREGATE_KEYS = (
    "inputs",
    "storage_order",
    "coordinate_permutation",
    "inverse_permutation",
    "content_derangement",
    "ca3_weights",
    "instruction_codes",
    "condition_instructions",
    "recalled_ca1",
    "outputs",
    "final_weights",
    "target_ca1_cosine",
    "metric_raw_cosine",
    "metric_topk_overlap",
    "metric_identity_correct",
    "metric_chance_corrected_cosine",
    "metric_mse",
)


def _interval(values: np.ndarray, critical: float) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    se = sd / np.sqrt(len(values))
    half_width = critical * se
    return {
        "mean": mean,
        "sd": sd,
        "se": se,
        "interval": [mean - half_width, mean + half_width],
        "values": values.tolist(),
    }


def evaluate_gate_a(
    seed_condition_means: np.ndarray,
    condition_names: list[str],
    inference_cfg: dict[str, float],
) -> dict[str, Any]:
    if seed_condition_means.ndim != 2:
        raise ValueError("Expected seed × condition means")
    indices = {name: condition_names.index(name) for name in condition_names}
    aligned = seed_condition_means[:, indices["aligned"]]
    contrasts = {
        "aligned_minus_fixed": aligned
        - seed_condition_means[:, indices["fixed_permutation"]],
        "rescue_minus_fixed": seed_condition_means[
            :, indices["matched_decoder_rescue"]
        ]
        - seed_condition_means[:, indices["fixed_permutation"]],
        "rescue_minus_aligned": seed_condition_means[
            :, indices["matched_decoder_rescue"]
        ]
        - aligned,
        "aligned_minus_random": aligned
        - seed_condition_means[:, indices["random_content_matched"]],
        "aligned_minus_no_plasticity": aligned
        - seed_condition_means[:, indices["no_plasticity"]],
    }
    superiority_names = (
        "aligned_minus_fixed",
        "rescue_minus_fixed",
        "aligned_minus_random",
        "aligned_minus_no_plasticity",
    )
    summaries = {
        name: _interval(
            values,
            inference_cfg[
                "equivalence_t_critical_df19"
                if name == "rescue_minus_aligned"
                else "superiority_t_critical_df19"
            ],
        )
        for name, values in contrasts.items()
    }
    margin = inference_cfg["rescue_equivalence_margin"]
    checks = {
        f"{name}_ci_above_zero": bool(summaries[name]["interval"][0] > 0.0)
        for name in superiority_names
    }
    checks["aligned_minus_fixed_practical_effect"] = bool(
        summaries["aligned_minus_fixed"]["mean"]
        >= inference_cfg["minimum_alignment_effect"]
    )
    equivalence_interval = summaries["rescue_minus_aligned"]["interval"]
    checks["rescue_equivalent_to_aligned"] = bool(
        equivalence_interval[0] >= -margin and equivalence_interval[1] <= margin
    )
    checks["complete_final_sample"] = len(seed_condition_means) == len(FINAL_SEEDS)
    return {
        "contrasts": summaries,
        "checks": checks,
        "gate_a_pass": bool(all(checks.values())),
    }


def _combine_source_tables(seed_dirs: list[Path], output_path: Path) -> None:
    wrote_header = False
    with output_path.open("w", encoding="utf-8", newline="") as output_handle:
        writer = None
        for seed_dir in seed_dirs:
            with (seed_dir / "source_data.csv").open(encoding="utf-8", newline="") as input_handle:
                reader = csv.DictReader(input_handle)
                if writer is None:
                    writer = csv.DictWriter(output_handle, fieldnames=reader.fieldnames)
                if not wrote_header:
                    writer.writeheader()
                    wrote_header = True
                writer.writerows(reader)


def run(
    config_path: Path, output_dir: Path, resume_existing: bool = False
) -> dict[str, Any]:
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    assert_seed_sets_disjoint()
    seeds = tuple(int(seed) for seed in config["root_seeds"])
    if seeds != FINAL_SEEDS:
        raise ValueError("Final E1 must use the complete frozen final seed set")
    if output_dir.exists() and any(output_dir.iterdir()) and not resume_existing:
        raise FileExistsError(f"Refusing to overwrite nonempty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_root = output_dir / "seeds"
    seed_root.mkdir(exist_ok=resume_existing)
    config_root = output_dir / "resolved_seed_configs"
    config_root.mkdir(exist_ok=resume_existing)

    seed_arrays: list[dict[str, np.ndarray]] = []
    seed_reports: dict[str, Any] = {}
    seed_dirs: list[Path] = []
    for position, seed in enumerate(seeds, start=1):
        print(f"final E1 seed {seed} ({position}/{len(seeds)})", flush=True)
        seed_config_path = config_root / f"seed_{seed}.json"
        seed_dir = seed_root / str(seed)
        if resume_existing:
            if not seed_config_path.exists() or not (seed_dir / "report.json").exists():
                raise FileNotFoundError(f"Incomplete frozen artifact for final seed {seed}")
            seed_report = json.loads((seed_dir / "report.json").read_text(encoding="utf-8"))
        else:
            seed_config = json.loads(json.dumps(config))
            seed_config.pop("root_seeds", None)
            seed_config.pop("inference", None)
            seed_config["root_seed"] = seed
            seed_config["experiment"] = "e1_final_seed"
            _json_dump(seed_config_path, seed_config)
            seed_report = run_seed(seed_config_path, seed_dir)
        if not seed_report["all_checks_pass"]:
            raise RuntimeError(f"Invariant failure for final seed {seed}")
        with np.load(seed_dir / "arrays.npz", allow_pickle=False) as loaded:
            seed_arrays.append({key: loaded[key] for key in AGGREGATE_KEYS})
            condition_names = loaded["condition_names"].copy()
        seed_reports[str(seed)] = {
            "scientific_digest": seed_report["scientific_digest"],
            "autoencoder_quality": seed_report["autoencoder_quality"],
            "checks": seed_report["checks"],
        }
        seed_dirs.append(seed_dir)

    arrays: dict[str, np.ndarray] = {
        "root_seeds": np.asarray(seeds, dtype=np.int64),
        "condition_names": condition_names,
    }
    for key in AGGREGATE_KEYS:
        arrays[key] = np.stack([item[key] for item in seed_arrays])
    seed_condition_means = arrays["metric_raw_cosine"].mean(axis=-1)
    gate_a = evaluate_gate_a(
        seed_condition_means,
        condition_names.tolist(),
        config["inference"],
    )
    condition_summaries = {
        condition: _interval(
            seed_condition_means[:, index],
            config["inference"]["superiority_t_critical_df19"],
        )
        for index, condition in enumerate(condition_names.tolist())
    }
    scientific_digest = _array_digest(arrays)
    arrays_path = output_dir / "arrays.npz"
    config_out = output_dir / "config.json"
    source_path = output_dir / "source_data.csv"
    report_path = output_dir / "report.json"
    manifest_path = output_dir / "manifest.json"
    np.savez_compressed(arrays_path, **arrays)
    _json_dump(config_out, config)
    _combine_source_tables(seed_dirs, source_path)
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": "e1_final",
        "scientific_digest": scientific_digest,
        "all_seed_invariants_pass": True,
        "seed_reports": seed_reports,
        "condition_summaries": condition_summaries,
        "gate_a": gate_a,
    }
    _json_dump(report_path, report)
    repo_root = Path(__file__).resolve().parents[3]
    source_files = sorted(Path(__file__).resolve().parent.glob("*.py"))
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": PROTOCOL_VERSION,
        "experiment": "e1_final",
        "root_seeds": list(seeds),
        "git": _git_metadata(repo_root),
        "source_files": {
            str(path.relative_to(repo_root)): _sha256(path) for path in source_files
        },
        "config_source": {"path": str(config_path), "sha256": _sha256(config_path)},
        "seed_artifacts": {
            str(seed): {
                "scientific_digest": seed_reports[str(seed)]["scientific_digest"],
                "manifest_sha256": _sha256(seed_root / str(seed) / "manifest.json"),
            }
            for seed in seeds
        },
        "artifacts": {
            path.name: _sha256(path)
            for path in (arrays_path, config_out, source_path, report_path)
        },
        "scientific_digest": scientific_digest,
    }
    _json_dump(manifest_path, manifest)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Aggregate an intact complete per-seed run without rerunning simulations.",
    )
    args = parser.parse_args()
    report = run(args.config, args.output, resume_existing=args.resume_existing)
    print(json.dumps(report["gate_a"], indent=2, sort_keys=True, allow_nan=False))
    if not report["gate_a"]["gate_a_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
