"""Run the frozen matched BASE/ERR2 development comparison."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.paper import PROTOCOL_VERSION
from experiments.paper.runner import (
    CONDITIONS,
    _array_digest,
    _balanced_ca3_weights,
    _git_metadata,
    _json_dump,
    _sha256,
    _simulate,
    _train_autoencoder,
    _unique_sparse_patterns,
    _validate_config,
)
from experiments.paper.seeds import (
    DEVELOPMENT_SEEDS,
    SeedStreams,
    assert_seed_sets_disjoint,
    random_derangement,
    random_nonidentity_permutation,
)


METRIC_NAMES = (
    "raw_cosine",
    "topk_overlap",
    "identity_correct",
    "chance_corrected_cosine",
    "mse",
)


def _mean_and_se(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean, se


def select_development_configuration(
    raw_cosine: np.ndarray,
    rule_names: list[str],
    alphas: np.ndarray,
    condition_names: list[str],
    selection_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Apply the frozen feasibility and one-standard-error rule.

    ``raw_cosine`` has shape rule × alpha × seed × condition × query.
    """

    aligned_index = condition_names.index("aligned")
    fixed_index = condition_names.index("fixed_permutation")
    rescue_index = condition_names.index("matched_decoder_rescue")
    candidates: dict[str, list[dict[str, Any]]] = {}
    selected_rates: dict[str, float | None] = {}

    for rule_index, rule in enumerate(rule_names):
        rule_candidates = []
        for alpha_index, alpha in enumerate(alphas):
            seed_condition_means = raw_cosine[rule_index, alpha_index].mean(axis=-1)
            aligned = seed_condition_means[:, aligned_index]
            fixed = seed_condition_means[:, fixed_index]
            rescue = seed_condition_means[:, rescue_index]
            effect = aligned - fixed
            equivalence = rescue - aligned
            equivalence_mean, equivalence_se = _mean_and_se(equivalence)
            equivalence_half_width = (
                selection_cfg["equivalence_t_critical_df7"] * equivalence_se
            )
            equivalence_low = equivalence_mean - equivalence_half_width
            equivalence_high = equivalence_mean + equivalence_half_width
            candidate = {
                "alpha": float(alpha),
                "positive_effect_seeds": int(np.sum(effect > 0.0)),
                "mean_aligned_cosine": float(aligned.mean()),
                "se_aligned_cosine": float(aligned.std(ddof=1) / np.sqrt(len(aligned))),
                "mean_aligned_minus_fixed": float(effect.mean()),
                "mean_rescue_minus_aligned": equivalence_mean,
                "rescue_equivalence_90_ci": [equivalence_low, equivalence_high],
            }
            margin = selection_cfg["rescue_equivalence_margin"]
            candidate["feasible"] = bool(
                candidate["positive_effect_seeds"]
                >= selection_cfg["minimum_positive_seeds"]
                and candidate["mean_aligned_cosine"]
                >= selection_cfg["minimum_aligned_cosine"]
                and candidate["mean_aligned_minus_fixed"]
                >= selection_cfg["minimum_alignment_effect"]
                and equivalence_low >= -margin
                and equivalence_high <= margin
            )
            rule_candidates.append(candidate)
        candidates[rule] = rule_candidates
        feasible = [candidate for candidate in rule_candidates if candidate["feasible"]]
        if not feasible:
            selected_rates[rule] = None
            continue
        best = max(feasible, key=lambda item: item["mean_aligned_cosine"])
        cutoff = best["mean_aligned_cosine"] - best["se_aligned_cosine"]
        within_one_se = [
            candidate for candidate in feasible if candidate["mean_aligned_cosine"] >= cutoff
        ]
        selected_rates[rule] = float(min(within_one_se, key=lambda item: item["alpha"])["alpha"])

    if selected_rates.get("base") is not None:
        primary_rule = "base"
    elif selected_rates.get("err2") is not None:
        primary_rule = "err2"
    else:
        primary_rule = None
    return {
        "candidates": candidates,
        "selected_rates": selected_rates,
        "primary_rule": primary_rule,
        "primary_alpha": selected_rates.get(primary_rule) if primary_rule else None,
        "step3_pass": primary_rule is not None,
    }


def _write_source_data(
    path: Path,
    arrays: dict[str, np.ndarray],
    rule_names: list[str],
    alphas: np.ndarray,
    seeds: list[int],
) -> None:
    conditions = arrays["condition_names"].tolist()
    fields = [
        "rule",
        "alpha",
        "root_seed",
        "condition",
        "memory_index",
        *METRIC_NAMES,
        "target_ca1_cosine",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rule_index, rule in enumerate(rule_names):
            for alpha_index, alpha in enumerate(alphas):
                for seed_index, seed in enumerate(seeds):
                    for condition_index, condition in enumerate(conditions):
                        for memory_index in range(arrays["inputs"].shape[1]):
                            row: dict[str, Any] = {
                                "rule": rule,
                                "alpha": float(alpha),
                                "root_seed": seed,
                                "condition": condition,
                                "memory_index": memory_index,
                                "target_ca1_cosine": float(
                                    arrays["target_ca1_cosine"][
                                        rule_index,
                                        alpha_index,
                                        seed_index,
                                        condition_index,
                                        memory_index,
                                    ]
                                ),
                            }
                            for metric in METRIC_NAMES:
                                row[metric] = float(
                                    arrays[f"metric_{metric}"][
                                        rule_index,
                                        alpha_index,
                                        seed_index,
                                        condition_index,
                                        memory_index,
                                    ]
                                )
                            writer.writerow(row)


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    assert_seed_sets_disjoint()
    seeds = [int(seed) for seed in config["root_seeds"]]
    if tuple(seeds) != DEVELOPMENT_SEEDS:
        raise ValueError("Development config must use the complete frozen seed set")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "autoencoders"
    checkpoint_dir.mkdir()

    rules = [str(rule) for rule in config["development"]["rules"]]
    alphas = np.asarray(config["development"]["alphas"], dtype=np.float64)
    data_cfg = config["data"]
    collected: dict[tuple[int, int], list[dict[str, np.ndarray]]] = {
        (rule_index, alpha_index): []
        for rule_index in range(len(rules))
        for alpha_index in range(len(alphas))
    }
    diagnostics: dict[str, dict[str, Any]] = {}
    common_inputs = []
    common_orders = []
    common_permutations = []
    common_derangements = []
    ae_quality_rows = []
    checkpoint_hashes: dict[str, str] = {}

    for seed_position, seed in enumerate(seeds, start=1):
        print(f"development seed {seed} ({seed_position}/{len(seeds)}): training AE", flush=True)
        streams = SeedStreams(seed)
        training, training_seen = _unique_sparse_patterns(
            data_cfg["training_size"],
            data_cfg["dimension"],
            data_cfg["active"],
            streams.numpy("ae_training_patterns"),
        )
        validation, validation_seen = _unique_sparse_patterns(
            data_cfg["validation_size"],
            data_cfg["dimension"],
            data_cfg["active"],
            streams.numpy("ae_validation_patterns"),
            forbidden=training_seen,
        )
        memories, _ = _unique_sparse_patterns(
            data_cfg["memory_count"],
            data_cfg["dimension"],
            data_cfg["active"],
            streams.numpy("memory_bank"),
            forbidden=validation_seen,
        )
        model, quality, _, _ = _train_autoencoder(config, streams, training, validation)
        if not quality["pass"]:
            raise RuntimeError(f"AE gate failed for seed {seed}: {quality}")
        checkpoint_path = checkpoint_dir / f"seed_{seed}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        checkpoint_hashes[str(seed)] = _sha256(checkpoint_path)
        ae_quality_rows.append([quality["mse"], quality["mean_cosine"]])

        ca3_weights = _balanced_ca3_weights(
            data_cfg["dimension"],
            config["memory"]["ca3_inputs_per_unit"],
            streams.numpy("ca3_wiring"),
        )
        storage_order = streams.numpy("storage_order").permutation(data_cfg["memory_count"])
        permutation = random_nonidentity_permutation(
            config["autoencoder"]["latent_dimension"],
            streams.numpy("coordinate_permutation"),
        )
        derangement = random_derangement(
            data_cfg["memory_count"], streams.numpy("content_derangement")
        )
        common_inputs.append(memories)
        common_orders.append(storage_order)
        common_permutations.append(permutation)
        common_derangements.append(derangement)

        for rule_index, rule in enumerate(rules):
            for alpha_index, alpha in enumerate(alphas):
                run_config = copy.deepcopy(config)
                run_config["root_seed"] = seed
                run_config["memory"]["plasticity_rule"] = rule
                run_config["memory"]["alpha"] = float(alpha)
                result_arrays, result_diagnostics = _simulate(
                    run_config,
                    model,
                    memories,
                    ca3_weights,
                    storage_order,
                    permutation,
                    derangement,
                )
                if not result_diagnostics["all_checks_pass"]:
                    failed = [
                        name
                        for name, passed in result_diagnostics["checks"].items()
                        if not passed
                    ]
                    raise RuntimeError(
                        f"Invariant failure seed={seed}, rule={rule}, alpha={alpha}: {failed}"
                    )
                collected[(rule_index, alpha_index)].append(result_arrays)
                diagnostics[f"{seed}/{rule}/{alpha:g}"] = result_diagnostics
        print(f"development seed {seed}: complete", flush=True)

    arrays: dict[str, np.ndarray] = {
        "rule_names": np.asarray(rules),
        "alphas": alphas,
        "root_seeds": np.asarray(seeds, dtype=np.int64),
        "condition_names": np.asarray(CONDITIONS),
        "inputs": np.stack(common_inputs).astype(np.float32),
        "storage_orders": np.stack(common_orders).astype(np.int64),
        "coordinate_permutations": np.stack(common_permutations).astype(np.int64),
        "content_derangements": np.stack(common_derangements).astype(np.int64),
        "autoencoder_quality": np.asarray(ae_quality_rows, dtype=np.float64),
    }
    result_keys = [
        "outputs",
        "recalled_ca1",
        "final_weights",
        "condition_instructions",
        "target_ca1_cosine",
        *(f"metric_{name}" for name in METRIC_NAMES),
    ]
    for key in result_keys:
        arrays[key] = np.stack(
            [
                np.stack(
                    [
                        np.stack([item[key] for item in collected[(rule_index, alpha_index)]])
                        for alpha_index in range(len(alphas))
                    ]
                )
                for rule_index in range(len(rules))
            ]
        )

    selection = select_development_configuration(
        arrays["metric_raw_cosine"],
        rules,
        alphas,
        list(CONDITIONS),
        config["development"],
    )
    scientific_digest = _array_digest(arrays)
    arrays_path = output_dir / "arrays.npz"
    config_out = output_dir / "config.json"
    report_path = output_dir / "report.json"
    source_path = output_dir / "source_data.csv"
    manifest_path = output_dir / "manifest.json"
    np.savez_compressed(arrays_path, **arrays)
    _json_dump(config_out, config)
    _write_source_data(source_path, arrays, rules, alphas, seeds)
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": config["experiment"],
        "scientific_digest": scientific_digest,
        "autoencoder_quality": {
            str(seed): {
                "mse": float(ae_quality_rows[index][0]),
                "mean_cosine": float(ae_quality_rows[index][1]),
            }
            for index, seed in enumerate(seeds)
        },
        "all_invariants_pass": True,
        "diagnostics": diagnostics,
        "selection": selection,
    }
    _json_dump(report_path, report)
    repo_root = Path(__file__).resolve().parents[3]
    source_files = sorted(Path(__file__).resolve().parent.glob("*.py"))
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": PROTOCOL_VERSION,
        "experiment": config["experiment"],
        "root_seeds": seeds,
        "seed_manifests": {str(seed): SeedStreams(seed).manifest() for seed in seeds},
        "git": _git_metadata(repo_root),
        "source_files": {
            str(path.relative_to(repo_root)): _sha256(path) for path in source_files
        },
        "config_source": {"path": str(config_path), "sha256": _sha256(config_path)},
        "checkpoint_hashes": checkpoint_hashes,
        "artifacts": {
            path.name: _sha256(path)
            for path in (arrays_path, config_out, report_path, source_path)
        },
        "scientific_digest": scientific_digest,
    }
    _json_dump(manifest_path, manifest)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run(args.config, args.output)
    print(json.dumps(report["selection"], indent=2, sort_keys=True, allow_nan=False))
    if not report["selection"]["step3_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
