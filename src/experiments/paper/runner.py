"""Configuration-driven E0/E1 runner.

Run from the repository root with::

    PYTHONPATH=src python3 -m experiments.paper.runner \
        --config src/experiments/paper/configs/e0_smoke.json \
        --output results/paper/v1/e0_smoke

The output directory must be absent or empty. Figures are built separately by
``experiments.paper.figures`` and never import the simulation model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ``core.functions`` imports Matplotlib even though the paper runner only uses
# its activation function. Keep cache writes out of the user's home directory.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kam-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "kam-cache"))

import numpy as np
import torch

from core.functions import sparsemoid
from core.models import Autoencoder
from experiments.paper import PROTOCOL_VERSION
from experiments.paper.metrics import (
    evaluate_outputs,
    metric_sanity_checks,
    row_cosine,
)
from experiments.paper.seeds import (
    DEVELOPMENT_SEEDS,
    FACTORIAL_SEEDS,
    FINAL_SEEDS,
    SeedStreams,
    assert_seed_sets_disjoint,
    random_derangement,
    random_nonidentity_permutation,
)


CONDITIONS = (
    "aligned",
    "fixed_permutation",
    "matched_decoder_rescue",
    "random_content_matched",
    "no_plasticity",
)


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_digest(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else "unavailable"

    status = run("status", "--short")
    return {
        "revision": run("rev-parse", "HEAD"),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "dirty": bool(status and status != "unavailable"),
    }


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "protocol_version",
        "experiment",
        "root_seed",
        "data",
        "autoencoder",
        "memory",
        "conditions",
        "tolerances",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Configuration is missing keys: {sorted(missing)}")
    if config["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("Configuration protocol version does not match package")
    if tuple(config["conditions"]) != CONDITIONS:
        raise ValueError(f"Conditions must be exactly {CONDITIONS}")
    if config["device"] != "cpu" or config["dtype"] != "float32":
        raise ValueError("Paper v1 requires CPU float32")
    data = config["data"]
    ae = config["autoencoder"]
    memory = config["memory"]
    if data["dimension"] != ae["latent_dimension"]:
        raise ValueError("Paper v1 requires equal input and latent dimensions")
    if data["dimension"] != memory["ca3_dimension"]:
        raise ValueError("Paper v1 requires equal input and CA3 dimensions")
    if data["active"] != memory["k_ca3"] or data["active"] != memory["k_ca1"]:
        raise ValueError("Paper v1 requires data sparsity to match CA3 and CA1 K")
    if memory["plasticity_rule"] not in {"base", "err2", "err1"}:
        raise ValueError("Unsupported paper plasticity rule")
    if int(config["root_seed"]) in set(DEVELOPMENT_SEEDS + FINAL_SEEDS + FACTORIAL_SEEDS):
        if config["experiment"] == "e0_smoke":
            raise ValueError("E0 smoke runs may not access reserved seeds")


def _unique_sparse_patterns(
    count: int,
    dimension: int,
    active: int,
    rng: np.random.Generator,
    forbidden: set[tuple[int, ...]] | None = None,
) -> tuple[np.ndarray, set[tuple[int, ...]]]:
    if not 0 < active < dimension:
        raise ValueError("active must be between zero and dimension")
    seen = set() if forbidden is None else set(forbidden)
    produced: list[np.ndarray] = []
    while len(produced) < count:
        indices = tuple(sorted(rng.choice(dimension, size=active, replace=False).tolist()))
        if indices in seen:
            continue
        seen.add(indices)
        row = np.zeros(dimension, dtype=np.float32)
        row[list(indices)] = 1.0
        produced.append(row)
    return np.stack(produced), seen


def _balanced_ca3_weights(
    dimension: int, inputs_per_unit: int, rng: np.random.Generator
) -> np.ndarray:
    if not 1 <= inputs_per_unit <= dimension:
        raise ValueError("inputs_per_unit outside valid range")
    assignments: list[list[int]] = [[] for _ in range(dimension)]
    for _ in range(inputs_per_unit):
        while True:
            permutation = rng.permutation(dimension)
            if all(int(permutation[row]) not in assignments[row] for row in range(dimension)):
                break
        for row, source in enumerate(permutation):
            assignments[row].append(int(source))
    weights = np.zeros((dimension, dimension), dtype=np.float32)
    for row, sources in enumerate(assignments):
        weights[row, sources] = 1.0 / dimension
    return weights


def _train_autoencoder(
    config: dict[str, Any],
    streams: SeedStreams,
    training: np.ndarray,
    validation: np.ndarray,
    seed_overrides: dict[str, int] | None = None,
) -> tuple[Autoencoder, dict[str, float], np.ndarray, np.ndarray]:
    data_cfg = config["data"]
    ae_cfg = config["autoencoder"]
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    seed_overrides = {} if seed_overrides is None else seed_overrides
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(
            seed_overrides.get(
                "ae_initialization", streams.integer("ae_initialization")
            )
        )
        model = Autoencoder(
            dim_ei=data_cfg["dimension"],
            dim_ca1=ae_cfg["latent_dimension"],
            K_ca1=data_cfg["active"],
            K_eo=data_cfg["active"],
            beta_ei=ae_cfg["beta_latent"],
            beta_eo=ae_cfg["beta_output"],
            use_bias=ae_cfg["use_bias"],
        ).cpu()

    optimizer = torch.optim.Adam(model.parameters(), lr=ae_cfg["learning_rate"])
    criterion = torch.nn.MSELoss()
    tensor = torch.tensor(training, dtype=torch.float32)
    order_generator = torch.Generator(device="cpu")
    order_generator.manual_seed(
        seed_overrides.get(
            "ae_minibatch_order", streams.integer("ae_minibatch_order")
        )
    )

    model.train()
    batch_size = int(ae_cfg["batch_size"])
    for _ in range(int(ae_cfg["epochs"])):
        order = torch.randperm(len(tensor), generator=order_generator)
        for start in range(0, len(tensor), batch_size):
            batch = tensor[order[start : start + batch_size]]
            output = model(batch)
            loss = criterion(output, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        validation_tensor = torch.tensor(validation, dtype=torch.float32)
        validation_output, validation_code = model(validation_tensor, ca1=True)
    reconstructed = validation_output.cpu().numpy().astype(np.float32)
    codes = validation_code.cpu().numpy().astype(np.float32)
    quality = {
        "mse": float(np.mean((reconstructed - validation) ** 2)),
        "mean_cosine": float(row_cosine(reconstructed, validation).mean()),
    }
    quality["pass"] = bool(
        quality["mse"] <= ae_cfg["quality_mse_max"]
        and quality["mean_cosine"] >= ae_cfg["quality_cosine_min"]
    )
    return model, quality, reconstructed, codes


def _sparsemoid_vector(values: torch.Tensor, k: int, beta: float) -> torch.Tensor:
    return sparsemoid(values.reshape(1, -1), K=k, beta=beta).reshape(-1)


def _ca3_code(
    input_vector: torch.Tensor,
    weights: torch.Tensor,
    k: int,
    beta: float,
) -> torch.Tensor:
    return _sparsemoid_vector(weights @ input_vector, k, beta)


def _update_weights(
    weights: torch.Tensor,
    ca3: torch.Tensor,
    instruction: torch.Tensor,
    rule: str,
    alpha: float,
    k_ca1: int,
    beta_ca1: float,
) -> torch.Tensor:
    if rule == "base":
        return (1.0 - alpha * instruction[:, None]) * weights + alpha * (
            instruction[:, None] @ ca3[None, :]
        )
    recalled = _sparsemoid_vector(weights @ ca3, k_ca1, beta_ca1)
    if rule == "err2":
        positive = torch.relu(instruction - recalled)[:, None] @ ca3[None, :]
        negative = torch.relu(recalled - instruction)[:, None] @ ca3[None, :]
        return (weights + alpha * positive * (1.0 - weights) - alpha * negative * weights).clamp(0.0, 1.0)
    if rule == "err1":
        error = instruction - recalled
        normalizer = ca3.square().sum().clamp_min(1e-6)
        return (weights + alpha * (error[:, None] @ ca3[None, :]) / normalizer).clamp(0.0, 1.0)
    raise ValueError(f"Unsupported rule: {rule}")


def _recall(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    ca3_weights: torch.Tensor,
    decoder: torch.Tensor,
    memory_cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    ca1_rows = []
    output_rows = []
    for input_vector in inputs:
        ca3 = _ca3_code(
            input_vector,
            ca3_weights,
            memory_cfg["k_ca3"],
            memory_cfg["beta_ca3"],
        )
        ca1 = _sparsemoid_vector(
            weights @ ca3,
            memory_cfg["k_ca1"],
            memory_cfg["beta_ca1"],
        )
        output = torch.sigmoid(memory_cfg["beta_output"] * (decoder @ ca1))
        ca1_rows.append(ca1)
        output_rows.append(output)
    return torch.stack(output_rows), torch.stack(ca1_rows)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes()
    ).hexdigest()


def _simulate(
    config: dict[str, Any],
    model: Autoencoder,
    memories: np.ndarray,
    ca3_weights_np: np.ndarray,
    storage_order: np.ndarray,
    permutation: np.ndarray,
    derangement: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    memory_cfg = config["memory"]
    tolerances = config["tolerances"]
    inputs = torch.tensor(memories, dtype=torch.float32)
    ca3_weights = torch.tensor(ca3_weights_np, dtype=torch.float32)
    encoder = model.encoder[0].weight.detach().cpu()
    decoder = model.decoder[0].weight.detach().cpu()

    with torch.no_grad():
        encoded = sparsemoid(
            (inputs @ encoder.T),
            K=memory_cfg["k_ca1"],
            beta=float(config["autoencoder"]["beta_latent"]),
        )

    p = torch.tensor(permutation, dtype=torch.long)
    q = torch.tensor(derangement, dtype=torch.long)
    instructions = {
        "aligned": encoded,
        "fixed_permutation": encoded[:, p],
        "matched_decoder_rescue": encoded[:, p],
        "random_content_matched": encoded[q],
        "no_plasticity": encoded,
    }

    dimension = encoded.shape[1]
    learned_weights: dict[str, torch.Tensor] = {}
    for condition in ("aligned", "fixed_permutation", "random_content_matched"):
        weights = torch.zeros((dimension, ca3_weights.shape[0]), dtype=torch.float32)
        with torch.no_grad():
            for item_index in storage_order:
                ca3 = _ca3_code(
                    inputs[int(item_index)],
                    ca3_weights,
                    memory_cfg["k_ca3"],
                    memory_cfg["beta_ca3"],
                )
                weights = _update_weights(
                    weights,
                    ca3,
                    instructions[condition][int(item_index)],
                    memory_cfg["plasticity_rule"],
                    memory_cfg["alpha"],
                    memory_cfg["k_ca1"],
                    memory_cfg["beta_ca1"],
                )
        learned_weights[condition] = weights
    learned_weights["matched_decoder_rescue"] = learned_weights["fixed_permutation"]
    learned_weights["no_plasticity"] = torch.zeros_like(learned_weights["aligned"])

    decoders = {condition: decoder for condition in CONDITIONS}
    decoders["matched_decoder_rescue"] = decoder[:, p]

    outputs: dict[str, torch.Tensor] = {}
    recalled_ca1: dict[str, torch.Tensor] = {}
    recall_hashes: dict[str, dict[str, str]] = {}
    for condition in CONDITIONS:
        before = _tensor_sha256(learned_weights[condition])
        with torch.no_grad():
            output, ca1 = _recall(
                inputs,
                learned_weights[condition],
                ca3_weights,
                decoders[condition],
                memory_cfg,
            )
        after = _tensor_sha256(learned_weights[condition])
        outputs[condition] = output
        recalled_ca1[condition] = ca1
        recall_hashes[condition] = {"before": before, "after": after}

    condition_outputs = np.stack([outputs[name].numpy() for name in CONDITIONS])
    condition_ca1 = np.stack([recalled_ca1[name].numpy() for name in CONDITIONS])
    condition_weights = np.stack([learned_weights[name].numpy() for name in CONDITIONS])
    condition_instructions = np.stack([instructions[name].numpy() for name in CONDITIONS])

    metric_arrays: dict[str, list[np.ndarray]] = {}
    for output in condition_outputs:
        for name, values in evaluate_outputs(output, memories, config["data"]["active"]).items():
            metric_arrays.setdefault(name, []).append(values)
    metrics = {name: np.stack(values) for name, values in metric_arrays.items()}
    target_ca1_cosine = np.stack(
        [row_cosine(condition_ca1[index], condition_instructions[index]) for index in range(len(CONDITIONS))]
    )

    probe_rng = np.random.default_rng(20260826)
    probe = torch.tensor(probe_rng.normal(size=dimension), dtype=torch.float32)
    decoder_error = float(torch.max(torch.abs(decoder @ probe - decoder[:, p] @ probe[p])).item())
    fixed_index = CONDITIONS.index("fixed_permutation")
    rescue_index = CONDITIONS.index("matched_decoder_rescue")
    aligned_index = CONDITIONS.index("aligned")
    signal_delta = float(
        np.max(
            np.abs(
                np.sort(condition_instructions[aligned_index], axis=1)
                - np.sort(condition_instructions[fixed_index], axis=1)
            )
        )
    )
    fixed_rescue_weight_delta = float(
        np.max(np.abs(condition_weights[fixed_index] - condition_weights[rescue_index]))
    )
    equivariance_delta = float(
        np.max(
            np.abs(
                condition_weights[fixed_index]
                - condition_weights[aligned_index][permutation]
            )
        )
    )
    checks: dict[str, bool] = {
        **metric_sanity_checks(),
        "reserved_seed_sets_disjoint": True,
        "paired_inputs_and_storage_order_shared": True,
        "decoder_orientation": decoder_error <= tolerances["decoder_identity"],
        "fixed_rescue_weight_identity": fixed_rescue_weight_delta <= tolerances["weight_equality"],
        "coordinate_equivariance": equivariance_delta <= tolerances["coordinate_equivariance"],
        "signal_statistics_match": signal_delta <= tolerances["signal_match"],
        "coordinate_permutation_nonidentity": not np.array_equal(permutation, np.arange(dimension)),
        "content_derangement_has_no_matches": bool(np.all(derangement != np.arange(len(memories)))),
        "recall_frozen": all(value["before"] == value["after"] for value in recall_hashes.values()),
        "actual_ca1_separate_from_instruction": bool(
            not np.shares_memory(condition_ca1, condition_instructions)
            and np.max(np.abs(condition_ca1[-1] - condition_instructions[-1])) > 1e-7
        ),
    }

    arrays = {
        "condition_names": np.asarray(CONDITIONS),
        "inputs": memories.astype(np.float32),
        "storage_order": storage_order.astype(np.int64),
        "coordinate_permutation": permutation.astype(np.int64),
        "inverse_permutation": np.argsort(permutation).astype(np.int64),
        "content_derangement": derangement.astype(np.int64),
        "ca3_weights": ca3_weights_np.astype(np.float32),
        "instruction_codes": encoded.numpy().astype(np.float32),
        "condition_instructions": condition_instructions.astype(np.float32),
        "recalled_ca1": condition_ca1.astype(np.float32),
        "outputs": condition_outputs.astype(np.float32),
        "final_weights": condition_weights.astype(np.float32),
        "target_ca1_cosine": target_ca1_cosine.astype(np.float64),
        **{f"metric_{name}": values.astype(np.float64) for name, values in metrics.items()},
    }
    diagnostics = {
        "checks": checks,
        "all_checks_pass": bool(all(checks.values())),
        "decoder_identity_max_abs": decoder_error,
        "fixed_rescue_weight_max_abs": fixed_rescue_weight_delta,
        "coordinate_equivariance_max_abs": equivariance_delta,
        "signal_sorted_value_max_abs": signal_delta,
        "recall_weight_hashes": recall_hashes,
    }
    return arrays, diagnostics


def _write_source_csv(path: Path, arrays: dict[str, np.ndarray], root_seed: int) -> None:
    names = arrays["condition_names"].tolist()
    metric_names = sorted(name.removeprefix("metric_") for name in arrays if name.startswith("metric_"))
    fields = ["root_seed", "condition", "memory_index", *metric_names, "target_ca1_cosine"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition_index, condition in enumerate(names):
            for memory_index in range(arrays["inputs"].shape[0]):
                row: dict[str, Any] = {
                    "root_seed": root_seed,
                    "condition": condition,
                    "memory_index": memory_index,
                    "target_ca1_cosine": float(arrays["target_ca1_cosine"][condition_index, memory_index]),
                }
                for metric in metric_names:
                    row[metric] = float(arrays[f"metric_{metric}"][condition_index, memory_index])
                writer.writerow(row)


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    assert_seed_sets_disjoint()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    streams = SeedStreams(int(config["root_seed"]))
    data_cfg = config["data"]
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
    model, ae_quality, validation_outputs, validation_codes = _train_autoencoder(
        config, streams, training, validation
    )
    if not ae_quality["pass"]:
        raise RuntimeError(f"Autoencoder technical gate failed: {ae_quality}")

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
    arrays, diagnostics = _simulate(
        config,
        model,
        memories,
        ca3_weights,
        storage_order,
        permutation,
        derangement,
    )
    arrays["ae_validation_targets"] = validation.astype(np.float32)
    arrays["ae_validation_outputs"] = validation_outputs
    arrays["ae_validation_codes"] = validation_codes
    arrays["ae_training_targets"] = training.astype(np.float32)
    scientific_digest = _array_digest(arrays)

    resolved_config_path = output_dir / "config.json"
    arrays_path = output_dir / "arrays.npz"
    checkpoint_path = output_dir / "autoencoder.pt"
    source_path = output_dir / "source_data.csv"
    report_path = output_dir / "report.json"
    manifest_path = output_dir / "manifest.json"
    _json_dump(resolved_config_path, config)
    np.savez_compressed(arrays_path, **arrays)
    torch.save(model.state_dict(), checkpoint_path)
    _write_source_csv(source_path, arrays, int(config["root_seed"]))

    summaries = {}
    for index, condition in enumerate(CONDITIONS):
        summaries[condition] = {
            name.removeprefix("metric_"): float(values[index].mean())
            for name, values in arrays.items()
            if name.startswith("metric_")
        }
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": config["experiment"],
        "root_seed": int(config["root_seed"]),
        "scientific_digest": scientific_digest,
        "autoencoder_quality": ae_quality,
        "condition_summaries": summaries,
        **diagnostics,
    }
    _json_dump(report_path, report)

    repo_root = Path(__file__).resolve().parents[3]
    source_files = sorted(Path(__file__).resolve().parent.glob("*.py"))
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": PROTOCOL_VERSION,
        "experiment": config["experiment"],
        "seed_manifest": streams.manifest(),
        "reserved_seed_sets": {
            "development": list(DEVELOPMENT_SEEDS),
            "final": list(FINAL_SEEDS),
            "factorial": list(FACTORIAL_SEEDS),
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "dtype": "float32",
        },
        "git": _git_metadata(repo_root),
        "source_files": {
            str(path.relative_to(repo_root)): _sha256(path) for path in source_files
        },
        "config_source": {
            "path": str(config_path),
            "sha256": _sha256(config_path),
        },
        "artifacts": {
            path.name: _sha256(path)
            for path in (
                resolved_config_path,
                arrays_path,
                checkpoint_path,
                source_path,
                report_path,
            )
        },
        "scientific_digest": scientific_digest,
    }
    _json_dump(manifest_path, manifest)
    if not diagnostics["all_checks_pass"]:
        failed = [name for name, passed in diagnostics["checks"].items() if not passed]
        raise RuntimeError(f"E0 invariants failed: {failed}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run(args.config, args.output)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
