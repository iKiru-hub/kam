"""One-shot recall under matched and mismatched CA1 decoder coordinates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from core import functions
from experiments.preprint.artifacts import create_artifact
from experiments.preprint.config import read_config
from experiments.preprint.metrics import output_metrics
from experiments.preprint.model_factory import balanced_ca3_weights, ca3_code, recall, train_autoencoder, update_weights
from experiments.preprint.seeds import SeedStreams, derangement, nonidentity_permutation
from experiments.preprint.stimuli import sparse_patterns


CONDITIONS = ("aligned", "fixed_permutation", "matched_decoder", "random_content", "no_plasticity")


def run_seed(config: dict, root_seed: int) -> tuple[dict[str, np.ndarray], list[dict], dict]:
    streams = SeedStreams(root_seed)
    data = config["data"]
    memory = config["memory"]
    training, seen = sparse_patterns(data["training_size"], data["dimension"], data["active"], streams.numpy("ae_train"))
    validation, seen = sparse_patterns(data["validation_size"], data["dimension"], data["active"], streams.numpy("ae_valid"), seen)
    memories, _ = sparse_patterns(data["memory_count"], data["dimension"], data["active"], streams.numpy("memory"), seen)
    autoencoder, quality = train_autoencoder(training, validation, config, streams.integer("ae_init"), streams.integer("ae_batches"))
    inputs = torch.as_tensor(memories)
    encoder = autoencoder.encoder[0].weight.detach()
    decoder = autoencoder.decoder[0].weight.detach()
    with torch.no_grad():
        instruction = functions.sparsemoid(inputs @ encoder.T, K=memory["k_ca1"], beta=config["autoencoder"]["beta_latent"])
    permutation = nonidentity_permutation(len(instruction[0]), streams.numpy("permutation"))
    content = derangement(len(inputs), streams.numpy("content_control"))
    keys = ca3_code(inputs, balanced_ca3_weights(data["dimension"], memory["ca3_inputs_per_unit"], streams.numpy("ca3_wiring")), memory["k_ca3"], memory["beta_ca3"])
    order = streams.numpy("storage_order").permutation(len(inputs))
    instructions = {
        "aligned": instruction,
        "fixed_permutation": instruction[:, permutation],
        "matched_decoder": instruction[:, permutation],
        "random_content": instruction[content],
        "no_plasticity": instruction,
    }
    decoders = {name: decoder for name in CONDITIONS}
    decoders["matched_decoder"] = decoder[:, permutation]
    arrays = {name: [] for name in ("outputs", "ca1", "cosine", "mse", "topk", "identity")}
    rows = []
    for condition in CONDITIONS:
        weights = torch.zeros((data["dimension"], data["dimension"]), dtype=torch.float32)
        if condition != "no_plasticity":
            weights = update_weights(weights, keys[order], instructions[condition][order], memory["alpha"])
        output, recalled = recall(keys, weights, decoders[condition], memory)
        metrics = output_metrics(output, memories, data["active"])
        arrays["outputs"].append(output)
        arrays["ca1"].append(recalled)
        for name, values in metrics.items():
            arrays[name].append(values)
        rows.extend({"root_seed": root_seed, "condition": condition, "memory_index": index, **{name: float(values[index]) for name, values in metrics.items()}} for index in range(len(memories)))
    result = {name: np.stack(values) for name, values in arrays.items()}
    result.update({"inputs": memories, "instruction": instruction.numpy(), "permutation": permutation, "storage_order": order})
    return result, rows, quality


def run(config: dict, output: Path) -> Path:
    per_seed, rows, quality = [], [], []
    for position, root_seed in enumerate(config["root_seeds"], start=1):
        print(f"compatibility seed {root_seed} ({position}/{len(config['root_seeds'])})", flush=True)
        result, seed_rows, seed_quality = run_seed(config, int(root_seed))
        per_seed.append(result)
        rows.extend(seed_rows)
        quality.append(seed_quality)
    arrays = {name: np.stack([result[name] for result in per_seed]) for name in ("outputs", "ca1", "cosine", "mse", "topk", "identity", "inputs", "instruction", "permutation", "storage_order")}
    arrays["root_seeds"] = np.asarray(config["root_seeds"], dtype=np.int64)
    arrays["condition_names"] = np.asarray(CONDITIONS)
    report = {"experiment": "compatibility", "autoencoder_quality": quality, "condition_mean_cosine": {name: float(arrays["cosine"][:, index].mean()) for index, name in enumerate(CONDITIONS)}}
    return create_artifact(output, config, arrays, rows, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(run(read_config(args.config), args.output))


if __name__ == "__main__":
    main()
