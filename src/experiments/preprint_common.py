"""Small, explicit helpers shared by the two preprint experiments.

The scientific model remains in :mod:`core.models` and the MEC/LEC stimulus
generator remains in :mod:`core.datagen`.  This file only makes those pieces
deterministic and writes simple, inspectable result artifacts.
"""

from __future__ import annotations

import csv
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from core import ae_tools, datagen, models, mtl_tools


# Shared controlled configuration used for the main representational and
# degradation comparisons. In reference mode, base and err2 differ only in
# their plasticity equation. Evolved mode is applied separately below.
REFERENCE_AE_PARAMETERS = {
    "K_ca1": 5,
    "beta_latent": 25.0,
    "beta_output": 25.0,
}
REFERENCE_MTL_PARAMETERS = {
    "ca3_inputs_per_unit": 29,
    "k_ca3": 3,
    "k_ca1": 5,
    "beta_ca3": 170.6,
    "beta_ca1": 41.8,
    "alpha": 0.092,
}
PARAMETER_MODES = ("reference", "evolved")


@contextmanager
def legacy_numpy_seed(seed: int):
    """Temporarily seed legacy helpers that use NumPy's global RNG."""

    state = np.random.get_state()
    np.random.seed(int(seed) % (2**32 - 1))
    try:
        yield
    finally:
        np.random.set_state(state)


def sparse_patterns(
    count: int,
    dimension: int,
    active: int,
    rng: np.random.Generator,
    seen: set[tuple[int, ...]] | None = None,
) -> tuple[np.ndarray, set[tuple[int, ...]]]:
    """Draw unique binary sparse patterns for encoder pretraining or recall."""

    seen = set() if seen is None else set(seen)
    rows = []
    while len(rows) < count:
        indices = tuple(sorted(rng.choice(dimension, size=active, replace=False)))
        if indices in seen:
            continue
        seen.add(indices)
        row = np.zeros(dimension, dtype=np.float32)
        row[list(indices)] = 1.0
        rows.append(row)
    return np.stack(rows), seen


def cue_track(
    laps: int,
    track: dict,
    assignments: list[list[int]],
    seed: int,
) -> np.ndarray:
    """Make MEC+LEC laps with an explicit cue-identity schedule."""

    with legacy_numpy_seed(seed):
        cues = datagen.make_cues(2, track["size"] // 2, fixed=True)
        arguments = {
            "n": laps,
            "length": track["lap_length"],
            "cues_positions": track["cue_positions"],
            "cues_patterns": cues,
            "cues_sequence": assignments,
            "cue_sigma": track["cue_sigma"],
            "cue_beta": track["cue_beta"],
            "cue_alpha": track["cue_alpha"],
            "mec_binarized": track["mec_binarized"],
        }
        values, _ = datagen.sparse_stimulus_generator_sensory(
            laps=arguments,
            mec_size=track["size"] // 2,
            mec_sigma=track["mec_sigma"],
            lec_sigma=track["lec_sigma"],
        )
    return values.astype(np.float32)


def train_autoencoder(
    training: np.ndarray,
    validation: np.ndarray,
    settings: dict,
    seed: int,
) -> tuple[models.Autoencoder, float]:
    """Train the one-layer EC--CA1--EC encoder used as the frozen basis."""

    autoencoder_settings = settings["autoencoder"]
    torch.manual_seed(seed)
    model = models.Autoencoder(
        dim_ei=settings["dimension"],
        dim_ca1=autoencoder_settings["latent_dimension"],
        K_ca1=settings["active"],
        K_eo=settings["active"],
        beta_ei=autoencoder_settings["beta_latent"],
        beta_eo=autoencoder_settings["beta_output"],
        use_bias=False,
    ).cpu()
    # Use the project's established trainer.  The paper only needs the final
    # frozen encoder, so validation is calculated once below instead of after
    # every epoch.
    ae_tools.train_autoencoder(
        training_data=training,
        test_data=validation,
        autoencoder=model,
        epochs=autoencoder_settings["epochs"],
        batch_size=autoencoder_settings["batch_size"],
        learning_rate=autoencoder_settings["learning_rate"],
        disable=True,
        test_every=None,
        device="cpu",
    )
    model.eval()
    with torch.no_grad():
        reconstruction = model(torch.as_tensor(validation, dtype=torch.float32)).numpy()
    mse = float(np.mean((reconstruction - validation) ** 2))
    return model, mse


def build_mtl(autoencoder: models.Autoencoder, memory: dict, seed: int) -> models.MTL:
    """Instantiate the existing CA3-key/CA1-plasticity model reproducibly."""

    encoder, decoder, _, _ = autoencoder.get_weights(bias=False)
    with legacy_numpy_seed(seed):
        return models.MTL(
            W_ei_ca1=encoder.detach().clone(),
            W_ca1_eo=decoder.detach().clone(),
            K_ca1=autoencoder._K_ca1,
            K_eo=autoencoder._K_eo,
            K_ca3=memory["k_ca3"],
            dim_ca3=memory["ca3_dimension"],
            beta_is=autoencoder._beta_ei,
            beta_ca3=memory["beta_ca3"],
            beta_ca1=memory["beta_ca1"],
            beta_eo=autoencoder._beta_eo,
            alpha=memory["alpha"],
            nb_ei_ca3=memory["ca3_inputs_per_unit"],
            plasticity=memory["plasticity_rule"],
        )


def run_mtl(model: models.MTL, inputs: np.ndarray, learn: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one sequence and return EC output, CA1 activity, and CA3 keys."""

    return mtl_tools.run_sequence(model, inputs, learn)


def row_cosine(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Cosine similarity for matching rows of two matrices."""

    numerator = np.sum(first * second, axis=-1)
    denominator = np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1)
    return numerator / np.maximum(denominator, 1e-12)


def write_artifact(output: Path, config: dict, arrays: dict[str, np.ndarray], rows: list[dict]) -> Path:
    """Write data required for plotting: arrays, parameters, and tidy rows."""

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "arrays.npz", **arrays)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if rows:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        with (output / "source_data.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    return output


def apply_reference_parameters(settings: dict) -> dict:
    """Apply one shared parameter point to every selected plasticity rule."""

    settings["active"] = REFERENCE_AE_PARAMETERS["K_ca1"]
    settings["autoencoder"]["beta_latent"] = REFERENCE_AE_PARAMETERS["beta_latent"]
    settings["autoencoder"]["beta_output"] = REFERENCE_AE_PARAMETERS["beta_output"]

    memory_defaults = settings.get("memory", {})
    shared_memory = {
        **memory_defaults,
        **REFERENCE_MTL_PARAMETERS,
    }
    rule_memories = {
        rule: {**shared_memory, "plasticity_rule": rule}
        for rule in settings.get("plasticity_rules", [])
    }
    settings["memory_by_rule"] = rule_memories
    if rule_memories:
        display_rule = "err2" if "err2" in rule_memories else next(iter(rule_memories))
        settings["memory"] = dict(rule_memories[display_rule])

    if "compatibility" in settings:
        settings["compatibility_by_rule"] = {
            rule: {
                **settings["compatibility"],
                "ca3_inputs_per_unit": memory["ca3_inputs_per_unit"],
                "k_ca3": memory["k_ca3"],
                "k_ca1": memory["k_ca1"],
                "beta_ca3": memory["beta_ca3"],
                "beta_ca1": memory["beta_ca1"],
                "beta_output": settings["autoencoder"]["beta_output"],
                "write_alpha": memory["alpha"],
            }
            for rule, memory in rule_memories.items()
        }
    return settings


def apply_evolved_parameters(settings: dict, root_dir: Path) -> dict:
    """Update preprint settings with the best available evolved parameters.

    The figure scripts keep readable fallback defaults near the top of each
    file.  When evolution artifacts are present, this helper replaces those
    defaults with the selected autoencoder and rule-specific MTL parameters.
    Missing files are ignored, which keeps quick exploratory copies runnable.
    """

    ae_path = root_dir / "results/preprint/ae_evolution/best_parameters.json"
    if ae_path.exists():
        ae_best = json.loads(ae_path.read_text())
        settings["active"] = int(ae_best.get("K_ca1", settings["active"]))
        settings["autoencoder"]["beta_latent"] = float(
            ae_best.get("beta_latent", settings["autoencoder"]["beta_latent"]))
        settings["autoencoder"]["beta_output"] = float(
            ae_best.get("beta_output", settings["autoencoder"]["beta_output"]))

    memory_defaults = settings.get("memory", {})
    explicit_rule_memories = settings.get("memory_by_rule", {})
    rule_memories = {}
    compatibility_defaults = settings.get("compatibility")
    compatibility_by_rule = {}

    for rule in settings.get("plasticity_rules", []):
        # Prefer a rule-specific configuration written directly in the
        # experiment file.  The JSON artifact, when present, remains the
        # authoritative override.  This makes the figure scripts reproducible
        # even if the evolution output directory is moved or archived.
        memory = {
            **memory_defaults,
            **explicit_rule_memories.get(rule, {}),
            "plasticity_rule": rule,
        }
        best_path = root_dir / f"results/preprint/mtl_evolution/{rule}/best_parameters.json"
        if best_path.exists():
            best = json.loads(best_path.read_text())
            memory.update({
                "ca3_inputs_per_unit": int(best["ca3_inputs_per_unit"]),
                "k_ca3": int(best["k_ca3"]),
                "beta_ca3": float(best["beta_ca3"]),
                "beta_ca1": float(best["beta_ca1"]),
                "alpha": float(best["alpha"]),
            })
        memory["k_ca1"] = int(settings["active"])
        rule_memories[rule] = memory

        if compatibility_defaults is not None:
            config = dict(compatibility_defaults)
            config.update({
                "ca3_inputs_per_unit": memory["ca3_inputs_per_unit"],
                "k_ca3": memory["k_ca3"],
                "k_ca1": memory["k_ca1"],
                "beta_ca3": memory["beta_ca3"],
                "beta_ca1": memory["beta_ca1"],
                "beta_output": settings["autoencoder"]["beta_output"],
                "write_alpha": memory["alpha"],
            })
            compatibility_by_rule[rule] = config

    if rule_memories:
        settings["memory_by_rule"] = rule_memories
        display_rule = "err2" if "err2" in rule_memories else next(iter(rule_memories))
        settings["memory"] = dict(rule_memories[display_rule])
    if compatibility_by_rule:
        settings["compatibility_by_rule"] = compatibility_by_rule

    return settings
