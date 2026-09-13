"""Shared, paired evaluation helpers for the MTL paper plots.

The saved evolution fitness is useful for selecting parameters, but final
plots should evaluate every selected model on the same clean targets and the
same corruption draws.  This module keeps that pairing explicit.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch

import core.ae_tools as aect
import core.datagen as dg
import core.models as models
import core.mtl_tools as mtlct


PLASTICITY_VARIANTS = ("base", "err2", "btsp", "xbtsp")
RULE_COLORS = {
    "base": "#4C78A8",
    "err2": "#F58518",
    "btsp": "#54A24B",
    "xbtsp": "#E45756",
}
RULE_LABELS = {
    "base": "BASE (ET–IS)",
    "err2": "ERR2 (ET–(IS−X))",
    "btsp": "BTSP",
    "xbtsp": "xBTSP",
}


def _effective_bit_kind(settings: dict) -> int:
    """Interpret legacy sessions without metadata as the old bit-flip mode."""

    return int(settings.get("bit_kind", 0))


def find_saved_ae(*, dim_ca1: int, num_cue_patterns: int,
                  train_noise: float, bit_kind: int):
    """Return the best matching saved AE with an explicit legacy fallback."""

    candidates = aect.find_ae(
        dim_ca1=dim_ca1,
        num_cue_patterns=num_cue_patterns,
        noise_level=float(train_noise),
    )
    candidates = [
        item for item in candidates
        if _effective_bit_kind(item[2].get("settings_data", {})) == bit_kind
    ]
    if not candidates:
        raise FileNotFoundError(
            "no saved AE matches "
            f"dim_ca1={dim_ca1}, num_cue_patterns={num_cue_patterns}, "
            f"train_noise={train_noise:.3f}, bit_kind={bit_kind}"
        )
    if any("bit_kind" not in item[2].get("settings_data", {})
           for item in candidates):
        warnings.warn(
            "legacy AE metadata without bit_kind interpreted as bit_kind=0",
            stacklevel=2,
        )

    def validation_loss(item) -> float:
        values = item[2].get("results", {}).get("test", [])
        return float(np.mean(values[-10:])) if values else np.inf

    return min(candidates, key=validation_loss)


def find_saved_mtl(*, dim_ca1: int, num_cue_patterns: int,
                   train_noise: float, bit_kind: int, plasticity: str):
    """Return the highest-fitness matching evolution record."""

    candidates = mtlct.find_mtl(
        dim_ca1=dim_ca1,
        num_cue_patterns=num_cue_patterns,
        noise_level=float(train_noise),
        plasticity=plasticity,
    )
    candidates = [
        item for item in candidates
        if _effective_bit_kind(item[1].get("settings", {})) == bit_kind
    ]
    if not candidates:
        raise FileNotFoundError(
            "no evolved MTL matches "
            f"dim_ca1={dim_ca1}, num_cue_patterns={num_cue_patterns}, "
            f"train_noise={train_noise:.3f}, bit_kind={bit_kind}, "
            f"plasticity={plasticity}"
        )
    if any("bit_kind" not in item[1].get("settings", {})
           for item in candidates):
        warnings.warn(
            "legacy MTL metadata without bit_kind interpreted as bit_kind=0",
            stacklevel=2,
        )
    return max(candidates, key=lambda item: float(item[1]["fitness"]))


def make_cue_data(num_samples: int, num_cue_patterns: int) -> np.ndarray:
    return dg.make_cue_track_data(
        num_samples=num_samples,
        size=50,
        num_cue_patterns=num_cue_patterns,
        lap_length=50,
        cue_positions=[10.0, 30.0],
        cue_sigma=3.0,
        cue_beta=40.0,
        cue_alpha=0.2,
        mec_binarized=True,
        mec_sigma=4.0,
        cue_spacing=1,
    ).astype(np.float32)


def corrupt(data: np.ndarray, noise_level: float, bit_kind: int) -> np.ndarray:
    operation = dg.bitflip if bit_kind == 0 else dg.bitkill
    return np.asarray(
        [operation(sample, fraction=noise_level) for sample in data],
        dtype=np.float32,
    )


def cosine_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    numerator = np.sum(x * y, axis=1)
    denominator = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0,
    )


def build_mtl(autoencoder: models.Autoencoder, record: dict,
              plasticity: str) -> models.MTL:
    parameters = dict(record["best_parameters"])
    weights = autoencoder.get_weights(bias=False)
    return models.MTL(
        W_ei_ca1=weights[0].clone(),
        W_ca1_eo=weights[1].clone(),
        K_ca1=autoencoder._K_ca1,
        K_eo=autoencoder._K_eo,
        K_ca3=int(parameters["K_ca3"]),
        dim_ca3=50,
        beta_is=autoencoder._beta_ei,
        beta_ca3=parameters["beta_ca3"],
        beta_ca1=parameters["beta_ca1"],
        beta_eo=autoencoder._beta_eo,
        alpha=parameters["alpha"],
        alpha_plus=parameters.get("alpha_plus"),
        alpha_minus=parameters.get("alpha_minus"),
        a_plus=parameters.get("a_plus", 0.0),
        b_plus=parameters.get("b_plus", 1.0),
        a_minus=parameters.get("a_minus", 0.0),
        b_minus=parameters.get("b_minus", 1.0),
        nb_ei_ca3=int(parameters.get("nb_ei_ca3", 10)),
        num_swaps_ca1=0,
        num_swaps_ca3=0,
        random_IS=False,
        B_ei_ca1=weights[2],
        B_ca1_eo=weights[3],
        plasticity=plasticity,
    )


def _forward_rows(model, data: np.ndarray) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for sample in data:
            tensor = torch.as_tensor(sample, dtype=torch.float32).reshape(-1, 1)
            outputs.append(model(tensor).reshape(-1).cpu().numpy())
    return np.asarray(outputs)


def evaluate_condition(*, dim_ca1: int, num_cue_patterns: int,
                       train_noise: float, test_noise: float, bit_kind: int,
                       plasticity: str, repetitions: int=8,
                       num_samples: int | None=None,
                       seed: int=73000) -> list[dict]:
    """Evaluate a saved AE/MTL pair against clean targets on paired draws."""

    ae_name, autoencoder, _ = find_saved_ae(
        dim_ca1=dim_ca1,
        num_cue_patterns=num_cue_patterns,
        train_noise=train_noise,
        bit_kind=bit_kind,
    )
    mtl_name, record = find_saved_mtl(
        dim_ca1=dim_ca1,
        num_cue_patterns=num_cue_patterns,
        train_noise=train_noise,
        bit_kind=bit_kind,
        plasticity=plasticity,
    )
    if num_samples is None:
        num_samples = 50 * num_cue_patterns

    rows = []
    state = np.random.get_state()
    try:
        for repetition in range(repetitions):
            np.random.seed(seed + repetition)
            torch.manual_seed(seed + repetition)
            clean = make_cue_data(num_samples, num_cue_patterns)
            storage_input = corrupt(clean, train_noise, bit_kind)
            probe_input = corrupt(clean, test_noise, bit_kind)

            autoencoder.eval()
            with torch.no_grad():
                ae_output = autoencoder(
                    torch.as_tensor(probe_input, dtype=torch.float32)
                ).cpu().numpy()

            model = build_mtl(autoencoder, record, plasticity)
            model.resume_lr()
            _forward_rows(model, storage_input)
            model.pause_lr()
            mtl_output = _forward_rows(model, probe_input)

            rows.append({
                "repetition": repetition,
                "seed": seed + repetition,
                "dim_ca1": dim_ca1,
                "num_cue_patterns": num_cue_patterns,
                "bit_kind": bit_kind,
                "plasticity": plasticity,
                "train_noise": float(train_noise),
                "test_noise": float(test_noise),
                "ae_name": ae_name,
                "mtl_name": mtl_name,
                "evolution_fitness": float(record["fitness"]),
                "input_cosine": float(cosine_rows(probe_input, clean).mean()),
                "ae_cosine": float(cosine_rows(ae_output, clean).mean()),
                "mtl_cosine": float(cosine_rows(mtl_output, clean).mean()),
                "ae_mse": float(np.mean((ae_output - clean) ** 2)),
                "mtl_mse": float(np.mean((mtl_output - clean) ** 2)),
            })
    finally:
        np.random.set_state(state)
    return rows


def save_rows(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot save an empty result table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_figure(fig, stem: str | Path) -> tuple[Path, Path]:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return png, pdf


def load_evolution_catalog() -> list[dict]:
    """Read saved MTL metadata without loading neural-network checkpoints."""

    from core.constants import MTL_PATH

    rows = []
    for path in sorted(Path(MTL_PATH).glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            settings = record["settings"]
        except (OSError, KeyError, json.JSONDecodeError):
            continue
        if settings.get("plasticity") not in PLASTICITY_VARIANTS:
            continue
        rows.append({
            "name": path.name,
            "dim_ca1": settings.get("dim_ca1"),
            "num_cue_patterns": settings.get("num_cue_patterns"),
            "train_noise": settings.get("noise_level"),
            "bit_kind": _effective_bit_kind(settings),
            "bit_kind_recorded": "bit_kind" in settings,
            "plasticity": settings.get("plasticity"),
            "fitness": record.get("fitness"),
        })
    return rows


def style_axis(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.2, linewidth=0.7)

