"""Capacity and tuning analysis with multiple cue-pair memories.

The experiment stores an increasing number of ordered cue pairs at two fixed
track positions.  It then measures clean and degraded pair recall, serial
retention, CA3 key collisions/recruitment, and an operational partition of CA1
tuning.  Both plasticity rules use their own evolved MTL parameters and share
the same held-out inputs, pretrained autoencoders, and random seeds.

All experiment choices are visible below.  In particular, changing ``N_CUES``
changes the cue vocabulary and therefore the maximum number of ordered pairs:

    N_CUES * (N_CUES - 1)

The script uses existing core model/training/data functions, but all protocol,
metrics, multiprocessing, and plotting logic needed for this experiment lives
in this file.

Quick check::

    python src/experiments/preprint_multiple_cues.py --quick --workers 2

Full run::

    python src/experiments/preprint_multiple_cues.py --workers 8
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch


SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core import datagen, models  # noqa: E402
from experiments.preprint_common import (  # noqa: E402
    build_mtl,
    legacy_numpy_seed,
    run_mtl,
    train_autoencoder,
)


# =============================================================================
# Main scientific choices
# =============================================================================

N_CUES = 8
SEEDS = list(range(53001, 53011))  # holdout seeds, not evolution seeds
RULES = ("base", "err2")
RULE_LABELS = {"base": "Instructive-driven", "err2": "Error-driven"}

LAPS_PER_PAIR = 3
LEC_DROP_FRACTION = 0.50
MASKS_PER_PAIR = 4

# CA1 tuning categories are descriptive.  A unit below this response variance
# is called neutral; otherwise the strongest reliable components determine its
# label.  All continuous component strengths are also saved to CSV.
NEUTRAL_VARIANCE_THRESHOLD = 1e-3
MIXED_SECOND_COMPONENT_RATIO = 0.50

MTL_RESULTS = ROOT_DIR / "results/preprint/mtl_evolution"
OUTPUT = ROOT_DIR / "results/preprint/multiple_cues"


def capacity_levels(number_cues: int) -> list[int]:
    """Return the full 1..N capacity sweep for non-repeating ordered pairs.

    Earlier versions used a sparse sweep such as 1, 2, 4, 22, ... to make
    exploratory runs faster.  For the preprint panel it is easier to read the
    same metric directly when every possible stored-pair count is sampled.
    """

    maximum = number_cues * (number_cues - 1)
    return list(range(1, maximum + 1))


# =============================================================================
# Deterministic data and model preparation
# =============================================================================

def seed_value(seed: int, *stream: int) -> int:
    """Create a deterministic independent seed for one operation."""

    return int(np.random.SeedSequence([seed, *stream]).generate_state(1)[0])


def ordered_pairs(number_cues: int, seed: int) -> list[tuple[int, int]]:
    """Create all A/B assignments with different identities in balanced rounds.

    Each offset round uses every cue exactly once at A and once at B.  Shuffling
    within rounds varies serial order without destroying balance at capacities
    that include a complete round.
    """

    rng = np.random.default_rng(seed)
    pairs = []
    offsets = np.arange(1, number_cues)
    rng.shuffle(offsets)
    for offset in offsets:
        round_pairs = [(cue, (cue + int(offset)) % number_cues)
                       for cue in range(number_cues)]
        rng.shuffle(round_pairs)
        pairs.extend(round_pairs)
    return pairs


def cue_laps(pair_sequence: list[tuple[int, int]], cue_patterns: np.ndarray,
             track: dict[str, Any], seed: int) -> np.ndarray:
    """Generate MEC+LEC laps for an explicit sequence of cue pairs."""

    arguments = {
        "n": len(pair_sequence),
        "length": track["lap_length"],
        "cues_positions": track["cue_positions"],
        "cues_patterns": cue_patterns,
        "cues_sequence": [list(pair) for pair in pair_sequence],
        "cue_sigma": track["cue_sigma"],
        "cue_beta": track["cue_beta"],
        "cue_alpha": track["cue_alpha"],
        "mec_binarized": track["mec_binarized"],
    }
    with legacy_numpy_seed(seed):
        values, _ = datagen.sparse_stimulus_generator_sensory(
            laps=arguments,
            mec_size=track["size"] // 2,
            mec_sigma=track["mec_sigma"],
            lec_sigma=track["lec_sigma"],
        )
    return values.astype(np.float32)


def pack_autoencoder(model: models.Autoencoder) -> dict[str, Any]:
    """Serialize a frozen autoencoder into NumPy arrays for worker processes."""

    return {
        "state": {name: value.detach().cpu().numpy()
                  for name, value in model.state_dict().items()},
        "dim_ei": model._dim_ei,
        "dim_ca1": model._dim_ca1,
        "K_ca1": model._K_ca1,
        "K_eo": model._K_eo,
        "beta_ei": model._beta_ei,
        "beta_eo": model._beta_eo,
        "use_bias": model._use_bias,
    }


def unpack_autoencoder(bundle: dict[str, Any]) -> models.Autoencoder:
    """Reconstruct one frozen autoencoder inside a worker."""

    model = models.Autoencoder(
        dim_ei=bundle["dim_ei"], dim_ca1=bundle["dim_ca1"],
        K_ca1=bundle["K_ca1"], K_eo=bundle["K_eo"],
        beta_ei=bundle["beta_ei"], beta_eo=bundle["beta_eo"],
        use_bias=bundle["use_bias"],
    )
    model.load_state_dict({name: torch.as_tensor(value, dtype=torch.float32)
                           for name, value in bundle["state"].items()})
    model.eval()
    return model


def load_configuration(results: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load the evolved AE/MTL settings and best values for both rules."""

    settings = None
    memories = {}
    for rule in RULES:
        config = json.loads((results / rule / "config.json").read_text())
        rule_settings = config["settings"]
        if settings is None:
            settings = rule_settings
        elif rule_settings != settings:
            raise ValueError("the two evolved searches did not share model/data settings")

        best = json.loads((results / rule / "best_parameters.json").read_text())
        memories[rule] = {
            "ca3_dimension": settings["memory"]["ca3_dimension"],
            "ca3_inputs_per_unit": int(best["ca3_inputs_per_unit"]),
            "k_ca3": int(best["k_ca3"]),
            "k_ca1": settings["memory"]["k_ca1"],
            "beta_ca3": float(best["beta_ca3"]),
            "beta_ca1": float(best["beta_ca1"]),
            "alpha": float(best["alpha"]),
            "plasticity_rule": rule,
        }
    assert settings is not None
    return settings, memories


def prepare_seed(seed: int, settings: dict[str, Any], pairs: list[tuple[int, int]],
                 laps_per_pair: int) -> dict[str, Any]:
    """Generate all pair laps and pretrain one capacity-independent AE basis."""

    track = settings["track"]
    cue_patterns = datagen.make_cues(N_CUES, settings["dimension"] // 2, fixed=True)

    # One deterministic storage lap per (pair, repetition).  Capacity conditions
    # select from this common bank instead of regenerating different tracks.
    storage = np.empty((len(pairs), laps_per_pair, track["lap_length"], track["size"]),
                       dtype=np.float32)
    for pair_index, pair in enumerate(pairs):
        storage[pair_index] = cue_laps(
            [pair] * laps_per_pair, cue_patterns, track,
            seed_value(seed, 10, pair_index),
        )
    probes = np.stack([
        cue_laps([pair], cue_patterns, track, seed_value(seed, 20, pair_index))[0]
        for pair_index, pair in enumerate(pairs)
    ])
    # The AE sees the whole cue vocabulary before any capacity condition.  Its
    # encoding basis therefore stays fixed as the associative load increases.
    ae_training = storage.reshape(-1, track["size"])
    ae_validation = probes.reshape(-1, track["size"])
    autoencoder, ae_mse = train_autoencoder(
        ae_training, ae_validation, settings, seed_value(seed, 40)
    )
    return {
        "seed": seed,
        "pairs": pairs,
        "cue_patterns": cue_patterns,
        "storage": storage,
        "probes": probes,
        "autoencoder": pack_autoencoder(autoencoder),
        "ae_validation_mse": ae_mse,
    }


# =============================================================================
# Recall, key-code, and tuning metrics
# =============================================================================

def decode_pair(output: np.ndarray, cue_patterns: np.ndarray,
                positions: list[int]) -> tuple[int, int]:
    """Decode cue identity independently at positions A and B by cosine."""

    lec = output[np.asarray(positions), output.shape[1] // 2:]
    similarity = lec @ cue_patterns.T
    similarity /= np.maximum(
        np.linalg.norm(lec, axis=1)[:, None]
        * np.linalg.norm(cue_patterns, axis=1)[None, :],
        1e-12,
    )
    prediction = np.argmax(similarity, axis=1)
    return int(prediction[0]), int(prediction[1])


def degrade_lec(values: np.ndarray, fraction: float,
                rng: np.random.Generator) -> np.ndarray:
    """Remove one fixed subset of LEC units for the entire probe lap."""

    result = values.copy()
    half = values.shape[1] // 2
    candidates = np.arange(half, values.shape[1])
    count = int(round(fraction * len(candidates)))
    removed = rng.choice(candidates, size=count, replace=False)
    result[:, removed] = 0.0
    return result


def key_statistics(ca3_by_pair: np.ndarray, cue_positions: list[int],
                   k_ca3: int) -> tuple[float, float]:
    """Measure pair-signature collisions and CA3 population recruitment."""

    signatures = []
    recruited = set()
    for activity in ca3_by_pair:
        signature = []
        for position in cue_positions:
            winners = np.argsort(activity[position])[-k_ca3:]
            winners = tuple(sorted(int(index) for index in winners))
            signature.extend(winners)
            recruited.update(winners)
        signatures.append(tuple(signature))
    collision_rate = 1.0 - len(set(signatures)) / len(signatures)
    recruitment = len(recruited) / ca3_by_pair.shape[-1]
    return float(collision_rate), float(recruitment)


def tuning_partition(ca1: np.ndarray, no_cue_ca1: np.ndarray,
                     pairs: list[tuple[int, int]], positions: list[int]) -> tuple[dict[str, int], list[dict]]:
    """Assign transparent operational CA1 tuning labels.

    ``ca1`` has shape (pairs, positions, units).  Four continuous components
    are calculated per unit:

    * spatial: variance of the pair-averaged position curve,
    * cue-general: mean cue-evoked change relative to a no-cue lap,
    * cue-specific: variance among cue-identity means,
    * interaction: residual pair-by-position variance.

    Low-variance units are neutral.  A unit is mixed when its second-largest
    component is at least half its largest; otherwise its largest component
    supplies the label.  The continuous values are saved, so the categorical
    summary never hides its threshold definition.
    """

    pair_array = np.asarray(pairs)
    number_units = ca1.shape[-1]
    spatial = np.var(ca1.mean(axis=0), axis=0)
    position_mean = ca1.mean(axis=0, keepdims=True)
    pair_mean = ca1.mean(axis=1, keepdims=True)
    grand_mean = ca1.mean(axis=(0, 1), keepdims=True)
    interaction = np.mean((ca1 - position_mean - pair_mean + grand_mean) ** 2,
                          axis=(0, 1))

    event_responses = []
    identities = []
    for pair_index, pair in enumerate(pair_array):
        for event_index, position in enumerate(positions):
            event_responses.append(
                ca1[pair_index, position] - no_cue_ca1[pair_index, position]
            )
            identities.append(pair[event_index])
    event_responses = np.asarray(event_responses)
    identities = np.asarray(identities)
    # At low capacities the selected pair prefix may not yet contain every cue
    # identity.  Estimate selectivity only across identities actually observed;
    # absent identities carry no response samples and must not create NaNs.
    present_identities = np.unique(identities)
    identity_means = np.stack([
        event_responses[identities == cue].mean(axis=0)
        for cue in present_identities
    ])
    cue_general = np.mean(identity_means, axis=0) ** 2
    cue_specific = np.var(identity_means, axis=0)

    component_names = ("spatial", "cue-general", "cue-specific", "interaction")
    components = np.stack((spatial, cue_general, cue_specific, interaction), axis=1)
    total_variance = np.var(ca1, axis=(0, 1))
    counts = {name: 0 for name in ("spatial", "cue-general", "cue-specific", "mixed", "neutral")}
    rows = []
    for unit in range(number_units):
        strengths = components[unit]
        if total_variance[unit] < NEUTRAL_VARIANCE_THRESHOLD:
            category = "neutral"
        else:
            order = np.argsort(strengths)
            if strengths[order[-2]] >= MIXED_SECOND_COMPONENT_RATIO * max(strengths[order[-1]], 1e-12):
                category = "mixed"
            else:
                winner = component_names[order[-1]]
                category = "mixed" if winner == "interaction" else winner
        counts[category] += 1
        rows.append({
            "unit": unit,
            "category": category,
            "total_variance": total_variance[unit],
            **dict(zip(component_names, strengths)),
        })
    return counts, rows


# =============================================================================
# One independent simulation job
# =============================================================================

_PREPARED: dict[int, dict[str, Any]] | None = None
_SETTINGS: dict[str, Any] | None = None
_MEMORIES: dict[str, dict[str, Any]] | None = None


def initialize_worker(prepared: dict[int, dict[str, Any]], settings: dict[str, Any],
                      memories: dict[str, dict[str, Any]]) -> None:
    global _PREPARED, _SETTINGS, _MEMORIES
    _PREPARED, _SETTINGS, _MEMORIES = prepared, settings, memories
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def run_condition(job: tuple[str, int, int]) -> dict[str, Any]:
    """Store and probe one (rule, seed, capacity) condition."""

    if _PREPARED is None or _SETTINGS is None or _MEMORIES is None:
        raise RuntimeError("worker was not initialized")
    rule, seed, capacity = job
    item = _PREPARED[seed]
    track = _SETTINGS["track"]
    selected_pairs = item["pairs"][:capacity]

    model = build_mtl(
        unpack_autoencoder(item["autoencoder"]),
        _MEMORIES[rule],
        # Both rules receive the same random wiring stream.  Their evolved
        # fan-in/K values differ, but no extra rule-specific RNG difference is
        # introduced on top of those intended parameter differences.
        seed_value(seed, 100),
    )

    # Present every selected pair once per cycle.  The final-cycle order defines
    # memory age while maintaining equal exposure across pairs.
    storage_order = []
    for repetition in range(item["storage"].shape[1]):
        cycle = np.arange(capacity)
        np.random.default_rng(seed_value(seed, 110, repetition)).shuffle(cycle)
        for pair_index in cycle:
            run_mtl(model, item["storage"][pair_index, repetition], learn=True)
            storage_order.append(int(pair_index))
    last_seen = {pair_index: max(index for index, value in enumerate(storage_order)
                                 if value == pair_index)
                 for pair_index in range(capacity)}
    final_write = len(storage_order) - 1

    pair_rows = []
    ca1_by_pair, no_cue_ca1_by_pair, ca3_by_pair = [], [], []
    for pair_index, target_pair in enumerate(selected_pairs):
        probe = item["probes"][pair_index]
        clean_output, ca1, ca3 = run_mtl(model, probe, learn=False)
        clean_prediction = decode_pair(clean_output, item["cue_patterns"],
                                       track["cue_positions"])
        ca1_by_pair.append(ca1)
        ca3_by_pair.append(ca3)

        # Zeroing LEC in this exact probe retains its MEC realization.  The
        # resulting CA1 difference is therefore attributable to cue input, not
        # to an independently sampled spatial pattern.
        cue_absent_probe = probe.copy()
        cue_absent_probe[:, cue_absent_probe.shape[1] // 2:] = 0.0
        _, no_cue_ca1, _ = run_mtl(model, cue_absent_probe, learn=False)
        no_cue_ca1_by_pair.append(no_cue_ca1)

        degraded_predictions = []
        for mask_index in range(MASKS_PER_PAIR):
            degraded = degrade_lec(
                probe,
                LEC_DROP_FRACTION,
                np.random.default_rng(seed_value(seed, 120, pair_index, mask_index)),
            )
            output, _, _ = run_mtl(model, degraded, learn=False)
            degraded_predictions.append(
                decode_pair(output, item["cue_patterns"], track["cue_positions"])
            )

        pair_rows.append({
            "target_pair": tuple(target_pair),
            "clean_prediction": clean_prediction,
            "clean_exact": float(clean_prediction == tuple(target_pair)),
            "degraded_predictions": degraded_predictions,
            "degraded_exact": float(np.mean([
                prediction == tuple(target_pair) for prediction in degraded_predictions
            ])),
            "memory_age": final_write - last_seen[pair_index],
        })

    ca1_by_pair = np.asarray(ca1_by_pair)
    no_cue_ca1_by_pair = np.asarray(no_cue_ca1_by_pair)
    ca3_by_pair = np.asarray(ca3_by_pair)
    collision, recruitment = key_statistics(
        ca3_by_pair, track["cue_positions"], _MEMORIES[rule]["k_ca3"]
    )
    categories, tuning_rows = tuning_partition(
        ca1_by_pair, no_cue_ca1_by_pair, selected_pairs, track["cue_positions"]
    )
    return {
        "rule": rule,
        "seed": seed,
        "capacity": capacity,
        "ae_validation_mse": item["ae_validation_mse"],
        "clean_pair_accuracy": float(np.mean([row["clean_exact"] for row in pair_rows])),
        "degraded_pair_accuracy": float(np.mean([row["degraded_exact"] for row in pair_rows])),
        "collision_rate": collision,
        "recruitment_fraction": recruitment,
        "pair_rows": pair_rows,
        "categories": categories,
        "tuning_rows": tuning_rows,
    }


# =============================================================================
# Parallel execution and source-data organization
# =============================================================================

def run_parallel(jobs: list[tuple[str, int, int]], prepared: dict, settings: dict,
                 memories: dict, workers: int) -> list[dict[str, Any]]:
    """Run independent conditions in a spawn-based multicore pool."""

    if workers == 1:
        initialize_worker(prepared, settings, memories)
        return [run_condition(job) for job in jobs]

    variables = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
    previous = {name: os.environ.get(name) for name in variables}
    try:
        for name in variables:
            os.environ[name] = "1"
        with mp.get_context("spawn").Pool(
            processes=workers,
            initializer=initialize_worker,
            initargs=(prepared, settings, memories),
        ) as pool:
            return pool.map(run_condition, jobs, chunksize=1)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def flatten_results(results: list[dict[str, Any]]) -> tuple[list[dict], list[dict], list[dict]]:
    """Create condition, pair-level, and unit-level tidy tables."""

    conditions, pairs, units = [], [], []
    for result in results:
        common = {key: result[key] for key in ("rule", "seed", "capacity")}
        conditions.append({
            **common,
            "plasticity_label": RULE_LABELS[result["rule"]],
            "ae_validation_mse": result["ae_validation_mse"],
            "clean_pair_accuracy": result["clean_pair_accuracy"],
            "degraded_pair_accuracy": result["degraded_pair_accuracy"],
            "collision_rate": result["collision_rate"],
            "recruitment_fraction": result["recruitment_fraction"],
            **{f"fraction_{category}": count / 50
               for category, count in result["categories"].items()},
        })
        for row in result["pair_rows"]:
            target = row["target_pair"]
            for mask_index, prediction in enumerate(row["degraded_predictions"]):
                pairs.append({
                    **common,
                    "target_a": target[0], "target_b": target[1],
                    "clean_predicted_a": row["clean_prediction"][0],
                    "clean_predicted_b": row["clean_prediction"][1],
                    "clean_exact": row["clean_exact"],
                    "mask_index": mask_index,
                    "degraded_predicted_a": prediction[0],
                    "degraded_predicted_b": prediction[1],
                    "degraded_exact": float(prediction == target),
                    "memory_age": row["memory_age"],
                })
        for row in result["tuning_rows"]:
            units.append({**common, **row})
    return conditions, pairs, units


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Plotting
# =============================================================================

COLORS = {"base": "#4c78a8", "err2": "#e45756"}


def mean_sem(values: list[float]) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return float(values.mean()), float(sem)


def finish_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.18)


def save_plot(figure: plt.Figure, output: Path, name: str) -> None:
    figure.savefig(output / f"{name}.svg", bbox_inches="tight")
    figure.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_capacity(axis: plt.Axes, conditions: list[dict], levels: list[int]) -> None:
    for rule in RULES:
        for metric, linestyle, label in (
            ("clean_pair_accuracy", "-", "clean"),
            ("degraded_pair_accuracy", "--", f"{int(100 * LEC_DROP_FRACTION)}% LEC drop"),
        ):
            means, sems = [], []
            for capacity in levels:
                values = [row[metric] for row in conditions
                          if row["rule"] == rule and row["capacity"] == capacity]
                mean, sem = mean_sem(values)
                means.append(mean); sems.append(sem)
            axis.errorbar(levels, means, yerr=sems, marker="o", linestyle=linestyle,
                          color=COLORS[rule], capsize=2,
                          label=f"{RULE_LABELS[rule]}, {label}")
    axis.set(xlabel="Stored ordered cue pairs", ylabel="Exact pair accuracy",
             ylim=(-0.03, 1.03), title="A  Cue-pair capacity")
    axis.legend(frameon=False, fontsize=7, loc="upper right")
    finish_axis(axis)


def plot_preprint_capacity(output: Path, conditions: list[dict],
                           levels: list[int]) -> None:
    """Save a compact manuscript panel linking capacity and degraded recall.

    Color identifies the plasticity rule. Open solid markers show clean recall;
    filled dashed markers show the same memories after 50% LEC-unit dropout.
    The x-axis is kept linear so adjacent points differ by one stored pair.
    """

    figure, axis = plt.subplots(figsize=(5.0, 3.45), constrained_layout=True)
    for rule in RULES:
        for metric, linestyle, filled in (
            ("clean_pair_accuracy", "-", False),
            ("degraded_pair_accuracy", "--", True),
        ):
            means, sems = [], []
            for capacity in levels:
                values = [float(row[metric]) for row in conditions
                          if row["rule"] == rule and int(row["capacity"]) == capacity]
                mean, sem = mean_sem(values)
                means.append(mean); sems.append(sem)
            axis.errorbar(
                levels, means, yerr=sems,
                color=COLORS[rule], linestyle=linestyle, marker="o",
                markerfacecolor=COLORS[rule] if filled else "white",
                markeredgecolor=COLORS[rule], markeredgewidth=1.4,
                linewidth=1.8, markersize=5.5, capsize=2.2,
            )

    chance = 1.0 / (N_CUES ** 2)
    axis.axhline(chance, color="0.55", linestyle=":", linewidth=1)
    axis.text(levels[-1], chance + 0.018, f"chance = {chance:.2f}",
              color="0.45", fontsize=7, ha="right")
    axis.set_xlim(1, levels[-1])
    axis.set_xticks(np.linspace(1, levels[-1], 6, dtype=int))
    axis.set(
        xlabel="Stored ordered cue pairs",
        ylabel="Exact cue-pair recall",
        ylim=(-0.025, 1.04),
        title="Cue-pair load and partial-cue recall",
    )

    rule_handles = [
        Line2D([0], [0], color=COLORS[rule], linewidth=2,
               label=RULE_LABELS[rule])
        for rule in RULES
    ]
    condition_handles = [
        Line2D([0], [0], color="0.25", marker="o", markerfacecolor="white",
               linestyle="-", label="Clean input"),
        Line2D([0], [0], color="0.25", marker="o", markerfacecolor="0.25",
               linestyle="--", label=f"{int(100 * LEC_DROP_FRACTION)}% LEC drop"),
    ]
    first_legend = axis.legend(handles=rule_handles, frameon=False, fontsize=7,
                               loc="lower left")
    axis.add_artist(first_legend)
    axis.legend(handles=condition_handles, frameon=False, fontsize=7,
                loc="upper right")
    finish_axis(axis)
    save_plot(figure, output, "plot_multiple_cues_preprint_capacity")


def plot_confusion(axes: list[plt.Axes], pair_rows: list[dict], all_pairs: list[tuple[int, int]],
                   maximum: int) -> list:
    displayed_pairs = all_pairs[:maximum]
    lookup = {pair: index for index, pair in enumerate(displayed_pairs)}
    images = []
    for axis, rule in zip(axes, RULES):
        # The final column catches decoded repeated-cue pairs (for example
        # 2-2), which are not members of the stored non-repeating pair set.
        matrix = np.zeros((maximum, maximum + 1), dtype=float)
        totals = np.zeros(maximum, dtype=float)
        selected = [row for row in pair_rows
                    if row["rule"] == rule and row["capacity"] == maximum]
        for row in selected:
            target = (int(row["target_a"]), int(row["target_b"]))
            predicted = (int(row["degraded_predicted_a"]), int(row["degraded_predicted_b"]))
            target_index = lookup[target]
            totals[target_index] += 1
            if predicted in lookup:
                matrix[target_index, lookup[predicted]] += 1
            else:
                matrix[target_index, -1] += 1
        matrix /= np.maximum(totals[:, None], 1)
        image = axis.imshow(matrix, cmap="magma", vmin=0, vmax=1, aspect="auto")
        images.append(image)
        axis.set(xlabel="Decoded pair", ylabel="Stored pair",
                 title=RULE_LABELS[rule])
        ticks = np.arange(maximum)
        labels = [f"{a+1}-{b+1}" for a, b in displayed_pairs]
        axis.set_xticks(np.arange(maximum + 1), labels + ["other"],
                        rotation=90, fontsize=5)
        axis.set_yticks(ticks, labels, fontsize=5)
    return images


def plot_retention(axis: plt.Axes, pair_rows: list[dict], maximum: int) -> None:
    for rule in RULES:
        selected = [row for row in pair_rows
                    if row["rule"] == rule and row["capacity"] == maximum]
        ages = sorted(set(int(row["memory_age"]) for row in selected))
        means = [np.mean([row["degraded_exact"] for row in selected
                          if int(row["memory_age"]) == age]) for age in ages]
        axis.plot(ages, means, marker="o", color=COLORS[rule],
                  label=RULE_LABELS[rule])
    axis.set(xlabel="Writes since pair was last presented", ylabel="Exact degraded recall",
             ylim=(-0.03, 1.03), title="C  Serial retention")
    axis.legend(frameon=False, fontsize=7)
    finish_axis(axis)


def plot_key_code(axis: plt.Axes, conditions: list[dict], levels: list[int]) -> None:
    for rule in RULES:
        for metric, linestyle, marker, label in (
            ("collision_rate", "-", "o", "key collisions"),
            ("recruitment_fraction", "--", "s", "unit recruitment"),
        ):
            means = [np.mean([row[metric] for row in conditions
                              if row["rule"] == rule and row["capacity"] == capacity])
                     for capacity in levels]
            axis.plot(levels, means, linestyle=linestyle, marker=marker,
                      color=COLORS[rule], label=f"{RULE_LABELS[rule]}, {label}")
    axis.set(xlabel="Stored ordered cue pairs", ylabel="Fraction", ylim=(-0.03, 1.03),
             title="D  CA3 code utilization")
    axis.legend(frameon=False, fontsize=6)
    finish_axis(axis)


def plot_tuning(axis: plt.Axes, conditions: list[dict], maximum: int) -> None:
    categories = ("spatial", "cue-general", "cue-specific", "mixed", "neutral")
    colors = ("#4c78a8", "#59a14f", "#f28e2b", "#b07aa1", "#bab0ac")
    bottom = np.zeros(len(RULES))
    for category, color in zip(categories, colors):
        values = [np.mean([row[f"fraction_{category}"] for row in conditions
                           if row["rule"] == rule and row["capacity"] == maximum])
                  for rule in RULES]
        axis.bar(range(len(RULES)), values, bottom=bottom, color=color, label=category)
        bottom += values
    axis.set(xticks=range(len(RULES)), xticklabels=[RULE_LABELS[rule] for rule in RULES],
             ylabel="Fraction of CA1 units", ylim=(0, 1),
             title="E  CA1 tuning at maximum load")
    axis.legend(frameon=False, fontsize=6, ncol=2)
    finish_axis(axis)


def build_plots(output: Path, conditions: list[dict], pair_rows: list[dict],
                pairs: list[tuple[int, int]], levels: list[int]) -> None:
    maximum = levels[-1]

    figure, axis = plt.subplots(figsize=(5.0, 3.5), constrained_layout=True)
    plot_capacity(axis, conditions, levels)
    save_plot(figure, output, "plot_multiple_cues_a_capacity")
    plot_preprint_capacity(output, conditions, levels)

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)
    images = plot_confusion(list(axes), pair_rows, pairs, maximum)
    colorbar = figure.colorbar(images[0], ax=axes.ravel().tolist(), shrink=0.8)
    colorbar.set_label("Recall probability")
    figure.suptitle(f"B  Pair confusion at capacity {maximum} ({int(100 * LEC_DROP_FRACTION)}% LEC drop)")
    save_plot(figure, output, "plot_multiple_cues_b_confusion")

    figure, axis = plt.subplots(figsize=(5.0, 3.5), constrained_layout=True)
    plot_retention(axis, pair_rows, maximum)
    save_plot(figure, output, "plot_multiple_cues_c_retention")

    figure, axis = plt.subplots(figsize=(5.0, 3.5), constrained_layout=True)
    plot_key_code(axis, conditions, levels)
    save_plot(figure, output, "plot_multiple_cues_d_key_code")

    figure, axis = plt.subplots(figsize=(4.7, 3.5), constrained_layout=True)
    plot_tuning(axis, conditions, maximum)
    save_plot(figure, output, "plot_multiple_cues_e_tuning")


# =============================================================================
# Command-line entry point
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mtl-results", type=Path, default=MTL_RESULTS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if N_CUES < 2:
        raise ValueError("N_CUES must be at least 2")
    if N_CUES > 25:
        raise ValueError("fixed cue patterns require N_CUES <= LEC dimension (25)")
    if args.workers < 1:
        raise ValueError("workers must be positive")

    settings, memories = load_configuration(args.mtl_results)
    # Copy nested settings before quick-mode changes.
    settings = json.loads(json.dumps(settings))
    seeds = list(SEEDS)
    levels = capacity_levels(N_CUES)
    laps_per_pair = LAPS_PER_PAIR
    output = args.output
    if args.quick:
        seeds = seeds[:1]
        levels = levels[:3]
        laps_per_pair = 1
        settings["autoencoder"]["epochs"] = 3
        if args.output == OUTPUT:
            output = ROOT_DIR / "results/preprint/multiple_cues_quick"

    print(f"Preparing {N_CUES} cues and {len(seeds)} independent encoding bases...")
    pair_orders = {seed: ordered_pairs(N_CUES, seed_value(seed, 1)) for seed in seeds}
    prepared = {
        seed: prepare_seed(seed, settings, pair_orders[seed], laps_per_pair)
        for seed in seeds
    }
    jobs = [(rule, seed, capacity) for rule in RULES for seed in seeds for capacity in levels]
    workers = min(args.workers, len(jobs), os.cpu_count() or 1)
    print(f"Running {len(jobs)} conditions on {workers} workers...")
    results = run_parallel(jobs, prepared, settings, memories, workers)
    conditions, pair_rows, unit_rows = flatten_results(results)

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "condition_data.csv", conditions)
    write_csv(output / "pair_recall_data.csv", pair_rows)
    write_csv(output / "unit_tuning_data.csv", unit_rows)
    config = {
        "N_CUES": N_CUES,
        "maximum_ordered_pairs_without_repetition": N_CUES * (N_CUES - 1),
        "capacity_levels": levels,
        "seeds": seeds,
        "rules": list(RULES),
        "laps_per_pair": laps_per_pair,
        "lec_drop_fraction": LEC_DROP_FRACTION,
        "masks_per_pair": MASKS_PER_PAIR,
        "neutral_variance_threshold": NEUTRAL_VARIANCE_THRESHOLD,
        "mixed_second_component_ratio": MIXED_SECOND_COMPONENT_RATIO,
        "workers": workers,
        "mtl_results": str(args.mtl_results.resolve()),
        "memory_parameters": memories,
        "model_settings": settings,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    build_plots(output, conditions, pair_rows, pair_orders[seeds[0]], levels)
    print(f"Saved source data and plots to {output}")


if __name__ == "__main__":
    main()
