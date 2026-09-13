"""Measure CA1 reconfiguration dynamics around repeated cue exchanges.

This experiment extends the cue-schedule protocol from ``preprint_figure_2``
without changing the model.  It asks a more specific temporal question:

    After the two cue identities exchange positions, does recalled CA1
    activity immediately adopt a new map, change over several laps, or return
    to its previous state?

For each root seed, one pretrained EC-CA1-EC basis is shared by two schedules:

``swap``
    Cue identities exchange positions every ten laps.

``no_swap``
    Cue identities remain fixed, but the same lap boundaries are analyzed.

The schedules share their MEC trajectory, probe inputs, plasticity parameters,
and EC-CA3 wiring.  The instructive-driven (internal identifier ``base``) and
error-driven (``err2``) rules also receive these same inputs and wiring.

After every training lap, plasticity is paused and CA1 is probed in the
context scheduled for that lap.  Three event-aligned measurements are then
computed:

``old_map_similarity``
    Similarity to the last probe immediately before the exchange.

``new_map_similarity``
    Similarity to the final probe in the new ten-lap cue block.

``step_similarity``
    Similarity to the preceding lap, which measures the instantaneous size of
    each update without choosing an old or new reference map.

The output directory contains ``arrays.npz``, ``config.json``, tidy
``source_data.csv``, and four plot families suitable for figure composition:

``plot_remap_a``
    Lap-by-lap population-map similarity matrix.

``plot_remap_b``
    Event-aligned old-map persistence and new-map convergence.

``plot_remap_c``
    Event-aligned step similarity for swap and no-swap schedules.

``plot_remap_d``
    Seed-averaged distribution of unit-level old-to-endpoint similarity.

Run from the repository root:

    python src/experiments/preprint_remapping_dynamics.py

Use ``--quick`` for a two-seed smoke test.  Quick-run values must not be used
in the manuscript.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.preprint_common import (
    build_mtl,
    cue_track,
    row_cosine,
    run_mtl,
    train_autoencoder,
    write_artifact,
)


RULE_LABELS = {"base": "Instructive-driven", "err2": "Error-driven"}
SCHEDULE_LABELS = {"swap": "Cue exchange", "no_swap": "No-swap control"}
SCHEDULE_COLORS = {"swap": "#6a3d9a", "no_swap": "0.40"}


SETTINGS = {
    "seeds": list(range(51001, 51021)),
    "plasticity_rules": ["base", "err2"],
    "dimension": 50,
    "active": 5,
    "autoencoder": {
        "latent_dimension": 50,
        "beta_latent": 25.0,
        "beta_output": 25.0,
        "epochs": 256,
        "batch_size": 64,
        "learning_rate": 0.001,
    },
    "memory": {
        "ca3_dimension": 50,
        "ca3_inputs_per_unit": 29,
        "k_ca3": 3,
        "k_ca1": 5,
        "beta_ca3": 170.6,
        "beta_ca1": 41.8,
        "alpha": 0.092,
        "plasticity_rule": "err2",
    },
    "track": {
        "training_laps": 40,
        "validation_laps": 10,
        "lap_length": 50,
        "size": 50,
        "cue_positions": [10, 30],
        "swap_every": 10,
        "cue_sigma": 4.0,
        "cue_beta": 40.0,
        "cue_alpha": 0.1,
        "mec_binarized": True,
        "mec_sigma": 5.0,
        "lec_sigma": 5.0,
    },
    # Relative lap zero is the first lap learned after a cue exchange.  The
    # positive window ends at +9, the end of that ten-lap cue block.
    "analysis": {"relative_laps": list(range(-3, 10))},
}


def seed_value(root_seed: int, stream: int) -> int:
    """Derive deterministic, independent random streams from one root seed."""

    return int(np.random.SeedSequence([root_seed, stream]).generate_state(1)[0])


def context_schedule(laps: int, swap_every: int, swap: bool) -> list[list[int]]:
    """Return the cue identities occupying the two fixed cue positions."""

    if not swap:
        return [[0, 1]] * laps
    return [
        [0, 1] if (lap // swap_every) % 2 == 0 else [1, 0]
        for lap in range(laps)
    ]


def tuning_similarity(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Compare two position-by-unit maps, returning one value per CA1 unit."""

    first_centered = first - first.mean(axis=0, keepdims=True)
    second_centered = second - second.mean(axis=0, keepdims=True)
    return row_cosine(first_centered.T, second_centered.T)


def population_similarity(fields: np.ndarray) -> np.ndarray:
    """Return a lap-by-lap similarity matrix for complete CA1 population maps."""

    # Center every unit over track position before flattening.  The resulting
    # comparison is sensitive to field shape and position, not only mean rate.
    centered = fields - fields.mean(axis=1, keepdims=True)
    flattened = centered.reshape(centered.shape[0], -1)
    numerator = flattened @ flattened.T
    norm = np.linalg.norm(flattened, axis=1)
    return (numerator / np.maximum(norm[:, None] * norm[None, :], 1e-12)).astype(np.float32)


def prepare_seed(seed: int, settings: dict) -> dict:
    """Generate paired tracks, fixed probes, and one shared pretrained basis."""

    track = settings["track"]
    laps = track["training_laps"]
    swap_every = track["swap_every"]
    swap_schedule = context_schedule(laps, swap_every, swap=True)
    no_swap_schedule = context_schedule(laps, swap_every, swap=False)

    # A shared generator seed gives the schedules the same MEC trajectory.
    # Cue positions are also identical; only the cue identities are exchanged.
    track_seed = seed_value(seed, 1)
    swap_laps = cue_track(laps, track, swap_schedule, track_seed)
    no_swap_laps = cue_track(laps, track, no_swap_schedule, track_seed)
    mec_units = track["size"] // 2
    if not np.array_equal(swap_laps[:, :, :mec_units], no_swap_laps[:, :, :mec_units]):
        raise RuntimeError("The no-swap control did not preserve the MEC trajectory.")

    validation = cue_track(
        track["validation_laps"],
        track,
        [[0, 1]] * track["validation_laps"],
        seed_value(seed, 2),
    )
    autoencoder, validation_mse = train_autoencoder(
        swap_laps.reshape(-1, track["size"]),
        validation.reshape(-1, track["size"]),
        settings,
        seed_value(seed, 3),
    )

    # Both probes share their random MEC and cue-presence draws.  Reversing the
    # assignment changes cue identity while holding all other probe structure.
    probe_seed = seed_value(seed, 4)
    probe_a = cue_track(1, track, [[0, 1]], probe_seed)[0]
    probe_b = cue_track(1, track, [[1, 0]], probe_seed)[0]
    if not np.array_equal(probe_a[:, :mec_units], probe_b[:, :mec_units]):
        raise RuntimeError("The two cue-context probes did not preserve MEC input.")

    return {
        "autoencoder": autoencoder,
        "validation_mse": validation_mse,
        "laps": (swap_laps, no_swap_laps),
        "schedules": (swap_schedule, no_swap_schedule),
        "probes": np.stack((probe_a, probe_b)),
    }


def event_measurements(
    fields: np.ndarray,
    event_laps: np.ndarray,
    relative_laps: np.ndarray,
    block_length: int,
) -> dict[str, np.ndarray]:
    """Calculate event-aligned map persistence, convergence, and step size."""

    shape = (len(event_laps), len(relative_laps))
    old_similarity = np.empty(shape, dtype=np.float32)
    new_similarity = np.empty(shape, dtype=np.float32)
    step_similarity = np.empty(shape, dtype=np.float32)
    endpoint_units = np.empty((len(event_laps), fields.shape[-1]), dtype=np.float32)

    for event_index, event_lap in enumerate(event_laps):
        old_reference = fields[event_lap - 1]
        new_reference = fields[event_lap + block_length - 1]
        endpoint_units[event_index] = tuning_similarity(new_reference, old_reference)

        for relative_index, relative_lap in enumerate(relative_laps):
            lap = event_lap + relative_lap
            current = fields[lap]
            old_similarity[event_index, relative_index] = tuning_similarity(current, old_reference).mean()
            new_similarity[event_index, relative_index] = tuning_similarity(current, new_reference).mean()
            step_similarity[event_index, relative_index] = tuning_similarity(current, fields[lap - 1]).mean()

    return {
        "old_map_similarity": old_similarity,
        "new_map_similarity": new_similarity,
        "step_similarity": step_similarity,
        "endpoint_unit_similarity": endpoint_units,
    }


def run_rule(seed: int, prepared: dict, settings: dict, rule: str) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Train both schedules under one rule and retain every frozen CA1 probe."""

    memory = {**settings["memory"], "plasticity_rule": rule}
    relative_laps = np.asarray(settings["analysis"]["relative_laps"], dtype=int)
    block_length = settings["track"]["swap_every"]

    swap_context = np.asarray([0 if assignment == [0, 1] else 1 for assignment in prepared["schedules"][0]])
    event_laps = np.flatnonzero(swap_context[1:] != swap_context[:-1]) + 1
    if np.any(event_laps + block_length - 1 >= settings["track"]["training_laps"]):
        raise ValueError("Every cue exchange must be followed by one complete cue block.")
    if np.any(event_laps[:, None] + relative_laps[None, :] - 1 < 0):
        raise ValueError("The event window extends before the first available transition.")

    scheduled_fields = []
    measurements = []
    similarity_matrices = []

    for schedule_index, (laps, schedule) in enumerate(zip(prepared["laps"], prepared["schedules"])):
        # The exact same wiring seed is deliberately reused across schedules
        # and rules. CA3-CA1 weights begin at zero in every model.
        model = build_mtl(prepared["autoencoder"], memory, seed_value(seed, 20))
        lap_fields = []
        for lap_input, assignment in zip(laps, schedule):
            run_mtl(model, lap_input, learn=True)
            context_index = 0 if assignment == [0, 1] else 1
            lap_fields.append(run_mtl(model, prepared["probes"][context_index], learn=False)[1])
        lap_fields = np.stack(lap_fields).astype(np.float32)
        scheduled_fields.append(lap_fields)
        similarity_matrices.append(population_similarity(lap_fields))
        measurements.append(event_measurements(lap_fields, event_laps, relative_laps, block_length))

    result = {
        "scheduled_probe_ca1": np.stack(scheduled_fields),
        "population_similarity": np.stack(similarity_matrices),
        **{
            name: np.stack([measurement[name] for measurement in measurements])
            for name in measurements[0]
        },
    }

    rows = []
    for schedule_index, schedule_name in enumerate(("swap", "no_swap")):
        for event_index, event_lap in enumerate(event_laps):
            for relative_index, relative_lap in enumerate(relative_laps):
                rows.append(
                    {
                        "record_type": "event_timecourse",
                        "seed": seed,
                        "plasticity_rule": rule,
                        "schedule": schedule_name,
                        "event": event_index,
                        "event_lap": int(event_lap + 1),
                        "relative_lap": int(relative_lap),
                        "old_map_similarity": float(result["old_map_similarity"][schedule_index, event_index, relative_index]),
                        "new_map_similarity": float(result["new_map_similarity"][schedule_index, event_index, relative_index]),
                        "step_similarity": float(result["step_similarity"][schedule_index, event_index, relative_index]),
                    }
                )
            rows.extend(
                {
                    "record_type": "endpoint_unit",
                    "seed": seed,
                    "plasticity_rule": rule,
                    "schedule": schedule_name,
                    "event": event_index,
                    "event_lap": int(event_lap + 1),
                    "ca1_unit": unit,
                    "endpoint_unit_similarity": float(value),
                }
                for unit, value in enumerate(result["endpoint_unit_similarity"][schedule_index, event_index])
            )
    return result, rows


def mean_sem(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return mean and SEM across root seeds (axis zero)."""

    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return values.mean(axis=0), np.zeros(values.shape[1:], dtype=float)
    return values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(len(values))


def save_panel(figure, output: Path, name: str, rule: str, display_rule: str) -> None:
    """Save editable SVG and high-resolution PNG versions of one panel."""

    for suffix in ("png", "svg"):
        figure.savefig(output / f"{name}_{rule}.{suffix}", dpi=300, bbox_inches="tight")
        if rule == display_rule:
            figure.savefig(output / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def build_panels(arrays: dict[str, np.ndarray], output: Path, display_rule: str) -> None:
    """Create four independent plot families from the saved numerical arrays."""

    rules = arrays["plasticity_rules"].tolist()
    schedules = arrays["schedule_conditions"].tolist()
    relative_laps = arrays["relative_laps"]
    event_laps = arrays["event_laps"]

    for rule_index, rule in enumerate(rules):
        rule_label = RULE_LABELS.get(rule, rule)

        # A: average population similarity matrix for the cue-exchange run.
        matrix = arrays["population_similarity"][rule_index, :, 0].mean(axis=0)
        figure, axis = plt.subplots(figsize=(5.2, 4.4))
        image = axis.imshow(matrix, origin="lower", cmap="viridis", vmin=-0.2, vmax=1.0)
        for event_lap in event_laps:
            boundary = event_lap - 0.5
            axis.axvline(boundary, color="white", linestyle="--", linewidth=0.8, alpha=0.75)
            axis.axhline(boundary, color="white", linestyle="--", linewidth=0.8, alpha=0.75)
        figure.colorbar(image, ax=axis, label="Population-map similarity")
        axis.set(
            xlabel="Lap after learning",
            ylabel="Lap after learning",
            title=f"Cue-exchange map similarity ({rule_label})",
        )
        save_panel(figure, output, "plot_remap_a", rule, display_rule)

        # B: loss of the pre-exchange map and convergence to the block endpoint.
        swap_old = arrays["old_map_similarity"][rule_index, :, 0].mean(axis=1)
        swap_new = arrays["new_map_similarity"][rule_index, :, 0].mean(axis=1)
        figure, axis = plt.subplots(figsize=(5.2, 3.8))
        for values, label, color in (
            (swap_old, "Similarity to pre-exchange map", "#6a3d9a"),
            (swap_new, "Similarity to end-of-block map", "#e66101"),
        ):
            mean, sem = mean_sem(values)
            axis.plot(relative_laps, mean, marker="o", color=color, label=label, linewidth=2)
            axis.fill_between(relative_laps, mean - sem, mean + sem, color=color, alpha=0.16)
        axis.axvline(0, color="0.45", linestyle="--", linewidth=1)
        axis.set(
            xlabel="Laps relative to cue exchange",
            ylabel="Mean CA1 tuning similarity",
            ylim=(-0.05, 1.05),
            title=f"Old-to-new map dynamics ({rule_label})",
        )
        axis.legend(frameon=False, fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
        save_panel(figure, output, "plot_remap_b", rule, display_rule)

        # C: immediate lap-to-lap reconfiguration compared with matched control.
        figure, axis = plt.subplots(figsize=(5.2, 3.8))
        for schedule_index, schedule in enumerate(schedules):
            values = arrays["step_similarity"][rule_index, :, schedule_index].mean(axis=1)
            mean, sem = mean_sem(values)
            color = SCHEDULE_COLORS[schedule]
            axis.plot(relative_laps, mean, marker="o", color=color, label=SCHEDULE_LABELS[schedule], linewidth=2)
            axis.fill_between(relative_laps, mean - sem, mean + sem, color=color, alpha=0.16)
        axis.axvline(0, color="#6a3d9a", linestyle="--", linewidth=1)
        axis.set(
            xlabel="Laps relative to cue exchange",
            ylabel="Consecutive-lap tuning similarity",
            ylim=(-0.05, 1.05),
            title=f"Event-aligned reconfiguration ({rule_label})",
        )
        axis.legend(frameon=False, fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
        save_panel(figure, output, "plot_remap_c", rule, display_rule)

        # D: a seed-level empirical CDF avoids treating individual units as
        # independent replicates while retaining population heterogeneity.
        grid = np.linspace(-1.0, 1.0, 201)
        figure, axis = plt.subplots(figsize=(5.2, 3.8))
        for schedule_index, schedule in enumerate(schedules):
            values = arrays["endpoint_unit_similarity"][rule_index, :, schedule_index]
            seed_cdf = (values[..., None] <= grid).mean(axis=(1, 2))
            mean, sem = mean_sem(seed_cdf)
            color = SCHEDULE_COLORS[schedule]
            axis.plot(grid, mean, color=color, label=SCHEDULE_LABELS[schedule], linewidth=2)
            axis.fill_between(grid, mean - sem, mean + sem, color=color, alpha=0.16)
        axis.axvline(0, color="0.55", linestyle=":", linewidth=1)
        axis.set(
            xlabel="Unit tuning similarity: pre-exchange to block endpoint",
            ylabel="Cumulative fraction of CA1 units",
            xlim=(-1.0, 1.0),
            ylim=(0.0, 1.0),
            title=f"Endpoint tuning heterogeneity ({rule_label})",
        )
        axis.legend(frameon=False, fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
        save_panel(figure, output, "plot_remap_d", rule, display_rule)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "results/preprint/remapping_dynamics",
    )
    parser.add_argument(
        "--rules",
        nargs="+",
        choices=("base", "err2"),
        default=SETTINGS["plasticity_rules"],
        help="Internal rule identifiers; both rules are paired by default.",
    )
    parser.add_argument(
        "--display-rule",
        default=None,
        help="Rule copied to unsuffixed plot files; defaults to err2.",
    )
    parser.add_argument("--quick", action="store_true", help="Run a low-cost smoke test.")
    args = parser.parse_args()

    settings = copy.deepcopy(SETTINGS)
    settings["plasticity_rules"] = list(args.rules)
    if args.quick:
        settings["seeds"] = settings["seeds"][:2]
        settings["autoencoder"]["epochs"] = 8
        settings["track"]["training_laps"] = 20
        settings["track"]["validation_laps"] = 2

    results, rows = [], []
    for seed_index, seed in enumerate(settings["seeds"], start=1):
        print(f"Remapping dynamics seed {seed} ({seed_index}/{len(settings['seeds'])})", flush=True)
        prepared = prepare_seed(seed, settings)
        rule_results = []
        for rule in settings["plasticity_rules"]:
            result, rule_rows = run_rule(seed, prepared, settings, rule)
            rule_results.append(result)
            rows.extend(rule_rows)
        results.append(
            {
                name: np.stack([result[name] for result in rule_results])
                for name in rule_results[0]
            }
        )

    swap_context = np.asarray(
        [0 if assignment == [0, 1] else 1 for assignment in context_schedule(
            settings["track"]["training_laps"], settings["track"]["swap_every"], True
        )]
    )
    event_laps = np.flatnonzero(swap_context[1:] != swap_context[:-1]) + 1
    arrays = {
        "root_seeds": np.asarray(settings["seeds"]),
        "plasticity_rules": np.asarray(settings["plasticity_rules"]),
        "schedule_conditions": np.asarray(("swap", "no_swap")),
        "event_laps": event_laps,
        "relative_laps": np.asarray(settings["analysis"]["relative_laps"]),
        **{
            name: np.stack([result[name] for result in results]).swapaxes(0, 1)
            for name in results[0]
        },
    }

    display_rule = args.display_rule or (
        "err2" if "err2" in settings["plasticity_rules"] else settings["plasticity_rules"][0]
    )
    if display_rule not in settings["plasticity_rules"]:
        raise ValueError("--display-rule must be one of the selected --rules")

    artifact = write_artifact(args.output, settings, arrays, rows)
    build_panels(arrays, artifact, display_rule)
    print(artifact)


if __name__ == "__main__":
    main()
