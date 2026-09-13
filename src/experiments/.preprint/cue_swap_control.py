"""Matched cue-swap and no-swap controls for lap-resolved CA1 tuning."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.preprint.artifacts import create_artifact
from experiments.preprint.config import read_config
from experiments.preprint.model_factory import build_mtl, run_mtl, train_autoencoder
from experiments.preprint.seeds import SeedStreams
from experiments.preprint.stimuli import alternating_assignments, cue_track


CONDITION_NAMES = ("swap", "no_swap")


def _mean_unit_tuning_similarity(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    """Mean unit-wise correlation between two CA1 tuning populations."""

    first_centered = first - first.mean(axis=-2, keepdims=True)
    second_centered = second - second.mean(axis=-2, keepdims=True)
    numerator = np.sum(first_centered * second_centered, axis=-2)
    denominator = np.linalg.norm(first_centered, axis=-2) * np.linalg.norm(
        second_centered, axis=-2
    )
    unit_similarity = numerator / np.maximum(denominator, 1e-12)
    return unit_similarity.mean(axis=-1)


def _mean_ci95(values: np.ndarray) -> tuple[float, list[float]]:
    """Return a seed mean and normal-approximation 95% interval."""

    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    half_width = 1.96 * float(values.std(ddof=1)) / np.sqrt(len(values))
    return mean, [mean - half_width, mean + half_width]


def _context_indices(assignments: list[list[int]]) -> np.ndarray:
    """Map the two supported cue orders to context indices A=0 and B=1."""

    lookup = {(0, 1): 0, (1, 0): 1}
    return np.asarray([lookup[tuple(value)] for value in assignments], dtype=np.int64)


def _run_condition(
    autoencoder,
    memory: dict,
    wiring_seed: int,
    training_laps: np.ndarray,
    probe_laps: np.ndarray,
    context_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Train one matched model and probe its scheduled context after each lap."""

    model = build_mtl(autoencoder, memory, wiring_seed)
    training_ca1 = []
    scheduled_ca1 = []
    final_probes = None
    for lap_index, lap in enumerate(training_laps):
        _, ca1, _ = run_mtl(model, lap, learn=True)
        training_ca1.append(ca1)
        current_probes = []
        for probe in probe_laps:
            _, probe_ca1, _ = run_mtl(model, probe, learn=False)
            current_probes.append(probe_ca1)
        final_probes = np.stack(current_probes)
        scheduled_ca1.append(final_probes[context_indices[lap_index]])

    scheduled = np.stack(scheduled_ca1)
    transition_similarity = _mean_unit_tuning_similarity(
        scheduled[1:], scheduled[:-1]
    )
    template_similarity = np.stack([
        _mean_unit_tuning_similarity(
            scheduled,
            np.broadcast_to(template, scheduled.shape),
        )
        for template in final_probes
    ], axis=-1)
    return {
        "training_ca1": np.stack(training_ca1).astype(np.float32),
        "scheduled_ca1": scheduled.astype(np.float32),
        "final_probe_ca1": final_probes.astype(np.float32),
        "transition_similarity": transition_similarity.astype(np.float32),
        "template_similarity": template_similarity.astype(np.float32),
    }


def run_seed(
    config: dict,
    root_seed: int,
) -> tuple[dict[str, np.ndarray], list[dict], dict]:
    """Run paired schedules with shared inputs, pretraining, and initialization."""

    streams = SeedStreams(root_seed)
    track = config["track"]
    num_laps = int(track["training_laps"])
    swap_assignments = alternating_assignments(num_laps, int(track["swap_every"]))
    no_swap_assignments = [[0, 1]] * num_laps
    assignments = (swap_assignments, no_swap_assignments)

    track_seed = streams.integer("track")
    swap_laps = cue_track(num_laps, track, swap_assignments, track_seed)
    no_swap_laps = cue_track(num_laps, track, no_swap_assignments, track_seed)
    mec_size = int(track["size"]) // 2
    if not np.array_equal(swap_laps[:, :, :mec_size], no_swap_laps[:, :, :mec_size]):
        raise RuntimeError("Matched schedules produced different MEC trajectories")
    if not np.array_equal(
        swap_laps[: int(track["swap_every"])],
        no_swap_laps[: int(track["swap_every"])],
    ):
        raise RuntimeError("Matched schedules differ before the first cue swap")

    validation_laps = cue_track(
        int(track["validation_laps"]),
        track,
        [[0, 1]] * int(track["validation_laps"]),
        streams.integer("ae_valid"),
    )
    autoencoder, quality = train_autoencoder(
        swap_laps.reshape(-1, int(track["size"])),
        validation_laps.reshape(-1, int(track["size"])),
        config,
        streams.integer("ae_init"),
        streams.integer("ae_batches"),
    )

    probe_seed = streams.integer("ae_train")
    probe_a = cue_track(1, track, [[0, 1]], probe_seed)[0]
    probe_b = cue_track(1, track, [[1, 0]], probe_seed)[0]
    if not np.array_equal(probe_a[:, :mec_size], probe_b[:, :mec_size]):
        raise RuntimeError("Context probes must share the same MEC trajectory")
    probe_laps = np.stack([probe_a, probe_b])

    condition_results = []
    for laps, schedule in zip((swap_laps, no_swap_laps), assignments):
        condition_results.append(_run_condition(
            autoencoder,
            config["memory"],
            streams.integer("ca3_wiring"),
            laps,
            probe_laps,
            _context_indices(schedule),
        ))

    context_indices = np.stack([_context_indices(value) for value in assignments])
    cue_changed = context_indices[:, 1:] != context_indices[:, :-1]
    arrays = {
        name: np.stack([result[name] for result in condition_results])
        for name in condition_results[0]
    }
    arrays.update({
        "cue_assignments": np.asarray(assignments, dtype=np.int64),
        "context_indices": context_indices,
        "cue_changed": cue_changed,
        "condition_names": np.asarray(CONDITION_NAMES),
    })

    rows = []
    for condition_index, condition in enumerate(CONDITION_NAMES):
        for transition_index, similarity in enumerate(
            arrays["transition_similarity"][condition_index]
        ):
            rows.append({
                "root_seed": root_seed,
                "condition": condition,
                "from_lap": transition_index + 1,
                "to_lap": transition_index + 2,
                "cue_changed": bool(cue_changed[condition_index, transition_index]),
                "tuning_similarity": float(similarity),
            })
    return arrays, rows, quality


def run(config: dict, output: Path) -> Path:
    results, rows, quality = [], [], []
    for position, root_seed in enumerate(config["root_seeds"], start=1):
        print(
            f"cue swap control seed {root_seed} "
            f"({position}/{len(config['root_seeds'])})",
            flush=True,
        )
        result, seed_rows, seed_quality = run_seed(config, int(root_seed))
        results.append(result)
        rows.extend(seed_rows)
        quality.append(seed_quality)

    arrays = {
        name: np.stack([result[name] for result in results])
        for name in results[0]
    }
    arrays["root_seeds"] = np.asarray(config["root_seeds"], dtype=np.int64)
    condition_names = arrays["condition_names"][0].tolist()
    swap_index = condition_names.index("swap")
    no_swap_index = condition_names.index("no_swap")
    event_mask = arrays["cue_changed"][0, swap_index]
    swap_events = arrays["transition_similarity"][:, swap_index, event_mask].mean(axis=1)
    matched_control = arrays["transition_similarity"][:, no_swap_index, event_mask].mean(axis=1)
    within_swap = arrays["transition_similarity"][:, swap_index, ~event_mask].mean(axis=1)
    swap_mean, swap_ci = _mean_ci95(swap_events)
    control_mean, control_ci = _mean_ci95(matched_control)
    within_mean, within_ci = _mean_ci95(within_swap)
    effect_mean, effect_ci = _mean_ci95(swap_events - matched_control)
    event_laps = (np.flatnonzero(event_mask) + 2).tolist()
    report = {
        "experiment": "cue_swap_control",
        "autoencoder_quality": quality,
        "metric": "mean unit-wise mean-centered CA1 tuning similarity",
        "cue_swap_laps": event_laps,
        "mean_swap_event_similarity": swap_mean,
        "swap_event_ci95": swap_ci,
        "mean_matched_no_swap_similarity": control_mean,
        "matched_no_swap_ci95": control_ci,
        "mean_within_swap_similarity": within_mean,
        "within_swap_ci95": within_ci,
        "mean_paired_swap_effect": effect_mean,
        "paired_swap_effect_ci95": effect_ci,
    }
    resolved_config = {
        **config,
        "cue_swap_control": {
            "conditions": list(CONDITION_NAMES),
            "fixed_assignment": [0, 1],
            "probe_after_each_lap": True,
            "shared_between_conditions": [
                "root seed",
                "pretrained autoencoder",
                "CA3 wiring",
                "initial CA3-CA1 weights",
                "MEC trajectory",
                "lap count",
                "plasticity parameters",
            ],
        },
    }
    return create_artifact(output, resolved_config, arrays, rows, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(run(read_config(args.config), args.output))


if __name__ == "__main__":
    main()
