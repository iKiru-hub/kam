"""Deterministic inputs for the generic-memory and cue-track tasks."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from core import datagen


def sparse_patterns(count: int, dimension: int, active: int, rng: np.random.Generator, forbidden: set[tuple[int, ...]] | None = None) -> tuple[np.ndarray, set[tuple[int, ...]]]:
    """Draw unique binary sparse patterns without relying on global randomness."""

    seen = set() if forbidden is None else set(forbidden)
    rows = []
    while len(rows) < count:
        indices = tuple(sorted(rng.choice(dimension, size=active, replace=False).tolist()))
        if indices in seen:
            continue
        seen.add(indices)
        row = np.zeros(dimension, dtype=np.float32)
        row[list(indices)] = 1.0
        rows.append(row)
    return np.stack(rows), seen


@contextmanager
def numpy_seed(seed: int):
    """Temporarily seed legacy stimulus helpers that use ``np.random``."""

    state = np.random.get_state()
    np.random.seed(seed % (2**32 - 1))
    try:
        yield
    finally:
        np.random.set_state(state)


def cue_track(num_laps: int, settings: dict, assignments: list[list[int]], seed: int) -> np.ndarray:
    """Generate cue-track laps with a caller-supplied context schedule."""

    with numpy_seed(seed):
        cues = datagen.make_cues(2, int(settings["size"]) // 2, fixed=True)
        laps = {
            "n": num_laps,
            "length": int(settings["lap_length"]),
            "cues_positions": list(settings["cue_positions"]),
            "cues_patterns": cues,
            "cues_sequence": assignments,
            "cue_sigma": float(settings["cue_sigma"]),
            "cue_beta": float(settings["cue_beta"]),
            "cue_alpha": float(settings["cue_alpha"]),
            "mec_binarized": bool(settings["mec_binarized"]),
        }
        values, _ = datagen.sparse_stimulus_generator_sensory(
            laps=laps,
            mec_size=int(settings["size"]) // 2,
            mec_sigma=float(settings["mec_sigma"]),
            lec_sigma=float(settings["lec_sigma"]),
        )
    return values.astype(np.float32)


def alternating_assignments(num_laps: int, swap_every: int) -> list[list[int]]:
    return [[0, 1] if (lap // swap_every) % 2 == 0 else [1, 0] for lap in range(num_laps)]
