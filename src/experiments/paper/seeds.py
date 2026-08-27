"""Order-independent random streams for paper experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SCHEMA_SEED = 20260826

STREAM_IDS = {
    "ae_initialization": 1,
    "ae_training_patterns": 2,
    "ae_validation_patterns": 3,
    "ae_minibatch_order": 4,
    "ca3_wiring": 5,
    "memory_bank": 6,
    "storage_order": 7,
    "coordinate_permutation": 8,
    "content_derangement": 9,
    "corruptions_and_lures": 10,
    "baselines": 11,
    "analysis_resampling": 12,
    "cue_identity_codes": 13,
    "position_codes": 14,
    "cue_storage_order": 15,
    "cue_ae_restart": 16,
}

DEVELOPMENT_SEEDS = tuple(range(31001, 31009))
FINAL_SEEDS = tuple(range(41001, 41021))
FACTORIAL_SEEDS = tuple(range(61001, 61013))


@dataclass(frozen=True)
class SeedStreams:
    root_seed: int

    def sequence(self, name: str) -> np.random.SeedSequence:
        if name not in STREAM_IDS:
            raise KeyError(f"Unknown seed stream: {name}")
        return np.random.SeedSequence(
            [SCHEMA_SEED, int(self.root_seed), STREAM_IDS[name]]
        )

    def numpy(self, name: str) -> np.random.Generator:
        return np.random.default_rng(self.sequence(name))

    def integer(self, name: str) -> int:
        state = self.sequence(name).generate_state(2, dtype=np.uint32)
        value = (int(state[0]) << 32) | int(state[1])
        return value % (2**63 - 1)

    def manifest(self) -> dict[str, object]:
        return {
            "schema_seed": SCHEMA_SEED,
            "root_seed": int(self.root_seed),
            "streams": {
                name: {
                    "stream_id": stream_id,
                    "derived_seed": self.integer(name),
                }
                for name, stream_id in STREAM_IDS.items()
            },
        }


def assert_seed_sets_disjoint() -> None:
    sets = [set(DEVELOPMENT_SEEDS), set(FINAL_SEEDS), set(FACTORIAL_SEEDS)]
    labels = ["development", "final", "factorial"]
    for left in range(len(sets)):
        for right in range(left + 1, len(sets)):
            overlap = sets[left] & sets[right]
            if overlap:
                raise AssertionError(
                    f"{labels[left]} and {labels[right]} seeds overlap: "
                    f"{sorted(overlap)}"
                )


def random_nonidentity_permutation(
    size: int, rng: np.random.Generator
) -> np.ndarray:
    if size < 2:
        raise ValueError("A nonidentity permutation requires size >= 2")
    identity = np.arange(size)
    while True:
        candidate = rng.permutation(size)
        if not np.array_equal(candidate, identity):
            return candidate.astype(np.int64)


def random_derangement(size: int, rng: np.random.Generator) -> np.ndarray:
    if size < 2:
        raise ValueError("A derangement requires size >= 2")
    identity = np.arange(size)
    while True:
        candidate = rng.permutation(size)
        if np.all(candidate != identity):
            return candidate.astype(np.int64)
