"""Named random streams used by the preprint simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SCHEMA_SEED = 20260828
DEVELOPMENT_SEEDS = tuple(range(31001, 31009))
FINAL_SEEDS = tuple(range(51001, 51021))

STREAM_IDS = {
    "ae_init": 1,
    "ae_train": 2,
    "ae_valid": 3,
    "ae_batches": 4,
    "memory": 5,
    "ca3_wiring": 6,
    "storage_order": 7,
    "permutation": 8,
    "content_control": 9,
    "track": 10,
    "masks": 11,
}


@dataclass(frozen=True)
class SeedStreams:
    """Create order-independent NumPy and Torch seeds from one root seed."""

    root_seed: int

    def sequence(self, name: str) -> np.random.SeedSequence:
        if name not in STREAM_IDS:
            raise KeyError(f"Unknown seed stream: {name}")
        return np.random.SeedSequence([SCHEMA_SEED, int(self.root_seed), STREAM_IDS[name]])

    def numpy(self, name: str) -> np.random.Generator:
        return np.random.default_rng(self.sequence(name))

    def integer(self, name: str) -> int:
        state = self.sequence(name).generate_state(2, dtype=np.uint32)
        return ((int(state[0]) << 32) | int(state[1])) % (2**63 - 1)


def nonidentity_permutation(size: int, rng: np.random.Generator) -> np.ndarray:
    identity = np.arange(size)
    while True:
        candidate = rng.permutation(size)
        if not np.array_equal(candidate, identity):
            return candidate.astype(np.int64)


def derangement(size: int, rng: np.random.Generator) -> np.ndarray:
    identity = np.arange(size)
    while True:
        candidate = rng.permutation(size)
        if np.all(candidate != identity):
            return candidate.astype(np.int64)
