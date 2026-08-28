"""Shared plotting helpers for figures rebuilt from saved arrays."""

from __future__ import annotations

import numpy as np


def mean_sem(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    return values.mean(axis=axis), values.std(axis=axis, ddof=1) / np.sqrt(values.shape[axis])
