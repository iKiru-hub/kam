"""Metrics shared by all preprint simulations."""

from __future__ import annotations

import numpy as np


EPSILON = 1e-12


def row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > EPSILON)


def topk_overlap(left: np.ndarray, right: np.ndarray, k: int) -> np.ndarray:
    left_top = np.argpartition(left, -k, axis=1)[:, -k:]
    right_top = np.argpartition(right, -k, axis=1)[:, -k:]
    return np.asarray([len(set(a) & set(b)) / k for a, b in zip(left_top, right_top)])


def identity_correct(outputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    similarity = outputs @ targets.T
    similarity /= np.maximum(np.linalg.norm(outputs, axis=1)[:, None] * np.linalg.norm(targets, axis=1)[None, :], EPSILON)
    return (np.argmax(similarity, axis=1) == np.arange(len(outputs))).astype(np.float64)


def output_metrics(outputs: np.ndarray, targets: np.ndarray, k: int) -> dict[str, np.ndarray]:
    return {
        "cosine": row_cosine(outputs, targets),
        "mse": np.mean((outputs - targets) ** 2, axis=1),
        "topk": topk_overlap(outputs, targets, k),
        "identity": identity_correct(outputs, targets),
    }
