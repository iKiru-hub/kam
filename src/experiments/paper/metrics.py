"""Output-level metrics with explicit sparse-vector edge behavior."""

from __future__ import annotations

import numpy as np


EPSILON = 1e-12


def row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("row_cosine expects equally shaped 2D arrays")
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > EPSILON,
    )


def cosine_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("cosine_matrix expects compatible 2D arrays")
    numerator = left @ right.T
    denominator = np.linalg.norm(left, axis=1)[:, None] * np.linalg.norm(
        right, axis=1
    )[None, :]
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > EPSILON,
    )


def topk_overlap(outputs: np.ndarray, targets: np.ndarray, k: int) -> np.ndarray:
    outputs = np.asarray(outputs)
    targets = np.asarray(targets)
    if outputs.shape != targets.shape or outputs.ndim != 2:
        raise ValueError("topk_overlap expects equally shaped 2D arrays")
    if not 1 <= k <= outputs.shape[1]:
        raise ValueError("k must be within the vector dimension")
    output_top = np.argpartition(outputs, -k, axis=1)[:, -k:]
    target_top = np.argpartition(targets, -k, axis=1)[:, -k:]
    return np.asarray(
        [len(set(a.tolist()) & set(b.tolist())) / k for a, b in zip(output_top, target_top)],
        dtype=np.float64,
    )


def identity_correct(outputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    similarities = cosine_matrix(outputs, targets)
    predictions = np.argmax(similarities, axis=1)
    return (predictions == np.arange(len(outputs))).astype(np.float64)


def chance_corrected_cosine(
    outputs: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    if len(outputs) < 2:
        raise ValueError("Chance correction requires at least two targets")
    similarities = cosine_matrix(outputs, targets)
    matched = np.diag(similarities)
    chance = (similarities.sum(axis=1) - matched) / (len(targets) - 1)
    denominator = 1.0 - chance
    return np.divide(
        matched - chance,
        denominator,
        out=np.zeros_like(matched),
        where=np.abs(denominator) > EPSILON,
    )


def evaluate_outputs(
    outputs: np.ndarray, targets: np.ndarray, k: int
) -> dict[str, np.ndarray]:
    return {
        "raw_cosine": row_cosine(outputs, targets),
        "topk_overlap": topk_overlap(outputs, targets, k),
        "identity_correct": identity_correct(outputs, targets),
        "chance_corrected_cosine": chance_corrected_cosine(outputs, targets),
        "mse": np.mean((np.asarray(outputs) - np.asarray(targets)) ** 2, axis=1),
    }


def metric_sanity_checks() -> dict[str, bool]:
    targets = np.eye(4, dtype=np.float64)
    perfect = evaluate_outputs(targets, targets, 1)
    zero = evaluate_outputs(np.zeros_like(targets), targets, 1)
    constant = evaluate_outputs(np.full_like(targets, 0.5), targets, 1)
    return {
        "perfect_cosine": bool(np.allclose(perfect["raw_cosine"], 1.0)),
        "perfect_topk": bool(np.allclose(perfect["topk_overlap"], 1.0)),
        "perfect_identity": bool(np.allclose(perfect["identity_correct"], 1.0)),
        "perfect_mse": bool(np.allclose(perfect["mse"], 0.0)),
        "zero_cosine": bool(np.allclose(zero["raw_cosine"], 0.0)),
        "zero_metrics_finite": bool(
            all(np.isfinite(values).all() for values in zero.values())
        ),
        "constant_metrics_finite": bool(
            all(np.isfinite(values).all() for values in constant.values())
        ),
    }

