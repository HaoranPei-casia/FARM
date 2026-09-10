"""Dependency-free binary trajectory metrics."""

from __future__ import annotations

import numpy as np


def _arrays(labels: list[int], scores: list[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s) or len(y) == 0:
        raise ValueError("labels and scores must be non-empty one-dimensional arrays")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must contain only 0 and 1")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    return y, s


def binary_auroc(labels: list[int], scores: list[float]) -> float:
    """Compute AUROC with average ranks for tied scores."""
    y, s = _arrays(labels, scores)
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(ranks[y == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2) / (
        positives * negatives
    )


def average_precision(labels: list[int], scores: list[float]) -> float:
    """Compute average precision using distinct score thresholds."""
    y, s = _arrays(labels, scores)
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    s = s[order]
    cumulative_true = np.cumsum(y)
    threshold_ends = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    true_positive = cumulative_true[threshold_ends]
    precision = true_positive / (threshold_ends + 1)
    previous_recall_count = np.r_[0, true_positive[:-1]]
    return float(np.sum(precision * (true_positive - previous_recall_count)) / positives)


def binary_metrics(
    labels: list[int], scores: list[float], threshold: float = 0.5
) -> dict[str, float | int]:
    y, s = _arrays(labels, scores)
    predictions = s >= threshold
    return {
        "count": len(y),
        "positive_count": int(y.sum()),
        "auroc": binary_auroc(labels, scores),
        "average_precision": average_precision(labels, scores),
        "accuracy": float(np.mean(predictions == y)),
    }
