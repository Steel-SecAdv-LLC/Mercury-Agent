"""
Mercury Agent - Automatic threshold calibration per dataset.

Copyright (C) 2025 Steel Security Advisors LLC
License: GPL-3.0+

Provides threshold optimization to maximize F1 score on anomaly detection
tasks. When AUC-ROC is high but F1 is low, it typically indicates
that the default 0.5 threshold is suboptimal for the score distribution.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import f1_score


def find_optimal_threshold(scores: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]) -> float:
    """Find threshold that maximizes F1 score.

    Sweeps thresholds in [0, 1] and returns the one producing the highest
    F1 score. Useful when AUC-ROC is high but F1 at the default 0.5
    threshold is poor (common in imbalanced anomaly detection datasets).

    Args:
        scores: Anomaly scores in [0, 1], shape (n_samples,).
        labels: Binary labels {0, 1}, shape (n_samples,).

    Returns:
        Optimal threshold to maximize F1.
    """
    best_f1 = 0.0
    best_threshold = 0.5

    for threshold in np.linspace(0.0, 1.0, 101):
        predictions = (scores >= threshold).astype(int)
        f1 = f1_score(labels, predictions, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold


def find_optimal_threshold_fine(
    scores: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    coarse_steps: int = 101,
    fine_steps: int = 51,
) -> tuple[float, float]:
    """Two-pass threshold search: coarse sweep then fine-grained refinement.

    Args:
        scores: Anomaly scores in [0, 1], shape (n_samples,).
        labels: Binary labels {0, 1}, shape (n_samples,).
        coarse_steps: Number of steps in coarse sweep.
        fine_steps: Number of steps in fine refinement.

    Returns:
        Tuple of (optimal_threshold, best_f1_score).
    """
    # Coarse pass
    coarse_threshold = find_optimal_threshold(scores, labels)

    # Fine pass around the coarse optimum
    low = max(0.0, coarse_threshold - 0.05)
    high = min(1.0, coarse_threshold + 0.05)

    best_f1 = 0.0
    best_threshold = coarse_threshold

    for threshold in np.linspace(low, high, fine_steps):
        predictions = (scores >= threshold).astype(int)
        f1 = f1_score(labels, predictions, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold, float(best_f1)
