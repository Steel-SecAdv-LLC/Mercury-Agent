"""
Mercury Agent - Automatic threshold calibration per dataset.

Copyright (C) 2025 Steel Security Advisors LLC
License: GPL-3.0+

Provides threshold optimization to maximize F1 score on anomaly detection
tasks. When AUC-ROC is high but F1 is low, it typically indicates
that the default 0.5 threshold is suboptimal for the score distribution.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


def find_optimal_threshold(
    scores: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    fallback_threshold: float = 0.5,
    verbose: bool = False,
) -> float:
    """Find threshold that maximizes F1 score, with robust error handling.

    Sweeps thresholds in [0, 1] and returns the one producing the highest
    F1 score. Useful when AUC-ROC is high but F1 at the default 0.5
    threshold is poor (common in imbalanced anomaly detection datasets).

    Args:
        scores: Anomaly scores in [0, 1], shape (n_samples,).
        labels: Binary labels {0, 1}, shape (n_samples,).
        fallback_threshold: Value to return if optimization fails.
        verbose: Log threshold search progress.

    Returns:
        Optimal threshold to maximize F1, or fallback if search fails.

    Raises:
        ValueError: If inputs have mismatched shapes.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int32).ravel()

    if len(scores) != len(labels):
        raise ValueError(f"Shape mismatch: scores {len(scores)} vs labels {len(labels)}")

    if len(scores) == 0:
        logger.warning("Empty scores/labels; returning fallback threshold")
        return fallback_threshold

    # Check for single-class labels
    unique_labels = np.unique(labels)
    if len(unique_labels) == 1:
        logger.warning(
            "Single class detected: only %d in labels. "
            "Threshold optimization impossible; returning %.2f",
            unique_labels[0],
            fallback_threshold,
        )
        return fallback_threshold

    best_f1 = 0.0
    best_threshold = fallback_threshold

    for threshold in np.linspace(0.0, 1.0, 101):
        predictions = (scores >= threshold).astype(np.int32)
        f1 = f1_score(labels, predictions, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    if verbose:
        logger.info("Threshold search: best_f1=%.3f at threshold=%.3f", best_f1, best_threshold)

    if best_f1 < 0.1:
        logger.warning(
            "Best F1 very low: %.3f. Detector may not discriminate well on this dataset.",
            best_f1,
        )

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


class ThresholdOptimizer:
    """Optimizer for per-dataset thresholds with persistence."""

    def __init__(self) -> None:
        self.thresholds: dict[str, float] = {}

    def optimize(
        self,
        dataset_name: str,
        scores: np.ndarray[Any, Any],
        labels: np.ndarray[Any, Any],
    ) -> float:
        """Optimize and cache threshold for a dataset."""
        threshold = find_optimal_threshold(scores, labels, verbose=True)
        self.thresholds[dataset_name] = threshold
        logger.info("Optimized threshold for %s: %.3f", dataset_name, threshold)
        return threshold

    def get_threshold(self, dataset_name: str, default: float = 0.5) -> float:
        """Get cached threshold or default."""
        return self.thresholds.get(dataset_name, default)

    def save(self, path: str) -> None:
        """Save thresholds to JSON for production deployment."""
        with open(path, "w") as f:
            json.dump(self.thresholds, f, indent=2)
        logger.info("Saved thresholds to %s", path)

    def load(self, path: str) -> None:
        """Load thresholds from JSON."""
        with open(path) as f:
            self.thresholds = json.load(f)
        logger.info("Loaded thresholds from %s", path)
