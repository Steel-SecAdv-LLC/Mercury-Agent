"""Analytical verification of compute_auc against 4 known-answer cases.

Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

These tests verify the correctness of the trapezoidal AUC-ROC implementation
in benchmarks/domain_benchmark_base.py against analytically derivable answers.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.domain_benchmark_base import compute_auc


class TestComputeAucAnalytical:
    """Four analytical cases for compute_auc correctness."""

    def test_perfect_separation(self) -> None:
        """Case 1: Perfect classifier — AUC must be exactly 1.0.

        All positives have higher scores than all negatives.
        The ROC curve goes (0,0) -> (0,1) -> (1,1), area = 1.0.
        """
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        auc = compute_auc(y_true, y_scores)
        assert auc == pytest.approx(1.0, abs=1e-10), f"Perfect separation: expected 1.0, got {auc}"

    def test_inverse_predictions(self) -> None:
        """Case 2: Perfectly wrong classifier — AUC must be exactly 0.0.

        All negatives have higher scores than all positives.
        The ROC curve goes (0,0) -> (1,0) -> (1,1), area = 0.0.
        """
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        y_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        auc = compute_auc(y_true, y_scores)
        assert auc == pytest.approx(0.0, abs=1e-10), f"Inverse predictions: expected 0.0, got {auc}"

    def test_random_coin_flip(self) -> None:
        """Case 3: Interleaved labels with uniform scores — AUC = 0.5.

        Alternating labels with linearly spaced scores.  For perfectly
        interleaved positive/negative pairs the trapezoidal AUC is
        analytically 0.5 (the diagonal).
        """
        # 50 positives interleaved with 50 negatives, scores = rank order
        n = 100
        y_true = np.array([i % 2 for i in range(n)])
        y_scores = np.linspace(0, 1, n)
        auc = compute_auc(y_true, y_scores)
        assert auc == pytest.approx(
            0.5, abs=0.02
        ), f"Coin-flip interleaved: expected ~0.5, got {auc}"

    def test_single_class_returns_half(self) -> None:
        """Case 4: Only one class present — AUC must be 0.5 (undefined).

        When there are no positives (or no negatives), AUC-ROC is
        mathematically undefined.  The implementation must return 0.5
        (random baseline) per convention.
        """
        y_all_neg = np.zeros(100)
        y_scores = np.random.default_rng(42).uniform(0, 1, 100)
        assert compute_auc(y_all_neg, y_scores) == 0.5, "All-negative should return 0.5"

        y_all_pos = np.ones(100)
        assert compute_auc(y_all_pos, y_scores) == 0.5, "All-positive should return 0.5"

    def test_tied_scores_handled(self) -> None:
        """Bonus: Tied scores must not crash or produce AUC > 1.0."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_scores = np.array([0.5, 0.5, 0.5, 0.8, 0.8, 0.8])
        auc = compute_auc(y_true, y_scores)
        assert 0.0 <= auc <= 1.0, f"Tied scores: AUC out of range: {auc}"

    def test_length_mismatch_raises(self) -> None:
        """Input arrays of different lengths must raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            compute_auc(np.array([0, 1]), np.array([0.1, 0.2, 0.3]))
