"""
Mercury Agent - Edge case tests for threshold calibrator.

Copyright (C) 2025 Steel Security Advisors LLC
License: GPL-3.0+

Tests robustness of threshold optimization against degenerate inputs:
single-class labels, empty arrays, shape mismatches, etc.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.threshold_calibrator import (
    ThresholdOptimizer,
    find_optimal_threshold,
    find_optimal_threshold_fine,
)


class TestFindOptimalThreshold:
    """Tests for find_optimal_threshold edge cases."""

    def test_single_class_all_normal(self) -> None:
        """Should return fallback for all-normal labels."""
        scores = np.random.uniform(0.0, 0.3, 100)
        labels = np.zeros(100)
        threshold = find_optimal_threshold(scores, labels)
        assert threshold == 0.5, "Should return default for single-class"

    def test_single_class_all_anomalous(self) -> None:
        """Should return fallback for all-anomalous labels."""
        scores = np.random.uniform(0.7, 1.0, 100)
        labels = np.ones(100)
        threshold = find_optimal_threshold(scores, labels)
        assert threshold == 0.5, "Should return default for single-class"

    def test_empty_input(self) -> None:
        """Should return fallback for empty arrays."""
        scores = np.array([])
        labels = np.array([])
        threshold = find_optimal_threshold(scores, labels)
        assert threshold == 0.5, "Should return default for empty"

    def test_shape_mismatch(self) -> None:
        """Should raise ValueError for mismatched shapes."""
        with pytest.raises(ValueError, match="Shape mismatch"):
            find_optimal_threshold(np.array([0.5, 0.6]), np.array([0, 1, 0]))

    def test_custom_fallback(self) -> None:
        """Should use custom fallback threshold."""
        scores = np.array([])
        labels = np.array([])
        threshold = find_optimal_threshold(scores, labels, fallback_threshold=0.3)
        assert threshold == 0.3

    def test_perfect_separation(self) -> None:
        """Should find perfect threshold when scores perfectly separate classes."""
        scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
        labels = np.array([0, 0, 0, 1, 1, 1])
        threshold = find_optimal_threshold(scores, labels)
        # Threshold should be around 0.5-0.8
        assert 0.3 <= threshold <= 0.9
        # F1 should be 1.0 at optimal threshold
        predictions = (scores >= threshold).astype(int)
        from sklearn.metrics import f1_score

        f1 = f1_score(labels, predictions)
        assert f1 == 1.0

    def test_normal_case(self) -> None:
        """Standard case with mixed labels."""
        np.random.seed(42)
        scores = np.concatenate(
            [
                np.random.uniform(0.0, 0.5, 80),
                np.random.uniform(0.5, 1.0, 20),
            ]
        )
        labels = np.concatenate([np.zeros(80), np.ones(20)])
        threshold = find_optimal_threshold(scores, labels)
        assert 0.0 <= threshold <= 1.0

    def test_verbose_mode(self) -> None:
        """Verbose mode should not crash."""
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        threshold = find_optimal_threshold(scores, labels, verbose=True)
        assert 0.0 <= threshold <= 1.0


class TestFindOptimalThresholdFine:
    """Tests for fine-grained threshold search."""

    def test_fine_improves_or_matches(self) -> None:
        """Fine search should match or improve coarse result."""
        np.random.seed(42)
        scores = np.concatenate(
            [
                np.random.uniform(0.0, 0.6, 80),
                np.random.uniform(0.4, 1.0, 20),
            ]
        )
        labels = np.concatenate([np.zeros(80), np.ones(20)])

        coarse = find_optimal_threshold(scores, labels)
        fine, fine_f1 = find_optimal_threshold_fine(scores, labels)

        from sklearn.metrics import f1_score

        coarse_preds = (scores >= coarse).astype(int)
        coarse_f1 = f1_score(labels, coarse_preds, zero_division=0)

        assert fine_f1 >= coarse_f1 - 0.001  # Fine should be at least as good


class TestThresholdOptimizer:
    """Tests for ThresholdOptimizer class."""

    def test_optimize_and_retrieve(self) -> None:
        """Should cache optimized thresholds."""
        optimizer = ThresholdOptimizer()
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])

        threshold = optimizer.optimize("test_dataset", scores, labels)
        assert 0.0 <= threshold <= 1.0
        assert optimizer.get_threshold("test_dataset") == threshold

    def test_get_default(self) -> None:
        """Should return default for unknown datasets."""
        optimizer = ThresholdOptimizer()
        assert optimizer.get_threshold("unknown") == 0.5
        assert optimizer.get_threshold("unknown", default=0.3) == 0.3

    def test_save_and_load(self, tmp_path: pytest.TempPathFactory) -> None:
        """Should persist and reload thresholds."""
        optimizer = ThresholdOptimizer()
        optimizer.thresholds = {"cardio": 0.45, "thyroid": 0.40}

        path = str(tmp_path / "thresholds.json")
        optimizer.save(path)

        loaded = ThresholdOptimizer()
        loaded.load(path)
        assert loaded.thresholds == {"cardio": 0.45, "thyroid": 0.40}
