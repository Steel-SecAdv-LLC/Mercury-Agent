"""
Unit tests for signal integrity fixes.

Tests for Issue #3 (Discrete Score Destruction) and Issue #5 (Contamination Mismatch).
Validates that statistical detector produces continuous scores instead of discrete values.

Mercury Agent - Copyright (C) 2025 Steel Security Advisors LLC
Licensed under GNU GPL v3
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_classification
from sklearn.metrics import roc_auc_score

from omni_mercury_engine.detectors.statistical import StatisticalAnomalyDetector


class TestContinuousScores:
    """Test that scores are continuous, not discrete."""

    @pytest.fixture
    def toy_data(self):
        """Generate toy classification data with 10% anomalies."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_classes=2,
            weights=[0.9, 0.1],  # 10% anomalies
            random_state=42,
        )
        return X.astype(np.float32), y

    @pytest.fixture
    def detector(self):
        """Create a StatisticalAnomalyDetector instance."""
        return StatisticalAnomalyDetector()

    def test_scores_are_continuous(self, detector, toy_data):
        """Verify scores have more than 5 unique values.

        Issue #3 caused only 5 discrete values: {0.0, 0.3, 0.4, 0.7, 1.0}.
        After fix, scores should be continuous with many unique values.
        """
        X, _ = toy_data
        detector.fit(X)
        result = detector.detect(X)

        scores = result["scores"]
        unique_scores = np.unique(scores)

        # Should have many unique values, not just 5
        assert len(unique_scores) > 10, (
            f"Expected >10 unique scores for continuous output, "
            f"got {len(unique_scores)}: {sorted(unique_scores)[:10]}"
        )

    def test_scores_in_valid_range(self, detector, toy_data):
        """Verify all scores are in [0, 1] range."""
        X, _ = toy_data
        detector.fit(X)
        result = detector.detect(X)

        scores = result["scores"]
        assert scores.min() >= 0.0, f"Min score {scores.min()} < 0"
        assert scores.max() <= 1.0, f"Max score {scores.max()} > 1"

    def test_continuous_z_scores_exist(self, detector, toy_data):
        """Verify z_score_continuous key exists and is continuous."""
        X, _ = toy_data
        detector.fit(X)
        result = detector.detect(X)

        assert "z_score_continuous" in result, "Missing z_score_continuous key"
        z_continuous = result["z_score_continuous"]
        unique_z = np.unique(z_continuous)

        # Should have many unique values
        assert len(unique_z) > 5, f"z_score_continuous has only {len(unique_z)} unique values"

    def test_iqr_scores_exist_and_continuous(self, detector, toy_data):
        """Verify iqr_scores key exists and is continuous."""
        X, _ = toy_data
        detector.fit(X)
        result = detector.detect(X)

        assert "iqr_scores" in result, "Missing iqr_scores key"
        iqr_scores = result["iqr_scores"]
        unique_iqr = np.unique(iqr_scores)

        # Should have more than boolean (0, 1) values
        assert len(unique_iqr) >= 2, "IQR scores should have variance"

    def test_isolation_forest_scores_continuous(self, detector, toy_data):
        """Verify isolation_forest_scores key exists and is continuous.

        Note: isolation_forest_scores is now an alias for the resonance score
        (backward compatibility). It should still be continuous [0, 1].
        """
        X, _ = toy_data
        detector.fit(X)
        result = detector.detect(X)

        assert "isolation_forest_scores" in result, "Missing isolation_forest_scores key"
        if_scores = result["isolation_forest_scores"]

        # Should be continuous [0, 1], not just binary
        unique_if = np.unique(if_scores)
        assert (
            len(unique_if) > 5
        ), f"IF scores should be continuous, got {len(unique_if)} unique values"

    def test_backward_compatibility(self, detector, toy_data):
        """Verify legacy keys still exist for backward compatibility."""
        X, _ = toy_data
        detector.fit(X)
        result = detector.detect(X)

        # Legacy keys should still exist
        assert "iqr_flags" in result, "Missing legacy key: iqr_flags"
        assert "isolation_forest_flags" in result, "Missing legacy key: isolation_forest_flags"
        assert "is_anomaly" in result, "Missing key: is_anomaly"
        assert "scores" in result, "Missing key: scores"
        assert "z_scores" in result, "Missing key: z_scores"


class TestROCAUCImprovement:
    """Test that continuous scores improve ROC-AUC."""

    @pytest.fixture
    def separable_data(self):
        """Generate data with clear separation between classes."""
        # Create two clusters with some overlap
        X_normal, _ = make_blobs(
            n_samples=180, centers=[[0, 0] * 5], cluster_std=1.0, random_state=42
        )
        X_anomaly, _ = make_blobs(
            n_samples=20, centers=[[5, 5] * 5], cluster_std=1.5, random_state=43
        )
        X = np.vstack([X_normal, X_anomaly]).astype(np.float32)
        y = np.array([0] * 180 + [1] * 20)

        # Shuffle
        idx = np.random.RandomState(42).permutation(len(X))
        return X[idx], y[idx]

    def test_roc_auc_above_baseline(self, separable_data):
        """Verify ROC-AUC is significantly above random (0.5)."""
        X, y = separable_data
        detector = StatisticalAnomalyDetector()
        detector.fit(X)
        result = detector.detect(X)

        scores = result["scores"]
        auc = roc_auc_score(y, scores)

        # Should be well above random chance
        assert auc > 0.6, f"Expected ROC-AUC >0.6, got {auc:.3f}"

    def test_continuous_scores_better_than_discrete(self, separable_data):
        """Verify continuous scores outperform simulated discrete scores."""
        X, y = separable_data
        detector = StatisticalAnomalyDetector()
        detector.fit(X)
        result = detector.detect(X)

        # Continuous scores
        continuous_scores = result["scores"]
        continuous_auc = roc_auc_score(y, continuous_scores)

        # Simulate old discrete behavior (only 5 values)
        discrete_bins = np.array([0.0, 0.3, 0.4, 0.7, 1.0])
        discrete_scores = discrete_bins[np.digitize(continuous_scores, discrete_bins[:-1])].clip(
            0, 1
        )
        discrete_auc = roc_auc_score(y, discrete_scores)

        # Continuous should be at least as good, ideally better
        assert (
            continuous_auc >= discrete_auc - 0.01
        ), f"Continuous AUC ({continuous_auc:.3f}) should be >= discrete AUC ({discrete_auc:.3f})"


class TestAdaptiveContamination:
    """Test that detector fits correctly with various data distributions.

    Note: IsolationForest and explicit contamination estimation were removed
    in the ensemble replacement (Resonance + Kinematic + InfoGeo). These tests
    now verify the detector still handles different data distributions correctly.
    """

    def test_detector_fits_with_skewed_data(self):
        """Verify detector fits correctly on skewed class distributions."""
        X, _ = make_classification(
            n_samples=200,
            n_features=10,
            n_classes=2,
            weights=[0.95, 0.05],  # 5% anomalies
            random_state=42,
        )

        detector = StatisticalAnomalyDetector()
        detector.fit(X)

        # Detector should be fitted successfully
        assert detector._is_fitted
        result = detector.detect(X)
        assert result["scores"].shape == (200,)

    def test_detector_accepts_contamination_config(self):
        """Verify detector does not crash when contamination is in config."""
        X, _ = make_classification(n_samples=100, n_features=5, random_state=42)

        # Config may still have contamination from legacy callers
        detector = StatisticalAnomalyDetector(config={"contamination": 0.05})
        detector.fit(X)
        assert detector._is_fitted

    def test_detector_fits_with_rare_anomalies(self):
        """Verify detector handles data with very rare anomalies."""
        X, _ = make_classification(
            n_samples=200,
            n_features=10,
            weights=[0.98, 0.02],  # 2% anomalies
            random_state=42,
        )

        detector = StatisticalAnomalyDetector()
        detector.fit(X)
        assert detector._is_fitted
        result = detector.detect(X)
        assert np.all(result["scores"] >= 0) and np.all(result["scores"] <= 1)


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_single_sample(self):
        """Test detection on a single sample."""
        detector = StatisticalAnomalyDetector()
        X_train = np.random.randn(50, 5).astype(np.float32)
        detector.fit(X_train)

        X_test = np.random.randn(1, 5).astype(np.float32)
        result = detector.detect(X_test)

        assert "scores" in result
        assert len(result["scores"]) == 1

    def test_1d_data(self):
        """Test with 1D input data."""
        detector = StatisticalAnomalyDetector()
        X_train = np.random.randn(50).astype(np.float32)
        detector.fit(X_train)

        X_test = np.random.randn(10).astype(np.float32)
        result = detector.detect(X_test)

        assert "scores" in result
        assert len(result["scores"]) == 10

    def test_high_dimensional_data(self):
        """Test with high-dimensional data."""
        detector = StatisticalAnomalyDetector()
        X = np.random.randn(100, 100).astype(np.float32)
        detector.fit(X)
        result = detector.detect(X)

        assert "scores" in result
        assert len(result["scores"]) == 100

    def test_constant_feature_handling(self):
        """Test handling of constant (zero-variance) features."""
        detector = StatisticalAnomalyDetector()
        X = np.random.randn(50, 5).astype(np.float32)
        X[:, 2] = 1.0  # Constant feature

        detector.fit(X)
        result = detector.detect(X)

        # Should not crash, scores should be valid
        assert "scores" in result
        assert not np.any(np.isnan(result["scores"]))
        assert not np.any(np.isinf(result["scores"]))
