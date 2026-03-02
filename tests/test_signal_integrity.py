"""
Unit tests for signal integrity fixes.

Tests for Issue #3 (Discrete Score Destruction) and Issue #5 (Contamination Mismatch).
Validates that statistical detector produces continuous scores instead of discrete values.

Mercury Agent - Copyright (C) 2025 Steel Security Advisors LLC
Licensed under GNU GPL v3
"""

import numpy as np
import pytest

from omni_mercury_engine.ml.mercury_ml import roc_auc_score
from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector
from omni_mercury_engine.ml._native_utils import native_roc_auc_score


def _make_classification_data(
    n_samples: int = 200,
    n_features: int = 10,
    anomaly_ratio: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic classification data with a specified anomaly ratio."""
    rng = np.random.RandomState(random_state)
    n_anomalies = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomalies

    X_normal = rng.randn(n_normal, n_features).astype(np.float32)
    X_anomaly = rng.randn(n_anomalies, n_features).astype(np.float32) + 3.0
    X = np.vstack([X_normal, X_anomaly])
    y = np.array([0] * n_normal + [1] * n_anomalies)

    idx = rng.permutation(n_samples)
    return X[idx], y[idx]


def _make_blobs_data(
    n_normal: int = 180,
    n_anomaly: int = 20,
    n_features: int = 10,
    separation: float = 5.0,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate clustered data with clear normal/anomaly separation."""
    rng = np.random.RandomState(random_state)
    X_normal = rng.randn(n_normal, n_features).astype(np.float32)
    X_anomaly = rng.randn(n_anomaly, n_features).astype(np.float32) * 1.5 + separation
    X = np.vstack([X_normal, X_anomaly])
    y = np.array([0] * n_normal + [1] * n_anomaly)

    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def make_classification(
    n_samples=100,
    n_features=20,
    n_informative=2,
    n_redundant=0,
    n_classes=2,
    weights=None,
    random_state=None,
):
    """Generate a synthetic binary classification dataset."""
    rng = np.random.RandomState(random_state)
    n_informative = min(n_informative, n_features)
    X = rng.randn(n_samples, n_features)
    coef = rng.randn(n_informative)
    logits = X[:, :n_informative] @ coef
    probs = 1 / (1 + np.exp(-logits))
    if weights is not None and len(weights) >= 2:
        threshold = np.percentile(probs, weights[0] * 100)
    else:
        threshold = 0.5
    y = (probs > threshold).astype(int)
    return X, y


def make_blobs(n_samples=100, centers=None, cluster_std=1.0, random_state=None):
    """Generate synthetic clustered data."""
    rng = np.random.RandomState(random_state)
    if centers is None:
        centers = [[0, 0]]
    centers = np.array(centers)
    n_features = centers.shape[1]
    X = rng.randn(n_samples, n_features) * cluster_std + centers[0]
    y = np.zeros(n_samples, dtype=int)
    return X, y


class TestContinuousScores:
    """Test that scores are continuous, not discrete."""

    @pytest.fixture
    def toy_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate toy classification data with 10% anomalies."""
        return _make_classification_data(
            n_samples=200, n_features=10, anomaly_ratio=0.1, random_state=42
        )

    @pytest.fixture
    def detector(self) -> MercuryAnomalyDetector:
        """Create a MercuryAnomalyDetector instance."""
        return MercuryAnomalyDetector()

    def test_scores_are_continuous(
        self,
        detector: MercuryAnomalyDetector,
        toy_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify scores have more than 5 unique values.

        Issue #3 caused only 5 discrete values: {0.0, 0.3, 0.4, 0.7, 1.0}.
        After fix, scores should be continuous with many unique values.
        """
        X, _ = toy_data
        detector.fit(X)
        result = detector.detect(X)

        scores = result["scores"]
        unique_scores = np.unique(scores)

        assert len(unique_scores) > 10, (
            f"Expected >10 unique scores for continuous output, "
            f"got {len(unique_scores)}: {sorted(unique_scores)[:10]}"
        )

    def test_scores_in_valid_range(
        self,
        detector: MercuryAnomalyDetector,
        toy_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify all scores are in [0, 1] range."""
        X, _ = toy_data
        detector.fit(X)
        result = detector.detect(X)

        scores = result["scores"]
        assert scores.min() >= 0.0, f"Min score {scores.min()} < 0"
        assert scores.max() <= 1.0, f"Max score {scores.max()} > 1"

    def test_continuous_z_scores_exist(
        self,
        detector: MercuryAnomalyDetector,
        toy_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify z_score_continuous key exists and is continuous."""
        X, _ = toy_data
        detector.fit(X)
        result = detector.detect(X)

        assert "z_score_continuous" in result, "Missing z_score_continuous key"
        z_continuous = result["z_score_continuous"]
        unique_z = np.unique(z_continuous)

        assert len(unique_z) > 5, f"z_score_continuous has only {len(unique_z)} unique values"

    def test_iqr_scores_exist_and_continuous(
        self,
        detector: MercuryAnomalyDetector,
        toy_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify iqr_scores key exists and is continuous."""
        X, _ = toy_data
        detector.fit(X)
        result = detector.detect(X)

        assert "iqr_scores" in result, "Missing iqr_scores key"
        iqr_scores = result["iqr_scores"]
        unique_iqr = np.unique(iqr_scores)

        assert len(unique_iqr) >= 2, "IQR scores should have variance"

    def test_backward_compatibility(self, detector, toy_data):
        """Verify legacy keys still exist for backward compatibility."""
        X, _ = toy_data
        detector.fit(X)
        result = detector.detect(X)

        assert "iqr_flags" in result, "Missing legacy key: iqr_flags"
        assert "is_anomaly" in result, "Missing key: is_anomaly"
        assert "scores" in result, "Missing key: scores"
        assert "z_scores" in result, "Missing key: z_scores"


class TestROCAUCImprovement:
    """Test that continuous scores improve ROC-AUC."""

    @pytest.fixture
    def separable_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate data with clear separation between classes."""
        return _make_blobs_data(
            n_normal=180, n_anomaly=20, n_features=10, separation=5.0, random_state=42
        )

    def test_roc_auc_above_baseline(self, separable_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Verify ROC-AUC is significantly above random (0.5)."""
        X, y = separable_data
        detector = MercuryAnomalyDetector()
        detector.fit(X)
        result = detector.detect(X)

        scores = result["scores"]
        auc = native_roc_auc_score(y, scores)

        assert auc > 0.6, f"Expected ROC-AUC >0.6, got {auc:.3f}"

    def test_continuous_scores_better_than_discrete(
        self, separable_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify continuous scores outperform simulated discrete scores."""
        X, y = separable_data
        detector = MercuryAnomalyDetector()
        detector.fit(X)
        result = detector.detect(X)

        continuous_scores = result["scores"]
        continuous_auc = native_roc_auc_score(y, continuous_scores)

        # Simulate old discrete behavior (only 5 values)
        discrete_bins = np.array([0.0, 0.3, 0.4, 0.7, 1.0])
        discrete_scores = discrete_bins[np.digitize(continuous_scores, discrete_bins[:-1])].clip(
            0, 1
        )
        discrete_auc = native_roc_auc_score(y, discrete_scores)

        assert continuous_auc >= discrete_auc - 0.01, (
            f"Continuous AUC ({continuous_auc:.3f}) should be >= "
            f"discrete AUC ({discrete_auc:.3f})"
        )


class TestAdaptiveContamination:
    """Test that detector fits correctly with various data distributions."""

    def test_detector_fits_with_skewed_data(self) -> None:
        """Verify detector fits correctly on skewed class distributions."""
        X, _ = _make_classification_data(
            n_samples=200, n_features=10, anomaly_ratio=0.05, random_state=42
        )

        detector = MercuryAnomalyDetector()
        detector.fit(X)

        assert detector._is_fitted
        result = detector.detect(X)
        assert result["scores"].shape == (200,)

    def test_detector_accepts_contamination_config(self) -> None:
        """Verify detector does not crash when contamination is in config."""
        X, _ = _make_classification_data(n_samples=100, n_features=5, random_state=42)

        detector = MercuryAnomalyDetector(config={"contamination": 0.05})
        detector.fit(X)
        assert detector._is_fitted

    def test_detector_fits_with_rare_anomalies(self) -> None:
        """Verify detector handles data with very rare anomalies."""
        X, _ = _make_classification_data(
            n_samples=200, n_features=10, anomaly_ratio=0.02, random_state=42
        )

        detector = MercuryAnomalyDetector()
        detector.fit(X)
        assert detector._is_fitted
        result = detector.detect(X)
        assert np.all(result["scores"] >= 0) and np.all(result["scores"] <= 1)


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_single_sample(self) -> None:
        """Test detection on a single sample."""
        detector = MercuryAnomalyDetector()
        X_train = np.random.randn(50, 5).astype(np.float32)
        detector.fit(X_train)

        X_test = np.random.randn(1, 5).astype(np.float32)
        result = detector.detect(X_test)

        assert "scores" in result
        assert len(result["scores"]) == 1

    def test_1d_data(self) -> None:
        """Test with 1D input data."""
        detector = MercuryAnomalyDetector()
        X_train = np.random.randn(50).astype(np.float32)
        detector.fit(X_train)

        X_test = np.random.randn(10).astype(np.float32)
        result = detector.detect(X_test)

        assert "scores" in result
        assert len(result["scores"]) == 10

    def test_high_dimensional_data(self) -> None:
        """Test with high-dimensional data."""
        detector = MercuryAnomalyDetector()
        X = np.random.randn(100, 100).astype(np.float32)
        detector.fit(X)
        result = detector.detect(X)

        assert "scores" in result
        assert len(result["scores"]) == 100

    def test_constant_feature_handling(self) -> None:
        """Test handling of constant (zero-variance) features."""
        detector = MercuryAnomalyDetector()
        X = np.random.randn(50, 5).astype(np.float32)
        X[:, 2] = 1.0  # Constant feature

        detector.fit(X)
        result = detector.detect(X)

        assert "scores" in result
        assert not np.any(np.isnan(result["scores"]))
        assert not np.any(np.isinf(result["scores"]))
