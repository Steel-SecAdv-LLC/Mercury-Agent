"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisors LLC

Tests for Advanced Anomaly Detection Modules

Tests cover:
- MultiScaleTransformerDetector (time-series)
- ContrastiveLearningDetector (representation learning)
- AdversarialAutoencoderDetector (industrial control)
- COPODDetector (copula-based)
- GWOEnsembleDetector (optimized ensemble)
- PointAdjustmentEvaluator (evaluation protocol)
"""

from __future__ import annotations

import numpy as np
import pytest


# Test data generators
def generate_synthetic_timeseries(
    n_samples: int = 500,
    n_features: int = 10,
    anomaly_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic time-series data with anomalies."""
    rng = np.random.default_rng(seed)

    # Normal data
    X = rng.normal(0, 1, (n_samples, n_features))

    # Add temporal patterns
    for i in range(1, n_samples):
        X[i] = 0.7 * X[i - 1] + 0.3 * X[i]

    # Labels
    y = np.zeros(n_samples, dtype=int)

    # Inject anomalies
    n_anomalies = int(n_samples * anomaly_ratio)
    anomaly_starts = rng.choice(n_samples - 10, n_anomalies, replace=False)

    for start in anomaly_starts:
        segment_len = rng.integers(3, 10)
        end = min(start + segment_len, n_samples)
        y[start:end] = 1
        # Inject anomaly patterns
        X[start:end] += rng.normal(3, 1, (end - start, n_features))

    return X.astype(np.float32), y


def generate_industrial_data(
    n_samples: int = 500,
    n_sensors: int = 25,
    anomaly_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic industrial control system data."""
    rng = np.random.default_rng(seed)

    # Correlated sensor data
    mean = np.zeros(n_sensors)
    cov = np.eye(n_sensors)
    # Add correlations between adjacent sensors
    for i in range(n_sensors - 1):
        cov[i, i + 1] = 0.5
        cov[i + 1, i] = 0.5

    X = rng.multivariate_normal(mean, cov, n_samples)

    # Labels
    y = np.zeros(n_samples, dtype=int)

    # Inject attack patterns
    n_attacks = int(n_samples * anomaly_ratio)
    attack_idx = rng.choice(n_samples, n_attacks, replace=False)

    for idx in attack_idx:
        y[idx] = 1
        # Sensor manipulation attack
        attacked_sensors = rng.choice(n_sensors, rng.integers(1, 5), replace=False)
        X[idx, attacked_sensors] += rng.normal(4, 1, len(attacked_sensors))

    return X.astype(np.float32), y


class TestMultiScaleTransformerDetector:
    """Tests for MultiScaleTransformerDetector."""

    def test_initialization(self) -> None:
        """Test detector initialization."""
        from omni_mercury_engine.detectors.advanced import MultiScaleTransformerDetector

        detector = MultiScaleTransformerDetector(input_dim=10)
        assert detector is not None
        assert detector.config.input_dim == 10
        assert len(detector.config.window_sizes) == 3

    def test_fit_2d_input(self) -> None:
        """Test fitting with 2D input."""
        from omni_mercury_engine.detectors.advanced import MultiScaleTransformerDetector

        X, _ = generate_synthetic_timeseries(n_samples=200, n_features=10)

        detector = MultiScaleTransformerDetector(
            input_dim=10,
            epochs=2,
            batch_size=32,
            window_sizes=[5, 10],
        )
        detector.fit(X)

        assert detector._fitted
        assert detector.threshold > 0

    def test_predict_returns_scores(self) -> None:
        """Test prediction returns valid scores."""
        from omni_mercury_engine.detectors.advanced import MultiScaleTransformerDetector

        X_train, _ = generate_synthetic_timeseries(n_samples=200, n_features=10)
        X_test, _ = generate_synthetic_timeseries(n_samples=50, n_features=10, seed=123)

        detector = MultiScaleTransformerDetector(
            input_dim=10,
            epochs=2,
            batch_size=32,
            window_sizes=[5, 10],
        )
        detector.fit(X_train)
        scores = detector.predict(X_test)

        assert len(scores) > 0
        assert np.all(np.isfinite(scores))

    def test_detect_returns_dict(self) -> None:
        """Test detect returns proper dictionary."""
        from omni_mercury_engine.detectors.advanced import MultiScaleTransformerDetector

        X_train, _ = generate_synthetic_timeseries(n_samples=200, n_features=10)
        X_test, _ = generate_synthetic_timeseries(n_samples=50, n_features=10, seed=123)

        detector = MultiScaleTransformerDetector(
            input_dim=10,
            epochs=2,
            batch_size=32,
            window_sizes=[5, 10],
        )
        detector.fit(X_train)
        result = detector.detect(X_test)

        assert "anomaly_score" in result
        assert "predictions" in result
        assert "threshold" in result
        assert "detector_type" in result
        assert result["detector_type"] == "MultiScaleTransformer"


class TestContrastiveLearningDetector:
    """Tests for ContrastiveLearningDetector."""

    def test_initialization(self) -> None:
        """Test detector initialization."""
        from omni_mercury_engine.detectors.advanced import ContrastiveLearningDetector

        detector = ContrastiveLearningDetector(input_dim=10)
        assert detector is not None
        assert detector.config.input_dim == 10

    def test_fit_and_predict(self) -> None:
        """Test fitting and prediction."""
        from omni_mercury_engine.detectors.advanced import ContrastiveLearningDetector

        X_train, _ = generate_industrial_data(n_samples=200, n_sensors=10)
        X_test, _ = generate_industrial_data(n_samples=50, n_sensors=10, seed=123)

        detector = ContrastiveLearningDetector(
            input_dim=10,
            epochs=2,
            batch_size=32,
        )
        detector.fit(X_train)
        scores = detector.predict(X_test)

        assert len(scores) == len(X_test)
        assert np.all(np.isfinite(scores))

    def test_extract_features(self) -> None:
        """Test feature extraction."""
        from omni_mercury_engine.detectors.advanced import ContrastiveLearningDetector

        X_train, _ = generate_industrial_data(n_samples=200, n_sensors=10)

        detector = ContrastiveLearningDetector(
            input_dim=10,
            hidden_dim=64,
            epochs=2,
        )
        detector.fit(X_train)
        features = detector.extract_features(X_train[:50])

        assert features.shape[0] == 50
        assert features.shape[1] == 64  # hidden_dim


class TestAdversarialAutoencoderDetector:
    """Tests for AdversarialAutoencoderDetector."""

    def test_initialization(self) -> None:
        """Test detector initialization."""
        from omni_mercury_engine.detectors.advanced import AdversarialAutoencoderDetector

        detector = AdversarialAutoencoderDetector(input_dim=25)
        assert detector is not None
        assert detector.config.input_dim == 25

    def test_fit_2d_input(self) -> None:
        """Test fitting with 2D input."""
        from omni_mercury_engine.detectors.advanced import AdversarialAutoencoderDetector

        X, _ = generate_industrial_data(n_samples=200, n_sensors=25)

        detector = AdversarialAutoencoderDetector(
            input_dim=25,
            hidden_dims=[64, 32],
            latent_dim=8,
            epochs=2,
            batch_size=32,
        )
        detector.fit(X)

        assert detector._fitted
        assert detector.threshold > 0

    def test_predict_returns_scores(self) -> None:
        """Test prediction returns valid scores."""
        from omni_mercury_engine.detectors.advanced import AdversarialAutoencoderDetector

        X_train, _ = generate_industrial_data(n_samples=200, n_sensors=25)
        X_test, _ = generate_industrial_data(n_samples=50, n_sensors=25, seed=123)

        detector = AdversarialAutoencoderDetector(
            input_dim=25,
            hidden_dims=[64, 32],
            latent_dim=8,
            epochs=2,
        )
        detector.fit(X_train)
        scores = detector.predict(X_test)

        assert len(scores) == len(X_test)
        assert np.all(np.isfinite(scores))

    def test_extract_features(self) -> None:
        """Test latent space extraction."""
        from omni_mercury_engine.detectors.advanced import AdversarialAutoencoderDetector

        X_train, _ = generate_industrial_data(n_samples=200, n_sensors=25)

        detector = AdversarialAutoencoderDetector(
            input_dim=25,
            latent_dim=8,
            epochs=2,
        )
        detector.fit(X_train)
        features = detector.extract_features(X_train[:50])

        assert features.shape[0] == 50
        assert features.shape[1] == 8  # latent_dim


class TestCOPODDetector:
    """Tests for COPODDetector."""

    def test_initialization(self) -> None:
        """Test detector initialization."""
        from omni_mercury_engine.detectors.advanced import COPODDetector

        detector = COPODDetector()
        assert detector is not None

    def test_fit_and_predict(self) -> None:
        """Test fitting and prediction."""
        from omni_mercury_engine.detectors.advanced import COPODDetector

        X_train, _ = generate_industrial_data(n_samples=500, n_sensors=10)
        X_test, _ = generate_industrial_data(n_samples=100, n_sensors=10, seed=123)

        detector = COPODDetector()
        detector.fit(X_train)
        scores = detector.predict(X_test)

        assert len(scores) == len(X_test)
        assert np.all(np.isfinite(scores))

    def test_no_hyperparameters(self) -> None:
        """Test COPOD works without hyperparameter tuning."""
        from omni_mercury_engine.detectors.advanced import COPODDetector

        X, _ = generate_industrial_data(n_samples=500, n_sensors=20)

        # Should work with default config
        detector = COPODDetector()
        detector.fit(X)
        scores = detector.predict(X)

        assert len(scores) == len(X)

    def test_feature_importance(self) -> None:
        """Test per-feature anomaly scores."""
        from omni_mercury_engine.detectors.advanced import COPODDetector

        X, _ = generate_industrial_data(n_samples=500, n_sensors=10)

        detector = COPODDetector()
        detector.fit(X)
        importance = detector.get_feature_importance(X[:50])

        assert importance.shape == (50, 10)


class TestGWOEnsembleDetector:
    """Tests for GWOEnsembleDetector."""

    def test_initialization(self) -> None:
        """Test detector initialization."""
        from omni_mercury_engine.detectors.advanced import GWOEnsembleDetector

        detector = GWOEnsembleDetector()
        assert detector is not None
        assert len(detector.detectors) == 0

    def test_add_detectors(self) -> None:
        """Test adding detectors to ensemble."""
        from sklearn.ensemble import IsolationForest

        from omni_mercury_engine.detectors.advanced import GWOEnsembleDetector

        detector = GWOEnsembleDetector()
        detector.add_detector(IsolationForest(contamination=0.1, random_state=42))
        detector.add_detector(IsolationForest(contamination=0.05, random_state=42))

        assert len(detector.detectors) == 2

    def test_fit_without_labels(self) -> None:
        """Test fitting without validation labels (equal weights)."""
        from sklearn.ensemble import IsolationForest

        from omni_mercury_engine.detectors.advanced import GWOEnsembleDetector

        X, _ = generate_industrial_data(n_samples=200, n_sensors=10)

        detector = GWOEnsembleDetector()
        detector.add_detector(IsolationForest(contamination=0.1, random_state=42))
        detector.add_detector(IsolationForest(contamination=0.05, random_state=42))

        detector.fit(X)

        assert detector._fitted
        assert detector.weights is not None
        assert np.allclose(detector.weights, [0.5, 0.5])  # Equal weights

    def test_fit_with_labels(self) -> None:
        """Test fitting with validation labels (optimized weights)."""
        from sklearn.ensemble import IsolationForest

        from omni_mercury_engine.detectors.advanced import GWOEnsembleDetector

        X, y = generate_industrial_data(n_samples=200, n_sensors=10)

        detector = GWOEnsembleDetector(n_wolves=5, max_iterations=5)
        detector.add_detector(IsolationForest(contamination=0.1, random_state=42))
        detector.add_detector(IsolationForest(contamination=0.05, random_state=42))

        detector.fit(X, y_val=y)

        assert detector._fitted
        assert detector.weights is not None
        assert abs(detector.weights.sum() - 1.0) < 0.01  # Weights sum to 1

    def test_predict(self) -> None:
        """Test prediction."""
        from sklearn.ensemble import IsolationForest

        from omni_mercury_engine.detectors.advanced import GWOEnsembleDetector

        X_train, y = generate_industrial_data(n_samples=200, n_sensors=10)
        X_test, _ = generate_industrial_data(n_samples=50, n_sensors=10, seed=123)

        detector = GWOEnsembleDetector()
        detector.add_detector(IsolationForest(contamination=0.1, random_state=42))
        detector.fit(X_train, y_val=y)

        scores = detector.predict(X_test)
        assert len(scores) == len(X_test)


class TestPointAdjustmentEvaluator:
    """Tests for PointAdjustmentEvaluator."""

    def test_find_anomaly_segments(self) -> None:
        """Test finding contiguous anomaly segments."""
        from omni_mercury_engine.detectors.advanced import find_anomaly_segments

        labels = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
        segments = find_anomaly_segments(labels)

        assert len(segments) == 2
        assert segments[0].start == 2
        assert segments[0].end == 5
        assert segments[0].length == 3
        assert segments[1].start == 7
        assert segments[1].end == 9

    def test_adjust_predictions(self) -> None:
        """Test point-adjustment of predictions."""
        from omni_mercury_engine.detectors.advanced import adjust_predictions

        labels = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
        predictions = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])  # Only detect one point

        adjusted = adjust_predictions(predictions, labels)

        # Entire first segment should be marked as detected
        assert adjusted[2] == 1
        assert adjusted[3] == 1
        assert adjusted[4] == 1

        # Second segment not detected
        assert adjusted[7] == 0

    def test_compute_adjusted_metrics(self) -> None:
        """Test computing adjusted metrics."""
        from omni_mercury_engine.detectors.advanced import compute_adjusted_metrics

        labels = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
        predictions = np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 0])

        metrics = compute_adjusted_metrics(predictions, labels)

        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert metrics["recall"] > 0  # At least partial detection

    def test_evaluator_report(self) -> None:
        """Test evaluation report generation."""
        from omni_mercury_engine.detectors.advanced import PointAdjustmentEvaluator

        labels = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
        predictions = np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 0])

        evaluator = PointAdjustmentEvaluator()
        report = evaluator.report(predictions, labels)

        assert "POINT-ADJUSTED EVALUATION REPORT" in report
        assert "Precision:" in report
        assert "Recall:" in report
        assert "F1 Score:" in report


class TestIntegration:
    """Integration tests for advanced detectors."""

    def test_detector_pipeline(self) -> None:
        """Test full pipeline: fit, predict, evaluate."""
        from omni_mercury_engine.detectors.advanced import (
            COPODDetector,
            PointAdjustmentEvaluator,
        )

        X_train, _ = generate_synthetic_timeseries(n_samples=300, n_features=10)
        X_test, y_test = generate_synthetic_timeseries(n_samples=100, n_features=10, seed=123)

        # Fit detector
        detector = COPODDetector()
        detector.fit(X_train)

        # Predict
        scores = detector.predict(X_test)
        result = detector.detect(X_test)

        # Evaluate with point-adjustment
        evaluator = PointAdjustmentEvaluator()
        metrics = evaluator.evaluate(
            predictions=result["predictions"],
            labels=y_test,
            scores=scores,
        )

        assert metrics["f1"] >= 0
        assert metrics["precision"] >= 0
        assert metrics["recall"] >= 0

    def test_ensemble_with_advanced_detectors(self) -> None:
        """Test GWO ensemble with advanced detectors."""
        from omni_mercury_engine.detectors.advanced import (
            COPODDetector,
            GWOEnsembleDetector,
        )

        X_train, y = generate_industrial_data(n_samples=300, n_sensors=15)
        X_test, _ = generate_industrial_data(n_samples=50, n_sensors=15, seed=123)

        # Create ensemble
        ensemble = GWOEnsembleDetector(n_wolves=5, max_iterations=5)
        ensemble.add_detector(COPODDetector())

        # Fit with GWO optimization
        ensemble.fit(X_train, y_val=y)

        # Predict
        scores = ensemble.predict(X_test)
        result = ensemble.detect(X_test)

        assert len(scores) == len(X_test)
        assert "weights" in result


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_list_detectors(self) -> None:
        """Test list_detectors returns all detectors."""
        from omni_mercury_engine.detectors.advanced import list_detectors

        detectors = list_detectors()
        assert len(detectors) == 6
        assert "MultiScaleTransformerDetector" in detectors
        assert "COPODDetector" in detectors
        assert "GWOEnsembleDetector" in detectors

    def test_create_detector_copod(self) -> None:
        """Test creating COPOD detector via factory."""
        from omni_mercury_engine.detectors.advanced import COPODDetector, create_detector

        detector = create_detector("copod")
        assert isinstance(detector, COPODDetector)

    def test_create_detector_fast_alias(self) -> None:
        """Test 'fast' alias creates COPOD detector."""
        from omni_mercury_engine.detectors.advanced import COPODDetector, create_detector

        detector = create_detector("fast")
        assert isinstance(detector, COPODDetector)

    def test_create_detector_timeseries(self) -> None:
        """Test creating timeseries detector via factory."""
        from omni_mercury_engine.detectors.advanced import (
            MultiScaleTransformerDetector,
            create_detector,
        )

        detector = create_detector("timeseries", input_dim=10, window_sizes=[5], epochs=1)
        assert isinstance(detector, MultiScaleTransformerDetector)

    def test_create_detector_industrial(self) -> None:
        """Test creating industrial detector via factory."""
        from omni_mercury_engine.detectors.advanced import (
            AdversarialAutoencoderDetector,
            create_detector,
        )

        detector = create_detector("industrial", input_dim=10, epochs=1)
        assert isinstance(detector, AdversarialAutoencoderDetector)

    def test_create_detector_contrastive(self) -> None:
        """Test creating contrastive detector via factory."""
        from omni_mercury_engine.detectors.advanced import (
            ContrastiveLearningDetector,
            create_detector,
        )

        detector = create_detector("contrastive", input_dim=10, epochs=1)
        assert isinstance(detector, ContrastiveLearningDetector)

    def test_create_detector_ensemble(self) -> None:
        """Test creating ensemble detector via factory."""
        from omni_mercury_engine.detectors.advanced import GWOEnsembleDetector, create_detector

        detector = create_detector("ensemble", n_wolves=5)
        assert isinstance(detector, GWOEnsembleDetector)

    def test_create_detector_invalid_type(self) -> None:
        """Test invalid detector type raises error."""
        from omni_mercury_engine.detectors.advanced import create_detector

        with pytest.raises(ValueError, match="Unknown detector type"):
            create_detector("invalid_type")  # type: ignore


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
