"""
Tests for Adaptive Anomaly Detector module.

Validates:
- AdaptiveThresholdCalibrator (fixes covtype F1=0 issue)
- CovarianceAwareDetector (improves batadal AUC)
- TemporalPatternDetector (enhances smd detection)
- AdaptiveAnomalyDetector (unified interface)

Copyright (C) 2025 Steel Security Advisors LLC
"""

import numpy as np
import pytest

from omni_mercury_engine.core.adaptive_detector import (
    AdaptiveAnomalyDetector,
    AdaptiveThresholdCalibrator,
    CovarianceAwareDetector,
    DatasetProfile,
    DatasetSpecificEnsemble,
    DetectionResult,
    TemporalPatternDetector,
)


class TestAdaptiveThresholdCalibrator:
    """Tests for threshold calibration - addresses covtype F1=0 issue."""

    def test_percentile_calibration(self):
        """Test percentile-based calibration produces non-zero predictions."""
        calibrator = AdaptiveThresholdCalibrator(contamination=0.1)

        # Bimodal scores simulating normal + anomaly distributions
        normal_scores = np.random.normal(0.3, 0.1, 900)
        anomaly_scores = np.random.normal(0.8, 0.1, 100)
        scores = np.concatenate([normal_scores, anomaly_scores])

        threshold, predictions = calibrator.calibrate(scores, method="percentile")

        assert threshold > 0
        assert predictions.sum() > 0, "Should have non-zero predictions (fixes F1=0)"
        assert predictions.sum() < len(predictions), "Shouldn't flag everything"

    def test_otsu_calibration(self):
        """Test Otsu method finds optimal bimodal threshold."""
        calibrator = AdaptiveThresholdCalibrator(contamination=0.05)

        # Create clearly bimodal distribution
        normal_scores = np.random.normal(2.0, 0.5, 950)
        anomaly_scores = np.random.normal(8.0, 0.5, 50)
        scores = np.concatenate([normal_scores, anomaly_scores])

        threshold, predictions = calibrator.calibrate(scores, method="otsu")

        # Threshold should be between the two modes
        assert 3.0 < threshold < 7.0
        assert predictions.sum() > 0

    def test_mad_calibration(self):
        """Test MAD-based robust calibration."""
        calibrator = AdaptiveThresholdCalibrator(contamination=0.05)

        # Normal distribution with outliers
        scores = np.random.normal(5.0, 1.0, 1000)
        scores[:50] = np.random.normal(15.0, 1.0, 50)  # Outliers

        threshold, predictions = calibrator.calibrate(scores, method="mad")

        assert threshold > 5.0  # Should be above median
        assert predictions.sum() > 0

    def test_empty_predictions_fallback(self):
        """Test fallback when MAD produces no predictions."""
        calibrator = AdaptiveThresholdCalibrator(contamination=0.05, min_contamination=0.01)

        # Uniform scores (no clear outliers)
        scores = np.random.uniform(0, 1, 1000)

        threshold, predictions = calibrator.calibrate(scores, method="mad")

        # Should fall back to percentile and still produce predictions
        assert predictions.sum() >= 10  # At least 1% of 1000


class TestCovarianceAwareDetector:
    """Tests for covariance detection - addresses batadal AUC=0.5458 issue."""

    def test_fit_and_score(self):
        """Test basic fit and scoring."""
        detector = CovarianceAwareDetector(contamination=0.1)

        # Create data with covariance structure
        n_samples = 500
        X = np.random.multivariate_normal(
            mean=[0, 0, 0],
            cov=[[1, 0.8, 0.6], [0.8, 1, 0.5], [0.6, 0.5, 1]],
            size=n_samples,
        )

        detector.fit(X)
        scores = detector.score_samples(X)

        assert len(scores) == n_samples
        assert scores.min() >= 0  # Mahalanobis distance is non-negative

    def test_detect_outliers(self):
        """Test that outliers have higher scores."""
        detector = CovarianceAwareDetector(contamination=0.1)

        # Training data (normal)
        X_train = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]], size=500)

        detector.fit(X_train)

        # Normal test point
        normal_score = detector.score_samples(np.array([[0.5, 0.5]]))[0]

        # Outlier test point
        outlier_score = detector.score_samples(np.array([[10, -10]]))[0]

        assert outlier_score > normal_score

    def test_regularization(self):
        """Test detector handles near-singular covariance."""
        detector = CovarianceAwareDetector(contamination=0.1)

        # Create data with high correlation (near-singular covariance)
        n = 100
        x = np.random.normal(0, 1, n)
        X = np.column_stack([x, x + np.random.normal(0, 0.01, n)])  # Almost identical

        detector.fit(X)  # Should not raise
        scores = detector.score_samples(X)

        assert len(scores) == n
        assert np.isfinite(scores).all()


class TestTemporalPatternDetector:
    """Tests for temporal detection - addresses smd F1=0.06 issue."""

    def test_transform_adds_features(self):
        """Test that temporal transformation adds features."""
        detector = TemporalPatternDetector(window_sizes=[5, 10], lag_features=2, include_diff=True)

        X = np.random.randn(100, 5)  # 100 samples, 5 features
        X_transformed = detector.transform(X)

        # Should have more features than original
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] > X.shape[1]

    def test_lag_features(self):
        """Test lag features are computed correctly."""
        detector = TemporalPatternDetector(window_sizes=[], lag_features=1, include_diff=False)

        # Simple time series
        X = np.array([[1], [2], [3], [4], [5]], dtype=float)
        X_transformed = detector.transform(X)

        # Second column should be lag-1 of first
        expected_lag1 = np.array([[0], [1], [2], [3], [4]], dtype=float)
        np.testing.assert_array_equal(X_transformed[:, 1:2], expected_lag1)

    def test_rolling_stats(self):
        """Test rolling statistics are computed."""
        detector = TemporalPatternDetector(
            window_sizes=[3],
            lag_features=0,
            include_diff=False,
            include_rolling_stats=True,
        )

        X = np.ones((10, 2)) * 5  # Constant values
        X_transformed = detector.transform(X)

        # Rolling mean should be 5, rolling std should be ~0
        # Features: original(2) + rmean3(2) + rstd3(2) + rdev3(2) = 8
        assert X_transformed.shape[1] >= 8

    def test_feature_names(self):
        """Test feature names are generated."""
        detector = TemporalPatternDetector(lag_features=1, include_diff=True)

        X = np.random.randn(20, 3)
        detector.transform(X)

        names = detector.feature_names
        assert len(names) > 0
        assert any("lag" in n for n in names)
        assert any("diff" in n for n in names)


class TestAdaptiveAnomalyDetector:
    """Tests for unified adaptive detector."""

    def test_auto_profile_temporal(self):
        """Test auto-profiling detects temporal data."""
        detector = AdaptiveAnomalyDetector(auto_profile=True)

        # Create time series with strong autocorrelation
        n = 200
        X = np.zeros((n, 3))
        for i in range(n):
            if i > 0:
                X[i] = 0.9 * X[i - 1] + np.random.randn(3) * 0.1

        profile = detector.profile_dataset(X)

        assert profile == DatasetProfile.TEMPORAL

    def test_auto_profile_covariance(self):
        """Test auto-profiling detects covariance structure."""
        detector = AdaptiveAnomalyDetector(auto_profile=True)

        # Create data with strong correlations
        X = np.random.multivariate_normal(
            mean=[0, 0, 0, 0],
            cov=[
                [1, 0.9, 0.85, 0.8],
                [0.9, 1, 0.88, 0.82],
                [0.85, 0.88, 1, 0.87],
                [0.8, 0.82, 0.87, 1],
            ],
            size=300,
        )

        profile = detector.profile_dataset(X)

        assert profile == DatasetProfile.COVARIANCE_STRUCTURED

    def test_auto_profile_high_dim(self):
        """Test auto-profiling detects high-dimensional data."""
        detector = AdaptiveAnomalyDetector(auto_profile=True)

        # Create high-dimensional data (like covtype with 54 features)
        X = np.random.randn(200, 50)

        profile = detector.profile_dataset(X)

        assert profile == DatasetProfile.HIGH_DIMENSIONAL

    def test_detect_returns_result(self):
        """Test detection returns proper result object."""
        detector = AdaptiveAnomalyDetector(contamination=0.1)

        X = np.random.randn(100, 10)
        result = detector.detect(X)

        assert isinstance(result, DetectionResult)
        assert len(result.scores) == 100
        assert len(result.predictions) == 100
        assert result.threshold > 0
        assert 0 <= result.confidence <= 1
        assert result.calibration_method in [
            "percentile",
            "otsu",
            "mad",
            "bimodal",
        ]

    def test_detect_temporal_profile(self):
        """Test detection with temporal profile."""
        detector = AdaptiveAnomalyDetector(auto_profile=False)
        detector._profile = DatasetProfile.TEMPORAL

        # Create simple time series
        X = np.random.randn(100, 5)
        result = detector.detect(X)

        assert result.profile_used == DatasetProfile.TEMPORAL
        assert "n_temporal_features" in result.metadata

    def test_ethics_evaluation(self):
        """Test ethics evaluation checks pass."""
        detector = AdaptiveAnomalyDetector()

        X = np.random.randn(100, 10)
        result = detector.detect(X)
        ethics = detector.evaluate_ethics(result)

        assert "benevolence" in ethics
        assert "sigma_immutable" in ethics
        assert "passes" in ethics
        assert ethics["sigma_immutable"] >= 0.0


class TestDatasetSpecificEnsemble:
    """Tests for dataset-specific optimizations."""

    @pytest.mark.parametrize(
        "dataset_name,expected_profile",
        [
            ("covtype", DatasetProfile.HIGH_DIMENSIONAL),
            ("batadal", DatasetProfile.COVARIANCE_STRUCTURED),
            ("smd", DatasetProfile.TEMPORAL),
            ("nsl_kdd", DatasetProfile.NETWORK),
            ("breast_cancer", DatasetProfile.MEDICAL),
        ],
    )
    def test_dataset_profile_mapping(self, dataset_name, expected_profile):
        """Test correct profile is assigned for each dataset."""
        ensemble = DatasetSpecificEnsemble(contamination=0.1)

        detector = ensemble.create_detector_for_dataset(dataset_name)

        assert detector._profile == expected_profile

    def test_detect_with_dataset_hint(self):
        """Test detection with dataset hint."""
        ensemble = DatasetSpecificEnsemble(contamination=0.1)

        X = np.random.randn(100, 10)
        result = ensemble.detect_with_dataset_hint(X, "covtype")

        assert result.profile_used == DatasetProfile.HIGH_DIMENSIONAL

    def test_unknown_dataset_uses_generic(self):
        """Test unknown datasets use generic profile."""
        ensemble = DatasetSpecificEnsemble()

        detector = ensemble.create_detector_for_dataset("unknown_dataset")

        assert detector._profile == DatasetProfile.GENERIC


class TestIntegration:
    """Integration tests for realistic scenarios."""

    def test_covtype_scenario(self):
        """Test covtype-like data (high-dim) with Otsu calibration."""
        detector = AdaptiveAnomalyDetector(contamination=0.05, auto_profile=False)
        detector._profile = DatasetProfile.HIGH_DIMENSIONAL

        # Simulate covtype-like data: 54 features, ~2% anomaly
        n_normal = 1900
        n_anomaly = 100
        n_features = 54

        X_normal = np.random.randn(n_normal, n_features) * 0.5
        X_anomaly = np.random.randn(n_anomaly, n_features) * 0.5 + 3  # Shifted
        X = np.vstack([X_normal, X_anomaly])

        result = detector.detect(X)

        # Key test: We should have non-zero predictions (fixes F1=0 issue)
        assert result.predictions.sum() > 0, "Must produce positive predictions"

        # Predictions should primarily be in the anomaly region
        anomaly_pred_in_anomaly = result.predictions[n_normal:].sum()
        assert anomaly_pred_in_anomaly > n_anomaly * 0.3, "Should detect significant anomalies"

    def test_batadal_scenario(self):
        """Test batadal-like data (covariance structured)."""
        detector = AdaptiveAnomalyDetector(contamination=0.1, auto_profile=False)
        detector._profile = DatasetProfile.COVARIANCE_STRUCTURED

        # Simulate batadal: correlated sensor readings
        cov = np.array(
            [
                [1.0, 0.7, 0.5, 0.3],
                [0.7, 1.0, 0.6, 0.4],
                [0.5, 0.6, 1.0, 0.5],
                [0.3, 0.4, 0.5, 1.0],
            ]
        )
        X_normal = np.random.multivariate_normal([0, 0, 0, 0], cov, size=450)

        # Anomalies break the correlation structure
        X_anomaly = np.random.randn(50, 4) * 3

        X = np.vstack([X_normal, X_anomaly])
        detector.fit(X_normal)
        result = detector.detect(X)

        # Anomaly scores should be higher for anomalies
        normal_scores = result.scores[:450].mean()
        anomaly_scores = result.scores[450:].mean()

        assert anomaly_scores > normal_scores, "Anomalies should have higher scores"

    def test_smd_scenario(self):
        """Test smd-like data (temporal patterns)."""
        detector = AdaptiveAnomalyDetector(contamination=0.1, auto_profile=False)
        detector._profile = DatasetProfile.TEMPORAL

        # Simulate server metrics time series
        n = 300
        t = np.arange(n)

        # Normal: smooth with autocorrelation
        X_normal = np.column_stack(
            [
                np.sin(t / 10) + np.random.randn(n) * 0.1,  # CPU
                np.cos(t / 8) + np.random.randn(n) * 0.1,  # Memory
                np.sin(t / 12) + np.random.randn(n) * 0.1,  # Network
            ]
        )

        # Inject anomalies (sudden spikes)
        anomaly_indices = [50, 100, 150, 200, 250]
        for idx in anomaly_indices:
            X_normal[idx] = [5, -5, 5]  # Sudden spike

        result = detector.detect(X_normal)

        # Check that anomaly points have higher scores
        anomaly_scores = [result.scores[i] for i in anomaly_indices]
        normal_scores = [result.scores[i] for i in range(n) if i not in anomaly_indices]

        assert np.mean(anomaly_scores) > np.percentile(
            normal_scores, 80
        ), "Anomalies should score above 80th percentile"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
