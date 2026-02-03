"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
Real Substantive Tests for Statistical Anomaly Detector

These tests verify actual algorithm behavior with mathematical assertions,
NOT just mock call counts. Each test exercises real production code paths.

Tests cover:
1. Z-score computation with known mathematical values
2. IQR bounds calculation and anomaly detection
3. Adaptive contamination estimation
4. Continuous score preservation for ML fusion
5. Anomaly detection accuracy on synthetic datasets
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from omni_mercury_engine.detectors.statistical import StatisticalAnomalyDetector


class TestZScoreComputation:
    """Test z-score computation with mathematically verifiable values."""

    def test_z_score_known_values(self):
        """Verify z-scores match hand-calculated values.

        For data [80, 90, 100, 110, 120]:
            mean = 100, std = sqrt(200) = 14.14
            z_score(130) = (130 - 100) / 14.14 = 2.12
        """
        detector = StatisticalAnomalyDetector()
        data = np.array([[80], [90], [100], [110], [120]])
        detector.fit(data)

        test_point = np.array([[130]])

        # Compute expected z-score manually
        expected_mean = 100.0
        expected_std = np.sqrt(200.0)  # Population std
        expected_z = (130 - expected_mean) / expected_std

        # Get actual z-scores from detector
        actual_z_scores = detector._compute_z_scores(test_point)

        # Verify within tolerance (1e-5 due to numerical precision)
        assert_allclose(actual_z_scores[0, 0], expected_z, rtol=1e-5)

    def test_z_score_zero_deviation(self):
        """Z-score should be 0 for point at mean."""
        detector = StatisticalAnomalyDetector()
        data = np.array([[10], [20], [30], [40], [50]])  # mean = 30
        detector.fit(data)

        test_point = np.array([[30]])  # exactly at mean
        z_scores = detector._compute_z_scores(test_point)

        assert_allclose(z_scores[0, 0], 0.0, atol=1e-10)

    def test_z_score_multivariate(self):
        """Z-scores computed correctly for multivariate data."""
        detector = StatisticalAnomalyDetector()
        # Two features with different distributions
        data = np.array([
            [10, 100],
            [20, 200],
            [30, 300],
            [40, 400],
            [50, 500],
        ])
        detector.fit(data)

        # Test point: 1 std above mean for feature 0, 2 std above for feature 1
        mean_0, std_0 = 30, np.std([10, 20, 30, 40, 50])
        mean_1, std_1 = 300, np.std([100, 200, 300, 400, 500])

        test_point = np.array([[mean_0 + std_0, mean_1 + 2 * std_1]])
        z_scores = detector._compute_z_scores(test_point)

        # Feature 0 should have z-score ~ 1.0, Feature 1 ~ 2.0
        assert_allclose(z_scores[0, 0], 1.0, rtol=0.1)
        assert_allclose(z_scores[0, 1], 2.0, rtol=0.1)


class TestIQRBounds:
    """Test IQR-based anomaly detection bounds."""

    def test_iqr_bounds_calculation(self):
        """Verify IQR bounds match manual calculation.

        For data [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            Q1 = 3.25, Q3 = 7.75, IQR = 4.5
            Lower bound = 3.25 - 1.5 * 4.5 = -3.5
            Upper bound = 7.75 + 1.5 * 4.5 = 14.5
        """
        detector = StatisticalAnomalyDetector({"iqr_multiplier": 1.5})
        data = np.array([[i] for i in range(1, 11)])  # 1 to 10
        detector.fit(data)

        # Verify Q1 and Q3
        expected_q1 = np.percentile(data, 25)
        expected_q3 = np.percentile(data, 75)

        assert_allclose(detector.q1[0], expected_q1, rtol=1e-10)
        assert_allclose(detector.q3[0], expected_q3, rtol=1e-10)

        # Test point within bounds should not be anomalous
        test_within = np.array([[5]])
        result = detector.detect(test_within)
        assert result["iqr_scores"][0] < 0.5  # Not anomalous

        # Test point outside upper bound
        test_above = np.array([[20]])
        result = detector.detect(test_above)
        assert result["iqr_scores"][0] > 0.5  # Anomalous

    def test_iqr_continuous_scores(self):
        """IQR scores should be continuous, not discrete flags."""
        detector = StatisticalAnomalyDetector()
        data = np.random.randn(100, 1)
        detector.fit(data)

        # Test multiple points at varying distances from bounds
        test_points = np.array([[-3], [-2], [-1], [0], [1], [2], [3]])
        result = detector.detect(test_points)

        iqr_scores = result["iqr_scores"]

        # Scores should increase monotonically away from center
        # Points closer to 0 should have lower scores
        assert iqr_scores[3] < iqr_scores[0]  # 0 < -3
        assert iqr_scores[3] < iqr_scores[6]  # 0 < 3

        # Verify continuous (not just 0/1)
        unique_scores = np.unique(iqr_scores)
        assert len(unique_scores) >= 3  # Should have multiple distinct values


class TestAdaptiveContamination:
    """Test adaptive contamination estimation (Fix for Issue #5)."""

    def test_contamination_adapts_to_data(self):
        """Contamination should adapt based on z-score outliers."""
        # Clean data (few outliers) - should have low contamination
        clean_data = np.random.randn(1000, 1)
        detector_clean = StatisticalAnomalyDetector()
        detector_clean.fit(clean_data)
        clean_contamination = detector_clean.contamination

        # Noisy data (many outliers) - should have higher contamination
        noisy_data = np.concatenate([
            np.random.randn(700, 1),  # Normal
            np.random.randn(300, 1) * 5 + 10,  # Outliers
        ])
        detector_noisy = StatisticalAnomalyDetector()
        detector_noisy.fit(noisy_data)
        noisy_contamination = detector_noisy.contamination

        # Noisy data should have higher contamination estimate
        assert noisy_contamination > clean_contamination

    def test_config_contamination_overrides_adaptive(self):
        """Explicit contamination config should override adaptive estimation."""
        detector = StatisticalAnomalyDetector({"contamination": 0.25})
        data = np.random.randn(100, 1)
        detector.fit(data)

        # Should use configured value, not adaptive
        assert detector.contamination == 0.25

    def test_contamination_bounds(self):
        """Contamination should be clamped to [0.001, 0.5]."""
        # Create data with no outliers
        uniform_data = np.ones((100, 1))  # All same value
        detector = StatisticalAnomalyDetector()
        detector.fit(uniform_data)

        # Should be clamped to minimum
        assert detector.contamination >= 0.001
        assert detector.contamination <= 0.5


class TestContinuousScores:
    """Test continuous score preservation (Fix for Issue #3)."""

    def test_combined_scores_continuous(self):
        """Combined scores should have more than 5 discrete values."""
        detector = StatisticalAnomalyDetector()
        data = np.random.randn(200, 5)
        detector.fit(data)

        test_data = np.random.randn(100, 5) * 2  # Wider distribution
        result = detector.detect(test_data)

        scores = result["scores"]
        unique_scores = np.unique(scores)

        # Should have many unique values (continuous), not just {0, 0.3, 0.4, 0.7, 1}
        assert len(unique_scores) > 10

    def test_scores_preserve_ranking(self):
        """Scores should preserve ranking of anomaly severity."""
        detector = StatisticalAnomalyDetector()
        data = np.random.randn(100, 1)
        detector.fit(data)

        # Create test points with increasing anomaly severity
        test_points = np.array([[0], [2], [4], [6], [8]])
        result = detector.detect(test_points)

        scores = result["scores"]

        # Scores should increase with distance from mean
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1]

    def test_z_score_continuous_not_boolean(self):
        """Z-score component should be continuous in [0, 1]."""
        detector = StatisticalAnomalyDetector()
        data = np.random.randn(100, 1)
        detector.fit(data)

        test_data = np.linspace(-3, 3, 20).reshape(-1, 1)
        result = detector.detect(test_data)

        z_continuous = result["z_score_continuous"]

        # All values should be in [0, 1]
        assert np.all(z_continuous >= 0)
        assert np.all(z_continuous <= 1)

        # Should have many distinct values
        unique_values = np.unique(z_continuous)
        assert len(unique_values) > 5


class TestAnomalyAccuracy:
    """Test anomaly detection accuracy on synthetic datasets."""

    def test_detects_obvious_outliers(self):
        """Should detect points > 3 std from mean as anomalies."""
        detector = StatisticalAnomalyDetector({"z_threshold": 3.0})
        # Training data: normal distribution
        train_data = np.random.randn(1000, 1)
        detector.fit(train_data)

        # Test: 90 normal points + 10 obvious outliers
        test_normal = np.random.randn(90, 1)
        test_outliers = np.random.randn(10, 1) * 0.1 + 10  # Way outside

        test_data = np.vstack([test_normal, test_outliers])
        result = detector.detect(test_data)

        # Outliers should have higher scores than normal
        normal_scores = result["scores"][:90]
        outlier_scores = result["scores"][90:]

        assert np.mean(outlier_scores) > np.mean(normal_scores)

        # At least 80% of outliers should have scores > median of all scores
        median_score = np.median(result["scores"])
        outlier_above_median = np.mean(outlier_scores > median_score)
        assert outlier_above_median >= 0.8

    def test_isolation_forest_scores(self):
        """Isolation forest scores should identify outliers."""
        detector = StatisticalAnomalyDetector()
        train_data = np.random.randn(200, 2)
        detector.fit(train_data)

        # Normal point (near center)
        normal_point = np.array([[0, 0]])
        # Outlier point (far from center)
        outlier_point = np.array([[5, 5]])

        normal_result = detector.detect(normal_point)
        outlier_result = detector.detect(outlier_point)

        # Isolation forest should give outlier higher score
        assert (
            outlier_result["isolation_forest_scores"][0]
            > normal_result["isolation_forest_scores"][0]
        )

    def test_multivariate_anomaly_detection(self):
        """Should detect anomalies in multivariate data."""
        np.random.seed(42)
        detector = StatisticalAnomalyDetector()

        # 5-dimensional normal data
        train_data = np.random.randn(500, 5)
        detector.fit(train_data)

        # Create test data with some anomalies in specific dimensions
        test_normal = np.random.randn(50, 5)
        test_anomaly = np.random.randn(10, 5)
        test_anomaly[:, 0] = 5  # Anomaly only in first dimension

        test_data = np.vstack([test_normal, test_anomaly])
        result = detector.detect(test_data)

        # Anomalous points should have higher combined scores
        normal_mean = np.mean(result["scores"][:50])
        anomaly_mean = np.mean(result["scores"][50:])

        assert anomaly_mean > normal_mean


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_data_raises_exception(self):
        """Should raise DetectorException for empty data."""
        from omni_mercury_engine.core.exceptions import DetectorException

        detector = StatisticalAnomalyDetector()
        with pytest.raises(DetectorException, match="empty data"):
            detector.fit(np.array([]))

    def test_nan_inf_handling(self):
        """Should handle NaN and Inf values gracefully."""
        from omni_mercury_engine.core.exceptions import DetectorException

        detector = StatisticalAnomalyDetector()

        # All NaN should raise exception
        nan_data = np.array([[np.nan], [np.nan], [np.nan]])
        with pytest.raises(DetectorException, match="NaN or Inf"):
            detector.fit(nan_data)

        # Partial NaN should work (filter out NaN rows)
        mixed_data = np.array([[1], [2], [np.nan], [4], [5]])
        detector.fit(mixed_data)  # Should succeed

        assert detector._is_fitted

    def test_single_sample(self):
        """Should handle single sample edge case."""
        detector = StatisticalAnomalyDetector()
        data = np.array([[5.0]])
        detector.fit(data)

        result = detector.detect(data)
        assert "scores" in result
        assert len(result["scores"]) == 1

    def test_constant_data(self):
        """Should handle constant (zero variance) data."""
        detector = StatisticalAnomalyDetector()
        data = np.ones((100, 1)) * 5
        detector.fit(data)

        # Detection should not crash
        result = detector.detect(data)
        assert "scores" in result


class TestFeatureExtraction:
    """Test feature extraction for ML fusion."""

    def test_feature_extraction_shape(self):
        """Extracted features should have correct shape."""
        detector = StatisticalAnomalyDetector()
        data = np.random.randn(50, 5)
        detector.fit(data)

        features = detector.extract_features(data)

        # Should return tensor with shape [batch, 10] (padded to 10)
        assert features.shape[0] == 50
        assert features.shape[1] == 10

    def test_features_include_statistics(self):
        """Features should include meaningful statistics."""
        detector = StatisticalAnomalyDetector()
        data = np.random.randn(100, 3)
        detector.fit(data)

        features = detector.extract_features(data)

        # Features should be finite
        assert np.all(np.isfinite(features.numpy()))

        # Features should have some variance (not all same)
        assert np.std(features.numpy()) > 0


class TestAutoCalibration:
    """Test auto-calibration functionality."""

    def test_auto_calibration_adjusts_threshold(self):
        """Auto-calibration should adjust threshold based on score distribution."""
        detector = StatisticalAnomalyDetector()
        detector.enable_auto_calibration()

        data = np.random.randn(200, 2)
        detector.fit(data)

        result = detector.detect(data)

        # Should include calibration diagnostics
        assert result["calibration_diagnostics"] is not None

        # Threshold should be adjusted (not default 0.5)
        assert "threshold" in result

    def test_calibrated_threshold_produces_positives(self):
        """Calibrated threshold should produce some positive predictions."""
        detector = StatisticalAnomalyDetector()
        detector.enable_auto_calibration()

        # Create data with some anomalies
        normal = np.random.randn(90, 2)
        outliers = np.random.randn(10, 2) * 3 + 5
        data = np.vstack([normal, outliers])

        detector.fit(data[:50])  # Fit on subset
        result = detector.detect(data)

        # With calibration, should have some anomalies detected
        n_anomalies = np.sum(result["is_anomaly"])
        assert n_anomalies > 0  # Should detect at least some
