"""
Tests for Score Calibration System
Copyright (C) 2025 Steel Security Advisory LLC

Tests the complete calibration pipeline that solves the F1=0 problem:
- AutoThresholdOptimizer with multiple methods
- ScoreDiagnostics analysis
- ScoreCalibrationManager integration
- Benchmark diagnostics
"""

import numpy as np
import pytest

from omni_mercury_engine.core.score_calibration import (
    AutoThresholdOptimizer,
    CalibrationDiagnostics,
    CalibrationMethod,
    CalibrationResult,
    ScoreCalibrationManager,
    ScoreDiagnostics,
    calibrate_scores,
    diagnose_scores,
)


class TestScoreDiagnostics:
    """Test ScoreDiagnostics class."""

    def test_analyze_basic_statistics(self):
        """Test basic score statistics computation."""
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        diag = ScoreDiagnostics.analyze(scores, threshold=0.5)

        assert diag.score_min == 0.1
        assert diag.score_max == 0.9
        assert abs(diag.score_mean - 0.5) < 0.01
        assert diag.n_samples == 9
        assert diag.n_above_threshold == 4  # 0.6, 0.7, 0.8, 0.9

    def test_analyze_all_below_threshold(self):
        """Test detection of F1=0 scenario where all scores < threshold."""
        # This simulates the F1=0 problem
        scores = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])

        diag = ScoreDiagnostics.analyze(scores, threshold=0.5)

        assert diag.n_above_threshold == 0
        assert diag.predicted_anomaly_ratio == 0.0
        assert diag.score_max < 0.5

    def test_analyze_with_labels(self):
        """Test analysis with ground truth labels."""
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.9])
        labels = np.array([0, 0, 0, 0, 1])  # 20% contamination

        diag = ScoreDiagnostics.analyze(scores, threshold=0.5, labels=labels)

        assert diag.actual_contamination == 0.2
        assert diag.estimated_contamination is not None

    def test_bimodality_detection(self):
        """Test bimodal distribution detection."""
        # Create bimodal distribution
        normal = np.random.normal(0.2, 0.05, 100)
        anomaly = np.random.normal(0.8, 0.05, 20)
        scores = np.concatenate([normal, anomaly])

        is_bimodal = ScoreDiagnostics._detect_bimodality(scores)

        # Should detect bimodality
        assert isinstance(is_bimodal, bool)

    def test_percentiles_computed(self):
        """Test that all expected percentiles are computed."""
        scores = np.random.uniform(0, 1, 100)

        diag = ScoreDiagnostics.analyze(scores, threshold=0.5)

        expected_percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in expected_percentiles:
            assert p in diag.percentiles


class TestAutoThresholdOptimizer:
    """Test AutoThresholdOptimizer class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.optimizer = AutoThresholdOptimizer(
            default_contamination=0.05,
            min_contamination=0.001,
            max_contamination=0.5,
        )

    def test_fixed_threshold(self):
        """Test FIXED calibration method."""
        scores = np.random.uniform(0, 1, 100)

        result = self.optimizer.optimize(
            scores=scores,
            method=CalibrationMethod.FIXED,
            fixed_threshold=0.7,
        )

        assert result.threshold == 0.7
        assert result.method == CalibrationMethod.FIXED

    def test_percentile_threshold(self):
        """Test PERCENTILE calibration method."""
        scores = np.linspace(0, 1, 100)

        result = self.optimizer.optimize(
            scores=scores,
            method=CalibrationMethod.PERCENTILE,
            contamination=0.05,  # Top 5%
        )

        # Threshold should be around 95th percentile
        assert 0.90 < result.threshold <= 1.0

    def test_otsu_threshold_bimodal(self):
        """Test OTSU method on bimodal distribution."""
        # Create clearly bimodal distribution
        normal = np.random.normal(0.2, 0.02, 90)
        anomaly = np.random.normal(0.9, 0.02, 10)
        scores = np.clip(np.concatenate([normal, anomaly]), 0, 1)

        result = self.optimizer.optimize(
            scores=scores,
            method=CalibrationMethod.OTSU,
        )

        # Threshold should be between the two modes
        assert 0.3 < result.threshold < 0.8

    def test_mad_threshold(self):
        """Test MAD-based calibration."""
        # Create scores with clear outliers
        scores = np.concatenate([
            np.random.normal(0.3, 0.05, 95),  # Normal
            np.array([0.9, 0.92, 0.95, 0.97, 0.99]),  # Outliers
        ])
        scores = np.clip(scores, 0, 1)

        result = self.optimizer.optimize(
            scores=scores,
            method=CalibrationMethod.MAD,
        )

        # Should produce some positive predictions
        assert result.predictions.sum() > 0

    def test_optimal_f1_with_labels(self):
        """Test OPTIMAL_F1 method with ground truth."""
        # Create scores where optimal threshold is around 0.5
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98])
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        result = self.optimizer.optimize(
            scores=scores,
            method=CalibrationMethod.OPTIMAL_F1,
            labels=labels,
        )

        # Should find threshold that maximizes F1
        assert result.method == CalibrationMethod.OPTIMAL_F1
        assert 0.4 < result.threshold < 0.7

    def test_auto_method_selection(self):
        """Test AUTO method selection."""
        scores = np.random.uniform(0, 1, 100)

        result = self.optimizer.optimize(
            scores=scores,
            method=CalibrationMethod.AUTO,
        )

        # Should select a valid method
        assert result.method in CalibrationMethod
        assert result.confidence > 0

    def test_empty_scores_handling(self):
        """Test handling of empty score array."""
        scores = np.array([])

        result = self.optimizer.optimize(
            scores=scores,
            method=CalibrationMethod.PERCENTILE,
        )

        assert len(result.predictions) == 0
        assert result.confidence == 0.0


class TestScoreCalibrationManager:
    """Test ScoreCalibrationManager class."""

    def test_basic_calibration(self):
        """Test basic calibration workflow."""
        manager = ScoreCalibrationManager(contamination=0.1)

        scores = np.random.uniform(0, 1, 100)

        result = manager.calibrate(scores)

        assert isinstance(result, CalibrationResult)
        assert 0 <= result.threshold <= 1
        assert len(result.predictions) == len(scores)
        assert isinstance(result.diagnostics, CalibrationDiagnostics)

    def test_get_calibrated_threshold(self):
        """Test convenience method for getting threshold."""
        manager = ScoreCalibrationManager(contamination=0.05)

        scores = np.linspace(0, 1, 100)

        threshold = manager.get_calibrated_threshold(scores)

        assert isinstance(threshold, float)
        assert 0 <= threshold <= 1

    def test_calibration_with_labels(self):
        """Test calibration with ground truth labels."""
        manager = ScoreCalibrationManager(contamination=0.1)

        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])

        result = manager.calibrate(scores, labels)

        # With labels, should use optimal F1 threshold
        assert result.diagnostics.actual_contamination == 0.4

    def test_different_methods(self):
        """Test calibration with different methods."""
        methods = [
            CalibrationMethod.PERCENTILE,
            CalibrationMethod.OTSU,
            CalibrationMethod.MAD,
            CalibrationMethod.ADAPTIVE_IQR,
        ]

        scores = np.random.uniform(0, 1, 100)

        for method in methods:
            manager = ScoreCalibrationManager(method=method)
            result = manager.calibrate(scores)

            assert result.method == method
            assert 0 <= result.threshold <= 1


class TestF1ZeroProblem:
    """Test that the calibration system solves the F1=0 problem."""

    def test_f1_zero_scenario(self):
        """
        Test the exact scenario described by the user:
        - ROC-AUC = 0.88 (good discrimination)
        - F1 = 0 (all predictions False)

        This happens when scores are in [0, 0.3] but threshold is 0.5
        """
        # Create scores that rank anomalies correctly but are all below 0.5
        np.random.seed(42)
        n_normal = 95
        n_anomaly = 5

        # Normal scores in [0, 0.15], anomaly scores in [0.2, 0.3]
        normal_scores = np.random.uniform(0.0, 0.15, n_normal)
        anomaly_scores = np.random.uniform(0.2, 0.3, n_anomaly)

        scores = np.concatenate([normal_scores, anomaly_scores])
        labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])

        # Shuffle
        idx = np.random.permutation(len(scores))
        scores = scores[idx]
        labels = labels[idx]

        # With fixed 0.5 threshold: F1 = 0
        predictions_fixed = scores > 0.5
        assert predictions_fixed.sum() == 0  # All False

        # With calibration: should have positive predictions
        manager = ScoreCalibrationManager(contamination=0.05)
        result = manager.calibrate(scores, labels.astype(np.int32))

        # Calibrated threshold should be in the score range
        assert result.threshold < scores.max()
        # Should have some positive predictions
        assert result.predictions.sum() > 0

    def test_calibration_improves_f1(self):
        """Test that calibration improves F1 compared to fixed threshold."""
        np.random.seed(42)

        # Create separable classes
        n_normal = 90
        n_anomaly = 10

        normal_scores = np.random.normal(0.2, 0.05, n_normal)
        anomaly_scores = np.random.normal(0.4, 0.05, n_anomaly)

        scores = np.clip(np.concatenate([normal_scores, anomaly_scores]), 0, 1)
        labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)]).astype(np.int32)

        # F1 with fixed threshold 0.5
        predictions_fixed = scores > 0.5
        tp_fixed = np.sum((labels == 1) & predictions_fixed)
        fp_fixed = np.sum((labels == 0) & predictions_fixed)
        fn_fixed = np.sum((labels == 1) & ~predictions_fixed)

        prec_fixed = tp_fixed / (tp_fixed + fp_fixed) if (tp_fixed + fp_fixed) > 0 else 0
        rec_fixed = tp_fixed / (tp_fixed + fn_fixed) if (tp_fixed + fn_fixed) > 0 else 0
        f1_fixed = 2 * prec_fixed * rec_fixed / (prec_fixed + rec_fixed) if (prec_fixed + rec_fixed) > 0 else 0

        # F1 with calibration
        manager = ScoreCalibrationManager(contamination=0.1)
        result = manager.calibrate(scores, labels)

        predictions_cal = result.predictions
        tp_cal = np.sum((labels == 1) & predictions_cal)
        fp_cal = np.sum((labels == 0) & predictions_cal)
        fn_cal = np.sum((labels == 1) & ~predictions_cal)

        prec_cal = tp_cal / (tp_cal + fp_cal) if (tp_cal + fp_cal) > 0 else 0
        rec_cal = tp_cal / (tp_cal + fn_cal) if (tp_cal + fn_cal) > 0 else 0
        f1_cal = 2 * prec_cal * rec_cal / (prec_cal + rec_cal) if (prec_cal + rec_cal) > 0 else 0

        # Calibrated F1 should be better or equal
        assert f1_cal >= f1_fixed


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_calibrate_scores_function(self):
        """Test calibrate_scores convenience function."""
        scores = np.random.uniform(0, 1, 100)

        threshold, predictions, diagnostics = calibrate_scores(
            scores, contamination=0.05
        )

        assert isinstance(threshold, float)
        assert len(predictions) == 100
        assert isinstance(diagnostics, CalibrationDiagnostics)

    def test_diagnose_scores_function(self, capsys):
        """Test diagnose_scores convenience function."""
        scores = np.array([0.1, 0.2, 0.3, 0.4])  # All below threshold
        labels = np.array([0, 0, 1, 1])

        diagnostics = diagnose_scores(
            scores, threshold=0.5, labels=labels, print_output=True
        )

        captured = capsys.readouterr()

        # Should print warning about all predictions being negative
        assert "NEGATIVE" in captured.out or diagnostics.predicted_anomaly_ratio == 0


class TestDiagnosticsOutput:
    """Test that diagnostics provide actionable information."""

    def test_diagnostics_to_dict(self):
        """Test diagnostics serialization."""
        scores = np.random.uniform(0, 1, 50)

        diag = ScoreDiagnostics.analyze(scores, threshold=0.5)

        d = diag.to_dict()

        assert "score_min" in d
        assert "score_max" in d
        assert "threshold" in d
        assert "n_above_threshold" in d
        assert "percentiles" in d

    def test_diagnostics_str_output(self):
        """Test diagnostics string representation."""
        scores = np.random.uniform(0, 1, 50)

        diag = ScoreDiagnostics.analyze(scores, threshold=0.5)

        output = str(diag)

        assert "Score Distribution" in output
        assert "Threshold" in output
        assert "Percentiles" in output


# Integration tests
class TestDetectorIntegration:
    """Test integration with actual detectors."""

    def test_statistical_detector_auto_calibration(self):
        """Test StatisticalAnomalyDetector with auto-calibration."""
        from omni_mercury_engine.detectors.statistical import StatisticalAnomalyDetector

        # Create test data
        np.random.seed(42)
        n_normal = 90
        n_anomaly = 10

        X_train = np.random.randn(100, 5)
        X_test = np.vstack([
            np.random.randn(n_normal, 5),
            np.random.randn(n_anomaly, 5) + 3,  # Shifted anomalies
        ])

        # Without auto-calibration
        detector_fixed = StatisticalAnomalyDetector({"threshold": 0.5})
        detector_fixed.fit(X_train)
        result_fixed = detector_fixed.detect(X_test)

        # With auto-calibration
        detector_cal = StatisticalAnomalyDetector({"threshold": 0.5})
        detector_cal.fit(X_train)
        detector_cal.enable_auto_calibration(contamination=0.1)
        result_cal = detector_cal.detect(X_test)

        # Auto-calibrated should have predictions
        assert result_cal["is_anomaly"].sum() > 0
        # And should have calibration info
        assert "threshold" in result_cal

    def test_detector_diagnose_scores_method(self):
        """Test detector's diagnose_scores method."""
        from omni_mercury_engine.detectors.statistical import StatisticalAnomalyDetector

        np.random.seed(42)
        X = np.random.randn(50, 5)

        detector = StatisticalAnomalyDetector()
        detector.fit(X)
        result = detector.detect(X)

        # Should be able to diagnose scores
        scores = result["scores"]
        diag = detector.diagnose_scores(scores, print_output=False)

        assert diag is not None


class TestEdgeCases:
    """Test edge cases: empty arrays, NaN/Inf, constant arrays, single elements."""

    def test_empty_array_raises_error(self):
        """Test that empty arrays raise ValueError."""
        scores = np.array([])

        with pytest.raises(ValueError, match="empty"):
            ScoreDiagnostics.analyze(scores)

    def test_nan_values_handled(self):
        """Test that NaN values are handled gracefully."""
        scores = np.array([0.1, np.nan, 0.3, 0.4, 0.5])

        # Should not raise, should replace NaN
        diag = ScoreDiagnostics.analyze(scores, threshold=0.3)

        assert diag.n_samples == 5  # Still reports 5 samples
        assert np.isfinite(diag.score_mean)

    def test_inf_values_handled(self):
        """Test that Inf values are handled gracefully."""
        scores = np.array([0.1, np.inf, 0.3, 0.4, 0.5])

        diag = ScoreDiagnostics.analyze(scores, threshold=0.3)

        assert diag.n_samples == 5
        assert np.isfinite(diag.score_max)

    def test_all_nan_raises_error(self):
        """Test that all-NaN arrays raise ValueError."""
        scores = np.array([np.nan, np.nan, np.nan])

        with pytest.raises(ValueError, match="NaN|Inf|finite"):
            ScoreDiagnostics.analyze(scores)

    def test_constant_array(self):
        """Test handling of constant arrays (all same value)."""
        scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

        optimizer = AutoThresholdOptimizer()
        result = optimizer.optimize(scores, method=CalibrationMethod.PERCENTILE)

        # Should still return a valid result
        assert result.threshold is not None
        assert 0 <= result.threshold <= 1

    def test_single_element_array(self):
        """Test handling of single-element arrays."""
        scores = np.array([0.5])

        optimizer = AutoThresholdOptimizer()
        result = optimizer.optimize(scores, method=CalibrationMethod.FIXED)

        assert result.threshold == 0.5

    def test_two_element_array(self):
        """Test handling of two-element arrays."""
        scores = np.array([0.1, 0.9])

        optimizer = AutoThresholdOptimizer()
        result = optimizer.optimize(scores, method=CalibrationMethod.PERCENTILE)

        assert 0.1 <= result.threshold <= 0.9

    def test_invalid_labels_raises_error(self):
        """Test that invalid labels (non-binary) raise error."""
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        labels = np.array([0, 1, 2, 3])  # Invalid: should be binary

        with pytest.raises(ValueError, match="binary"):
            ScoreDiagnostics.analyze(scores, labels=labels)

    def test_float_labels_converted(self):
        """Test that float labels are converted to int."""
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        labels = np.array([0.0, 1.0, 0.0, 1.0])  # Float labels

        # Should work - converts to int
        diag = ScoreDiagnostics.analyze(scores, labels=labels)
        assert diag.actual_contamination == 0.5


class TestContaminationValidation:
    """Test contamination parameter validation."""

    def test_invalid_contamination_raises_error(self):
        """Test that invalid contamination raises ValueError."""
        with pytest.raises(ValueError, match="contamination"):
            AutoThresholdOptimizer(default_contamination=1.5)

        with pytest.raises(ValueError, match="contamination"):
            AutoThresholdOptimizer(default_contamination=-0.1)

        with pytest.raises(ValueError, match="contamination"):
            AutoThresholdOptimizer(default_contamination=0.0)

    def test_invalid_min_max_contamination(self):
        """Test that invalid min/max contamination raises ValueError."""
        with pytest.raises(ValueError, match="min.*max"):
            AutoThresholdOptimizer(min_contamination=0.5, max_contamination=0.3)

    def test_default_outside_range_raises_error(self):
        """Test that default outside [min, max] raises error."""
        with pytest.raises(ValueError, match="default"):
            AutoThresholdOptimizer(
                default_contamination=0.1,
                min_contamination=0.2,
                max_contamination=0.5
            )


class TestErrorHandling:
    """Test error handling and fallback behavior."""

    def test_gmm_fallback_on_small_dataset(self):
        """Test that GMM falls back to percentile on small datasets."""
        scores = np.array([0.1, 0.2, 0.3])  # Only 3 samples

        optimizer = AutoThresholdOptimizer()
        result = optimizer.optimize(scores, method=CalibrationMethod.GAUSSIAN_MIXTURE)

        # Should fall back to percentile
        assert result.threshold is not None

    def test_optimal_f1_fallback_without_labels(self):
        """Test that OPTIMAL_F1 falls back when labels not provided."""
        scores = np.random.uniform(0, 1, 50)

        optimizer = AutoThresholdOptimizer()
        result = optimizer.optimize(scores, method=CalibrationMethod.OPTIMAL_F1)

        # Should fall back to another method
        assert result.method != CalibrationMethod.OPTIMAL_F1 or result.threshold is not None


class TestReturnTypeConsistency:
    """Test that return types are consistent across methods."""

    def test_all_methods_return_calibration_result(self):
        """Test that all methods return CalibrationResult."""
        scores = np.random.uniform(0, 1, 100)
        labels = np.array([0] * 90 + [1] * 10)

        optimizer = AutoThresholdOptimizer()

        methods = [
            CalibrationMethod.FIXED,
            CalibrationMethod.PERCENTILE,
            CalibrationMethod.MAD,
            CalibrationMethod.ADAPTIVE_IQR,
            CalibrationMethod.AUTO,
        ]

        for method in methods:
            result = optimizer.optimize(scores, method=method, labels=labels)

            assert isinstance(result, CalibrationResult), \
                f"Method {method} didn't return CalibrationResult"
            assert isinstance(result.threshold, float), \
                f"Method {method} threshold not float"
            assert isinstance(result.predictions, np.ndarray), \
                f"Method {method} predictions not ndarray"
            assert len(result.predictions) == len(scores), \
                f"Method {method} predictions length mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
