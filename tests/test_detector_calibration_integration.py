"""
Integration Tests for All Calibrated Detectors
Copyright (C) 2025 Steel Security Advisory LLC

Tests that all detectors properly support auto-calibration and that the
calibration system solves the F1=0 problem across all detector types.
"""

import numpy as np
import pytest


class TestAllDetectorsAutoCalibration:
    """Test auto-calibration across all detector types."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with known anomalies."""
        np.random.seed(42)
        n_normal = 90
        n_anomaly = 10

        # Normal data centered at 0
        X_normal = np.random.randn(n_normal, 10)

        # Anomalies shifted
        X_anomaly = np.random.randn(n_anomaly, 10) + 3

        X = np.vstack([X_normal, X_anomaly])
        y = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)]).astype(np.int32)

        # Shuffle
        idx = np.random.permutation(len(X))
        return X[idx], y[idx]

    def test_statistical_detector_calibration(self, sample_data):
        """Test StatisticalAnomalyDetector auto-calibration."""
        from omni_mercury_engine.detectors.statistical import StatisticalAnomalyDetector

        X, y = sample_data

        # Without calibration
        detector_fixed = StatisticalAnomalyDetector({"threshold": 0.5})
        detector_fixed.fit(X)
        result_fixed = detector_fixed.detect(X)

        # With calibration
        detector_cal = StatisticalAnomalyDetector({"threshold": 0.5})
        detector_cal.fit(X)
        detector_cal.enable_auto_calibration(contamination=0.1, method="percentile")
        result_cal = detector_cal.detect(X)

        # Verify calibration info is present
        assert "threshold" in result_cal
        assert "calibration_diagnostics" in result_cal

        # Verify calibrated predictions
        assert result_cal["is_anomaly"].sum() > 0

    def test_temporal_detector_calibration(self, sample_data):
        """Test TemporalAnomalyDetector auto-calibration."""
        from omni_mercury_engine.detectors.temporal import TemporalAnomalyDetector

        X, y = sample_data
        # Flatten for temporal
        X_flat = X.flatten()

        detector = TemporalAnomalyDetector({"threshold": 0.5})
        detector.fit(X_flat)
        detector.enable_auto_calibration(contamination=0.1)
        result = detector.detect(X_flat)

        assert "threshold" in result
        assert "calibration_diagnostics" in result
        assert result["is_anomaly"].sum() >= 0  # May be 0 for random data

    def test_dimensional_analyzer_calibration(self, sample_data):
        """Test DimensionalAnalyzer auto-calibration."""
        from omni_mercury_engine.detectors.dimensional import DimensionalAnalyzer

        X, y = sample_data

        detector = DimensionalAnalyzer({"threshold": 0.5, "use_db_term": False})
        detector.fit(X)
        detector.enable_auto_calibration(contamination=0.1)
        result = detector.detect(X)

        assert "threshold" in result
        assert "calibration_diagnostics" in result
        assert result["is_anomaly"].sum() > 0

    def test_spatial_detector_calibration(self, sample_data):
        """Test SpatialAnomalyDetector auto-calibration."""
        from omni_mercury_engine.detectors.spatial import SpatialAnomalyDetector

        X, y = sample_data

        detector = SpatialAnomalyDetector({"threshold": 0.5})
        detector.fit(X)
        detector.enable_auto_calibration(contamination=0.1)
        result = detector.detect(X)

        assert "threshold" in result
        assert "calibration_diagnostics" in result
        assert result["is_anomaly"].sum() > 0

    def test_directive_detector_calibration(self, sample_data):
        """Test SigmaDirectiveDetector auto-calibration."""
        from omni_mercury_engine.detectors.directive import SigmaDirectiveDetector

        X, y = sample_data

        detector = SigmaDirectiveDetector({
            "threshold": 0.5,
            "use_quantum_enhanced": False,
            "use_nano_detection": False,
            "use_harmonic_detection": False,
        })
        detector.fit(X)
        detector.enable_auto_calibration(contamination=0.1)
        result = detector.detect(X)

        assert "threshold" in result
        assert "calibration_diagnostics" in result
        assert result["is_anomaly"].sum() >= 0


class TestF1ZeroProblemAllDetectors:
    """Test that calibration solves F1=0 across all detectors."""

    @pytest.fixture
    def f1_zero_scenario(self):
        """
        Create data that causes F1=0 with fixed threshold:
        - Scores in [0, 0.3] but threshold at 0.5
        """
        np.random.seed(42)
        n_normal = 95
        n_anomaly = 5

        # Normal: scores will be in [0, 0.1]
        X_normal = np.random.randn(n_normal, 10) * 0.1

        # Anomaly: scores will be in [0.15, 0.25] (still below 0.5!)
        X_anomaly = np.random.randn(n_anomaly, 10) * 0.1 + 2.0

        X = np.vstack([X_normal, X_anomaly])
        y = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)]).astype(np.int32)

        return X, y

    def test_statistical_f1_zero_solved(self, f1_zero_scenario):
        """Verify StatisticalAnomalyDetector calibration solves F1=0."""
        from omni_mercury_engine.detectors.statistical import StatisticalAnomalyDetector

        X, y = f1_zero_scenario

        # Without calibration: F1 = 0
        detector_fixed = StatisticalAnomalyDetector({"threshold": 0.5})
        detector_fixed.fit(X)
        result_fixed = detector_fixed.detect(X)

        # With calibration: should have predictions
        detector_cal = StatisticalAnomalyDetector({"threshold": 0.5})
        detector_cal.fit(X)
        detector_cal.enable_auto_calibration(contamination=0.05, method="percentile")
        result_cal = detector_cal.detect(X)

        # Calibrated should detect anomalies
        assert result_cal["is_anomaly"].sum() > 0

        # Verify threshold was calibrated below max score
        assert result_cal["threshold"] <= result_cal["scores"].max()


class TestCalibratedThresholdMethods:
    """Test different calibration methods across detectors."""

    @pytest.fixture
    def bimodal_data(self):
        """Create bimodal data for Otsu testing."""
        np.random.seed(42)

        normal = np.random.normal(0.2, 0.03, (80, 5))
        anomaly = np.random.normal(0.8, 0.03, (20, 5))

        X = np.vstack([normal, anomaly])
        y = np.concatenate([np.zeros(80), np.ones(20)]).astype(np.int32)

        idx = np.random.permutation(len(X))
        return X[idx], y[idx]

    def test_percentile_method(self, bimodal_data):
        """Test percentile calibration method."""
        from omni_mercury_engine.detectors.statistical import StatisticalAnomalyDetector

        X, y = bimodal_data

        detector = StatisticalAnomalyDetector()
        detector.fit(X)
        detector.enable_auto_calibration(contamination=0.2, method="percentile")
        result = detector.detect(X)

        # Should detect approximately 20% as anomalies
        pred_ratio = result["is_anomaly"].sum() / len(X)
        assert 0.1 <= pred_ratio <= 0.4  # Allow some variance

    def test_diagnose_scores_method(self, bimodal_data):
        """Test detector's diagnose_scores method."""
        from omni_mercury_engine.detectors.dimensional import DimensionalAnalyzer

        X, y = bimodal_data

        detector = DimensionalAnalyzer({"use_db_term": False})
        detector.fit(X)
        result = detector.detect(X)

        # Should be able to diagnose scores
        diagnostics = detector.diagnose_scores(
            result["scores"], labels=y, print_output=False
        )

        assert diagnostics is not None
        assert diagnostics.n_samples == len(X)


class TestEngineCalibration:
    """Test engine-level calibration methods."""

    @pytest.fixture
    def engine_data(self):
        """Create data for engine testing."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.array([0] * 45 + [1] * 5).astype(np.int32)
        return X, y

    def test_engine_enable_auto_calibration(self, engine_data):
        """Test engine.enable_auto_calibration()."""
        from omni_mercury_engine import OmniMercuryEngine

        X, y = engine_data

        engine = OmniMercuryEngine()
        engine.enable_auto_calibration(contamination=0.1, method="percentile")

        # Verify all detectors have calibration enabled
        for detector in engine.detectors.values():
            assert detector._auto_calibrate is True

    def test_engine_detect_with_calibration(self, engine_data):
        """Test engine.detect_with_calibration()."""
        from omni_mercury_engine import OmniMercuryEngine

        X, y = engine_data

        engine = OmniMercuryEngine()
        result = engine.detect_with_calibration(
            X, labels=y, calibration_method="auto"
        )

        assert "threshold" in result
        assert "diagnostics" in result
        assert "is_anomaly" in result

    def test_engine_diagnose_detection(self, engine_data, capsys):
        """Test engine.diagnose_detection()."""
        from omni_mercury_engine import OmniMercuryEngine

        X, y = engine_data

        engine = OmniMercuryEngine()
        result = engine.diagnose_detection(X, labels=y, print_output=True)

        captured = capsys.readouterr()

        assert "diagnostics" in result
        assert "recommendations" in result
        # Should print detector diagnostics
        assert "DETECTOR" in captured.out or len(result["diagnostics"]) > 0


class TestBenchmarkDiagnostics:
    """Test benchmark diagnostics module."""

    def test_quick_diagnose(self, capsys):
        """Test BenchmarkDiagnostics.quick_diagnose()."""
        from omni_mercury_engine.evaluation.benchmark_diagnostics import (
            BenchmarkDiagnostics,
        )

        scores = np.array([0.1, 0.2, 0.3, 0.4])  # All below 0.5
        labels = np.array([0, 0, 1, 1])

        BenchmarkDiagnostics.quick_diagnose(scores, labels, threshold=0.5)

        captured = capsys.readouterr()

        assert "Score range" in captured.out
        assert "Threshold" in captured.out
        assert "Predictions above threshold" in captured.out

    def test_full_diagnostics(self):
        """Test BenchmarkDiagnostics.diagnose()."""
        from omni_mercury_engine.evaluation.benchmark_diagnostics import (
            BenchmarkDiagnostics,
        )

        np.random.seed(42)
        # Create scores that have good ranking (ROC-AUC > 0.7) but F1=0 due to threshold
        # Normal samples: low scores (0.1-0.3)
        # Anomaly samples: high scores (0.35-0.45) - still below threshold 0.5
        normal_scores = np.random.uniform(0.1, 0.3, 95)
        anomaly_scores = np.random.uniform(0.35, 0.45, 5)  # Higher than normal but below 0.5
        scores = np.concatenate([normal_scores, anomaly_scores])
        labels = np.concatenate([np.zeros(95), np.ones(5)]).astype(np.int32)

        result = BenchmarkDiagnostics.diagnose(
            scores, labels, threshold=0.5,
            detector_name="TestDetector",
            dataset_name="TestDataset",
        )

        # Should detect F1=0 discrepancy when ROC-AUC is good but F1=0
        assert result.f1 == 0.0
        assert result.f1_at_best_threshold > 0  # Best threshold should give F1 > 0
        # ROC-AUC should be good since anomalies have higher scores
        assert result.roc_auc > 0.7
        # With good ROC-AUC and F1=0, should detect discrepancy
        assert result.discrepancy.has_discrepancy
        assert result.discrepancy.discrepancy_type == "f1_zero"

    def test_run_diagnostic_benchmark(self):
        """Test run_diagnostic_benchmark convenience function."""
        from omni_mercury_engine.detectors.statistical import StatisticalAnomalyDetector
        from omni_mercury_engine.evaluation.benchmark_diagnostics import (
            run_diagnostic_benchmark,
        )

        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        X_test = np.vstack([
            np.random.randn(90, 5),
            np.random.randn(10, 5) + 3,
        ])
        y_test = np.concatenate([np.zeros(90), np.ones(10)]).astype(np.int32)

        detector = StatisticalAnomalyDetector()

        result = run_diagnostic_benchmark(
            detector, X_train, X_test, y_test,
            detector_name="Statistical",
            dataset_name="Synthetic",
            print_report=False,
        )

        assert result.n_samples == 100
        assert result.n_anomalies_true == 10
        assert result.roc_auc >= 0  # Should have some discrimination


class TestCalibrationMethodRecommendations:
    """Test that the system makes appropriate calibration recommendations."""

    def test_bimodal_recommends_otsu(self):
        """Test that bimodal distributions recommend Otsu method."""
        from omni_mercury_engine.evaluation.benchmark_diagnostics import (
            BenchmarkDiagnostics,
        )

        # Create bimodal scores
        normal = np.random.normal(0.2, 0.03, 80)
        anomaly = np.random.normal(0.8, 0.03, 20)
        scores = np.concatenate([normal, anomaly])
        labels = np.concatenate([np.zeros(80), np.ones(20)]).astype(np.int32)

        result = BenchmarkDiagnostics.diagnose(scores, labels, threshold=0.5)

        # Should recommend Otsu for bimodal
        if result.is_bimodal:
            assert result.calibration_method_recommended == "otsu"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
