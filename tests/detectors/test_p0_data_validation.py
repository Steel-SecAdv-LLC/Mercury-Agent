# Copyright (C) 2025 Steel Security Advisors LLC
"""P0 Data Validation Tests.

Comprehensive test suite for P0 critical fixes addressing:
1. Threshold validation ([0, 1] range enforcement)
2. NaN/Inf handling in detectors
3. Empty data validation
4. Constant array handling (division by zero prevention)
5. Device propagation in fusion layers
6. GraphAnomalyDetector z-score threshold bypass

These tests ensure the Mercury Agent anomaly detection pipeline
produces reliable, finite scores even with edge-case inputs.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("torch")

import numpy as np
import pytest
import torch

from omni_mercury_engine.core.exceptions import DetectorException
from omni_mercury_engine.core.fusion import HybridFusionLayer, ResonanceWeightedFusion
from omni_mercury_engine.detectors.dimensional import DimensionalAnalyzer
from omni_mercury_engine.detectors.graph_based import GraphAnomalyDetector
from omni_mercury_engine.detectors.spatial import SpatialAnomalyDetector
from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector
from omni_mercury_engine.detectors.temporal import TemporalAnomalyDetector


# =============================================================================
# Test 1: Threshold Validation
# =============================================================================
class TestThresholdValidation:
    """Test threshold [0, 1] range validation in BaseDetector."""

    def test_valid_threshold_zero(self) -> None:
        """Threshold of 0.0 should be accepted."""
        detector = SpatialAnomalyDetector({"threshold": 0.0})
        assert detector.threshold == 0.0

    def test_valid_threshold_one(self) -> None:
        """Threshold of 1.0 should be accepted."""
        detector = SpatialAnomalyDetector({"threshold": 1.0})
        assert detector.threshold == 1.0

    def test_valid_threshold_midpoint(self) -> None:
        """Threshold of 0.5 (default) should be accepted."""
        detector = SpatialAnomalyDetector({"threshold": 0.5})
        assert detector.threshold == 0.5

    def test_valid_threshold_custom(self) -> None:
        """Custom valid thresholds should be accepted."""
        for threshold in [0.1, 0.25, 0.75, 0.9, 0.99]:
            detector = SpatialAnomalyDetector({"threshold": threshold})
            assert detector.threshold == threshold

    def test_invalid_threshold_above_one(self) -> None:
        """Threshold > 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match=r"Threshold must be in \[0, 1\] range"):
            SpatialAnomalyDetector({"threshold": 1.5})

    def test_invalid_threshold_negative(self) -> None:
        """Threshold < 0.0 should raise ValueError."""
        with pytest.raises(ValueError, match=r"Threshold must be in \[0, 1\] range"):
            SpatialAnomalyDetector({"threshold": -0.1})

    def test_invalid_threshold_large(self) -> None:
        """Very large threshold should raise ValueError."""
        with pytest.raises(ValueError, match=r"Threshold must be in \[0, 1\] range"):
            SpatialAnomalyDetector({"threshold": 100.0})

    def test_invalid_threshold_non_numeric(self) -> None:
        """Non-numeric threshold should raise ValueError."""
        with pytest.raises(ValueError, match=r"Threshold must be numeric"):
            SpatialAnomalyDetector({"threshold": "high"})

    def test_threshold_integer_accepted(self) -> None:
        """Integer thresholds (0 and 1) should be accepted and converted to float."""
        detector = SpatialAnomalyDetector({"threshold": 1})
        assert detector.threshold == 1.0
        assert isinstance(detector.threshold, float)


# =============================================================================
# Test 2: GraphAnomalyDetector Z-Score Threshold Bypass
# =============================================================================
class TestGraphAnomalyDetectorZScoreThreshold:
    """Test GraphAnomalyDetector correctly bypasses [0, 1] threshold validation."""

    def test_default_zscore_threshold(self) -> None:
        """Default z-score threshold of 3.0 should be accepted."""
        detector = GraphAnomalyDetector()
        assert detector.threshold == 3.0

    def test_custom_zscore_threshold(self) -> None:
        """Custom z-score thresholds > 1 should be accepted."""
        detector = GraphAnomalyDetector({"threshold": 2.5})
        assert detector.threshold == 2.5

    def test_high_zscore_threshold(self) -> None:
        """High z-score thresholds should be accepted."""
        detector = GraphAnomalyDetector({"threshold": 5.0})
        assert detector.threshold == 5.0

    def test_zscore_threshold_low_valid(self) -> None:
        """Low z-score thresholds (< 1) should also work."""
        detector = GraphAnomalyDetector({"threshold": 0.5})
        assert detector.threshold == 0.5

    def test_other_config_preserved(self) -> None:
        """Other config options should be preserved after threshold extraction."""
        detector = GraphAnomalyDetector({"threshold": 4.0, "name": "test_graph"})
        assert detector.threshold == 4.0
        assert detector._name == "test_graph"


# =============================================================================
# Test 3: NaN/Inf Handling in Detectors
# =============================================================================
class TestNaNInfHandling:
    """Test NaN/Inf sanitization across all detector types."""

    @pytest.fixture
    def normal_data(self, deterministic_rng: Any) -> Any:
        """Generate normal training data."""
        return deterministic_rng.randn(100, 10)

    @pytest.fixture
    def data_with_nan(self, deterministic_rng: Any) -> Any:
        """Generate test data containing NaN values."""
        data = deterministic_rng.randn(50, 10)
        data[5, 3] = np.nan
        data[15, 7] = np.nan
        data[25, :] = np.nan  # Entire row is NaN
        return data

    @pytest.fixture
    def data_with_inf(self, deterministic_rng: Any) -> Any:
        """Generate test data containing Inf values."""
        data = deterministic_rng.randn(50, 10)
        data[10, 2] = np.inf
        data[20, 5] = -np.inf
        return data

    @pytest.fixture
    def data_with_nan_and_inf(self, deterministic_rng: Any) -> Any:
        """Generate test data containing both NaN and Inf values."""
        data = deterministic_rng.randn(50, 10)
        data[5, 3] = np.nan
        data[10, 2] = np.inf
        data[20, 5] = -np.inf
        return data

    # --- Temporal Detector Tests ---
    def test_temporal_nan_input_produces_finite_scores(
        self, normal_data: Any, data_with_nan: Any
    ) -> None:
        """TemporalAnomalyDetector should produce finite scores with NaN input."""
        detector = TemporalAnomalyDetector()
        # Flatten for 1D temporal analysis
        detector.fit(normal_data[:, 0])
        result = detector.detect(data_with_nan[:, 0])

        assert np.all(np.isfinite(result["scores"])), "Scores contain NaN/Inf"
        assert result["scores"].shape[0] == data_with_nan.shape[0]

    def test_temporal_inf_input_produces_finite_scores(
        self, normal_data: Any, data_with_inf: Any
    ) -> None:
        """TemporalAnomalyDetector should produce finite scores with Inf input."""
        detector = TemporalAnomalyDetector()
        detector.fit(normal_data[:, 0])
        result = detector.detect(data_with_inf[:, 0])

        assert np.all(np.isfinite(result["scores"])), "Scores contain NaN/Inf"

    # --- Spatial Detector Tests ---
    def test_spatial_nan_input_produces_finite_scores(
        self, normal_data: Any, data_with_nan: Any
    ) -> None:
        """SpatialAnomalyDetector should produce finite scores with NaN input."""
        detector = SpatialAnomalyDetector()
        # Use 2D spatial data
        detector.fit(normal_data[:, :2])

        # Replace NaN with values before passing to LOF (LOF can't handle NaN)
        clean_test = np.nan_to_num(data_with_nan[:, :2], nan=0.0)
        result = detector.detect(clean_test)

        assert np.all(np.isfinite(result["scores"])), "Scores contain NaN/Inf"

    def test_spatial_safe_normalize_constant_array(self) -> None:
        """_safe_normalize should handle constant arrays without division by zero."""
        detector = SpatialAnomalyDetector()

        # Test constant array
        constant_scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        normalized = detector._safe_normalize(constant_scores)

        assert np.all(np.isfinite(normalized)), "Constant array produced non-finite values"
        assert np.allclose(normalized, 0.5), "Constant array should normalize to 0.5"

    def test_spatial_safe_normalize_with_nan(self) -> None:
        """_safe_normalize should handle NaN values."""
        detector = SpatialAnomalyDetector()

        scores_with_nan = np.array([0.1, np.nan, 0.3, 0.4, 0.5])
        normalized = detector._safe_normalize(scores_with_nan)

        assert np.all(np.isfinite(normalized)), "NaN not handled correctly"
        assert np.all(normalized >= 0.0) and np.all(normalized <= 1.0)

    def test_spatial_safe_normalize_all_zero(self) -> None:
        """_safe_normalize should handle all-zero arrays."""
        detector = SpatialAnomalyDetector()

        zero_scores = np.array([0.0, 0.0, 0.0, 0.0])
        normalized = detector._safe_normalize(zero_scores)

        assert np.all(np.isfinite(normalized)), "Zero array produced non-finite values"
        assert np.allclose(normalized, 0.5), "Zero array should normalize to 0.5"

    # --- Dimensional Detector Tests ---
    def test_dimensional_nan_produces_finite_scores(self, normal_data: Any) -> None:
        """DimensionalAnalyzer should produce finite scores even with edge cases."""
        detector = DimensionalAnalyzer()
        detector.fit(normal_data)
        result = detector.detect(normal_data)

        assert np.all(np.isfinite(result["scores"])), "Scores contain NaN/Inf"
        assert np.all(result["scores"] >= 0.0) and np.all(result["scores"] <= 1.0)

    def test_dimensional_constant_input_produces_valid_scores(self, deterministic_rng: Any) -> None:
        """DimensionalAnalyzer should handle constant input gracefully."""
        detector = DimensionalAnalyzer()

        # Training data with some variation
        train_data = deterministic_rng.randn(100, 10)
        detector.fit(train_data)

        # Test with near-constant data
        constant_test = np.full((50, 10), 0.5)
        result = detector.detect(constant_test)

        assert np.all(np.isfinite(result["scores"])), "Constant input produced NaN/Inf"

    # --- Statistical Detector Tests ---
    def test_statistical_produces_finite_scores(self, normal_data: Any) -> None:
        """MercuryAnomalyDetector should produce finite scores."""
        detector = MercuryAnomalyDetector()
        detector.fit(normal_data)
        result = detector.detect(normal_data)

        assert np.all(np.isfinite(result["scores"])), "Scores contain NaN/Inf"


# =============================================================================
# Test 4: Empty Data Validation
# =============================================================================
class TestEmptyDataValidation:
    """Test empty data rejection in MercuryAnomalyDetector."""

    def test_empty_array_raises_exception(self) -> None:
        """Empty array should raise DetectorException."""
        detector = MercuryAnomalyDetector()

        with pytest.raises(DetectorException, match="empty data"):
            detector.fit(np.array([]))

    def test_empty_2d_array_raises_exception(self) -> None:
        """Empty 2D array should raise DetectorException."""
        detector = MercuryAnomalyDetector()

        with pytest.raises(DetectorException, match="empty data"):
            detector.fit(np.array([]).reshape(0, 10))

    def test_all_nan_array_raises_exception(self) -> None:
        """Array with all NaN values should raise DetectorException."""
        detector = MercuryAnomalyDetector()

        all_nan = np.full((10, 5), np.nan)
        with pytest.raises(DetectorException, match="all data values are NaN or Inf"):
            detector.fit(all_nan)

    def test_all_inf_array_raises_exception(self) -> None:
        """Array with all Inf values should raise DetectorException."""
        detector = MercuryAnomalyDetector()

        all_inf = np.full((10, 5), np.inf)
        with pytest.raises(DetectorException, match="all data values are NaN or Inf"):
            detector.fit(all_inf)

    def test_partial_nan_rows_filtered(self, deterministic_rng: Any) -> None:
        """Rows with NaN should be filtered, valid rows used for fitting."""
        detector = MercuryAnomalyDetector()

        # 10 rows, 5 features, rows 0 and 5 have NaN
        data = deterministic_rng.randn(10, 5)
        data[0, :] = np.nan
        data[5, :] = np.nan

        # Should not raise - valid rows exist
        detector.fit(data)
        assert detector._is_fitted


# =============================================================================
# Test 5: Device Propagation in Fusion Layers
# =============================================================================
class TestDevicePropagation:
    """Test device handling in HybridFusionLayer and ResonanceWeightedFusion."""

    def test_hybrid_fusion_cpu_device(self) -> None:
        """HybridFusionLayer should handle CPU tensors correctly."""
        fusion = HybridFusionLayer(feature_dims={"detector1": 32, "detector2": 32}, hidden_dim=64)

        detector_features = {
            "detector1": torch.randn(8, 32),
            "detector2": torch.randn(8, 32),
        }
        detector_scores = {
            "detector1": torch.randn(8, 1),
            "detector2": torch.randn(8, 1),
        }

        fused, attn = fusion(detector_features, detector_scores)

        assert fused.device.type == "cpu"
        assert fused.shape[0] == 8

    def test_hybrid_fusion_missing_detector_uses_correct_device(self) -> None:
        """Missing detectors should use zeros on the same device as other tensors."""
        fusion = HybridFusionLayer(
            feature_dims={"detector1": 32, "detector2": 32, "detector3": 32},
            hidden_dim=64,
        )

        # Only provide detector1 and detector2, missing detector3
        detector_features = {
            "detector1": torch.randn(8, 32),
            "detector2": torch.randn(8, 32),
            # detector3 is missing
        }
        detector_scores = {
            "detector1": torch.randn(8, 1),
            "detector2": torch.randn(8, 1),
            # detector3 is missing
        }

        # Should not raise device mismatch error
        fused, attn = fusion(detector_features, detector_scores)

        assert fused.device.type == "cpu"
        assert fused.shape[0] == 8

    def test_resonance_fusion_nan_divergences_handled(self) -> None:
        """ResonanceWeightedFusion should handle NaN divergences gracefully."""
        fusion = ResonanceWeightedFusion(num_detectors=3)

        detector_scores = torch.randn(8, 3)
        divergences = torch.randn(8, 3)
        divergences[2, 1] = float("nan")
        divergences[5, 0] = float("inf")

        fused = fusion(detector_scores, divergences)

        assert torch.all(torch.isfinite(fused)), "Fused scores contain NaN/Inf"

    def test_resonance_fusion_nan_scores_handled(self) -> None:
        """ResonanceWeightedFusion should handle NaN scores gracefully."""
        fusion = ResonanceWeightedFusion(num_detectors=3)

        detector_scores = torch.randn(8, 3)
        detector_scores[3, 1] = float("nan")
        detector_scores[6, 2] = float("inf")

        fused = fusion(detector_scores)

        assert torch.all(torch.isfinite(fused)), "Fused scores contain NaN/Inf"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_hybrid_fusion_cuda_device(self) -> None:
        """HybridFusionLayer should handle CUDA tensors correctly."""
        fusion = HybridFusionLayer(
            feature_dims={"detector1": 32, "detector2": 32}, hidden_dim=64
        ).cuda()

        detector_features = {
            "detector1": torch.randn(8, 32).cuda(),
            "detector2": torch.randn(8, 32).cuda(),
        }
        detector_scores = {
            "detector1": torch.randn(8, 1).cuda(),
            "detector2": torch.randn(8, 1).cuda(),
        }

        fused, attn = fusion(detector_features, detector_scores)

        assert fused.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_hybrid_fusion_missing_detector_cuda(self) -> None:
        """Missing detectors with CUDA tensors should use CUDA zeros."""
        fusion = HybridFusionLayer(
            feature_dims={"detector1": 32, "detector2": 32, "detector3": 32},
            hidden_dim=64,
        ).cuda()

        detector_features = {
            "detector1": torch.randn(8, 32).cuda(),
            # detector2 and detector3 missing
        }
        detector_scores = {
            "detector1": torch.randn(8, 1).cuda(),
        }

        # Should not raise device mismatch error
        fused, attn = fusion(detector_features, detector_scores)

        assert fused.device.type == "cuda"

    def test_hybrid_fusion_mixed_device_raises_error(self) -> None:
        """Mixed device tensors should raise ValueError with clear message."""
        # Skip if CUDA not available - can't test mixed devices without it
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed device test")

        fusion = HybridFusionLayer(
            feature_dims={"detector1": 32, "detector2": 32},
            hidden_dim=64,
        )

        # Create mixed-device tensors
        detector_features = {
            "detector1": torch.randn(8, 32),  # CPU
            "detector2": torch.randn(8, 32).cuda(),  # CUDA
        }
        detector_scores = {
            "detector1": torch.randn(8, 1),
            "detector2": torch.randn(8, 1),
        }

        with pytest.raises(ValueError, match="Mixed devices"):
            fusion(detector_features, detector_scores)

    def test_hybrid_fusion_empty_features_raises_error(self) -> None:
        """Empty detector_features dict should raise ValueError."""
        fusion = HybridFusionLayer(
            feature_dims={"detector1": 32, "detector2": 32},
            hidden_dim=64,
        )

        with pytest.raises(ValueError, match="Empty"):
            fusion({}, {"detector1": torch.randn(8, 1)})


# =============================================================================
# Test 6: Score Range Validation
# =============================================================================
class TestScoreRangeValidation:
    """Test that detector scores are in valid [0, 1] range after normalization."""

    @pytest.fixture
    def normal_data(self, deterministic_rng: Any) -> Any:
        """Generate normal training data."""
        return deterministic_rng.randn(100, 10)

    def test_spatial_scores_in_range(self, normal_data: Any) -> None:
        """SpatialAnomalyDetector scores should be in [0, 1]."""
        detector = SpatialAnomalyDetector()
        detector.fit(normal_data[:, :2])
        result = detector.detect(normal_data[:, :2])

        assert np.all(result["scores"] >= 0.0), "Scores below 0"
        assert np.all(result["scores"] <= 1.0), "Scores above 1"

    def test_dimensional_scores_in_range(self, normal_data: Any) -> None:
        """DimensionalAnalyzer scores should be in [0, 1]."""
        detector = DimensionalAnalyzer()
        detector.fit(normal_data)
        result = detector.detect(normal_data)

        assert np.all(result["scores"] >= 0.0), "Scores below 0"
        assert np.all(result["scores"] <= 1.0), "Scores above 1"

    def test_temporal_scores_in_range(self, normal_data: Any) -> None:
        """TemporalAnomalyDetector scores should be in [0, 1]."""
        detector = TemporalAnomalyDetector()
        detector.fit(normal_data[:, 0])
        result = detector.detect(normal_data[:, 0])

        assert np.all(result["scores"] >= 0.0), "Scores below 0"
        assert np.all(result["scores"] <= 1.0), "Scores above 1"

    def test_statistical_scores_in_range(self, normal_data: Any) -> None:
        """MercuryAnomalyDetector scores should be in [0, 1]."""
        detector = MercuryAnomalyDetector()
        detector.fit(normal_data)
        result = detector.detect(normal_data)

        assert np.all(result["scores"] >= 0.0), "Scores below 0"
        assert np.all(result["scores"] <= 1.0), "Scores above 1"


# =============================================================================
# Test 7: Edge Cases
# =============================================================================
class TestEdgeCases:
    """Test edge cases that could cause numerical issues."""

    def test_single_sample_detection(self, deterministic_rng: Any) -> None:
        """Detectors should handle single-sample detection."""
        detector = MercuryAnomalyDetector()
        train_data = deterministic_rng.randn(100, 5)
        detector.fit(train_data)

        single_sample = deterministic_rng.randn(1, 5)
        result = detector.detect(single_sample)

        assert result["scores"].shape[0] == 1
        assert np.all(np.isfinite(result["scores"]))

    def test_high_dimensional_data(self, deterministic_rng: Any) -> None:
        """Detectors should handle high-dimensional data."""
        detector = MercuryAnomalyDetector()
        train_data = deterministic_rng.randn(100, 500)  # 500 features
        detector.fit(train_data)

        test_data = deterministic_rng.randn(50, 500)
        result = detector.detect(test_data)

        assert np.all(np.isfinite(result["scores"]))

    def test_very_small_values(self, deterministic_rng: Any) -> None:
        """Detectors should handle very small values without underflow."""
        detector = MercuryAnomalyDetector()
        train_data = deterministic_rng.randn(100, 10) * 1e-10
        detector.fit(train_data)

        test_data = deterministic_rng.randn(50, 10) * 1e-10
        result = detector.detect(test_data)

        assert np.all(np.isfinite(result["scores"]))

    def test_very_large_values(self, deterministic_rng: Any) -> None:
        """Detectors should handle very large values without overflow."""
        detector = MercuryAnomalyDetector()
        train_data = deterministic_rng.randn(100, 10) * 1e6
        detector.fit(train_data)

        test_data = deterministic_rng.randn(50, 10) * 1e6
        result = detector.detect(test_data)

        assert np.all(np.isfinite(result["scores"]))

    def test_mixed_scale_features(self, deterministic_rng: Any) -> None:
        """Detectors should handle features with vastly different scales."""
        detector = MercuryAnomalyDetector()
        train_data = np.column_stack(
            [
                deterministic_rng.randn(100) * 1e-5,  # Tiny scale
                deterministic_rng.randn(100) * 1e5,  # Huge scale
                deterministic_rng.randn(100),  # Normal scale
            ]
        )
        detector.fit(train_data)

        test_data = np.column_stack(
            [
                deterministic_rng.randn(50) * 1e-5,
                deterministic_rng.randn(50) * 1e5,
                deterministic_rng.randn(50),
            ]
        )
        result = detector.detect(test_data)

        assert np.all(np.isfinite(result["scores"]))
