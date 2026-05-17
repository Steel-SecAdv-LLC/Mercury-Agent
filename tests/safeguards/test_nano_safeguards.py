"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

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

"""Tests for Nano-Safeguards micro-anomaly detection."""

import numpy as np
import pytest

from omni_mercury_engine.core.exceptions import DetectorException
from omni_mercury_engine.safeguards.nano_safeguards import (
    HierarchicalMicroScanner,
    NanoSafeguardDetector,
    NanoSafeguardResult,
    ResonanceAnalyzer,
)


class TestNanoSafeguardResult:
    """Tests for NanoSafeguardResult dataclass."""

    def test_default_values(self) -> None:
        """Test default values of result dataclass."""
        result = NanoSafeguardResult(
            micro_anomaly_detected=False,
            confidence=0.0,
            alert_level="normal",
        )
        assert result.micro_anomaly_detected is False
        assert result.confidence == 0.0
        assert result.alert_level == "normal"

    def test_full_initialization(self) -> None:
        """Test full initialization with all fields."""
        result = NanoSafeguardResult(
            micro_anomaly_detected=True,
            confidence=0.95,
            alert_level="critical",
            convergence_score=0.005,
            dimensional_residual=0.15,
            hierarchical_scores=[0.1, 0.2, 0.3, 0.4],
            bit_level_anomalies=5,
            molecular_entropy=0.85,
            quantum_checksum=0.75,
            resonance_score=0.9,
            recursion_depth_reached=3,
            refactoring_suggestions=["Reduce learning rate"],
            threshold_violations=["Convergence below threshold"],
            recommended_actions=["Check data quality"],
        )
        assert result.micro_anomaly_detected is True
        assert result.confidence == 0.95
        assert result.convergence_score == 0.005
        assert len(result.hierarchical_scores) == 4
        assert len(result.recommended_actions) == 1


class TestHierarchicalMicroScanner:
    """Tests for HierarchicalMicroScanner."""

    @pytest.fixture
    def scanner(self):
        """Create HierarchicalMicroScanner instance."""
        return HierarchicalMicroScanner(input_dim=64, num_scales=4)

    def test_initialization(self, scanner) -> None:
        """Test scanner initialization."""
        assert scanner is not None
        assert scanner.num_scales == 4

    def test_forward_1d_input(self, scanner) -> None:
        """Test forward pass with 1D input."""
        import torch

        x = torch.randn(64)
        score, features = scanner(x)
        assert score is not None
        assert len(features) == 4

    def test_forward_2d_input(self, scanner) -> None:
        """Test forward pass with 2D input."""
        import torch

        x = torch.randn(8, 64)
        score, features = scanner(x)
        assert score is not None
        assert len(features) == 4


class TestResonanceAnalyzer:
    """Tests for ResonanceAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create ResonanceAnalyzer instance."""
        return ResonanceAnalyzer()

    def test_initialization(self, analyzer) -> None:
        """Test analyzer initialization."""
        assert analyzer is not None
        assert analyzer.fundamental_freq == 7.83

    def test_analyze_normal_signal(self, analyzer) -> None:
        """Test analysis of normal signal."""
        signal = np.random.randn(100)
        result = analyzer.analyze(signal)

        assert "resonance_score" in result
        assert "harmonic_ratio" in result
        assert result["resonance_score"] >= 0.0
        assert result["resonance_score"] <= 1.0

    def test_analyze_short_signal(self, analyzer) -> None:
        """Test analysis of short signal."""
        signal = np.array([1.0, 2.0, 3.0])
        result = analyzer.analyze(signal)

        assert result["resonance_score"] == 0.0
        assert result["harmonic_ratio"] == 0.0

    def test_analyze_empty_signal(self, analyzer) -> None:
        """Test analysis of empty signal."""
        signal = np.array([])
        result = analyzer.analyze(signal)

        assert result["resonance_score"] == 0.0


class TestNanoSafeguardDetector:
    """Tests for NanoSafeguardDetector."""

    @pytest.fixture
    def detector(self):
        """Create NanoSafeguardDetector instance."""
        return NanoSafeguardDetector()

    @pytest.fixture
    def detector_custom_config(self):
        """Create NanoSafeguardDetector with custom config."""
        return NanoSafeguardDetector(
            config={
                "convergence_threshold": 0.005,
                "micro_threshold": 0.1,
                "num_scales": 3,
            }
        )

    @pytest.fixture
    def normal_data(self):
        """Create normal data for testing."""
        return np.random.randn(100, 10)

    @pytest.fixture
    def anomalous_data(self):
        """Create anomalous data for testing."""
        data = np.random.randn(100, 10)
        data[50:60, :] = 100.0
        return data

    def test_initialization(self, detector) -> None:
        """Test detector initialization."""
        assert detector is not None
        assert hasattr(detector, "detect_micro_anomalies")
        assert hasattr(detector, "fit")
        assert hasattr(detector, "detect")

    def test_initialization_custom_config(self, detector_custom_config) -> None:
        """Test detector initialization with custom config."""
        assert detector_custom_config.convergence_threshold == 0.005
        assert detector_custom_config.micro_threshold == 0.1
        assert detector_custom_config.num_scales == 3

    def test_fit(self, detector, normal_data) -> None:
        """Test fitting detector to normal data."""
        detector.fit(normal_data)
        assert detector._is_fitted is True
        assert "mean" in detector.baseline_stats
        assert "std" in detector.baseline_stats

    def test_detect_micro_anomalies_unfitted(self, detector, normal_data) -> None:
        """Test detection raises error when not fitted."""
        with pytest.raises(DetectorException):
            detector.detect(normal_data)

    def test_detect_micro_anomalies_normal(self, detector, normal_data) -> None:
        """Test detection on normal data."""
        detector.fit(normal_data)
        result = detector.detect_micro_anomalies(normal_data)

        assert isinstance(result, NanoSafeguardResult)
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0

    def test_detect_micro_anomalies_anomalous(self, detector, normal_data, anomalous_data) -> None:
        """Test detection on anomalous data."""
        detector.fit(normal_data)
        result = detector.detect_micro_anomalies(anomalous_data)

        assert isinstance(result, NanoSafeguardResult)
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0

    def test_detect_interface(self, detector, normal_data) -> None:
        """Test detect interface returns dict."""
        detector.fit(normal_data)
        result = detector.detect(normal_data)

        assert isinstance(result, dict)
        assert "is_anomaly" in result
        assert "scores" in result
        assert "detector_type" in result
        assert result["detector_type"] == "nano_safeguard"

    def test_extract_features(self, detector, normal_data) -> None:
        """Test feature extraction."""
        features = detector.extract_features(normal_data)
        assert features is not None
        assert len(features) == 20

    def test_alert_levels(self, detector, normal_data) -> None:
        """Test alert level determination."""
        detector.fit(normal_data)
        result = detector.detect_micro_anomalies(normal_data)

        assert result.alert_level in ["normal", "moderate", "warning", "critical", "emergency"]

    def test_threshold_violations(self, detector, normal_data) -> None:
        """Test threshold violation tracking."""
        detector.fit(normal_data)
        result = detector.detect_micro_anomalies(normal_data)

        assert isinstance(result.threshold_violations, list)

    def test_recommended_actions(self, detector, normal_data) -> None:
        """Test recommended actions generation."""
        detector.fit(normal_data)
        result = detector.detect_micro_anomalies(normal_data)

        assert isinstance(result.recommended_actions, list)


class TestNanoSafeguardIntegration:
    """Integration tests for nano-safeguards."""

    def test_detector_instantiation(self) -> None:
        """Test that detector can be instantiated."""
        detector = NanoSafeguardDetector()
        assert detector is not None

    def test_full_pipeline(self) -> None:
        """Test full detection pipeline."""
        detector = NanoSafeguardDetector()

        train_data = np.random.randn(1000, 20)
        detector.fit(train_data)

        test_data = np.random.randn(100, 20)
        result = detector.detect_micro_anomalies(test_data)

        assert isinstance(result, NanoSafeguardResult)
        assert result.alert_level in ["normal", "moderate", "warning", "critical", "emergency"]

    def test_multiple_predictions(self) -> None:
        """Test multiple sequential predictions."""
        detector = NanoSafeguardDetector()
        train_data = np.random.randn(500, 10)
        detector.fit(train_data)

        for i in range(10):
            test_data = np.random.randn(50, 10)
            result = detector.detect_micro_anomalies(test_data)
            assert isinstance(result, NanoSafeguardResult)

    def test_confidence_bounds(self) -> None:
        """Test that confidence is always within [0, 1]."""
        detector = NanoSafeguardDetector()
        train_data = np.random.randn(500, 10)
        detector.fit(train_data)

        for _ in range(20):
            test_data = np.random.randn(50, 10) * np.random.uniform(0.1, 10.0)
            result = detector.detect_micro_anomalies(test_data)
            assert 0.0 <= result.confidence <= 1.0


class TestNanoSafeguardEdgeCases:
    """Edge case tests for nano-safeguards."""

    @pytest.fixture
    def detector(self):
        """Create NanoSafeguardDetector instance."""
        detector = NanoSafeguardDetector()
        detector.fit(np.random.randn(100, 10))
        return detector

    def test_single_sample(self, detector) -> None:
        """Test detection with single sample."""
        data = np.random.randn(1, 10)
        result = detector.detect_micro_anomalies(data)
        assert isinstance(result, NanoSafeguardResult)

    def test_high_dimensional_data(self, detector) -> None:
        """Test detection with high-dimensional data."""
        detector_hd = NanoSafeguardDetector()
        train_data = np.random.randn(100, 100)
        detector_hd.fit(train_data)
        test_data = np.random.randn(50, 100)
        result = detector_hd.detect_micro_anomalies(test_data)
        assert isinstance(result, NanoSafeguardResult)

    def test_1d_data(self, detector) -> None:
        """Test detection with 1D data."""
        detector_1d = NanoSafeguardDetector()
        train_data = np.random.randn(100)
        detector_1d.fit(train_data)
        test_data = np.random.randn(50)
        result = detector_1d.detect_micro_anomalies(test_data)
        assert isinstance(result, NanoSafeguardResult)

    def test_constant_data(self, detector) -> None:
        """Test detection with constant data."""
        detector_const = NanoSafeguardDetector()
        train_data = np.ones((100, 10))
        detector_const.fit(train_data)
        test_data = np.ones((50, 10))
        result = detector_const.detect_micro_anomalies(test_data)
        assert isinstance(result, NanoSafeguardResult)
