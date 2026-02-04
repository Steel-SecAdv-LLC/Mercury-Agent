"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

Tests for Advanced Physics Integration Module.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from omni_mercury_engine.detectors.advanced_physics_integration import (
    AdvancedPhysicsIntegratedDetector,
    PhysicsGOSNNScalars,
    create_dynamics_detector,
    create_integrated_detector,
    create_spectral_detector,
    create_uiux_detector,
)
from omni_mercury_engine.detectors.uiux_anomaly import (
    InteractionType,
    UserInteraction,
)


def create_test_time_series(length: int = 1000) -> np.ndarray:
    """Create test time series data."""
    t = np.linspace(0, 10, length)
    signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 25 * t)
    signal += np.random.randn(length) * 0.1
    return signal


def create_test_interactions(count: int = 30) -> list[UserInteraction]:
    """Create test user interactions."""
    interactions = []
    for i in range(count):
        interactions.append(
            UserInteraction(
                timestamp=i * 0.5,
                interaction_type=InteractionType.CLICK,
                x=100 + i * 10,
                y=200 + i * 5,
                element_id=f"element_{i % 5}",
                page_url=f"/page_{i % 3}",
            )
        )
    return interactions


class TestAdvancedPhysicsIntegratedDetector:
    """Tests for AdvancedPhysicsIntegratedDetector."""

    def test_init_default_config(self) -> None:
        """Test initialization with default configuration."""
        detector = AdvancedPhysicsIntegratedDetector()
        assert detector is not None
        assert not detector.is_fitted()

    def test_init_with_all_detectors(self) -> None:
        """Test initialization with all detectors enabled."""
        config = {
            "enabled_detectors": ["all"],
            "threshold": 0.6,
        }
        detector = AdvancedPhysicsIntegratedDetector(config)
        assert detector._spectral_detector is not None
        assert detector._dynamics_detector is not None
        assert detector._uiux_detector is not None

    def test_init_with_specific_detectors(self) -> None:
        """Test initialization with specific detectors."""
        config = {
            "enabled_detectors": ["spectral_vibration", "acceleration_dynamics"],
        }
        detector = AdvancedPhysicsIntegratedDetector(config)
        assert detector._spectral_detector is not None
        assert detector._dynamics_detector is not None
        assert detector._uiux_detector is None

    def test_init_with_3r_disabled(self) -> None:
        """Test initialization with 3R enhancement disabled."""
        config = {
            "use_3r_enhancement": False,
        }
        detector = AdvancedPhysicsIntegratedDetector(config)
        assert detector._fusion_equation is None
        assert detector._recursion_engine is None

    def test_fit_time_series(self) -> None:
        """Test fitting on time series data."""
        detector = AdvancedPhysicsIntegratedDetector(
            {
                "enabled_detectors": ["spectral_vibration", "acceleration_dynamics"],
            }
        )

        signal = create_test_time_series()
        detector.fit(signal, data_type="time_series")

        assert detector.is_fitted()
        status = detector.get_detector_status()
        assert status["spectral_vibration"]
        assert status["acceleration_dynamics"]

    def test_fit_interactions(self) -> None:
        """Test fitting on user interactions."""
        detector = AdvancedPhysicsIntegratedDetector(
            {
                "enabled_detectors": ["uiux_anomaly"],
            }
        )

        interactions = create_test_interactions(count=50)
        detector.fit(interactions, data_type="interactions")

        assert detector.is_fitted()
        status = detector.get_detector_status()
        assert status["uiux_anomaly"]

    def test_fit_mixed_data(self) -> None:
        """Test fitting on mixed data types."""
        detector = AdvancedPhysicsIntegratedDetector(
            {
                "enabled_detectors": ["all"],
            }
        )

        mixed_data = {
            "time_series": create_test_time_series(),
            "interactions": create_test_interactions(count=50),
        }

        detector.fit(mixed_data, data_type="mixed")
        assert detector.is_fitted()

    def test_detect_time_series(self) -> None:
        """Test detection on time series."""
        detector = AdvancedPhysicsIntegratedDetector(
            {
                "enabled_detectors": ["spectral_vibration", "acceleration_dynamics"],
                "threshold": 0.6,
            }
        )

        train_signal = create_test_time_series()
        detector.fit(train_signal, data_type="time_series")

        test_signal = create_test_time_series()
        result = detector.detect(test_signal, data_type="time_series")

        assert "is_anomaly" in result
        assert "anomaly_score" in result
        assert "spectral_result" in result
        assert "dynamics_result" in result
        assert "component_scores" in result
        assert "ethical_scaling" in result

    def test_detect_with_3r_enhancement(self) -> None:
        """Test detection with 3R enhancement."""
        detector = AdvancedPhysicsIntegratedDetector(
            {
                "enabled_detectors": ["spectral_vibration", "acceleration_dynamics"],
                "use_3r_enhancement": True,
            }
        )

        signal = create_test_time_series()
        detector.fit(signal, data_type="time_series")
        result = detector.detect(signal, data_type="time_series")

        assert "fusion_result" in result
        assert "recursion_score" in result
        assert "resonance_score" in result

    def test_detect_with_gosnn_scaling(self) -> None:
        """Test detection with GOSNN ethical scaling."""
        detector = AdvancedPhysicsIntegratedDetector(
            {
                "enabled_detectors": ["all"],
                "use_gosnn_scaling": True,
            }
        )

        mixed_data = {
            "time_series": create_test_time_series(),
            "interactions": create_test_interactions(count=50),
        }
        detector.fit(mixed_data, data_type="mixed")

        result = detector.detect(mixed_data, data_type="mixed")

        assert "ethical_scaling" in result
        assert 0 < result["ethical_scaling"] <= 1

    def test_detect_returns_recommendations(self) -> None:
        """Test that detection returns recommendations."""
        detector = AdvancedPhysicsIntegratedDetector()

        signal = create_test_time_series()
        detector.fit(signal, data_type="time_series")
        result = detector.detect(signal, data_type="time_series")

        assert "recommendations" in result
        assert isinstance(result["recommendations"], list)

    def test_extract_features(self) -> None:
        """Test feature extraction."""
        detector = AdvancedPhysicsIntegratedDetector(
            {
                "enabled_detectors": ["spectral_vibration", "acceleration_dynamics"],
            }
        )

        signal = create_test_time_series()
        detector.fit(signal, data_type="time_series")

        features = detector.extract_features(signal, data_type="time_series")
        assert isinstance(features, torch.Tensor)

    def test_get_detector_status(self) -> None:
        """Test getting detector status."""
        detector = AdvancedPhysicsIntegratedDetector(
            {
                "enabled_detectors": ["spectral_vibration"],
            }
        )

        status = detector.get_detector_status()
        assert "spectral_vibration" in status
        assert "acceleration_dynamics" in status
        assert "uiux_anomaly" in status

    def test_get_gosnn_scalars(self) -> None:
        """Test getting GOSNN scalars."""
        detector = AdvancedPhysicsIntegratedDetector()
        scalars = detector.get_gosnn_scalars()

        assert "SPECTRAL_INTEGRITY" in scalars
        assert "KINEMATIC_PRECISION" in scalars
        assert "UIUX_FAIRNESS" in scalars

    def test_detect_not_fitted_raises(self) -> None:
        """Test that detection before fitting raises exception."""
        detector = AdvancedPhysicsIntegratedDetector()
        with pytest.raises((ValueError, RuntimeError)):
            detector.detect(create_test_time_series(), data_type="time_series")


class TestPhysicsGOSNNScalars:
    """Tests for PhysicsGOSNNScalars."""

    def test_get_all_scalars(self) -> None:
        """Test getting all scalars."""
        scalars = PhysicsGOSNNScalars.get_all_scalars()

        assert len(scalars) > 0
        for name, value in scalars.items():
            assert isinstance(value, float)
            assert 0 <= value <= 1

    def test_scalar_values(self) -> None:
        """Test individual scalar values."""
        assert PhysicsGOSNNScalars.SPECTRAL_INTEGRITY == 0.95
        assert PhysicsGOSNNScalars.KINEMATIC_PRECISION == 0.94
        assert PhysicsGOSNNScalars.HUMANITARIAN_IMPACT == 0.97

    def test_compute_ethical_scaling_basic(self) -> None:
        """Test basic ethical scaling computation."""
        context = {}
        scaling = PhysicsGOSNNScalars.compute_ethical_scaling(context)

        assert 0 < scaling <= 1

    def test_compute_ethical_scaling_with_uiux(self) -> None:
        """Test ethical scaling with UIUX context."""
        context = {
            "uiux_enabled": True,
            "anomaly_detected": False,
            "maintenance_context": False,
        }
        scaling = PhysicsGOSNNScalars.compute_ethical_scaling(context)

        assert 0 < scaling <= 1

    def test_compute_ethical_scaling_with_anomaly(self) -> None:
        """Test ethical scaling when anomaly detected."""
        context = {
            "uiux_enabled": False,
            "anomaly_detected": True,
            "maintenance_context": False,
        }
        scaling = PhysicsGOSNNScalars.compute_ethical_scaling(context)

        assert 0 < scaling <= 1


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_spectral_detector(self) -> None:
        """Test spectral detector factory."""
        detector = create_spectral_detector()
        assert detector is not None

        detector_with_config = create_spectral_detector({"threshold": 0.7})
        assert detector_with_config.threshold == 0.7

    def test_create_dynamics_detector(self) -> None:
        """Test dynamics detector factory."""
        detector = create_dynamics_detector()
        assert detector is not None

        detector_with_config = create_dynamics_detector({"time_step": 0.01})
        assert detector_with_config._dynamics_config.time_step == 0.01

    def test_create_uiux_detector(self) -> None:
        """Test UIUX detector factory."""
        detector = create_uiux_detector()
        assert detector is not None

        detector_with_config = create_uiux_detector({"rage_click_count": 5})
        assert detector_with_config._uiux_config.rage_click_count == 5

    def test_create_integrated_detector(self) -> None:
        """Test integrated detector factory."""
        detector = create_integrated_detector()
        assert detector is not None

        detector_with_config = create_integrated_detector(
            {
                "enabled_detectors": ["spectral_vibration"],
                "threshold": 0.8,
            }
        )
        assert detector_with_config.threshold == 0.8


class TestIntegrationScenarios:
    """Integration tests for real-world scenarios."""

    def test_predictive_maintenance_scenario(self) -> None:
        """Test predictive maintenance scenario with spectral analysis."""
        detector = AdvancedPhysicsIntegratedDetector(
            {
                "enabled_detectors": ["spectral_vibration"],
                "spectral_config": {
                    "sample_rate": 10000,
                    "fft_size": 2048,
                },
            }
        )

        # Simulate normal vibration signal
        t = np.linspace(0, 1, 10000)
        normal_signal = np.sin(2 * np.pi * 100 * t)  # 100 Hz fundamental
        normal_signal += 0.1 * np.random.randn(len(t))

        detector.fit(normal_signal, data_type="time_series")

        # Test with bearing fault signature (additional frequency)
        fault_signal = normal_signal.copy()
        fault_signal += 0.3 * np.sin(2 * np.pi * 42 * t)  # Fault frequency

        result = detector.detect(fault_signal, data_type="time_series")

        assert result["spectral_result"] is not None
        # Should detect the changed frequency content

    def test_system_monitoring_scenario(self) -> None:
        """Test system monitoring scenario with dynamics analysis."""
        detector = AdvancedPhysicsIntegratedDetector(
            {
                "enabled_detectors": ["acceleration_dynamics"],
                "dynamics_config": {
                    "time_step": 1.0,  # 1 second intervals
                },
            }
        )

        # Simulate normal metric behavior
        t = np.arange(1000)
        normal_metrics = 50 + 5 * np.sin(2 * np.pi * t / 100)  # Slight oscillation
        normal_metrics += np.random.randn(1000) * 2

        detector.fit(normal_metrics, data_type="time_series")

        # Test with sudden spike (anomaly)
        anomaly_metrics = normal_metrics.copy()
        anomaly_metrics[500:510] = 100  # Sudden spike

        result = detector.detect(anomaly_metrics, data_type="time_series")

        assert result["dynamics_result"] is not None
        assert "anomaly_timestamps" in result["dynamics_result"]

    def test_user_experience_scenario(self) -> None:
        """Test user experience monitoring scenario."""
        detector = AdvancedPhysicsIntegratedDetector(
            {
                "enabled_detectors": ["uiux_anomaly"],
            }
        )

        # Normal user session
        normal_interactions = []
        for i in range(50):
            normal_interactions.append(
                UserInteraction(
                    timestamp=i * (0.5 + 0.3 * np.random.rand()),
                    interaction_type=InteractionType.CLICK if i % 3 else InteractionType.SCROLL,
                    x=100 + np.random.randint(0, 1000),
                    y=100 + np.random.randint(0, 500),
                    page_url=f"/page_{i % 5}",
                )
            )

        detector.fit(normal_interactions, data_type="interactions")

        # Frustrated user session (rage clicks)
        frustrated_interactions = []
        for i in range(30):
            frustrated_interactions.append(
                UserInteraction(
                    timestamp=i * 0.1,  # Very fast
                    interaction_type=InteractionType.CLICK,
                    x=500,
                    y=300,
                    page_url="/stuck_page",
                )
            )

        result = detector.detect(frustrated_interactions, data_type="interactions")

        assert result["uiux_result"] is not None
        # Should detect frustration signals

    def test_full_system_integration(self) -> None:
        """Test full system with all detectors and 3R enhancement."""
        detector = AdvancedPhysicsIntegratedDetector(
            {
                "enabled_detectors": ["all"],
                "use_3r_enhancement": True,
                "use_gosnn_scaling": True,
                "ethical_compliance_threshold": 0.96,
            }
        )

        # Mixed training data
        train_data = {
            "time_series": create_test_time_series(length=2000),
            "interactions": create_test_interactions(count=100),
        }
        detector.fit(train_data, data_type="mixed")

        # Mixed test data
        test_data = {
            "time_series": create_test_time_series(length=1000),
            "interactions": create_test_interactions(count=50),
        }
        result = detector.detect(test_data, data_type="mixed")

        # Verify all components are present
        assert result["spectral_result"] is not None
        assert result["dynamics_result"] is not None
        assert result["uiux_result"] is not None
        assert result["fusion_result"] is not None
        assert result["ethical_scaling"] > 0
        assert len(result["recommendations"]) > 0
