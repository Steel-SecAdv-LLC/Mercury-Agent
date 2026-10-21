# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for EMP Detector."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("torch")

import pytest
import torch

from omni_mercury_engine.detectors.energy.emp_detector import (
    E1PulseDetector,
    E3PulseDetector,
    EMPDetector,
    EMPPredictionResult,
    EMPType,
    IntentionalEMIDetector,
    ThreatLevel,
)


class TestEMPEnums:
    """Tests for EMP enumerations."""

    def test_emp_type_values(self) -> None:
        """Test EMPType enum values."""
        assert EMPType.HEMP.value == "high_altitude_emp"
        assert EMPType.NUCLEAR_EMP.value == "nuclear_emp"
        assert EMPType.NON_NUCLEAR_EMP.value == "non_nuclear_emp"
        assert EMPType.LIGHTNING.value == "lightning_surge"
        assert EMPType.SOLAR_GIC.value == "solar_geomagnetic_current"
        assert EMPType.INTENTIONAL_EMI.value == "intentional_emi"

    def test_threat_level_values(self) -> None:
        """Test ThreatLevel enum values."""
        assert ThreatLevel.BENIGN.value == "benign"
        assert ThreatLevel.ANOMALOUS.value == "anomalous"
        assert ThreatLevel.SUSPICIOUS.value == "suspicious"
        assert ThreatLevel.THREAT.value == "threat"
        assert ThreatLevel.CRITICAL.value == "critical"


class TestEMPPredictionResult:
    """Tests for EMPPredictionResult dataclass."""

    def test_default_values(self) -> None:
        """Test default values of result dataclass."""
        result = EMPPredictionResult(
            emp_detected=False,
            confidence=0.0,
            emp_type="benign",
            threat_level="benign",
        )
        assert result.e1_component_detected is False
        assert result.e2_component_detected is False
        assert result.e3_component_detected is False
        assert result.grid_impact_assessment == "none"
        # None until a trained IEMI network is consulted (0.0 claimed "no attack").
        assert result.intentional_attack_probability is None
        assert result.protective_actions == []
        assert result.recovery_actions == []


class TestE1PulseDetector:
    """Tests for E1PulseDetector."""

    @pytest.fixture
    def detector(self):
        """Create E1PulseDetector instance."""
        return E1PulseDetector()

    def test_no_e1_pulse_normal_conditions(self, detector: Any) -> None:
        """Test detection with normal EM conditions."""
        sensor_data = {
            "field_strength_vm": 100.0,
            "rise_time_ns": 1000.0,
            "peak_frequency_mhz": 1.0,
        }
        result = detector.detect_e1_pulse(sensor_data)

        assert result["e1_detected"] is False
        assert result["severity"] == "low"

    def test_e1_pulse_detected(self, detector: Any) -> None:
        """Test detection of E1 pulse."""
        sensor_data = {
            "field_strength_vm": 15000.0,
            "rise_time_ns": 5.0,
            "peak_frequency_mhz": 100.0,
        }
        result = detector.detect_e1_pulse(sensor_data)

        assert result["e1_detected"] is True
        assert result["severity"] == "critical"

    def test_e1_detection_threshold_boundary(self, detector: Any) -> None:
        """Test E1 detection at threshold boundaries."""
        # Just above threshold
        sensor_data = {
            "field_strength_vm": 10001.0,
            "rise_time_ns": 9.0,
            "peak_frequency_mhz": 50.0,
        }
        result = detector.detect_e1_pulse(sensor_data)
        assert result["e1_detected"] is True

    def test_field_strength_captured(self, detector: Any) -> None:
        """Test that field strength is captured in result."""
        sensor_data = {"field_strength_vm": 5000.0}
        result = detector.detect_e1_pulse(sensor_data)

        assert result["field_strength_vm"] == 5000.0


class TestE3PulseDetector:
    """Tests for E3PulseDetector."""

    @pytest.fixture
    def detector(self):
        """Create E3PulseDetector instance."""
        return E3PulseDetector()

    def test_no_e3_pulse_normal_conditions(self, detector: Any) -> None:
        """Test detection with normal geomagnetic conditions."""
        # 0.5 nT/s = 30 nT/min; 100 nT/s is 300 times the largest storm rate
        # observed (~20 nT/s, March 1989), not a normal condition.
        magnetometer_data = {
            "db_dt_nt_s": 0.5,
            "duration_seconds": 10.0,
        }
        result = detector.detect_e3_pulse(magnetometer_data)

        assert result["e3_detected"] is False
        assert result["grid_impact"] == "low"

    def test_e3_pulse_detected(self, detector: Any) -> None:
        """Test detection of E3 pulse."""
        magnetometer_data = {
            "db_dt_nt_s": 3000.0,
            "duration_seconds": 120.0,
        }
        result = detector.detect_e3_pulse(magnetometer_data)

        assert result["e3_detected"] is True

    def test_grid_impact_levels(self, detector: Any) -> None:
        """Test different grid impact levels."""
        # The GIC detector's 1-minute tiers: 90 / 300 / 480 nT/min.
        # Low impact
        result = detector.detect_e3_pulse({"db_dt_nt_s": 1.0, "duration_seconds": 120.0})
        assert result["grid_impact"] == "low"

        # Moderate impact
        result = detector.detect_e3_pulse({"db_dt_nt_s": 2.0, "duration_seconds": 120.0})
        assert result["grid_impact"] == "moderate"

        # High impact
        result = detector.detect_e3_pulse({"db_dt_nt_s": 6.0, "duration_seconds": 120.0})
        assert result["grid_impact"] == "high"

        # Critical impact
        result = detector.detect_e3_pulse({"db_dt_nt_s": 9.0, "duration_seconds": 120.0})
        assert result["grid_impact"] == "critical"

    def test_peak_dbdt_reported(self, detector: Any) -> None:
        """The measured |dB/dt| is reported in its unit.

        It replaces ``gic_amplitude_a`` (0.1 x |dB/dt| labelled amperes): a GIC
        in amperes depends on the network and ground model, not on dB/dt alone.
        """
        magnetometer_data = {
            "db_dt_nt_s": -1000.0,
            "duration_seconds": 120.0,
        }
        result = detector.detect_e3_pulse(magnetometer_data)

        assert result["peak_dbdt_nt_per_s"] == 1000.0


class TestIntentionalEMIDetector:
    """Tests for IntentionalEMIDetector neural network."""

    @pytest.fixture
    def detector(self):
        """Create IntentionalEMIDetector instance."""
        return IntentionalEMIDetector(input_dim=64)

    def test_initialization(self, detector: Any) -> None:
        """Test model initialization."""
        assert detector.signature_analyzer is not None
        assert detector.attack_classifier is not None

    def test_forward_pass(self, detector: Any) -> None:
        """Test forward pass through model."""
        input_tensor = torch.randn(10, 64)
        output = detector(input_tensor)

        assert output.shape == (10, 1)
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)

    def test_batch_processing(self, detector: Any) -> None:
        """Test processing of batched inputs."""
        for batch_size in [1, 5, 20]:
            input_tensor = torch.randn(batch_size, 64)
            output = detector(input_tensor)
            assert output.shape == (batch_size, 1)


class TestEMPDetector:
    """Tests for comprehensive EMPDetector."""

    @pytest.fixture
    def detector(self):
        """Create EMPDetector instance."""
        return EMPDetector()

    @pytest.fixture
    def normal_emp_data(self):
        """Create normal (non-EMP) conditions data."""
        return {
            "sensor_data": {
                "field_strength_vm": 100.0,
                "rise_time_ns": 1000.0,
                "peak_frequency_mhz": 1.0,
            },
            "magnetometer_data": {
                # Quiet field (30 nT/min); 50 nT/s is a "critical" grid impact.
                "db_dt_nt_s": 0.5,
                "duration_seconds": 10.0,
            },
            "signature_data": {"repetition_rate_hz": 0.0},
        }

    @pytest.fixture
    def e1_emp_data(self):
        """Create E1 EMP event data."""
        return {
            "sensor_data": {
                "field_strength_vm": 20000.0,
                "rise_time_ns": 5.0,
                "peak_frequency_mhz": 100.0,
            },
            "signature_data": {"repetition_rate_hz": 0.0},
        }

    @pytest.fixture
    def storm_gic_data(self):
        """Create severe-storm GIC data (Gannon storm, Sitka: 553.7 nT/min)."""
        return {
            "magnetometer_data": {
                "db_dt_nt_s": 553.7 / 60.0,
                "duration_seconds": 600.0,
            },
            "solar_data": {"storm_active": True},
        }

    def test_initialization_all_enabled(self, detector: Any) -> None:
        """Test initialization with all detectors enabled."""
        assert detector.e1_detector is not None
        assert detector.e3_detector is not None
        assert detector.emi_detector is not None

    def test_initialization_selective_enablement(self) -> None:
        """Test selective detector enablement."""
        detector = EMPDetector(
            enable_e1_detection=True,
            enable_e3_detection=False,
            enable_attack_classification=False,
        )

        assert detector.e1_detector is not None
        assert detector.e3_detector is None
        assert detector.emi_detector is None

    def test_predict_normal_conditions(self, detector: Any, normal_emp_data: Any) -> None:
        """Test prediction with normal conditions."""
        result = detector.predict_emp(normal_emp_data)

        assert isinstance(result, EMPPredictionResult)
        assert result.emp_detected is False
        assert result.threat_level == "benign"

    def test_predict_e1_event(self, detector: Any, e1_emp_data: Any) -> None:
        """Test prediction of E1 event."""
        result = detector.predict_emp(e1_emp_data)

        assert result.emp_detected is True
        assert result.e1_component_detected is True
        # E1 is the HEMP early-time pulse (IEC 61000-2-9), not "non-nuclear".
        assert result.emp_type == "high_altitude_emp"

    def test_predict_storm_gic_event(self, detector: Any, storm_gic_data: Any) -> None:
        """Test prediction of a solar-storm GIC event.

        A storm's dB/dt (~20 nT/s at most observed) is far below E3 (>= 450
        nT/s), so it is a geomagnetic disturbance with a grid impact, not an
        EMP; the former 3000 nT/s "solar" input was no storm.
        """
        result = detector.predict_emp(storm_gic_data)

        assert result.emp_detected is False
        assert result.e3_component_detected is False
        assert result.emp_type == "solar_geomagnetic_current"
        assert result.grid_impact_assessment == "critical"

    def test_predict_nuclear_emp(self, detector: Any) -> None:
        """Test prediction of nuclear EMP (E1 + E3)."""
        data = {
            "sensor_data": {
                "field_strength_vm": 20000.0,
                "rise_time_ns": 5.0,
                "peak_frequency_mhz": 100.0,
            },
            "magnetometer_data": {
                "db_dt_nt_s": 3000.0,
                "duration_seconds": 120.0,
            },
        }
        result = detector.predict_emp(data)

        assert result.e1_component_detected is True
        assert result.e3_component_detected is True
        assert result.emp_type == "nuclear_emp"
        assert result.threat_level == "critical"

    def test_affected_infrastructure_identified(self, detector: Any, e1_emp_data: Any) -> None:
        """Test that affected infrastructure is identified."""
        result = detector.predict_emp(e1_emp_data)

        assert len(result.affected_infrastructure) > 0
        assert "Electronics and semiconductors" in result.affected_infrastructure

    def test_protective_actions_generated(self, detector: Any, e1_emp_data: Any) -> None:
        """Test that protective actions are generated for threats."""
        result = detector.predict_emp(e1_emp_data)

        # Should generate some protective actions
        assert len(result.protective_actions) >= 0

    def test_recovery_actions_generated(self, detector: Any, e1_emp_data: Any) -> None:
        """Test that recovery actions are generated."""
        result = detector.predict_emp(e1_emp_data)

        assert len(result.recovery_actions) >= 0

    def test_empty_data_handling(self, detector: Any) -> None:
        """Test handling of empty EMP data."""
        result = detector.predict_emp({})

        assert result.emp_detected is False
        assert result.threat_level == "benign"

    def test_threat_level_assessment(self, detector: Any) -> None:
        """Test threat level assessment function."""
        result = EMPPredictionResult(
            emp_detected=True,
            confidence=0.9,
            emp_type="nuclear_emp",
            threat_level="critical",
            intentional_attack_probability=0.9,
        )

        level = detector._assess_threat_level(result, 2)
        assert level == "critical"

    def test_source_localization_field(self, detector: Any, normal_emp_data: Any) -> None:
        """Test source localization field in result."""
        result = detector.predict_emp(normal_emp_data)

        # Source localization may be None if not calculated
        assert hasattr(result, "source_localization")
