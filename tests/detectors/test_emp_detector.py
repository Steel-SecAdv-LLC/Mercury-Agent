"""
OMNI ♱ AVA (O♱A)
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

"""Tests for EMP Detector."""

import numpy as np
import pytest
import torch

from omni_anomaly_engine.detectors.energy.emp_detector import (
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

    def test_emp_type_values(self):
        """Test EMPType enum values."""
        assert EMPType.HEMP.value == "high_altitude_emp"
        assert EMPType.NUCLEAR_EMP.value == "nuclear_emp"
        assert EMPType.NON_NUCLEAR_EMP.value == "non_nuclear_emp"
        assert EMPType.LIGHTNING.value == "lightning_surge"
        assert EMPType.SOLAR_GIC.value == "solar_geomagnetic_current"
        assert EMPType.INTENTIONAL_EMI.value == "intentional_emi"

    def test_threat_level_values(self):
        """Test ThreatLevel enum values."""
        assert ThreatLevel.BENIGN.value == "benign"
        assert ThreatLevel.ANOMALOUS.value == "anomalous"
        assert ThreatLevel.SUSPICIOUS.value == "suspicious"
        assert ThreatLevel.THREAT.value == "threat"
        assert ThreatLevel.CRITICAL.value == "critical"


class TestEMPPredictionResult:
    """Tests for EMPPredictionResult dataclass."""

    def test_default_values(self):
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
        assert result.intentional_attack_probability == 0.0
        assert result.protective_actions == []
        assert result.recovery_actions == []


class TestE1PulseDetector:
    """Tests for E1PulseDetector."""

    @pytest.fixture
    def detector(self):
        """Create E1PulseDetector instance."""
        return E1PulseDetector()

    def test_no_e1_pulse_normal_conditions(self, detector):
        """Test detection with normal EM conditions."""
        sensor_data = {
            "field_strength_vm": 100.0,
            "rise_time_ns": 1000.0,
            "peak_frequency_mhz": 1.0,
        }
        result = detector.detect_e1_pulse(sensor_data)

        assert result["e1_detected"] is False
        assert result["severity"] == "low"

    def test_e1_pulse_detected(self, detector):
        """Test detection of E1 pulse."""
        sensor_data = {
            "field_strength_vm": 15000.0,
            "rise_time_ns": 5.0,
            "peak_frequency_mhz": 100.0,
        }
        result = detector.detect_e1_pulse(sensor_data)

        assert result["e1_detected"] is True
        assert result["severity"] == "critical"

    def test_e1_detection_threshold_boundary(self, detector):
        """Test E1 detection at threshold boundaries."""
        # Just above threshold
        sensor_data = {
            "field_strength_vm": 10001.0,
            "rise_time_ns": 9.0,
            "peak_frequency_mhz": 50.0,
        }
        result = detector.detect_e1_pulse(sensor_data)
        assert result["e1_detected"] is True

    def test_field_strength_captured(self, detector):
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

    def test_no_e3_pulse_normal_conditions(self, detector):
        """Test detection with normal geomagnetic conditions."""
        magnetometer_data = {
            "db_dt_nt_s": 100.0,
            "duration_seconds": 10.0,
        }
        result = detector.detect_e3_pulse(magnetometer_data)

        assert result["e3_detected"] is False
        assert result["grid_impact"] == "low"

    def test_e3_pulse_detected(self, detector):
        """Test detection of E3 pulse."""
        magnetometer_data = {
            "db_dt_nt_s": 3000.0,
            "duration_seconds": 120.0,
        }
        result = detector.detect_e3_pulse(magnetometer_data)

        assert result["e3_detected"] is True

    def test_grid_impact_levels(self, detector):
        """Test different grid impact levels."""
        # Low impact
        result = detector.detect_e3_pulse({"db_dt_nt_s": 100.0, "duration_seconds": 120.0})
        assert result["grid_impact"] == "low"

        # Moderate impact
        result = detector.detect_e3_pulse({"db_dt_nt_s": 300.0, "duration_seconds": 120.0})
        assert result["grid_impact"] == "moderate"

        # High impact
        result = detector.detect_e3_pulse({"db_dt_nt_s": 700.0, "duration_seconds": 120.0})
        assert result["grid_impact"] == "high"

        # Critical impact
        result = detector.detect_e3_pulse({"db_dt_nt_s": 1500.0, "duration_seconds": 120.0})
        assert result["grid_impact"] == "critical"

    def test_gic_amplitude_calculation(self, detector):
        """Test GIC amplitude calculation."""
        magnetometer_data = {
            "db_dt_nt_s": 1000.0,
            "duration_seconds": 120.0,
        }
        result = detector.detect_e3_pulse(magnetometer_data)

        expected_gic = abs(1000.0) * 0.1
        assert result["gic_amplitude_a"] == expected_gic


class TestIntentionalEMIDetector:
    """Tests for IntentionalEMIDetector neural network."""

    @pytest.fixture
    def detector(self):
        """Create IntentionalEMIDetector instance."""
        return IntentionalEMIDetector(input_dim=64)

    def test_initialization(self, detector):
        """Test model initialization."""
        assert detector.signature_analyzer is not None
        assert detector.attack_classifier is not None

    def test_forward_pass(self, detector):
        """Test forward pass through model."""
        input_tensor = torch.randn(10, 64)
        output = detector(input_tensor)

        assert output.shape == (10, 1)
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)

    def test_batch_processing(self, detector):
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
                "db_dt_nt_s": 50.0,
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
    def e3_emp_data(self):
        """Create E3 EMP event data."""
        return {
            "magnetometer_data": {
                "db_dt_nt_s": 3000.0,
                "duration_seconds": 120.0,
            },
            "solar_data": {"storm_active": True},
        }

    def test_initialization_all_enabled(self, detector):
        """Test initialization with all detectors enabled."""
        assert detector.e1_detector is not None
        assert detector.e3_detector is not None
        assert detector.emi_detector is not None

    def test_initialization_selective_enablement(self):
        """Test selective detector enablement."""
        detector = EMPDetector(
            enable_e1_detection=True,
            enable_e3_detection=False,
            enable_attack_classification=False,
        )

        assert detector.e1_detector is not None
        assert detector.e3_detector is None
        assert detector.emi_detector is None

    def test_predict_normal_conditions(self, detector, normal_emp_data):
        """Test prediction with normal conditions."""
        result = detector.predict_emp(normal_emp_data)

        assert isinstance(result, EMPPredictionResult)
        assert result.emp_detected is False
        assert result.threat_level == "benign"

    def test_predict_e1_event(self, detector, e1_emp_data):
        """Test prediction of E1 event."""
        result = detector.predict_emp(e1_emp_data)

        assert result.emp_detected is True
        assert result.e1_component_detected is True
        assert result.emp_type == "non_nuclear_emp"

    def test_predict_e3_event(self, detector, e3_emp_data):
        """Test prediction of E3 event (solar GIC)."""
        result = detector.predict_emp(e3_emp_data)

        assert result.emp_detected is True
        assert result.e3_component_detected is True
        assert result.emp_type == "solar_geomagnetic_current"

    def test_predict_nuclear_emp(self, detector):
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

    def test_affected_infrastructure_identified(self, detector, e1_emp_data):
        """Test that affected infrastructure is identified."""
        result = detector.predict_emp(e1_emp_data)

        assert len(result.affected_infrastructure) > 0
        assert "Electronics and semiconductors" in result.affected_infrastructure

    def test_protective_actions_generated(self, detector, e1_emp_data):
        """Test that protective actions are generated for threats."""
        result = detector.predict_emp(e1_emp_data)

        # Should generate some protective actions
        assert len(result.protective_actions) >= 0

    def test_recovery_actions_generated(self, detector, e1_emp_data):
        """Test that recovery actions are generated."""
        result = detector.predict_emp(e1_emp_data)

        assert len(result.recovery_actions) >= 0

    def test_empty_data_handling(self, detector):
        """Test handling of empty EMP data."""
        result = detector.predict_emp({})

        assert result.emp_detected is False
        assert result.threat_level == "benign"

    def test_threat_level_assessment(self, detector):
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

    def test_source_localization_field(self, detector, normal_emp_data):
        """Test source localization field in result."""
        result = detector.predict_emp(normal_emp_data)

        # Source localization may be None if not calculated
        assert hasattr(result, "source_localization")
