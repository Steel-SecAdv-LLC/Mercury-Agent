"""
Mercury Agent ♱
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

"""
Tests for CISA Critical Infrastructure sector modules.
"""

import numpy as np

from omni_mercury_engine.infrastructure import (
    ChemicalNuclearDetector,
    CISASector,
    CommunicationsITDetector,
    DamType,
    EnergyDamsDetector,
    EnergySubsector,
    HealthcareEmergencyDetector,
    PatientStatus,
)


class TestChemicalNuclearDetector:
    """Tests for Chemical & Nuclear infrastructure detector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = ChemicalNuclearDetector(CISASector.CHEMICAL)
        assert detector.sector == CISASector.CHEMICAL
        assert len(detector.safety_thresholds) > 0

    def test_process_anomaly_detection_normal(self):
        """Test anomaly detection with normal parameters."""
        detector = ChemicalNuclearDetector(CISASector.CHEMICAL)

        sensor_data = np.array([[25.0, 100.0, 7.0, 10.0]])
        parameter_names = ["temperature_celsius", "pressure_psi", "ph_level", "leak_rate_ppm"]

        result = detector.detect_process_anomaly(sensor_data, parameter_names)

        assert result["safety_status"] == "NORMAL"
        assert len(result["anomalies"]) == 0

    def test_process_anomaly_detection_violation(self):
        """Test anomaly detection with parameter violations."""
        detector = ChemicalNuclearDetector(CISASector.CHEMICAL)

        sensor_data = np.array([[250.0, 100.0, 7.0, 10.0]])
        parameter_names = ["temperature_celsius", "pressure_psi", "ph_level", "leak_rate_ppm"]

        result = detector.detect_process_anomaly(sensor_data, parameter_names)

        assert "temperature_celsius" in result["anomalies"]
        assert result["safety_status"] in ["WARNING", "CRITICAL"]

    def test_nuclear_sector(self):
        """Test nuclear sector specific thresholds."""
        detector = ChemicalNuclearDetector(CISASector.NUCLEAR)

        assert "radiation_mrem_hr" in detector.safety_thresholds
        assert "core_temperature_celsius" in detector.safety_thresholds


class TestCommunicationsITDetector:
    """Tests for Communications & IT infrastructure detector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = CommunicationsITDetector()
        assert len(detector.traffic_history) == 0
        assert detector.baseline_window == 3600

    def test_network_anomaly_learning_phase(self):
        """Test detector learning phase."""
        detector = CommunicationsITDetector()

        traffic_data = {"packets_per_sec": 1000, "bytes_per_sec": 100000, "connections_per_sec": 50}

        result = detector.detect_network_anomaly(traffic_data)

        assert result["status"] == "LEARNING"

    def test_network_anomaly_with_baseline(self):
        """Test anomaly detection after baseline established."""
        detector = CommunicationsITDetector()

        for _ in range(200):
            detector.traffic_history.append(
                {"packets_per_sec": 1000, "bytes_per_sec": 100000, "connections_per_sec": 50}
            )

        traffic_data = {"packets_per_sec": 1000, "bytes_per_sec": 100000, "connections_per_sec": 50}

        result = detector.detect_network_anomaly(traffic_data)

        assert "anomalies" in result
        assert "overall_risk" in result


class TestEnergyDamsDetector:
    """Tests for Energy & Dams infrastructure detector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = EnergyDamsDetector()
        assert detector.subsector == EnergySubsector.ELECTRICITY
        assert "frequency_hz" in detector.grid_parameters

    def test_grid_anomaly_normal(self):
        """Test grid anomaly detection with normal frequency."""
        detector = EnergyDamsDetector()

        grid_data = {"frequency_hz": np.array([60.0, 60.01, 59.99])}

        result = detector.detect_grid_anomaly(grid_data)

        assert result["grid_status"] == "NORMAL"
        assert len(result["anomalies"]) == 0

    def test_grid_anomaly_frequency_deviation(self):
        """Test grid anomaly detection with frequency deviation."""
        detector = EnergyDamsDetector()

        grid_data = {"frequency_hz": np.array([60.0, 60.1, 60.2])}

        result = detector.detect_grid_anomaly(grid_data)

        assert "frequency" in result["anomalies"]
        assert result["grid_status"] in ["ALERT", "EMERGENCY"]

    def test_dam_anomaly_normal(self):
        """Test dam anomaly detection with normal parameters."""
        detector = EnergyDamsDetector()

        dam_data = {"seepage_gpm": 20, "displacement_mm": 5, "water_level_ft": 300}

        result = detector.detect_dam_anomaly(dam_data, DamType.HYDROELECTRIC)

        assert result["safety_status"] == "SAFE"
        assert len(result["anomalies"]) == 0

    def test_dam_anomaly_excessive_seepage(self):
        """Test dam anomaly detection with excessive seepage."""
        detector = EnergyDamsDetector()

        dam_data = {"seepage_gpm": 150, "displacement_mm": 5, "water_level_ft": 300}

        result = detector.detect_dam_anomaly(dam_data)

        assert "seepage" in result["anomalies"]
        assert result["safety_status"] in ["WARNING", "CRITICAL"]


class TestHealthcareEmergencyDetector:
    """Tests for Healthcare & Emergency Services infrastructure detector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = HealthcareEmergencyDetector()
        assert len(detector.vital_sign_ranges) > 0

    def test_patient_deterioration_stable(self):
        """Test patient deterioration detection with stable vitals."""
        detector = HealthcareEmergencyDetector()

        vital_signs = {
            "heart_rate_bpm": 75,
            "blood_pressure_systolic": 110,
            "oxygen_saturation_pct": 98,
            "temperature_f": 98.6,
            "respiratory_rate_bpm": 16,
        }

        result = detector.detect_patient_deterioration(vital_signs)

        assert result["patient_status"] == PatientStatus.STABLE.value
        assert result["early_warning_score"] == 0
        assert not result["requires_intervention"]

    def test_patient_deterioration_critical(self):
        """Test patient deterioration detection with critical vitals."""
        detector = HealthcareEmergencyDetector()

        vital_signs = {
            "heart_rate_bpm": 140,
            "blood_pressure_systolic": 70,
            "oxygen_saturation_pct": 80,
            "temperature_f": 104.0,
            "respiratory_rate_bpm": 35,
        }

        result = detector.detect_patient_deterioration(vital_signs)

        assert result["patient_status"] in [
            PatientStatus.CRITICAL.value,
            PatientStatus.DETERIORATING.value,
        ]
        assert result["early_warning_score"] >= 5
        assert result["requires_intervention"]

    def test_emergency_call_anomaly_normal(self):
        """Test emergency call anomaly detection with normal volume."""
        detector = HealthcareEmergencyDetector()

        call_data = {"total_calls": 95, "medical_calls": 50, "fire_calls": 20, "police_calls": 25}

        result = detector.detect_emergency_call_anomaly(call_data)

        assert result["call_volume_status"] == "NORMAL"
        assert result["estimated_event_type"] == "normal_operations"

    def test_emergency_call_anomaly_surge(self):
        """Test emergency call anomaly detection with call surge."""
        detector = HealthcareEmergencyDetector()

        call_data = {"total_calls": 250, "medical_calls": 150, "fire_calls": 50, "police_calls": 50}

        result = detector.detect_emergency_call_anomaly(call_data)

        assert result["call_volume_status"] in ["HIGH", "CRITICAL"]
        assert "call_surge" in result["anomalies"]
