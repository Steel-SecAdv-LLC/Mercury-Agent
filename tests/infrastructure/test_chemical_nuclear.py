# Copyright (C) 2025 Steel Security Advisors LLC
"""Tests for Chemical and Nuclear Infrastructure Detector."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.infrastructure.chemical_nuclear import ChemicalNuclearDetector, CISASector


class TestCISASector:
    """Tests for CISASector enumeration."""

    def test_sector_values(self) -> None:
        """Test CISASector enum values."""
        assert CISASector.CHEMICAL.value == "chemical"
        assert CISASector.NUCLEAR.value == "nuclear_reactors_materials_waste"


class TestChemicalNuclearDetectorInitialization:
    """Tests for ChemicalNuclearDetector initialization."""

    def test_chemical_sector_initialization(self) -> None:
        """Test initialization for chemical sector."""
        detector = ChemicalNuclearDetector(sector=CISASector.CHEMICAL)

        assert detector.sector == CISASector.CHEMICAL
        assert "temperature_celsius" in detector.safety_thresholds
        assert "pressure_psi" in detector.safety_thresholds
        assert "ph_level" in detector.safety_thresholds
        assert "leak_rate_ppm" in detector.safety_thresholds

    def test_nuclear_sector_initialization(self) -> None:
        """Test initialization for nuclear sector."""
        detector = ChemicalNuclearDetector(sector=CISASector.NUCLEAR)

        assert detector.sector == CISASector.NUCLEAR
        assert "radiation_mrem_hr" in detector.safety_thresholds
        assert "core_temperature_celsius" in detector.safety_thresholds
        assert "coolant_flow_gpm" in detector.safety_thresholds
        assert "neutron_flux" in detector.safety_thresholds

    def test_interdependency_map_chemical(self) -> None:
        """Test interdependency map for chemical sector."""
        detector = ChemicalNuclearDetector(sector=CISASector.CHEMICAL)

        assert "energy" in detector.interdependency_map[CISASector.CHEMICAL]
        assert "water" in detector.interdependency_map[CISASector.CHEMICAL]

    def test_interdependency_map_nuclear(self) -> None:
        """Test interdependency map for nuclear sector."""
        detector = ChemicalNuclearDetector(sector=CISASector.NUCLEAR)

        assert "energy" in detector.interdependency_map[CISASector.NUCLEAR]
        assert "communications" in detector.interdependency_map[CISASector.NUCLEAR]


class TestChemicalSectorDetection:
    """Tests for chemical sector anomaly detection."""

    @pytest.fixture
    def chemical_detector(self):
        """Create chemical sector detector."""
        return ChemicalNuclearDetector(sector=CISASector.CHEMICAL)

    @pytest.fixture
    def normal_chemical_data(self):
        """Create normal chemical sensor data."""
        return np.array(
            [
                [25.0, 500.0, 7.0, 10.0],  # temp, pressure, pH, leak_rate
                [30.0, 550.0, 6.5, 15.0],
                [28.0, 480.0, 7.5, 12.0],
            ]
        )

    @pytest.fixture
    def anomalous_chemical_data(self):
        """Create anomalous chemical sensor data (temperature violation)."""
        return np.array(
            [
                [25.0, 500.0, 7.0, 10.0],
                [250.0, 550.0, 6.5, 15.0],  # Temperature above 200°C threshold
                [28.0, 480.0, 7.5, 12.0],
            ]
        )

    def test_detect_normal_conditions(
        self, chemical_detector: Any, normal_chemical_data: Any
    ) -> None:
        """Test detection with normal chemical conditions."""
        param_names = ["temperature_celsius", "pressure_psi", "ph_level", "leak_rate_ppm"]
        result = chemical_detector.detect(normal_chemical_data, parameter_names=param_names)

        assert result["safety_status"] == "NORMAL"
        assert result["sector"] == "chemical"
        assert len(result["anomalies"]) == 0

    def test_detect_temperature_violation(
        self, chemical_detector: Any, anomalous_chemical_data: Any
    ) -> None:
        """Test detection of temperature threshold violation."""
        param_names = ["temperature_celsius", "pressure_psi", "ph_level", "leak_rate_ppm"]
        result = chemical_detector.detect(anomalous_chemical_data, parameter_names=param_names)

        assert result["safety_status"] == "WARNING"
        assert "temperature_celsius" in result["anomalies"]

    def test_detect_multiple_violations(self, chemical_detector: Any) -> None:
        """Test detection of multiple violations."""
        data = np.array(
            [
                [250.0, 1500.0, 1.0, 150.0],  # All params violate thresholds
            ]
        )
        param_names = ["temperature_celsius", "pressure_psi", "ph_level", "leak_rate_ppm"]
        result = chemical_detector.detect(data, parameter_names=param_names)

        assert result["safety_status"] in ["WARNING", "CRITICAL"]
        assert len(result["anomalies"]) > 1

    def test_detect_leak_rate_violation(self, chemical_detector: Any) -> None:
        """Test detection of high leak rate (emergency condition)."""
        data = np.array(
            [
                [25.0, 500.0, 7.0, 150.0],  # Leak rate above 100 ppm
                [25.0, 500.0, 7.0, 180.0],
                [25.0, 500.0, 7.0, 200.0],
                [25.0, 500.0, 7.0, 220.0],
            ]
        )
        param_names = ["temperature_celsius", "pressure_psi", "ph_level", "leak_rate_ppm"]
        result = chemical_detector.detect(data, parameter_names=param_names)

        assert "leak_rate_ppm" in result["anomalies"]

    def test_default_parameter_names(self, chemical_detector: Any) -> None:
        """Test detection with default parameter names."""
        data = np.array([[25.0, 500.0, 7.0, 10.0]])
        result = chemical_detector.detect(data)

        # Should use default param_0, param_1, etc.
        assert result["sector"] == "chemical"


class TestNuclearSectorDetection:
    """Tests for nuclear sector anomaly detection."""

    @pytest.fixture
    def nuclear_detector(self):
        """Create nuclear sector detector."""
        return ChemicalNuclearDetector(sector=CISASector.NUCLEAR)

    @pytest.fixture
    def normal_nuclear_data(self):
        """Create normal nuclear sensor data."""
        return np.array(
            [
                [2.0, 300.0, 100000.0, 1.0],  # radiation, core_temp, coolant_flow, neutron_flux
                [3.0, 310.0, 95000.0, 1.05],
                [2.5, 305.0, 98000.0, 0.95],
            ]
        )

    @pytest.fixture
    def anomalous_nuclear_data(self):
        """Create anomalous nuclear sensor data (radiation spike)."""
        return np.array(
            [
                [2.0, 300.0, 100000.0, 1.0],
                [10.0, 340.0, 45000.0, 1.5],  # Multiple violations
                [2.5, 305.0, 98000.0, 0.95],
                [15.0, 350.0, 40000.0, 1.6],  # More violations
            ]
        )

    def test_detect_normal_conditions(
        self, nuclear_detector: Any, normal_nuclear_data: Any
    ) -> None:
        """Test detection with normal nuclear conditions."""
        param_names = [
            "radiation_mrem_hr",
            "core_temperature_celsius",
            "coolant_flow_gpm",
            "neutron_flux",
        ]
        result = nuclear_detector.detect(normal_nuclear_data, parameter_names=param_names)

        assert result["safety_status"] == "NORMAL"
        assert result["sector"] == "nuclear_reactors_materials_waste"

    def test_detect_radiation_spike(
        self, nuclear_detector: Any, anomalous_nuclear_data: Any
    ) -> None:
        """Test detection of radiation threshold violation."""
        param_names = [
            "radiation_mrem_hr",
            "core_temperature_celsius",
            "coolant_flow_gpm",
            "neutron_flux",
        ]
        result = nuclear_detector.detect(anomalous_nuclear_data, parameter_names=param_names)

        assert result["safety_status"] in ["WARNING", "CRITICAL"]
        assert "radiation_mrem_hr" in result["anomalies"]

    def test_detect_coolant_flow_loss(self, nuclear_detector: Any) -> None:
        """Test detection of coolant flow loss (emergency condition)."""
        data = np.array(
            [
                [2.0, 300.0, 40000.0, 1.0],  # Low coolant flow
                [2.0, 310.0, 35000.0, 1.0],
                [2.0, 320.0, 30000.0, 1.0],
                [2.0, 330.0, 25000.0, 1.0],
            ]
        )
        param_names = [
            "radiation_mrem_hr",
            "core_temperature_celsius",
            "coolant_flow_gpm",
            "neutron_flux",
        ]
        result = nuclear_detector.detect(data, parameter_names=param_names)

        assert "coolant_flow_gpm" in result["anomalies"]
        assert bool(result["anomalies"]["coolant_flow_gpm"]["requires_emergency_response"]) is True

    def test_detect_core_temperature_high(self, nuclear_detector: Any) -> None:
        """Test detection of high core temperature."""
        data = np.array(
            [
                [2.0, 340.0, 100000.0, 1.0],  # Above 330°C threshold
            ]
        )
        param_names = [
            "radiation_mrem_hr",
            "core_temperature_celsius",
            "coolant_flow_gpm",
            "neutron_flux",
        ]
        result = nuclear_detector.detect(data, parameter_names=param_names)

        assert "core_temperature_celsius" in result["anomalies"]


class TestSeverityCalculation:
    """Tests for severity calculation."""

    @pytest.fixture
    def detector(self):
        """Create detector for testing."""
        return ChemicalNuclearDetector(sector=CISASector.CHEMICAL)

    def test_severity_none(self, detector: Any) -> None:
        """Test severity calculation with no violations."""
        severity = detector._calculate_severity(np.array([]), {"lower": 0, "upper": 100})
        assert severity == "NONE"

    def test_severity_low(self, detector: Any) -> None:
        """Test low severity calculation."""
        violations = np.array([105.0])  # 5% over 100 threshold
        severity = detector._calculate_severity(violations, {"lower": 0, "upper": 100})
        assert severity == "LOW"

    def test_severity_medium(self, detector: Any) -> None:
        """Test medium severity calculation."""
        violations = np.array([115.0])  # 15% over threshold
        severity = detector._calculate_severity(violations, {"lower": 0, "upper": 100})
        assert severity == "MEDIUM"

    def test_severity_high(self, detector: Any) -> None:
        """Test high severity calculation."""
        violations = np.array([130.0])  # 30% over threshold
        severity = detector._calculate_severity(violations, {"lower": 0, "upper": 100})
        assert severity == "HIGH"

    def test_severity_critical(self, detector: Any) -> None:
        """Test critical severity calculation."""
        violations = np.array([180.0])  # 80% over threshold
        severity = detector._calculate_severity(violations, {"lower": 0, "upper": 100})
        assert severity == "CRITICAL"


class TestCrossSectorImpact:
    """Tests for cross-sector impact assessment."""

    @pytest.fixture
    def detector(self):
        """Create detector for testing."""
        return ChemicalNuclearDetector(sector=CISASector.CHEMICAL)

    def test_no_impact_no_anomalies(self, detector: Any) -> None:
        """Test no impact when no anomalies."""
        impact = detector._assess_cross_sector_impact({})

        assert impact["affected_sectors"] == []
        assert impact["impact_level"] == "NONE"

    def test_medium_impact(self, detector: Any) -> None:
        """Test medium impact with non-critical anomalies."""
        anomalies = {"temperature_celsius": {"severity": "MEDIUM"}}
        impact = detector._assess_cross_sector_impact(anomalies)

        assert len(impact["affected_sectors"]) > 0
        assert impact["impact_level"] == "MEDIUM"

    def test_high_impact_critical_anomaly(self, detector: Any) -> None:
        """Test high impact with critical anomalies."""
        anomalies = {"leak_rate_ppm": {"severity": "CRITICAL"}}
        impact = detector._assess_cross_sector_impact(anomalies)

        assert impact["impact_level"] == "HIGH"
        assert impact["cascading_risk"] is True


class TestRecommendations:
    """Tests for recommendation generation."""

    @pytest.fixture
    def detector(self):
        """Create detector for testing."""
        return ChemicalNuclearDetector(sector=CISASector.NUCLEAR)

    def test_recommendations_no_anomalies(self, detector: Any) -> None:
        """Test recommendations when no anomalies."""
        recommendations = detector._generate_recommendations({})

        assert "Continue normal operations" in recommendations

    def test_recommendations_emergency(self, detector: Any) -> None:
        """Test recommendations for emergency conditions."""
        anomalies = {
            "radiation_mrem_hr": {
                "requires_emergency_response": True,
                "severity": "CRITICAL",
            }
        }
        recommendations = detector._generate_recommendations(anomalies)

        assert any("URGENT" in rec for rec in recommendations)
        assert any("regulatory" in rec.lower() for rec in recommendations)

    def test_recommendations_high_severity(self, detector: Any) -> None:
        """Test recommendations for high severity anomalies."""
        anomalies = {
            "core_temperature_celsius": {
                "requires_emergency_response": False,
                "severity": "HIGH",
            }
        }
        recommendations = detector._generate_recommendations(anomalies)

        assert any("Investigate" in rec for rec in recommendations)

    def test_recommendations_low_severity(self, detector: Any) -> None:
        """Test recommendations for low severity anomalies."""
        anomalies = {
            "neutron_flux": {
                "requires_emergency_response": False,
                "severity": "LOW",
            }
        }
        recommendations = detector._generate_recommendations(anomalies)

        assert any("Monitor" in rec for rec in recommendations)
