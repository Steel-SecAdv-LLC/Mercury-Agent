# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for Marine Biodiversity Detector."""

from __future__ import annotations

from typing import Any

import pytest

from omni_mercury_engine.detectors.marine.biodiversity_detector import (
    BiodiversityPredictionResult,
    CoralBleachingDetector,
    EcosystemHealth,
    MarineBiodiversityDetector,
)


class TestEcosystemHealthEnum:
    """Tests for EcosystemHealth enumeration."""

    def test_ecosystem_health_values(self) -> None:
        """Test EcosystemHealth enum values."""
        assert EcosystemHealth.THRIVING.value == "thriving"
        assert EcosystemHealth.HEALTHY.value == "healthy"
        assert EcosystemHealth.STRESSED.value == "stressed"
        assert EcosystemHealth.DEGRADED.value == "degraded"
        assert EcosystemHealth.CRITICAL.value == "critical"
        assert EcosystemHealth.COLLAPSED.value == "collapsed"


class TestBiodiversityPredictionResult:
    """Tests for BiodiversityPredictionResult dataclass."""

    def test_default_values(self) -> None:
        """Test default values of result dataclass."""
        result = BiodiversityPredictionResult(
            ecosystem_threatened=False,
            confidence=0.0,
            health_status="healthy",
        )
        assert result.species_decline_detected is False
        assert result.coral_bleaching_detected is False
        assert result.ocean_acidification is False
        assert result.marine_heatwave is False
        assert result.threatened_species == []
        assert result.conservation_actions == []


class TestCoralBleachingDetector:
    """Tests for CoralBleachingDetector."""

    @pytest.fixture
    def detector(self):
        """Create CoralBleachingDetector instance."""
        return CoralBleachingDetector()

    def test_no_bleaching_normal_conditions(self, detector: Any) -> None:
        """Test detection with normal coral conditions."""
        coral_data = {
            "sst_anomaly_c": 0.5,
            "degree_heating_weeks": 1.0,
        }
        result = detector.detect_coral_bleaching(coral_data)

        assert result["bleaching_detected"] is False
        assert result["severity"] == "mild"

    def test_bleaching_detected_high_temp_anomaly(self, detector: Any) -> None:
        """Test bleaching detection with high temperature anomaly."""
        coral_data = {
            "sst_anomaly_c": 1.5,
            "degree_heating_weeks": 2.0,
        }
        result = detector.detect_coral_bleaching(coral_data)

        assert result["bleaching_detected"] is True
        assert result["severity"] == "mild"

    def test_bleaching_detected_high_dhw(self, detector: Any) -> None:
        """Test bleaching detection with high degree heating weeks."""
        coral_data = {
            "sst_anomaly_c": 0.5,
            "degree_heating_weeks": 5.0,
        }
        result = detector.detect_coral_bleaching(coral_data)

        assert result["bleaching_detected"] is True
        assert result["severity"] == "severe"

    def test_bleaching_severity_levels(self, detector: Any) -> None:
        """Test different bleaching severity levels."""
        # Mild severity
        result = detector.detect_coral_bleaching(
            {
                "sst_anomaly_c": 1.5,
                "degree_heating_weeks": 1.0,
            }
        )
        assert result["severity"] == "mild"

        # Moderate severity
        result = detector.detect_coral_bleaching(
            {
                "sst_anomaly_c": 1.5,
                "degree_heating_weeks": 3.0,
            }
        )
        assert result["severity"] == "moderate"

        # Severe severity
        result = detector.detect_coral_bleaching(
            {
                "sst_anomaly_c": 1.5,
                "degree_heating_weeks": 5.0,
            }
        )
        assert result["severity"] == "severe"

    def test_dhw_captured_in_result(self, detector: Any) -> None:
        """Test that degree heating weeks is captured in result."""
        coral_data = {
            "sst_anomaly_c": 0.5,
            "degree_heating_weeks": 3.5,
        }
        result = detector.detect_coral_bleaching(coral_data)

        assert result["degree_heating_weeks"] == 3.5


class TestMarineBiodiversityDetector:
    """Tests for comprehensive MarineBiodiversityDetector."""

    @pytest.fixture
    def detector(self):
        """Create MarineBiodiversityDetector instance."""
        return MarineBiodiversityDetector()

    @pytest.fixture
    def healthy_marine_data(self):
        """Create healthy marine ecosystem data."""
        return {
            "coral_data": {
                "sst_anomaly_c": 0.3,
                "degree_heating_weeks": 1.0,
            },
            "chemistry_data": {"ph": 8.1},
            "temperature_data": {"anomaly_c": 0.5},
        }

    @pytest.fixture
    def stressed_marine_data(self):
        """Create stressed marine ecosystem data."""
        return {
            "coral_data": {
                "sst_anomaly_c": 1.5,
                "degree_heating_weeks": 5.0,
            },
            "chemistry_data": {"ph": 7.8},
            "temperature_data": {"anomaly_c": 2.5},
        }

    def test_initialization(self, detector: Any) -> None:
        """Test detector initialization."""
        assert detector.coral_detector is not None

    def test_predict_healthy_ecosystem(self, detector: Any, healthy_marine_data: Any) -> None:
        """Test prediction for healthy ecosystem."""
        result = detector.predict_biodiversity_threat(healthy_marine_data)

        assert isinstance(result, BiodiversityPredictionResult)
        assert result.ecosystem_threatened is False
        assert result.health_status == "healthy"
        assert result.coral_bleaching_detected is False
        assert result.ocean_acidification is False
        assert result.marine_heatwave is False

    def test_predict_stressed_ecosystem(self, detector: Any, stressed_marine_data: Any) -> None:
        """Test prediction for stressed ecosystem."""
        result = detector.predict_biodiversity_threat(stressed_marine_data)

        assert result.ecosystem_threatened is True
        assert result.coral_bleaching_detected is True
        assert result.ocean_acidification is True
        assert result.marine_heatwave is True
        assert result.health_status == "critical"

    def test_coral_bleaching_integration(self, detector: Any) -> None:
        """Test coral bleaching detection integration."""
        data = {
            "coral_data": {
                "sst_anomaly_c": 1.5,
                "degree_heating_weeks": 5.0,
            }
        }
        result = detector.predict_biodiversity_threat(data)

        assert result.coral_bleaching_detected is True
        assert result.confidence == 0.8
        assert result.ecosystem_threatened is True

    def test_ocean_acidification_detection(self, detector: Any) -> None:
        """Test ocean acidification detection."""
        data = {
            "chemistry_data": {"ph": 7.7},
        }
        result = detector.predict_biodiversity_threat(data)

        assert result.ocean_acidification is True
        assert result.ph_level == 7.7

    def test_ocean_acidification_normal_ph(self, detector: Any) -> None:
        """Test no acidification with normal pH."""
        data = {
            "chemistry_data": {"ph": 8.1},
        }
        result = detector.predict_biodiversity_threat(data)

        assert result.ocean_acidification is False
        assert result.ph_level == 8.1

    def test_marine_heatwave_detection(self, detector: Any) -> None:
        """Test marine heatwave detection."""
        data = {
            "temperature_data": {"anomaly_c": 3.0},
        }
        result = detector.predict_biodiversity_threat(data)

        assert result.marine_heatwave is True
        assert result.temperature_anomaly_c == 3.0

    def test_marine_heatwave_normal_temp(self, detector: Any) -> None:
        """Test no heatwave with normal temperature."""
        data = {
            "temperature_data": {"anomaly_c": 1.0},
        }
        result = detector.predict_biodiversity_threat(data)

        assert result.marine_heatwave is False

    def test_health_status_levels(self, detector: Any) -> None:
        """Test different health status determinations."""
        # Healthy - no threats
        result = detector.predict_biodiversity_threat({})
        assert result.health_status == "healthy"

        # Stressed - one threat
        result = detector.predict_biodiversity_threat({"chemistry_data": {"ph": 7.7}})
        assert result.health_status == "stressed"

        # Critical - multiple threats
        result = detector.predict_biodiversity_threat(
            {
                "chemistry_data": {"ph": 7.7},
                "temperature_data": {"anomaly_c": 3.0},
            }
        )
        assert result.health_status == "critical"

    def test_conservation_actions_generated(self, detector: Any, stressed_marine_data: Any) -> None:
        """Test that conservation actions are generated."""
        result = detector.predict_biodiversity_threat(stressed_marine_data)

        assert len(result.conservation_actions) > 0

    def test_conservation_actions_for_bleaching(self, detector: Any) -> None:
        """Test conservation actions for coral bleaching."""
        data = {
            "coral_data": {
                "sst_anomaly_c": 1.5,
                "degree_heating_weeks": 5.0,
            }
        }
        result = detector.predict_biodiversity_threat(data)

        assert any("reef" in action.lower() for action in result.conservation_actions)

    def test_conservation_actions_for_acidification(self, detector: Any) -> None:
        """Test conservation actions for ocean acidification."""
        data = {"chemistry_data": {"ph": 7.7}}
        result = detector.predict_biodiversity_threat(data)

        assert any("CO2" in action for action in result.conservation_actions)

    def test_empty_data_handling(self, detector: Any) -> None:
        """Test handling of empty marine data."""
        result = detector.predict_biodiversity_threat({})

        assert result.ecosystem_threatened is False
        assert result.health_status == "healthy"
        assert result.coral_bleaching_detected is False
        assert result.ocean_acidification is False
        assert result.marine_heatwave is False

    def test_partial_data_handling(self, detector: Any) -> None:
        """Test handling of partial marine data."""
        # Only coral data provided
        result = detector.predict_biodiversity_threat(
            {
                "coral_data": {
                    "sst_anomaly_c": 1.5,
                    "degree_heating_weeks": 5.0,
                }
            }
        )

        assert result.coral_bleaching_detected is True
        assert result.ph_level is None
        assert result.temperature_anomaly_c is None
