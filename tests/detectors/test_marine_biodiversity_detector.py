# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for Marine Biodiversity Detector."""

from __future__ import annotations

from typing import Any

import pytest

from omni_mercury_engine.core.exceptions import InvalidObservationError
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

    def test_hotspot_alone_is_not_bleaching(self, detector: Any) -> None:
        """A hot day without accumulated stress is a CRW Warning, not an alert.

        The retired rule fired on either signal alone; NOAA CRW requires
        HotSpot >= 1 degC AND DHW >= 4 for Alert Level 1 (reef-wide
        bleaching risk).
        """
        result = detector.detect_coral_bleaching(
            {"sst_anomaly_c": 1.5, "degree_heating_weeks": 2.0}
        )
        assert result["bleaching_detected"] is False
        assert result["alert_level"] == "warning"
        assert result["severity"] == "mild"

    def test_residual_dhw_with_subsided_hotspot_is_watch(self, detector: Any) -> None:
        """Accumulated DHW with the HotSpot back under 1 degC is a Watch.

        Per the CRW Bleaching Alert Area table every level above Watch
        requires the current HotSpot >= 1 degC.
        """
        result = detector.detect_coral_bleaching(
            {"sst_anomaly_c": 0.5, "degree_heating_weeks": 5.0}
        )
        assert result["bleaching_detected"] is False
        assert result["alert_level"] == "watch"

    @pytest.mark.parametrize(
        ("hotspot", "dhw", "alert_level", "detected", "severity"),
        [
            (-0.5, 0.0, "no_stress", False, "mild"),
            (0.5, 0.0, "watch", False, "mild"),
            (1.5, 3.9, "warning", False, "mild"),
            (1.5, 4.0, "alert_level_1", True, "moderate"),
            (1.5, 8.0, "alert_level_2", True, "severe"),
            (1.5, 12.0, "alert_level_3", True, "severe"),
            (1.5, 16.0, "alert_level_4", True, "severe"),
            (1.5, 20.0, "alert_level_5", True, "severe"),
        ],
    )
    def test_crw_alert_ladder(
        self,
        detector: Any,
        hotspot: float,
        dhw: float,
        alert_level: str,
        detected: bool,
        severity: str,
    ) -> None:
        """The full NOAA CRW Bleaching Alert Area table (Dec 2023)."""
        result = detector.detect_coral_bleaching(
            {"sst_anomaly_c": hotspot, "degree_heating_weeks": dhw}
        )
        assert result["alert_level"] == alert_level
        assert result["bleaching_detected"] is detected
        assert result["severity"] == severity

    def test_decision_source_is_published(self, detector: Any) -> None:
        """The bleaching rule cites NOAA Coral Reef Watch."""
        assert detector.decision_source.kind == "published"
        assert "coralreefwatch.noaa.gov" in detector.decision_source.citation

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
        # The one assessed indicator fires (was a constant 0.8).
        assert result.confidence == 1.0
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

    def test_anomaly_only_heatwave_is_a_labelled_screen(self, detector: Any) -> None:
        """Without local percentile climatology the verdict says so."""
        result = detector.predict_biodiversity_threat({"temperature_data": {"anomaly_c": 3.0}})
        assert result.marine_heatwave is True
        assert result.heatwave_basis == "screening_2c_unsourced"
        assert result.heatwave_category is None

    def test_hobday_definition_needs_five_days_at_p90(self, detector: Any) -> None:
        """Hobday et al. (2016): >= 5 days at/above the local 90th percentile."""
        base = {"anomaly_c": 1.6, "p90_anomaly_c": 1.5}
        short = detector.predict_biodiversity_threat(
            {"temperature_data": {**base, "days_at_or_above_p90": 4.0}}
        )
        assert short.marine_heatwave is False
        assert short.heatwave_basis == "hobday_2016"
        event = detector.predict_biodiversity_threat(
            {"temperature_data": {**base, "days_at_or_above_p90": 5.0}}
        )
        assert event.marine_heatwave is True
        assert event.heatwave_category == "moderate"

    @pytest.mark.parametrize(
        ("anomaly", "category"),
        [(1.5, "moderate"), (3.0, "strong"), (4.5, "severe"), (6.0, "extreme")],
    )
    def test_hobday_2018_categories(self, detector: Any, anomaly: float, category: str) -> None:
        """Categories are multiples of the climatology-to-p90 gap (Hobday 2018)."""
        result = detector.predict_biodiversity_threat(
            {
                "temperature_data": {
                    "anomaly_c": anomaly,
                    "p90_anomaly_c": 1.5,
                    "days_at_or_above_p90": 10.0,
                }
            }
        )
        assert result.marine_heatwave is True
        assert result.heatwave_category == category

    def test_acidification_uses_supplied_local_baseline(self, detector: Any) -> None:
        """EPA Gold Book departure (> 0.2 units) against the local baseline."""
        # Upwelling site with a naturally low baseline: 7.85 is within 0.2.
        normal = detector.predict_biodiversity_threat(
            {"chemistry_data": {"ph": 7.85, "baseline_ph": 8.0}}
        )
        assert normal.ocean_acidification is False
        assert normal.ph_baseline == 8.0
        event = detector.predict_biodiversity_threat(
            {"chemistry_data": {"ph": 7.79, "baseline_ph": 8.0}}
        )
        assert event.ocean_acidification is True

    def test_acidification_default_baseline_is_global_mean(self, detector: Any) -> None:
        """Without baseline_ph the Jiang et al. (2019) global mean applies."""
        result = detector.predict_biodiversity_threat({"chemistry_data": {"ph": 7.86}})
        assert result.ph_baseline == pytest.approx(8.07)
        assert result.ocean_acidification is True  # 7.86 < 8.07 - 0.2
        ok = detector.predict_biodiversity_threat({"chemistry_data": {"ph": 7.88}})
        assert ok.ocean_acidification is False  # 7.88 >= 7.87

    def test_health_status_levels(self, detector: Any) -> None:
        """Test different health status determinations."""
        # Healthy - measured, no threats
        result = detector.predict_biodiversity_threat({"chemistry_data": {"ph": 8.1}})
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
        """Empty marine data is refused: it used to grade as a healthy ecosystem."""
        with pytest.raises(InvalidObservationError, match="no indicator measured"):
            detector.predict_biodiversity_threat({})

    @pytest.mark.parametrize(
        "data",
        [
            {"coral_data": {}},  # coral read as 0 degC / 0 DHW: a measured all-clear
            {"coral_data": {"sst_anomaly_c": 0.2}},  # DHW defaulted to 0
            {"chemistry_data": {"pH": 7.5}},  # misspelt: acidification unassessed
            {"temperature_data": {"anomaly": 3.0}},  # misspelt: heatwave unassessed
        ],
    )
    def test_incomplete_or_misspelt_section_is_refused(self, detector: Any, data: Any) -> None:
        with pytest.raises(InvalidObservationError, match=r"missing|unknown keys"):
            detector.predict_biodiversity_threat(data)

    @pytest.mark.parametrize(
        ("section", "key"),
        [("chemistry_data", "ph"), ("temperature_data", "anomaly_c")],
    )
    def test_non_finite_indicator_is_refused(self, detector: Any, section: str, key: str) -> None:
        """A NaN failed every threshold comparison and read as healthy."""
        with pytest.raises(InvalidObservationError):
            detector.predict_biodiversity_threat({section: {key: float("nan")}})

    def test_chemistry_without_ph_reports_no_fabricated_ph(self, detector: Any) -> None:
        """A chemistry section without pH used to report pH 8.1 nobody measured."""
        with pytest.raises(InvalidObservationError, match="missing"):
            detector.predict_biodiversity_threat({"chemistry_data": {}})

    def test_status_and_decision_agree(self, detector: Any) -> None:
        """Acidification plus a heatwave was 'critical' yet not threatened."""
        result = detector.predict_biodiversity_threat(
            {"chemistry_data": {"ph": 7.7}, "temperature_data": {"anomaly_c": 3.0}}
        )
        assert result.health_status == "critical"
        assert result.ecosystem_threatened is True
        assert result.confidence == 1.0

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
