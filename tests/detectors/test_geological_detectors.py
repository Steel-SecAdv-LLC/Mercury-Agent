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

import pytest

pytest.importorskip("torch")

"""Tests for Geological Disaster Detectors (Tornado, Hurricane, Flood)."""

import numpy as np
import pytest

from omni_mercury_engine.detectors.geological.flood_detector import (
    FloodDetector,
    FloodPredictionResult,
    FloodSeverity,
    FloodType,
)
from omni_mercury_engine.detectors.geological.hurricane_detector import (
    CycloneType,
    HurricaneDetector,
    HurricanePredictionResult,
    SaffirSimpsonCategory,
)
from omni_mercury_engine.detectors.geological.tornado_detector import (
    TornadoDetector,
    TornadoIntensity,
    TornadoPredictionResult,
    TornadoThreatLevel,
)


class TestTornadoIntensityEnum:
    """Tests for TornadoIntensity enumeration."""

    def test_intensity_values(self):
        """Test TornadoIntensity enum values."""
        assert TornadoIntensity.EF0.value == "ef0_weak"
        assert TornadoIntensity.EF1.value == "ef1_moderate"
        assert TornadoIntensity.EF2.value == "ef2_significant"
        assert TornadoIntensity.EF3.value == "ef3_severe"
        assert TornadoIntensity.EF4.value == "ef4_devastating"
        assert TornadoIntensity.EF5.value == "ef5_incredible"
        assert TornadoIntensity.NO_TORNADO.value == "no_tornado"


class TestTornadoThreatLevelEnum:
    """Tests for TornadoThreatLevel enumeration."""

    def test_threat_level_values(self):
        """Test TornadoThreatLevel enum values."""
        assert TornadoThreatLevel.NONE.value == "none"
        assert TornadoThreatLevel.MARGINAL.value == "marginal"
        assert TornadoThreatLevel.SLIGHT.value == "slight"
        assert TornadoThreatLevel.ENHANCED.value == "enhanced"
        assert TornadoThreatLevel.MODERATE.value == "moderate"
        assert TornadoThreatLevel.HIGH.value == "high"


class TestTornadoPredictionResult:
    """Tests for TornadoPredictionResult dataclass."""

    def test_default_values(self):
        """Test default values of result dataclass."""
        result = TornadoPredictionResult(
            tornado_likely=False,
            confidence=0.0,
            threat_level="none",
            estimated_intensity="no_tornado",
        )
        assert result.tornado_likely is False
        assert result.confidence == 0.0
        assert result.threat_level == "none"
        assert result.estimated_intensity == "no_tornado"

    def test_full_initialization(self):
        """Test full initialization with all fields."""
        result = TornadoPredictionResult(
            tornado_likely=True,
            confidence=0.95,
            threat_level="high",
            estimated_intensity="ef4_devastating",
            mesocyclone_detected=True,
            rotation_velocity_ms=30.0,
            pressure_drop_mb=10.0,
            cape_value=3500.0,
            helicity_value=200.0,
            wind_shear_detected=True,
            resonance_score=0.9,
            recursion_depth=3,
            harmonic_anomalies=[0.1, 0.5, 1.0],
            tornado_alley_correlation=0.85,
            time_to_touchdown_minutes=15.0,
            warning_actions=["Seek shelter immediately"],
            shelter_recommendations=["Go to basement"],
        )
        assert result.tornado_likely is True
        assert result.confidence == 0.95
        assert result.mesocyclone_detected is True
        assert result.cape_value == 3500.0
        assert len(result.warning_actions) == 1


class TestTornadoDetector:
    """Tests for TornadoDetector."""

    @pytest.fixture
    def detector(self):
        """Create TornadoDetector instance."""
        return TornadoDetector()

    @pytest.fixture
    def normal_weather_data(self):
        """Create normal weather conditions data."""
        return {
            "atmospheric_data": {
                "cape_j_kg": 500.0,
                "cin_j_kg": -50.0,
                "srh_m2_s2": 50.0,
                "bulk_shear_kt": 20.0,
                "lcl_m": 1500.0,
            },
            "pressure_data": {
                "pressure_mb": np.array([1013.0, 1013.1, 1013.0, 1012.9]),
            },
        }

    @pytest.fixture
    def tornado_conditions_data(self):
        """Create tornado-favorable conditions data."""
        radar_seq = np.random.randn(1, 10, 64).astype(np.float32)
        return {
            "radar_sequence": radar_seq,
            "atmospheric_data": {
                "cape_j_kg": 3500.0,
                "cin_j_kg": -20.0,
                "srh_m2_s2": 250.0,
                "bulk_shear_kt": 50.0,
                "lcl_m": 800.0,
            },
            "pressure_data": {
                "pressure_mb": np.array([1010.0, 1008.0, 1005.0, 1000.0]),
            },
            "location": {"state": "OK"},
        }

    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector is not None
        assert hasattr(detector, "predict_tornado")

    def test_predict_normal_conditions(self, detector, normal_weather_data):
        """Test prediction with normal weather conditions."""
        result = detector.predict_tornado(normal_weather_data)

        assert isinstance(result, TornadoPredictionResult)
        assert result.tornado_likely is False

    def test_predict_tornado_conditions(self, detector, tornado_conditions_data):
        """Test prediction with tornado-favorable conditions."""
        result = detector.predict_tornado(tornado_conditions_data)

        assert isinstance(result, TornadoPredictionResult)
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0

    def test_empty_data_handling(self, detector):
        """Test handling of empty data."""
        result = detector.predict_tornado({})

        assert isinstance(result, TornadoPredictionResult)
        assert result.tornado_likely is False


class TestSaffirSimpsonCategoryEnum:
    """Tests for SaffirSimpsonCategory enumeration."""

    def test_category_values(self):
        """Test SaffirSimpsonCategory enum values."""
        assert SaffirSimpsonCategory.TROPICAL_DEPRESSION.value == "tropical_depression"
        assert SaffirSimpsonCategory.TROPICAL_STORM.value == "tropical_storm"
        assert SaffirSimpsonCategory.CATEGORY_1.value == "category_1"
        assert SaffirSimpsonCategory.CATEGORY_2.value == "category_2"
        assert SaffirSimpsonCategory.CATEGORY_3.value == "category_3"
        assert SaffirSimpsonCategory.CATEGORY_4.value == "category_4"
        assert SaffirSimpsonCategory.CATEGORY_5.value == "category_5"
        assert SaffirSimpsonCategory.NO_CYCLONE.value == "no_cyclone"


class TestCycloneTypeEnum:
    """Tests for CycloneType enumeration."""

    def test_cyclone_type_values(self):
        """Test CycloneType enum values."""
        assert CycloneType.HURRICANE.value == "hurricane"
        assert CycloneType.TYPHOON.value == "typhoon"
        assert CycloneType.CYCLONE.value == "cyclone"
        assert CycloneType.NO_CYCLONE.value == "no_cyclone"


class TestHurricanePredictionResult:
    """Tests for HurricanePredictionResult dataclass."""

    def test_default_values(self):
        """Test default values of result dataclass."""
        result = HurricanePredictionResult(
            cyclone_detected=False,
            confidence=0.0,
            category="no_cyclone",
            cyclone_type="no_cyclone",
        )
        assert result.cyclone_detected is False
        assert result.confidence == 0.0
        assert result.category == "no_cyclone"

    def test_full_initialization(self):
        """Test full initialization with all fields."""
        result = HurricanePredictionResult(
            cyclone_detected=True,
            confidence=0.92,
            category="category_4",
            cyclone_type="hurricane",
            max_wind_speed_kt=130.0,
            min_pressure_mb=940.0,
            eye_diameter_nm=25.0,
            sst_anomaly_c=2.5,
            wind_shear_kt=10.0,
            ocean_heat_content=80.0,
            rapid_intensification=True,
            intensification_rate_kt_24h=35.0,
            resonance_score=0.85,
            frequency_amplification=1.5,
            harmonic_patterns=[0.05, 0.1, 0.2],
            storm_surge_risk="high",
            rainfall_potential_inches=15.0,
            landfall_probability=0.75,
            warning_actions=["Evacuate coastal areas"],
            evacuation_zones=["Zone A", "Zone B"],
        )
        assert result.cyclone_detected is True
        assert result.max_wind_speed_kt == 130.0
        assert len(result.evacuation_zones) == 2


class TestHurricaneDetector:
    """Tests for HurricaneDetector."""

    @pytest.fixture
    def detector(self):
        """Create HurricaneDetector instance."""
        return HurricaneDetector()

    @pytest.fixture
    def normal_ocean_data(self):
        """Create normal ocean conditions data."""
        return {
            "sst_data": {
                "sst_celsius": 25.0,
                "climatology_celsius": 26.0,
                "depth_26c_m": 30.0,
            },
            "pressure_data": {
                "central_pressure_mb": 1010.0,
                "environmental_pressure_mb": 1013.0,
            },
            "basin": "atlantic",
        }

    @pytest.fixture
    def hurricane_conditions_data(self):
        """Create hurricane-favorable conditions data."""
        return {
            "sst_data": {
                "sst_celsius": 29.5,
                "climatology_celsius": 27.0,
                "depth_26c_m": 80.0,
            },
            "pressure_data": {
                "central_pressure_mb": 945.0,
                "environmental_pressure_mb": 1013.0,
                "pressure_history_mb": [970.0, 960.0, 950.0, 945.0],
            },
            "signal_data": np.random.randn(100),
            "basin": "atlantic",
        }

    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector is not None
        assert hasattr(detector, "predict_hurricane")

    def test_predict_normal_conditions(self, detector, normal_ocean_data):
        """Test prediction with normal ocean conditions."""
        result = detector.predict_hurricane(normal_ocean_data)

        assert isinstance(result, HurricanePredictionResult)
        assert result.cyclone_detected is False

    def test_predict_hurricane_conditions(self, detector, hurricane_conditions_data):
        """Test prediction with hurricane conditions."""
        result = detector.predict_hurricane(hurricane_conditions_data)

        assert isinstance(result, HurricanePredictionResult)
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0

    def test_empty_data_handling(self, detector):
        """Test handling of empty data."""
        result = detector.predict_hurricane({})

        assert isinstance(result, HurricanePredictionResult)
        assert result.cyclone_detected is False

    def test_saffir_simpson_classification(self, detector):
        """Test Saffir-Simpson scale classification."""
        test_cases = [
            (20, "no_cyclone"),
            (30, "tropical_depression"),
            (50, "tropical_storm"),
            (80, "category_1"),
            (100, "category_2"),
            (115, "category_3"),
            (140, "category_4"),
            (160, "category_5"),
        ]
        for wind_speed, expected_category in test_cases:
            category = detector._classify_category(wind_speed)
            assert category == expected_category

    def test_cyclone_type_detection(self, detector):
        """Test cyclone type detection based on basin."""
        assert detector._determine_cyclone_type("atlantic") == "hurricane"
        assert detector._determine_cyclone_type("eastern_pacific") == "hurricane"
        assert detector._determine_cyclone_type("western_pacific") == "typhoon"
        assert detector._determine_cyclone_type("indian") == "cyclone"


class TestFloodSeverityEnum:
    """Tests for FloodSeverity enumeration."""

    def test_severity_values(self):
        """Test FloodSeverity enum values."""
        assert FloodSeverity.MINOR.value == "minor"
        assert FloodSeverity.MODERATE.value == "moderate"
        assert FloodSeverity.MAJOR.value == "major"
        assert FloodSeverity.RECORD.value == "record"
        assert FloodSeverity.NO_FLOOD.value == "no_flood"


class TestFloodTypeEnum:
    """Tests for FloodType enumeration."""

    def test_flood_type_values(self):
        """Test FloodType enum values."""
        assert FloodType.FLASH.value == "flash"
        assert FloodType.RIVER.value == "river"
        assert FloodType.COASTAL.value == "coastal"
        assert FloodType.URBAN.value == "urban"
        assert FloodType.DAM_FAILURE.value == "dam_failure"
        assert FloodType.GROUNDWATER.value == "groundwater"
        assert FloodType.NO_FLOOD.value == "no_flood"


class TestFloodPredictionResult:
    """Tests for FloodPredictionResult dataclass."""

    def test_default_values(self):
        """Test default values of result dataclass."""
        result = FloodPredictionResult(
            flood_likely=False,
            confidence=0.0,
            severity="no_flood",
            flood_type="no_flood",
        )
        assert result.flood_likely is False
        assert result.confidence == 0.0
        assert result.severity == "no_flood"

    def test_full_initialization(self):
        """Test full initialization with all fields."""
        result = FloodPredictionResult(
            flood_likely=True,
            confidence=0.88,
            severity="major",
            flood_type="flash",
            river_stage_ft=20.0,
            flood_stage_ft=15.0,
            stage_trend="rising_rapidly",
            precipitation_24h_inches=8.5,
            precipitation_forecast_inches=4.0,
            soil_saturation_pct=95.0,
            runoff_coefficient=0.8,
            time_to_peak_hours=6.0,
            peak_discharge_cfs=50000.0,
            refactoring_score=0.9,
            model_optimization_iterations=5,
            prediction_uncertainty=0.15,
            affected_area_sq_mi=100.0,
            population_at_risk=50000,
            warning_actions=["Evacuate low-lying areas"],
            evacuation_routes=["Route 1", "Route 2"],
            shelter_locations=["Community Center"],
        )
        assert result.flood_likely is True
        assert result.river_stage_ft == 20.0
        assert len(result.evacuation_routes) == 2


class TestFloodDetector:
    """Tests for FloodDetector."""

    @pytest.fixture
    def detector(self):
        """Create FloodDetector instance."""
        return FloodDetector()

    @pytest.fixture
    def normal_conditions_data(self):
        """Create normal hydrological conditions data."""
        return {
            "precip_data": {
                "precipitation_1h_inches": 0.1,
                "precipitation_6h_inches": 0.5,
                "precipitation_24h_inches": 1.0,
                "forecast_24h_inches": 0.5,
            },
            "gauge_data": {
                "current_stage_ft": 5.0,
                "action_stage_ft": 10.0,
                "flood_stage_ft": 15.0,
                "moderate_flood_stage_ft": 20.0,
                "major_flood_stage_ft": 25.0,
            },
            "soil_data": {
                "soil_moisture_pct": 40.0,
                "soil_type": "loam",
            },
        }

    @pytest.fixture
    def flood_conditions_data(self):
        """Create flood-favorable conditions data."""
        return {
            "precip_data": {
                "precipitation_1h_inches": 3.0,
                "precipitation_6h_inches": 8.0,
                "precipitation_24h_inches": 12.0,
                "forecast_24h_inches": 4.0,
            },
            "gauge_data": {
                "current_stage_ft": 22.0,
                "action_stage_ft": 10.0,
                "flood_stage_ft": 15.0,
                "moderate_flood_stage_ft": 20.0,
                "major_flood_stage_ft": 25.0,
                "stage_history_ft": [15.0, 17.0, 19.0, 22.0],
            },
            "soil_data": {
                "soil_moisture_pct": 95.0,
                "soil_type": "clay",
            },
        }

    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector is not None
        assert hasattr(detector, "predict_flood")

    def test_predict_normal_conditions(self, detector, normal_conditions_data):
        """Test prediction with normal conditions."""
        result = detector.predict_flood(normal_conditions_data)

        assert isinstance(result, FloodPredictionResult)
        assert result.flood_likely is False

    def test_predict_flood_conditions(self, detector, flood_conditions_data):
        """Test prediction with flood conditions."""
        result = detector.predict_flood(flood_conditions_data)

        assert isinstance(result, FloodPredictionResult)
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0

    def test_empty_data_handling(self, detector):
        """Test handling of empty data."""
        result = detector.predict_flood({})

        assert isinstance(result, FloodPredictionResult)
        assert result.flood_likely is False


class TestCrossDomainFusion:
    """Tests for cross-domain fusion between disaster detectors."""

    @pytest.fixture
    def tornado_detector(self):
        """Create TornadoDetector instance."""
        return TornadoDetector()

    @pytest.fixture
    def hurricane_detector(self):
        """Create HurricaneDetector instance."""
        return HurricaneDetector()

    @pytest.fixture
    def flood_detector(self):
        """Create FloodDetector instance."""
        return FloodDetector()

    def test_hurricane_flood_correlation(self, hurricane_detector, flood_detector):
        """Test correlation between hurricane and flood predictions."""
        hurricane_data = {
            "sst_data": {"sst_celsius": 29.0, "climatology_celsius": 27.0},
            "pressure_data": {"central_pressure_mb": 960.0, "environmental_pressure_mb": 1013.0},
            "basin": "atlantic",
        }
        flood_data = {
            "precip_data": {"precipitation_24h_inches": 10.0, "forecast_24h_inches": 5.0},
            "gauge_data": {"current_stage_ft": 20.0, "flood_stage_ft": 15.0},
        }

        hurricane_result = hurricane_detector.predict_hurricane(hurricane_data)
        flood_result = flood_detector.predict_flood(flood_data)

        assert isinstance(hurricane_result, HurricanePredictionResult)
        assert isinstance(flood_result, FloodPredictionResult)

    def test_tornado_outbreak_detection(self, tornado_detector):
        """Test detection of tornado outbreak conditions."""
        outbreak_data = {
            "atmospheric_data": {
                "cape_j_kg": 4000.0,
                "srh_m2_s2": 300.0,
                "bulk_shear_kt": 60.0,
                "lcl_m": 500.0,
            },
        }

        result = tornado_detector.predict_tornado(outbreak_data)
        assert isinstance(result, TornadoPredictionResult)


class TestDetectorIntegration:
    """Integration tests for geological detectors."""

    def test_all_detectors_instantiate(self):
        """Test that all detectors can be instantiated."""
        tornado = TornadoDetector()
        hurricane = HurricaneDetector()
        flood = FloodDetector()

        assert tornado is not None
        assert hurricane is not None
        assert flood is not None

    def test_all_detectors_predict_empty(self):
        """Test that all detectors handle empty data gracefully."""
        tornado = TornadoDetector()
        hurricane = HurricaneDetector()
        flood = FloodDetector()

        tornado_result = tornado.predict_tornado({})
        hurricane_result = hurricane.predict_hurricane({})
        flood_result = flood.predict_flood({})

        assert tornado_result.tornado_likely is False
        assert hurricane_result.cyclone_detected is False
        assert flood_result.flood_likely is False

    def test_detector_confidence_bounds(self):
        """Test that all detector confidences are within [0, 1]."""
        tornado = TornadoDetector()
        hurricane = HurricaneDetector()
        flood = FloodDetector()

        for _ in range(10):
            tornado_result = tornado.predict_tornado({})
            hurricane_result = hurricane.predict_hurricane({})
            flood_result = flood.predict_flood({})

            assert 0.0 <= tornado_result.confidence <= 1.0
            assert 0.0 <= hurricane_result.confidence <= 1.0
            assert 0.0 <= flood_result.confidence <= 1.0
