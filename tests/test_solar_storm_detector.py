"""
Tests for Solar Storm Detector.

Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC
"""

import pytest

pytest.importorskip("torch")

import numpy as np
import torch

from omni_mercury_engine.space.solar_storm_detector import (
    CMETracker,
    GeomagneticStormPredictor,
    GeostormScale,
    SolarFlareClass,
    SolarFlareDetector,
    SolarStormDetector,
    SolarStormPredictionResult,
)


class TestSolarFlareClass:
    """Tests for SolarFlareClass enum."""

    def test_enum_values(self) -> None:
        """Test enum values exist."""
        assert SolarFlareClass.A.value == "A"
        assert SolarFlareClass.B.value == "B"
        assert SolarFlareClass.C.value == "C"
        assert SolarFlareClass.M.value == "M"
        assert SolarFlareClass.X.value == "X"


class TestGeostormScale:
    """Tests for GeostormScale enum."""

    def test_enum_values(self) -> None:
        """Test enum values exist."""
        assert GeostormScale.G0.value == "none"
        assert GeostormScale.G1.value == "minor"
        assert GeostormScale.G2.value == "moderate"
        assert GeostormScale.G3.value == "strong"
        assert GeostormScale.G4.value == "severe"
        assert GeostormScale.G5.value == "extreme"


class TestSolarStormPredictionResult:
    """Tests for SolarStormPredictionResult dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        result = SolarStormPredictionResult(
            solar_storm_imminent=False,
            confidence=0.5,
            storm_severity="G0",
        )
        assert result.solar_storm_imminent is False
        assert result.confidence == 0.5
        assert result.flare_detected is False
        assert result.cme_detected is False
        assert result.power_grid_risk == "low"

    def test_custom_values(self) -> None:
        """Test custom values."""
        result = SolarStormPredictionResult(
            solar_storm_imminent=True,
            confidence=0.9,
            storm_severity="G4",
            flare_detected=True,
            flare_class="X",
            cme_detected=True,
            cme_speed_km_s=1500.0,
        )
        assert result.solar_storm_imminent is True
        assert result.flare_class == "X"
        assert result.cme_speed_km_s == 1500.0


class TestSolarFlareDetector:
    """Tests for SolarFlareDetector class."""

    def test_init(self) -> None:
        """Test initialization."""
        detector = SolarFlareDetector()
        assert detector is not None

    def test_detect_no_flare(self) -> None:
        """Test detection with no flare (A-class)."""
        detector = SolarFlareDetector()
        xray_data = {"flux_short_wm2": 1e-9, "flux_long_wm2": 1e-9}
        result = detector.detect_solar_flare(xray_data)
        assert result["flare_detected"] is False
        assert result["flare_class"] == "A"
        assert result["severity"] == "low"

    def test_detect_b_class_flare(self) -> None:
        """Test B-class flare detection."""
        detector = SolarFlareDetector()
        xray_data = {"flux_short_wm2": 5e-7, "flux_long_wm2": 1e-9}
        result = detector.detect_solar_flare(xray_data)
        assert result["flare_class"] == "B"
        assert result["flare_detected"] is False

    def test_detect_c_class_flare(self) -> None:
        """Test C-class flare detection."""
        detector = SolarFlareDetector()
        xray_data = {"flux_short_wm2": 5e-6, "flux_long_wm2": 1e-9}
        result = detector.detect_solar_flare(xray_data)
        assert result["flare_detected"] is True
        assert result["flare_class"] == "C"
        assert result["severity"] == "moderate"

    def test_detect_m_class_flare(self) -> None:
        """Test M-class flare detection."""
        detector = SolarFlareDetector()
        xray_data = {"flux_short_wm2": 5e-5, "flux_long_wm2": 1e-9}
        result = detector.detect_solar_flare(xray_data)
        assert result["flare_detected"] is True
        assert result["flare_class"] == "M"
        assert result["severity"] == "high"

    def test_detect_x_class_flare(self) -> None:
        """Test X-class flare detection."""
        detector = SolarFlareDetector()
        xray_data = {"flux_short_wm2": 5e-4, "flux_long_wm2": 1e-9}
        result = detector.detect_solar_flare(xray_data)
        assert result["flare_detected"] is True
        assert result["flare_class"] == "X"
        assert result["severity"] == "extreme"

    def test_classify_flare(self) -> None:
        """Test flare classification."""
        detector = SolarFlareDetector()
        assert detector._classify_flare(1e-4)[0] == "X"
        assert detector._classify_flare(1e-5)[0] == "M"
        assert detector._classify_flare(1e-6)[0] == "C"
        assert detector._classify_flare(1e-7)[0] == "B"
        assert detector._classify_flare(1e-9)[0] == "A"


class TestCMETracker:
    """Tests for CMETracker class."""

    def test_init(self) -> None:
        """Test initialization."""
        tracker = CMETracker()
        assert tracker is not None

    def test_track_no_cme(self) -> None:
        """Test tracking with no Earth-directed CME."""
        tracker = CMETracker()
        cme_data = {
            "speed_km_s": 200,
            "angular_width_deg": 30,
            "direction_longitude_deg": 90,
            "direction_latitude_deg": 0,
        }
        result = tracker.track_cme(cme_data)
        assert not result["cme_detected"]

    def test_track_earth_directed_cme(self) -> None:
        """Test tracking Earth-directed CME."""
        tracker = CMETracker()
        cme_data = {
            "speed_km_s": 1000,
            "angular_width_deg": 60,
            "direction_longitude_deg": 0,
            "direction_latitude_deg": 0,
        }
        result = tracker.track_cme(cme_data)
        assert result["cme_detected"] is True
        assert result["arrival_time_hours"] is not None
        assert result["speed_km_s"] == 1000

    def test_track_halo_cme(self) -> None:
        """Test tracking halo CME."""
        tracker = CMETracker()
        cme_data = {
            "speed_km_s": 1500,
            "angular_width_deg": 360,
            "direction_longitude_deg": 0,
            "direction_latitude_deg": 0,
        }
        result = tracker.track_cme(cme_data)
        assert result["halo_cme"] is True

    def test_track_slow_cme(self) -> None:
        """Test tracking slow CME (not Earth-directed)."""
        tracker = CMETracker()
        cme_data = {
            "speed_km_s": 200,
            "angular_width_deg": 60,
            "direction_longitude_deg": 0,
            "direction_latitude_deg": 0,
        }
        result = tracker.track_cme(cme_data)
        assert result["cme_detected"] is False


class TestGeomagneticStormPredictor:
    """Tests for GeomagneticStormPredictor class."""

    def test_init(self) -> None:
        """Test initialization."""
        predictor = GeomagneticStormPredictor()
        assert isinstance(predictor, torch.nn.Module)

    def test_forward(self) -> None:
        """Test forward pass."""
        predictor = GeomagneticStormPredictor(input_dim=32)
        features = torch.randn(4, 32)
        storm_prob, kp_estimate = predictor(features)
        assert storm_prob.shape == (4, 1)
        assert kp_estimate.shape == (4, 1)

    def test_output_ranges(self) -> None:
        """Test output value ranges."""
        predictor = GeomagneticStormPredictor(input_dim=32)
        features = torch.randn(4, 32)
        storm_prob, kp_estimate = predictor(features)
        assert (storm_prob >= 0).all() and (storm_prob <= 1).all()
        assert (kp_estimate >= 0).all() and (kp_estimate <= 9).all()

    def test_batch_sizes(self) -> None:
        """Test different batch sizes."""
        predictor = GeomagneticStormPredictor(input_dim=32)
        predictor.eval()
        for batch_size in [2, 8, 16]:
            features = torch.randn(batch_size, 32)
            storm_prob, kp_estimate = predictor(features)
            assert storm_prob.shape == (batch_size, 1)


class TestSolarStormDetector:
    """Tests for SolarStormDetector class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        detector = SolarStormDetector()
        assert detector.enable_flare is True
        assert detector.enable_cme is True
        assert detector.enable_geomag is True

    def test_init_disabled_components(self) -> None:
        """Test initialization with disabled components."""
        detector = SolarStormDetector(
            enable_flare_detection=False,
            enable_cme_tracking=False,
            enable_geomag_prediction=False,
        )
        assert detector.flare_detector is None
        assert detector.cme_tracker is None
        assert detector.geomag_predictor is None

    def test_predict_no_storm(self) -> None:
        """Test prediction with no storm data."""
        detector = SolarStormDetector()
        storm_data = {}
        result = detector.predict_solar_storm(storm_data)
        assert result.solar_storm_imminent is False
        assert result.confidence == 0.0

    def test_predict_with_flare(self) -> None:
        """Test prediction with flare data."""
        detector = SolarStormDetector()
        storm_data = {"xray_data": {"flux_short_wm2": 5e-5, "flux_long_wm2": 1e-9}}
        result = detector.predict_solar_storm(storm_data)
        assert result.flare_detected is True
        assert result.flare_class == "M"

    def test_predict_with_cme(self) -> None:
        """Test prediction with CME data."""
        detector = SolarStormDetector()
        storm_data = {
            "cme_data": {
                "speed_km_s": 1000,
                "angular_width_deg": 60,
                "direction_longitude_deg": 0,
                "direction_latitude_deg": 0,
            }
        }
        result = detector.predict_solar_storm(storm_data)
        assert result.cme_detected is True
        assert result.solar_storm_imminent is True

    def test_predict_with_magnetosphere(self) -> None:
        """Test prediction with magnetosphere data."""
        detector = SolarStormDetector()
        storm_data = {
            "magnetosphere_data": {
                "solar_wind_speed_km_s": 600,
                "bz_imf_nt": -10,
            }
        }
        result = detector.predict_solar_storm(storm_data)
        assert result.kp_index is not None

    def test_predict_full_data(self) -> None:
        """Test prediction with full data."""
        detector = SolarStormDetector()
        storm_data = {
            "xray_data": {"flux_short_wm2": 5e-4, "flux_long_wm2": 1e-9},
            "cme_data": {
                "speed_km_s": 1500,
                "angular_width_deg": 180,
                "direction_longitude_deg": 0,
                "direction_latitude_deg": 0,
            },
            "magnetosphere_data": {
                "solar_wind_speed_km_s": 800,
                "bz_imf_nt": -20,
            },
            "geomagnetic_indices": {"dst_index": -100},
            "schumann_data": np.array([7.5, 7.6, 7.7, 7.8, 7.9]),
        }
        result = detector.predict_solar_storm(storm_data)
        assert result.flare_detected is True
        assert result.cme_detected is True
        assert result.dst_index == -100
        assert result.schumann_correlation is not None

    def test_classify_geostorm(self) -> None:
        """Test geomagnetic storm classification."""
        detector = SolarStormDetector()
        assert detector._classify_geostorm(9.0) == "extreme"
        assert detector._classify_geostorm(8.0) == "severe"
        assert detector._classify_geostorm(7.0) == "strong"
        assert detector._classify_geostorm(6.0) == "moderate"
        assert detector._classify_geostorm(5.0) == "minor"
        assert detector._classify_geostorm(3.0) == "none"

    def test_assess_grid_risk(self) -> None:
        """Test power grid risk assessment."""
        detector = SolarStormDetector()
        result = SolarStormPredictionResult(
            solar_storm_imminent=True,
            confidence=0.9,
            storm_severity="extreme",
        )
        assert detector._assess_grid_risk(result) == "critical"

    def test_assess_satellite_risk(self) -> None:
        """Test satellite risk assessment."""
        detector = SolarStormDetector()
        result = SolarStormPredictionResult(
            solar_storm_imminent=True,
            confidence=0.9,
            storm_severity="extreme",
            radiation_storm=True,
        )
        assert detector._assess_satellite_risk(result) == "critical"

    def test_assess_comm_risk(self) -> None:
        """Test communication risk assessment."""
        detector = SolarStormDetector()
        result = SolarStormPredictionResult(
            solar_storm_imminent=True,
            confidence=0.9,
            storm_severity="extreme",
            radio_blackout=True,
        )
        assert detector._assess_comm_risk(result) == "critical"

    def test_correlate_schumann(self) -> None:
        """Test Schumann resonance correlation."""
        detector = SolarStormDetector()
        schumann_data = np.array([7.83, 7.83, 7.83])
        correlation = detector._correlate_schumann(schumann_data)
        assert correlation >= 0.0 and correlation <= 1.0

    def test_generate_protective_actions(self) -> None:
        """Test protective action generation."""
        detector = SolarStormDetector()
        result = SolarStormPredictionResult(
            solar_storm_imminent=True,
            confidence=0.9,
            storm_severity="extreme",
            power_grid_risk="critical",
            satellite_risk="critical",
            communication_disruption="critical",
        )
        actions = detector._generate_protective_actions(result)
        assert len(actions) > 0

    def test_generate_infrastructure_alerts(self) -> None:
        """Test infrastructure alert generation."""
        detector = SolarStormDetector()
        result = SolarStormPredictionResult(
            solar_storm_imminent=True,
            confidence=0.9,
            storm_severity="extreme",
        )
        alerts = detector._generate_infrastructure_alerts(result)
        assert len(alerts) > 0
        assert any("EXTREME" in alert for alert in alerts)
