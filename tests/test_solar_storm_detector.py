# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for Solar Storm Detector."""

from typing import Any

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

    def test_off_modality_matrix_rejected_cleanly(self) -> None:
        """A generic 2-D feature matrix is not a GOES XRS flux series."""
        detector = SolarFlareDetector()
        matrix = np.zeros((200, 8))
        with pytest.raises(ValueError, match=r"GOES XRS.*1-D time series.*\(200, 8\)"):
            detector.predict_solar_flare(matrix)
        with pytest.raises(ValueError, match=r"GOES XRS.*1-D time series.*\(200, 8\)"):
            detector.extract_features(matrix)

    def test_empty_flux_series_rejected_cleanly(self) -> None:
        """An empty flux series carries no measurement to classify."""
        detector = SolarFlareDetector()
        with pytest.raises(ValueError, match="empty X-ray flux series"):
            detector.predict_solar_flare(np.array([]))

    def test_scalar_flux_accepted_as_single_measurement(self) -> None:
        """A 0-d numpy array (np.array(1e-5)) and a Python float are single
        GOES XRS measurements: accepted and classified, not rejected as a
        malformed shape. Regression for _validate_xray_flux treating a 0-d
        array as non-1-D and crashing on the trend path.
        """
        detector = SolarFlareDetector()
        for flux, expected in [(1e-9, "A"), (5e-6, "C"), (1e-5, "M"), (1e-4, "X")]:
            zero_d = detector.predict_solar_flare(np.array(flux))
            py_float = detector.predict_solar_flare(flux)
            assert zero_d.flare_class == expected
            assert py_float.flare_class == expected
        # extract_features must produce a finite feature vector for a 0-d input.
        features = np.asarray(detector.extract_features(np.array(1e-5)))
        assert features.ndim == 1 and np.all(np.isfinite(features))

    def test_x_class_flux_curve_drive(self) -> None:
        """A rising GOES-style flux curve peaking at X2.5 classifies as X."""
        detector = SolarFlareDetector()
        t = np.arange(201)
        flux = 1e-7 + 2.5e-4 * np.exp(-0.5 * ((t - 200) / 30.0) ** 2)

        offline = detector.predict_solar_flare(flux)
        assert offline.flare_detected is True
        assert offline.flare_class == "X"
        assert offline.flux_class_index == 4
        assert offline.confidence == pytest.approx(1.0)
        assert offline.x_ray_flux == pytest.approx(flux[-1])
        # Offline the storm-forecast fields are never fabricated.
        assert offline.kp_index_predicted is None
        assert offline.dst_index_predicted is None
        assert offline.geomagnetic_storm_probability is None

        observed = detector.predict_solar_flare(flux, observed_kp=7.0, kp_source="test_kp")
        assert observed.kp_index_predicted == pytest.approx(7.0)
        assert observed.dst_index_predicted == pytest.approx(-150.0)
        assert observed.geomagnetic_storm_probability == pytest.approx(0.75)
        assert observed.storm_forecast_source == "test_kp"

        features = detector.extract_features(flux)
        assert features.shape == (20,)
        assert features[2] == pytest.approx(float(np.max(flux)))
        assert features[5] == pytest.approx(1.0)  # confidence
        assert features[6] == pytest.approx(1.0)  # X class index / 4


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
        storm_data: dict[str, Any] = {}
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

    def test_geomag_accepts_batched_features_and_rejects_multirow(self) -> None:
        """Explicit ``features`` of shape ``(1, W)`` is an already-batched
        single sample and must run the trained predictor, not crash BatchNorm1d
        with a 3-D tensor. A genuine multi-row ``(N>1, W)`` batch and any
        mis-shaped array fall back to physics. Regression for the width guard
        that accepted ``(1, W)`` and then ``unsqueeze(0)``'d it to 3-D.
        """
        detector = SolarStormDetector(load_shipped_weights=True)
        predictor = detector.geomag_predictor
        if predictor is None:  # pragma: no cover - config guard
            pytest.skip("geomag predictor disabled")
        width = int(predictor.feature_fusion[0].in_features)

        batched = detector._predict_geomagnetic_storm(
            {"features": np.zeros((1, width), dtype=np.float32)}
        )
        flat = detector._predict_geomagnetic_storm({"features": np.zeros(width, dtype=np.float32)})
        assert batched["method"] == flat["method"] == "neural"

        # Multi-row batch, wrong width, and a 3-D array all degrade to physics
        # instead of crashing the network.
        for bad in (
            np.zeros((5, width), dtype=np.float32),
            np.zeros((1, width + 32), dtype=np.float32),
            np.zeros((1, 1, width), dtype=np.float32),
        ):
            result = detector._predict_geomagnetic_storm({"features": bad})
            assert result["method"].startswith("physics")

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
