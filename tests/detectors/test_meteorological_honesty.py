# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Meteorological/terrain detectors must derive from observed physics, never fabricate.

Regression locks for the anti-theater fixes:

* ``TornadoDetector`` — mesocyclone probability and rotation velocity came from
  an UNTRAINED DopplerRadarAnalyzer (rotation = net·50). Now: the Doppler
  velocity couplet ``V_rot = (V_max − V_min)/2`` with the ~15 m/s operational
  threshold, median over frames.
* ``WildfireDetector`` — fire probability came from an untrained CNN. Now:
  VIIRS/MODIS-style brightness-temperature criteria (absolute 360 K, contextual
  330 K + 4·MAD contrast).
* ``LandslideDetector`` — failure probability/type came from an untrained
  SlopeStabilityModel fed an opaque vector; without ``slope_features`` the
  detector could NEVER declare a landslide imminent. Now: geotechnical physics
  from the observed slope angle, saturation, displacement rate, and triggers.
* ``HurricanePredictionResult`` — ``track_forecast``/``landfall_probability``/
  ``time_to_landfall_hours`` were declared but never computed (removed), and the
  supplied wind field was silently ignored (the untrained WindPatternAnalyzer
  was never called). Now: deterministic kinematics — observed max wind and
  relative vorticity ζ = ∂v/∂x − ∂u/∂y with a closed-circulation criterion.
"""

from __future__ import annotations

import numpy as np
import torch

from omni_mercury_engine.detectors.geological.hurricane_detector import (
    HurricaneDetector,
    HurricanePredictionResult,
)
from omni_mercury_engine.detectors.geological.landslide import LandslideDetector
from omni_mercury_engine.detectors.geological.tornado_detector import TornadoDetector
from omni_mercury_engine.detectors.geological.wildfire import WildfireDetector


class TestTornadoHonesty:
    def test_default_detector_serves_the_shipped_winner(self) -> None:
        assert TornadoDetector()._neural_trained is True

    def test_physics_configuration_is_untrained(self) -> None:
        assert TornadoDetector(load_shipped_weights=False)._neural_trained is False

    def test_velocity_couplet_detects_mesocyclone_and_rejects_calm(self) -> None:
        rng = np.random.default_rng(0)
        det = TornadoDetector(load_shipped_weights=False)
        couplet = np.stack([np.linspace(-30, 30, 64) + rng.normal(0, 2, 64) for _ in range(10)])
        calm = rng.normal(0, 3, (10, 64))
        meso = det._analyze_radar(couplet)
        quiet = det._analyze_radar(calm)
        assert meso["mesocyclone_detected"] is True
        assert 25.0 < meso["rotation_velocity"] < 40.0  # physical m/s, not net·50
        assert quiet["mesocyclone_detected"] is False

    def test_untrained_path_ignores_neural_model_weights(self) -> None:
        rng = np.random.default_rng(1)
        det = TornadoDetector(load_shipped_weights=False)
        field = np.stack([np.linspace(-20, 20, 64) for _ in range(6)])
        before = det._analyze_radar(field)["rotation_velocity"]
        assert det.radar_analyzer is not None
        with torch.no_grad():
            for p in det.radar_analyzer.parameters():
                p.mul_(0).add_(3.0)
        after = det._analyze_radar(field)["rotation_velocity"]
        assert before == after
        _ = rng  # rng reserved for future field jitter


class TestWildfireHonesty:
    def test_default_detector_serves_the_shipped_winner(self) -> None:
        assert WildfireDetector()._neural_trained is True

    def test_physics_configuration_is_untrained(self) -> None:
        assert WildfireDetector(load_shipped_weights=False)._neural_trained is False

    def test_brightness_temperature_detects_fire_and_rejects_ambient(self) -> None:
        rng = np.random.default_rng(0)
        det = WildfireDetector()
        scene = rng.normal(295, 3, (64, 64))
        scene[30:34, 30:34] = 420.0  # active combustion
        fire = det.predict_wildfire({"thermal_image": scene})
        ambient = det.predict_wildfire({"thermal_image": rng.normal(295, 3, (64, 64))})
        assert fire.fire_detected is True
        assert fire.thermal_hotspots > 0
        assert ambient.fire_detected is False

    def test_untrained_path_ignores_neural_model_weights(self) -> None:
        rng = np.random.default_rng(2)
        det = WildfireDetector()
        scene = rng.normal(295, 3, (32, 32))
        scene[10, 10] = 400.0
        before = det.predict_wildfire({"thermal_image": scene.copy()}).confidence
        assert det.ignition_detector is not None
        with torch.no_grad():
            for p in det.ignition_detector.parameters():
                p.mul_(0).add_(2.0)
        after = det.predict_wildfire({"thermal_image": scene.copy()}).confidence
        assert before == after


class TestLandslideHonesty:
    def test_default_detector_serves_the_shipped_winner(self) -> None:
        assert LandslideDetector()._neural_trained is True

    def test_physics_configuration_is_untrained(self) -> None:
        assert LandslideDetector(load_shipped_weights=False)._neural_trained is False

    def test_physics_fires_without_opaque_feature_vector(self) -> None:
        """Previously landslide_imminent could NEVER be True without
        slope_features; the physics path works from the real fields."""
        det = LandslideDetector()
        result = det.predict_landslide(
            {
                "rainfall_data": {
                    "intensity_mm_hr": 25.0,
                    "duration_hours": 12.0,
                    "antecedent_7day_mm": 150.0,
                },
                "sensor_data": {"soil_saturation_pct": 90.0, "displacement_rate_mm_day": 35.0},
                "slope_data": {"slope_angle_deg": 38.0},
            }
        )
        assert result.slope_failure_probability > 0.6
        assert result.landslide_imminent is True
        assert result.landslide_type in {"mud_flow", "debris_flow"}  # rainfall + saturation

    def test_dry_flat_slope_is_stable(self) -> None:
        det = LandslideDetector()
        result = det.predict_landslide(
            {
                "sensor_data": {"soil_saturation_pct": 20.0, "displacement_rate_mm_day": 0.1},
                "slope_data": {"slope_angle_deg": 8.0},
            }
        )
        assert result.landslide_imminent is False
        assert result.slope_failure_probability < 0.2

    def test_untrained_path_ignores_neural_model_weights(self) -> None:
        det = LandslideDetector()
        payload = {
            "sensor_data": {"soil_saturation_pct": 70.0, "displacement_rate_mm_day": 10.0},
            "slope_data": {"slope_angle_deg": 30.0},
        }
        before = det.predict_landslide(dict(payload)).slope_failure_probability
        assert det.stability_model is not None
        with torch.no_grad():
            for p in det.stability_model.parameters():
                p.mul_(0).add_(1.0)
        after = det.predict_landslide(dict(payload)).slope_failure_probability
        assert before == after

    def test_snowmelt_yields_avalanche_type(self) -> None:
        det = LandslideDetector()
        result = det.predict_landslide(
            {
                "weather_data": {"snowmelt_mm_day": 30.0},
                "slope_data": {"slope_angle_deg": 40.0},
                "sensor_data": {"soil_saturation_pct": 50.0, "displacement_rate_mm_day": 5.0},
            }
        )
        assert result.landslide_type == "snow_avalanche"


class TestHurricaneHonesty:
    def test_dead_track_fields_are_gone(self) -> None:
        """The uncomputed track/landfall fields were removed, not left as theater."""
        fields = set(HurricanePredictionResult.__dataclass_fields__)
        assert "track_forecast" not in fields
        assert "landfall_probability" not in fields
        assert "time_to_landfall_hours" not in fields

    def test_wind_field_is_no_longer_ignored(self) -> None:
        """A rotating wind field registers as a closed circulation with the
        analytically correct vorticity (solid-body: ζ = 2ω)."""
        n = 50
        x = np.linspace(-1, 1, n)
        X, Y = np.meshgrid(x, x)
        omega, half_domain = 1e-3, 100_000.0
        det = HurricaneDetector()
        vortex = det._analyze_wind_field(
            {
                "u": -omega * Y * half_domain,
                "v": omega * X * half_domain,
                "grid_spacing_m": 2 * half_domain / (n - 1),
            }
        )
        uniform = det._analyze_wind_field(
            {
                "u": np.full((n, n), 10.0),
                "v": np.zeros((n, n)),
                "grid_spacing_m": 2 * half_domain / (n - 1),
            }
        )
        assert abs(vortex["max_relative_vorticity_s1"] - 2 * omega) < 1e-4
        assert vortex["closed_circulation"] is True
        assert uniform["max_relative_vorticity_s1"] < 1e-9
        assert uniform["closed_circulation"] is False

    def test_predict_uses_the_wind_field(self) -> None:
        n = 50
        x = np.linspace(-1, 1, n)
        X, Y = np.meshgrid(x, x)
        omega, half_domain = 1e-3, 100_000.0
        det = HurricaneDetector()
        result = det.predict_hurricane(
            {
                "pressure_data": {
                    "central_pressure_mb": 950.0,
                    "environmental_pressure_mb": 1013.0,
                },
                "wind_field": {
                    "u": -omega * Y * half_domain,
                    "v": omega * X * half_domain,
                    "grid_spacing_m": 2 * half_domain / (n - 1),
                },
            }
        )
        assert result.max_relative_vorticity_s1 is not None
        assert result.closed_circulation is True
        assert result.cyclone_detected is True

    def test_malformed_wind_field_contributes_nothing(self) -> None:
        det = HurricaneDetector()
        out = det._analyze_wind_field({"u": np.zeros((3, 4)), "v": np.zeros((2, 2))})
        assert out["max_wind_speed_kt"] == 0.0
        assert out["closed_circulation"] is False
