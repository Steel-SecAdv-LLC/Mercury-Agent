# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Dust-storm detector: WMO visibility classes, Shao-Lu / Fecan friction velocity, haboob gust-front signature, and NWS dust-alert wiring.

Formulation anchors:
* WMO SDS visibility classes (< 200 m severe dust storm, < 1000 m dust
  storm, < 10 km blowing dust) with the dust-raising wind gate.
* Shao & Lu (2000) dry threshold friction velocity: documented curve
  minimum ~0.2 m/s near d = 75-100 um for quartz sand in air.
* Fecan et al. (1999) soil-moisture correction with hand-computed worked
  example (clay 10 %: w' = 0.0014*100 + 0.17*10 = 1.84 %).
* Neutral log-profile friction velocity, hand-computed worked example
  (u* = 0.4 * 12 / ln(10/0.001) = 0.52115 m/s).

NWS wiring tests run against the recorded real Dust Storm Warning fixture
(``tests/fixtures/meteorological/nws_dust_storm_warnings.json``: five real
warnings for the San Luis Valley, CO, issued 2026-07-08; provenance in
PROVENANCE.json).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

# The dev venv's editable install may point at a sibling worktree that
# predates ``dust_storm_detector``; ``unused-ignore`` keeps a correctly
# installed tree (CI) clean.
from omni_mercury_engine.detectors.meteorological.dust_storm_detector import (  # type: ignore[import-not-found,unused-ignore]
    DustEventClass,
    DustStormDetector,
)

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "meteorological"


@pytest.fixture(scope="module")
def detector() -> DustStormDetector:
    """One physics-core detector for the whole module (stateless)."""
    return DustStormDetector()


@pytest.fixture(scope="module")
def dust_alerts() -> dict[str, Any]:
    """Recorded real Dust Storm Warning FeatureCollection."""
    with open(FIXTURE_DIR / "nws_dust_storm_warnings.json") as f:
        payload: dict[str, Any] = json.load(f)
    return payload


class TestVisibilityClassification:
    """WMO SDS visibility classes with the dust-raising wind gate."""

    @pytest.mark.parametrize(
        ("vis_m", "wind_ms", "expected"),
        [
            (150.0, 15.0, DustEventClass.SEVERE_DUST_STORM),
            (500.0, 14.0, DustEventClass.DUST_STORM),
            (999.0, 10.0, DustEventClass.DUST_STORM),
            (5000.0, 12.0, DustEventClass.BLOWING_DUST),
            (20000.0, 15.0, DustEventClass.NONE),
        ],
    )
    def test_classes(
        self, detector: DustStormDetector, vis_m: float, wind_ms: float, expected: str
    ) -> None:
        assert detector.classify_visibility(vis_m, wind_ms) == expected

    def test_wind_gate_rejects_fog_lookalikes(self, detector: DustStormDetector) -> None:
        """Low visibility in calm air is fog/haze, never a dust event."""
        assert detector.classify_visibility(300.0, 4.0) == DustEventClass.NONE

    def test_series_classification(self, detector: DustStormDetector) -> None:
        labels = detector.classify_series([150.0, 5000.0, 20000.0], [15.0, 12.0, 15.0])
        assert labels == [
            DustEventClass.SEVERE_DUST_STORM,
            DustEventClass.BLOWING_DUST,
            DustEventClass.NONE,
        ]

    def test_fail_loud(self, detector: DustStormDetector) -> None:
        with pytest.raises(ValueError):
            detector.classify_visibility(-1.0, 15.0)
        with pytest.raises(ValueError):
            detector.classify_visibility(500.0, float("nan"))
        with pytest.raises(ValueError):
            detector.classify_series([100.0, 200.0], [15.0])
        with pytest.raises(ValueError):
            DustStormDetector(wind_threshold_ms=0.0)


class TestShaoLuThreshold:
    """Shao & Lu (2000) dry threshold friction velocity."""

    def test_documented_minimum_location_and_value(self, detector: DustStormDetector) -> None:
        """The curve minimum sits near d = 75-100 um at u*t ~ 0.2 m/s."""
        diameters = np.logspace(math.log10(10e-6), math.log10(1000e-6), 400)
        ut = np.array([detector.threshold_friction_velocity_dry(float(d)) for d in diameters])
        d_min = float(diameters[int(np.argmin(ut))])
        assert 50e-6 <= d_min <= 150e-6
        assert 0.18 <= float(np.min(ut)) <= 0.28

    def test_hand_computed_value_at_75um(self, detector: DustStormDetector) -> None:
        """u*t(75 um) = sqrt(0.0123 * ((2650/1.23)*9.80665*75e-6
        + 3e-4/(1.23*75e-6))) = 0.24386 m/s (hand-computed)."""
        sigma_term = (2650.0 / 1.23) * 9.80665 * 75e-6
        cohesion_term = 3.0e-4 / (1.23 * 75e-6)
        expected = math.sqrt(0.0123 * (sigma_term + cohesion_term))
        assert detector.threshold_friction_velocity_dry(75e-6) == pytest.approx(expected, rel=1e-12)

    def test_both_branches_raise_threshold(self, detector: DustStormDetector) -> None:
        """Gravity dominates large grains; cohesion dominates small ones."""
        mid = detector.threshold_friction_velocity_dry(80e-6)
        assert detector.threshold_friction_velocity_dry(500e-6) > mid
        assert detector.threshold_friction_velocity_dry(5e-6) > mid

    def test_fail_loud(self, detector: DustStormDetector) -> None:
        with pytest.raises(ValueError):
            detector.threshold_friction_velocity_dry(0.0)
        with pytest.raises(ValueError):
            detector.threshold_friction_velocity_dry(75e-6, air_density_kg_m3=-1.0)


class TestFecanMoistureCorrection:
    """Fecan et al. (1999) parameterization."""

    def test_residual_moisture_worked_example(self, detector: DustStormDetector) -> None:
        """clay = 10 %: w' = 0.0014*10^2 + 0.17*10 = 1.84 %."""
        ratio, w_prime = detector.fecan_moisture_ratio(soil_moisture_pct=0.5, clay_pct=10.0)
        assert w_prime == pytest.approx(1.84, rel=1e-12)
        assert ratio == 1.0  # w below w': no correction

    def test_wet_soil_ratio_hand_computed(self, detector: DustStormDetector) -> None:
        """w = 5 %, clay = 10 %: ratio = sqrt(1 + 1.21*(5-1.84)^0.68)."""
        expected = math.sqrt(1.0 + 1.21 * (5.0 - 1.84) ** 0.68)
        ratio, _ = detector.fecan_moisture_ratio(5.0, 10.0)
        assert ratio == pytest.approx(expected, rel=1e-12)
        assert ratio > 1.0

    def test_monotone_in_moisture(self, detector: DustStormDetector) -> None:
        ratios = [detector.fecan_moisture_ratio(w, 10.0)[0] for w in (2.0, 4.0, 8.0, 16.0)]
        assert ratios == sorted(ratios)

    def test_fail_loud(self, detector: DustStormDetector) -> None:
        with pytest.raises(ValueError):
            detector.fecan_moisture_ratio(-1.0, 10.0)
        with pytest.raises(ValueError):
            detector.fecan_moisture_ratio(5.0, 150.0)


class TestEmissionPotential:
    """Log-profile u* against threshold with moisture correction."""

    def test_hand_computed_friction_velocity(self, detector: DustStormDetector) -> None:
        """u* = 0.4 * 12 / ln(10 / 0.001) = 0.521153 m/s."""
        result = detector.emission_potential(
            wind_speed_ms=12.0,
            measurement_height_m=10.0,
            roughness_length_m=0.001,
            soil_moisture_pct=0.5,
            clay_pct=10.0,
        )
        expected_ustar = 0.4 * 12.0 / math.log(10.0 / 0.001)
        assert result.friction_velocity_ms == pytest.approx(expected_ustar, rel=1e-12)
        assert result.emission_favorable
        assert result.excess_ratio > 1.0

    def test_wet_soil_suppresses_emission(self, detector: DustStormDetector) -> None:
        dry = detector.emission_potential(
            wind_speed_ms=7.0,
            measurement_height_m=10.0,
            roughness_length_m=0.001,
            soil_moisture_pct=0.5,
            clay_pct=10.0,
        )
        wet = detector.emission_potential(
            wind_speed_ms=7.0,
            measurement_height_m=10.0,
            roughness_length_m=0.001,
            soil_moisture_pct=12.0,
            clay_pct=10.0,
        )
        assert dry.threshold_wet_ms == dry.threshold_dry_ms
        assert wet.threshold_wet_ms > wet.threshold_dry_ms
        assert wet.excess_ratio < dry.excess_ratio

    def test_missing_inputs_fail_loud(self, detector: DustStormDetector) -> None:
        """Every required input must be supplied -- nothing is defaulted."""
        with pytest.raises(ValueError, match="wind_speed_ms"):
            detector.emission_potential(
                measurement_height_m=10.0,
                roughness_length_m=0.001,
                soil_moisture_pct=0.5,
                clay_pct=10.0,
            )
        with pytest.raises(ValueError, match="soil_moisture_pct"):
            detector.emission_potential(
                wind_speed_ms=12.0,
                measurement_height_m=10.0,
                roughness_length_m=0.001,
                clay_pct=10.0,
            )
        with pytest.raises(ValueError, match="clay_pct"):
            detector.emission_potential(
                wind_speed_ms=12.0,
                measurement_height_m=10.0,
                roughness_length_m=0.001,
                soil_moisture_pct=0.5,
            )

    def test_unphysical_geometry_fails_loud(self, detector: DustStormDetector) -> None:
        with pytest.raises(ValueError, match="exceed"):
            detector.emission_potential(
                wind_speed_ms=12.0,
                measurement_height_m=0.0005,
                roughness_length_m=0.001,
                soil_moisture_pct=0.5,
                clay_pct=10.0,
            )
        with pytest.raises(ValueError):
            detector.emission_potential(
                wind_speed_ms=-3.0,
                measurement_height_m=10.0,
                roughness_length_m=0.001,
                soil_moisture_pct=0.5,
                clay_pct=10.0,
            )


class TestHaboobSignature:
    """Gust-front (density current) signature in obs time series."""

    @staticmethod
    def _gust_front_series() -> dict[str, Any]:
        """Constructed unit input: a textbook outflow passage at t=30 min
        (pre-frontal desert afternoon, then pressure jump, temperature
        crash, wind veer + surge; magnitudes within the Idso et al. 1972
        haboob signature envelope)."""
        t = np.arange(0.0, 7200.0, 300.0)  # 5-min cadence, 2 h
        n = t.size
        pressure = np.full(n, 1005.0)
        temp = np.full(n, 41.0)
        wspd = np.full(n, 4.0)
        wdir = np.full(n, 120.0)
        front = 6  # t = 30 min
        pressure[front:] = 1007.5
        temp[front:] = 34.0
        wspd[front:] = 18.0
        wdir[front:] = 250.0
        return {
            "times_s": t,
            "pressure_hpa": pressure,
            "temperature_c": temp,
            "wind_speed_ms": wspd,
            "wind_direction_deg": wdir,
        }

    def test_detects_gust_front(self, detector: DustStormDetector) -> None:
        result = detector.detect_haboob_signature(**self._gust_front_series())
        assert result.detected
        assert result.onset_index is not None
        assert result.pressure_jump_hpa >= 1.0
        assert result.temp_drop_c >= 3.0
        assert result.wind_shift_deg >= 30.0
        assert result.wind_surge_ms >= 5.0

    def test_quiet_series_not_flagged(self, detector: DustStormDetector) -> None:
        series = self._gust_front_series()
        n = series["times_s"].size
        series["pressure_hpa"] = np.linspace(1005.0, 1005.4, n)  # diurnal drift
        series["temperature_c"] = np.linspace(41.0, 39.5, n)
        series["wind_speed_ms"] = np.full(n, 5.0)
        series["wind_direction_deg"] = np.full(n, 120.0)
        result = detector.detect_haboob_signature(**series)
        assert not result.detected
        assert result.onset_index is None

    def test_partial_signature_not_flagged(self, detector: DustStormDetector) -> None:
        """Pressure jump + temp drop without the wind shift (e.g. a dry
        frontal passage with no outflow surge) must not be called a haboob."""
        series = self._gust_front_series()
        series["wind_speed_ms"] = np.full(series["times_s"].size, 4.0)
        series["wind_direction_deg"] = np.full(series["times_s"].size, 120.0)
        result = detector.detect_haboob_signature(**series)
        assert not result.detected

    def test_fail_loud_series(self, detector: DustStormDetector) -> None:
        series = self._gust_front_series()
        with pytest.raises(ValueError):
            detector.detect_haboob_signature(
                series["times_s"],
                series["pressure_hpa"][:-1],
                series["temperature_c"],
                series["wind_speed_ms"],
                series["wind_direction_deg"],
            )
        bad_t = series["times_s"].copy()
        bad_t[5] = bad_t[4]
        with pytest.raises(ValueError):
            detector.detect_haboob_signature(
                bad_t,
                series["pressure_hpa"],
                series["temperature_c"],
                series["wind_speed_ms"],
                series["wind_direction_deg"],
            )
        with pytest.raises(ValueError, match="window_s"):
            detector.detect_haboob_signature(**self._gust_front_series(), window_s=1.0)


class TestNWSDustWiring:
    """Wiring against the recorded real Dust Storm Warning feed."""

    def test_recorded_feed(self, detector: DustStormDetector, dust_alerts: dict[str, Any]) -> None:
        result = detector.cross_check_nws_alerts(dust_alerts)
        assert result["n_dust_alerts"] == 5
        assert result["dust_storm_warned"] is True
        assert result["events"] == ["Dust Storm Warning"]

    def test_severe_thunderstorm_feed_matches_nothing(self, detector: DustStormDetector) -> None:
        with open(FIXTURE_DIR / "nws_severe_thunderstorm_warnings.json") as f:
            payload = json.load(f)
        result = detector.cross_check_nws_alerts(payload)
        assert result["n_dust_alerts"] == 0
        assert not result["dust_storm_warned"]

    def test_bad_payload_fails_loud(self, detector: DustStormDetector) -> None:
        with pytest.raises(TypeError):
            detector.cross_check_nws_alerts(3.14)


class TestExtractFeatures:
    """Fusion feature interface."""

    def test_dict_visibility_and_emission_paths(self, detector: DustStormDetector) -> None:
        features = detector.extract_features(
            {
                "visibility_m": 500.0,
                "wind_speed_ms": 14.0,
                "measurement_height_m": 10.0,
                "roughness_length_m": 0.001,
                "soil_moisture_pct": 0.5,
                "clay_pct": 10.0,
            }
        )
        assert isinstance(features, torch.Tensor)
        assert features.shape == (DustStormDetector.FEATURE_DIM,)
        assert torch.isfinite(features).all()
        assert features[0].item() == 2.0  # dust_storm ordinal

    def test_dict_without_physics_inputs_fails_loud(self, detector: DustStormDetector) -> None:
        with pytest.raises(ValueError, match="no recognized"):
            detector.extract_features({"unrelated": 1.0})

    def test_array_path(self, detector: DustStormDetector) -> None:
        features = detector.extract_features(np.linspace(0.0, 2.0, 9))
        assert features.shape == (DustStormDetector.FEATURE_DIM,)
        assert features[0].item() == pytest.approx(1.0, abs=1e-6)
