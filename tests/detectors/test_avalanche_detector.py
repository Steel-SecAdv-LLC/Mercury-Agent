# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the snow avalanche detector (SK38 core + weather factors).

Literature worked examples covered:

- Föhn (1987) line-load skier stress at the 38° reference: the maximising
  load angle is ~54.3° and the stress is ~0.152 kPa per metre of slab for
  R ≈ 490 N/m (85 kg on 1.7 m skis) — the reference values documented in the
  operational SNOWPACK implementation.
- Full SK38 worked example (0.4 m / 200 kg/m³ slab over 250 kg/m³ facets)
  computed independently here from the published formulas: strength
  18.5 kPa·(250/917)^2.11 ≈ 1.19 kPa, slab stress ≈ 0.381 kPa, skier stress
  at h_eff ≈ 0.227 m ≈ 0.67 kPa → SK38 ≈ 1.13 (transitional class).
- Schweizer et al. (2003) critical new-snow rates (30 cm/24 h natural,
  10 cm skier-critical) and the wind-slab multiplier.
- TG > 10 K/m faceting threshold; Conway & Raymond (1993) rain-on-snow.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from omni_mercury_engine.detectors.geological.avalanche_detector import (
    CRITICAL_NEW_SNOW_24H_CM,
    RHO_ICE,
    TG_FACETING_THRESHOLD_K_M,
    AvalancheDangerLevel,
    AvalancheDetector,
    SnowLayer,
)


def _slab_over_facets(
    slab_thickness: float = 0.4,
    slab_density: float = 200.0,
    weak_density: float = 250.0,
) -> list[SnowLayer]:
    """Canonical profile: one slab layer over a faceted weak layer over base."""
    return [
        SnowLayer(slab_thickness, slab_density, -6.0),
        SnowLayer(0.02, weak_density, -5.0, grain_type="facets"),
        SnowLayer(0.8, 320.0, -2.0),
    ]


class TestFoehnSkierStress:
    """Föhn (1987) line-load solution against the SNOWPACK reference values."""

    def test_reference_stress_and_angle_at_38_degrees(self) -> None:
        """85 kg / 1.7 m skis => R=490.5 N/m: 0.1523 kPa/m at alpha ~54.3 deg."""
        det = AvalancheDetector(skier_line_load_n_m=85.0 * 9.81 / 1.7)
        stress, alpha = det._foehn_skier_stress(1.0, np.deg2rad(38.0))
        assert stress == pytest.approx(152.3, rel=0.005)
        assert alpha == pytest.approx(54.3, abs=0.3)

    def test_stress_scales_inversely_with_depth(self) -> None:
        det = AvalancheDetector()
        s1, _ = det._foehn_skier_stress(0.5, np.deg2rad(38.0))
        s2, _ = det._foehn_skier_stress(1.0, np.deg2rad(38.0))
        assert s1 == pytest.approx(2.0 * s2, rel=1e-6)

    def test_zero_depth_raises(self) -> None:
        with pytest.raises(ValueError, match="effective slab depth"):
            AvalancheDetector()._foehn_skier_stress(0.0, np.deg2rad(38.0))


class TestSK38WorkedExample:
    """Full SK38 computation against independent hand calculation."""

    def test_worked_example_transitional(self) -> None:
        """0.4 m 200 kg/m³ slab over facets (250 kg/m³): SK38 ≈ 1.13, fair."""
        det = AvalancheDetector()
        result = det.compute_sk38(_slab_over_facets(), weak_layer_index=1)

        # Independent hand computation from the published formulas:
        psi = math.radians(38.0)
        tau_xz = 200.0 * 9.81 * 0.4 * math.sin(psi) * math.cos(psi)
        assert result.tau_xz_pa == pytest.approx(tau_xz, rel=1e-9)

        penetration = 0.8 * 43.3 / 200.0  # J&J 1998 / SNOWPACK adaptation
        assert result.penetration_m == pytest.approx(penetration, rel=1e-9)
        h_eff = 0.4 - penetration
        # SNOWPACK reference: 0.1523 kPa per metre for R=490.5; here R=500.
        expected_skier = (500.0 / 490.5) * 152.3 / h_eff
        assert result.delta_tau_skier_pa == pytest.approx(expected_skier, rel=0.01)

        strength = 18.5e3 * (250.0 / RHO_ICE) ** 2.11
        assert result.weak_layer_strength_pa == pytest.approx(strength, rel=1e-9)
        assert result.strength_source == "jamieson_johnston_2001"

        expected_sk38 = strength / (tau_xz + expected_skier)
        assert result.sk38 == pytest.approx(expected_sk38, rel=0.01)
        assert 1.0 < result.sk38 < 1.5
        assert result.stability_class == "fair"

    def test_dense_slab_over_strong_facets_is_stable(self) -> None:
        """A dense slab over well-sintered (450 kg/m³) facets: SK38 good."""
        layers = [
            SnowLayer(1.0, 280.0, -5.0),
            SnowLayer(0.02, 450.0, -4.0, grain_type="facets"),
            SnowLayer(0.5, 400.0, -2.0),
        ]
        result = AvalancheDetector().compute_sk38(layers, weak_layer_index=1)
        assert result.sk38 > 1.5
        assert result.stability_class == "good"

    def test_shallow_slab_over_weak_facets_is_poor(self) -> None:
        """Thin soft slab over very light facets: skier stress dominates."""
        layers = [
            SnowLayer(0.3, 150.0, -8.0),
            SnowLayer(0.02, 150.0, -7.0, grain_type="depth_hoar"),
            SnowLayer(0.6, 350.0, -2.0),
        ]
        result = AvalancheDetector().compute_sk38(layers, weak_layer_index=1)
        assert result.sk38 < 1.0
        assert result.stability_class == "poor"

    def test_measured_strength_preferred(self) -> None:
        result = AvalancheDetector().compute_sk38(
            _slab_over_facets(), weak_layer_index=1, measured_strength_pa=800.0
        )
        assert result.weak_layer_strength_pa == 800.0
        assert result.strength_source == "measured"

    def test_nonpersistent_grain_without_measurement_raises(self) -> None:
        """The J&J 2001 power law must not be applied out of scope."""
        layers = [
            SnowLayer(0.4, 200.0, -6.0),
            SnowLayer(0.02, 250.0, -5.0, grain_type="new_snow"),
            SnowLayer(0.8, 320.0, -2.0),
        ]
        with pytest.raises(ValueError, match="persistent grain types"):
            AvalancheDetector().compute_sk38(layers, weak_layer_index=1)

    def test_weak_layer_index_bounds(self) -> None:
        with pytest.raises(ValueError, match="weak_layer_index"):
            AvalancheDetector().compute_sk38(_slab_over_facets(), weak_layer_index=0)
        with pytest.raises(ValueError, match="weak_layer_index"):
            AvalancheDetector().compute_sk38(_slab_over_facets(), weak_layer_index=3)

    def test_layer_validation_fails_loud(self) -> None:
        with pytest.raises(ValueError, match="density"):
            SnowLayer(0.4, 5.0, -6.0)
        with pytest.raises(ValueError, match="thickness"):
            SnowLayer(-0.1, 200.0, -6.0)
        with pytest.raises(ValueError, match="temperature"):
            SnowLayer(0.4, 200.0, 12.0)


class TestNewSnowLoading:
    """Schweizer et al. (2003) critical loading + wind-slab multiplier."""

    def test_critical_constants_match_literature(self) -> None:
        assert CRITICAL_NEW_SNOW_24H_CM == 30.0
        assert TG_FACETING_THRESHOLD_K_M == 10.0

    def test_below_critical_calm(self) -> None:
        loading = AvalancheDetector().assess_new_snow_loading(15.0, 2.0)
        assert not loading["critical"]
        assert loading["skier_critical"]
        assert loading["multiplier"] == 1.0

    def test_wind_multiplier_promotes_to_critical(self) -> None:
        """15 cm HN24 with drift-capable wind => effective 30 cm: critical."""
        loading = AvalancheDetector().assess_new_snow_loading(15.0, 8.0)
        assert loading["wind_slab_active"]
        assert loading["effective_new_snow_24h_cm"] == pytest.approx(30.0)
        assert loading["critical"]

    def test_natural_critical_without_wind(self) -> None:
        loading = AvalancheDetector().assess_new_snow_loading(35.0, 0.0)
        assert loading["critical"]
        assert not loading["wind_slab_active"]

    def test_negative_inputs_raise(self) -> None:
        with pytest.raises(ValueError, match="new_snow_24h_cm"):
            AvalancheDetector().assess_new_snow_loading(-1.0)
        with pytest.raises(ValueError, match="wind_speed_10m_ms"):
            AvalancheDetector().assess_new_snow_loading(5.0, -3.0)


class TestTemperatureGradient:
    """Kinetic-growth metamorphism threshold (TG > 10 K/m)."""

    def test_strong_gradient_flags_faceting(self) -> None:
        # 6 K over ~0.21 m between mid-points => ~28 K/m
        layers = [SnowLayer(0.4, 200.0, -8.0), SnowLayer(0.02, 250.0, -2.0, "facets")]
        tg = AvalancheDetector().assess_temperature_gradient(layers)
        assert tg["max_gradient_k_m"] > TG_FACETING_THRESHOLD_K_M
        assert tg["faceting_risk"]

    def test_weak_gradient_no_flag(self) -> None:
        layers = [SnowLayer(0.5, 250.0, -2.0), SnowLayer(0.5, 300.0, -1.0)]
        tg = AvalancheDetector().assess_temperature_gradient(layers)
        assert tg["max_gradient_k_m"] < TG_FACETING_THRESHOLD_K_M
        assert not tg["faceting_risk"]

    def test_persistence_gate(self) -> None:
        layers = [SnowLayer(0.4, 200.0, -8.0), SnowLayer(0.02, 250.0, -2.0, "facets")]
        det = AvalancheDetector()
        assert not det.assess_temperature_gradient(layers, gradient_duration_days=1.0)[
            "faceting_risk"
        ]
        assert det.assess_temperature_gradient(layers, gradient_duration_days=5.0)["faceting_risk"]

    def test_single_layer_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 2 layers"):
            AvalancheDetector().assess_temperature_gradient([SnowLayer(0.4, 200.0, -8.0)])


class TestEAWSMapping:
    """Danger-level mapping and full prediction plumbing."""

    def test_calm_stable_pack_is_low(self) -> None:
        layers = [
            SnowLayer(1.0, 280.0, -3.0),
            SnowLayer(0.02, 450.0, -2.5, grain_type="facets"),
            SnowLayer(0.5, 400.0, -2.0),
        ]
        result = AvalancheDetector().predict_avalanche(layers=layers, weak_layer_index=1)
        assert result.danger_level == AvalancheDangerLevel.LOW.value
        assert not result.avalanche_likely

    def test_poor_stability_plus_critical_loading_is_very_high(self) -> None:
        layers = [
            SnowLayer(0.3, 150.0, -8.0),
            SnowLayer(0.02, 150.0, -7.0, grain_type="depth_hoar"),
            SnowLayer(0.6, 350.0, -2.0),
        ]
        result = AvalancheDetector().predict_avalanche(
            layers=layers,
            weak_layer_index=1,
            new_snow_24h_cm=40.0,
            wind_speed_10m_ms=10.0,
        )
        assert result.danger_level == AvalancheDangerLevel.VERY_HIGH.value
        assert result.avalanche_likely
        assert result.new_snow_loading_flag
        assert result.wind_slab_flag
        assert result.confidence > 0.5
        criteria = {e["criterion"] for e in result.evidence}
        assert {"sk38", "new_snow_loading", "temperature_gradient"} <= criteria

    def test_critical_loading_alone_is_considerable(self) -> None:
        result = AvalancheDetector().predict_avalanche(new_snow_24h_cm=35.0)
        assert result.danger_level == AvalancheDangerLevel.CONSIDERABLE.value
        assert result.sk38 is None  # no profile: no SK38, honestly absent

    def test_rain_on_snow_escalates(self) -> None:
        result = AvalancheDetector().predict_avalanche(rain_mm_24h=12.0, air_temperature_c=2.0)
        assert result.rain_on_snow_flag
        assert result.danger_level >= AvalancheDangerLevel.CONSIDERABLE.value

    def test_subzero_precipitation_is_not_rain(self) -> None:
        result = AvalancheDetector().predict_avalanche(rain_mm_24h=12.0, air_temperature_c=-4.0)
        assert not result.rain_on_snow_flag

    def test_faceting_alone_is_moderate(self) -> None:
        """Strong TG (~31 K/m) in an otherwise stable pack (SK38 good)."""
        layers = [
            SnowLayer(0.5, 250.0, -12.0),
            SnowLayer(0.02, 420.0, -4.0, grain_type="facets"),
            SnowLayer(0.5, 400.0, -2.0),
        ]
        result = AvalancheDetector().predict_avalanche(layers=layers, weak_layer_index=1)
        assert result.sk38 is not None and result.sk38.stability_class == "good"
        assert result.faceting_risk_flag
        assert result.danger_level == AvalancheDangerLevel.MODERATE.value

    def test_layers_without_weak_index_raise(self) -> None:
        with pytest.raises(ValueError, match="weak_layer_index is required"):
            AvalancheDetector().predict_avalanche(layers=_slab_over_facets())

    def test_negative_rain_raises(self) -> None:
        with pytest.raises(ValueError, match="rain_mm_24h"):
            AvalancheDetector().predict_avalanche(rain_mm_24h=-2.0)


class TestLandslideCarveOut:
    """landslide.py keeps soil scope and points here for avalanches."""

    def test_landslide_docstring_defers_to_avalanche_detector(self) -> None:
        from omni_mercury_engine.detectors.geological import landslide

        doc = landslide.__doc__ or ""
        assert "avalanche_detector" in doc
        assert "Snow avalanche forecasting no longer lives here" in doc
