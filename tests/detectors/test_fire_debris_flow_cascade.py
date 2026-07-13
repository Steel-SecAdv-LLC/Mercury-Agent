# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the wildfire → debris-flow cascade (published USGS models).

Published-coefficient checks:

- The embedded M1 coefficient table must equal USGS OFR 2016-1106 Table 1
  (= Staley et al. 2017 Table 2) verbatim.
- M1 algebra: p = 0.5 exactly at the inverted threshold accumulation; for a
  documented severely burned steep southern-California basin profile
  (T = 0.6, mean dNBR = 350, KF = 0.25) the 15-min threshold intensity is
  ~22 mm/h, inside the 15-30 mm/h range of objectively defined southern
  California post-fire thresholds (Staley et al. 2013, Landslides 10,
  547-562) that the M1 training region exhibits.
- Cannon et al. (2008): I = 12.5 D^-0.4 (southern California) and
  I = 9.5 D^-0.7 (Colorado) reproduced at spot durations.
- Gartner et al. (2014): ln V = 4.22 + 0.39 sqrt(i15) + 0.36 ln(Bmh) +
  0.13 sqrt(relief) recomputed independently in the test.
"""

from __future__ import annotations

import itertools
import math
from typing import TypedDict

import numpy as np
import pytest

# The dev venv's editable install may point at a sibling worktree that
# predates ``fire_debris_flow_cascade``; ``unused-ignore`` keeps a
# correctly installed tree (CI) clean.
from omni_mercury_engine.detectors.geological.fire_debris_flow_cascade import (  # type: ignore[import-not-found,unused-ignore]
    CANNON_ID_THRESHOLDS,
    GARTNER_2014,
    M1_COEFFICIENTS,
    FireDebrisFlowCascadeDetector,
    cannon_id_threshold_mm_h,
    gartner_2014_volume_m3,
    staley_m1_likelihood,
    staley_m1_threshold_intensity,
)


# Documented severely-burned steep SoCal basin profile used across tests.
# TypedDict so ``**SEVERE_BASIN`` unpacks type-safely into the M1 helpers'
# float-typed basin parameters (a plain dict[str, float] could also bind
# ``duration_min: int``, which mypy rejects).
class _BasinProfile(TypedDict):
    t_proportion: float
    dnbr_mean: float
    kf_factor: float


SEVERE_BASIN: _BasinProfile = {"t_proportion": 0.6, "dnbr_mean": 350.0, "kf_factor": 0.25}


class TestPublishedCoefficients:
    """Embedded coefficients equal the primary sources verbatim."""

    def test_m1_table_matches_ofr_2016_1106(self) -> None:
        assert M1_COEFFICIENTS[15] == {"beta": -3.63, "c1": 0.41, "c2": 0.67, "c3": 0.70}
        assert M1_COEFFICIENTS[30] == {"beta": -3.61, "c1": 0.26, "c2": 0.39, "c3": 0.50}
        assert M1_COEFFICIENTS[60] == {"beta": -3.21, "c1": 0.17, "c2": 0.20, "c3": 0.22}

    def test_cannon_curves_match_geomorphology_96(self) -> None:
        assert CANNON_ID_THRESHOLDS["southern_california"] == (12.5, -0.4)
        assert CANNON_ID_THRESHOLDS["colorado"] == (9.5, -0.7)

    def test_gartner_coefficients_match_engineering_geology_176(self) -> None:
        assert GARTNER_2014["intercept"] == 4.22
        assert GARTNER_2014["i15"] == 0.39
        assert GARTNER_2014["bmh"] == 0.36
        assert GARTNER_2014["relief"] == 0.13


class TestStaleyM1:
    """M1 likelihood and threshold algebra."""

    def test_likelihood_is_half_at_inverted_threshold(self) -> None:
        """p(R_0.5) = 0.5 exactly, for every published duration."""
        for duration in (15, 30, 60):
            intensity = staley_m1_threshold_intensity(
                **SEVERE_BASIN, likelihood=0.5, duration_min=duration
            )
            accum = intensity * duration / 60.0
            p = staley_m1_likelihood(**SEVERE_BASIN, rain_accum_mm=accum, duration_min=duration)
            assert p == pytest.approx(0.5, abs=1e-12), f"duration={duration}"

    def test_severe_socal_basin_threshold_in_published_range(self) -> None:
        """T=0.6, dNBR=350, KF=0.25: hand computation gives
        R = 3.63 / (0.41*0.6 + 0.67*0.35 + 0.70*0.25) = 5.538 mm / 15 min
        => 22.15 mm/h, inside the 15-30 mm/h published SoCal range."""
        intensity = staley_m1_threshold_intensity(**SEVERE_BASIN)
        expected = (3.63 / (0.41 * 0.6 + 0.67 * 0.35 + 0.70 * 0.25)) * 4.0
        assert intensity == pytest.approx(expected, rel=1e-12)
        assert intensity == pytest.approx(22.15, abs=0.05)
        assert 15.0 <= intensity <= 30.0

    def test_severe_basin_high_likelihood_at_40mm_h(self) -> None:
        """X = -3.63 + 10 mm * 0.6555 = 2.925 -> p = 0.949."""
        p = staley_m1_likelihood(**SEVERE_BASIN, rain_accum_mm=10.0)
        assert p == pytest.approx(1.0 / (1.0 + math.exp(-2.925)), rel=1e-9)
        assert p > 0.9

    def test_lightly_burned_basin_low_likelihood(self) -> None:
        p = staley_m1_likelihood(
            t_proportion=0.05, dnbr_mean=100.0, kf_factor=0.1, rain_accum_mm=6.0
        )
        assert p < 0.1

    def test_likelihood_monotone_in_rainfall(self) -> None:
        accums = np.linspace(0.0, 20.0, 41)
        probs = [staley_m1_likelihood(**SEVERE_BASIN, rain_accum_mm=a) for a in accums]
        assert all(b > a for a, b in itertools.pairwise(probs))

    def test_unsupported_duration_raises(self) -> None:
        with pytest.raises(ValueError, match="durations"):
            staley_m1_likelihood(**SEVERE_BASIN, rain_accum_mm=5.0, duration_min=10)

    def test_out_of_range_terrain_raises(self) -> None:
        with pytest.raises(ValueError, match="t_proportion"):
            staley_m1_likelihood(1.5, 350.0, 0.25, 5.0)
        with pytest.raises(ValueError, match="dnbr_mean"):
            staley_m1_likelihood(0.6, 5000.0, 0.25, 5.0)
        with pytest.raises(ValueError, match="kf_factor"):
            staley_m1_likelihood(0.6, 350.0, 0.9, 5.0)

    def test_unburned_basin_threshold_out_of_scope(self) -> None:
        with pytest.raises(ValueError, match="burned areas only"):
            staley_m1_threshold_intensity(0.0, 0.0, 0.0)


class TestCannonThresholds:
    """Cannon et al. (2008) I-D curves."""

    def test_socal_spot_values(self) -> None:
        assert cannon_id_threshold_mm_h(1.0) == pytest.approx(12.5)
        assert cannon_id_threshold_mm_h(0.25) == pytest.approx(12.5 * 0.25**-0.4, rel=1e-12)

    def test_colorado_spot_values(self) -> None:
        assert cannon_id_threshold_mm_h(1.0, "colorado") == pytest.approx(9.5)
        assert cannon_id_threshold_mm_h(2.0, "colorado") == pytest.approx(
            9.5 * 2.0**-0.7, rel=1e-12
        )

    def test_bad_inputs_raise(self) -> None:
        with pytest.raises(ValueError, match="duration_h"):
            cannon_id_threshold_mm_h(0.0)
        with pytest.raises(ValueError, match="region"):
            cannon_id_threshold_mm_h(1.0, "iceland")


class TestGartnerVolume:
    """Gartner et al. (2014) volume model."""

    def test_worked_example_recomputed_independently(self) -> None:
        """i15=24 mm/h, Bmh=1 km², relief=500 m."""
        volume, volume_class = gartner_2014_volume_m3(24.0, 1.0, 500.0)
        ln_v = 4.22 + 0.39 * math.sqrt(24.0) + 0.36 * math.log(1.0) + 0.13 * math.sqrt(500.0)
        assert volume == pytest.approx(math.exp(ln_v), rel=1e-12)
        assert 8000.0 < volume < 9000.0  # ~8.4e3 m³
        assert volume_class == 2

    def test_volume_class_bins(self) -> None:
        # Tiny basin, weak storm -> class 1 (< 1e3 m³).
        v_small, c_small = gartner_2014_volume_m3(1.0, 0.01, 50.0)
        assert v_small < 1e3 and c_small == 1
        # Large severe basin, intense storm -> class 4 (>= 1e5 m³).
        v_big, c_big = gartner_2014_volume_m3(60.0, 20.0, 1500.0)
        assert v_big >= 1e5 and c_big == 4

    def test_unburned_watershed_out_of_scope(self) -> None:
        with pytest.raises(ValueError, match="burned_mh_km2"):
            gartner_2014_volume_m3(24.0, 0.0, 500.0)


class TestRainfallReduction:
    """Rolling-accumulation storm metrics."""

    def test_peak_15min_accumulation_exact(self) -> None:
        # 5-min gauge: burst of 8+8+8 mm in three consecutive steps.
        rain = np.array([0.0, 0.0, 2.0, 8.0, 8.0, 8.0, 2.0, 0.0])
        analysis = FireDebrisFlowCascadeDetector().analyze_rainfall(rain, 5.0)
        assert analysis.peak_accum_mm[15] == pytest.approx(24.0)
        assert analysis.peak_intensity_mm_h[15] == pytest.approx(96.0)
        assert analysis.storm_duration_h == pytest.approx(25.0 / 60.0)
        assert analysis.storm_mean_intensity_mm_h == pytest.approx(28.0 / (25.0 / 60.0))

    def test_dry_series_raises(self) -> None:
        with pytest.raises(ValueError, match="entirely dry"):
            FireDebrisFlowCascadeDetector().analyze_rainfall(np.zeros(12), 5.0)

    def test_step_must_divide_15(self) -> None:
        with pytest.raises(ValueError, match="divide 15"):
            FireDebrisFlowCascadeDetector().analyze_rainfall(np.ones(10), 7.0)

    def test_negative_rain_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            FireDebrisFlowCascadeDetector().analyze_rainfall(np.array([1.0, -1.0]), 5.0)


class TestCascadeComposition:
    """Staged composition with transparent omissions."""

    def _storm(self) -> np.ndarray:
        """5-min gauge storm peaking at 24 mm / 15 min (96 mm/h i15)."""
        return np.array([0.0, 1.0, 2.0, 8.0, 8.0, 8.0, 2.0, 1.0, 0.0])

    def test_severe_basin_intense_storm_very_high(self) -> None:
        result = FireDebrisFlowCascadeDetector().assess(
            **SEVERE_BASIN,
            rain_mm=self._storm(),
            step_minutes=5.0,
            burned_mh_km2=2.5,
            relief_m=600.0,
        )
        assert result.likelihood > 0.9
        assert result.likelihood_class == "very_high"
        assert result.threshold_exceeded
        assert result.cannon_exceeded
        assert result.volume_m3 is not None and result.volume_m3 > 1e3
        assert result.volume_class in (2, 3, 4)
        assert result.volume_omitted_reason == ""
        stages = [e["stage"] for e in result.evidence]
        assert stages == [
            "burn_evidence",
            "rainfall",
            "m1_likelihood",
            "cannon_id_threshold",
            "gartner_volume",
        ]

    def test_volume_omitted_honestly_without_inputs(self) -> None:
        result = FireDebrisFlowCascadeDetector().assess(
            **SEVERE_BASIN, rain_mm=self._storm(), step_minutes=5.0
        )
        assert result.volume_m3 is None
        assert result.volume_class is None
        assert "does not guess" in result.volume_omitted_reason
        assert "burned_mh_km2" in result.volume_omitted_reason
        assert "relief_m" in result.volume_omitted_reason

    def test_light_drizzle_low_likelihood(self) -> None:
        drizzle = np.full(24, 0.1)  # 0.1 mm / 5 min = 1.2 mm/h for 2 h
        result = FireDebrisFlowCascadeDetector().assess(
            **SEVERE_BASIN, rain_mm=drizzle, step_minutes=5.0
        )
        assert result.likelihood < 0.25
        assert result.likelihood_class == "low"
        assert not result.threshold_exceeded
        assert not result.cannon_exceeded

    def test_wildfire_context_recorded_not_used(self) -> None:
        """The context flag lands in evidence; the likelihood is unchanged."""
        detector = FireDebrisFlowCascadeDetector()
        with_ctx = detector.assess(
            **SEVERE_BASIN,
            rain_mm=self._storm(),
            step_minutes=5.0,
            wildfire_context={"fire_detected": True},
        )
        without_ctx = detector.assess(**SEVERE_BASIN, rain_mm=self._storm(), step_minutes=5.0)
        assert with_ctx.likelihood == without_ctx.likelihood
        burn = with_ctx.evidence[0]["detail"]
        assert burn["wildfire_context"]["fire_detected"] is True
        assert "untrained" in burn["wildfire_context"]["note"]

    def test_colorado_region_uses_colorado_curve(self) -> None:
        detector = FireDebrisFlowCascadeDetector(cannon_region="colorado")
        result = detector.assess(**SEVERE_BASIN, rain_mm=self._storm(), step_minutes=5.0)
        expected = 9.5 * result.rainfall.storm_duration_h**-0.7
        assert result.cannon_threshold_mm_h == pytest.approx(expected, rel=1e-12)

    def test_bad_constructor_args_raise(self) -> None:
        with pytest.raises(ValueError, match="durations"):
            FireDebrisFlowCascadeDetector(primary_duration_min=45)
        with pytest.raises(ValueError, match="cannon_region"):
            FireDebrisFlowCascadeDetector(cannon_region="alps")

    def test_evidence_carries_citations(self) -> None:
        result = FireDebrisFlowCascadeDetector().assess(
            **SEVERE_BASIN, rain_mm=self._storm(), step_minutes=5.0
        )
        cited = [e for e in result.evidence if "citation" in e]
        assert any("OFR 2016-1106" in e["citation"] for e in cited)
        assert any("Cannon et al. 2008" in e["citation"] for e in cited)
