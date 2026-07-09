# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Winter / ice-storm detector: partial thickness, Stull wet bulb, FRAM-like
accretion, SPIA tiers, NWS blizzard criteria, and winter-alert wiring.

Formulation anchors:
* Partial-thickness thresholds (1300 m / 1540 m / 1560 m) per the
  Keeter & Cline (1991) lineage of operational thickness forecasting.
* Stull (2011) wet-bulb approximation, including the paper's worked
  example Tw(T=20 C, RH=50 %) = 13.7 C.
* Sanders & Barjenbruch (2016) mean flat-surface ILR (0.72) structure.
* NWS Glossary blizzard definition (>= 35 mph wind + < 1/4 mi visibility
  for >= 3 h).

NWS wiring tests use the recorded real fixtures in
``tests/fixtures/meteorological/`` (see PROVENANCE.json there): the July
winter-event query response is genuinely empty (api.weather.gov retention
does not reach the previous winter), and the populated-feed filter path is
exercised against the schema-identical recorded dust-warning feed plus
clearly-marked constructed unit inputs for the positive winter branch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from omni_mercury_engine.detectors.meteorological.winter_storm_detector import (
    PrecipType,
    WinterStormDetector,
)

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "meteorological"


@pytest.fixture(scope="module")
def detector() -> WinterStormDetector:
    """One physics-core detector for the whole module (stateless)."""
    return WinterStormDetector()


class TestPartialThickness:
    """Decision matrix on example profiles for each published quadrant."""

    def test_snow_cold_column(self, detector: WinterStormDetector) -> None:
        """Cold low levels + no melting layer aloft -> snow."""
        ptype = detector.precip_type_partial_thickness(
            thickness_1000_850_m=1285.0, thickness_850_700_m=1525.0
        )
        assert ptype == PrecipType.SNOW

    def test_sleet_partial_melting_band(self, detector: WinterStormDetector) -> None:
        """850-700 thickness in the 1540-1560 m band + cold low levels -> sleet."""
        ptype = detector.precip_type_partial_thickness(
            thickness_1000_850_m=1280.0, thickness_850_700_m=1550.0
        )
        assert ptype == PrecipType.SLEET

    def test_freezing_rain_warm_nose(self, detector: WinterStormDetector) -> None:
        """Deep warm layer aloft (> 1560 m) + cold low levels -> freezing rain."""
        ptype = detector.precip_type_partial_thickness(
            thickness_1000_850_m=1280.0, thickness_850_700_m=1570.0
        )
        assert ptype == PrecipType.FREEZING_RAIN

    def test_rain_warm_column(self, detector: WinterStormDetector) -> None:
        """Warm low levels -> rain regardless of the mid-level band."""
        for mid in (1525.0, 1550.0, 1575.0):
            ptype = detector.precip_type_partial_thickness(
                thickness_1000_850_m=1320.0, thickness_850_700_m=mid
            )
            assert ptype == PrecipType.RAIN

    def test_surface_temp_downgrades_freezing_rain(self, detector: WinterStormDetector) -> None:
        """A surface above 0 C cannot accrete ice: freezing rain -> rain."""
        ptype = detector.precip_type_partial_thickness(
            thickness_1000_850_m=1280.0,
            thickness_850_700_m=1570.0,
            surface_temp_c=2.5,
        )
        assert ptype == PrecipType.RAIN

    def test_heights_path_equivalent(self, detector: WinterStormDetector) -> None:
        """Supplying geopotential heights must match supplying thicknesses."""
        from_heights = detector.precip_type_partial_thickness(
            z1000_m=100.0, z850_m=1385.0, z700_m=2910.0
        )  # thicknesses: 1285 / 1525
        from_thickness = detector.precip_type_partial_thickness(
            thickness_1000_850_m=1285.0, thickness_850_700_m=1525.0
        )
        assert from_heights == from_thickness == PrecipType.SNOW

    def test_missing_inputs_fail_loud(self, detector: WinterStormDetector) -> None:
        with pytest.raises(ValueError, match="refusing to guess"):
            detector.precip_type_partial_thickness(thickness_1000_850_m=1285.0)
        with pytest.raises(ValueError, match="refusing to guess"):
            detector.precip_type_partial_thickness(z1000_m=100.0, z850_m=1385.0)

    def test_unphysical_thickness_fails_loud(self, detector: WinterStormDetector) -> None:
        """Feet-vs-metres unit mistakes must raise, not silently classify."""
        with pytest.raises(ValueError, match="plausible"):
            detector.precip_type_partial_thickness(
                thickness_1000_850_m=4215.0, thickness_850_700_m=5000.0
            )


class TestStullWetBulb:
    """Stull (2011) approximation and validity enforcement."""

    def test_paper_worked_example(self, detector: WinterStormDetector) -> None:
        """Stull (2011): T=20 C, RH=50 % -> Tw = 13.7 C."""
        assert detector.wet_bulb_stull(20.0, 50.0) == pytest.approx(13.7, abs=0.05)

    def test_wet_bulb_below_dry_bulb(self, detector: WinterStormDetector) -> None:
        for t, rh in [(0.0, 90.0), (10.0, 40.0), (-5.0, 80.0), (30.0, 20.0)]:
            assert detector.wet_bulb_stull(t, rh) <= t

    @pytest.mark.parametrize(("t", "rh"), [(-30.0, 50.0), (60.0, 50.0), (10.0, 2.0), (10.0, 100.0)])
    def test_outside_validity_fails_loud(
        self, detector: WinterStormDetector, t: float, rh: float
    ) -> None:
        with pytest.raises(ValueError, match="Stull"):
            detector.wet_bulb_stull(t, rh)

    def test_surface_precip_type(self, detector: WinterStormDetector) -> None:
        """Wet-bulb rain/snow split, including the above-freezing snow case.

        At T=+2 C / RH=60 % the wet bulb is about -1.2 C: the wet-bulb
        method correctly calls snow where a dry-bulb threshold would call
        rain.
        """
        assert detector.precip_type_surface(2.0, 60.0) == PrecipType.SNOW
        assert detector.precip_type_surface(5.0, 95.0) == PrecipType.RAIN

    def test_surface_method_never_emits_ice_types(self, detector: WinterStormDetector) -> None:
        """Documented limitation: surface-only data cannot see a warm nose."""
        for t in np.linspace(-15.0, 10.0, 11):
            ptype = detector.precip_type_surface(float(t), 80.0)
            assert ptype in (PrecipType.SNOW, PrecipType.RAIN)


class TestIceAccretion:
    """FRAM-like flat-surface accretion (S&B 2016 ILR structure)."""

    def test_cold_steady_event_full_ilr(self, detector: WinterStormDetector) -> None:
        """2 mm/h for 6 h at Tw=-3 C: 12 mm liquid x 0.72 ILR = 8.64 mm ice.

        In inches: liquid 0.47244, ice 0.34016.
        """
        result = detector.ice_accretion(
            precip_rate_mm_hr=2.0, wet_bulb_c=-3.0, duration_hr=6.0, wind_speed_mph=20.0
        )
        assert result.liquid_equivalent_in == pytest.approx(12.0 / 25.4, rel=1e-9)
        assert result.mean_ilr == pytest.approx(0.72, rel=1e-9)
        assert result.flat_ice_in == pytest.approx(12.0 * 0.72 / 25.4, rel=1e-9)
        # 0.34 in ice with 20 mph wind: SPIA band 15-25 mph, >= 0.25 in -> 2.
        assert result.spia_index == 2

    def test_no_accretion_above_freezing_wet_bulb(self, detector: WinterStormDetector) -> None:
        result = detector.ice_accretion(
            precip_rate_mm_hr=5.0, wet_bulb_c=0.5, duration_hr=4.0, wind_speed_mph=10.0
        )
        assert result.flat_ice_in == 0.0
        assert result.spia_index == 0

    def test_near_zero_wet_bulb_reduces_ilr(self, detector: WinterStormDetector) -> None:
        """S&B: accretion efficiency collapses as the wet bulb nears 0 C."""
        near = detector.ice_accretion(2.0, -0.25, 6.0, 10.0)
        cold = detector.ice_accretion(2.0, -3.0, 6.0, 10.0)
        assert near.mean_ilr < cold.mean_ilr
        assert near.mean_ilr == pytest.approx(0.72 * 0.25, rel=1e-9)

    def test_heavy_rate_reduces_ilr(self, detector: WinterStormDetector) -> None:
        """S&B: heavy precipitation sheds liquid before it freezes."""
        heavy = detector.ice_accretion(10.0, -3.0, 1.0, 10.0)
        light = detector.ice_accretion(2.0, -3.0, 5.0, 10.0)
        assert heavy.mean_ilr < light.mean_ilr

    def test_interval_arrays(self, detector: WinterStormDetector) -> None:
        """Per-interval series accumulate; warm intervals contribute nothing."""
        result = detector.ice_accretion(
            precip_rate_mm_hr=[2.0, 2.0, 2.0],
            wet_bulb_c=[-3.0, 0.5, -3.0],
            duration_hr=[2.0, 2.0, 2.0],
            wind_speed_mph=10.0,
        )
        assert result.flat_ice_in == pytest.approx(8.0 * 0.72 / 25.4, rel=1e-9)

    def test_fail_loud_inputs(self, detector: WinterStormDetector) -> None:
        with pytest.raises(ValueError):
            detector.ice_accretion([2.0, 3.0], [-3.0], [1.0, 2.0, 3.0], 10.0)
        with pytest.raises(ValueError):
            detector.ice_accretion(-1.0, -3.0, 2.0, 10.0)
        with pytest.raises(ValueError):
            detector.ice_accretion(2.0, np.nan, 2.0, 10.0)
        with pytest.raises(ValueError):
            detector.ice_accretion(2.0, -3.0, 0.0, 10.0)
        with pytest.raises(ValueError):
            detector.ice_accretion(2.0, -3.0, 2.0, None)


class TestSPIATiers:
    """Sperry-Piltz matrix spot checks across wind bands."""

    @pytest.mark.parametrize(
        ("ice", "wind", "expected"),
        [
            (0.05, 10.0, 0),
            (0.05, 30.0, 0),
            (0.30, 10.0, 1),
            (0.30, 20.0, 2),
            (0.30, 30.0, 3),
            (0.60, 10.0, 2),
            (0.60, 20.0, 3),
            (0.60, 30.0, 4),
            (0.80, 30.0, 5),
            (1.20, 20.0, 5),
            (1.60, 5.0, 5),
        ],
    )
    def test_matrix(
        self, detector: WinterStormDetector, ice: float, wind: float, expected: int
    ) -> None:
        index, description = detector.spia_tier(ice, wind)
        assert index == expected
        assert description

    def test_fail_loud(self, detector: WinterStormDetector) -> None:
        with pytest.raises(ValueError):
            detector.spia_tier(-0.1, 10.0)
        with pytest.raises(ValueError):
            detector.spia_tier(float("nan"), 10.0)


class TestBlizzardCriteria:
    """NWS definition: >= 35 mph + < 400 m visibility for >= 3 h."""

    @staticmethod
    def _series(hours: float, step_min: float = 30.0) -> np.ndarray:
        return np.arange(0.0, hours * 3600.0 + 1.0, step_min * 60.0)

    def test_qualifying_event(self, detector: WinterStormDetector) -> None:
        t = self._series(4.0)
        wind = np.full(t.size, 18.0)
        vis = np.full(t.size, 300.0)
        result = detector.check_blizzard_criteria(t, wind, vis)
        assert result.blizzard
        assert result.longest_qualifying_hours >= 3.0

    def test_too_short_event(self, detector: WinterStormDetector) -> None:
        t = self._series(2.5)
        wind = np.full(t.size, 20.0)
        vis = np.full(t.size, 200.0)
        result = detector.check_blizzard_criteria(t, wind, vis)
        assert not result.blizzard

    def test_gap_resets_the_run(self, detector: WinterStormDetector) -> None:
        """4 h of wind/vis with a mid-event lull: no contiguous 3 h stretch."""
        t = self._series(4.0)
        wind = np.full(t.size, 18.0)
        vis = np.full(t.size, 300.0)
        wind[t.size // 2] = 5.0  # lull below 35 mph at the 2 h mark
        result = detector.check_blizzard_criteria(t, wind, vis)
        assert not result.blizzard

    def test_wind_alone_or_visibility_alone_insufficient(
        self, detector: WinterStormDetector
    ) -> None:
        t = self._series(4.0)
        result_wind_only = detector.check_blizzard_criteria(
            t, np.full(t.size, 20.0), np.full(t.size, 2000.0)
        )
        result_vis_only = detector.check_blizzard_criteria(
            t, np.full(t.size, 10.0), np.full(t.size, 200.0)
        )
        assert not result_wind_only.blizzard
        assert not result_vis_only.blizzard

    def test_fail_loud_series(self, detector: WinterStormDetector) -> None:
        t = self._series(4.0)
        with pytest.raises(ValueError):
            detector.check_blizzard_criteria(t, np.full(t.size - 1, 18.0), np.full(t.size, 300.0))
        bad_t = t.copy()
        bad_t[3] = bad_t[2]  # non-monotonic
        with pytest.raises(ValueError):
            detector.check_blizzard_criteria(bad_t, np.full(t.size, 18.0), np.full(t.size, 300.0))
        wind = np.full(t.size, 18.0)
        wind[0] = np.nan
        with pytest.raises(ValueError):
            detector.check_blizzard_criteria(t, wind, np.full(t.size, 300.0))


class TestNWSWinterWiring:
    """Alert filtering over recorded real feeds + constructed unit inputs."""

    def test_recorded_july_winter_query_is_empty(self, detector: WinterStormDetector) -> None:
        """The recorded July response genuinely has zero winter alerts."""
        with open(FIXTURE_DIR / "nws_winter_alerts_query_july.json") as f:
            payload = json.load(f)
        result = detector.cross_check_nws_alerts(payload)
        assert result["n_winter_alerts"] == 0
        assert result["events"] == []
        assert not result["ice_storm_warned"]

    def test_recorded_dust_feed_matches_no_winter_products(
        self, detector: WinterStormDetector
    ) -> None:
        """Schema-identical populated CAP feed: filter must reject all."""
        with open(FIXTURE_DIR / "nws_dust_storm_warnings.json") as f:
            payload = json.load(f)
        result = detector.cross_check_nws_alerts(payload)
        assert result["n_winter_alerts"] == 0

    def test_constructed_winter_products_are_matched(self, detector: WinterStormDetector) -> None:
        """Positive filter branch on constructed unit inputs (not recorded
        data: no real winter alert exists in the July retention window)."""
        records = [
            {"event": "Ice Storm Warning", "severity": "Severe"},
            {"event": "Blizzard Warning", "severity": "Severe"},
            {"event": "Winter Weather Advisory", "severity": "Minor"},
            {"event": "Severe Thunderstorm Warning", "severity": "Severe"},
        ]
        result = detector.cross_check_nws_alerts(records)
        assert result["n_winter_alerts"] == 3
        assert result["ice_storm_warned"]
        assert result["blizzard_warned"]
        assert not result["winter_storm_warned"]

    def test_bad_payload_fails_loud(self, detector: WinterStormDetector) -> None:
        with pytest.raises(TypeError):
            detector.cross_check_nws_alerts("not alerts")


class TestExtractFeatures:
    """Fusion feature interface."""

    def test_dict_thickness_and_accretion_paths(self, detector: WinterStormDetector) -> None:
        features = detector.extract_features(
            {
                "thickness_1000_850_m": 1280.0,
                "thickness_850_700_m": 1570.0,
                "precip_rate_mm_hr": 2.0,
                "wet_bulb_c": -3.0,
                "duration_hr": 6.0,
                "wind_speed_mph": 20.0,
            }
        )
        assert isinstance(features, torch.Tensor)
        assert features.shape == (WinterStormDetector.FEATURE_DIM,)
        assert torch.isfinite(features).all()
        # One-hot: freezing rain is index 2 of [snow, sleet, zr, rain].
        assert features[2].item() == 1.0

    def test_dict_without_physics_inputs_fails_loud(self, detector: WinterStormDetector) -> None:
        with pytest.raises(ValueError, match="no recognized"):
            detector.extract_features({"unrelated": 1.0})

    def test_array_path(self, detector: WinterStormDetector) -> None:
        features = detector.extract_features(np.linspace(-5.0, 5.0, 21))
        assert features.shape == (WinterStormDetector.FEATURE_DIM,)
        assert features[0].item() == pytest.approx(0.0, abs=1e-6)
