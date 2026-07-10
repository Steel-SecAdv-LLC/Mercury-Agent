# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hail / severe-convective detector: SHIP formulation, tiers, NWS wiring.

SHIP expectations are hand-computed from the SPC formulation
(SHIP = MUCAPE * MUMR * LR75 * (-T500) * SHR6 / 44e6 with the SPC clamps
and low-end corrections; https://www.spc.noaa.gov/exper/mesoanalysis/help/help_sigh.html).

NWS wiring tests run against the recorded real alert fixture
``tests/fixtures/meteorological/nws_severe_thunderstorm_warnings.json``
(provenance: tests/fixtures/meteorological/PROVENANCE.json — verbatim
api.weather.gov response recorded 2026-07-09).
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
# predates these severe-storm modules; ``unused-ignore`` keeps a
# correctly installed tree (CI) clean.
from omni_mercury_engine.detectors.meteorological.hail_detector import (  # type: ignore[import-not-found,unused-ignore]
    HailDetector,
)
from omni_mercury_engine.detectors.meteorological.severe_storm_alerts import (  # type: ignore[import-not-found,unused-ignore]
    filter_alerts_by_event,
    normalize_alert_records,
    parse_max_hail_size_in,
    parse_max_wind_gust_mph,
    parse_threat_tag,
)

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "meteorological"

#: A strong significant-hail environment (all SHIP components inside the
#: SPC clamps, no low-end corrections active).
STRONG_ENV = {
    "mucape_j_kg": 3500.0,
    "mu_mixing_ratio_g_kg": 13.6,
    "lapse_rate_700_500_c_km": 8.0,
    "temp_500_c": -12.0,
    "shear_0_6km_ms": 25.0,
    "freezing_level_m": 3800.0,
}


@pytest.fixture(scope="module")
def detector() -> HailDetector:
    """One physics-core detector for the whole module (stateless)."""
    return HailDetector()


@pytest.fixture(scope="module")
def svr_alerts() -> dict[str, Any]:
    """Recorded real Severe Thunderstorm Warning FeatureCollection."""
    with open(FIXTURE_DIR / "nws_severe_thunderstorm_warnings.json") as f:
        alerts: dict[str, Any] = json.load(f)
    return alerts


class TestShipFormulation:
    """SHIP against hand-computed values from the SPC formulation."""

    def test_ship_strong_environment_hand_computed(self, detector: HailDetector) -> None:
        """No clamps active: SHIP = 3500*13.6*8*12*25 / 44e6 = 2.59636..."""
        result = detector.compute_ship(**STRONG_ENV)
        expected = (3500.0 * 13.6 * 8.0 * 12.0 * 25.0) / 44_000_000.0
        assert result.ship == pytest.approx(expected, rel=1e-12)
        assert result.ship == pytest.approx(2.5963636, abs=1e-6)
        assert result.low_cape_factor == 1.0
        assert result.low_lapse_factor == 1.0
        assert result.low_fzl_factor == 1.0

    def test_ship_significant_threshold_sanity(self, detector: HailDetector) -> None:
        """SPC: SHIP > 1 discriminates significant-hail environments.

        A strong environment must land above 1; a modest one below.
        """
        assert detector.compute_ship(**STRONG_ENV).ship > 1.0
        modest = detector.compute_ship(
            mucape_j_kg=2000.0,
            mu_mixing_ratio_g_kg=13.0,
            lapse_rate_700_500_c_km=7.0,
            temp_500_c=-10.0,
            shear_0_6km_ms=20.0,
            freezing_level_m=3500.0,
        )
        # Hand-computed: 2000*13*7*10*20/44e6 = 0.827272...
        assert modest.ship == pytest.approx(0.8272727, abs=1e-6)
        assert modest.ship < 1.0

    def test_shear_clamped_to_7_27_ms(self, detector: HailDetector) -> None:
        """SPC confines the 0-6 km shear term to 7-27 m/s."""
        high = detector.compute_ship(**{**STRONG_ENV, "shear_0_6km_ms": 40.0})
        assert high.shear_ms == 27.0
        low = detector.compute_ship(**{**STRONG_ENV, "shear_0_6km_ms": 3.0})
        assert low.shear_ms == 7.0
        # Clamping means shear 40 and shear 27 give identical SHIP.
        ref = detector.compute_ship(**{**STRONG_ENV, "shear_0_6km_ms": 27.0})
        assert high.ship == pytest.approx(ref.ship, rel=1e-12)

    def test_mixing_ratio_clamped_to_11_13p6(self, detector: HailDetector) -> None:
        """SPC confines the MU mixing ratio term to 11-13.6 g/kg."""
        rich = detector.compute_ship(**{**STRONG_ENV, "mu_mixing_ratio_g_kg": 16.0})
        assert rich.mixing_ratio_g_kg == 13.6
        dry = detector.compute_ship(**{**STRONG_ENV, "mu_mixing_ratio_g_kg": 8.0})
        assert dry.mixing_ratio_g_kg == 11.0

    def test_t500_warm_limit(self, detector: HailDetector) -> None:
        """SPC sets T500 warmer than -5.5 C to -5.5 C."""
        warm = detector.compute_ship(**{**STRONG_ENV, "temp_500_c": -3.0})
        assert warm.temp_500_c == -5.5
        cold = detector.compute_ship(**{**STRONG_ENV, "temp_500_c": -20.0})
        assert cold.temp_500_c == -20.0

    def test_low_cape_correction(self, detector: HailDetector) -> None:
        """MUCAPE < 1300 J/kg scales SHIP by MUCAPE/1300."""
        result = detector.compute_ship(**{**STRONG_ENV, "mucape_j_kg": 650.0})
        assert result.low_cape_factor == pytest.approx(0.5)
        expected = (650.0 * 13.6 * 8.0 * 12.0 * 25.0) / 44_000_000.0 * 0.5
        assert result.ship == pytest.approx(expected, rel=1e-12)

    def test_low_lapse_correction(self, detector: HailDetector) -> None:
        """LR75 < 5.8 C/km scales SHIP by LR75/5.8."""
        result = detector.compute_ship(**{**STRONG_ENV, "lapse_rate_700_500_c_km": 2.9})
        assert result.low_lapse_factor == pytest.approx(0.5)

    def test_low_freezing_level_correction(self, detector: HailDetector) -> None:
        """FZL < 2400 m scales SHIP by FZL/2400."""
        result = detector.compute_ship(**{**STRONG_ENV, "freezing_level_m": 1200.0})
        assert result.low_fzl_factor == pytest.approx(0.5)

    @pytest.mark.parametrize(
        ("key", "bad"),
        [
            ("mucape_j_kg", float("nan")),
            ("mucape_j_kg", -100.0),
            ("mu_mixing_ratio_g_kg", 0.0),
            ("lapse_rate_700_500_c_km", -1.0),
            ("temp_500_c", 30.0),
            ("shear_0_6km_ms", float("inf")),
            ("freezing_level_m", -5.0),
        ],
    )
    def test_invalid_inputs_fail_loud(self, detector: HailDetector, key: str, bad: float) -> None:
        """Non-finite / unphysical inputs raise instead of degrading."""
        with pytest.raises(ValueError):
            detector.compute_ship(**{**STRONG_ENV, key: bad})


class TestTierLadder:
    """Documented SHIP / CAPE-shear tier ladder."""

    def test_extreme(self, detector: HailDetector) -> None:
        assert detector.classify_tier(4.5, 5000.0, 30.0) == "extreme"

    def test_significant_likely(self, detector: HailDetector) -> None:
        assert detector.classify_tier(1.7, 3000.0, 25.0) == "significant_hail_likely"

    def test_significant_favorable(self, detector: HailDetector) -> None:
        assert detector.classify_tier(1.1, 2500.0, 20.0) == "significant_hail_favorable"

    def test_large_hail_possible_needs_supercell_shear(self, detector: HailDetector) -> None:
        assert detector.classify_tier(0.5, 1500.0, 20.0) == "large_hail_possible"
        # Same CAPE without supercell-capable shear: only marginal.
        assert detector.classify_tier(0.5, 1500.0, 10.0) == "marginal"

    def test_none_below_marginal_cape(self, detector: HailDetector) -> None:
        assert detector.classify_tier(0.0, 300.0, 25.0) == "none"

    def test_tier_nan_fails_loud(self, detector: HailDetector) -> None:
        with pytest.raises(ValueError):
            detector.classify_tier(float("nan"), 1000.0, 20.0)


class TestAssess:
    """Full assessment path."""

    def test_assess_strong_environment(self, detector: HailDetector) -> None:
        result = detector.assess(dict(STRONG_ENV))
        assert result.significant_hail_favorable
        assert result.supercell_capable_shear
        assert result.tier == "significant_hail_likely"
        assert result.nws_cross_check is None

    def test_assess_missing_key_fails_loud(self, detector: HailDetector) -> None:
        data = dict(STRONG_ENV)
        del data["freezing_level_m"]
        with pytest.raises(ValueError, match="freezing_level_m"):
            detector.assess(data)

    def test_assess_with_recorded_alerts(
        self, detector: HailDetector, svr_alerts: dict[str, Any]
    ) -> None:
        result = detector.assess({**STRONG_ENV, "nws_alerts": svr_alerts})
        assert result.nws_cross_check is not None
        assert result.nws_cross_check["n_warnings"] == 25


class TestNWSCrossCheck:
    """Wiring against the recorded real Severe Thunderstorm Warning feed."""

    def test_counts_and_max_hail_size(
        self, detector: HailDetector, svr_alerts: dict[str, Any]
    ) -> None:
        """The recorded feed has 25 warnings, largest tagged hail 1.00 in."""
        result = detector.cross_check_nws_alerts(svr_alerts)
        assert result["n_warnings"] == 25
        assert result["max_hail_size_in"] == pytest.approx(1.0)
        assert result["severe_hail_warned"] is True
        assert result["significant_hail_warned"] is False
        assert "RADAR INDICATED" in result["hail_threat_tags"]

    def test_accepts_feature_list_and_property_dicts(
        self, detector: HailDetector, svr_alerts: dict[str, Any]
    ) -> None:
        """FeatureCollection, feature list, and flat property dicts agree."""
        from_collection = detector.cross_check_nws_alerts(svr_alerts)
        from_features = detector.cross_check_nws_alerts(svr_alerts["features"])
        from_props = detector.cross_check_nws_alerts(
            [f["properties"] for f in svr_alerts["features"]]
        )
        assert from_collection == from_features == from_props

    def test_accepts_datapoint_like_objects(self, detector: HailDetector) -> None:
        """NWSWeatherAlertsSource DataPoints expose a .data property dict."""

        class _FakeDataPoint:
            def __init__(self, data: dict[str, Any]) -> None:
                self.data = data

        points = [
            _FakeDataPoint({"event": "Severe Thunderstorm Warning", "severity": "Severe"}),
            _FakeDataPoint({"event": "Flood Warning", "severity": "Moderate"}),
        ]
        result = detector.cross_check_nws_alerts(points)
        assert result["n_warnings"] == 1
        assert result["max_hail_size_in"] is None

    def test_unrecognized_payload_fails_loud(self, detector: HailDetector) -> None:
        with pytest.raises(TypeError):
            detector.cross_check_nws_alerts(42)
        with pytest.raises(TypeError):
            detector.cross_check_nws_alerts({"not_features": []})
        with pytest.raises(TypeError):
            detector.cross_check_nws_alerts([{"neither": 1}])


class TestAlertParsers:
    """Shared CAP parsers on recorded and constructed unit inputs."""

    def test_parse_up_to_string(self) -> None:
        """The recorded feed contains 'Up to .75' maxHailSize strings."""
        record = {
            "event": "Severe Thunderstorm Warning",
            "parameters": {"maxHailSize": ["Up to .75"]},
        }
        assert parse_max_hail_size_in(record) == pytest.approx(0.75)

    def test_parse_description_fallback(self) -> None:
        """IBW text tags in `description` are a documented fallback channel.

        (Constructed unit input: the recorded API fixtures carry the value
        in `parameters`, so the fallback needs a text-tag example.)
        """
        record = {
            "event": "Severe Thunderstorm Warning",
            "description": "HAIL THREAT...OBSERVED; MAX HAIL SIZE...1.75 IN",
        }
        assert parse_max_hail_size_in(record) == pytest.approx(1.75)

    def test_parse_wind_gust_from_recorded_feed(self, svr_alerts: dict[str, Any]) -> None:
        records = normalize_alert_records(svr_alerts)
        gusts = [g for g in (parse_max_wind_gust_mph(r) for r in records) if g is not None]
        assert gusts, "recorded feed should carry maxWindGust tags"
        assert all(20.0 <= g <= 120.0 for g in gusts)

    def test_threat_tag_and_filter(self, svr_alerts: dict[str, Any]) -> None:
        records = normalize_alert_records(svr_alerts)
        warnings = filter_alerts_by_event(records, ["Severe Thunderstorm Warning"])
        assert len(warnings) == 25
        assert filter_alerts_by_event(records, ["Winter Storm Warning"]) == []
        tags = {parse_threat_tag(r, "hailThreat") for r in warnings}
        assert "RADAR INDICATED" in tags


class TestExtractFeatures:
    """Fusion feature interface."""

    def test_dict_path_physics_features(self, detector: HailDetector) -> None:
        features = detector.extract_features(dict(STRONG_ENV))
        assert isinstance(features, torch.Tensor)
        assert features.shape == (HailDetector.FEATURE_DIM,)
        assert torch.isfinite(features).all()
        assert features[0].item() == pytest.approx(2.5963636, abs=1e-4)  # SHIP

    def test_array_path_summary_stats(self, detector: HailDetector) -> None:
        arr = np.linspace(0.0, 10.0, 50)
        features = detector.extract_features(arr)
        assert features.shape == (HailDetector.FEATURE_DIM,)
        assert features[0].item() == pytest.approx(5.0, abs=1e-5)  # mean

    def test_empty_or_nonfinite_array_fails_loud(self, detector: HailDetector) -> None:
        with pytest.raises(ValueError):
            detector.extract_features(np.array([]))
        with pytest.raises(ValueError):
            detector.extract_features(np.array([1.0, math.nan]))
