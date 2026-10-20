# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioral tests for the humanitarian detectors.

Every test pins exact outputs against the modules' documented baselines
(z-scores, thresholds, severities, recommendations), replacing the
previous instantiation-only checks. Values are hand-computed from the
shipped baselines: agrifood yield 5.0 +/- 1.0 and price 100 +/- 15,
climate temperature 15 +/- 10 and precipitation 66.2 +/- 13.5 (the measured
NOAA 1991-2020 CONUS monthly climatology), economic
market 100 +/- 15 and unemployment 5 +/- 2, education achievement
75 +/- 10, neural activity 50 +/- 15.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from omni_mercury_engine.infrastructure.humanitarian import (
    AgriFoodSecurityDetector,
    ClimateResilienceDetector,
    EconomicResilienceDetector,
    EducationEquityDetector,
    NeuroscienceDetector,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class TestAgriFoodSecurityDetector:
    """Yield and price anomalies against the documented baselines."""

    def test_yield_failure_two_sigma(self) -> None:
        """A 2 t/ha yield is a high-severity crop failure at z = -3."""
        result = AgriFoodSecurityDetector().detect_yield_anomaly(
            np.array([2.0, 2.0]), crop_type="wheat"
        )

        assert result["anomaly_detected"]
        assert result["threat_type"] == "crop_failure"
        assert result["severity"] == "high"
        assert result["metrics"]["mean_yield_tons_per_hectare"] == pytest.approx(2.0)
        assert result["metrics"]["yield_deficit_pct"] == pytest.approx(60.0)
        assert result["metrics"]["z_score"] == pytest.approx(-3.0)
        assert result["food_insecurity_risk"] == "acute_shortage"
        assert result["affected_population_estimate"] == 500000
        assert result["recommendations"] == [
            "Activate emergency food reserves",
            "Coordinate with WFP for food assistance",
            "Focus interventions on wheat production",
        ]

    def test_yield_collapse_is_critical_famine_risk(self) -> None:
        """A 1 t/ha yield (z = -4) is critical with famine risk."""
        result = AgriFoodSecurityDetector().detect_yield_anomaly(np.array([1.0]))

        assert result["severity"] == "critical"
        assert result["food_insecurity_risk"] == "famine_risk"
        assert result["affected_population_estimate"] == 1000000
        assert result["recommendations"] == [
            "Activate emergency food reserves",
            "Coordinate with WFP for food assistance",
            "URGENT: Declare agricultural emergency",
            "Begin emergency food distribution",
        ]

    def test_yield_at_exact_two_sigma_boundary_is_not_flagged(self) -> None:
        """z = -2 exactly stays below the strict detection threshold."""
        result = AgriFoodSecurityDetector().detect_yield_anomaly(np.array([3.0]))

        assert not result["anomaly_detected"]
        assert result["threat_type"] == "normal"
        assert result["severity"] == "low"
        assert result["food_insecurity_risk"] == "moderate_shortage"
        assert result["affected_population_estimate"] == 0

    def test_normal_yield(self) -> None:
        """A baseline yield reports adequate supply and routine monitoring."""
        result = AgriFoodSecurityDetector().detect_yield_anomaly(np.array([5.0]))

        assert not result["anomaly_detected"]
        assert result["food_insecurity_risk"] == "adequate"
        assert result["recommendations"] == ["Continue routine agricultural monitoring"]

    def test_price_spike_high_and_critical(self) -> None:
        """Prices at z = 3 grade high; above z = 3 grade critical and severe."""
        detector = AgriFoodSecurityDetector()

        high = detector.detect_price_anomaly(np.array([145.0]))
        assert high["anomaly_detected"]
        assert high["threat_type"] == "price_spike"
        assert high["severity"] == "high"
        assert high["metrics"]["price_increase_pct"] == pytest.approx(45.0)
        assert high["metrics"]["z_score"] == pytest.approx(3.0)
        assert high["affordability_impact"] == "moderate"
        assert high["recommendations"] == [
            "Implement price stabilization measures",
            "Release strategic food reserves",
            "Provide subsidies for vulnerable populations",
        ]

        critical = detector.detect_price_anomaly(np.array([160.0]))
        assert critical["severity"] == "critical"
        assert critical["affordability_impact"] == "severe"

    def test_empty_data_returns_bare_negative(self) -> None:
        """Empty inputs return exactly the bare negative result."""
        detector = AgriFoodSecurityDetector()
        assert detector.detect_yield_anomaly(np.array([])) == {"anomaly_detected": False}
        assert detector.detect_price_anomaly(np.array([])) == {"anomaly_detected": False}

    def test_detect_routes_by_type(self) -> None:
        """detect() routes 'price' to the price path and defaults to yield."""
        detector = AgriFoodSecurityDetector()

        price = detector.detect(np.array([100.0]), detection_type="price")
        assert "current_price_usd" in price["metrics"]

        default = detector.detect(np.array([5.0]))
        assert "mean_yield_tons_per_hectare" in default["metrics"]

        # soil_health is implemented: it routes to the FAO-threshold path.
        soil = detector.detect(np.array([2.5, 6.5, 1.3, 1.0]), detection_type="soil_health")
        assert "thresholds_breached" in soil["metrics"]

        # An unknown type still falls back to the yield path (shipped routing).
        fallback = detector.detect(np.array([5.0]), detection_type="unknown_type")
        assert "mean_yield_tons_per_hectare" in fallback["metrics"]

    def test_price_threshold_boundaries(self) -> None:
        """The spike cut is strictly above baseline + 2 sigma = 130."""
        detector = AgriFoodSecurityDetector()

        at_cut = detector.detect_price_anomaly(np.array([130.0]))
        assert at_cut["threat_type"] == "normal"
        assert at_cut["severity"] == "low"
        assert not at_cut["anomaly_detected"]
        assert at_cut["metrics"]["z_score"] == pytest.approx(2.0)

        below_cut = detector.detect_price_anomaly(np.array([120.0]))
        assert below_cut["threat_type"] == "normal"
        assert not below_cut["anomaly_detected"]

    def test_degenerate_baselines_stay_finite(self) -> None:
        """Zero- and sub-unit baselines exercise every division guard."""
        detector = AgriFoodSecurityDetector()

        detector.yield_baseline = {"mean": 5.0, "std": 0.0}
        zero_std = detector.detect_yield_anomaly(np.array([4.0]))
        assert zero_std["metrics"]["z_score"] == pytest.approx(0.0)
        assert not zero_std["anomaly_detected"]

        detector.yield_baseline = {"mean": 1.0, "std": 0.5}
        sub_unit = detector.detect_yield_anomaly(np.array([0.5]))
        assert sub_unit["metrics"]["z_score"] == pytest.approx(-1.0)
        assert sub_unit["metrics"]["yield_deficit_pct"] == pytest.approx(50.0)

        detector.yield_baseline = {"mean": 0.0, "std": 1.0}
        zero_mean = detector.detect_yield_anomaly(np.array([0.0]))
        assert zero_mean["metrics"]["yield_deficit_pct"] == pytest.approx(0.0)

        detector.price_baseline = {"mean": 1.0, "std": 0.0}
        price_sub_unit = detector.detect_price_anomaly(np.array([0.5]))
        assert price_sub_unit["metrics"]["z_score"] == pytest.approx(0.0)
        assert price_sub_unit["metrics"]["price_increase_pct"] == pytest.approx(-50.0)

        detector.price_baseline = {"mean": 0.0, "std": 1.0}
        price_zero_mean = detector.detect_price_anomaly(np.array([0.0]))
        assert price_zero_mean["metrics"]["price_increase_pct"] == pytest.approx(0.0)

        detector.price_baseline = {"mean": 100.0, "std": 0.5}
        price_sub_unit_std = detector.detect_price_anomaly(np.array([100.25]))
        assert price_sub_unit_std["metrics"]["z_score"] == pytest.approx(0.5)


class TestClimateResilienceDetector:
    """Temperature and precipitation events against the documented baselines."""

    def test_heatwave_is_critical(self) -> None:
        """A 50 C reading (baseline 15 +/- 10) is a critical heatwave."""
        result = ClimateResilienceDetector().detect_temperature_anomaly(np.array([50.0, 20.0]))

        assert result["anomaly_detected"]
        assert result["event_type"] == "heatwave"
        assert result["severity"] == "critical"
        assert result["metrics"]["max_temperature_c"] == pytest.approx(50.0)
        assert result["metrics"]["z_score"] == pytest.approx(2.0)
        assert result["disaster_risk"] == "imminent"
        assert result["recommendations"] == [
            "Activate heat emergency protocols",
            "Open cooling centers for vulnerable populations",
            "URGENT: Issue heat warnings, restrict outdoor activities",
        ]

    def test_cold_snap_is_not_a_heatwave(self) -> None:
        """A -20 C reading is a cold snap with cold-weather guidance.

        Regression: the cold branch previously returned 'heatwave' and
        heat-emergency recommendations (cooling centers) for extreme cold.
        """
        result = ClimateResilienceDetector().detect_temperature_anomaly(np.array([-20.0, 10.0]))

        assert result["anomaly_detected"]
        assert result["event_type"] == "cold_snap"
        assert result["severity"] == "high"
        assert result["disaster_risk"] == "elevated"
        assert result["recommendations"] == [
            "Activate cold weather emergency protocols",
            "Open warming centers for vulnerable populations",
        ]

    def test_normal_temperature(self) -> None:
        """A 25 C mean (z = 1) is normal with routine monitoring."""
        result = ClimateResilienceDetector().detect_temperature_anomaly(np.array([25.0, 25.0]))

        assert not result["anomaly_detected"]
        assert result["event_type"] == "normal"
        assert result["severity"] == "low"
        assert result["metrics"]["z_score"] == pytest.approx(1.0)
        assert result["disaster_risk"] == "low"
        assert result["recommendations"] == ["Continue routine climate monitoring"]

    def test_rainless_window_is_a_detected_drought(self) -> None:
        """0 mm total rainfall is the maximal drought and grades high."""
        result = ClimateResilienceDetector().detect_precipitation_anomaly(np.array([0.0, 0.0]))

        assert result["anomaly_detected"]
        assert result["event_type"] == "drought"
        assert result["severity"] == "high"
        assert result["metrics"]["total_precipitation_mm"] == pytest.approx(0.0)
        assert result["metrics"]["z_score"] == pytest.approx(-66.2 / 13.5)
        assert result["disaster_risk"] == "elevated"
        assert result["recommendations"] == [
            "Implement water conservation measures",
            "Monitor agricultural impacts and food security",
        ]

    def test_graded_drought_is_reachable_from_positive_rainfall(self) -> None:
        """The measured baseline opens the moderate and severe drought grades.

        Regression: the 100 +/- 50 placeholder baseline put the two-sigma
        deficit at exactly 0 mm, so no positive-rainfall series could ever
        grade as drought and the "medium" grade was unreachable.
        """
        detector = ClimateResilienceDetector()

        moderate = detector.detect_precipitation_anomaly(np.array([30.0]))
        assert moderate["event_type"] == "drought"
        assert moderate["severity"] == "medium"
        assert moderate["anomaly_detected"]

        severe = detector.detect_precipitation_anomaly(np.array([25.0]))
        assert severe["event_type"] == "drought"
        assert severe["severity"] == "high"
        assert severe["metrics"]["z_score"] == pytest.approx((25.0 - 66.2) / 13.5)

    def test_drought_boundary_is_inclusive_at_two_sigma(self) -> None:
        """Exactly the 2-sigma deficit (39.2 mm) grades as moderate drought."""
        detector = ClimateResilienceDetector()

        at_boundary = detector.detect_precipitation_anomaly(np.array([39.2]))
        assert at_boundary["event_type"] == "drought"
        assert at_boundary["severity"] == "medium"
        assert at_boundary["anomaly_detected"]
        assert at_boundary["metrics"]["z_score"] == pytest.approx(-2.0)

        just_inside = detector.detect_precipitation_anomaly(np.array([39.3]))
        assert just_inside["event_type"] == "normal"
        assert not just_inside["anomaly_detected"]

    def test_baseline_constants_match_the_recorded_noaa_series(self) -> None:
        """The shipped constants ARE the measurement, recomputed from the fixture.

        tests/fixtures/humanitarian/noaa_cag_conus_pcp_199101_202012.csv is
        NOAA NCEI Climate at a Glance (nClimGrid), contiguous U.S. monthly
        precipitation, 1991-2020 (inches).
        """
        from pathlib import Path

        fixture = (
            Path(__file__).resolve().parent.parent
            / "fixtures"
            / "humanitarian"
            / "noaa_cag_conus_pcp_199101_202012.csv"
        )
        values = []
        for line in fixture.read_text().splitlines():
            if line.startswith("#") or line.startswith("Date") or not line.strip():
                continue
            _date, value = line.split(",")
            values.append(float(value) * 25.4)  # inches -> mm
        assert len(values) == 360  # 30 years x 12 months
        baseline = ClimateResilienceDetector().precip_baseline
        assert baseline["mean"] == pytest.approx(np.mean(values), abs=0.05)
        assert baseline["std"] == pytest.approx(np.std(values), abs=0.05)

    def test_flood_critical_above_three_sigma(self) -> None:
        """300 mm total (z = 4) is a critical flood with evacuations."""
        result = ClimateResilienceDetector().detect_precipitation_anomaly(np.array([300.0]))

        assert result["anomaly_detected"]
        assert result["event_type"] == "flood"
        assert result["severity"] == "critical"
        assert result["disaster_risk"] == "imminent"
        assert result["recommendations"] == [
            "Activate flood response teams",
            "Issue evacuation advisories for flood zones",
            "CRITICAL: Begin emergency evacuations",
        ]

    def test_flood_boundary_is_inclusive(self) -> None:
        """Exactly the 2-sigma surplus (93.2 mm) detects as a high-severity flood."""
        result = ClimateResilienceDetector().detect_precipitation_anomaly(np.array([93.2]))

        assert result["anomaly_detected"]
        assert result["event_type"] == "flood"
        assert result["severity"] == "high"

    def test_empty_data_returns_bare_negative(self) -> None:
        """Empty inputs return exactly the bare negative result."""
        detector = ClimateResilienceDetector()
        assert detector.detect_temperature_anomaly(np.array([])) == {"anomaly_detected": False}
        assert detector.detect_precipitation_anomaly(np.array([])) == {"anomaly_detected": False}

    def test_detect_routes_by_type(self) -> None:
        """detect() routes 'precipitation' and defaults to temperature."""
        detector = ClimateResilienceDetector()

        precip = detector.detect(np.array([100.0]), detection_type="precipitation")
        assert "total_precipitation_mm" in precip["metrics"]

        default = detector.detect(np.array([15.0]))
        assert "mean_temperature_c" in default["metrics"]

    def test_temperature_threshold_boundaries(self) -> None:
        """Heat and cold cuts are strict at +/- 3 sigma; z grading at 2."""
        detector = ClimateResilienceDetector()

        heat_at_cut = detector.detect_temperature_anomaly(np.array([45.0, 5.0]))
        assert heat_at_cut["event_type"] == "normal"
        assert heat_at_cut["severity"] == "low"
        assert not heat_at_cut["anomaly_detected"]

        heat_below_cut = detector.detect_temperature_anomaly(np.array([40.0, 10.0]))
        assert heat_below_cut["event_type"] == "normal"

        cold_at_cut = detector.detect_temperature_anomaly(np.array([-15.0, 25.0]))
        assert cold_at_cut["event_type"] == "normal"
        assert cold_at_cut["severity"] == "low"

        cold_above_cut = detector.detect_temperature_anomaly(np.array([0.0, 30.0]))
        assert cold_above_cut["event_type"] == "normal"

        z_at_two = detector.detect_temperature_anomaly(np.array([35.0, 35.0]))
        assert z_at_two["severity"] == "low"
        assert not z_at_two["anomaly_detected"]

        z_above_two = detector.detect_temperature_anomaly(np.array([37.0, 37.0]))
        assert z_above_two["severity"] == "medium"
        assert z_above_two["anomaly_detected"]

    def test_precipitation_threshold_boundaries(self) -> None:
        """Dry and wet mid-bands stay normal; the critical cut is above z = 3."""
        detector = ClimateResilienceDetector()

        dry_normal = detector.detect_precipitation_anomaly(np.array([45.0]))
        assert dry_normal["event_type"] == "normal"
        assert dry_normal["severity"] == "low"
        assert not dry_normal["anomaly_detected"]

        wet_normal = detector.detect_precipitation_anomaly(np.array([80.0]))
        assert wet_normal["event_type"] == "normal"
        assert not wet_normal["anomaly_detected"]

        # z = 3 exactly (106.7 mm) is still "high"; critical needs z > 3.
        flood_at_three_sigma = detector.detect_precipitation_anomaly(np.array([106.7]))
        assert flood_at_three_sigma["event_type"] == "flood"
        assert flood_at_three_sigma["severity"] == "high"
        assert flood_at_three_sigma["metrics"]["z_score"] == pytest.approx(3.0)

        flood_above_three_sigma = detector.detect_precipitation_anomaly(np.array([110.0]))
        assert flood_above_three_sigma["severity"] == "critical"

    def test_degenerate_baselines_stay_finite(self) -> None:
        """Zero-variance baselines exercise both division guards."""
        detector = ClimateResilienceDetector()

        detector.temp_baseline = {"mean": 15.0, "std": 0.0}
        temp = detector.detect_temperature_anomaly(np.array([15.0]))
        assert temp["metrics"]["z_score"] == pytest.approx(0.0)
        assert temp["event_type"] == "normal"
        assert not temp["anomaly_detected"]

        detector.precip_baseline = {"mean": 100.0, "std": 0.0}
        precip = detector.detect_precipitation_anomaly(np.array([50.0]))
        assert precip["metrics"]["z_score"] == pytest.approx(0.0)
        assert not precip["anomaly_detected"]

        detector.temp_baseline = {"mean": 15.0, "std": 0.5}
        temp_sub_unit = detector.detect_temperature_anomaly(np.array([16.0]))
        assert temp_sub_unit["metrics"]["z_score"] == pytest.approx(2.0)

        detector.precip_baseline = {"mean": 100.0, "std": 0.5}
        precip_sub_unit = detector.detect_precipitation_anomaly(np.array([100.5]))
        assert precip_sub_unit["metrics"]["z_score"] == pytest.approx(1.0)

    def test_drought_severity_with_wetter_baselines(self) -> None:
        """Drought severity grades on z = -3 or a fully rainless window.

        Custom baselines pin the severity cuts at round numbers,
        independent of the shipped measured climatology.
        """
        detector = ClimateResilienceDetector()

        detector.precip_baseline = {"mean": 300.0, "std": 50.0}
        at_three_sigma = detector.detect_precipitation_anomaly(np.array([150.0]))
        assert at_three_sigma["event_type"] == "drought"
        assert at_three_sigma["severity"] == "medium"

        inside_band = detector.detect_precipitation_anomaly(np.array([125.0]))
        assert inside_band["event_type"] == "drought"
        assert inside_band["severity"] == "high"

        detector.precip_baseline = {"mean": 101.0, "std": 50.0}
        near_zero = detector.detect_precipitation_anomaly(np.array([0.5]))
        assert near_zero["event_type"] == "drought"
        assert near_zero["severity"] == "medium"


class TestEconomicResilienceDetector:
    """Market and employment anomalies against the documented baselines."""

    def test_market_crash_high_severity(self) -> None:
        """An index at 60 (z = -2.67) is a high-severity crash."""
        result = EconomicResilienceDetector().detect_market_anomaly(np.array([60.0]))

        assert result["anomaly_detected"]
        assert result["threat_type"] == "market_crash"
        assert result["severity"] == "high"
        assert result["metrics"]["z_score"] == pytest.approx(-8 / 3)
        assert result["metrics"]["change_pct"] == pytest.approx(-40.0)
        assert result["systemic_risk"] == "elevated"
        assert result["economic_impact_estimate"] == "$1B-10B"
        assert result["recommendations"] == [
            "Activate market stabilization measures",
            "Coordinate with central bank for liquidity support",
        ]

    def test_market_crash_critical(self) -> None:
        """An index at 50 (z = -3.33) is critical with circuit breakers."""
        result = EconomicResilienceDetector().detect_market_anomaly(np.array([50.0]))

        assert result["severity"] == "critical"
        assert result["systemic_risk"] == "critical"
        assert result["economic_impact_estimate"] == "$10B+"
        assert result["recommendations"][-1] == "URGENT: Implement emergency circuit breakers"

    def test_volatility_alone_flags_crash(self) -> None:
        """A flat mean with volatility above 2x baseline std still flags."""
        result = EconomicResilienceDetector().detect_market_anomaly(np.array([60.0, 140.0]))

        assert result["anomaly_detected"]
        assert result["threat_type"] == "market_crash"
        assert result["severity"] == "high"
        assert result["metrics"]["z_score"] == pytest.approx(0.0)
        assert result["metrics"]["volatility"] == pytest.approx(40.0)
        assert result["systemic_risk"] == "elevated"
        assert result["economic_impact_estimate"] == "<$1B"

    def test_unemployment_spike_critical(self) -> None:
        """12% unemployment (z = 3.5) is a critical spike."""
        result = EconomicResilienceDetector().detect_employment_anomaly(np.array([12.0]))

        assert result["anomaly_detected"]
        assert result["threat_type"] == "unemployment_spike"
        assert result["severity"] == "critical"
        assert result["labor_market_stress"] == "severe"
        assert result["recommendations"] == [
            "Deploy job creation programs",
            "Provide unemployment assistance",
            "CRITICAL: Emergency economic stimulus required",
        ]

    def test_unemployment_at_exact_boundary_is_not_flagged(self) -> None:
        """9% unemployment (z = 2 exactly) stays below the strict cut."""
        result = EconomicResilienceDetector().detect_employment_anomaly(np.array([9.0]))

        assert not result["anomaly_detected"]
        assert result["threat_type"] == "normal"

    def test_empty_data_returns_bare_negative(self) -> None:
        """Empty inputs return exactly the bare negative result."""
        detector = EconomicResilienceDetector()
        assert detector.detect_market_anomaly(np.array([])) == {"anomaly_detected": False}
        assert detector.detect_employment_anomaly(np.array([])) == {"anomaly_detected": False}

    def test_detect_routes_by_type(self) -> None:
        """detect() routes 'employment' and defaults to market."""
        detector = EconomicResilienceDetector()

        employment = detector.detect(np.array([5.0]), detection_type="employment")
        assert "unemployment_rate_pct" in employment["metrics"]

        default = detector.detect(np.array([100.0]))
        assert "current_value" in default["metrics"]

    def test_market_threshold_boundaries(self) -> None:
        """Crash, volatility, impact and systemic cuts at their exact edges."""
        detector = EconomicResilienceDetector()

        at_two_sigma = detector.detect_market_anomaly(np.array([70.0]))
        assert at_two_sigma["threat_type"] == "normal"
        assert not at_two_sigma["anomaly_detected"]
        assert at_two_sigma["metrics"]["z_score"] == pytest.approx(-2.0)
        assert at_two_sigma["economic_impact_estimate"] == "<$1B"
        assert at_two_sigma["systemic_risk"] == "low"

        mid_band = detector.detect_market_anomaly(np.array([80.0]))
        assert mid_band["threat_type"] == "normal"
        assert not mid_band["anomaly_detected"]

        at_three_sigma = detector.detect_market_anomaly(np.array([55.0]))
        assert at_three_sigma["threat_type"] == "market_crash"
        assert at_three_sigma["severity"] == "high"
        assert at_three_sigma["economic_impact_estimate"] == "$1B-10B"
        assert at_three_sigma["systemic_risk"] == "elevated"

        vol_at_two_sigma = detector.detect_market_anomaly(np.array([70.0, 130.0]))
        assert vol_at_two_sigma["threat_type"] == "normal"
        assert not vol_at_two_sigma["anomaly_detected"]
        assert vol_at_two_sigma["systemic_risk"] == "low"

        vol_mid_band = detector.detect_market_anomaly(np.array([80.0, 120.0]))
        assert vol_mid_band["threat_type"] == "normal"
        assert vol_mid_band["systemic_risk"] == "low"

        vol_at_three_sigma = detector.detect_market_anomaly(np.array([55.0, 145.0]))
        assert vol_at_three_sigma["threat_type"] == "market_crash"
        assert vol_at_three_sigma["systemic_risk"] == "elevated"

        vol_above_three_sigma = detector.detect_market_anomaly(np.array([50.0, 150.0]))
        assert vol_above_three_sigma["systemic_risk"] == "critical"

    def test_employment_threshold_boundaries(self) -> None:
        """The spike cut is strict at 9%; critical is strictly above z = 3."""
        detector = EconomicResilienceDetector()

        at_critical_cut = detector.detect_employment_anomaly(np.array([11.0]))
        assert at_critical_cut["threat_type"] == "unemployment_spike"
        assert at_critical_cut["severity"] == "high"
        assert at_critical_cut["metrics"]["z_score"] == pytest.approx(3.0)

        above_spike_cut = detector.detect_employment_anomaly(np.array([10.0]))
        assert above_spike_cut["threat_type"] == "unemployment_spike"
        assert above_spike_cut["severity"] == "high"
        assert above_spike_cut["anomaly_detected"]

        detector.unemployment_baseline = {"mean": 5.0, "std": 3.0}
        custom = detector.detect_employment_anomaly(np.array([10.5]))
        assert custom["threat_type"] == "normal"
        assert not custom["anomaly_detected"]

    def test_unemployment_z_score_is_exact(self) -> None:
        """12% against baseline 5 +/- 2 is exactly z = 3.5."""
        result = EconomicResilienceDetector().detect_employment_anomaly(np.array([12.0]))
        assert result["metrics"]["z_score"] == pytest.approx(3.5)

    def test_degenerate_baselines_stay_finite(self) -> None:
        """Zero- and sub-unit baselines exercise every division guard."""
        detector = EconomicResilienceDetector()

        detector.market_baseline = {"mean": 1.0, "std": 0.0}
        sub_unit = detector.detect_market_anomaly(np.array([0.5]))
        assert sub_unit["metrics"]["z_score"] == pytest.approx(0.0)
        assert sub_unit["metrics"]["change_pct"] == pytest.approx(-50.0)

        detector.market_baseline = {"mean": 0.0, "std": 1.0}
        zero_mean = detector.detect_market_anomaly(np.array([0.0]))
        assert zero_mean["metrics"]["change_pct"] == pytest.approx(0.0)

        detector.market_baseline = {"mean": 100.0, "std": 0.5}
        market_sub_unit = detector.detect_market_anomaly(np.array([100.25]))
        assert market_sub_unit["metrics"]["z_score"] == pytest.approx(0.5)

        detector.unemployment_baseline = {"mean": 5.0, "std": 0.0}
        employment = detector.detect_employment_anomaly(np.array([6.0]))
        assert employment["metrics"]["z_score"] == pytest.approx(0.0)
        assert not employment["anomaly_detected"]

        detector.unemployment_baseline = {"mean": 5.0, "std": 0.5}
        employment_sub_unit = detector.detect_employment_anomaly(np.array([5.25]))
        assert employment_sub_unit["metrics"]["z_score"] == pytest.approx(0.5)


class TestEducationEquityDetector:
    """Achievement gaps and dropout risk against the documented baselines."""

    def test_achievement_gap_high_severity(self) -> None:
        """Mean 50 (z = -2.5) is a high-severity achievement gap."""
        result = EducationEquityDetector().detect_achievement_gap(np.array([50.0, 50.0]))

        assert result["anomaly_detected"]
        assert result["threat_type"] == "achievement_gap"
        assert result["severity"] == "high"
        assert result["metrics"]["z_score"] == pytest.approx(-2.5)
        assert result["equity_risk"] == "moderate_inequity"
        assert result["affected_students_estimate"] == 200
        assert result["recommendations"] == [
            "Implement targeted interventions for underperforming groups",
            "Review curriculum for cultural bias",
        ]

    def test_achievement_collapse_critical(self) -> None:
        """Mean 40 (z = -3.5) is critical with emergency tutoring."""
        result = EducationEquityDetector().detect_achievement_gap(np.array([40.0]))

        assert result["severity"] == "critical"
        assert result["equity_risk"] == "severe_inequity"
        assert result["affected_students_estimate"] == 500
        assert result["recommendations"][-1] == "URGENT: Deploy emergency tutoring programs"

    def test_disparity_alone_flags_without_threat(self) -> None:
        """High spread at a normal mean flags on the disparity score."""
        result = EducationEquityDetector().detect_achievement_gap(np.array([40.0, 110.0]))

        assert result["anomaly_detected"]
        assert result["threat_type"] == "normal"
        assert result["metrics"]["disparity_score"] == pytest.approx(35.0 / 75.0)
        assert result["equity_risk"] == "severe_inequity"

    def test_dropout_risk_one_dimensional(self) -> None:
        """Two of five students under the 0.3 indicator mean 40% risk."""
        result = EducationEquityDetector().detect_dropout_risk(np.array([0.1, 0.2, 0.9, 0.9, 0.9]))

        assert result["anomaly_detected"]
        assert result["threat_type"] == "dropout_risk"
        assert result["severity"] == "high"
        assert result["metrics"]["dropout_risk_rate"] == pytest.approx(0.4)
        assert result["metrics"]["at_risk_percentage"] == pytest.approx(40.0)
        assert result["intervention_urgency"] == "standard"
        assert result["recommendations"] == [
            "Activate student retention programs",
            "Provide counseling and mentorship",
        ]

    def test_dropout_risk_two_dimensional_critical(self) -> None:
        """2-D input averages per column; 50% risk is critical and immediate."""
        result = EducationEquityDetector().detect_dropout_risk(np.array([[0.2, 0.9], [0.2, 0.9]]))

        assert result["severity"] == "critical"
        assert result["metrics"]["dropout_risk_rate"] == pytest.approx(0.5)
        assert result["intervention_urgency"] == "immediate"
        assert result["recommendations"][-1] == "CRITICAL: Immediate one-on-one interventions"

    def test_no_dropout_risk(self) -> None:
        """Strong indicators report normal with routine monitoring."""
        result = EducationEquityDetector().detect_dropout_risk(np.array([0.9] * 5))

        assert not result["anomaly_detected"]
        assert result["threat_type"] == "normal"
        assert result["recommendations"] == ["Continue routine educational monitoring"]

    def test_empty_data_returns_bare_negative(self) -> None:
        """Empty inputs return exactly the bare negative result."""
        detector = EducationEquityDetector()
        assert detector.detect_achievement_gap(np.array([])) == {"anomaly_detected": False}
        assert detector.detect_dropout_risk(np.array([])) == {"anomaly_detected": False}

    def test_detect_routes_by_type(self) -> None:
        """detect() routes 'dropout_risk' and defaults to achievement."""
        detector = EducationEquityDetector()

        dropout = detector.detect(np.array([0.9]), detection_type="dropout_risk")
        assert "dropout_risk_rate" in dropout["metrics"]

        default = detector.detect(np.array([75.0]))
        assert "mean_achievement" in default["metrics"]

    def test_achievement_threshold_boundaries(self) -> None:
        """Gap, anomaly, severity and disparity cuts at their exact edges."""
        detector = EducationEquityDetector()

        at_two_sigma = detector.detect_achievement_gap(np.array([55.0]))
        assert at_two_sigma["threat_type"] == "normal"
        assert not at_two_sigma["anomaly_detected"]
        assert at_two_sigma["affected_students_estimate"] == 0

        mid_band = detector.detect_achievement_gap(np.array([60.0]))
        assert mid_band["threat_type"] == "normal"
        assert not mid_band["anomaly_detected"]

        at_three_sigma = detector.detect_achievement_gap(np.array([45.0]))
        assert at_three_sigma["threat_type"] == "achievement_gap"
        assert at_three_sigma["severity"] == "high"
        assert at_three_sigma["affected_students_estimate"] == 200

        # Disparity exactly 0.3 stays below the strict anomaly cut.
        disparity_at_cut = detector.detect_achievement_gap(np.array([52.5, 97.5]))
        assert disparity_at_cut["metrics"]["disparity_score"] == pytest.approx(0.3)
        assert not disparity_at_cut["anomaly_detected"]
        assert disparity_at_cut["equity_risk"] == "acceptable"

        disparity_mid_band = detector.detect_achievement_gap(np.array([48.0, 102.0]))
        assert disparity_mid_band["metrics"]["disparity_score"] == pytest.approx(0.36)
        assert disparity_mid_band["anomaly_detected"]

        disparity_at_severe_cut = detector.detect_achievement_gap(np.array([45.0, 105.0]))
        assert disparity_at_severe_cut["metrics"]["disparity_score"] == pytest.approx(0.4)
        assert disparity_at_severe_cut["equity_risk"] == "moderate_inequity"

    def test_dropout_threshold_boundaries(self) -> None:
        """Risk cuts at 0.15/0.2/0.4 and the 0.3 indicator cut, all exact."""
        detector = EducationEquityDetector()

        at_anomaly_cut = detector.detect_dropout_risk(np.array([0.1] * 3 + [0.9] * 17))
        assert at_anomaly_cut["metrics"]["dropout_risk_rate"] == pytest.approx(0.15)
        assert not at_anomaly_cut["anomaly_detected"]
        assert at_anomaly_cut["threat_type"] == "normal"

        at_threat_cut = detector.detect_dropout_risk(np.array([0.1] + [0.9] * 4))
        assert at_threat_cut["metrics"]["dropout_risk_rate"] == pytest.approx(0.2)
        assert at_threat_cut["threat_type"] == "normal"
        assert at_threat_cut["anomaly_detected"]

        above_threat_cut = detector.detect_dropout_risk(np.array([0.1] + [0.9] * 3))
        assert above_threat_cut["metrics"]["dropout_risk_rate"] == pytest.approx(0.25)
        assert above_threat_cut["threat_type"] == "dropout_risk"
        assert above_threat_cut["severity"] == "high"

        # Indicators at exactly 0.3 and inside (0.3, 0.4) are not at risk.
        indicator_cut = detector.detect_dropout_risk(np.array([0.3, 0.35, 0.9, 0.9]))
        assert indicator_cut["metrics"]["dropout_risk_rate"] == pytest.approx(0.0)
        assert indicator_cut["threat_type"] == "normal"

        # 2-D input averages per column before the indicator cut.
        column_means = detector.detect_dropout_risk(np.array([[0.2, 0.9], [0.4, 0.9]]))
        assert column_means["metrics"]["dropout_risk_rate"] == pytest.approx(0.0)
        assert column_means["threat_type"] == "normal"

    def test_degenerate_baselines_stay_finite(self) -> None:
        """Zero-variance and sub-unit baselines exercise both guards."""
        detector = EducationEquityDetector()

        detector.achievement_baseline = {"mean": 1.0, "std": 0.0}
        result = detector.detect_achievement_gap(np.array([0.5, 1.5]))
        assert result["metrics"]["z_score"] == pytest.approx(0.0)
        assert result["metrics"]["disparity_score"] == pytest.approx(0.5)

        zero_mean = detector.detect_achievement_gap(np.array([0.0]))
        assert zero_mean["metrics"]["disparity_score"] == pytest.approx(0.0)

        detector.achievement_baseline = {"mean": 75.0, "std": 0.5}
        sub_unit_std = detector.detect_achievement_gap(np.array([75.25]))
        assert sub_unit_std["metrics"]["z_score"] == pytest.approx(0.5)


class TestNeuroscienceDetector:
    """Neural activity and cognitive anomalies against the documented baselines."""

    def test_cognitive_decline_medium(self) -> None:
        """A 0.6 cognitive score is a medium decline, no intervention yet."""
        result = NeuroscienceDetector().detect_cognitive_anomaly(np.array([0.6]))

        assert result["anomaly_detected"]
        assert result["threat_type"] == "cognitive_decline"
        assert result["severity"] == "medium"
        assert result["metrics"]["performance_percentile"] == pytest.approx(60.0)
        assert result["intervention_needed"] is False
        assert result["recommendations"] == [
            "Schedule comprehensive cognitive assessment",
            "Review medications and medical history",
        ]

    def test_cognitive_decline_high_needs_intervention(self) -> None:
        """A 0.4 cognitive score grades high and adds the consultation."""
        result = NeuroscienceDetector().detect_cognitive_anomaly(np.array([0.4]))

        assert result["severity"] == "high"
        assert result["intervention_needed"] is True
        assert result["recommendations"][-1] == "Consider neurological consultation"

    def test_cognitive_normal(self) -> None:
        """A 0.8 cognitive score is normal with routine monitoring."""
        result = NeuroscienceDetector().detect_cognitive_anomaly(np.array([0.8]))

        assert not result["anomaly_detected"]
        assert result["threat_type"] == "normal"
        assert result["recommendations"] == ["Continue routine monitoring"]

    def test_seizure_pattern_is_critical_and_immediate(self) -> None:
        """Two >3-sigma jumps in 20 samples saturate the seizure risk."""
        data = np.array([0.0] * 18 + [1000.0, 0.0])
        result = NeuroscienceDetector().detect_neural_activity_anomaly(data)

        assert result["anomaly_detected"]
        assert result["threat_type"] == "seizure_pattern"
        assert result["severity"] == "critical"
        assert result["metrics"]["seizure_risk"] == pytest.approx(1.0)
        assert result["neurological_risk"] == "critical"
        assert result["intervention_urgency"] == "immediate"
        assert result["recommendations"] == [
            "URGENT: Alert medical staff immediately",
            "Prepare emergency seizure response",
            "Monitor continuously for 24 hours",
        ]

    def test_sustained_high_activity_is_cognitive_decline(self) -> None:
        """Flat activity at z = 4 grades as high-severity decline."""
        result = NeuroscienceDetector().detect_neural_activity_anomaly(np.array([110.0] * 10))

        assert result["anomaly_detected"]
        assert result["threat_type"] == "cognitive_decline"
        assert result["severity"] == "high"
        assert result["metrics"]["z_score"] == pytest.approx(4.0)
        assert result["metrics"]["seizure_risk"] == pytest.approx(0.0)
        assert result["neurological_risk"] == "critical"
        # Shipped behavior: only seizure patterns escalate the urgency.
        assert result["intervention_urgency"] == "routine"

    def test_short_series_cannot_grade_seizure(self) -> None:
        """Under ten samples the seizure score stays 0 by design."""
        result = NeuroscienceDetector().detect_neural_activity_anomaly(np.array([0.0, 1000.0] * 4))

        assert result["metrics"]["seizure_risk"] == pytest.approx(0.0)
        assert result["threat_type"] == "cognitive_decline"

    def test_neural_activity_normal(self) -> None:
        """Baseline activity reports normal, low risk, routine monitoring."""
        result = NeuroscienceDetector().detect_neural_activity_anomaly(np.array([50.0] * 10))

        assert not result["anomaly_detected"]
        assert result["threat_type"] == "normal"
        assert result["neurological_risk"] == "low"

    def test_empty_data_returns_bare_negative(self) -> None:
        """Empty inputs return exactly the bare negative result."""
        detector = NeuroscienceDetector()
        assert detector.detect_neural_activity_anomaly(np.array([])) == {"anomaly_detected": False}
        assert detector.detect_cognitive_anomaly(np.array([])) == {"anomaly_detected": False}

    def test_detect_routes_by_type(self) -> None:
        """detect() routes 'cognitive' and defaults to neural activity."""
        detector = NeuroscienceDetector()

        cognitive = detector.detect(np.array([0.9]), detection_type="cognitive")
        assert "cognitive_score" in cognitive["metrics"]

        default = detector.detect(np.array([50.0] * 10))
        assert "mean_activity" in default["metrics"]

    def test_neural_z_score_boundaries(self) -> None:
        """Anomaly, risk and decline cuts sit strictly at z = 2 and z = 3."""
        detector = NeuroscienceDetector()

        at_two = detector.detect_neural_activity_anomaly(np.array([80.0] * 10))
        assert at_two["metrics"]["z_score"] == pytest.approx(2.0)
        assert not at_two["anomaly_detected"]
        assert at_two["neurological_risk"] == "low"

        above_two = detector.detect_neural_activity_anomaly(np.array([85.0] * 10))
        assert above_two["anomaly_detected"]
        assert above_two["threat_type"] == "normal"
        assert above_two["neurological_risk"] == "elevated"

        at_three = detector.detect_neural_activity_anomaly(np.array([95.0] * 10))
        assert at_three["metrics"]["z_score"] == pytest.approx(3.0)
        assert at_three["threat_type"] == "normal"
        assert at_three["neurological_risk"] == "elevated"

    def test_seizure_risk_boundaries(self) -> None:
        """The seizure grades sit strictly at risk 0.5 and 0.3."""
        detector = NeuroscienceDetector()

        # One >3-sigma jump in 20 samples: risk exactly 0.5 stays below
        # the strict critical cut.
        at_half = detector.detect_neural_activity_anomaly(np.array([0.0] * 19 + [1000.0]))
        assert at_half["metrics"]["seizure_risk"] == pytest.approx(0.5)
        assert at_half["threat_type"] == "normal"
        assert at_half["anomaly_detected"]
        assert at_half["neurological_risk"] == "elevated"

        # Three isolated spikes in 100 samples: risk 0.6 crosses the cut.
        data = np.zeros(100)
        data[[20, 50, 80]] = 1000.0
        above_half = detector.detect_neural_activity_anomaly(data)
        assert above_half["metrics"]["seizure_risk"] == pytest.approx(0.6)
        assert above_half["threat_type"] == "seizure_pattern"
        assert above_half["neurological_risk"] == "critical"

        # One jump in 30 samples: risk 1/3 sits inside (0.3, 0.4).
        at_third = detector.detect_neural_activity_anomaly(np.array([0.0] * 29 + [1000.0]))
        assert at_third["metrics"]["seizure_risk"] == pytest.approx(1 / 3)
        assert at_third["anomaly_detected"]
        assert at_third["neurological_risk"] == "elevated"

    def test_seizure_spike_definition(self) -> None:
        """A spike is a jump strictly above 3x the series' own sigma."""
        detector = NeuroscienceDetector()

        # Two isolated 1000-jumps in 20 samples: sigma = 300, so the jumps
        # sit between 3 sigma (900) and 4 sigma (1200) and must count.
        data = np.zeros(20)
        data[[5, 14]] = 1000.0
        counted = detector.detect_neural_activity_anomaly(data)
        assert counted["metrics"]["seizure_risk"] == pytest.approx(1.0)
        assert counted["threat_type"] == "seizure_pattern"

        # A small oscillation (jumps of 8 against sigma 4) never counts:
        # 8 < 3 sigma = 12, so the flagged threat is the z-score decline.
        oscillation = detector.detect_neural_activity_anomaly(np.array([0.0, 8.0] * 10))
        assert oscillation["metrics"]["seizure_risk"] == pytest.approx(0.0)
        assert oscillation["threat_type"] == "cognitive_decline"

        # Exactly ten samples is enough for the seizure score.
        shortest = detector.detect_neural_activity_anomaly(np.array([0.0] * 8 + [1000.0, 0.0]))
        assert shortest["metrics"]["seizure_risk"] == pytest.approx(1.0)
        assert shortest["threat_type"] == "seizure_pattern"

    def test_cognitive_threshold_boundaries(self) -> None:
        """The decline cut is strict at 0.7 and the high grade strict at 0.5."""
        detector = NeuroscienceDetector()

        at_cut = detector.detect_cognitive_anomaly(np.array([0.7]))
        assert at_cut["threat_type"] == "normal"
        assert not at_cut["anomaly_detected"]

        inside_band = detector.detect_cognitive_anomaly(np.array([0.75]))
        assert inside_band["threat_type"] == "normal"

        at_high_cut = detector.detect_cognitive_anomaly(np.array([0.5]))
        assert at_high_cut["threat_type"] == "cognitive_decline"
        assert at_high_cut["severity"] == "medium"

    def test_degenerate_baseline_stays_finite(self) -> None:
        """A zero-variance neural baseline exercises the division guard."""
        detector = NeuroscienceDetector()
        detector.neural_baseline = {"mean": 50.0, "std": 0.0}

        result = detector.detect_neural_activity_anomaly(np.array([60.0] * 10))
        assert result["metrics"]["z_score"] == pytest.approx(0.0)
        assert not result["anomaly_detected"]

        detector.neural_baseline = {"mean": 50.0, "std": 0.5}
        sub_unit_std = detector.detect_neural_activity_anomaly(np.array([50.25] * 10))
        assert sub_unit_std["metrics"]["z_score"] == pytest.approx(0.5)

    def test_anomaly_cut_at_exact_seizure_boundary(self) -> None:
        """Risk exactly 0.3 with z exactly -2 sits below both strict cuts.

        Three counted jumps in 100 samples (one isolated spike is two
        jumps, one end spike is one) give risk 0.3, and the two 1000s
        pull the mean to 20, so z = -2 exactly.
        """
        data = np.zeros(100)
        data[50] = 1000.0
        data[99] = 1000.0

        result = NeuroscienceDetector().detect_neural_activity_anomaly(data)

        assert result["metrics"]["seizure_risk"] == pytest.approx(0.3)
        assert result["metrics"]["z_score"] == pytest.approx(-2.0)
        assert not result["anomaly_detected"]
        assert result["neurological_risk"] == "low"


class TestResultTypeHygiene:
    """``anomaly_detected`` is a Python bool and timestamps are UTC-aware.

    The bare numpy comparisons previously leaked ``np.bool_`` into the
    results (which fails ``isinstance(..., bool)`` and does not survive
    JSON), and ``datetime.now()`` produced naive local timestamps against
    the repo-wide UTC convention. These fail on that code.
    """

    _CASES = [
        pytest.param(
            lambda: NeuroscienceDetector().detect(np.full(20, 50.0), "neural_activity"),
            False,
            id="neural_activity-normal",
        ),
        pytest.param(
            lambda: NeuroscienceDetector().detect(np.full(20, 120.0), "neural_activity"),
            True,
            id="neural_activity-anomalous",
        ),
        pytest.param(
            lambda: NeuroscienceDetector().detect(np.full(10, 0.9), "cognitive"),
            False,
            id="cognitive-normal",
        ),
        pytest.param(
            lambda: NeuroscienceDetector().detect(np.full(10, 0.5), "cognitive"),
            True,
            id="cognitive-anomalous",
        ),
        pytest.param(
            lambda: ClimateResilienceDetector().detect(np.full(10, 15.0), "temperature"),
            False,
            id="temperature-normal",
        ),
        pytest.param(
            lambda: ClimateResilienceDetector().detect(np.full(10, 60.0), "temperature"),
            True,
            id="temperature-anomalous",
        ),
        pytest.param(
            lambda: ClimateResilienceDetector().detect(np.full(10, 6.62), "precipitation"),
            False,
            id="precipitation-normal",
        ),
        pytest.param(
            lambda: ClimateResilienceDetector().detect(np.zeros(3), "precipitation"),
            True,
            id="precipitation-anomalous",
        ),
        pytest.param(
            lambda: EconomicResilienceDetector().detect(np.full(10, 100.0), "market"),
            False,
            id="market-normal",
        ),
        pytest.param(
            lambda: EconomicResilienceDetector().detect(np.full(10, 20.0), "market"),
            True,
            id="market-anomalous",
        ),
        pytest.param(
            lambda: EconomicResilienceDetector().detect(np.full(10, 5.0), "employment"),
            False,
            id="employment-normal",
        ),
        pytest.param(
            lambda: EconomicResilienceDetector().detect(np.full(10, 12.0), "employment"),
            True,
            id="employment-anomalous",
        ),
        pytest.param(
            lambda: EducationEquityDetector().detect(np.full(10, 75.0), "achievement"),
            False,
            id="achievement-normal",
        ),
        pytest.param(
            lambda: EducationEquityDetector().detect(np.full(10, 40.0), "achievement"),
            True,
            id="achievement-anomalous",
        ),
        pytest.param(
            lambda: EducationEquityDetector().detect(np.full(10, 0.9), "dropout_risk"),
            False,
            id="dropout-normal",
        ),
        pytest.param(
            lambda: EducationEquityDetector().detect(np.full(10, 0.1), "dropout_risk"),
            True,
            id="dropout-anomalous",
        ),
        pytest.param(
            lambda: AgriFoodSecurityDetector().detect(np.full(10, 5.0), "yield"),
            False,
            id="yield-normal",
        ),
        pytest.param(
            lambda: AgriFoodSecurityDetector().detect(np.full(10, 2.0), "yield"),
            True,
            id="yield-anomalous",
        ),
        pytest.param(
            lambda: AgriFoodSecurityDetector().detect(np.full(10, 100.0), "price"),
            False,
            id="price-normal",
        ),
        pytest.param(
            lambda: AgriFoodSecurityDetector().detect(np.full(10, 150.0), "price"),
            True,
            id="price-anomalous",
        ),
    ]

    @pytest.mark.parametrize(("run", "expected"), _CASES)
    def test_anomaly_detected_is_a_python_bool(
        self, run: Callable[[], dict[str, Any]], expected: bool
    ) -> None:
        result = run()
        assert isinstance(result["anomaly_detected"], bool)
        assert result["anomaly_detected"] is expected

    @pytest.mark.parametrize(("run", "expected"), _CASES)
    def test_timestamp_is_utc_aware(
        self, run: Callable[[], dict[str, Any]], expected: bool
    ) -> None:
        from datetime import UTC

        result = run()
        assert result["timestamp"].tzinfo is UTC


class TestMentalHealthScreening:
    """PHQ-9 / GAD-7 / ASRS v1.1 scoring on their published rules.

    Cutpoints, bands and per-item thresholds follow the instruments'
    primary sources (Kroenke 2001; Spitzer 2006; Kessler 2005); PHQ-9
    item 9 escalates per the instrument's published guidance.
    """

    def _phq9(self, total: int, item9: int = 0) -> dict[str, Any]:
        items = np.zeros(9)
        remaining = total - item9
        for i in range(8):
            items[i] = min(3, remaining)
            remaining -= items[i]
            if remaining <= 0:
                break
        items[8] = item9
        assert items.sum() == total
        return NeuroscienceDetector().detect(items, "mental_health")

    @pytest.mark.parametrize(
        ("total", "band", "anomaly"),
        [
            (0, "minimal", False),
            (4, "minimal", False),
            (5, "mild", False),
            (9, "mild", False),
            (10, "moderate", True),
            (14, "moderate", True),
            (15, "moderately_severe", True),
            (19, "moderately_severe", True),
            (20, "severe", True),
        ],
    )
    def test_phq9_bands_and_cutpoint(self, total: int, band: str, anomaly: bool) -> None:
        result = self._phq9(total)
        assert result["metrics"]["severity_band"] == band
        assert result["anomaly_detected"] is anomaly
        assert result["metrics"]["cutpoint"] == 10

    def test_phq9_item9_escalates_to_crisis(self) -> None:
        """Any item-9 endorsement is a crisis, whatever the total score."""
        result = self._phq9(1, item9=1)
        assert result["threat_type"] == "mental_health_crisis"
        assert result["severity"] == "critical"
        assert result["intervention_urgency"] == "immediate"
        assert result["metrics"]["crisis_item_positive"] is True
        assert result["anomaly_detected"] is True
        assert any("suicide-risk assessment" in r for r in result["recommendations"])

    def test_phq9_severe_band_is_crisis_high(self) -> None:
        result = self._phq9(20)
        assert result["threat_type"] == "mental_health_crisis"
        assert result["severity"] == "high"
        assert result["intervention_urgency"] == "routine"
        assert result["metrics"]["crisis_item_positive"] is False

    def test_phq9_positive_screen_refers_to_clinician(self) -> None:
        result = self._phq9(12)
        assert result["threat_type"] == "normal"
        assert result["severity"] == "medium"
        assert result["recommendations"][0] == (
            "Refer for clinical assessment by a licensed mental health professional"
        )

    def test_every_result_carries_the_clinical_disclaimer(self) -> None:
        for total in (0, 12, 20):
            result = self._phq9(total)
            assert result["clinical_disclaimer"] == (
                "Screening output only, not a diagnosis; defer to evaluation by a "
                "licensed mental health clinician."
            )
            assert result["metrics"]["published_sensitivity"] == 0.88
            assert result["metrics"]["published_specificity"] == 0.88
            assert "Kroenke" in result["method_citation"]

    @pytest.mark.parametrize(
        ("total", "band", "threat", "anomaly"),
        [
            (4, "minimal", "normal", False),
            (5, "mild", "normal", False),
            (10, "moderate", "normal", True),
            (15, "severe", "mental_health_crisis", True),
        ],
    )
    def test_gad7_bands(self, total: int, band: str, threat: str, anomaly: bool) -> None:
        items = np.zeros(7)
        remaining = total
        for i in range(7):
            items[i] = min(3, remaining)
            remaining -= items[i]
        result = NeuroscienceDetector().detect(items, "mental_health", {"instrument": "gad7"})
        assert result["metrics"]["severity_band"] == band
        assert result["threat_type"] == threat
        assert result["anomaly_detected"] is anomaly
        assert result["metrics"]["published_sensitivity"] == 0.89

    def test_asrs_positive_screen_needs_four_shaded_items(self) -> None:
        detector = NeuroscienceDetector()
        positive = detector.detect(
            np.array([2, 2, 2, 3, 0, 0]), "mental_health", {"instrument": "asrs6"}
        )
        assert positive["threat_type"] == "attention_deficit"
        assert positive["anomaly_detected"] is True
        assert positive["metrics"]["positive_items"] == 4
        assert positive["recommendations"][0].startswith("Refer for a comprehensive adult ADHD")

        three_only = detector.detect(
            np.array([2, 2, 2, 2, 2, 2]), "mental_health", {"instrument": "asrs6"}
        )
        assert three_only["metrics"]["positive_items"] == 3  # items 4-6 need >= 3
        assert three_only["threat_type"] == "normal"
        assert three_only["anomaly_detected"] is False

    def test_asrs_item_thresholds_differ_by_position(self) -> None:
        detector = NeuroscienceDetector()
        # Items 1-3 count from 2 ("sometimes"); a 1 never counts.
        low_first_half = detector.detect(
            np.array([1, 1, 1, 3, 3, 3]), "mental_health", {"instrument": "asrs6"}
        )
        assert low_first_half["metrics"]["positive_items"] == 3
        # Items 4-6 count from 3 ("often"); a 4 counts everywhere it can.
        maxed = detector.detect(
            np.array([4, 4, 4, 4, 4, 4]), "mental_health", {"instrument": "asrs6"}
        )
        assert maxed["metrics"]["positive_items"] == 6
        assert maxed["anomaly_detected"] is True

    def test_invalid_instrument_inputs_are_refused(self) -> None:
        detector = NeuroscienceDetector()
        with pytest.raises(ValueError, match="unknown screening instrument"):
            detector.detect(np.zeros(9), "mental_health", {"instrument": "mmpi"})
        with pytest.raises(ValueError, match="exactly 9 item scores"):
            detector.detect(np.zeros(3), "mental_health")
        with pytest.raises(ValueError, match=r"integers in 0\.\.3"):
            detector.detect(np.array([0, 0, 0, 0, 0, 0, 0, 0, 4]), "mental_health")
        with pytest.raises(ValueError, match=r"integers in 0\.\.3"):
            detector.detect(np.array([0.5, 0, 0, 0, 0, 0, 0, 0, 0]), "mental_health")
        with pytest.raises(ValueError, match=r"integers in 0\.\.3"):
            detector.detect(np.array([-1, 0, 0, 0, 0, 0, 0, 0, 0]), "mental_health")

    def test_empty_input_returns_bare_negative(self) -> None:
        assert NeuroscienceDetector().detect(np.array([]), "mental_health") == {
            "anomaly_detected": False
        }

    def test_timestamps_are_utc(self) -> None:
        from datetime import UTC

        assert self._phq9(12)["timestamp"].tzinfo is UTC


class TestSeaLevelDetection:
    """Trend/acceleration vs the global-mean altimetry references."""

    @staticmethod
    def _series(trend_mm_yr: float, months: int = 350, seasonal_m: float = 0.05) -> np.ndarray:
        t = np.arange(months) / 12.0
        return trend_mm_yr / 1000.0 * t + seasonal_m * np.cos(2 * np.pi * t)

    def test_trend_recovered_through_seasonal_cycle(self) -> None:
        """The harmonic design recovers the trend on a non-integer year span."""
        result = ClimateResilienceDetector().detect(self._series(3.0), "sea_level")
        assert result["metrics"]["trend_mm_per_year"] == pytest.approx(3.0, abs=0.05)
        assert result["anomaly_detected"] is False  # 3.0 < global 3.4
        assert result["coastal_flood_risk"] == "baseline"

    def test_above_global_trend_detects(self) -> None:
        result = ClimateResilienceDetector().detect(self._series(5.0), "sea_level")
        assert result["anomaly_detected"] is True
        assert result["severity"] == "medium"
        assert result["coastal_flood_risk"] == "elevated"

    def test_hotspot_above_twice_global_is_high(self) -> None:
        result = ClimateResilienceDetector().detect(self._series(7.0), "sea_level")
        assert result["severity"] == "high"

    def test_acceleration_alone_detects(self) -> None:
        t = np.arange(360) / 12.0
        series = 0.0005 * t**2  # 1 mm/yr^2
        result = ClimateResilienceDetector().detect(series, "sea_level")
        assert result["metrics"]["acceleration_mm_per_year2"] == pytest.approx(1.0, abs=0.01)
        assert result["anomaly_detected"] is True

    def test_flat_record_is_normal(self) -> None:
        result = ClimateResilienceDetector().detect(np.full(60, 1.0), "sea_level")
        assert result["metrics"]["trend_mm_per_year"] == pytest.approx(0.0, abs=1e-9)
        assert result["anomaly_detected"] is False

    def test_flood_threshold_context_grades_critical(self) -> None:
        series = self._series(5.0) + 1.0
        result = ClimateResilienceDetector().detect_sea_level_anomaly(
            series, {"minor_flood_threshold_m": 1.0}
        )
        assert result["event_type"] == "flood"
        assert result["severity"] == "critical"

    def test_short_or_bad_records_are_refused_not_scored(self) -> None:
        detector = ClimateResilienceDetector()
        assert detector.detect(np.array([]), "sea_level") == {"anomaly_detected": False}
        short = detector.detect(np.full(23, 1.0), "sea_level")
        assert short["anomaly_detected"] is False
        assert "at least 24 monthly means" in short["note"]
        with pytest.raises(ValueError, match="non-finite"):
            detector.detect(np.array([1.0] * 30 + [np.nan]), "sea_level")

    def test_battery_fixture_reproduces_recorded_trend(self) -> None:
        """NOAA CO-OPS 8518750 (The Battery), monthly MSL 2000-2024.

        Recorded from api.tidesandcurrents.noaa.gov (product=monthly_mean,
        datum=MSL, metric); sha256
        d1bd577994f58786c74ff506fc115314428d8aa88ca76ecf39798949927d9fbf.
        The measured 2000-2024 trend (6.32 mm/yr) sits above NOAA's
        published 1856-2024 long-term rate (2.98 mm/yr), the documented
        post-2000 acceleration on the US Northeast coast.
        """
        import json
        from pathlib import Path

        fixture = (
            Path(__file__).resolve().parent.parent
            / "fixtures"
            / "humanitarian"
            / "coops_8518750_monthly_msl_2000_2024.json"
        )
        rows = json.loads(fixture.read_text())["data"]
        assert [(int(r["year"]), int(r["month"])) for r in rows] == [
            (y, m) for y in range(2000, 2025) for m in range(1, 13)
        ]
        msl = np.array([float(r["MSL"]) for r in rows])
        result = ClimateResilienceDetector().detect(msl, "sea_level")
        assert result["metrics"]["trend_mm_per_year"] == pytest.approx(6.3242, abs=0.01)
        assert result["anomaly_detected"] is True
        assert result["coastal_flood_risk"] == "elevated"


class TestWindAndFireWeather:
    """Saffir-Simpson wind thresholds and the Angström fire-weather index."""

    def test_hurricane_force_boundary(self) -> None:
        detector = ClimateResilienceDetector()
        below = detector.detect(np.array([32.9]), "wind")
        assert below["event_type"] == "normal"
        assert below["anomaly_detected"] is False
        at_force = detector.detect(np.array([64 * 0.514444]), "wind")
        assert at_force["event_type"] == "hurricane"
        assert at_force["severity"] == "high"
        assert at_force["anomaly_detected"] is True

    def test_major_hurricane_is_critical(self) -> None:
        result = ClimateResilienceDetector().detect(np.array([30.0, 50.0]), "wind")
        assert result["event_type"] == "hurricane"
        assert result["severity"] == "critical"
        assert result["recommendations"][-1].startswith("CRITICAL: Major hurricane")

    def test_wind_validation(self) -> None:
        detector = ClimateResilienceDetector()
        assert detector.detect(np.array([]), "wind") == {"anomaly_detected": False}
        with pytest.raises(ValueError, match="finite, non-negative"):
            detector.detect(np.array([-5.0]), "wind")

    @pytest.mark.parametrize(
        ("temp", "humidity", "event", "severity"),
        [
            (35.0, 15.0, "wildfire", "high"),  # I = -0.05: very likely
            (17.1, 20.0, "wildfire", "high"),  # I = 1.99, just under 2.0
            (17.0, 20.0, "wildfire", "medium"),  # I = 2.0 exactly: favorable band
            (25.0, 45.0, "wildfire", "medium"),  # I = 2.45
            (10.0, 80.0, "normal", "low"),  # I = 5.7
        ],
    )
    def test_angstrom_index_ladder(
        self, temp: float, humidity: float, event: str, severity: str
    ) -> None:
        result = ClimateResilienceDetector().detect(np.array([[temp, humidity]]), "fire_weather")
        assert result["event_type"] == event
        assert result["severity"] == severity
        assert result["metrics"]["worst_angstrom_index"] == pytest.approx(
            humidity / 20.0 + (27.0 - temp) / 10.0
        )

    def test_worst_observation_decides(self) -> None:
        result = ClimateResilienceDetector().detect(
            np.array([[10.0, 80.0], [35.0, 15.0]]), "fire_weather"
        )
        assert result["event_type"] == "wildfire"
        assert result["severity"] == "high"

    def test_single_pair_is_accepted(self) -> None:
        result = ClimateResilienceDetector().detect(np.array([35.0, 15.0]), "fire_weather")
        assert result["event_type"] == "wildfire"

    def test_fire_weather_validation(self) -> None:
        detector = ClimateResilienceDetector()
        with pytest.raises(ValueError, match="rows"):
            detector.detect(np.zeros((2, 3)), "fire_weather")
        with pytest.raises(ValueError, match="humidity"):
            detector.detect(np.array([[20.0, 150.0]]), "fire_weather")


class TestInflationAndRecession:
    """Published inflation thresholds and the technical-recession rule."""

    @pytest.mark.parametrize(
        ("rate", "threat", "severity", "anomaly"),
        [
            (9.99, "normal", "low", False),
            (10.0, "inflation_surge", "medium", True),
            (14.99, "inflation_surge", "medium", True),
            (15.0, "inflation_surge", "high", True),
            (39.99, "inflation_surge", "high", True),
            (40.0, "inflation_surge", "critical", True),
        ],
    )
    def test_inflation_threshold_ladder(
        self, rate: float, threat: str, severity: str, anomaly: bool
    ) -> None:
        result = EconomicResilienceDetector().detect(np.array([rate]), "inflation")
        assert result["threat_type"] == threat
        assert result["severity"] == severity
        assert result["anomaly_detected"] is anomaly

    def test_latest_value_is_graded_and_acceleration_reported(self) -> None:
        result = EconomicResilienceDetector().detect(np.array([50.0, 8.0]), "inflation")
        assert result["anomaly_detected"] is False  # only the latest counts
        assert result["metrics"]["acceleration_pp"] == pytest.approx(-42.0)

    def test_turkiye_2022_fixture_is_a_crisis(self) -> None:
        """World Bank FP.CPI.TOTL.ZG for TUR (recorded fixture, sha256
        fd6b475ceb1a0cde261dd916c7e396e938c92f18d51e5f6e...): 72.3% in 2022
        exceeds the Bruno-Easterly 40% crisis threshold."""
        import json
        from pathlib import Path

        fixture = (
            Path(__file__).resolve().parent.parent
            / "fixtures"
            / "humanitarian"
            / "wb_fp_cpi_totl_zg_tur.json"
        )
        rows = json.loads(fixture.read_text())[1]
        by_year = {r["date"]: r["value"] for r in rows if r["value"] is not None}
        series = np.array([by_year[y] for y in ("2020", "2021", "2022")])
        result = EconomicResilienceDetector().detect(series, "inflation")
        assert result["severity"] == "critical"
        assert result["metrics"]["latest_inflation_pct"] == pytest.approx(72.31, abs=0.01)

    @pytest.mark.parametrize(
        ("growth", "threat", "severity", "streak"),
        [
            ([1.0, -0.5], "normal", "low", 1),
            ([1.0, -0.5, -1.0], "recession_risk", "high", 2),
            ([-1.0, -0.5, -0.2, -0.1], "recession_risk", "critical", 4),
            ([-1.0, -0.5, 0.3], "normal", "low", 0),
        ],
    )
    def test_technical_recession_rule(
        self, growth: list[float], threat: str, severity: str, streak: int
    ) -> None:
        result = EconomicResilienceDetector().detect(np.array(growth), "gdp")
        assert result["threat_type"] == threat
        assert result["severity"] == severity
        assert result["metrics"]["negative_quarter_streak"] == streak

    def test_economic_validation(self) -> None:
        detector = EconomicResilienceDetector()
        assert detector.detect(np.array([]), "inflation") == {"anomaly_detected": False}
        with pytest.raises(ValueError, match="non-finite"):
            detector.detect(np.array([np.inf]), "gdp")


class TestEducationAccessResourcesParity:
    """UIS world benchmark, EFA pupil-teacher benchmark, UIS parity band."""

    @pytest.mark.parametrize(
        ("rate", "threat", "severity"),
        [
            (10.3, "normal", "low"),  # the world rate itself is not a barrier
            (10.4, "access_barrier", "medium"),
            (20.6, "access_barrier", "medium"),  # exactly 2x world
            (20.7, "access_barrier", "high"),
            (41.2, "access_barrier", "high"),  # exactly 4x world
            (41.3, "access_barrier", "critical"),
        ],
    )
    def test_access_barrier_ladder(self, rate: float, threat: str, severity: str) -> None:
        result = EducationEquityDetector().detect(np.array([rate]), "access")
        assert result["threat_type"] == threat
        assert result["severity"] == severity

    def test_world_benchmark_matches_the_recorded_uis_series(self) -> None:
        """The 10.3% constant IS the latest UIS world value (ROFST.1.CP,
        geoUnit 'SDG: World'; recorded fixture sha256
        49b32bef43425aaab82e65accdcb03c097654f07c917801e39acdc4996b6a3cc)."""
        import json
        from pathlib import Path

        fixture = (
            Path(__file__).resolve().parent.parent
            / "fixtures"
            / "humanitarian"
            / "uis_rofst1cp_sdg_world.json"
        )
        records = json.loads(fixture.read_text())["records"]
        latest = max(records, key=lambda r: r["year"])
        assert pytest.approx(latest["value"], abs=0.05) == (
            EducationEquityDetector.WORLD_OUT_OF_SCHOOL_PCT
        )

    @pytest.mark.parametrize(
        ("ratio", "threat", "severity"),
        [
            (40.0, "normal", "low"),
            (40.1, "resource_inequity", "high"),
            (60.0, "resource_inequity", "high"),
            (60.1, "resource_inequity", "critical"),
        ],
    )
    def test_pupil_teacher_ratio_benchmark(self, ratio: float, threat: str, severity: str) -> None:
        result = EducationEquityDetector().detect(np.array([ratio]), "resources")
        assert result["threat_type"] == threat
        assert result["severity"] == severity

    @pytest.mark.parametrize(
        ("gpi", "threat", "severity", "group"),
        [
            (1.0, "normal", "low", "none"),
            (0.97, "normal", "low", "none"),  # the band is inclusive
            (1.03, "normal", "low", "none"),
            (0.96, "bias_detected", "medium", "female"),
            (1.04, "bias_detected", "medium", "male"),
            (0.89, "bias_detected", "high", "female"),
            (1.11, "bias_detected", "high", "male"),
        ],
    )
    def test_gender_parity_band(self, gpi: float, threat: str, severity: str, group: str) -> None:
        result = EducationEquityDetector().detect(np.array([gpi]), "parity")
        assert result["threat_type"] == threat
        assert result["severity"] == severity
        assert result["metrics"]["disadvantaged_group"] == group

    def test_world_gpi_fixture_is_within_parity(self) -> None:
        """World primary GPI (WB SE.ENR.PRIM.FM.ZS, recorded fixture sha256
        cfeabbc2aa2bd3e4ea3f06ed93d5966cf1c92011142c87cec94a8bf32754a26c)
        sits inside the UIS parity band."""
        import json
        from pathlib import Path

        fixture = (
            Path(__file__).resolve().parent.parent
            / "fixtures"
            / "humanitarian"
            / "wb_se_enr_prim_fm_zs_wld.json"
        )
        rows = json.loads(fixture.read_text())[1]
        latest = next(r["value"] for r in rows if r["value"] is not None)
        result = EducationEquityDetector().detect(np.array([latest]), "parity")
        assert result["anomaly_detected"] is False

    def test_education_validation(self) -> None:
        detector = EducationEquityDetector()
        assert detector.detect(np.array([]), "access") == {"anomaly_detected": False}
        with pytest.raises(ValueError, match="percentages"):
            detector.detect(np.array([120.0]), "access")
        with pytest.raises(ValueError, match="positive"):
            detector.detect(np.array([0.0]), "resources")
        with pytest.raises(ValueError, match="positive"):
            detector.detect(np.array([-0.5]), "parity")


class TestSoilPestSupply:
    """FAO/USDA soil thresholds, locust control density, stocks-to-use."""

    _HEALTHY = [2.5, 6.5, 1.3, 1.0]

    def test_healthy_soil_is_normal(self) -> None:
        result = AgriFoodSecurityDetector().detect(np.array(self._HEALTHY), "soil_health")
        assert result["anomaly_detected"] is False
        assert result["severity"] == "low"
        assert result["metrics"]["thresholds_breached"] == []

    @pytest.mark.parametrize(
        ("vector", "breach"),
        [
            ([1.9, 6.5, 1.3, 1.0], "soil_organic_carbon_below_2pct"),
            ([2.5, 5.4, 1.3, 1.0], "ph_outside_optimal_range"),
            ([2.5, 7.6, 1.3, 1.0], "ph_outside_optimal_range"),
            ([2.5, 6.5, 1.7, 1.0], "bulk_density_root_restrictive"),
            ([2.5, 6.5, 1.3, 4.0], "saline_soil"),
        ],
    )
    def test_each_single_breach_grades_medium(self, vector: list[float], breach: str) -> None:
        result = AgriFoodSecurityDetector().detect(np.array(vector), "soil_health")
        assert result["metrics"]["thresholds_breached"] == [breach]
        assert result["severity"] == "medium"
        assert result["threat_type"] == "normal"
        assert result["anomaly_detected"] is True

    def test_soil_threshold_boundaries_exact(self) -> None:
        detector = AgriFoodSecurityDetector()
        # SOC 2.0 is NOT breached (strict <); bulk density 1.6 is NOT (strict >);
        # pH 5.5 and 7.5 are inside (inclusive); salinity 4.0 IS saline (>=).
        boundary = detector.detect(np.array([2.0, 5.5, 1.6, 3.9]), "soil_health")
        assert boundary["metrics"]["thresholds_breached"] == []
        at_75 = detector.detect(np.array([2.0, 7.5, 1.6, 3.9]), "soil_health")
        assert at_75["metrics"]["thresholds_breached"] == []

    def test_multiple_breaches_escalate(self) -> None:
        detector = AgriFoodSecurityDetector()
        two = detector.detect(np.array([1.5, 5.0, 1.3, 1.0]), "soil_health")
        assert two["severity"] == "high"
        assert two["threat_type"] == "crop_failure"
        four = detector.detect(np.array([1.5, 8.2, 1.7, 5.0]), "soil_health")
        assert four["severity"] == "critical"
        assert len(four["metrics"]["thresholds_breached"]) == 4

    def test_sample_matrix_is_judged_by_means(self) -> None:
        samples = np.array([[1.0, 6.5, 1.3, 1.0], [3.0, 6.5, 1.3, 1.0]])  # mean SOC 2.0
        result = AgriFoodSecurityDetector().detect(samples, "soil_health")
        assert result["metrics"]["soil_organic_carbon_pct"] == pytest.approx(2.0)
        assert result["metrics"]["thresholds_breached"] == []

    def test_soil_validation(self) -> None:
        detector = AgriFoodSecurityDetector()
        with pytest.raises(ValueError, match="rows"):
            detector.detect(np.array([1.0, 2.0, 3.0]), "soil_health")
        with pytest.raises(ValueError, match="finite"):
            detector.detect(np.array([np.nan, 6.5, 1.3, 1.0]), "soil_health")

    @pytest.mark.parametrize(
        ("density", "threat", "severity"),
        [
            (499.9, "normal", "low"),
            (500.0, "pest_outbreak", "high"),
            (4999.9, "pest_outbreak", "high"),
            (5000.0, "pest_outbreak", "critical"),
        ],
    )
    def test_locust_control_threshold(self, density: float, threat: str, severity: str) -> None:
        result = AgriFoodSecurityDetector().detect(np.array([density]), "pest")
        assert result["threat_type"] == threat
        assert result["severity"] == severity

    def test_pest_grades_the_latest_observation(self) -> None:
        result = AgriFoodSecurityDetector().detect(np.array([600.0, 100.0]), "pest")
        assert result["anomaly_detected"] is False

    @pytest.mark.parametrize(
        ("ratio", "threat", "severity", "anomaly"),
        [
            (18.0, "normal", "low", False),
            (17.9, "normal", "medium", False),  # tight band
            (17.0, "normal", "medium", False),
            (16.9, "supply_disruption", "high", True),
            (14.0, "supply_disruption", "high", True),
            (13.9, "supply_disruption", "critical", True),
        ],
    )
    def test_stocks_to_use_ladder(
        self, ratio: float, threat: str, severity: str, anomaly: bool
    ) -> None:
        result = AgriFoodSecurityDetector().detect(np.array([ratio]), "supply")
        assert result["threat_type"] == threat
        assert result["severity"] == severity
        assert result["anomaly_detected"] is anomaly

    def test_supply_disruption_recommendations(self) -> None:
        result = AgriFoodSecurityDetector().detect(np.array([13.0]), "supply")
        assert result["recommendations"][0] == "Release strategic reserves to stabilize supply"
        assert result["recommendations"][-1].startswith("CRITICAL: Coordinate international")


class TestMutationKillPins:
    """Close the gaps the 2026-10-20 full enumeration measured (644 sites).

    Each test pins behavior a surviving mutant falsified: empty-input
    contracts, zero-value validity at ``< 0`` guards, exact published
    boundaries, output units, and recommendation branch selection. The
    survivors these leave open are classified in the floor lane's comment.
    """

    # ---- agrifood: soil ----

    def test_soil_empty_input_is_the_bare_negative(self) -> None:
        result = AgriFoodSecurityDetector().detect_soil_health_anomaly(np.array([]))
        assert result == {"anomaly_detected": False}

    def test_soil_single_row_list_is_fully_scored(self) -> None:
        result = AgriFoodSecurityDetector().detect_soil_health_anomaly(
            np.array([[2.5, 6.5, 1.2, 0.5]])
        )
        assert result["anomaly_detected"] is False
        assert result["threat_type"] == "normal"
        assert "metrics" in result

    def test_soil_zero_salinity_is_valid(self) -> None:
        result = AgriFoodSecurityDetector().detect_soil_health_anomaly(
            np.array([2.5, 6.5, 1.2, 0.0])
        )
        assert result["anomaly_detected"] is False

    def test_soil_three_breaches_grade_critical(self) -> None:
        result = AgriFoodSecurityDetector().detect_soil_health_anomaly(
            np.array([1.0, 4.0, 1.7, 0.5])
        )
        assert result["severity"] == "critical"
        assert result["threat_type"] == "crop_failure"
        assert result["metrics"]["thresholds_breached"] == [
            "soil_organic_carbon_below_2pct",
            "ph_outside_optimal_range",
            "bulk_density_root_restrictive",
        ]

    # ---- agrifood: pests and supply ----

    def test_pest_empty_input_is_the_bare_negative(self) -> None:
        result = AgriFoodSecurityDetector().detect_pest_outbreak(np.array([]))
        assert result == {"anomaly_detected": False}

    def test_pest_nan_is_refused(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            AgriFoodSecurityDetector().detect_pest_outbreak(np.array([float("nan")]))

    def test_pest_zero_density_is_valid_and_normal(self) -> None:
        result = AgriFoodSecurityDetector().detect_pest_outbreak(np.array([0.0]))
        assert result["anomaly_detected"] is False
        assert result["threat_type"] == "normal"

    def test_pest_outbreak_recommendations_are_the_pest_branch(self) -> None:
        result = AgriFoodSecurityDetector().detect_pest_outbreak(np.array([600.0]))
        assert result["recommendations"] == [
            "Launch coordinated pest control operations",
            "Survey adjacent areas for spread",
        ]

    def test_supply_empty_input_is_the_bare_negative(self) -> None:
        result = AgriFoodSecurityDetector().detect_supply_disruption(np.array([]))
        assert result == {"anomaly_detected": False}

    def test_supply_nan_is_refused(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            AgriFoodSecurityDetector().detect_supply_disruption(np.array([float("nan")]))

    def test_supply_zero_ratio_is_valid_and_critical(self) -> None:
        result = AgriFoodSecurityDetector().detect_supply_disruption(np.array([0.0]))
        assert result["anomaly_detected"] is True
        assert result["severity"] == "critical"

    # ---- climate: precipitation ----

    def test_rainless_window_grades_high_from_the_zero_total_alone(self) -> None:
        """total <= 0 decides when the z-score does not reach -3."""
        detector = ClimateResilienceDetector()
        detector.precip_baseline = {"mean": 50.0, "std": 20.0}
        result = detector.detect_precipitation_anomaly(np.zeros(5))
        assert result["metrics"]["z_score"] == pytest.approx(-2.5)
        assert result["event_type"] == "drought"
        assert result["severity"] == "high"

    def test_midband_total_is_normal(self) -> None:
        """85 mm sits between +1.4 and +2 sigma: neither flood nor anomaly."""
        result = ClimateResilienceDetector().detect_precipitation_anomaly(np.full(10, 8.5))
        assert result["event_type"] == "normal"
        assert result["anomaly_detected"] is False

    # ---- climate: sea level ----

    @staticmethod
    def _msl_series(months: int, trend_m_yr: float, accel_m_yr2: float = 0.0) -> Any:
        """Exact trend + full annual/semiannual harmonics, no noise."""
        t = np.arange(months, dtype=float) / 12.0
        return (
            trend_m_yr * t
            + 0.5 * accel_m_yr2 * t**2
            + 0.03 * np.cos(2 * np.pi * t)
            + 0.015 * np.sin(2 * np.pi * t)
            + 0.02 * np.cos(4 * np.pi * t)
            + 0.01 * np.sin(4 * np.pi * t)
        )

    def test_sea_level_exact_harmonic_recovery(self) -> None:
        """The Zervas design recovers an exact synthetic trend to 1e-6 mm/yr.

        Kills any mutation of the harmonic design columns or of the
        m-to-mm scale factors: an unmodeled seasonal component leaks into
        the trend estimate far beyond this tolerance.
        """
        result = ClimateResilienceDetector().detect_sea_level_anomaly(self._msl_series(36, 0.005))
        assert result["metrics"]["trend_mm_per_year"] == pytest.approx(5.0, abs=1e-6)
        assert result["metrics"]["acceleration_mm_per_year2"] == pytest.approx(0.0, abs=1e-6)
        assert result["severity"] == "medium"
        assert result["coastal_flood_risk"] == "elevated"

    def test_sea_level_acceleration_branch_grades_medium(self) -> None:
        """Trend below global, acceleration 0.15 mm/yr^2 above Nerem 0.084."""
        result = ClimateResilienceDetector().detect_sea_level_anomaly(
            self._msl_series(36, 0.002, 1.5e-4)
        )
        assert result["metrics"]["acceleration_mm_per_year2"] == pytest.approx(0.15, abs=1e-6)
        assert result["severity"] == "medium"
        assert result["anomaly_detected"] is True
        assert result["coastal_flood_risk"] == "baseline"

    def test_sea_level_trend_six_grades_medium_not_high(self) -> None:
        """High needs twice the global trend (6.8); 6.0 stays medium."""
        result = ClimateResilienceDetector().detect_sea_level_anomaly(self._msl_series(36, 0.006))
        assert result["metrics"]["trend_mm_per_year"] == pytest.approx(6.0, abs=1e-6)
        assert result["severity"] == "medium"

    def test_sea_level_exactly_24_months_is_scored(self) -> None:
        result = ClimateResilienceDetector().detect_sea_level_anomaly(self._msl_series(24, 0.001))
        assert "note" not in result
        assert result["metrics"]["n_months"] == 24

    def test_flood_threshold_uses_the_latest_twelve_months_inclusively(self) -> None:
        """Month -13 is far below threshold; the 12-month mean equals it."""
        series = np.zeros(24)
        series[11] = -13.0
        series[12:] = 1.0
        result = ClimateResilienceDetector().detect_sea_level_anomaly(
            series, {"minor_flood_threshold_m": 1.0}
        )
        assert result["metrics"]["latest_12mo_mean_m"] == pytest.approx(1.0)
        assert result["event_type"] == "flood"
        assert result["severity"] == "critical"

    # ---- climate: wind ----

    def test_calm_zero_wind_is_valid(self) -> None:
        result = ClimateResilienceDetector().detect_wind_anomaly(np.array([0.0]))
        assert result["anomaly_detected"] is False
        assert result["event_type"] == "normal"

    def test_major_hurricane_boundary_is_inclusive_at_96_kt(self) -> None:
        result = ClimateResilienceDetector().detect_wind_anomaly(np.array([96 * 0.514444]))
        assert result["severity"] == "critical"
        assert result["event_type"] == "hurricane"

    def test_wind_kt_conversion_pins_the_nhc_factor(self) -> None:
        result = ClimateResilienceDetector().detect_wind_anomaly(np.array([51.4444]))
        assert result["metrics"]["max_sustained_wind_kt"] == pytest.approx(100.0, rel=1e-9)

    # ---- climate: fire weather ----

    def test_fire_empty_input_is_the_bare_negative(self) -> None:
        result = ClimateResilienceDetector().detect_fire_weather_anomaly(np.array([]))
        assert result == {"anomaly_detected": False}

    def test_fire_single_scalar_fails_with_the_shape_message(self) -> None:
        with pytest.raises(ValueError, match="temp_humidity takes"):
            ClimateResilienceDetector().detect_fire_weather_anomaly(np.array([5.0]))

    def test_fire_range_boundaries_are_valid(self) -> None:
        """Humidity 0 and 100, temperature -90 and 60 are all accepted."""
        result = ClimateResilienceDetector().detect_fire_weather_anomaly(
            np.array([[60.0, 0.0], [-90.0, 100.0]])
        )
        assert result["event_type"] == "wildfire"

    @pytest.mark.parametrize(
        "row",
        [
            [20.0, 100.5],
            [20.0, -0.5],
            [60.5, 50.0],
            [-90.5, 50.0],
        ],
    )
    def test_fire_out_of_range_observations_are_refused(self, row: list[float]) -> None:
        with pytest.raises(ValueError, match="temperature must be finite"):
            ClimateResilienceDetector().detect_fire_weather_anomaly(np.array([row]))

    def test_angstrom_exactly_two_is_favorable_not_very_likely(self) -> None:
        """I = 40/20 + (27-27)/10 = 2.0 exactly: medium, not high."""
        result = ClimateResilienceDetector().detect_fire_weather_anomaly(np.array([[27.0, 40.0]]))
        assert result["severity"] == "medium"
        assert result["event_type"] == "wildfire"

    def test_fire_recommendations_are_the_wildfire_branch(self) -> None:
        high = ClimateResilienceDetector().detect_fire_weather_anomaly(np.array([[35.0, 10.0]]))
        assert high["recommendations"] == [
            "Raise fire-weather readiness and restrict ignition sources",
            "Pre-position firefighting resources",
            "Issue fire-weather warnings to exposed communities",
        ]
        medium = ClimateResilienceDetector().detect_fire_weather_anomaly(np.array([[27.0, 40.0]]))
        assert medium["recommendations"] == [
            "Raise fire-weather readiness and restrict ignition sources",
            "Pre-position firefighting resources",
        ]

    # ---- economic ----

    def test_single_rate_reports_zero_acceleration(self) -> None:
        result = EconomicResilienceDetector().detect_inflation_anomaly(np.array([50.0]))
        assert result["metrics"]["acceleration_pp"] == 0.0

    def test_inflation_crisis_recommendations(self) -> None:
        result = EconomicResilienceDetector().detect_inflation_anomaly(np.array([50.0]))
        assert result["recommendations"] == [
            "Tighten monetary policy coordination with the central bank",
            "Protect purchasing power of vulnerable households",
            "CRITICAL: Inflation above the crisis threshold — stabilization program needed",
        ]

    def test_recession_empty_input_is_the_bare_negative(self) -> None:
        result = EconomicResilienceDetector().detect_recession_risk(np.array([]))
        assert result == {"anomaly_detected": False}

    def test_zero_growth_breaks_the_negative_streak(self) -> None:
        result = EconomicResilienceDetector().detect_recession_risk(np.array([2.0, -1.0, 0.0]))
        assert result["anomaly_detected"] is False
        assert result["metrics"]["negative_quarter_streak"] == 0

    def test_recession_reports_the_latest_quarter(self) -> None:
        result = EconomicResilienceDetector().detect_recession_risk(np.array([1.0, -1.0, -2.5]))
        assert result["metrics"]["latest_growth_pct"] == -2.5
        assert result["severity"] == "high"
        assert result["recommendations"] == [
            "Prepare counter-cyclical fiscal measures",
            "Expand social safety nets ahead of labor-market impact",
        ]

    def test_prolonged_recession_recommendations_escalate(self) -> None:
        result = EconomicResilienceDetector().detect_recession_risk(
            np.array([-1.0, -1.0, -1.0, -2.0])
        )
        assert result["severity"] == "critical"
        assert (
            result["recommendations"][-1]
            == "CRITICAL: Prolonged contraction — coordinate full response"
        )

    # ---- education ----

    def test_zero_out_of_school_is_valid_and_normal(self) -> None:
        result = EducationEquityDetector().detect_access_barrier(np.array([0.0]))
        assert result["anomaly_detected"] is False

    def test_hundred_percent_out_of_school_is_valid(self) -> None:
        result = EducationEquityDetector().detect_access_barrier(np.array([100.0]))
        assert result["anomaly_detected"] is True

    def test_out_of_school_above_hundred_is_refused(self) -> None:
        with pytest.raises(ValueError, match="percentages"):
            EducationEquityDetector().detect_access_barrier(np.array([100.5]))

    def test_ptr_empty_input_is_the_bare_negative(self) -> None:
        result = EducationEquityDetector().detect_resource_inequity(np.array([]))
        assert result == {"anomaly_detected": False}

    def test_ptr_of_one_is_valid_and_normal(self) -> None:
        result = EducationEquityDetector().detect_resource_inequity(np.array([1.0]))
        assert result["anomaly_detected"] is False

    def test_ptr_recommendations_are_the_resource_branch(self) -> None:
        result = EducationEquityDetector().detect_resource_inequity(np.array([45.0]))
        assert result["recommendations"] == [
            "Recruit and deploy additional qualified teachers",
            "Rebalance resource allocation toward under-served schools",
        ]

    def test_gpi_empty_input_is_the_bare_negative(self) -> None:
        result = EducationEquityDetector().detect_gender_parity_bias(np.array([]))
        assert result == {"anomaly_detected": False}

    def test_gpi_zero_is_refused(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            EducationEquityDetector().detect_gender_parity_bias(np.array([0.0]))

    @pytest.mark.parametrize("gpi", [0.9, 1.1])
    def test_gpi_severe_boundaries_are_exclusive(self, gpi: float) -> None:
        """Exactly 0.9 / 1.1 is outside parity but below the high grade."""
        result = EducationEquityDetector().detect_gender_parity_bias(np.array([gpi]))
        assert result["threat_type"] == "bias_detected"
        assert result["severity"] == "medium"

    # ---- neuroscience ----

    def test_gad7_item_above_three_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"integers in 0\.\.3"):
            NeuroscienceDetector().detect(
                np.array([4, 0, 0, 0, 0, 0, 0]), "mental_health", {"instrument": "gad7"}
            )

    def test_asrs_item_above_four_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"integers in 0\.\.4"):
            NeuroscienceDetector().detect(
                np.array([5, 0, 0, 0, 0, 0]), "mental_health", {"instrument": "asrs6"}
            )

    def test_gad7_reports_its_published_operating_point(self) -> None:
        result = NeuroscienceDetector().detect(
            np.array([3, 3, 3, 1, 0, 0, 0]), "mental_health", {"instrument": "gad7"}
        )
        assert result["metrics"]["published_sensitivity"] == 0.89
        assert result["metrics"]["published_specificity"] == 0.82

    def test_asrs_reports_its_published_operating_point(self) -> None:
        result = NeuroscienceDetector().detect(
            np.array([2, 2, 2, 3, 0, 0]), "mental_health", {"instrument": "asrs6"}
        )
        assert result["metrics"]["published_sensitivity"] == 0.687
        assert result["metrics"]["published_specificity"] == 0.995

    def test_gad7_exact_cutpoint_grades_medium_with_referral_first(self) -> None:
        result = NeuroscienceDetector().detect(
            np.array([3, 3, 3, 1, 0, 0, 0]), "mental_health", {"instrument": "gad7"}
        )
        assert result["severity"] == "medium"
        assert result["recommendations"][0] == (
            "Refer for clinical assessment by a licensed mental health professional"
        )

    def test_below_cutpoint_screen_carries_no_referral(self) -> None:
        result = NeuroscienceDetector().detect(
            np.array([1, 1, 1, 1, 1, 0, 0, 0, 0]), "mental_health"
        )
        assert result["severity"] == "low"
        assert all("Refer for clinical assessment" not in r for r in result["recommendations"])

    def test_angstrom_exactly_two_point_five_is_normal(self) -> None:
        """I = 50/20 + (27-27)/10 = 2.5 exactly: the favorable band is exclusive."""
        result = ClimateResilienceDetector().detect_fire_weather_anomaly(np.array([[27.0, 50.0]]))
        assert result["event_type"] == "normal"
        assert result["severity"] == "low"
        assert result["anomaly_detected"] is False

    def test_access_barrier_critical_escalation_is_severity_gated(self) -> None:
        """The CRITICAL access line appears at critical severity and only there."""
        detector = EducationEquityDetector()
        critical = detector.detect_access_barrier(np.array([50.0]))
        assert critical["severity"] == "critical"
        assert critical["recommendations"][-1] == (
            "CRITICAL: Deploy emergency education access programs"
        )
        moderate = detector.detect_access_barrier(np.array([15.0]))
        assert moderate["severity"] != "critical"
        assert all("CRITICAL" not in r for r in moderate["recommendations"])
