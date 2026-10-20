# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioral tests for the humanitarian detectors.

Every test pins exact outputs against the modules' documented baselines
(z-scores, thresholds, severities, recommendations), replacing the
previous instantiation-only checks. Values are hand-computed from the
shipped baselines: agrifood yield 5.0 +/- 1.0 and price 100 +/- 15,
climate temperature 15 +/- 10 and precipitation 100 +/- 50, economic
market 100 +/- 15 and unemployment 5 +/- 2, education achievement
75 +/- 10, neural activity 50 +/- 15.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.infrastructure.humanitarian import (
    AgriFoodSecurityDetector,
    ClimateResilienceDetector,
    EconomicResilienceDetector,
    EducationEquityDetector,
    NeuroscienceDetector,
)


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

        # Documented-but-unimplemented types fall back to the yield path
        # (shipped routing, reported to the owner).
        fallback = detector.detect(np.array([5.0]), detection_type="soil_health")
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
        """0 mm total rainfall is the maximal drought and grades high.

        Regression: with baseline 100 +/- 50 the two-sigma deficit sits
        exactly at 0 mm, so the previous strict comparison made drought
        unreachable for any non-negative rainfall series — a bone-dry
        window was reported normal with routine monitoring.
        """
        result = ClimateResilienceDetector().detect_precipitation_anomaly(np.array([0.0, 0.0]))

        assert result["anomaly_detected"]
        assert result["event_type"] == "drought"
        assert result["severity"] == "high"
        assert result["metrics"]["total_precipitation_mm"] == pytest.approx(0.0)
        assert result["metrics"]["z_score"] == pytest.approx(-2.0)
        assert result["disaster_risk"] == "elevated"
        assert result["recommendations"] == [
            "Implement water conservation measures",
            "Monitor agricultural impacts and food security",
        ]

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
        """Exactly 200 mm (z = 2) detects as a high-severity flood."""
        result = ClimateResilienceDetector().detect_precipitation_anomaly(np.array([200.0]))

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

        dry_normal = detector.detect_precipitation_anomaly(np.array([30.0]))
        assert dry_normal["event_type"] == "normal"
        assert dry_normal["severity"] == "low"
        assert not dry_normal["anomaly_detected"]

        wet_normal = detector.detect_precipitation_anomaly(np.array([170.0]))
        assert wet_normal["event_type"] == "normal"
        assert not wet_normal["anomaly_detected"]

        flood_at_three_sigma = detector.detect_precipitation_anomaly(np.array([250.0]))
        assert flood_at_three_sigma["event_type"] == "flood"
        assert flood_at_three_sigma["severity"] == "high"

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

        The shipped baseline (100 +/- 50) pins the drought branch to its
        0 mm edge, so wetter custom baselines open the branch and pin the
        severity cuts directly.
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
