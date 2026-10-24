# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for infrastructure/humanitarian/crisis_monitoring/crisis_monitor.py module. Comprehensive test coverage for crisis monitoring functionality."""

from __future__ import annotations

import math

import numpy as np
import pytest

from omni_mercury_engine.infrastructure.humanitarian.crisis_monitoring.crisis_monitor import (
    CrisisAlert,
    CrisisMonitor,
)


class TestCrisisAlert:
    """Tests for CrisisAlert dataclass."""

    def test_basic_alert(self) -> None:
        """Test basic crisis alert creation."""
        alert = CrisisAlert(
            crisis_detected=True,
            crisis_type="natural_disaster",
            severity="high",
            affected_population=10000,
            vulnerable_groups=["elderly", "children"],
            survivor_priorities=["medical_aid", "shelter"],
            geoint_indicators=["flooding", "infrastructure_damage"],
            recommended_response=["evacuate", "deploy_aid"],
        )
        assert alert.crisis_detected is True
        assert alert.crisis_type == "natural_disaster"
        assert alert.severity == "high"
        assert alert.affected_population == 10000

    def test_no_crisis_alert(self) -> None:
        """Test alert when no crisis detected."""
        alert = CrisisAlert(
            crisis_detected=False,
            crisis_type="none",
            severity="low",
            affected_population=0,
            vulnerable_groups=[],
            survivor_priorities=[],
            geoint_indicators=[],
            recommended_response=[],
        )
        assert alert.crisis_detected is False
        assert len(alert.vulnerable_groups) == 0

    def test_alert_with_vulnerable_groups(self) -> None:
        """Test alert with vulnerable groups identified."""
        alert = CrisisAlert(
            crisis_detected=True,
            crisis_type="pandemic",
            severity="critical",
            affected_population=1000000,
            vulnerable_groups=["elderly", "immunocompromised", "children", "pregnant_women"],
            survivor_priorities=["vaccines", "ventilators", "hospital_beds"],
            geoint_indicators=["hospital_overflow", "morgue_capacity"],
            recommended_response=["lockdown", "vaccine_distribution", "hospital_expansion"],
        )
        assert len(alert.vulnerable_groups) == 4
        assert "elderly" in alert.vulnerable_groups

    def test_alert_survivor_priorities(self) -> None:
        """Test alert with survivor priorities."""
        alert = CrisisAlert(
            crisis_detected=True,
            crisis_type="humanitarian_emergency",
            severity="high",
            affected_population=50000,
            vulnerable_groups=["refugees", "displaced_persons"],
            survivor_priorities=["water", "food", "shelter", "medical_care"],
            geoint_indicators=["camp_overcrowding", "water_shortage"],
            recommended_response=["aid_convoy", "water_purification", "tent_deployment"],
        )
        assert "water" in alert.survivor_priorities
        assert "food" in alert.survivor_priorities


class TestCrisisMonitorInitialization:
    """Tests for CrisisMonitor initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        monitor = CrisisMonitor()
        assert monitor.config == {}
        assert monitor.severity_threshold == 0.7
        assert len(monitor.crisis_types) > 0

    def test_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = {"severity_threshold": 0.5, "alert_level": "high"}
        monitor = CrisisMonitor(config=config)
        assert monitor.severity_threshold == 0.5

    def test_crisis_types_available(self) -> None:
        """Test crisis types are defined."""
        monitor = CrisisMonitor()
        assert "natural_disaster" in monitor.crisis_types
        assert "pandemic" in monitor.crisis_types
        assert "humanitarian_emergency" in monitor.crisis_types

    def test_logger_initialized(self) -> None:
        """Test logger is initialized."""
        monitor = CrisisMonitor()
        assert monitor.logger is not None


class TestCrisisMonitorMonitoring:
    """Score fusion, the strict detection threshold, and the default alert."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.monitor = CrisisMonitor()

    def test_monitor_crisis_no_data_full_default_alert(self) -> None:
        """No reports: nothing detected, nothing invented."""
        alert = self.monitor.monitor_crisis()
        assert alert.crisis_detected is False
        assert alert.crisis_type == "humanitarian_emergency"
        assert alert.severity == "low"
        assert alert.affected_population is None
        assert alert.geoint_indicators == []
        assert alert.gdacs_level is None and alert.gdacs_score is None
        assert alert.vulnerable_groups == [
            "Elderly",
            "Children",
            "Disabled individuals",
            "Low-income families",
        ]
        # Exposure is unknown, so the response asks for it before scaling.
        assert any("exposed population" in r for r in alert.recommended_response)
        assert alert.recommended_response[-1] == (
            "Prioritize vulnerable populations in all response efforts"
        )

    def test_irrelevant_geoint_keys_carry_no_score(self) -> None:
        """Only ``threat_score`` scores a report; other keys never do."""
        alert = self.monitor.monitor_crisis(
            geoint_data={"flood_level": 0.95, "infrastructure_damage": 0.9}
        )
        assert alert.crisis_detected is False
        assert alert.severity == "low"

    def test_osint_score_and_indicators_fuse(self) -> None:
        alert = self.monitor.monitor_crisis(
            osint_data={"threat_score": 0.8, "indicators": ["river flooding"]}
        )
        assert alert.crisis_detected is True
        assert alert.crisis_type == "natural_disaster"
        assert alert.severity == "high"

    def test_fusion_takes_the_highest_report_score(self) -> None:
        alert = self.monitor.monitor_crisis(
            geoint_data={"threat_score": 0.6},
            osint_data={"threat_score": 0.95},
        )
        assert alert.severity == "critical"

    def test_detection_threshold_is_strict(self) -> None:
        at = self.monitor.monitor_crisis(geoint_data={"threat_score": 0.7})
        assert at.crisis_detected is False
        over = self.monitor.monitor_crisis(geoint_data={"threat_score": 0.71})
        assert over.crisis_detected is True

    def test_custom_threshold_is_honored(self) -> None:
        monitor = CrisisMonitor(config={"severity_threshold": 0.9})
        assert monitor.monitor_crisis(geoint_data={"threat_score": 0.85}).crisis_detected is False
        assert monitor.monitor_crisis(geoint_data={"threat_score": 0.95}).crisis_detected is True


class TestCrisisMonitorClassification:
    """Indicator patterns type the crisis; no match falls back."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.monitor = CrisisMonitor()

    def test_natural_disaster_classification(self) -> None:
        alert = self.monitor.monitor_crisis(
            geoint_data={"indicators": ["earthquake M7.5", "tsunami warning"]}
        )
        assert alert.crisis_type == "natural_disaster"

    def test_pandemic_classification(self) -> None:
        alert = self.monitor.monitor_crisis(geoint_data={"indicators": ["measles epidemic"]})
        assert alert.crisis_type == "pandemic"

    def test_infrastructure_failure_classification(self) -> None:
        alert = self.monitor.monitor_crisis(
            geoint_data={"indicators": ["regional power outage", "grid failure"]}
        )
        assert alert.crisis_type == "infrastructure_failure"

    def test_mass_casualty_classification(self) -> None:
        alert = self.monitor.monitor_crisis(geoint_data={"indicators": ["MCI declared"]})
        assert alert.crisis_type == "mass_casualty"

    def test_unmatched_indicators_fall_back(self) -> None:
        alert = self.monitor.monitor_crisis(geoint_data={"indicators": ["unrelated note"]})
        assert alert.crisis_type == "humanitarian_emergency"


class TestCrisisMonitorVulnerableGroups:
    """The group list is the base four plus crisis- and severity-specific adds."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.monitor = CrisisMonitor()

    _BASE = ["Elderly", "Children", "Disabled individuals", "Low-income families"]

    def test_low_severity_keeps_the_base_groups(self) -> None:
        alert = self.monitor.monitor_crisis(geoint_data={"threat_score": 0.2})
        assert alert.vulnerable_groups == self._BASE

    def test_pandemic_adds_medical_groups(self) -> None:
        alert = self.monitor.monitor_crisis(
            geoint_data={"threat_score": 0.2, "indicators": ["cholera outbreak"]}
        )
        assert alert.vulnerable_groups == self._BASE + [
            "Immunocompromised individuals",
            "Healthcare workers",
        ]

    def test_natural_disaster_adds_exposure_groups(self) -> None:
        alert = self.monitor.monitor_crisis(
            geoint_data={"threat_score": 0.2, "indicators": ["flood"]}
        )
        assert alert.vulnerable_groups == self._BASE + [
            "Homeless population",
            "Rural communities",
        ]

    def test_high_severity_adds_general_population(self) -> None:
        alert = self.monitor.monitor_crisis(geoint_data={"threat_score": 0.8})
        assert alert.vulnerable_groups[-1] == "General population"
        medium = self.monitor.monitor_crisis(geoint_data={"threat_score": 0.6})
        assert "General population" not in medium.vulnerable_groups


class TestCrisisMonitorRecommendations:
    """Response actions keyed to severity, exposure, and crisis type."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.monitor = CrisisMonitor()

    def test_high_severity_activates_national_response(self) -> None:
        alert = self.monitor.monitor_crisis(geoint_data={"threat_score": 0.8})
        assert alert.recommended_response[:3] == [
            "Activate national emergency response (FEMA, Red Cross)",
            "Deploy search and rescue teams",
            "Establish emergency medical facilities",
        ]
        medium = self.monitor.monitor_crisis(geoint_data={"threat_score": 0.6})
        assert "Deploy search and rescue teams" not in medium.recommended_response

    def test_international_aid_needs_more_than_100k_exposed(self) -> None:
        at = self.monitor.monitor_crisis(geoint_data={"affected_population": 100_000})
        assert "Request international humanitarian aid" not in at.recommended_response
        over = self.monitor.monitor_crisis(geoint_data={"affected_population": 100_001})
        assert "Request international humanitarian aid" in over.recommended_response
        assert "Coordinate with UN agencies (UNHCR, WHO, WFP)" in over.recommended_response

    def test_type_specific_actions(self) -> None:
        disaster = self.monitor.monitor_crisis(geoint_data={"indicators": ["wildfire"]})
        assert "Establish temporary shelters" in disaster.recommended_response
        pandemic = self.monitor.monitor_crisis(geoint_data={"indicators": ["flu epidemic"]})
        assert "Activate pandemic response protocols" in pandemic.recommended_response
        assert "Deploy mobile testing and vaccination units" in pandemic.recommended_response


class TestCrisisMonitorSurvivorPriorities:
    """Tests for survivor priority identification."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.monitor = CrisisMonitor()

    def test_non_pandemic_priorities_exact_order(self) -> None:
        alert = self.monitor.monitor_crisis(geoint_data={"threat_score": 0.9})
        assert alert.survivor_priorities == [
            "Immediate medical attention for critically injured",
            "Evacuation of vulnerable groups: Elderly, Children, Disabled individuals",
            "Secure food, water, and shelter for displaced populations",
            "Establish communication channels for family reunification",
        ]

    def test_civilization_first_prioritization(self) -> None:
        """The most-exposed people are prioritised first, in a fixed, auditable order."""
        geoint_data = {
            "mass_casualty_event": True,
            "severity": 0.95,
        }
        alert = self.monitor.monitor_crisis(geoint_data=geoint_data)
        priorities = alert.survivor_priorities
        # Life-critical care leads; evacuation names the groups actually identified
        # as vulnerable for this crisis, ahead of logistics and reunification.
        assert priorities[0].startswith("Immediate medical attention")
        evacuation = next(p for p in priorities if p.startswith("Evacuation of vulnerable groups"))
        assert priorities.index(evacuation) == 1
        for group in alert.vulnerable_groups[:3]:
            assert group in evacuation
        assert priorities[-1].startswith("Establish communication channels")

    def test_pandemic_puts_containment_ahead_of_everything(self) -> None:
        """In a pandemic, quarantine and treatment of the infected precede all other aid."""
        alert = self.monitor.monitor_crisis(
            geoint_data={"severity": 0.9, "indicators": ["pandemic outbreak cluster"]},
        )
        assert alert.crisis_type == "pandemic"
        assert alert.survivor_priorities[0].startswith("Quarantine and treatment")
        assert alert.survivor_priorities[1].startswith("Immediate medical attention")


class TestCrisisMonitorGEOINTIndicators:
    """Indicators pass through as strings, in order."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.monitor = CrisisMonitor()

    def test_indicators_are_stringified_in_order(self) -> None:
        alert = self.monitor.monitor_crisis(
            geoint_data={"indicators": ["thermal signature", 42, "subsidence"]}
        )
        assert alert.geoint_indicators == ["thermal signature", "42", "subsidence"]

    def test_a_single_string_is_one_indicator(self) -> None:
        alert = self.monitor.monitor_crisis(geoint_data={"indicators": "flash flooding"})
        assert alert.geoint_indicators == ["flash flooding"]

    def test_no_indicators_is_an_empty_list(self) -> None:
        alert = self.monitor.monitor_crisis(geoint_data={"threat_score": 0.5})
        assert alert.geoint_indicators == []


class TestCrisisMonitorAffectedPopulation:
    """Exposure arithmetic: supplied count wins, else density x area, else None."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.monitor = CrisisMonitor()

    def test_density_times_area_exact(self) -> None:
        alert = self.monitor.monitor_crisis(
            geoint_data={"affected_area_km2": 100, "population_density": 1000}
        )
        assert alert.affected_population == 100_000

    def test_high_population_density_area(self) -> None:
        alert = self.monitor.monitor_crisis(
            geoint_data={"affected_area_km2": 50, "population_density": 10000, "severity": 0.9}
        )
        assert alert.affected_population == 500_000

    def test_result_is_rounded_not_truncated(self) -> None:
        alert = self.monitor.monitor_crisis(
            geoint_data={"affected_area_km2": 1.0, "population_density": 999.6}
        )
        assert alert.affected_population == 1000

    def test_density_without_area_is_unknown(self) -> None:
        alert = self.monitor.monitor_crisis(geoint_data={"population_density": 1000})
        assert alert.affected_population is None


class TestCrisisMonitorSeverityLevels:
    """The exact score-to-severity ladder with strict boundaries."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.monitor = CrisisMonitor()

    @pytest.mark.parametrize(
        ("score", "severity"),
        [
            (0.95, "critical"),
            (0.9, "high"),  # strict > 0.9
            (0.71, "high"),
            (0.7, "medium"),  # strict > 0.7
            (0.55, "medium"),
            (0.5, "low"),  # strict > 0.5
            (0.2, "low"),
        ],
    )
    def test_severity_ladder(self, score: float, severity: str) -> None:
        alert = self.monitor.monitor_crisis(geoint_data={"threat_score": score})
        assert alert.severity == severity


class TestCrisisMonitorEdgeCases:
    """Hostile and missing inputs degrade to the documented defaults."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.monitor = CrisisMonitor()

    def test_empty_reports_equal_the_no_data_alert(self) -> None:
        baseline = self.monitor.monitor_crisis()
        empty = self.monitor.monitor_crisis(geoint_data={}, osint_data={})
        assert empty == baseline

    def test_none_inputs(self) -> None:
        alert = self.monitor.monitor_crisis(geoint_data=None, osint_data=None)
        assert alert.crisis_detected is False
        assert alert.severity == "low"

    def test_overrange_score_clips_and_negative_population_is_ignored(self) -> None:
        alert = self.monitor.monitor_crisis(
            geoint_data={"threat_score": 1.5, "affected_population": -100}
        )
        assert alert.severity == "critical"  # clipped to 1.0
        assert alert.affected_population is None

    def test_very_large_values(self) -> None:
        alert = self.monitor.monitor_crisis(
            geoint_data={"affected_area_km2": 1_000_000, "population_density": 100_000}
        )
        assert alert.affected_population == 100_000_000_000


class TestCrisisMonitorCorrectness:
    """Exposure comes from the report, and both reports type the crisis.

    The released monitor multiplied a default density of 10,000 by the score
    (a lone score of 0.8 meant 80,000 affected), ignored OSINT indicators and
    the ``affected_area_km2`` the tests above pass, and could never return
    ``mass_casualty``. The tests marked as such fail on the released code.
    """

    def test_score_alone_gives_unknown_exposure(self) -> None:
        """Fails on the released monitor (80,000)."""
        alert = CrisisMonitor().monitor_crisis(geoint_data={"threat_score": 0.8})
        assert alert.crisis_detected
        assert alert.affected_population is None
        assert any("exposed population" in r for r in alert.recommended_response)

    def test_density_times_area(self) -> None:
        """Fails on the released monitor (density x score x 10)."""
        alert = CrisisMonitor().monitor_crisis(
            geoint_data={"threat_score": 0.8, "population_density": 1000, "affected_area_km2": 100}
        )
        assert alert.affected_population == 100_000

    def test_supplied_exposure_wins_and_invalid_values_are_ignored(self) -> None:
        monitor = CrisisMonitor()
        direct = {"affected_population": 2500, "population_density": 10, "affected_area_km2": 1}
        assert monitor.monitor_crisis(geoint_data=direct).affected_population == 2500
        bad = {"affected_population": -100, "population_density": float("nan")}
        assert monitor.monitor_crisis(geoint_data=bad).affected_population is None

    def test_osint_indicators_type_the_crisis(self) -> None:
        """Fails on the released monitor (``humanitarian_emergency``)."""
        alert = CrisisMonitor().monitor_crisis(
            geoint_data={"threat_score": 0.95}, osint_data={"indicators": ["Earthquake M7.1"]}
        )
        assert alert.crisis_type == "natural_disaster"

    def test_mass_casualty_is_reachable(self) -> None:
        """Fails on the released monitor, which never returned it."""
        alert = CrisisMonitor().monitor_crisis(
            geoint_data={"threat_score": 0.95, "indicators": ["mass_casualty_incident"]}
        )
        assert alert.crisis_type == "mass_casualty"
        assert any("triage" in r for r in alert.recommended_response)

    def test_cause_wins_over_mass_casualty(self) -> None:
        alert = CrisisMonitor().monitor_crisis(
            geoint_data={"indicators": ["mass casualties", "tsunami inundation"]}
        )
        assert alert.crisis_type == "natural_disaster"

    def test_matching_is_per_indicator_and_on_word_boundaries(self) -> None:
        monitor = CrisisMonitor()
        assert monitor._classify_crisis({"indicators": ["commercial fire"]}, None) == (
            "humanitarian_emergency"
        )
        assert monitor._classify_crisis({"indicators": "flash flooding"}, None) == (
            "natural_disaster"
        )

    def test_outbreak_of_violence_is_not_a_pandemic(self) -> None:
        monitor = CrisisMonitor()
        for text in ("outbreak of violence", "outbreak_of_fighting", "outbreak of hostilities"):
            assert monitor._classify_crisis({}, {"indicators": [text]}) != "pandemic"
        assert monitor._classify_crisis({"indicators": ["cholera outbreak"]}, None) == "pandemic"

    def test_invalid_scores_are_clipped_or_ignored(self) -> None:
        monitor = CrisisMonitor()
        assert monitor.monitor_crisis(geoint_data={"threat_score": 1.5}).severity == "critical"
        assert monitor.monitor_crisis(geoint_data={"threat_score": float("nan")}).severity == "low"
        assert monitor.monitor_crisis(geoint_data={"threat_score": "high"}).severity == "low"
        assert monitor.monitor_crisis(geoint_data={"threat_score": -2}).severity == "low"
        assert monitor.monitor_crisis(geoint_data={"threat_score": True}).severity == "low"


class TestCrisisMonitorGdacs:
    """GDACS levels from the inputs of its published models (see ``test_gdacs``)."""

    _MMI = {"VII": 0, "VIII": 0, "IX": 1_000_000}  # global score -0.59 + 0.53 x 7 = 3.12

    def test_red_earthquake_is_a_crisis_without_a_report_score(self) -> None:
        alert = CrisisMonitor().monitor_crisis(
            geoint_data={"hazard": "Earthquake", "population_by_mmi": self._MMI, "inform_lcc": 5.1}
        )
        assert alert.gdacs_level == "red"
        assert alert.gdacs_score == pytest.approx(3.12)
        assert alert.crisis_detected
        assert alert.recommended_response[0].startswith("GDACS Red")

    def test_country_terms_and_coping_capacity_apply(self) -> None:
        exposure = {"VII": 100_000, "VIII": 20_000, "IX": 1_000}  # global score 1.849
        base = {"hazard": "earthquake", "population_by_mmi": exposure, "inform_lcc": 5.1}
        monitor = CrisisMonitor()
        assert monitor.monitor_crisis(geoint_data=base).gdacs_level == "orange"
        japan = monitor.monitor_crisis(geoint_data={**base, "country_iso3": "JPN"})
        assert japan.gdacs_level == "green"
        assert not japan.crisis_detected
        fragile = monitor.monitor_crisis(geoint_data={**base, "coping_capacity_factor": 1.5})
        assert fragile.gdacs_level == "red"
        assert fragile.gdacs_score == pytest.approx(1.5 * (-0.59 + 0.53 * math.log10(40_000)))

    def test_overrides_replace_country_terms(self) -> None:
        alert = CrisisMonitor().monitor_crisis(
            geoint_data={
                "hazard": "earthquake",
                "population_by_mmi": {"VII": 100_000, "VIII": 20_000, "IX": 1_000},
                "country_iso3": "JPN",
                "gdacs_coefficients": {"vulnerability_shakemap": 0.0},
                "inform_lcc": 5.1,
            }
        )
        assert alert.gdacs_level == "orange"

    def test_missing_country_and_coping_capacity_are_stated(self) -> None:
        alert = CrisisMonitor().monitor_crisis(
            geoint_data={"hazard": "earthquake", "population_by_mmi": self._MMI}
        )
        assert "GDACS score uses the global model (no country_iso3 supplied)" in (
            alert.recommended_response
        )
        assert any("no coping-capacity factor" in r for r in alert.recommended_response)

    def test_parameter_model_without_shakemap(self) -> None:
        alert = CrisisMonitor().monitor_crisis(
            geoint_data={
                "indicators": ["earthquake M7.0"],
                "magnitude": 7.0,
                "depth_km": 10.0,
                "population_within_km": {
                    "20": 50_000,
                    "50": 200_000,
                    "75": 400_000,
                    "100": 600_000,
                },
                "country_iso3": "USA",
                "inform_lcc": 5.1,
            }
        )
        assert alert.gdacs_level == "orange"
        assert alert.gdacs_score == pytest.approx(1.753927, abs=1e-6)

    def test_flood_level_from_deaths_or_displaced(self) -> None:
        monitor = CrisisMonitor()
        orange = monitor.monitor_crisis(
            geoint_data={"indicators": ["river flooding"], "deaths": 150}
        )
        assert orange.gdacs_level == "orange" and orange.crisis_detected
        assert orange.gdacs_score is None
        red = monitor.monitor_crisis(geoint_data={"hazard": "flood", "displaced": 900_000})
        assert red.gdacs_level == "red"

    @pytest.mark.parametrize(
        ("extra", "reason"),
        [
            ({"population_by_mmi": {"7": 100}}, "takes VII, VIII and IX"),
            ({"population_by_mmi": [1, 2, 3]}, "must be a mapping"),
            ({"population_by_mmi": {"IX": 10}, "gdacs_coefficients": {"c3": 1}}, "unknown"),
            ({"population_by_mmi": {"IX": 10}, "country_iso3": "Japan"}, "alpha-3"),
            ({"population_by_mmi": {"IX": 10}, "inform_lcc": 12}, "inform_lcc"),
            ({"magnitude": 7.0, "population_within_km": [20, 50]}, "must be a mapping"),
            (
                {"magnitude": 7.0, "population_within_km": dict.fromkeys((20, 50, 75, 100), 1)},
                "number",
            ),
        ],
    )
    def test_invalid_inputs_give_no_level_and_a_reason(
        self, extra: dict[str, object], reason: str
    ) -> None:
        alert = CrisisMonitor().monitor_crisis(geoint_data={"hazard": "earthquake", **extra})
        assert alert.gdacs_level is None and alert.gdacs_score is None
        assert any(
            r.startswith("GDACS level not computed") and reason in r
            for r in alert.recommended_response
        )

    def test_level_sets_severity_and_type(self) -> None:
        """Red is critical and Orange high, with no score or indicators supplied."""
        monitor = CrisisMonitor()
        red = monitor.monitor_crisis(
            geoint_data={"hazard": "earthquake", "population_by_mmi": self._MMI, "inform_lcc": 9.3}
        )
        assert (red.severity, red.crisis_type) == ("critical", "natural_disaster")
        assert "Activate national emergency response (FEMA, Red Cross)" in red.recommended_response
        assert "General population" in red.vulnerable_groups
        orange = monitor.monitor_crisis(geoint_data={"hazard": "flood", "deaths": 150})
        assert orange.severity == "high"
        stronger = monitor.monitor_crisis(
            geoint_data={"hazard": "flood", "deaths": 150, "threat_score": 0.95}
        )
        assert stronger.severity == "critical"  # the score can raise it further

    def test_flood_levels_are_described_as_floods(self) -> None:
        alert = CrisisMonitor().monitor_crisis(geoint_data={"hazard": "flood", "deaths": 900})
        assert alert.recommended_response[0] == (
            "GDACS Orange flood: more than 100 dead or 80,000 displaced"
        )

    def test_hazard_follows_the_inputs(self) -> None:
        monitor = CrisisMonitor()
        quake = monitor.monitor_crisis(
            geoint_data={"population_by_mmi": self._MMI, "country_iso3": "TUR", "inform_lcc": 5}
        )
        assert quake.gdacs_level == "red" and quake.crisis_type == "natural_disaster"
        flood = monitor.monitor_crisis(
            geoint_data={"indicators": ["earthquake", "flooding"], "deaths": 5000, "displaced": 1e6}
        )
        assert flood.gdacs_level == "red"
        assert flood.recommended_response[0].startswith("GDACS Red flood")

    def test_incomplete_inputs_are_named(self) -> None:
        alert = CrisisMonitor().monitor_crisis(
            geoint_data={"hazard": "earthquake", "magnitude": 7.8, "depth_km": 10}
        )
        assert alert.gdacs_level is None
        assert (
            "GDACS earthquake level needs shakemap_scaled_population, population_by_mmi, "
            "or magnitude, depth_km and population_within_km"
        ) in alert.recommended_response
        flood = CrisisMonitor().monitor_crisis(geoint_data={"hazard": "flood"})
        assert "GDACS flood level needs deaths or displaced" in flood.recommended_response

    @pytest.mark.parametrize(
        ("geoint", "reason"),
        [
            ({"hazard": "flood", "deaths": 10**400}, "too large"),
            ({"population_by_mmi": {"IX": 10**400}}, "too large"),
            ({"population_by_mmi": {"IX": np.True_}}, "expected a number"),
            (
                {"magnitude": 7.0, "depth_km": 10, "population_within_km": {math.inf: 1}},
                "finite",
            ),
            (
                {
                    "magnitude": 7.0,
                    "depth_km": 10,
                    "population_within_km": {20.9: 1, 50: 2, 75: 3, 100: 4},
                },
                "whole number",
            ),
            (
                {
                    "magnitude": 7.0,
                    "depth_km": 10,
                    "population_within_km": {20: 5, "20": 0, 50: 6, 75: 7, 100: 8},
                },
                "twice",
            ),
        ],
    )
    def test_hostile_numbers_give_a_reason_not_a_crash(
        self, geoint: dict[str, object], reason: str
    ) -> None:
        """Fails on the first GDACS integration (OverflowError, silent truncation)."""
        alert = CrisisMonitor().monitor_crisis(geoint_data=geoint)
        assert alert.gdacs_level is None
        assert any(
            r.startswith("GDACS level not computed") and reason in r
            for r in alert.recommended_response
        )

    # Melissa (2025) at Jamaica: 125 kt, 996,963 people in the 64 kt buffer; GDACS Red.
    _JAMAICA = {
        "max_wind_kt": 125,
        "population_34kt": 2_500_000,
        "population_64kt": 996_963,
        "country_iso3": "JAM",
    }

    def test_one_cyclone_point_on_the_report(self) -> None:
        alert = CrisisMonitor().monitor_crisis(geoint_data={"hazard": "Hurricane", **self._JAMAICA})
        assert (alert.gdacs_level, alert.gdacs_score) == ("red", 2.5)
        assert (alert.severity, alert.crisis_type) == ("critical", "natural_disaster")
        assert alert.crisis_detected
        assert alert.recommended_response[0].startswith("GDACS Red tropical cyclone (wind)")
        assert any("10%-of-country thresholds not applied" in r for r in alert.recommended_response)

    def test_track_points_take_the_highest_and_cap_far_reds(self) -> None:
        far_red = {**self._JAMAICA, "lead_time_hours": 93}
        weak = {
            "max_wind_kt": 45,
            "population_34kt": 1e5,
            "population_64kt": 0,
            "country_iso3": "JAM",
        }
        monitor = CrisisMonitor()
        capped = monitor.monitor_crisis(
            geoint_data={"cyclone_points": [weak, far_red], "country_population": 2.8e6}
        )
        assert capped.gdacs_level == "orange"
        assert not any("10%" in r for r in capped.recommended_response)
        landfall = monitor.monitor_crisis(geoint_data={"cyclone_points": [weak, self._JAMAICA]})
        assert landfall.gdacs_level == "red"

    def test_storm_surge_is_reported_beside_the_wind_level(self) -> None:
        alert = CrisisMonitor().monitor_crisis(
            geoint_data={"indicators": ["Typhoon"], **self._JAMAICA, "max_storm_surge_m": 3.4}
        )
        assert alert.gdacs_score == 2.5
        assert (
            "GDACS storm surge level red (3.4 m), reported beside the wind level and outside "
            "its score"
        ) in alert.recommended_response

    @pytest.mark.parametrize(
        ("geoint", "reason"),
        [
            ({"hazard": "typhoon"}, "GDACS tropical cyclone level needs cyclone_points"),
            (
                {"max_wind_kt": 120, "population_34kt": 1e6, "population_64kt": 2e5},
                "needs country_iso3 or tc_vulnerability",
            ),
            ({"cyclone_points": {"max_wind_kt": 120}}, "must be a list"),
            ({"cyclone_points": [{"max_wind_kt": 120}]}, "needs population_34kt, population_64kt"),
            ({**_JAMAICA, "tc_vulnerability": "extreme"}, "vulnerability must be one of"),
        ],
    )
    def test_cyclone_inputs_are_checked(self, geoint: dict[str, object], reason: str) -> None:
        alert = CrisisMonitor().monitor_crisis(geoint_data=geoint)
        assert alert.gdacs_level is None
        assert any(reason in r for r in alert.recommended_response)

    def test_open_ocean_points_need_no_country(self) -> None:
        alert = CrisisMonitor().monitor_crisis(
            geoint_data={
                "cyclone_points": [{"max_wind_kt": 140, "population_34kt": 0, "population_64kt": 0}]
            }
        )
        assert alert.gdacs_level == "green" and alert.gdacs_score == 0.5

    def test_gdacs_scaled_population_is_accepted(self) -> None:
        """GDACS's own ScaledPopulation reproduces its score (event 1508467, AFG: 3.048)."""
        alert = CrisisMonitor().monitor_crisis(
            geoint_data={
                "hazard": "earthquake",
                "shakemap_scaled_population": 10**3.69931730102138,
                "country_iso3": "AFG",
                "coping_capacity_factor": 1.28571428571429,
            }
        )
        assert alert.gdacs_score == pytest.approx(3.04796336083887, rel=1e-12)
        assert alert.gdacs_level == "red"

    def test_explicit_none_scaled_population_falls_through(self) -> None:
        alert = CrisisMonitor().monitor_crisis(
            geoint_data={
                "shakemap_scaled_population": None,
                "population_by_mmi": self._MMI,
                "country_iso3": "TUR",
                "inform_lcc": 5,
            }
        )
        assert alert.gdacs_level == "red"

    @pytest.mark.parametrize("surge", [-1.0, "high", math.nan])
    def test_a_bad_storm_surge_keeps_the_wind_level(self, surge: object) -> None:
        alert = CrisisMonitor().monitor_crisis(
            geoint_data={**self._JAMAICA, "max_storm_surge_m": surge}
        )
        assert (alert.gdacs_level, alert.gdacs_score) == ("red", 2.5)
        assert any(
            r.startswith("GDACS storm surge level not computed") for r in alert.recommended_response
        )

    def test_none_on_a_point_falls_back_to_the_report(self) -> None:
        point = {**self._JAMAICA, "country_iso3": None, "country_population": None}
        alert = CrisisMonitor().monitor_crisis(
            geoint_data={
                "cyclone_points": [point],
                "country_iso3": "JAM",
                "country_population": 2.8e6,
            }
        )
        assert alert.gdacs_level == "red"
        assert not any("10%" in r for r in alert.recommended_response)

    def test_earthquake_and_cyclone_inputs_need_the_hazard_named(self) -> None:
        both = {**self._JAMAICA, "population_by_mmi": self._MMI}
        monitor = CrisisMonitor()
        unresolved = monitor.monitor_crisis(geoint_data=both)
        assert unresolved.gdacs_level is None
        assert any("state hazard" in r for r in unresolved.recommended_response)
        named = monitor.monitor_crisis(geoint_data={**both, "indicators": ["earthquake"]})
        assert named.recommended_response[0].startswith("GDACS Red earthquake")
        stated = monitor.monitor_crisis(geoint_data={**both, "hazard": "hurricane"})
        assert stated.recommended_response[0].startswith("GDACS Red tropical cyclone")

    def test_bool_counts_are_not_numbers(self) -> None:
        alert = CrisisMonitor().monitor_crisis(geoint_data={"hazard": "flood", "deaths": True})
        assert alert.gdacs_level is None

    def test_reports_without_gdacs_inputs_are_unchanged(self) -> None:
        alert = CrisisMonitor().monitor_crisis(
            geoint_data={"threat_score": 0.3, "indicators": ["earthquake"]}
        )
        assert alert.gdacs_level is None and not alert.crisis_detected
        assert not any("GDACS" in r for r in alert.recommended_response)


class TestMutationKillPins:
    """Pins that close the gaps the full-enumeration mutation run measured."""

    def test_affected_population_of_zero_is_reported_not_unknown(self) -> None:
        """An explicit zero exposure is a statement, not missing data."""
        alert = CrisisMonitor().monitor_crisis(geoint_data={"affected_population": 0})
        assert alert.affected_population == 0

    def test_cyclone_points_as_string_is_refused(self) -> None:
        alert = CrisisMonitor().monitor_crisis(
            geoint_data={"hazard": "typhoon", "cyclone_points": "abc"}
        )
        assert alert.gdacs_level is None
        assert any("must be a list" in r for r in alert.recommended_response)

    def test_either_exposure_band_requires_a_country(self) -> None:
        """34 kt-only and 64 kt-only exposure each need vulnerability data.

        A single exposed person is exposure: the requirement starts at one,
        not at some higher count.
        """
        monitor = CrisisMonitor()
        for exposure in (
            {"population_34kt": 1, "population_64kt": 0},
            {"population_34kt": 0, "population_64kt": 1},
        ):
            one_person = monitor.monitor_crisis(geoint_data={"max_wind_kt": 120, **exposure})
            assert one_person.gdacs_level is None
            assert any("needs country_iso3" in r for r in one_person.recommended_response)
        p34_only = monitor.monitor_crisis(
            geoint_data={"max_wind_kt": 120, "population_34kt": 1e6, "population_64kt": 0}
        )
        assert p34_only.gdacs_level is None
        assert any("needs country_iso3" in r for r in p34_only.recommended_response)
        p64_only = monitor.monitor_crisis(
            geoint_data={"max_wind_kt": 120, "population_34kt": 0, "population_64kt": 2e5}
        )
        assert p64_only.gdacs_level is None
        assert any("needs country_iso3" in r for r in p64_only.recommended_response)

    def test_country_population_note_tracks_64kt_exposure_exactly(self) -> None:
        monitor = CrisisMonitor()
        # Nobody in the 64 kt band: the 10%-of-country note must NOT appear.
        open_ocean = monitor.monitor_crisis(
            geoint_data={
                "cyclone_points": [{"max_wind_kt": 140, "population_34kt": 0, "population_64kt": 0}]
            }
        )
        assert open_ocean.gdacs_level == "green"
        assert not any("10%" in r for r in open_ocean.recommended_response)
        # A single exposed person without a country population triggers it.
        one_person = monitor.monitor_crisis(
            geoint_data={
                "cyclone_points": [
                    {
                        "max_wind_kt": 140,
                        "population_34kt": 0,
                        "population_64kt": 1,
                        "country_iso3": "JAM",
                    }
                ]
            }
        )
        assert any("10%" in r for r in one_person.recommended_response)

    def test_flood_inputs_without_any_flood_indication_stay_ungraded(self) -> None:
        """Deaths alone, with no stated hazard and no indicators, name no hazard."""
        alert = CrisisMonitor().monitor_crisis(geoint_data={"deaths": 150})
        assert alert.gdacs_level is None
        assert not any(r.startswith("GDACS") for r in alert.recommended_response)
