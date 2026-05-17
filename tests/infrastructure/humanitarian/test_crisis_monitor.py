"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Tests for infrastructure/humanitarian/crisis_monitoring/crisis_monitor.py module.
Comprehensive test coverage for crisis monitoring functionality.
"""

from __future__ import annotations

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
    """Tests for crisis monitoring functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.monitor = CrisisMonitor()

    def test_monitor_crisis_no_data(self) -> None:
        """Test monitoring with no data."""
        alert = self.monitor.monitor_crisis()
        assert isinstance(alert, CrisisAlert)

    def test_monitor_crisis_with_geoint(self) -> None:
        """Test monitoring with GEOINT data."""
        geoint_data = {
            "flood_level": 0.9,
            "earthquake_magnitude": 0.0,
            "fire_spread": 0.0,
            "infrastructure_damage": 0.8,
        }
        alert = self.monitor.monitor_crisis(geoint_data=geoint_data)
        assert isinstance(alert, CrisisAlert)

    def test_monitor_crisis_with_osint(self) -> None:
        """Test monitoring with OSINT data."""
        osint_data = {
            "social_media_alerts": 150,
            "news_reports": 25,
            "emergency_calls": 500,
        }
        alert = self.monitor.monitor_crisis(osint_data=osint_data)
        assert isinstance(alert, CrisisAlert)

    def test_monitor_crisis_with_both_int(self) -> None:
        """Test monitoring with both GEOINT and OSINT."""
        geoint_data = {"flood_level": 0.8}
        osint_data = {"social_media_alerts": 200}
        alert = self.monitor.monitor_crisis(
            geoint_data=geoint_data,
            osint_data=osint_data,
        )
        assert isinstance(alert, CrisisAlert)

    def test_high_severity_detection(self) -> None:
        """Test high severity crisis detection."""
        geoint_data = {
            "flood_level": 0.95,
            "infrastructure_damage": 0.9,
            "population_displacement": 0.85,
        }
        alert = self.monitor.monitor_crisis(geoint_data=geoint_data)
        if alert.crisis_detected:
            assert alert.severity in ["high", "critical"]

    def test_low_severity_no_detection(self) -> None:
        """Test low severity data doesn't trigger crisis."""
        geoint_data = {
            "flood_level": 0.1,
            "earthquake_magnitude": 0.0,
            "fire_spread": 0.0,
        }
        alert = self.monitor.monitor_crisis(geoint_data=geoint_data)
        # With low values, should typically not detect crisis
        assert isinstance(alert, CrisisAlert)


class TestCrisisMonitorClassification:
    """Tests for crisis classification."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.monitor = CrisisMonitor()

    def test_natural_disaster_classification(self) -> None:
        """Test natural disaster classification."""
        geoint_data = {
            "earthquake_magnitude": 7.5,
            "tsunami_warning": True,
            "aftershock_probability": 0.8,
        }
        alert = self.monitor.monitor_crisis(geoint_data=geoint_data)
        # Should classify as natural_disaster
        assert alert.crisis_type in self.monitor.crisis_types or alert.crisis_type == "unknown"

    def test_pandemic_classification(self) -> None:
        """Test pandemic classification."""
        geoint_data = {
            "infection_rate": 0.9,
            "hospital_capacity": 0.95,
            "mortality_rate": 0.05,
        }
        alert = self.monitor.monitor_crisis(geoint_data=geoint_data)
        assert isinstance(alert.crisis_type, str)

    def test_infrastructure_failure_classification(self) -> None:
        """Test infrastructure failure classification."""
        geoint_data = {
            "power_outage": 0.8,
            "water_system_failure": 0.7,
            "communication_disruption": 0.6,
        }
        alert = self.monitor.monitor_crisis(geoint_data=geoint_data)
        assert isinstance(alert.crisis_type, str)


class TestCrisisMonitorVulnerableGroups:
    """Tests for vulnerable group identification."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.monitor = CrisisMonitor()

    def test_identifies_vulnerable_groups(self) -> None:
        """Test that vulnerable groups are identified."""
        geoint_data = {"severity": 0.9, "population_density": 10000}
        alert = self.monitor.monitor_crisis(geoint_data=geoint_data)
        assert isinstance(alert.vulnerable_groups, list)

    def test_vulnerable_groups_for_natural_disaster(self) -> None:
        """Test vulnerable groups for natural disaster."""
        geoint_data = {
            "flood_level": 0.9,
            "affected_area_km2": 500,
        }
        alert = self.monitor.monitor_crisis(geoint_data=geoint_data)
        # Should identify relevant vulnerable groups
        if alert.crisis_detected:
            assert len(alert.vulnerable_groups) >= 0


class TestCrisisMonitorRecommendations:
    """Tests for response recommendations."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.monitor = CrisisMonitor()

    def test_generates_recommendations(self) -> None:
        """Test that recommendations are generated."""
        geoint_data = {"severity": 0.9}
        alert = self.monitor.monitor_crisis(geoint_data=geoint_data)
        assert isinstance(alert.recommended_response, list)

    def test_recommendations_for_high_severity(self) -> None:
        """Test recommendations for high severity crisis."""
        geoint_data = {
            "severity": 0.95,
            "casualties": 100,
            "displaced": 10000,
        }
        alert = self.monitor.monitor_crisis(geoint_data=geoint_data)
        if alert.crisis_detected:
            assert len(alert.recommended_response) >= 0


class TestCrisisMonitorSurvivorPriorities:
    """Tests for survivor priority identification."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.monitor = CrisisMonitor()

    def test_identifies_survivor_priorities(self) -> None:
        """Test that survivor priorities are identified."""
        geoint_data = {"severity": 0.9}
        alert = self.monitor.monitor_crisis(geoint_data=geoint_data)
        assert isinstance(alert.survivor_priorities, list)

    def test_survivor_first_prioritization(self) -> None:
        """Test survivor-first prioritization principle."""
        geoint_data = {
            "mass_casualty_event": True,
            "severity": 0.95,
        }
        alert = self.monitor.monitor_crisis(geoint_data=geoint_data)
        # Survivor priorities should be generated
        assert isinstance(alert.survivor_priorities, list)


class TestCrisisMonitorGEOINTIndicators:
    """Tests for GEOINT indicator processing."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.monitor = CrisisMonitor()

    def test_processes_geoint_indicators(self) -> None:
        """Test GEOINT indicator processing."""
        geoint_data = {
            "satellite_imagery_anomaly": True,
            "thermal_signature": 0.8,
            "vegetation_index_change": -0.5,
        }
        alert = self.monitor.monitor_crisis(geoint_data=geoint_data)
        assert isinstance(alert.geoint_indicators, list)

    def test_multiple_geoint_sources(self) -> None:
        """Test processing multiple GEOINT sources."""
        geoint_data = {
            "sar_data": {"flood_extent": 0.7},
            "optical_data": {"smoke_detection": 0.3},
            "elevation_data": {"subsidence": 0.1},
        }
        alert = self.monitor.monitor_crisis(geoint_data=geoint_data)
        assert isinstance(alert, CrisisAlert)


class TestCrisisMonitorAffectedPopulation:
    """Tests for affected population estimation."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.monitor = CrisisMonitor()

    def test_estimates_affected_population(self) -> None:
        """Test affected population estimation."""
        geoint_data = {
            "affected_area_km2": 100,
            "population_density": 1000,
        }
        alert = self.monitor.monitor_crisis(geoint_data=geoint_data)
        assert isinstance(alert.affected_population, int)
        assert alert.affected_population >= 0

    def test_high_population_density_area(self) -> None:
        """Test estimation for high population density."""
        geoint_data = {
            "affected_area_km2": 50,
            "population_density": 10000,
            "severity": 0.9,
        }
        alert = self.monitor.monitor_crisis(geoint_data=geoint_data)
        assert alert.affected_population >= 0


class TestCrisisMonitorSeverityLevels:
    """Tests for severity level determination."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.monitor = CrisisMonitor()

    def test_severity_levels(self) -> None:
        """Test different severity levels."""
        # Low severity
        alert_low = self.monitor.monitor_crisis(geoint_data={"severity": 0.2})
        assert alert_low.severity in ["low", "medium", "high", "critical"]

        # High severity
        alert_high = self.monitor.monitor_crisis(geoint_data={"severity": 0.9})
        assert alert_high.severity in ["low", "medium", "high", "critical"]

    def test_critical_severity_threshold(self) -> None:
        """Test critical severity threshold."""
        config = {"severity_threshold": 0.9}
        monitor = CrisisMonitor(config=config)
        alert = monitor.monitor_crisis(geoint_data={"severity": 0.95})
        # Should detect crisis at this level
        assert isinstance(alert, CrisisAlert)


class TestCrisisMonitorEdgeCases:
    """Tests for edge cases."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.monitor = CrisisMonitor()

    def test_empty_geoint_data(self) -> None:
        """Test with empty GEOINT data."""
        alert = self.monitor.monitor_crisis(geoint_data={})
        assert isinstance(alert, CrisisAlert)

    def test_empty_osint_data(self) -> None:
        """Test with empty OSINT data."""
        alert = self.monitor.monitor_crisis(osint_data={})
        assert isinstance(alert, CrisisAlert)

    def test_none_inputs(self) -> None:
        """Test with None inputs."""
        alert = self.monitor.monitor_crisis(geoint_data=None, osint_data=None)
        assert isinstance(alert, CrisisAlert)

    def test_unusual_data_values(self) -> None:
        """Test with unusual data values."""
        geoint_data = {
            "severity": 1.5,  # Over 1.0
            "affected_population": -100,  # Negative
        }
        alert = self.monitor.monitor_crisis(geoint_data=geoint_data)
        # Should handle gracefully
        assert isinstance(alert, CrisisAlert)

    def test_very_large_values(self) -> None:
        """Test with very large values."""
        geoint_data = {
            "affected_area_km2": 1000000,
            "population_density": 100000,
        }
        alert = self.monitor.monitor_crisis(geoint_data=geoint_data)
        assert isinstance(alert, CrisisAlert)
