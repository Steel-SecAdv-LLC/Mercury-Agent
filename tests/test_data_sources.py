# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for STEM Alert & Anomaly Detection Data Sources.

This test suite covers:
- Base classes and interfaces
- Space Weather sources (NASA DONKI, NeoWs, NOAA SWPC, EONET, Solar System)
- Geomagnetic sources (USGS Geomagnetism, INTERMAGNET, SuperMAG, HeartMath, BGS ELF)
- Earth Science sources (USGS Earthquake/Volcano, NOAA NWPS/CO-OPS, NWS, EPA)
- Consciousness Research sources (GCP, GCPDot)
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("httpx")

from datetime import UTC, datetime
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest

from omni_mercury_engine.data_sources import (
    AlertLevel,
    CacheConfig,
    DataPoint,
    DataSourceConfig,
    DataSourceError,
    DataSourceManager,
    DataSourceType,
    FetchResult,
    RateLimitConfig,
)
from omni_mercury_engine.data_sources.consciousness import (
    GCPDataSource,
    GCPDotSource,
    chi_square_deviation,
    cumulative_deviation,
    inter_egg_correlation,
    stouffer_z_score,
)
from omni_mercury_engine.data_sources.earth_science import (
    EPAAirNowSource,
    NOAACOOPSSource,
    NOAANWPSSource,
    NWSWeatherAlertsSource,
    USGSEarthquakeSource,
    USGSVolcanoSource,
)
from omni_mercury_engine.data_sources.geomagnetic import (
    BGSELFStationSource,
    HeartMathGCMSSource,
    HeartMathSite,
    INTERMAGNETSource,
    SuperMAGSource,
    USGSGeomagnetismSource,
    USGSObservatory,
)
from omni_mercury_engine.data_sources.space_weather import (
    DONKIEventType,
    NASADONKISource,
    NASAEONETSource,
    NASANeoWsSource,
    NOAASWPCSource,
    SolarSystemOpenDataSource,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def data_source_config() -> DataSourceConfig:
    """Create a test data source configuration."""
    return DataSourceConfig(
        api_key="test_api_key",
        timeout_seconds=10.0,
        retry_attempts=2,
        rate_limit=RateLimitConfig(requests_per_hour=100, min_interval_seconds=0.1),
        cache=CacheConfig(enabled=True, ttl_seconds=60),
    )


@pytest.fixture
def sample_data_point() -> DataPoint:
    """Create a sample data point."""
    return DataPoint(
        source_id="test_source",
        source_type=DataSourceType.EARTHQUAKE,
        event_id="test_event_001",
        timestamp=datetime.now(UTC),
        data={"magnitude": 5.5, "depth_km": 10.0},
        location=(34.0522, -118.2437, 10.0),
        alert_level=AlertLevel.MODERATE,
        confidence=0.9,
        metadata={"test": True},
    )


@pytest.fixture
def mock_http_response() -> MagicMock:
    """Create a mock HTTP response."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.json.return_value = {}
    response.raise_for_status = MagicMock()
    return response


# =============================================================================
# Base Module Tests
# =============================================================================


class TestDataSourceType:
    """Tests for DataSourceType enum."""

    def test_all_types_defined(self) -> None:
        """Verify all expected data source types are defined."""
        expected_types = [
            "SOLAR_FLARE",
            "CME",
            "GEOMAGNETIC_STORM",
            "SOLAR_WIND",
            "SOLAR_ENERGETIC_PARTICLE",
            "NEAR_EARTH_OBJECT",
            "NATURAL_EVENT",
            "CELESTIAL_BODY",
            "MAGNETOMETER",
            "SCHUMANN_RESONANCE",
            "IONOSPHERIC",
            "ELF_VLF",
            "EARTHQUAKE",
            "VOLCANO",
            "WEATHER_ALERT",
            "FLOOD",
            "TIDE",
            "AIR_QUALITY",
            "RANDOM_NUMBER_GENERATOR",
            "GLOBAL_COHERENCE",
            "CUSTOM",
        ]
        for type_name in expected_types:
            assert hasattr(DataSourceType, type_name)


class TestAlertLevel:
    """Tests for AlertLevel enum."""

    def test_alert_level_ordering(self) -> None:
        """Verify alert levels are properly ordered."""
        assert AlertLevel.NONE.value < AlertLevel.MINOR.value
        assert AlertLevel.MINOR.value < AlertLevel.MODERATE.value
        assert AlertLevel.MODERATE.value < AlertLevel.STRONG.value
        assert AlertLevel.STRONG.value < AlertLevel.SEVERE.value
        assert AlertLevel.SEVERE.value < AlertLevel.EXTREME.value

    def test_from_noaa_g_scale(self) -> None:
        """Test NOAA G-scale conversion."""
        assert AlertLevel.from_noaa_g_scale(0) == AlertLevel.NONE
        assert AlertLevel.from_noaa_g_scale(1) == AlertLevel.MINOR
        assert AlertLevel.from_noaa_g_scale(5) == AlertLevel.EXTREME

    def test_from_nws_severity(self) -> None:
        """Test NWS severity conversion."""
        assert AlertLevel.from_nws_severity("minor") == AlertLevel.MINOR
        assert AlertLevel.from_nws_severity("Moderate") == AlertLevel.MODERATE
        assert AlertLevel.from_nws_severity("EXTREME") == AlertLevel.EXTREME


class TestDataPoint:
    """Tests for DataPoint dataclass."""

    def test_data_point_creation(self, sample_data_point: DataPoint) -> None:
        """Test DataPoint creation."""
        assert sample_data_point.source_id == "test_source"
        assert sample_data_point.source_type == DataSourceType.EARTHQUAKE
        assert sample_data_point.confidence == 0.9

    def test_to_dict(self, sample_data_point: DataPoint) -> None:
        """Test DataPoint serialization."""
        data = sample_data_point.to_dict()
        assert data["source_id"] == "test_source"
        assert data["source_type"] == "earthquake"
        assert data["alert_level"] == AlertLevel.MODERATE.value

    def test_from_dict(self, sample_data_point: DataPoint) -> None:
        """Test DataPoint deserialization."""
        data = sample_data_point.to_dict()
        restored = DataPoint.from_dict(data)
        assert restored.source_id == sample_data_point.source_id
        assert restored.source_type == sample_data_point.source_type


class TestDataSourceConfig:
    """Tests for DataSourceConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = DataSourceConfig()
        assert config.timeout_seconds == 30.0
        assert config.retry_attempts == 3
        assert config.verify_ssl is True

    def test_custom_config(self, data_source_config: DataSourceConfig) -> None:
        """Test custom configuration."""
        assert data_source_config.api_key == "test_api_key"
        assert data_source_config.timeout_seconds == 10.0


class TestFetchResult:
    """Tests for FetchResult."""

    def test_successful_result(self, sample_data_point: DataPoint) -> None:
        """Test successful fetch result."""
        result = FetchResult(
            success=True,
            data_points=[sample_data_point],
            fetch_time_ms=100.0,
        )
        assert result.success is True
        assert len(result.data_points) == 1
        assert result.error is None

    def test_failed_result(self) -> None:
        """Test failed fetch result."""
        result = FetchResult(
            success=False,
            error="API error",
            fetch_time_ms=50.0,
        )
        assert result.success is False
        assert result.error == "API error"
        assert len(result.data_points) == 0


# =============================================================================
# DataSourceManager Tests
# =============================================================================


class TestDataSourceManager:
    """Tests for DataSourceManager."""

    def test_register_source(self) -> None:
        """Test source registration."""
        manager = DataSourceManager()
        source = USGSEarthquakeSource()
        manager.register_source(source)

        assert source.source_id in manager.list_sources()
        assert source.source_id in manager.list_enabled_sources()

    def test_disable_source(self) -> None:
        """Test source disable/enable."""
        manager = DataSourceManager()
        source = USGSEarthquakeSource()
        manager.register_source(source)

        manager.disable_source(source.source_id)
        assert source.source_id not in manager.list_enabled_sources()

        manager.enable_source(source.source_id)
        assert source.source_id in manager.list_enabled_sources()

    def test_unregister_source(self) -> None:
        """Test source unregistration."""
        manager = DataSourceManager()
        source = USGSEarthquakeSource()
        manager.register_source(source)
        manager.unregister_source(source.source_id)

        assert source.source_id not in manager.list_sources()

    def test_get_all_data_points(self, sample_data_point: DataPoint) -> None:
        """Test data point aggregation."""
        manager = DataSourceManager()
        results = {
            "source1": FetchResult(success=True, data_points=[sample_data_point]),
            "source2": FetchResult(success=False, error="Error"),
        }
        points = manager.get_all_data_points(results)
        assert len(points) == 1

    def test_get_metrics(self) -> None:
        """Test metrics collection."""
        manager = DataSourceManager()
        source = USGSEarthquakeSource()
        manager.register_source(source)

        metrics = manager.get_metrics()
        assert metrics["total_sources"] == 1
        assert metrics["enabled_sources"] == 1


# =============================================================================
# Space Weather Source Tests
# =============================================================================


class TestNASADONKISource:
    """Tests for NASA DONKI data source."""

    def test_initialization(self) -> None:
        """Test source initialization."""
        source = NASADONKISource(api_key="test_key")
        assert source.source_id == "nasa_donki"
        assert DataSourceType.SOLAR_FLARE in source.default_source_types

    def test_event_type_mapping(self) -> None:
        """Test DONKI event type to DataSourceType mapping."""
        source = NASADONKISource()
        assert (
            source._event_type_to_source_type(DONKIEventType.SOLAR_FLARE)
            == DataSourceType.SOLAR_FLARE
        )
        assert source._event_type_to_source_type(DONKIEventType.CME) == DataSourceType.CME

    def test_flare_class_parsing(self) -> None:
        """Test solar flare class parsing."""
        source = NASADONKISource()
        assert source._parse_flare_class("X10.0") == AlertLevel.EXTREME
        assert source._parse_flare_class("X5.0") == AlertLevel.SEVERE
        assert source._parse_flare_class("X1.0") == AlertLevel.STRONG
        assert source._parse_flare_class("M5.0") == AlertLevel.MODERATE
        assert source._parse_flare_class("C3.0") == AlertLevel.MINOR

    def test_kp_index_parsing(self) -> None:
        """Test Kp index parsing."""
        source = NASADONKISource()
        assert source._parse_kp_index(9) == AlertLevel.EXTREME
        assert source._parse_kp_index(8) == AlertLevel.SEVERE
        assert source._parse_kp_index(7) == AlertLevel.STRONG
        assert source._parse_kp_index(6) == AlertLevel.MODERATE
        assert source._parse_kp_index(5) == AlertLevel.MINOR
        assert source._parse_kp_index(4) == AlertLevel.NONE


class TestNASANeoWsSource:
    """Tests for NASA NeoWs data source."""

    def test_initialization(self) -> None:
        """Test source initialization."""
        source = NASANeoWsSource(api_key="test_key")
        assert source.source_id == "nasa_neows"
        assert DataSourceType.NEAR_EARTH_OBJECT in source.default_source_types

    def test_hazard_level_calculation(self) -> None:
        """Test hazard level calculation from NEO data."""
        source = NASANeoWsSource()

        # Non-hazardous
        neo: dict[str, Any] = {"is_potentially_hazardous_asteroid": False}
        assert source._calculate_hazard_level(neo) == AlertLevel.NONE

        # Hazardous with close approach
        neo = {
            "is_potentially_hazardous_asteroid": True,
            "close_approach_data": [{"miss_distance": {"kilometers": "100000"}}],  # Very close
        }
        assert source._calculate_hazard_level(neo) == AlertLevel.EXTREME


class TestNOAASWPCSource:
    """Tests for NOAA SWPC data source."""

    def test_initialization(self) -> None:
        """Test source initialization."""
        source = NOAASWPCSource()
        assert source.source_id == "noaa_swpc"
        assert DataSourceType.SOLAR_WIND in source.default_source_types

    def test_kp_alert_level(self) -> None:
        """Test Kp to alert level mapping."""
        source = NOAASWPCSource()
        assert source._parse_kp_alert_level(9) == AlertLevel.EXTREME
        assert source._parse_kp_alert_level(5) == AlertLevel.MINOR
        assert source._parse_kp_alert_level(3) == AlertLevel.NONE


class TestNASAEONETSource:
    """Tests for NASA EONET data source."""

    def test_initialization(self) -> None:
        """Test source initialization."""
        source = NASAEONETSource()
        assert source.source_id == "nasa_eonet"
        assert DataSourceType.NATURAL_EVENT in source.default_source_types

    def test_category_alert_level(self) -> None:
        """Test category to alert level mapping."""
        source = NASAEONETSource()
        assert source._category_to_alert_level("volcanoes", []) == AlertLevel.STRONG
        assert source._category_to_alert_level("wildfires", []) == AlertLevel.MODERATE


class TestSolarSystemOpenDataSource:
    """Tests for Solar System OpenData source."""

    def test_initialization(self) -> None:
        """Test source initialization."""
        source = SolarSystemOpenDataSource()
        assert source.source_id == "solar_system_opendata"
        assert DataSourceType.CELESTIAL_BODY in source.default_source_types


# =============================================================================
# Geomagnetic Source Tests
# =============================================================================


class TestUSGSGeomagnetismSource:
    """Tests for USGS Geomagnetism data source."""

    def test_initialization(self) -> None:
        """Test source initialization."""
        source = USGSGeomagnetismSource()
        assert source.source_id == "usgs_geomagnetism"
        assert DataSourceType.MAGNETOMETER in source.default_source_types

    def test_observatory_coords(self) -> None:
        """Test observatory coordinate lookup."""
        coords = USGSGeomagnetismSource.OBSERVATORY_COORDS
        assert USGSObservatory.BOULDER in coords
        assert coords[USGSObservatory.BOULDER] == (40.137, -105.237)

    def test_disturbance_level(self) -> None:
        """Test magnetic disturbance level calculation."""
        source = USGSGeomagnetismSource()

        # Normal conditions (within 50 nT of typical 20000)
        assert source._calculate_disturbance_level({"H": 20000}) == AlertLevel.NONE

        # Storm conditions - deviation of 1000 nT triggers SEVERE (>500)
        assert source._calculate_disturbance_level({"H": 19000}) == AlertLevel.SEVERE

        # Strong storm - deviation around 350 nT (>300)
        assert source._calculate_disturbance_level({"H": 19650}) == AlertLevel.STRONG

        # Moderate storm - deviation around 200 nT (>150)
        assert source._calculate_disturbance_level({"H": 19800}) == AlertLevel.MODERATE

        # Minor - deviation around 75 nT (>50)
        assert source._calculate_disturbance_level({"H": 19925}) == AlertLevel.MINOR


class TestINTERMAGNETSource:
    """Tests for INTERMAGNET data source."""

    def test_initialization(self) -> None:
        """Test source initialization."""
        source = INTERMAGNETSource()
        assert source.source_id == "intermagnet"
        assert DataSourceType.MAGNETOMETER in source.default_source_types


class TestSuperMAGSource:
    """Tests for SuperMAG data source."""

    def test_initialization(self) -> None:
        """Test source initialization."""
        source = SuperMAGSource()
        assert source.source_id == "supermag"

    def test_index_description(self) -> None:
        """Test index description retrieval."""
        source = SuperMAGSource()
        desc = source._get_index_description("SME")
        assert "electrojet" in desc.lower()


class TestHeartMathGCMSSource:
    """Tests for HeartMath GCMS data source."""

    def test_initialization(self) -> None:
        """Test source initialization."""
        source = HeartMathGCMSSource()
        assert source.source_id == "heartmath_gcms"
        assert DataSourceType.SCHUMANN_RESONANCE in source.default_source_types

    def test_schumann_frequencies(self) -> None:
        """Test Schumann resonance frequencies."""
        source = HeartMathGCMSSource()
        assert 7.83 in source.SCHUMANN_FREQUENCIES
        assert len(source.SCHUMANN_FREQUENCIES) == 6

    def test_site_coordinates(self) -> None:
        """Test site coordinate lookup."""
        coords = HeartMathGCMSSource.SITE_COORDS
        assert HeartMathSite.CALIFORNIA in coords


class TestBGSELFStationSource:
    """Tests for BGS ELF Station data source."""

    def test_initialization(self) -> None:
        """Test source initialization."""
        source = BGSELFStationSource()
        assert source.source_id == "bgs_elf"
        assert DataSourceType.SCHUMANN_RESONANCE in source.default_source_types

    def test_schumann_resonances(self) -> None:
        """Test Schumann resonance definitions."""
        resonances = BGSELFStationSource.SCHUMANN_RESONANCES
        assert resonances["SR1"] == 7.83
        assert len(resonances) == 6


# =============================================================================
# Earth Science Source Tests
# =============================================================================


class TestUSGSEarthquakeSource:
    """Tests for USGS Earthquake data source."""

    def test_initialization(self) -> None:
        """Test source initialization."""
        source = USGSEarthquakeSource(min_magnitude=4.0)
        assert source.source_id == "usgs_earthquake"
        assert DataSourceType.EARTHQUAKE in source.default_source_types

    def test_magnitude_alert_level(self) -> None:
        """Test magnitude to alert level conversion."""
        source = USGSEarthquakeSource()
        assert source._magnitude_to_alert_level(8.5) == AlertLevel.EXTREME
        assert source._magnitude_to_alert_level(7.0) == AlertLevel.SEVERE
        assert source._magnitude_to_alert_level(6.0) == AlertLevel.STRONG
        assert source._magnitude_to_alert_level(5.0) == AlertLevel.MODERATE
        assert source._magnitude_to_alert_level(4.0) == AlertLevel.MINOR
        assert source._magnitude_to_alert_level(3.0) == AlertLevel.NONE


class TestUSGSVolcanoSource:
    """Tests for USGS Volcano data source."""

    def test_initialization(self) -> None:
        """Test source initialization."""
        source = USGSVolcanoSource()
        assert source.source_id == "usgs_volcano"
        assert DataSourceType.VOLCANO in source.default_source_types

    def test_us_volcanoes(self) -> None:
        """Test US volcano definitions."""
        volcanoes = USGSVolcanoSource.US_VOLCANOES
        assert "kilauea" in volcanoes
        assert "yellowstone" in volcanoes


class TestNOAANWPSSource:
    """Tests for NOAA NWPS data source."""

    def test_initialization(self) -> None:
        """Test source initialization."""
        source = NOAANWPSSource()
        assert source.source_id == "noaa_nwps"
        assert DataSourceType.FLOOD in source.default_source_types

    def test_flood_stage_alert(self) -> None:
        """Test flood stage to alert level."""
        source = NOAANWPSSource()
        assert source._flood_stage_to_alert("major flood") == AlertLevel.SEVERE
        assert source._flood_stage_to_alert("moderate flood") == AlertLevel.STRONG
        assert source._flood_stage_to_alert("minor flood") == AlertLevel.MODERATE


class TestNOAACOOPSSource:
    """Tests for NOAA CO-OPS data source."""

    def test_initialization(self) -> None:
        """Test source initialization."""
        source = NOAACOOPSSource(station_id="8518750")
        assert "noaa_coops" in source.source_id
        assert DataSourceType.TIDE in source.default_source_types

    def test_sample_stations(self) -> None:
        """Test sample station definitions."""
        stations = NOAACOOPSSource.SAMPLE_STATIONS
        assert "8518750" in stations  # The Battery, NY


class TestNWSWeatherAlertsSource:
    """Tests for NWS Weather Alerts data source."""

    def test_initialization(self) -> None:
        """Test source initialization."""
        source = NWSWeatherAlertsSource(state="CA")
        assert "nws_alerts" in source.source_id
        assert DataSourceType.WEATHER_ALERT in source.default_source_types

    def test_severity_alert_level(self) -> None:
        """Test severity to alert level conversion."""
        source = NWSWeatherAlertsSource()
        assert source._severity_to_alert_level("Extreme") == AlertLevel.EXTREME
        assert source._severity_to_alert_level("Severe") == AlertLevel.SEVERE
        assert source._severity_to_alert_level("Moderate") == AlertLevel.STRONG


class TestEPAAirNowSource:
    """Tests for EPA AirNow data source."""

    def test_initialization(self) -> None:
        """Test source initialization."""
        source = EPAAirNowSource(api_key="test_key", latitude=37.7749, longitude=-122.4194)
        assert source.source_id == "epa_airnow"
        assert DataSourceType.AIR_QUALITY in source.default_source_types

    def test_aqi_alert_level(self) -> None:
        """Test AQI to alert level conversion."""
        source = EPAAirNowSource(api_key="test")
        assert source._aqi_to_alert_level(350) == AlertLevel.EXTREME
        assert source._aqi_to_alert_level(250) == AlertLevel.SEVERE
        assert source._aqi_to_alert_level(175) == AlertLevel.STRONG
        assert source._aqi_to_alert_level(125) == AlertLevel.MODERATE
        assert source._aqi_to_alert_level(75) == AlertLevel.MINOR
        assert source._aqi_to_alert_level(25) == AlertLevel.NONE

    def test_aqi_category(self) -> None:
        """Test AQI category determination."""
        source = EPAAirNowSource(api_key="test")
        assert source._get_aqi_category(25) == "Good"
        assert source._get_aqi_category(75) == "Moderate"
        assert source._get_aqi_category(125) == "Unhealthy for Sensitive Groups"
        assert source._get_aqi_category(175) == "Unhealthy"


# =============================================================================
# Consciousness Research Source Tests
# =============================================================================


class TestStatisticalFunctions:
    """Tests for GCP statistical analysis functions."""

    def test_stouffer_z_score(self) -> None:
        """Test Stouffer Z-score calculation."""
        # Empty list
        assert stouffer_z_score([]) == 0.0

        # Single value
        assert stouffer_z_score([2.0]) == 2.0

        # Multiple values
        z = stouffer_z_score([1.0, 1.0, 1.0, 1.0])
        assert abs(z - 2.0) < 0.01  # 4 * 1.0 / sqrt(4) = 2.0

    def test_chi_square_deviation(self) -> None:
        """Test chi-square deviation calculation."""
        # Test with realistic binomial data (variance should be near 50)
        np.random.seed(42)
        observed = list(np.random.binomial(200, 0.5, 100))
        chi_sq, p_value = chi_square_deviation(observed)
        # Chi-square statistic should be reasonable for df=99
        assert 50 < chi_sq < 200  # Reasonable range for df=99
        # P-value should indicate non-significant deviation
        assert p_value > 0.001

    def test_cumulative_deviation(self) -> None:
        """Test cumulative deviation calculation."""
        # All at expected value
        trials = [100] * 10
        deviations = cumulative_deviation(trials)
        assert all(d == 0 for d in deviations)

        # Positive deviation
        trials = [101] * 10
        deviations = cumulative_deviation(trials)
        assert deviations[-1] == 10

    def test_inter_egg_correlation(self) -> None:
        """Test inter-EGG correlation calculation."""
        # Independent random
        np.random.seed(42)
        egg_data = {f"egg_{i}": list(np.random.binomial(200, 0.5, 100)) for i in range(5)}
        corr = inter_egg_correlation(egg_data)
        assert abs(corr) < 0.3  # Should be near zero


class TestGCPDataSource:
    """Tests for GCP data source."""

    def test_initialization(self) -> None:
        """Test source initialization."""
        source = GCPDataSource()
        assert source.source_id == "gcp_noosphere"
        assert DataSourceType.GLOBAL_COHERENCE in source.default_source_types

    def test_z_score_alert_level(self) -> None:
        """Test Z-score to alert level conversion."""
        source = GCPDataSource()
        assert source._z_score_to_alert_level(5.0) == AlertLevel.EXTREME
        assert source._z_score_to_alert_level(3.5) == AlertLevel.SEVERE
        assert source._z_score_to_alert_level(3.0) == AlertLevel.STRONG
        assert source._z_score_to_alert_level(2.0) == AlertLevel.MODERATE
        assert source._z_score_to_alert_level(1.0) == AlertLevel.NONE

    def test_egg_data_simulation(self) -> None:
        """Test EGG data simulation."""
        source = GCPDataSource()
        egg_data = source._simulate_egg_data(n_samples=100, n_eggs=5)

        assert len(egg_data) == 5
        for data in egg_data.values():
            assert len(data) == 100
            # Check values are in valid range
            assert all(0 <= v <= 200 for v in data)

    def test_network_analysis(self) -> None:
        """Test network analysis."""
        source = GCPDataSource()
        egg_data = source._simulate_egg_data(n_samples=100, n_eggs=5)
        analysis = source._analyze_network(egg_data)

        assert analysis["n_eggs"] == 5
        assert analysis["n_samples"] == 100
        assert "network_variance" in analysis["analyses"]
        assert "stouffer_z" in analysis["analyses"]


class TestGCPDotSource:
    """Tests for GCPDot data source."""

    def test_initialization(self) -> None:
        """Test source initialization."""
        source = GCPDotSource()
        assert source.source_id == "gcpdot"
        assert DataSourceType.GLOBAL_COHERENCE in source.default_source_types

    def test_deviation_to_color(self) -> None:
        """Test deviation to color mapping."""
        source = GCPDotSource()

        assert source._deviation_to_color(3.0).value == "blue"
        assert source._deviation_to_color(1.0).value == "green"
        assert source._deviation_to_color(0.0).value == "yellow"
        assert source._deviation_to_color(-1.0).value == "red"


# =============================================================================
# Integration Tests
# =============================================================================


class TestDataSourceIntegration:
    """Integration tests for data source system."""

    @pytest.mark.asyncio
    async def test_manager_fetch_all(self) -> None:
        """Test fetching from multiple sources via manager."""
        manager = DataSourceManager()

        # Register multiple sources
        manager.register_source(USGSEarthquakeSource())
        manager.register_source(USGSVolcanoSource())

        # Fetch all (will use simulated/cached data)
        # Note: In real tests, mock the HTTP calls
        assert len(manager.list_enabled_sources()) == 2

    def test_data_point_filtering(self, sample_data_point: DataPoint) -> None:
        """Test data point filtering by type."""
        manager = DataSourceManager()
        results = {"source": FetchResult(success=True, data_points=[sample_data_point])}

        # Filter by matching type
        points = manager.get_all_data_points(results, filter_types=[DataSourceType.EARTHQUAKE])
        assert len(points) == 1

        # Filter by non-matching type
        points = manager.get_all_data_points(results, filter_types=[DataSourceType.VOLCANO])
        assert len(points) == 0

    def test_confidence_filtering(self, sample_data_point: DataPoint) -> None:
        """Test data point filtering by confidence."""
        manager = DataSourceManager()
        results = {"source": FetchResult(success=True, data_points=[sample_data_point])}

        # Below threshold
        points = manager.get_all_data_points(results, min_confidence=0.95)
        assert len(points) == 0

        # Above threshold
        points = manager.get_all_data_points(results, min_confidence=0.5)
        assert len(points) == 1


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_data_source_error(self) -> None:
        """Test DataSourceError exception."""
        error = DataSourceError(
            "Test error",
            source_id="test_source",
            status_code=500,
            retryable=True,
        )
        assert str(error) == "Test error"
        assert error.source_id == "test_source"
        assert error.status_code == 500
        assert error.retryable is True

    def test_fetch_result_error(self) -> None:
        """Test failed FetchResult."""
        result = FetchResult(
            success=False,
            error="Connection timeout",
            rate_limited=False,
        )
        assert not result.success
        assert result.error == "Connection timeout"
        assert len(result.data_points) == 0


# =============================================================================
# Caching Tests
# =============================================================================


class TestCaching:
    """Tests for caching functionality."""

    def test_cache_config(self) -> None:
        """Test cache configuration."""
        config = CacheConfig(enabled=True, ttl_seconds=120, max_entries=500)
        assert config.enabled is True
        assert config.ttl_seconds == 120
        assert config.max_entries == 500

    def test_rate_limit_config(self) -> None:
        """Test rate limit configuration."""
        config = RateLimitConfig(
            requests_per_hour=500,
            min_interval_seconds=2.0,
            burst_limit=20,
        )
        assert config.requests_per_hour == 500
        assert config.min_interval_seconds == 2.0
        assert config.burst_limit == 20


# =============================================================================
# Feature Vector Tests
# =============================================================================


class TestFeatureVector:
    """Tests for DataPoint.to_feature_vector() method."""

    def test_feature_vector_default_dim(self, sample_data_point: DataPoint) -> None:
        """Test feature vector with default dimension."""
        features = sample_data_point.to_feature_vector()
        assert features.shape == (32,)
        assert features.dtype == np.float32

    def test_feature_vector_custom_dim(self, sample_data_point: DataPoint) -> None:
        """Test feature vector with custom dimension."""
        features = sample_data_point.to_feature_vector(feature_dim=16)
        assert features.shape == (16,)

    def test_feature_vector_alert_level_normalized(self, sample_data_point: DataPoint) -> None:
        """Test that alert level is normalized to [0, 1]."""
        features = sample_data_point.to_feature_vector()
        # AlertLevel.MODERATE = 2, normalized: 2/5 = 0.4
        assert abs(features[0] - 0.4) < 0.01

    def test_feature_vector_confidence(self, sample_data_point: DataPoint) -> None:
        """Test that confidence is included."""
        features = sample_data_point.to_feature_vector()
        # Confidence = 0.9 (index 1)
        assert features[1] == 0.9

    def test_feature_vector_location(self, sample_data_point: DataPoint) -> None:
        """Test that location is included and normalized."""
        features = sample_data_point.to_feature_vector()
        # Location features start at index 10 (after alert, confidence, 8 one-hot)
        lat_normalized = 34.0522 / 90.0
        lon_normalized = -118.2437 / 180.0
        alt_normalized = min(10.0 / 1000.0, 1.0)
        assert abs(features[10] - lat_normalized) < 0.01
        assert abs(features[11] - lon_normalized) < 0.01
        assert abs(features[12] - alt_normalized) < 0.01

    def test_feature_vector_no_location(self) -> None:
        """Test feature vector when location is None."""
        point = DataPoint(
            source_id="test",
            source_type=DataSourceType.EARTHQUAKE,
            event_id="test_001",
            timestamp=datetime.now(UTC),
            data={},
            location=None,
        )
        features = point.to_feature_vector()
        # Location features should be zeros
        assert features[10] == 0.0
        assert features[11] == 0.0
        assert features[12] == 0.0


# =============================================================================
# Location Validation Tests
# =============================================================================


class TestLocationValidation:
    """Tests for DataPoint location tuple validation."""

    def test_from_dict_with_3_element_location(self) -> None:
        """Test from_dict with 3-element location tuple."""
        data = {
            "source_id": "test",
            "source_type": "earthquake",
            "event_id": "test_001",
            "timestamp": datetime.now(UTC).isoformat(),
            "data": {},
            "location": [34.0, -118.0, 10.0],
        }
        point = DataPoint.from_dict(data)
        assert point.location == (34.0, -118.0, 10.0)

    def test_from_dict_with_2_element_location(self) -> None:
        """Test from_dict with 2-element location tuple (adds default altitude)."""
        data = {
            "source_id": "test",
            "source_type": "earthquake",
            "event_id": "test_001",
            "timestamp": datetime.now(UTC).isoformat(),
            "data": {},
            "location": [34.0, -118.0],
        }
        point = DataPoint.from_dict(data)
        assert point.location == (34.0, -118.0, 0.0)

    def test_from_dict_with_no_location(self) -> None:
        """Test from_dict with no location."""
        data = {
            "source_id": "test",
            "source_type": "earthquake",
            "event_id": "test_001",
            "timestamp": datetime.now(UTC).isoformat(),
            "data": {},
        }
        point = DataPoint.from_dict(data)
        assert point.location is None

    def test_from_dict_with_empty_location(self) -> None:
        """Test from_dict with empty location list."""
        data = {
            "source_id": "test",
            "source_type": "earthquake",
            "event_id": "test_001",
            "timestamp": datetime.now(UTC).isoformat(),
            "data": {},
            "location": [],
        }
        point = DataPoint.from_dict(data)
        assert point.location is None


# =============================================================================
# Async Context Manager Tests
# =============================================================================


class TestAsyncContextManager:
    """Tests for async context manager functionality."""

    @pytest.mark.asyncio
    async def test_async_context_manager_entry_exit(self) -> None:
        """Test async context manager enter and exit."""
        source = USGSEarthquakeSource()

        async with source as src:
            assert src is source
            # Client should be available but may not be created yet

        # After exit, client should be closed
        assert source._client is None

    @pytest.mark.asyncio
    async def test_async_context_manager_cleanup_on_exception(self) -> None:
        """Test that resources are cleaned up on exception."""
        source = USGSEarthquakeSource()

        try:
            async with source as src:
                # Force client creation
                await src._get_client()
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Client should still be closed
        assert source._client is None


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Tests for health check functionality."""

    def test_is_healthy_initial_state(self) -> None:
        """Test initial health state is True."""
        source = USGSEarthquakeSource()
        assert source.is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_with_mock_success(self) -> None:
        """Test health check with successful fetch."""
        source = USGSEarthquakeSource()

        # Mock the fetch method to return success
        async def mock_fetch(*args: Any, **kwargs: Any) -> FetchResult:
            return FetchResult(success=True, data_points=[])

        source.fetch = mock_fetch  # type: ignore[method-assign]
        result = await source.health_check()

        assert result is True
        assert source.is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_with_mock_failure(self) -> None:
        """Test health check with failed fetch."""
        source = USGSEarthquakeSource()

        # Mock the fetch method to raise an exception
        async def mock_fetch(*args: Any, **kwargs: Any) -> FetchResult:
            raise Exception("Connection failed")

        source.fetch = mock_fetch  # type: ignore[method-assign]
        result = await source.health_check()

        assert result is False
        assert source.is_healthy is False

    def test_get_metrics_includes_health(self) -> None:
        """Test that get_metrics includes health status."""
        source = USGSEarthquakeSource()
        metrics = source.get_metrics()
        assert "is_healthy" in metrics
        assert metrics["is_healthy"] is True


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety features."""

    def test_cache_lock_exists(self) -> None:
        """Test that cache lock is initialized."""
        source = USGSEarthquakeSource()
        import threading

        assert isinstance(source._cache_lock, type(threading.Lock()))

    def test_rate_limit_lock_exists(self) -> None:
        """Test that rate limit lock is initialized."""
        source = USGSEarthquakeSource()
        import threading

        assert isinstance(source._rate_limit_lock, type(threading.Lock()))

    def test_metrics_lock_exists(self) -> None:
        """Test that metrics lock is initialized."""
        source = USGSEarthquakeSource()
        import threading

        assert isinstance(source._metrics_lock, type(threading.Lock()))

    def test_concurrent_cache_access(self) -> None:
        """Test concurrent cache access doesn't raise exceptions."""
        import concurrent.futures

        source = USGSEarthquakeSource()

        def cache_operation(i: int) -> None:
            # Simulate cache access
            key = f"test_key_{i}"
            source._set_cached(key, [])
            source._get_cached(key)
            source.clear_cache()

        # Run multiple threads accessing cache
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(cache_operation, i) for i in range(20)]
            for future in concurrent.futures.as_completed(futures):
                future.result()  # Will raise if there was an exception


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
