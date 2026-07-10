# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""STEM Alert & Anomaly Detection API Integration.

Production-grade data ingestion modules for multi-domain anomaly detection:
- Space Weather & Solar Physics (NASA DONKI, NeoWs, NOAA SWPC, EONET)
- Geomagnetic & Electromagnetic Monitoring (USGS, INTERMAGNET, HeartMath)
- Earth Science & Hazards (USGS Earthquake/Volcano, NOAA, NWS, EPA)
- Consciousness Research (Global Consciousness Project)

Usage:
    from omni_mercury_engine.data_sources import DataSourceManager, DataSourceType
    from omni_mercury_engine.data_sources.space_weather import NASADONKISource

    manager = DataSourceManager()
    manager.register_source(NASADONKISource(api_key="your_key"))
    data = await manager.fetch_all()
"""

from omni_mercury_engine.data_sources.base import (
    AlertLevel,
    CacheConfig,
    CircuitBreakerConfig,
    DataPoint,
    DataSourceBase,
    DataSourceConfig,
    DataSourceError,
    DataSourceManager,
    DataSourceType,
    FetchResult,
    RateLimitConfig,
    SourceUnreachableError,
)
from omni_mercury_engine.data_sources.consciousness import (
    GCPDataSource,
    GCPDotSource,
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
    INTERMAGNETSource,
    SuperMAGSource,
    USGSGeomagnetismSource,
)
from omni_mercury_engine.data_sources.jpl_ssd import (
    CloseApproachEvent,
    FireballEvent,
    JPLFireballSource,
    JPLSentrySource,
    SentryImpactRisk,
    close_approaches_from_neows_datapoints,
    fireball_events_from_datapoints,
    sentry_risks_from_datapoints,
)
from omni_mercury_engine.data_sources.live_ingestion import (
    LiveDataError,
    LiveFetch,
    SimulatedDataError,
    fetch_live_datapoints,
)
from omni_mercury_engine.data_sources.space_weather import (
    NASADONKISource,
    NASAEONETSource,
    NASANeoWsSource,
    NOAASWPCSource,
    SolarSystemOpenDataSource,
)

__all__ = [
    # Base types
    "AlertLevel",
    # Geomagnetic
    "BGSELFStationSource",
    "CacheConfig",
    "CircuitBreakerConfig",
    # NASA/JPL SSD
    "CloseApproachEvent",
    "DataPoint",
    "DataSourceBase",
    "DataSourceConfig",
    "DataSourceError",
    "DataSourceManager",
    "DataSourceType",
    # Earth Science
    "EPAAirNowSource",
    "FetchResult",
    "FireballEvent",
    # Consciousness Research
    "GCPDataSource",
    "GCPDotSource",
    "HeartMathGCMSSource",
    "INTERMAGNETSource",
    "JPLFireballSource",
    "JPLSentrySource",
    # Live ingestion seam
    "LiveDataError",
    "LiveFetch",
    # Space Weather
    "NASADONKISource",
    "NASAEONETSource",
    "NASANeoWsSource",
    "NOAACOOPSSource",
    "NOAANWPSSource",
    "NOAASWPCSource",
    "NWSWeatherAlertsSource",
    "RateLimitConfig",
    "SentryImpactRisk",
    "SimulatedDataError",
    "SolarSystemOpenDataSource",
    "SourceUnreachableError",
    "SuperMAGSource",
    "USGSEarthquakeSource",
    "USGSGeomagnetismSource",
    "USGSVolcanoSource",
    "close_approaches_from_neows_datapoints",
    "fetch_live_datapoints",
    "fireball_events_from_datapoints",
    "sentry_risks_from_datapoints",
]
