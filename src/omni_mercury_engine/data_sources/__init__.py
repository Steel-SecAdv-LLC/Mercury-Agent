"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

STEM Alert & Anomaly Detection API Integration

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
    "DataPoint",
    "DataSourceBase",
    "DataSourceConfig",
    "DataSourceError",
    "DataSourceManager",
    "DataSourceType",
    # Earth Science
    "EPAAirNowSource",
    "FetchResult",
    # Consciousness Research
    "GCPDataSource",
    "GCPDotSource",
    "HeartMathGCMSSource",
    "INTERMAGNETSource",
    # Space Weather
    "NASADONKISource",
    "NASAEONETSource",
    "NASANeoWsSource",
    "NOAACOOPSSource",
    "NOAANWPSSource",
    "NOAASWPCSource",
    "NWSWeatherAlertsSource",
    "RateLimitConfig",
    "SolarSystemOpenDataSource",
    "SuperMAGSource",
    "USGSEarthquakeSource",
    "USGSGeomagnetismSource",
    "USGSVolcanoSource",
]
