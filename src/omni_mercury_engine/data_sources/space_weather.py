"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Space Weather & Solar Physics Data Sources

Production-grade integrations for:
- NASA DONKI (Space Weather Database Of Notifications, Knowledge, Information)
- NASA NeoWs (Near Earth Object Web Service)
- NOAA SWPC (Space Weather Prediction Center)
- NASA EONET (Earth Observatory Natural Event Tracker)
- Solar System OpenData (Le Système Solaire)

API Documentation:
- DONKI: https://api.nasa.gov/DONKI/
- NeoWs: https://api.nasa.gov/neo/rest/v1/
- SWPC: https://services.swpc.noaa.gov/
- EONET: https://eonet.gsfc.nasa.gov/api/v3/
- Solar System: https://api.le-systeme-solaire.net/rest/

Rate Limits:
- NASA APIs: 1000 requests/hour with API key, 30/hour with DEMO_KEY
- NOAA/SWPC: No stated limits, use respectful polling (≥60s intervals)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from omni_mercury_engine.data_sources.base import (
    AlertLevel,
    CacheConfig,
    DataPoint,
    DataSourceBase,
    DataSourceConfig,
    DataSourceType,
    RateLimitConfig,
)


logger = logging.getLogger(__name__)


# =============================================================================
# NASA DONKI - Space Weather Database
# =============================================================================


class DONKIEventType(Enum):
    """DONKI event types with their API endpoints."""

    SOLAR_FLARE = "FLR"
    CME = "CME"
    CME_ANALYSIS = "CMEAnalysis"
    GEOMAGNETIC_STORM = "GST"
    INTERPLANETARY_SHOCK = "IPS"
    SOLAR_ENERGETIC_PARTICLE = "SEP"
    MAGNETOPAUSE_CROSSING = "MPC"
    RADIATION_BELT_ENHANCEMENT = "RBE"
    HIGH_SPEED_STREAM = "HSS"
    NOTIFICATIONS = "notifications"


@dataclass
class DONKIConfig:
    """Configuration specific to NASA DONKI source."""

    api_key: str = "DEMO_KEY"
    event_types: list[DONKIEventType] | None = None  # None = all types
    days_back: int = 7


class NASADONKISource(DataSourceBase):
    """NASA DONKI (Space Weather Database) data source.

    Provides access to space weather events including:
    - Solar flares (FLR): M and X class flares
    - Coronal mass ejections (CME): Earth-directed CMEs
    - Geomagnetic storms (GST): G1-G5 scale events
    - Interplanetary shocks (IPS): Shock wave arrivals
    - Solar energetic particles (SEP): S1-S5 scale events
    - Magnetopause crossings (MPC): Magnetic boundary events
    - Radiation belt enhancements (RBE): Van Allen belt changes
    - High speed streams (HSS): Solar wind speed increases

    Example:
        >>> source = NASADONKISource(api_key="your_api_key")
        >>> result = await source.fetch(
        ...     start_time=datetime.now() - timedelta(days=7),
        ...     event_types=[DONKIEventType.SOLAR_FLARE, DONKIEventType.CME]
        ... )
    """

    DEFAULT_BASE_URL = "https://api.nasa.gov/DONKI/"

    def __init__(
        self,
        api_key: str = "DEMO_KEY",
        event_types: list[DONKIEventType] | None = None,
        days_back: int = 7,
        config: DataSourceConfig | None = None,
    ) -> None:
        """Initialize NASA DONKI data source.

        Args:
            api_key: NASA API key (get free key at api.nasa.gov)
            event_types: Event types to fetch (None = all)
            days_back: Default number of days to look back
            config: Optional base configuration
        """
        base_config = config or DataSourceConfig()
        base_config.api_key = api_key
        base_config.rate_limit = RateLimitConfig(
            requests_per_hour=1000 if api_key != "DEMO_KEY" else 30,
            min_interval_seconds=1.0,
        )
        base_config.cache = CacheConfig(ttl_seconds=300)  # 5 min cache

        super().__init__(base_config)

        self._api_key = api_key
        self._event_types = event_types or list(DONKIEventType)
        self._days_back = days_back

    @property
    def source_id(self) -> str:
        return "nasa_donki"

    @property
    def default_source_types(self) -> list[DataSourceType]:
        return [
            DataSourceType.SOLAR_FLARE,
            DataSourceType.CME,
            DataSourceType.GEOMAGNETIC_STORM,
            DataSourceType.SOLAR_ENERGETIC_PARTICLE,
        ]

    def _event_type_to_source_type(self, event_type: DONKIEventType) -> DataSourceType:
        """Map DONKI event type to DataSourceType."""
        mapping = {
            DONKIEventType.SOLAR_FLARE: DataSourceType.SOLAR_FLARE,
            DONKIEventType.CME: DataSourceType.CME,
            DONKIEventType.CME_ANALYSIS: DataSourceType.CME,
            DONKIEventType.GEOMAGNETIC_STORM: DataSourceType.GEOMAGNETIC_STORM,
            DONKIEventType.INTERPLANETARY_SHOCK: DataSourceType.SOLAR_WIND,
            DONKIEventType.SOLAR_ENERGETIC_PARTICLE: DataSourceType.SOLAR_ENERGETIC_PARTICLE,
            DONKIEventType.MAGNETOPAUSE_CROSSING: DataSourceType.MAGNETOMETER,
            DONKIEventType.RADIATION_BELT_ENHANCEMENT: DataSourceType.SOLAR_ENERGETIC_PARTICLE,
            DONKIEventType.HIGH_SPEED_STREAM: DataSourceType.SOLAR_WIND,
        }
        return mapping.get(event_type, DataSourceType.CUSTOM)

    def _parse_flare_class(self, class_type: str | None) -> AlertLevel:
        """Parse solar flare class to alert level."""
        if not class_type:
            return AlertLevel.NONE

        # Extract magnitude (e.g., "M1.5" -> "M", "X2.1" -> "X")
        class_letter = class_type[0].upper() if class_type else ""

        if class_letter == "X":
            # X-class: Major flare
            try:
                magnitude = float(class_type[1:])
                if magnitude >= 10:
                    return AlertLevel.EXTREME
                elif magnitude >= 5:
                    return AlertLevel.SEVERE
                else:
                    return AlertLevel.STRONG
            except (ValueError, IndexError):
                return AlertLevel.STRONG
        elif class_letter == "M":
            return AlertLevel.MODERATE
        elif class_letter == "C":
            return AlertLevel.MINOR
        return AlertLevel.NONE

    def _parse_kp_index(self, kp_index: str | float | None) -> AlertLevel:
        """Parse Kp index to alert level (G-scale)."""
        if kp_index is None:
            return AlertLevel.NONE

        try:
            kp = float(str(kp_index).replace("+", ".33").replace("-", ""))
            if kp >= 9:
                return AlertLevel.EXTREME  # G5
            elif kp >= 8:
                return AlertLevel.SEVERE  # G4
            elif kp >= 7:
                return AlertLevel.STRONG  # G3
            elif kp >= 6:
                return AlertLevel.MODERATE  # G2
            elif kp >= 5:
                return AlertLevel.MINOR  # G1
        except ValueError:
            pass
        return AlertLevel.NONE

    async def _fetch_event_type(
        self,
        event_type: DONKIEventType,
        start_time: datetime,
        end_time: datetime,
    ) -> list[DataPoint]:
        """Fetch a specific event type from DONKI."""
        params = {
            "startDate": start_time.strftime("%Y-%m-%d"),
            "endDate": end_time.strftime("%Y-%m-%d"),
            "api_key": self._api_key,
        }

        try:
            response = await self._http_get(event_type.value, params=params)
            events = response.json()

            if not isinstance(events, list):
                return []

            data_points: list[DataPoint] = []
            source_type = self._event_type_to_source_type(event_type)

            for event in events:
                data_point = self._parse_event(event, event_type, source_type)
                if data_point:
                    data_points.append(data_point)

            return data_points

        except Exception as e:
            logger.warning(f"DONKI {event_type.value} fetch failed: {e}")
            return []

    def _parse_event(
        self,
        event: dict[str, Any],
        event_type: DONKIEventType,
        source_type: DataSourceType,
    ) -> DataPoint | None:
        """Parse a DONKI event to DataPoint."""
        try:
            # Common fields
            event_id = event.get("activityID") or event.get("messageID") or ""

            # Parse timestamp - DONKI uses various date field names
            time_str = (
                event.get("beginTime")
                or event.get("time21_5")
                or event.get("eventTime")
                or event.get("startTime")
            )

            if not time_str:
                return None

            # Handle DONKI timestamp format (ISO with or without Z)
            timestamp = self._parse_donki_timestamp(time_str)

            # Determine alert level based on event type
            alert_level = AlertLevel.NONE
            confidence = 0.8

            if event_type == DONKIEventType.SOLAR_FLARE:
                alert_level = self._parse_flare_class(event.get("classType"))
                confidence = 0.95  # Well-observed events

            elif event_type == DONKIEventType.GEOMAGNETIC_STORM:
                kp_index = event.get("kpIndex") or event.get("allKpIndex", [{}])[0].get("kpIndex")
                alert_level = self._parse_kp_index(kp_index)

            elif event_type == DONKIEventType.CME:
                # CME speed indicates potential impact
                speed = event.get("speed")
                if speed:
                    if speed > 2000:
                        alert_level = AlertLevel.EXTREME
                    elif speed > 1500:
                        alert_level = AlertLevel.SEVERE
                    elif speed > 1000:
                        alert_level = AlertLevel.STRONG
                    elif speed > 500:
                        alert_level = AlertLevel.MODERATE

            elif event_type == DONKIEventType.SOLAR_ENERGETIC_PARTICLE:
                # S-scale from DONKI
                s_scale = event.get("sScale")
                if s_scale:
                    try:
                        alert_level = AlertLevel.from_noaa_s_scale(int(s_scale[1:]))
                    except (ValueError, IndexError):
                        pass

            # Location (for CMEs and some other events)
            location = None
            if "latitude" in event and "longitude" in event:
                location = (
                    float(event.get("latitude", 0)),
                    float(event.get("longitude", 0)),
                    0.0,  # Solar events don't have altitude in km
                )
            elif "sourceLocation" in event:
                # Parse solar coordinates like "N15W30"
                loc_str = event["sourceLocation"]
                location = self._parse_solar_location(loc_str)

            return DataPoint(
                source_id=self.source_id,
                source_type=source_type,
                event_id=event_id,
                timestamp=timestamp,
                data=event,
                location=location,
                alert_level=alert_level,
                confidence=confidence,
                metadata={
                    "event_type": event_type.value,
                    "api_version": "DONKI",
                },
            )

        except Exception as e:
            logger.debug(f"Failed to parse DONKI event: {e}")
            return None

    def _parse_donki_timestamp(self, time_str: str) -> datetime:
        """Parse DONKI timestamp formats."""
        # Try ISO format first
        try:
            if time_str.endswith("Z"):
                return datetime.fromisoformat(time_str.replace("Z", "+00:00"))
            elif "+" in time_str or time_str.count("-") > 2:
                return datetime.fromisoformat(time_str)
            else:
                # Assume UTC
                return datetime.fromisoformat(time_str).replace(tzinfo=UTC)
        except ValueError:
            pass

        # Try DONKI-specific format
        for fmt in ["%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
            try:
                return datetime.strptime(time_str, fmt).replace(tzinfo=UTC)
            except ValueError:
                continue

        raise ValueError(f"Cannot parse timestamp: {time_str}")

    def _parse_solar_location(self, loc_str: str) -> tuple[float, float, float] | None:
        """Parse solar location string like 'N15W30' to coordinates."""
        if not loc_str:
            return None

        match = re.match(r"([NS])(\d+)([EW])(\d+)", loc_str.upper())
        if match:
            ns, lat, ew, lon = match.groups()
            latitude = float(lat) * (1 if ns == "N" else -1)
            longitude = float(lon) * (1 if ew == "E" else -1)
            return (latitude, longitude, 0.0)

        return None

    async def _fetch_impl(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[DONKIEventType] | None = None,
        **kwargs: Any,
    ) -> list[DataPoint]:
        """Fetch space weather events from NASA DONKI."""
        end_time = end_time or datetime.now(UTC)
        start_time = start_time or (end_time - timedelta(days=self._days_back))

        types_to_fetch = event_types or self._event_types

        # Don't include NOTIFICATIONS by default as it's a meta-type
        if DONKIEventType.NOTIFICATIONS in types_to_fetch:
            types_to_fetch = [t for t in types_to_fetch if t != DONKIEventType.NOTIFICATIONS]

        all_data_points: list[DataPoint] = []

        for event_type in types_to_fetch:
            data_points = await self._fetch_event_type(event_type, start_time, end_time)
            all_data_points.extend(data_points)

        logger.info(
            f"DONKI: Fetched {len(all_data_points)} events from {len(types_to_fetch)} types"
        )
        return all_data_points


# =============================================================================
# NASA NeoWs - Near Earth Objects
# =============================================================================


class NASANeoWsSource(DataSourceBase):
    """NASA NeoWs (Near Earth Object Web Service) data source.

    Provides access to near-Earth asteroid and comet data including:
    - Orbital data and trajectory information
    - Hazard assessments (potentially hazardous asteroids)
    - Close approach data and dates
    - Physical characteristics (diameter, magnitude)

    Example:
        >>> source = NASANeoWsSource(api_key="your_api_key")
        >>> result = await source.fetch(
        ...     start_time=datetime.now(),
        ...     end_time=datetime.now() + timedelta(days=7)
        ... )
    """

    DEFAULT_BASE_URL = "https://api.nasa.gov/neo/rest/v1/"

    def __init__(
        self,
        api_key: str = "DEMO_KEY",
        days_forward: int = 7,
        config: DataSourceConfig | None = None,
    ) -> None:
        """Initialize NASA NeoWs data source.

        Args:
            api_key: NASA API key
            days_forward: Default number of days to look forward
            config: Optional base configuration
        """
        base_config = config or DataSourceConfig()
        base_config.api_key = api_key
        base_config.rate_limit = RateLimitConfig(
            requests_per_hour=1000 if api_key != "DEMO_KEY" else 30,
            min_interval_seconds=1.0,
        )
        base_config.cache = CacheConfig(ttl_seconds=3600)  # 1 hour cache

        super().__init__(base_config)

        self._api_key = api_key
        self._days_forward = days_forward

    @property
    def source_id(self) -> str:
        return "nasa_neows"

    @property
    def default_source_types(self) -> list[DataSourceType]:
        return [DataSourceType.NEAR_EARTH_OBJECT]

    def _calculate_hazard_level(self, neo: dict[str, Any]) -> AlertLevel:
        """Calculate hazard level from NEO data."""
        is_hazardous = neo.get("is_potentially_hazardous_asteroid", False)

        if not is_hazardous:
            return AlertLevel.NONE

        # Check close approach distance
        close_approaches = neo.get("close_approach_data", [])
        if close_approaches:
            # Get minimum miss distance
            min_distance_km = float("inf")
            for approach in close_approaches:
                miss_distance = approach.get("miss_distance", {})
                distance_km = float(miss_distance.get("kilometers", float("inf")))
                min_distance_km = min(min_distance_km, distance_km)

            # Lunar distance is ~384,400 km
            lunar_distance = 384400

            if min_distance_km < lunar_distance * 0.5:
                return AlertLevel.EXTREME
            elif min_distance_km < lunar_distance:
                return AlertLevel.SEVERE
            elif min_distance_km < lunar_distance * 5:
                return AlertLevel.STRONG
            elif min_distance_km < lunar_distance * 10:
                return AlertLevel.MODERATE

        return AlertLevel.MINOR if is_hazardous else AlertLevel.NONE

    async def _fetch_impl(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        **kwargs: Any,
    ) -> list[DataPoint]:
        """Fetch NEO data from NASA NeoWs."""
        start_time = start_time or datetime.now(UTC)
        end_time = end_time or (start_time + timedelta(days=self._days_forward))

        # NeoWs has a 7-day limit per request
        max_days = 7
        all_data_points: list[DataPoint] = []

        current_start = start_time
        while current_start < end_time:
            current_end = min(current_start + timedelta(days=max_days), end_time)

            params = {
                "start_date": current_start.strftime("%Y-%m-%d"),
                "end_date": current_end.strftime("%Y-%m-%d"),
                "api_key": self._api_key,
            }

            try:
                response = await self._http_get("feed", params=params)
                data = response.json()

                neo_objects = data.get("near_earth_objects", {})
                for date_str, neos in neo_objects.items():
                    for neo in neos:
                        data_point = self._parse_neo(neo, date_str)
                        if data_point:
                            all_data_points.append(data_point)

            except Exception as e:
                logger.warning(f"NeoWs fetch failed for {current_start}: {e}")

            current_start = current_end

        logger.info(f"NeoWs: Fetched {len(all_data_points)} near-Earth objects")
        return all_data_points

    def _parse_neo(self, neo: dict[str, Any], date_str: str) -> DataPoint | None:
        """Parse NEO data to DataPoint."""
        try:
            neo_id = neo.get("id", "")
            name = neo.get("name", "Unknown")

            # Parse close approach date
            close_approaches = neo.get("close_approach_data", [])
            if close_approaches:
                approach = close_approaches[0]
                timestamp = datetime.strptime(
                    approach.get("close_approach_date_full", date_str + " 00:00"), "%Y-%b-%d %H:%M"
                ).replace(tzinfo=UTC)
            else:
                timestamp = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC)

            # Get diameter estimates
            diameter = neo.get("estimated_diameter", {})
            diameter_km = diameter.get("kilometers", {})
            min_diameter = diameter_km.get("estimated_diameter_min", 0)
            max_diameter = diameter_km.get("estimated_diameter_max", 0)
            avg_diameter = (min_diameter + max_diameter) / 2

            # Confidence based on diameter estimate precision
            if max_diameter > 0:
                diameter_precision = 1 - (max_diameter - min_diameter) / max_diameter
                confidence = 0.5 + 0.4 * diameter_precision
            else:
                confidence = 0.5

            return DataPoint(
                source_id=self.source_id,
                source_type=DataSourceType.NEAR_EARTH_OBJECT,
                event_id=neo_id,
                timestamp=timestamp,
                data={
                    "name": name,
                    "neo_reference_id": neo.get("neo_reference_id"),
                    "nasa_jpl_url": neo.get("nasa_jpl_url"),
                    "absolute_magnitude_h": neo.get("absolute_magnitude_h"),
                    "estimated_diameter_km": {
                        "min": min_diameter,
                        "max": max_diameter,
                        "avg": avg_diameter,
                    },
                    "is_potentially_hazardous": neo.get("is_potentially_hazardous_asteroid", False),
                    "close_approach_data": close_approaches,
                    "is_sentry_object": neo.get("is_sentry_object", False),
                },
                alert_level=self._calculate_hazard_level(neo),
                confidence=confidence,
                metadata={"api_version": "NeoWs v1"},
            )

        except Exception as e:
            logger.debug(f"Failed to parse NEO: {e}")
            return None


# =============================================================================
# NOAA SWPC - Space Weather Prediction Center
# =============================================================================


class SWPCProduct(Enum):
    """NOAA SWPC data products."""

    SOLAR_WIND_PLASMA = "solar-wind/plasma-7-day.json"
    SOLAR_WIND_MAG = "solar-wind/mag-7-day.json"
    KP_INDEX = "noaa-planetary-k-index.json"
    ALERTS = "alerts.json"
    XRAY_FLUX = "primary/xrays-7-day.json"
    REALTIME_SOLAR_WIND = "rtsw/rtsw_mag_1m.json"
    GEOSPACE_PRED = "geospace/geospace-pred-7-day.json"
    SOLAR_CYCLE = "solar-cycle-observed.json"


class NOAASWPCSource(DataSourceBase):
    """NOAA Space Weather Prediction Center data source.

    Provides real-time and historical space weather data:
    - Solar wind plasma parameters (density, speed, temperature)
    - Interplanetary Magnetic Field (IMF) data
    - Planetary Kp index (geomagnetic activity)
    - Active space weather alerts
    - X-ray flux measurements
    - Real-time solar wind data

    No authentication required.

    Example:
        >>> source = NOAASWPCSource()
        >>> result = await source.fetch(products=[SWPCProduct.KP_INDEX, SWPCProduct.ALERTS])
    """

    DEFAULT_BASE_URL = "https://services.swpc.noaa.gov/"

    def __init__(
        self,
        products: list[SWPCProduct] | None = None,
        config: DataSourceConfig | None = None,
    ) -> None:
        """Initialize NOAA SWPC data source.

        Args:
            products: SWPC products to fetch (None = key products)
            config: Optional base configuration
        """
        base_config = config or DataSourceConfig()
        base_config.rate_limit = RateLimitConfig(
            requests_per_hour=0,  # No stated limit
            min_interval_seconds=60.0,  # Respectful polling
        )
        base_config.cache = CacheConfig(ttl_seconds=60)  # 1 min cache for real-time

        super().__init__(base_config)

        self._products = products or [
            SWPCProduct.KP_INDEX,
            SWPCProduct.ALERTS,
            SWPCProduct.SOLAR_WIND_PLASMA,
        ]

    @property
    def source_id(self) -> str:
        return "noaa_swpc"

    @property
    def default_source_types(self) -> list[DataSourceType]:
        return [
            DataSourceType.SOLAR_WIND,
            DataSourceType.GEOMAGNETIC_STORM,
            DataSourceType.SOLAR_FLARE,
        ]

    def _product_to_source_type(self, product: SWPCProduct) -> DataSourceType:
        """Map SWPC product to DataSourceType."""
        mapping = {
            SWPCProduct.SOLAR_WIND_PLASMA: DataSourceType.SOLAR_WIND,
            SWPCProduct.SOLAR_WIND_MAG: DataSourceType.SOLAR_WIND,
            SWPCProduct.KP_INDEX: DataSourceType.GEOMAGNETIC_STORM,
            SWPCProduct.ALERTS: DataSourceType.WEATHER_ALERT,
            SWPCProduct.XRAY_FLUX: DataSourceType.SOLAR_FLARE,
            SWPCProduct.REALTIME_SOLAR_WIND: DataSourceType.SOLAR_WIND,
        }
        return mapping.get(product, DataSourceType.CUSTOM)

    def _parse_kp_alert_level(self, kp: float) -> AlertLevel:
        """Parse Kp index to G-scale alert level."""
        if kp >= 9:
            return AlertLevel.EXTREME  # G5
        elif kp >= 8:
            return AlertLevel.SEVERE  # G4
        elif kp >= 7:
            return AlertLevel.STRONG  # G3
        elif kp >= 6:
            return AlertLevel.MODERATE  # G2
        elif kp >= 5:
            return AlertLevel.MINOR  # G1
        return AlertLevel.NONE

    async def _fetch_product(self, product: SWPCProduct) -> list[DataPoint]:
        """Fetch a specific SWPC product."""
        endpoint = f"products/{product.value}"

        if product == SWPCProduct.XRAY_FLUX:
            endpoint = f"json/goes/{product.value}"
        elif product == SWPCProduct.REALTIME_SOLAR_WIND:
            endpoint = f"products/{product.value}"

        try:
            response = await self._http_get(endpoint)
            data = response.json()
            return self._parse_product_data(product, data)

        except Exception as e:
            logger.warning(f"SWPC {product.value} fetch failed: {e}")
            return []

    def _parse_product_data(
        self,
        product: SWPCProduct,
        data: Any,
    ) -> list[DataPoint]:
        """Parse SWPC product data to DataPoints."""
        data_points: list[DataPoint] = []
        source_type = self._product_to_source_type(product)

        if product == SWPCProduct.KP_INDEX:
            data_points = self._parse_kp_index(data, source_type)
        elif product == SWPCProduct.ALERTS:
            data_points = self._parse_alerts(data, source_type)
        elif product in (SWPCProduct.SOLAR_WIND_PLASMA, SWPCProduct.SOLAR_WIND_MAG):
            data_points = self._parse_solar_wind(data, source_type, product)
        elif product == SWPCProduct.XRAY_FLUX:
            data_points = self._parse_xray_flux(data, source_type)

        return data_points

    def _parse_kp_index(
        self,
        data: list[list[Any]],
        source_type: DataSourceType,
    ) -> list[DataPoint]:
        """Parse Kp index data."""
        data_points: list[DataPoint] = []

        # Skip header row
        for row in data[1:]:
            try:
                # Format: [time_tag, Kp, estimated_Kp, a_running, station_count]
                if len(row) < 2:
                    continue

                time_str = row[0]
                kp_value = float(row[1]) if row[1] else 0.0

                timestamp = datetime.fromisoformat(time_str.replace("Z", "+00:00"))

                data_points.append(
                    DataPoint(
                        source_id=self.source_id,
                        source_type=source_type,
                        event_id=f"kp_{timestamp.isoformat()}",
                        timestamp=timestamp,
                        data={
                            "kp_index": kp_value,
                            "estimated_kp": row[2] if len(row) > 2 else None,
                            "a_running": row[3] if len(row) > 3 else None,
                            "station_count": row[4] if len(row) > 4 else None,
                        },
                        alert_level=self._parse_kp_alert_level(kp_value),
                        confidence=0.95,
                        metadata={"product": "kp_index"},
                    )
                )

            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to parse Kp row: {e}")
                continue

        return data_points

    def _parse_alerts(
        self,
        data: list[dict[str, Any]],
        source_type: DataSourceType,
    ) -> list[DataPoint]:
        """Parse SWPC alerts."""
        data_points: list[DataPoint] = []

        for alert in data:
            try:
                issue_time = alert.get("issue_datetime", "")
                if not issue_time:
                    continue

                timestamp = datetime.fromisoformat(issue_time.replace("Z", "+00:00"))
                message = alert.get("message", "")
                product_id = alert.get("product_id", "")

                # Determine alert level from message content
                alert_level = AlertLevel.NONE
                if "Warning" in message or "G4" in message or "G5" in message:
                    alert_level = AlertLevel.SEVERE
                elif "Watch" in message or "G3" in message:
                    alert_level = AlertLevel.STRONG
                elif "Alert" in message or "G2" in message:
                    alert_level = AlertLevel.MODERATE
                elif "G1" in message:
                    alert_level = AlertLevel.MINOR

                data_points.append(
                    DataPoint(
                        source_id=self.source_id,
                        source_type=DataSourceType.WEATHER_ALERT,
                        event_id=f"swpc_alert_{product_id}_{timestamp.isoformat()}",
                        timestamp=timestamp,
                        data={
                            "product_id": product_id,
                            "message": message[:1000],  # Truncate long messages
                            "serial_number": alert.get("serial_number"),
                        },
                        alert_level=alert_level,
                        confidence=0.99,
                        metadata={"product": "alerts"},
                    )
                )

            except (ValueError, KeyError) as e:
                logger.debug(f"Failed to parse SWPC alert: {e}")
                continue

        return data_points

    def _parse_solar_wind(
        self,
        data: list[list[Any]],
        source_type: DataSourceType,
        product: SWPCProduct,
    ) -> list[DataPoint]:
        """Parse solar wind data (plasma or magnetic)."""
        data_points: list[DataPoint] = []

        # Get last 24 entries (every 5 min = 288/day, so ~2 hours of data)
        recent_data = data[1:25] if len(data) > 25 else data[1:]

        for row in recent_data:
            try:
                if len(row) < 2:
                    continue

                time_str = row[0]
                timestamp = datetime.fromisoformat(time_str.replace("Z", "+00:00"))

                if product == SWPCProduct.SOLAR_WIND_PLASMA:
                    # Format: [time, density, speed, temperature]
                    point_data = {
                        "density": float(row[1]) if row[1] else None,
                        "speed": float(row[2]) if len(row) > 2 and row[2] else None,
                        "temperature": float(row[3]) if len(row) > 3 and row[3] else None,
                    }
                else:  # SOLAR_WIND_MAG
                    # Format: [time, bx, by, bz, bt, lat, lon]
                    point_data = {
                        "bx": float(row[1]) if row[1] else None,
                        "by": float(row[2]) if len(row) > 2 and row[2] else None,
                        "bz": float(row[3]) if len(row) > 3 and row[3] else None,
                        "bt": float(row[4]) if len(row) > 4 and row[4] else None,
                    }

                # Calculate alert level based on speed or Bz
                alert_level = AlertLevel.NONE
                speed = point_data.get("speed")
                bz = point_data.get("bz")

                if speed and speed > 800:
                    alert_level = AlertLevel.STRONG
                elif (speed and speed > 600) or (bz and bz < -10):
                    alert_level = AlertLevel.MODERATE

                data_points.append(
                    DataPoint(
                        source_id=self.source_id,
                        source_type=source_type,
                        event_id=f"sw_{product.value}_{timestamp.isoformat()}",
                        timestamp=timestamp,
                        data=point_data,
                        alert_level=alert_level,
                        confidence=0.9,
                        metadata={"product": product.value},
                    )
                )

            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to parse solar wind row: {e}")
                continue

        return data_points

    def _parse_xray_flux(
        self,
        data: list[list[Any]],
        source_type: DataSourceType,
    ) -> list[DataPoint]:
        """Parse X-ray flux data."""
        data_points: list[DataPoint] = []

        # Get recent data
        recent_data = data[1:25] if len(data) > 25 else data[1:]

        for row in recent_data:
            try:
                if len(row) < 3:
                    continue

                time_str = row[0]
                timestamp = datetime.fromisoformat(time_str.replace("Z", "+00:00"))

                short_flux = float(row[1]) if row[1] else 0
                long_flux = float(row[2]) if row[2] else 0

                # Determine flare class from flux
                # M-class: 1e-5 to 1e-4 W/m^2
                # X-class: > 1e-4 W/m^2
                alert_level = AlertLevel.NONE
                if long_flux > 1e-4:
                    alert_level = AlertLevel.STRONG  # X-class
                elif long_flux > 1e-5:
                    alert_level = AlertLevel.MODERATE  # M-class
                elif long_flux > 1e-6:
                    alert_level = AlertLevel.MINOR  # C-class

                data_points.append(
                    DataPoint(
                        source_id=self.source_id,
                        source_type=source_type,
                        event_id=f"xray_{timestamp.isoformat()}",
                        timestamp=timestamp,
                        data={
                            "short_flux": short_flux,
                            "long_flux": long_flux,
                        },
                        alert_level=alert_level,
                        confidence=0.95,
                        metadata={"product": "xray_flux"},
                    )
                )

            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to parse X-ray row: {e}")
                continue

        return data_points

    async def _fetch_impl(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        products: list[SWPCProduct] | None = None,
        **kwargs: Any,
    ) -> list[DataPoint]:
        """Fetch space weather data from NOAA SWPC."""
        products_to_fetch = products or self._products
        all_data_points: list[DataPoint] = []

        for product in products_to_fetch:
            data_points = await self._fetch_product(product)
            all_data_points.extend(data_points)

        logger.info(
            f"SWPC: Fetched {len(all_data_points)} data points from {len(products_to_fetch)} products"
        )
        return all_data_points


# =============================================================================
# NASA EONET - Earth Observatory Natural Events
# =============================================================================


class EONETCategory(Enum):
    """EONET event categories."""

    DROUGHT = "drought"
    DUST_HAZE = "dustHaze"
    EARTHQUAKES = "earthquakes"
    FLOODS = "floods"
    LANDSLIDES = "landslides"
    MANMADE = "manmade"
    SEA_LAKE_ICE = "seaLakeIce"
    SEVERE_STORMS = "severeStorms"
    SNOW = "snow"
    TEMPERATURE_EXTREMES = "tempExtremes"
    VOLCANOES = "volcanoes"
    WATER_COLOR = "waterColor"
    WILDFIRES = "wildfires"


class NASAEONETSource(DataSourceBase):
    """NASA EONET (Earth Observatory Natural Event Tracker) data source.

    Provides access to natural events including:
    - Wildfires
    - Severe storms
    - Volcanoes
    - Sea/lake ice
    - Earthquakes
    - Floods
    - Landslides

    No authentication required.

    Example:
        >>> source = NASAEONETSource()
        >>> result = await source.fetch(categories=[EONETCategory.WILDFIRES, EONETCategory.VOLCANOES])
    """

    DEFAULT_BASE_URL = "https://eonet.gsfc.nasa.gov/api/v3/"

    def __init__(
        self,
        categories: list[EONETCategory] | None = None,
        days_back: int = 30,
        config: DataSourceConfig | None = None,
    ) -> None:
        """Initialize NASA EONET data source.

        Args:
            categories: Event categories to fetch (None = all)
            days_back: Number of days to look back
            config: Optional base configuration
        """
        base_config = config or DataSourceConfig()
        base_config.rate_limit = RateLimitConfig(
            requests_per_hour=0,  # No stated limit
            min_interval_seconds=5.0,
        )
        base_config.cache = CacheConfig(ttl_seconds=600)  # 10 min cache

        super().__init__(base_config)

        self._categories = categories
        self._days_back = days_back

    @property
    def source_id(self) -> str:
        return "nasa_eonet"

    @property
    def default_source_types(self) -> list[DataSourceType]:
        return [DataSourceType.NATURAL_EVENT]

    def _category_to_alert_level(self, category: str, geometry: list[Any]) -> AlertLevel:
        """Determine alert level based on category and event characteristics."""
        # Base levels by category
        category_levels = {
            "wildfires": AlertLevel.MODERATE,
            "volcanoes": AlertLevel.STRONG,
            "severeStorms": AlertLevel.MODERATE,
            "earthquakes": AlertLevel.MODERATE,
            "floods": AlertLevel.MODERATE,
            "landslides": AlertLevel.MODERATE,
        }

        base_level = category_levels.get(category, AlertLevel.MINOR)

        # Increase level for ongoing events with many geometry points (larger/longer)
        if len(geometry) > 10:
            if base_level.value < AlertLevel.SEVERE.value:
                return AlertLevel(base_level.value + 1)

        return base_level

    async def _fetch_impl(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        categories: list[EONETCategory] | None = None,
        **kwargs: Any,
    ) -> list[DataPoint]:
        """Fetch natural events from NASA EONET."""
        params: dict[str, Any] = {
            "status": "all",
            "limit": 100,
        }

        # Add category filter
        cats = categories or self._categories
        if cats:
            params["category"] = ",".join(c.value for c in cats)

        # Add date filter
        if start_time:
            params["start"] = start_time.strftime("%Y-%m-%d")
        if end_time:
            params["end"] = end_time.strftime("%Y-%m-%d")

        response = await self._http_get("events", params=params)
        data = response.json()

        data_points: list[DataPoint] = []

        for event in data.get("events", []):
            try:
                event_id = event.get("id", "")
                title = event.get("title", "")
                categories_list = event.get("categories", [])
                geometry = event.get("geometry", [])
                sources = event.get("sources", [])

                if not geometry:
                    continue

                # Get the most recent geometry point
                latest_geo = geometry[-1]
                geo_date = latest_geo.get("date", "")
                coords = latest_geo.get("coordinates", [0, 0])

                timestamp = datetime.fromisoformat(geo_date.replace("Z", "+00:00"))

                category_name = (
                    categories_list[0].get("id", "unknown") if categories_list else "unknown"
                )

                location = (
                    float(coords[1]) if len(coords) > 1 else 0.0,  # lat
                    float(coords[0]) if len(coords) > 0 else 0.0,  # lon
                    0.0,
                )

                data_points.append(
                    DataPoint(
                        source_id=self.source_id,
                        source_type=DataSourceType.NATURAL_EVENT,
                        event_id=event_id,
                        timestamp=timestamp,
                        data={
                            "title": title,
                            "category": category_name,
                            "geometry_count": len(geometry),
                            "sources": [s.get("url") for s in sources],
                            "closed": event.get("closed"),
                        },
                        location=location,
                        alert_level=self._category_to_alert_level(category_name, geometry),
                        confidence=0.85,
                        metadata={"api_version": "EONET v3"},
                    )
                )

            except (ValueError, KeyError, IndexError) as e:
                logger.debug(f"Failed to parse EONET event: {e}")
                continue

        logger.info(f"EONET: Fetched {len(data_points)} natural events")
        return data_points


# =============================================================================
# Solar System OpenData
# =============================================================================


class SolarSystemOpenDataSource(DataSourceBase):
    """Solar System OpenData (Le Système Solaire) data source.

    Provides access to celestial body data:
    - Planets and moons
    - Asteroids and comets
    - Physical characteristics (mass, gravity, dimensions)
    - Orbital parameters
    - Discovery information

    No authentication required.

    Example:
        >>> source = SolarSystemOpenDataSource()
        >>> result = await source.fetch(body_type="planet")
    """

    DEFAULT_BASE_URL = "https://api.le-systeme-solaire.net/rest/"

    def __init__(
        self,
        config: DataSourceConfig | None = None,
    ) -> None:
        """Initialize Solar System OpenData source."""
        base_config = config or DataSourceConfig()
        base_config.rate_limit = RateLimitConfig(
            requests_per_hour=0,
            min_interval_seconds=1.0,
        )
        base_config.cache = CacheConfig(ttl_seconds=86400)  # 24 hour cache (static data)

        super().__init__(base_config)

    @property
    def source_id(self) -> str:
        return "solar_system_opendata"

    @property
    def default_source_types(self) -> list[DataSourceType]:
        return [DataSourceType.CELESTIAL_BODY]

    async def _fetch_impl(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        body_type: str | None = None,
        body_id: str | None = None,
        **kwargs: Any,
    ) -> list[DataPoint]:
        """Fetch celestial body data."""
        if body_id:
            # Fetch specific body
            response = await self._http_get(f"bodies/{body_id}")
            data = response.json()
            bodies = [data]
        else:
            # Fetch all bodies (or filtered by type)
            params: dict[str, Any] = {}
            if body_type:
                params["filter[]"] = f"bodyType,eq,{body_type}"

            response = await self._http_get("bodies", params=params)
            data = response.json()
            bodies = data.get("bodies", [])

        data_points: list[DataPoint] = []

        for body in bodies:
            try:
                body_name = body.get("englishName", body.get("name", "Unknown"))
                body_id_val = body.get("id", body_name.lower().replace(" ", "_"))

                # Use discovery date or current time
                discovery = body.get("discoveryDate")
                if discovery:
                    try:
                        timestamp = datetime.strptime(discovery, "%d/%m/%Y").replace(tzinfo=UTC)
                    except ValueError:
                        timestamp = datetime.now(UTC)
                else:
                    timestamp = datetime.now(UTC)

                data_points.append(
                    DataPoint(
                        source_id=self.source_id,
                        source_type=DataSourceType.CELESTIAL_BODY,
                        event_id=body_id_val,
                        timestamp=timestamp,
                        data={
                            "name": body_name,
                            "body_type": body.get("bodyType"),
                            "is_planet": body.get("isPlanet", False),
                            "mass": body.get("mass"),
                            "vol": body.get("vol"),
                            "density": body.get("density"),
                            "gravity": body.get("gravity"),
                            "escape": body.get("escape"),
                            "mean_radius": body.get("meanRadius"),
                            "equa_radius": body.get("equaRadius"),
                            "polar_radius": body.get("polarRadius"),
                            "flattening": body.get("flattening"),
                            "dimension": body.get("dimension"),
                            "sideral_orbit": body.get("sideralOrbit"),
                            "sideral_rotation": body.get("sideralRotation"),
                            "around_planet": body.get("aroundPlanet"),
                            "moons": body.get("moons"),
                            "discovered_by": body.get("discoveredBy"),
                            "discovery_date": discovery,
                            "axial_tilt": body.get("axialTilt"),
                            "avg_temp": body.get("avgTemp"),
                            "aphelion": body.get("aphelion"),
                            "perihelion": body.get("perihelion"),
                            "semi_major_axis": body.get("semimajorAxis"),
                            "eccentricity": body.get("eccentricity"),
                            "inclination": body.get("inclination"),
                        },
                        confidence=0.99,  # Well-established astronomical data
                        metadata={"api_version": "Solar System OpenData"},
                    )
                )

            except (ValueError, KeyError) as e:
                logger.debug(f"Failed to parse celestial body: {e}")
                continue

        logger.info(f"Solar System: Fetched {len(data_points)} celestial bodies")
        return data_points
