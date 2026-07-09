# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Earth Science & Hazards Data Sources.

Production-grade integrations for:
- USGS Earthquake Hazards
- USGS Volcano Hazards
- NOAA NWPS (National Water Prediction Service)
- NOAA CO-OPS (Center for Operational Oceanographic Products and Services)
- NWS Weather Alerts (National Weather Service)
- EPA AirNow

API Documentation:
- USGS Earthquake: https://earthquake.usgs.gov/fdsnws/event/1/
- USGS Volcano: https://volcanoes.usgs.gov/vhp/api/volcanoApi/
- NOAA NWPS: https://api.water.noaa.gov/nwps/v1/
- NOAA CO-OPS: https://api.tidesandcurrents.noaa.gov/api/prod/datagetter
- NWS: https://api.weather.gov/alerts
- EPA AirNow: https://www.airnowapi.org/aq/
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from omni_mercury_engine.data_sources.base import (
    AlertLevel,
    CacheConfig,
    DataPoint,
    DataSourceBase,
    DataSourceConfig,
    DataSourceError,
    DataSourceType,
    RateLimitConfig,
)

logger = logging.getLogger(__name__)

# =============================================================================
# USGS Earthquake Hazards
# =============================================================================


class USGSEarthquakeSource(DataSourceBase):
    """USGS Earthquake Hazards Program data source.

    Provides real-time earthquake data worldwide:
    - Event search with magnitude/location filters
    - Formats: GeoJSON, QuakeML, CSV, KML
    - Historical and real-time data

    No authentication required.

    Example:
        >>> source = USGSEarthquakeSource(min_magnitude=4.0)
        >>> result = await source.fetch(
        ...     start_time=datetime.now() - timedelta(days=7)
        ... )
    """

    DEFAULT_BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/"

    def __init__(
        self,
        min_magnitude: float = 2.5,
        max_results: int = 100,
        days_back: int = 7,
        config: DataSourceConfig | None = None,
    ) -> None:
        """Initialize USGS Earthquake data source.

        Args:
            min_magnitude: Minimum earthquake magnitude to fetch
            max_results: Maximum number of results
            days_back: Default number of days to look back
            config: Optional base configuration
        """
        base_config = config or DataSourceConfig()
        base_config.rate_limit = RateLimitConfig(
            requests_per_hour=0,  # No stated limit
            min_interval_seconds=5.0,
        )
        base_config.cache = CacheConfig(ttl_seconds=60)

        super().__init__(base_config)

        self._min_magnitude = min_magnitude
        self._max_results = max_results
        self._days_back = days_back

    @property
    def source_id(self) -> str:
        """Source id."""
        return "usgs_earthquake"

    @property
    def default_source_types(self) -> list[DataSourceType]:
        """Default source types."""
        return [DataSourceType.EARTHQUAKE]

    def _magnitude_to_alert_level(self, magnitude: float) -> AlertLevel:
        """Convert earthquake magnitude to alert level."""
        if magnitude >= 8.0:
            return AlertLevel.EXTREME  # Great earthquake
        elif magnitude >= 7.0:
            return AlertLevel.SEVERE  # Major earthquake
        elif magnitude >= 6.0:
            return AlertLevel.STRONG  # Strong earthquake
        elif magnitude >= 5.0:
            return AlertLevel.MODERATE  # Moderate earthquake
        elif magnitude >= 4.0:
            return AlertLevel.MINOR  # Light earthquake
        return AlertLevel.NONE

    async def _fetch_impl(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        min_magnitude: float | None = None,
        max_latitude: float | None = None,
        min_latitude: float | None = None,
        max_longitude: float | None = None,
        min_longitude: float | None = None,
        **kwargs: Any,
    ) -> list[DataPoint]:
        """Fetch earthquake data from USGS."""
        end_time = end_time or datetime.now(UTC)
        start_time = start_time or (end_time - timedelta(days=self._days_back))

        params: dict[str, Any] = {
            "format": "geojson",
            "starttime": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "endtime": end_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "minmagnitude": min_magnitude or self._min_magnitude,
            "limit": self._max_results,
            "orderby": "time",
        }

        # Add geographic bounds if specified
        if max_latitude is not None:
            params["maxlatitude"] = max_latitude
        if min_latitude is not None:
            params["minlatitude"] = min_latitude
        if max_longitude is not None:
            params["maxlongitude"] = max_longitude
        if min_longitude is not None:
            params["minlongitude"] = min_longitude

        response = await self._http_get("query", params=params)
        data = response.json()

        # Contract check: FDSN GeoJSON always carries a "features" array. A
        # payload without it is endpoint drift and must fail loud -- returning
        # an empty success here would read as "no earthquakes this week".
        if not isinstance(data, dict) or "features" not in data:
            raise DataSourceError(
                "USGS FDSN payload has no 'features' array "
                f"(got {type(data).__name__}); endpoint contract drift",
                source_id=self.source_id,
                retryable=False,
            )
        features = data["features"]

        data_points: list[DataPoint] = []
        parse_failures = 0

        for feature in features:
            try:
                props = feature.get("properties", {})
                coords = feature.get("geometry", {}).get("coordinates", [0, 0, 0])

                event_id = feature.get("id", "")
                magnitude = float(props.get("mag", 0) or 0)

                # Parse timestamp (milliseconds since epoch)
                time_ms = props.get("time", 0)
                timestamp = datetime.fromtimestamp(time_ms / 1000, tz=UTC)

                # Confidence based on magnitude type quality
                mag_type = props.get("magType", "")
                confidence = 0.9 if mag_type in ["mw", "mww", "mwb"] else 0.8

                location = (
                    float(coords[1]) if len(coords) > 1 else 0.0,  # latitude
                    float(coords[0]) if len(coords) > 0 else 0.0,  # longitude
                    float(coords[2]) if len(coords) > 2 else 0.0,  # depth in km
                )

                data_points.append(
                    DataPoint(
                        source_id=self.source_id,
                        source_type=DataSourceType.EARTHQUAKE,
                        event_id=event_id,
                        timestamp=timestamp,
                        data={
                            "magnitude": magnitude,
                            "magnitude_type": mag_type,
                            "depth_km": location[2],
                            "place": props.get("place", "Unknown"),
                            "tsunami": props.get("tsunami", 0) == 1,
                            "felt_reports": props.get("felt"),
                            "cdi": props.get("cdi"),  # Community internet intensity
                            "mmi": props.get("mmi"),  # Modified Mercalli Intensity
                            "alert": props.get("alert"),
                            "significance": props.get("sig"),
                            "status": props.get("status"),
                            "url": props.get("url"),
                        },
                        location=location,
                        alert_level=self._magnitude_to_alert_level(magnitude),
                        confidence=confidence,
                        metadata={"api_version": "FDSN 1.0"},
                    )
                )

            except (ValueError, KeyError, TypeError) as e:
                parse_failures += 1
                logger.warning(f"Failed to parse earthquake feature: {e}")
                continue

        if features and not data_points:
            # Every feature failed to parse: that is schema drift, not a
            # quiet catalog. Refuse to return a fabricated empty success.
            raise DataSourceError(
                f"USGS FDSN returned {len(features)} features but none parsed "
                f"({parse_failures} parse failures); schema drift",
                source_id=self.source_id,
                retryable=False,
            )

        logger.info(
            f"USGS Earthquake: Fetched {len(data_points)} earthquakes (M>={self._min_magnitude})"
        )
        return data_points


# =============================================================================
# USGS Volcano Hazards
# =============================================================================


class VolcanoAlertLevel(Enum):
    """USGS Volcano alert levels."""

    NORMAL = "normal"
    ADVISORY = "advisory"
    WATCH = "watch"
    WARNING = "warning"


class USGSVolcanoSource(DataSourceBase):
    """USGS Volcano Hazards Program data source (real HANS public API).

    Fetches the official alert state of every U.S. monitored volcano from the
    USGS Hazard Alert Notification System (HANS) public API:

    - ``getMonitoredVolcanoes`` — every monitored volcano with its current
      alert level (NORMAL / ADVISORY / WATCH / WARNING / UNASSIGNED) and
      aviation color code (GREEN / YELLOW / ORANGE / RED / UNASSIGNED)
    - ``getElevatedVolcanoes`` — only volcanoes currently above NORMAL

    API: https://volcanoes.usgs.gov/hans-public/api/volcano/

    The HANS list endpoints do not include coordinates, so ``DataPoint.location``
    is None; per-volcano coordinates are available from the ``getVolcano/<vnum>``
    detail endpoint if a consumer needs them.

    Example:
        >>> source = USGSVolcanoSource()
        >>> result = await source.fetch()                      # all monitored
        >>> result = await source.fetch(elevated_only=True)    # elevated only
    """

    DEFAULT_BASE_URL = "https://volcanoes.usgs.gov/hans-public/api/volcano/"

    MONITORED_ENDPOINT = "getMonitoredVolcanoes"
    ELEVATED_ENDPOINT = "getElevatedVolcanoes"

    def __init__(
        self,
        config: DataSourceConfig | None = None,
    ) -> None:
        """Initialize USGS Volcano data source."""
        base_config = config or DataSourceConfig()
        base_config.rate_limit = RateLimitConfig(
            requests_per_hour=60,
            min_interval_seconds=60.0,
        )
        base_config.cache = CacheConfig(ttl_seconds=3600)  # 1 hour cache

        super().__init__(base_config)

    @property
    def source_id(self) -> str:
        """Source id."""
        return "usgs_volcano"

    @property
    def default_source_types(self) -> list[DataSourceType]:
        """Default source types."""
        return [DataSourceType.VOLCANO]

    def _alert_level_to_level(self, alert: str) -> AlertLevel:
        """Convert USGS volcano alert to AlertLevel."""
        mapping = {
            "warning": AlertLevel.SEVERE,
            "watch": AlertLevel.STRONG,
            "advisory": AlertLevel.MODERATE,
            "normal": AlertLevel.NONE,
        }
        return mapping.get(alert.lower(), AlertLevel.NONE)

    def _parse_hans_entry(self, entry: dict[str, Any]) -> DataPoint | None:
        """Parse one HANS volcano entry into a DataPoint.

        Args:
            entry: One element of a HANS ``getMonitoredVolcanoes`` /
                ``getElevatedVolcanoes`` response.

        Returns:
            Parsed DataPoint, or None when the entry is malformed.
        """
        try:
            name = entry["volcano_name"]
            vnum = entry.get("vnum", "")
            alert = str(entry.get("alert_level", "") or "UNASSIGNED")
            color = str(entry.get("color_code", "") or "UNASSIGNED")

            sent_unix = entry.get("sent_unixtime")
            if sent_unix:
                timestamp = datetime.fromtimestamp(int(sent_unix), tz=UTC)
            else:
                timestamp = datetime.now(UTC)

            return DataPoint(
                source_id=self.source_id,
                source_type=DataSourceType.VOLCANO,
                event_id=f"usgs_volcano_{vnum or name}_{int(timestamp.timestamp())}",
                timestamp=timestamp,
                data={
                    "volcano_id": vnum,
                    "name": name,
                    "alert_level": alert.lower(),
                    "aviation_color_code": color.lower(),
                    "monitoring_status": "monitored",
                    "observatory": entry.get("obs_fullname"),
                    "observatory_abbr": entry.get("obs_abbr"),
                    "notice_type": entry.get("notice_type_cd"),
                    "notice_identifier": entry.get("notice_identifier"),
                    "notice_url": entry.get("notice_url"),
                },
                location=None,  # HANS list endpoints carry no coordinates
                alert_level=self._alert_level_to_level(alert),
                confidence=0.98,  # Official observatory alert statements
                metadata={"monitoring_network": "USGS", "api": "HANS"},
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.debug(f"Failed to parse HANS volcano entry: {e}")
            return None

    async def _fetch_impl(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        elevated_only: bool = False,
        **kwargs: Any,
    ) -> list[DataPoint]:
        """Fetch live volcano alert levels from the USGS HANS public API.

        Args:
            start_time: Unused (HANS returns the current alert state).
            end_time: Unused (HANS returns the current alert state).
            elevated_only: Fetch only volcanoes currently above NORMAL.
            **kwargs: Unused source-specific parameters.

        Returns:
            One data point per monitored volcano with its official alert
            level and aviation color code.
        """
        endpoint = self.ELEVATED_ENDPOINT if elevated_only else self.MONITORED_ENDPOINT
        response = await self._http_get(endpoint)
        entries = response.json()

        if not isinstance(entries, list):
            raise DataSourceError(
                f"USGS Volcano: unexpected HANS payload for {endpoint}: " f"{str(entries)[:200]}",
                source_id=self.source_id,
                retryable=False,
            )

        data_points: list[DataPoint] = []
        for entry in entries:
            point = self._parse_hans_entry(entry)
            if point is not None:
                data_points.append(point)

        logger.info(
            f"USGS Volcano: Fetched {len(data_points)} volcano alert entries "
            f"({'elevated only' if elevated_only else 'all monitored'})"
        )
        return data_points


# =============================================================================
# NOAA National Water Prediction Service
# =============================================================================


class NOAANWPSSource(DataSourceBase):
    """NOAA National Water Prediction Service data source (river gauges).

    Provides water-related data from the NWPS v1 API:
    - Stream gauge readings (observed stage / flow)
    - Flood category per gauge (no_flooding / action / minor / moderate / major)
    - Forecast stage where available

    The ``/gauges`` endpoint requires a bounding box; without one the API
    returns an empty gauge list, so a ``bbox`` must be configured (or passed
    per-fetch) for the source to produce data.

    No authentication required.

    Example:
        >>> source = NOAANWPSSource(bbox=(-96.0, 28.0, -93.0, 31.0))  # Houston area
        >>> result = await source.fetch()
    """

    DEFAULT_BASE_URL = "https://api.water.noaa.gov/nwps/v1/"

    # NWPS sentinel for "no data" numeric fields.
    _MISSING = -999

    def __init__(
        self,
        bbox: tuple[float, float, float, float] | None = None,
        max_gauges: int = 50,
        config: DataSourceConfig | None = None,
    ) -> None:
        """Initialize NOAA NWPS data source.

        Args:
            bbox: Bounding box (xmin, ymin, xmax, ymax) in EPSG:4326 degrees.
                Required by the NWPS ``/gauges`` endpoint for non-empty output.
            max_gauges: Maximum gauges to parse per fetch.
            config: Optional base configuration.
        """
        base_config = config or DataSourceConfig()
        base_config.rate_limit = RateLimitConfig(
            requests_per_hour=0,
            min_interval_seconds=30.0,
        )
        base_config.cache = CacheConfig(ttl_seconds=300)

        super().__init__(base_config)

        self._bbox = bbox
        self._max_gauges = max_gauges

    @property
    def source_id(self) -> str:
        """Source id."""
        return "noaa_nwps"

    @property
    def default_source_types(self) -> list[DataSourceType]:
        """Default source types."""
        return [DataSourceType.FLOOD]

    def _flood_category_to_alert(self, category: str | None) -> AlertLevel:
        """Convert an NWPS ``floodCategory`` string to AlertLevel.

        Handles both the exact NWPS v1 category tokens (``no_flooding`` /
        ``action`` / ``minor`` / ``moderate`` / ``major``) and descriptive
        strings ("major flood"). The ``no_flooding`` token is checked before
        any substring rules: it contains the substring "flood", so naive
        substring matching would misreport a quiet gauge as MODERATE.
        """
        if not category:
            return AlertLevel.NONE

        cat = category.lower()
        if "no_flood" in cat or "not_defined" in cat or "obs_not_current" in cat:
            return AlertLevel.NONE
        if "major" in cat:
            return AlertLevel.SEVERE
        if "moderate" in cat:
            return AlertLevel.STRONG
        if "minor" in cat or "flood" in cat:
            return AlertLevel.MODERATE
        if "action" in cat:
            return AlertLevel.MINOR
        return AlertLevel.NONE

    # Backwards-compatible alias for the pre-NWPS-v1 method name.
    def _flood_stage_to_alert(self, stage: str | None) -> AlertLevel:
        """Deprecated alias for :meth:`_flood_category_to_alert`."""
        return self._flood_category_to_alert(stage)

    @classmethod
    def _numeric_or_none(cls, value: Any) -> float | None:
        """Return float(value) unless it is missing or the -999 sentinel."""
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return None if numeric == cls._MISSING else numeric

    async def _fetch_impl(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        **kwargs: Any,
    ) -> list[DataPoint]:
        """Fetch river gauge readings from NOAA NWPS.

        Args:
            start_time: Unused (NWPS returns current gauge state).
            end_time: Unused (NWPS returns current gauge state).
            bbox: Optional per-fetch bounding-box override.
            **kwargs: Unused source-specific parameters.

        Returns:
            One data point per gauge with observed stage/flow and flood
            category.
        """
        effective_bbox = bbox or self._bbox
        params: dict[str, Any] = {"srid": "EPSG_4326"}
        if effective_bbox is not None:
            xmin, ymin, xmax, ymax = effective_bbox
            params.update(
                {
                    "bbox.xmin": xmin,
                    "bbox.ymin": ymin,
                    "bbox.xmax": xmax,
                    "bbox.ymax": ymax,
                }
            )
        else:
            logger.warning(
                "NOAA NWPS: no bbox configured; the /gauges endpoint returns an "
                "empty list without one"
            )

        data_points: list[DataPoint] = []

        response = await self._http_get("gauges", params=params)
        data = response.json()

        # Contract check: the NWPS v1 /gauges payload always carries a
        # "gauges" array (possibly empty for an unpopulated bbox). A payload
        # without it is endpoint drift and must fail loud.
        if not isinstance(data, dict) or "gauges" not in data:
            raise DataSourceError(
                f"NWPS payload has no 'gauges' array (got {type(data).__name__}); "
                "endpoint contract drift",
                source_id=self.source_id,
                retryable=False,
            )
        gauges = data["gauges"]

        for gauge in gauges[: self._max_gauges]:
            try:
                gauge_id = gauge.get("lid", "")
                name = gauge.get("name", "Unknown")

                lat = gauge.get("latitude", 0)
                lon = gauge.get("longitude", 0)

                status = gauge.get("status", {})
                observed = status.get("observed", {})
                forecast = status.get("forecast", {})

                flood_category = observed.get("floodCategory")
                observed_value = self._numeric_or_none(observed.get("primary"))
                forecast_value = self._numeric_or_none(forecast.get("primary"))

                valid_time = observed.get("validTime")
                if valid_time and not str(valid_time).startswith("0001"):
                    timestamp = datetime.fromisoformat(str(valid_time).replace("Z", "+00:00"))
                else:
                    timestamp = datetime.now(UTC)

                data_points.append(
                    DataPoint(
                        source_id=self.source_id,
                        source_type=DataSourceType.FLOOD,
                        event_id=f"nwps_{gauge_id}",
                        timestamp=timestamp,
                        data={
                            "gauge_id": gauge_id,
                            "name": name,
                            "observed_value": observed_value,
                            "observed_unit": observed.get("primaryUnit"),
                            "observed_secondary": self._numeric_or_none(observed.get("secondary")),
                            "observed_secondary_unit": observed.get("secondaryUnit"),
                            "forecast_value": forecast_value,
                            "forecast_unit": forecast.get("primaryUnit"),
                            "flood_category": flood_category,
                            "forecast_flood_category": forecast.get("floodCategory"),
                            "wfo": gauge.get("wfo", {}).get("abbreviation"),
                            "state": gauge.get("state", {}).get("abbreviation"),
                        },
                        location=(float(lat), float(lon), 0.0),
                        alert_level=self._flood_category_to_alert(flood_category),
                        confidence=0.9,
                        metadata={"api_version": "NWPS v1"},
                    )
                )

            except (ValueError, KeyError, TypeError) as e:
                logger.debug(f"Failed to parse gauge: {e}")
                continue

        logger.info(f"NOAA NWPS: Fetched {len(data_points)} gauge readings")
        return data_points


# =============================================================================
# NOAA CO-OPS (Tides & Currents)
# =============================================================================


class COOPSProduct(Enum):
    """NOAA CO-OPS data products."""

    WATER_LEVEL = "water_level"
    PREDICTIONS = "predictions"
    CURRENTS = "currents"
    AIR_TEMPERATURE = "air_temperature"
    WATER_TEMPERATURE = "water_temperature"
    WIND = "wind"
    AIR_PRESSURE = "air_pressure"


class NOAACOOPSSource(DataSourceBase):
    """NOAA CO-OPS (Tides & Currents) data source.

    Provides tidal and oceanographic data:
    - Water levels
    - Tide predictions
    - Currents
    - Air/water temperature
    - Wind data
    - Air pressure

    No authentication required.

    Example:
        >>> source = NOAACOOPSSource(station_id="8454000")  # Providence, RI
        >>> result = await source.fetch(products=[COOPSProduct.WATER_LEVEL])
    """

    DEFAULT_BASE_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/"

    # Sample NOAA tide stations
    SAMPLE_STATIONS: dict[str, tuple[str, float, float]] = {
        "8454000": ("Providence, RI", 41.807, -71.401),
        "8518750": ("The Battery, NY", 40.700, -74.014),
        "9410230": ("La Jolla, CA", 32.867, -117.258),
        "1612340": ("Honolulu, HI", 21.307, -157.867),
        "8665530": ("Charleston, SC", 32.782, -79.924),
    }

    def __init__(
        self,
        station_id: str = "8518750",
        products: list[COOPSProduct] | None = None,
        config: DataSourceConfig | None = None,
    ) -> None:
        """Initialize NOAA CO-OPS data source.

        Args:
            station_id: NOAA station ID
            products: Products to fetch (None = water level)
            config: Optional base configuration
        """
        base_config = config or DataSourceConfig()
        base_config.rate_limit = RateLimitConfig(
            requests_per_hour=0,
            min_interval_seconds=10.0,
        )
        base_config.cache = CacheConfig(ttl_seconds=360)  # 6 min cache

        super().__init__(base_config)

        self._station_id = station_id
        self._products = products or [COOPSProduct.WATER_LEVEL]

    @property
    def source_id(self) -> str:
        """Source id."""
        return f"noaa_coops_{self._station_id}"

    @property
    def default_source_types(self) -> list[DataSourceType]:
        """Default source types."""
        return [DataSourceType.TIDE]

    async def _fetch_product(
        self,
        product: COOPSProduct,
        start_time: datetime,
        end_time: datetime,
    ) -> list[DataPoint]:
        """Fetch a specific CO-OPS product."""
        params = {
            "begin_date": start_time.strftime("%Y%m%d %H:%M"),
            "end_date": end_time.strftime("%Y%m%d %H:%M"),
            "station": self._station_id,
            "product": product.value,
            "datum": "MLLW",  # Mean Lower Low Water
            "units": "metric",
            "time_zone": "gmt",
            "application": "MercuryAgent",
            "format": "json",
        }

        data_points: list[DataPoint] = []

        # A fetch failure propagates loudly (DataSourceError from _http_get):
        # silently returning [] here used to convert an outage or contract
        # drift into a fabricated "no readings" result.
        response = await self._http_get("datagetter", params=params)
        data = response.json()

        # CO-OPS reports errors as {"error": {"message": ...}} with HTTP 200.
        if isinstance(data, dict) and "error" in data:
            raise DataSourceError(
                f"CO-OPS {product.value} error for station {self._station_id}: "
                f"{data['error'].get('message', data['error'])}",
                source_id=self.source_id,
                retryable=False,
            )

        # Get station info
        station_info = self.SAMPLE_STATIONS.get(self._station_id, ("Unknown Station", 0.0, 0.0))
        station_name, lat, lon = station_info

        readings = data.get("data", [])

        for reading in readings[-20:]:  # Last 20 readings
            try:
                time_str = reading.get("t", "")
                value = reading.get("v")

                if not time_str or value is None:
                    continue

                timestamp = datetime.strptime(time_str, "%Y-%m-%d %H:%M").replace(tzinfo=UTC)

                data_points.append(
                    DataPoint(
                        source_id=self.source_id,
                        source_type=DataSourceType.TIDE,
                        event_id=f"coops_{self._station_id}_{product.value}_{timestamp.isoformat()}",
                        timestamp=timestamp,
                        data={
                            "station_id": self._station_id,
                            "station_name": station_name,
                            "product": product.value,
                            "value": float(value),
                            "unit": "meters" if product == COOPSProduct.WATER_LEVEL else None,
                            "quality": reading.get("q"),
                        },
                        location=(lat, lon, 0.0),
                        confidence=0.95 if reading.get("q") == "v" else 0.8,
                        metadata={"datum": "MLLW"},
                    )
                )

            except (ValueError, KeyError, TypeError) as e:
                logger.warning(f"Failed to parse CO-OPS reading: {e}")
                continue

        return data_points

    async def _fetch_impl(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        **kwargs: Any,
    ) -> list[DataPoint]:
        """Fetch tidal data from NOAA CO-OPS."""
        end_time = end_time or datetime.now(UTC)
        start_time = start_time or (end_time - timedelta(hours=24))

        all_data_points: list[DataPoint] = []

        for product in self._products:
            data_points = await self._fetch_product(product, start_time, end_time)
            all_data_points.extend(data_points)

        logger.info(
            f"NOAA CO-OPS: Fetched {len(all_data_points)} readings from station {self._station_id}"
        )
        return all_data_points


# =============================================================================
# NWS Weather Alerts
# =============================================================================


class NWSAlertSeverity(Enum):
    """NWS alert severity levels."""

    EXTREME = "Extreme"
    SEVERE = "Severe"
    MODERATE = "Moderate"
    MINOR = "Minor"
    UNKNOWN = "Unknown"


class NWSWeatherAlertsSource(DataSourceBase):
    """National Weather Service Weather Alerts API data source.

    Provides active weather alerts in GeoJSON and CAP 1.2 XML formats:
    - Watches, warnings, advisories
    - All NWS hazardous weather products
    - Real-time alert updates

    No authentication required (User-Agent recommended).

    Example:
        >>> source = NWSWeatherAlertsSource(state="CA")
        >>> result = await source.fetch()
    """

    DEFAULT_BASE_URL = "https://api.weather.gov/"
    DEFAULT_USER_AGENT = "MercuryAgent/1.7.0 (steel.sa.llc@gmail.com)"

    def __init__(
        self,
        state: str | None = None,
        zone: str | None = None,
        event_types: list[str] | None = None,
        config: DataSourceConfig | None = None,
    ) -> None:
        """Initialize NWS Weather Alerts data source.

        Args:
            state: Two-letter state code (e.g., "CA", "TX")
            zone: NWS zone ID (e.g., "CAZ006")
            event_types: Filter by event types (e.g., ["Tornado Warning"])
            config: Optional base configuration
        """
        base_config = config or DataSourceConfig()
        base_config.rate_limit = RateLimitConfig(
            requests_per_hour=0,
            min_interval_seconds=30.0,
        )
        base_config.cache = CacheConfig(ttl_seconds=60)
        base_config.headers = {
            "User-Agent": self.DEFAULT_USER_AGENT,
            "Accept": "application/geo+json",
        }

        super().__init__(base_config)

        self._state = state
        self._zone = zone
        self._event_types = event_types

    @property
    def source_id(self) -> str:
        """Source id."""
        suffix = self._state or self._zone or "all"
        return f"nws_alerts_{suffix}"

    @property
    def default_source_types(self) -> list[DataSourceType]:
        """Default source types."""
        return [DataSourceType.WEATHER_ALERT]

    def _severity_to_alert_level(self, severity: str) -> AlertLevel:
        """Convert NWS severity to AlertLevel."""
        mapping = {
            "extreme": AlertLevel.EXTREME,
            "severe": AlertLevel.SEVERE,
            "moderate": AlertLevel.STRONG,
            "minor": AlertLevel.MODERATE,
        }
        return mapping.get(severity.lower(), AlertLevel.MINOR)

    async def _fetch_impl(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        **kwargs: Any,
    ) -> list[DataPoint]:
        """Fetch active weather alerts from NWS."""
        params: dict[str, str] = {"status": "actual"}

        if self._state:
            params["area"] = self._state
        if self._zone:
            params["zone"] = self._zone
        if self._event_types:
            params["event"] = ",".join(self._event_types)

        response = await self._http_get("alerts/active", params=params)
        data = response.json()

        data_points: list[DataPoint] = []

        for feature in data.get("features", []):
            try:
                props = feature.get("properties", {})
                geometry = feature.get("geometry")

                alert_id = props.get("id", "")
                event = props.get("event", "Unknown")
                severity = props.get("severity", "Unknown")
                certainty = props.get("certainty", "Unknown")
                urgency = props.get("urgency", "Unknown")

                # Parse timestamps
                effective = props.get("effective", "")
                expires = props.get("expires", "")

                if effective:
                    timestamp = datetime.fromisoformat(effective.replace("Z", "+00:00"))
                else:
                    timestamp = datetime.now(UTC)

                # Get centroid location from geometry
                location = None
                if geometry and geometry.get("type") == "Polygon":
                    coords = geometry.get("coordinates", [[]])[0]
                    if coords:
                        # Calculate centroid
                        lons = [c[0] for c in coords]
                        lats = [c[1] for c in coords]
                        location = (
                            sum(lats) / len(lats),
                            sum(lons) / len(lons),
                            0.0,
                        )

                data_points.append(
                    DataPoint(
                        source_id=self.source_id,
                        source_type=DataSourceType.WEATHER_ALERT,
                        event_id=alert_id,
                        timestamp=timestamp,
                        data={
                            "event": event,
                            "severity": severity,
                            "certainty": certainty,
                            "urgency": urgency,
                            "headline": props.get("headline", "")[:200],
                            "description": props.get("description", "")[:500],
                            "instruction": props.get("instruction", "")[:500],
                            "area_desc": props.get("areaDesc", ""),
                            "effective": effective,
                            "expires": expires,
                            "sender": props.get("senderName", "NWS"),
                            "status": props.get("status"),
                            "message_type": props.get("messageType"),
                        },
                        location=location,
                        alert_level=self._severity_to_alert_level(severity),
                        confidence=0.99,  # Official NWS alerts
                        metadata={"format": "CAP 1.2", "api_version": "NWS API"},
                    )
                )

            except (ValueError, KeyError, TypeError) as e:
                logger.debug(f"Failed to parse NWS alert: {e}")
                continue

        logger.info(f"NWS Alerts: Fetched {len(data_points)} active alerts")
        return data_points


# =============================================================================
# EPA AirNow
# =============================================================================


class AQICategory(Enum):
    """EPA Air Quality Index categories."""

    GOOD = (0, 50, "Good")
    MODERATE = (51, 100, "Moderate")
    UNHEALTHY_SENSITIVE = (101, 150, "Unhealthy for Sensitive Groups")
    UNHEALTHY = (151, 200, "Unhealthy")
    VERY_UNHEALTHY = (201, 300, "Very Unhealthy")
    HAZARDOUS = (301, 500, "Hazardous")


class EPAAirNowSource(DataSourceBase):
    """EPA AirNow API data source.

    Provides air quality data:
    - Current AQI observations
    - AQI forecasts
    - Pollutant data: PM2.5, PM10, ozone, wildfire smoke

    Requires free API key from AirNow.

    Example:
        >>> source = EPAAirNowSource(api_key="your_key", latitude=37.7749, longitude=-122.4194)
        >>> result = await source.fetch()
    """

    DEFAULT_BASE_URL = "https://www.airnowapi.org/aq/"

    def __init__(
        self,
        api_key: str,
        latitude: float | None = None,
        longitude: float | None = None,
        zip_code: str | None = None,
        config: DataSourceConfig | None = None,
    ) -> None:
        """Initialize EPA AirNow data source.

        Args:
            api_key: AirNow API key
            latitude: Latitude for location-based queries
            longitude: Longitude for location-based queries
            zip_code: ZIP code for location-based queries
            config: Optional base configuration
        """
        base_config = config or DataSourceConfig()
        base_config.api_key = api_key
        base_config.rate_limit = RateLimitConfig(
            requests_per_hour=500,
            min_interval_seconds=2.0,
        )
        base_config.cache = CacheConfig(ttl_seconds=1800)  # 30 min cache

        super().__init__(base_config)

        self._api_key = api_key
        self._latitude = latitude
        self._longitude = longitude
        self._zip_code = zip_code

    @property
    def source_id(self) -> str:
        """Source id."""
        return "epa_airnow"

    @property
    def default_source_types(self) -> list[DataSourceType]:
        """Default source types."""
        return [DataSourceType.AIR_QUALITY]

    def _aqi_to_alert_level(self, aqi: int) -> AlertLevel:
        """Convert AQI value to alert level."""
        if aqi > 300:
            return AlertLevel.EXTREME  # Hazardous
        elif aqi > 200:
            return AlertLevel.SEVERE  # Very Unhealthy
        elif aqi > 150:
            return AlertLevel.STRONG  # Unhealthy
        elif aqi > 100:
            return AlertLevel.MODERATE  # Unhealthy for Sensitive
        elif aqi > 50:
            return AlertLevel.MINOR  # Moderate
        return AlertLevel.NONE  # Good

    def _get_aqi_category(self, aqi: int) -> str:
        """Get AQI category name."""
        for category in AQICategory:
            min_val, max_val, name = category.value
            if min_val <= aqi <= max_val:
                return name
        return "Unknown"

    async def _fetch_impl(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        **kwargs: Any,
    ) -> list[DataPoint]:
        """Fetch air quality data from EPA AirNow."""
        data_points: list[DataPoint] = []

        # Build params based on location type
        params: dict[str, Any] = {
            "API_KEY": self._api_key,
            "format": "application/json",
        }

        endpoint = "observation/latLong/current/"

        if self._latitude is not None and self._longitude is not None:
            params["latitude"] = self._latitude
            params["longitude"] = self._longitude
        elif self._zip_code:
            endpoint = "observation/zipCode/current/"
            params["zipCode"] = self._zip_code
        else:
            # Default to a location (San Francisco)
            params["latitude"] = 37.7749
            params["longitude"] = -122.4194

        params["distance"] = 25  # miles

        try:
            response = await self._http_get(endpoint, params=params)
            data = response.json()

            for obs in data:
                try:
                    aqi = int(obs.get("AQI", 0) or 0)
                    parameter = obs.get("ParameterName", "")
                    reporting_area = obs.get("ReportingArea", "")

                    # Parse date
                    date_observed = obs.get("DateObserved", "")
                    hour_observed = obs.get("HourObserved", 0)

                    if date_observed:
                        timestamp = datetime.strptime(
                            f"{date_observed} {hour_observed}:00", "%Y-%m-%d %H:%M"
                        ).replace(tzinfo=UTC)
                    else:
                        timestamp = datetime.now(UTC)

                    # Get location
                    lat = obs.get("Latitude", self._latitude or 0)
                    lon = obs.get("Longitude", self._longitude or 0)

                    data_points.append(
                        DataPoint(
                            source_id=self.source_id,
                            source_type=DataSourceType.AIR_QUALITY,
                            event_id=f"airnow_{reporting_area}_{parameter}_{timestamp.isoformat()}",
                            timestamp=timestamp,
                            data={
                                "aqi": aqi,
                                "parameter": parameter,
                                "category": self._get_aqi_category(aqi),
                                "category_number": obs.get("Category", {}).get("Number"),
                                "reporting_area": reporting_area,
                                "state_code": obs.get("StateCode"),
                                "date_observed": date_observed,
                                "hour_observed": hour_observed,
                            },
                            location=(float(lat), float(lon), 0.0),
                            alert_level=self._aqi_to_alert_level(aqi),
                            confidence=0.9,
                            metadata={"api_version": "AirNow API"},
                        )
                    )

                except (ValueError, KeyError, TypeError) as e:
                    logger.debug(f"Failed to parse AirNow observation: {e}")
                    continue

        except Exception as e:
            logger.warning(f"EPA AirNow fetch failed: {e}")

        logger.info(f"EPA AirNow: Fetched {len(data_points)} air quality readings")
        return data_points
