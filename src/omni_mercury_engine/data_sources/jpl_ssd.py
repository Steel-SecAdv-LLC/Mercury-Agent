# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""NASA/JPL Solar System Dynamics (SSD) data sources.

Production-grade integrations for the JPL SSD public APIs:

- Fireball API — bolides/fireballs detected by US Government sensors
  (https://ssd-api.jpl.nasa.gov/doc/fireball.html)
- Sentry API — long-term Earth-impact risk monitoring
  (https://ssd-api.jpl.nasa.gov/doc/sentry.html)

Both APIs are keyless. These clients replace the private module-level HTTP
loaders that previously lived inside
``detectors/geological/disaster_detectors.py`` so that meteor/NEO ingestion
goes through the standard :class:`~omni_mercury_engine.data_sources.base.DataSourceBase`
resilience stack (rate limiting, caching, circuit breaker) and emits
standardized :class:`~omni_mercury_engine.data_sources.base.DataPoint` objects.

Close-approach data intentionally has NO client here: the existing
:class:`~omni_mercury_engine.data_sources.space_weather.NASANeoWsSource`
already provides it (miss distance, relative velocity, diameter estimates and
hazard flags); :func:`close_approaches_from_neows_datapoints` converts its
data points into :class:`CloseApproachEvent` records.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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

_AU_KM = 149597870.7


@dataclass
class FireballEvent:
    """NASA CNEOS Fireball event data.

    Represents a bolide (fireball) detected by US Government sensors.
    Data source: NASA JPL Center for Near Earth Object Studies (CNEOS)
    https://cneos.jpl.nasa.gov/fireballs/
    """

    date: datetime
    latitude: float | None
    longitude: float | None
    altitude_km: float | None
    velocity_km_s: float | None
    total_radiated_energy_j: float | None
    calculated_total_impact_energy_kt: float | None

    @property
    def estimated_size_m(self) -> float | None:
        """Estimate size from impact energy using empirical relation.

        Based on: E = 4.185 x 10^10 x D^3 (Brown et al., 2002)
        Where E is energy in Joules and D is diameter in meters.
        """
        if self.calculated_total_impact_energy_kt is None:
            return None
        # Convert kt TNT to Joules (1 kt = 4.184e12 J)
        energy_j = self.calculated_total_impact_energy_kt * 4.184e12
        # Solve for diameter: D = (E / 4.185e10)^(1/3)
        diameter = (energy_j / 4.185e10) ** (1 / 3)
        return float(diameter)


@dataclass
class CloseApproachEvent:
    """Near-Earth object close approach event.

    Built from :class:`~omni_mercury_engine.data_sources.space_weather.NASANeoWsSource`
    data points (see :func:`close_approaches_from_neows_datapoints`).
    """

    designation: str
    close_approach_date: datetime
    nominal_distance_au: float
    nominal_distance_km: float
    relative_velocity_km_s: float
    absolute_magnitude_h: float | None
    estimated_diameter_km: float | None


@dataclass
class SentryImpactRisk:
    """NASA Sentry impact monitoring data.

    Represents a potential future Earth impact event monitored by Sentry.
    Data source: NASA JPL Sentry Impact Monitoring System
    https://cneos.jpl.nasa.gov/sentry/
    """

    designation: str
    potential_impacts: int
    impact_probability: float
    palermo_scale: float
    torino_scale: int
    estimated_diameter_km: float | None
    next_impact_date: datetime | None


class JPLFireballSource(DataSourceBase):
    """NASA/JPL CNEOS Fireball API data source.

    Provides atmospheric bolide events with location, velocity, altitude and
    energy estimates. Keyless; fetched via ``https://ssd-api.jpl.nasa.gov/fireball.api``.

    Example:
        >>> source = JPLFireballSource(days_back=30)
        >>> result = await source.fetch()
    """

    DEFAULT_BASE_URL = "https://ssd-api.jpl.nasa.gov/"

    def __init__(
        self,
        days_back: int = 30,
        min_energy_kt: float = 0.0,
        config: DataSourceConfig | None = None,
    ) -> None:
        """Initialize the JPL Fireball data source.

        Args:
            days_back: Default look-back window in days.
            min_energy_kt: Minimum calculated impact energy (kt TNT) filter.
            config: Optional base configuration.
        """
        base_config = config or DataSourceConfig()
        base_config.rate_limit = RateLimitConfig(
            requests_per_hour=60,
            min_interval_seconds=5.0,
        )
        # 6 h cache: fireball catalog updates are infrequent, and this
        # preserves the historical MeteorDetector refresh cadence.
        base_config.cache = CacheConfig(ttl_seconds=21600)

        super().__init__(base_config)

        self._days_back = days_back
        self._min_energy_kt = min_energy_kt

    @property
    def source_id(self) -> str:
        """Source id."""
        return "jpl_fireball"

    @property
    def default_source_types(self) -> list[DataSourceType]:
        """Default source types."""
        return [DataSourceType.NEAR_EARTH_OBJECT]

    @staticmethod
    def _energy_to_alert_level(impact_energy_kt: float | None) -> AlertLevel:
        """Order-of-magnitude impact-energy tiers to AlertLevel.

        Anchors: Chelyabinsk (2013) ~440 kt -> SEVERE; typical detected
        bolides are well below 1 kt -> NONE/MINOR.
        """
        if impact_energy_kt is None:
            return AlertLevel.NONE
        if impact_energy_kt >= 5000:
            return AlertLevel.EXTREME
        if impact_energy_kt >= 500:
            return AlertLevel.SEVERE
        if impact_energy_kt >= 50:
            return AlertLevel.STRONG
        if impact_energy_kt >= 5:
            return AlertLevel.MODERATE
        if impact_energy_kt >= 0.5:
            return AlertLevel.MINOR
        return AlertLevel.NONE

    async def _fetch_impl(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        min_energy_kt: float | None = None,
        **kwargs: Any,
    ) -> list[DataPoint]:
        """Fetch fireball events from the JPL Fireball API."""
        end_time = end_time or datetime.now(UTC)
        start_time = start_time or (end_time - timedelta(days=self._days_back))
        energy_floor = self._min_energy_kt if min_energy_kt is None else min_energy_kt

        params = {
            "date-min": start_time.strftime("%Y-%m-%d"),
            "date-max": end_time.strftime("%Y-%m-%d"),
            "req-loc": "false",
        }

        response = await self._http_get("fireball.api", params=params)
        payload = response.json()

        rows = payload.get("data") or []
        fields = {name: i for i, name in enumerate(payload.get("fields", []))}
        if rows and ("date" not in fields or "impact-e" not in fields):
            raise DataSourceError(
                f"JPL Fireball: unexpected field set {sorted(fields)}",
                source_id=self.source_id,
                retryable=False,
            )

        def _column(row: list[Any], name: str) -> Any:
            idx = fields.get(name)
            return row[idx] if idx is not None and idx < len(row) else None

        data_points: list[DataPoint] = []
        for row in rows:
            try:
                date_str = _column(row, "date")
                event_date = datetime.strptime(str(date_str), "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=UTC
                )

                lat_raw = _column(row, "lat")
                lon_raw = _column(row, "lon")
                lat = float(lat_raw) if lat_raw not in (None, "") else None
                lon = float(lon_raw) if lon_raw not in (None, "") else None
                if lat is not None and _column(row, "lat-dir") == "S":
                    lat = -lat
                if lon is not None and _column(row, "lon-dir") == "W":
                    lon = -lon

                alt_raw = _column(row, "alt")
                vel_raw = _column(row, "vel")
                energy_raw = _column(row, "energy")
                impact_raw = _column(row, "impact-e")

                altitude_km = float(alt_raw) if alt_raw not in (None, "") else None
                velocity = float(vel_raw) if vel_raw not in (None, "") else None
                radiated_j = float(energy_raw) * 1e10 if energy_raw not in (None, "") else None
                impact_kt = float(impact_raw) if impact_raw not in (None, "") else None

                if energy_floor > 0 and (impact_kt is None or impact_kt < energy_floor):
                    continue

                event = FireballEvent(
                    date=event_date,
                    latitude=lat,
                    longitude=lon,
                    altitude_km=altitude_km,
                    velocity_km_s=velocity,
                    total_radiated_energy_j=radiated_j,
                    calculated_total_impact_energy_kt=impact_kt,
                )

                location = None
                if lat is not None and lon is not None:
                    location = (lat, lon, altitude_km or 0.0)

                data_points.append(
                    DataPoint(
                        source_id=self.source_id,
                        source_type=DataSourceType.NEAR_EARTH_OBJECT,
                        event_id=f"fireball_{event_date.isoformat()}",
                        timestamp=event_date,
                        data={
                            "velocity_km_s": velocity,
                            "altitude_km": altitude_km,
                            "total_radiated_energy_j": radiated_j,
                            "impact_energy_kt": impact_kt,
                            "estimated_size_m": event.estimated_size_m,
                        },
                        location=location,
                        alert_level=self._energy_to_alert_level(impact_kt),
                        confidence=0.9,  # USG sensor detections
                        metadata={"api": "JPL Fireball", "event_kind": "bolide"},
                    )
                )
            except (ValueError, IndexError, KeyError, TypeError) as e:
                logger.debug(f"Skipping malformed fireball record: {e}")
                continue

        logger.info(f"JPL Fireball: Fetched {len(data_points)} bolide events")
        return data_points


class JPLSentrySource(DataSourceBase):
    """NASA/JPL Sentry impact-risk monitoring data source.

    Fetches the Sentry summary table: every object with a non-zero computed
    probability of Earth impact within the next ~100 years, with Palermo and
    Torino scale ratings. Keyless; fetched via
    ``https://ssd-api.jpl.nasa.gov/sentry.api`` (summary mode).

    Note: the summary-mode fields are ``ps_cum``/``ps_max`` (Palermo,
    cumulative/max) and ``ts_max`` (Torino) — NOT ``ps``/``ts`` as the legacy
    in-detector loader assumed (that loader therefore always emitted its
    ``-10``/``0`` fallbacks).

    Example:
        >>> source = JPLSentrySource()
        >>> result = await source.fetch(min_palermo=-3.0)
    """

    DEFAULT_BASE_URL = "https://ssd-api.jpl.nasa.gov/"

    def __init__(
        self,
        config: DataSourceConfig | None = None,
    ) -> None:
        """Initialize the JPL Sentry data source."""
        base_config = config or DataSourceConfig()
        base_config.rate_limit = RateLimitConfig(
            requests_per_hour=60,
            min_interval_seconds=5.0,
        )
        # 6 h cache: Sentry risk-table updates are infrequent, and this
        # preserves the historical MeteorDetector refresh cadence.
        base_config.cache = CacheConfig(ttl_seconds=21600)

        super().__init__(base_config)

    @property
    def source_id(self) -> str:
        """Source id."""
        return "jpl_sentry"

    @property
    def default_source_types(self) -> list[DataSourceType]:
        """Default source types."""
        return [DataSourceType.NEAR_EARTH_OBJECT]

    @staticmethod
    def _risk_to_alert_level(palermo_cum: float, torino: int) -> AlertLevel:
        """Map Palermo/Torino ratings to AlertLevel.

        Torino >= 5 is a certain-threat band (SEVERE+); Torino 1-4 merits
        attention; Palermo >= -2 means within two orders of magnitude of the
        background hazard.
        """
        if torino >= 8:
            return AlertLevel.EXTREME
        if torino >= 5:
            return AlertLevel.SEVERE
        if torino >= 2:
            return AlertLevel.STRONG
        if torino >= 1 or palermo_cum >= 0:
            return AlertLevel.MODERATE
        if palermo_cum >= -2:
            return AlertLevel.MINOR
        return AlertLevel.NONE

    async def _fetch_impl(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        min_palermo: float | None = None,
        **kwargs: Any,
    ) -> list[DataPoint]:
        """Fetch the Sentry impact-risk summary table.

        Args:
            start_time: Unused (Sentry is a current risk table).
            end_time: Unused (Sentry is a current risk table).
            min_palermo: Optional cumulative-Palermo-scale floor filter.
            **kwargs: Unused source-specific parameters.

        Returns:
            One data point per monitored object.
        """
        response = await self._http_get("sentry.api")
        payload = response.json()

        objects = payload.get("data") or []
        fetched_at = datetime.now(UTC)

        data_points: list[DataPoint] = []
        for obj in objects:
            try:
                designation = str(obj.get("des", "Unknown"))
                impact_probability = float(obj.get("ip", 0.0) or 0.0)
                palermo_cum = float(obj.get("ps_cum", -10.0) or -10.0)
                palermo_max = float(obj.get("ps_max", -10.0) or -10.0)
                torino = int(float(obj.get("ts_max", 0) or 0))
                n_imp = int(obj.get("n_imp", 0) or 0)
                diameter = float(obj["diameter"]) if obj.get("diameter") else None

                if min_palermo is not None and palermo_cum < min_palermo:
                    continue

                next_impact = None
                impact_range = obj.get("range")
                if impact_range:
                    try:
                        first_year = str(impact_range).split("-")[0].strip()
                        next_impact = datetime(int(first_year), 1, 1, tzinfo=UTC)
                    except (ValueError, AttributeError):
                        pass

                data_points.append(
                    DataPoint(
                        source_id=self.source_id,
                        source_type=DataSourceType.NEAR_EARTH_OBJECT,
                        event_id=f"sentry_{designation}",
                        timestamp=fetched_at,
                        data={
                            "designation": designation,
                            "fullname": obj.get("fullname"),
                            "potential_impacts": n_imp,
                            "impact_probability": impact_probability,
                            "palermo_scale_cumulative": palermo_cum,
                            "palermo_scale_max": palermo_max,
                            "torino_scale": torino,
                            "estimated_diameter_km": diameter,
                            "impact_year_range": impact_range,
                            "next_impact_year": next_impact.year if next_impact else None,
                            "last_observation": obj.get("last_obs"),
                            "absolute_magnitude_h": (float(obj["h"]) if obj.get("h") else None),
                        },
                        alert_level=self._risk_to_alert_level(palermo_cum, torino),
                        confidence=0.95,  # JPL orbit-determination products
                        metadata={"api": "JPL Sentry", "mode": "summary"},
                    )
                )
            except (ValueError, KeyError, TypeError) as e:
                logger.debug(f"Skipping malformed Sentry record: {e}")
                continue

        logger.info(f"JPL Sentry: Fetched {len(data_points)} impact-risk objects")
        return data_points


def fireball_events_from_datapoints(points: list[DataPoint]) -> list[FireballEvent]:
    """Convert JPLFireballSource data points into FireballEvent records.

    Args:
        points: Data points emitted by :class:`JPLFireballSource`.

    Returns:
        FireballEvent records, one per bolide data point.
    """
    events: list[FireballEvent] = []
    for dp in points:
        if dp.metadata.get("event_kind") != "bolide":
            continue
        lat, lon, alt = dp.location or (None, None, None)
        events.append(
            FireballEvent(
                date=dp.timestamp,
                latitude=lat,
                longitude=lon,
                altitude_km=dp.data.get("altitude_km", alt),
                velocity_km_s=dp.data.get("velocity_km_s"),
                total_radiated_energy_j=dp.data.get("total_radiated_energy_j"),
                calculated_total_impact_energy_kt=dp.data.get("impact_energy_kt"),
            )
        )
    return events


def sentry_risks_from_datapoints(points: list[DataPoint]) -> list[SentryImpactRisk]:
    """Convert JPLSentrySource data points into SentryImpactRisk records.

    Args:
        points: Data points emitted by :class:`JPLSentrySource`.

    Returns:
        SentryImpactRisk records, one per monitored object.
    """
    risks: list[SentryImpactRisk] = []
    for dp in points:
        if dp.metadata.get("api") != "JPL Sentry":
            continue
        next_impact = None
        impact_range = dp.data.get("impact_year_range")
        if impact_range:
            try:
                next_impact = datetime(int(str(impact_range).split("-")[0]), 1, 1, tzinfo=UTC)
            except (ValueError, AttributeError):
                next_impact = None
        risks.append(
            SentryImpactRisk(
                designation=str(dp.data.get("designation", "Unknown")),
                potential_impacts=int(dp.data.get("potential_impacts", 0)),
                impact_probability=float(dp.data.get("impact_probability", 0.0)),
                palermo_scale=float(dp.data.get("palermo_scale_cumulative", -10.0)),
                torino_scale=int(dp.data.get("torino_scale", 0)),
                estimated_diameter_km=dp.data.get("estimated_diameter_km"),
                next_impact_date=next_impact,
            )
        )
    return risks


def close_approaches_from_neows_datapoints(points: list[DataPoint]) -> list[CloseApproachEvent]:
    """Convert NASANeoWsSource data points into CloseApproachEvent records.

    Uses each NEO's first close-approach entry (the one NeoWs keyed the feed
    date on). Data points without close-approach data are skipped.

    Args:
        points: Data points emitted by
            :class:`~omni_mercury_engine.data_sources.space_weather.NASANeoWsSource`.

    Returns:
        CloseApproachEvent records sorted by approach date.
    """
    events: list[CloseApproachEvent] = []
    for dp in points:
        if dp.source_type != DataSourceType.NEAR_EARTH_OBJECT:
            continue
        approaches = dp.data.get("close_approach_data") or []
        if not approaches:
            continue
        try:
            approach = approaches[0]
            miss = approach.get("miss_distance", {})
            velocity = approach.get("relative_velocity", {})
            distance_km = float(miss.get("kilometers"))
            distance_au = (
                float(miss["astronomical"]) if miss.get("astronomical") else distance_km / _AU_KM
            )
            diameter = dp.data.get("estimated_diameter_km") or {}
            events.append(
                CloseApproachEvent(
                    designation=str(dp.data.get("name", dp.event_id)),
                    close_approach_date=dp.timestamp,
                    nominal_distance_au=distance_au,
                    nominal_distance_km=distance_km,
                    relative_velocity_km_s=float(velocity.get("kilometers_per_second", 0.0)),
                    absolute_magnitude_h=dp.data.get("absolute_magnitude_h"),
                    estimated_diameter_km=diameter.get("avg"),
                )
            )
        except (ValueError, KeyError, TypeError) as e:
            logger.debug(f"Skipping malformed NeoWs close-approach record: {e}")
            continue
    events.sort(key=lambda e: e.close_approach_date)
    return events
