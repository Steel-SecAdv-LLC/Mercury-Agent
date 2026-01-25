"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Base classes and types for STEM Alert & Anomaly Detection data sources.

This module provides:
- DataSourceType: Enum for categorizing data sources
- DataPoint: Standardized data container for all sources
- DataSourceBase: Abstract base class for all data source implementations
- DataSourceManager: Unified manager for multiple data sources
- Resilience patterns: Caching, rate limiting, circuit breaker integration

Design Principles:
1. All data sources return standardized DataPoint objects
2. Built-in resilience with exponential backoff and circuit breakers
3. Configurable caching to respect API rate limits
4. Async-first design with synchronous wrapper support
5. Comprehensive error handling with graceful degradation
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, TypeVar

import httpx

logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """Categories of data sources for anomaly detection.

    Each type represents a distinct domain with specific data characteristics
    and anomaly patterns relevant to Mercury Agent's multi-domain detection.
    """

    # Space Weather & Solar Physics
    SOLAR_FLARE = "solar_flare"
    CME = "coronal_mass_ejection"
    GEOMAGNETIC_STORM = "geomagnetic_storm"
    SOLAR_WIND = "solar_wind"
    SOLAR_ENERGETIC_PARTICLE = "solar_energetic_particle"
    NEAR_EARTH_OBJECT = "near_earth_object"
    NATURAL_EVENT = "natural_event"
    CELESTIAL_BODY = "celestial_body"

    # Geomagnetic & Electromagnetic
    MAGNETOMETER = "magnetometer"
    SCHUMANN_RESONANCE = "schumann_resonance"
    IONOSPHERIC = "ionospheric"
    ELF_VLF = "elf_vlf"

    # Earth Science & Hazards
    EARTHQUAKE = "earthquake"
    VOLCANO = "volcano"
    WEATHER_ALERT = "weather_alert"
    FLOOD = "flood"
    TIDE = "tide"
    AIR_QUALITY = "air_quality"

    # Consciousness Research
    RANDOM_NUMBER_GENERATOR = "random_number_generator"
    GLOBAL_COHERENCE = "global_coherence"

    # Generic
    CUSTOM = "custom"


class AlertLevel(Enum):
    """Standardized alert severity levels.

    Maps to various agency-specific scales:
    - NOAA: G1-G5 (geomagnetic), S1-S5 (solar radiation), R1-R5 (radio blackout)
    - NWS: Minor, Moderate, Severe, Extreme
    - USGS: Normal, Advisory, Watch, Warning
    """

    NONE = 0
    MINOR = 1  # G1, S1, R1, Minor
    MODERATE = 2  # G2, S2, R2, Moderate
    STRONG = 3  # G3, S3, R3, Severe
    SEVERE = 4  # G4, S4, R4
    EXTREME = 5  # G5, S5, R5, Extreme

    @classmethod
    def from_noaa_g_scale(cls, level: int) -> AlertLevel:
        """Convert NOAA G-scale (geomagnetic) to AlertLevel."""
        return cls(min(level, 5))

    @classmethod
    def from_noaa_s_scale(cls, level: int) -> AlertLevel:
        """Convert NOAA S-scale (solar radiation) to AlertLevel."""
        return cls(min(level, 5))

    @classmethod
    def from_noaa_r_scale(cls, level: int) -> AlertLevel:
        """Convert NOAA R-scale (radio blackout) to AlertLevel."""
        return cls(min(level, 5))

    @classmethod
    def from_nws_severity(cls, severity: str) -> AlertLevel:
        """Convert NWS severity string to AlertLevel."""
        mapping = {
            "minor": cls.MINOR,
            "moderate": cls.MODERATE,
            "severe": cls.STRONG,
            "extreme": cls.EXTREME,
        }
        return mapping.get(severity.lower(), cls.NONE)


@dataclass
class DataPoint:
    """Standardized data container for all data sources.

    All data sources normalize their output to this format to enable
    consistent processing in the anomaly detection pipeline.

    Attributes:
        source_id: Unique identifier for the data source instance
        source_type: Category of the data source
        event_id: Unique identifier for this specific event/measurement
        timestamp: UTC timestamp of the event/measurement
        data: Source-specific data payload
        location: Geographic location if applicable (lat, lon, alt)
        alert_level: Standardized alert severity
        confidence: Confidence score [0, 1] in data quality/accuracy
        metadata: Additional source-specific metadata
    """

    source_id: str
    source_type: DataSourceType
    event_id: str
    timestamp: datetime
    data: dict[str, Any]
    location: tuple[float, float, float] | None = None  # (lat, lon, alt_km)
    alert_level: AlertLevel = AlertLevel.NONE
    confidence: float = 0.8
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_id": self.source_id,
            "source_type": self.source_type.value,
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "location": self.location,
            "alert_level": self.alert_level.value,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataPoint:
        """Create from dictionary."""
        return cls(
            source_id=data["source_id"],
            source_type=DataSourceType(data["source_type"]),
            event_id=data["event_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            data=data["data"],
            location=tuple(data["location"]) if data.get("location") else None,
            alert_level=AlertLevel(data.get("alert_level", 0)),
            confidence=data.get("confidence", 0.8),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RateLimitConfig:
    """Rate limiting configuration.

    Attributes:
        requests_per_hour: Maximum requests per hour (0 = unlimited)
        min_interval_seconds: Minimum seconds between requests
        burst_limit: Maximum burst of requests allowed
    """

    requests_per_hour: int = 1000
    min_interval_seconds: float = 1.0
    burst_limit: int = 10


@dataclass
class CacheConfig:
    """Caching configuration.

    Attributes:
        enabled: Whether caching is enabled
        ttl_seconds: Time-to-live for cached data
        max_entries: Maximum cache entries
    """

    enabled: bool = True
    ttl_seconds: int = 300  # 5 minutes default
    max_entries: int = 1000


@dataclass
class DataSourceConfig:
    """Configuration for a data source.

    Attributes:
        api_key: API key if required
        base_url: Override base URL
        timeout_seconds: Request timeout
        retry_attempts: Number of retry attempts
        retry_backoff: Backoff multiplier for retries
        rate_limit: Rate limiting configuration
        cache: Caching configuration
        headers: Additional HTTP headers
        verify_ssl: Whether to verify SSL certificates
    """

    api_key: str | None = None
    base_url: str | None = None
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    retry_backoff: float = 2.0
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    headers: dict[str, str] = field(default_factory=dict)
    verify_ssl: bool = True


@dataclass
class FetchResult:
    """Result of a data fetch operation.

    Attributes:
        success: Whether the fetch succeeded
        data_points: List of fetched data points
        error: Error message if failed
        cached: Whether the result was from cache
        fetch_time_ms: Time taken to fetch in milliseconds
        rate_limited: Whether request was rate limited
    """

    success: bool
    data_points: list[DataPoint] = field(default_factory=list)
    error: str | None = None
    cached: bool = False
    fetch_time_ms: float = 0.0
    rate_limited: bool = False


class DataSourceError(Exception):
    """Exception for data source errors."""

    def __init__(
        self,
        message: str,
        source_id: str | None = None,
        status_code: int | None = None,
        retryable: bool = True,
    ):
        super().__init__(message)
        self.source_id = source_id
        self.status_code = status_code
        self.retryable = retryable


T = TypeVar("T", bound="DataSourceBase")


class DataSourceBase(ABC):
    """Abstract base class for all data sources.

    Provides common functionality for:
    - HTTP request handling with resilience patterns
    - Rate limiting
    - Caching
    - Error handling with exponential backoff

    Subclasses must implement:
    - _fetch_impl(): Actual data fetching logic
    - source_id: Property returning unique source identifier
    - default_source_types: Property returning list of DataSourceTypes

    Example:
        >>> class MySource(DataSourceBase):
        ...     @property
        ...     def source_id(self) -> str:
        ...         return "my_source"
        ...
        ...     @property
        ...     def default_source_types(self) -> list[DataSourceType]:
        ...         return [DataSourceType.CUSTOM]
        ...
        ...     async def _fetch_impl(self) -> list[DataPoint]:
        ...         response = await self._http_get("/endpoint")
        ...         return self._parse_response(response)
    """

    DEFAULT_BASE_URL: str = ""
    DEFAULT_TIMEOUT: float = 30.0
    DEFAULT_USER_AGENT: str = "MercuryAgent/1.1.0 (steel.sa.llc@gmail.com)"

    def __init__(self, config: DataSourceConfig | None = None) -> None:
        """Initialize data source.

        Args:
            config: Configuration options. If None, uses defaults.
        """
        self.config = config or DataSourceConfig()
        self._client: httpx.AsyncClient | None = None
        self._sync_client: httpx.Client | None = None

        # Rate limiting state
        self._last_request_time: float = 0.0
        self._request_count_hour: int = 0
        self._hour_start_time: float = time.time()

        # Cache state
        self._cache: dict[str, tuple[float, list[DataPoint]]] = {}

        # Metrics
        self._total_requests: int = 0
        self._successful_requests: int = 0
        self._failed_requests: int = 0
        self._total_latency_ms: float = 0.0

    @property
    @abstractmethod
    def source_id(self) -> str:
        """Unique identifier for this data source instance."""
        pass

    @property
    @abstractmethod
    def default_source_types(self) -> list[DataSourceType]:
        """List of data source types this source can produce."""
        pass

    @property
    def base_url(self) -> str:
        """Base URL for API requests."""
        return self.config.base_url or self.DEFAULT_BASE_URL

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for requests."""
        headers = {
            "User-Agent": self.DEFAULT_USER_AGENT,
            "Accept": "application/json",
        }
        headers.update(self.config.headers)
        return headers

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.config.timeout_seconds,
                headers=self._get_headers(),
                verify=self.config.verify_ssl,
            )
        return self._client

    def _get_sync_client(self) -> httpx.Client:
        """Get or create sync HTTP client."""
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                timeout=self.config.timeout_seconds,
                headers=self._get_headers(),
                verify=self.config.verify_ssl,
            )
        return self._sync_client

    async def _http_get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Make HTTP GET request with resilience patterns.

        Args:
            endpoint: API endpoint (relative to base_url)
            params: Query parameters

        Returns:
            HTTP response

        Raises:
            DataSourceError: If request fails after retries
        """
        await self._check_rate_limit()

        url = f"{self.base_url}{endpoint}"
        client = await self._get_client()

        last_error: Exception | None = None
        delay = 1.0

        for attempt in range(1, self.config.retry_attempts + 1):
            try:
                self._total_requests += 1
                start_time = time.time()

                response = await client.get(url, params=params)
                response.raise_for_status()

                self._successful_requests += 1
                self._total_latency_ms += (time.time() - start_time) * 1000
                self._last_request_time = time.time()
                self._request_count_hour += 1

                return response

            except httpx.HTTPStatusError as e:
                last_error = e
                self._failed_requests += 1

                if e.response.status_code < 500:
                    # Client error - don't retry
                    raise DataSourceError(
                        f"HTTP {e.response.status_code}: {e.response.text[:200]}",
                        source_id=self.source_id,
                        status_code=e.response.status_code,
                        retryable=False,
                    ) from e

                if attempt < self.config.retry_attempts:
                    logger.warning(
                        f"{self.source_id}: HTTP {e.response.status_code}, "
                        f"retry {attempt}/{self.config.retry_attempts} in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * self.config.retry_backoff, 60.0)

            except httpx.RequestError as e:
                last_error = e
                self._failed_requests += 1

                if attempt < self.config.retry_attempts:
                    logger.warning(
                        f"{self.source_id}: Request error: {e}, "
                        f"retry {attempt}/{self.config.retry_attempts} in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * self.config.retry_backoff, 60.0)

        raise DataSourceError(
            f"Request failed after {self.config.retry_attempts} attempts: {last_error}",
            source_id=self.source_id,
            retryable=True,
        )

    def _http_get_sync(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Synchronous version of _http_get."""
        self._check_rate_limit_sync()

        url = f"{self.base_url}{endpoint}"
        client = self._get_sync_client()

        last_error: Exception | None = None
        delay = 1.0

        for attempt in range(1, self.config.retry_attempts + 1):
            try:
                self._total_requests += 1
                start_time = time.time()

                response = client.get(url, params=params)
                response.raise_for_status()

                self._successful_requests += 1
                self._total_latency_ms += (time.time() - start_time) * 1000
                self._last_request_time = time.time()
                self._request_count_hour += 1

                return response

            except httpx.HTTPStatusError as e:
                last_error = e
                self._failed_requests += 1

                if e.response.status_code < 500:
                    raise DataSourceError(
                        f"HTTP {e.response.status_code}: {e.response.text[:200]}",
                        source_id=self.source_id,
                        status_code=e.response.status_code,
                        retryable=False,
                    ) from e

                if attempt < self.config.retry_attempts:
                    logger.warning(
                        f"{self.source_id}: HTTP {e.response.status_code}, "
                        f"retry {attempt}/{self.config.retry_attempts} in {delay:.1f}s"
                    )
                    time.sleep(delay)
                    delay = min(delay * self.config.retry_backoff, 60.0)

            except httpx.RequestError as e:
                last_error = e
                self._failed_requests += 1

                if attempt < self.config.retry_attempts:
                    logger.warning(
                        f"{self.source_id}: Request error: {e}, "
                        f"retry {attempt}/{self.config.retry_attempts} in {delay:.1f}s"
                    )
                    time.sleep(delay)
                    delay = min(delay * self.config.retry_backoff, 60.0)

        raise DataSourceError(
            f"Request failed after {self.config.retry_attempts} attempts: {last_error}",
            source_id=self.source_id,
            retryable=True,
        )

    async def _check_rate_limit(self) -> None:
        """Check and enforce rate limits."""
        # Reset hourly counter if needed
        current_time = time.time()
        if current_time - self._hour_start_time >= 3600:
            self._request_count_hour = 0
            self._hour_start_time = current_time

        # Check hourly limit
        if (
            self.config.rate_limit.requests_per_hour > 0
            and self._request_count_hour >= self.config.rate_limit.requests_per_hour
        ):
            wait_time = 3600 - (current_time - self._hour_start_time)
            logger.warning(f"{self.source_id}: Rate limit reached, waiting {wait_time:.0f}s")
            raise DataSourceError(
                f"Rate limit exceeded ({self.config.rate_limit.requests_per_hour}/hour)",
                source_id=self.source_id,
                retryable=True,
            )

        # Enforce minimum interval
        time_since_last = current_time - self._last_request_time
        if time_since_last < self.config.rate_limit.min_interval_seconds:
            await asyncio.sleep(self.config.rate_limit.min_interval_seconds - time_since_last)

    def _check_rate_limit_sync(self) -> None:
        """Synchronous rate limit check."""
        current_time = time.time()
        if current_time - self._hour_start_time >= 3600:
            self._request_count_hour = 0
            self._hour_start_time = current_time

        if (
            self.config.rate_limit.requests_per_hour > 0
            and self._request_count_hour >= self.config.rate_limit.requests_per_hour
        ):
            raise DataSourceError(
                f"Rate limit exceeded ({self.config.rate_limit.requests_per_hour}/hour)",
                source_id=self.source_id,
                retryable=True,
            )

        time_since_last = current_time - self._last_request_time
        if time_since_last < self.config.rate_limit.min_interval_seconds:
            time.sleep(self.config.rate_limit.min_interval_seconds - time_since_last)

    def _get_cache_key(self, params: dict[str, Any] | None = None) -> str:
        """Generate cache key for request parameters."""
        key_data = f"{self.source_id}:{params or {}}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def _get_cached(self, cache_key: str) -> list[DataPoint] | None:
        """Get cached data if valid."""
        if not self.config.cache.enabled:
            return None

        if cache_key in self._cache:
            timestamp, data = self._cache[cache_key]
            if time.time() - timestamp < self.config.cache.ttl_seconds:
                return data

            # Expired, remove
            del self._cache[cache_key]

        return None

    def _set_cached(self, cache_key: str, data: list[DataPoint]) -> None:
        """Store data in cache."""
        if not self.config.cache.enabled:
            return

        # Enforce max entries
        if len(self._cache) >= self.config.cache.max_entries:
            # Remove oldest entry
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][0])
            del self._cache[oldest_key]

        self._cache[cache_key] = (time.time(), data)

    @abstractmethod
    async def _fetch_impl(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        **kwargs: Any,
    ) -> list[DataPoint]:
        """Implementation of data fetching logic.

        Subclasses must implement this method.

        Args:
            start_time: Optional start of time range
            end_time: Optional end of time range
            **kwargs: Source-specific parameters

        Returns:
            List of fetched data points
        """
        pass

    async def fetch(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> FetchResult:
        """Fetch data from the source.

        Args:
            start_time: Optional start of time range
            end_time: Optional end of time range
            use_cache: Whether to use cached data if available
            **kwargs: Source-specific parameters

        Returns:
            FetchResult with data points or error information
        """
        fetch_start = time.time()

        # Check cache
        cache_key = self._get_cache_key(
            {"start": str(start_time), "end": str(end_time), **kwargs}
        )

        if use_cache:
            cached = self._get_cached(cache_key)
            if cached is not None:
                logger.debug(f"{self.source_id}: Returning {len(cached)} cached data points")
                return FetchResult(
                    success=True,
                    data_points=cached,
                    cached=True,
                    fetch_time_ms=(time.time() - fetch_start) * 1000,
                )

        try:
            data_points = await self._fetch_impl(start_time, end_time, **kwargs)
            self._set_cached(cache_key, data_points)

            logger.info(f"{self.source_id}: Fetched {len(data_points)} data points")
            return FetchResult(
                success=True,
                data_points=data_points,
                fetch_time_ms=(time.time() - fetch_start) * 1000,
            )

        except DataSourceError as e:
            logger.warning(f"{self.source_id}: Fetch failed: {e}")
            return FetchResult(
                success=False,
                error=str(e),
                fetch_time_ms=(time.time() - fetch_start) * 1000,
                rate_limited="Rate limit" in str(e),
            )

        except Exception as e:
            logger.error(f"{self.source_id}: Unexpected error: {e}")
            return FetchResult(
                success=False,
                error=f"Unexpected error: {e}",
                fetch_time_ms=(time.time() - fetch_start) * 1000,
            )

    def fetch_sync(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> FetchResult:
        """Synchronous version of fetch."""
        return asyncio.get_event_loop().run_until_complete(
            self.fetch(start_time, end_time, use_cache, **kwargs)
        )

    def get_metrics(self) -> dict[str, Any]:
        """Get source metrics."""
        return {
            "source_id": self.source_id,
            "total_requests": self._total_requests,
            "successful_requests": self._successful_requests,
            "failed_requests": self._failed_requests,
            "success_rate": (
                self._successful_requests / self._total_requests
                if self._total_requests > 0
                else 0.0
            ),
            "avg_latency_ms": (
                self._total_latency_ms / self._successful_requests
                if self._successful_requests > 0
                else 0.0
            ),
            "cache_entries": len(self._cache),
            "requests_this_hour": self._request_count_hour,
        }

    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._cache.clear()

    async def close(self) -> None:
        """Close HTTP clients and release resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None

    def __del__(self) -> None:
        """Cleanup on deletion."""
        if self._sync_client:
            self._sync_client.close()


class DataSourceManager:
    """Unified manager for multiple data sources.

    Provides:
    - Registration and lifecycle management for data sources
    - Concurrent fetching from multiple sources
    - Aggregation and filtering of data points
    - Cross-source correlation support

    Example:
        >>> manager = DataSourceManager()
        >>> manager.register_source(NASADONKISource(api_key="key"))
        >>> manager.register_source(USGSEarthquakeSource())
        >>> results = await manager.fetch_all()
        >>> for result in results.values():
        ...     if result.success:
        ...         process(result.data_points)
    """

    def __init__(self) -> None:
        """Initialize the data source manager."""
        self._sources: dict[str, DataSourceBase] = {}
        self._enabled_sources: set[str] = set()

    def register_source(
        self,
        source: DataSourceBase,
        enabled: bool = True,
    ) -> None:
        """Register a data source.

        Args:
            source: Data source instance
            enabled: Whether to enable the source immediately
        """
        source_id = source.source_id
        self._sources[source_id] = source

        if enabled:
            self._enabled_sources.add(source_id)

        logger.info(
            f"Registered data source: {source_id} "
            f"(enabled={enabled}, types={[t.value for t in source.default_source_types]})"
        )

    def unregister_source(self, source_id: str) -> None:
        """Unregister a data source."""
        if source_id in self._sources:
            del self._sources[source_id]
            self._enabled_sources.discard(source_id)

    def enable_source(self, source_id: str) -> None:
        """Enable a registered source."""
        if source_id in self._sources:
            self._enabled_sources.add(source_id)

    def disable_source(self, source_id: str) -> None:
        """Disable a source without unregistering."""
        self._enabled_sources.discard(source_id)

    def get_source(self, source_id: str) -> DataSourceBase | None:
        """Get a source by ID."""
        return self._sources.get(source_id)

    def list_sources(self) -> list[str]:
        """List all registered source IDs."""
        return list(self._sources.keys())

    def list_enabled_sources(self) -> list[str]:
        """List enabled source IDs."""
        return list(self._enabled_sources)

    async def fetch_source(
        self,
        source_id: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        **kwargs: Any,
    ) -> FetchResult:
        """Fetch from a specific source.

        Args:
            source_id: Source identifier
            start_time: Optional start time
            end_time: Optional end time
            **kwargs: Source-specific parameters

        Returns:
            FetchResult from the source
        """
        source = self._sources.get(source_id)
        if source is None:
            return FetchResult(success=False, error=f"Unknown source: {source_id}")

        return await source.fetch(start_time, end_time, **kwargs)

    async def fetch_all(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        source_types: list[DataSourceType] | None = None,
        **kwargs: Any,
    ) -> dict[str, FetchResult]:
        """Fetch from all enabled sources concurrently.

        Args:
            start_time: Optional start time
            end_time: Optional end time
            source_types: Optional filter by source types
            **kwargs: Source-specific parameters

        Returns:
            Dictionary mapping source_id to FetchResult
        """
        # Filter sources
        sources_to_fetch: list[DataSourceBase] = []
        for source_id in self._enabled_sources:
            source = self._sources[source_id]

            if source_types:
                # Check if source produces any of the requested types
                if not any(t in source_types for t in source.default_source_types):
                    continue

            sources_to_fetch.append(source)

        if not sources_to_fetch:
            return {}

        # Fetch concurrently
        tasks = [source.fetch(start_time, end_time, **kwargs) for source in sources_to_fetch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build result dictionary
        result_dict: dict[str, FetchResult] = {}
        for source, result in zip(sources_to_fetch, results, strict=False):
            if isinstance(result, Exception):
                result_dict[source.source_id] = FetchResult(
                    success=False, error=str(result)
                )
            else:
                result_dict[source.source_id] = result

        return result_dict

    def fetch_all_sync(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        source_types: list[DataSourceType] | None = None,
        **kwargs: Any,
    ) -> dict[str, FetchResult]:
        """Synchronous version of fetch_all."""
        return asyncio.get_event_loop().run_until_complete(
            self.fetch_all(start_time, end_time, source_types, **kwargs)
        )

    def get_all_data_points(
        self,
        results: dict[str, FetchResult],
        filter_types: list[DataSourceType] | None = None,
        min_confidence: float = 0.0,
    ) -> list[DataPoint]:
        """Extract and filter data points from fetch results.

        Args:
            results: Results from fetch_all
            filter_types: Optional type filter
            min_confidence: Minimum confidence threshold

        Returns:
            Combined list of data points
        """
        all_points: list[DataPoint] = []

        for result in results.values():
            if not result.success:
                continue

            for point in result.data_points:
                if filter_types and point.source_type not in filter_types:
                    continue
                if point.confidence < min_confidence:
                    continue
                all_points.append(point)

        # Sort by timestamp
        all_points.sort(key=lambda p: p.timestamp, reverse=True)
        return all_points

    def get_metrics(self) -> dict[str, Any]:
        """Get aggregated metrics from all sources."""
        metrics: dict[str, Any] = {
            "total_sources": len(self._sources),
            "enabled_sources": len(self._enabled_sources),
            "sources": {},
        }

        for source_id, source in self._sources.items():
            metrics["sources"][source_id] = source.get_metrics()

        return metrics

    async def close_all(self) -> None:
        """Close all data sources."""
        for source in self._sources.values():
            await source.close()

    def clear_all_caches(self) -> None:
        """Clear caches for all sources."""
        for source in self._sources.values():
            source.clear_cache()
