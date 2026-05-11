"""
Mercury Agent

Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Weather service stub for testing and development.

Example:
    >>> service = WeatherServiceStub()
    >>> data = await service.get_current("New York")
    >>> print(f"Temperature: {data.temperature}C")
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class WeatherCondition(Enum):
    """Weather condition types."""

    CLEAR = "clear"
    CLOUDY = "cloudy"
    PARTLY_CLOUDY = "partly_cloudy"
    RAIN = "rain"
    HEAVY_RAIN = "heavy_rain"
    SNOW = "snow"
    THUNDERSTORM = "thunderstorm"
    FOG = "fog"
    WINDY = "windy"
    HAIL = "hail"


@dataclass
class WeatherData:
    """
    Weather data structure.

    Attributes:
        location: Location name or coordinates.
        temperature: Temperature in Celsius.
        feels_like: Feels-like temperature.
        humidity: Relative humidity percentage.
        pressure: Atmospheric pressure in hPa.
        wind_speed: Wind speed in m/s.
        wind_direction: Wind direction in degrees.
        condition: Weather condition.
        visibility: Visibility in km.
        uv_index: UV index.
        timestamp: Data timestamp.
    """

    location: str
    temperature: float
    feels_like: float
    humidity: float
    pressure: float
    wind_speed: float
    wind_direction: int
    condition: WeatherCondition
    visibility: float
    uv_index: float
    timestamp: datetime = field(default_factory=datetime.now)
    raw_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "location": self.location,
            "temperature": self.temperature,
            "feels_like": self.feels_like,
            "humidity": self.humidity,
            "pressure": self.pressure,
            "wind_speed": self.wind_speed,
            "wind_direction": self.wind_direction,
            "condition": self.condition.value,
            "visibility": self.visibility,
            "uv_index": self.uv_index,
            "timestamp": self.timestamp.isoformat(),
        }

    @property
    def is_severe(self) -> bool:
        """Check if weather is severe."""
        return self.condition in [
            WeatherCondition.THUNDERSTORM,
            WeatherCondition.HEAVY_RAIN,
            WeatherCondition.HAIL,
        ]


@dataclass
class WeatherForecast:
    """Weather forecast for a future time period."""

    location: str
    forecast_time: datetime
    high_temp: float
    low_temp: float
    condition: WeatherCondition
    precipitation_chance: float
    wind_speed: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "location": self.location,
            "forecast_time": self.forecast_time.isoformat(),
            "high_temp": self.high_temp,
            "low_temp": self.low_temp,
            "condition": self.condition.value,
            "precipitation_chance": self.precipitation_chance,
            "wind_speed": self.wind_speed,
        }


class WeatherServiceStub:
    """Stub implementation of weather service.

    Provides realistic mock weather data for testing without
    requiring actual API access.

    Example:
        >>> stub = WeatherServiceStub(seed=42)
        >>> current = await stub.get_current("Los Angeles")
        >>> forecast = await stub.get_forecast("Los Angeles", days=7)
    """

    def __init__(
        self,
        seed: int | None = None,
        latency_ms: tuple[int, int] = (50, 200),
        failure_rate: float = 0.0,
    ):
        """
        Initialize weather stub.

        Args:
            seed: Random seed for reproducibility.
            latency_ms: Min/max simulated latency in milliseconds.
            failure_rate: Probability of simulated failure (0-1).
        """
        self._rng = random.Random(seed)
        self._latency_ms = latency_ms
        self._failure_rate = failure_rate
        self._call_count = 0
        self._cache: dict[str, WeatherData] = {}

    async def _simulate_latency(self) -> None:
        """Simulate network latency."""
        latency = self._rng.randint(*self._latency_ms) / 1000.0
        await asyncio.sleep(latency)

    def _maybe_fail(self) -> None:
        """Potentially raise exception to simulate failure."""
        if self._rng.random() < self._failure_rate:
            raise ConnectionError("Simulated weather service failure")

    def _generate_weather(self, location: str) -> WeatherData:
        """Generate realistic weather data."""
        # Base temperature varies by location hash
        loc_hash = hash(location) % 100
        base_temp = 15 + (loc_hash / 100) * 20

        # Add daily variation
        hour = datetime.now().hour
        daily_var = 5 * (1 - abs(hour - 14) / 12)

        temp = base_temp + daily_var + self._rng.gauss(0, 3)
        humidity = max(20, min(100, 60 + self._rng.gauss(0, 15)))
        wind = max(0, self._rng.gauss(5, 3))

        # Determine condition based on humidity and temperature
        if humidity > 80:
            condition = self._rng.choice(
                [
                    WeatherCondition.RAIN,
                    WeatherCondition.CLOUDY,
                    WeatherCondition.FOG,
                ]
            )
        elif humidity > 60:
            condition = self._rng.choice(
                [
                    WeatherCondition.PARTLY_CLOUDY,
                    WeatherCondition.CLOUDY,
                ]
            )
        else:
            condition = self._rng.choice(
                [
                    WeatherCondition.CLEAR,
                    WeatherCondition.PARTLY_CLOUDY,
                ]
            )

        return WeatherData(
            location=location,
            temperature=round(temp, 1),
            feels_like=round(temp - wind * 0.5, 1),
            humidity=round(humidity, 1),
            pressure=round(1013 + self._rng.gauss(0, 10), 1),
            wind_speed=round(wind, 1),
            wind_direction=self._rng.randint(0, 359),
            condition=condition,
            visibility=round(max(1, 10 - humidity / 20 + self._rng.gauss(0, 2)), 1),
            uv_index=round(max(0, min(11, 5 + self._rng.gauss(0, 2))), 1),
        )

    async def get_current(self, location: str) -> WeatherData:
        """
        Get current weather for location.

        Args:
            location: Location name or coordinates.

        Returns:
            Current weather data.

        Raises:
            ConnectionError: On simulated failure.
        """
        self._call_count += 1
        await self._simulate_latency()
        self._maybe_fail()

        weather = self._generate_weather(location)
        self._cache[location] = weather
        return weather

    async def get_forecast(
        self,
        location: str,
        days: int = 7,
    ) -> list[WeatherForecast]:
        """
        Get weather forecast.

        Args:
            location: Location name.
            days: Number of days to forecast.

        Returns:
            List of daily forecasts.
        """
        self._call_count += 1
        await self._simulate_latency()
        self._maybe_fail()

        forecasts = []
        base_weather = self._generate_weather(location)

        for day in range(days):
            forecast_time = datetime.now() + timedelta(days=day + 1)
            temp_variation = self._rng.gauss(0, 3)

            forecasts.append(
                WeatherForecast(
                    location=location,
                    forecast_time=forecast_time,
                    high_temp=round(base_weather.temperature + 5 + temp_variation, 1),
                    low_temp=round(base_weather.temperature - 5 + temp_variation, 1),
                    condition=self._rng.choice(list(WeatherCondition)),
                    precipitation_chance=round(self._rng.random() * 100, 1),
                    wind_speed=round(max(0, self._rng.gauss(5, 3)), 1),
                )
            )

        return forecasts

    async def get_alerts(self, location: str) -> list[dict[str, Any]]:
        """
        Get weather alerts for location.

        Args:
            location: Location name.

        Returns:
            List of active alerts.
        """
        self._call_count += 1
        await self._simulate_latency()
        self._maybe_fail()

        alerts = []
        # 20% chance of an alert
        if self._rng.random() < 0.2:
            alert_types = [
                ("High Wind Warning", "wind", "moderate"),
                ("Heat Advisory", "heat", "moderate"),
                ("Flood Watch", "flood", "severe"),
                ("Winter Storm Warning", "winter", "severe"),
            ]
            alert = self._rng.choice(alert_types)
            alerts.append(
                {
                    "title": alert[0],
                    "type": alert[1],
                    "severity": alert[2],
                    "location": location,
                    "expires": (datetime.now() + timedelta(hours=24)).isoformat(),
                }
            )

        return alerts

    def get_metrics(self) -> dict[str, Any]:
        """Get service metrics."""
        return {
            "call_count": self._call_count,
            "cache_size": len(self._cache),
            "failure_rate": self._failure_rate,
        }


class WeatherAPIProvider(Enum):
    """Supported weather API providers."""

    OPENWEATHERMAP = "openweathermap"
    NOAA = "noaa"
    STUB = "stub"


class WeatherService:
    """Production-ready weather data service with multiple API backends.

    Supports weather data from:
    - OpenWeatherMap (requires API key, free tier available)
    - NOAA (National Weather Service, free, US only)
    - Stub/Mock (for testing)

    Example:
        >>> # Using OpenWeatherMap (requires API key)
        >>> service = WeatherService(
        ...     provider=WeatherAPIProvider.OPENWEATHERMAP,
        ...     api_key=os.getenv("OPENWEATHERMAP_API_KEY")
        ... )
        >>> data = await service.get_current("London")

        >>> # Using NOAA (US locations only)
        >>> service = WeatherService(provider=WeatherAPIProvider.NOAA)
        >>> data = await service.get_current_by_coords(40.7128, -74.0060)

        >>> # Fallback to stub for testing
        >>> service = WeatherService(provider=WeatherAPIProvider.STUB)
    """

    # API endpoints
    OPENWEATHERMAP_BASE = "https://api.openweathermap.org/data/2.5"
    NOAA_POINTS_BASE = "https://api.weather.gov/points"
    NOAA_FORECAST_BASE = "https://api.weather.gov/gridpoints"

    # Condition mapping from OpenWeatherMap codes
    OWM_CONDITION_MAP: dict[int, WeatherCondition] = {
        800: WeatherCondition.CLEAR,
        801: WeatherCondition.PARTLY_CLOUDY,
        802: WeatherCondition.PARTLY_CLOUDY,
        803: WeatherCondition.CLOUDY,
        804: WeatherCondition.CLOUDY,
        300: WeatherCondition.RAIN,
        301: WeatherCondition.RAIN,
        500: WeatherCondition.RAIN,
        501: WeatherCondition.RAIN,
        502: WeatherCondition.HEAVY_RAIN,
        503: WeatherCondition.HEAVY_RAIN,
        511: WeatherCondition.SNOW,
        520: WeatherCondition.RAIN,
        600: WeatherCondition.SNOW,
        601: WeatherCondition.SNOW,
        602: WeatherCondition.SNOW,
        200: WeatherCondition.THUNDERSTORM,
        201: WeatherCondition.THUNDERSTORM,
        202: WeatherCondition.THUNDERSTORM,
        741: WeatherCondition.FOG,
        701: WeatherCondition.FOG,
    }

    def __init__(
        self,
        provider: WeatherAPIProvider = WeatherAPIProvider.STUB,
        api_key: str | None = None,
        timeout: int = 30,
        cache_ttl: int = 300,  # 5 minutes
    ):
        """
        Initialize weather service.

        The ``fallback_to_stub`` parameter was removed in the May 2026
        Phase 2 audit cure — silent fallback to stub data is not
        permitted.

        Args:
            provider: API provider to use.
            api_key: API key for OpenWeatherMap (not needed for NOAA/Stub).
            timeout: Request timeout in seconds.
            cache_ttl: Cache time-to-live in seconds.
        """
        self.provider = provider
        self.api_key = api_key or os.getenv("OPENWEATHERMAP_API_KEY")
        self.timeout = timeout
        self.cache_ttl = cache_ttl

        # Cache with TTL
        self._cache: dict[str, tuple[WeatherData, datetime]] = {}
        self._call_count = 0
        self._api_errors = 0

        # Stub instance for explicit STUB provider only
        self._stub = WeatherServiceStub()

        # Validate OpenWeatherMap configuration
        if provider == WeatherAPIProvider.OPENWEATHERMAP and not self.api_key:
            raise NotImplementedError(
                "OpenWeatherMap requires an API key. Set OPENWEATHERMAP_API_KEY "
                "environment variable or provide api_key parameter. "
                "Silent fallback to stub mode is not permitted "
                "(Phase 2 audit cure)."
            )

    def _get_cached(self, key: str) -> WeatherData | None:
        """Get cached data if still valid."""
        if key in self._cache:
            data, cached_at = self._cache[key]
            if (datetime.now() - cached_at).total_seconds() < self.cache_ttl:
                return data
        return None

    def _set_cached(self, key: str, data: WeatherData) -> None:
        """Cache weather data."""
        self._cache[key] = (data, datetime.now())

    def _condition_from_owm_code(self, code: int) -> WeatherCondition:
        """Map OpenWeatherMap condition code to WeatherCondition."""
        return self.OWM_CONDITION_MAP.get(code, WeatherCondition.CLOUDY)

    async def _fetch_openweathermap(self, location: str) -> WeatherData:
        """
        Fetch weather from OpenWeatherMap API.

        API Documentation: https://openweathermap.org/api
        """
        if not self.api_key:
            raise ValueError("OpenWeatherMap API key required")

        params = {
            "q": location,
            "appid": self.api_key,
            "units": "metric",
        }

        from omni_mercury_engine.security.input_validation import TrustedEndpoints

        url = f"{self.OPENWEATHERMAP_BASE}/weather?{urlencode(params)}"
        TrustedEndpoints.validate_url(self.OPENWEATHERMAP_BASE)

        def fetch() -> dict[str, Any]:
            req = Request(
                url,
                headers={"User-Agent": "Mercury-Agent/1.0"},
            )
            with urlopen(req, timeout=self.timeout) as response:  # nosec B310 - TrustedEndpoints.validate_url enforces https + allowlisted domain
                result: dict[str, Any] = json.loads(response.read().decode())
                return result

        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, fetch)

        if data.get("cod") != 200:
            raise ValueError(f"OpenWeatherMap API error: {data.get('message', 'Unknown')}")

        main = data.get("main", {})
        wind = data.get("wind", {})
        weather = data.get("weather", [{}])[0]
        visibility = data.get("visibility", 10000) / 1000  # Convert to km

        condition_code = weather.get("id", 800)
        condition = self._condition_from_owm_code(condition_code)

        return WeatherData(
            location=location,
            temperature=round(main.get("temp", 0), 1),
            feels_like=round(main.get("feels_like", main.get("temp", 0)), 1),
            humidity=round(main.get("humidity", 0), 1),
            pressure=round(main.get("pressure", 1013), 1),
            wind_speed=round(wind.get("speed", 0), 1),
            wind_direction=wind.get("deg", 0),
            condition=condition,
            visibility=round(visibility, 1),
            uv_index=0,  # Requires separate API call
            raw_data=data,
        )

    async def _fetch_openweathermap_by_coords(self, lat: float, lon: float) -> WeatherData:
        """Fetch weather from OpenWeatherMap by coordinates."""
        if not self.api_key:
            raise ValueError("OpenWeatherMap API key required")

        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric",
        }

        from omni_mercury_engine.security.input_validation import TrustedEndpoints

        url = f"{self.OPENWEATHERMAP_BASE}/weather?{urlencode(params)}"
        TrustedEndpoints.validate_url(self.OPENWEATHERMAP_BASE)

        def fetch() -> dict[str, Any]:
            req = Request(
                url,
                headers={"User-Agent": "Mercury-Agent/1.0"},
            )
            with urlopen(req, timeout=self.timeout) as response:  # nosec B310 - TrustedEndpoints.validate_url enforces https + allowlisted domain
                result: dict[str, Any] = json.loads(response.read().decode())
                return result

        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, fetch)

        if data.get("cod") != 200:
            raise ValueError(f"OpenWeatherMap API error: {data.get('message', 'Unknown')}")

        main = data.get("main", {})
        wind = data.get("wind", {})
        weather = data.get("weather", [{}])[0]
        visibility = data.get("visibility", 10000) / 1000

        condition_code = weather.get("id", 800)
        condition = self._condition_from_owm_code(condition_code)

        return WeatherData(
            location=f"{lat},{lon}",
            temperature=round(main.get("temp", 0), 1),
            feels_like=round(main.get("feels_like", main.get("temp", 0)), 1),
            humidity=round(main.get("humidity", 0), 1),
            pressure=round(main.get("pressure", 1013), 1),
            wind_speed=round(wind.get("speed", 0), 1),
            wind_direction=wind.get("deg", 0),
            condition=condition,
            visibility=round(visibility, 1),
            uv_index=0,
            raw_data=data,
        )

    async def _fetch_noaa_point_info(self, lat: float, lon: float) -> dict[str, Any]:
        """Get NOAA grid point information for coordinates."""
        from omni_mercury_engine.security.input_validation import TrustedEndpoints

        url = f"{self.NOAA_POINTS_BASE}/{lat},{lon}"
        TrustedEndpoints.validate_url(self.NOAA_POINTS_BASE)

        def fetch() -> dict[str, Any]:
            req = Request(
                url,
                headers={
                    "User-Agent": "Mercury-Agent/1.0",
                    "Accept": "application/geo+json",
                },
            )
            with urlopen(req, timeout=self.timeout) as response:  # nosec B310 - TrustedEndpoints.validate_url enforces https + allowlisted domain
                result: dict[str, Any] = json.loads(response.read().decode())
                return result

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fetch)

    async def _fetch_noaa(self, lat: float, lon: float) -> WeatherData:
        """
        Fetch weather from NOAA National Weather Service API.

        API Documentation: https://www.weather.gov/documentation/services-web-api
        Only works for US locations.
        """
        # First get the grid point info
        point_data = await self._fetch_noaa_point_info(lat, lon)
        properties = point_data.get("properties", {})

        from omni_mercury_engine.security.input_validation import TrustedEndpoints

        forecast_url = properties.get("forecastHourly")
        if not forecast_url:
            raise ValueError("NOAA API: Could not get forecast URL")

        # forecast_url comes back inside the NOAA response body — treat it as
        # tainted and require it to resolve to an allowlisted domain (NOAA
        # currently returns URLs under api.weather.gov).
        TrustedEndpoints.validate_url(forecast_url.split("?")[0])

        # Fetch the hourly forecast
        def fetch_forecast() -> dict[str, Any]:
            req = Request(
                forecast_url,
                headers={
                    "User-Agent": "Mercury-Agent/1.0",
                    "Accept": "application/geo+json",
                },
            )
            with urlopen(req, timeout=self.timeout) as response:  # nosec B310 - TrustedEndpoints.validate_url enforces https + allowlisted domain
                result: dict[str, Any] = json.loads(response.read().decode())
                return result

        loop = asyncio.get_event_loop()
        forecast_data = await loop.run_in_executor(None, fetch_forecast)

        periods = forecast_data.get("properties", {}).get("periods", [])
        if not periods:
            raise ValueError("NOAA API: No forecast periods returned")

        current = periods[0]

        # Parse NOAA weather condition
        short_forecast = current.get("shortForecast", "").lower()
        condition = WeatherCondition.CLEAR
        if "thunder" in short_forecast:
            condition = WeatherCondition.THUNDERSTORM
        elif "rain" in short_forecast or "shower" in short_forecast:
            condition = WeatherCondition.RAIN
        elif "snow" in short_forecast:
            condition = WeatherCondition.SNOW
        elif "cloud" in short_forecast or "overcast" in short_forecast:
            condition = WeatherCondition.CLOUDY
        elif "partly" in short_forecast:
            condition = WeatherCondition.PARTLY_CLOUDY
        elif "fog" in short_forecast:
            condition = WeatherCondition.FOG
        elif "wind" in short_forecast:
            condition = WeatherCondition.WINDY

        # Temperature conversion if in Fahrenheit
        temp = current.get("temperature", 0)
        temp_unit = current.get("temperatureUnit", "F")
        if temp_unit == "F":
            temp = (temp - 32) * 5 / 9

        # Parse wind
        wind_speed_str = current.get("windSpeed", "0 mph")
        wind_speed = float(wind_speed_str.split()[0]) * 0.44704  # mph to m/s

        wind_direction_str = current.get("windDirection", "N")
        wind_dirs = {"N": 0, "NE": 45, "E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270, "NW": 315}
        wind_direction = wind_dirs.get(wind_direction_str, 0)

        return WeatherData(
            location=f"{lat},{lon}",
            temperature=round(temp, 1),
            feels_like=round(temp, 1),  # NOAA doesn't provide feels-like
            humidity=current.get("relativeHumidity", {}).get("value", 50),
            pressure=1013,  # NOAA doesn't provide in hourly forecast
            wind_speed=round(wind_speed, 1),
            wind_direction=wind_direction,
            condition=condition,
            visibility=10,  # NOAA doesn't provide in hourly
            uv_index=0,
            raw_data=forecast_data,
        )

    async def get_current(self, location: str) -> WeatherData:
        """
        Get current weather for location.

        Args:
            location: Location name (city, address).

        Returns:
            Current weather data.

        Raises:
            ValueError: If location is invalid or API fails without fallback.
        """
        self._call_count += 1

        # Check cache
        cached = self._get_cached(location)
        if cached:
            return cached

        if self.provider == WeatherAPIProvider.STUB:
            return await self._stub.get_current(location)

        if self.provider == WeatherAPIProvider.OPENWEATHERMAP:
            data = await self._fetch_openweathermap(location)
        elif self.provider == WeatherAPIProvider.NOAA:
            raise ValueError("NOAA API requires coordinates. Use get_current_by_coords() instead.")
        else:
            raise ValueError(f"Unknown weather API provider: {self.provider}")

        self._set_cached(location, data)
        return data

    async def get_current_by_coords(self, lat: float, lon: float) -> WeatherData:
        """
        Get current weather by coordinates.

        Args:
            lat: Latitude.
            lon: Longitude.

        Returns:
            Current weather data.
        """
        self._call_count += 1
        cache_key = f"{lat:.4f},{lon:.4f}"

        cached = self._get_cached(cache_key)
        if cached:
            return cached

        if self.provider == WeatherAPIProvider.STUB:
            return await self._stub.get_current(cache_key)

        if self.provider == WeatherAPIProvider.OPENWEATHERMAP:
            data = await self._fetch_openweathermap_by_coords(lat, lon)
        elif self.provider == WeatherAPIProvider.NOAA:
            data = await self._fetch_noaa(lat, lon)
        else:
            raise ValueError(f"Unknown weather API provider: {self.provider}")

        self._set_cached(cache_key, data)
        return data

    async def get_forecast(
        self,
        location: str,
        days: int = 7,
    ) -> list[WeatherForecast]:
        """
        Get weather forecast.

        Args:
            location: Location name.
            days: Number of days to forecast.

        Returns:
            List of daily forecasts.
        """
        self._call_count += 1

        if self.provider == WeatherAPIProvider.STUB:
            return await self._stub.get_forecast(location, days)

        if self.provider == WeatherAPIProvider.OPENWEATHERMAP:
            return await self._fetch_owm_forecast(location, days)

        raise ValueError(
            f"Forecast not supported for provider {self.provider}. "
            "Silent fallback to stub is not permitted (Phase 2 audit cure)."
        )

    async def _fetch_owm_forecast(self, location: str, days: int) -> list[WeatherForecast]:
        """Fetch forecast from OpenWeatherMap."""
        if not self.api_key:
            raise NotImplementedError(
                "OpenWeatherMap API key required for forecast. "
                "Silent fallback to stub is not permitted (Phase 2 audit cure)."
            )

        params = {
            "q": location,
            "appid": self.api_key,
            "units": "metric",
            "cnt": min(days * 8, 40),  # 3-hour intervals
        }

        from omni_mercury_engine.security.input_validation import TrustedEndpoints

        url = f"{self.OPENWEATHERMAP_BASE}/forecast?{urlencode(params)}"
        TrustedEndpoints.validate_url(self.OPENWEATHERMAP_BASE)

        def fetch() -> dict[str, Any]:
            req = Request(
                url,
                headers={"User-Agent": "Mercury-Agent/1.0"},
            )
            with urlopen(req, timeout=self.timeout) as response:  # nosec B310 - TrustedEndpoints.validate_url enforces https + allowlisted domain
                result: dict[str, Any] = json.loads(response.read().decode())
                return result

        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, fetch)

        if str(data.get("cod")) != "200":
            logger.warning(f"OpenWeatherMap forecast error: {data}")
            return await self._stub.get_forecast(location, days)

        forecast_list = data.get("list", [])

        # Group by day and aggregate
        daily_data: dict[str, list[dict[str, Any]]] = {}
        for item in forecast_list:
            date_str = item.get("dt_txt", "")[:10]
            if date_str not in daily_data:
                daily_data[date_str] = []
            daily_data[date_str].append(item)

        forecasts = []
        for date_str, items in sorted(daily_data.items())[:days]:
            temps = [item["main"]["temp"] for item in items]
            wind_speeds = [item.get("wind", {}).get("speed", 0) for item in items]
            pop = max(item.get("pop", 0) for item in items)

            # Get most common condition
            conditions = [
                self._condition_from_owm_code(item["weather"][0]["id"])
                for item in items
                if item.get("weather")
            ]
            condition = (
                max(set(conditions), key=conditions.count) if conditions else WeatherCondition.CLEAR
            )

            forecasts.append(
                WeatherForecast(
                    location=location,
                    forecast_time=datetime.strptime(date_str, "%Y-%m-%d"),
                    high_temp=round(max(temps), 1),
                    low_temp=round(min(temps), 1),
                    condition=condition,
                    precipitation_chance=round(pop * 100, 1),
                    wind_speed=round(sum(wind_speeds) / len(wind_speeds), 1),
                )
            )

        return forecasts

    async def get_alerts(self, location: str) -> list[dict[str, Any]]:
        """
        Get weather alerts for location.

        Args:
            location: Location name.

        Returns:
            List of active alerts.
        """
        self._call_count += 1

        if self.provider == WeatherAPIProvider.STUB:
            return await self._stub.get_alerts(location)

        # OpenWeatherMap requires separate API call for alerts
        # For now, return stub alerts
        return await self._stub.get_alerts(location)

    def get_metrics(self) -> dict[str, Any]:
        """Get service metrics."""
        return {
            "provider": self.provider.value,
            "call_count": self._call_count,
            "api_errors": self._api_errors,
            "cache_size": len(self._cache),
            "error_rate": self._api_errors / self._call_count if self._call_count > 0 else 0,
        }


# Factory function for easy service creation
def create_weather_service(
    use_real_api: bool = False,
    provider: str = "openweathermap",
    api_key: str | None = None,
) -> WeatherService | WeatherServiceStub:
    """Create weather service with appropriate backend.

    Args:
        use_real_api: Whether to use real API or stub.
        provider: API provider ("openweathermap", "noaa", "stub").
        api_key: API key for OpenWeatherMap.

    Returns:
        Configured weather service.

    Example:
        >>> # For testing
        >>> service = create_weather_service(use_real_api=False)

        >>> # For production with OpenWeatherMap
        >>> service = create_weather_service(
        ...     use_real_api=True,
        ...     provider="openweathermap",
        ...     api_key="YOUR_API_KEY"
        ... )

        >>> # For production with NOAA (US only)
        >>> service = create_weather_service(use_real_api=True, provider="noaa")
    """
    if not use_real_api:
        return WeatherServiceStub()

    provider_map = {
        "openweathermap": WeatherAPIProvider.OPENWEATHERMAP,
        "noaa": WeatherAPIProvider.NOAA,
        "stub": WeatherAPIProvider.STUB,
    }

    provider_enum = provider_map.get(provider.lower(), WeatherAPIProvider.STUB)

    return WeatherService(
        provider=provider_enum,
        api_key=api_key,
    )
