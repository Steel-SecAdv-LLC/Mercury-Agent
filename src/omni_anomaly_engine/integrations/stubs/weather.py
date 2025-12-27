"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

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
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


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
    """Weather data structure.

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
        """Initialize weather stub.

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
        """Get current weather for location.

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
        """Get weather forecast.

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
        """Get weather alerts for location.

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
