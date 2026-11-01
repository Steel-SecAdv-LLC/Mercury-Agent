# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""OpenWeatherMap and NOAA report an absent field as ``None``, never a constant.

The Open-Meteo change made every numeric ``WeatherData`` field nullable and
removed NOAA's invented pressure/visibility/UV, but two parses still filled
gaps with plausible numbers:

* OpenWeatherMap: absent ``temp``/``humidity``/``speed``/``deg`` became 0,
  absent ``pressure`` became 1013 hPa, absent ``visibility`` became 10 km,
  UV was always 0, and an absent condition code became 800 ("clear sky").
* NOAA: an absent ``temperature`` became 0 F (reported as -17.8 C), an absent
  ``windSpeed`` became calm, and the 8-point direction table sent every
  intermediate compass point the NWS reports (NNE, ENE, ..., NNW) to north.

Offline: the HTTP client is replaced with canned bodies.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from omni_mercury_engine.integrations.stubs import weather as weather_mod
from omni_mercury_engine.integrations.stubs.weather import (
    WeatherAPIProvider,
    WeatherCondition,
    WeatherService,
    _parse_nws_wind_speed,
)


def _owm_service() -> WeatherService:
    return WeatherService(provider=WeatherAPIProvider.OPENWEATHERMAP, api_key="k")


class TestOpenWeatherMapAbsentFields:
    """A sparse body yields ``None`` for what it does not carry."""

    def test_sparse_body_invents_nothing(self) -> None:
        body: dict[str, Any] = {"cod": 200, "main": {}, "wind": {}, "weather": [{}]}
        data = _owm_service()._parse_openweathermap(body, "X")
        assert data.temperature is None
        assert data.feels_like is None
        assert data.humidity is None
        assert data.pressure is None  # was 1013
        assert data.wind_speed is None  # was 0
        assert data.wind_direction is None  # was 0
        assert data.visibility is None  # was 10 km
        assert data.uv_index is None  # was 0
        assert data.condition is WeatherCondition.CLOUDY  # was CLEAR (code 800)

    def test_full_body_is_reported(self) -> None:
        body = {
            "cod": 200,
            "main": {"temp": 21.44, "feels_like": 20.0, "humidity": 55, "pressure": 1008},
            "wind": {"speed": 3.21, "deg": 270},
            "weather": [{"id": 500}],
            "visibility": 8000,
        }
        data = _owm_service()._parse_openweathermap(body, "X")
        assert (data.temperature, data.feels_like, data.humidity) == (21.4, 20.0, 55.0)
        assert (data.pressure, data.wind_speed, data.wind_direction) == (1008.0, 3.2, 270)
        assert data.visibility == 8.0
        assert data.uv_index is None
        assert data.condition is WeatherCondition.RAIN

    def test_non_numeric_values_are_not_readings(self) -> None:
        body = {"cod": 200, "main": {"temp": True, "humidity": "high"}, "wind": {}}
        data = _owm_service()._parse_openweathermap(body, "X")
        assert data.temperature is None
        assert data.humidity is None

    def test_fetch_routes_through_the_shared_parse(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            weather_mod.SafeHTTPClient,
            "get_json",
            staticmethod(lambda *a, **k: {"cod": 200, "main": {}, "weather": [{}]}),
        )
        service = _owm_service()
        by_name = asyncio.run(service._fetch_openweathermap("X"))
        by_coords = asyncio.run(service._fetch_openweathermap_by_coords(1.0, 2.0))
        for data in (by_name, by_coords):
            assert data.pressure is None and data.visibility is None
        assert by_coords.location == "1.0,2.0"


def _noaa_fetch(monkeypatch: pytest.MonkeyPatch, period: dict[str, Any]) -> Any:
    def fake_get_json(url: str, **_: Any) -> dict[str, Any]:
        if "points" in url:
            return {"properties": {"forecastHourly": "https://api.weather.gov/f"}}
        return {"properties": {"periods": [period]}}

    monkeypatch.setattr(weather_mod.SafeHTTPClient, "get_json", staticmethod(fake_get_json))
    service = WeatherService(provider=WeatherAPIProvider.NOAA)
    return asyncio.run(service._fetch_noaa(40.0, -75.0))


class TestNoaaAbsentFields:
    """NOAA's hourly period: absent stays absent, compass stays a compass."""

    def test_absent_temperature_and_wind_are_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        data = _noaa_fetch(monkeypatch, {"shortForecast": "Sunny"})
        assert data.temperature is None  # was -17.8 (0 F)
        assert data.wind_speed is None  # was 0 (calm)
        assert data.wind_direction is None  # was 0 (north)

    @pytest.mark.parametrize(
        ("direction", "degrees"),
        [("N", 0), ("NNE", 22), ("ENE", 68), ("SSW", 202), ("WSW", 248), ("NNW", 338)],
    )
    def test_sixteen_point_compass(
        self, monkeypatch: pytest.MonkeyPatch, direction: str, degrees: int
    ) -> None:
        period = {
            "temperature": 50,
            "temperatureUnit": "F",
            "windSpeed": "10 mph",
            "windDirection": direction,
        }
        data = _noaa_fetch(monkeypatch, period)
        assert data.wind_direction == degrees
        assert data.temperature == 10.0
        assert data.wind_speed == 4.5


class TestNwsWindSpeed:
    """``windSpeed`` strings as the NWS formats them."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("10 mph", 4.4704),
            ("10 to 15 mph", 6.7056),  # a range reports its upper bound
            ("18 km/h", 5.0),
            ("", None),
            ("calm", None),
            (None, None),
        ],
    )
    def test_parse(self, raw: Any, expected: float | None) -> None:
        got = _parse_nws_wind_speed(raw)
        if expected is None:
            assert got is None
        else:
            assert got == pytest.approx(expected)
