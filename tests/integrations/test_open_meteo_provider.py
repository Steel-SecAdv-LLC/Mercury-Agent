# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Open-Meteo serves by-name weather for real, and nobody invents readings.

``docs/ARCHITECTURE.md`` documented Open-Meteo as the default weather provider
before it existed. What the constructor actually defaulted to was
``WeatherAPIProvider.STUB``, so ``WeatherService().get_current("Boston")``
returned ``gauss``-drawn temperature and humidity -- a by-name lookup answered
with fabricated data under a real provider's documentation.

Open-Meteo is now implemented and is the default: keyless current conditions,
a keyless geocoder (the only one here that can turn a place name into
coordinates), and a daily forecast.

The second half of this file is the companion rule. Providers disagree about
what they carry -- the NWS hourly forecast has no pressure, visibility or UV
index, and Open-Meteo's ``current`` block has no visibility or UV -- and the
old dataclass forced those gaps to be filled: 1013 hPa, 10 km, UV 0, humidity
50, ``feels_like`` copied from temperature. A caller charting pressure from
NOAA was charting a constant. Every numeric field is nullable now, and these
tests pin that the absent ones are ``None`` rather than plausible.

The WMO mapping and nullability are offline. The live lookups are marked
``network``.
"""

from __future__ import annotations

import asyncio

import pytest

from omni_mercury_engine.integrations.stubs.weather import (
    WeatherAPIProvider,
    WeatherCondition,
    WeatherService,
)


class TestDefaultProvider:
    """The default must be real, keyless, and able to resolve a name."""

    def test_default_is_open_meteo(self) -> None:
        """Not the stub, and not a provider that cannot do by-name."""
        assert WeatherService().provider is WeatherAPIProvider.OPEN_METEO

    def test_open_meteo_is_a_declared_provider(self) -> None:
        assert WeatherAPIProvider.OPEN_METEO.value == "open_meteo"


class TestWmoCodeMapping:
    """WMO 4677 codes map to conditions without inventing optimism."""

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            (0, WeatherCondition.CLEAR),
            (1, WeatherCondition.PARTLY_CLOUDY),
            (2, WeatherCondition.PARTLY_CLOUDY),
            (3, WeatherCondition.CLOUDY),
            (45, WeatherCondition.FOG),
            (48, WeatherCondition.FOG),
            (61, WeatherCondition.RAIN),
            (65, WeatherCondition.HEAVY_RAIN),
            (71, WeatherCondition.SNOW),
            (86, WeatherCondition.SNOW),
            (95, WeatherCondition.THUNDERSTORM),
            (99, WeatherCondition.THUNDERSTORM),
        ],
    )
    def test_known_codes(self, code: int, expected: WeatherCondition) -> None:
        assert WeatherService._condition_from_wmo_code(code) is expected

    @pytest.mark.parametrize("code", [None, 7, 42, 123, -1])
    def test_unknown_code_is_cloudy_not_clear(self, code: int | None) -> None:
        """An unknown code must not read as fair weather.

        ``CLEAR`` is the one answer a fallback must never give: it turns "we
        do not recognise this code" into a benign-looking reading, and
        ``is_severe`` consumers would see calm. ``CLOUDY`` is the neutral
        member of the vocabulary.
        """
        assert WeatherService._condition_from_wmo_code(code) is WeatherCondition.CLOUDY


class TestNoFabricatedReadings:
    """A field the provider does not carry is ``None``, never a constant."""

    def test_weather_data_fields_are_nullable(self) -> None:
        """The dataclass permits "not reported" at the type level."""
        from typing import get_type_hints

        from omni_mercury_engine.integrations.stubs.weather import WeatherData

        hints = get_type_hints(WeatherData)
        for field in (
            "temperature",
            "feels_like",
            "humidity",
            "pressure",
            "wind_speed",
            "wind_direction",
            "visibility",
            "uv_index",
        ):
            assert "None" in str(hints[field]), f"{field} cannot express 'not reported'"

    def test_noaa_does_not_invent_the_fields_nws_omits(self) -> None:
        """The four constants are gone from the NOAA parse.

        Pinned on the source rather than a live call so the rule holds in the
        offline lane: the NWS hourly forecast carries temperature, humidity,
        wind and a text forecast, and nothing else this dataclass holds.
        """
        import inspect

        src = inspect.getsource(WeatherService._fetch_noaa)
        for invented in ("pressure=1013", "visibility=10", "uv_index=0"):
            assert invented not in src, f"NOAA still fabricates {invented}"
        assert "feels_like=None" in src, "NOAA should not copy temperature into feels_like"
        for honest in ("pressure=None", "visibility=None", "uv_index=None"):
            assert honest in src, f"NOAA should report {honest}"

    def test_open_meteo_does_not_invent_visibility_or_uv(self) -> None:
        """Open-Meteo's current block has neither; neither is filled in."""
        import inspect

        src = inspect.getsource(WeatherService._fetch_open_meteo)
        assert "visibility=None" in src
        assert "uv_index=None" in src


@pytest.mark.network
class TestOpenMeteoLive:
    """The route actually resolves names and returns real readings."""

    def test_by_name_lookup_returns_real_readings(self) -> None:
        service = WeatherService()
        data = asyncio.run(service.get_current("Boston"))

        assert "Boston" in data.location
        assert data.temperature is not None and -90.0 < data.temperature < 60.0
        assert data.humidity is not None and 0.0 <= data.humidity <= 100.0
        assert data.pressure is not None and 800.0 < data.pressure < 1100.0
        assert isinstance(data.condition, WeatherCondition)
        # Absent upstream, so absent here.
        assert data.visibility is None
        assert data.uv_index is None

    def test_unresolvable_name_raises_rather_than_guessing(self) -> None:
        """A name that does not geocode is an error, not a default coordinate."""
        service = WeatherService()
        with pytest.raises(ValueError, match="found no match"):
            asyncio.run(service.get_current("Xyzzyville Nowhere Qqqq"))

    def test_pressure_varies_by_location(self) -> None:
        """Proof the pressure is measured, not the 1013 constant NOAA used.

        Nairobi sits near 1,795 m, so its surface pressure is far below sea
        level's. Two locations agreeing to the decimal would mean a constant.
        """
        service = WeatherService()
        low = asyncio.run(service.get_current("Nairobi")).pressure
        sea = asyncio.run(service.get_current("Boston")).pressure

        assert low is not None and sea is not None
        assert low < sea - 50.0, f"expected altitude to separate these: {low} vs {sea}"

    def test_forecast_returns_the_requested_span(self) -> None:
        service = WeatherService()
        days = asyncio.run(service.get_forecast("Boston", days=3))

        assert len(days) == 3
        for entry in days:
            assert entry.high_temp is not None and entry.low_temp is not None
            assert entry.high_temp >= entry.low_temp
            assert isinstance(entry.condition, WeatherCondition)
