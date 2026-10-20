# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""NHC Peak Storm Surge source against NHC's own recorded products.

The KML fixture is Hurricane Milton's advisory-17 Peak Storm Surge file,
byte-verbatim from the NHC GIS archive (tests/fixtures/nhc/PROVENANCE.json);
the query fixture is the live ArcGIS layer with no active advisory. Every
expected value below is read from the product, not invented.
"""

from __future__ import annotations

import asyncio
import dataclasses
import gzip
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from omni_mercury_engine.data_sources.base import AlertLevel, DataSourceError, DataSourceType
from omni_mercury_engine.data_sources.storm_surge import (
    FEET_TO_METERS,
    NHCPeakStormSurgeSource,
    PeakSurgeForecast,
    parse_peak_surge_kml,
    parse_peak_surge_name,
)

FIXTURES = Path(__file__).parent / "fixtures" / "nhc"


@pytest.fixture(scope="module")
def milton_forecasts() -> list[PeakSurgeForecast]:
    data = gzip.open(FIXTURES / "AL142024_PeakStormSurge_017adv.kml.gz", "rb").read()
    return parse_peak_surge_kml(data)


class TestMiltonAdvisory17:
    """The real Milton advisory-17 product parses to its published values."""

    def test_counts(self, milton_forecasts: list[PeakSurgeForecast]) -> None:
        kinds: dict[str, int] = {}
        for forecast in milton_forecasts:
            kinds[forecast.kind] = kinds.get(forecast.kind, 0) + 1
        assert kinds == {"inundation_area": 14, "coastal_segment": 13}

    def test_tampa_bay_is_the_10_15_ft_red(self, milton_forecasts: list[PeakSurgeForecast]) -> None:
        (tampa,) = [f for f in milton_forecasts if f.area == "Tampa Bay"]
        assert tampa.kind == "inundation_area"
        assert (tampa.min_surge_ft, tampa.max_surge_ft) == (10.0, 15.0)
        assert tampa.min_surge_m == pytest.approx(3.048)
        assert tampa.max_surge_m == pytest.approx(4.572)
        assert tampa.gdacs_level() == "red"

    def test_coastal_segment_values(self, milton_forecasts: list[PeakSurgeForecast]) -> None:
        (egmont,) = [f for f in milton_forecasts if f.area == "Egmont Key, FL to Boca Grande, FL"]
        assert egmont.kind == "coastal_segment"
        assert (egmont.min_surge_ft, egmont.max_surge_ft) == (10.0, 15.0)

    def test_gdacs_levels_from_the_manual_thresholds(
        self, milton_forecasts: list[PeakSurgeForecast]
    ) -> None:
        levels: dict[str, int] = {}
        for forecast in milton_forecasts:
            level = forecast.gdacs_level()
            assert level is not None  # every Milton area states a positive surge
            levels[level] = levels.get(level, 0) + 1
        # Red >= 3 m (>= 9.85 ft upper bound), Orange >= 1 m, Green below.
        assert levels == {"red": 5, "orange": 12, "green": 10}
        for forecast in milton_forecasts:
            expected = (
                "red"
                if forecast.max_surge_m >= 3.0
                else "orange" if forecast.max_surge_m >= 1.0 else "green"
            )
            assert forecast.gdacs_level() == expected

    def test_label_and_breakpoint_cartography_is_not_data(
        self, milton_forecasts: list[PeakSurgeForecast]
    ) -> None:
        assert all("..." not in f.area for f in milton_forecasts)
        assert all(f.min_surge_ft <= f.max_surge_ft for f in milton_forecasts)


class TestNameParsing:
    def test_feet_to_meters_is_the_international_foot(self) -> None:
        assert FEET_TO_METERS == 0.3048

    def test_label_names_are_none(self) -> None:
        assert parse_peak_surge_name("4-7 ft", "coastal_segment") is None

    def test_inverted_range_refused(self) -> None:
        with pytest.raises(ValueError, match="inverted"):
            parse_peak_surge_name("Tampa Bay...15-10 ft", "inundation_area")

    def test_unknown_kind_refused(self) -> None:
        with pytest.raises(ValueError, match="kind"):
            parse_peak_surge_name("Tampa Bay...1-3 ft", "surge_zone")

    def test_not_this_product_refused(self) -> None:
        with pytest.raises(ValueError, match="not a Peak Storm Surge KML"):
            parse_peak_surge_kml(b"<kml xmlns='http://www.opengis.net/kml/2.2'/>")


class TestLiveQueryParsing:
    """The recorded live layer answer and the service's failure shapes."""

    def test_source_registration(self) -> None:
        source = NHCPeakStormSurgeSource()
        assert source.source_id == "nhc_peak_storm_surge"
        assert source.default_source_types == [DataSourceType.STORM_SURGE]
        assert source.base_url.startswith("https://mapservices.weather.noaa.gov/")

    def test_quiet_feed_is_empty_not_zero(self) -> None:
        payload = json.loads((FIXTURES / "peakstormsurge_query_layer2_20261020.json").read_text())
        source = NHCPeakStormSurgeSource()
        now = datetime.now(UTC)
        assert source._points_from_query(payload, "inundation_area", now) == []

    def test_active_feature_becomes_a_graded_datapoint(self) -> None:
        source = NHCPeakStormSurgeSource()
        payload = {
            "features": [
                {
                    "attributes": {
                        "name": "Tampa Bay...10-15 ft",
                        "folderpath": "Peak Storm Surge/Polygons",
                        "idp_filedate": 1728432000000,
                    }
                }
            ]
        }
        (point,) = source._points_from_query(payload, "inundation_area", datetime.now(UTC))
        assert point.source_type is DataSourceType.STORM_SURGE
        assert point.data["area"] == "Tampa Bay"
        assert point.data["max_surge_m"] == pytest.approx(4.572)
        assert point.data["gdacs_level"] == "red"
        assert point.alert_level is AlertLevel.SEVERE
        assert point.timestamp == datetime(2024, 10, 9, 0, 0, tzinfo=UTC)

    def test_service_error_body_is_an_error_not_quiet(self) -> None:
        source = NHCPeakStormSurgeSource()
        with pytest.raises(DataSourceError, match="service error"):
            source._points_from_query(
                {"error": {"code": 500}}, "inundation_area", datetime.now(UTC)
            )

    def test_unreadable_data_layer_name_is_an_error(self) -> None:
        source = NHCPeakStormSurgeSource()
        with pytest.raises(DataSourceError, match="unreadable"):
            source._points_from_query(
                {"features": [{"attributes": {"name": "4-7 ft"}}]},
                "inundation_area",
                datetime.now(UTC),
            )


class TestMutationClosedGaps:
    """Cases a 14-mutant sweep (42.9% killed) proved untested."""

    def test_fetch_queries_polygons_then_lines(self) -> None:
        """The live fetch reads layer 2 (polygons) then layer 1 (lines)."""
        source = NHCPeakStormSurgeSource()
        seen: list[str] = []
        payloads = {
            "/2/query": {
                "features": [
                    {
                        "attributes": {
                            "name": "Tampa Bay...10-15 ft",
                            "idp_filedate": 1728432000000,
                        }
                    }
                ]
            },
            "/1/query": {"features": []},
        }

        class _Response:
            def __init__(self, payload: dict) -> None:
                self._payload = payload

            def json(self) -> dict:
                return self._payload

        async def _get(endpoint: str, params: dict | None = None) -> _Response:
            seen.append(endpoint)
            assert params is not None and params["f"] == "json"
            return _Response(payloads[endpoint])

        source._http_get = _get  # type: ignore[method-assign]
        points = asyncio.run(source._fetch_impl())
        assert seen == ["/2/query", "/1/query"]
        (point,) = points
        assert point.data["kind"] == "inundation_area"
        assert point.data["area"] == "Tampa Bay"

    def test_equal_range_bounds_are_a_valid_statement(self) -> None:
        forecast = parse_peak_surge_name("Mobile Bay...3-3 ft", "inundation_area")
        assert forecast is not None
        assert (forecast.min_surge_ft, forecast.max_surge_ft) == (3.0, 3.0)

    def test_zero_surge_grades_to_no_gdacs_level_and_alert_none(self) -> None:
        forecast = parse_peak_surge_name("Dry Basin...0-0 ft", "inundation_area")
        assert forecast is not None and forecast.gdacs_level() is None
        source = NHCPeakStormSurgeSource()
        (point,) = source._points_from_query(
            {"features": [{"attributes": {"name": "Dry Basin...0-0 ft"}}]},
            "inundation_area",
            datetime.now(UTC),
        )
        assert point.data["gdacs_level"] is None
        assert point.alert_level is AlertLevel.NONE

    def test_kml_with_empty_data_folders_is_a_quiet_product(self) -> None:
        kml = (
            "<kml xmlns='http://www.opengis.net/kml/2.2'><Document>"
            "<Folder><name>Polygons</name></Folder>"
            "<Folder><name>Lines</name></Folder>"
            "</Document></kml>"
        )
        assert parse_peak_surge_kml(kml.encode()) == []

    def test_forecast_records_are_immutable(self) -> None:
        forecast = parse_peak_surge_name("Tampa Bay...10-15 ft", "inundation_area")
        assert forecast is not None
        with pytest.raises(dataclasses.FrozenInstanceError):
            forecast.max_surge_ft = 99.0  # type: ignore[misc]

    def test_document_without_data_folders_is_not_this_product(self) -> None:
        kml = (
            "<kml xmlns='http://www.opengis.net/kml/2.2'><Document>"
            "<Folder><name>Breakpoints</name></Folder>"
            "</Document></kml>"
        )
        with pytest.raises(ValueError, match="no Polygons or Lines folder"):
            parse_peak_surge_kml(kml.encode())
