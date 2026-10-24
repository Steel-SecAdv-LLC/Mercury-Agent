# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
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
from typing import Any

import pytest

from omni_mercury_engine.data_sources.base import AlertLevel, DataSourceError, DataSourceType
from omni_mercury_engine.data_sources.storm_surge import (
    FEET_TO_METERS,
    SURGEDAT_GLOBAL_URL,
    SURGEDAT_GULF_URL,
    NHCPeakStormSurgeSource,
    PeakSurgeForecast,
    SurgeObservation,
    parse_peak_surge_kml,
    parse_peak_surge_name,
    parse_surgedat_csv,
)

FIXTURES = Path(__file__).parent / "fixtures" / "nhc"
SURGEDAT = Path(__file__).parent / "fixtures" / "surgedat"


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
        payloads: dict[str, dict[str, Any]] = {
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
            def __init__(self, payload: dict[str, Any]) -> None:
                self._payload = payload

            def json(self) -> dict[str, Any]:
                return self._payload

        async def _get(endpoint: str, params: dict[str, Any] | None = None) -> _Response:
            seen.append(endpoint)
            assert params is not None and params["f"] == "json"
            return _Response(payloads[endpoint])

        source._http_get = _get  # type: ignore[method-assign, assignment, unused-ignore]
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
        frozen: Any = forecast  # the write below is deliberately illegal
        with pytest.raises(dataclasses.FrozenInstanceError):
            frozen.max_surge_ft = 99.0

    def test_document_without_data_folders_is_not_this_product(self) -> None:
        kml = (
            "<kml xmlns='http://www.opengis.net/kml/2.2'><Document>"
            "<Folder><name>Breakpoints</name></Folder>"
            "</Document></kml>"
        )
        with pytest.raises(ValueError, match="no Polygons or Lines folder"):
            parse_peak_surge_kml(kml.encode())


def _surgedat(name: str) -> bytes:
    return gzip.open(SURGEDAT / name, "rb").read()


@pytest.fixture(scope="module")
def surgedat_global() -> list[SurgeObservation]:
    return parse_surgedat_csv(_surgedat("globalpeaksurgedb.csv.gz"))


@pytest.fixture(scope="module")
def surgedat_gulf() -> list[SurgeObservation]:
    return parse_surgedat_csv(_surgedat("gompeaksurgedb.csv.gz"))


class TestSurgedatRecordedFiles:
    """The two SURGEDAT databases, byte-verbatim (tests/fixtures/surgedat/PROVENANCE.json)."""

    def test_fixtures_are_the_recorded_bytes(self) -> None:
        import hashlib

        provenance = json.loads((SURGEDAT / "PROVENANCE.json").read_text())
        for name, record in provenance["fixtures"].items():
            digest = hashlib.sha256(_surgedat(name)).hexdigest()
            assert f"sha256 of the uncompressed CSV: {digest}" in record["method"]
        urls = {record["url"] for record in provenance["fixtures"].values()}
        assert urls == {SURGEDAT_GLOBAL_URL, SURGEDAT_GULF_URL}

    def test_classic_mac_form(self) -> None:
        raw = _surgedat("globalpeaksurgedb.csv.gz")
        assert raw.count(b"\r") == 745 and b"\n" not in raw
        assert {b for b in raw if b > 0x7F} == {0xD0, 0xD3}
        with pytest.raises(UnicodeDecodeError):
            raw.decode("utf-8")

    def test_counts_and_span(
        self, surgedat_global: list[SurgeObservation], surgedat_gulf: list[SurgeObservation]
    ) -> None:
        assert len(surgedat_global) == 745
        assert len(surgedat_gulf) == 240
        assert {o.database for o in surgedat_global} == {"global"}
        assert {o.database for o in surgedat_gulf} == {"gulf"}
        assert (min(o.year for o in surgedat_global), max(o.year for o in surgedat_global)) == (
            1737,
            2014,
        )
        assert (min(o.year for o in surgedat_gulf), max(o.year for o in surgedat_gulf)) == (
            1880,
            2011,
        )
        assert sum(o.latitude is None for o in surgedat_global) == 158
        assert sum(o.latitude is None for o in surgedat_gulf) == 12

    def test_first_global_row_as_stated(self, surgedat_global: list[SurgeObservation]) -> None:
        first = surgedat_global[0]
        assert (first.year, first.storm, first.country) == (
            1897,
            "Typhoon of Samar and Leyte",
            "PH",
        )
        assert (first.latitude, first.longitude) == (11.412, 125.549)
        assert first.surge is None
        assert first.storm_tide is not None and first.storm_tide.metres == 7.3
        assert first.tropical is True and first.confidence == 3

    def test_mac_roman_names_decoded(self, surgedat_global: list[SurgeObservation]) -> None:
        framed = [o.storm for o in surgedat_global if o.storm.startswith("\u2013")]
        assert len(framed) == 11
        assert "\u2013Indianola\u201d" in framed

    def test_largest_global_surge_is_mahina(self, surgedat_global: list[SurgeObservation]) -> None:
        stated = [o for o in surgedat_global if o.surge is not None and o.surge.metres is not None]
        top = max(stated, key=lambda o: o.surge.metres if o.surge and o.surge.metres else 0.0)
        assert (top.year, top.storm) == (1899, "Mahina")
        assert top.surge is not None
        assert (top.surge.stated_ft, top.surge.stated_m) == ("44.95", "13.7")
        # 44.95 ft (+/- 0.005 ft) is the more precise reading of the pair.
        assert top.surge.metres == pytest.approx(44.95 * FEET_TO_METERS)
        assert top.gdacs_level() == "red"

    def test_gulf_rows_are_us_and_untagged(self, surgedat_gulf: list[SurgeObservation]) -> None:
        assert {o.country for o in surgedat_gulf} == {"US"}
        assert {o.tropical for o in surgedat_gulf} == {None}
        assert {o.storm_tide_waves for o in surgedat_gulf} == {None}
        beulah = max(
            (o for o in surgedat_gulf if o.surge is not None and o.surge.metres is not None),
            key=lambda o: o.surge.metres if o.surge and o.surge.metres else 0.0,
        )
        assert (beulah.year, beulah.storm, beulah.state) == (1967, "Beulah", "TX")
        assert beulah.surge is not None and beulah.surge.metres == 5.49

    def test_gdacs_levels_of_the_recorded_surges(
        self, surgedat_global: list[SurgeObservation], surgedat_gulf: list[SurgeObservation]
    ) -> None:
        def tally(rows: list[SurgeObservation]) -> dict[str | None, int]:
            counts: dict[str | None, int] = {}
            for row in rows:
                level = row.gdacs_level()
                counts[level] = counts.get(level, 0) + 1
            return counts

        assert tally(surgedat_global) == {"red": 95, "orange": 203, "green": 73, None: 374}
        assert tally(surgedat_gulf) == {"red": 8, "orange": 40, "green": 25, None: 167}


class TestSurgedatUnitConflicts:
    """A feet/metres pair is consistent only when its rounding intervals overlap."""

    def test_the_four_recorded_conflicts(self, surgedat_global: list[SurgeObservation]) -> None:
        conflicts = sorted(
            (o.year, o.storm, kind, h.stated_ft, h.stated_m)
            for o in surgedat_global
            for kind, h in (("surge", o.surge), ("storm_tide", o.storm_tide))
            if h is not None and h.conflict
        )
        assert conflicts == [
            (1940, "Unnamed", "storm_tide", "11.3", "3.94"),
            (1947, "George", "storm_tide", "10.8", "3.33"),
            (1985, "Unnamed", "storm_tide", "16.4", "4.5"),
            (1993, "Roger", "surge", "1.3", "0.7"),
        ]

    def test_gulf_file_is_self_consistent(self, surgedat_gulf: list[SurgeObservation]) -> None:
        heights = [h for o in surgedat_gulf for h in (o.surge, o.storm_tide) if h is not None]
        assert len(heights) == 251
        assert not any(h.conflict for h in heights)

    def test_conflict_has_no_single_value_but_keeps_both_readings(
        self, surgedat_global: list[SurgeObservation]
    ) -> None:
        (roger,) = [o for o in surgedat_global if o.storm == "Roger" and o.year == 1993]
        assert roger.surge is not None
        assert roger.surge.metres is None
        assert roger.surge.low_m == pytest.approx(1.3 * FEET_TO_METERS)
        assert roger.surge.high_m == 0.7
        assert roger.gdacs_level() == "green"  # both readings below 1 m

    def test_stated_range_is_kept_as_a_range(self, surgedat_global: list[SurgeObservation]) -> None:
        (georges,) = [o for o in surgedat_global if o.surge and o.surge.stated_m == "1.5-1.8"]
        assert (georges.year, georges.storm, georges.location) == (1998, "Georges", "Cudjoe Key")
        assert georges.surge is not None
        assert (georges.surge.low_m, georges.surge.high_m, georges.surge.metres) == (1.5, 1.8, None)
        assert georges.gdacs_level() == "orange"


_GULF_HEADER = (
    "Name,Year,Dates,Surge_FT,Surge_M,Storm_Tide_FT,Storm_Tide_M,Location,State,Lat,Lon,Confidence"
)


def _gulf_row(**cells: str) -> bytes:
    row = {
        "Name": "Test",
        "Year": "2000",
        "Dates": "Sep 1-2",
        "Surge_FT": "",
        "Surge_M": "",
        "Storm_Tide_FT": "",
        "Storm_Tide_M": "",
        "Location": "Somewhere",
        "State": "TX",
        "Lat": "29.0",
        "Lon": "-95.0",
        "Confidence": "2",
    }
    row.update(cells)
    return (_GULF_HEADER + "\r" + ",".join(row.values()) + "\r").encode("ascii")


class TestSurgedatParsing:
    """Constructed rows in the published Gulf layout; values chosen at the edges."""

    @pytest.mark.parametrize(
        ("ft", "m", "conflict", "metres"),
        [
            ("10", "3.05", False, 3.05),  # 3.048 m: inside 3.05 +/- 0.005
            ("10", "3.20", False, 3.2),  # 10 ft spans 2.896-3.200 m: the edge overlaps
            ("10.0", "3.20", True, None),  # 10.0 ft spans 3.033-3.063 m: no overlap
            ("1.3", "0.7", True, None),
            ("23", "7", False, 7.0104),  # +/- 0.15 m beats +/- 0.5 m: feet is the finer reading
            ("", "2.5", False, 2.5),
            ("8.2", "", False, 8.2 * FEET_TO_METERS),
        ],
    )
    def test_pair_consistency(self, ft: str, m: str, conflict: bool, metres: float | None) -> None:
        (row,) = parse_surgedat_csv(_gulf_row(Surge_FT=ft, Surge_M=m))
        assert row.surge is not None
        assert row.surge.conflict is conflict
        if metres is None:
            assert row.surge.metres is None
        else:
            assert row.surge.metres == pytest.approx(metres)

    def test_finer_metres_cell_is_kept(self) -> None:
        # 1 ft half-width is 0.1524 m; 0.3 m half-width is 0.05 m, so metres wins.
        (row,) = parse_surgedat_csv(_gulf_row(Surge_FT="1", Surge_M="0.3"))
        assert row.surge is not None and row.surge.metres == 0.3

    def test_band_edge_straddled_by_a_consistent_pair_has_no_level(self) -> None:
        (row,) = parse_surgedat_csv(_gulf_row(Surge_FT="10", Surge_M="2.99"))
        assert row.surge is not None and not row.surge.conflict
        assert row.gdacs_level() is None  # 3.048 m is red, 2.99 m is orange

    def test_storm_tide_alone_is_not_graded(self) -> None:
        (row,) = parse_surgedat_csv(_gulf_row(Storm_Tide_M="5.0"))
        assert row.storm_tide is not None and row.storm_tide.metres == 5.0
        assert row.gdacs_level() is None

    def test_blank_heights_are_none(self) -> None:
        (row,) = parse_surgedat_csv(_gulf_row())
        assert (row.surge, row.storm_tide, row.storm_tide_waves) == (None, None, None)

    @pytest.mark.parametrize(
        ("cells", "match"),
        [
            ({"Surge_M": "about 2"}, "unreadable SURGEDAT height"),
            ({"Surge_M": "-1"}, "unreadable SURGEDAT height"),
            ({"Surge_M": "2.0-1.5"}, "inverted"),
            ({"Lat": "91"}, "latitude"),
            ({"Lon": "-180.5"}, "longitude"),
            ({"Lon": ""}, "only one coordinate"),
            ({"Year": ""}, "invalid literal"),
        ],
    )
    def test_unreadable_cells_refused(self, cells: dict[str, str], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            parse_surgedat_csv(_gulf_row(**cells))

    def test_unknown_layout_refused(self) -> None:
        with pytest.raises(ValueError, match="not a SURGEDAT layout"):
            parse_surgedat_csv(b"Year,Name\r2000,Test\r")

    def test_empty_document_refused(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            parse_surgedat_csv(b"")

    def test_short_row_refused(self) -> None:
        with pytest.raises(ValueError, match="cells"):
            parse_surgedat_csv((_GULF_HEADER + "\rTest,2000\r").encode())

    def test_unix_line_ends_parse_identically(self) -> None:
        raw = _gulf_row(Surge_M="2.5")
        assert parse_surgedat_csv(raw.replace(b"\r", b"\n")) == parse_surgedat_csv(raw)
        assert parse_surgedat_csv(raw.replace(b"\r", b"\r\n")) == parse_surgedat_csv(raw)

    def test_global_tropical_flag_must_be_stated(self) -> None:
        raw = _surgedat("globalpeaksurgedb.csv.gz").decode("mac_roman")
        header, first, *_ = raw.split("\r")
        cells = first.split(",")
        cells[header.split(",").index("Tropical")] = "maybe"
        with pytest.raises(ValueError, match="tropical flag"):
            parse_surgedat_csv((header + "\r" + ",".join(cells) + "\r").encode("mac_roman"))


class TestSurgedatPrecisionEdges:
    """Pairs placed at the exact edges of the rounding-interval rule.

    Each case sits within a few micrometres of the overlap boundary, so a
    change to the half-width (0.5 of the last stated digit, in the cell's
    own unit) or to the slack flips it.
    """

    @pytest.mark.parametrize(
        ("ft", "m", "conflict"),
        [
            # 10 ft spans 2.8956-3.2004 m; 3.21 m spans 3.205-3.215: 4.6 mm apart.
            ("10", "3.21", True),
            # 10.0 ft spans 3.03276-3.06324 m; 3.0632 m spans 3.06315-3.06325.
            ("10.0", "3.0632", False),
            ("10.0", "3.0634", True),  # 3.06335 m: 0.11 mm past the feet edge
            # The metres side's own half-width: 3.04 m spans 3.035-3.045 and
            # 10.0 ft starts at 3.03276; 3.02 m (3.015-3.025) misses it.
            ("10.0", "3.04", False),
            ("10.0", "3.02", True),
            # A conflict narrower than 0.1 m is still a conflict.
            ("10.0", "3.12", True),
            # A stated range's upper end carries its own precision.
            ("9.0-10.0", "3.0632", False),
            ("9.0-10.0", "3.0634", True),
            ("9-10", "3.21", True),
        ],
    )
    def test_overlap_boundary(self, ft: str, m: str, conflict: bool) -> None:
        (row,) = parse_surgedat_csv(_gulf_row(Surge_FT=ft, Surge_M=m))
        assert row.surge is not None
        assert row.surge.conflict is conflict

    @pytest.mark.parametrize(
        ("ft", "m", "conflict"),
        [
            # A feet range whose ends differ in precision: 9 ft (+/- 0.1524 m)
            # starts the interval at 2.5908 m, 10.0 ft (+/- 0.01524 m) ends it
            # at 3.06324 m; 2.65 m (to 2.655) overlaps the low end only.
            ("9-10.0", "2.65", False),
            ("9-10.0", "2.58", True),  # 2.585 m: short of the 2.5908 m start
            # A metres range: 3 m (+/- 0.5) opens at 2.5, 3.10 m (+/- 0.005)
            # closes at 3.105; 8 ft ends at 2.5908 m, inside.
            ("8", "3-3.10", False),
            ("7", "3-3.10", True),  # 7 ft ends at 2.286 m, below 2.5
        ],
    )
    def test_range_ends_carry_their_own_precision(self, ft: str, m: str, conflict: bool) -> None:
        (row,) = parse_surgedat_csv(_gulf_row(Surge_FT=ft, Surge_M=m))
        assert row.surge is not None
        assert row.surge.conflict is conflict

    def test_range_bounds_in_metres(self) -> None:
        (row,) = parse_surgedat_csv(_gulf_row(Surge_FT="9.0-10.0"))
        assert row.surge is not None
        assert (row.surge.low_m, row.surge.high_m) == pytest.approx((2.7432, 3.048))
        assert row.surge.metres is None

    def test_records_are_immutable(self, surgedat_gulf: list[SurgeObservation]) -> None:
        row: Any = surgedat_gulf[0]  # the writes below are deliberately illegal
        assert row.surge is not None
        with pytest.raises(dataclasses.FrozenInstanceError):
            row.year = 1900
        with pytest.raises(dataclasses.FrozenInstanceError):
            row.surge.metres = 0.0

    @pytest.mark.parametrize(("lat", "lon"), [("90", "180"), ("-90", "-180"), ("90.0", "-180.0")])
    def test_coordinate_limits_are_inclusive(self, lat: str, lon: str) -> None:
        (row,) = parse_surgedat_csv(_gulf_row(Lat=lat, Lon=lon))
        assert (row.latitude, row.longitude) == (float(lat), float(lon))

    @pytest.mark.parametrize(
        ("cells", "name"), [({"Lat": "90.05"}, "latitude"), ({"Lon": "180.05"}, "longitude")]
    )
    def test_just_past_the_limits_refused(self, cells: dict[str, str], name: str) -> None:
        with pytest.raises(ValueError, match=name):
            parse_surgedat_csv(_gulf_row(**cells))

    def test_errors_name_the_file_row(self) -> None:
        # Line 1 is the header, so the second data row is file row 3.
        short = (_GULF_HEADER + "\rTest,2000,Sep,,,,,,,,,2\rTest,2000\r").encode()
        with pytest.raises(ValueError, match="SURGEDAT row 3 has 2 cells"):
            parse_surgedat_csv(short)

    def test_non_tropical_rows_are_flagged(self, surgedat_global: list[SurgeObservation]) -> None:
        flags = [o.tropical for o in surgedat_global]
        assert (flags.count(True), flags.count(False)) == (737, 8)
