# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``NASAEONETSource`` locates Polygon geometries instead of failing the whole fetch.

EONET v3 geometry ``coordinates`` are a GeoJSON position (``[lon, lat]``) for
a ``Point`` but a list of rings of positions for a ``Polygon`` (GDACS flood
extents are published that way). The parser read ``coordinates[1]`` as the
latitude unconditionally, so one Polygon event raised ``TypeError: float()
argument must be a string or a real number, not 'list'`` -- which the
per-event handler does not catch -- and the entire fetch failed: 7 Polygon
events among 100 took the other 93 down with them.

The payload below holds two events recorded verbatim from
``https://eonet.gsfc.nasa.gov/api/v3/events?status=all&limit=100`` on
2026-11-14 (one Point wildfire, one 21-vertex Polygon flood) inside the
response envelope. The transport is patched to serve it; nothing here touches
the network.
"""

from __future__ import annotations

import copy
from datetime import UTC, datetime
from typing import Any

import pytest

from omni_mercury_engine.data_sources.base import DataSourceError, DataSourceType
from omni_mercury_engine.data_sources.space_weather import NASAEONETSource

_POLYGON_RING: list[list[float]] = [
    [-16.286083, 33.1242927],
    [-16.2853705, 33.1248284],
    [-16.2843386, 33.1252453],
    [-16.2839827, 33.1261046],
    [-16.284478, 33.1266107],
    [-16.28445, 33.1268032],
    [-16.2849212, 33.1274469],
    [-16.2849367, 33.1278037],
    [-16.2854832, 33.1281601],
    [-16.2856, 33.127989],
    [-16.2858847, 33.1280277],
    [-16.2860824, 33.127338],
    [-16.2864333, 33.1270312],
    [-16.2859954, 33.1266283],
    [-16.2859372, 33.1263067],
    [-16.2863396, 33.1258192],
    [-16.2866672, 33.1259891],
    [-16.2868934, 33.1258378],
    [-16.2865797, 33.1253381],
    [-16.2865045, 33.1244078],
    [-16.286083, 33.1242927],
]

_EONET_PAYLOAD: dict[str, Any] = {
    "title": "EONET Events",
    "description": "Natural events from EONET.",
    "link": "https://eonet.gsfc.nasa.gov/api/v3/events",
    "events": [
        {
            "id": "EONET_26331",
            "title": "Wildfire Red Hill, Perry, Mississippi",
            "description": None,
            "link": "https://eonet.gsfc.nasa.gov/api/v3/events/EONET_26331",
            "closed": None,
            "categories": [{"id": "wildfires", "title": "Wildfires"}],
            "sources": [
                {
                    "id": "IRWIN",
                    "url": "https://irwin.doi.gov/observer/incidents/2026-MSMNF-000713",
                }
            ],
            "geometry": [
                {
                    "magnitudeValue": 519.0,
                    "magnitudeUnit": "acres",
                    "date": "2026-11-12T00:22:00Z",
                    "type": "Point",
                    "coordinates": [-89.063853, 31.130069],
                }
            ],
        },
        {
            "id": "EONET_26164",
            "title": "Flood in Portugal 1104276",
            "description": None,
            "link": "https://eonet.gsfc.nasa.gov/api/v3/events/EONET_26164",
            "closed": "2026-11-08T00:00:00Z",
            "categories": [{"id": "floods", "title": "Floods"}],
            "sources": [
                {
                    "id": "GDACS",
                    "url": "https://www.gdacs.org/report.aspx?eventtype=FL&eventid=1104276",
                }
            ],
            "geometry": [
                {
                    "magnitudeValue": None,
                    "magnitudeUnit": None,
                    "date": "2026-11-06T20:00:00Z",
                    "type": "Polygon",
                    "coordinates": [_POLYGON_RING],
                }
            ],
        },
    ],
}


class _Response:
    """Stands in for the ``httpx.Response`` the transport returns."""

    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def json(self) -> Any:
        return self._payload


async def _fetch_points(monkeypatch: pytest.MonkeyPatch, payload: Any) -> list[Any]:
    source = NASAEONETSource()

    async def _serve(endpoint: str, params: dict[str, Any] | None = None) -> _Response:
        assert endpoint == "events"
        return _Response(payload)

    monkeypatch.setattr(source, "_http_get", _serve)
    return await source._fetch_impl()


async def test_point_and_polygon_events_both_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Was: ``TypeError`` from the Polygon event failed the fetch, Point event included."""
    points = await _fetch_points(monkeypatch, _EONET_PAYLOAD)

    assert [p.event_id for p in points] == ["EONET_26331", "EONET_26164"]
    assert all(p.source_type == DataSourceType.NATURAL_EVENT for p in points)

    wildfire, flood = points
    assert wildfire.location == pytest.approx((31.130069, -89.063853, 0.0))
    assert wildfire.data["geometry_type"] == "Point"
    assert wildfire.timestamp == datetime(2026, 11, 12, 0, 22, tzinfo=UTC)

    # The ring is closed (first vertex repeated last), so the representative
    # point is the mean of its 20 distinct vertices.
    distinct = _POLYGON_RING[:-1]
    expected_lat = sum(lat for _lon, lat in distinct) / len(distinct)
    expected_lon = sum(lon for lon, _lat in distinct) / len(distinct)
    assert flood.location == pytest.approx((expected_lat, expected_lon, 0.0))
    assert flood.data["geometry_type"] == "Polygon"
    assert flood.data["category"] == "floods"
    # ... and it lies inside the ring's own bounding box.
    assert min(lat for _lon, lat in distinct) < flood.location[0] < max(lat for _, lat in distinct)
    assert min(lon for lon, _lat in distinct) < flood.location[1] < max(lon for lon, _ in distinct)


async def test_unreadable_coordinates_skip_one_event_not_the_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An event whose coordinates cannot be read is dropped; its neighbours survive."""
    wildfire, flood = _EONET_PAYLOAD["events"]
    broken = {**flood, "id": "EONET_BROKEN", "geometry": [{**flood["geometry"][0]}]}
    broken["geometry"][0]["coordinates"] = [[["not-a-number", 33.1]]]
    payload = {**_EONET_PAYLOAD, "events": [broken, wildfire]}

    points = await _fetch_points(monkeypatch, payload)

    assert [p.event_id for p in points] == ["EONET_26331"]


@pytest.mark.parametrize(
    ("coordinates", "expected_lon_lat"),
    [
        ([10.0, 20.0], (10.0, 20.0)),
        ([10.0, 20.0, 150.0], (10.0, 20.0)),  # position with altitude
        ([[[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0], [0.0, 0.0]]], (2.0, 1.0)),
        # Outer ring only: the hole must not move the point.
        (
            [
                [[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0], [0.0, 0.0]],
                [[3.0, 1.0], [3.5, 1.0], [3.5, 1.5], [3.0, 1.0]],
            ],
            (2.0, 1.0),
        ),
        # A ring straddling the antimeridian averages across it, not to 0 deg.
        ([[[179.0, 0.0], [-179.0, 0.0], [-179.0, 2.0], [179.0, 2.0], [179.0, 0.0]]], (180.0, 1.0)),
    ],
)
def test_representative_position(
    coordinates: list[Any], expected_lon_lat: tuple[float, float]
) -> None:
    """Point, Polygon, Polygon-with-hole and antimeridian shapes reduce to one position."""
    lon, lat = NASAEONETSource._representative_lon_lat(coordinates)

    assert (abs(lon), lat) == pytest.approx((abs(expected_lon_lat[0]), expected_lon_lat[1]))


@pytest.mark.parametrize("coordinates", [[], None, "POINT(1 2)", [[]], [5.0]])
def test_unreadable_coordinates_are_rejected(coordinates: Any) -> None:
    """Shapes with no readable position raise instead of defaulting to (0, 0)."""
    with pytest.raises(ValueError, match="EONET"):
        NASAEONETSource._representative_lon_lat(coordinates)


# ---------------------------------------------------------------------------
# One malformed event, schema drift, and positions that are not places.
# ---------------------------------------------------------------------------


def _with_second_event(**changes: Any) -> dict[str, Any]:
    """The recorded payload with fields of its second (flood) event replaced."""
    payload = copy.deepcopy(_EONET_PAYLOAD)
    payload["events"][1].update(changes)
    return payload


@pytest.mark.parametrize(
    "changes",
    [
        {"categories": ["floods"]},
        {"geometry": [{"date": None, "type": "Point", "coordinates": [1.0, 2.0]}]},
        {"geometry": {"date": "2026-11-06T20:00:00Z", "coordinates": [1.0, 2.0]}},
        {"geometry": [{"date": "2026-11-06T20:00:00Z", "coordinates": [500.0, 95.0]}]},
        {"geometry": [{"date": "2026-11-06T20:00:00Z", "coordinates": [float("nan"), 5.0]}]},
    ],
)
async def test_one_malformed_event_does_not_cost_the_well_formed_one(
    monkeypatch: pytest.MonkeyPatch, changes: dict[str, Any]
) -> None:
    """Was: an uncaught TypeError/AttributeError failed the fetch, or a point at latitude 95."""
    points = await _fetch_points(monkeypatch, _with_second_event(**changes))

    assert [p.event_id for p in points] == ["EONET_26331"]


async def test_a_null_sources_list_keeps_the_event(monkeypatch: pytest.MonkeyPatch) -> None:
    """Was: ``'NoneType' object is not iterable`` failed the whole fetch."""
    points = await _fetch_points(monkeypatch, _with_second_event(sources=None))

    assert [p.event_id for p in points] == ["EONET_26331", "EONET_26164"]
    assert points[1].data["sources"] == []


@pytest.mark.parametrize("payload", [{}, {"events": None}, ["not", "an", "envelope"]])
async def test_a_response_without_an_events_array_is_an_error(
    monkeypatch: pytest.MonkeyPatch, payload: Any
) -> None:
    """Was: ``success=True`` with zero points, indistinguishable from a quiet day."""
    with pytest.raises(DataSourceError, match="no 'events' array"):
        await _fetch_points(monkeypatch, payload)


async def test_events_that_all_fail_to_parse_are_schema_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Was: ``success=True`` with zero points and the reasons logged at DEBUG."""
    payload = {
        "events": [
            {"id": "a", "geometry": [{"type": "Point", "coordinates": [1.0, 2.0]}]},
            {"id": "b", "geometry": [{"date": "2026-11-06T20:00:00Z", "coordinates": [[]]}]},
        ]
    }

    with pytest.raises(DataSourceError, match="2 events but none parsed"):
        await _fetch_points(monkeypatch, payload)


async def test_an_empty_events_array_is_still_a_quiet_day(monkeypatch: pytest.MonkeyPatch) -> None:
    assert await _fetch_points(monkeypatch, {"events": []}) == []


async def test_a_date_without_a_zone_is_read_as_utc(monkeypatch: pytest.MonkeyPatch) -> None:
    """Was: a naive timestamp, which raises when compared with the aware ones."""
    geometry = [{"date": "2026-11-06T20:00:00", "type": "Point", "coordinates": [1.0, 2.0]}]
    points = await _fetch_points(monkeypatch, _with_second_event(geometry=geometry))

    assert points[1].timestamp == datetime(2026, 11, 6, 20, 0, tzinfo=UTC)
    assert points[0].timestamp < points[1].timestamp or points[0].timestamp >= points[1].timestamp
