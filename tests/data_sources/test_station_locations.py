# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A reading is placed where its provider says it was taken, or nowhere.

Three sources completed a missing position with latitude 0, longitude 0 (a
point in the Gulf of Guinea):

* ``NOAACOOPSSource`` knew five stations; any other was "Unknown Station" at
  ``(0, 0)``, although every CO-OPS response carries the station's name and
  position in ``metadata``.
* ``USGSGeomagnetismSource`` did the same for an observatory missing from its
  table, although the response states the observatory's coordinates.
* ``USGSEarthquakeSource`` gave a feature without geometry ``(0, 0)``, one
  without an origin time 1970-01-01, and one without a magnitude 0.0.

The payloads are the shapes the live services returned on 2026-11-16; the
transport is patched, so nothing here touches the network.
"""

from __future__ import annotations

from typing import Any

import pytest

from omni_mercury_engine.data_sources.base import DataSourceError
from omni_mercury_engine.data_sources.earth_science import NOAACOOPSSource, USGSEarthquakeSource
from omni_mercury_engine.data_sources.geomagnetic import USGSGeomagnetismSource, USGSObservatory


class _Response:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def json(self) -> Any:
        return self._payload


def _serving(monkeypatch: pytest.MonkeyPatch, source: Any, payload: Any) -> None:
    async def _serve(*_args: Any, **_kwargs: Any) -> _Response:
        return _Response(payload)

    monkeypatch.setattr(source, "_http_get", _serve)


_COOPS = {
    "metadata": {"id": "9414290", "name": "San Francisco", "lat": "37.8063", "lon": "-122.4659"},
    "data": [{"t": "2026-11-16 01:36", "v": "0.597", "s": "0.022", "f": "1,0,0,0", "q": "p"}],
}


async def test_a_coops_station_outside_the_table_is_placed_from_the_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Was: "Unknown Station" at latitude 0, longitude 0."""
    source = NOAACOOPSSource(station_id="9414290")
    assert "9414290" not in NOAACOOPSSource.SAMPLE_STATIONS
    _serving(monkeypatch, source, _COOPS)

    (point,) = await source._fetch_impl()

    assert point.location == (37.8063, -122.4659, 0.0)
    assert point.data["station_name"] == "San Francisco"
    assert point.data["value"] == 0.597


async def test_a_coops_station_with_no_position_anywhere_has_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = NOAACOOPSSource(station_id="9414290")
    _serving(monkeypatch, source, {"data": _COOPS["data"]})

    (point,) = await source._fetch_impl()

    assert point.location is None
    assert point.data["station_name"] == "Station 9414290"


async def test_a_tabled_coops_station_still_resolves_without_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = NOAACOOPSSource(station_id="8518750")
    _serving(monkeypatch, source, {"data": _COOPS["data"]})

    (point,) = await source._fetch_impl()

    assert point.location == (40.700, -74.014, 0.0)


def test_an_observatory_is_placed_from_the_response_with_its_elevation() -> None:
    """Was: ``(0, 0)`` for an observatory missing from the table."""
    source = USGSGeomagnetismSource()
    response = {
        "metadata": {
            "intermagnet": {
                "imo": {
                    "iaga_code": "BOU",
                    "name": "Boulder",
                    "coordinates": [254.763, 40.137, 1682.0],
                }
            }
        }
    }

    location = source._observatory_location(USGSObservatory.BOULDER, response)

    # 254.763 degrees east is 105.237 degrees west; 1,682 m is 1.682 km.
    assert location == pytest.approx((40.137, -105.237, 1.682))
    assert source._observatory_location(USGSObservatory.BOULDER, {}) == (40.137, -105.237, 0.0)


def _feature(**changes: Any) -> dict[str, Any]:
    feature: dict[str, Any] = {
        "id": "us7000abcd",
        "properties": {"mag": 5.4, "time": 1794790000000, "magType": "mww", "place": "somewhere"},
        "geometry": {"type": "Point", "coordinates": [142.37, 38.32, 29.0]},
    }
    feature.update(changes)
    return feature


@pytest.mark.parametrize(
    "broken",
    [
        _feature(geometry=None),
        _feature(geometry={"type": "Point", "coordinates": []}),
        _feature(properties={"mag": 5.4, "magType": "mww"}),
        _feature(properties={"mag": None, "time": 1794790000000}),
    ],
)
async def test_an_earthquake_missing_a_field_is_skipped_not_completed(
    monkeypatch: pytest.MonkeyPatch, broken: dict[str, Any]
) -> None:
    """Was: a point at (0, 0), dated 1970, or of magnitude 0.0 (or an uncaught AttributeError)."""
    source = USGSEarthquakeSource()
    _serving(monkeypatch, source, {"features": [_feature(), broken]})

    points = await source._fetch_impl()

    assert [p.event_id for p in points] == ["us7000abcd"]
    assert points[0].location == (38.32, 142.37, 29.0)


async def test_earthquakes_that_all_lack_a_field_are_schema_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = USGSEarthquakeSource()
    _serving(monkeypatch, source, {"features": [_feature(geometry=None)]})

    with pytest.raises(DataSourceError, match="none parsed"):
        await source._fetch_impl()


@pytest.mark.parametrize(
    ("coordinates", "expected"),
    [
        ([254.763, 40.137, "x"], (40.137, -105.237, 0.0)),
        ([254.763, 40.137, float("nan")], (40.137, -105.237, 0.0)),
        ([254.763, 40.137], (40.137, -105.237, 0.0)),
    ],
)
def test_an_unreadable_elevation_does_not_cost_the_position(
    coordinates: list[Any], expected: tuple[float, float, float]
) -> None:
    """Was: the table fallback (None for an unlisted observatory), or an altitude of NaN."""
    response = {"metadata": {"intermagnet": {"imo": {"coordinates": coordinates}}}}

    location = USGSGeomagnetismSource()._observatory_location(USGSObservatory.BOULDER, response)

    assert location == pytest.approx(expected)


def test_a_longitude_outside_the_globe_is_not_wrapped_into_it() -> None:
    """``[400, 10, 0]`` was accepted as longitude 40."""
    response = {"metadata": {"intermagnet": {"imo": {"coordinates": [400.0, 10.0, 0.0]}}}}

    location = USGSGeomagnetismSource()._observatory_location(USGSObservatory.BOULDER, response)

    assert location == (40.137, -105.237, 0.0)  # the table's entry, not the response's
