# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""National flood status from NOAA's National Water Prediction Service.

The flood loader's "real-time" read was three stream gauges, the ones its
benchmark events use. On 2026-11-17 the National Weather Service classed seven
gauges as in flood, three of them major, and none of the three was read.
``fetch_flood_status`` reads every NWS gauge with the category its forecast
office assigns the latest reading; ``fetch_gauge_record`` reads one gauge's
flood stages and its observed record.

The service is stubbed at ``_fetch_json`` with payloads in the shape it
returned on that date.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from omni_mercury_engine.loaders import (
    FloodLoader,
    flood_loader as module,
)
from omni_mercury_engine.security.input_validation import TrustedEndpoints

if TYPE_CHECKING:
    from pathlib import Path


def _gauge(lid: str, category: str, stage: float = 17.09, **fields: Any) -> dict[str, Any]:
    return {
        "lid": lid,
        "name": f" River near {lid} ",
        "state": {"abbreviation": "FL", "name": "Florida"},
        "latitude": 28.37,
        "longitude": -80.87,
        "status": {
            "observed": {
                "primary": stage,
                "primaryUnit": "ft",
                "floodCategory": category,
                "validTime": "2026-11-17T00:15:00Z",
            },
            "forecast": {
                "primary": -999,
                "floodCategory": "fcst_not_current",
                "validTime": "0001-01-01T00:00:00Z",
            },
        },
        **fields,
    }


class _Service:
    """Answers the gauge listing per tile and records what was asked."""

    def __init__(self, tiles: list[Any]) -> None:
        self.tiles = tiles
        self.asked: list[tuple[str, dict[str, str] | None]] = []

    def __call__(self, url: str, params: dict[str, str] | None = None) -> Any:
        self.asked.append((url, params))
        answer = self.tiles[(len(self.asked) - 1) % len(self.tiles)]
        if isinstance(answer, Exception):
            raise answer
        return answer


@pytest.fixture()
def loader(tmp_path: Path) -> FloodLoader:
    return FloodLoader(cache_dir=tmp_path, max_retries=1)


def test_the_service_host_is_trusted_by_name() -> None:
    assert TrustedEndpoints.validate_url_host("api.water.noaa.gov")
    with pytest.raises(ValueError, match="not in trusted allowlist"):
        TrustedEndpoints.validate_url_host("api.water.noaa.gov.example.org")


def test_every_tile_is_read_and_a_gauge_on_an_edge_is_counted_once(
    loader: FloodLoader, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _Service(
        [
            {"gauges": [_gauge("COCF1", "major"), _gauge("EDGE1", "minor", 4.0)]},
            {"gauges": [_gauge("EDGE1", "minor", 4.0), _gauge("QUIET1", "no_flooding", 2.0)]},
        ]
    )
    monkeypatch.setattr(loader, "_fetch_json", service)

    frame = loader.fetch_flood_status()

    assert len(service.asked) == len(module._NWPS_TILES) == 6
    assert {params["srid"] for _, params in service.asked if params} == {"EPSG_4326"}
    assert len({tuple(sorted(params.items())) for _, params in service.asked if params}) == 6
    assert sorted(frame["gauge_id"]) == ["COCF1", "EDGE1", "QUIET1"]
    row = frame.set_index("gauge_id").loc["COCF1"]
    assert (row["name"], row["state"], row["category"]) == ("River near COCF1", "Florida", "major")
    assert row["stage"] == 17.09 and row["unit"] == "ft"
    assert row["observed_at"] == pd.Timestamp("2026-11-17T00:15:00Z")
    assert row["forecast_category"] == "fcst_not_current"


def test_the_tiles_cover_the_states_and_puerto_rico() -> None:
    places = {
        "Seattle": (-122.3, 47.6), "Miami": (-80.2, 25.8), "Caribou, Maine": (-68.0, 46.9),
        "Brownsville": (-97.5, 25.9), "Fairbanks": (-147.7, 64.8), "Hilo": (-155.1, 19.7),
        "San Juan": (-66.1, 18.4), "Denver": (-105.0, 39.7), "St. Louis": (-90.2, 38.6),
    }  # fmt: skip
    for place, (longitude, latitude) in places.items():
        assert any(
            west <= longitude <= east and south <= latitude <= north
            for west, south, east, north in module._NWPS_TILES
        ), place


def test_the_services_placeholders_are_not_readings(
    loader: FloodLoader, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The service writes -999 for no stage and year 1 for no time."""
    silent = _gauge("DOWN1", "out_of_service", -999)
    silent["status"]["observed"]["validTime"] = "0001-01-01T00:00:00Z"
    bare = {"lid": "BARE1"}
    monkeypatch.setattr(
        loader, "_fetch_json", _Service([{"gauges": [silent, bare, {"name": "no id"}, "junk"]}])
    )

    frame = loader.fetch_flood_status().set_index("gauge_id")

    assert list(frame.index) == ["DOWN1", "BARE1"]
    assert np.isnan(frame.loc["DOWN1", "stage"]) and pd.isna(frame.loc["DOWN1", "observed_at"])
    assert frame.loc["BARE1", "category"] == "not_defined" and frame.loc["BARE1", "name"] == "BARE1"


def test_a_tile_that_fails_fails_the_read_and_nothing_partial_is_cached(
    loader: FloodLoader, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A country with a region missing must not be reported as the country."""
    good = {"gauges": [_gauge("COCF1", "major")]}
    monkeypatch.setattr(loader, "_fetch_json", _Service([good, ConnectionError("504")]))

    with pytest.raises(ConnectionError):
        loader.fetch_flood_status()

    healthy = _Service([good])
    monkeypatch.setattr(loader, "_fetch_json", healthy)
    assert len(loader.fetch_flood_status()) == 1 and len(healthy.asked) == 6


def test_a_listing_in_another_shape_is_refused(
    loader: FloodLoader, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loader, "_fetch_json", _Service([{"message": "Internal error"}]))

    with pytest.raises(ValueError, match="not the documented shape"):
        loader.fetch_flood_status()


def test_a_read_is_cached_with_its_times(
    loader: FloodLoader, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _Service([{"gauges": [_gauge("COCF1", "major")]}])
    monkeypatch.setattr(loader, "_fetch_json", service)

    first, again = loader.fetch_flood_status(), loader.fetch_flood_status()

    assert len(service.asked) == 6
    assert again["observed_at"].iloc[0] == first["observed_at"].iloc[0]
    assert str(again["observed_at"].dtype).endswith("UTC]")


def _record(
    loader: FloodLoader, monkeypatch: pytest.MonkeyPatch, flood: Any, series: Any
) -> tuple[dict[str, float], pd.DataFrame]:
    asked: list[str] = []

    def answer(url: str, params: dict[str, str] | None = None) -> Any:
        asked.append(url)
        return series if url.endswith("/stageflow/observed") else {"lid": "COCF1", "flood": flood}

    monkeypatch.setattr(loader, "_fetch_json", answer)
    result = loader.fetch_gauge_record("COCF1")
    assert asked == [
        "https://api.water.noaa.gov/nwps/v1/gauges/COCF1",
        "https://api.water.noaa.gov/nwps/v1/gauges/COCF1/stageflow/observed",
    ]
    return result


_FLOOD = {
    "stageUnits": "ft",
    "flowUnits": "cfs",
    "categories": {
        "action": {"stage": 14, "flow": 9000},
        "minor": {"stage": 15, "flow": 12000},
        "moderate": {"stage": 16, "flow": -9999},
        "major": {"stage": 17, "flow": -9999},
    },
}


def test_a_gauges_record_comes_with_its_flood_stages(
    loader: FloodLoader, monkeypatch: pytest.MonkeyPatch
) -> None:
    series = {
        "primaryUnits": "ft",
        "data": [
            {"validTime": "2026-11-17T00:15:00Z", "primary": 17.09, "secondary": -999},
            {"validTime": "2026-11-17T00:00:00Z", "primary": 17.08, "secondary": -999},
            {"validTime": "2026-11-16T23:45:00Z", "primary": -999, "secondary": -999},
            {"validTime": "not a time", "primary": 3.0},
        ],
    }

    stages, record = _record(loader, monkeypatch, _FLOOD, series)

    assert stages == {"action": 14.0, "minor": 15.0, "moderate": 16.0, "major": 17.0}
    assert record["stage"].tolist() == [17.08, 17.09]  # oldest first, placeholders dropped
    assert record["observed_at"].iloc[-1] == pd.Timestamp("2026-11-17T00:15:00Z")


def test_a_record_kept_as_flow_takes_the_flow_stages_in_its_own_unit(
    loader: FloodLoader, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record is in thousands of cubic feet per second; the stages are in cubic feet."""
    series = {
        "primaryUnits": "kcfs",
        "data": [{"validTime": "2026-11-17T00:15:00Z", "primary": 12.4}],
    }

    stages, _ = _record(loader, monkeypatch, _FLOOD, series)

    assert stages == {"action": 9.0, "minor": 12.0}  # the unset categories are left out


@pytest.mark.parametrize(
    ("flood", "series"),
    [
        (None, {"primaryUnits": "ft", "data": []}),
        ({"categories": "none"}, {"primaryUnits": "m", "data": None}),
        (_FLOOD, "not a record"),
    ],
)
def test_a_gauge_without_stages_or_readings_has_neither(
    loader: FloodLoader, monkeypatch: pytest.MonkeyPatch, flood: Any, series: Any
) -> None:
    stages, record = _record(loader, monkeypatch, flood, series)

    assert stages == {} or series == {"primaryUnits": "ft", "data": []}
    assert record.empty and list(record.columns) == ["observed_at", "stage"]


def test_only_a_gauge_identifier_reaches_the_url(loader: FloodLoader) -> None:
    with pytest.raises(ValueError, match="not an NWS gauge identifier"):
        loader.fetch_gauge_record("COCF1/../../admin")
