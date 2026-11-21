# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The volcanic loader reads the USGS Hazard Notification System.

Its endpoints were ``/vsc/api/volcanoApi/alerts`` and ``/volcanoList``, which
USGS retired: both answered HTTP 404 "Resource not found in API" on
2026-11-17, so every live read failed while three volcanoes stood above
NORMAL. The loader now reads the public HANS API. The payloads below are in
the shape that API returned on that date.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from omni_mercury_engine.loaders import (
    VolcanicLoader,
    volcanic_loader as module,
)

if TYPE_CHECKING:
    from pathlib import Path

MONITORED = [
    {
        "volcano_name": "Great Sitkin",
        "vnum": "311120",
        "sent_utc": "2026-11-13 21:04:48",
        "alert_level": "WATCH",
        "color_code": "ORANGE",
        "obs_fullname": "Alaska Volcano Observatory",
        "notice_identifier": "DOI-USGS-AVO-2026-11-13T19:12:01+00:00",
    },
    {
        "volcano_name": "Akutan",
        "vnum": "311320",
        "sent_utc": "2019-09-06 18:29:39",
        "alert_level": "NORMAL",
        "color_code": "GREEN",
        "obs_fullname": "Alaska Volcano Observatory",
        "notice_identifier": "DOI-USGS-AVO-2019-09-04T15:47:55-08:00",
    },
]
US_VOLCANOES = [
    {"volcano_name": "Great Sitkin", "latitude": 52.0765, "longitude": -176.1109,
     "elevation_meters": 1740},
    {"volcano_name": "Akutan", "latitude": 54.1331, "longitude": -165.9856,
     "elevation_meters": 1303},
]  # fmt: skip


@pytest.fixture()
def loader(tmp_path: Path) -> VolcanicLoader:
    return VolcanicLoader(cache_dir=tmp_path, max_retries=1)


def test_the_endpoints_are_the_ones_usgs_serves() -> None:
    assert module._ALERTS_URL == (
        "https://volcanoes.usgs.gov/hans-public/api/volcano/getMonitoredVolcanoes"
    )
    assert module._VOLCANO_LIST_URL.endswith("/hans-public/api/volcano/getUSVolcanoes")
    assert "vsc/api/volcanoApi" not in module._ALERTS_URL + module._VOLCANO_LIST_URL


def test_a_live_read_carries_each_volcanos_level_notice_and_place(
    loader: VolcanicLoader, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[str] = []

    def answer(url: str, params: dict[str, str] | None = None) -> Any:
        asked.append(url)
        return MONITORED if url == module._ALERTS_URL else US_VOLCANOES

    monkeypatch.setattr(loader, "_fetch_json", answer)

    frame = loader.fetch_realtime().set_index("volcano_name")

    assert asked == [module._ALERTS_URL, module._VOLCANO_LIST_URL]
    sitkin = frame.loc["Great Sitkin"]
    assert (sitkin["alert_level"], sitkin["color_code"]) == ("WATCH", "ORANGE")
    assert (sitkin["alert_level_numeric"], sitkin["color_code_numeric"]) == (2, 2)
    assert sitkin["alert_date"] == "2026-11-13 21:04:48"  # was empty: the key is sent_utc
    assert sitkin["volcano_number"] == "311120"
    assert sitkin["observatory"] == "Alaska Volcano Observatory"
    assert sitkin["notice_id"] == "DOI-USGS-AVO-2026-11-13T19:12:01+00:00"
    assert (sitkin["latitude"], sitkin["longitude"]) == (52.0765, -176.1109)
    assert sitkin["elevation"] == 1740  # was NaN: the key is elevation_meters
    assert frame.loc["Akutan", "alert_level_numeric"] == 0


def test_a_notice_is_fetched_by_its_identifier(
    loader: VolcanicLoader, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[str] = []

    def answer(url: str, params: dict[str, str] | None = None) -> Any:
        asked.append(url)
        return {"notice_type_title": "Weekly Update", "notice_sections": []}

    monkeypatch.setattr(loader, "_fetch_json", answer)

    notice = loader.fetch_notice("DOI-USGS-AVO-2026-11-13T19:12:01+00:00")

    assert notice["notice_type_title"] == "Weekly Update"
    assert asked == [
        "https://volcanoes.usgs.gov/hans-public/api/notice/getNotice/"
        "DOI-USGS-AVO-2026-11-13T19:12:01+00:00"
    ]


@pytest.mark.parametrize("identifier", ["", "../../admin", "DOI USGS", "a?b=c", "x" * 200])
def test_only_a_notice_identifier_reaches_the_url(loader: VolcanicLoader, identifier: str) -> None:
    with pytest.raises(ValueError, match="not a USGS notice identifier"):
        loader.fetch_notice(identifier)


def test_a_notice_in_another_shape_is_not_a_notice(
    loader: VolcanicLoader, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loader, "_fetch_json", lambda url, params=None: ["not", "a", "notice"])

    with pytest.raises(ConnectionError, match="not the documented shape"):
        loader.fetch_notice("DOI-USGS-AVO-2026-11-13T19:12:01+00:00")
