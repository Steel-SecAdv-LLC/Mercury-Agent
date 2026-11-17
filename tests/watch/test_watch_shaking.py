# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""An earthquake goes to the people where USGS says it was felt.

A subscriber was told of an earthquake when its epicentre lay within their
radius (200 km unless changed), whatever its size. The magnitude 6.6
earthquake off Tual, Indonesia on 2026-11-16 (``us7000tymu``) was reported
felt in Darwin, 708 km away: a subscriber there was not told, and one 190 km
from a magnitude 4.5 nobody felt was.

The files under ``tests/watch/data`` are USGS's own products for that
earthquake as served on 2026-11-17: its event detail (cut down to the two
products read), ShakeMap's low-resolution intensity grid, and the "Did You
Feel It?" 10 km cells.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from omni_mercury_engine.watch import Severity, WatchStore
from omni_mercury_engine.watch.feeds import EarthquakeFeed
from omni_mercury_engine.watch.guidance import with_guidance
from omni_mercury_engine.watch.model import Finding
from omni_mercury_engine.watch.notify import Notifier, audience, notice, shaking_note
from omni_mercury_engine.watch.shaking import (
    Felt,
    _cells,
    _grid,
    felt_at,
    intensity_at,
    intensity_words,
    read_reach,
    reported_at,
)
from omni_mercury_engine.watch.sms import text_message

DATA = Path(__file__).parent / "data"
NOW = datetime(2026, 11, 16, 22, 0, tzinfo=UTC)
EVENT = "us7000tymu"
EPICENTRE = (-6.0804, 131.4868)
DARWIN = (-12.43, 130.88)  # inside the cell five people reported from, 708 km away


def _json(name: str) -> Any:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


@pytest.fixture()
def reach() -> dict[str, Any]:
    return {
        "event": EVENT,
        "grid": _grid(_json("usgs_us7000tymu_mmi_low_res.covjson")),
        "cells": _cells(_json("usgs_us7000tymu_dyfi_geo_10km.geojson")),
    }


@pytest.fixture()
def store(tmp_path: Path) -> WatchStore:
    return WatchStore(tmp_path)


def _quake(magnitude: float = 6.6) -> Finding:
    return with_guidance(
        Finding(
            feed="earthquakes",
            kind="earthquake",
            subject=EVENT,
            severity=Severity.SIGNIFICANT,
            headline=f"Magnitude {magnitude} earthquake, 148 km WSW of Tual, Indonesia",
            detail="USGS located it.",
            observed_at=NOW,
            source="U.S. Geological Survey",
            basis="USGS magnitude class: strong",
            value=magnitude,
            unit="magnitude",
            latitude=EPICENTRE[0],
            longitude=EPICENTRE[1],
            ongoing=False,
        )
    )


def _person(
    email: str, place: str, point: tuple[float, float], radius: float = 200.0
) -> dict[str, Any]:
    return {
        "id": abs(hash(email)) % 1000, "email": email, "place": place, "latitude": point[0],
        "longitude": point[1], "radius_km": radius, "min_severity": 1,
    }  # fmt: skip


class TestWhatUsgsSays:
    def test_the_grid_gives_usgss_estimate_at_a_place(self, reach: dict[str, Any]) -> None:
        assert intensity_at(reach, *EPICENTRE) == pytest.approx(5.6, abs=0.15)
        assert intensity_at(reach, -7.0, 130.55) == 3.9  # a corner of the map
        assert intensity_at(reach, *DARWIN) is None  # off the map
        assert intensity_at(None, *EPICENTRE) is None

    def test_rows_run_south_to_north(self) -> None:
        """Checked against USGS's contours on three earthquakes; the other order is wrong."""
        coverage = {
            "domain": {"axes": {"x": {"start": 10.0, "stop": 11.0, "num": 2},
                                "y": {"start": -5.0, "stop": -4.0, "num": 2}}},
            "ranges": {"MMI": {"axisNames": ["y", "x"], "shape": [2, 2],
                               "values": [1.0, 2.0, 7.0, 8.0]}},
        }  # fmt: skip
        grid = {"grid": _grid(coverage)}

        assert intensity_at(grid, -5.0, 10.0) == 1.0  # the first value is the south-west corner
        assert intensity_at(grid, -4.0, 10.0) == 7.0
        assert intensity_at(grid, -4.0, 11.0) == 8.0

    def test_a_map_across_the_date_line_is_read_from_either_side(self) -> None:
        coverage = {
            "domain": {"axes": {"x": {"start": 179.0, "stop": 181.0, "num": 3},
                                "y": {"start": -18.0, "stop": -16.0, "num": 3}}},
            "ranges": {"MMI": {"axisNames": ["y", "x"], "shape": [3, 3], "values": [4.0] * 9}},
        }  # fmt: skip

        assert intensity_at({"grid": _grid(coverage)}, -17.0, -179.5) == 4.0

    def test_a_grid_that_is_not_what_was_expected_is_not_used(self) -> None:
        good = _json("usgs_us7000tymu_mmi_low_res.covjson")
        wrong_axes = json.loads(json.dumps(good))
        wrong_axes["ranges"]["MMI"]["axisNames"] = ["x", "y"]
        short = json.loads(json.dumps(good))
        short["ranges"]["MMI"]["values"] = short["ranges"]["MMI"]["values"][:-1]

        assert _grid(wrong_axes) is None and _grid(short) is None and _grid({}) is None

    def test_the_cells_give_what_people_reported(self, reach: dict[str, Any]) -> None:
        assert len(reach["cells"]) == 5
        assert reported_at(reach, *DARWIN) == (2.8, 5)
        assert reported_at(reach, *EPICENTRE) is None  # nobody reported from the sea

    def test_felt_is_usgss_estimate_or_peoples_reports(self, reach: dict[str, Any]) -> None:
        darwin, near = felt_at(reach, *DARWIN), felt_at(reach, *EPICENTRE)

        assert darwin == Felt(None, 2.8, 5) and darwin.felt
        assert near is not None and near.estimated is not None and near.felt
        assert felt_at(reach, 35.68, 139.69) is None  # Tokyo: USGS said nothing of it
        assert Felt(estimated=2.4).felt is False and Felt(estimated=2.5).felt is True
        assert Felt(reported=1.0, reports=3).felt is False

    @pytest.mark.parametrize(
        ("mmi", "words"),
        [(1.0, "I (not felt)"), (2.8, "III (weak shaking)"), (3.9, "IV (light shaking)"),
         (5.6, "VI (strong shaking)"), (9.7, "X+ (extreme shaking)")],
    )  # fmt: skip
    def test_an_intensity_is_said_as_usgss_legend_says_it(self, mmi: float, words: str) -> None:
        assert intensity_words(mmi) == words


class TestReading:
    @staticmethod
    def _serve(monkeypatch: pytest.MonkeyPatch, detail: Any) -> list[str]:
        from omni_mercury_engine.security.safe_http import SafeHTTPClient

        asked: list[str] = []

        def get_json(url: str, **kwargs: Any) -> Any:
            asked.append(
                url + ("?eventid=" + kwargs["params"]["eventid"] if "params" in kwargs else "")
            )
            if url.endswith("coverage_mmi_low_res.covjson"):
                return _json("usgs_us7000tymu_mmi_low_res.covjson")
            if url.endswith("dyfi_geo_10km.geojson"):
                return _json("usgs_us7000tymu_dyfi_geo_10km.geojson")
            return detail

        monkeypatch.setattr(SafeHTTPClient, "get_json", staticmethod(get_json))
        return asked

    def test_both_products_are_read_from_the_events_own_record(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        asked = self._serve(monkeypatch, _json("usgs_us7000tymu_detail.json"))

        reach = read_reach(EVENT)

        assert asked[0] == "https://earthquake.usgs.gov/fdsnws/event/1/query?eventid=us7000tymu"
        assert len(asked) == 3 and all(
            url.startswith("https://earthquake.usgs.gov/") for url in asked
        )
        assert reach["grid"]["nx"] == reach["grid"]["ny"] == 55 and len(reach["cells"]) == 5

    def test_an_earthquake_usgs_has_published_nothing_for_has_no_reach(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        detail = _json("usgs_us7000tymu_detail.json")
        detail["properties"]["products"] = {}
        self._serve(monkeypatch, detail)

        assert read_reach(EVENT) == {"event": EVENT, "grid": None, "cells": []}

    def test_a_product_that_is_not_on_usgss_host_is_not_fetched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        detail = _json("usgs_us7000tymu_detail.json")
        contents = detail["properties"]["products"]["shakemap"][0]["contents"]
        contents["download/coverage_mmi_low_res.covjson"]["url"] = "https://evil.example/x.covjson"
        asked = self._serve(monkeypatch, detail)

        reach = read_reach(EVENT)

        assert reach["grid"] is None and not any("evil.example" in url for url in asked)

    @pytest.mark.parametrize("bad", ["", "../etc", "us7000tymu&x=1", "US7000TYMU", "a b"])
    def test_what_is_not_an_event_id_is_never_asked_for(self, bad: str) -> None:
        with pytest.raises(ValueError, match="event id"):
            read_reach(bad)


class TestWhoIsTold:
    def test_someone_where_it_was_felt_is_told_however_far_away(
        self, reach: dict[str, Any]
    ) -> None:
        """Was: Darwin, 708 km away with five reports of shaking, was told nothing."""
        darwin = _person("d@example.org", "Darwin", DARWIN)
        quake = _quake().to_dict()

        assert audience(quake, [darwin]) == []  # by radius alone
        ((told, away),) = audience(quake, [darwin], reach)
        assert told is darwin and away == pytest.approx(709, abs=3)

    def test_someone_within_their_radius_where_usgs_estimates_no_felt_shaking_is_not_told(
        self,
    ) -> None:
        """Was: a magnitude 4.5 at 190 km, felt by no one there, was sent to them."""
        weak = {
            "event": EVENT,
            "cells": [],
            "grid": {"x0": 130.0, "x1": 133.0, "nx": 2, "y0": -8.0, "y1": -4.0, "ny": 2,
                     "mmi": [2.0, 2.0, 2.0, 2.0]},
        }  # fmt: skip
        near = _person("n@example.org", "Saumlaki", (-7.5, 131.3))
        quake = _quake(4.5).to_dict()

        assert len(audience(quake, [near])) == 1  # 160 km: inside the radius
        assert audience(quake, [near], weak) == []

    def test_where_usgs_has_said_nothing_the_subscribers_radius_still_decides(
        self, reach: dict[str, Any]
    ) -> None:
        inside = _person("a@example.org", "Ambon", (-3.70, 128.18), radius=500)  # off the map
        outside = _person("j@example.org", "Jakarta", (-6.2, 106.8))
        quake = _quake().to_dict()

        assert felt_at(reach, -3.70, 128.18) is None
        assert [s["email"] for s, _ in audience(quake, [inside, outside], reach)] == [
            "a@example.org"
        ]

    def test_only_earthquakes_are_decided_this_way_and_no_place_still_means_everything(
        self, reach: dict[str, Any]
    ) -> None:
        everything = {"id": 9, "email": "all@example.org", "place": "", "latitude": None,
                      "longitude": None, "radius_km": 200.0, "min_severity": 1}  # fmt: skip
        flood = {**_quake().to_dict(), "kind": "river_flood"}
        darwin = _person("d@example.org", "Darwin", DARWIN)

        assert audience(flood, [darwin], reach) == []  # 708 km from a flood: not told
        assert audience(_quake().to_dict(), [everything], reach) == [(everything, None)]

    def test_the_message_says_what_usgs_says_of_the_readers_place(
        self, reach: dict[str, Any]
    ) -> None:
        quake = _quake().to_dict()
        darwin = _person("d@example.org", "Darwin", DARWIN)
        tual = _person("t@example.org", "Tual", (-5.64, 132.4))

        reported = shaking_note(quake, darwin, reach)
        estimated = shaking_note(quake, tual, reach)

        assert reported == (
            '5 people near "Darwin" told USGS they felt it (intensity III (weak shaking)).'
        )
        assert estimated == 'USGS estimates intensity IV (light shaking) at "Tual".'
        _, body = notice(quake, darwin, 708.0, reported)
        assert 'About 708 km from "Darwin".\n\n5 people near "Darwin" told USGS' in body
        assert "5 people near" in text_message(quake, darwin, 708.0, reported)
        assert shaking_note(quake, darwin, None) == ""
        assert shaking_note({**quake, "kind": "river_flood"}, darwin, reach) == ""

    def test_the_delivery_pass_uses_the_reach_the_store_keeps(
        self, store: WatchStore, reach: dict[str, Any]
    ) -> None:
        sent: list[tuple[str, str]] = []

        class _Mail:
            def send(self, *, to: str, subject: str, body: str, **_: Any) -> None:
                sent.append((to, body))

        store.add_subscriber(
            "d@example.org", place="Darwin", latitude=DARWIN[0], longitude=DARWIN[1]
        )
        store.report(_quake(), NOW)
        notifier = Notifier(store, mailer=_Mail(), clock=lambda: NOW)

        assert notifier.sync() == {"sent": 0, "failed": 0}  # USGS has published nothing yet
        store.save_reach(EVENT, reach, NOW + timedelta(minutes=20))
        assert notifier.sync() == {"sent": 1, "failed": 0}  # its reports reach Darwin

        ((to, body),) = sent
        assert to == "d@example.org" and "told USGS they felt it" in body
        assert notifier.sync() == {"sent": 0, "failed": 0}

    def test_the_reach_is_kept_beside_the_finding_not_in_what_the_route_serves(
        self, store: WatchStore, reach: dict[str, Any]
    ) -> None:
        store.report(_quake(), NOW)
        store.save_reach(EVENT, reach, NOW)

        (served,) = store.findings(now=NOW)

        assert "grid" not in json.dumps(served) and store.reaches()[EVENT]["grid"]["nx"] == 55
        assert store.reach_read_at(EVENT) == NOW and store.reach_read_at("other") is None
        store.save_reach(
            "later", {"event": "later", "grid": None, "cells": []}, NOW + timedelta(days=15)
        )
        assert list(store.reaches()) == ["later"]  # a fortnight on, the old one is dropped


class TestTheFeedReadsIt:
    @staticmethod
    def _feed(
        store: WatchStore, monkeypatch: pytest.MonkeyPatch, fail: set[str] | None = None
    ) -> tuple[EarthquakeFeed, list[str]]:
        from omni_mercury_engine.watch import shaking

        read: list[str] = []

        def fake(event_id: str) -> dict[str, Any]:
            read.append(event_id)
            if event_id in (fail or set()):
                raise ConnectionError("simulated: 503")
            return {"event": event_id, "grid": None, "cells": []}

        monkeypatch.setattr(shaking, "read_reach", fake)
        feed = EarthquakeFeed()
        feed.bind(store)
        return feed, read

    def test_it_is_read_for_reportable_earthquakes_and_for_those_still_standing(
        self, store: WatchStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        feed, read = self._feed(store, monkeypatch)
        store.report(_quake(), NOW - timedelta(hours=3))  # left the hourly feed; still standing
        frame = pd.DataFrame({"event_id": ["small", "big"], "magnitude": [3.1, 5.2]})

        assert feed._refresh_reach(frame, NOW) == 2
        assert read == ["big", EVENT]  # not the magnitude 3.1
        assert feed._refresh_reach(frame, NOW + timedelta(minutes=5)) == 0  # read ten minutes apart
        assert feed._refresh_reach(frame, NOW + timedelta(minutes=11)) == 2

    def test_it_is_read_every_ten_minutes_at_first_and_hourly_once_the_reports_are_in(
        self, store: WatchStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Was: every ten minutes for the whole day it stands, 144 reads an earthquake.

        Measured on 385 earthquakes: 93% of first ShakeMaps and most felt
        reports arrive within three hours (research/watch_thresholds).
        """
        feed, read = self._feed(store, monkeypatch)
        store.report(_quake(), NOW)  # happened at NOW
        nothing = pd.DataFrame()

        times = [10 * n for n in range(0, 6 * 24 + 1)]  # a read every ten minutes for a day
        for minutes in times:
            feed._refresh_reach(nothing, NOW + timedelta(minutes=minutes))

        first_three_hours = sum(1 for m in times if m < 180)
        assert len(read) == first_three_hours + 21  # 18 reads, then one an hour to the day's end
        assert len(read) == 39

    def test_one_earthquakes_products_failing_costs_no_other_and_not_the_read(
        self, store: WatchStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        feed, read = self._feed(store, monkeypatch, fail={"bad"})
        frame = pd.DataFrame({"event_id": ["bad", "good"], "magnitude": [6.0, 6.1]})

        assert feed._refresh_reach(frame, NOW) == 1
        assert read == ["bad", "good"] and list(store.reaches()) == ["good"]

    def test_without_a_store_nothing_is_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from omni_mercury_engine.watch import shaking

        monkeypatch.setattr(shaking, "read_reach", lambda event: pytest.fail("no store, no read"))

        assert (
            EarthquakeFeed()._refresh_reach(
                pd.DataFrame({"event_id": ["x"], "magnitude": [7.0]}), NOW
            )
            == 0
        )
