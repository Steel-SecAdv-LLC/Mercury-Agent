# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Feeds from outside the United States: national authorities' alerts, and GDACS.

Outside the United States a subscriber was told of earthquakes, tsunami
buoys and geomagnetic storms, and of nothing their own weather service had
warned of.

The files under ``tests/watch/data`` are what the services returned on
2026-11-17: the WMO Severe Weather Information Centre's map service for a
point in Zurich and one in Cocoa, Florida; the CAP documents of a MeteoSwiss
gale warning (five languages) and a Deutscher Wetterdienst wind warning
(1,205 areas, of which six are kept); and GDACS's Orange and Red events.
Polygon vertices, which the parser does not read, are removed from the CAP
recordings.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from omni_mercury_engine.watch import FEEDS, Severity, WatchStore
from omni_mercury_engine.watch.guidance import GUIDANCE, with_guidance
from omni_mercury_engine.watch.notify import audience
from omni_mercury_engine.watch.world import GdacsFeed, WorldWarningsFeed, parse_cap

DATA = Path(__file__).parent / "data"
NOW = datetime(2026, 11, 17, 18, 30, tzinfo=UTC)
GALE = "ch-meteoswiss-de/2026/11/17/05/51/55-b6da6800ffd46ee8cd4062cda506296e.xml"


def _text(name: str) -> str:
    return (DATA / name).read_text(encoding="utf-8")


@pytest.fixture()
def store(tmp_path: Path) -> WatchStore:
    return WatchStore(tmp_path)


class TestCap:
    def test_an_alert_is_read_as_its_authority_wrote_it(self) -> None:
        alert = parse_cap(_text("meteoswiss_strong_gale.cap.xml"))

        assert alert["identifier"] == "2.49.0.0.756.0.CH.261117065158c984b8de9d00cda692d2"
        assert (alert["status"], alert["msg_type"]) == ("Actual", "Alert")
        assert (alert["event"], alert["sender"], alert["language"]) == (
            "Strong gale", "MeteoSwiss", "en",
        )  # fmt: skip
        assert (alert["severity"], alert["urgency"], alert["certainty"]) == (
            "Severe", "Future", "Likely",
        )  # fmt: skip
        assert alert["expires"] == "2026-11-18T08:00:00+00:00"
        assert alert["instruction"].startswith("Avoid forests, tree-lined avenues")
        assert alert["area"] == "Thalwil-Wetzikon"

    def test_the_readers_language_is_taken_when_the_alert_has_it(self) -> None:
        document = _text("meteoswiss_strong_gale.cap.xml")

        assert parse_cap(document, prefer="fr")["language"].startswith("fr")
        assert parse_cap(document, prefer="fr")["event"] != "Strong gale"
        # A language it does not have: the first block, never no alert.
        assert parse_cap(document, prefer="ja")["event"]

    def test_an_area_named_only_by_its_polygon_is_not_said_and_a_long_list_is_counted(
        self,
    ) -> None:
        """One German warning names 1,205 municipalities; an email is not a gazetteer."""
        alert = parse_cap(_text("dwd_wind_gusts.cap.xml"))

        assert alert["sender"] == "Deutscher Wetterdienst"
        assert "polygonal event area" not in alert["area"]
        assert alert["area"].split("; ")[:2] == ["Gemeinde Nordhalben", "Stadt Roßwein"]
        many = "<alert><info>" + "".join(
            f"<area><areaDesc>Town {n}</areaDesc></area>" for n in range(40)
        ) + "</info></alert>"  # fmt: skip
        assert parse_cap(many)["area"].endswith("Town 7; and 32 more areas")

    @pytest.mark.parametrize(
        "document",
        ["not xml at all", "<feed/>", "<alert><identifier>x</identifier></alert>"],
    )
    def test_what_is_not_a_cap_alert_is_refused(self, document: str) -> None:
        with pytest.raises(ValueError, match=r"XML|CAP"):
            parse_cap(document)

    def test_an_entity_bomb_is_not_expanded(self) -> None:
        bomb = (
            '<?xml version="1.0"?><!DOCTYPE a [<!ENTITY x "xxxxxxxxxx">'
            '<!ENTITY y "&x;&x;&x;&x;&x;&x;&x;&x;">]><alert><info><event>&y;</event></info></alert>'
        )
        with pytest.raises(ValueError, match="XML"):
            parse_cap(bomb)


class TestWorldWarnings:
    @staticmethod
    def _feed(
        store: WatchStore,
        monkeypatch: pytest.MonkeyPatch,
        covering: dict[tuple[float, float], Any],
        documents: dict[str, Any],
    ) -> tuple[WorldWarningsFeed, list[str]]:
        feed, fetched = WorldWarningsFeed(), []
        feed.bind(store)

        def cover(latitude: float, longitude: float) -> list[str]:
            answer = covering.get((latitude, longitude), [])
            if isinstance(answer, Exception):
                raise answer
            return list(answer)

        def document(path: str) -> str:
            fetched.append(path)
            answer = documents[path]
            if isinstance(answer, Exception):
                raise answer
            return str(answer)

        monkeypatch.setattr(feed, "_covering", cover)
        monkeypatch.setattr(feed, "_document", document)
        return feed, fetched

    def test_the_map_services_answer_is_read_for_its_alerts_paths(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from omni_mercury_engine.security.safe_http import SafeHTTPClient

        asked: list[dict[str, Any]] = []

        def get_json(url: str, **kwargs: Any) -> Any:
            asked.append({"url": url, **kwargs})
            return json.loads(_text("wmo_wfs_cocoa.json"))

        monkeypatch.setattr(SafeHTTPClient, "get_json", staticmethod(get_json))

        paths = WorldWarningsFeed()._covering(28.37, -80.87)

        (call,) = asked
        assert call["url"] == "https://severeweather.wmo.int/f/wfs"
        assert call["params"]["cql_filter"] == (
            "INTERSECTS(wkb_geometry, POINT(-80.87 28.37)) AND row_type <> 'BOUNDARY'"
        )
        assert call["params"]["version"] == "1.0.0"  # longitude first
        # The warning has a polygon and a multipolygon row: one alert, once.
        assert len(paths) == 2 and all(path.startswith("us-noaa-nws-en/") for path in paths)

    def test_a_path_that_is_not_a_cap_documents_is_never_fetched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from omni_mercury_engine.security.safe_http import SafeHTTPClient

        hostile = {
            "features": [
                {"properties": {"capurl": "../../etc/passwd"}},
                {"properties": {"capurl": "https://evil.example/x.xml"}},
                {"properties": {"capurl": "ch-meteoswiss-de/2026/11/17/05/51/../../x.xml"}},
                {"properties": {"capurl": GALE}},
                {"properties": {}},
            ]
        }
        monkeypatch.setattr(SafeHTTPClient, "get_json", staticmethod(lambda url, **_: hostile))

        assert WorldWarningsFeed()._covering(47.37, 8.54) == [GALE]

    def test_an_alert_in_effect_at_a_subscribers_place_is_a_finding_for_them(
        self, store: WatchStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        zurich = store.add_subscriber(
            "z@example.org", place="Zurich", latitude=47.37, longitude=8.54
        )
        store.add_subscriber("t@example.org", place="Tokyo", latitude=35.68, longitude=139.69)
        feed, fetched = self._feed(
            store, monkeypatch, {(47.37, 8.54): [GALE]},
            {GALE: _text("meteoswiss_strong_gale.cap.xml")},
        )  # fmt: skip

        judgment = feed.judge(feed.read(), NOW)
        feed.judge(feed.read(), NOW)

        (finding,) = judgment.findings
        assert finding.headline == "Strong gale: Thalwil-Wetzikon"
        assert finding.severity is Severity.SIGNIFICANT and finding.place_ids == (zurich,)
        assert finding.source == "MeteoSwiss" and finding.action_source == "MeteoSwiss"
        assert finding.action.startswith("Avoid forests")  # the authority's words
        assert "In effect until 08:00 UTC on 18 November 2026" in finding.detail
        assert "MeteoSwiss rates it severe, future, likely" in finding.detail
        assert judgment.summary == "1 alert in effect at 2 subscribed places checked"
        assert judgment.partial is False
        assert fetched == [GALE]  # a published alert is read once

    def test_it_goes_to_the_subscriber_there_and_to_no_one_else(
        self, store: WatchStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store.add_subscriber("z@example.org", place="Zurich", latitude=47.37, longitude=8.54)
        store.add_subscriber("t@example.org", place="Tokyo", latitude=35.68, longitude=139.69)
        store.add_subscriber("all@example.org")
        feed, _ = self._feed(
            store, monkeypatch, {(47.37, 8.54): [GALE]},
            {GALE: _text("meteoswiss_strong_gale.cap.xml")},
        )  # fmt: skip
        (finding,) = feed.judge(feed.read(), NOW).findings

        told = [s["email"] for s, _ in audience(finding.to_dict(), store.subscribers())]

        assert told == ["z@example.org", "all@example.org"]

    def test_the_united_states_alerts_are_left_to_the_services_own_feed(
        self, store: WatchStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The centre carries them too; relayed from both they would be said twice."""
        store.add_subscriber("c@example.org", place="Cocoa", latitude=28.37, longitude=-80.87)
        paths = sorted(
            {
                feature["properties"]["capurl"]
                for feature in json.loads(_text("wmo_wfs_cocoa.json"))["features"]
            }
        )
        feed, fetched = self._feed(store, monkeypatch, {(28.37, -80.87): paths}, {})

        judgment = feed.judge(feed.read(), NOW)

        assert judgment.findings == [] and fetched == []
        assert judgment.summary == "0 alerts in effect at 1 subscribed place checked"

    def test_an_expired_or_cancelled_alert_or_an_exercise_is_not_a_finding(
        self, store: WatchStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store.add_subscriber("z@example.org", place="Zurich", latitude=47.37, longitude=8.54)
        gale = _text("meteoswiss_strong_gale.cap.xml")
        variants = {
            "ch-a-en/2026/11/17/05/51/cancel.xml": gale.replace(
                "<msgType>Alert</msgType>", "<msgType>Cancel</msgType>"
            ),
            "ch-a-en/2026/11/17/05/51/exercise.xml": gale.replace(
                "<status>Actual</status>", "<status>Exercise</status>"
            ),
        }
        feed, _ = self._feed(
            store, monkeypatch, {(47.37, 8.54): [GALE, *variants]}, {GALE: gale, **variants}
        )

        in_effect = feed.judge(feed.read(), NOW)
        after = feed.judge(feed.read(), NOW + timedelta(days=1))

        assert [f.headline for f in in_effect.findings] == ["Strong gale: Thalwil-Wetzikon"]
        assert after.findings == []  # it expired at 08:00 UTC on the 18th

    def test_a_lookup_that_fails_ends_nothing_and_costs_no_other_place_its_alerts(
        self, store: WatchStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store.add_subscriber("z@example.org", place="Zurich", latitude=47.37, longitude=8.54)
        store.add_subscriber("v@example.org", place="Vienna", latitude=48.21, longitude=16.37)
        feed, _ = self._feed(
            store, monkeypatch,
            {(47.37, 8.54): [GALE], (48.21, 16.37): ConnectionError("simulated: timed out")},
            {GALE: _text("meteoswiss_strong_gale.cap.xml")},
        )  # fmt: skip

        judgment = feed.judge(feed.read(), NOW)

        assert len(judgment.findings) == 1 and judgment.partial is True
        assert judgment.summary == (
            "1 alert in effect at 1 subscribed place checked; "
            "1 place could not be checked this time"
        )

    def test_an_alert_that_cannot_be_read_is_said_and_the_read_is_partial(
        self, store: WatchStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store.add_subscriber("z@example.org", place="Zurich", latitude=47.37, longitude=8.54)
        feed, _ = self._feed(
            store, monkeypatch, {(47.37, 8.54): [GALE]}, {GALE: ConnectionError("simulated: 503")}
        )

        judgment = feed.judge(feed.read(), NOW)

        assert judgment.findings == [] and judgment.partial is True
        assert judgment.summary.endswith("1 alert could not be read this time")

    def test_every_lookup_failing_is_down_and_no_place_subscribed_is_said(
        self, store: WatchStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        feed, _ = self._feed(store, monkeypatch, {}, {})
        assert feed.judge(feed.read(), NOW).summary == (
            "no place is subscribed; alerts are looked up for subscribed places"
        )

        store.add_subscriber("z@example.org", place="Zurich", latitude=47.37, longitude=8.54)
        feed, _ = self._feed(
            store, monkeypatch, {(47.37, 8.54): ConnectionError("simulated: 503")}, {}
        )
        with pytest.raises(ConnectionError, match="all 1 subscribed places"):
            feed.read()

    def test_an_alert_without_an_instruction_says_whom_to_follow(
        self, store: WatchStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store.add_subscriber("z@example.org", place="Zurich", latitude=47.37, longitude=8.54)
        import re

        bare = re.sub(
            r"<instruction>.*?</instruction>",
            "",
            _text("meteoswiss_strong_gale.cap.xml"),
            flags=re.S,
        )
        feed, _ = self._feed(store, monkeypatch, {(47.37, 8.54): [GALE]}, {GALE: bare})

        finding = with_guidance(feed.judge(feed.read(), NOW).findings[0])

        assert finding.action == GUIDANCE["cap_alert"][0]
        assert "authority that issued this alert" in finding.action


class TestGdacs:
    @staticmethod
    def _frame(
        monkeypatch: pytest.MonkeyPatch, payload: Any = None
    ) -> tuple[Any, list[dict[str, Any]]]:
        from omni_mercury_engine.security.safe_http import SafeHTTPClient

        asked: list[dict[str, Any]] = []
        answer = payload if payload is not None else json.loads(_text("gdacs_orange_red.json"))

        def get_json(url: str, **kwargs: Any) -> Any:
            asked.append({"url": url, **kwargs})
            return answer

        monkeypatch.setattr(SafeHTTPClient, "get_json", staticmethod(get_json))
        return GdacsFeed().read(), asked

    def test_only_orange_and_red_events_of_its_kinds_are_asked_for(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, asked = self._frame(monkeypatch)

        (call,) = asked
        assert call["url"] == "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH"
        assert call["params"]["alertlevel"] == "Orange;Red"
        assert call["params"]["eventlist"] == "TC;FL;VO;WF"  # not earthquakes, not droughts

    def test_a_current_orange_event_is_a_finding_and_one_that_is_over_is_not(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frame, _ = self._frame(monkeypatch)

        judgment = GdacsFeed().judge(frame, NOW)

        (finding,) = judgment.findings  # AKVAN-26 is current; MURASU-26 (Red) ended on the 3rd
        assert finding.headline == "Tropical Cyclone AKVAN-26: GDACS Orange alert"
        assert finding.severity is Severity.SIGNIFICANT and finding.subject == "TC1001351"
        assert (finding.latitude, finding.longitude) == (12.2, 87.3)
        assert "Tropical Storm (maximum wind speed of 83 km/h)" in finding.detail
        assert "they are not a public warning" in finding.detail
        assert finding.reference.startswith("https://www.gdacs.org/report.aspx?eventid=1001351")
        assert judgment.summary == "1 current event rated Orange or Red (0 red, 1 orange)"
        assert judgment.latest_reading == datetime(2026, 11, 17, 18, 20, 34, tzinfo=UTC)

    def test_red_is_severe_and_it_goes_to_subscribers_within_their_radius(
        self, store: WatchStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = json.loads(_text("gdacs_orange_red.json"))
        for feature in payload["features"]:
            feature["properties"].update(alertlevel="Red", iscurrent="true")
        frame, _ = self._frame(monkeypatch, payload)
        store.add_subscriber(
            "near@example.org", place="Port Blair", latitude=11.62, longitude=92.73, radius_km=700
        )
        store.add_subscriber("far@example.org", place="Delhi", latitude=28.61, longitude=77.21)

        findings = GdacsFeed().judge(frame, NOW).findings

        assert [f.severity for f in findings] == [Severity.SEVERE, Severity.SEVERE]
        akvan = next(f for f in findings if "AKVAN" in f.headline)
        told = [s["email"] for s, _ in audience(akvan.to_dict(), store.subscribers())]
        assert told == ["near@example.org"]

    def test_a_quiet_fortnight_and_a_wrong_shape_are_each_said(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frame, _ = self._frame(monkeypatch, {"type": "FeatureCollection", "features": []})
        assert (
            GdacsFeed().judge(frame, NOW).summary == "no Orange or Red event in the past two weeks"
        )

        with pytest.raises(ValueError, match="features"):
            self._frame(monkeypatch, {"error": "maintenance"})


def test_the_world_feeds_are_feeds_of_the_watch_and_say_what_to_do() -> None:
    assert {"world_warnings", "gdacs"} <= set(FEEDS)
    assert FEEDS["world_warnings"] is WorldWarningsFeed and FEEDS["gdacs"] is GdacsFeed
    for kind in ("cap_alert", "gdacs_event"):
        action, source = GUIDANCE[kind]
        assert len(action) > 40 and source
