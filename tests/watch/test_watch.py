# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The watch: Mercury reads its live feeds, judges them and says what matters.

Before it, every live source could be read and every batch scored, and nothing
did either unless called. Scored against themselves, the feeds as they stood
on 2026-11-16 flagged 1,036 tsunami-buoy readings, 1,178 hurricane rows and 99
river-gauge readings on an ordinary day: about one row in twenty, by
construction. And a "real-time" read was cached for an hour (a day, for the
buoys), so anything polling saw old data.

These tests feed each feed frames shaped like its loader's and pin: what is a
finding and what is not, that a finding is reported once, that a feed that
fails does not stop the others, and that every read reaches the source.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from omni_mercury_engine.watch import FEEDS, Severity, Watcher, WatchStore
from omni_mercury_engine.watch.feeds import (
    DeclarationFeed,
    EarthquakeFeed,
    Feed,
    Judgment,
    RiverGaugeFeed,
    SpaceWeatherFeed,
    TsunamiBuoyFeed,
    VolcanoFeed,
)
from omni_mercury_engine.watch.model import Finding
from omni_mercury_engine.watch.voice import age, brief, status_line

if TYPE_CHECKING:
    from pathlib import Path

NOW = datetime(2026, 11, 16, 21, 0, tzinfo=UTC)


@pytest.fixture()
def store(tmp_path: Path) -> WatchStore:
    return WatchStore(tmp_path)


def _quakes(*magnitudes: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [
                int((NOW - timedelta(minutes=10 * i)).timestamp() * 1000)
                for i in range(len(magnitudes))
            ],
            "latitude": [38.3] * len(magnitudes),
            "longitude": [142.4] * len(magnitudes),
            "depth": [29.0] * len(magnitudes),
            "magnitude": list(magnitudes),
            "place": ["130 km E of Sendai, Japan"] * len(magnitudes),
            "event_id": [f"us{i:04d}" for i in range(len(magnitudes))],
        }
    )


# --------------------------------------------------------------------------- #
# Judgment
# --------------------------------------------------------------------------- #


class TestEarthquakes:
    def test_an_ordinary_hour_is_nothing_notable(self) -> None:
        """Was: a batch of small quakes scored against itself always flagged some."""
        judgment = EarthquakeFeed().judge(_quakes(1.6, 0.9, 0.7, 2.4, 3.9), NOW)

        assert judgment.findings == []
        assert "largest magnitude 3.9" in judgment.summary
        assert "none at magnitude 4.5 or above" in judgment.summary
        assert judgment.readings == 5

    @pytest.mark.parametrize(
        ("magnitude", "severity", "word"),
        [
            (4.6, Severity.NOTABLE, "light"),
            (5.8, Severity.NOTABLE, "moderate"),
            (6.3, Severity.SIGNIFICANT, "strong"),
            (7.4, Severity.SEVERE, "major"),
            (9.1, Severity.SEVERE, "great"),
        ],
    )
    def test_severity_follows_the_usgs_magnitude_class(
        self, magnitude: float, severity: Severity, word: str
    ) -> None:
        (finding,) = EarthquakeFeed().judge(_quakes(magnitude, 1.2), NOW).findings

        assert finding.severity is severity
        assert f"magnitude {magnitude:.1f} ({word})" in finding.detail
        assert (
            finding.headline == f"Magnitude {magnitude:.1f} earthquake, 130 km E of Sendai, Japan"
        )
        assert "preliminary" in finding.detail  # says what it cannot tell
        assert finding.latitude == 38.3 and finding.value == magnitude
        assert finding.ongoing is False

    def test_one_earthquake_is_one_earthquake(self) -> None:
        """Was: "1 earthquakes in the past hour"."""
        assert (
            EarthquakeFeed()
            .judge(_quakes(0.9), NOW)
            .summary.startswith("1 earthquake in the past hour, the largest magnitude 0.9;")
        )

    def test_no_earthquakes_is_said_not_raised(self) -> None:
        judgment = EarthquakeFeed().judge(_quakes(), NOW)

        assert judgment.findings == [] and "no earthquake" in judgment.summary


class TestSpaceWeather:
    @staticmethod
    def _frame(*kp: float) -> pd.DataFrame:
        stamps = [NOW - timedelta(hours=3 * (len(kp) - 1 - i)) for i in range(len(kp))]
        return pd.DataFrame(
            {"timestamp": stamps, "kp": list(kp), "solar_wind_speed": [640.0] * len(kp)}
        )

    def test_quiet_conditions_say_how_far_from_a_storm(self) -> None:
        judgment = SpaceWeatherFeed().judge(self._frame(1.0, 0.67), NOW)

        assert judgment.findings == []
        assert (
            judgment.summary == "Kp 0.7, below storm level (G1 begins at Kp 5); solar wind 640 km/s"
        )

    @pytest.mark.parametrize(
        ("kp", "level", "severity"),
        [(5.0, 1, Severity.NOTABLE), (6.33, 2, Severity.NOTABLE), (7.0, 3, Severity.SIGNIFICANT),
         (8.33, 4, Severity.SEVERE), (9.0, 5, Severity.SEVERE)],
    )  # fmt: skip
    def test_a_storm_is_named_on_the_noaa_g_scale(
        self, kp: float, level: int, severity: Severity
    ) -> None:
        (finding,) = SpaceWeatherFeed().judge(self._frame(2.0, kp), NOW).findings

        assert finding.severity is severity
        assert finding.headline.startswith(f"G{level} ")
        assert finding.ongoing is True and finding.subject == "planetary"

    def test_the_latest_reading_decides_not_the_largest(self) -> None:
        """A storm that has passed is not a standing finding."""
        assert SpaceWeatherFeed().judge(self._frame(8.0, 3.0), NOW).findings == []

    def test_a_missing_latest_kp_falls_back_to_the_last_measured(self) -> None:
        judgment = SpaceWeatherFeed().judge(self._frame(6.0, float("nan")), NOW)

        assert len(judgment.findings) == 1


class TestTsunamiBuoys:
    @staticmethod
    def _frame(event_station: str | None, hours_ago: float = 1.0) -> pd.DataFrame:
        rows = []
        for station in ("21418", "46407"):
            for step in range(40):
                rows.append((NOW - timedelta(minutes=15 * step), 5662.8, 1, station))
        if event_station:
            begun = NOW - timedelta(hours=hours_ago)
            for second in range(0, 600, 15):
                rows.append((begun + timedelta(seconds=second), 5662.8 + 0.4 * np.sin(second), 2,
                             event_station))  # fmt: skip
        return pd.DataFrame(rows, columns=["timestamp", "bpr", "sample_type", "station_id"])

    def test_buoys_at_their_standard_cadence_are_nothing_notable(self) -> None:
        """Was: 1,036 of 32,845 standard readings flagged on a quiet day."""
        judgment = TsunamiBuoyFeed().judge(self._frame(None), NOW)

        assert judgment.findings == []
        assert judgment.summary == "2 buoys reporting; none in event mode"

    def test_a_buoy_in_event_mode_is_reported_as_what_it_is(self) -> None:
        (finding,) = TsunamiBuoyFeed().judge(self._frame("46407"), NOW).findings

        assert finding.headline == "DART buoy 46407 is in event mode"
        assert finding.severity is Severity.NOTABLE
        assert "not a tsunami warning" in finding.detail
        assert "a test" in finding.detail  # the causes it cannot tell apart

    def test_event_mode_days_ago_is_not_now(self) -> None:
        """The NDBC file holds 45 days; an event last month is history."""
        assert TsunamiBuoyFeed().judge(self._frame("46407", hours_ago=60), NOW).findings == []


class TestRiverGauges:
    @staticmethod
    def _frame(*gauges: tuple[str, str, float, str], hours_ago: float = 0.5) -> pd.DataFrame:
        """Gauges as ``(id, category, stage, forecast category)`` plus two ordinary ones."""
        rows = [*gauges, ("QUIET1", "no_flooding", 3.1, ""), ("BARE1", "not_defined", 1.0, "")]
        return pd.DataFrame(
            {
                "gauge_id": [row[0] for row in rows],
                "name": [f"Test River at {row[0]}" for row in rows],
                "state": "Florida",
                "latitude": 28.0,
                "longitude": -81.0,
                "stage": [row[2] for row in rows],
                "unit": "ft",
                "category": [row[1] for row in rows],
                "observed_at": NOW - timedelta(hours=hours_ago),
                "forecast_category": [row[3] for row in rows],
            }
        )

    def test_gauges_below_flood_are_counted_not_reported(self) -> None:
        """Action stage is below flood: the NWS does not call it flooding, so Mercury does not."""
        judgment = RiverGaugeFeed().judge(self._frame(("ACTN1", "action", 9.0, "action")), NOW)

        assert judgment.findings == []
        assert judgment.summary == (
            "3 gauges reporting; none in flood; 1 at action stage, below flood"
        )
        assert judgment.subjects == 3

    @pytest.mark.parametrize(
        ("category", "severity"),
        [
            ("minor", Severity.NOTABLE),
            ("moderate", Severity.SIGNIFICANT),
            ("major", Severity.SEVERE),
        ],
    )
    def test_severity_is_the_nws_flood_category(self, category: str, severity: Severity) -> None:
        (finding,) = (
            RiverGaugeFeed().judge(self._frame(("COCF1", category, 17.09, category)), NOW).findings
        )

        assert finding.severity is severity
        assert finding.headline == f"{category.capitalize()} flooding: Test River at COCF1, Florida"
        assert finding.value == 17.09 and finding.unit == "ft" and finding.ongoing
        assert f"classes as {category} flooding at this gauge" in finding.detail
        assert "forecast reaches" not in finding.detail  # the forecast says nothing more
        assert "flood warnings come from the National Weather Service" in finding.detail

    def test_a_forecast_of_worse_is_said(self) -> None:
        (finding,) = (
            RiverGaugeFeed().judge(self._frame(("ASTF1", "minor", 2.62, "moderate")), NOW).findings
        )

        assert "Its forecast reaches moderate flooding." in finding.detail

    def test_the_summary_counts_each_category(self) -> None:
        judgment = RiverGaugeFeed().judge(
            self._frame(
                ("A1", "major", 9.0, ""), ("B1", "minor", 4.0, ""), ("C1", "minor", 5.0, "")
            ),
            NOW,
        )

        assert judgment.summary == "5 gauges reporting; 3 in flood (1 major, 2 minor)"
        assert [finding.subject for finding in judgment.findings] == ["A1", "B1", "C1"]

    def test_a_reading_hours_old_is_not_the_rivers_present_state(self) -> None:
        """Was not possible to tell: a gauge that stopped reporting in flood stays 'major'."""
        judgment = RiverGaugeFeed().judge(
            self._frame(("OLD1", "major", 30.0, ""), hours_ago=9), NOW
        )

        assert judgment.findings == []
        assert judgment.summary == "no gauge has a current reading"
        assert judgment.latest_reading == NOW - timedelta(hours=9)

    def test_a_gauge_with_no_reading_at_all_is_not_counted(self) -> None:
        frame = self._frame(("COCF1", "major", 17.09, ""))
        frame.loc[frame["gauge_id"] == "BARE1", "observed_at"] = pd.NaT

        judgment = RiverGaugeFeed().judge(frame, NOW)

        assert judgment.summary == "2 gauges reporting; 1 in flood (1 major)"

    def test_a_flood_that_worsens_is_reported_again_and_one_that_recedes_ends(
        self, store: WatchStore
    ) -> None:
        feed = RiverGaugeFeed()
        said: list[Finding] = []
        moment = [NOW]
        watcher = Watcher(store, [feed], on_finding=said.append, clock=lambda: moment[0])
        frames = iter(
            [
                self._frame(("COCF1", "minor", 15.0, "")),
                self._frame(("COCF1", "major", 17.1, ""), hours_ago=-0.5),
                self._frame(("COCF1", "no_flooding", 12.0, ""), hours_ago=-1.5),
            ]
        )
        feed.read = lambda: next(frames)  # type: ignore[method-assign]

        for hours in range(3):
            moment[0] = NOW + timedelta(hours=hours)
            watcher.read_feed(feed)

        assert [finding.headline.split(":")[0] for finding in said] == [
            "Minor flooding",
            "Major flooding",
        ]
        assert store.findings()[0]["ended"] == (NOW + timedelta(hours=2)).isoformat()


class TestVolcanoes:
    @staticmethod
    def _frame(*levels: str) -> pd.DataFrame:
        colours = {"NORMAL": "GREEN", "ADVISORY": "YELLOW", "WATCH": "ORANGE", "WARNING": "RED"}
        numeric = {"NORMAL": 0, "ADVISORY": 1, "WATCH": 2, "WARNING": 3, "UNASSIGNED": 0}
        return pd.DataFrame(
            {
                "volcano_name": [f"Volcano {i}" for i in range(len(levels))],
                "alert_level": list(levels),
                "color_code": [colours.get(level, "") for level in levels],
                "alert_level_numeric": [numeric[level] for level in levels],
                "color_code_numeric": [numeric[level] for level in levels],
                "alert_date": "2026-11-13 21:04:48",
                "volcano_number": [str(311120 + i) for i in range(len(levels))],
                "observatory": "Alaska Volcano Observatory",
                "notice_id": "DOI-USGS-AVO-2026-11-13T19:12:01+00:00",
                "latitude": 52.0765,
                "longitude": -176.1109,
                "elevation": 1740.0,
            }
        )

    def test_volcanoes_at_normal_are_counted_not_reported(self) -> None:
        judgment = VolcanoFeed().judge(self._frame("NORMAL", "NORMAL", "UNASSIGNED"), NOW)

        assert judgment.findings == []
        assert judgment.summary == "3 monitored volcanoes; all at NORMAL"
        assert judgment.latest_reading is None  # a quiet volcano's last notice can be years old

    @pytest.mark.parametrize(
        ("level", "severity", "meaning"),
        [
            ("ADVISORY", Severity.NOTABLE, "elevated unrest above its known background"),
            ("WATCH", Severity.SIGNIFICANT, "heightened or escalating unrest"),
            ("WARNING", Severity.SEVERE, "hazardous eruption that is imminent, under way"),
        ],
    )
    def test_severity_is_the_usgs_volcano_alert_level(
        self, level: str, severity: Severity, meaning: str
    ) -> None:
        (finding,) = VolcanoFeed().judge(self._frame("NORMAL", level), NOW).findings

        assert finding.severity is severity and finding.ongoing
        assert finding.headline == f"Volcano 1 volcano: USGS alert level {level}"
        assert finding.subject == "311121" and finding.value == float(severity)
        assert finding.reference == "DOI-USGS-AVO-2026-11-13T19:12:01+00:00"
        assert (finding.latitude, finding.longitude) == (52.0765, -176.1109)
        assert "by the notice of 21:04 UTC on 13 November 2026 (Alaska Volcano Observatory)" in (
            finding.detail
        )
        assert meaning in finding.detail and "does not assess the volcano" in finding.detail
        assert finding.observed_at == datetime(2026, 11, 13, 21, 4, 48, tzinfo=UTC)

    def test_the_summary_counts_each_level(self) -> None:
        judgment = VolcanoFeed().judge(self._frame("ADVISORY", "WATCH", "ADVISORY", "NORMAL"), NOW)

        assert judgment.summary == "4 monitored volcanoes; 3 above NORMAL (1 WATCH, 2 ADVISORY)"

    def test_a_frame_without_the_notice_columns_filled_still_reads(self) -> None:
        """A response that carries no observatory, number or date is still a level."""
        frame = self._frame("WATCH").assign(
            observatory="", volcano_number="", alert_date="", latitude=np.nan
        )

        (finding,) = VolcanoFeed().judge(frame, NOW).findings

        assert finding.subject == "Volcano 0" and finding.observed_at == NOW
        assert finding.latitude is None and "by the notice of" not in finding.detail


class TestDeclarations:
    @staticmethod
    def _frame(recent: int, earlier_peak: int = 40) -> pd.DataFrame:
        days = pd.date_range("2026-01-01", NOW.date(), freq="D", tz="UTC")
        counts = np.zeros(len(days), dtype=int)
        counts[200] = earlier_peak
        counts[-2] = recent
        return pd.DataFrame(
            {"timestamp": days, "designations": counts, "states": (counts > 0) * 2,
             "disasters": (counts > 0) * 1}
        )  # fmt: skip

    def test_no_recent_declaration_is_nothing_notable(self) -> None:
        judgment = DeclarationFeed().judge(self._frame(0), NOW)

        assert judgment.findings == []
        assert (
            judgment.summary
            == "40 tropical-cyclone designations this year; none in the past 3 days"
        )

    def test_new_declarations_are_reported(self) -> None:
        (finding,) = DeclarationFeed().judge(self._frame(12), NOW).findings

        assert finding.severity is Severity.NOTABLE
        assert finding.headline == "FEMA declared 12 county designations for tropical cyclones"
        assert "not the storm's arrival" in finding.detail

    def test_the_busiest_week_of_the_year_is_significant(self) -> None:
        (finding,) = DeclarationFeed().judge(self._frame(107), NOW).findings

        assert finding.severity is Severity.SIGNIFICANT
        assert finding.headline.endswith("the busiest week of the year")


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #


def _finding(
    severity: Severity = Severity.NOTABLE, *, ongoing: bool = True, feed: str = "space_weather"
) -> Finding:
    return Finding(
        feed=feed, kind="geomagnetic_storm", subject="planetary", severity=severity,
        headline="G1 storm", detail="detail", observed_at=NOW, source="NOAA", basis="G scale",
        ongoing=ongoing,
    )  # fmt: skip


class TestStore:
    def test_a_finding_is_new_once_then_seen_again_then_can_escalate(
        self, store: WatchStore
    ) -> None:
        assert store.report(_finding(), NOW) is True
        assert store.report(_finding(), NOW + timedelta(minutes=15)) is False
        assert store.report(_finding(Severity.SEVERE), NOW + timedelta(minutes=30)) is True
        assert store.report(_finding(Severity.NOTABLE), NOW + timedelta(minutes=45)) is False

        (recorded,) = store.findings()
        assert recorded["first_seen"] == NOW.isoformat()
        assert recorded["last_seen"] == (NOW + timedelta(minutes=45)).isoformat()

    def test_an_escalation_is_checked_afresh(self, store: WatchStore) -> None:
        """Was: 'confirmed at minor flooding' would have stood beside 'major flooding'."""
        from omni_mercury_engine.watch.model import Outcome

        store.report(_finding(), NOW)
        store.record_outcome(_finding().key, Outcome("confirmed", "n", "r", final=True), NOW)

        store.report(_finding(Severity.NOTABLE), NOW + timedelta(minutes=30))  # seen again
        assert store.findings()[0]["outcome"]["verdict"] == "confirmed"

        store.report(_finding(Severity.SEVERE), NOW + timedelta(hours=1))
        (finding,) = store.findings()
        assert finding["outcome"] is None and finding["reviewed_at"] is None
        assert len(store.awaiting_review("space_weather", NOW + timedelta(hours=2), NOW)) == 1

    def test_a_condition_that_clears_ends_and_can_recur(self, store: WatchStore) -> None:
        store.report(_finding(), NOW)

        assert store.close_absent("space_weather", set(), NOW + timedelta(hours=3)) == 1
        assert store.findings()[0]["ended"] == (NOW + timedelta(hours=3)).isoformat()
        assert store.findings(open_only=True) == []
        assert store.report(_finding(), NOW + timedelta(hours=9)) is True  # a new storm

    def test_a_single_occurrence_is_not_ended_by_the_next_read(self, store: WatchStore) -> None:
        """An earthquake leaves the hourly feed; it did not stop having happened."""
        store.report(_finding(ongoing=False), NOW)

        assert store.close_absent("space_weather", set(), NOW + timedelta(hours=2)) == 0

    def test_findings_are_filtered_by_feed_and_by_open_in_any_combination(
        self, store: WatchStore
    ) -> None:
        """One fixed statement serves every filter (it was assembled per call)."""
        storm = _finding()
        quake = Finding(
            feed="earthquakes", kind="earthquake", subject="us7000abcd",
            severity=Severity.SIGNIFICANT, headline="M 6.4 earthquake", detail="detail",
            observed_at=NOW, source="USGS", basis="magnitude class", ongoing=False,
        )  # fmt: skip
        flood = Finding(
            feed="river_gauges", kind="river_flood", subject="COCF1", severity=Severity.SEVERE,
            headline="Major flooding", detail="detail", observed_at=NOW, source="NWS",
            basis="NWS flood category",
        )  # fmt: skip
        store.report(storm, NOW)
        store.report(quake, NOW + timedelta(minutes=1))
        store.report(flood, NOW + timedelta(minutes=2))
        store.close_absent("space_weather", set(), NOW + timedelta(hours=3))

        def keys(**filters: Any) -> list[str]:
            return [finding["key"] for finding in store.findings(**filters)]

        assert keys() == [flood.key, quake.key, storm.key]  # newest first
        assert keys(open_only=True) == [flood.key, quake.key]
        assert keys(feed="space_weather") == [storm.key]
        assert keys(feed="space_weather", open_only=True) == []
        assert keys(feed="river_gauges", open_only=True) == [flood.key]
        assert keys(feed="no_such_feed") == []
        assert keys(limit=1) == [flood.key]
        assert keys(feed="earthquakes", limit=0) == []
        # A feed name is data, never SQL.
        assert keys(feed="x' OR '1'='1") == []


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #


class _Scripted(Feed):
    name = "scripted"
    title = "Scripted feed"
    source = "test"
    cadence = timedelta(minutes=5)
    stale_after = timedelta(hours=1)

    def __init__(self, judgments: list[Judgment | Exception]) -> None:
        self._judgments = judgments
        self.reads = 0

    def read(self) -> pd.DataFrame:
        self.reads += 1
        return pd.DataFrame()

    def judge(self, frame: pd.DataFrame, now: datetime) -> Judgment:
        outcome = self._judgments[min(self.reads, len(self._judgments)) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class TestWatcher:
    def test_a_finding_is_announced_once_across_reads(self, store: WatchStore) -> None:
        announced: list[str] = []
        storm = Judgment([_finding(feed="scripted")], "Kp 5.3, G1", NOW, 1, 1)
        feed = _Scripted([storm, storm, storm])
        watcher = Watcher(store, [feed], on_finding=lambda f: announced.append(f.headline),
                          clock=lambda: NOW)  # fmt: skip

        statuses = [watcher.run_once()[0] for _ in range(3)]

        assert announced == ["G1 storm"]
        assert [status.new_findings for status in statuses] == [1, 0, 0]
        assert statuses[-1].open_findings == 1 and statuses[-1].state == "ok"

    def test_a_feed_that_fails_is_down_and_the_others_are_still_read(
        self, store: WatchStore
    ) -> None:
        broken = _Scripted([ConnectionError("simulated: unreachable")])
        working = _Scripted([Judgment(summary="nothing notable", latest_reading=NOW)])
        working.name = "working"  # type: ignore[misc]
        watcher = Watcher(store, [broken, working], clock=lambda: NOW)

        down, ok = watcher.run_once()

        assert down.state == "down" and "ConnectionError" in down.error
        assert ok.state == "ok"
        assert {status["feed"]: status["state"] for status in store.statuses()} == {
            "scripted": "down",
            "working": "ok",
        }

    def test_a_store_that_fails_once_does_not_end_the_watch(self, store: WatchStore) -> None:
        """Was: one 'database is locked' ended the loop's thread; the server served on."""
        import sqlite3
        import threading

        storm = Judgment([_finding(feed="scripted")], "Kp 5.3, G1", NOW, 1, 1)
        feed = _Scripted([storm, storm])
        feed.cadence = timedelta(0)  # type: ignore[misc]
        report, calls, stop = store.report, [0], threading.Event()

        def locked_once(finding: Finding, now: datetime) -> bool:
            calls[0] += 1
            if calls[0] == 1:
                raise sqlite3.OperationalError("database is locked")
            stop.set()
            return report(finding, now)

        store.report = locked_once  # type: ignore[method-assign]
        Watcher(store, [feed], clock=lambda: NOW).run_forever(tick_seconds=0.0, stop=stop)

        assert calls[0] == 2 and [found["headline"] for found in store.findings()] == ["G1 storm"]

    def test_a_feed_whose_reading_cannot_be_recorded_is_down_and_the_rest_are_read(
        self, store: WatchStore
    ) -> None:
        import sqlite3

        storm = Judgment([_finding(feed="scripted")], "Kp 5.3, G1", NOW, 1, 1)
        working = _Scripted([Judgment(summary="nothing notable", latest_reading=NOW)])
        working.name = "working"  # type: ignore[misc]

        def locked(finding: Finding, now: datetime) -> bool:
            raise sqlite3.OperationalError("database is locked")

        store.report = locked  # type: ignore[method-assign]
        down, ok = Watcher(store, [_Scripted([storm]), working], clock=lambda: NOW).run_once()

        assert down.state == "down" and "database is locked" in down.error
        assert down.summary == "its reading could not be recorded"
        assert ok.state == "ok"
        assert {status["feed"]: status["state"] for status in store.statuses()} == {
            "scripted": "down",
            "working": "ok",
        }

    def test_a_feed_with_only_old_readings_is_stale(self, store: WatchStore) -> None:
        old = Judgment(summary="nothing notable", latest_reading=NOW - timedelta(hours=5))
        (status,) = Watcher(store, [_Scripted([old])], clock=lambda: NOW).run_once()

        assert status.state == "stale"

    def test_a_feed_is_read_at_its_cadence_not_at_every_tick(self, store: WatchStore) -> None:
        moment = [NOW]
        feed = _Scripted([Judgment()])
        watcher = Watcher(store, [feed], clock=lambda: moment[0])

        watcher.run_once(force=False)
        moment[0] = NOW + timedelta(minutes=2)
        watcher.run_once(force=False)
        moment[0] = NOW + timedelta(minutes=6)
        watcher.run_once(force=False)

        assert feed.reads == 2

    def test_a_cleared_condition_is_ended_by_the_read_that_no_longer_shows_it(
        self, store: WatchStore
    ) -> None:
        feed = _Scripted(
            [Judgment([_finding(feed="scripted")], "storm", NOW), Judgment(summary="quiet")]
        )
        watcher = Watcher(store, [feed], clock=lambda: NOW)

        watcher.run_once()
        (status,) = watcher.run_once()

        assert status.open_findings == 0
        assert store.findings()[0]["ended"] is not None


def test_every_feed_reads_fresh_at_its_cadence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Was: a "real-time" read was cached for an hour, the buoys' for a day."""
    monkeypatch.setenv("MERCURY_WATCH_DIR", str(tmp_path))
    loaders = {
        "earthquakes": "EarthquakeLoader",
        "space_weather": "EnergyLoader",
        "tsunami_buoys": "TsunamiLoader",
        "river_gauges": "FloodLoader",
        "volcanoes": "VolcanicLoader",
        "fema_declarations": "FEMAActivityLoader",
    }
    assert set(FEEDS) == {*loaders, "warnings"}  # warnings are read uncached, every time
    for name, loader_name in loaders.items():
        feed = FEEDS[name]()
        loader: Any = feed._loader(loader_name)

        assert feed.cadence.total_seconds() > loader.CACHE_TTL, name
        assert loader.cache_dir == tmp_path / "cache" / name
        assert feed.stale_after > feed.cadence


# --------------------------------------------------------------------------- #
# Voice
# --------------------------------------------------------------------------- #


class TestVoice:
    def test_nothing_notable_is_said_and_every_feed_has_a_line(self) -> None:
        statuses = [
            {"feed": "a", "title": "Earthquakes", "state": "ok", "summary": "4 in the past hour",
             "latest_reading": (NOW - timedelta(minutes=32)).isoformat()},
            {"feed": "b", "title": "Buoys", "state": "down", "error": "ConnectionError: reset"},
            {"feed": "c", "title": "Gauges", "state": "waiting"},
        ]  # fmt: skip

        text = brief(statuses, [], NOW)

        assert text.splitlines()[0] == "Nothing notable on any feed."
        assert "Feeds (1 of 3 answering)" in text
        assert "ok    Earthquakes: 4 in the past hour (latest reading 32 minutes ago)" in text
        assert "down  Buoys: could not be read (ConnectionError: reset)" in text
        assert "wait  Gauges: not read yet" in text

    def test_the_most_severe_standing_finding_comes_first_and_ended_ones_are_left_out(
        self,
    ) -> None:
        def entry(headline: str, severity: str, ended: str | None = None) -> dict[str, Any]:
            return {"headline": headline, "detail": "d", "source": "s", "basis": "b",
                    "severity": severity, "first_seen": NOW.isoformat(), "ended": ended}  # fmt: skip

        text = brief(
            [],
            [entry("minor", "notable"), entry("major", "severe"),
             entry("over", "severe", NOW.isoformat())],
            NOW,
        )  # fmt: skip

        assert text.startswith("2 findings standing")
        assert text.index("!!! major") < text.index("!   minor")
        assert "over" not in text

    def test_ages_read_as_a_person_would_say_them(self) -> None:
        assert age((NOW - timedelta(seconds=20)).isoformat(), NOW) == "just now"
        assert age((NOW - timedelta(minutes=1)).isoformat(), NOW) == "1 minute ago"
        assert age((NOW - timedelta(hours=5, minutes=59)).isoformat(), NOW) == "5 hours ago"
        assert age((NOW - timedelta(days=3)).isoformat(), NOW) == "3 days ago"
        assert age(None, NOW) == "never"
        assert status_line({"title": "X", "state": "stale", "summary": "s"}).startswith(
            "stale X: s"
        )
