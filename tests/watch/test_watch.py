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
    def test_an_ordinary_hour_is_nothing_notable(self, store: WatchStore) -> None:
        """Was: a batch of small quakes scored against itself always flagged some."""
        judgment = EarthquakeFeed().judge(_quakes(1.6, 0.9, 0.7, 2.4, 3.9), store, NOW)

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
        self, store: WatchStore, magnitude: float, severity: Severity, word: str
    ) -> None:
        (finding,) = EarthquakeFeed().judge(_quakes(magnitude, 1.2), store, NOW).findings

        assert finding.severity is severity
        assert f"magnitude {magnitude:.1f} ({word})" in finding.detail
        assert (
            finding.headline == f"Magnitude {magnitude:.1f} earthquake, 130 km E of Sendai, Japan"
        )
        assert "preliminary" in finding.detail  # says what it cannot tell
        assert finding.latitude == 38.3 and finding.value == magnitude
        assert finding.ongoing is False

    def test_no_earthquakes_is_said_not_raised(self, store: WatchStore) -> None:
        judgment = EarthquakeFeed().judge(_quakes(), store, NOW)

        assert judgment.findings == [] and "no earthquake" in judgment.summary


class TestSpaceWeather:
    @staticmethod
    def _frame(*kp: float) -> pd.DataFrame:
        stamps = [NOW - timedelta(hours=3 * (len(kp) - 1 - i)) for i in range(len(kp))]
        return pd.DataFrame(
            {"timestamp": stamps, "kp": list(kp), "solar_wind_speed": [640.0] * len(kp)}
        )

    def test_quiet_conditions_say_how_far_from_a_storm(self, store: WatchStore) -> None:
        judgment = SpaceWeatherFeed().judge(self._frame(1.0, 0.67), store, NOW)

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
        self, store: WatchStore, kp: float, level: int, severity: Severity
    ) -> None:
        (finding,) = SpaceWeatherFeed().judge(self._frame(2.0, kp), store, NOW).findings

        assert finding.severity is severity
        assert finding.headline.startswith(f"G{level} ")
        assert finding.ongoing is True and finding.subject == "planetary"

    def test_the_latest_reading_decides_not_the_largest(self, store: WatchStore) -> None:
        """A storm that has passed is not a standing finding."""
        assert SpaceWeatherFeed().judge(self._frame(8.0, 3.0), store, NOW).findings == []

    def test_a_missing_latest_kp_falls_back_to_the_last_measured(self, store: WatchStore) -> None:
        judgment = SpaceWeatherFeed().judge(self._frame(6.0, float("nan")), store, NOW)

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

    def test_buoys_at_their_standard_cadence_are_nothing_notable(self, store: WatchStore) -> None:
        """Was: 1,036 of 32,845 standard readings flagged on a quiet day."""
        judgment = TsunamiBuoyFeed().judge(self._frame(None), store, NOW)

        assert judgment.findings == []
        assert judgment.summary == "2 buoys reporting; none in event mode"

    def test_a_buoy_in_event_mode_is_reported_as_what_it_is(self, store: WatchStore) -> None:
        (finding,) = TsunamiBuoyFeed().judge(self._frame("46407"), store, NOW).findings

        assert finding.headline == "DART buoy 46407 is in event mode"
        assert finding.severity is Severity.NOTABLE
        assert "not a tsunami warning" in finding.detail
        assert "a test" in finding.detail  # the causes it cannot tell apart

    def test_event_mode_days_ago_is_not_now(self, store: WatchStore) -> None:
        """The NDBC file holds 45 days; an event last month is history."""
        assert (
            TsunamiBuoyFeed().judge(self._frame("46407", hours_ago=60), store, NOW).findings == []
        )


class TestRiverGauges:
    @staticmethod
    def _frame(levels: list[float], end: datetime = NOW) -> pd.DataFrame:
        stamps = [end - timedelta(minutes=15 * (len(levels) - 1 - i)) for i in range(len(levels))]
        return pd.DataFrame(
            {
                "datetime": stamps,
                "site_id": "03451500",
                "site_name": "FRENCH BROAD RIVER AT ASHEVILLE, NC",
                "gauge_height_ft": levels,
                "discharge_cfs": 800.0,
            }
        )

    def test_first_sight_of_a_gauge_only_builds_its_record(self, store: WatchStore) -> None:
        judgment = RiverGaugeFeed().judge(self._frame([1.2] * 300), store, NOW)

        assert judgment.findings == []
        assert judgment.summary == "1 gauges reporting; none above its record"

    def test_a_gauge_above_everything_recorded_is_reported(self, store: WatchStore) -> None:
        feed = RiverGaugeFeed()
        steady = list(1.2 + 0.02 * np.sin(np.arange(300)))
        feed.judge(self._frame(steady), store, NOW)

        later = NOW + timedelta(hours=1)
        rising = self._frame([1.3, 2.9, 4.6, 6.1], end=later)
        (finding,) = feed.judge(rising, store, later).findings

        assert (
            finding.headline
            == "French Broad River At Asheville, Nc: highest level Mercury has recorded"
        )
        assert finding.value == 6.1
        assert "does not have this gauge's flood stage" in finding.detail
        assert "300 readings" in finding.detail

    def test_ordinary_variation_is_not_a_record(self, store: WatchStore) -> None:
        feed = RiverGaugeFeed()
        steady = list(1.2 + 0.02 * np.sin(np.arange(300)))
        feed.judge(self._frame(steady), store, NOW)

        later = NOW + timedelta(hours=1)
        assert (
            feed.judge(self._frame([1.21, 1.225, 1.23, 1.22], end=later), store, later).findings
            == []
        )

    def test_a_short_record_cannot_be_exceeded(self, store: WatchStore) -> None:
        feed = RiverGaugeFeed()
        feed.judge(self._frame([1.2] * 50), store, NOW)

        later = NOW + timedelta(hours=1)
        assert feed.judge(self._frame([9.0], end=later), store, later).findings == []


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

    def test_no_recent_declaration_is_nothing_notable(self, store: WatchStore) -> None:
        judgment = DeclarationFeed().judge(self._frame(0), store, NOW)

        assert judgment.findings == []
        assert (
            judgment.summary
            == "40 tropical-cyclone designations this year; none in the past 3 days"
        )

    def test_new_declarations_are_reported(self, store: WatchStore) -> None:
        (finding,) = DeclarationFeed().judge(self._frame(12), store, NOW).findings

        assert finding.severity is Severity.NOTABLE
        assert finding.headline == "FEMA declared 12 county designations for tropical cyclones"
        assert "not the storm's arrival" in finding.detail

    def test_the_busiest_week_of_the_year_is_significant(self, store: WatchStore) -> None:
        (finding,) = DeclarationFeed().judge(self._frame(107), store, NOW).findings

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
    def test_a_reading_is_counted_once_however_often_it_is_read(self, store: WatchStore) -> None:
        """Pulls overlap: the same 15-minute readings come back every time."""
        times = [NOW + timedelta(minutes=15 * i) for i in range(4)]
        first = store.remember("f", "s", "x", times, [1.0, 2.0, 3.0, 4.0])
        again = store.remember("f", "s", "x", times, [1.0, 2.0, 3.0, 4.0])
        more = store.remember("f", "s", "x", [*times, times[-1] + timedelta(minutes=15)],
                              [1.0, 2.0, 3.0, 4.0, 9.0])  # fmt: skip
        after = store.remember("f", "s", "x", [], [])

        assert len(first) == 0 and first.latest is None
        assert again.values.tolist() == [1.0, 2.0, 3.0, 4.0]
        assert more.values.tolist() == [1.0, 2.0, 3.0, 4.0]  # the record before the new reading
        assert after.values.tolist() == [1.0, 2.0, 3.0, 4.0, 9.0]
        assert after.maximum == 9.0 and after.median == 3.0

    def test_non_finite_readings_are_not_remembered(self, store: WatchStore) -> None:
        store.remember("f", "s", "x", [NOW, NOW + timedelta(minutes=1)], [float("nan"), 2.0])

        assert store.remember("f", "s", "x", [], []).values.tolist() == [2.0]

    def test_the_record_is_bounded(self, store: WatchStore) -> None:
        times = [NOW + timedelta(minutes=i) for i in range(1500)]
        store.remember("f", "s", "x", times, np.arange(1500.0))

        record = store.remember("f", "s", "x", [], [])
        assert len(record) == 1000 and record.values[0] == 500.0

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

    def judge(self, frame: pd.DataFrame, store: WatchStore, now: datetime) -> Judgment:
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
        "fema_declarations": "FEMAActivityLoader",
    }
    assert set(FEEDS) == set(loaders)
    for name, feed_class in FEEDS.items():
        feed = feed_class()
        loader: Any = feed._loader(loaders[name])

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
