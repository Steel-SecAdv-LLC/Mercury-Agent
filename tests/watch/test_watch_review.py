# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The watch checks its own findings against the records that settle them.

Before this, a finding was reported and never looked at again: a preliminary
magnitude USGS later revised, or an automatic solution it deleted, stood in
the store as first said, and nothing counted how often the watch was right.

These tests give each feed the later record it consults (stubbed at the
network boundary) and pin the verdict, when a finding is checked, when the
last answer becomes final, and that the record being unreachable costs
nothing but a later retry.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

from omni_mercury_engine.loaders.base import FetchHTTPError
from omni_mercury_engine.watch import Severity, Watcher, WatchStore
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
from omni_mercury_engine.watch.model import Finding, Outcome
from omni_mercury_engine.watch.voice import brief, finding_text, record_line

if TYPE_CHECKING:
    from pathlib import Path

NOW = datetime(2026, 11, 16, 21, 0, tzinfo=UTC)


@pytest.fixture()
def store(tmp_path: Path) -> WatchStore:
    return WatchStore(tmp_path)


def _finding(
    feed: str = "earthquakes",
    *,
    subject: str = "us7000abcd",
    value: float | None = 6.6,
    ongoing: bool = False,
    observed_at: datetime = NOW,
) -> Finding:
    return Finding(
        feed=feed, kind="k", subject=subject, severity=Severity.SIGNIFICANT,
        headline=f"{feed} {subject}", detail="detail", observed_at=observed_at, source="src",
        basis="basis", value=value, ongoing=ongoing,
    )  # fmt: skip


def _stored(finding: Finding) -> dict[str, Any]:
    return {**finding.to_dict(), "key": finding.key, "first_seen": NOW.isoformat()}


class _Record:
    """Stands in for a loader: answers ``_fetch_json`` with a fixed record or error."""

    def __init__(self, answer: Any) -> None:
        self.answer = answer
        self.calls: list[tuple[str, dict[str, str]]] = []

    def _fetch_json(self, url: str, params: dict[str, str]) -> Any:
        self.calls.append((url, params))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def _with_record(feed: Feed, answer: Any, monkeypatch: pytest.MonkeyPatch) -> _Record:
    record = _Record(answer)
    monkeypatch.setattr(feed, "_loader", lambda _name: record)
    return record


def _http(status: int) -> FetchHTTPError:
    return FetchHTTPError(f"HTTP {status}", status_code=status)


# --------------------------------------------------------------------------- #
# What each record says
# --------------------------------------------------------------------------- #


class TestEarthquakeReview:
    @pytest.mark.parametrize(
        ("properties", "verdict", "final", "says"),
        [
            ({"mag": 6.6, "status": "reviewed"}, "confirmed", True, "reviewed by a seismologist"),
            ({"mag": 6.62, "status": "automatic"}, "confirmed", False, "still an automatic"),
            ({"mag": 6.3, "status": "reviewed"}, "revised", True, "now lists it at magnitude 6.3"),
            ({"mag": 4.2, "status": "reviewed"}, "revised", True, "below the 4.5"),
            ({"mag": 6.6, "status": "deleted"}, "withdrawn", True, "deleted"),
        ],
    )
    def test_the_catalogue_entry_decides(
        self, properties: dict[str, Any], verdict: str, final: bool, says: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:  # fmt: skip
        feed = EarthquakeFeed()
        record = _with_record(feed, {"properties": properties}, monkeypatch)

        outcome = feed.review(_stored(_finding()), NOW + timedelta(hours=3))

        assert outcome is not None
        assert (outcome.verdict, outcome.final) == (verdict, final)
        assert says in outcome.note
        assert record.calls[0][1]["eventid"] == "us7000abcd"

    @pytest.mark.parametrize("status", [404, 409])
    def test_an_event_the_catalogue_dropped_is_withdrawn(
        self, status: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        feed = EarthquakeFeed()
        _with_record(feed, _http(status), monkeypatch)

        outcome = feed.review(_stored(_finding()), NOW)

        assert outcome is not None and (outcome.verdict, outcome.final) == ("withdrawn", True)

    def test_the_catalogue_being_down_is_not_a_verdict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        feed = EarthquakeFeed()
        _with_record(feed, _http(503), monkeypatch)

        with pytest.raises(FetchHTTPError):
            feed.review(_stored(_finding()), NOW)

    def test_an_entry_without_a_magnitude_says_nothing_yet(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        feed = EarthquakeFeed()
        _with_record(feed, {"properties": {"mag": None, "status": "automatic"}}, monkeypatch)

        assert feed.review(_stored(_finding()), NOW) is None


class TestSpaceWeatherReview:
    @staticmethod
    def _review(kp: float | None, reported: float, days_later: float = 0.5) -> Outcome | None:
        feed = SpaceWeatherFeed()
        frame = pd.DataFrame({"timestamp": [NOW - timedelta(hours=3), NOW], "kp": [2.0, kp]})
        feed.read = lambda: frame  # type: ignore[method-assign]
        finding = _stored(_finding("space_weather", subject="planetary", value=reported))
        return feed.review(finding, NOW + timedelta(days=days_later))

    def test_the_same_storm_level_is_confirmed(self) -> None:
        outcome = self._review(6.3, 6.0)

        assert outcome is not None and outcome.verdict == "confirmed" and not outcome.final
        assert "G2, as reported" in outcome.note

    def test_a_kp_noaa_lowered_out_of_storm_level_is_revised(self) -> None:
        outcome = self._review(4.0, 5.0)

        assert outcome is not None and outcome.verdict == "revised"
        assert "below storm level" in outcome.note and "reported as G1" in outcome.note

    def test_the_answer_is_final_once_the_interval_is_two_days_old(self) -> None:
        outcome = self._review(6.0, 6.0, days_later=2.0)

        assert outcome is not None and outcome.final

    def test_an_interval_noaa_no_longer_lists_says_nothing(self) -> None:
        assert self._review(None, 6.0) is None


class TestTsunamiReview:
    @staticmethod
    def _item(**fields: Any) -> dict[str, Any]:
        return {"year": 2026, "month": 11, "day": 16, "hour": 14, "minute": 5,
                "locationName": "KURIL ISLANDS", "eventValidity": 4,
                "maxWaterHeight": 1.25, **fields}  # fmt: skip

    def _review(self, items: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch) -> Outcome:
        feed = TsunamiBuoyFeed()
        _with_record(feed, {"items": items}, monkeypatch)
        outcome = feed.review(_stored(_finding("tsunami_buoys", subject="46407", value=None)), NOW)
        assert outcome is not None
        return outcome

    def test_a_listed_tsunami_within_a_day_confirms_the_trigger(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        outcome = self._review([self._item()], monkeypatch)

        assert (outcome.verdict, outcome.final) == ("confirmed", True)
        assert "definite tsunami at Kuril Islands on 16 November 2026, 14:05 UTC" in outcome.note
        assert "1.25 m" in outcome.note

    @pytest.mark.parametrize(
        "item",
        [
            {"day": 12},  # four days before the trigger
            {"day": 17},  # after the trigger
            {"eventValidity": 1},  # NCEI calls it very doubtful
            {"month": None},  # undated
        ],
    )
    def test_an_unrelated_doubtful_or_undated_event_does_not(
        self, item: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        outcome = self._review([self._item(**item)], monkeypatch)

        assert (outcome.verdict, outcome.final) == ("unconfirmed", False)

    def test_an_event_dated_only_to_the_day_counts_for_that_whole_day(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        outcome = self._review([self._item(hour=None, minute=None)], monkeypatch)

        assert outcome.verdict == "confirmed" and "UTC" not in outcome.note

    def test_an_empty_database_leaves_it_unconfirmed_and_open(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        outcome = self._review([], monkeypatch)

        assert (outcome.verdict, outcome.final) == ("unconfirmed", False)


class _Gauge:
    """Stands in for the flood loader: one gauge's flood stages and observed record."""

    def __init__(self, stages: dict[str, float], levels: dict[int, float]) -> None:
        self.stages = stages
        self.record = pd.DataFrame(
            {
                "observed_at": [pd.Timestamp(NOW + timedelta(minutes=m)) for m in levels],
                "stage": list(levels.values()),
            }
        )
        self.asked: list[str] = []

    def fetch_gauge_record(self, gauge_id: str) -> tuple[dict[str, float], pd.DataFrame]:
        self.asked.append(gauge_id)
        return self.stages, self.record


class TestRiverGaugeReview:
    STAGES = {"action": 14.0, "minor": 15.0, "moderate": 16.0, "major": 17.0}

    def _review(
        self,
        levels: dict[int, float],
        monkeypatch: pytest.MonkeyPatch,
        *,
        reported: float | None = 15.4,
        stages: dict[str, float] | None = None,
        ended: bool = False,
    ) -> Outcome | None:
        """``levels`` maps minutes after the finding to the stage the NWS record shows now."""
        feed = RiverGaugeFeed()
        gauge = _Gauge(self.STAGES if stages is None else stages, levels)
        monkeypatch.setattr(feed, "_loader", lambda _name: gauge)
        finding = _stored(_finding("river_gauges", subject="COCF1", value=reported))
        finding.update(unit="ft", ended=(NOW + timedelta(hours=5)).isoformat() if ended else None)
        outcome = feed.review(finding, NOW + timedelta(hours=7))
        assert gauge.asked == ([] if reported is None else ["COCF1"])
        return outcome

    def test_the_same_reading_is_confirmed_with_how_far_the_river_went(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        outcome = self._review({0: 15.4, 15: 17.3, 30: 16.2}, monkeypatch)

        assert outcome is not None and (outcome.verdict, outcome.final) == ("confirmed", False)
        assert outcome.note == (
            "The NWS record still shows 15.4 ft for that time. It went on to 17.3 ft at "
            "21:15 UTC on 16 November 2026 (major flooding) and stands at 16.2 ft "
            "(moderate flooding)."
        )

    def test_a_reading_that_was_the_crest_is_said_to_be(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        outcome = self._review({0: 15.4, 15: 14.2}, monkeypatch, ended=True)

        assert outcome is not None and (outcome.verdict, outcome.final) == ("confirmed", True)
        assert outcome.note.endswith(
            "That is the highest since; it stands at 14.2 ft (below flood stage)."
        )

    def test_a_reading_the_nws_corrected_but_still_in_flood_is_revised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        outcome = self._review({0: 15.1, 15: 15.2}, monkeypatch)

        assert outcome is not None and outcome.verdict == "revised"
        assert "now shows 15.1 ft for that time; 15.4 ft was reported" in outcome.note

    def test_a_reading_corrected_to_below_flood_stage_is_withdrawn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        outcome = self._review({0: 12.0, 15: 12.1}, monkeypatch)

        assert outcome is not None and outcome.verdict == "withdrawn"
        assert "below this gauge's minor flood stage of 15 ft; 15.4 ft was reported" in outcome.note

    def test_a_reading_the_nws_removed_is_withdrawn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        outcome = self._review({-15: 15.0, 15: 15.2}, monkeypatch)

        assert outcome is not None and outcome.verdict == "withdrawn"
        assert "no longer holds a reading for that time" in outcome.note

    def test_a_record_that_stops_before_the_finding_cannot_say_yet(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert self._review({-30: 15.0}, monkeypatch) is None
        assert self._review({}, monkeypatch) is None

    def test_a_gauge_without_flood_stages_is_compared_on_the_reading_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        outcome = self._review({0: 12.0, 15: 12.4}, monkeypatch, stages={})

        assert outcome is not None and outcome.verdict == "revised"
        assert "flooding" not in outcome.note and "flood stage" not in outcome.note

    def test_a_finding_without_a_stage_is_not_looked_up(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert self._review({0: 15.4}, monkeypatch, reported=None) is None


class _Notices:
    """Stands in for the volcanic loader: answers ``fetch_notice`` with a notice or an error."""

    def __init__(self, answer: Any) -> None:
        self.answer = answer
        self.asked: list[str] = []

    def fetch_notice(self, notice_id: str) -> Any:
        self.asked.append(notice_id)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


class TestVolcanoReview:
    NOTICE = "DOI-USGS-AVO-2026-11-13T19:12:01+00:00"

    def _review(
        self, answer: Any, monkeypatch: pytest.MonkeyPatch, *, reference: str | None = None
    ) -> Outcome | None:
        feed, notices = VolcanoFeed(), _Notices(answer)
        monkeypatch.setattr(feed, "_loader", lambda _name: notices)
        finding = _stored(_finding("volcanoes", subject="311120", value=2.0, ongoing=True))
        finding.update(
            basis="USGS volcano alert level WATCH (ADVISORY and above is reported)",
            reference=self.NOTICE if reference is None else reference,
        )
        outcome = feed.review(finding, NOW)
        assert notices.asked == ([self.NOTICE] if reference is None else [])
        return outcome

    @staticmethod
    def _notice(level: str, vnum: str = "311120") -> dict[str, Any]:
        return {
            "notice_type_title": "Weekly Update",
            "notice_sections": [
                {"vnum": "311240", "alertLevel": "NORMAL", "synopsis": "another volcano"},
                {
                    "vnum": vnum,
                    "alertLevel": level,
                    "synopsis": "AVO Great Sitkin ORANGE/WATCH -  Slow eruption of lava\n continues",
                },
            ],
        }

    def test_the_notice_that_gives_the_level_confirms_it_in_the_observatorys_words(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        outcome = self._review(self._notice("WATCH"), monkeypatch)

        assert outcome is not None and (outcome.verdict, outcome.final) == ("confirmed", True)
        assert outcome.note == (
            "The observatory's weekly update gives WATCH, as reported. It reads: "
            '"AVO Great Sitkin ORANGE/WATCH - Slow eruption of lava continues"'
        )

    def test_a_notice_that_gives_another_level_revises_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        outcome = self._review(self._notice("ADVISORY"), monkeypatch)

        assert outcome is not None and outcome.verdict == "revised"
        assert "gives ADVISORY; WATCH was reported" in outcome.note

    @pytest.mark.parametrize("status", [400, 404])
    def test_a_notice_usgs_no_longer_holds_withdraws_it(
        self, status: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        outcome = self._review(_http(status), monkeypatch)

        assert outcome is not None and (outcome.verdict, outcome.final) == ("withdrawn", True)

    def test_usgs_being_down_is_not_a_verdict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(FetchHTTPError):
            self._review(_http(503), monkeypatch)

    def test_a_notice_that_does_not_mention_the_volcano_says_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert self._review(self._notice("WATCH", vnum="999999"), monkeypatch) is None
        assert self._review({"notice_sections": "none"}, monkeypatch) is None

    def test_a_finding_without_a_notice_is_not_looked_up(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert self._review(self._notice("WATCH"), monkeypatch, reference="") is None


class TestDeclarationReview:
    @staticmethod
    def _review(designations: list[int], reported: int, days_later: int = 1) -> Outcome | None:
        feed = DeclarationFeed()
        day = pd.Timestamp(NOW).floor("D")
        frame = pd.DataFrame(
            {
                "timestamp": [day - pd.Timedelta(days=i) for i in range(len(designations))],
                "designations": designations,
            }
        )
        feed.read = lambda: frame  # type: ignore[method-assign]
        finding = _stored(_finding("declarations", subject="recent", value=float(reported)))
        return feed.review(finding, NOW + timedelta(days=days_later))

    def test_the_same_count_is_confirmed(self) -> None:
        outcome = self._review([5, 4, 3, 2, 50], reported=14)

        assert outcome is not None and (outcome.verdict, outcome.final) == ("confirmed", False)

    def test_counties_added_later_revise_it(self) -> None:
        outcome = self._review([9, 4, 3, 2], reported=14)

        assert outcome is not None and outcome.verdict == "revised"
        assert "now lists 18 designations for those days; 14 were reported" in outcome.note

    def test_the_count_is_final_after_ten_days(self) -> None:
        outcome = self._review([5, 4, 3, 2], reported=14, days_later=11)

        assert outcome is not None and outcome.final


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #


class TestStoreReview:
    def test_a_store_written_before_self_checking_is_migrated_in_place(
        self, tmp_path: Path
    ) -> None:
        with closing(sqlite3.connect(tmp_path / "watch.sqlite")) as old, old:
            old.execute(
                "CREATE TABLE findings (key TEXT PRIMARY KEY, feed TEXT NOT NULL, "
                "severity INTEGER NOT NULL, first_seen TEXT NOT NULL, last_seen TEXT NOT NULL, "
                "ended TEXT, body TEXT NOT NULL)"
            )
            finding = _finding()
            old.execute(
                "INSERT INTO findings VALUES (?, 'earthquakes', 2, ?, ?, NULL, ?)",
                (finding.key, NOW.isoformat(), NOW.isoformat(),
                 __import__("json").dumps(finding.to_dict())),
            )  # fmt: skip

        store = WatchStore(tmp_path)
        WatchStore(tmp_path)  # opening it again changes nothing

        (kept,) = store.findings()
        assert kept["headline"] == finding.headline and kept["outcome"] is None
        assert store.track_record()["unchecked"] == 1

    def test_a_finding_is_due_only_after_its_record_has_had_time(self, store: WatchStore) -> None:
        store.report(_finding(), NOW)

        early = store.awaiting_review("earthquakes", NOW - timedelta(hours=2), NOW)
        due = store.awaiting_review("earthquakes", NOW, NOW - timedelta(hours=12))
        other = store.awaiting_review("space_weather", NOW, NOW)

        assert early == [] and other == []
        assert [finding["subject"] for finding in due] == ["us7000abcd"]

    def test_an_open_answer_is_asked_again_and_a_final_one_is_not(self, store: WatchStore) -> None:
        finding = _finding()
        store.report(finding, NOW)
        later = NOW + timedelta(hours=3)
        store.record_outcome(finding.key, Outcome("confirmed", "note", "record"), later)

        assert store.awaiting_review("earthquakes", later, later - timedelta(hours=12)) == []
        assert len(store.awaiting_review("earthquakes", later, later)) == 1

        store.record_outcome(finding.key, Outcome("revised", "n", "r", final=True), later)

        assert store.awaiting_review("earthquakes", later, later + timedelta(days=9)) == []
        (kept,) = store.findings()
        assert kept["outcome"] == {"verdict": "revised", "note": "n", "record": "r", "final": True}
        assert kept["reviewed_at"] == later.isoformat()

    def test_a_recurrence_does_not_inherit_the_earlier_occurrences_outcome(
        self, store: WatchStore
    ) -> None:
        """Was: one row per key, so a second storm would have arrived already 'confirmed'."""
        storm = _finding("space_weather", subject="planetary", ongoing=True)
        store.report(storm, NOW)
        store.record_outcome(storm.key, Outcome("confirmed", "n", "r", final=True), NOW)
        store.close_absent("space_weather", set(), NOW + timedelta(hours=3))

        assert store.report(storm, NOW + timedelta(days=2)) is True

        again, first = store.findings()
        assert again["key"] == storm.key and again["outcome"] is None and again["ended"] is None
        assert first["key"] == f"{storm.key}@{NOW.isoformat()}"
        assert first["outcome"]["verdict"] == "confirmed" and first["ended"] is not None
        assert store.track_record() == {
            "confirmed": 1, "revised": 0, "withdrawn": 0, "unconfirmed": 0, "unchecked": 1,
        }  # fmt: skip


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #


class _Reviewed(Feed):
    name = "earthquakes"
    title = "Scripted"
    source = "the scripted record"
    cadence = timedelta(minutes=5)
    stale_after = timedelta(days=1)
    review_after = timedelta(hours=2)
    review_every = timedelta(hours=12)
    review_until = timedelta(days=2)

    def __init__(self, answers: list[Outcome | Exception | None]) -> None:
        self.answers = answers
        self.asked: list[datetime] = []

    def read(self) -> pd.DataFrame:
        return pd.DataFrame()

    def judge(self, frame: pd.DataFrame, now: datetime) -> Judgment:
        return Judgment([], "quiet", now, 0, 0)

    def review(self, finding: dict[str, Any], now: datetime) -> Outcome | None:
        self.asked.append(now)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class _Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> None:
        self.now += timedelta(**delta)


class TestWatcherReview:
    @staticmethod
    def _watch(
        store: WatchStore, answers: list[Outcome | Exception | None]
    ) -> tuple[Watcher, _Reviewed, _Clock]:
        feed, clock = _Reviewed(answers), _Clock()
        store.report(_finding(), NOW)
        return Watcher(store, [feed], clock=clock), feed, clock

    def test_a_finding_is_checked_after_its_delay_then_at_intervals_until_final(
        self, store: WatchStore
    ) -> None:
        open_answer = Outcome("confirmed", "automatic", "record")
        watcher, feed, clock = self._watch(
            store, [open_answer, Outcome("revised", "reviewed", "record", final=True)]
        )

        watcher.run_once()  # at once: too soon
        clock.advance(hours=2)
        watcher.run_once()  # first check
        clock.advance(hours=6)
        watcher.run_once()  # checked six hours ago: not yet again
        clock.advance(hours=6)
        watcher.run_once()  # second check, final
        clock.advance(hours=24)
        watcher.run_once()  # final: never asked again

        assert feed.asked == [NOW + timedelta(hours=2), NOW + timedelta(hours=14)]
        assert store.findings()[0]["outcome"]["verdict"] == "revised"
        assert store.track_record()["revised"] == 1

    def test_a_record_that_cannot_be_reached_is_tried_again(self, store: WatchStore) -> None:
        watcher, feed, clock = self._watch(
            store, [ConnectionError("down"), None, Outcome("confirmed", "n", "r", final=True)]
        )
        clock.advance(hours=2)

        assert watcher.review_feed(feed) == 0  # unreachable
        assert watcher.review_feed(feed) == 0  # reachable, cannot say yet
        assert store.findings()[0]["outcome"] is None
        assert watcher.review_feed(feed) == 1

        assert store.findings()[0]["outcome"]["verdict"] == "confirmed"

    def test_at_the_end_of_the_review_period_the_last_answer_stands(
        self, store: WatchStore
    ) -> None:
        watcher, feed, clock = self._watch(store, [Outcome("confirmed", "automatic", "record")])
        clock.advance(days=2)

        watcher.review_feed(feed)
        clock.advance(days=30)

        assert watcher.review_feed(feed) == 0 and len(feed.asked) == 1
        assert store.findings()[0]["outcome"]["final"] is True

    def test_a_finding_no_record_ever_spoke_to_is_closed_as_unconfirmed(
        self, store: WatchStore
    ) -> None:
        watcher, feed, clock = self._watch(store, [None])
        clock.advance(days=2)

        assert watcher.review_feed(feed) == 1

        outcome = store.findings()[0]["outcome"]
        assert (outcome["verdict"], outcome["final"]) == ("unconfirmed", True)
        assert outcome["record"] == "the scripted record"

    def test_a_feed_that_is_down_is_not_asked_to_check_anything(self, store: WatchStore) -> None:
        watcher, feed, clock = self._watch(store, [])
        feed.read = lambda: (_ for _ in ()).throw(ConnectionError("down"))  # type: ignore[method-assign]
        clock.advance(hours=3)

        (status,) = watcher.run_once()

        assert status.state == "down" and feed.asked == []


# --------------------------------------------------------------------------- #
# Voice
# --------------------------------------------------------------------------- #


class TestVoiceOfTheRecord:
    def test_a_checked_finding_says_what_the_record_said(self, store: WatchStore) -> None:
        finding = _finding()
        store.report(finding, NOW)
        store.record_outcome(
            finding.key,
            Outcome("revised", "USGS now lists it at magnitude 6.3.", "USGS earthquake catalogue"),
            NOW + timedelta(hours=2),
        )

        text = finding_text(store.findings()[0], NOW + timedelta(hours=3))

        assert text.splitlines()[-1].strip() == (
            "Checked 1 hour ago against the USGS earthquake catalogue: revised. "
            "USGS now lists it at magnitude 6.3."
        )

    def test_an_unchecked_finding_claims_no_check(self, store: WatchStore) -> None:
        store.report(_finding(), NOW)

        assert "Checked" not in finding_text(store.findings()[0], NOW)

    @pytest.mark.parametrize(
        ("tally", "line"),
        [
            ({"unchecked": 0}, "Track record: no findings yet."),
            ({"unchecked": 1}, "Track record: 1 finding not yet checked against a later record."),
            (
                {"confirmed": 7, "revised": 2, "withdrawn": 1, "unconfirmed": 0, "unchecked": 3},
                "Track record: of 10 checked against later records, 7 confirmed, 2 revised, "
                "1 withdrawn; 3 not yet checked.",
            ),
            ({"confirmed": 1, "unchecked": 0},
             "Track record: of 1 checked against later records, 1 confirmed."),
        ],
    )  # fmt: skip
    def test_the_track_record_is_counted_not_characterised(
        self, tally: dict[str, int], line: str
    ) -> None:
        assert record_line(tally) == line

    def test_the_brief_ends_with_the_track_record_when_given_one(self) -> None:
        assert brief([], [], NOW, tally={"unchecked": 0}).splitlines()[-1] == (
            "Track record: no findings yet."
        )
        assert "Track record" not in brief([], [], NOW)
