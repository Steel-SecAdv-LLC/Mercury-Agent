# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What a review of the second round of watch fixes found, each as the failure it was.

The fixes of the round before were themselves reviewed. Two of what follows
were made by them: keying an alert by its warning merged the pieces a warning
is in effect as, and keeping a finding's current severity re-told a gauge
wavering between two stages at every climb. Each test failed on that code.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

from omni_mercury_engine.watch import Severity, Watcher, WatchStore
from omni_mercury_engine.watch.feeds import (
    DeclarationFeed,
    Feed,
    Judgment,
    WarningsFeed,
    _alert_thread,
)
from omni_mercury_engine.watch.model import Finding, finding_key
from omni_mercury_engine.watch.notify import Notifier, address_entry
from omni_mercury_engine.watch.sms import MAX_TEXT, MAX_TEXT_UCS2, text_message, two_segments
from omni_mercury_engine.watch.voice import brief
from omni_mercury_engine.watch.world import WorldWarningsFeed

if TYPE_CHECKING:
    from pathlib import Path

NOW = datetime(2026, 11, 18, 9, 0, tzinfo=UTC)
ANA = "ana@example.org"
MIAMI = (25.76, -80.19)


@pytest.fixture()
def store(tmp_path: Path) -> WatchStore:
    return WatchStore(tmp_path)


def _finding(severity: Severity = Severity.SEVERE, **fields: Any) -> Finding:
    values: dict[str, Any] = {
        "feed": "river_gauges",
        "kind": "river_flood",
        "subject": "COCF1",
        "severity": severity,
        "headline": f"{severity.label} flooding: St. Johns River near Cocoa",
        "detail": "It read 17 ft.",
        "observed_at": NOW,
        "source": "NOAA National Weather Service",
        "basis": "NWS flood category",
        "latitude": 25.8,
        "longitude": -80.2,
        "ongoing": True,
    }
    values.update(fields)
    return Finding(**values)


class _Mail:
    def __init__(self) -> None:
        self.sent: list[str] = []

    def send(self, *, to: str, subject: str, body: str, **_: Any) -> None:
        self.sent.append(to)


# --------------------------------------------------------------------------- #
# A warning in effect as several pieces is one finding that reaches all of them
# --------------------------------------------------------------------------- #


def _alerts_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    base = {
        "event": "Gale Watch",
        "severity": "Severe",
        "urgency": "Future",
        "certainty": "Possible",
        "headline": "Gale Watch issued November 18",
        "instruction": "",
        "sender": "NWS Marquette MI",
        "sent": "2026-11-18T03:31:00-05:00",
        "until": "2026-11-19T12:00:00-05:00",
    }
    frame = pd.DataFrame([{**base, **row} for row in rows], columns=WarningsFeed._COLUMNS)
    frame.attrs["places"] = 3
    return frame


class TestOneWarningInSeveralPieces:
    """23 of the 527 NWS alerts in effect on 2026-11-18 were pieces of 8 warnings."""

    def _pieces(self) -> pd.DataFrame:
        return _alerts_frame(
            [
                {"id": "w.001.3", "thread": "w.003.1", "place_ids": [1],
                 "area": "Saxon Harbor WI to Black River MI"},
                {"id": "w.002.3", "thread": "w.003.1", "place_ids": [2], "severity": "Extreme",
                 "area": "Huron Islands to Marquette MI; Marquette to Munising MI"},
                {"id": "w.003.3", "thread": "w.003.1", "place_ids": [3],
                 "area": "Marquette to Munising MI"},
            ]
        )  # fmt: skip

    def test_the_pieces_are_one_finding_for_everyone_in_any_of_them(self) -> None:
        """Was: three findings under one key; the last read replaced the others' places."""
        judgment = WarningsFeed().judge(self._pieces(), NOW)

        (finding,) = judgment.findings
        assert finding.place_ids == (1, 2, 3)
        assert finding.area == (
            "Saxon Harbor WI to Black River MI; Huron Islands to Marquette MI; "
            "Marquette to Munising MI"
        )
        assert finding.severity is Severity.SEVERE  # the most severe piece speaks for it
        assert finding.reference == "w.002.3"

    def test_the_store_keeps_every_piece_s_places(self, store: WatchStore) -> None:
        for finding in WarningsFeed().judge(self._pieces(), NOW).findings:
            store.report(finding, NOW)

        (standing,) = store.standing(now=NOW)
        assert standing["place_ids"] == [1, 2, 3]

    def test_the_order_the_pieces_are_read_in_changes_nothing(self) -> None:
        """Was: the stored severity was the last piece's, so it "rose" at the next read."""
        forward = WarningsFeed().judge(self._pieces(), NOW).findings
        backward = WarningsFeed().judge(self._pieces().iloc[::-1], NOW).findings

        assert [f.severity for f in forward] == [f.severity for f in backward]
        assert [f.reference for f in forward] == [f.reference for f in backward]

    def test_a_warning_s_first_message_is_the_earliest_by_time_not_by_text(self) -> None:
        """Was: compared as text, so across a change of UTC offset the name changed."""
        references = [
            {"identifier": "first", "sent": "2026-11-01T01:30:00-05:00"},  # 06:30 UTC
            {"identifier": "second", "sent": "2026-11-01T01:10:00-06:00"},  # 07:10 UTC
        ]

        assert _alert_thread("third", references) == "first"
        assert (
            _alert_thread(
                "third", "s,second,2026-11-01T01:10:00-06:00 s,first,2026-11-01T01:30:00-05:00"
            )
            == "first"
        )
        assert _alert_thread("alone", []) == "alone"


def _cap(identifier: str, *, sent: str, references: str = "", msg_type: str = "Alert",
         area: str = "Uusimaa", severity: str = "Severe") -> str:  # fmt: skip
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>{identifier}</identifier><sender>fmi.fi</sender><sent>{sent}</sent>
  <status>Actual</status><msgType>{msg_type}</msgType><scope>Public</scope>
  <references>{references}</references>
  <info><language>en</language><event>Wind warning</event><urgency>Expected</urgency>
    <severity>{severity}</severity><certainty>Likely</certainty>
    <expires>2026-11-20T00:00:00+00:00</expires><senderName>FMI</senderName>
    <headline>Wind warning</headline><area><areaDesc>{area}</areaDesc></area></info>
</alert>"""


class _World(WorldWarningsFeed):
    """The world feed over documents held here instead of at the centre."""

    def __init__(self, documents: dict[str, str]) -> None:
        super().__init__()
        self.documents = documents

    def _covering(self, latitude: float, longitude: float) -> list[str]:
        return sorted(self.documents)

    def _document(self, path: str) -> str:
        return self.documents[path]


class TestAWarningFromAnAuthorityThatNamesOnlyTheMessageBefore:
    """Of 34 updates read from 17 authorities on 2026-11-18, 26 named one earlier message."""

    def _read(self, store: WatchStore, documents: dict[str, str]) -> tuple[_World, Judgment]:
        store.add_subscriber(ANA, place="Helsinki", latitude=60.17, longitude=24.94)
        feed = _World(documents)
        feed.bind(store)
        return feed, feed.judge(feed.read(), NOW)

    def test_the_third_message_is_still_the_first_one_s_warning(self, store: WatchStore) -> None:
        """Was: its earliest reference is the second message, so it was a new warning."""
        documents = {
            "fi/2026/11/18/01/00/a.xml": _cap("m1", sent="2026-11-18T01:00:00+00:00"),
        }
        feed, first = self._read(store, documents)
        feed.documents = {
            "fi/2026/11/18/02/00/b.xml": _cap(
                "m2", sent="2026-11-18T02:00:00+00:00", msg_type="Update",
                references="fmi.fi,m1,2026-11-18T01:00:00+00:00",
            ),
        }  # fmt: skip
        second = feed.judge(feed.read(), NOW)
        feed.documents = {
            "fi/2026/11/18/03/00/c.xml": _cap(
                "m3", sent="2026-11-18T03:00:00+00:00", msg_type="Update",
                references="fmi.fi,m2,2026-11-18T02:00:00+00:00",
            ),
        }  # fmt: skip
        third = feed.judge(feed.read(), NOW)

        keys = {judgment.findings[0].key for judgment in (first, second, third)}
        assert keys == {finding_key("world_warnings", "cap_alert", "m1")}

    def test_pieces_of_one_warning_are_one_finding(self, store: WatchStore) -> None:
        documents = {
            "fi/2026/11/18/02/00/b.xml": _cap(
                "m2", sent="2026-11-18T02:00:00+00:00", msg_type="Update", area="Uusimaa",
                references="fmi.fi,m1,2026-11-18T01:00:00+00:00",
            ),
            "fi/2026/11/18/02/00/c.xml": _cap(
                "m3", sent="2026-11-18T02:00:00+00:00", msg_type="Update", area="Kymenlaakso",
                references="fmi.fi,m1,2026-11-18T01:00:00+00:00",
            ),
        }  # fmt: skip
        _, judgment = self._read(store, documents)

        (finding,) = judgment.findings
        assert finding.area == "Uusimaa; Kymenlaakso"

    def test_a_warning_since_cancelled_is_not_confirmed_from_its_first_message(
        self, store: WatchStore
    ) -> None:
        """Was: the first cached message of the warning was re-read: "confirmed"."""
        documents = {
            "fi/2026/11/18/01/00/a.xml": _cap("m1", sent="2026-11-18T01:00:00+00:00"),
            "fi/2026/11/18/04/00/z.xml": _cap(
                "m4", sent="2026-11-18T04:00:00+00:00", msg_type="Cancel",
                references="fmi.fi,m1,2026-11-18T01:00:00+00:00",
            ),
        }  # fmt: skip
        feed, _ = self._read(store, documents)

        outcome = feed.review({"subject": "m1"}, NOW)

        assert outcome is not None and outcome.verdict == "withdrawn"


# --------------------------------------------------------------------------- #
# A severity it has just been told at is not told again at every climb back
# --------------------------------------------------------------------------- #


class TestAFindingWaveringBetweenTwoSeverities:
    EPISODE = timedelta(hours=26)

    def test_a_climb_back_within_the_episode_is_not_told_again(self, store: WatchStore) -> None:
        """Was: major, moderate, major, moderate, major: three alerts to everyone."""
        told = [
            store.report(_finding(severity), NOW + timedelta(minutes=10 * step), self.EPISODE)
            for step, severity in enumerate(
                [Severity.SEVERE, Severity.SIGNIFICANT, Severity.SEVERE, Severity.SIGNIFICANT,
                 Severity.SEVERE]
            )
        ]  # fmt: skip

        assert told == [True, False, False, False, False]

    def test_a_climb_past_what_was_told_is_told(self, store: WatchStore) -> None:
        store.report(_finding(Severity.SIGNIFICANT), NOW, self.EPISODE)
        store.report(_finding(Severity.NOTABLE), NOW + timedelta(minutes=10), self.EPISODE)

        assert store.report(_finding(Severity.SEVERE), NOW + timedelta(minutes=20), self.EPISODE)

    def test_a_return_after_the_episode_is_told(self, store: WatchStore) -> None:
        """A volcano back at WARNING months after it fell to ADVISORY is news again."""
        store.report(_finding(Severity.SEVERE), NOW, self.EPISODE)
        store.report(_finding(Severity.NOTABLE), NOW + timedelta(hours=1), self.EPISODE)
        later = NOW + timedelta(days=60)
        store.report(_finding(Severity.NOTABLE), later - timedelta(minutes=5), self.EPISODE)

        assert store.report(_finding(Severity.SEVERE), later, self.EPISODE)

    def test_a_feed_with_no_episode_tells_every_climb_as_before(self, store: WatchStore) -> None:
        store.report(_finding(Severity.SEVERE), NOW)
        store.report(_finding(Severity.NOTABLE), NOW + timedelta(minutes=5))

        assert store.report(_finding(Severity.SEVERE), NOW + timedelta(minutes=10))


# --------------------------------------------------------------------------- #
# Deliveries
# --------------------------------------------------------------------------- #


class TestTheLedgerIsTheAddresss:
    def test_removing_a_place_does_not_send_what_stands_again(self, store: WatchStore) -> None:
        """Was: the entry was the address's oldest place; remove it and everything went again."""
        home = store.add_subscriber(ANA, place="home", latitude=MIAMI[0], longitude=MIAMI[1])
        store.add_subscriber(ANA, place="work", latitude=MIAMI[0] + 0.1, longitude=MIAMI[1])
        store.report(_finding(), NOW)
        mail = _Mail()
        notifier = Notifier(store, mailer=mail, clock=lambda: NOW)

        assert notifier.sync()["sent"] == 1
        store.remove_subscription(ANA, home)

        assert notifier.sync() == {"sent": 0, "failed": 0}
        assert mail.sent == [ANA]

    def test_what_was_recorded_under_a_place_is_still_counted(self, store: WatchStore) -> None:
        """A store written before the entry was the address's does not send it all again."""
        place = store.add_subscriber(ANA, place="home", latitude=MIAMI[0], longitude=MIAMI[1])
        finding = _finding()
        store.report(finding, NOW)
        store.record_delivery(finding.key, 3, place, "email", error=None, now=NOW)
        mail = _Mail()

        assert Notifier(store, mailer=mail, clock=lambda: NOW).sync()["sent"] == 0
        assert address_entry(ANA) < 0 and address_entry(ANA) == address_entry(" Ana@Example.org ")

    def test_a_pass_ends_when_the_lease_is_gone(self, store: WatchStore) -> None:
        """Was: it went on sending from the ledger it had read, beside the watch that took over."""
        for name in ("ana", "ben", "cy"):
            store.add_subscriber(
                f"{name}@example.org", place="home", latitude=MIAMI[0], longitude=MIAMI[1]
            )
        store.report(_finding(), NOW)
        mail = _Mail()

        done = Notifier(store, mailer=mail, clock=lambda: NOW).sync(heartbeat=lambda: False)

        assert done["sent"] == 1 and len(mail.sent) == 1


class _Quiet(Feed):
    name = "volcanoes"
    title = "Volcanoes"
    source = "test"
    cadence = timedelta(minutes=30)
    stale_after = timedelta(days=1)

    def __init__(self) -> None:
        self.read_count = 0

    def read(self) -> pd.DataFrame:
        self.read_count += 1
        return pd.DataFrame()

    def judge(self, frame: pd.DataFrame, now: datetime) -> Judgment:
        return Judgment([], "nothing", now, 1, 1)


class TestSeveralWatchesOfOneStore:
    def test_a_feed_another_watch_has_read_is_not_due(self, store: WatchStore) -> None:
        """Was: due by this watcher's own reads, so each worker read it when the lease lapsed.

        A lease lasts fifteen minutes and this feed is read every thirty.
        """
        clock = [NOW]
        first = Watcher(store, feeds=[_Quiet()], clock=lambda: clock[0])
        feed = _Quiet()
        second = Watcher(store, feeds=[feed], clock=lambda: clock[0])
        first._pass(force=False)

        clock[0] = NOW + timedelta(minutes=16)  # the first watcher's lease has lapsed
        second._pass(force=False)
        assert feed.read_count == 0

        clock[0] = NOW + timedelta(minutes=31)
        second._pass(force=False)
        assert feed.read_count == 1

    def test_the_watch_that_tells_people_tells_what_another_watch_read(
        self, store: WatchStore
    ) -> None:
        """Was: it delivered only on a tick when it had itself read a feed."""
        store.add_subscriber(ANA, place="home", latitude=MIAMI[0], longitude=MIAMI[1])
        mail = _Mail()
        clock = [NOW]
        teller = Watcher(
            store,
            feeds=[],
            clock=lambda: clock[0],
            notifier=Notifier(store, mailer=mail, clock=lambda: clock[0]),
        )
        teller._pass(force=False)  # nothing to say yet
        assert mail.sent == []

        clock[0] = NOW + timedelta(seconds=30)
        store.report(_finding(), clock[0])  # another watch's read
        teller._pass(force=False)

        assert mail.sent == [ANA]

    def test_stores_opened_together_on_an_old_file_all_open(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Was: two that both found a column missing both added it; the second failed.

        Each worker of ``serve --workers 4`` opens the store as it starts. The
        two here are held where each has read which columns exist, until both
        have (or, once that reading is done under the write lock, until the
        one waiting gives up).
        """
        import sqlite3
        from contextlib import suppress

        directory = tmp_path / "watch"
        directory.mkdir()
        with sqlite3.connect(directory / "watch.sqlite") as connection:
            connection.execute(
                "CREATE TABLE findings (key TEXT PRIMARY KEY, feed TEXT NOT NULL, severity "
                "INTEGER NOT NULL, first_seen TEXT NOT NULL, last_seen TEXT NOT NULL, "
                "ended TEXT, body TEXT NOT NULL)"
            )
        both_have_read = threading.Barrier(2)

        class _Held(sqlite3.Connection):
            def execute(self, sql: str, *parameters: Any) -> sqlite3.Cursor:
                cursor = super().execute(sql, *parameters)
                if sql.startswith("PRAGMA table_info(findings)"):
                    rows = cursor.fetchall()
                    with suppress(threading.BrokenBarrierError):
                        both_have_read.wait(timeout=1.0)
                    return super().execute(
                        "SELECT ? AS name" + " UNION ALL SELECT ?" * (len(rows) - 1),
                        [row["name"] for row in rows],
                    )
                return cursor

        def held(self: WatchStore) -> sqlite3.Connection:
            connection = sqlite3.connect(self.path, timeout=30.0, factory=_Held)
            connection.row_factory = sqlite3.Row
            return connection

        monkeypatch.setattr(WatchStore, "_connect", held)
        failures: list[BaseException] = []

        def open_it() -> None:
            try:
                WatchStore(directory)
            except BaseException as exc:
                failures.append(exc)

        threads = [threading.Thread(target=open_it) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        assert failures == []


class TestAWarningTheSourceNoLongerLists:
    def test_it_ends_even_at_a_place_that_could_not_be_checked(self, store: WatchStore) -> None:
        """Was: one place failing at every read kept an expired warning standing for good."""
        kept = _finding(feed="nws_alerts", kind="nws_alert", subject="still", place_ids=(7,))
        gone = _finding(feed="nws_alerts", kind="nws_alert", subject="expired", place_ids=(7,))
        store.report(kept, NOW)
        store.report(gone, NOW)

        ended = store.close_absent(
            "nws_alerts", set(), NOW, frozenset({7}), listed=frozenset({kept.key})
        )

        assert ended == 1
        assert [f["subject"] for f in store.standing(now=NOW)] == ["still"]

    def test_the_warnings_feed_says_what_the_nationwide_list_holds(self) -> None:
        frame = _alerts_frame([{"id": "a.1", "thread": "a.1", "place_ids": [], "area": "X"}])

        judgment = WarningsFeed().judge(frame, NOW)

        assert judgment.findings == []  # not Extreme and at no subscribed place
        assert judgment.listed == frozenset({finding_key(WarningsFeed.name, "nws_alert", "a.1")})


# --------------------------------------------------------------------------- #
# Phones
# --------------------------------------------------------------------------- #


class TestANumberThatOptsOutWhileAnotherIsAwaited:
    OLD, NEW = "+14155550123", "+14155550199"

    def _changing(self, store: WatchStore) -> str:
        code = store.request_phone(ANA, self.OLD, now=NOW)
        assert store.confirm_phone(ANA, code, now=NOW)
        return store.request_phone(ANA, self.NEW, now=NOW + timedelta(minutes=2))

    def test_the_awaited_number_can_still_be_confirmed(self, store: WatchStore) -> None:
        """Was: the opt-out voided the new number's code, and the page said texts "still go" to the old."""
        code = self._changing(store)

        assert store.unverify_phone(self.OLD) == 1
        held = store.phone_of(ANA)
        assert held["verified_at"] is None and held["pending_number"] == self.NEW
        assert store.confirm_phone(ANA, code, now=NOW + timedelta(minutes=3))
        assert store.phone_of(ANA)["number"] == self.NEW

    def test_the_account_does_not_say_it_is_texting_a_number_that_opted_out(
        self, store: WatchStore
    ) -> None:
        from omni_mercury_engine.api.routes.accounts import _watch_phone_view

        self._changing(store)
        assert _watch_phone_view(store.phone_of(ANA)).texting == self.OLD

        store.unverify_phone(self.OLD)

        assert _watch_phone_view(store.phone_of(ANA)).texting is None

    def test_an_awaited_number_that_opts_out_is_no_longer_awaited(self, store: WatchStore) -> None:
        self._changing(store)

        assert store.unverify_phone(self.NEW) == 1
        held = store.phone_of(ANA)
        assert held["pending_number"] is None and held["number"] == self.OLD
        assert held["verified_at"] is not None


class TestATextIsTwoSegmentsInTheAlphabetItIsSentIn:
    def test_a_message_with_a_character_outside_the_gsm_alphabet(self) -> None:
        """Was: cut to 306 characters, which in 16-bit units of 67 is five segments."""
        text = two_segments("Aviso de inundação para São Paulo e região metropolitana. " * 10)

        assert len(text.encode("utf-16-le")) // 2 <= MAX_TEXT_UCS2
        assert text.endswith("...")

    def test_a_message_inside_it_keeps_all_306(self) -> None:
        text = two_segments("Flood warning for the river at Cedar Rapids. " * 12)

        assert MAX_TEXT - 45 < len(text) <= MAX_TEXT

    def test_a_character_that_takes_two_places_counts_twice(self) -> None:
        text = two_segments("€" * 200)

        assert 2 * text.count("€") + 3 <= MAX_TEXT

    def test_the_alert_text_is_cut_the_same_way(self) -> None:
        finding = _finding(headline="Inundação severa: Rio Tietê em São Paulo " * 8).to_dict()

        text = text_message(finding, None, None, "")

        assert len(text.encode("utf-16-le")) // 2 <= MAX_TEXT_UCS2


# --------------------------------------------------------------------------- #
# Declarations at the turn of the year
# --------------------------------------------------------------------------- #


class TestDeclarationsInJanuary:
    def _frame(self, counts: dict[str, int]) -> pd.DataFrame:
        days = pd.date_range("2026-12-14", "2027-01-10", freq="D", tz="UTC")
        return pd.DataFrame(
            {
                "timestamp": days,
                "designations": [counts.get(day.date().isoformat(), 0) for day in days],
                "states": 1,
                "disasters": 1,
            }
        )

    def test_december_is_not_an_earlier_week_of_this_year(self) -> None:
        """Was: with the end of December read too, January's first weeks held its count."""
        frame = self._frame({"2026-12-30": 5, "2027-01-10": 8})

        judgment = DeclarationFeed().judge(frame, datetime(2027, 1, 10, 12, tzinfo=UTC))

        (finding,) = judgment.findings
        assert "busiest" not in finding.headline
        assert finding.severity is Severity.NOTABLE

    def test_a_week_busier_than_an_earlier_one_this_year_still_is(self) -> None:
        days = pd.date_range("2027-01-01", "2027-03-10", freq="D", tz="UTC")
        counts = {"2027-02-01": 4, "2027-03-10": 9}
        frame = pd.DataFrame(
            {
                "timestamp": days,
                "designations": [counts.get(day.date().isoformat(), 0) for day in days],
                "states": 1,
                "disasters": 1,
            }
        )

        judgment = DeclarationFeed().judge(frame, datetime(2027, 3, 10, 12, tzinfo=UTC))

        assert "busiest" in judgment.findings[0].headline

    def test_the_read_reaches_back_as_far_as_a_review_looks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Was: fourteen days, where a finding checked on its fourteenth day counts three more."""
        asked: list[int] = []

        class _Loader:
            def fetch_realtime(self, days_before_year: int = 0) -> pd.DataFrame:
                asked.append(days_before_year)
                return pd.DataFrame()

        feed = DeclarationFeed()
        monkeypatch.setattr(feed, "_loader", lambda _name: _Loader())

        feed.read()

        assert asked == [feed.review_until.days + feed.RECENT_DAYS + 1]


def test_the_command_lists_what_stands_in_the_page_s_order(store: WatchStore) -> None:
    """Was: the page listed the newest first within a severity, the command the oldest."""
    store.report(_finding(subject="older"), NOW - timedelta(hours=2))
    store.report(_finding(subject="newer", headline="severe flooding: the newer one"), NOW)

    page = [finding["subject"] for finding in store.overview(now=NOW)]
    said = brief([], store.overview(now=NOW), NOW)

    assert page == ["newer", "older"]
    assert said.index("the newer one") < said.index("St. Johns River")


# --------------------------------------------------------------------------- #
# What a review of the fixes above found
# --------------------------------------------------------------------------- #


class TestTheLedgerGoesWithTheAddress:
    def _told(self, store: WatchStore) -> tuple[Notifier, _Mail]:
        store.add_subscriber(ANA, place="home", latitude=MIAMI[0], longitude=MIAMI[1])
        store.report(_finding(), NOW)
        mail = _Mail()
        notifier = Notifier(store, mailer=mail, clock=lambda: NOW)
        assert notifier.sync()["sent"] == 1
        return notifier, mail

    def test_a_change_of_address_sends_nothing_again(self, store: WatchStore) -> None:
        """Was: the record stayed under the old address, so the new one was sent it all."""
        notifier, mail = self._told(store)

        store.move_subscriber(ANA, "ana@new.example.org")

        assert notifier.sync() == {"sent": 0, "failed": 0}
        assert mail.sent == [ANA]

    def test_an_address_that_leaves_takes_its_record_with_it(self, store: WatchStore) -> None:
        """Was: what it had been sent stayed in the store under a number made from it."""
        notifier, mail = self._told(store)
        finding = _finding()
        assert store.delivery(finding.key, 3, address_entry(ANA), "email")

        store.remove_subscriber(ANA)

        assert not store.delivery(finding.key, 3, address_entry(ANA), "email")
        store.add_subscriber(ANA, place="home", latitude=MIAMI[0], longitude=MIAMI[1])
        assert notifier.sync()["sent"] == 1  # it asked again, and is told what stands


class TestAMessageAnotherReplaces:
    def test_a_downgrade_is_not_outvoted_by_the_message_it_replaces(self) -> None:
        """Was: the most severe message listed spoke for the warning, replaced or not."""
        frame = _alerts_frame(
            [
                {"id": "w.1", "thread": "w.1", "references": [], "place_ids": [1],
                 "severity": "Extreme", "area": "Dade", "sent": "2026-11-18T01:00:00-05:00"},
                {"id": "w.2", "thread": "w.1", "references": ["w.1"], "place_ids": [1],
                 "severity": "Severe", "area": "Dade", "sent": "2026-11-18T03:00:00-05:00"},
            ]
        )  # fmt: skip

        (finding,) = WarningsFeed().judge(frame, NOW).findings

        assert finding.reference == "w.2"
        assert finding.severity is Severity.SIGNIFICANT

    def test_the_latest_of_two_pieces_is_the_latest_by_time(self) -> None:
        """Was: by the text of the time, the mistake fixed for references in the same change."""
        frame = _alerts_frame(
            [
                {"id": "w.a", "thread": "w.0", "references": ["w.0"], "place_ids": [1],
                 "area": "East", "sent": "2026-11-01T01:30:00-05:00"},  # 06:30 UTC
                {"id": "w.b", "thread": "w.0", "references": ["w.0"], "place_ids": [2],
                 "area": "West", "sent": "2026-11-01T01:10:00-06:00"},  # 07:10 UTC, the later
            ]
        )  # fmt: skip

        (finding,) = WarningsFeed().judge(frame, NOW).findings

        assert finding.reference == "w.b"

    def test_an_update_listed_beside_its_original_speaks_for_the_warning(
        self, store: WatchStore
    ) -> None:
        store.add_subscriber(ANA, place="Helsinki", latitude=60.17, longitude=24.94)
        feed = _World(
            {
                "fi/2026/11/18/01/00/a.xml": _cap(
                    "m1", sent="2026-11-18T01:00:00+00:00", severity="Extreme"
                ),
                "fi/2026/11/18/02/00/b.xml": _cap(
                    "m2", sent="2026-11-18T02:00:00+00:00", msg_type="Update", severity="Severe",
                    references="fmi.fi,m1,2026-11-18T01:00:00+00:00",
                ),
            }
        )  # fmt: skip
        feed.bind(store)

        (finding,) = feed.judge(feed.read(), NOW).findings

        assert finding.severity is Severity.SIGNIFICANT


class TestAWarningsNameOutlastsOneFailedLookup:
    def test_the_chain_is_not_forgotten_when_its_place_cannot_be_checked(
        self, store: WatchStore
    ) -> None:
        """Was: names were kept only for warnings in the read just made."""
        store.add_subscriber(ANA, place="Helsinki", latitude=60.17, longitude=24.94)
        store.add_subscriber("ben@example.org", place="Turku", latitude=60.45, longitude=22.27)
        second = _cap(
            "m2", sent="2026-11-18T02:00:00+00:00", msg_type="Update",
            references="fmi.fi,m1,2026-11-18T01:00:00+00:00",
        )  # fmt: skip
        third = _cap(
            "m3", sent="2026-11-18T03:00:00+00:00", msg_type="Update",
            references="fmi.fi,m2,2026-11-18T02:00:00+00:00",
        )  # fmt: skip
        other = _cap("o1", sent="2026-11-18T01:00:00+00:00", area="Turku")
        down = [False]

        class _Flaky(_World):
            def _covering(self, latitude: float, longitude: float) -> list[str]:
                if latitude < 60.3:  # Helsinki
                    if down[0]:
                        raise ConnectionError("simulated: the centre did not answer")
                    return [
                        path for path in self.documents if "/h/" in path or path.endswith("m.xml")
                    ]
                return [path for path in self.documents if path.endswith("o.xml")]

        feed = _Flaky({"fi/2026/11/18/02/00/m.xml": second, "fi/2026/11/18/01/00/o.xml": other})
        feed.bind(store)
        before = {f.key for f in feed.judge(feed.read(), NOW).findings}

        down[0] = True
        feed.judge(feed.read(), NOW)  # Helsinki's lookup fails; only Turku's alert is read
        down[0] = False
        feed.documents = {"fi/2026/11/18/03/00/m.xml": third, "fi/2026/11/18/01/00/o.xml": other}
        after = {f.key for f in feed.judge(feed.read(), NOW).findings}

        assert finding_key("world_warnings", "cap_alert", "m1") in before
        assert after == before


class _Counting(Notifier):
    def __init__(self, store: WatchStore, clock: Any) -> None:
        self.mail = _Mail()
        super().__init__(store, mailer=self.mail, clock=clock)
        self.passes = 0

    def sync(self, heartbeat: Any = None, feed: str | None = None) -> dict[str, int]:
        self.passes += feed is None
        return super().sync(heartbeat=heartbeat, feed=feed)


def test_a_full_delivery_pass_is_made_when_there_may_be_something_to_send(
    store: WatchStore,
) -> None:
    """Was: once a minute whatever had happened, loading everyone and everything each time."""
    store.add_subscriber(ANA, place="home", latitude=MIAMI[0], longitude=MIAMI[1])
    clock = [NOW]
    notifier = _Counting(store, lambda: clock[0])
    watcher = Watcher(store, feeds=[], clock=lambda: clock[0], notifier=notifier)

    watcher._pass(force=False)
    for minute in range(1, 6):  # five quiet minutes
        clock[0] = NOW + timedelta(minutes=minute)
        watcher._pass(force=False)
    assert notifier.passes == 1

    store.add_subscriber("ben@example.org", place="home", latitude=MIAMI[0], longitude=MIAMI[1])
    clock[0] = NOW + timedelta(minutes=6, seconds=30)
    watcher._pass(force=False)
    assert notifier.passes == 2  # a subscriber since


def test_a_character_outside_the_alphabet_in_the_part_cut_off_costs_nothing() -> None:
    """Was: the 134-character budget was chosen from the whole text, cut part included."""
    text = (
        "Flood warning for the river at Cedar Rapids. " * 6
        + "\u201cMove to higher ground\u201d now."
    )

    kept = two_segments(text + " More follows here." * 10)

    assert len(kept) > MAX_TEXT_UCS2 and len(kept) <= MAX_TEXT
    assert "\u201c" not in kept


# --------------------------------------------------------------------------- #
# And of those
# --------------------------------------------------------------------------- #


def test_a_message_that_fits_is_not_cut_for_a_character_at_its_end() -> None:
    """Was, for one commit: the text before a late curly quote, with "..." added, was preferred."""
    text = (
        "Flood warning for the river valley. Move to higher ground now. "
        "Shelter is open at Lincoln High\u2019s"
    )

    assert len(text) < MAX_TEXT_UCS2
    assert two_segments(text) == text


def test_a_changed_place_is_a_reason_for_a_delivery_pass(store: WatchStore) -> None:
    """Was, for one commit: only a new subscriber or number was; a place moved was not."""
    far = (47.6, -122.3)
    store.add_subscriber(ANA, place="home", latitude=far[0], longitude=far[1], radius_km=50.0)
    store.report(_finding(), NOW)
    clock = [NOW]
    notifier = _Counting(store, lambda: clock[0])
    watcher = Watcher(store, feeds=[], clock=lambda: clock[0], notifier=notifier)
    watcher._pass(force=False)
    assert notifier.mail.sent == []

    store.add_subscriber(ANA, place="home", latitude=MIAMI[0], longitude=MIAMI[1], radius_km=50.0)
    clock[0] = NOW + timedelta(seconds=30)
    watcher._pass(force=False)

    assert notifier.passes == 2
    assert notifier.mail.sent == [ANA]


def test_an_address_change_keeps_what_the_new_address_was_sent(store: WatchStore) -> None:
    """Was, for one commit: the old address's failed attempt replaced the new one's "sent"."""
    new = "ana@new.example.org"
    store.add_subscriber(ANA, place="home", latitude=MIAMI[0], longitude=MIAMI[1])
    store.add_subscriber(new, place="work", latitude=MIAMI[0], longitude=MIAMI[1])
    finding = _finding()
    store.report(finding, NOW)
    store.record_delivery(finding.key, 3, address_entry(new), "email", error=None, now=NOW)
    store.record_delivery(finding.key, 3, address_entry(ANA), "email", error="refused", now=NOW)

    store.move_subscriber(ANA, new)

    assert store.delivery(finding.key, 3, address_entry(new), "email")["status"] == "sent"
    assert not store.delivery(finding.key, 3, address_entry(ANA), "email")


# --------------------------------------------------------------------------- #
# And of those, in their turn
# --------------------------------------------------------------------------- #


def test_an_address_written_another_way_keeps_its_record(store: WatchStore) -> None:
    """Was, for one commit: a change of case was "the old address", and its record was deleted."""
    store.add_subscriber(ANA, place="home", latitude=MIAMI[0], longitude=MIAMI[1])
    finding = _finding()
    store.report(finding, NOW)
    store.record_delivery(finding.key, 3, address_entry(ANA), "email", error=None, now=NOW)

    store.move_subscriber(ANA, ANA.title())

    assert address_entry(ANA) == address_entry(ANA.title())
    assert store.delivery(finding.key, 3, address_entry(ANA), "email")["status"] == "sent"


def test_two_changes_that_cancel_in_a_sum_are_still_a_change(store: WatchStore) -> None:
    """Was, for one commit: the mark was sums, and one severity up and one down left it alone."""
    ben = "ben@example.org"
    store.add_subscriber(ANA, place="home", latitude=MIAMI[0], longitude=MIAMI[1], min_severity=1)
    store.add_subscriber(ben, place="home", latitude=MIAMI[0], longitude=MIAMI[1], min_severity=2)
    before = store.delivery_mark()

    store.add_subscriber(ANA, place="home", latitude=MIAMI[0], longitude=MIAMI[1], min_severity=2)
    store.add_subscriber(ben, place="home", latitude=MIAMI[0], longitude=MIAMI[1], min_severity=1)

    assert store.delivery_mark() != before
    assert store.delivery_mark() == store.delivery_mark()


def test_a_pass_that_lost_its_lease_is_not_made_by_taking_it_back(store: WatchStore) -> None:
    """Was, for one commit: asked afterwards whether it led, which took the lease back and said yes."""
    for name in ("ana", "ben", "cy"):
        store.add_subscriber(
            f"{name}@example.org", place="home", latitude=MIAMI[0], longitude=MIAMI[1]
        )
    store.report(_finding(), NOW)
    clock = [NOW]
    notifier = _Counting(store, lambda: clock[0])
    answers = iter([True, False])  # takes the pass; loses it after the first message

    class _LosesItOnce(Watcher):
        def leads(self) -> bool:
            return next(answers, True)

    watcher = _LosesItOnce(store, feeds=[], clock=lambda: clock[0], notifier=notifier)

    watcher._pass(force=False)

    assert len(notifier.mail.sent) == 1
    assert watcher._delivered_at is None  # cut short: owed again
    watcher._pass(force=False)
    assert len(notifier.mail.sent) == 3


def test_the_mark_moves_with_who_is_told_and_with_nothing_else(store: WatchStore) -> None:
    """A count the store keeps as those rows change; it was every row, read on every tick."""
    store.add_subscriber(ANA, place="home", latitude=MIAMI[0], longitude=MIAMI[1])
    store.report(_finding(), NOW)
    before = store.delivery_mark()

    store.add_subscriber(ANA, place="home", latitude=MIAMI[0], longitude=MIAMI[1])  # as it was
    store.record_delivery(_finding().key, 3, address_entry(ANA), "email", error=None, now=NOW)
    assert store.delivery_mark() == before

    store.add_subscriber(ANA, place="home", latitude=MIAMI[0], longitude=MIAMI[1], radius_km=9.0)
    assert store.delivery_mark() != before
