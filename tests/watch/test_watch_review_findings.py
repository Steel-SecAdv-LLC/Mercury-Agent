# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What a line-by-line review of the watch found, each as the failure it was.

Every test here failed on the code as reviewed. The review reproduced each one
before it was changed; the docstrings say what was observed.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest
import requests

from omni_mercury_engine.narrative.alert_bus import get_alert_bus
from omni_mercury_engine.watch import (
    Severity,
    Watcher,
    WatchStore,
    sms as sms_module,
)
from omni_mercury_engine.watch.feeds import Feed, Judgment
from omni_mercury_engine.watch.model import (
    EVERYWHERE_KINDS,
    EXTREME_BASIS,
    RANKS,
    Finding,
    public_view,
)
from omni_mercury_engine.watch.notify import (
    MAX_ATTEMPTS,
    RETRY_CAP,
    RETRY_FIRST,
    Notifier,
    address_entry,
    audience,
    configured_channels,
    retry_wait,
)
from omni_mercury_engine.watch.sms import SmsUnreachableError, TwilioSms
from omni_mercury_engine.watch.store import PhoneLimitError, open_store

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

NOW = datetime(2026, 11, 17, 1, 0, tzinfo=UTC)
NUMBER = "+14155550123"
TOKEN = "-".join(("auth", "token"))  # a stand-in, not a credential


def _entry(store: WatchStore) -> int:
    """The ledger entry of the store's one subscribed address."""
    return address_entry(str(store.subscribers()[0]["email"]))


@pytest.fixture()
def store(tmp_path: Path) -> WatchStore:
    return WatchStore(tmp_path)


def _finding(
    severity: Severity = Severity.SEVERE, *, kind: str = "river_flood", subject: str = "COCF1",
    feed: str = "river_gauges", latitude: float | None = 28.36,
    longitude: float | None = -80.80, **fields: Any,
) -> Finding:  # fmt: skip
    fields.setdefault("basis", "NWS flood category")
    return Finding(
        feed=feed, kind=kind, subject=subject, severity=severity,
        headline=f"{severity.label} flooding: St. Johns River near Cocoa", detail="It read 17 ft.",
        observed_at=NOW, source="NOAA National Weather Service",
        latitude=latitude, longitude=longitude, ongoing=True, **fields,
    )  # fmt: skip


class _Mail:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []
        self.down = False

    def send(self, *, to: str, subject: str, body: str, **_: Any) -> None:
        if self.down:
            raise ConnectionError("simulated: the mail server is down")
        self.sent.append((to, subject))


class _Fixed(Feed):
    """A feed that reports what it is given."""

    name = "river_gauges"
    title = "River gauges"
    source = "test"
    cadence = timedelta(minutes=5)
    stale_after = timedelta(hours=1)

    def __init__(self, findings: list[Finding]) -> None:
        self.findings = findings
        self.read_count = 0

    def read(self) -> pd.DataFrame:
        import pandas as pd

        self.read_count += 1
        return pd.DataFrame()

    def judge(self, frame: pd.DataFrame, now: datetime) -> Judgment:
        return Judgment(list(self.findings), "read", now, 1, 1)


def _fixed(findings: list[Finding], name: str) -> _Fixed:
    """A test feed under another feed's name."""
    named = type("_Named", (_Fixed,), {"name": name})
    feed: _Fixed = named(findings)
    return feed


# --------------------------------------------------------------------------- #
# A delivery that fails is tried again later, not five times in one pass
# --------------------------------------------------------------------------- #


class TestAFailedDeliveryIsTriedAgainLater:
    def test_a_short_outage_does_not_cost_the_alert(self, store: WatchStore) -> None:
        """Was: nine feed reads in one pass spent all five attempts; the alert was never sent.

        The watch delivers after every feed it reads. With the mail server
        down for that one pass the ledger read ``failed, 5 attempts`` at its
        end, and three more passes with the server back sent nothing.
        """
        store.add_subscriber("ana@example.org")
        store.report(_finding(), NOW)
        mail = _Mail()
        clock = [NOW]
        notifier = Notifier(store, mailer=mail, clock=lambda: clock[0])

        mail.down = True
        for _ in range(9):  # one pass: a delivery attempt after each of nine feeds
            notifier.sync()
            clock[0] += timedelta(seconds=4)
        record = store.delivery(_finding().key, 3, _entry(store), "email")
        assert (record["status"], record["attempts"]) == ("failed", 1)

        mail.down = False
        clock[0] += timedelta(minutes=5)  # the next pass
        assert notifier.sync() == {"sent": 1, "failed": 0}
        assert [to for to, _ in mail.sent] == ["ana@example.org"]

    def test_the_wait_doubles_to_an_hour_and_the_trying_lasts_about_six_hours(self) -> None:
        waits = [retry_wait(failures) for failures in range(1, MAX_ATTEMPTS)]

        assert waits[0] == RETRY_FIRST == timedelta(minutes=1)
        assert waits[:7] == [timedelta(minutes=2**n) for n in range(6)] + [RETRY_CAP]
        assert max(waits) == RETRY_CAP == timedelta(hours=1)
        assert timedelta(hours=5) < sum(waits, timedelta()) < timedelta(hours=7)

    def test_an_attempt_is_not_made_before_its_wait_has_passed(self, store: WatchStore) -> None:
        store.add_subscriber("ana@example.org")
        store.report(_finding(), NOW)
        mail = _Mail()
        mail.down = True
        clock = [NOW]
        notifier = Notifier(store, mailer=mail, clock=lambda: clock[0])
        made = []
        for _ in range(200):  # every half minute for an hour and forty minutes
            made.append(notifier.sync()["failed"])
            clock[0] += timedelta(seconds=30)

        # at 0, then 1, 2, 4, 8, 16, 32 minutes after each failure: 0, 1, 3, 7, 15, 31, 63
        assert [i // 2 for i, tried in enumerate(made) if tried] == [0, 1, 3, 7, 15, 31, 63]


# --------------------------------------------------------------------------- #
# A flood that is falling is not three alerts
# --------------------------------------------------------------------------- #


class TestWhatWasToldIsNotToldAgainWeaker:
    def test_a_finding_that_becomes_less_severe_is_not_sent_again(self, store: WatchStore) -> None:
        """Was: "Severe: Major flooding", then "Significant", then "Notable": three emails.

        ``report`` returned True, False, False, as it should. The ledger is
        keyed by the severity in the finding's text, which follows the latest
        reading down, so each step down had no delivery recorded and was sent.
        """
        store.add_subscriber("ana@example.org")
        mail = _Mail()
        clock = [NOW]
        notifier = Notifier(store, mailer=mail, clock=lambda: clock[0])
        new = []
        for severity in (Severity.SEVERE, Severity.SIGNIFICANT, Severity.NOTABLE):
            new.append(store.report(_finding(severity), clock[0], timedelta(hours=26)))
            notifier.sync()
            clock[0] += timedelta(hours=6)

        assert new == [True, False, False]
        assert [subject for _, subject in mail.sent] == [
            "[Mercury watch] Severe: severe flooding: St. Johns River near Cocoa"
        ]

    def test_one_that_becomes_more_severe_is(self, store: WatchStore) -> None:
        store.add_subscriber("ana@example.org")
        mail = _Mail()
        clock = [NOW]
        notifier = Notifier(store, mailer=mail, clock=lambda: clock[0])
        for severity in (Severity.NOTABLE, Severity.SEVERE, Severity.SIGNIFICANT):
            store.report(_finding(severity), clock[0], timedelta(hours=26))
            notifier.sync()
            clock[0] += timedelta(hours=6)

        assert [subject.split(":")[0] for _, subject in mail.sent] == [
            "[Mercury watch] Notable",
            "[Mercury watch] Severe",
        ]


# --------------------------------------------------------------------------- #
# A hazard that lost its coordinates has not become everyone's
# --------------------------------------------------------------------------- #


class TestALocalHazardWithoutCoordinatesIsNotSentEverywhere:
    def test_a_volcano_whose_location_lookup_failed_is_not_sent_to_florida(
        self, store: WatchStore
    ) -> None:
        """Was: sent to every subscriber on Earth, whatever their place and radius."""
        store.add_subscriber(
            "fl@example.org", place="Cocoa", latitude=28.37, longitude=-80.87, radius_km=50
        )
        store.add_subscriber("all@example.org")  # no place: everything
        volcano = _finding(
            kind="volcano_alert", feed="volcanoes", subject="Great Sitkin",
            latitude=None, longitude=None,
        ).to_dict()  # fmt: skip

        reached = [who["email"] for who, _ in audience(volcano, store.subscribers())]

        assert reached == ["all@example.org"]

    @pytest.mark.parametrize("kind", sorted(EVERYWHERE_KINDS))
    def test_what_is_nowhere_in_particular_still_goes_to_everyone(
        self, store: WatchStore, kind: str
    ) -> None:
        store.add_subscriber("fl@example.org", place="Cocoa", latitude=28.37, longitude=-80.87)
        finding = _finding(kind=kind, latitude=None, longitude=None).to_dict()

        assert [who["email"] for who, _ in audience(finding, store.subscribers())] == [
            "fl@example.org"
        ]

    def test_every_kind_a_feed_makes_is_either_somewhere_or_everywhere(self) -> None:
        """A new kind must be placed on one side or the other, not left to a default."""
        import inspect
        import re

        from omni_mercury_engine.watch import feeds, world

        source = inspect.getsource(feeds) + inspect.getsource(world)
        kinds = set(re.findall(r'kind="(\w+)"', source))
        somewhere = {
            "earthquake", "river_flood", "volcano_alert", "gdacs_event", "nws_alert", "cap_alert",
        }  # fmt: skip

        assert kinds == somewhere | EVERYWHERE_KINDS
        assert not somewhere & EVERYWHERE_KINDS


# --------------------------------------------------------------------------- #
# Where a subscriber is, is not published
# --------------------------------------------------------------------------- #


def _alert(kind: str, severity: Severity, feed: str) -> Finding:
    return _finding(
        severity, kind=kind, feed=feed, subject="urn:alert:1", latitude=None, longitude=None,
        area="Kanton Zürich", place_ids=(7,),
        basis="an alert its national authority has in effect at a subscribed place",
    )  # fmt: skip


class TestAPlaceBoundAlertIsNotPublic:
    def test_a_foreign_alert_is_never_public_however_severe(self) -> None:
        """Was: "severe" exempted it, so one subscriber's place published it to everyone."""
        for severity in Severity:
            assert public_view(_alert("cap_alert", severity, "world_warnings").to_dict()) is None

    def test_an_extreme_nws_alert_is_public_as_the_nationwide_report_it_also_is(self) -> None:
        shown = public_view(_alert("nws_alert", Severity.SEVERE, "warnings").to_dict())

        assert shown is not None
        assert "place_ids" not in shown
        assert shown["basis"] == EXTREME_BASIS  # not "at a subscribed place"
        assert public_view(_alert("nws_alert", Severity.SIGNIFICANT, "warnings").to_dict()) is None

    def test_a_finding_that_names_no_place_is_shown_as_it_is(self) -> None:
        flood = _finding().to_dict()

        assert public_view(flood) == {k: v for k, v in flood.items() if k != "place_ids"}

    def test_the_public_snapshot_leaves_it_out(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from omni_mercury_engine.api.routes.watch import snapshot

        monkeypatch.setenv("MERCURY_WATCH_DIR", str(tmp_path))
        store = WatchStore(tmp_path)
        store.report(_alert("cap_alert", Severity.SEVERE, "world_warnings"), NOW)
        store.report(_finding(subject="open"), NOW)

        shown = snapshot()["findings"]

        assert [finding["kind"] for finding in shown] == ["river_flood"]
        assert all("place_ids" not in finding for finding in shown)

    def test_it_is_not_put_on_the_bus_every_open_console_hears(
        self, store: WatchStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Was: published with ``principal=None``, place ids and area in the event."""
        heard: list[dict[str, Any]] = []
        monkeypatch.setattr(get_alert_bus(), "publish", heard.append)
        notifier = Notifier(store, clock=lambda: NOW)
        private = notifier.broadcast(_alert("cap_alert", Severity.SEVERE, "world_warnings"))
        public = notifier.broadcast(_finding())

        assert (private, public) == (False, True)
        assert [entry["alert"]["event"]["kind"] for entry in heard] == ["river_flood"]
        assert "place_ids" not in heard[0]["alert"]["event"]


# --------------------------------------------------------------------------- #
# Every process tells its own consoles
# --------------------------------------------------------------------------- #


class TestEveryWatcherTellsItsOwnConsoles:
    @staticmethod
    def _heard(watcher: Watcher, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        said: list[str] = []

        def broadcast(finding: dict[str, Any]) -> bool:
            said.append(str(finding["subject"]))
            return True

        monkeypatch.setattr(watcher.notifier, "broadcast", broadcast)
        return said

    def test_a_watcher_standing_by_announces_what_the_leading_one_found(
        self, store: WatchStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Was: only the process that read the feed published, on its own in-process bus.

        With ``serve --workers 4`` the consoles of three workers in four heard
        nothing, and with ``mercury-agent watch`` beside a server none did.
        """
        leader = Watcher(store, [_Fixed([_finding(subject="COCF1")])], clock=lambda: NOW)
        standby = Watcher(store, [_Fixed([])], clock=lambda: NOW)
        leader_heard = self._heard(leader, monkeypatch)
        standby_heard = self._heard(standby, monkeypatch)

        leader.run_once()

        assert leader_heard == ["COCF1"]
        assert standby_heard == []
        assert standby.announce() == 1
        assert standby_heard == ["COCF1"]
        assert standby.announce() == 0  # once

    def test_what_was_found_before_a_process_started_is_not_replayed(
        self, store: WatchStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        Watcher(store, [_Fixed([_finding(subject="old")])], clock=lambda: NOW).run_once()

        late = Watcher(store, [_Fixed([])], clock=lambda: NOW)
        heard = self._heard(late, monkeypatch)

        assert late.announce() == 0 and heard == []
        store.report(_finding(subject="new"), NOW + timedelta(minutes=5))
        assert late.announce() == 1 and heard == ["new"]

    def test_a_finding_is_announced_when_new_and_when_more_severe_only(
        self, store: WatchStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        watcher = Watcher(store, [_Fixed([])], clock=lambda: NOW)
        heard = self._heard(watcher, monkeypatch)
        for minutes, severity in enumerate(
            (Severity.NOTABLE, Severity.NOTABLE, Severity.SEVERE, Severity.SIGNIFICANT)
        ):
            store.report(_finding(severity), NOW + timedelta(minutes=minutes), timedelta(hours=1))
            watcher.announce()

        assert heard == ["COCF1", "COCF1"]

    def test_announcements_do_not_pile_up(self, store: WatchStore) -> None:
        store.report(_finding(subject="then"), NOW)
        store.report(_finding(subject="now"), NOW + store.ANNOUNCEMENTS_KEPT + timedelta(hours=1))

        assert [f["subject"] for _, f in store.announcements_after(0)] == ["now"]


# --------------------------------------------------------------------------- #
# The limit on verification texts
# --------------------------------------------------------------------------- #


class TestTheCodeLimitCannotBeReset:
    def test_removing_the_number_does_not_start_the_count_again(self, store: WatchStore) -> None:
        """Was: 21 codes to one number in under a second by removing and adding it."""
        issued = 0
        clock = NOW
        with pytest.raises(PhoneLimitError, match="codes have been sent today"):
            for _ in range(21):
                store.request_phone("ana@example.org", NUMBER, now=clock)
                issued += 1
                store.remove_phone("ana@example.org")
                clock += timedelta(minutes=2)

        assert issued == store.CODE_SENDS_PER_DAY == 5

    def test_removing_it_does_not_skip_the_minute_between_two_either(
        self, store: WatchStore
    ) -> None:
        store.request_phone("ana@example.org", NUMBER, now=NOW)
        store.remove_phone("ana@example.org")

        with pytest.raises(PhoneLimitError, match="a moment ago"):
            store.request_phone("ana@example.org", NUMBER, now=NOW + timedelta(seconds=5))

    def test_one_number_is_limited_whichever_addresses_ask(self, store: WatchStore) -> None:
        """Five accounts, one stranger's phone: the count is the number's too."""
        with pytest.raises(PhoneLimitError):
            for n in range(8):
                store.request_phone(f"a{n}@example.org", NUMBER, now=NOW + timedelta(minutes=2 * n))

        asked = [bool(store.phone_of(f"a{n}@example.org")) for n in range(8)]
        assert asked == [True] * 5 + [False] * 3

    def test_the_count_is_kept_without_the_address_or_the_number(self, store: WatchStore) -> None:
        import sqlite3

        store.request_phone("ana@example.org", NUMBER, now=NOW)
        store.remove_phone("ana@example.org")

        with sqlite3.connect(store.path) as connection:
            kept = " ".join(str(row) for row in connection.execute("SELECT * FROM code_sends"))
        assert "ana@example.org" not in kept and NUMBER not in kept and "4155550123" not in kept

    def test_a_code_that_never_left_is_not_counted(self, store: WatchStore) -> None:
        """The provider could not be reached: nobody was texted and nothing was spent."""
        for n in range(8):  # eight failures in a row, no waiting, none counted
            store.request_phone("ana@example.org", NUMBER, now=NOW + timedelta(seconds=n))
            store.release_code_send("ana@example.org", NUMBER)

        code = store.request_phone("ana@example.org", NUMBER, now=NOW + timedelta(seconds=9))
        assert store.confirm_phone("ana@example.org", code, NOW + timedelta(seconds=10))


# --------------------------------------------------------------------------- #
# The provider being unreachable is a named failure
# --------------------------------------------------------------------------- #


class TestAProviderThatCannotBeReached:
    @pytest.mark.parametrize(
        "failure", [requests.ConnectionError("refused"), requests.Timeout("20 s"), OSError("dns")]
    )
    def test_it_is_one_error_a_caller_can_expect(
        self, failure: Exception, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Was: the HTTP library's own exception, which no caller caught: a 500, a traceback."""
        from omni_mercury_engine.security.safe_http import SafeHTTPClient

        def unreachable(*_a: Any, **_k: Any) -> Any:
            raise failure

        monkeypatch.setattr(SafeHTTPClient, "post_form", unreachable)
        sender = TwilioSms(account_sid="AC" + "0" * 32, auth_token=TOKEN, from_number=NUMBER)

        with pytest.raises(SmsUnreachableError, match=type(failure).__name__) as raised:
            sender.send(to=NUMBER, body="Mercury watch: your code is 123456.")
        assert "123456" not in str(raised.value)  # nothing of the request is quoted

    def test_the_command_says_so_and_does_not_count_the_code(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from click.testing import CliRunner

        from omni_mercury_engine.cli import main

        class Down:
            def send(self, *, to: str, body: str) -> None:
                raise SmsUnreachableError("Twilio could not be reached (Timeout)")

        monkeypatch.setattr(sms_module, "build_sms_sender", Down)
        WatchStore(tmp_path).add_subscriber("ana@example.org")
        ask = ["alerts", "phone", "--email", "ana@example.org", "--number", NUMBER,
               "--state-dir", str(tmp_path)]  # fmt: skip

        results = [CliRunner().invoke(main, ask) for _ in range(7)]

        assert all(result.exit_code == 1 for result in results)
        assert all("could not be reached" in result.output for result in results)
        assert all("Traceback" not in result.output for result in results)
        assert all("wait a minute" not in result.output for result in results)


# --------------------------------------------------------------------------- #
# One rule for what an address is; one table for a severity's rank
# --------------------------------------------------------------------------- #


class TestOneRule:
    @pytest.mark.parametrize(
        "address", ["o'brien@example.com", "user@exämple.de", "a_b@sub_domain.example.com"]
    )
    def test_an_address_an_account_may_have_is_one_the_watch_can_tell(
        self, store: WatchStore, address: str
    ) -> None:
        """Was: verified by the account system, refused by the watch's stricter pattern."""
        from omni_mercury_engine.api.auth_service import _is_valid_email

        assert _is_valid_email(address)
        store.add_subscriber(address)
        store.request_phone(address, NUMBER, now=NOW)
        assert store.move_subscriber("old@example.org", address) == 0

    @pytest.mark.parametrize("address", ["", "no-at-sign", "two@@example.com", "a b@example.com"])
    def test_what_is_not_an_address_is_refused_by_both(
        self, store: WatchStore, address: str
    ) -> None:
        from omni_mercury_engine.api.auth_service import _is_valid_email

        assert not _is_valid_email(address)
        with pytest.raises(ValueError, match="not an email address"):
            store.add_subscriber(address)

    def test_a_severitys_rank_is_the_same_wherever_it_is_named(self) -> None:
        from omni_mercury_engine import cli
        from omni_mercury_engine.api.routes import accounts
        from omni_mercury_engine.watch import notify

        assert RANKS == {"notable": 1, "significant": 2, "severe": 3}
        assert cli._SEVERITIES == accounts._WATCH_SEVERITY_RANK == notify._RANK == RANKS

    def test_a_phone_number_is_judged_by_one_pattern(self) -> None:
        from omni_mercury_engine.watch import store as store_module

        assert getattr(store_module, "E164") is sms_module.E164  # noqa: B009 - not re-exported
        assert not hasattr(store_module, "_E164")


# --------------------------------------------------------------------------- #
# The public page does not rebuild the store on every request
# --------------------------------------------------------------------------- #


class TestTheSnapshotIsCheap:
    def test_the_store_is_opened_once_in_a_process(self, tmp_path: Path) -> None:
        WatchStore(tmp_path)

        assert open_store(tmp_path) is open_store(tmp_path)

    def test_a_request_runs_no_schema_and_builds_no_sender(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Was: seven connections a request, and an ERROR logged on each when texts are misconfigured."""
        from omni_mercury_engine.api.routes.watch import snapshot

        monkeypatch.setenv("MERCURY_WATCH_DIR", str(tmp_path))
        monkeypatch.setenv(sms_module.ACCOUNT_SID_ENV, "not-a-sid")
        monkeypatch.setenv(sms_module.AUTH_TOKEN_ENV, TOKEN)
        WatchStore(tmp_path).report(_finding(), NOW)
        snapshot()  # the first request opens the store and works the channels out
        made: list[Path] = []
        monkeypatch.setattr(WatchStore, "__init__", lambda self, *a, **k: made.append(tmp_path))

        caplog.clear()  # the first request said so once; that is right
        with caplog.at_level(logging.ERROR, logger="omni_mercury_engine.watch.notify"):
            shown = [snapshot() for _ in range(5)]

        assert made == []
        assert not [r for r in caplog.records if "misconfigured" in r.getMessage()]
        assert shown[-1]["telling"]["sms"] is False
        assert len(shown[-1]["findings"]) == 1

    def test_the_channels_are_worked_out_again_when_the_settings_change(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MERCURY_SMTP_HOST", raising=False)
        assert configured_channels()["email"] is False

        monkeypatch.setenv("MERCURY_SMTP_HOST", "smtp.example.org")
        assert configured_channels()["email"] is True


# =========================================================================== #
# The second pass over the same diff
# =========================================================================== #


class TestAFindingThatClimbsBackIsToldAgain:
    def test_a_volcano_back_at_warning_after_months_at_advisory_is_announced_and_sent(
        self, store: WatchStore
    ) -> None:
        """Was: never again. The row kept the highest severity it had ever reached.

        WARNING on day 0, ADVISORY from day 30, WARNING on day 90, never
        NORMAL in between, so one standing finding: ``report`` answered False
        on day 90 and nobody was told.
        """
        store.add_subscriber("ana@example.org")
        mail = _Mail()
        clock = [NOW]
        notifier = Notifier(store, mailer=mail, clock=lambda: clock[0])
        new = []
        for day, severity in ((0, Severity.SEVERE), (30, Severity.NOTABLE), (90, Severity.SEVERE)):
            clock[0] = NOW + timedelta(days=day)
            new.append(store.report(_finding(severity), clock[0], timedelta(hours=26)))
            notifier.sync()

        assert new == [True, False, True]
        assert [subject.split(":")[0] for _, subject in mail.sent] == [
            "[Mercury watch] Severe",
            "[Mercury watch] Severe",
        ]
        assert len(store.announcements_after(0)) == 1  # the first is older than a day


class TestWhatIsShown:
    def test_private_alerts_do_not_crowd_the_public_page(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Was: one public flood and sixty place-bound alerts gave a page of nothing."""
        from omni_mercury_engine.api.routes.watch import snapshot

        monkeypatch.setenv("MERCURY_WATCH_DIR", str(tmp_path))
        store = WatchStore(tmp_path)
        store.report(_finding(subject="flood"), NOW)
        for n in range(60):
            alert = _alert("cap_alert", Severity.NOTABLE, "world_warnings")
            store.report(
                Finding(**{**alert.__dict__, "subject": f"urn:alert:{n}"}),
                NOW + timedelta(minutes=1),
            )

        shown = snapshot(limit=50)["findings"]

        assert [finding["subject"] for finding in shown] == ["flood"]

    def test_what_stands_comes_first_the_most_severe_first(self, store: WatchStore) -> None:
        """Was: newest sighting first, so fifty gauges hid a magnitude 7 of two hours ago."""
        quake = _finding(Severity.SEVERE, kind="earthquake", feed="earthquakes", subject="us7000")
        store.report(Finding(**{**quake.__dict__, "ongoing": False}), NOW - timedelta(hours=2))
        for n in range(60):
            store.report(_finding(Severity.NOTABLE, subject=f"gauge{n}"), NOW)

        shown = store.overview(limit=50, now=NOW)

        assert shown[0]["subject"] == "us7000"
        assert len(store.overview(open_only=True, now=NOW)) == 61

    def test_the_commands_first_line_counts_everything_that_stands(self, tmp_path: Path) -> None:
        """Was: "50 findings standing" of 181."""
        from click.testing import CliRunner

        from omni_mercury_engine.cli import main

        store = WatchStore(tmp_path)
        for n in range(181):
            store.report(_finding(Severity.NOTABLE, subject=f"gauge{n}"), datetime.now(UTC))

        said = CliRunner().invoke(main, ["watch", "--status", "--state-dir", str(tmp_path)])

        assert said.exit_code == 0, said.output
        assert said.output.splitlines()[0] == "181 findings standing"


class TestEachFeedHasItsOwnReader:
    def test_a_watch_of_one_feed_does_not_stop_the_others_being_read(
        self, store: WatchStore
    ) -> None:
        """Was: ``watch --feed earthquakes`` took the one lease; eight feeds went unread."""
        import threading

        quakes, floods = _fixed([], "earthquakes"), _fixed([], "river_gauges")
        servers_quakes = _fixed([], "earthquakes")
        only_quakes = Watcher(store, [quakes], clock=lambda: NOW)
        server = Watcher(store, [servers_quakes, floods], clock=lambda: NOW)
        assert only_quakes.reads(quakes)  # the one-feed watch is running

        class Once(threading.Event):
            def wait(self, timeout: float | None = None) -> bool:
                self.set()
                return True

        server.run_forever(tick_seconds=0.0, stop=Once())

        assert (servers_quakes.read_count, floods.read_count) == (0, 1)

    def test_a_command_standing_by_prints_what_the_server_found(self, store: WatchStore) -> None:
        """Was: "New findings appear below", and none ever did: a standby never reads."""
        printed: list[str] = []
        command = Watcher(
            store, [_Fixed([])], clock=lambda: NOW,
            on_announced=lambda entry: printed.append(str(entry["subject"])),
        )  # fmt: skip
        server = Watcher(store, [_Fixed([_finding(subject="COCF1")])], clock=lambda: NOW)
        own: list[str] = []
        server.on_announced = lambda entry: own.append(str(entry["subject"]))

        server.run_once()
        command.announce()

        assert printed == ["COCF1"]
        assert own == []  # its own finding is not said to it twice

    def test_a_watcher_that_can_tell_no_one_does_not_hold_the_lease_for_telling(
        self, store: WatchStore
    ) -> None:
        """Was: a command run without the server's mail settings took it, and nobody was told."""
        mute = Watcher(store, [_Fixed([])], clock=lambda: NOW)
        able = Watcher(
            store, [_Fixed([])], clock=lambda: NOW,
            notifier=Notifier(store, mailer=_Mail(), clock=lambda: NOW),
        )  # fmt: skip

        assert mute.leads() is False
        assert able.leads() is True

    def test_stopping_gives_the_leases_up_at_once(self, store: WatchStore) -> None:
        """Was: a server restarted mid-read left them to lapse: fifteen minutes with no watch."""
        feed = _Fixed([])
        first = Watcher(
            store, [feed], clock=lambda: NOW,
            notifier=Notifier(store, mailer=_Mail(), clock=lambda: NOW),
        )  # fmt: skip
        assert first.leads() and first.reads(feed)

        first.stop()

        assert store.lease() == {} and store.lease("feed:river_gauges") == {}
        assert not first.leads() and not first.reads(feed)  # and takes none again
        second = Watcher(
            store, [feed], clock=lambda: NOW,
            notifier=Notifier(store, mailer=_Mail(), clock=lambda: NOW),
        )  # fmt: skip
        assert second.leads() and second.reads(feed)

    def test_a_failure_while_checking_past_findings_does_not_mark_the_feed_down(
        self, store: WatchStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        watcher = Watcher(store, [_Fixed([_finding()])], clock=lambda: NOW)

        def locked(*_a: Any, **_k: Any) -> Any:
            raise RuntimeError("database is locked")

        monkeypatch.setattr(watcher, "review_feed", locked)

        (status,) = watcher.run_once()

        assert status.state == "ok"
        assert store.statuses()[0]["state"] == "ok"


class TestOneUnansweredPlaceDoesNotKeepEverythingStanding:
    def test_only_what_is_in_effect_at_the_unchecked_place_is_kept(
        self, store: WatchStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Was: one failed lookup in 250 skipped the closing of every finding of the feed."""
        here = _alert("nws_alert", Severity.NOTABLE, "warnings")
        at_seven = Finding(**{**here.__dict__, "subject": "a", "place_ids": (7,)})
        at_eight = Finding(**{**here.__dict__, "subject": "b", "place_ids": (8,)})
        nationwide = Finding(**{**here.__dict__, "subject": "c", "place_ids": ()})
        feed = _fixed([at_seven, at_eight, nationwide], "warnings")
        watcher = Watcher(store, [feed], clock=lambda: NOW)
        watcher.run_once()

        # The next read shows none of them, and could not look at place 8.
        monkeypatch.setattr(
            feed,
            "judge",
            lambda frame, now: Judgment(
                [], "read", now, 0, 0, partial=True, unchecked_places=frozenset({8})
            ),
        )
        watcher.run_once()

        assert sorted(f["subject"] for f in store.standing(now=NOW)) == ["b"]

    def test_a_read_that_cannot_say_where_it_did_not_look_still_ends_nothing(
        self, store: WatchStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        feed = _Fixed([_finding(subject="a")])
        watcher = Watcher(store, [feed], clock=lambda: NOW)
        watcher.run_once()
        monkeypatch.setattr(
            feed, "judge", lambda frame, now: Judgment([], "read", now, 0, 0, partial=True)
        )

        watcher.run_once()

        assert [f["subject"] for f in store.standing(now=NOW)] == ["a"]


class TestAWarningIsOneFindingAcrossItsUpdates:
    def test_the_thread_is_the_first_message_of_the_warning(self) -> None:
        """Measured 2026-11-17: 194 of 517 alerts in effect were updates, each with a new id."""
        from omni_mercury_engine.watch.feeds import _alert_thread

        first, second, third = "urn:oid:1.1", "urn:oid:1.2", "urn:oid:1.3"
        references = [
            {"identifier": second, "sent": "2026-11-17T12:00:00-09:00"},
            {"identifier": first, "sent": "2026-11-17T06:00:00-09:00"},
        ]

        assert _alert_thread(first, []) == first
        assert _alert_thread(second, references[1:]) == first
        assert _alert_thread(third, references) == first

    def test_an_update_from_the_service_is_the_same_finding(
        self, store: WatchStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Was: keyed by message, so each update of a warning was told as a new alert."""
        from omni_mercury_engine.watch.feeds import WarningsFeed

        def alert(identifier: str, references: list[dict[str, str]]) -> dict[str, Any]:
            return {
                "id": identifier, "event": "Flood Warning", "severity": "Extreme",
                "areaDesc": "Brevard, FL", "headline": "Flood Warning", "messageType": "Update",
                "senderName": "NWS Melbourne FL", "sent": "2026-11-17T01:00:00+00:00",
                "references": references,
            }  # fmt: skip

        feed = WarningsFeed()
        feed.bind(store)
        keys = []
        for identifier, references in (
            ("urn:oid:1.1", []),
            ("urn:oid:1.2", [{"identifier": "urn:oid:1.1", "sent": "2026-11-17T00:00:00+00:00"}]),
        ):
            issued = [alert(identifier, references)]
            monkeypatch.setattr(feed, "_alerts", lambda params, issued=issued: issued)
            (finding,) = feed.judge(feed.read(), NOW).findings
            keys.append(finding.key)
            assert finding.reference == identifier  # the message it is now

        assert keys[0] == keys[1]

    def test_a_foreign_authoritys_update_is_the_same_finding_too(self) -> None:
        """The recorded Deutscher Wetterdienst alert is itself an update of an earlier one."""
        from pathlib import Path as _Path

        from omni_mercury_engine.watch.world import parse_cap

        cap = (_Path(__file__).parent / "data" / "dwd_wind_gusts.cap.xml").read_text()

        parsed = parse_cap(cap)

        assert parsed["msg_type"] == "Update"
        assert parsed["thread"] != parsed["identifier"]
        assert parsed["thread"].endswith("b71f5037-8ec8-448b-bce4-21d51d533a92.ENG")


class TestAnEarthquakeUnderAnotherId:
    def test_an_event_already_recorded_under_one_of_its_ids_keeps_it(
        self, store: WatchStore
    ) -> None:
        """USGS lists every id an event has; the preferred one can change between reads."""
        import pandas as pd

        from omni_mercury_engine.watch.feeds import EarthquakeFeed

        feed = EarthquakeFeed()
        feed.bind(store)

        def frame(preferred: str) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "time": NOW.timestamp() * 1000, "latitude": -5.6, "longitude": 132.7,
                        "depth": 10.0, "magnitude": 6.6, "place": "Tual, Indonesia",
                        "event_id": preferred, "event_ids": "pt26322050,us7000tywi",
                    }
                ]
            )  # fmt: skip

        (first,) = feed.judge(frame("pt26322050"), NOW).findings
        store.report(first, NOW)
        (second,) = feed.judge(frame("us7000tywi"), NOW).findings

        assert first.key == second.key
        assert store.report(second, NOW + timedelta(minutes=5)) is False


class TestDeclarations:
    @staticmethod
    def _frame(designations: dict[str, int]) -> Any:
        import pandas as pd

        days = pd.date_range("2026-09-01", "2026-09-10", freq="D", tz="UTC")
        return pd.DataFrame(
            {
                "timestamp": days,
                "designations": [designations.get(f"{d:%m-%d}", 0) for d in days],
                "states": 1,
                "disasters": 1,
            }
        )

    def test_the_review_counts_the_days_the_finding_said(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Was: 10 on 6 Sep and 5 on 8 Sep, judged on the 10th, reported 5; reviewed as 15, "revised"."""
        from omni_mercury_engine.watch.feeds import DeclarationFeed

        feed = DeclarationFeed()
        frame = self._frame({"09-06": 10, "09-08": 5})
        monkeypatch.setattr(feed, "read", lambda: frame)
        now = datetime(2026, 9, 10, 12, tzinfo=UTC)

        (finding,) = feed.judge(frame, now).findings
        outcome = feed.review(finding.to_dict(), now + timedelta(days=2))

        assert finding.value == 15.0 and "In the 3 days to 08 September 2026" in finding.detail
        assert outcome is not None and outcome.verdict == "confirmed"

    def test_the_first_declaration_of_a_year_is_not_its_busiest_week(self) -> None:
        """Was: one designation, no earlier one, rated significant and called the busiest week."""
        from omni_mercury_engine.watch.feeds import DeclarationFeed

        frame = self._frame({"09-09": 1})

        (finding,) = DeclarationFeed().judge(frame, datetime(2026, 9, 10, 12, tzinfo=UTC)).findings

        assert finding.severity is Severity.NOTABLE
        assert "busiest" not in finding.headline

    def test_the_last_days_of_december_are_still_read_in_january(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Was: from 1 January the record began on 1 January."""
        import pandas as pd

        from omni_mercury_engine.loaders.fema_activity_loader import FEMAActivityLoader

        loader = FEMAActivityLoader(cache_dir=tmp_path, max_retries=0)
        asked: list[int] = []

        def declarations(year: int) -> pd.DataFrame:
            asked.append(year)
            return pd.DataFrame({"declarationDate": [], "state": [], "disasterNumber": []})

        monkeypatch.setattr(loader, "_fetch_tropical_declarations", declarations)
        monkeypatch.setattr(
            pd.Timestamp, "now", classmethod(lambda cls, tz=None: pd.Timestamp("2027-01-02", tz=tz))
        )

        frame = loader.fetch_realtime(days_before_year=14)

        assert asked == [2027, 2026]
        assert str(frame["timestamp"].min().date()) == "2026-12-18"
        assert str(frame["timestamp"].max().date()) == "2027-01-02"
        assert list(loader.fetch_realtime()["timestamp"].dt.year.unique()) == [2027]


class TestOneMessageForAnAddress:
    def test_the_nearest_place_changing_does_not_send_the_finding_again(
        self, store: WatchStore
    ) -> None:
        """Was: the ledger entry was the place's, so "home" then "work" was two emails."""
        store.add_subscriber("ana@example.org", place="Home", latitude=28.37, longitude=-80.87)
        store.add_subscriber("ana@example.org", place="Work", latitude=28.60, longitude=-81.20)
        mail = _Mail()
        notifier = Notifier(store, mailer=mail, clock=lambda: NOW)
        near_home = _finding(latitude=28.36, longitude=-80.80)
        store.report(near_home, NOW)
        notifier.sync()

        # Seen again, now nearer the other place.
        store.report(
            Finding(**{**near_home.__dict__, "latitude": 28.61, "longitude": -81.21}),
            NOW + timedelta(minutes=5),
        )
        notifier.sync()

        assert [to for to, _ in mail.sent] == ["ana@example.org"]


class TestTheLedgerReadIsOfWhatStands:
    def test_an_earthquake_of_last_week_is_not_loaded_on_every_pass(
        self, store: WatchStore
    ) -> None:
        """Was: a single occurrence is never given an end, so its deliveries were loaded for ever."""
        quake = _finding(kind="earthquake", feed="earthquakes", subject="us7000")
        store.report(Finding(**{**quake.__dict__, "ongoing": False}), NOW - timedelta(days=7))
        store.record_delivery(quake.key, 3, 1, "email", error=None, now=NOW - timedelta(days=7))
        flood = _finding(subject="COCF1")
        store.report(flood, NOW)
        store.record_delivery(flood.key, 3, 1, "email", error=None, now=NOW)

        assert list(store.standing_deliveries(NOW)) == [(flood.key, 3, 1, "email")]


class TestSmallThings:
    def test_a_name_that_does_not_resolve_is_the_provider_being_unreachable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Was: the egress gate's ``ValueError``, so the code was counted for a text never sent."""
        from omni_mercury_engine.security.safe_http import SafeHTTPClient, UnsafeURLError

        def no_dns(*_a: Any, **_k: Any) -> Any:
            raise UnsafeURLError("host 'api.twilio.com' did not resolve")

        monkeypatch.setattr(SafeHTTPClient, "post_form", no_dns)
        sender = TwilioSms(account_sid="AC" + "0" * 32, auth_token=TOKEN, from_number=NUMBER)

        with pytest.raises(SmsUnreachableError, match="UnsafeURLError"):
            sender.send(to=NUMBER, body="x")

    def test_the_link_in_a_message_is_the_deployments_own_address(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Was: read from a setting nothing documented, so no alert carried the link."""
        from omni_mercury_engine.watch.notify import notice

        monkeypatch.delenv("MERCURY_PUBLIC_URL", raising=False)
        monkeypatch.setenv("MERCURY_PUBLIC_BASE_URL", "https://mercury.example.org/")

        _subject, body = notice(_finding().to_dict())

        assert "What Mercury is watching now: https://mercury.example.org/watch" in body
