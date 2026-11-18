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

    def __init__(self, findings: list[Finding], name: str = "river_gauges") -> None:
        self.findings = findings
        self.name = name

    def read(self) -> pd.DataFrame:
        import pandas as pd

        return pd.DataFrame()

    def judge(self, frame: pd.DataFrame, now: datetime) -> Judgment:
        return Judgment(list(self.findings), "read", now, 1, 1)


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
        record = store.delivery(_finding().key, 3, 1, "email")
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

    def test_it_is_not_put_on_the_bus_every_open_console_hears(self, store: WatchStore) -> None:
        """Was: published with ``principal=None``, place ids and area in the event."""
        heard: list[dict[str, Any]] = []
        bus = get_alert_bus()
        real = bus.publish
        bus.publish = heard.append  # type: ignore[method-assign]
        try:
            notifier = Notifier(store, clock=lambda: NOW)
            private = notifier.broadcast(_alert("cap_alert", Severity.SEVERE, "world_warnings"))
            public = notifier.broadcast(_finding())
        finally:
            bus.publish = real  # type: ignore[method-assign]

        assert (private, public) == (False, True)
        assert [entry["alert"]["event"]["kind"] for entry in heard] == ["river_flood"]
        assert "place_ids" not in heard[0]["alert"]["event"]


# --------------------------------------------------------------------------- #
# Every process tells its own consoles
# --------------------------------------------------------------------------- #


class TestEveryWatcherTellsItsOwnConsoles:
    @staticmethod
    def _heard(watcher: Watcher) -> list[str]:
        said: list[str] = []
        watcher.notifier.broadcast = lambda finding: (  # type: ignore[method-assign]
            said.append(str(finding["subject"])) or True
        )
        return said

    def test_a_watcher_standing_by_announces_what_the_leading_one_found(
        self, store: WatchStore
    ) -> None:
        """Was: only the process that read the feed published, on its own in-process bus.

        With ``serve --workers 4`` the consoles of three workers in four heard
        nothing, and with ``mercury-agent watch`` beside a server none did.
        """
        leader = Watcher(store, [_Fixed([_finding(subject="COCF1")])], clock=lambda: NOW)
        standby = Watcher(store, [_Fixed([])], clock=lambda: NOW)
        leader_heard, standby_heard = self._heard(leader), self._heard(standby)

        assert leader.leads() and not standby.leads()
        leader.run_once()

        assert leader_heard == ["COCF1"]
        assert standby_heard == []
        assert standby.announce() == 1
        assert standby_heard == ["COCF1"]
        assert standby.announce() == 0  # once

    def test_what_was_found_before_a_process_started_is_not_replayed(
        self, store: WatchStore
    ) -> None:
        Watcher(store, [_Fixed([_finding(subject="old")])], clock=lambda: NOW).run_once()

        late = Watcher(store, [_Fixed([])], clock=lambda: NOW)
        heard = self._heard(late)

        assert late.announce() == 0 and heard == []
        store.report(_finding(subject="new"), NOW + timedelta(minutes=5))
        assert late.announce() == 1 and heard == ["new"]

    def test_a_finding_is_announced_when_new_and_when_more_severe_only(
        self, store: WatchStore
    ) -> None:
        watcher = Watcher(store, [_Fixed([])], clock=lambda: NOW)
        heard = self._heard(watcher)
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

        assert store_module.E164 is sms_module.E164
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
