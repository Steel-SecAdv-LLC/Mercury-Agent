# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The watch tells the people a finding concerns, and what to do.

Before this the watch reported a river in major flood and told no one: its
findings were printed where it ran, stored and served to whoever came to
look, and none said what a person there should do. A consumer was left
uninformed and unprepared.

These tests pin: that every kind of finding carries the responsible agency's
action; who a finding is sent to; that it is sent once, retried when it
fails and never claimed when it could not be sent; that someone who
subscribes during a flood is told of it; and that the National Weather
Service's own alerts are relayed with its instruction, word for word.
"""

from __future__ import annotations

import json
import stat
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest
from click.testing import CliRunner

from omni_mercury_engine.api.routes.watch import snapshot
from omni_mercury_engine.cli import main
from omni_mercury_engine.narrative.alert_bus import to_alert
from omni_mercury_engine.watch import FEEDS, Severity, Watcher, WatchStore
from omni_mercury_engine.watch.feeds import Feed, Judgment, WarningsFeed
from omni_mercury_engine.watch.guidance import EMERGENCY_LINE, GUIDANCE, with_guidance
from omni_mercury_engine.watch.model import Finding
from omni_mercury_engine.watch.notify import MAX_ATTEMPTS, Notifier, audience, distance_km, notice
from omni_mercury_engine.watch.voice import finding_text

if TYPE_CHECKING:
    from pathlib import Path

NOW = datetime(2026, 11, 17, 1, 0, tzinfo=UTC)
COCOA = (28.37, -80.87)


@pytest.fixture()
def store(tmp_path: Path) -> WatchStore:
    return WatchStore(tmp_path)


def _finding(
    severity: Severity = Severity.SEVERE, *, kind: str = "river_flood", subject: str = "COCF1",
    latitude: float | None = 28.36, longitude: float | None = -80.80, **fields: Any,
) -> Finding:  # fmt: skip
    return Finding(
        feed="river_gauges", kind=kind, subject=subject, severity=severity,
        headline="Major flooding: St. Johns River near Cocoa, Florida", detail="It read 17.09 ft.",
        observed_at=NOW, source="NOAA National Weather Service", basis="NWS flood category",
        latitude=latitude, longitude=longitude, ongoing=True, **fields,
    )  # fmt: skip


class _Mail:
    """A mailer that records, and fails for the addresses it is told to."""

    def __init__(self, failing: set[str] | None = None) -> None:
        self.sent: list[tuple[str, str, str]] = []
        self.failing = failing or set()

    def send(self, *, to: str, subject: str, body: str, **_: Any) -> None:
        if to in self.failing:
            raise ConnectionError("simulated: mailbox unavailable")
        self.sent.append((to, subject, body))


# --------------------------------------------------------------------------- #
# What to do
# --------------------------------------------------------------------------- #


class TestGuidance:
    def test_every_kind_of_finding_a_feed_makes_says_what_to_do(self) -> None:
        import inspect

        from omni_mercury_engine.watch import feeds

        kinds = set(__import__("re").findall(r'kind="(\w+)"', inspect.getsource(feeds)))
        assert kinds == set(GUIDANCE) and len(kinds) == 7
        for kind, (action, source) in GUIDANCE.items():
            assert len(action) > 40 and source, kind

    def test_the_action_is_the_agencys_and_is_attributed(self) -> None:
        flood = with_guidance(_finding())

        assert "turn around, don't drown" in flood.action
        assert flood.action_source == "National Weather Service flood safety guidance"
        assert "1-800-621-3362" in GUIDANCE["declarations"][0]
        assert "tsunami.gov" in GUIDANCE["buoy_event_mode"][0]

    def test_an_action_the_source_gave_is_kept_word_for_word(self) -> None:
        given = _finding(kind="nws_alert", action="Move to higher ground now.", action_source="NWS")

        assert with_guidance(given) is given

    def test_the_watch_attaches_it_before_the_finding_is_stored_or_said(
        self, store: WatchStore
    ) -> None:
        said: list[Finding] = []
        Watcher(store, [_Fixed([_finding()])], on_finding=said.append, clock=lambda: NOW).run_once()

        assert "turn around" in said[0].action and "turn around" in store.findings()[0]["action"]
        text = finding_text(store.findings()[0], NOW)
        assert "    What to do: Never walk or drive through flood water" in text
        assert "(National Weather Service flood safety guidance)" in text


class _Fixed(Feed):
    name = "river_gauges"
    title = "Fixed"
    source = "test"
    cadence = timedelta(minutes=5)
    stale_after = timedelta(days=1)

    def __init__(self, findings: list[Finding]) -> None:
        self.findings = findings

    def read(self) -> pd.DataFrame:
        return pd.DataFrame()

    def judge(self, frame: pd.DataFrame, now: datetime) -> Judgment:
        return Judgment(list(self.findings), "fixed", now, 1, 1)


# --------------------------------------------------------------------------- #
# Who
# --------------------------------------------------------------------------- #


def _subscriber(
    ident: int, place: tuple[float, float] | None = COCOA, **fields: Any
) -> dict[str, Any]:
    return {
        "id": ident, "email": f"p{ident}@example.org", "place": "Home" if place else "",
        "latitude": place[0] if place else None, "longitude": place[1] if place else None,
        "radius_km": 60.0, "min_severity": 1, **fields,
    }  # fmt: skip


class TestAudience:
    def test_distance_is_great_circle(self) -> None:
        assert distance_km(28.37, -80.87, 28.37, -80.87) == 0.0
        assert distance_km(0.0, 0.0, 0.0, 1.0) == pytest.approx(111.195, abs=0.01)
        assert distance_km(52.0765, -176.1109, 19.421, -155.287) == pytest.approx(4040, rel=0.02)

    def test_a_located_finding_goes_to_those_within_their_radius(self) -> None:
        near, far, everywhere = _subscriber(1), _subscriber(2, (47.6, -122.3)), _subscriber(3, None)

        told = audience(_finding().to_dict(), [near, far, everywhere])

        assert [(s["id"], None if d is None else round(d)) for s, d in told] == [(1, 7), (3, None)]

    def test_severity_below_what_was_asked_for_is_not_sent(self) -> None:
        only_severe = _subscriber(1, min_severity=3)

        assert audience(_finding(Severity.NOTABLE).to_dict(), [only_severe]) == []
        assert len(audience(_finding(Severity.SEVERE).to_dict(), [only_severe])) == 1

    def test_an_alert_goes_to_the_place_the_service_says_it_is_in_effect_at(self) -> None:
        """A tornado warning in Kansas must not wake someone in Florida."""
        here, elsewhere, everywhere = _subscriber(1), _subscriber(2), _subscriber(3, None)
        local = _finding(kind="nws_alert", latitude=None, longitude=None, place_ids=(1,)).to_dict()
        national = _finding(kind="nws_alert", latitude=None, longitude=None).to_dict()

        assert [s["id"] for s, _ in audience(local, [here, elsewhere, everywhere])] == [1, 3]
        assert [s["id"] for s, _ in audience(national, [here, elsewhere, everywhere])] == [3]

    def test_a_finding_with_no_place_goes_to_everyone(self) -> None:
        storm = _finding(kind="geomagnetic_storm", latitude=None, longitude=None).to_dict()

        assert len(audience(storm, [_subscriber(1), _subscriber(2, None)])) == 2


class TestNotice:
    def test_it_says_what_happened_where_what_to_do_and_who_says_so(self) -> None:
        finding = with_guidance(_finding()).to_dict()

        subject, body = notice(finding, _subscriber(1), 7.2)

        assert (
            subject == "[Mercury watch] Severe: Major flooding: St. Johns River near Cocoa, Florida"
        )
        assert 'About 7 km from "Home".' in body
        assert "WHAT TO DO\nNever walk or drive through flood water" in body
        assert "(National Weather Service flood safety guidance)" in body
        assert EMERGENCY_LINE in body and "911" in body
        assert 'p1@example.org is subscribed to this Mercury watch for "Home"' in body

    def test_an_alert_in_effect_at_the_place_says_so_and_names_its_area(self) -> None:
        alert = _finding(kind="nws_alert", area="Brevard, FL", action="Do not drive through water.")

        _, body = notice(alert.to_dict(), _subscriber(1), 0.0)

        assert 'This is in effect at "Home".' in body and "Where: Brevard, FL" in body

    def test_the_public_address_is_given_when_the_deployment_has_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MERCURY_PUBLIC_URL", "https://mercury.example.org/")

        assert "https://mercury.example.org/watch" in notice(_finding().to_dict())[1]


# --------------------------------------------------------------------------- #
# Delivery
# --------------------------------------------------------------------------- #


class TestSubscribers:
    def test_a_subscription_is_kept_and_updated_not_duplicated(self, store: WatchStore) -> None:
        first = store.add_subscriber(
            "a@example.org", place="Home", latitude=28.37, longitude=-80.87
        )
        again = store.add_subscriber(
            "a@example.org",
            place="Home",
            latitude=28.4,
            longitude=-80.9,
            radius_km=50,
            min_severity=2,
        )
        store.add_subscriber("a@example.org")  # and for everything

        assert first == again
        home, everything = store.subscribers()
        assert (home["latitude"], home["radius_km"], home["min_severity"]) == (28.4, 50.0, 2)
        assert everything["latitude"] is None and everything["place"] == ""
        assert store.remove_subscriber("a@example.org", "Home") == 1
        assert store.remove_subscriber("a@example.org") == 1 and store.subscribers() == []

    @pytest.mark.parametrize(
        "fields",
        [
            {"email": "not an address"},
            {"email": "a@example.org\nBcc: x@example.org"},
            {"latitude": 91.0, "longitude": 0.0},
            {"latitude": 28.0},
            {"radius_km": 0.0},
            {"min_severity": 4},
        ],
    )
    def test_what_cannot_be_a_subscription_is_refused(
        self, store: WatchStore, fields: dict[str, Any]
    ) -> None:
        with pytest.raises(ValueError):
            store.add_subscriber(**({"email": "a@example.org"} | fields))

    def test_the_store_is_readable_by_its_owner_only(self, store: WatchStore) -> None:
        """It holds people's addresses and where they live."""
        assert stat.S_IMODE(store.path.stat().st_mode) == 0o600


class TestDelivery:
    @staticmethod
    def _standing(
        store: WatchStore, finding: Finding | None = None, when: datetime = NOW
    ) -> Finding:
        made = with_guidance(finding or _finding())
        store.report(made, when)
        return made

    def test_a_finding_is_sent_once_to_each_person_it_concerns(self, store: WatchStore) -> None:
        store.add_subscriber("near@example.org", place="Home", latitude=28.37, longitude=-80.87)
        store.add_subscriber("far@example.org", place="Home", latitude=47.6, longitude=-122.3)
        self._standing(store)
        mail = _Mail()
        notifier = Notifier(store, mailer=mail, clock=lambda: NOW)

        assert notifier.sync() == {"sent": 1, "failed": 0}
        assert notifier.sync() == {"sent": 0, "failed": 0}

        ((to, subject, body),) = mail.sent
        assert to == "near@example.org" and subject.startswith("[Mercury watch] Severe:")
        assert "WHAT TO DO" in body

    def test_someone_who_subscribes_during_a_flood_is_told_of_it(self, store: WatchStore) -> None:
        """Was not possible: nothing was delivered at all, least of all what already stood."""
        self._standing(store)
        mail = _Mail()
        notifier = Notifier(store, mailer=mail, clock=lambda: NOW + timedelta(days=3))
        assert notifier.sync() == {"sent": 0, "failed": 0}

        store.add_subscriber("late@example.org", place="Home", latitude=28.37, longitude=-80.87)

        assert notifier.sync() == {"sent": 1, "failed": 0} and mail.sent[0][0] == "late@example.org"

    def test_a_finding_that_worsens_is_sent_again_and_one_that_ended_is_not(
        self, store: WatchStore
    ) -> None:
        store.add_subscriber("a@example.org")
        mail = _Mail()
        notifier = Notifier(store, mailer=mail, clock=lambda: NOW)
        minor = self._standing(store, _finding(Severity.NOTABLE))
        notifier.sync()
        self._standing(store, _finding(Severity.SEVERE), NOW + timedelta(hours=1))
        notifier.sync()
        store.close_absent("river_gauges", set(), NOW + timedelta(hours=2))
        store.add_subscriber("b@example.org")

        assert notifier.sync() == {"sent": 0, "failed": 0}
        assert [subject.split(":")[0] for _, subject, _ in mail.sent] == [
            "[Mercury watch] Notable",
            "[Mercury watch] Severe",
        ]
        assert minor.key == store.findings()[0]["key"]

    def test_a_failure_is_recorded_retried_and_costs_no_one_else_their_message(
        self, store: WatchStore
    ) -> None:
        store.add_subscriber("down@example.org")
        store.add_subscriber("up@example.org")
        made = self._standing(store)
        mail = _Mail(failing={"down@example.org"})
        notifier = Notifier(store, mailer=mail, clock=lambda: NOW)

        assert notifier.sync() == {"sent": 1, "failed": 1}
        assert [to for to, _, _ in mail.sent] == ["up@example.org"]
        record = store.delivery(made.key, 3, 1, "email")
        assert record["status"] == "failed" and "mailbox unavailable" in record["error"]

        mail.failing.clear()
        assert notifier.sync() == {"sent": 1, "failed": 0}
        assert store.delivery(made.key, 3, 1, "email")["attempts"] == 2
        assert store.delivery_counts() == {"sent": 2, "failed": 0}

    def test_a_line_break_in_a_source_text_does_not_make_the_finding_undeliverable(
        self, store: WatchStore
    ) -> None:
        """Was: the subject was refused as a mail header on all five attempts."""
        from email.message import EmailMessage

        store.add_subscriber("a@example.org")
        finding = _finding(kind="nws_alert", subject="urn:oid:1", action="Move.", action_source="N")
        broken = {**finding.to_dict(), "headline": "Flood Warning:\r\n Brevard,\n\tFL"}

        subject, body = notice(broken, store.subscribers()[0], 7.0)
        message = EmailMessage()
        message["Subject"] = subject  # raises on a line break

        assert subject == "[Mercury watch] Severe: Flood Warning: Brevard, FL"
        assert body.startswith("Flood Warning:\r\n Brevard")  # the body is the source's text

    def test_a_pass_with_nothing_to_send_reads_the_ledger_once(self, store: WatchStore) -> None:
        """Was: one connection per finding per subscriber, on every feed read.

        Measured at 2,000 subscribers and 20 standing findings: 3.01 s a pass
        before, 0.17 s after.
        """
        for index in range(40):
            store.add_subscriber(f"p{index}@example.org")
        for index in range(5):
            self._standing(store, _finding(subject=f"G{index}"))
        notifier = Notifier(store, mailer=_Mail(), clock=lambda: NOW)
        assert notifier.sync() == {"sent": 200, "failed": 0}
        assert len(store.standing_deliveries()) == 200

        connect, opened = store._connect, [0]

        def counting() -> Any:
            opened[0] += 1
            return connect()

        store._connect = counting  # type: ignore[method-assign]
        assert notifier.sync() == {"sent": 0, "failed": 0}
        assert opened[0] <= 3  # subscribers, the ledger, the findings: not 200

    def test_the_ledger_read_leaves_out_findings_that_have_ended(self, store: WatchStore) -> None:
        store.add_subscriber("a@example.org")
        made = self._standing(store)
        Notifier(store, mailer=_Mail(), clock=lambda: NOW).sync()
        assert store.standing_deliveries() == {
            (made.key, 3, 1, "email"): {
                "status": "sent", "attempts": 1, "at": NOW.isoformat(), "error": None,
            }
        }  # fmt: skip

        store.close_absent("river_gauges", set(), NOW + timedelta(hours=2))

        assert store.standing_deliveries() == {}
        assert store.delivery(made.key, 3, 1, "email")["status"] == "sent"  # still on record

    def test_one_mailbox_is_one_subscriber_however_it_is_capitalised(
        self, store: WatchStore
    ) -> None:
        """Was: 'A@Example.org' beside 'a@example.org' was two rows and two copies."""
        first = store.add_subscriber("Ana@Example.org", place="Home", latitude=1.0, longitude=2.0)
        again = store.add_subscriber("ana@example.org", place="Home", latitude=3.0, longitude=4.0)

        assert first == again
        (row,) = store.subscribers()
        assert (row["email"], row["latitude"]) == ("ana@example.org", 3.0)
        assert store.remove_subscriber("ANA@example.org") == 1

    def test_subscriptions_are_listed_ended_and_carried_by_address(self, store: WatchStore) -> None:
        home = store.add_subscriber("old@example.org", place="Home", latitude=1.0, longitude=2.0)
        work = store.add_subscriber("old@example.org", place="Work", latitude=5.0, longitude=6.0)
        kept = store.add_subscriber("new@example.org", place="Home", latitude=9.0, longitude=9.0)
        other = store.add_subscriber("other@example.org")

        assert [row["id"] for row in store.subscriptions_of("OLD@example.org")] == [home, work]
        assert store.remove_subscription("other@example.org", home) is False  # not theirs
        assert store.remove_subscription("old@example.org", 9999) is False
        assert len(store.subscribers()) == 4

        # "Home" exists at the new address already and keeps its settings there.
        assert store.move_subscriber("old@example.org", "new@example.org") == 1
        assert store.subscriptions_of("old@example.org") == []
        assert [(row["id"], row["place"], row["latitude"]) for row in
                store.subscriptions_of("new@example.org")] == [(work, "Work", 5.0), (kept, "Home", 9.0)]  # fmt: skip
        assert store.remove_subscription("new@example.org", work) is True
        assert [row["id"] for row in store.subscribers()] == [kept, other]
        with pytest.raises(ValueError, match="not an email address"):
            store.move_subscriber("new@example.org", "nobody")

    def test_changing_only_the_capitals_of_an_address_keeps_its_subscriptions(
        self, store: WatchStore
    ) -> None:
        store.add_subscriber("ana@example.org", place="Home", latitude=1.0, longitude=2.0)

        assert store.move_subscriber("ana@example.org", "Ana@Example.org") == 1
        assert [row["email"] for row in store.subscribers()] == ["Ana@Example.org"]

    def test_places_are_distinct_to_about_a_kilometre(self, store: WatchStore) -> None:
        one = store.add_subscriber(
            "a@example.org", place="Home", latitude=28.371, longitude=-80.874
        )
        two = store.add_subscriber("b@example.org", place="Work", latitude=28.37, longitude=-80.87)
        three = store.add_subscriber("c@example.org", place="Far", latitude=47.6, longitude=-122.3)
        store.add_subscriber("d@example.org")

        assert store.places() == {(28.37, -80.87): [one, two], (47.6, -122.3): [three]}

    def test_someone_who_subscribes_is_not_sent_every_earthquake_the_store_remembers(
        self, store: WatchStore
    ) -> None:
        """Was: a new subscriber was mailed each past earthquake, months old, as news."""
        for days, subject in ((90, "old"), (3, "recent"), (0, "today")):
            when = NOW - timedelta(days=days, hours=1)
            quake = _finding(kind="earthquake", subject=subject, latitude=None, longitude=None)
            store.report(Finding(**{**quake.__dict__, "ongoing": False}), when)
        self._standing(store)  # and a flood under way
        store.add_subscriber("new@example.org")
        mail = _Mail()

        assert Notifier(store, mailer=mail, clock=lambda: NOW).sync() == {"sent": 2, "failed": 0}
        assert len(mail.sent) == 2  # the flood, and the earthquake an hour ago

    def test_a_long_pass_says_it_is_alive_after_every_message(self, store: WatchStore) -> None:
        """The watch renews its lease from this, so a slow mail server is not taken for a dead watch."""
        for name in ("a", "b", "c"):
            store.add_subscriber(f"{name}@example.org")
        self._standing(store)
        beats: list[int] = []
        mail = _Mail(failing={"b@example.org"})

        Notifier(store, mailer=mail, clock=lambda: NOW).sync(lambda: beats.append(len(mail.sent)))

        assert beats == [1, 1, 2]  # after each attempt, sent or failed

    def test_a_hazard_that_recurs_is_told_again(self, store: WatchStore) -> None:
        """Was: whoever was told of the first flood at a gauge was never told of the next.

        The ended occurrence was archived under ``key@began`` and its
        deliveries were left under the bare key, where they read as
        deliveries of the new occurrence.
        """
        store.add_subscriber("a@example.org")
        mail = _Mail()
        notifier = Notifier(store, mailer=mail, post=lambda url, body: None, clock=lambda: NOW)
        first = self._standing(store)
        assert notifier.sync() == {"sent": 1, "failed": 0}
        store.close_absent("river_gauges", set(), NOW + timedelta(days=2))

        month = NOW + timedelta(days=30)
        assert store.report(with_guidance(_finding()), month) is True  # it floods again
        later = Notifier(store, mailer=mail, clock=lambda: month)

        assert later.sync() == {"sent": 1, "failed": 0}
        assert later.sync() == {"sent": 0, "failed": 0}  # once per occurrence
        assert len(mail.sent) == 2
        # Each occurrence keeps its own record.
        archived = f"{first.key}@{NOW.isoformat()}"
        assert store.delivery(archived, 3, 1, "email")["at"] == NOW.isoformat()
        assert store.delivery(first.key, 3, 1, "email")["at"] == month.isoformat()

    def test_a_store_whose_deliveries_stayed_under_the_bare_key_tells_the_recurrence_too(
        self, store: WatchStore
    ) -> None:
        """A store written before the fix: the old record must not pass for a new delivery."""
        import sqlite3
        from contextlib import closing

        store.add_subscriber("a@example.org")
        made = self._standing(store, when=NOW + timedelta(days=30))  # the occurrence that stands
        with closing(sqlite3.connect(store.path)) as connection, connection:
            connection.execute(  # what the first occurrence left behind: failed five times
                "INSERT INTO deliveries VALUES (?, 3, 1, 'email', 'failed', 5, ?, 'old')",
                (made.key, NOW.isoformat()),
            )
        mail = _Mail()
        notifier = Notifier(store, mailer=mail, clock=lambda: NOW + timedelta(days=30))

        assert store.standing_deliveries() == {}
        assert notifier.sync() == {"sent": 1, "failed": 0}
        record = store.delivery(made.key, 3, 1, "email")
        assert (record["status"], record["attempts"]) == ("sent", 1)  # this occurrence's first
        assert notifier.sync() == {"sent": 0, "failed": 0}

    def test_a_delivery_that_keeps_failing_is_left_failed_not_retried_for_ever(
        self, store: WatchStore
    ) -> None:
        store.add_subscriber("down@example.org")
        self._standing(store)
        notifier = Notifier(store, mailer=_Mail(failing={"down@example.org"}), clock=lambda: NOW)

        attempts = [notifier.sync()["failed"] for _ in range(MAX_ATTEMPTS + 2)]

        assert attempts == [1] * MAX_ATTEMPTS + [0, 0]
        assert store.delivery_counts() == {"sent": 0, "failed": 1}

    def test_without_a_mail_server_nothing_is_claimed_sent(
        self, store: WatchStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Was the risk: the default mailer files messages in a folder and reports success."""
        monkeypatch.delenv("MERCURY_SMTP_HOST", raising=False)
        monkeypatch.delenv("MERCURY_WATCH_WEBHOOK_URL", raising=False)
        store.add_subscriber("a@example.org")
        self._standing(store)
        notifier = Notifier(store)

        assert notifier.channels() == {"email": False, "webhook": False}
        assert notifier.sync() == {"sent": 0, "failed": 0}
        assert store.delivery_counts() == {"sent": 0, "failed": 0}

    def test_the_webhook_gets_significant_findings_once_with_a_topic_when_set(
        self, store: WatchStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MERCURY_WATCH_WEBHOOK_URL", "https://push.example.org/")
        monkeypatch.setenv("MERCURY_WATCH_WEBHOOK_TOPIC", "mercury")
        self._standing(store)
        self._standing(store, _finding(Severity.NOTABLE, subject="MINOR1"))
        posted: list[tuple[str, dict[str, Any]]] = []
        notifier = Notifier(
            store,
            mailer=_Mail(),
            post=lambda url, body: posted.append((url, body)),
            clock=lambda: NOW,
        )

        assert notifier.sync() == {"sent": 1, "failed": 0}
        assert notifier.sync() == {"sent": 0, "failed": 0}

        ((url, body),) = posted
        assert url == "https://push.example.org/" and body["topic"] == "mercury"
        assert body["title"].startswith("[Mercury watch] Severe:") and body["priority"] == 5
        assert "WHAT TO DO" in body["message"] and body["text"].startswith("*[Mercury watch]")

    def test_the_webhook_leaves_through_the_egress_gate(self) -> None:
        from omni_mercury_engine.security.safe_http import SafeHTTPClient, UnsafeURLError

        with pytest.raises(UnsafeURLError):
            SafeHTTPClient.post("http://169.254.169.254/x", json_body={}, user_configured=True)

    def test_a_delivery_fault_does_not_stop_the_watch(self, store: WatchStore) -> None:
        class _Broken(Notifier):
            def sync(self, heartbeat: Any = None) -> dict[str, int]:
                raise RuntimeError("ledger unavailable")

        watcher = Watcher(store, [_Fixed([_finding()])], notifier=_Broken(store), clock=lambda: NOW)

        (status,) = watcher.run_once()

        assert status.state == "ok" and len(store.findings()) == 1


class TestConsoleAlert:
    def test_a_new_finding_reaches_open_consoles_worded_with_what_to_do(
        self, store: WatchStore
    ) -> None:
        from omni_mercury_engine.narrative.alert_bus import get_alert_bus

        published: list[dict[str, Any]] = []
        bus = get_alert_bus()
        original = bus.publish

        def record(entry: dict[str, Any]) -> int:
            published.append(entry)
            return 0

        bus.publish = record  # type: ignore[method-assign]
        try:
            Watcher(store, [_Fixed([_finding()])], clock=lambda: NOW).run_once()
            Watcher(store, [_Fixed([_finding()])], clock=lambda: NOW).run_once()  # seen again
        finally:
            bus.publish = original  # type: ignore[method-assign]

        (entry,) = published
        assert entry["detection"]["anomaly_detected"] and entry["principal"] is None
        alert = to_alert(entry)
        assert alert["initiative_type"] == "watch_severe" and alert["severity"] == 1.0
        assert alert["message"].startswith(
            "Major flooding: St. Johns River near Cocoa, Florida. Never"
        )
        assert alert["event"]["action_source"].startswith("National Weather Service")


# --------------------------------------------------------------------------- #
# The Service's own warnings
# --------------------------------------------------------------------------- #


def _alert(ident: str, event: str, severity: str, **fields: Any) -> dict[str, Any]:
    return {
        "properties": {
            "id": f"urn:oid:2.49.0.1.840.0.{ident}.001.1", "event": event, "severity": severity,
            "urgency": "Immediate", "certainty": "Observed", "areaDesc": "Brevard, FL",
            "headline": f"{event} issued November 16 at 7:31PM EST by NWS Melbourne FL",
            "instruction": "Turn around, don't drown\nwhen encountering flooded roads.",
            "senderName": "NWS Melbourne FL", "sent": "2026-11-17T00:31:00+00:00",
            "ends": "2026-11-19T01:00:00+00:00", "messageType": "Alert", "status": "Actual",
            **fields,
        }
    }  # fmt: skip


class TestWarnings:
    @staticmethod
    def _feed(
        store: WatchStore, monkeypatch: pytest.MonkeyPatch, national: list[Any], local: list[Any]
    ) -> tuple[WarningsFeed, list[dict[str, str]]]:
        feed, asked = WarningsFeed(), []
        feed.bind(store)

        def answer(params: dict[str, str]) -> list[dict[str, Any]]:
            asked.append(params)
            return [a["properties"] for a in (local if "point" in params else national)]

        monkeypatch.setattr(feed, "_alerts", answer)
        return feed, asked

    def test_nationwide_only_extreme_alerts_are_findings_and_the_rest_are_counted(
        self, store: WatchStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        national = [
            _alert("aa", "Tornado Warning", "Extreme"),
            _alert("bb", "Storm Warning", "Severe"),
            _alert("cc", "Frost Advisory", "Minor"),
            _alert("dd", "Tornado Warning", "Extreme", messageType="Cancel"),
        ]
        feed, asked = self._feed(store, monkeypatch, national, [])

        judgment = feed.judge(feed.read(), NOW)

        assert asked == [{"status": "actual"}]
        (finding,) = judgment.findings
        assert (
            finding.headline == "Tornado Warning: Brevard, FL"
            and finding.severity is Severity.SEVERE
        )
        assert finding.action == "Turn around, don't drown when encountering flooded roads."
        assert finding.action_source == "NWS Melbourne FL" and finding.place_ids == ()
        assert (
            "relayed as issued" in finding.detail
            and "extreme, immediate, observed" in finding.detail
        )
        assert judgment.summary == "3 alerts in effect nationwide (1 extreme, 1 severe)"

    def test_every_alert_in_effect_at_a_subscribers_place_is_a_finding_for_them(
        self, store: WatchStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        one = store.add_subscriber(
            "a@example.org", place="Home", latitude=28.371, longitude=-80.874
        )
        two = store.add_subscriber("b@example.org", place="Work", latitude=28.37, longitude=-80.87)
        store.add_subscriber("c@example.org")  # no place: no lookup
        local = [_alert("ee", "Flood Warning", "Severe"), _alert("ff", "Flood Watch", "Moderate")]
        feed, asked = self._feed(store, monkeypatch, [], local)

        judgment = feed.judge(feed.read(), NOW)

        assert asked == [{"status": "actual"}, {"status": "actual", "point": "28.37,-80.87"}]
        warning, watch = judgment.findings
        assert warning.place_ids == (one, two) and warning.severity is Severity.SIGNIFICANT
        assert watch.severity is Severity.NOTABLE
        assert warning.basis == "an NWS alert in effect at a subscribed place"
        assert judgment.summary.endswith(
            "1 subscribed place checked; 2 in effect at subscribed places"
        )

    def test_a_place_the_service_will_not_answer_for_costs_no_one_else_their_warnings(
        self, store: WatchStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Was: one failed lookup raised, and the whole feed, Extreme alerts included, was down."""
        home = store.add_subscriber("a@example.org", place="Home", latitude=28.37, longitude=-80.87)
        store.add_subscriber("b@example.org", place="Abroad", latitude=35.68, longitude=139.69)
        feed = WarningsFeed()
        feed.bind(store)

        def answer(params: dict[str, str]) -> list[dict[str, Any]]:
            if params.get("point") == "35.68,139.69":
                raise ValueError("HTTP 400: point is outside the Service's area")
            source = [_alert("ee", "Flood Warning", "Severe")] if "point" in params else [
                _alert("aa", "Tornado Warning", "Extreme")
            ]  # fmt: skip
            return [alert["properties"] for alert in source]

        monkeypatch.setattr(feed, "_alerts", answer)
        judgment = feed.judge(feed.read(), NOW)

        assert sorted(finding.place_ids for finding in judgment.findings) == [(), (home,)]
        assert "1 subscribed place checked; 1 place could not be checked this time" in (
            judgment.summary
        )

    def test_more_places_than_one_read_looks_up_are_said_and_the_rest_are_still_checked(
        self, store: WatchStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Was: the 251st place raised, and every subscriber lost every warning."""
        monkeypatch.setattr(WarningsFeed, "MAX_PLACES", 2)
        ids = [
            store.add_subscriber(f"p{n}@example.org", place="P", latitude=30.0 + n, longitude=-90.0)
            for n in range(3)
        ]
        feed, asked = self._feed(
            store, monkeypatch, [_alert("aa", "Tornado Warning", "Extreme")],
            [_alert("ee", "Flood Warning", "Severe")],
        )  # fmt: skip

        judgment = feed.judge(feed.read(), NOW)

        assert [params.get("point") for params in asked] == [None, "30.0,-90.0", "31.0,-90.0"]
        local = next(finding for finding in judgment.findings if finding.place_ids)
        assert local.place_ids == tuple(ids[:2])  # the longest-subscribed places first
        assert judgment.summary.endswith(
            "2 subscribed places checked; 1 place NOT checked: more are subscribed than the "
            "2 one read looks up; 1 in effect at subscribed places"
        )

    def test_the_point_lookup_failing_everywhere_is_down_not_no_alert_in_effect(
        self, store: WatchStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store.add_subscriber("a@example.org", place="Home", latitude=28.37, longitude=-80.87)
        feed = WarningsFeed()
        feed.bind(store)

        def answer(params: dict[str, str]) -> list[dict[str, Any]]:
            if "point" in params:
                raise ConnectionError("simulated: 503")
            return []

        monkeypatch.setattr(feed, "_alerts", answer)

        with pytest.raises(ConnectionError, match="all 1 subscribed places"):
            feed.read()

    def test_a_quiet_day_is_said(self, store: WatchStore, monkeypatch: pytest.MonkeyPatch) -> None:
        feed, _ = self._feed(store, monkeypatch, [], [])

        assert feed.judge(feed.read(), NOW).summary == "no alert in effect"

    def test_a_local_alert_is_delivered_to_that_subscriber_with_the_services_instruction(
        self, store: WatchStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store.add_subscriber("here@example.org", place="Home", latitude=28.37, longitude=-80.87)
        store.add_subscriber("there@example.org", place="Home", latitude=47.6, longitude=-122.3)
        feed = WarningsFeed()
        mail = _Mail()
        watcher = Watcher(store, [feed], notifier=Notifier(store, mailer=mail), clock=lambda: NOW)

        def answer(params: dict[str, str]) -> list[dict[str, Any]]:
            hit = params.get("point") == "28.37,-80.87"
            return [_alert("ee", "Flood Warning", "Severe")["properties"]] if hit else []

        monkeypatch.setattr(feed, "_alerts", answer)

        watcher.run_once()

        ((to, subject, body),) = mail.sent
        assert to == "here@example.org" and subject.endswith("Flood Warning: Brevard, FL")
        assert "WHAT TO DO\nTurn around, don't drown when encountering flooded roads." in body
        assert "(NWS Melbourne FL)" in body and 'This is in effect at "Home".' in body
        served = snapshot_of(store)
        assert "place_ids" not in json.dumps(served)
        # Served to everyone, it would say where a subscriber is.
        assert served["findings"] == [] and len(store.findings()) == 1

    def test_the_services_record_of_the_alert_confirms_or_withdraws_it(
        self, store: WatchStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from omni_mercury_engine.security.safe_http import SafeHTTPClient

        feed = WarningsFeed()
        held: Any = _alert("ee", "Flood Warning", "Severe")
        asked: list[str] = []

        def get_json(url: str, **_: Any) -> Any:
            asked.append(url)
            return held

        monkeypatch.setattr(SafeHTTPClient, "get_json", staticmethod(get_json))
        stored = {
            "reference": "urn:oid:2.49.0.1.840.0.ee.001.1",
            "headline": "Flood Warning: Brevard, FL",
        }

        confirmed = feed.review(stored, NOW)
        held = _alert("ee", "Flood Warning", "Severe", status="Test")
        withdrawn = feed.review(stored, NOW)

        assert asked[0] == "https://api.weather.gov/alerts/urn:oid:2.49.0.1.840.0.ee.001.1"
        assert confirmed is not None and (confirmed.verdict, confirmed.final) == ("confirmed", True)
        assert "whether the hazard occurred is the Service's to verify" in confirmed.note
        assert withdrawn is not None and withdrawn.verdict == "withdrawn"
        assert feed.review({"reference": "urn:oid:../../x", "headline": "x"}, NOW) is None


def snapshot_of(store: WatchStore) -> dict[str, Any]:
    import os

    previous = os.environ.get("MERCURY_WATCH_DIR")
    os.environ["MERCURY_WATCH_DIR"] = str(store.directory)
    try:
        return snapshot()
    finally:
        if previous is None:
            del os.environ["MERCURY_WATCH_DIR"]
        else:
            os.environ["MERCURY_WATCH_DIR"] = previous


# --------------------------------------------------------------------------- #
# Surfaces
# --------------------------------------------------------------------------- #


class TestSurfaces:
    def test_the_route_says_how_many_are_told_and_never_who_or_where(
        self, store: WatchStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MERCURY_SMTP_HOST", raising=False)
        store.add_subscriber("private@example.org", place="Home", latitude=28.37, longitude=-80.87)

        data = snapshot_of(store)

        assert data["telling"] == {
            "subscribers": 1, "email": False, "webhook": False, "sent": 0, "failed": 0,
        }  # fmt: skip
        assert "private@example.org" not in json.dumps(data) and "28.37" not in json.dumps(data)

    def test_the_command_says_when_a_place_will_not_be_checked_for_alerts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(WarningsFeed, "MAX_PLACES", 1)
        run, where = CliRunner(), ["--state-dir", str(tmp_path)]

        def add(email: str, lat: str) -> str:
            result = run.invoke(
                main, ["alerts", "add", "--email", email, "--lat", lat, "--lon", "-80.87", *where]
            )
            assert result.exit_code == 0, result.output
            return str(result.output)

        assert "not checked" not in add("a@example.org", "28.37")
        assert "1 more distinct place(s) are subscribed than the 1" in add("b@example.org", "40.0")

    def test_the_command_subscribes_lists_and_unsubscribes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MERCURY_SMTP_HOST", raising=False)
        run = CliRunner()
        where = ["--state-dir", str(tmp_path)]

        added = run.invoke(
            main,
            ["alerts", "add", "--email", "a@example.org", "--place", "Home", "--lat", "28.37",
             "--lon", "-80.87", "--radius-km", "60", *where],
        )  # fmt: skip
        listed = run.invoke(main, ["alerts", "list", *where])
        refused = run.invoke(main, ["alerts", "add", "--email", "nope", *where])
        unreachable = run.invoke(main, ["alerts", "test", "--email", "a@example.org", *where])
        removed = run.invoke(main, ["alerts", "remove", "--email", "a@example.org", *where])

        assert added.exit_code == 0, added.output
        assert "will be told of findings within 60 km of Home, notable and above" in added.output
        assert (
            "will not be reached until one is" in added.output
        )  # no mail server: said, not hidden
        assert (
            "a@example.org: within 60 km of Home (28.37, -80.87); notable and above"
            in listed.output
        )
        assert "NOT CONFIGURED" in listed.output
        assert refused.exit_code == 2 and "not an email address" in refused.output
        assert unreachable.exit_code == 1 and "no email can be sent" in unreachable.output
        assert removed.exit_code == 0 and "Ended 1 subscription" in removed.output

    def test_the_page_and_the_feed_list_carry_it(self) -> None:
        from omni_mercury_engine.api import frontend

        page = (frontend._PAGES_DIR / "watch.html").read_text(encoding="utf-8")
        script = (frontend._STATIC_DIR / "watch.js").read_text(encoding="utf-8")

        assert 'id="telling"' in page and "finding.action" in script and "innerHTML" not in script
        assert "warnings" in FEEDS and len(FEEDS) == 7
