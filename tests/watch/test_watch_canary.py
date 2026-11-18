# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A test message is proven only by what comes back from the other end.

The watch recorded a delivery as sent when a mail server or a text provider
took it, and nothing told a message that was taken from one that arrived.
Here a canary goes out through the deployment's own mailer to a listening
SMTP server, and is then looked for the way the deployment looks: over IMAP,
with TLS, in the mailbox it was addressed to. Where the server drops it, the
watch says it is unproven.

What this cannot show is a real provider and a real inbox: that is what
``mercury-agent alerts canary --wait`` is, run by the operator.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest
from click.testing import CliRunner

from omni_mercury_engine.cli import main
from omni_mercury_engine.watch import (
    Watcher,
    WatchStore,
    canary as canary_module,
)
from omni_mercury_engine.watch.canary import (
    HEADER,
    Canary,
    ImapMailbox,
    proof_of_delivery,
    targets,
)
from omni_mercury_engine.watch.notify import Notifier
from omni_mercury_engine.watch.sms import SmsRefusedError, TwilioSms

from .imap_server import ImapServer
from .smtp_server import OPENSSL, SmtpServer, mint_certificate

if TYPE_CHECKING:
    from collections.abc import Iterator
    from email.message import EmailMessage
    from pathlib import Path

pytestmark = pytest.mark.skipif(
    OPENSSL is None, reason="openssl CLI not on PATH: the servers' certificate is minted with it"
)

NOW = datetime(2026, 11, 18, 9, 0, tzinfo=UTC)
OPERATOR = "operator@example.org"
NUMBER = "+14155550123"
SID = "AC" + "0" * 32
MESSAGE_SID = "SM" + "a" * 32
#: Not credentials for anything: what the stand-in servers are told to accept.
TOKEN = "1" * 32


class Clock:
    """A clock a test moves by hand."""

    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> datetime:
        return self.now

    def forward(self, **by: float) -> None:
        self.now += timedelta(**by)


@pytest.fixture()
def clock() -> Clock:
    return Clock()


@pytest.fixture()
def certificate(tmp_path: Path) -> tuple[Path, Path]:
    return mint_certificate(tmp_path)


@pytest.fixture()
def smtp(monkeypatch: pytest.MonkeyPatch, certificate: tuple[Path, Path]) -> Iterator[SmtpServer]:
    with SmtpServer(*certificate) as server:
        monkeypatch.setenv("MERCURY_SMTP_HOST", "localhost")
        monkeypatch.setenv("MERCURY_SMTP_PORT", str(server.port))
        monkeypatch.setenv("MERCURY_SMTP_USERNAME", server.credentials[0])
        monkeypatch.setenv("MERCURY_SMTP_PASSWORD", server.credentials[1])
        monkeypatch.setenv("MERCURY_SMTP_FROM", "Mercury watch <watch@example.org>")
        monkeypatch.setenv("MERCURY_SMTP_STARTTLS", "true")
        monkeypatch.setenv("SSL_CERT_FILE", str(certificate[0]))  # the test's own authority
        yield server


def _inbox(smtp: SmtpServer, address: str = OPERATOR) -> list[EmailMessage]:
    """What the mail server accepted for ``address``: that address's mailbox."""
    return [received.message for received in smtp.received if address in received.recipients]


@pytest.fixture()
def imap(
    monkeypatch: pytest.MonkeyPatch, certificate: tuple[Path, Path], smtp: SmtpServer
) -> Iterator[ImapServer]:
    with ImapServer(*certificate, messages=lambda: _inbox(smtp)) as server:
        monkeypatch.setenv("MERCURY_WATCH_CANARY_IMAP_HOST", "localhost")
        monkeypatch.setenv("MERCURY_WATCH_CANARY_IMAP_PORT", str(server.port))
        monkeypatch.setenv("MERCURY_WATCH_CANARY_IMAP_USERNAME", server.credentials[0])
        monkeypatch.setenv("MERCURY_WATCH_CANARY_IMAP_PASSWORD", server.credentials[1])
        yield server


@pytest.fixture()
def store(tmp_path: Path) -> WatchStore:
    return WatchStore(tmp_path / "watch")


class Texts:
    """A text provider: takes messages, and says what became of each when asked."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []
        self.became = "queued"
        self.asked: list[str] = []

    def send(self, *, to: str, body: str) -> str:
        self.sent.append((to, body))
        return MESSAGE_SID

    def status(self, message_sid: str) -> str:
        self.asked.append(message_sid)
        return self.became


# --------------------------------------------------------------------------- #
# Email: out by SMTP, found again by IMAP
# --------------------------------------------------------------------------- #


class TestEmail:
    def test_a_canary_goes_the_way_an_alert_does_and_is_found_in_the_mailbox(
        self, store: WatchStore, smtp: SmtpServer, imap: ImapServer, clock: Clock
    ) -> None:
        canary = Canary(store, Notifier(store, clock=clock), clock=clock)

        (sent,) = canary.send(email=OPERATOR)

        assert sent["error"] is None
        (received,) = smtp.received
        assert received.encrypted and received.authenticated_as == smtp.credentials[0]
        assert received.recipients == [OPERATOR]
        assert received.message[HEADER] == sent["id"]
        assert received.message["Auto-Submitted"] == "auto-generated"
        assert "Nothing has happened" in received.message.get_content()
        # Taken by the server is not arrived: nothing has said so yet.
        assert proof_of_delivery(store, clock())["email"]["state"] == "awaiting"

        assert canary.settle() == {"proven": 1, "failed": 0}

        assert imap.searches == [["HEADER", HEADER, sent["id"]]]
        shown = proof_of_delivery(store, clock())["email"]
        assert (shown["state"], shown["by"]) == ("proven", "mailbox")
        assert shown["proven_at"] == shown["last_proven_at"] == NOW.isoformat()

    def test_one_the_server_took_and_did_not_deliver_is_unproven_and_said_so(
        self,
        store: WatchStore,
        smtp: SmtpServer,
        imap: ImapServer,
        clock: Clock,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Was: "sent" in the ledger, and that was the end of what the watch knew."""
        monkeypatch.setenv("MERCURY_WATCH_CANARY_EMAIL", OPERATOR)
        canary = Canary(store, Notifier(store, clock=clock), clock=clock)
        canary.run()  # none has been sent: one is due
        assert len(smtp.received) == 1
        smtp.received.clear()  # the server took it and lost it

        clock.forward(minutes=5)
        canary.run()
        assert proof_of_delivery(store, clock())["email"]["state"] == "awaiting"

        clock.forward(hours=1)
        with caplog.at_level(logging.ERROR, logger=canary_module.__name__):
            canary.run()
            clock.forward(minutes=10)
            canary.run()

        assert proof_of_delivery(store, clock())["email"]["state"] == "unproven"
        said = [record.getMessage() for record in caplog.records]
        assert len(said) == 1 and "email canary" in said[0] and "unproven" in said[0]
        assert smtp.received == []  # and no second one before its time

    def test_it_is_looked_for_where_it_was_sent_and_nowhere_else(
        self, store: WatchStore, smtp: SmtpServer, imap: ImapServer, clock: Clock
    ) -> None:
        """A canary to another address is not in the operator's mailbox."""
        canary = Canary(store, Notifier(store, clock=clock), clock=clock)
        canary.send(email="someone.else@example.org")

        assert canary.settle() == {"proven": 0, "failed": 0}
        assert proof_of_delivery(store, clock())["email"]["state"] == "awaiting"

    def test_a_mailbox_that_cannot_be_read_proves_nothing_either_way(
        self,
        store: WatchStore,
        smtp: SmtpServer,
        imap: ImapServer,
        clock: Clock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MERCURY_WATCH_CANARY_IMAP_PASSWORD", "wrong")
        canary = Canary(store, Notifier(store, clock=clock), clock=clock)
        canary.send(email=OPERATOR)

        assert canary.settle() == {"proven": 0, "failed": 0}
        assert proof_of_delivery(store, clock())["email"]["state"] == "awaiting"

    def test_a_refused_recipient_is_failing_and_the_reason_stays_off_the_public_page(
        self, store: WatchStore, smtp: SmtpServer, clock: Clock
    ) -> None:
        smtp.refuse.add(OPERATOR)
        canary = Canary(store, Notifier(store, clock=clock), clock=clock)

        (sent,) = canary.send(email=OPERATOR)

        assert sent["error"] and "SMTPRecipientsRefused" in sent["error"]
        public = proof_of_delivery(store, clock())["email"]
        assert public["state"] == "failing" and "error" not in public and "id" not in public
        assert OPERATOR in proof_of_delivery(store, clock(), detail=True)["email"]["error"]

    def test_the_mailbox_asks_only_for_an_id_it_issued(
        self, certificate: tuple[Path, Path]
    ) -> None:
        mailbox = ImapMailbox(host="localhost", username="u", password=TOKEN, port=1)

        with pytest.raises(ValueError, match="not a canary's id"):
            mailbox.holds('x" OR ALL')


# --------------------------------------------------------------------------- #
# Texts: the carrier's own word
# --------------------------------------------------------------------------- #


class TestText:
    def test_delivered_to_the_handset_is_proof_and_queued_is_not(
        self, store: WatchStore, clock: Clock
    ) -> None:
        texts = Texts()
        canary = Canary(store, Notifier(store, sms=texts, clock=clock), clock=clock)

        (sent,) = canary.send(phone=NUMBER)

        ((to, body),) = texts.sent
        assert to == NUMBER and sent["code"] in body and "Nothing has happened" in body
        assert canary.settle() == {"proven": 0, "failed": 0}  # queued: taken, not arrived
        assert proof_of_delivery(store, clock())["sms"]["state"] == "awaiting"

        texts.became = "delivered"
        clock.forward(minutes=2)
        assert canary.settle() == {"proven": 1, "failed": 0}
        assert texts.asked == [MESSAGE_SID, MESSAGE_SID]
        assert proof_of_delivery(store, clock())["sms"]["by"] == "carrier"

    def test_undelivered_is_the_carriers_answer_too(self, store: WatchStore, clock: Clock) -> None:
        texts = Texts()
        texts.became = "undelivered"
        canary = Canary(store, Notifier(store, sms=texts, clock=clock), clock=clock)
        canary.send(phone=NUMBER)

        assert canary.settle() == {"proven": 0, "failed": 1}
        shown = proof_of_delivery(store, clock(), detail=True)["sms"]
        assert shown["state"] == "failing" and "undelivered" in shown["error"]
        clock.forward(minutes=5)
        canary.settle()
        assert len(texts.asked) == 1  # answered: it is not asked about again

    def test_an_open_canary_is_asked_about_less_often_as_it_ages(
        self, store: WatchStore, clock: Clock
    ) -> None:
        texts = Texts()
        canary = Canary(store, Notifier(store, sms=texts, clock=clock), clock=clock)
        canary.send(phone=NUMBER)

        for _ in range(60):  # a tick every 30 seconds for half an hour
            canary.settle()
            clock.forward(seconds=30)
        early = len(texts.asked)
        clock.forward(hours=10)
        for _ in range(120):  # and for an hour, ten hours on
            canary.settle()
            clock.forward(seconds=30)

        assert 10 <= early <= 30
        assert len(texts.asked) - early <= 2
        clock.forward(hours=14)
        asked = len(texts.asked)
        canary.settle()
        assert len(texts.asked) == asked  # a day old: no longer asked about

    def test_twilio_gives_the_message_its_sid_and_answers_for_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from omni_mercury_engine.security.safe_http import SafeHTTPClient

        calls: list[dict[str, Any]] = []

        def post_form(url: str, **kwargs: Any) -> Any:
            return {"sid": MESSAGE_SID, "status": "queued"}

        def get_json(url: str, **kwargs: Any) -> Any:
            calls.append({"url": url, **kwargs})
            return {"sid": MESSAGE_SID, "status": "delivered"}

        monkeypatch.setattr(SafeHTTPClient, "post_form", staticmethod(post_form))
        monkeypatch.setattr(SafeHTTPClient, "get_json", staticmethod(get_json))
        sender = TwilioSms(account_sid=SID, auth_token=TOKEN, from_number="+14155550100")

        assert sender.send(to=NUMBER, body="Mercury watch test.") == MESSAGE_SID
        assert sender.status(MESSAGE_SID) == "delivered"
        (call,) = calls
        assert call["url"] == (
            f"https://api.twilio.com/2010-04-01/Accounts/{SID}/Messages/{MESSAGE_SID}.json"
        )
        assert call["headers"]["Authorization"].startswith("Basic ")
        with pytest.raises(ValueError, match="message SID"):
            sender.status("../Accounts")  # nothing but a SID is put in the address

    def test_a_number_that_is_not_a_number_is_refused_before_anything_is_recorded(
        self, store: WatchStore, clock: Clock
    ) -> None:
        canary = Canary(store, Notifier(store, sms=Texts(), clock=clock), clock=clock)

        with pytest.raises(ValueError, match=r"\+14155550123"):
            canary.send(phone="4155550123")
        assert store.latest_canaries() == {}

    def test_a_refusal_by_the_provider_is_recorded_as_one(
        self, store: WatchStore, clock: Clock
    ) -> None:
        class Refusing(Texts):
            def send(self, *, to: str, body: str) -> str:
                raise SmsRefusedError("Twilio error 21610: unsubscribed recipient", 21610)

        canary = Canary(store, Notifier(store, sms=Refusing(), clock=clock), clock=clock)

        (sent,) = canary.send(phone=NUMBER)

        assert "21610" in sent["error"]
        assert proof_of_delivery(store, clock())["sms"]["state"] == "failing"


# --------------------------------------------------------------------------- #
# Acknowledged by a person: the link and the code
# --------------------------------------------------------------------------- #


class TestAcknowledged:
    def _sent(
        self, store: WatchStore, clock: Clock, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        monkeypatch.setenv("MERCURY_WATCH_WEBHOOK_URL", "https://ntfy.example.org/")
        monkeypatch.setenv("MERCURY_WATCH_WEBHOOK_TOPIC", "mercury")
        monkeypatch.setenv("MERCURY_PUBLIC_BASE_URL", "https://mercury.example.org/")
        posted: list[dict[str, Any]] = []
        notifier = Notifier(
            store, clock=clock, post=lambda url, payload: posted.append({"url": url, **payload})
        )
        (sent,) = Canary(store, notifier, clock=clock).send(webhook=True)
        return sent, posted

    def test_a_webhook_that_takes_it_has_accepted_it_and_the_link_proves_it(
        self, store: WatchStore, clock: Clock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("fastapi")
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from omni_mercury_engine.api.routes import watch as watch_routes

        monkeypatch.setenv("MERCURY_WATCH_DIR", str(store.directory))
        sent, posted = self._sent(store, clock, monkeypatch)
        (payload,) = posted
        assert payload["url"] == "https://ntfy.example.org/" and payload["topic"] == "mercury"
        link = re.search(
            r"https://mercury\.example\.org(/api/v1/watch/ack/[\w-]+)", payload["message"]
        )
        assert link is not None
        assert proof_of_delivery(store, clock())["webhook"]["state"] == "awaiting"
        app = FastAPI()
        app.include_router(watch_routes.router)
        client = TestClient(app)

        # Opening it is what a mail scanner or a link preview does: it proves nothing.
        page = client.get(link.group(1))
        assert page.status_code == 200 and 'method="post"' in page.text
        assert page.headers["cache-control"] == "no-store"
        assert proof_of_delivery(store, clock())["webhook"]["state"] == "awaiting"

        assert client.post(link.group(1)).status_code == 200
        shown = client.get("/api/v1/watch").json()["proof_of_delivery"]["webhook"]
        assert (shown["state"], shown["by"]) == ("proven", "acknowledged")
        assert client.post(link.group(1)).status_code == 200  # pressed twice: still proven
        assert client.post("/api/v1/watch/ack/" + "A" * 32).status_code == 404
        assert client.post("/api/v1/watch/ack/short").status_code == 422
        assert sent["id"] not in json.dumps(client.get("/api/v1/watch").json())

    def test_the_snapshot_says_never_before_any_canary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("fastapi")
        from omni_mercury_engine.api.routes.watch import snapshot

        monkeypatch.setenv("MERCURY_WATCH_DIR", str(tmp_path / "nothing-here"))

        assert snapshot()["proof_of_delivery"] == {
            "email": {"state": "never"}, "sms": {"state": "never"}, "webhook": {"state": "never"},
        }  # fmt: skip

    def test_the_code_proves_it_by_hand_for_a_day(
        self, store: WatchStore, clock: Clock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sent, _ = self._sent(store, clock, monkeypatch)
        wrong = f"{(int(sent['code']) + 1) % 1_000_000:06d}"

        assert not store.prove_canary(code=wrong, by="acknowledged", now=clock())
        clock.forward(hours=25)
        assert not store.prove_canary(code=sent["code"], by="acknowledged", now=clock())
        clock.forward(hours=-2)
        assert store.prove_canary(code=f" {sent['code']} ", by="acknowledged", now=clock())
        assert proof_of_delivery(store, clock())["webhook"]["by"] == "acknowledged"

    def test_the_store_keeps_no_address_token_or_code(
        self, store: WatchStore, smtp: SmtpServer, clock: Clock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MERCURY_PUBLIC_BASE_URL", "https://mercury.example.org")
        Canary(store, Notifier(store, clock=clock), clock=clock).send(email=OPERATOR)
        body = smtp.received[0].message.get_content()
        token = re.search(r"/api/v1/watch/ack/([\w-]+)", body)
        code = re.search(r"code (\d{6})", body)
        assert token is not None and code is not None

        kept = store.path.read_bytes()

        for secret in (OPERATOR, token.group(1), code.group(1)):
            assert secret.encode() not in kept


# --------------------------------------------------------------------------- #
# The watch sends them, and the commands
# --------------------------------------------------------------------------- #


class TestTheWatch:
    def test_the_watch_sends_one_when_due_and_only_the_watch_that_tells_people(
        self, store: WatchStore, clock: Clock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MERCURY_WATCH_CANARY_PHONE", NUMBER)
        monkeypatch.setenv("MERCURY_WATCH_CANARY_HOURS", "6")
        texts, other = Texts(), Texts()
        texts.became = other.became = "delivered"
        first = Watcher(
            store, feeds=[], clock=clock, notifier=Notifier(store, sms=texts, clock=clock)
        )
        second = Watcher(
            store, feeds=[], clock=clock, notifier=Notifier(store, sms=other, clock=clock)
        )

        first._pass(force=False)  # takes the lease, and sends the first
        second._pass(force=False)
        for _ in range(8):  # two hours of ticks
            clock.forward(minutes=15)
            first._pass(force=False)
            second._pass(force=False)

        assert len(texts.sent) == 1 and other.sent == []
        assert proof_of_delivery(store, clock())["sms"]["state"] == "proven"

        clock.forward(hours=4, minutes=1)
        first._pass(force=False)
        assert len(texts.sent) == 2

    def test_a_channel_that_refused_one_is_tried_again_within_the_hour_and_the_others_are_not(
        self, store: WatchStore, smtp: SmtpServer, clock: Clock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A mail server down for an hour is not "failing" for a day."""
        monkeypatch.setenv("MERCURY_WATCH_CANARY_EMAIL", OPERATOR)
        monkeypatch.setenv("MERCURY_WATCH_CANARY_PHONE", NUMBER)
        texts = Texts()
        texts.became = "delivered"
        canary = Canary(store, Notifier(store, sms=texts, clock=clock), clock=clock)
        smtp.refuse.add(OPERATOR)
        canary.run()
        assert proof_of_delivery(store, clock())["email"]["state"] == "failing"

        smtp.refuse.clear()
        clock.forward(minutes=30)
        canary.run()
        assert smtp.received == []  # not yet
        clock.forward(minutes=31)
        canary.run()

        assert len(smtp.received) == 1 and len(texts.sent) == 1
        assert proof_of_delivery(store, clock())["email"]["state"] == "awaiting"
        assert proof_of_delivery(store, clock())["sms"]["state"] == "proven"

    def test_with_none_asked_for_the_watch_does_not_touch_the_store_for_it(
        self, store: WatchStore, clock: Clock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in ("EMAIL", "PHONE", "WEBHOOK"):
            monkeypatch.delenv(f"MERCURY_WATCH_CANARY_{name}", raising=False)
        canary = Canary(store, Notifier(store, sms=Texts(), clock=clock), clock=clock)
        monkeypatch.setattr(store, "open_canaries", lambda since: pytest.fail("store was read"))
        monkeypatch.setattr(store, "latest_canaries", lambda: pytest.fail("store was read"))

        canary.run()

        assert not targets()

    def test_settings_that_are_wrong_are_said_and_left_out(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("MERCURY_WATCH_CANARY_EMAIL", OPERATOR)
        monkeypatch.setenv("MERCURY_WATCH_CANARY_PHONE", "555-0123")
        monkeypatch.setenv("MERCURY_WATCH_CANARY_HOURS", "0.01")

        with caplog.at_level(logging.ERROR, logger=canary_module.__name__):
            wanted = targets()

        assert (wanted.email, wanted.phone, wanted.every) == (OPERATOR, "", timedelta(hours=24))
        assert len(caplog.records) == 2

    def test_the_commands_send_wait_show_and_acknowledge(
        self, tmp_path: Path, smtp: SmtpServer, imap: ImapServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = ["--state-dir", str(tmp_path / "watch")]
        runner = CliRunner()

        nowhere = runner.invoke(main, ["alerts", "canary", *state])
        assert nowhere.exit_code == 2 and "say where the test goes" in nowhere.output

        sent = runner.invoke(
            main, ["alerts", "canary", "--email", OPERATOR, "--wait", "30", *state]
        )
        assert sent.exit_code == 0, sent.output
        assert "email: PROVEN (mailbox)." in sent.output

        shown = runner.invoke(main, ["alerts", "proof", "--json", *state])
        assert shown.exit_code == 0
        assert json.loads(shown.output)["email"]["by"] == "mailbox"

        # A second, to an address whose mailbox is not the one that is read:
        # taken, not proven, and the command says so with its exit status.
        other = runner.invoke(
            main, ["alerts", "canary", "--email", "other@example.org", "--wait", "1", *state]
        )
        assert other.exit_code == 1 and "nothing has yet said it arrived" in other.output
        code = re.search(r"alerts ack (\d{6})", other.output)
        assert code is not None
        wrong = f"{(int(code.group(1)) + 1) % 1_000_000:06d}"
        refused = runner.invoke(main, ["alerts", "ack", wrong, *state])
        assert refused.exit_code == 1 and "has that code" in refused.output
        acked = runner.invoke(main, ["alerts", "ack", code.group(1), *state])
        assert acked.exit_code == 0 and "that test arrived" in acked.output
        text = runner.invoke(main, ["alerts", "proof", *state])
        assert text.exit_code == 0 and "proven" in text.output and "(acknowledged)" in text.output

    def test_a_channel_the_deployment_does_not_have_is_an_error_not_a_silence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MERCURY_SMTP_HOST", raising=False)

        result = CliRunner().invoke(
            main, ["alerts", "canary", "--email", OPERATOR, "--state-dir", str(tmp_path / "w")]
        )

        assert result.exit_code == 1 and "cannot send by email" in result.output
