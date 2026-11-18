# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The watch's email goes through a real SMTP server.

Until now a message was "sent" to an object that recorded the call. Here the
deployment's mailer, built from ``MERCURY_SMTP_*`` exactly as a deployment
builds it, connects to a listening server, negotiates TLS, authenticates and
hands the message over, and the test reads what arrived.

What this cannot show is a provider accepting the message and a mailbox
receiving it: that needs the operator's server and an inbox, and is what
``mercury-agent alerts test`` is for.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner

from omni_mercury_engine.cli import main
from omni_mercury_engine.watch import Severity, WatchStore
from omni_mercury_engine.watch.guidance import with_guidance
from omni_mercury_engine.watch.model import Finding
from omni_mercury_engine.watch.notify import Notifier

from .smtp_server import OPENSSL, SmtpServer, mint_certificate

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

pytestmark = pytest.mark.skipif(
    OPENSSL is None, reason="openssl CLI not on PATH: the server's certificate is minted with it"
)

NOW = datetime(2026, 11, 17, 1, 0, tzinfo=UTC)


@pytest.fixture()
def certificate(tmp_path: Path) -> tuple[Path, Path]:
    return mint_certificate(tmp_path)


def _configure(
    monkeypatch: pytest.MonkeyPatch, server: SmtpServer, trusted: Path | None, starttls: bool
) -> None:
    """Point the deployment's SMTP settings at ``server``, as an operator would."""
    monkeypatch.setenv("MERCURY_SMTP_HOST", "localhost")
    monkeypatch.setenv("MERCURY_SMTP_PORT", str(server.port))
    monkeypatch.setenv("MERCURY_SMTP_USERNAME", server.credentials[0])
    monkeypatch.setenv("MERCURY_SMTP_PASSWORD", server.credentials[1])
    monkeypatch.setenv("MERCURY_SMTP_FROM", "Mercury watch <watch@example.org>")
    monkeypatch.setenv("MERCURY_SMTP_STARTTLS", "true" if starttls else "false")
    if trusted is not None:
        # The certificate authority the client trusts: here, the test's own.
        monkeypatch.setenv("SSL_CERT_FILE", str(trusted))
    else:
        monkeypatch.delenv("SSL_CERT_FILE", raising=False)


@pytest.fixture()
def smtp(monkeypatch: pytest.MonkeyPatch, certificate: tuple[Path, Path]) -> Iterator[SmtpServer]:
    with SmtpServer(*certificate) as server:
        _configure(monkeypatch, server, certificate[0], starttls=True)
        yield server


def _flood(
    store: WatchStore,
    headline: str = "Major flooding: St. Johns River near Cocoa, Florida",
    detail: str = "It read 17.09 ft; major flooding begins at 17 ft.",
) -> Finding:
    finding = with_guidance(
        Finding(
            feed="river_gauges",
            kind="river_flood",
            subject="COCF1",
            severity=Severity.SEVERE,
            headline=headline,
            detail=detail,
            observed_at=NOW,
            source="NOAA National Weather Service",
            basis="NWS flood category",
            latitude=28.36,
            longitude=-80.80,
            ongoing=True,
        )
    )
    store.report(finding, NOW)
    return finding


class TestDelivery:
    def test_a_finding_is_delivered_over_tls_with_the_deployments_credentials(
        self, tmp_path: Path, smtp: SmtpServer
    ) -> None:
        store = WatchStore(tmp_path / "watch")
        store.add_subscriber("ana@example.org", place="Home", latitude=28.37, longitude=-80.87)
        finding = _flood(store)

        done = Notifier(store, clock=lambda: NOW).sync()

        assert done == {"sent": 1, "failed": 0}
        (got,) = smtp.received
        assert got.encrypted and got.authenticated_as == "watch@example.org"
        assert (got.sender, got.recipients) == ("watch@example.org", ["ana@example.org"])
        message = got.message
        assert message["To"] == "ana@example.org"
        assert message["From"] == "Mercury watch <watch@example.org>"
        assert message["Subject"] == (
            "[Mercury watch] Severe: Major flooding: St. Johns River near Cocoa, Florida"
        )
        body = message.get_content()
        assert "It read 17.09 ft" in body and 'About 7 km from "Home".' in body
        assert "WHAT TO DO" in body and "turn around, don't drown" in body
        assert store.delivery(finding.key, 3, 1, "email")["status"] == "sent"

    def test_the_message_is_one_a_mail_server_and_a_mail_client_will_take(
        self, tmp_path: Path, smtp: SmtpServer
    ) -> None:
        """Was: no Date and no Message-ID (RFC 5322 requires the first), and nothing
        marking it as automatic, so an out-of-office reply answered every alert."""
        store = WatchStore(tmp_path / "watch")
        for name in ("ana", "bo"):
            store.add_subscriber(f"{name}@example.org")
        _flood(store)

        Notifier(store, clock=lambda: NOW).sync()

        first, second = (got.message for got in smtp.received)
        sent_at = parsedate_to_datetime(first["Date"])
        assert abs(datetime.now(UTC) - sent_at) < timedelta(minutes=5)
        assert first["Message-ID"].endswith("@example.org>")
        assert first["Message-ID"] != second["Message-ID"]
        assert first["Auto-Submitted"] == "auto-generated"

    def test_text_that_is_not_ascii_arrives_as_written(
        self, tmp_path: Path, smtp: SmtpServer
    ) -> None:
        store = WatchStore(tmp_path / "watch")
        store.add_subscriber("ana@example.org")
        _flood(store, "Major flooding: Río Grande at Peñitas", "El río está a 17,09 ft; 28.37° N.")

        Notifier(store, clock=lambda: NOW).sync()

        (got,) = smtp.received
        assert got.message["Subject"].endswith("Major flooding: Río Grande at Peñitas")
        assert "El río está a 17,09 ft; 28.37° N." in got.message.get_content()

    def test_an_address_the_server_refuses_is_recorded_and_costs_no_one_else_theirs(
        self, tmp_path: Path, smtp: SmtpServer
    ) -> None:
        store = WatchStore(tmp_path / "watch")
        store.add_subscriber("gone@example.org")
        store.add_subscriber("ana@example.org")
        finding = _flood(store)
        smtp.refuse.add("gone@example.org")
        clock = [NOW]
        notifier = Notifier(store, clock=lambda: clock[0])

        assert notifier.sync() == {"sent": 1, "failed": 1}
        assert [got.recipients for got in smtp.received] == [["ana@example.org"]]
        refused = store.delivery(finding.key, 3, 1, "email")
        assert refused["status"] == "failed" and "does not exist" in refused["error"]

        smtp.refuse.clear()  # the mailbox comes back
        clock[0] += timedelta(minutes=1)  # the wait after a first failure
        assert notifier.sync() == {"sent": 1, "failed": 0}
        assert store.delivery(finding.key, 3, 1, "email")["attempts"] == 2

    def test_a_server_whose_certificate_is_not_trusted_is_not_sent_the_message(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, certificate: tuple[Path, Path]
    ) -> None:
        """An address and a place are not handed to whoever answers on the port."""
        store = WatchStore(tmp_path / "watch")
        store.add_subscriber("ana@example.org")
        finding = _flood(store)
        with SmtpServer(*certificate) as server:
            _configure(monkeypatch, server, trusted=None, starttls=True)

            done = Notifier(store, clock=lambda: NOW).sync()

            assert done == {"sent": 0, "failed": 1} and server.received == []
        assert "CERTIFICATE_VERIFY_FAILED" in store.delivery(finding.key, 3, 1, "email")["error"]

    def test_tls_from_the_first_byte_is_used_when_starttls_is_off(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, certificate: tuple[Path, Path]
    ) -> None:
        store = WatchStore(tmp_path / "watch")
        store.add_subscriber("ana@example.org")
        _flood(store)
        with SmtpServer(*certificate, implicit_tls=True) as server:
            _configure(monkeypatch, server, certificate[0], starttls=False)

            assert Notifier(store, clock=lambda: NOW).sync() == {"sent": 1, "failed": 0}
            assert server.received[0].encrypted


class TestTheOperatorsCheck:
    def test_alerts_test_sends_a_real_message_to_a_subscriber(
        self, tmp_path: Path, smtp: SmtpServer
    ) -> None:
        where = ["--state-dir", str(tmp_path / "watch")]
        WatchStore(tmp_path / "watch").add_subscriber("Ana@Example.org")

        result = CliRunner().invoke(main, ["alerts", "test", "--email", "ana@example.org", *where])

        assert result.exit_code == 0, result.output
        assert "Test message sent to ana@example.org." in result.output
        (got,) = smtp.received
        assert got.encrypted and got.recipients == ["Ana@Example.org"]
        assert got.message["Subject"] == "[Mercury watch] Test: you can be reached"
        assert "Nothing has happened." in got.message.get_content()

    def test_alerts_test_says_why_when_the_server_will_not_take_it(
        self, tmp_path: Path, smtp: SmtpServer
    ) -> None:
        where = ["--state-dir", str(tmp_path / "watch")]
        WatchStore(tmp_path / "watch").add_subscriber("gone@example.org")
        smtp.refuse.add("gone@example.org")

        result = CliRunner().invoke(main, ["alerts", "test", "--email", "gone@example.org", *where])

        assert result.exit_code == 1
        assert "could not be sent" in result.output and "does not exist" in result.output
