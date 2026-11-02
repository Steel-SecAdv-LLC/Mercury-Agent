# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for maintenance sweeps, mailer resilience, and the audit seam.

Maintenance must prune every kind of expiring/append-only state, apply the
TOTP sealing migration, and isolate per-store failures. The mailer path must
never fail a committed account change (best-effort delivery), and auth events
must reach the audit sink without ever raising into the request.
"""

from __future__ import annotations

import gc
import re
import warnings
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from omni_mercury_engine.api import totp
from omni_mercury_engine.api.auth_audit import AuthAuditor
from omni_mercury_engine.api.auth_service import AuthService
from omni_mercury_engine.api.identity_store import (
    EmailToken,
    InMemoryIdentityStore,
    Session,
    SqliteIdentityStore,
)
from omni_mercury_engine.api.maintenance import run_maintenance_sweep, usage_retention
from omni_mercury_engine.api.usage_ledger import SqliteUsageLedger, UsageEvent

if TYPE_CHECKING:
    from pathlib import Path


class ExplodingMailer:
    """A mailer that always raises (a hard SMTP outage)."""

    def send(self, **_kwargs: object) -> None:
        """Fail every send."""
        raise RuntimeError("SMTP is down")


class RecordingMailer:
    """Records messages."""

    def __init__(self) -> None:
        """Empty outbox."""
        self.sent: list[dict[str, str]] = []

    def send(
        self,
        *,
        to: str,
        subject: str,
        body: str,
        html_body: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Record the message."""
        self.sent.append(
            {
                "to": to,
                "subject": subject,
                "body": body,
                "html_body": html_body or "",
                "headers": str(headers or {}),
            }
        )

    def last_token(self) -> str:
        """Extract the token from the most recent email."""
        return re.search(r"token=([\w\-]+)", self.sent[-1]["body"]).group(1)  # type: ignore[union-attr]


class FixedClock:
    """A clock pinned to one instant."""

    def __init__(self) -> None:
        """Pin the time."""
        self.now = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        """Return the time."""
        return self.now


class TestMailerResilience:
    """A committed account change must survive a mailer outage."""

    def test_register_commits_despite_mailer_failure(self) -> None:
        """Registration persists even when the verification email throws."""
        store = InMemoryIdentityStore()
        service = AuthService(store, ExplodingMailer(), clock=FixedClock())
        account = service.register("u@b.com", "a-strong-password")
        # The account is committed and independently verifiable/loginable...
        assert store.get_account_by_id(account.id) is not None
        # ...via a token the caller can mint through resend once mail recovers.
        assert store.get_account_by_email("u@b.com") is not None

    def test_reset_request_swallows_mailer_failure(self) -> None:
        """A reset request never surfaces the SMTP error to the caller."""
        store = InMemoryIdentityStore()
        mailer = RecordingMailer()
        service = AuthService(store, mailer, clock=FixedClock())
        service.register("u@b.com", "a-strong-password")
        service.verify_email(mailer.last_token())
        service._mailer = ExplodingMailer()  # SMTP goes down mid-life
        service.request_password_reset("u@b.com")  # must not raise
        # The account survives the failed send and remains loginable.
        assert store.get_account_by_email("u@b.com") is not None

    def test_emails_carry_html_and_unsubscribe(self) -> None:
        """Transactional mail includes an HTML part and List-Unsubscribe."""
        mailer = RecordingMailer()
        service = AuthService(InMemoryIdentityStore(), mailer, clock=FixedClock())
        service.register("u@b.com", "a-strong-password")
        sent = mailer.sent[-1]
        assert sent["html_body"]
        assert "List-Unsubscribe" in sent["headers"]


class TestAuditSeam:
    """Auth events reach the sink and never raise into the caller."""

    def test_events_recorded(self) -> None:
        """Login success/failure produce audit records."""
        records: list[dict[str, object]] = []

        class CapturingLogger:
            def log_authentication(self, **kwargs: object) -> str:
                records.append(dict(kwargs))
                return "id"

        mailer = RecordingMailer()
        service = AuthService(
            InMemoryIdentityStore(),
            mailer,
            clock=FixedClock(),
            auditor=AuthAuditor(CapturingLogger()),  # type: ignore[arg-type]
        )
        service.register("u@b.com", "a-strong-password")
        service.verify_email(mailer.last_token())
        service.login("u@b.com", "a-strong-password")
        try:
            service.login("u@b.com", "wrong-password")
        except Exception:
            pass
        actions = [r["action"] for r in records]
        assert "register" in actions
        assert "login" in actions
        outcomes = {(r["action"], r["outcome"]) for r in records}
        assert ("login", "success") in outcomes
        assert ("login", "failure") in outcomes

    def test_broken_sink_never_breaks_auth(self) -> None:
        """An audit sink that raises does not fail the login."""

        class BrokenLogger:
            def log_authentication(self, **_kwargs: object) -> str:
                raise RuntimeError("audit backend down")

        mailer = RecordingMailer()
        service = AuthService(
            InMemoryIdentityStore(),
            mailer,
            clock=FixedClock(),
            auditor=AuthAuditor(BrokenLogger()),  # type: ignore[arg-type]
        )
        service.register("u@b.com", "a-strong-password")
        service.verify_email(mailer.last_token())
        assert service.login("u@b.com", "a-strong-password").account is not None


class TestMaintenanceSweep:
    """The sweep prunes every store and applies the sealing migration."""

    def test_sweep_closes_every_store_it_opens(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: each sweep leaked four SQLite connections.

        The periodic loop runs the sweep in the long-lived API process, so the
        identity store, usage ledger and both rate-limit stores it opened were
        file descriptors leaked per sweep, without bound.
        """
        # One file backs the identity store, ledger and both rate-limit stores.
        monkeypatch.setenv("MERCURY_KEYSTORE_PATH", str(tmp_path / "platform.db"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run_maintenance_sweep(now=datetime(2026, 6, 1, tzinfo=UTC))
            gc.collect()
        leaked = [str(w.message) for w in caught if "unclosed database" in str(w.message)]
        assert leaked == [], leaked

    def test_prunes_expired_sessions_and_tokens(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Expired sessions and consumed/expired tokens are removed."""
        db = tmp_path / "identity.db"
        monkeypatch.setenv("MERCURY_KEYSTORE_PATH", str(db))
        store = SqliteIdentityStore(db)
        past = datetime(2025, 1, 1, tzinfo=UTC)
        now = datetime(2026, 6, 1, tzinfo=UTC)
        store.create_session(Session("live", "acct", now, now + timedelta(days=1)))
        store.create_session(Session("dead", "acct", past, past + timedelta(days=1)))
        store.create_email_token(
            EmailToken("expired", "acct", "verify", past, past + timedelta(hours=1))
        )
        store.create_email_token(
            EmailToken("fresh", "acct", "verify", now, now + timedelta(hours=1))
        )
        store.close()

        results = run_maintenance_sweep(now=now)
        assert results["expired_sessions"] == 1
        assert results["email_tokens"] == 1

        reopened = SqliteIdentityStore(db)
        assert reopened.get_session("live") is not None
        assert reopened.get_session("dead") is None
        assert reopened.get_email_token("fresh") is not None
        assert reopened.get_email_token("expired") is None
        reopened.close()

    def test_prunes_aged_usage_events(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Usage rows past the retention window are deleted."""
        db = tmp_path / "identity.db"
        monkeypatch.setenv("MERCURY_KEYSTORE_PATH", str(db))
        now = datetime(2026, 6, 1, tzinfo=UTC)
        ledger = SqliteUsageLedger(db)
        ledger.record(UsageEvent("acct", now - usage_retention() - timedelta(days=1), "/d", 1.0))
        ledger.record(UsageEvent("acct", now, "/d", 1.0))
        ledger.close()

        results = run_maintenance_sweep(now=now)
        assert results["usage_events"] == 1

    def test_migrates_plaintext_totp_secret(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A plaintext TOTP secret is sealed in place by the sweep."""
        db = tmp_path / "identity.db"
        monkeypatch.setenv("MERCURY_KEYSTORE_PATH", str(db))
        monkeypatch.setenv("MERCURY_DATA_ENC_KEY", "ab" * 32)
        store = SqliteIdentityStore(db)
        mailer = RecordingMailer()
        service = AuthService(store, mailer, clock=FixedClock())
        account = service.register("u@b.com", "a-strong-password")
        service.verify_email(mailer.last_token())
        # Write a legacy plaintext secret directly.
        acct = store.get_account_by_id(account.id)
        acct.totp_secret = totp.generate_secret()  # type: ignore[union-attr]
        acct.totp_enabled = True  # type: ignore[union-attr]
        store.update_account(acct)  # type: ignore[arg-type]
        store.close()

        results = run_maintenance_sweep(now=datetime(2026, 6, 1, tzinfo=UTC))
        assert results["sealed_totp_secrets"] == 1

        reopened = SqliteIdentityStore(db)
        migrated = reopened.get_account_by_id(account.id)
        assert migrated is not None and migrated.totp_secret is not None
        assert migrated.totp_secret.startswith("enc$")
        reopened.close()

    def test_in_memory_sweep_is_noop_safe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no durable store the sweep runs cleanly over empty state."""
        monkeypatch.delenv("MERCURY_KEYSTORE_PATH", raising=False)
        results = run_maintenance_sweep(now=datetime(2026, 6, 1, tzinfo=UTC))
        assert results["expired_sessions"] == 0
        assert results["usage_events"] == 0


class TestConsoleMailerOutbox:
    """The development mailer delivers to a file, so the link in it is usable."""

    def test_files_the_message_with_the_link_intact(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The outbox file carries the full link; the log names the file, not the body."""
        from omni_mercury_engine.api.mailer import ConsoleMailer

        mailer = ConsoleMailer(tmp_path / "outbox")
        link = "https://mercuryagent.global/verify-email?token=abc123SECRET"
        with caplog.at_level("INFO", logger="omni_mercury_engine.api.mailer"):
            mailer.send(
                to="who@example.test",
                subject="Verify",
                body=f"Open {link} within 24 hours.",
                html_body=f'<a href="{link}">verify</a>',
                headers={"List-Unsubscribe": "<mailto:contact@example.test>"},
            )

        files = sorted((tmp_path / "outbox").iterdir())
        assert [f.suffix for f in files] == [".html", ".txt"]
        text = files[1].read_text(encoding="utf-8")
        assert text.startswith("To: who@example.test\nSubject: Verify\nDate: ")
        assert "List-Unsubscribe: <mailto:contact@example.test>" in text
        assert f"\n\nOpen {link} within 24 hours." in text
        assert link in files[0].read_text(encoding="utf-8")
        assert all(f.stat().st_mode & 0o077 == 0 for f in files)
        assert (tmp_path / "outbox").stat().st_mode & 0o077 == 0
        record = caplog.records[-1].getMessage()
        assert str(files[1]) in record
        assert "SECRET" not in record

    def test_environment_and_default_locations(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MERCURY_MAIL_OUTBOX wins; otherwise the XDG state root Mercury already inventories."""
        from omni_mercury_engine.api.mailer import OUTBOX_ENV, ConsoleMailer, build_mailer

        monkeypatch.delenv("MERCURY_SMTP_HOST", raising=False)
        monkeypatch.setenv(OUTBOX_ENV, str(tmp_path / "box"))
        built = build_mailer()
        assert isinstance(built, ConsoleMailer)
        assert built.outbox == tmp_path / "box"

        monkeypatch.delenv(OUTBOX_ENV)
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
        assert ConsoleMailer().outbox == tmp_path / "state" / "mercury-agent" / "outbox"

    def test_registration_link_is_recoverable_from_the_outbox(self, tmp_path: Path) -> None:
        """The flow a developer runs: register, open the outbox, follow the link."""
        from omni_mercury_engine.api.mailer import ConsoleMailer

        store = InMemoryIdentityStore()
        service = AuthService(store, ConsoleMailer(tmp_path), clock=FixedClock())
        service.register("dev@example.test", "Correct-Horse-Battery-9!")

        [message] = [f for f in tmp_path.iterdir() if f.suffix == ".txt"]
        match = re.search(r"/verify-email\?token=([A-Za-z0-9_.~-]+)", message.read_text("utf-8"))
        assert match, message.read_text("utf-8")
        service.verify_email(match.group(1))
        account = store.get_account_by_email("dev@example.test")
        assert account is not None and account.is_verified

    def test_unwritable_outbox_is_a_delivery_failure(self, tmp_path: Path) -> None:
        """A write error surfaces to the caller's delivery audit instead of vanishing."""
        from omni_mercury_engine.api.mailer import ConsoleMailer

        blocker = tmp_path / "file-not-dir"
        blocker.write_text("x", encoding="utf-8")
        mailer = ConsoleMailer(blocker / "outbox")
        with pytest.raises(OSError):
            mailer.send(to="a@b.test", subject="s", body="b")

        store = InMemoryIdentityStore()
        service = AuthService(store, mailer, clock=FixedClock())
        service.register("dev@example.test", "Correct-Horse-Battery-9!")
        assert store.get_account_by_email("dev@example.test") is not None
