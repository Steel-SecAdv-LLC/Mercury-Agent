# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for account lifecycle flows, CSRF defense-in-depth, and sessions.

Covers the authenticated change-password (session-rotating), the two-step
email change with re-verification of the NEW address, account deletion, data
export, the CSRF double-submit contract on every state-changing POST, and the
hardened session policy (idle timeout vs absolute lifetime, remember-me, and
rotation on privilege change).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omni_mercury_engine.api.auth_service import (
    AuthService,
    EmailAlreadyRegisteredError,
    InvalidCredentialsError,
)
from omni_mercury_engine.api.identity_store import InMemoryIdentityStore
from omni_mercury_engine.api.routes import accounts

if TYPE_CHECKING:
    from collections.abc import Iterator


class RecordingMailer:
    """Collects messages so tests can extract tokens."""

    def __init__(self) -> None:
        """Start with an empty outbox."""
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
        self.sent.append({"to": to, "subject": subject, "body": body})

    def last_token(self) -> str:
        """Extract the token from the most recent email."""
        match = re.search(r"token=([\w\-]+)", self.sent[-1]["body"])
        assert match is not None
        return match.group(1)


class FakeClock:
    """Movable clock."""

    def __init__(self) -> None:
        """Start at a fixed instant."""
        self.now = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        """Return the fake time."""
        return self.now

    def advance(self, **kwargs: float) -> None:
        """Move forward by the given timedelta components."""
        self.now += timedelta(**kwargs)


#: The tuple the ``setup`` fixture yields; named so every test annotates it fully
#: (the strict test-mypy gate rejects a bare ``tuple``).
Setup = tuple[TestClient, RecordingMailer, AuthService, FakeClock]


@pytest.fixture
def setup(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Setup]:
    """TestClient + service over an in-memory store with a movable clock."""
    monkeypatch.setenv("MERCURY_SESSION_COOKIE_SECURE", "false")
    mailer = RecordingMailer()
    clock = FakeClock()
    service = AuthService(InMemoryIdentityStore(), mailer, clock=clock)
    app = FastAPI()
    app.include_router(accounts.router)
    app.dependency_overrides[accounts.get_auth_service] = lambda: service
    with TestClient(app) as client:
        yield client, mailer, service, clock


def _register_verify_login(
    client: TestClient, mailer: RecordingMailer, email: str = "u@b.com"
) -> str:
    """Register, verify, and log in; return the CSRF token."""
    assert (
        client.post(
            "/api/v1/auth/register", json={"email": email, "password": "a-strong-pw"}
        ).status_code
        == 201
    )
    assert (
        client.post("/api/v1/auth/verify-email", json={"token": mailer.last_token()}).status_code
        == 200
    )
    resp = client.post("/api/v1/auth/login", json={"email": email, "password": "a-strong-pw"})
    assert resp.status_code == 200
    return str(resp.json()["csrf_token"])


class TestCsrfDefenseInDepth:
    """State-changing POSTs demand the double-submit header."""

    def test_missing_header_is_403(self, setup: Setup) -> None:
        """An authenticated POST without X-CSRF-Token is rejected."""
        client, mailer, _service, _clock = setup
        _register_verify_login(client, mailer)
        resp = client.post(
            "/api/v1/auth/password/change",
            json={"current_password": "a-strong-pw", "new_password": "next-strong-pw"},
        )
        assert resp.status_code == 403
        assert "X-CSRF-Token" in str(resp.json()["detail"])

    def test_wrong_token_is_403(self, setup: Setup) -> None:
        """A forged token value is rejected."""
        client, mailer, _service, _clock = setup
        _register_verify_login(client, mailer)
        resp = client.post(
            "/api/v1/auth/2fa/enroll", headers={"X-CSRF-Token": "forged-token-value"}
        )
        assert resp.status_code == 403

    def test_correct_token_passes(self, setup: Setup) -> None:
        """The token issued at login authorises the request."""
        client, mailer, _service, _clock = setup
        csrf = _register_verify_login(client, mailer)
        assert (
            client.post("/api/v1/auth/2fa/enroll", headers={"X-CSRF-Token": csrf}).status_code
            == 200
        )

    def test_csrf_cookie_is_set_at_login(self, setup: Setup) -> None:
        """The readable CSRF cookie rides alongside the httpOnly session."""
        client, mailer, _service, _clock = setup
        csrf = _register_verify_login(client, mailer)
        assert client.cookies.get(accounts.CSRF_COOKIE) == csrf

    def test_enforcement_can_be_disabled_for_api_clients(
        self, setup: Setup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MERCURY_CSRF_PROTECTION=false waives the header (documented escape)."""
        client, mailer, _service, _clock = setup
        _register_verify_login(client, mailer)
        monkeypatch.setenv("MERCURY_CSRF_PROTECTION", "false")
        assert client.post("/api/v1/auth/2fa/enroll").status_code == 200


class TestChangePassword:
    """Authenticated password change: re-auth, rotation, continuity."""

    def test_change_rotates_sessions_and_keeps_caller_signed_in(self, setup: Setup) -> None:
        """Other sessions die; the caller continues on fresh cookies."""
        client, mailer, service, _clock = setup
        csrf = _register_verify_login(client, mailer)
        # A second session (another device / an attacker with the cookie).
        hijacked = service.login("u@b.com", "a-strong-pw").session_token

        resp = client.post(
            "/api/v1/auth/password/change",
            json={"current_password": "a-strong-pw", "new_password": "next-strong-pw"},
            headers={"X-CSRF-Token": csrf},
        )
        assert resp.status_code == 200
        # The other session is dead; the caller's refreshed cookie works.
        assert service.authenticate_session(hijacked) is None
        assert client.get("/api/v1/auth/me").status_code == 200
        # Old password out, new password in.
        client.post("/api/v1/auth/logout")
        assert (
            client.post(
                "/api/v1/auth/login", json={"email": "u@b.com", "password": "a-strong-pw"}
            ).status_code
            == 401
        )
        assert (
            client.post(
                "/api/v1/auth/login", json={"email": "u@b.com", "password": "next-strong-pw"}
            ).status_code
            == 200
        )

    def test_wrong_current_password_is_401(self, setup: Setup) -> None:
        """A hijacked cookie alone cannot change the password."""
        client, mailer, _service, _clock = setup
        csrf = _register_verify_login(client, mailer)
        resp = client.post(
            "/api/v1/auth/password/change",
            json={"current_password": "not-the-password", "new_password": "next-strong-pw"},
            headers={"X-CSRF-Token": csrf},
        )
        assert resp.status_code == 401

    def test_change_invalidates_outstanding_reset_tokens(self, setup: Setup) -> None:
        """A pre-issued reset link dies when the password changes."""
        client, mailer, service, _clock = setup
        csrf = _register_verify_login(client, mailer)
        service.request_password_reset("u@b.com")
        stale_reset = mailer.last_token()
        client.post(
            "/api/v1/auth/password/change",
            json={"current_password": "a-strong-pw", "new_password": "next-strong-pw"},
            headers={"X-CSRF-Token": csrf},
        )
        resp = client.post(
            "/api/v1/auth/password-reset/confirm",
            json={"token": stale_reset, "new_password": "attacker-chosen-pw"},
        )
        assert resp.status_code == 400


class TestChangeEmail:
    """Two-step email change with re-verification of the new address."""

    def test_full_flow(self, setup: Setup) -> None:
        """Request → confirmation link to NEW address → address flips."""
        client, mailer, service, _clock = setup
        csrf = _register_verify_login(client, mailer)
        resp = client.post(
            "/api/v1/auth/email-change/request",
            json={"new_email": "new@b.com", "current_password": "a-strong-pw"},
            headers={"X-CSRF-Token": csrf},
        )
        assert resp.status_code == 202
        assert mailer.sent[-1]["to"] == "new@b.com"  # link goes to the NEW box

        confirm = client.post(
            "/api/v1/auth/email-change/confirm", json={"token": mailer.last_token()}
        )
        assert confirm.status_code == 200
        assert confirm.json()["email"] == "new@b.com"
        # Address change is a privilege change: sessions were dropped.
        assert client.get("/api/v1/auth/me").status_code == 401
        # Old email is free again; new email logs in.
        assert service.login("new@b.com", "a-strong-pw").account is not None
        with pytest.raises(InvalidCredentialsError):
            service.login("u@b.com", "a-strong-pw")

    def test_taken_address_is_409(self, setup: Setup) -> None:
        """A new address owned by someone else is rejected up front."""
        client, mailer, service, _clock = setup
        service.register("taken@b.com", "another-strong-pw")
        csrf = _register_verify_login(client, mailer)
        resp = client.post(
            "/api/v1/auth/email-change/request",
            json={"new_email": "taken@b.com", "current_password": "a-strong-pw"},
            headers={"X-CSRF-Token": csrf},
        )
        assert resp.status_code == 409

    def test_race_on_confirm_is_rejected(self, setup: Setup) -> None:
        """Uniqueness is re-checked at commit: a raced claim loses cleanly."""
        client, mailer, service, _clock = setup
        csrf = _register_verify_login(client, mailer)
        client.post(
            "/api/v1/auth/email-change/request",
            json={"new_email": "contested@b.com", "current_password": "a-strong-pw"},
            headers={"X-CSRF-Token": csrf},
        )
        change_token = mailer.last_token()
        # Someone registers the address between request and confirm.
        service.register("contested@b.com", "third-strong-pw")
        with pytest.raises(EmailAlreadyRegisteredError):
            service.confirm_email_change(change_token)


class TestDeletionAndExport:
    """Account deletion (re-authenticated) and data export."""

    def test_delete_requires_password_and_removes_account(self, setup: Setup) -> None:
        """Deletion re-authenticates, then the account and sessions are gone."""
        client, mailer, service, _clock = setup
        csrf = _register_verify_login(client, mailer)
        wrong = client.post(
            "/api/v1/auth/account/delete",
            json={"current_password": "not-it"},
            headers={"X-CSRF-Token": csrf},
        )
        assert wrong.status_code == 401

        ok = client.post(
            "/api/v1/auth/account/delete",
            json={"current_password": "a-strong-pw"},
            headers={"X-CSRF-Token": csrf},
        )
        assert ok.status_code == 200
        assert client.get("/api/v1/auth/me").status_code == 401
        with pytest.raises(InvalidCredentialsError):
            service.login("u@b.com", "a-strong-pw")
        # The email is registrable again (hard delete, not a tombstone).
        assert service.register("u@b.com", "brand-new-pw123").email == "u@b.com"

    def test_export_returns_profile_without_secrets(self, setup: Setup) -> None:
        """The export carries the profile and never any secret material."""
        client, mailer, _service, _clock = setup
        _register_verify_login(client, mailer)
        resp = client.get("/api/v1/auth/export")
        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == "u@b.com"
        assert data["is_verified"] is True
        assert "password" not in str(data).lower() or "password_hash" not in data
        assert "totp_secret" not in data


class TestSessionHardening:
    """Idle timeout, absolute lifetime, and remember-me semantics."""

    def _service_with_clock(self) -> tuple[AuthService, RecordingMailer, FakeClock]:
        mailer = RecordingMailer()
        clock = FakeClock()
        service = AuthService(
            InMemoryIdentityStore(),
            mailer,
            clock=clock,
            session_ttl=timedelta(days=14),
            session_ttl_short=timedelta(days=1),
            session_idle_timeout=timedelta(hours=6),
        )
        service.register("u@b.com", "a-strong-pw")
        service.verify_email(mailer.last_token())
        return service, mailer, clock

    def test_idle_timeout_kills_inactive_session(self) -> None:
        """A session unused past the idle window stops authenticating."""
        service, _mailer, clock = self._service_with_clock()
        token = service.login("u@b.com", "a-strong-pw").session_token
        clock.advance(hours=7)
        assert service.authenticate_session(token) is None

    def test_activity_keeps_session_alive_within_absolute_ttl(self) -> None:
        """Regular activity resets idleness but not the absolute clock."""
        service, _mailer, clock = self._service_with_clock()
        token = service.login("u@b.com", "a-strong-pw").session_token
        for _ in range(4):
            clock.advance(hours=5)  # always within the 6h idle window
            assert service.authenticate_session(token) is not None

    def test_absolute_ttl_caps_even_active_sessions(self) -> None:
        """No amount of activity extends a session past its absolute TTL."""
        service, _mailer, clock = self._service_with_clock()
        token = service.login("u@b.com", "a-strong-pw").session_token
        for _ in range(14 * 5):  # touch every ~4.8h for 14 days
            clock.advance(hours=4, minutes=48)
            service.authenticate_session(token)
        assert service.authenticate_session(token) is None

    def test_remember_me_off_uses_short_ttl(self) -> None:
        """Without remember-me the absolute lifetime is the short one."""
        service, _mailer, clock = self._service_with_clock()
        result = service.login("u@b.com", "a-strong-pw", remember_me=False)
        assert result.persistent is False
        clock.advance(hours=5)
        assert service.authenticate_session(result.session_token) is not None
        clock.advance(hours=20)  # past the 1-day short TTL
        assert service.authenticate_session(result.session_token) is None

    def test_remember_me_off_sets_browser_session_cookie(self, setup: Setup) -> None:
        """The cookie for a non-remembered login carries no Max-Age."""
        client, mailer, _service, _clock = setup
        client.post("/api/v1/auth/register", json={"email": "s@b.com", "password": "a-strong-pw"})
        client.post("/api/v1/auth/verify-email", json={"token": mailer.last_token()})
        resp = client.post(
            "/api/v1/auth/login",
            json={"email": "s@b.com", "password": "a-strong-pw", "remember_me": False},
        )
        set_cookie = resp.headers.get_list("set-cookie")
        session_cookie = next(c for c in set_cookie if c.startswith(accounts.SESSION_COOKIE))
        assert "Max-Age" not in session_cookie
