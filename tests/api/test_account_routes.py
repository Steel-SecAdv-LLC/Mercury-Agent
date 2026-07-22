# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""HTTP-level tests for the self-service account routes.

Drives the FastAPI router end-to-end with a TestClient over an in-memory auth
service (recording mailer + fixed clock), exercising the cookie session flow,
error-to-status mapping, and the 2FA challenge — no network, no secrets.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omni_mercury_engine.api import totp
from omni_mercury_engine.api.auth_service import AuthService
from omni_mercury_engine.api.identity_store import InMemoryIdentityStore
from omni_mercury_engine.api.routes import accounts

if TYPE_CHECKING:
    from collections.abc import Iterator


class RecordingMailer:
    """Mailer that records messages so tests can read the emailed token."""

    def __init__(self) -> None:
        """Start with an empty outbox."""
        self.sent: list[dict[str, str]] = []

    def send(self, *, to: str, subject: str, body: str) -> None:
        """Record instead of delivering."""
        self.sent.append({"to": to, "subject": subject, "body": body})

    def last_token(self) -> str:
        """Extract the token from the most recent email."""
        match = re.search(r"token=([A-Za-z0-9_\-]+)", self.sent[-1]["body"])
        assert match is not None
        return match.group(1)


class FakeClock:
    """A fixed clock so TOTP codes are deterministic."""

    def __init__(self, start: datetime) -> None:
        """Start the clock at ``start``."""
        self.now = start

    def __call__(self) -> datetime:
        """Return the current fake time."""
        return self.now


@pytest.fixture
def client_and_mail(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[TestClient, RecordingMailer, AuthService]]:
    """A TestClient wired to an in-memory auth service via dependency override."""
    # Allow the session cookie to ride back over the TestClient's http transport.
    monkeypatch.setenv("MERCURY_SESSION_COOKIE_SECURE", "false")
    mailer = RecordingMailer()
    service = AuthService(
        InMemoryIdentityStore(),
        mailer,
        clock=FakeClock(datetime(2026, 1, 1, tzinfo=UTC)),
    )
    app = FastAPI()
    app.include_router(accounts.router)
    app.dependency_overrides[accounts.get_auth_service] = lambda: service
    with TestClient(app) as client:
        yield client, mailer, service


def _register_and_verify(client: TestClient, mailer: RecordingMailer, email: str) -> None:
    """Register and verify an account through the HTTP routes."""
    resp = client.post("/api/v1/auth/register", json={"email": email, "password": "a-strong-pw"})
    assert resp.status_code == 201
    verify = client.post("/api/v1/auth/verify-email", json={"token": mailer.last_token()})
    assert verify.status_code == 200
    assert verify.json()["is_verified"] is True


def test_register_validation(
    client_and_mail: tuple[TestClient, RecordingMailer, AuthService],
) -> None:
    """Bad email -> 400, weak password -> 400, duplicate -> 409."""
    client, _, _ = client_and_mail
    assert (
        client.post(
            "/api/v1/auth/register", json={"email": "x", "password": "a-strong-pw"}
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/api/v1/auth/register", json={"email": "a@b.com", "password": "short"}
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/api/v1/auth/register", json={"email": "dup@b.com", "password": "a-strong-pw"}
        ).status_code
        == 201
    )
    assert (
        client.post(
            "/api/v1/auth/register", json={"email": "dup@b.com", "password": "a-strong-pw"}
        ).status_code
        == 409
    )


def test_login_requires_verification_then_succeeds(
    client_and_mail: tuple[TestClient, RecordingMailer, AuthService],
) -> None:
    """Login is 403 before verification and 200 (with a session) after."""
    client, mailer, _ = client_and_mail
    client.post("/api/v1/auth/register", json={"email": "u@b.com", "password": "a-strong-pw"})
    early = client.post("/api/v1/auth/login", json={"email": "u@b.com", "password": "a-strong-pw"})
    assert early.status_code == 403

    client.post("/api/v1/auth/verify-email", json={"token": mailer.last_token()})
    ok = client.post("/api/v1/auth/login", json={"email": "u@b.com", "password": "a-strong-pw"})
    assert ok.status_code == 200
    assert accounts.SESSION_COOKIE in ok.cookies


def test_me_requires_session(
    client_and_mail: tuple[TestClient, RecordingMailer, AuthService],
) -> None:
    """/me is 401 without a session and 200 with one."""
    client, mailer, _ = client_and_mail
    assert client.get("/api/v1/auth/me").status_code == 401
    _register_and_verify(client, mailer, "u@b.com")
    client.post("/api/v1/auth/login", json={"email": "u@b.com", "password": "a-strong-pw"})
    me = client.get("/api/v1/auth/me")
    assert me.status_code == 200
    assert me.json()["email"] == "u@b.com"


def test_wrong_password_is_401(
    client_and_mail: tuple[TestClient, RecordingMailer, AuthService],
) -> None:
    """A wrong password yields 401."""
    client, mailer, _ = client_and_mail
    _register_and_verify(client, mailer, "u@b.com")
    resp = client.post("/api/v1/auth/login", json={"email": "u@b.com", "password": "wrong-pw!!"})
    assert resp.status_code == 401


def test_logout_clears_session(
    client_and_mail: tuple[TestClient, RecordingMailer, AuthService],
) -> None:
    """After logout, /me is 401 again."""
    client, mailer, _ = client_and_mail
    _register_and_verify(client, mailer, "u@b.com")
    client.post("/api/v1/auth/login", json={"email": "u@b.com", "password": "a-strong-pw"})
    assert client.get("/api/v1/auth/me").status_code == 200
    assert client.post("/api/v1/auth/logout").status_code == 200
    assert client.get("/api/v1/auth/me").status_code == 401


def test_password_reset_flow(
    client_and_mail: tuple[TestClient, RecordingMailer, AuthService],
) -> None:
    """Reset request -> confirm -> old password fails, new works."""
    client, mailer, _ = client_and_mail
    _register_and_verify(client, mailer, "u@b.com")

    req = client.post("/api/v1/auth/password-reset/request", json={"email": "u@b.com"})
    assert req.status_code == 202
    confirm = client.post(
        "/api/v1/auth/password-reset/confirm",
        json={"token": mailer.last_token(), "new_password": "a-new-strong-pw"},
    )
    assert confirm.status_code == 200

    assert (
        client.post(
            "/api/v1/auth/login", json={"email": "u@b.com", "password": "a-strong-pw"}
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/api/v1/auth/login", json={"email": "u@b.com", "password": "a-new-strong-pw"}
        ).status_code
        == 200
    )


def test_password_reset_unknown_email_is_202_and_silent(
    client_and_mail: tuple[TestClient, RecordingMailer, AuthService],
) -> None:
    """Reset for an unknown email is 202 and sends nothing (no enumeration)."""
    client, mailer, _ = client_and_mail
    resp = client.post("/api/v1/auth/password-reset/request", json={"email": "ghost@b.com"})
    assert resp.status_code == 202
    assert mailer.sent == []


def test_two_factor_challenge(
    client_and_mail: tuple[TestClient, RecordingMailer, AuthService],
) -> None:
    """Enrolling 2FA makes login require a code; the code path is exercised."""
    client, mailer, service = client_and_mail
    _register_and_verify(client, mailer, "u@b.com")
    client.post("/api/v1/auth/login", json={"email": "u@b.com", "password": "a-strong-pw"})

    enroll = client.post("/api/v1/auth/2fa/enroll")
    assert enroll.status_code == 200
    secret = enroll.json()["secret"]
    code = totp.generate_totp(secret, at=datetime(2026, 1, 1, tzinfo=UTC).timestamp())
    assert client.post("/api/v1/auth/2fa/confirm", json={"code": code}).status_code == 200

    client.post("/api/v1/auth/logout")
    # Password alone is now rejected with the two_factor_required signal.
    challenge = client.post(
        "/api/v1/auth/login", json={"email": "u@b.com", "password": "a-strong-pw"}
    )
    assert challenge.status_code == 401
    assert challenge.json()["detail"]["code"] == "two_factor_required"
    # Password + valid code succeeds.
    good = totp.generate_totp(secret, at=datetime(2026, 1, 1, tzinfo=UTC).timestamp())
    ok = client.post(
        "/api/v1/auth/login",
        json={"email": "u@b.com", "password": "a-strong-pw", "totp_code": good},
    )
    assert ok.status_code == 200
