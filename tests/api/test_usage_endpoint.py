# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for ``GET /api/v1/auth/usage`` (the dashboard's usage/limits read).

Covers both caller kinds (browser session and ``X-API-Key``), the
unauthenticated 401, and the effective-limit resolution order
(override > tier > default) surfacing through the endpoint.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omni_mercury_engine.api import auth
from omni_mercury_engine.api.auth import APIKeyStore, Permission
from omni_mercury_engine.api.auth_service import AuthService
from omni_mercury_engine.api.identity_store import InMemoryIdentityStore
from omni_mercury_engine.api.quota import (
    InMemoryQuotaOverrideStore,
    QuotaConfig,
    QuotaEnforcer,
    get_shared_quota_enforcer,
)
from omni_mercury_engine.api.routes import accounts
from omni_mercury_engine.api.usage_ledger import InMemoryUsageLedger, UsageEvent

if TYPE_CHECKING:
    from collections.abc import Iterator


class RecordingMailer:
    """Mailer that records messages so tests can read the emailed token."""

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
        """Record instead of delivering."""
        self.sent.append({"to": to, "subject": subject, "body": body})

    def last_token(self) -> str:
        """Extract the token from the most recent email."""
        import re

        match = re.search(r"token=([A-Za-z0-9_\-]+)", self.sent[-1]["body"])
        assert match is not None
        return match.group(1)


class FakeClock:
    """A movable clock for deterministic window arithmetic."""

    def __init__(self, start: datetime) -> None:
        """Start the clock at ``start``."""
        self.now = start

    def __call__(self) -> datetime:
        """Return the current fake time."""
        return self.now


_T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

#: What the ``setup`` fixture yields.
Setup = tuple[TestClient, RecordingMailer, AuthService, QuotaEnforcer, InMemoryUsageLedger]


@pytest.fixture
def setup(monkeypatch: pytest.MonkeyPatch) -> Iterator[Setup]:
    """A TestClient with in-memory auth service, key store, and quota enforcer."""
    monkeypatch.setenv("MERCURY_SESSION_COOKIE_SECURE", "false")
    monkeypatch.setattr(auth, "_api_key_store", APIKeyStore())
    mailer = RecordingMailer()
    service = AuthService(InMemoryIdentityStore(), mailer)
    ledger = InMemoryUsageLedger()
    enforcer = QuotaEnforcer(
        ledger,
        QuotaConfig(window_seconds=3600, max_requests=100, max_compute_ms=60_000.0),
        clock=FakeClock(_T0),
        tiers={
            "free": QuotaConfig(window_seconds=3600, max_requests=100, max_compute_ms=60_000.0),
            "supporter": QuotaConfig(
                window_seconds=3600, max_requests=5000, max_compute_ms=3_600_000.0
            ),
        },
        overrides=InMemoryQuotaOverrideStore(),
    )
    app = FastAPI()
    app.include_router(accounts.router)
    app.dependency_overrides[accounts.get_auth_service] = lambda: service
    app.dependency_overrides[get_shared_quota_enforcer] = lambda: enforcer
    with TestClient(app) as client:
        yield client, mailer, service, enforcer, ledger


def _register_verify_login(client: TestClient, mailer: RecordingMailer, email: str) -> str:
    """Register, verify, and log in through the HTTP routes; return the account id."""
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
    assert (
        client.post(
            "/api/v1/auth/login", json={"email": email, "password": "a-strong-pw"}
        ).status_code
        == 200
    )
    return str(client.get("/api/v1/auth/me").json()["id"])


def _set_tier(service: AuthService, account_id: str, tier: str) -> None:
    """Move an account onto ``tier`` directly in the identity store."""
    account = service.get_account(account_id)
    assert account is not None
    account.tier = tier
    service._store.update_account(account)


def test_usage_requires_authentication(setup: Setup) -> None:
    """An anonymous caller gets 401, not an empty usage report."""
    client, _, _, _, _ = setup
    assert client.get("/api/v1/auth/usage").status_code == 401


def test_usage_for_session_caller(setup: Setup) -> None:
    """A logged-in browser session reads its own usage and effective limits."""
    client, mailer, _service, _, ledger = setup
    account_id = _register_verify_login(client, mailer, "u@b.com")
    ledger.record(UsageEvent(account_id, _T0 - timedelta(seconds=10), "/api/v1/detect", 250.0))
    ledger.record(UsageEvent(account_id, _T0 - timedelta(seconds=5), "/api/v1/detect", 100.0))

    resp = client.get("/api/v1/auth/usage")
    assert resp.status_code == 200
    body = resp.json()
    assert body["tier"] == "free"
    assert body["window_seconds"] == 3600
    assert body["requests_used"] == 2
    assert body["compute_ms_used"] == pytest.approx(350.0)
    assert body["requests_limit"] == 100
    assert body["compute_ms_limit"] == pytest.approx(60_000.0)
    window_start = datetime.fromisoformat(body["window_start"])
    window_end = datetime.fromisoformat(body["window_end"])
    assert window_end - window_start == timedelta(seconds=3600)
    assert window_end == _T0


def test_usage_for_api_key_caller_reads_owning_account(setup: Setup) -> None:
    """An ``X-API-Key`` caller reads the owning account's usage at its tier."""
    client, mailer, service, _, ledger = setup
    account_id = _register_verify_login(client, mailer, "u@b.com")
    _set_tier(service, account_id, "supporter")

    raw_key, _ = auth.get_api_key_store().create_key(
        name="ci", user_id=account_id, permissions={Permission.READ}
    )
    ledger.record(UsageEvent(account_id, _T0 - timedelta(seconds=1), "/api/v1/detect", 42.0))

    fresh = TestClient(client.app)  # no session cookie: the key alone must authenticate
    resp = fresh.get("/api/v1/auth/usage", headers={"X-API-Key": raw_key})
    assert resp.status_code == 200
    body = resp.json()
    assert body["tier"] == "supporter"
    assert body["requests_used"] == 1
    assert body["requests_limit"] == 5000
    assert body["compute_ms_used"] == pytest.approx(42.0)


def test_usage_rejects_bad_api_key(setup: Setup) -> None:
    """An unknown or revoked key is 401 — same as no credentials at all."""
    client, _, _, _, _ = setup
    assert (
        client.get("/api/v1/auth/usage", headers={"X-API-Key": "not-a-real-key"}).status_code == 401
    )


def test_usage_override_beats_tier(setup: Setup) -> None:
    """A per-account override wins over the tier config in the reported limits."""
    client, mailer, service, enforcer, _ = setup
    account_id = _register_verify_login(client, mailer, "u@b.com")
    _set_tier(service, account_id, "supporter")

    enforcer.override_store.set_override(
        account_id,
        QuotaConfig(window_seconds=60, max_requests=7, max_compute_ms=99.0),
    )
    resp = client.get("/api/v1/auth/usage")
    assert resp.status_code == 200
    body = resp.json()
    assert body["tier"] == "supporter"  # the account's tier name still reports
    assert body["requests_limit"] == 7  # ...but the override's ceilings win
    assert body["compute_ms_limit"] == pytest.approx(99.0)
    assert body["window_seconds"] == 60
