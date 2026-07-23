# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Route-level tests for the per-action auth throttles.

Each sensitive endpoint (login, register, password-reset request,
resend-verification) must 429 — with a ``Retry-After`` header — once its
action budget is spent, per client IP and per targeted account, and failed
attempts must count (an attacker cannot probe for free). Limits are injected
tight so tests stay fast; production values are configuration.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omni_mercury_engine.api.auth_service import AuthService
from omni_mercury_engine.api.identity_store import InMemoryIdentityStore
from omni_mercury_engine.api.rate_limit_store import (
    ActionRateLimiter,
    ActionRule,
    InMemoryCounterStore,
)
from omni_mercury_engine.api.routes import accounts

if TYPE_CHECKING:
    from collections.abc import Iterator


class NullMailer:
    """Discards messages (delivery is irrelevant to throttle behaviour)."""

    def send(self, **_kwargs: object) -> None:
        """Discard the message."""


def _tight_limiter(**rules: tuple[int, int]) -> ActionRateLimiter:
    """Build an in-memory limiter with the given (max, window) per action."""
    return ActionRateLimiter(
        InMemoryCounterStore(),
        rules={
            name: ActionRule(max_attempts=mx, window_seconds=win)
            for name, (mx, win) in rules.items()
        },
    )


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A TestClient over an in-memory service with default (loose) limits."""
    monkeypatch.setenv("MERCURY_SESSION_COOKIE_SECURE", "false")
    service = AuthService(
        InMemoryIdentityStore(),
        NullMailer(),
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )
    app = FastAPI()
    app.include_router(accounts.router)
    app.dependency_overrides[accounts.get_auth_service] = lambda: service
    with TestClient(app) as client:
        yield client


def _install_limiter(limiter: ActionRateLimiter) -> None:
    """Install a specific limiter as the route-layer singleton."""
    accounts._action_limiter = limiter


class TestLoginThrottles:
    """Online password guessing is bounded per IP and per account."""

    def test_login_ip_limit_yields_429_with_retry_after(self, app_client: TestClient) -> None:
        """The (N+1)-th login attempt from one IP is 429 + Retry-After."""
        _install_limiter(_tight_limiter(login_ip=(3, 300)))
        for _ in range(3):
            resp = app_client.post(
                "/api/v1/auth/login", json={"email": "a@b.com", "password": "wrong-pw-1"}
            )
            assert resp.status_code == 401  # wrong creds, but not throttled yet
        blocked = app_client.post(
            "/api/v1/auth/login", json={"email": "a@b.com", "password": "wrong-pw-1"}
        )
        assert blocked.status_code == 429
        assert int(blocked.headers["Retry-After"]) > 0

    def test_login_account_limit_spans_source_ips(self, app_client: TestClient) -> None:
        """The per-account bucket trips even as (spoof-resolved) IPs vary.

        The TestClient peer is constant, so this exercises the account key:
        a distributed guesser rotating source addresses still exhausts the
        target account's budget.
        """
        _install_limiter(_tight_limiter(login_account=(2, 900)))
        target = {"email": "victim@b.com", "password": "wrong-pw-1"}
        assert app_client.post("/api/v1/auth/login", json=target).status_code == 401
        assert app_client.post("/api/v1/auth/login", json=target).status_code == 401
        assert app_client.post("/api/v1/auth/login", json=target).status_code == 429
        # A different account is unaffected.
        other = app_client.post(
            "/api/v1/auth/login", json={"email": "other@b.com", "password": "wrong-pw-1"}
        )
        assert other.status_code == 401

    def test_failed_attempts_count(self, app_client: TestClient) -> None:
        """Attempts count on entry: N failures then the right password is 429.

        Counting before the credential check is what stops an attacker from
        learning anything once throttled — and from probing free of charge.
        """
        _install_limiter(_tight_limiter(login_ip=(2, 300)))
        app_client.post("/api/v1/auth/login", json={"email": "x@b.com", "password": "wrong-1x"})
        app_client.post("/api/v1/auth/login", json={"email": "x@b.com", "password": "wrong-2x"})
        third = app_client.post(
            "/api/v1/auth/login", json={"email": "x@b.com", "password": "anything-now"}
        )
        assert third.status_code == 429


class TestSignupAndEmailThrottles:
    """Signup flooding and email bombing are bounded."""

    def test_register_ip_limit(self, app_client: TestClient) -> None:
        """Mass signup from one address trips the register budget."""
        _install_limiter(_tight_limiter(register_ip=(2, 3600)))
        for i in range(2):
            resp = app_client.post(
                "/api/v1/auth/register",
                json={"email": f"u{i}@b.com", "password": "a-strong-pw"},
            )
            assert resp.status_code == 201
        blocked = app_client.post(
            "/api/v1/auth/register", json={"email": "u9@b.com", "password": "a-strong-pw"}
        )
        assert blocked.status_code == 429

    def test_reset_request_limits_per_account(self, app_client: TestClient) -> None:
        """Reset-email bombing of one mailbox trips the per-account budget."""
        _install_limiter(_tight_limiter(reset_account=(2, 3600)))
        for _ in range(2):
            resp = app_client.post(
                "/api/v1/auth/password-reset/request", json={"email": "victim@b.com"}
            )
            assert resp.status_code == 202
        blocked = app_client.post(
            "/api/v1/auth/password-reset/request", json={"email": "victim@b.com"}
        )
        assert blocked.status_code == 429
        # A different mailbox still gets its 202.
        other = app_client.post(
            "/api/v1/auth/password-reset/request", json={"email": "other@b.com"}
        )
        assert other.status_code == 202

    def test_resend_verification_limits(self, app_client: TestClient) -> None:
        """Verification-email resends are bounded per account."""
        _install_limiter(_tight_limiter(resend_account=(1, 3600)))
        first = app_client.post("/api/v1/auth/resend-verification", json={"email": "victim@b.com"})
        assert first.status_code == 202
        blocked = app_client.post(
            "/api/v1/auth/resend-verification", json={"email": "victim@b.com"}
        )
        assert blocked.status_code == 429

    def test_throttle_response_reveals_nothing(self, app_client: TestClient) -> None:
        """The 429 body is identical for existing and ghost accounts."""
        _install_limiter(_tight_limiter(reset_account=(1, 3600)))
        app_client.post("/api/v1/auth/password-reset/request", json={"email": "ghost@b.com"})
        blocked_ghost = app_client.post(
            "/api/v1/auth/password-reset/request", json={"email": "ghost@b.com"}
        )
        _install_limiter(_tight_limiter(reset_account=(1, 3600)))
        app_client.post(
            "/api/v1/auth/register", json={"email": "real@b.com", "password": "a-strong-pw"}
        )
        app_client.post("/api/v1/auth/password-reset/request", json={"email": "real@b.com"})
        blocked_real = app_client.post(
            "/api/v1/auth/password-reset/request", json={"email": "real@b.com"}
        )
        assert blocked_ghost.status_code == blocked_real.status_code == 429
        assert blocked_ghost.json() == blocked_real.json()
