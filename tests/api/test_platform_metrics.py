# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the platform Prometheus counters.

Covers registrations, logins, throttles, quotas, mail, and maintenance.
Each test drives the real event path (HTTP route, middleware, service, or
sweep) and asserts the counter moved in the ``/metrics`` exposition — deltas,
not absolutes, because the prometheus default registry is process-wide. A
final test proves the module stays importable and inert with
``prometheus_client`` absent (the core-lane contract).
"""

from __future__ import annotations

import builtins
import importlib
import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omni_mercury_engine.api import platform_metrics, totp
from omni_mercury_engine.api.auth_service import AuthService
from omni_mercury_engine.api.identity_store import InMemoryIdentityStore
from omni_mercury_engine.api.quota import QuotaConfig, QuotaEnforcer
from omni_mercury_engine.api.routes import accounts
from omni_mercury_engine.api.usage_ledger import InMemoryUsageLedger

if TYPE_CHECKING:
    from collections.abc import Iterator

prometheus_client = pytest.importorskip(
    "prometheus_client", reason="scrape assertions need the real registry"
)


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
        match = re.search(r"token=([A-Za-z0-9_\-]+)", self.sent[-1]["body"])
        assert match is not None
        return match.group(1)


class FailingMailer:
    """Mailer whose every send raises (exercises the failure counter)."""

    def send(
        self,
        *,
        to: str,
        subject: str,
        body: str,
        html_body: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Simulate an SMTP outage."""
        raise RuntimeError("smtp down")


class FakeClock:
    """A movable clock so TOTP codes are deterministic."""

    def __init__(self, start: datetime) -> None:
        """Start the clock at ``start``."""
        self.now = start

    def __call__(self) -> datetime:
        """Return the current fake time."""
        return self.now

    def advance(self, delta: timedelta) -> None:
        """Move the clock forward by ``delta``."""
        self.now += delta


def _metric_value(name: str, labels: dict[str, str] | None = None) -> float:
    """Read a counter's current value straight from the default registry."""
    from prometheus_client import REGISTRY

    value = REGISTRY.get_sample_value(name, labels or {})
    return float(value) if value is not None else 0.0


#: What the ``client_setup`` fixture yields.
ClientSetup = tuple[TestClient, RecordingMailer, AuthService, FakeClock]


@pytest.fixture
def client_setup(monkeypatch: pytest.MonkeyPatch) -> Iterator[ClientSetup]:
    """A TestClient over the accounts router with an in-memory service."""
    monkeypatch.setenv("MERCURY_SESSION_COOKIE_SECURE", "false")
    mailer = RecordingMailer()
    clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
    service = AuthService(InMemoryIdentityStore(), mailer, clock=clock)
    app = FastAPI()
    app.include_router(accounts.router)
    app.dependency_overrides[accounts.get_auth_service] = lambda: service
    with TestClient(app) as client:
        yield client, mailer, service, clock


def _register_verify(client: TestClient, mailer: RecordingMailer, email: str) -> None:
    """Register and verify an account through the HTTP routes."""
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


def test_registration_and_login_counters(client_setup: ClientSetup) -> None:
    """Register + login-ok + login-fail each move their counter (and email=sent)."""
    client, mailer, _, _ = client_setup
    reg_before = _metric_value("mercury_platform_registrations_total")
    ok_before = _metric_value("mercury_platform_logins_total", {"outcome": "ok"})
    fail_before = _metric_value("mercury_platform_logins_total", {"outcome": "fail"})
    sent_before = _metric_value("mercury_platform_emails_total", {"outcome": "sent"})

    _register_verify(client, mailer, "m@b.com")
    assert (
        client.post(
            "/api/v1/auth/login", json={"email": "m@b.com", "password": "a-strong-pw"}
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/v1/auth/login", json={"email": "m@b.com", "password": "wrong-pw!!"}
        ).status_code
        == 401
    )

    assert _metric_value("mercury_platform_registrations_total") == reg_before + 1
    assert _metric_value("mercury_platform_logins_total", {"outcome": "ok"}) == ok_before + 1
    assert _metric_value("mercury_platform_logins_total", {"outcome": "fail"}) == fail_before + 1
    # The verification email went through the recording mailer successfully.
    assert _metric_value("mercury_platform_emails_total", {"outcome": "sent"}) >= sent_before + 1


def test_2fa_challenge_counter(client_setup: ClientSetup) -> None:
    """A password-only login against a 2FA account counts as 2fa_challenged."""
    client, mailer, _, clock = client_setup
    challenged_before = _metric_value(
        "mercury_platform_logins_total", {"outcome": "2fa_challenged"}
    )

    _register_verify(client, mailer, "m2@b.com")
    login = client.post("/api/v1/auth/login", json={"email": "m2@b.com", "password": "a-strong-pw"})
    csrf = str(login.json()["csrf_token"])
    enroll = client.post("/api/v1/auth/2fa/enroll", headers={"X-CSRF-Token": csrf})
    assert enroll.status_code == 200
    secret = str(enroll.json()["secret"])
    code = totp.generate_totp(secret, at=clock.now.timestamp())
    assert (
        client.post(
            "/api/v1/auth/2fa/confirm", json={"code": code}, headers={"X-CSRF-Token": csrf}
        ).status_code
        == 200
    )

    challenge = client.post(
        "/api/v1/auth/login", json={"email": "m2@b.com", "password": "a-strong-pw"}
    )
    assert challenge.status_code == 401
    assert challenge.json()["detail"]["code"] == "two_factor_required"
    assert (
        _metric_value("mercury_platform_logins_total", {"outcome": "2fa_challenged"})
        == challenged_before + 1
    )


def test_throttle_denial_counter(
    client_setup: ClientSetup, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Breaching a per-action throttle moves the denial counter for that action."""
    client, _, _, _ = client_setup
    monkeypatch.setenv("MERCURY_AUTH_RATE_REGISTER_IP", "1/3600")
    denials_before = _metric_value(
        "mercury_platform_throttle_denials_total", {"action": "register_ip"}
    )
    client.post("/api/v1/auth/register", json={"email": "t1@b.com", "password": "a-strong-pw"})
    denied = client.post(
        "/api/v1/auth/register", json={"email": "t2@b.com", "password": "a-strong-pw"}
    )
    assert denied.status_code == 429
    assert (
        _metric_value("mercury_platform_throttle_denials_total", {"action": "register_ip"})
        == denials_before + 1
    )


def test_quota_denial_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    """A quota 429 on a metered route moves the quota-denial counter."""
    from omni_mercury_engine.api.quota_middleware import QuotaMiddleware

    monkeypatch.setenv("MERCURY_QUOTA_ENABLED", "true")
    enforcer = QuotaEnforcer(
        InMemoryUsageLedger(),
        QuotaConfig(window_seconds=3600, max_requests=1, max_compute_ms=1e12),
    )
    app = FastAPI()

    @app.post("/api/v1/detect/probe")
    def probe() -> dict[str, bool]:
        return {"ok": True}

    app.add_middleware(QuotaMiddleware, enforcer=enforcer)
    denials_before = _metric_value("mercury_platform_quota_denials_total", {"reason": "requests"})
    with TestClient(app) as client:
        assert client.post("/api/v1/detect/probe").status_code == 200
        assert client.post("/api/v1/detect/probe").status_code == 429
    assert (
        _metric_value("mercury_platform_quota_denials_total", {"reason": "requests"})
        == denials_before + 1
    )


def test_mailer_failure_counter() -> None:
    """A mailer exception is swallowed by the flow but counted as failed."""
    service = AuthService(InMemoryIdentityStore(), FailingMailer())
    failed_before = _metric_value("mercury_platform_emails_total", {"outcome": "failed"})
    service.register("f@b.com", "a-strong-pw")  # must not raise despite the mailer
    assert (
        _metric_value("mercury_platform_emails_total", {"outcome": "failed"}) == failed_before + 1
    )


def test_maintenance_sweep_counters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sweep results feed the pruned counter; -1 steps feed the error counter."""
    from omni_mercury_engine.api.maintenance import run_maintenance_sweep

    pruned_before = _metric_value(
        "mercury_platform_maintenance_pruned_total", {"step": "usage_events"}
    )
    errors_before = _metric_value(
        "mercury_platform_maintenance_errors_total", {"step": "audit_segments"}
    )
    platform_metrics.record_maintenance_sweep({"usage_events": 3, "audit_segments": -1})
    assert (
        _metric_value("mercury_platform_maintenance_pruned_total", {"step": "usage_events"})
        == pruned_before + 3
    )
    assert (
        _metric_value("mercury_platform_maintenance_errors_total", {"step": "audit_segments"})
        == errors_before + 1
    )

    # The real sweep (in-memory backends; env unset) records without error and
    # materialises the per-step series.
    monkeypatch.delenv("MERCURY_KEYSTORE_PATH", raising=False)
    monkeypatch.delenv("MERCURY_AUDIT_LOG_DIR", raising=False)
    results = run_maintenance_sweep()
    assert "expired_sessions" in results


def test_metrics_surface_in_exposition(client_setup: ClientSetup) -> None:
    """The platform counters appear on the /metrics scrape target."""
    from omni_mercury_engine.api.health import health_metrics

    client, mailer, _, _ = client_setup
    _register_verify(client, mailer, "s@b.com")
    app = client.app
    assert isinstance(app, FastAPI)
    app.add_api_route("/metrics", health_metrics, methods=["GET"])
    scrape = client.get("/metrics")
    assert scrape.status_code == 200
    assert "mercury_platform_registrations_total" in scrape.text


def test_module_is_inert_without_prometheus(monkeypatch: pytest.MonkeyPatch) -> None:
    """With prometheus_client absent the module imports and no-ops (core lane)."""
    real_import = builtins.__import__
    # Snapshot the module state; restoring it (rather than reloading a second
    # time) keeps the already-registered collectors and their cache intact —
    # a fresh reload would try to re-register them and the default registry
    # rejects duplicates.
    saved_state = dict(platform_metrics.__dict__)

    def _blocked(name: str, *args: object, **kwargs: object) -> object:
        if name == "prometheus_client":
            raise ImportError("blocked for the core-lane simulation")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _blocked)
    try:
        reloaded = importlib.reload(platform_metrics)
        assert reloaded.PROMETHEUS_AVAILABLE is False
        # Every recorder is a silent no-op.
        reloaded.record_registration()
        reloaded.record_login("ok")
        reloaded.record_throttle_denial("login_ip")
        reloaded.record_throttle_config_mismatch("ghost_action")
        reloaded.record_quota_denial("requests")
        reloaded.record_email("sent")
        reloaded.record_maintenance_sweep({"usage_events": 1, "broken": -1})
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)
        platform_metrics.__dict__.update(saved_state)
    assert platform_metrics.PROMETHEUS_AVAILABLE is True
