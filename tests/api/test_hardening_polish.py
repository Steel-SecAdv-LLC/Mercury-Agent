# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the hardening-polish invariants.

Covers the dispatched-throttle-action ↔ rule-table invariant (static test +
boot-time cross-check), the ``MERCURY_QUOTA_FAIL_CLOSED`` deny-on-quota-outage
option, and the platform contact default.
"""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omni_mercury_engine.api.auth_service import AuthService, build_auth_service
from omni_mercury_engine.api.quota_middleware import QuotaMiddleware
from omni_mercury_engine.api.rate_limit_store import DEFAULT_ACTION_RULES
from omni_mercury_engine.api.routes import accounts

if TYPE_CHECKING:
    from omni_mercury_engine.api.quota import QuotaDecision, QuotaEnforcer


class RaisingEnforcer:
    """An enforcer whose reserve always fails (simulated quota-store outage)."""

    def reserve(self, account_id: str, endpoint: str, tier: str = "free") -> QuotaDecision:
        """Blow up like a broken SQLite file would."""
        raise RuntimeError("quota store unavailable")


def _metered_app() -> FastAPI:
    """A minimal app with one metered route behind the quota middleware."""
    app = FastAPI()

    @app.post("/api/v1/detect/probe")
    def probe() -> dict[str, bool]:
        return {"ok": True}

    # RaisingEnforcer only implements the surface dispatch() touches before
    # the failure; the cast keeps the middleware signature honest.
    app.add_middleware(QuotaMiddleware, enforcer=cast("QuotaEnforcer", RaisingEnforcer()))
    return app


# --------------------------------------------------------------------------- #
# throttle-registration boot invariant
# --------------------------------------------------------------------------- #
def test_dispatched_actions_match_default_rules() -> None:
    """Every dispatched action has a default rule, and vice versa.

    A dispatched action without a rule would silently allow-on-unknown; a
    rule nothing dispatches is dead configuration. Both directions must stay
    in lockstep when handlers gain or lose throttle checks.
    """
    assert set(accounts.DISPATCHED_THROTTLE_ACTIONS) == set(DEFAULT_ACTION_RULES)


def test_limiter_build_flags_unknown_dispatched_action(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A dispatched action missing from the rules logs an error, never crashes."""
    monkeypatch.setattr(
        accounts,
        "DISPATCHED_THROTTLE_ACTIONS",
        (*accounts.DISPATCHED_THROTTLE_ACTIONS, "ghost_action"),
    )
    accounts._action_limiter = None
    with caplog.at_level(logging.ERROR, logger="omni_mercury_engine.api.routes.accounts"):
        limiter = accounts.get_action_limiter()
    assert limiter is not None  # the server still builds and serves
    assert any("ghost_action" in record.message for record in caplog.records)
    # A clean table logs nothing.
    monkeypatch.setattr(accounts, "DISPATCHED_THROTTLE_ACTIONS", tuple(DEFAULT_ACTION_RULES))
    accounts._action_limiter = None
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger="omni_mercury_engine.api.routes.accounts"):
        accounts.get_action_limiter()
    assert not caplog.records


def test_limiter_build_mismatch_moves_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    """The boot-time mismatch also feeds the platform metric (when available)."""
    prometheus_client = pytest.importorskip("prometheus_client")
    monkeypatch.setattr(
        accounts,
        "DISPATCHED_THROTTLE_ACTIONS",
        (*accounts.DISPATCHED_THROTTLE_ACTIONS, "ghost_metric_action"),
    )
    accounts._action_limiter = None

    def _value() -> float:
        value = prometheus_client.REGISTRY.get_sample_value(
            "mercury_platform_throttle_rule_mismatch_total",
            {"action": "ghost_metric_action"},
        )
        return float(value) if value is not None else 0.0

    before = _value()
    accounts.get_action_limiter()
    assert _value() == before + 1


# --------------------------------------------------------------------------- #
# MERCURY_QUOTA_FAIL_CLOSED
# --------------------------------------------------------------------------- #
def test_quota_infrastructure_failure_admits_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default posture: a broken quota store admits the request unmetered."""
    monkeypatch.setenv("MERCURY_QUOTA_ENABLED", "true")
    monkeypatch.delenv("MERCURY_QUOTA_FAIL_CLOSED", raising=False)
    with TestClient(_metered_app()) as client:
        assert client.post("/api/v1/detect/probe").status_code == 200


def test_quota_infrastructure_failure_denies_when_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MERCURY_QUOTA_FAIL_CLOSED=true turns the same outage into a 503."""
    monkeypatch.setenv("MERCURY_QUOTA_ENABLED", "true")
    monkeypatch.setenv("MERCURY_QUOTA_FAIL_CLOSED", "true")
    with TestClient(_metered_app()) as client:
        denied = client.post("/api/v1/detect/probe")
    assert denied.status_code == 503
    assert denied.json()["error"] == "quota_unavailable"
    assert int(denied.headers["Retry-After"]) > 0


def test_fail_closed_does_not_touch_unmetered_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail-closed only applies inside the metered prefixes."""
    monkeypatch.setenv("MERCURY_QUOTA_ENABLED", "true")
    monkeypatch.setenv("MERCURY_QUOTA_FAIL_CLOSED", "true")
    app = _metered_app()

    @app.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200


# --------------------------------------------------------------------------- #
# contact default
# --------------------------------------------------------------------------- #
def test_contact_defaults_to_platform_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fallback contact is the platform mailbox, not a personal address."""
    default = inspect.signature(AuthService.__init__).parameters["contact"].default
    assert default == "contact@mercuryagent.global"

    monkeypatch.delenv("MERCURY_CONTACT_EMAIL", raising=False)
    monkeypatch.delenv("MERCURY_KEYSTORE_PATH", raising=False)
    monkeypatch.delenv("MERCURY_SMTP_HOST", raising=False)
    service = build_auth_service()
    assert service._contact == "contact@mercuryagent.global"

    monkeypatch.setenv("MERCURY_CONTACT_EMAIL", "ops@example.org")
    assert build_auth_service()._contact == "ops@example.org"
