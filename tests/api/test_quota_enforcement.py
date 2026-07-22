# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for quota enforcement: tiers, overrides, atomic reservation, wiring.

Pins the properties a fair free service needs:

* **Hard request ceiling** — ``reserve`` counts and inserts atomically, so
  concurrent requests against a nearly full window admit exactly the number of
  free slots (no overrun). The compute ceiling is soft-by-one and documented.
* **Policy precedence** — per-account override beats tier beats default.
* **HTTP wiring** — the metered middleware returns 429 + ``Retry-After`` once
  the ceiling is hit and back-fills measured compute onto the reserved row.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from omni_mercury_engine.api.quota import (
    InMemoryQuotaOverrideStore,
    QuotaConfig,
    QuotaEnforcer,
    SqliteQuotaOverrideStore,
)
from omni_mercury_engine.api.usage_ledger import (
    InMemoryUsageLedger,
    SqliteUsageLedger,
    UsageEvent,
)

if TYPE_CHECKING:
    from pathlib import Path


class FixedClock:
    """A clock pinned to one instant."""

    def __init__(self) -> None:
        """Pin the time."""
        self.now = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        """Return the fixed time."""
        return self.now


class TestAtomicReservation:
    """The request ceiling is hard under concurrency."""

    @pytest.fixture(params=["memory", "sqlite"])
    def ledger(self, request: pytest.FixtureRequest, tmp_path: Path):
        """Both ledger backends."""
        if request.param == "memory":
            yield InMemoryUsageLedger()
        else:
            ledger = SqliteUsageLedger(tmp_path / "usage.db")
            yield ledger
            ledger.close()

    def test_concurrent_reserve_never_overruns(self, ledger) -> None:
        """40 threads racing a 10-request window admit exactly 10."""
        config = QuotaConfig(window_seconds=3600, max_requests=10, max_compute_ms=1e12)
        enforcer = QuotaEnforcer(ledger, config, clock=FixedClock())
        allowed: list[bool] = []
        lock = threading.Lock()

        def worker() -> None:
            decision = enforcer.reserve("acct", "/detect")
            with lock:
                allowed.append(decision.allowed)

        threads = [threading.Thread(target=worker) for _ in range(40)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sum(allowed) == 10

    def test_reserve_then_commit_records_compute(self, ledger) -> None:
        """Commit back-fills the measured cost onto the reserved row."""
        clock = FixedClock()
        config = QuotaConfig(window_seconds=3600, max_requests=100, max_compute_ms=1e9)
        enforcer = QuotaEnforcer(ledger, config, clock=clock)
        decision = enforcer.reserve("acct", "/detect")
        assert decision.allowed and decision.event_id is not None
        enforcer.commit(decision.event_id, 250.0)
        since = clock.now - timedelta(seconds=3600)
        summary = ledger.summary_since("acct", since)
        assert summary.request_count == 1
        assert summary.compute_ms == 250.0

    def test_compute_ceiling_denies(self, ledger) -> None:
        """Once the compute budget is spent, reservation is denied."""
        clock = FixedClock()
        config = QuotaConfig(window_seconds=3600, max_requests=1000, max_compute_ms=500.0)
        enforcer = QuotaEnforcer(ledger, config, clock=clock)
        first = enforcer.reserve("acct", "/detect")
        enforcer.commit(first.event_id, 600.0)  # blow the compute budget
        denied = enforcer.reserve("acct", "/detect")
        assert denied.allowed is False
        assert "compute" in (denied.reason or "")
        assert denied.retry_after_seconds == 3600


class TestPolicyPrecedence:
    """override > tier > default resolution."""

    def _enforcer(self, overrides=None):
        default = QuotaConfig(window_seconds=3600, max_requests=100, max_compute_ms=1e9)
        tiers = {
            "free": default,
            "supporter": QuotaConfig(window_seconds=3600, max_requests=5000, max_compute_ms=1e10),
        }
        return QuotaEnforcer(
            InMemoryUsageLedger(),
            default,
            clock=FixedClock(),
            tiers=tiers,
            overrides=overrides or InMemoryQuotaOverrideStore(),
        )

    def test_tier_lookup(self) -> None:
        """A tiered account gets the tier's ceilings."""
        enforcer = self._enforcer()
        assert enforcer.config_for("acct", "supporter").max_requests == 5000
        assert enforcer.config_for("acct", "free").max_requests == 100

    def test_unknown_tier_falls_back_to_default(self) -> None:
        """An unrecognised tier name uses the default (fail-closed)."""
        enforcer = self._enforcer()
        assert enforcer.config_for("acct", "platinum").max_requests == 100

    def test_override_beats_tier(self) -> None:
        """A per-account override wins over the tier."""
        overrides = InMemoryQuotaOverrideStore()
        enforcer = self._enforcer(overrides)
        overrides.set_override(
            "vip", QuotaConfig(window_seconds=60, max_requests=999999, max_compute_ms=1e12)
        )
        assert enforcer.config_for("vip", "free").max_requests == 999999
        # Clearing it restores tier/default resolution.
        overrides.set_override("vip", None)
        assert enforcer.config_for("vip", "free").max_requests == 100

    def test_env_tiers_parsed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MERCURY_QUOTA_TIER_* definitions load; malformed ones are skipped."""
        from omni_mercury_engine.api.quota import build_quota_enforcer

        monkeypatch.delenv("MERCURY_KEYSTORE_PATH", raising=False)
        monkeypatch.setenv("MERCURY_QUOTA_TIER_SUPPORTER", "5000,3600000,3600")
        monkeypatch.setenv("MERCURY_QUOTA_TIER_BROKEN", "not,a,number")
        enforcer = build_quota_enforcer()
        assert enforcer.config_for("a", "supporter").max_requests == 5000
        assert (
            enforcer.config_for("a", "broken").max_requests
            == enforcer.config_for("a", "free").max_requests
        )


class TestSqliteOverrideStore:
    """The durable override store is shared and persistent."""

    def test_persist_and_clear(self, tmp_path: Path) -> None:
        """Overrides survive reopen and clear cleanly."""
        path = tmp_path / "ov.db"
        store = SqliteQuotaOverrideStore(path)
        store.set_override(
            "acct", QuotaConfig(window_seconds=60, max_requests=42, max_compute_ms=1.0)
        )
        store.close()
        reopened = SqliteQuotaOverrideStore(path)
        got = reopened.get_override("acct")
        assert got is not None and got.max_requests == 42
        reopened.set_override("acct", None)
        assert reopened.get_override("acct") is None
        reopened.close()


class TestCheckReadOnly:
    """The read-only check path (for callers that only report)."""

    def test_check_does_not_consume(self) -> None:
        """check() reports the state without charging a slot."""
        ledger = InMemoryUsageLedger()
        clock = FixedClock()
        enforcer = QuotaEnforcer(
            ledger,
            QuotaConfig(window_seconds=3600, max_requests=1, max_compute_ms=1e9),
            clock=clock,
        )
        assert enforcer.check("acct").allowed is True
        assert enforcer.check("acct").allowed is True  # still allowed, nothing spent
        ledger.record(UsageEvent("acct", clock.now, "/detect", 0.0))
        assert enforcer.check("acct").allowed is False


class TestQuotaMiddleware:
    """End-to-end HTTP wiring over a metered route."""

    def _app(self, enforcer: QuotaEnforcer):
        from fastapi import FastAPI

        from omni_mercury_engine.api.quota_middleware import QuotaMiddleware

        app = FastAPI()

        @app.post("/api/v1/detect/probe")
        def probe() -> dict[str, bool]:
            return {"ok": True}

        @app.get("/health")
        def health() -> dict[str, bool]:
            return {"ok": True}

        app.add_middleware(QuotaMiddleware, enforcer=enforcer)
        return app

    def test_429_after_ceiling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A metered route 429s with Retry-After once the ceiling is hit."""
        from fastapi.testclient import TestClient

        monkeypatch.setenv("MERCURY_QUOTA_ENABLED", "true")
        enforcer = QuotaEnforcer(
            InMemoryUsageLedger(),
            QuotaConfig(window_seconds=3600, max_requests=2, max_compute_ms=1e12),
            clock=FixedClock(),
        )
        app = self._app(enforcer)
        with TestClient(app) as client:
            assert client.post("/api/v1/detect/probe").status_code == 200
            assert client.post("/api/v1/detect/probe").status_code == 200
            blocked = client.post("/api/v1/detect/probe")
        assert blocked.status_code == 429
        assert int(blocked.headers["Retry-After"]) > 0
        assert "X-Quota-Requests-Used" in blocked.headers

    def test_unmetered_paths_untouched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-metered paths (health) never consume quota."""
        from fastapi.testclient import TestClient

        monkeypatch.setenv("MERCURY_QUOTA_ENABLED", "true")
        enforcer = QuotaEnforcer(
            InMemoryUsageLedger(),
            QuotaConfig(window_seconds=3600, max_requests=1, max_compute_ms=1e12),
            clock=FixedClock(),
        )
        app = self._app(enforcer)
        with TestClient(app) as client:
            for _ in range(5):
                assert client.get("/health").status_code == 200

    def test_disabled_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without MERCURY_QUOTA_ENABLED the middleware is inert."""
        from fastapi.testclient import TestClient

        monkeypatch.delenv("MERCURY_QUOTA_ENABLED", raising=False)
        enforcer = QuotaEnforcer(
            InMemoryUsageLedger(),
            QuotaConfig(window_seconds=3600, max_requests=1, max_compute_ms=1e12),
            clock=FixedClock(),
        )
        app = self._app(enforcer)
        with TestClient(app) as client:
            for _ in range(5):
                assert client.post("/api/v1/detect/probe").status_code == 200
