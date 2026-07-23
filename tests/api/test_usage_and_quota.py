# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for per-account usage metering and quota enforcement (brick 4).

Covers the usage ledger (in-memory + durable SQLite parity, windowing, per-account
isolation, restart durability) and the quota engine (allow under budget, deny on
each ceiling, and rolling-window rolloff via an injected clock).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from omni_mercury_engine.api.quota import QuotaConfig, QuotaEnforcer
from omni_mercury_engine.api.usage_ledger import (
    InMemoryUsageLedger,
    SqliteUsageLedger,
    UsageEvent,
    UsageLedger,
    build_usage_ledger,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


class FakeClock:
    """A movable clock for deterministic window tests."""

    def __init__(self, start: datetime) -> None:
        """Start the clock at ``start``."""
        self.now = start

    def __call__(self) -> datetime:
        """Return the current fake time."""
        return self.now

    def advance(self, delta: timedelta) -> None:
        """Move the clock forward."""
        self.now += delta


# --------------------------------------------------------------------------- #
# usage ledger — parity across both backends
# --------------------------------------------------------------------------- #
@pytest.fixture(params=["memory", "sqlite"])
def ledger(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[UsageLedger]:
    """Yield each ledger backend so parity tests run against both."""
    if request.param == "memory":
        yield InMemoryUsageLedger()
    else:
        store = SqliteUsageLedger(tmp_path / "usage.db")
        yield store
        store.close()


def test_summary_counts_and_sums(ledger: UsageLedger) -> None:
    """summary_since counts requests and sums compute for the account."""
    ledger.record(UsageEvent("acct-1", _T0, "/detect", 10.0))
    ledger.record(UsageEvent("acct-1", _T0 + timedelta(seconds=1), "/detect", 5.5))
    summary = ledger.summary_since("acct-1", _T0 - timedelta(seconds=1))
    assert summary.request_count == 2
    assert summary.compute_ms == pytest.approx(15.5)


def test_summary_respects_window_and_account(ledger: UsageLedger) -> None:
    """Only events at/after `since` and for the given account are counted."""
    ledger.record(UsageEvent("acct-1", _T0 - timedelta(hours=2), "/detect", 100.0))  # too old
    ledger.record(UsageEvent("acct-1", _T0, "/detect", 10.0))  # in window
    ledger.record(UsageEvent("acct-2", _T0, "/detect", 99.0))  # other account

    summary = ledger.summary_since("acct-1", _T0 - timedelta(hours=1))
    assert summary.request_count == 1
    assert summary.compute_ms == pytest.approx(10.0)


def test_prune_before(ledger: UsageLedger) -> None:
    """prune_before removes only events older than the cutoff."""
    ledger.record(UsageEvent("acct-1", _T0 - timedelta(days=2), "/detect", 1.0))
    ledger.record(UsageEvent("acct-1", _T0, "/detect", 1.0))
    removed = ledger.prune_before(_T0 - timedelta(days=1))
    assert removed == 1
    assert ledger.summary_since("acct-1", _T0 - timedelta(days=10)).request_count == 1


def test_ledger_survives_reopen(tmp_path: Path) -> None:
    """Usage persists across a SQLite close/reopen (restart)."""
    db_path = tmp_path / "durable-usage.db"
    first = SqliteUsageLedger(db_path)
    first.record(UsageEvent("acct-1", _T0, "/detect", 42.0))
    first.close()

    second = SqliteUsageLedger(db_path)
    try:
        summary = second.summary_since("acct-1", _T0 - timedelta(seconds=1))
        assert summary.request_count == 1
        assert summary.compute_ms == pytest.approx(42.0)
    finally:
        second.close()


def test_build_usage_ledger_selects_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The env var selects the durable backend; unset is in-memory."""
    monkeypatch.delenv("MERCURY_KEYSTORE_PATH", raising=False)
    assert isinstance(build_usage_ledger(), InMemoryUsageLedger)
    monkeypatch.setenv("MERCURY_KEYSTORE_PATH", str(tmp_path / "u.db"))
    assert isinstance(build_usage_ledger(), SqliteUsageLedger)


# --------------------------------------------------------------------------- #
# quota enforcement
# --------------------------------------------------------------------------- #
@pytest.fixture
def enforcer_setup() -> tuple[QuotaEnforcer, FakeClock]:
    """A quota enforcer over an in-memory ledger with a small, fixed budget."""
    clock = FakeClock(_T0)
    config = QuotaConfig(window_seconds=3600, max_requests=3, max_compute_ms=100.0)
    return QuotaEnforcer(InMemoryUsageLedger(), config, clock=clock), clock


def test_allows_under_budget(enforcer_setup: tuple[QuotaEnforcer, FakeClock]) -> None:
    """A fresh account is allowed and sees zero prior usage."""
    enforcer, _ = enforcer_setup
    decision = enforcer.check("acct-1")
    assert decision.allowed is True
    assert decision.request_count == 0


def test_denies_on_request_ceiling(
    enforcer_setup: tuple[QuotaEnforcer, FakeClock],
) -> None:
    """Hitting the request count denies further work with a retry hint."""
    enforcer, _ = enforcer_setup
    for _ in range(3):  # max_requests == 3
        assert enforcer.check("acct-1").allowed is True
        enforcer.record("acct-1", "/detect", 1.0)
    denied = enforcer.check("acct-1")
    assert denied.allowed is False
    assert denied.reason == "request quota exceeded"
    assert denied.retry_after_seconds == 3600


def test_denies_on_compute_ceiling(
    enforcer_setup: tuple[QuotaEnforcer, FakeClock],
) -> None:
    """Exceeding the compute budget denies even under the request count."""
    enforcer, _ = enforcer_setup
    enforcer.record("acct-1", "/detect", 120.0)  # over max_compute_ms (100) in one shot
    denied = enforcer.check("acct-1")
    assert denied.allowed is False
    assert denied.reason == "compute quota exceeded"


def test_window_rolls_off(enforcer_setup: tuple[QuotaEnforcer, FakeClock]) -> None:
    """Usage outside the rolling window no longer counts once time advances."""
    enforcer, clock = enforcer_setup
    for _ in range(3):
        enforcer.record("acct-1", "/detect", 1.0)
    assert enforcer.check("acct-1").allowed is False

    clock.advance(timedelta(seconds=3601))  # push the old events out of the window
    fresh = enforcer.check("acct-1")
    assert fresh.allowed is True
    assert fresh.request_count == 0


def test_quota_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """QuotaConfig reads ceilings from the environment."""
    monkeypatch.setenv("MERCURY_QUOTA_WINDOW_SECONDS", "60")
    monkeypatch.setenv("MERCURY_QUOTA_MAX_REQUESTS", "5")
    monkeypatch.setenv("MERCURY_QUOTA_MAX_COMPUTE_MS", "250.5")
    config = QuotaConfig.from_env()
    assert config.window_seconds == 60
    assert config.max_requests == 5
    assert config.max_compute_ms == pytest.approx(250.5)
