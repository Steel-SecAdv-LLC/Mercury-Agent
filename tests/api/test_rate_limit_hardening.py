# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the shared rate-limit store and the middleware bypass fixes.

Three properties are pinned here, each mapped to a real attack:

* **Cross-worker sharing + restart persistence** — bucket and counter state in
  the shared SQLite file is visible from a second store instance (a second
  worker) and survives close/reopen (a restart), so neither multiplies an
  attacker's budget.
* **Atomicity** — concurrent consumers can never jointly overdraw a bucket or
  under-count a fixed window (the race that would let a burst of parallel
  login attempts slip past the limit).
* **Header-spoof immunity** — the server middleware keyed on rotating
  ``X-Forwarded-For`` values must keep hitting ONE bucket (regression test for
  the left-most-hop bypass).
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import pytest

from omni_mercury_engine.api.rate_limit_store import (
    ActionRateLimiter,
    ActionRule,
    InMemoryCounterStore,
    SqliteCounterStore,
    SqliteRateLimitBackend,
)
from omni_mercury_engine.security.rate_limiting import RateLimiter, RateLimitInfo

if TYPE_CHECKING:
    from pathlib import Path


class TestSqliteBucketBackend:
    """The shared token-bucket backend: atomic, shared, persistent."""

    def test_atomic_consume_never_overspends(self, tmp_path: Path) -> None:
        """32 threads racing 10 tokens spend exactly 10."""
        backend = SqliteRateLimitBackend(tmp_path / "rl.db")
        now = time.time()
        allowed: list[bool] = []
        lock = threading.Lock()

        def worker() -> None:
            ok, _ = backend.consume_token("k", refill_rate=0.0, burst=10, now=now)
            with lock:
                allowed.append(ok)

        threads = [threading.Thread(target=worker) for _ in range(32)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sum(allowed) == 10
        backend.close()

    def test_state_shared_across_instances(self, tmp_path: Path) -> None:
        """A second instance on the same file (a second worker) sees the spend."""
        path = tmp_path / "rl.db"
        first = SqliteRateLimitBackend(path)
        second = SqliteRateLimitBackend(path)
        now = time.time()
        for _ in range(5):
            first.consume_token("k", refill_rate=0.0, burst=5, now=now)
        ok, _ = second.consume_token("k", refill_rate=0.0, burst=5, now=now)
        assert ok is False  # the other "worker" already drained the bucket
        first.close()
        second.close()

    def test_state_survives_reopen(self, tmp_path: Path) -> None:
        """Bucket state persists across close/reopen (a restart)."""
        path = tmp_path / "rl.db"
        backend = SqliteRateLimitBackend(path)
        now = time.time()
        for _ in range(5):
            backend.consume_token("k", refill_rate=0.0, burst=5, now=now)
        backend.close()

        reopened = SqliteRateLimitBackend(path)
        ok, _ = reopened.consume_token("k", refill_rate=0.0, burst=5, now=now)
        assert ok is False  # a restart no longer refills the world
        reopened.close()

    def test_refill_over_time(self, tmp_path: Path) -> None:
        """Tokens refill at the configured rate after a quiet period."""
        backend = SqliteRateLimitBackend(tmp_path / "rl.db")
        now = time.time()
        for _ in range(5):
            backend.consume_token("k", refill_rate=1.0, burst=5, now=now)
        ok, _ = backend.consume_token("k", refill_rate=1.0, burst=5, now=now)
        assert ok is False
        ok, _ = backend.consume_token("k", refill_rate=1.0, burst=5, now=now + 3)
        assert ok is True  # 3 seconds * 1 token/s refilled
        backend.close()

    def test_unified_limiter_uses_atomic_path(self, tmp_path: Path) -> None:
        """RateLimiter over the SQLite backend routes through consume_token."""
        backend = SqliteRateLimitBackend(tmp_path / "rl.db")
        limiter = RateLimiter(requests_per_minute=60, burst_size=3, backend=backend)
        results = [limiter.check("client").allowed for _ in range(4)]
        assert results[:3] == [True, True, True]
        assert results[3] is False
        info = limiter.check("client")
        assert info.retry_after is not None
        backend.close()

    def test_prune_stale_buckets(self, tmp_path: Path) -> None:
        """Idle buckets are prunable (maintenance sweep hook)."""
        backend = SqliteRateLimitBackend(tmp_path / "rl.db")
        old = time.time() - 7200
        backend.consume_token("stale", refill_rate=0.0, burst=5, now=old)
        backend.consume_token("fresh", refill_rate=0.0, burst=5, now=time.time())
        assert backend.prune_stale(time.time() - 3600) == 1
        assert backend.get("stale") is None
        assert backend.get("fresh") is not None
        backend.close()


class TestDeliveredRate:
    """The limiter must deliver the *configured* rate, on every backend.

    Regression cover for the truncating in-memory refill: the fallback path
    used to add ``int(elapsed * refill_rate)`` tokens while unconditionally
    advancing ``last_time`` to ``now``, so every call that arrived before a
    whole token had accrued threw away the elapsed time that produced the
    fraction. Any client polling faster than ``rpm / 60`` was permanently
    starved after its opening burst, delivering ~2 req/min against a
    configured 100 — while the SQLite backend, whose ``consume_token``
    always refilled fractionally, delivered the full 100 from the same
    configuration. The two backends must be indistinguishable in rate.
    """

    #: 100 rpm, 20-token burst — the shipped ``RateLimiter`` defaults.
    _RPM = 100
    _BURST = 20

    @staticmethod
    def _drive(
        limiter: RateLimiter,
        monkeypatch: pytest.MonkeyPatch,
        *,
        offered_per_second: float,
        duration_s: float,
        start: float = 1_700_000_000.0,
    ) -> int:
        """Offer requests at a fixed rate over an injected clock; count grants.

        ``RateLimiter`` reads ``time.time()`` from its own module namespace,
        so patching it there drives the whole refill computation from a
        deterministic virtual clock — no sleeping, no wall-clock flake.
        """
        clock = {"now": start}
        monkeypatch.setattr(
            "omni_mercury_engine.security.rate_limiting.time.time",
            lambda: clock["now"],
        )
        step = 1.0 / offered_per_second
        granted = 0
        offered = int(duration_s * offered_per_second)
        for _ in range(offered):
            if limiter.check("client").allowed:
                granted += 1
            clock["now"] += step
        return granted

    @pytest.mark.parametrize("offered_per_second", [10.0, 5.0, 1.0])
    def test_in_memory_delivers_configured_rate(
        self, monkeypatch: pytest.MonkeyPatch, offered_per_second: float
    ) -> None:
        """However fast the client polls, it gets ~rpm requests per minute."""
        limiter = RateLimiter(requests_per_minute=self._RPM, burst_size=self._BURST)
        duration_s = 600.0
        granted = self._drive(
            limiter,
            monkeypatch,
            offered_per_second=offered_per_second,
            duration_s=duration_s,
        )
        # Ideal service is the offered load, capped by what the bucket can
        # release: the opening burst plus one token per refill period.
        offered = duration_s * offered_per_second
        expected = min(offered, self._BURST + self._RPM / 60.0 * duration_s)
        assert granted == pytest.approx(expected, abs=2.0), (
            f"offered {offered_per_second}/s for {duration_s}s: granted {granted}, "
            f"expected ~{expected:.0f} ({granted / (duration_s / 60):.1f}/min "
            f"vs configured {self._RPM}/min)"
        )

    @pytest.mark.parametrize("offered_per_second", [10.0, 5.0, 1.0])
    def test_backends_agree_on_delivered_rate(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        offered_per_second: float,
    ) -> None:
        """In-memory and SQLite grant the same count from the same config."""
        duration_s = 600.0
        in_memory = RateLimiter(requests_per_minute=self._RPM, burst_size=self._BURST)
        memory_granted = self._drive(
            in_memory,
            monkeypatch,
            offered_per_second=offered_per_second,
            duration_s=duration_s,
        )

        backend = SqliteRateLimitBackend(tmp_path / "rl.db")
        try:
            sqlite_limiter = RateLimiter(
                requests_per_minute=self._RPM, burst_size=self._BURST, backend=backend
            )
            sqlite_granted = self._drive(
                sqlite_limiter,
                monkeypatch,
                offered_per_second=offered_per_second,
                duration_s=duration_s,
            )
        finally:
            backend.close()

        assert memory_granted == sqlite_granted

    def test_denied_retry_after_reflects_partial_refill(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Retry-After is the wait for the *shortfall*, not a flat 60/rpm."""
        clock = {"now": 1_700_000_000.0}
        monkeypatch.setattr(
            "omni_mercury_engine.security.rate_limiting.time.time",
            lambda: clock["now"],
        )
        # 60 rpm => 1 token/s. Drain the burst, then wait 0.75s.
        limiter = RateLimiter(requests_per_minute=60, burst_size=2)
        assert limiter.check("c").allowed
        assert limiter.check("c").allowed
        clock["now"] += 0.75
        info = limiter.check("c")
        assert info.allowed is False
        assert info.retry_after == pytest.approx(0.25, abs=1e-6)

    def test_retry_after_header_never_tells_a_client_to_retry_now(self) -> None:
        """``Retry-After: 0`` is an invitation to a hot retry loop.

        The header is whole seconds (RFC 9110 delay-seconds), and the value
        was truncated with ``int()``. At the shipped 100 rpm default the wait
        for the next token is 0.6s, so *every* 429 the limiter produced
        advertised ``Retry-After: 0`` -- a conforming client reads that as
        "retry immediately" against the limiter that just denied it. Present
        on ``main`` as well; the fractional refill only varied the value.
        """
        for wait in (0.0, 0.05, 0.2, 0.6, 0.999):
            info = RateLimitInfo(
                allowed=False, limit=100, remaining=0, reset_at=0, retry_after=wait
            )
            assert info.to_headers()["Retry-After"] == "1", wait

    def test_retry_after_header_rounds_up_never_down(self) -> None:
        """Under-stating the wait is the failure mode; over-stating costs one poll."""
        for wait, expected in ((1.0, "1"), (1.01, "2"), (1.5, "2"), (59.1, "60")):
            info = RateLimitInfo(
                allowed=False, limit=100, remaining=0, reset_at=0, retry_after=wait
            )
            assert info.to_headers()["Retry-After"] == expected, wait

    def test_retry_after_absent_when_allowed(self) -> None:
        info = RateLimitInfo(allowed=True, limit=100, remaining=5, reset_at=0)
        assert "Retry-After" not in info.to_headers()

    def test_shipped_defaults_emit_a_usable_retry_after(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end on 100 rpm / burst 20 -- the configuration that shipped."""
        clock = {"now": 1_700_000_000.0}
        monkeypatch.setattr(
            "omni_mercury_engine.security.rate_limiting.time.time",
            lambda: clock["now"],
        )
        limiter = RateLimiter(requests_per_minute=self._RPM, burst_size=self._BURST)
        for _ in range(self._BURST):
            assert limiter.check("client").allowed
        denied = limiter.check("client")
        assert denied.allowed is False
        assert int(denied.to_headers()["Retry-After"]) >= 1

    def test_fractional_balance_survives_the_store(self) -> None:
        """A sub-token balance round-trips through the backend unrounded."""
        from omni_mercury_engine.security.rate_limiting import InMemoryBackend

        backend = InMemoryBackend()
        backend.set("k", 1_700_000_000.0, 0.4, 300)
        state = backend.get("k")
        assert state is not None
        assert state[1] == pytest.approx(0.4)


class TestCounterStores:
    """Fixed-window counters: atomic increments, window resets, pruning."""

    @pytest.fixture(params=["memory", "sqlite"])
    def counter_store(
        self, request: pytest.FixtureRequest, tmp_path: Path
    ) -> InMemoryCounterStore | SqliteCounterStore:
        """Both counter backends, for parity."""
        if request.param == "memory":
            return InMemoryCounterStore()
        return SqliteCounterStore(tmp_path / "counters.db")

    def test_concurrent_increments_never_undercount(
        self, counter_store: InMemoryCounterStore | SqliteCounterStore
    ) -> None:
        """N racing increments land exactly N (no lost updates)."""
        window = 1_000_000
        results: list[int] = []
        lock = threading.Lock()

        def worker() -> None:
            value = counter_store.increment("k", window)
            with lock:
                results.append(value)

        threads = [threading.Thread(target=worker) for _ in range(40)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert max(results) == 40
        assert sorted(results) == list(range(1, 41))

    def test_new_window_resets_count(
        self, counter_store: InMemoryCounterStore | SqliteCounterStore
    ) -> None:
        """A later window starts counting from 1 again."""
        assert counter_store.increment("k", 100) == 1
        assert counter_store.increment("k", 100) == 2
        assert counter_store.increment("k", 200) == 1

    def test_prune_stale_counters(
        self, counter_store: InMemoryCounterStore | SqliteCounterStore
    ) -> None:
        """Old-window rows are prunable."""
        counter_store.increment("old", 100)
        counter_store.increment("new", 200)
        assert counter_store.prune_stale(150) == 1

    def test_sqlite_counters_shared_and_persistent(self, tmp_path: Path) -> None:
        """Counters are shared across instances and survive reopen."""
        path = tmp_path / "counters.db"
        first = SqliteCounterStore(path)
        assert first.increment("k", 100) == 1
        second = SqliteCounterStore(path)
        assert second.increment("k", 100) == 2  # second worker, same window
        first.close()
        second.close()
        third = SqliteCounterStore(path)
        assert third.increment("k", 100) == 3  # restart keeps the count
        third.close()


class TestActionRateLimiter:
    """Per-action rule evaluation and environment overrides."""

    def test_denies_over_ceiling_with_retry_after(self) -> None:
        """The attempt over the ceiling is denied and told when to return."""
        limiter = ActionRateLimiter(
            InMemoryCounterStore(),
            rules={"login_ip": ActionRule(max_attempts=3, window_seconds=60)},
            clock=lambda: 1000.0,
        )
        assert [limiter.check("login_ip", "ip1")[0] for _ in range(3)] == [True] * 3
        allowed, retry_after = limiter.check("login_ip", "ip1")
        assert allowed is False
        assert 0 < retry_after <= 60

    def test_keys_are_isolated(self) -> None:
        """One caller exhausting its budget does not affect another."""
        limiter = ActionRateLimiter(
            InMemoryCounterStore(),
            rules={"login_ip": ActionRule(max_attempts=1, window_seconds=60)},
            clock=lambda: 1000.0,
        )
        assert limiter.check("login_ip", "a")[0] is True
        assert limiter.check("login_ip", "a")[0] is False
        assert limiter.check("login_ip", "b")[0] is True

    def test_window_rollover_frees_budget(self) -> None:
        """After the window rolls, attempts are allowed again."""
        now = {"t": 1000.0}
        limiter = ActionRateLimiter(
            InMemoryCounterStore(),
            rules={"reset_ip": ActionRule(max_attempts=1, window_seconds=60)},
            clock=lambda: now["t"],
        )
        assert limiter.check("reset_ip", "a")[0] is True
        assert limiter.check("reset_ip", "a")[0] is False
        now["t"] = 1061.0
        assert limiter.check("reset_ip", "a")[0] is True

    def test_env_override_and_malformed_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MERCURY_AUTH_RATE_* overrides apply; junk keeps the strict default."""
        monkeypatch.setenv("MERCURY_AUTH_RATE_LOGIN_IP", "2/120")
        monkeypatch.setenv("MERCURY_AUTH_RATE_REGISTER_IP", "not-a-rule")
        monkeypatch.setenv("MERCURY_AUTH_RATE_RESET_IP", "0/60")  # zero is invalid
        rules = ActionRateLimiter.rules_from_env()
        assert rules["login_ip"] == ActionRule(max_attempts=2, window_seconds=120)
        assert rules["register_ip"] == ActionRule(max_attempts=5, window_seconds=3600)
        assert rules["reset_ip"] == ActionRule(max_attempts=5, window_seconds=3600)

    def test_unknown_action_allows(self) -> None:
        """A rule-less action never locks users out (typo containment)."""
        limiter = ActionRateLimiter(InMemoryCounterStore(), rules={})
        assert limiter.check("no_such_action", "k") == (True, 0)


class TestMiddlewareSpoofImmunity:
    """End-to-end regression: rotating XFF no longer mints fresh buckets."""

    def test_rotating_xff_hits_one_bucket(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no trusted proxy tier, spoofed XFF headers share the peer bucket."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from omni_mercury_engine.api.server import RateLimitMiddleware

        # Other API test modules disable the limiter process-wide at import;
        # force it on for this middleware-behaviour test.
        monkeypatch.setenv("OMNI_RATE_LIMIT_ENABLED", "true")
        monkeypatch.delenv("MERCURY_TRUSTED_PROXY_HOPS", raising=False)
        app = FastAPI()

        @app.get("/probe")
        def probe() -> dict[str, bool]:
            return {"ok": True}

        app.add_middleware(
            RateLimitMiddleware,  # type: ignore[arg-type, unused-ignore]
            requests_per_minute=60,
            burst_size=5,
        )

        with TestClient(app) as client:
            statuses = [
                client.get("/probe", headers={"X-Forwarded-For": f"10.0.0.{i}"}).status_code
                for i in range(10)
            ]
        # Pre-fix: every request minted a new bucket → all 200 (bypass).
        # Post-fix: one shared bucket → the burst cap bites at request 6.
        assert statuses[:5] == [200] * 5
        assert set(statuses[5:]) == {429}

    def test_trusted_hop_still_distinguishes_real_clients(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With one trusted hop, distinct real clients get distinct buckets."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from omni_mercury_engine.api.server import RateLimitMiddleware

        monkeypatch.setenv("OMNI_RATE_LIMIT_ENABLED", "true")
        monkeypatch.setenv("MERCURY_TRUSTED_PROXY_HOPS", "1")
        app = FastAPI()

        @app.get("/probe")
        def probe() -> dict[str, bool]:
            return {"ok": True}

        app.add_middleware(
            RateLimitMiddleware,  # type: ignore[arg-type, unused-ignore]
            requests_per_minute=60,
            burst_size=2,
        )

        with TestClient(app) as client:
            a1 = client.get("/probe", headers={"X-Forwarded-For": "198.51.100.7"})
            a2 = client.get("/probe", headers={"X-Forwarded-For": "198.51.100.7"})
            a3 = client.get("/probe", headers={"X-Forwarded-For": "198.51.100.7"})
            b1 = client.get("/probe", headers={"X-Forwarded-For": "198.51.100.8"})
        assert (a1.status_code, a2.status_code, a3.status_code) == (200, 200, 429)
        assert b1.status_code == 200  # a different real client is unaffected
