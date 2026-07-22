# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-account quota enforcement over the usage ledger.

Turns a windowed :class:`~omni_mercury_engine.api.usage_ledger.UsageSummary`
into an allow/deny decision, so a free service throttles heavy accounts fairly
and automatically instead of being shut off wholesale. Two independent ceilings
apply over a rolling window: a request count and a compute-millisecond budget —
whichever is hit first denies further work until the window rolls forward.

Limits are configuration, not code (``MERCURY_QUOTA_*``), so they can be dialed
as the cost budget changes without a redeploy. This module is framework-free
(no FastAPI import) and fully testable with an in-memory ledger and an injected
clock; the HTTP 429 wiring lives at the route layer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from omni_mercury_engine.api.usage_ledger import UsageEvent, build_usage_ledger

if TYPE_CHECKING:
    from collections.abc import Callable

    from omni_mercury_engine.api.usage_ledger import UsageLedger

__all__ = [
    "QuotaConfig",
    "QuotaDecision",
    "QuotaEnforcer",
    "build_quota_enforcer",
]


@dataclass(frozen=True)
class QuotaConfig:
    """Rolling-window quota ceilings for one account tier."""

    window_seconds: int = 3600
    max_requests: int = 1000
    max_compute_ms: float = 600_000.0

    @classmethod
    def from_env(cls) -> QuotaConfig:
        """Build a config from ``MERCURY_QUOTA_*`` (falling back to defaults)."""
        return cls(
            window_seconds=int(os.getenv("MERCURY_QUOTA_WINDOW_SECONDS", "3600")),
            max_requests=int(os.getenv("MERCURY_QUOTA_MAX_REQUESTS", "1000")),
            max_compute_ms=float(os.getenv("MERCURY_QUOTA_MAX_COMPUTE_MS", "600000")),
        )


@dataclass
class QuotaDecision:
    """The outcome of a quota check, including the usage it was based on."""

    allowed: bool
    request_count: int
    compute_ms: float
    reason: str | None = None
    retry_after_seconds: int | None = None


class QuotaEnforcer:
    """Checks and records per-account usage against a :class:`QuotaConfig`."""

    def __init__(
        self,
        ledger: UsageLedger,
        config: QuotaConfig,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Wire the enforcer to a ledger, a config, and an injectable clock.

        Args:
            ledger: The usage ledger to read/record against.
            config: The quota ceilings to enforce.
            clock: Time source (defaults to ``datetime.now(UTC)``); injected in
                tests for deterministic window rolloff.
        """
        self._ledger = ledger
        self._config = config
        self._clock = clock or (lambda: datetime.now(UTC))

    def check(self, account_id: str) -> QuotaDecision:
        """Decide whether ``account_id`` may make another request right now.

        The decision reads only usage inside the rolling window; either ceiling
        (request count or compute budget) being met denies the request.

        Args:
            account_id: The account to check.

        Returns:
            A :class:`QuotaDecision`. When denied, ``retry_after_seconds`` is the
            window length (an upper bound on when budget frees up).
        """
        since = self._clock() - timedelta(seconds=self._config.window_seconds)
        summary = self._ledger.summary_since(account_id, since)
        if summary.request_count >= self._config.max_requests:
            return QuotaDecision(
                allowed=False,
                request_count=summary.request_count,
                compute_ms=summary.compute_ms,
                reason="request quota exceeded",
                retry_after_seconds=self._config.window_seconds,
            )
        if summary.compute_ms >= self._config.max_compute_ms:
            return QuotaDecision(
                allowed=False,
                request_count=summary.request_count,
                compute_ms=summary.compute_ms,
                reason="compute quota exceeded",
                retry_after_seconds=self._config.window_seconds,
            )
        return QuotaDecision(
            allowed=True,
            request_count=summary.request_count,
            compute_ms=summary.compute_ms,
        )

    def record(self, account_id: str, endpoint: str, compute_ms: float) -> None:
        """Record one unit of metered work for an account.

        Typical flow is check-before, record-after: reject if already at the
        ceiling, do the work, then record the actual compute cost.

        Args:
            account_id: The charged account.
            endpoint: The endpoint/operation label (for auditing/attribution).
            compute_ms: Wall-clock compute milliseconds to charge.
        """
        self._ledger.record(
            UsageEvent(
                account_id=account_id,
                ts=self._clock(),
                endpoint=endpoint,
                compute_ms=compute_ms,
            )
        )


def build_quota_enforcer() -> QuotaEnforcer:
    """Construct a :class:`QuotaEnforcer` wired from the environment.

    Uses :func:`build_usage_ledger` (durable when ``MERCURY_KEYSTORE_PATH`` is
    set) and :meth:`QuotaConfig.from_env`.

    Returns:
        A ready-to-use quota enforcer.
    """
    return QuotaEnforcer(build_usage_ledger(), QuotaConfig.from_env())
