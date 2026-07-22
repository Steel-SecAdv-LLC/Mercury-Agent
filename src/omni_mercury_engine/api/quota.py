# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-account quota enforcement over the usage ledger.

Turns windowed usage into an allow/deny decision, so a free service throttles
heavy accounts fairly and automatically instead of being shut off wholesale.
Two independent ceilings apply over a rolling window: a request count and a
compute-millisecond budget — whichever is hit first denies further work until
the window rolls forward.

Limits are configuration, not code, resolved in precedence order:

1. **Per-account override** — a row in the ``quota_overrides`` table (shared
   SQLite file), settable at runtime via :meth:`QuotaOverrideStore.set_override`
   or plain SQL; lets an operator lift or pinch one account without a deploy.
2. **Tier** — the account's ``tier`` column, matched against
   ``MERCURY_QUOTA_TIER_<NAME>`` (``"<max_requests>,<max_compute_ms>,
   <window_seconds>"``), e.g. ``MERCURY_QUOTA_TIER_SUPPORTER=5000,3600000,3600``.
3. **Default** — the base ``MERCURY_QUOTA_*`` variables.

Enforcement has two modes: :meth:`QuotaEnforcer.check` (read-only decision,
kept for callers that only report) and the **hard** path —
:meth:`QuotaEnforcer.reserve` + :meth:`QuotaEnforcer.commit` — which counts
and inserts atomically in the ledger so concurrent requests cannot overrun
the request ceiling, then back-fills the measured compute cost. This module
is framework-free (no FastAPI import) and fully testable with an in-memory
ledger and an injected clock; the HTTP 429 wiring lives in the server's
quota middleware.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from omni_mercury_engine.api.usage_ledger import UsageEvent, build_usage_ledger

if TYPE_CHECKING:
    from collections.abc import Callable

    from omni_mercury_engine.api.usage_ledger import UsageLedger

logger = logging.getLogger(__name__)

__all__ = [
    "InMemoryQuotaOverrideStore",
    "QuotaConfig",
    "QuotaDecision",
    "QuotaEnforcer",
    "QuotaOverrideStore",
    "SqliteQuotaOverrideStore",
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
        """Build the default config from ``MERCURY_QUOTA_*`` (with fallbacks)."""
        return cls(
            window_seconds=int(os.getenv("MERCURY_QUOTA_WINDOW_SECONDS", "3600")),
            max_requests=int(os.getenv("MERCURY_QUOTA_MAX_REQUESTS", "1000")),
            max_compute_ms=float(os.getenv("MERCURY_QUOTA_MAX_COMPUTE_MS", "600000")),
        )


def _tier_configs_from_env(default: QuotaConfig) -> dict[str, QuotaConfig]:
    """Collect ``MERCURY_QUOTA_TIER_<NAME>`` tier definitions.

    Each value is ``"<max_requests>,<max_compute_ms>[,<window_seconds>]"``;
    a malformed entry is logged and skipped (the tier then resolves to the
    default — fail-closed toward the stricter base policy).
    """
    tiers: dict[str, QuotaConfig] = {"free": default}
    prefix = "MERCURY_QUOTA_TIER_"
    for name, raw in os.environ.items():
        if not name.startswith(prefix):
            continue
        tier_name = name[len(prefix) :].lower()
        parts = [p.strip() for p in raw.split(",")]
        try:
            max_requests = int(parts[0])
            max_compute_ms = float(parts[1])
            window = int(parts[2]) if len(parts) > 2 else default.window_seconds
        except (IndexError, ValueError):
            logger.warning("ignoring malformed quota tier %s=%r", name, raw)
            continue
        tiers[tier_name] = QuotaConfig(
            window_seconds=window, max_requests=max_requests, max_compute_ms=max_compute_ms
        )
    return tiers


@dataclass
class QuotaDecision:
    """The outcome of a quota check, including the usage it was based on."""

    allowed: bool
    request_count: int
    compute_ms: float
    reason: str | None = None
    retry_after_seconds: int | None = None
    #: Ledger row backing an allowed :meth:`QuotaEnforcer.reserve`; pass to
    #: :meth:`QuotaEnforcer.commit` with the measured cost.
    event_id: int | None = None


# --------------------------------------------------------------------------- #
# per-account overrides
# --------------------------------------------------------------------------- #
@runtime_checkable
class QuotaOverrideStore(Protocol):
    """Contract for per-account quota overrides."""

    def get_override(self, account_id: str) -> QuotaConfig | None:
        """Return the override for ``account_id``, or ``None``."""
        ...

    def set_override(self, account_id: str, config: QuotaConfig | None) -> None:
        """Set (or with ``None`` clear) the override for ``account_id``."""
        ...


class InMemoryQuotaOverrideStore:
    """Process-local override store (dev/test default)."""

    def __init__(self) -> None:
        """Initialise the empty override map."""
        self._overrides: dict[str, QuotaConfig] = {}
        self._lock = threading.Lock()

    def get_override(self, account_id: str) -> QuotaConfig | None:
        """Return the override for ``account_id``, or ``None``."""
        with self._lock:
            return self._overrides.get(account_id)

    def set_override(self, account_id: str, config: QuotaConfig | None) -> None:
        """Set (or clear) the override for ``account_id``."""
        with self._lock:
            if config is None:
                self._overrides.pop(account_id, None)
            else:
                self._overrides[account_id] = config


class SqliteQuotaOverrideStore:
    """Durable override store on the shared SQLite file."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS quota_overrides (
        account_id     TEXT PRIMARY KEY,
        window_seconds INTEGER NOT NULL,
        max_requests   INTEGER NOT NULL,
        max_compute_ms REAL NOT NULL
    );
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        """Open (creating if needed) the override store at ``path``."""
        self._path = Path(path)
        if self._path.parent and not self._path.parent.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.isolation_level = None
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(self._SCHEMA)

    def close(self) -> None:
        """Close the underlying connection."""
        with self._lock:
            self._conn.close()

    def get_override(self, account_id: str) -> QuotaConfig | None:
        """Return the override for ``account_id``, or ``None``."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM quota_overrides WHERE account_id = ?", (account_id,)
            ).fetchone()
        if row is None:
            return None
        return QuotaConfig(
            window_seconds=int(row["window_seconds"]),
            max_requests=int(row["max_requests"]),
            max_compute_ms=float(row["max_compute_ms"]),
        )

    def set_override(self, account_id: str, config: QuotaConfig | None) -> None:
        """Set (or clear) the override for ``account_id``."""
        with self._lock:
            if config is None:
                self._conn.execute(
                    "DELETE FROM quota_overrides WHERE account_id = ?", (account_id,)
                )
            else:
                self._conn.execute(
                    "INSERT INTO quota_overrides "
                    "(account_id, window_seconds, max_requests, max_compute_ms) "
                    "VALUES (?, ?, ?, ?) ON CONFLICT(account_id) DO UPDATE SET "
                    "window_seconds = excluded.window_seconds, "
                    "max_requests = excluded.max_requests, "
                    "max_compute_ms = excluded.max_compute_ms",
                    (
                        account_id,
                        config.window_seconds,
                        config.max_requests,
                        config.max_compute_ms,
                    ),
                )


# --------------------------------------------------------------------------- #
# enforcement
# --------------------------------------------------------------------------- #
class QuotaEnforcer:
    """Checks, reserves, and records per-account usage against resolved configs."""

    def __init__(
        self,
        ledger: UsageLedger,
        config: QuotaConfig,
        clock: Callable[[], datetime] | None = None,
        *,
        tiers: dict[str, QuotaConfig] | None = None,
        overrides: QuotaOverrideStore | None = None,
    ) -> None:
        """Wire the enforcer to a ledger, policy sources, and a clock.

        Args:
            ledger: The usage ledger to read/record against.
            config: The default quota ceilings.
            clock: Time source (defaults to ``datetime.now(UTC)``); injected in
                tests for deterministic window rolloff.
            tiers: Named tier configs (the ``"free"`` tier defaults to
                ``config``); resolved via each account's tier name.
            overrides: Per-account override store (highest precedence).
        """
        self._ledger = ledger
        self._config = config
        self._clock = clock or (lambda: datetime.now(UTC))
        self._tiers = tiers if tiers is not None else {"free": config}
        self._overrides = overrides or InMemoryQuotaOverrideStore()

    @property
    def override_store(self) -> QuotaOverrideStore:
        """The per-account override store (operator tooling hooks in here)."""
        return self._overrides

    def config_for(self, account_id: str, tier: str = "free") -> QuotaConfig:
        """Resolve the effective config: override > tier > default.

        Args:
            account_id: The account being charged.
            tier: The account's tier name (unknown names fall back to default).

        Returns:
            The effective :class:`QuotaConfig`.
        """
        override = self._overrides.get_override(account_id)
        if override is not None:
            return override
        return self._tiers.get(tier.lower(), self._config)

    def check(self, account_id: str, tier: str = "free") -> QuotaDecision:
        """Decide (read-only) whether ``account_id`` may make another request.

        The decision reads only usage inside the rolling window; either ceiling
        (request count or compute budget) being met denies the request. This
        does not reserve — concurrent callers can race it. Use
        :meth:`reserve` for hard enforcement.

        Args:
            account_id: The account to check.
            tier: The account's tier name.

        Returns:
            A :class:`QuotaDecision`. When denied, ``retry_after_seconds`` is
            the window length (an upper bound on when budget frees up).
        """
        config = self.config_for(account_id, tier)
        since = self._clock() - timedelta(seconds=config.window_seconds)
        summary = self._ledger.summary_since(account_id, since)
        if summary.request_count >= config.max_requests:
            return QuotaDecision(
                allowed=False,
                request_count=summary.request_count,
                compute_ms=summary.compute_ms,
                reason="request quota exceeded",
                retry_after_seconds=config.window_seconds,
            )
        if summary.compute_ms >= config.max_compute_ms:
            return QuotaDecision(
                allowed=False,
                request_count=summary.request_count,
                compute_ms=summary.compute_ms,
                reason="compute quota exceeded",
                retry_after_seconds=config.window_seconds,
            )
        return QuotaDecision(
            allowed=True,
            request_count=summary.request_count,
            compute_ms=summary.compute_ms,
        )

    def reserve(self, account_id: str, endpoint: str, tier: str = "free") -> QuotaDecision:
        """Atomically check the ceilings and charge one request slot.

        The request-count ceiling is *hard*: the count and the insert happen
        in one ledger critical section, so racing requests cannot jointly
        exceed it. On success, finish with :meth:`commit` to back-fill the
        measured compute cost onto the reserved row.

        Args:
            account_id: The account to charge.
            endpoint: The endpoint/operation label (auditing/attribution).
            tier: The account's tier name.

        Returns:
            A :class:`QuotaDecision`; ``event_id`` is set when allowed.
        """
        config = self.config_for(account_id, tier)
        now = self._clock()
        since = now - timedelta(seconds=config.window_seconds)
        result = self._ledger.reserve(
            account_id, endpoint, now, since, config.max_requests, config.max_compute_ms
        )
        if not result.allowed:
            reason = (
                "request quota exceeded"
                if result.denied_by == "requests"
                else "compute quota exceeded"
            )
            return QuotaDecision(
                allowed=False,
                request_count=result.request_count,
                compute_ms=result.compute_ms,
                reason=reason,
                retry_after_seconds=config.window_seconds,
            )
        return QuotaDecision(
            allowed=True,
            request_count=result.request_count,
            compute_ms=result.compute_ms,
            event_id=result.event_id,
        )

    def commit(self, event_id: int, compute_ms: float) -> None:
        """Back-fill the measured compute cost onto a reserved ledger row.

        Args:
            event_id: The ``event_id`` from an allowed :meth:`reserve`.
            compute_ms: Wall-clock compute milliseconds actually spent.
        """
        self._ledger.set_compute(event_id, compute_ms)

    def record(self, account_id: str, endpoint: str, compute_ms: float) -> None:
        """Record one unit of metered work for an account (soft path).

        Kept for callers that meter without reserving (check-before,
        record-after). For hard enforcement use :meth:`reserve`/:meth:`commit`.

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
    set), :meth:`QuotaConfig.from_env`, ``MERCURY_QUOTA_TIER_*`` tier
    definitions, and a durable override store on the same file.

    Returns:
        A ready-to-use quota enforcer.
    """
    default = QuotaConfig.from_env()
    path = os.getenv("MERCURY_KEYSTORE_PATH", "").strip()
    overrides: QuotaOverrideStore = (
        SqliteQuotaOverrideStore(path) if path else InMemoryQuotaOverrideStore()
    )
    return QuotaEnforcer(
        build_usage_ledger(),
        default,
        tiers=_tier_configs_from_env(default),
        overrides=overrides,
    )
