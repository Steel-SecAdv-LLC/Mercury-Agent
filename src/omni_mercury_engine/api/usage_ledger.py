# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-account usage metering.

An append-only ledger of metered work — one row per request, carrying the
account, timestamp, endpoint, and compute-milliseconds — plus a windowed
aggregation used by the quota engine. It follows the same shape as the other
stores: a :class:`UsageLedger` Protocol with an in-memory backend (dev/tests)
and a durable SQLite backend selected by the shared ``MERCURY_KEYSTORE_PATH``
file.

Metering is deliberately separate from enforcement: this module only records
and summarises usage; :mod:`quota` turns a summary into an allow/deny decision.
That split keeps the ledger a plain, auditable fact table and lets the policy
change without touching the data model.

**Hard request-count enforcement** needs one operation the split alone cannot
give: an atomic check-and-reserve. :meth:`UsageLedger.reserve` counts the
window and inserts the new row inside a single critical section (a lock for
the in-memory backend, ``BEGIN IMMEDIATE`` for SQLite), so two concurrent
requests cannot both slip under a nearly full ceiling. The compute-ms ceiling
stays *soft* by design — a request's cost is only known after it runs, so an
in-flight request can overshoot the compute budget once before the window
closes on the account (see the quota notes in ``docs/PLATFORM_HARDENING.md``).
"""

from __future__ import annotations

import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from datetime import datetime

__all__ = [
    "InMemoryUsageLedger",
    "ReserveResult",
    "SqliteUsageLedger",
    "UsageEvent",
    "UsageLedger",
    "UsageSummary",
    "build_usage_ledger",
]

#: Shared SQLite database file (same variable as the key/identity stores, so one
#: file holds keys, accounts, and usage). Unset selects the in-memory backend.
USAGE_PATH_ENV = "MERCURY_KEYSTORE_PATH"


@dataclass
class UsageEvent:
    """One metered unit of work charged to an account."""

    account_id: str
    ts: datetime
    endpoint: str
    compute_ms: float


@dataclass
class UsageSummary:
    """Aggregated usage for an account over a time window."""

    request_count: int
    compute_ms: float


@dataclass
class ReserveResult:
    """Outcome of an atomic check-and-reserve against the window ceilings."""

    allowed: bool
    #: Ledger id of the reserved row (fill its compute cost via
    #: :meth:`UsageLedger.set_compute` when the work finishes); ``None`` when
    #: denied.
    event_id: int | None
    #: Window usage *before* this reservation.
    request_count: int
    compute_ms: float
    #: ``"requests"`` or ``"compute"`` when denied; ``None`` when allowed.
    denied_by: str | None = None


@runtime_checkable
class UsageLedger(Protocol):
    """Contract for recording and summarising per-account usage."""

    def record(self, event: UsageEvent) -> int:
        """Append a usage event; return its ledger id."""
        ...

    def set_compute(self, event_id: int, compute_ms: float) -> None:
        """Set the measured compute cost of a previously recorded event."""
        ...

    def reserve(
        self,
        account_id: str,
        endpoint: str,
        ts: datetime,
        since: datetime,
        max_requests: int,
        max_compute_ms: float,
    ) -> ReserveResult:
        """Atomically check the window ceilings and insert the row if allowed."""
        ...

    def summary_since(self, account_id: str, since: datetime) -> UsageSummary:
        """Summarise an account's usage with ``ts >= since``."""
        ...

    def prune_before(self, cutoff: datetime) -> int:
        """Delete events older than ``cutoff``; return how many were removed."""
        ...


class InMemoryUsageLedger:
    """Process-local usage ledger (dev/test default; not durable)."""

    def __init__(self) -> None:
        """Initialise an empty event list."""
        self._events: dict[int, UsageEvent] = {}
        self._next_id = 1
        self._lock = threading.Lock()

    def record(self, event: UsageEvent) -> int:
        """Append a usage event; return its ledger id."""
        with self._lock:
            event_id = self._next_id
            self._next_id += 1
            self._events[event_id] = event
            return event_id

    def set_compute(self, event_id: int, compute_ms: float) -> None:
        """Set the measured compute cost of a previously recorded event."""
        with self._lock:
            event = self._events.get(event_id)
            if event is not None:
                event.compute_ms = compute_ms

    def _summary_locked(self, account_id: str, since: datetime) -> UsageSummary:
        """Aggregate under the caller-held lock."""
        relevant = [
            e for e in self._events.values() if e.account_id == account_id and e.ts >= since
        ]
        return UsageSummary(
            request_count=len(relevant),
            compute_ms=float(sum(e.compute_ms for e in relevant)),
        )

    def reserve(
        self,
        account_id: str,
        endpoint: str,
        ts: datetime,
        since: datetime,
        max_requests: int,
        max_compute_ms: float,
    ) -> ReserveResult:
        """Atomically check the window ceilings and insert the row if allowed."""
        with self._lock:
            summary = self._summary_locked(account_id, since)
            denied = _deny_reason(summary, max_requests, max_compute_ms)
            if denied is not None:
                return ReserveResult(
                    allowed=False,
                    event_id=None,
                    request_count=summary.request_count,
                    compute_ms=summary.compute_ms,
                    denied_by=denied,
                )
            event_id = self._next_id
            self._next_id += 1
            self._events[event_id] = UsageEvent(
                account_id=account_id, ts=ts, endpoint=endpoint, compute_ms=0.0
            )
            return ReserveResult(
                allowed=True,
                event_id=event_id,
                request_count=summary.request_count,
                compute_ms=summary.compute_ms,
            )

    def summary_since(self, account_id: str, since: datetime) -> UsageSummary:
        """Summarise an account's usage with ``ts >= since``."""
        with self._lock:
            return self._summary_locked(account_id, since)

    def prune_before(self, cutoff: datetime) -> int:
        """Delete events older than ``cutoff``; return how many were removed."""
        with self._lock:
            stale = [eid for eid, e in self._events.items() if e.ts < cutoff]
            for event_id in stale:
                del self._events[event_id]
            return len(stale)


def _deny_reason(summary: UsageSummary, max_requests: int, max_compute_ms: float) -> str | None:
    """Which ceiling (if any) the summarised usage has already met."""
    if summary.request_count >= max_requests:
        return "requests"
    if summary.compute_ms >= max_compute_ms:
        return "compute"
    return None


class SqliteUsageLedger:
    """Durable usage ledger backed by stdlib ``sqlite3`` (WAL, lock-guarded)."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS usage_events (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        account_id TEXT NOT NULL,
        ts         TEXT NOT NULL,
        endpoint   TEXT NOT NULL,
        compute_ms REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_usage_account_ts ON usage_events (account_id, ts);
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        """Open (creating if needed) the usage ledger at ``path``."""
        self._path = Path(path)
        if self._path.parent and not self._path.parent.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.isolation_level = None  # explicit transactions only
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(self._SCHEMA)

    def close(self) -> None:
        """Close the underlying connection."""
        with self._lock:
            self._conn.close()

    def record(self, event: UsageEvent) -> int:
        """Append a usage event; return its ledger id."""
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO usage_events (account_id, ts, endpoint, compute_ms) "
                "VALUES (?, ?, ?, ?)",
                (event.account_id, event.ts.isoformat(), event.endpoint, event.compute_ms),
            )
            return int(cur.lastrowid or 0)

    def set_compute(self, event_id: int, compute_ms: float) -> None:
        """Set the measured compute cost of a previously recorded event."""
        with self._lock:
            self._conn.execute(
                "UPDATE usage_events SET compute_ms = ? WHERE id = ?",
                (compute_ms, event_id),
            )

    def reserve(
        self,
        account_id: str,
        endpoint: str,
        ts: datetime,
        since: datetime,
        max_requests: int,
        max_compute_ms: float,
    ) -> ReserveResult:
        """Atomically check the window ceilings and insert the row if allowed.

        ``BEGIN IMMEDIATE`` takes the database write lock before the count, so
        concurrent workers serialise here and the request ceiling is *hard*:
        N racing requests against a window with one slot left admit exactly
        one.
        """
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT COUNT(*) AS n, COALESCE(SUM(compute_ms), 0.0) AS total "
                    "FROM usage_events WHERE account_id = ? AND ts >= ?",
                    (account_id, since.isoformat()),
                ).fetchone()
                summary = UsageSummary(request_count=int(row["n"]), compute_ms=float(row["total"]))
                denied = _deny_reason(summary, max_requests, max_compute_ms)
                if denied is not None:
                    self._conn.execute("COMMIT")
                    return ReserveResult(
                        allowed=False,
                        event_id=None,
                        request_count=summary.request_count,
                        compute_ms=summary.compute_ms,
                        denied_by=denied,
                    )
                cur = self._conn.execute(
                    "INSERT INTO usage_events (account_id, ts, endpoint, compute_ms) "
                    "VALUES (?, ?, ?, 0.0)",
                    (account_id, ts.isoformat(), endpoint),
                )
                event_id = int(cur.lastrowid or 0)
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
        return ReserveResult(
            allowed=True,
            event_id=event_id,
            request_count=summary.request_count,
            compute_ms=summary.compute_ms,
        )

    def summary_since(self, account_id: str, since: datetime) -> UsageSummary:
        """Summarise an account's usage with ``ts >= since``."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(compute_ms), 0.0) AS total "
                "FROM usage_events WHERE account_id = ? AND ts >= ?",
                (account_id, since.isoformat()),
            ).fetchone()
        return UsageSummary(request_count=int(row["n"]), compute_ms=float(row["total"]))

    def prune_before(self, cutoff: datetime) -> int:
        """Delete events older than ``cutoff``; return how many were removed."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM usage_events WHERE ts < ?", (cutoff.isoformat(),))
            return int(cur.rowcount)


def build_usage_ledger() -> UsageLedger:
    """Construct the configured usage-ledger backend from the environment.

    Returns:
        A :class:`SqliteUsageLedger` when ``MERCURY_KEYSTORE_PATH`` is set
        (durable; shares the auth database file), otherwise an
        :class:`InMemoryUsageLedger` (dev/test default).
    """
    path = os.getenv(USAGE_PATH_ENV, "").strip()
    if not path:
        return InMemoryUsageLedger()
    return SqliteUsageLedger(path)
