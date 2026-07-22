# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-account usage metering.

An append-only ledger of metered work — one row per request, carrying the
account, timestamp, endpoint, and compute-milliseconds — plus a windowed
aggregation used by the quota engine. It follows the same shape as the other
stores: an :class:`UsageLedger` Protocol with an in-memory backend (dev/tests)
and a durable SQLite backend selected by the shared ``MERCURY_KEYSTORE_PATH``
file.

Metering is deliberately separate from enforcement: this module only records and
summarises usage; :mod:`quota` turns a summary into an allow/deny decision. That
split keeps the ledger a plain, auditable fact table and lets the policy change
without touching the data model.
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


@runtime_checkable
class UsageLedger(Protocol):
    """Contract for recording and summarising per-account usage."""

    def record(self, event: UsageEvent) -> None:
        """Append a usage event."""
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
        self._events: list[UsageEvent] = []
        self._lock = threading.Lock()

    def record(self, event: UsageEvent) -> None:
        """Append a usage event."""
        with self._lock:
            self._events.append(event)

    def summary_since(self, account_id: str, since: datetime) -> UsageSummary:
        """Summarise an account's usage with ``ts >= since``."""
        with self._lock:
            relevant = [e for e in self._events if e.account_id == account_id and e.ts >= since]
        return UsageSummary(
            request_count=len(relevant),
            compute_ms=float(sum(e.compute_ms for e in relevant)),
        )

    def prune_before(self, cutoff: datetime) -> int:
        """Delete events older than ``cutoff``; return how many were removed."""
        with self._lock:
            before = len(self._events)
            self._events = [e for e in self._events if e.ts >= cutoff]
            return before - len(self._events)


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
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(self._SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        """Close the underlying connection."""
        with self._lock:
            self._conn.close()

    def record(self, event: UsageEvent) -> None:
        """Append a usage event."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO usage_events (account_id, ts, endpoint, compute_ms) "
                "VALUES (?, ?, ?, ?)",
                (event.account_id, event.ts.isoformat(), event.endpoint, event.compute_ms),
            )
            self._conn.commit()

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
            self._conn.commit()
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
