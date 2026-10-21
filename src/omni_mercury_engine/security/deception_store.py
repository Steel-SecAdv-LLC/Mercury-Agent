# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cross-worker tripwire state for the deception sensor.

With N uvicorn workers, an in-process :class:`~omni_mercury_engine.security.
deception.DeceptionSensor` gives a prober N tarpit allowances, N first-touch
audit records for the same decoy, and a campaign signal that each worker sees
at 1/N strength. :class:`SqliteTripwireStore` closes that split the same way
the platform's rate limiter does (``api/rate_limit_store.py``): state lives in
the shared SQLite file named by ``MERCURY_KEYSTORE_PATH``, every
:meth:`~SqliteTripwireStore.record_touch` runs inside one ``BEGIN IMMEDIATE``
transaction so concurrent workers serialize on the write lock, and without the
variable :func:`build_tripwire_store` returns ``None`` -- the existing
single-process behaviour, unchanged.

Time here is the wall clock (:func:`time.time`), not the sensor's monotonic
clock: rows are read by other processes and across restarts, where a monotonic
reading has no shared meaning. The two clocks never mix -- the sensor keeps
its own for the in-process fallback.

Bounded like the sensor: per-key hit history is capped (oldest rows dropped),
rows older than the window are pruned on every touch for the keys involved,
a full prune of all stale rows runs opportunistically, and the sources and
aggregates on record are each capped at the sensor's ``max_sources`` -- past
it the least recently seen key is evicted with its hits and decoy touches,
the in-process sensor's LRU rule. Without the cap, every rotated address
added rows forever and every touch paid a scan over all of them.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "TRIPWIRE_STORE_PATH_ENV",
    "SqliteTripwireStore",
    "TouchRecord",
    "TripwireStore",
    "build_tripwire_store",
]

#: The shared SQLite file -- the same variable the key store, identity store
#: and rate limiter use, so one file backs all cross-worker platform state.
#: Unset keeps tripwire state in-process.
TRIPWIRE_STORE_PATH_ENV = "MERCURY_KEYSTORE_PATH"

#: Hits per key beyond which a full stale prune is attempted on a touch; the
#: per-key prune on every touch keeps the involved keys exact regardless.
_FULL_PRUNE_EVERY = 256


@dataclass(frozen=True)
class TouchRecord:
    """Cluster-wide counts for one recorded decoy touch.

    Attributes:
        hits: The source's hits inside the window, this one included.
        aggregate_hits: The aggregate's hits inside the window, ditto.
        distinct_decoys: Distinct decoys the source has touched on record.
        first_touch: Whether no worker had recorded this source on this decoy.
        audit_due: Whether the (source, decoy) pair is due an audit record --
            its first touch, or a window elapsed since its last audit.
        sources_in_window: Distinct sources that touched any decoy inside the
            window, across all workers.
    """

    hits: int
    aggregate_hits: int
    distinct_decoys: int
    first_touch: bool
    audit_due: bool
    sources_in_window: int


@runtime_checkable
class TripwireStore(Protocol):
    """What the deception sensor needs from a shared state backend."""

    def record_touch(
        self,
        key: str,
        aggregate: str,
        path: str,
        *,
        window_s: float,
        max_hits: int,
        max_sources: int = 10_000,
    ) -> TouchRecord:
        """Record one touch and return the cluster-wide counts."""
        ...  # pragma: no cover - protocol signature

    def distinct_decoys(self, key: str) -> int:
        """Return the distinct decoys the source has touched on record."""
        ...  # pragma: no cover - protocol signature

    def close(self) -> None:
        """Release the backing resources."""
        ...  # pragma: no cover - protocol signature


class SqliteTripwireStore:
    """Durable, cross-worker tripwire state on the shared SQLite file.

    Args:
        path: The SQLite file (created, with parents, when missing).
        clock: Wall clock, injectable for tests.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS deception_hits (
        scope     TEXT NOT NULL,
        scope_key TEXT NOT NULL,
        hit_s     REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_deception_hits
        ON deception_hits (scope, scope_key, hit_s);
    CREATE TABLE IF NOT EXISTS deception_touches (
        source_key TEXT NOT NULL,
        decoy_path TEXT NOT NULL,
        first_s    REAL NOT NULL,
        audited_s  REAL NOT NULL,
        PRIMARY KEY (source_key, decoy_path)
    );
    CREATE TABLE IF NOT EXISTS deception_keys (
        scope     TEXT NOT NULL,
        scope_key TEXT NOT NULL,
        last_s    REAL NOT NULL,
        PRIMARY KEY (scope, scope_key)
    );
    CREATE INDEX IF NOT EXISTS idx_deception_keys_last
        ON deception_keys (scope, last_s);
    """

    # A store written before deception_keys existed: put its keys on record
    # so the cap sees them. Idempotent; a no-op once every key is listed.
    _BACKFILL = """
    INSERT OR IGNORE INTO deception_keys (scope, scope_key, last_s)
        SELECT scope, scope_key, MAX(hit_s) FROM deception_hits GROUP BY scope, scope_key;
    INSERT OR IGNORE INTO deception_keys (scope, scope_key, last_s)
        SELECT 'src', source_key, MAX(audited_s) FROM deception_touches GROUP BY source_key;
    """

    def __init__(
        self, path: str | os.PathLike[str], *, clock: Callable[[], float] = time.time
    ) -> None:
        """Open (creating if needed) the tripwire store at ``path``."""
        self._path = Path(path)
        if self._path.parent and not self._path.parent.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._lock = threading.Lock()
        self._touches_since_prune = 0
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.isolation_level = None
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(self._SCHEMA)
            self._conn.executescript(self._BACKFILL)

    def close(self) -> None:
        """Close the underlying connection."""
        with self._lock:
            self._conn.close()

    def __del__(self) -> None:
        """Close on collection, so a dropped store never leaks its connection."""
        try:
            self.close()
        except Exception:  # pragma: no cover - interpreter-shutdown best effort
            pass

    def _count_scope(self, scope: str, scope_key: str, floor_s: float, max_hits: int) -> int:
        """Prune and cap one (scope, key)'s hits; return its window count."""
        self._conn.execute(
            "DELETE FROM deception_hits WHERE scope = ? AND scope_key = ? AND hit_s <= ?",
            (scope, scope_key, floor_s),
        )
        self._conn.execute(
            "DELETE FROM deception_hits WHERE rowid IN ("
            "  SELECT rowid FROM deception_hits WHERE scope = ? AND scope_key = ?"
            "  ORDER BY hit_s DESC LIMIT -1 OFFSET ?)",
            (scope, scope_key, max_hits),
        )
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM deception_hits WHERE scope = ? AND scope_key = ?",
            (scope, scope_key),
        ).fetchone()
        return int(row["n"])

    def _touch_key(self, scope: str, scope_key: str, now: float, max_keys: int) -> None:
        """Mark a key seen now; past ``max_keys`` keys, evict the least recently seen.

        The key being recorded is never a candidate: the wall clock can step
        backwards, and a fresh key stamped earlier than every other must not
        evict itself (and its just-written hits) mid-touch.
        """
        known = self._conn.execute(
            "UPDATE deception_keys SET last_s = ? WHERE scope = ? AND scope_key = ?",
            (now, scope, scope_key),
        ).rowcount
        if known:
            return
        self._conn.execute(
            "INSERT INTO deception_keys (scope, scope_key, last_s) VALUES (?, ?, ?)",
            (scope, scope_key, now),
        )
        excess = (
            int(
                self._conn.execute(
                    "SELECT COUNT(*) AS n FROM deception_keys WHERE scope = ?", (scope,)
                ).fetchone()["n"]
            )
            - max_keys
        )
        if excess <= 0:
            return
        evicted = [
            str(row["scope_key"])
            for row in self._conn.execute(
                "SELECT scope_key FROM deception_keys WHERE scope = ? AND scope_key != ? "
                "ORDER BY last_s ASC, scope_key ASC LIMIT ?",
                (scope, scope_key, excess),
            ).fetchall()
        ]
        for old in evicted:
            self._conn.execute(
                "DELETE FROM deception_keys WHERE scope = ? AND scope_key = ?", (scope, old)
            )
            self._conn.execute(
                "DELETE FROM deception_hits WHERE scope = ? AND scope_key = ?", (scope, old)
            )
            if scope == "src":
                self._conn.execute("DELETE FROM deception_touches WHERE source_key = ?", (old,))

    def distinct_decoys(self, key: str) -> int:
        """Return the distinct decoys the source has touched on record.

        Args:
            key: The :func:`~omni_mercury_engine.security.deception.source_key`.

        Returns:
            The cluster-wide count; 0 for a source not on record (never seen,
            or evicted by the ``max_sources`` cap).

        Raises:
            sqlite3.Error: The database refused the read; the caller (the
                sensor) logs it and answers from its in-process state.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM deception_touches WHERE source_key = ?", (key,)
            ).fetchone()
        return int(row["n"])

    def record_touch(
        self,
        key: str,
        aggregate: str,
        path: str,
        *,
        window_s: float,
        max_hits: int,
        max_sources: int = 10_000,
    ) -> TouchRecord:
        """Record one touch atomically and return the cluster-wide counts.

        Args:
            key: The :func:`~omni_mercury_engine.security.deception.source_key`.
            aggregate: The matching
                :func:`~omni_mercury_engine.security.deception.aggregate_key`.
            path: The decoy path touched.
            window_s: The tarpit window (> 0).
            max_hits: Per-key hit-history cap (>= 1).
            max_sources: Cap on sources, and separately on aggregates, kept
                on record (>= 1); the least recently seen is evicted.

        Returns:
            The :class:`TouchRecord`.

        Raises:
            ValueError: ``window_s``, ``max_hits`` or ``max_sources`` out of
                range.
            sqlite3.Error: The database refused the transaction; the caller
                (the sensor) logs it and serves from its in-process state.
        """
        if not window_s > 0:
            raise ValueError("window_s must be positive")
        if max_hits < 1:
            raise ValueError("max_hits must be >= 1")
        if max_sources < 1:
            raise ValueError("max_sources must be >= 1")
        now = self._clock()
        floor_s = now - window_s
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "INSERT INTO deception_hits (scope, scope_key, hit_s) VALUES ('src', ?, ?)",
                    (key, now),
                )
                self._conn.execute(
                    "INSERT INTO deception_hits (scope, scope_key, hit_s) VALUES ('agg', ?, ?)",
                    (aggregate, now),
                )
                self._touch_key("src", key, now, max_sources)
                self._touch_key("agg", aggregate, now, max_sources)
                hits = self._count_scope("src", key, floor_s, max_hits)
                aggregate_hits = self._count_scope("agg", aggregate, floor_s, max_hits)
                touch = self._conn.execute(
                    "SELECT audited_s FROM deception_touches "
                    "WHERE source_key = ? AND decoy_path = ?",
                    (key, path),
                ).fetchone()
                first_touch = touch is None
                audit_due = first_touch or now - float(touch["audited_s"]) > window_s
                if first_touch:
                    self._conn.execute(
                        "INSERT INTO deception_touches "
                        "(source_key, decoy_path, first_s, audited_s) VALUES (?, ?, ?, ?)",
                        (key, path, now, now),
                    )
                elif audit_due:
                    self._conn.execute(
                        "UPDATE deception_touches SET audited_s = ? "
                        "WHERE source_key = ? AND decoy_path = ?",
                        (now, key, path),
                    )
                distinct = int(
                    self._conn.execute(
                        "SELECT COUNT(*) AS n FROM deception_touches WHERE source_key = ?",
                        (key,),
                    ).fetchone()["n"]
                )
                sources = int(
                    self._conn.execute(
                        "SELECT COUNT(*) AS n FROM deception_keys "
                        "WHERE scope = 'src' AND last_s > ?",
                        (floor_s,),
                    ).fetchone()["n"]
                )
                self._touches_since_prune += 1
                if self._touches_since_prune >= _FULL_PRUNE_EVERY:
                    self._touches_since_prune = 0
                    self._conn.execute("DELETE FROM deception_hits WHERE hit_s <= ?", (floor_s,))
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
        return TouchRecord(
            hits=hits,
            aggregate_hits=aggregate_hits,
            distinct_decoys=distinct,
            first_touch=first_touch,
            audit_due=audit_due,
            sources_in_window=sources,
        )


def build_tripwire_store() -> SqliteTripwireStore | None:
    """Return the shared tripwire store, or ``None`` for in-process state.

    Returns:
        A :class:`SqliteTripwireStore` on the file named by
        :data:`TRIPWIRE_STORE_PATH_ENV`; ``None`` when the variable is unset
        or blank, keeping the sensor's per-process behaviour unchanged.
    """
    path = os.getenv(TRIPWIRE_STORE_PATH_ENV, "").strip()
    if not path:
        return None
    return SqliteTripwireStore(path)
