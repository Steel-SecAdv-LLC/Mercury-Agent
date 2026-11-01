# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Durable store for the agent's detection memory.

The conversational agent answers "what was the latest anomaly?" and "how many
anomalies today?" from the retriever's detection log. That log lived only in
the process: every restart emptied it, and the first answer after a deploy was
always "no detection runs are recorded in memory yet". The export history the
log mirrors is a volatile convenience view by design (``routes/export.py``),
so nothing durable held the runs.

This module keeps the log in SQLite (the interpreter's own ``sqlite3``, the
same library the rate-limit and key stores use) when
``MERCURY_AGENT_MEMORY_PATH`` points at a file. Unset, nothing changes: the
retriever keeps its in-process list and a deployment that never asked for
durable memory behaves exactly as before. Set, the retriever loads the newest
entries at construction and appends every logged detection, so recall
survives restarts and is shared by every worker on the same path.

Only fixed statements with bound parameters are issued and no virtual table
of any kind is created, which is what the ``libsqlite3-0`` acceptance in
``.trivyignore`` rests on (``tests/security/test_sqlite_cve_acceptance_conditions.py``).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_KEEP", "ENV_VAR", "DetectionMemoryStore"]

#: Environment variable naming the SQLite file; unset means in-process only.
ENV_VAR = "MERCURY_AGENT_MEMORY_PATH"

#: Entries kept on disk and loaded at start (the retriever's own window).
DEFAULT_KEEP = 1000

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS detection_memory ("
    " id INTEGER PRIMARY KEY AUTOINCREMENT,"
    " ts REAL NOT NULL,"
    " domain TEXT,"
    " principal TEXT,"
    " detection TEXT NOT NULL)"
)
_INDEX = "CREATE INDEX IF NOT EXISTS detection_memory_ts ON detection_memory (ts)"


class DetectionMemoryStore:
    """Append-only SQLite log of detection entries, bounded to the newest ``keep``.

    Each entry is the retriever's own shape: ``{"detection": {...}, "domain":
    str | None, "principal": str | None, "timestamp": float}``. The detection
    mapping is stored as JSON; anything that is not JSON-serialisable is
    rendered with ``str`` so a record can never fail to persist.
    """

    def __init__(self, path: str | Path, keep: int = DEFAULT_KEEP) -> None:
        """Open (creating if needed) the store at ``path``."""
        self.path = Path(path).expanduser()
        self.keep = max(1, int(keep))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        with self._conn:
            self._conn.execute(_SCHEMA)
            self._conn.execute(_INDEX)

    @classmethod
    def from_env(cls, keep: int = DEFAULT_KEEP) -> DetectionMemoryStore | None:
        """The store named by :data:`ENV_VAR`, or ``None`` when it is unset or unusable.

        An unusable path (unwritable directory, a directory where a file was
        expected) is logged and disables durability for this process rather
        than taking the voice down with it.
        """
        raw = os.environ.get(ENV_VAR, "").strip()
        if not raw:
            return None
        try:
            return cls(raw, keep=keep)
        except (OSError, sqlite3.Error) as exc:
            logger.warning(
                "%s=%r is not usable (%s); the agent's detection memory stays in-process",
                ENV_VAR,
                raw,
                exc,
            )
            return None

    def append(self, entry: dict[str, Any]) -> None:
        """Persist one retriever entry and drop rows beyond ``keep``."""
        payload = json.dumps(entry.get("detection", {}), default=str, sort_keys=True)
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO detection_memory (ts, domain, principal, detection) VALUES (?, ?, ?, ?)",
                (
                    float(entry.get("timestamp") or 0.0),
                    entry.get("domain"),
                    entry.get("principal"),
                    payload,
                ),
            )
            self._conn.execute(
                "DELETE FROM detection_memory WHERE id NOT IN "
                "(SELECT id FROM detection_memory ORDER BY ts DESC, id DESC LIMIT ?)",
                (self.keep,),
            )

    def load(self, limit: int | None = None) -> list[dict[str, Any]]:
        """The newest ``limit`` entries, oldest first (the retriever's log order)."""
        n = self.keep if limit is None else max(0, int(limit))
        with self._lock:
            rows = self._conn.execute(
                "SELECT ts, domain, principal, detection FROM detection_memory "
                "ORDER BY ts DESC, id DESC LIMIT ?",
                (n,),
            ).fetchall()
        entries: list[dict[str, Any]] = []
        for ts, domain, principal, detection in reversed(rows):
            try:
                parsed = json.loads(detection)
            except ValueError:
                continue
            entries.append(
                {
                    "detection": parsed if isinstance(parsed, dict) else {},
                    "domain": domain,
                    "principal": principal,
                    "timestamp": float(ts),
                }
            )
        return entries

    def count(self) -> int:
        """Rows currently stored."""
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM detection_memory").fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        """Close the connection; further use is an error."""
        with self._lock:
            self._conn.close()
