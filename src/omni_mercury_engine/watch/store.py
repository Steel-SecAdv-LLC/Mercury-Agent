# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The watch's memory: series history, findings and feed statuses, in SQLite.

One file, three tables. Writes are small and each is its own transaction, so
a watcher that is killed loses at most the read in progress, and a reader
(the API, ``watch status``) can open the file while the watcher runs.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from omni_mercury_engine.watch.model import FeedStatus, Finding

__all__ = ["SeriesRecord", "WatchStore", "default_state_dir"]

#: Readings kept per series. At a 15-minute cadence this is about ten days.
SERIES_CAPACITY = 1000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS series (
    feed TEXT NOT NULL, subject TEXT NOT NULL, signal TEXT NOT NULL,
    readings TEXT NOT NULL, latest TEXT NOT NULL,
    PRIMARY KEY (feed, subject, signal)
);
CREATE TABLE IF NOT EXISTS findings (
    key TEXT PRIMARY KEY, feed TEXT NOT NULL, severity INTEGER NOT NULL,
    first_seen TEXT NOT NULL, last_seen TEXT NOT NULL, ended TEXT,
    body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS findings_last_seen ON findings (last_seen);
CREATE TABLE IF NOT EXISTS feeds (feed TEXT PRIMARY KEY, body TEXT NOT NULL);
"""


def default_state_dir() -> Path:
    """Where the watch keeps its memory: ``MERCURY_WATCH_DIR`` or ``~/.mercury/watch``."""
    override = os.environ.get("MERCURY_WATCH_DIR")
    return Path(override).expanduser() if override else Path.home() / ".mercury" / "watch"


class SeriesRecord:
    """What the store remembers of one series before a new read is added.

    Attributes:
        values: Past readings, oldest first.
        latest: Time of the newest reading remembered, or ``None``.
    """

    def __init__(self, values: Sequence[float], latest: datetime | None) -> None:
        """Hold ``values`` and the time of the newest of them."""
        self.values = np.asarray(values, dtype=np.float64)
        self.latest = latest

    def __len__(self) -> int:
        """How many readings are remembered."""
        return int(self.values.size)

    @property
    def median(self) -> float:
        """Median of the record; NaN when it is empty."""
        return float(np.median(self.values)) if self.values.size else float("nan")

    @property
    def spread(self) -> float:
        """Robust standard deviation (1.4826 x MAD); NaN when the record is empty."""
        if not self.values.size:
            return float("nan")
        return float(1.4826 * np.median(np.abs(self.values - np.median(self.values))))

    @property
    def maximum(self) -> float:
        """Largest reading remembered; NaN when the record is empty."""
        return float(self.values.max()) if self.values.size else float("nan")


class WatchStore:
    """SQLite-backed memory of the watch."""

    def __init__(self, directory: Path | None = None) -> None:
        """Open (and create) the store under ``directory``."""
        self.directory = directory if directory is not None else default_state_dir()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "watch.sqlite"
        self._lock = threading.Lock()
        with closing(self._connect()) as connection, connection:
            connection.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    # ------------------------------------------------------------------ series

    def remember(
        self,
        feed: str,
        subject: str,
        signal: str,
        times: Iterable[datetime],
        values: Iterable[float],
    ) -> SeriesRecord:
        """Add the readings newer than the record, and return the record as it was.

        A pull usually overlaps the one before it. Only readings later than
        the newest one remembered are added, so the same reading is never
        counted twice, and what is returned is the history a new reading
        should be judged against: it does not contain that reading.

        Args:
            feed: Feed name.
            subject: Station, gauge or other series owner.
            signal: Which quantity of the subject.
            times: Reading times (UTC), in any order.
            values: Readings, aligned with ``times``. Non-finite ones are dropped.

        Returns:
            The record before this call.
        """
        pairs = sorted(
            (time, float(value))
            for time, value in zip(times, values, strict=True)
            if np.isfinite(value)
        )
        with self._lock, closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT readings, latest FROM series WHERE feed=? AND subject=? AND signal=?",
                (feed, subject, signal),
            ).fetchone()
            past: list[float] = json.loads(row["readings"]) if row else []
            latest = datetime.fromisoformat(row["latest"]) if row else None
            before = SeriesRecord(past, latest)
            fresh = [(time, value) for time, value in pairs if latest is None or time > latest]
            if fresh:
                kept = (past + [value for _, value in fresh])[-SERIES_CAPACITY:]
                connection.execute(
                    "INSERT INTO series (feed, subject, signal, readings, latest) "
                    "VALUES (?, ?, ?, ?, ?) ON CONFLICT (feed, subject, signal) "
                    "DO UPDATE SET readings=excluded.readings, latest=excluded.latest",
                    (feed, subject, signal, json.dumps(kept), fresh[-1][0].isoformat()),
                )
        return before

    # ---------------------------------------------------------------- findings

    def report(self, finding: Finding, now: datetime) -> bool:
        """Record a finding. Returns ``True`` when it is new.

        A finding with the same key that is still open is the same finding
        seen again: its last-seen time and body are updated (a storm's
        strongest reading so far) and it is not new. One that had ended and
        recurs is new again.
        """
        with self._lock, closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT ended, severity FROM findings WHERE key=?", (finding.key,)
            ).fetchone()
            body = json.dumps(finding.to_dict())
            if row is not None and row["ended"] is None:
                escalated = int(finding.severity) > int(row["severity"])
                connection.execute(
                    "UPDATE findings SET last_seen=?, body=?, severity=? WHERE key=?",
                    (
                        now.isoformat(),
                        body,
                        max(int(finding.severity), int(row["severity"])),
                        finding.key,
                    ),
                )
                return escalated
            connection.execute(
                "INSERT OR REPLACE INTO findings "
                "(key, feed, severity, first_seen, last_seen, ended, body) "
                "VALUES (?, ?, ?, ?, ?, NULL, ?)",
                (
                    finding.key,
                    finding.feed,
                    int(finding.severity),
                    now.isoformat(),
                    now.isoformat(),
                    body,
                ),
            )
            return True

    def close_absent(self, feed: str, standing: set[str], now: datetime) -> int:
        """End the feed's ongoing findings that the latest read no longer shows.

        Args:
            feed: Feed name.
            standing: Keys of the ongoing findings the read produced.
            now: Time of the read.

        Returns:
            How many findings ended.
        """
        with self._lock, closing(self._connect()) as connection, connection:
            rows = connection.execute(
                "SELECT key, body FROM findings WHERE feed=? AND ended IS NULL", (feed,)
            ).fetchall()
            ended = [
                row["key"]
                for row in rows
                if json.loads(row["body"]).get("ongoing") and row["key"] not in standing
            ]
            connection.executemany(
                "UPDATE findings SET ended=? WHERE key=?", [(now.isoformat(), key) for key in ended]
            )
            return len(ended)

    def findings(
        self, *, limit: int = 50, open_only: bool = False, feed: str | None = None
    ) -> list[dict[str, Any]]:
        """Recorded findings, newest first, each with when it was first and last seen."""
        clauses, arguments = [], []
        if open_only:
            clauses.append("(ended IS NULL)")
        if feed is not None:
            clauses.append("feed=?")
            arguments.append(feed)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"SELECT * FROM findings {where} ORDER BY last_seen DESC, key LIMIT ?",  # noqa: S608
                (*arguments, int(limit)),
            ).fetchall()
        out = []
        for row in rows:
            body = json.loads(row["body"])
            body.update(
                first_seen=row["first_seen"], last_seen=row["last_seen"], ended=row["ended"]
            )
            out.append(body)
        return out

    def open_count(self, feed: str) -> int:
        """How many of the feed's findings are standing.

        A single occurrence (an earthquake) stands for a day after it was
        last seen; an ongoing one stands until a read no longer shows it.
        """
        horizon = datetime.now(UTC).timestamp() - 86400.0
        count = 0
        for finding in self.findings(limit=500, open_only=True, feed=feed):
            if (
                finding.get("ongoing")
                or datetime.fromisoformat(finding["last_seen"]).timestamp() >= horizon
            ):
                count += 1
        return count

    # ------------------------------------------------------------------- feeds

    def save_status(self, status: FeedStatus) -> None:
        """Record what happened when a feed was last read."""
        with self._lock, closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT INTO feeds (feed, body) VALUES (?, ?) "
                "ON CONFLICT (feed) DO UPDATE SET body=excluded.body",
                (status.feed, json.dumps(status.to_dict())),
            )

    def statuses(self) -> list[dict[str, Any]]:
        """The last recorded status of every feed that has been read."""
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT body FROM feeds ORDER BY feed").fetchall()
        return [json.loads(row["body"]) for row in rows]
