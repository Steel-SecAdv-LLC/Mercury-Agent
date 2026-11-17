# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The watch's memory: findings, what later records said of them, and feed statuses.

One SQLite file, two tables. Writes are small and each is its own transaction, so
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

if TYPE_CHECKING:

    from omni_mercury_engine.watch.model import FeedStatus, Finding, Outcome

__all__ = ["WatchStore", "default_state_dir"]

_SCHEMA = """
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
            # Stores written before the watch checked itself lack these columns.
            present = {row["name"] for row in connection.execute("PRAGMA table_info(findings)")}
            for column, kind in (
                ("outcome", "TEXT"),
                ("reviewed_at", "TEXT"),
                ("review_final", "INTEGER NOT NULL DEFAULT 0"),
            ):
                if column not in present:
                    connection.execute(f"ALTER TABLE findings ADD COLUMN {column} {kind}")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

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
                if escalated:
                    # What a later record said was said of the milder statement; the
                    # stronger one has not been checked.
                    connection.execute(
                        "UPDATE findings SET outcome=NULL, reviewed_at=NULL, review_final=0 "
                        "WHERE key=?",
                        (finding.key,),
                    )
                return escalated
            if row is not None:
                # It ended and has recurred. The earlier occurrence keeps its
                # own row, with its outcome, under a key that names when it
                # began; this one starts clean.
                began = connection.execute(
                    "SELECT first_seen FROM findings WHERE key=?", (finding.key,)
                ).fetchone()["first_seen"]
                connection.execute(
                    "UPDATE findings SET key=? WHERE key=?",
                    (f"{finding.key}@{began}", finding.key),
                )
            connection.execute(
                "INSERT INTO findings "
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
            out.append(self._finding(row))
        return out

    @staticmethod
    def _finding(row: sqlite3.Row) -> dict[str, Any]:
        body: dict[str, Any] = json.loads(row["body"])
        body.update(
            key=row["key"],
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
            ended=row["ended"],
            outcome=json.loads(row["outcome"]) if row["outcome"] else None,
            reviewed_at=row["reviewed_at"],
        )
        return body

    # ------------------------------------------------------------------ review

    def awaiting_review(
        self, feed: str, first_seen_before: datetime, not_since: datetime
    ) -> list[dict[str, Any]]:
        """The feed's findings that are due a check against a later record.

        Args:
            feed: Feed name.
            first_seen_before: Only findings first seen before this: a record
                needs time to exist.
            not_since: Only findings not checked since this.

        Returns:
            Findings whose outcome is not final, oldest first.
        """
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM findings WHERE feed=? AND review_final=0 AND first_seen<=? "
                "AND (reviewed_at IS NULL OR reviewed_at<=?) ORDER BY first_seen",
                (feed, first_seen_before.isoformat(), not_since.isoformat()),
            ).fetchall()
        return [self._finding(row) for row in rows]

    def record_outcome(self, key: str, outcome: Outcome, now: datetime) -> None:
        """Record what a later record says about a finding."""
        with self._lock, closing(self._connect()) as connection, connection:
            connection.execute(
                "UPDATE findings SET outcome=?, reviewed_at=?, review_final=? WHERE key=?",
                (json.dumps(outcome.to_dict()), now.isoformat(), int(outcome.final), key),
            )

    def track_record(self) -> dict[str, int]:
        """How the findings turned out: a count per verdict, and those not yet checked."""
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT outcome FROM findings").fetchall()
        tally = {"confirmed": 0, "revised": 0, "withdrawn": 0, "unconfirmed": 0, "unchecked": 0}
        for row in rows:
            verdict = json.loads(row["outcome"])["verdict"] if row["outcome"] else "unchecked"
            tally[verdict] = tally.get(verdict, 0) + 1
        return tally

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
