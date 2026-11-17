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
import re
import sqlite3
import threading
from contextlib import closing, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:

    from omni_mercury_engine.watch.model import FeedStatus, Finding, Outcome

__all__ = [
    "SubscriptionLimitError",
    "WatchStore",
    "carry_address",
    "default_state_dir",
    "existing_store",
    "forget_address",
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS findings (
    key TEXT PRIMARY KEY, feed TEXT NOT NULL, severity INTEGER NOT NULL,
    first_seen TEXT NOT NULL, last_seen TEXT NOT NULL, ended TEXT,
    body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS findings_last_seen ON findings (last_seen);
CREATE TABLE IF NOT EXISTS feeds (feed TEXT PRIMARY KEY, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS subscribers (
    id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT NOT NULL, place TEXT NOT NULL DEFAULT '',
    latitude REAL, longitude REAL, radius_km REAL NOT NULL DEFAULT 200,
    min_severity INTEGER NOT NULL DEFAULT 1, created TEXT NOT NULL,
    UNIQUE (email, place)
);
CREATE TABLE IF NOT EXISTS deliveries (
    finding_key TEXT NOT NULL, severity INTEGER NOT NULL, subscriber INTEGER NOT NULL,
    channel TEXT NOT NULL, status TEXT NOT NULL, attempts INTEGER NOT NULL, at TEXT NOT NULL,
    error TEXT, PRIMARY KEY (finding_key, severity, subscriber, channel)
);
"""


#: An address a mail server could accept; delivery is the real test.
_EMAIL = re.compile(
    r"[^@\s<>,;\"']{1,64}@[A-Za-z0-9](?:[A-Za-z0-9.-]{0,253}[A-Za-z0-9])?\.[A-Za-z]{2,63}"
)


def default_state_dir() -> Path:
    """Where the watch keeps its memory: ``MERCURY_WATCH_DIR`` or ``~/.mercury/watch``."""
    override = os.environ.get("MERCURY_WATCH_DIR")
    return Path(override).expanduser() if override else Path.home() / ".mercury" / "watch"


class SubscriptionLimitError(ValueError):
    """A subscription was refused because a limit would be passed.

    Attributes:
        scope: ``"address"`` (the address holds as many as it may) or
            ``"places"`` (the store holds as many distinct places as it may).
        limit: The limit.
    """

    def __init__(self, scope: str, limit: int) -> None:
        """Record which limit, and its value."""
        super().__init__(f"the {scope} limit of {limit} subscriptions is reached")
        self.scope = scope
        self.limit = limit


def existing_store() -> WatchStore | None:
    """The watch's store, when this deployment has one. Never creates it."""
    directory = default_state_dir()
    return WatchStore(directory) if (directory / "watch.sqlite").exists() else None


def forget_address(email: str) -> None:
    """End every subscription of ``email``: its account is being deleted."""
    store = existing_store()
    if store is not None:
        store.remove_subscriber(email)


def carry_address(previous: str, current: str) -> None:
    """Carry ``previous``'s subscriptions to ``current``: its account changed address."""
    store = existing_store()
    if store is not None:
        store.move_subscriber(previous, current)


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
        # The store holds subscribers' addresses and places: readable by its owner only.
        with suppress(OSError):
            self.path.chmod(0o600)
        with closing(self._connect()) as connection, connection:
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
        # One fixed statement: a filter that is not asked for is switched off by
        # its own bound parameter, so no part of the SQL is ever assembled.
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM findings"
                " WHERE (:open_only = 0 OR ended IS NULL)"
                " AND (:feed IS NULL OR feed = :feed)"
                " ORDER BY last_seen DESC, key LIMIT :limit",
                {"open_only": int(bool(open_only)), "feed": feed, "limit": int(limit)},
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

    # ------------------------------------------------------------- subscribers

    def add_subscriber(
        self,
        email: str,
        *,
        place: str = "",
        latitude: float | None = None,
        longitude: float | None = None,
        radius_km: float = 200.0,
        min_severity: int = 1,
        now: datetime | None = None,
        address_limit: int | None = None,
        place_limit: int | None = None,
    ) -> int:
        """Subscribe ``email``, for one place or (with no place) for everything.

        Subscribing the same address to the same place again updates it.

        Args:
            email: The address to tell.
            place: What the subscriber calls the place.
            latitude: Where it is.
            longitude: Where it is.
            radius_km: How far from it a finding still concerns them.
            min_severity: The least severity to be told of (1 to 3).
            now: When; defaults to the present.
            address_limit: Refuse a subscription that would be the
                address's first beyond this many. ``None``: no limit.
            place_limit: Refuse one that would add a distinct place (see
                :meth:`places`) beyond this many. ``None``: no limit.

        Returns:
            The subscriber's id.

        Raises:
            SubscriptionLimitError: If a limit would be passed. The limits
                are checked in the transaction that writes, so concurrent
                requests cannot pass them together.
            ValueError: If the address, the coordinates or the radius cannot
                be what they claim to be.
        """
        address = email.strip()
        if not _EMAIL.fullmatch(address):
            raise ValueError(f"not an email address: {email!r}")
        located = latitude is not None or longitude is not None
        if located and (
            latitude is None
            or longitude is None
            or not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0)
        ):
            raise ValueError("a place needs a latitude in [-90, 90] and a longitude in [-180, 180]")
        if not 1.0 <= radius_km <= 2000.0:
            raise ValueError("the radius must be between 1 and 2,000 km")
        if min_severity not in (1, 2, 3):
            raise ValueError("the least severity is 1 (notable), 2 (significant) or 3 (severe)")
        label = place.strip() or ("" if not located else f"{latitude:.2f}, {longitude:.2f}")
        moment = (now or datetime.now(UTC)).isoformat()
        with self._lock, closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")  # the checks below and the write are one step
            # A mailbox is one subscriber however it is capitalised: the
            # table's own uniqueness compares bytes, and "A@x.org" beside
            # "a@x.org" was two rows and two copies of every message.
            row = connection.execute(
                "SELECT id, latitude, longitude FROM subscribers "
                "WHERE email = ? COLLATE NOCASE AND place = ?",
                (address, label),
            ).fetchone()
            if address_limit is not None and row is None:
                held = connection.execute(
                    "SELECT COUNT(*) AS n FROM subscribers WHERE email = ? COLLATE NOCASE",
                    (address,),
                ).fetchone()
                if int(held["n"]) >= address_limit:
                    raise SubscriptionLimitError("address", address_limit)
            if place_limit is not None and latitude is not None and longitude is not None:
                points = {
                    (round(float(other["latitude"]), 2), round(float(other["longitude"]), 2))
                    for other in connection.execute(
                        "SELECT latitude, longitude FROM subscribers "
                        "WHERE latitude IS NOT NULL AND longitude IS NOT NULL"
                    )
                }
                if (round(latitude, 2), round(longitude, 2)) not in points and (
                    len(points) >= place_limit
                ):
                    raise SubscriptionLimitError("places", place_limit)
            if row is not None:
                connection.execute(
                    "UPDATE subscribers SET email=?, latitude=?, longitude=?, radius_km=?, "
                    "min_severity=? WHERE id=?",
                    (address, latitude, longitude, float(radius_km), min_severity, row["id"]),
                )
                return int(row["id"])
            cursor = connection.execute(
                "INSERT INTO subscribers "
                "(email, place, latitude, longitude, radius_km, min_severity, created) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (address, label, latitude, longitude, float(radius_km), min_severity, moment),
            )
            return int(cursor.lastrowid or 0)

    def remove_subscriber(self, email: str, place: str | None = None) -> int:
        """Unsubscribe ``email`` from ``place``, or from everything. Returns how many ended."""
        with self._lock, closing(self._connect()) as connection, connection:
            if place is None:
                cursor = connection.execute(
                    "DELETE FROM subscribers WHERE email = ? COLLATE NOCASE", (email.strip(),)
                )
            else:
                cursor = connection.execute(
                    "DELETE FROM subscribers WHERE email = ? COLLATE NOCASE AND place = ?",
                    (email.strip(), place.strip()),
                )
            return int(cursor.rowcount)

    def subscriptions_of(self, email: str) -> list[dict[str, Any]]:
        """The subscriptions of one address, oldest first."""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM subscribers WHERE email = ? COLLATE NOCASE ORDER BY id",
                (email.strip(),),
            ).fetchall()
        return [dict(row) for row in rows]

    def remove_subscription(self, email: str, subscription: int) -> bool:
        """End one subscription, if it is ``email``'s. Returns whether it was."""
        with self._lock, closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "DELETE FROM subscribers WHERE id = ? AND email = ? COLLATE NOCASE",
                (int(subscription), email.strip()),
            )
            return cursor.rowcount == 1

    def move_subscriber(self, previous: str, current: str) -> int:
        """Carry an address's subscriptions to the address that replaces it.

        A place the new address is already subscribed to keeps the new
        address's settings; the old one's copy ends. Returns how many moved.

        Raises:
            ValueError: If ``current`` is not an email address.
        """
        old, new = previous.strip(), current.strip()
        if not _EMAIL.fullmatch(new):
            raise ValueError(f"not an email address: {current!r}")
        with self._lock, closing(self._connect()) as connection, connection:
            connection.execute(
                "DELETE FROM subscribers WHERE email = :old COLLATE NOCASE AND place IN "
                "(SELECT place FROM subscribers WHERE email = :new COLLATE NOCASE "
                "AND email <> :old COLLATE NOCASE)",
                {"old": old, "new": new},
            )
            cursor = connection.execute(
                "UPDATE subscribers SET email = :new WHERE email = :old COLLATE NOCASE",
                {"old": old, "new": new},
            )
            return int(cursor.rowcount)

    def places(self) -> dict[tuple[float, float], list[int]]:
        """The distinct places subscribed, each with its subscribers' ids.

        A place is its coordinates to two decimals (about a kilometre): the
        precision an alert lookup is made at. Oldest subscription first.
        """
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT id, latitude, longitude FROM subscribers "
                "WHERE latitude IS NOT NULL AND longitude IS NOT NULL ORDER BY id"
            ).fetchall()
        points: dict[tuple[float, float], list[int]] = {}
        for row in rows:
            point = (round(float(row["latitude"]), 2), round(float(row["longitude"]), 2))
            points.setdefault(point, []).append(int(row["id"]))
        return points

    def subscribers(self) -> list[dict[str, Any]]:
        """Every subscription, oldest first."""
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT * FROM subscribers ORDER BY id").fetchall()
        return [dict(row) for row in rows]

    # -------------------------------------------------------------- deliveries

    def delivery(self, key: str, severity: int, subscriber: int, channel: str) -> dict[str, Any]:
        """What is recorded of one delivery: its status and attempts (``{}`` when none)."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT status, attempts, at, error FROM deliveries WHERE finding_key=? AND "
                "severity=? AND subscriber=? AND channel=?",
                (key, severity, subscriber, channel),
            ).fetchone()
        return dict(row) if row else {}

    def standing_deliveries(self) -> dict[tuple[str, int, int, str], dict[str, Any]]:
        """The ledger for every finding that stands, in one read.

        A delivery pass asks about each standing finding for each subscriber;
        asked one at a time that is a connection per pair on every feed read.

        Returns:
            ``(finding key, severity, subscriber, channel)`` to its status and attempts.
        """
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT d.finding_key, d.severity, d.subscriber, d.channel, d.status, "
                "d.attempts, d.at, d.error FROM deliveries AS d "
                "JOIN findings AS f ON f.key = d.finding_key WHERE f.ended IS NULL"
            ).fetchall()
        return {
            (
                str(row["finding_key"]),
                int(row["severity"]),
                int(row["subscriber"]),
                str(row["channel"]),
            ): {name: row[name] for name in ("status", "attempts", "at", "error")}
            for row in rows
        }

    def record_delivery(
        self,
        key: str,
        severity: int,
        subscriber: int,
        channel: str,
        *,
        error: str | None,
        now: datetime,
    ) -> None:
        """Record an attempt to deliver a finding: sent, or failed with why."""
        with self._lock, closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT INTO deliveries VALUES (?, ?, ?, ?, ?, 1, ?, ?) ON CONFLICT "
                "(finding_key, severity, subscriber, channel) DO UPDATE SET "
                "status=excluded.status, attempts=attempts+1, at=excluded.at, error=excluded.error",
                (
                    key,
                    severity,
                    subscriber,
                    channel,
                    "failed" if error else "sent",
                    now.isoformat(),
                    error,
                ),
            )

    def delivery_counts(self) -> dict[str, int]:
        """How many deliveries were sent and how many last failed."""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS n FROM deliveries GROUP BY status"
            ).fetchall()
        counts = {"sent": 0, "failed": 0}
        counts.update({str(row["status"]): int(row["n"]) for row in rows})
        return counts

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
