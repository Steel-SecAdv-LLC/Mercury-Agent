# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The watch's memory: findings, what later records said of them, and feed statuses.

One SQLite file, two tables. Writes are small and each is its own transaction, so
a watcher that is killed loses at most the read in progress, and a reader
(the API, ``watch status``) can open the file while the watcher runs.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
from contextlib import closing, suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from omni_mercury_engine.utils.email_address import is_email_address
from omni_mercury_engine.watch.model import most_severe_first
from omni_mercury_engine.watch.sms import E164

if TYPE_CHECKING:

    from omni_mercury_engine.watch.model import FeedStatus, Finding, Outcome

__all__ = [
    "PhoneLimitError",
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
CREATE TABLE IF NOT EXISTS phones (
    id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT NOT NULL UNIQUE COLLATE NOCASE,
    number TEXT NOT NULL, min_severity INTEGER NOT NULL DEFAULT 2,
    code_hash TEXT, code_expires TEXT, code_attempts INTEGER NOT NULL DEFAULT 0,
    sends INTEGER NOT NULL DEFAULT 0, sends_since TEXT, last_send TEXT,
    verified_at TEXT, created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reach (
    subject TEXT PRIMARY KEY, body TEXT NOT NULL, read_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS lease (name TEXT PRIMARY KEY, holder TEXT NOT NULL, until TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS announcements (
    id INTEGER PRIMARY KEY AUTOINCREMENT, finding_key TEXT NOT NULL, at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS code_sends (
    subject TEXT PRIMARY KEY, sends INTEGER NOT NULL, since TEXT NOT NULL, last TEXT
);
CREATE TABLE IF NOT EXISTS deliveries (
    finding_key TEXT NOT NULL, severity INTEGER NOT NULL, subscriber INTEGER NOT NULL,
    channel TEXT NOT NULL, status TEXT NOT NULL, attempts INTEGER NOT NULL, at TEXT NOT NULL,
    error TEXT, PRIMARY KEY (finding_key, severity, subscriber, channel)
);
CREATE TABLE IF NOT EXISTS marks (name TEXT PRIMARY KEY, n INTEGER NOT NULL);
INSERT OR IGNORE INTO marks (name, n) VALUES ('told', 0);
CREATE TRIGGER IF NOT EXISTS told_subscriber_added AFTER INSERT ON subscribers
BEGIN UPDATE marks SET n = n + 1 WHERE name = 'told'; END;
CREATE TRIGGER IF NOT EXISTS told_subscriber_gone AFTER DELETE ON subscribers
BEGIN UPDATE marks SET n = n + 1 WHERE name = 'told'; END;
CREATE TRIGGER IF NOT EXISTS told_subscriber_changed
AFTER UPDATE OF email, latitude, longitude, radius_km, min_severity ON subscribers
WHEN OLD.email IS NOT NEW.email OR OLD.latitude IS NOT NEW.latitude
  OR OLD.longitude IS NOT NEW.longitude OR OLD.radius_km IS NOT NEW.radius_km
  OR OLD.min_severity IS NOT NEW.min_severity
BEGIN UPDATE marks SET n = n + 1 WHERE name = 'told'; END;
CREATE TRIGGER IF NOT EXISTS told_phone_changed
AFTER UPDATE OF email, number, min_severity, verified_at ON phones
WHEN OLD.email IS NOT NEW.email OR OLD.number IS NOT NEW.number
  OR OLD.min_severity IS NOT NEW.min_severity OR OLD.verified_at IS NOT NEW.verified_at
BEGIN UPDATE marks SET n = n + 1 WHERE name = 'told'; END;
CREATE TRIGGER IF NOT EXISTS told_phone_gone AFTER DELETE ON phones
BEGIN UPDATE marks SET n = n + 1 WHERE name = 'told'; END;
CREATE TRIGGER IF NOT EXISTS told_reach_read AFTER INSERT ON reach
BEGIN UPDATE marks SET n = n + 1 WHERE name = 'told'; END;
CREATE TRIGGER IF NOT EXISTS told_reach_changed AFTER UPDATE OF body ON reach
WHEN OLD.body IS NOT NEW.body
BEGIN UPDATE marks SET n = n + 1 WHERE name = 'told'; END;
"""


_handles: dict[Path, WatchStore] = {}
_handles_guard = threading.Lock()


def open_store(directory: Path | None = None) -> WatchStore:
    """This process's handle on the store in ``directory``, made once.

    Making a :class:`WatchStore` creates its directory, runs the schema,
    migrates old columns and sets the file's mode. That is right once in a
    process and was being done on every request for the public snapshot.
    """
    where = (directory if directory is not None else default_state_dir()).resolve()
    with _handles_guard:
        held = _handles.get(where)
        if held is None or not held.path.exists():
            held = _handles[where] = WatchStore(where)
        return held


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


class PhoneLimitError(ValueError):
    """A verification code was refused: too many were asked for, or too soon.

    Attributes:
        retry_after: Seconds until another may be asked for.
    """

    def __init__(self, message: str, retry_after: float) -> None:
        """Record why, and how long to wait."""
        super().__init__(message)
        self.retry_after = retry_after


def existing_store() -> WatchStore | None:
    """The watch's store, when this deployment has one. Never creates it."""
    directory = default_state_dir()
    return open_store(directory) if (directory / "watch.sqlite").exists() else None


def address_entry(address: str) -> int:
    """An address's entry in the delivery ledger: its own, whatever places it holds.

    Negative, so it is never a subscription's id, a phone's, or the webhook's 0.
    """
    digest = hashlib.sha256(b"address:" + address.strip().lower().encode()).digest()
    return -(int.from_bytes(digest[:7], "big") + 1)


def forget_address(email: str) -> None:
    """End every subscription of ``email``, and forget its phone: its account is being deleted."""
    store = existing_store()
    if store is not None:
        store.remove_subscriber(email)
        store.remove_phone(email)


def carry_address(previous: str, current: str) -> None:
    """Carry ``previous``'s subscriptions to ``current``: its account changed address."""
    store = existing_store()
    if store is not None:
        store.move_subscriber(previous, current)
        store.move_phone(previous, current)


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
            # One process at a time. Every worker of ``serve --workers 4`` opens
            # the store as it starts; two that both found a column missing both
            # added it, the second failed on "duplicate column name", and that
            # worker served without a watch until it was restarted.
            connection.execute("BEGIN IMMEDIATE")
            # Stores written before the watch checked itself lack these columns.
            present = {row["name"] for row in connection.execute("PRAGMA table_info(findings)")}
            for column, kind in (
                ("outcome", "TEXT"),
                ("reviewed_at", "TEXT"),
                ("review_final", "INTEGER NOT NULL DEFAULT 0"),
                # Whether it can persist across reads, and how it turned out: both were
                # only inside the JSON, so "what stands" and the track record were
                # answered by parsing every finding ever recorded.
                ("ongoing", "INTEGER NOT NULL DEFAULT 0"),
                ("verdict", "TEXT"),
                # When it last became more severe than it was (or was first
                # seen): what was sent before then was sent of a milder thing.
                ("raised_at", "TEXT"),
                # The severity it was told at then, and when it was last seen
                # at that severity or above: a climb back to it within the
                # feed's episode is the same thing told, not a new one.
                ("raised_severity", "INTEGER"),
                ("peak_seen", "TEXT"),
            ):
                if column not in present:
                    connection.execute(f"ALTER TABLE findings ADD COLUMN {column} {kind}")
            if "ongoing" not in present or "verdict" not in present:
                connection.executemany(
                    "UPDATE findings SET ongoing=?, verdict=? WHERE key=?",
                    [
                        (
                            int(bool(json.loads(row["body"]).get("ongoing"))),
                            json.loads(row["outcome"]).get("verdict") if row["outcome"] else None,
                            row["key"],
                        )
                        for row in connection.execute(
                            "SELECT key, body, outcome FROM findings"
                        ).fetchall()
                    ],
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS findings_standing "
                "ON findings (feed, ongoing, last_seen) WHERE ended IS NULL"
            )
            held = {row["name"] for row in connection.execute("PRAGMA table_info(phones)")}
            for column, kind in (("pending_number", "TEXT"), ("pending_min_severity", "INTEGER")):
                if column not in held:
                    connection.execute(f"ALTER TABLE phones ADD COLUMN {column} {kind}")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    # ---------------------------------------------------------------- findings

    def report(
        self, finding: Finding, now: datetime, same_episode: timedelta = timedelta(0)
    ) -> bool:
        """Record a finding. Returns ``True`` when it is new.

        A finding with the same key that is still open is the same finding
        seen again: its last-seen time and body are updated (a storm's
        strongest reading so far) and it is not new. One that ended less
        than ``same_episode`` ago is reopened as the same finding. One that
        ended longer ago and recurs is new again.

        Args:
            finding: What the feed found.
            now: Time of the read.
            same_episode: How soon after clearing a return is still the
                same episode. It is the feed's to say, because it is a
                property of what the feed measures: a tide gauge floods
                again at the next high tide, a storm index dips for one
                interval (:attr:`Feed.same_episode
                <omni_mercury_engine.watch.feeds.Feed.same_episode>`). The
                default merges nothing.
        """
        with self._lock, closing(self._connect()) as connection, connection:
            # Reading whether it is known and writing it are one step. The
            # lock above is this handle's; another process's watch read
            # "unknown" at the same moment and both inserted: one of them
            # failed on the key and its whole pass was recorded as down.
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT ended, severity, raised_severity, peak_seen FROM findings WHERE key=?",
                (finding.key,),
            ).fetchone()
            body = json.dumps(finding.to_dict())
            # Open, or ended so recently that this is the same episode: a gauge
            # hovering at its flood stage, a storm dipping under its threshold
            # for an hour. It is the same finding seen again, not a new one to
            # tell everyone about each time it crosses back.
            resumed = row is not None and (
                row["ended"] is None or now - datetime.fromisoformat(row["ended"]) < same_episode
            )
            if row is not None and resumed:
                # More severe than it is now, not than it has ever been. The
                # row used to keep the highest severity it had reached, so a
                # volcano that went from WARNING to ADVISORY and, months
                # later, back to WARNING was never announced the second time.
                #
                # And more severe than it was told at, unless that was an
                # episode ago. A gauge wavering between moderate and major
                # climbed "back" at every other read, and each climb was told
                # again to everyone: the feed's own window for "the same
                # episode" now covers a return to a severity as it covers a
                # return of the finding.
                severity = int(finding.severity)
                told_at = int(row["raised_severity"] or row["severity"])
                at_peak = row["peak_seen"]
                lately = at_peak is not None and (
                    now - datetime.fromisoformat(at_peak) < same_episode
                )
                escalated = severity > int(row["severity"]) and (severity > told_at or not lately)
                connection.execute(
                    "UPDATE findings SET ended=NULL, last_seen=?, body=?, severity=?, ongoing=? "
                    "WHERE key=?",
                    (now.isoformat(), body, severity, int(finding.ongoing), finding.key),
                )
                if escalated:
                    connection.execute(
                        "UPDATE findings SET raised_at=?, raised_severity=? WHERE key=?",
                        (now.isoformat(), severity, finding.key),
                    )
                if severity >= (severity if escalated else told_at):
                    connection.execute(
                        "UPDATE findings SET peak_seen=? WHERE key=?",
                        (now.isoformat(), finding.key),
                    )
                if escalated:
                    # What a later record said was said of the milder statement; the
                    # stronger one has not been checked.
                    connection.execute(
                        "UPDATE findings SET outcome=NULL, verdict=NULL, reviewed_at=NULL, "
                        "review_final=0 WHERE key=?",
                        (finding.key,),
                    )
                    self._announce(connection, finding.key, now)
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
                # Its deliveries go with it. Left under the bare key they read
                # as deliveries of this occurrence, and nobody who was told of
                # the first flood at a gauge was ever told of the next.
                connection.execute(
                    "UPDATE OR REPLACE deliveries SET finding_key=? WHERE finding_key=?",
                    (f"{finding.key}@{began}", finding.key),
                )
            connection.execute(
                "INSERT INTO findings (key, feed, severity, first_seen, last_seen, ended, body, "
                "ongoing, raised_at, raised_severity, peak_seen) "
                "VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?)",
                (
                    finding.key,
                    finding.feed,
                    int(finding.severity),
                    now.isoformat(),
                    now.isoformat(),
                    body,
                    int(finding.ongoing),
                    now.isoformat(),
                    int(finding.severity),
                    now.isoformat(),
                ),
            )
            self._announce(connection, finding.key, now)
            return True

    # ----------------------------------------------------------- announcements

    #: How long an announcement is kept for a process that has yet to read it.
    ANNOUNCEMENTS_KEPT: ClassVar[timedelta] = timedelta(days=1)

    def _announce(self, connection: sqlite3.Connection, key: str, now: datetime) -> None:
        """Note, in the transaction that records it, that a finding is new or more severe.

        Every process that serves consoles reads these and tells its own
        (:meth:`announcements_after`). The watch used to tell the consoles
        itself, on the alert bus of the one process that read the feed: with
        four workers three of them never heard, and with ``mercury-agent
        watch`` running beside a server none did.
        """
        connection.execute(
            "DELETE FROM announcements WHERE at < ?",
            ((now - self.ANNOUNCEMENTS_KEPT).isoformat(),),
        )
        connection.execute(
            "INSERT INTO announcements (finding_key, at) VALUES (?, ?)", (key, now.isoformat())
        )

    def last_announcement(self) -> int:
        """The newest announcement's number (0 when there is none)."""
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT MAX(id) AS newest FROM announcements").fetchone()
        return int(row["newest"] or 0)

    def announcements_after(self, number: int) -> list[tuple[int, dict[str, Any]]]:
        """Findings announced since announcement ``number``, oldest first, each with its number."""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT a.id AS number, f.* FROM announcements AS a "
                "JOIN findings AS f ON f.key = a.finding_key WHERE a.id > ? ORDER BY a.id",
                (number,),
            ).fetchall()
        return [(int(row["number"]), self._finding(row)) for row in rows]

    def close_absent(
        self,
        feed: str,
        standing: set[str],
        now: datetime,
        unchecked_places: frozenset[int] = frozenset(),
        listed: frozenset[str] | None = None,
    ) -> int:
        """End the feed's ongoing findings that the latest read no longer shows.

        Args:
            feed: Feed name.
            standing: Keys of the ongoing findings the read produced.
            now: Time of the read.
            unchecked_places: Subscribed places the read could not look at. A
                finding in effect at one of them is left standing: the read
                did not show it because it could not look, not because it
                has ended.
            listed: Keys of everything the source itself still has in effect,
                when it gives such a list. A finding at an unchecked place is
                kept only while it is on it.

        Returns:
            How many findings ended.
        """
        with self._lock, closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT key, body FROM findings WHERE feed=? AND ended IS NULL AND ongoing=1",
                (feed,),
            ).fetchall()
            ended = [
                row["key"]
                for row in rows
                if row["key"] not in standing
                and not (
                    unchecked_places
                    and (listed is None or row["key"] in listed)
                    and unchecked_places.intersection(
                        json.loads(row["body"]).get("place_ids") or ()
                    )
                )
            ]
            connection.executemany(
                "UPDATE findings SET ended=? WHERE key=?", [(now.isoformat(), key) for key in ended]
            )
            return len(ended)

    #: How long a single occurrence (an earthquake) stands after it was last seen.
    #: While it stands it is news, a new subscriber is told of it, and USGS's
    #: account of where it was felt is still read. Measured on the 385
    #: earthquakes of magnitude 4.5 and above between 2026-09-01 and 2026-11-16
    #: for which USGS published either product (research/watch_thresholds): 91%
    #: of first ShakeMaps come within an hour; of the earthquakes with ten or
    #: more "Did You Feel It?" reports, 58% had nine in ten of their reports
    #: by 12 hours, 85% by 24 hours and 86% by 48. A day is where it levels off.
    OCCURRENCE_STANDS: ClassVar[timedelta] = timedelta(days=1)

    def findings(
        self,
        *,
        limit: int = 50,
        open_only: bool = False,
        feed: str | None = None,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Recorded findings, newest first, each with when it was first and last seen.

        An ongoing finding (a flood) stands until a read no longer shows it.
        A single occurrence (an earthquake) is never ended by a read; it
        stands for :attr:`OCCURRENCE_STANDS` after it was last seen and is
        then reported as ended at that moment. It was reported as standing
        for as long as the store kept it: the page called a three-month-old
        earthquake "standing", and a new subscriber was sent every one.

        Args:
            limit: Most findings to return.
            open_only: Only findings that stand at ``now``.
            feed: Only this feed's.
            now: The moment to judge standing at; defaults to the present.
        """
        horizon = ((now or datetime.now(UTC)) - self.OCCURRENCE_STANDS).isoformat()
        # One fixed statement: a filter that is not asked for is switched off by
        # its own bound parameter, so no part of the SQL is ever assembled.
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM findings"
                " WHERE (:open_only = 0"
                " OR (ended IS NULL AND (ongoing = 1 OR last_seen >= :horizon)))"
                " AND (:feed IS NULL OR feed = :feed)"
                " ORDER BY last_seen DESC, key LIMIT :limit",
                {
                    "open_only": int(bool(open_only)),
                    "horizon": horizon,
                    "feed": feed,
                    "limit": int(limit),
                },
            ).fetchall()
        return [self._finding(row, horizon) for row in rows]

    def standing(
        self, *, now: datetime | None = None, limit: int = 1000, feed: str | None = None
    ) -> list[dict[str, Any]]:
        """The findings that stand at ``now``, most recently seen first."""
        return self.findings(limit=limit, open_only=True, feed=feed, now=now)

    def overview(
        self, *, limit: int = 50, open_only: bool = False, now: datetime | None = None
    ) -> list[dict[str, Any]]:
        """What to show someone: everything that stands, then what ended most recently.

        The newest ``limit`` rows by last sighting is not that. A river in
        flood is seen again on every read, so fifty gauges pushed a magnitude
        7 earthquake of two hours ago off the page, and the command's first
        line said "50 findings standing" of 181. What stands comes first, the
        most severe first and the newest first within a severity; what has
        ended fills what room is left. ``limit`` is applied by the caller,
        after anything it must leave out.
        """
        standing = most_severe_first(self.standing(now=now))
        if open_only:
            return standing
        held = {str(finding["key"]) for finding in standing}
        ended = [
            finding
            for finding in self.findings(limit=limit + len(standing), now=now)
            if str(finding["key"]) not in held
        ]
        return standing + ended

    @classmethod
    def _finding(cls, row: sqlite3.Row, horizon: str | None = None) -> dict[str, Any]:
        """A stored finding as a mapping; with ``horizon``, a lapsed occurrence has ended."""
        ended = row["ended"]
        if horizon is not None and ended is None and not row["ongoing"]:
            if row["last_seen"] < horizon:
                ended = (
                    datetime.fromisoformat(row["last_seen"]) + cls.OCCURRENCE_STANDS
                ).isoformat()
        body: dict[str, Any] = json.loads(row["body"])
        body.update(
            key=row["key"],
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
            ended=ended,
            outcome=json.loads(row["outcome"]) if row["outcome"] else None,
            reviewed_at=row["reviewed_at"],
            raised_at=row["raised_at"] or row["first_seen"],
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
                "UPDATE findings SET outcome=?, verdict=?, reviewed_at=?, review_final=? "
                "WHERE key=?",
                (
                    json.dumps(outcome.to_dict()),
                    outcome.verdict,
                    now.isoformat(),
                    int(outcome.final),
                    key,
                ),
            )

    def track_record(self) -> dict[str, int]:
        """How the findings turned out: a count per verdict, and those not yet checked."""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT COALESCE(verdict, 'unchecked') AS verdict, COUNT(*) AS n "
                "FROM findings GROUP BY 1"
            ).fetchall()
        tally = {"confirmed": 0, "revised": 0, "withdrawn": 0, "unconfirmed": 0, "unchecked": 0}
        tally.update({str(row["verdict"]): int(row["n"]) for row in rows})
        return tally

    def subscriber_count(self) -> int:
        """How many subscriptions there are."""
        with closing(self._connect()) as connection:
            return int(connection.execute("SELECT COUNT(*) FROM subscribers").fetchone()[0])

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
        if not is_email_address(address):
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
            self._forget_deliveries_if_gone(connection, email)
            return int(cursor.rowcount)

    @staticmethod
    def _forget_deliveries_if_gone(connection: sqlite3.Connection, email: str) -> None:
        """Drop the record of what an address was sent, once it holds no subscription.

        The record is kept under a number made from the address. Left behind,
        it said which findings an address that had left was sent, to anyone
        who could try an address against the store; and an address that came
        back was not told what stood, because it "had been".
        """
        held = connection.execute(
            "SELECT 1 FROM subscribers WHERE email = ? COLLATE NOCASE LIMIT 1", (email.strip(),)
        ).fetchone()
        if held is None:
            connection.execute(
                "DELETE FROM deliveries WHERE subscriber = ? AND channel = 'email'",
                (address_entry(email),),
            )

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
            self._forget_deliveries_if_gone(connection, email)
            return cursor.rowcount == 1

    def move_subscriber(self, previous: str, current: str) -> int:
        """Carry an address's subscriptions to the address that replaces it.

        A place the new address is already subscribed to keeps the new
        address's settings; the old one's copy ends. Returns how many moved.

        Raises:
            ValueError: If ``current`` is not an email address.
        """
        old, new = previous.strip(), current.strip()
        if not is_email_address(new):
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
            # What the old address was sent, the new one has been sent: the
            # record is kept under the address, and left under the old one
            # everything standing went again to the new.
            # Where both addresses hold a record of one finding, "sent" stands:
            # replaced by the old address's failed attempt it went out again.
            moved = {"new": address_entry(new), "old": address_entry(old)}
            if moved["new"] == moved["old"]:
                # The same address written another way (a change of case):
                # the record is already under it, and deleted as "the old
                # address's" everything standing went out again.
                return int(cursor.rowcount)
            connection.execute(
                "DELETE FROM deliveries WHERE subscriber = :new AND channel = 'email' "
                "AND status <> 'sent' AND EXISTS (SELECT 1 FROM deliveries AS was WHERE "
                "was.subscriber = :old AND was.channel = 'email' AND was.status = 'sent' "
                "AND was.finding_key = deliveries.finding_key "
                "AND was.severity = deliveries.severity)",
                moved,
            )
            connection.execute(
                "UPDATE OR IGNORE deliveries SET subscriber = :new WHERE subscriber = :old "
                "AND channel = 'email'",
                moved,
            )
            connection.execute(
                "DELETE FROM deliveries WHERE subscriber = :old AND channel = 'email'", moved
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

    # ------------------------------------------------------------------ phones

    #: A verification code is good for this long.
    CODE_LIFETIME: ClassVar[timedelta] = timedelta(minutes=10)
    #: Wrong guesses at one code before it is void.
    CODE_ATTEMPTS: ClassVar[int] = 5
    #: Codes one address may be sent in a day, and the least gap between two.
    #: A text costs the operator money and a stranger's phone its peace: a
    #: form that sends one on request is a known way to abuse both.
    CODE_SENDS_PER_DAY: ClassVar[int] = 5
    CODE_GAP: ClassVar[timedelta] = timedelta(seconds=60)

    @staticmethod
    def _code_hash(email: str, number: str, code: str) -> str:
        """What is kept of a code: never the code. Bound to the address and number."""
        return hashlib.sha256(f"{email.strip().lower()}\n{number}\n{code}".encode()).hexdigest()

    def request_phone(
        self, email: str, number: str, *, min_severity: int = 2, now: datetime | None = None
    ) -> str:
        """Record ``number`` for ``email``, unverified, and issue a code to text to it.

        A number its address has already verified keeps its verification
        when only the least severity changes; no code is issued then.

        Returns:
            The six-digit code to send to the number, or ``""`` when none is needed.

        Raises:
            PhoneLimitError: If a code was sent under a minute ago, or five in the past day.
            ValueError: If the address, the number or the severity is not valid.
        """
        address = email.strip()
        if not is_email_address(address):
            raise ValueError(f"not an email address: {email!r}")
        if not E164.fullmatch(number):
            raise ValueError("a phone number must be written like +14155550123")
        if min_severity not in (1, 2, 3):
            raise ValueError("the least severity is 1 (notable), 2 (significant) or 3 (severe)")
        moment = now or datetime.now(UTC)
        with self._lock, closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM phones WHERE email = ? COLLATE NOCASE", (address,)
            ).fetchone()
            if row is not None and row["number"] == number and row["verified_at"]:
                # The verified number again: only the severity can have changed,
                # and a change that was waiting for a code is called off.
                connection.execute(
                    "UPDATE phones SET min_severity=?, pending_number=NULL, "
                    "pending_min_severity=NULL, code_hash=NULL, code_expires=NULL WHERE id=?",
                    (min_severity, row["id"]),
                )
                return ""
            # The limit is kept for the address and for the number, each apart
            # from the phone's own row. It was kept in that row, and removing
            # the number deleted it: remove, add again, and the count began
            # at nought. Twenty-one codes went to one number in a second.
            counted = [
                self._count_code_send(connection, subject, moment)
                for subject in self._code_subjects(address, number)
            ]
            sends, since = max(counted, key=lambda count: count[0])
            code = f"{secrets.randbelow(1_000_000):06d}"
            for subject, (count, began) in zip(
                self._code_subjects(address, number), counted, strict=True
            ):
                connection.execute(
                    "INSERT INTO code_sends (subject, sends, since, last) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT (subject) DO UPDATE SET sends=excluded.sends, "
                    "since=excluded.since, last=excluded.last",
                    (subject, count + 1, began.isoformat(), moment.isoformat()),
                )
            details = {
                "email": address,
                "number": number,
                "least": min_severity,
                "hash": self._code_hash(address, number, code),
                "expires": (moment + self.CODE_LIFETIME).isoformat(),
                "sends": sends + 1,
                "since": since.isoformat(),
                "now": moment.isoformat(),
            }
            if row is not None and row["verified_at"]:
                # A different number for an address that has a verified one. The
                # verified one stays, and goes on being texted, until the new
                # one is confirmed. It used to be replaced at once: a mistyped
                # number, or a code never entered, and the flood texts stopped.
                connection.execute(
                    "UPDATE phones SET pending_number=:number, pending_min_severity=:least, "
                    "code_hash=:hash, code_expires=:expires, code_attempts=0, sends=:sends, "
                    "sends_since=:since, last_send=:now WHERE email = :email COLLATE NOCASE",
                    details,
                )
                return code
            connection.execute(
                "INSERT INTO phones (email, number, min_severity, code_hash, code_expires, "
                "code_attempts, sends, sends_since, last_send, verified_at, created) "
                "VALUES (:email, :number, :least, :hash, :expires, 0, :sends, :since, :now, "
                "NULL, :now) ON CONFLICT (email) DO UPDATE SET email=excluded.email, "
                "number=excluded.number, min_severity=excluded.min_severity, "
                "code_hash=excluded.code_hash, code_expires=excluded.code_expires, "
                "code_attempts=0, sends=excluded.sends, sends_since=excluded.sends_since, "
                "last_send=excluded.last_send, verified_at=NULL, pending_number=NULL, "
                "pending_min_severity=NULL",
                details,
            )
            return code

    @staticmethod
    def _code_subjects(email: str, number: str) -> tuple[str, str]:
        """Who a code is counted against: the asking address and the number texted.

        Kept as digests, so the count outlives the address and the number
        without the store holding either.
        """
        return (
            "address:" + hashlib.sha256(email.strip().lower().encode()).hexdigest(),
            "number:" + hashlib.sha256(number.encode()).hexdigest(),
        )

    def _count_code_send(
        self, connection: sqlite3.Connection, subject: str, moment: datetime
    ) -> tuple[int, datetime]:
        """Codes counted against ``subject`` in the current day, and when that day began.

        Raises:
            PhoneLimitError: If one was sent under a minute ago, or five in the past day.
        """
        row = connection.execute(
            "SELECT sends, since, last FROM code_sends WHERE subject = ?", (subject,)
        ).fetchone()
        if row is None:
            return 0, moment
        if row["last"]:
            wait = self.CODE_GAP - (moment - datetime.fromisoformat(row["last"]))
            if wait > timedelta(0):
                raise PhoneLimitError(
                    "a code was sent a moment ago; wait a minute before asking again",
                    wait.total_seconds(),
                )
        began = datetime.fromisoformat(row["since"])
        if moment - began >= timedelta(days=1):
            return 0, moment
        if int(row["sends"]) >= self.CODE_SENDS_PER_DAY:
            raise PhoneLimitError(
                f"{self.CODE_SENDS_PER_DAY} codes have been sent today; try again tomorrow",
                (began + timedelta(days=1) - moment).total_seconds(),
            )
        return int(row["sends"]), began

    def release_code_send(self, email: str, number: str) -> None:
        """Take back the count of a code that never left: the provider could not be reached.

        Nothing was sent, so nobody was texted and nothing was spent; the
        requester is not made to wait a minute, or charged one of the day's
        five, for it.
        """
        with self._lock, closing(self._connect()) as connection, connection:
            for subject in self._code_subjects(email, number):
                connection.execute(
                    "UPDATE code_sends SET sends = MAX(sends - 1, 0), last = NULL "
                    "WHERE subject = ?",
                    (subject,),
                )
            connection.execute(
                "UPDATE phones SET code_hash=NULL, code_expires=NULL, pending_number=NULL, "
                "pending_min_severity=NULL WHERE email = ? COLLATE NOCASE "
                "AND (number = ? OR pending_number = ?)",
                (email.strip(), number, number),
            )

    def confirm_phone(self, email: str, code: str, now: datetime | None = None) -> bool:
        """Verify ``email``'s number with the code texted to it. Returns whether it was right.

        A code is void once it has expired or been guessed wrong
        :attr:`CODE_ATTEMPTS` times; a right code is used once.
        """
        moment = (now or datetime.now(UTC)).isoformat()
        with self._lock, closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM phones WHERE email = ? COLLATE NOCASE", (email.strip(),)
            ).fetchone()
            if (
                row is None
                or not row["code_hash"]
                or row["code_expires"] <= moment
                or int(row["code_attempts"]) >= self.CODE_ATTEMPTS
            ):
                return False
            awaited = str(row["pending_number"] or row["number"])
            given = self._code_hash(str(row["email"]), awaited, code.strip())
            if not hmac.compare_digest(given, str(row["code_hash"])):
                connection.execute(
                    "UPDATE phones SET code_attempts = code_attempts + 1 WHERE id=?", (row["id"],)
                )
                return False
            connection.execute(
                "UPDATE phones SET number=?, min_severity=COALESCE(pending_min_severity, "
                "min_severity), pending_number=NULL, pending_min_severity=NULL, verified_at=?, "
                "code_hash=NULL, code_expires=NULL, code_attempts=0 WHERE id=?",
                (awaited, moment, row["id"]),
            )
            return True

    def phone_of(self, email: str) -> dict[str, Any]:
        """The number recorded for ``email`` and whether it is verified (``{}`` when none).

        ``pending_number`` is a different number its owner has asked for and
        not yet confirmed; ``number`` goes on being texted until they do.
        """
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT id, email, number, min_severity, verified_at, created, pending_number, "
                "pending_min_severity FROM phones WHERE email = ? COLLATE NOCASE",
                (email.strip(),),
            ).fetchone()
        return dict(row) if row else {}

    def verified_phones(self) -> dict[str, dict[str, Any]]:
        """Every verified number, by its address in lower case."""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT id, email, number, min_severity FROM phones WHERE verified_at IS NOT NULL"
            ).fetchall()
        return {str(row["email"]).lower(): dict(row) for row in rows}

    def remove_phone(self, email: str) -> bool:
        """Forget ``email``'s number. Returns whether there was one."""
        with self._lock, closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "DELETE FROM phones WHERE email = ? COLLATE NOCASE", (email.strip(),)
            )
            return cursor.rowcount == 1

    def unverify_phone(self, number: str) -> int:
        """Stop texting ``number``: its owner told the provider to stop. Returns how many.

        A change of number that is waiting for its code is another number's
        business and keeps its code; it was voided with the opt-out, and the
        account went on saying "texts still go to" the number that had opted
        out. A number that opts out while it is the one awaited is no longer
        awaited.
        """
        with self._lock, closing(self._connect()) as connection, connection:
            stopped = connection.execute(
                "UPDATE phones SET verified_at=NULL, code_hash=CASE WHEN pending_number IS NULL "
                "THEN NULL ELSE code_hash END WHERE number=?",
                (number,),
            )
            awaited = connection.execute(
                "UPDATE phones SET pending_number=NULL, pending_min_severity=NULL, "
                "code_hash=NULL, code_expires=NULL WHERE pending_number=?",
                (number,),
            )
            return int(stopped.rowcount) + int(awaited.rowcount)

    def move_phone(self, previous: str, current: str) -> None:
        """Carry an address's number to the address that replaces it."""
        old, new = previous.strip(), current.strip()
        with self._lock, closing(self._connect()) as connection, connection:
            if old.lower() != new.lower():
                connection.execute(
                    "DELETE FROM phones WHERE email = :new COLLATE NOCASE AND EXISTS "
                    "(SELECT 1 FROM phones WHERE email = :old COLLATE NOCASE)",
                    {"old": old, "new": new},
                )
            connection.execute(
                "UPDATE phones SET email = :new WHERE email = :old COLLATE NOCASE",
                {"old": old, "new": new},
            )

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

    def standing_deliveries(
        self, now: datetime | None = None
    ) -> dict[tuple[str, int, int, str], dict[str, Any]]:
        """The ledger for every finding that stands at ``now``, in one read.

        A delivery pass asks about each standing finding for each subscriber;
        asked one at a time that is a connection per pair on every feed read.
        Only deliveries of the occurrence that stands are returned. "Stands"
        is what :meth:`findings` means by it: a single occurrence is never
        given an end, so every delivery of every earthquake ever sent used to
        be loaded on each pass.

        Returns:
            ``(finding key, severity, subscriber, channel)`` to its status and attempts.
        """
        horizon = ((now or datetime.now(UTC)) - self.OCCURRENCE_STANDS).isoformat()
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT d.finding_key, d.severity, d.subscriber, d.channel, d.status, "
                "d.attempts, d.at, d.error FROM deliveries AS d "
                "JOIN findings AS f ON f.key = d.finding_key "
                # A delivery made before the finding was first seen is of an
                # earlier occurrence (a store written before deliveries
                # followed their occurrence into the archive).
                "WHERE f.ended IS NULL AND (f.ongoing = 1 OR f.last_seen >= ?) "
                "AND d.at >= f.first_seen",
                (horizon,),
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
                "status=excluded.status, at=excluded.at, error=excluded.error, "
                # An attempt on a record left by an earlier occurrence, or from before
                # the finding last became more severe, is this one's first.
                "attempts = CASE WHEN deliveries.at < COALESCE((SELECT COALESCE(raised_at, "
                "first_seen) FROM findings WHERE key = excluded.finding_key), '') "
                "THEN 1 ELSE deliveries.attempts + 1 END",
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

    def delivery_mark(self) -> tuple[int, tuple[Any, ...]]:
        """What a delivery pass depends on, cheaply: has anything changed that it would act on.

        Returns:
            How many deliveries last failed, and a mark that differs whenever
            who is told, or where, has changed: a subscription added, removed
            or re-saved with another place, radius or severity; a number
            verified or its severity changed; an earthquake's felt extent read
            and found different. The mark is a count the store keeps as those
            rows change (the ``told_`` triggers). Made of their sums, one
            severity raised and another lowered left it as it was; made of
            when the extents were last read, every reading changed it; made
            of the rows themselves, it read every one of them on every tick.
        """
        with closing(self._connect()) as connection:
            failed = connection.execute(
                "SELECT COUNT(*) FROM deliveries WHERE status = 'failed'"
            ).fetchone()[0]
            told = connection.execute("SELECT n FROM marks WHERE name = 'told'").fetchone()
        return int(failed), (int(told[0]) if told else 0,)

    def delivery_counts(self) -> dict[str, int]:
        """How many deliveries were sent and how many last failed."""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS n FROM deliveries GROUP BY status"
            ).fetchall()
        counts = {"sent": 0, "failed": 0}
        counts.update({str(row["status"]): int(row["n"]) for row in rows})
        return counts

    def open_count(self, feed: str, now: datetime | None = None) -> int:
        """How many of the feed's findings are standing (see :meth:`standing`)."""
        horizon = (now or datetime.now(UTC)) - self.OCCURRENCE_STANDS
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM findings WHERE feed = ? AND ended IS NULL "
                "AND (ongoing = 1 OR last_seen >= ?)",
                (feed, horizon.isoformat()),
            ).fetchone()
        return int(row[0])

    # ------------------------------------------------------------------- reach

    def save_reach(self, subject: str, reach: dict[str, Any], now: datetime) -> None:
        """Keep where an earthquake was felt, as its source said at ``now``.

        Kept beside the finding, not in it: the grid is thousands of numbers
        and the finding is what the public route serves. Reach older than a
        fortnight is dropped; its earthquake stopped standing after a day.
        """
        with self._lock, closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT INTO reach (subject, body, read_at) VALUES (?, ?, ?) "
                "ON CONFLICT (subject) DO UPDATE SET body=excluded.body, read_at=excluded.read_at",
                (subject, json.dumps(reach), now.isoformat()),
            )
            connection.execute(
                "DELETE FROM reach WHERE read_at < ?", ((now - timedelta(days=14)).isoformat(),)
            )

    def reach_read_at(self, subject: str) -> datetime | None:
        """When ``subject``'s reach was last read, or ``None`` if it never was."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT read_at FROM reach WHERE subject = ?", (subject,)
            ).fetchone()
        return datetime.fromisoformat(row["read_at"]) if row else None

    def reaches(self) -> dict[str, dict[str, Any]]:
        """Every reach kept, by its finding's subject."""
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT subject, body FROM reach").fetchall()
        return {str(row["subject"]): json.loads(row["body"]) for row in rows}

    # ------------------------------------------------------------------- lease

    def hold_lease(self, holder: str, now: datetime, ttl: timedelta, name: str = "watch") -> bool:
        """Take or renew the lease ``name`` for ``holder`` until ``now + ttl``.

        There is one lease for telling people (``"watch"``) and one for
        reading each feed (``"feed:<name>"``). There used to be only the
        first, held by whichever watch started first whatever it read: a
        ``mercury-agent watch --feed earthquakes`` took it and the server's
        watch, standing by, read none of the other eight feeds.

        Every process that opens this store may run a watch (each worker of
        ``serve --workers 4`` does, and so does ``mercury-agent watch`` beside
        a server). Unco-ordinated, each read every source and each sent every
        message. The lease is held by one at a time: it is taken when free or
        lapsed, renewed by its holder, and refused to anyone else.

        Returns:
            Whether ``holder`` holds the lease now.
        """
        with self._lock, closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "INSERT INTO lease (name, holder, until) VALUES (:name, :holder, :until) "
                "ON CONFLICT (name) DO UPDATE SET holder = excluded.holder, until = excluded.until "
                "WHERE lease.holder = excluded.holder OR lease.until <= :now",
                {
                    "name": name,
                    "holder": holder,
                    "until": (now + ttl).isoformat(),
                    "now": now.isoformat(),
                },
            )
            return cursor.rowcount == 1

    def release_lease(self, holder: str) -> None:
        """Give up every lease ``holder`` has, so another watch can take them at once."""
        with self._lock, closing(self._connect()) as connection, connection:
            connection.execute("DELETE FROM lease WHERE holder = ?", (holder,))

    def lease(self, name: str = "watch") -> dict[str, str]:
        """Who holds the lease ``name`` and until when (``{}`` when no one does)."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT holder, until FROM lease WHERE name = ?", (name,)
            ).fetchone()
        return dict(row) if row else {}

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
