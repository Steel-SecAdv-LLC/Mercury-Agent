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
import re
import secrets
import sqlite3
import threading
from contextlib import closing, suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

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
CREATE TABLE IF NOT EXISTS lease (name TEXT PRIMARY KEY, holder TEXT NOT NULL, until TEXT NOT NULL);
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


#: An international phone number as E.164 writes it.
_E164 = re.compile(r"\+[1-9]\d{6,14}")


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
    return WatchStore(directory) if (directory / "watch.sqlite").exists() else None


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
            # Reading whether it is known and writing it are one step. The
            # lock above is this handle's; another process's watch read
            # "unknown" at the same moment and both inserted: one of them
            # failed on the key and its whole pass was recorded as down.
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT ended, severity FROM findings WHERE key=?", (finding.key,)
            ).fetchone()
            body = json.dumps(finding.to_dict())
            if row is not None and row["ended"] is None:
                escalated = int(finding.severity) > int(row["severity"])
                connection.execute(
                    "UPDATE findings SET last_seen=?, body=?, severity=?, ongoing=? WHERE key=?",
                    (
                        now.isoformat(),
                        body,
                        max(int(finding.severity), int(row["severity"])),
                        int(finding.ongoing),
                        finding.key,
                    ),
                )
                if escalated:
                    # What a later record said was said of the milder statement; the
                    # stronger one has not been checked.
                    connection.execute(
                        "UPDATE findings SET outcome=NULL, verdict=NULL, reviewed_at=NULL, "
                        "review_final=0 WHERE key=?",
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
                # Its deliveries go with it. Left under the bare key they read
                # as deliveries of this occurrence, and nobody who was told of
                # the first flood at a gauge was ever told of the next.
                connection.execute(
                    "UPDATE OR REPLACE deliveries SET finding_key=? WHERE finding_key=?",
                    (f"{finding.key}@{began}", finding.key),
                )
            connection.execute(
                "INSERT INTO findings "
                "(key, feed, severity, first_seen, last_seen, ended, body, ongoing) "
                "VALUES (?, ?, ?, ?, ?, NULL, ?, ?)",
                (
                    finding.key,
                    finding.feed,
                    int(finding.severity),
                    now.isoformat(),
                    now.isoformat(),
                    body,
                    int(finding.ongoing),
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
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT key FROM findings WHERE feed=? AND ended IS NULL AND ongoing=1", (feed,)
            ).fetchall()
            ended = [row["key"] for row in rows if row["key"] not in standing]
            connection.executemany(
                "UPDATE findings SET ended=? WHERE key=?", [(now.isoformat(), key) for key in ended]
            )
            return len(ended)

    #: How long a single occurrence (an earthquake) stands after it was last seen.
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
        if not _EMAIL.fullmatch(address):
            raise ValueError(f"not an email address: {email!r}")
        if not _E164.fullmatch(number):
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
                connection.execute(
                    "UPDATE phones SET min_severity=? WHERE id=?", (min_severity, row["id"])
                )
                return ""
            sends, since = 0, moment
            if row is not None:
                if row["last_send"]:
                    wait = self.CODE_GAP - (moment - datetime.fromisoformat(row["last_send"]))
                    if wait > timedelta(0):
                        raise PhoneLimitError(
                            "a code was sent a moment ago; wait a minute before asking again",
                            wait.total_seconds(),
                        )
                if row["sends_since"]:
                    began = datetime.fromisoformat(row["sends_since"])
                    if moment - began < timedelta(days=1):
                        sends, since = int(row["sends"]), began
                if sends >= self.CODE_SENDS_PER_DAY:
                    raise PhoneLimitError(
                        f"{self.CODE_SENDS_PER_DAY} codes have been sent today; try again tomorrow",
                        (since + timedelta(days=1) - moment).total_seconds(),
                    )
            code = f"{secrets.randbelow(1_000_000):06d}"
            connection.execute(
                "INSERT INTO phones (email, number, min_severity, code_hash, code_expires, "
                "code_attempts, sends, sends_since, last_send, verified_at, created) "
                "VALUES (:email, :number, :least, :hash, :expires, 0, :sends, :since, :now, "
                "NULL, :now) ON CONFLICT (email) DO UPDATE SET email=excluded.email, "
                "number=excluded.number, min_severity=excluded.min_severity, "
                "code_hash=excluded.code_hash, code_expires=excluded.code_expires, "
                "code_attempts=0, sends=excluded.sends, sends_since=excluded.sends_since, "
                "last_send=excluded.last_send, verified_at=NULL",
                {
                    "email": address,
                    "number": number,
                    "least": min_severity,
                    "hash": self._code_hash(address, number, code),
                    "expires": (moment + self.CODE_LIFETIME).isoformat(),
                    "sends": sends + 1,
                    "since": since.isoformat(),
                    "now": moment.isoformat(),
                },
            )
            return code

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
            given = self._code_hash(str(row["email"]), str(row["number"]), code.strip())
            if not hmac.compare_digest(given, str(row["code_hash"])):
                connection.execute(
                    "UPDATE phones SET code_attempts = code_attempts + 1 WHERE id=?", (row["id"],)
                )
                return False
            connection.execute(
                "UPDATE phones SET verified_at=?, code_hash=NULL, code_expires=NULL, "
                "code_attempts=0 WHERE id=?",
                (moment, row["id"]),
            )
            return True

    def phone_of(self, email: str) -> dict[str, Any]:
        """The number recorded for ``email`` and whether it is verified (``{}`` when none)."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT id, email, number, min_severity, verified_at, created FROM phones "
                "WHERE email = ? COLLATE NOCASE",
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
        """Stop texting ``number``: its owner told the provider to stop. Returns how many."""
        with self._lock, closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "UPDATE phones SET verified_at=NULL, code_hash=NULL WHERE number=?", (number,)
            )
            return int(cursor.rowcount)

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

    def standing_deliveries(self) -> dict[tuple[str, int, int, str], dict[str, Any]]:
        """The ledger for every finding that stands, in one read.

        A delivery pass asks about each standing finding for each subscriber;
        asked one at a time that is a connection per pair on every feed read.
        Only deliveries of the occurrence that stands are returned.

        Returns:
            ``(finding key, severity, subscriber, channel)`` to its status and attempts.
        """
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT d.finding_key, d.severity, d.subscriber, d.channel, d.status, "
                "d.attempts, d.at, d.error FROM deliveries AS d "
                "JOIN findings AS f ON f.key = d.finding_key "
                # A delivery made before the finding was first seen is of an
                # earlier occurrence (a store written before deliveries
                # followed their occurrence into the archive).
                "WHERE f.ended IS NULL AND d.at >= f.first_seen"
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
                # An attempt on a record left by an earlier occurrence is this one's first.
                "attempts = CASE WHEN deliveries.at < COALESCE((SELECT first_seen FROM findings "
                "WHERE key = excluded.finding_key), '') THEN 1 ELSE deliveries.attempts + 1 END",
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

    # ------------------------------------------------------------------- lease

    def hold_lease(self, holder: str, now: datetime, ttl: timedelta) -> bool:
        """Take or renew the watch's lease for ``holder`` until ``now + ttl``.

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
                "INSERT INTO lease (name, holder, until) VALUES ('watch', :holder, :until) "
                "ON CONFLICT (name) DO UPDATE SET holder = excluded.holder, until = excluded.until "
                "WHERE lease.holder = excluded.holder OR lease.until <= :now",
                {"holder": holder, "until": (now + ttl).isoformat(), "now": now.isoformat()},
            )
            return cursor.rowcount == 1

    def release_lease(self, holder: str) -> None:
        """Give the lease up, if ``holder`` has it, so another watch can take it at once."""
        with self._lock, closing(self._connect()) as connection, connection:
            connection.execute("DELETE FROM lease WHERE name = 'watch' AND holder = ?", (holder,))

    def lease(self) -> dict[str, str]:
        """Who holds the watch's lease and until when (``{}`` when no one does)."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT holder, until FROM lease WHERE name = 'watch'"
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
