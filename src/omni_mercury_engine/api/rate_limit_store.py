# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared, persistent rate-limit state + per-action auth throttles.

Two gaps make an in-process rate limiter bypassable in a real deployment:

* **Per-worker state.** With N uvicorn workers an attacker gets N times the
  limit, and a restart (or a deploy) resets every bucket to full.
* **No per-action limits.** A global 100 req/min ceiling still allows 100
  password guesses a minute against one account, 100 signups a minute from
  one address, and an unbounded stream of reset emails.

This module closes both. :class:`SqliteRateLimitBackend` keeps token-bucket
state in the shared SQLite file (``MERCURY_KEYSTORE_PATH``) so the *same*
bucket is decremented no matter which worker serves the request and the state
survives restarts; consumption is a single ``BEGIN IMMEDIATE`` transaction so
two workers cannot both spend the last token. :class:`ActionRateLimiter`
layers named fixed-window counters on the same storage for the sensitive auth
actions (login, register, password-reset, resend-verification), keyed per-IP
and per-account, with limits that are configuration
(``MERCURY_AUTH_RATE_<ACTION>`` = ``"<max>/<window-seconds>"``), not code.

Without ``MERCURY_KEYSTORE_PATH`` both fall back to in-memory state — the
existing solo/self-host behaviour, where there is only one process anyway.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

__all__ = [
    "RATE_LIMIT_PATH_ENV",
    "ActionRateLimiter",
    "ActionRule",
    "InMemoryCounterStore",
    "SqliteCounterStore",
    "SqliteRateLimitBackend",
    "build_action_rate_limiter",
    "build_shared_bucket_backend",
]

#: The shared SQLite file (same variable as the key/identity/usage stores, so
#: one file backs all platform state). Unset selects in-memory counters.
RATE_LIMIT_PATH_ENV = "MERCURY_KEYSTORE_PATH"


# --------------------------------------------------------------------------- #
# token-bucket backend (plugs into security.rate_limiting.RateLimiter)
# --------------------------------------------------------------------------- #
class SqliteRateLimitBackend:
    """Cross-worker, restart-persistent token-bucket state in SQLite.

    Satisfies the :class:`~omni_mercury_engine.security.rate_limiting.
    RateLimitBackend` protocol (``get``/``set``/``delete``) and additionally
    provides :meth:`consume_token`, an *atomic* refill-and-spend that the
    unified ``RateLimiter`` prefers when present. The plain ``get``/``set``
    pair is inherently read-modify-write and therefore racy across processes;
    ``consume_token`` does the whole operation inside one ``BEGIN IMMEDIATE``
    transaction, so concurrent workers serialize on the database write lock
    and the bucket can never be double-spent.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS rate_buckets (
        bucket_key TEXT PRIMARY KEY,
        last_time  REAL NOT NULL,
        tokens     REAL NOT NULL
    );
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        """Open (creating if needed) the bucket store at ``path``."""
        self._path = Path(path)
        if self._path.parent and not self._path.parent.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Autocommit mode: transactions are opened explicitly (BEGIN IMMEDIATE)
        # so the atomic sections are exactly the ones marked as such.
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

    # -- RateLimitBackend protocol surface -------------------------------- #
    def get(self, key: str) -> tuple[float, int] | None:
        """Return ``(last_update_time, tokens)`` for ``key``, or ``None``."""
        with self._lock:
            row = self._conn.execute(
                "SELECT last_time, tokens FROM rate_buckets WHERE bucket_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return float(row["last_time"]), int(row["tokens"])

    def set(self, key: str, last_time: float, tokens: int, ttl: int) -> None:
        """Upsert bucket state (``ttl`` is advisory; pruning is time-based)."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO rate_buckets (bucket_key, last_time, tokens) "
                "VALUES (?, ?, ?) ON CONFLICT(bucket_key) DO UPDATE SET "
                "last_time = excluded.last_time, tokens = excluded.tokens",
                (key, last_time, tokens),
            )

    def delete(self, key: str) -> None:
        """Delete one bucket."""
        with self._lock:
            self._conn.execute("DELETE FROM rate_buckets WHERE bucket_key = ?", (key,))

    # -- atomic extension -------------------------------------------------- #
    def consume_token(
        self,
        key: str,
        *,
        refill_rate: float,
        burst: int,
        now: float,
    ) -> tuple[bool, float]:
        """Atomically refill ``key``'s bucket and spend one token if available.

        Args:
            key: Bucket identifier.
            refill_rate: Tokens added per second.
            burst: Bucket capacity.
            now: Current UNIX time (injected for deterministic tests).

        Returns:
            ``(allowed, tokens_remaining)`` — ``tokens_remaining`` is the
            balance *after* the spend (or the unspendable balance on deny).
        """
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT last_time, tokens FROM rate_buckets WHERE bucket_key = ?",
                    (key,),
                ).fetchone()
                if row is None:
                    tokens = float(burst)
                else:
                    elapsed = max(0.0, now - float(row["last_time"]))
                    tokens = min(float(burst), float(row["tokens"]) + elapsed * refill_rate)
                allowed = tokens >= 1.0
                if allowed:
                    tokens -= 1.0
                self._conn.execute(
                    "INSERT INTO rate_buckets (bucket_key, last_time, tokens) "
                    "VALUES (?, ?, ?) ON CONFLICT(bucket_key) DO UPDATE SET "
                    "last_time = excluded.last_time, tokens = excluded.tokens",
                    (key, now, tokens),
                )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
        return allowed, tokens

    def prune_stale(self, older_than: float) -> int:
        """Delete buckets not touched since ``older_than``; return the count."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM rate_buckets WHERE last_time < ?", (older_than,))
        return int(cur.rowcount)


# --------------------------------------------------------------------------- #
# fixed-window counters (per-action auth throttles)
# --------------------------------------------------------------------------- #
@runtime_checkable
class CounterStore(Protocol):
    """Contract for atomic fixed-window counter storage."""

    def increment(self, key: str, window_start: int) -> int:
        """Atomically bump ``key``'s counter for ``window_start``; return it.

        A stored row from an older window is reset to 1 for the new window in
        the same atomic step.
        """
        ...

    def prune_stale(self, before_window_start: int) -> int:
        """Delete counters from windows older than ``before_window_start``."""
        ...


class InMemoryCounterStore:
    """Process-local counter store (dev/test default; not shared)."""

    def __init__(self) -> None:
        """Initialise the empty counter map."""
        self._counters: dict[str, tuple[int, int]] = {}  # key -> (window_start, count)
        self._lock = threading.Lock()

    def increment(self, key: str, window_start: int) -> int:
        """Atomically bump (or window-reset) ``key``'s counter."""
        with self._lock:
            stored = self._counters.get(key)
            if stored is None or stored[0] != window_start:
                self._counters[key] = (window_start, 1)
                return 1
            count = stored[1] + 1
            self._counters[key] = (window_start, count)
            return count

    def prune_stale(self, before_window_start: int) -> int:
        """Delete counters from windows older than ``before_window_start``."""
        with self._lock:
            stale = [k for k, (ws, _) in self._counters.items() if ws < before_window_start]
            for key in stale:
                del self._counters[key]
            return len(stale)


class SqliteCounterStore:
    """Durable, cross-worker counter store on the shared SQLite file.

    ``increment`` runs inside ``BEGIN IMMEDIATE`` so two workers bumping the
    same key serialize on the write lock — the counter can never under-count
    (the failure mode that would let racing login attempts slip past the
    limit).
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS rate_counters (
        counter_key  TEXT PRIMARY KEY,
        window_start INTEGER NOT NULL,
        count        INTEGER NOT NULL
    );
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        """Open (creating if needed) the counter store at ``path``."""
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

    def increment(self, key: str, window_start: int) -> int:
        """Atomically bump (or window-reset) ``key``'s counter; return it."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "INSERT INTO rate_counters (counter_key, window_start, count) "
                    "VALUES (?, ?, 1) ON CONFLICT(counter_key) DO UPDATE SET "
                    "count = CASE WHEN rate_counters.window_start = excluded.window_start "
                    "THEN rate_counters.count + 1 ELSE 1 END, "
                    "window_start = excluded.window_start",
                    (key, window_start),
                )
                row = self._conn.execute(
                    "SELECT count FROM rate_counters WHERE counter_key = ?", (key,)
                ).fetchone()
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
        return int(row["count"])

    def prune_stale(self, before_window_start: int) -> int:
        """Delete counters from windows older than ``before_window_start``."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM rate_counters WHERE window_start < ?",
                (before_window_start,),
            )
        return int(cur.rowcount)


@dataclass(frozen=True)
class ActionRule:
    """One action's fixed-window ceiling: at most ``max_attempts``/``window``."""

    max_attempts: int
    window_seconds: int


#: Default per-action rules. Values chosen against the attack each blocks:
#: online password guessing (login), signup flooding (register), reset-email
#: bombing (reset), verification-email bombing (resend), and change-email
#: bombing toward an attacker-supplied address (email_change). All are
#: overridable per deployment via ``MERCURY_AUTH_RATE_<ACTION>``.
DEFAULT_ACTION_RULES: dict[str, ActionRule] = {
    "login_ip": ActionRule(max_attempts=10, window_seconds=300),
    "login_account": ActionRule(max_attempts=10, window_seconds=900),
    "register_ip": ActionRule(max_attempts=5, window_seconds=3600),
    "reset_ip": ActionRule(max_attempts=5, window_seconds=3600),
    "reset_account": ActionRule(max_attempts=3, window_seconds=3600),
    "resend_ip": ActionRule(max_attempts=5, window_seconds=3600),
    "resend_account": ActionRule(max_attempts=3, window_seconds=3600),
    # Email change sends a confirmation link to a *new*, caller-chosen address;
    # without a ceiling a logged-in account is an open relay toward arbitrary
    # mailboxes. Keyed per IP and per acting account (mirrors the reset limits).
    "email_change_ip": ActionRule(max_attempts=5, window_seconds=3600),
    "email_change_account": ActionRule(max_attempts=3, window_seconds=3600),
}


def _rule_from_env(action: str, default: ActionRule) -> ActionRule:
    """Parse ``MERCURY_AUTH_RATE_<ACTION>`` (``"<max>/<seconds>"``) or default.

    Malformed values keep the (stricter, known-good) default rather than
    silently disabling the throttle.
    """
    raw = os.getenv(f"MERCURY_AUTH_RATE_{action.upper()}", "").strip()
    if not raw:
        return default
    try:
        max_part, window_part = raw.split("/", 1)
        max_attempts = int(max_part)
        window_seconds = int(window_part)
    except ValueError:
        return default
    if max_attempts < 1 or window_seconds < 1:
        return default
    return ActionRule(max_attempts=max_attempts, window_seconds=window_seconds)


class ActionRateLimiter:
    """Named fixed-window throttles over an atomic :class:`CounterStore`.

    ``check("login_ip", client_ip)`` counts this attempt and reports whether
    the caller is still under that action's ceiling, plus how long until the
    window resets. Counting *before* the guarded work runs is deliberate:
    failed attempts must count, and an attacker must not learn whether the
    work would have succeeded.
    """

    def __init__(
        self,
        store: CounterStore,
        rules: dict[str, ActionRule] | None = None,
        clock: object | None = None,
    ) -> None:
        """Wire the limiter to storage, rules, and an injectable clock.

        Args:
            store: Atomic counter storage (in-memory or shared SQLite).
            rules: Per-action ceilings; defaults to :data:`DEFAULT_ACTION_RULES`
                with ``MERCURY_AUTH_RATE_*`` overrides applied.
            clock: Zero-arg callable returning UNIX seconds (tests inject a
                fake; defaults to ``time.time``).
        """
        self._store = store
        self._rules = rules if rules is not None else self.rules_from_env()
        self._clock = clock if callable(clock) else time.time

    @staticmethod
    def rules_from_env() -> dict[str, ActionRule]:
        """Resolve the effective rules: defaults + environment overrides."""
        return {
            action: _rule_from_env(action, default)
            for action, default in DEFAULT_ACTION_RULES.items()
        }

    def check(self, action: str, key: str) -> tuple[bool, int]:
        """Count one attempt of ``action`` by ``key``; report allow/deny.

        Args:
            action: Rule name (e.g. ``"login_ip"``). Unknown actions allow —
                a typo must not lock every user out — but are a programming
                error caught by tests.
            key: The bucket discriminator (client IP or account id).

        Returns:
            ``(allowed, retry_after_seconds)``; ``retry_after_seconds`` is 0
            when allowed, otherwise the seconds until the window rolls over.
        """
        rule = self._rules.get(action)
        if rule is None:
            return True, 0
        now = int(self._clock())
        window_start = now - (now % rule.window_seconds)
        count = self._store.increment(f"{action}:{key}", window_start)
        if count <= rule.max_attempts:
            return True, 0
        return False, max(1, window_start + rule.window_seconds - now)


# --------------------------------------------------------------------------- #
# environment-driven builders
# --------------------------------------------------------------------------- #
def build_shared_bucket_backend() -> SqliteRateLimitBackend | None:
    """Return the shared SQLite bucket backend, or ``None`` for in-memory.

    ``None`` tells the caller to keep the unified limiter's default
    ``InMemoryBackend`` (unchanged single-process behaviour).
    """
    path = os.getenv(RATE_LIMIT_PATH_ENV, "").strip()
    if not path:
        return None
    return SqliteRateLimitBackend(path)


def build_action_rate_limiter() -> ActionRateLimiter:
    """Construct the per-action limiter on the configured counter store."""
    path = os.getenv(RATE_LIMIT_PATH_ENV, "").strip()
    store: CounterStore = SqliteCounterStore(path) if path else InMemoryCounterStore()
    return ActionRateLimiter(store)
