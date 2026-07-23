# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Durable storage for user accounts, sessions, email tokens, recovery codes.

This is the persistence layer the self-service auth flows sit on. It mirrors the
key-store design: an :class:`IdentityStore` Protocol with an in-memory backend
(dev/tests) and a durable SQLite backend selected by the same
``MERCURY_KEYSTORE_PATH`` file, so a single database file holds API keys,
accounts, sessions, tokens, and usage.

Four record types are stored:

* :class:`Account` — a user: unique lower-cased email, password hash, verified /
  active flags, quota tier, and optional TOTP 2FA state (the secret is stored
  *sealed* by the auth service — see :mod:`~omni_mercury_engine.api.
  secret_sealer` — together with the last-used time step for replay rejection).
* :class:`Session` — a browser login. Only the SHA-256 of the opaque session
  token is stored (the raw token lives only in the user's cookie), so a database
  read cannot mint sessions. Carries the CSRF token hash bound to the session
  and a last-seen stamp for idle timeout.
* :class:`EmailToken` — a single-use, expiring token for email verification,
  password reset, or email change (the optional ``payload`` carries the
  pending new address). Again only the hash is stored.
* **Recovery codes** — single-use 2FA backup codes, stored only as hashes.

Nothing here sends email or hashes passwords; that is the auth service's job.
This module only persists and retrieves. The SQLite backend migrates older
database files forward in place (purely additive ``ALTER TABLE ... ADD
COLUMN`` steps), so a file created by an earlier build keeps working.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

__all__ = [
    "Account",
    "EmailToken",
    "IdentityStore",
    "InMemoryIdentityStore",
    "Session",
    "SqliteIdentityStore",
    "TokenPurpose",
    "build_identity_store",
    "hash_token",
    "identity_store_is_durable",
]

#: Environment variable naming the shared SQLite database file. Identical to the
#: key store's variable so one file backs all auth state; unset selects the
#: in-memory backend (dev/test default).
IDENTITY_PATH_ENV = "MERCURY_KEYSTORE_PATH"

#: The purposes an :class:`EmailToken` can serve.
TokenPurpose = str  # "verify" | "reset" | "email_change"


def hash_token(raw_token: str) -> str:
    """Hash an opaque high-entropy token for storage.

    Session and email tokens are 256-bit random values, so a single SHA-256 is
    the right primitive (unlike low-entropy passwords, which need a slow salted
    KDF). Only the hash is ever persisted.

    Args:
        raw_token: The raw token as issued to the user.

    Returns:
        Hex SHA-256 digest.
    """
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


@dataclass
class Account:
    """A user account."""

    id: str
    email: str
    password_hash: str
    is_verified: bool = False
    is_active: bool = True
    totp_secret: str | None = None
    totp_enabled: bool = False
    #: Last accepted TOTP time step (``timestamp // period``). A code from a
    #: step at or below this is a replay and must be rejected.
    totp_last_step: int | None = None
    #: Quota tier name resolved by the quota policy ("free" default).
    tier: str = "free"
    created_at: datetime = datetime.min


@dataclass
class Session:
    """A browser login session (only the token hash is stored)."""

    token_hash: str
    account_id: str
    created_at: datetime
    expires_at: datetime
    #: SHA-256 of the CSRF token issued alongside this session (double-submit
    #: defense-in-depth); empty for legacy rows.
    csrf_hash: str = ""
    #: Last authenticated use, for idle timeout; ``None`` on legacy rows means
    #: "treat created_at as last seen".
    last_seen_at: datetime | None = None


@dataclass
class EmailToken:
    """A single-use, expiring email token (only the token hash is stored)."""

    token_hash: str
    account_id: str
    purpose: TokenPurpose
    created_at: datetime
    expires_at: datetime
    consumed_at: datetime | None = None
    #: Purpose-specific data (the pending new address for ``email_change``).
    payload: str | None = None


@runtime_checkable
class IdentityStore(Protocol):
    """Contract for durable account/session/token/recovery-code storage."""

    def create_account(self, account: Account) -> None:
        """Persist a new account. Raises on a duplicate email."""
        ...

    def get_account_by_email(self, email: str) -> Account | None:
        """Return the account with ``email`` (case-insensitive), or ``None``."""
        ...

    def get_account_by_id(self, account_id: str) -> Account | None:
        """Return the account with ``account_id``, or ``None``."""
        ...

    def update_account(self, account: Account) -> None:
        """Persist mutable account fields (incl. email; raises on duplicate)."""
        ...

    def delete_account(self, account_id: str) -> None:
        """Hard-delete an account plus its sessions, tokens, recovery codes."""
        ...

    def iter_accounts(self) -> list[Account]:
        """Return a snapshot of all accounts (admin/migration sweeps)."""
        ...

    def create_session(self, session: Session) -> None:
        """Persist a new session."""
        ...

    def get_session(self, token_hash: str) -> Session | None:
        """Return the session for ``token_hash``, or ``None``."""
        ...

    def touch_session(self, token_hash: str, last_seen_at: datetime) -> None:
        """Update a session's last-seen stamp (idle-timeout bookkeeping)."""
        ...

    def delete_session(self, token_hash: str) -> None:
        """Delete one session (logout)."""
        ...

    def delete_sessions_for_account(self, account_id: str) -> None:
        """Delete every session for an account (logout-all / after reset)."""
        ...

    def prune_expired_sessions(self, now: datetime) -> int:
        """Delete sessions with ``expires_at <= now``; return how many."""
        ...

    def create_email_token(self, token: EmailToken) -> None:
        """Persist a new email token."""
        ...

    def get_email_token(self, token_hash: str) -> EmailToken | None:
        """Return the email token for ``token_hash``, or ``None``."""
        ...

    def consume_email_token(self, token_hash: str, consumed_at: datetime) -> None:
        """Mark an email token consumed so it cannot be reused."""
        ...

    def delete_email_tokens_for_account(self, account_id: str, purpose: str | None = None) -> None:
        """Delete an account's outstanding tokens (optionally one purpose)."""
        ...

    def prune_email_tokens(self, now: datetime) -> int:
        """Delete consumed or expired email tokens; return how many."""
        ...

    def replace_recovery_codes(
        self, account_id: str, code_hashes: list[str], created_at: datetime
    ) -> None:
        """Replace the account's recovery-code set with ``code_hashes``."""
        ...

    def consume_recovery_code(self, account_id: str, code_hash: str, used_at: datetime) -> bool:
        """Atomically consume one unused recovery code; report success."""
        ...

    def count_unused_recovery_codes(self, account_id: str) -> int:
        """Return how many recovery codes remain unused for the account."""
        ...


class DuplicateEmailError(ValueError):
    """Raised when creating an account whose email already exists."""


def _normalize_email(email: str) -> str:
    """Lower-case and strip an email for case-insensitive uniqueness."""
    return email.strip().lower()


class InMemoryIdentityStore:
    """Process-local identity store (dev/test default; not durable)."""

    def __init__(self) -> None:
        """Initialise empty in-memory maps."""
        self._accounts: dict[str, Account] = {}
        self._email_index: dict[str, str] = {}  # normalized email -> account id
        self._sessions: dict[str, Session] = {}
        self._email_tokens: dict[str, EmailToken] = {}
        self._recovery_codes: dict[str, list[dict[str, object]]] = {}
        self._lock = threading.Lock()

    def create_account(self, account: Account) -> None:
        """Persist a new account, rejecting a duplicate email."""
        key = _normalize_email(account.email)
        with self._lock:
            if key in self._email_index:
                raise DuplicateEmailError(f"email already registered: {account.email}")
            self._accounts[account.id] = account
            self._email_index[key] = account.id

    def get_account_by_email(self, email: str) -> Account | None:
        """Return the account with ``email`` (case-insensitive), or ``None``."""
        with self._lock:
            account_id = self._email_index.get(_normalize_email(email))
            return self._accounts.get(account_id) if account_id else None

    def get_account_by_id(self, account_id: str) -> Account | None:
        """Return the account with ``account_id``, or ``None``."""
        with self._lock:
            return self._accounts.get(account_id)

    def update_account(self, account: Account) -> None:
        """Replace the stored record, keeping the email index consistent.

        Raises:
            DuplicateEmailError: If the (possibly changed) email now collides
                with a different account.
        """
        new_key = _normalize_email(account.email)
        with self._lock:
            existing_owner = self._email_index.get(new_key)
            if existing_owner is not None and existing_owner != account.id:
                raise DuplicateEmailError(f"email already registered: {account.email}")
            # Reconcile by account id, not by re-reading the stored object's
            # email: callers may mutate the live stored instance in place
            # (``get_account_by_id`` returns a reference), so any prior index
            # entry pointing at this id under a different key is stale.
            for stale_key in [
                key
                for key, owner in self._email_index.items()
                if owner == account.id and key != new_key
            ]:
                del self._email_index[stale_key]
            self._accounts[account.id] = account
            self._email_index[new_key] = account.id

    def delete_account(self, account_id: str) -> None:
        """Hard-delete the account and everything hanging off it."""
        with self._lock:
            account = self._accounts.pop(account_id, None)
            if account is not None:
                self._email_index.pop(_normalize_email(account.email), None)
            self._sessions = {
                th: s for th, s in self._sessions.items() if s.account_id != account_id
            }
            self._email_tokens = {
                th: t for th, t in self._email_tokens.items() if t.account_id != account_id
            }
            self._recovery_codes.pop(account_id, None)

    def iter_accounts(self) -> list[Account]:
        """Return a snapshot of all accounts."""
        with self._lock:
            return list(self._accounts.values())

    def create_session(self, session: Session) -> None:
        """Persist a new session."""
        with self._lock:
            self._sessions[session.token_hash] = session

    def get_session(self, token_hash: str) -> Session | None:
        """Return the session for ``token_hash``, or ``None``."""
        with self._lock:
            return self._sessions.get(token_hash)

    def touch_session(self, token_hash: str, last_seen_at: datetime) -> None:
        """Update a session's last-seen stamp."""
        with self._lock:
            session = self._sessions.get(token_hash)
            if session is not None:
                self._sessions[token_hash] = replace(session, last_seen_at=last_seen_at)

    def delete_session(self, token_hash: str) -> None:
        """Delete one session."""
        with self._lock:
            self._sessions.pop(token_hash, None)

    def delete_sessions_for_account(self, account_id: str) -> None:
        """Delete every session for an account."""
        with self._lock:
            for token_hash in [
                th for th, s in self._sessions.items() if s.account_id == account_id
            ]:
                del self._sessions[token_hash]

    def prune_expired_sessions(self, now: datetime) -> int:
        """Delete sessions with ``expires_at <= now``; return how many."""
        with self._lock:
            stale = [th for th, s in self._sessions.items() if s.expires_at <= now]
            for token_hash in stale:
                del self._sessions[token_hash]
            return len(stale)

    def create_email_token(self, token: EmailToken) -> None:
        """Persist a new email token."""
        with self._lock:
            self._email_tokens[token.token_hash] = token

    def get_email_token(self, token_hash: str) -> EmailToken | None:
        """Return the email token for ``token_hash``, or ``None``."""
        with self._lock:
            return self._email_tokens.get(token_hash)

    def consume_email_token(self, token_hash: str, consumed_at: datetime) -> None:
        """Mark an email token consumed."""
        with self._lock:
            token = self._email_tokens.get(token_hash)
            if token is not None:
                token.consumed_at = consumed_at

    def delete_email_tokens_for_account(self, account_id: str, purpose: str | None = None) -> None:
        """Delete an account's outstanding tokens (optionally one purpose)."""
        with self._lock:
            self._email_tokens = {
                th: t
                for th, t in self._email_tokens.items()
                if not (t.account_id == account_id and (purpose is None or t.purpose == purpose))
            }

    def prune_email_tokens(self, now: datetime) -> int:
        """Delete consumed or expired email tokens; return how many."""
        with self._lock:
            stale = [
                th
                for th, t in self._email_tokens.items()
                if t.consumed_at is not None or t.expires_at <= now
            ]
            for token_hash in stale:
                del self._email_tokens[token_hash]
            return len(stale)

    def replace_recovery_codes(
        self, account_id: str, code_hashes: list[str], created_at: datetime
    ) -> None:
        """Replace the account's recovery-code set."""
        with self._lock:
            self._recovery_codes[account_id] = [
                {"hash": h, "created_at": created_at, "used_at": None} for h in code_hashes
            ]

    def consume_recovery_code(self, account_id: str, code_hash: str, used_at: datetime) -> bool:
        """Atomically consume one unused recovery code; report success."""
        with self._lock:
            for entry in self._recovery_codes.get(account_id, []):
                if entry["hash"] == code_hash and entry["used_at"] is None:
                    entry["used_at"] = used_at
                    return True
            return False

    def count_unused_recovery_codes(self, account_id: str) -> int:
        """Return how many recovery codes remain unused."""
        with self._lock:
            return sum(1 for e in self._recovery_codes.get(account_id, []) if e["used_at"] is None)


class SqliteIdentityStore:
    """Durable identity store backed by stdlib ``sqlite3`` (WAL, lock-guarded)."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS accounts (
        id             TEXT PRIMARY KEY,
        email          TEXT NOT NULL UNIQUE,
        password_hash  TEXT NOT NULL,
        is_verified    INTEGER NOT NULL,
        is_active      INTEGER NOT NULL,
        totp_secret    TEXT,
        totp_enabled   INTEGER NOT NULL,
        totp_last_step INTEGER,
        tier           TEXT NOT NULL DEFAULT 'free',
        created_at     TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS sessions (
        token_hash   TEXT PRIMARY KEY,
        account_id   TEXT NOT NULL,
        created_at   TEXT NOT NULL,
        expires_at   TEXT NOT NULL,
        csrf_hash    TEXT NOT NULL DEFAULT '',
        last_seen_at TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_sessions_account ON sessions (account_id);
    CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions (expires_at);
    CREATE TABLE IF NOT EXISTS email_tokens (
        token_hash  TEXT PRIMARY KEY,
        account_id  TEXT NOT NULL,
        purpose     TEXT NOT NULL,
        created_at  TEXT NOT NULL,
        expires_at  TEXT NOT NULL,
        consumed_at TEXT,
        payload     TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_email_tokens_account ON email_tokens (account_id);
    CREATE TABLE IF NOT EXISTS recovery_codes (
        code_hash  TEXT PRIMARY KEY,
        account_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        used_at    TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_recovery_codes_account ON recovery_codes (account_id);
    """

    #: Purely additive migrations for database files created before a column
    #: existed. Applied idempotently at open; ``ALTER TABLE ADD COLUMN`` on an
    #: existing column is skipped via a PRAGMA check.
    _MIGRATIONS: tuple[tuple[str, str, str], ...] = (
        ("accounts", "totp_last_step", "INTEGER"),
        ("accounts", "tier", "TEXT NOT NULL DEFAULT 'free'"),
        ("sessions", "csrf_hash", "TEXT NOT NULL DEFAULT ''"),
        ("sessions", "last_seen_at", "TEXT"),
        ("email_tokens", "payload", "TEXT"),
    )

    def __init__(self, path: str | os.PathLike[str]) -> None:
        """Open (creating if needed) the identity store at ``path``."""
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
            self._apply_migrations()
            self._conn.commit()

    def _apply_migrations(self) -> None:
        """Add any columns missing from an older database file (idempotent)."""
        for table, column, declaration in self._MIGRATIONS:
            present = {row["name"] for row in self._conn.execute(f"PRAGMA table_info({table})")}
            if column not in present:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    def close(self) -> None:
        """Close the underlying connection."""
        with self._lock:
            self._conn.close()

    @staticmethod
    def _iso(moment: datetime | None) -> str | None:
        """ISO-8601 form of a timestamp, or ``None``."""
        return moment.isoformat() if moment is not None else None

    @staticmethod
    def _from_iso(raw: str | None) -> datetime | None:
        """Parse an ISO-8601 timestamp, or ``None``."""
        return datetime.fromisoformat(raw) if raw else None

    @classmethod
    def _row_to_account(cls, row: sqlite3.Row) -> Account:
        """Reconstruct an :class:`Account` from a row."""
        return Account(
            id=row["id"],
            email=row["email"],
            password_hash=row["password_hash"],
            is_verified=bool(row["is_verified"]),
            is_active=bool(row["is_active"]),
            totp_secret=row["totp_secret"],
            totp_enabled=bool(row["totp_enabled"]),
            totp_last_step=(
                int(row["totp_last_step"]) if row["totp_last_step"] is not None else None
            ),
            tier=row["tier"] or "free",
            created_at=cls._from_iso(row["created_at"]) or datetime.min,
        )

    def create_account(self, account: Account) -> None:
        """Persist a new account, mapping a UNIQUE clash to DuplicateEmailError."""
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO accounts (id, email, password_hash, is_verified, "
                    "is_active, totp_secret, totp_enabled, totp_last_step, tier, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        account.id,
                        _normalize_email(account.email),
                        account.password_hash,
                        int(account.is_verified),
                        int(account.is_active),
                        account.totp_secret,
                        int(account.totp_enabled),
                        account.totp_last_step,
                        account.tier,
                        self._iso(account.created_at),
                    ),
                )
                self._conn.commit()
            except sqlite3.IntegrityError as exc:
                raise DuplicateEmailError(f"email already registered: {account.email}") from exc

    def get_account_by_email(self, email: str) -> Account | None:
        """Return the account with ``email`` (case-insensitive), or ``None``."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM accounts WHERE email = ?", (_normalize_email(email),)
            ).fetchone()
        return self._row_to_account(row) if row is not None else None

    def get_account_by_id(self, account_id: str) -> Account | None:
        """Return the account with ``account_id``, or ``None``."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM accounts WHERE id = ?", (account_id,)
            ).fetchone()
        return self._row_to_account(row) if row is not None else None

    def update_account(self, account: Account) -> None:
        """Persist mutable account fields (including a changed email).

        Raises:
            DuplicateEmailError: If the new email collides with another row.
        """
        with self._lock:
            try:
                self._conn.execute(
                    "UPDATE accounts SET email = ?, password_hash = ?, is_verified = ?, "
                    "is_active = ?, totp_secret = ?, totp_enabled = ?, totp_last_step = ?, "
                    "tier = ? WHERE id = ?",
                    (
                        _normalize_email(account.email),
                        account.password_hash,
                        int(account.is_verified),
                        int(account.is_active),
                        account.totp_secret,
                        int(account.totp_enabled),
                        account.totp_last_step,
                        account.tier,
                        account.id,
                    ),
                )
                self._conn.commit()
            except sqlite3.IntegrityError as exc:
                raise DuplicateEmailError(f"email already registered: {account.email}") from exc

    def delete_account(self, account_id: str) -> None:
        """Hard-delete the account and everything hanging off it."""
        with self._lock:
            self._conn.execute("DELETE FROM sessions WHERE account_id = ?", (account_id,))
            self._conn.execute("DELETE FROM email_tokens WHERE account_id = ?", (account_id,))
            self._conn.execute("DELETE FROM recovery_codes WHERE account_id = ?", (account_id,))
            self._conn.execute("DELETE FROM accounts WHERE id = ?", (account_id,))
            self._conn.commit()

    def iter_accounts(self) -> list[Account]:
        """Return a snapshot of all accounts."""
        with self._lock:
            rows = self._conn.execute("SELECT * FROM accounts").fetchall()
        return [self._row_to_account(row) for row in rows]

    def create_session(self, session: Session) -> None:
        """Persist a new session."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions (token_hash, account_id, created_at, expires_at, "
                "csrf_hash, last_seen_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session.token_hash,
                    session.account_id,
                    self._iso(session.created_at),
                    self._iso(session.expires_at),
                    session.csrf_hash,
                    self._iso(session.last_seen_at),
                ),
            )
            self._conn.commit()

    def get_session(self, token_hash: str) -> Session | None:
        """Return the session for ``token_hash``, or ``None``."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE token_hash = ?", (token_hash,)
            ).fetchone()
        if row is None:
            return None
        return Session(
            token_hash=row["token_hash"],
            account_id=row["account_id"],
            created_at=self._from_iso(row["created_at"]) or datetime.min,
            expires_at=self._from_iso(row["expires_at"]) or datetime.min,
            csrf_hash=row["csrf_hash"] or "",
            last_seen_at=self._from_iso(row["last_seen_at"]),
        )

    def touch_session(self, token_hash: str, last_seen_at: datetime) -> None:
        """Update a session's last-seen stamp."""
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET last_seen_at = ? WHERE token_hash = ?",
                (self._iso(last_seen_at), token_hash),
            )
            self._conn.commit()

    def delete_session(self, token_hash: str) -> None:
        """Delete one session."""
        with self._lock:
            self._conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
            self._conn.commit()

    def delete_sessions_for_account(self, account_id: str) -> None:
        """Delete every session for an account."""
        with self._lock:
            self._conn.execute("DELETE FROM sessions WHERE account_id = ?", (account_id,))
            self._conn.commit()

    def prune_expired_sessions(self, now: datetime) -> int:
        """Delete sessions with ``expires_at <= now``; return how many."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM sessions WHERE expires_at <= ?", (self._iso(now),)
            )
            self._conn.commit()
            return int(cur.rowcount)

    def create_email_token(self, token: EmailToken) -> None:
        """Persist a new email token."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO email_tokens (token_hash, account_id, purpose, created_at, "
                "expires_at, consumed_at, payload) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    token.token_hash,
                    token.account_id,
                    token.purpose,
                    self._iso(token.created_at),
                    self._iso(token.expires_at),
                    self._iso(token.consumed_at),
                    token.payload,
                ),
            )
            self._conn.commit()

    def get_email_token(self, token_hash: str) -> EmailToken | None:
        """Return the email token for ``token_hash``, or ``None``."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM email_tokens WHERE token_hash = ?", (token_hash,)
            ).fetchone()
        if row is None:
            return None
        return EmailToken(
            token_hash=row["token_hash"],
            account_id=row["account_id"],
            purpose=row["purpose"],
            created_at=self._from_iso(row["created_at"]) or datetime.min,
            expires_at=self._from_iso(row["expires_at"]) or datetime.min,
            consumed_at=self._from_iso(row["consumed_at"]),
            payload=row["payload"],
        )

    def consume_email_token(self, token_hash: str, consumed_at: datetime) -> None:
        """Mark an email token consumed."""
        with self._lock:
            self._conn.execute(
                "UPDATE email_tokens SET consumed_at = ? WHERE token_hash = ?",
                (self._iso(consumed_at), token_hash),
            )
            self._conn.commit()

    def delete_email_tokens_for_account(self, account_id: str, purpose: str | None = None) -> None:
        """Delete an account's outstanding tokens (optionally one purpose)."""
        with self._lock:
            if purpose is None:
                self._conn.execute("DELETE FROM email_tokens WHERE account_id = ?", (account_id,))
            else:
                self._conn.execute(
                    "DELETE FROM email_tokens WHERE account_id = ? AND purpose = ?",
                    (account_id, purpose),
                )
            self._conn.commit()

    def prune_email_tokens(self, now: datetime) -> int:
        """Delete consumed or expired email tokens; return how many."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM email_tokens WHERE consumed_at IS NOT NULL OR expires_at <= ?",
                (self._iso(now),),
            )
            self._conn.commit()
            return int(cur.rowcount)

    def replace_recovery_codes(
        self, account_id: str, code_hashes: list[str], created_at: datetime
    ) -> None:
        """Replace the account's recovery-code set."""
        with self._lock:
            self._conn.execute("DELETE FROM recovery_codes WHERE account_id = ?", (account_id,))
            self._conn.executemany(
                "INSERT INTO recovery_codes (code_hash, account_id, created_at, used_at) "
                "VALUES (?, ?, ?, NULL)",
                [(code_hash, account_id, self._iso(created_at)) for code_hash in code_hashes],
            )
            self._conn.commit()

    def consume_recovery_code(self, account_id: str, code_hash: str, used_at: datetime) -> bool:
        """Atomically consume one unused recovery code; report success.

        The single conditional ``UPDATE`` is the atomicity guarantee: two
        concurrent logins presenting the same code race on the row, exactly
        one UPDATE matches ``used_at IS NULL``, and the loser is rejected.
        """
        with self._lock:
            cur = self._conn.execute(
                "UPDATE recovery_codes SET used_at = ? "
                "WHERE account_id = ? AND code_hash = ? AND used_at IS NULL",
                (self._iso(used_at), account_id, code_hash),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def count_unused_recovery_codes(self, account_id: str) -> int:
        """Return how many recovery codes remain unused."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM recovery_codes "
                "WHERE account_id = ? AND used_at IS NULL",
                (account_id,),
            ).fetchone()
        return int(row["n"])


def identity_store_is_durable() -> bool:
    """Whether the environment selects the durable (SQLite) identity backend."""
    return bool(os.getenv(IDENTITY_PATH_ENV, "").strip())


def build_identity_store() -> IdentityStore:
    """Construct the configured identity backend from the environment.

    Returns:
        A :class:`SqliteIdentityStore` when ``MERCURY_KEYSTORE_PATH`` is set
        (durable; shares the key store's file), otherwise an
        :class:`InMemoryIdentityStore` (dev/test default).
    """
    path = os.getenv(IDENTITY_PATH_ENV, "").strip()
    if not path:
        return InMemoryIdentityStore()
    return SqliteIdentityStore(path)
