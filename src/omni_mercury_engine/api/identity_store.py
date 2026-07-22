# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Durable storage for user accounts, sessions, and email tokens.

This is the persistence layer the self-service auth flows sit on. It mirrors the
key-store design: an :class:`IdentityStore` Protocol with an in-memory backend
(dev/tests) and a durable SQLite backend selected by the same
``MERCURY_KEYSTORE_PATH`` file, so a single database file holds API keys,
accounts, sessions, and tokens.

Three record types are stored:

* :class:`Account` — a user: unique lower-cased email, password hash, verified /
  active flags, and optional TOTP 2FA secret.
* :class:`Session` — a browser login. Only the SHA-256 of the opaque session
  token is stored (the raw token lives only in the user's cookie), so a database
  read cannot mint sessions.
* :class:`EmailToken` — a single-use, expiring token for email verification or
  password reset. Again only the hash is stored.

Nothing here sends email or hashes passwords; that is the auth service's job.
This module only persists and retrieves.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
from dataclasses import dataclass
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
]

#: Environment variable naming the shared SQLite database file. Identical to the
#: key store's variable so one file backs all auth state; unset selects the
#: in-memory backend (dev/test default).
IDENTITY_PATH_ENV = "MERCURY_KEYSTORE_PATH"

#: The two purposes an :class:`EmailToken` can serve.
TokenPurpose = str  # "verify" | "reset"


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
    created_at: datetime = datetime.min


@dataclass
class Session:
    """A browser login session (only the token hash is stored)."""

    token_hash: str
    account_id: str
    created_at: datetime
    expires_at: datetime


@dataclass
class EmailToken:
    """A single-use, expiring email token (only the token hash is stored)."""

    token_hash: str
    account_id: str
    purpose: TokenPurpose
    created_at: datetime
    expires_at: datetime
    consumed_at: datetime | None = None


@runtime_checkable
class IdentityStore(Protocol):
    """Contract for durable account/session/token storage."""

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
        """Persist mutable account fields (verified/active/password/TOTP)."""
        ...

    def create_session(self, session: Session) -> None:
        """Persist a new session."""
        ...

    def get_session(self, token_hash: str) -> Session | None:
        """Return the session for ``token_hash``, or ``None``."""
        ...

    def delete_session(self, token_hash: str) -> None:
        """Delete one session (logout)."""
        ...

    def delete_sessions_for_account(self, account_id: str) -> None:
        """Delete every session for an account (logout-all / after reset)."""
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
        """Replace the stored account record with ``account``."""
        with self._lock:
            self._accounts[account.id] = account

    def create_session(self, session: Session) -> None:
        """Persist a new session."""
        with self._lock:
            self._sessions[session.token_hash] = session

    def get_session(self, token_hash: str) -> Session | None:
        """Return the session for ``token_hash``, or ``None``."""
        with self._lock:
            return self._sessions.get(token_hash)

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


class SqliteIdentityStore:
    """Durable identity store backed by stdlib ``sqlite3`` (WAL, lock-guarded)."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS accounts (
        id            TEXT PRIMARY KEY,
        email         TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        is_verified   INTEGER NOT NULL,
        is_active     INTEGER NOT NULL,
        totp_secret   TEXT,
        totp_enabled  INTEGER NOT NULL,
        created_at    TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS sessions (
        token_hash TEXT PRIMARY KEY,
        account_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        expires_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_sessions_account ON sessions (account_id);
    CREATE TABLE IF NOT EXISTS email_tokens (
        token_hash  TEXT PRIMARY KEY,
        account_id  TEXT NOT NULL,
        purpose     TEXT NOT NULL,
        created_at  TEXT NOT NULL,
        expires_at  TEXT NOT NULL,
        consumed_at TEXT
    );
    """

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
            self._conn.commit()

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
            created_at=cls._from_iso(row["created_at"]) or datetime.min,
        )

    def create_account(self, account: Account) -> None:
        """Persist a new account, mapping a UNIQUE clash to DuplicateEmailError."""
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO accounts (id, email, password_hash, is_verified, "
                    "is_active, totp_secret, totp_enabled, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        account.id,
                        _normalize_email(account.email),
                        account.password_hash,
                        int(account.is_verified),
                        int(account.is_active),
                        account.totp_secret,
                        int(account.totp_enabled),
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
        """Persist mutable account fields."""
        with self._lock:
            self._conn.execute(
                "UPDATE accounts SET password_hash = ?, is_verified = ?, is_active = ?, "
                "totp_secret = ?, totp_enabled = ? WHERE id = ?",
                (
                    account.password_hash,
                    int(account.is_verified),
                    int(account.is_active),
                    account.totp_secret,
                    int(account.totp_enabled),
                    account.id,
                ),
            )
            self._conn.commit()

    def create_session(self, session: Session) -> None:
        """Persist a new session."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions (token_hash, account_id, created_at, expires_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    session.token_hash,
                    session.account_id,
                    self._iso(session.created_at),
                    self._iso(session.expires_at),
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
        )

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

    def create_email_token(self, token: EmailToken) -> None:
        """Persist a new email token."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO email_tokens (token_hash, account_id, purpose, created_at, "
                "expires_at, consumed_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    token.token_hash,
                    token.account_id,
                    token.purpose,
                    self._iso(token.created_at),
                    self._iso(token.expires_at),
                    self._iso(token.consumed_at),
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
        )

    def consume_email_token(self, token_hash: str, consumed_at: datetime) -> None:
        """Mark an email token consumed."""
        with self._lock:
            self._conn.execute(
                "UPDATE email_tokens SET consumed_at = ? WHERE token_hash = ?",
                (self._iso(consumed_at), token_hash),
            )
            self._conn.commit()


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
