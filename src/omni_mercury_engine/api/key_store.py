# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Durable API-key storage backends.

The auth layer historically kept API keys in a process-local
:class:`~omni_mercury_engine.api.auth.APIKeyStore` (a plain ``dict``), so every
key vanished on restart — fine for tests, unusable for a real deployment. This
module introduces the storage seam without changing a single caller:

* :class:`KeyStore` is the structural contract every backend satisfies — the
  exact public surface the auth layer already calls
  (``create_key`` / ``get_by_key`` / ``get_by_id`` / ``revoke`` /
  ``update_last_used``). ``APIKeyStore`` already conforms; it stays the
  in-memory backend and the default for dev and tests.
* :class:`SqliteKeyStore` is a durable backend built on the stdlib ``sqlite3``
  module (WAL journalling, multi-process safe). Keys created here survive a
  restart, a second worker, and a redeploy. SQLite is the "start" backend; a
  Postgres backend is a future sibling implementation of the same
  :class:`KeyStore` contract — no caller changes when it lands.
* :func:`build_key_store` selects the backend from the environment
  (``MERCURY_KEYSTORE_PATH``): unset → in-memory (unchanged behaviour), set →
  SQLite at that path.

Key hashing is deliberately shared: :class:`SqliteKeyStore` hashes through
``APIKeyStore.hash_key`` so a key issued by one backend validates byte-identically
against the other (same PBKDF2 salt + iteration policy, single source of truth).
Only the *hash* is persisted — the raw key is returned once at creation and never
stored, exactly as before.
"""

from __future__ import annotations

import os
import secrets
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from omni_mercury_engine.api.auth import APIKey, APIKeyStore, Permission

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["KeyStore", "SqliteKeyStore", "build_key_store"]

#: Environment variable selecting a durable key store. When set to a filesystem
#: path, :func:`build_key_store` returns a :class:`SqliteKeyStore` rooted there;
#: when unset the in-memory :class:`APIKeyStore` is used (dev/test default, so
#: existing behaviour is unchanged).
KEYSTORE_PATH_ENV = "MERCURY_KEYSTORE_PATH"


@runtime_checkable
class KeyStore(Protocol):
    """Structural contract for an API-key backend.

    This is exactly the surface the auth layer already depends on; both
    :class:`APIKeyStore` (in-memory) and :class:`SqliteKeyStore` (durable)
    satisfy it, so ``get_api_key_store()`` can return either with no caller
    change.
    """

    def create_key(
        self,
        name: str,
        user_id: str,
        permissions: set[Permission] | None = None,
        expires_in_days: int | None = None,
        rate_limit: int = 100,
    ) -> tuple[str, APIKey]:
        """Create a key; return ``(raw_key, APIKey)``. The raw key is shown once."""
        ...

    def get_by_key(self, raw_key: str) -> APIKey | None:
        """Return the stored key matching ``raw_key``'s hash, or ``None``."""
        ...

    def get_by_id(self, key_id: str) -> APIKey | None:
        """Return the stored key with ``key_id``, or ``None``."""
        ...

    def revoke(self, key_id: str) -> bool:
        """Deactivate ``key_id``; return whether a row was affected."""
        ...

    def update_last_used(self, key_id: str) -> None:
        """Stamp ``key_id``'s last-used time to now (best effort)."""
        ...


def _serialize_permissions(permissions: Iterable[Permission]) -> str:
    """Encode a permission set as a stable, comma-separated string of values."""
    return ",".join(sorted(p.value for p in permissions))


def _deserialize_permissions(raw: str) -> set[Permission]:
    """Decode the :func:`_serialize_permissions` form back into a permission set."""
    return {Permission(value) for value in raw.split(",") if value}


def _iso(moment: datetime | None) -> str | None:
    """ISO-8601 form of a timestamp, or ``None``."""
    return moment.isoformat() if moment is not None else None


def _from_iso(raw: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp written by :func:`_iso`, or ``None``."""
    return datetime.fromisoformat(raw) if raw else None


class SqliteKeyStore:
    """Durable, restart-surviving API-key store backed by stdlib ``sqlite3``.

    The backend is intentionally synchronous to match the existing
    :class:`KeyStore` surface exactly (the auth path is low-QPS and every query
    is a single-row primary-key/unique-index hit, so the on-loop cost is
    negligible). WAL journalling makes concurrent reads and cross-process access
    safe; all access is additionally guarded by a per-instance lock so the shared
    connection is used serially within a process.

    Hashing is delegated to :meth:`APIKeyStore.hash_key`, so a key created by
    this backend validates identically through the in-memory one and vice versa.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS api_keys (
        key_id       TEXT PRIMARY KEY,
        key_hash     TEXT NOT NULL UNIQUE,
        name         TEXT NOT NULL,
        user_id      TEXT NOT NULL,
        permissions  TEXT NOT NULL,
        created_at   TEXT NOT NULL,
        expires_at   TEXT,
        last_used_at TEXT,
        rate_limit   INTEGER NOT NULL,
        is_active    INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys (key_hash);
    CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys (user_id);
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        """Open (creating if needed) the SQLite key store at ``path``.

        Args:
            path: Filesystem path to the SQLite database file. Parent
                directories are created if missing.
        """
        self._path = Path(path)
        if self._path.parent and not self._path.parent.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # check_same_thread=False + explicit locking: the connection may be
        # touched from the event-loop thread and a worker threadpool; the lock
        # serialises use rather than relying on sqlite3's own thread checks.
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(self._SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        """Close the underlying connection (idempotent)."""
        with self._lock:
            self._conn.close()

    @staticmethod
    def _row_to_key(row: sqlite3.Row) -> APIKey:
        """Reconstruct an :class:`APIKey` from a database row."""
        return APIKey(
            key_id=row["key_id"],
            key_hash=row["key_hash"],
            name=row["name"],
            user_id=row["user_id"],
            permissions=_deserialize_permissions(row["permissions"]),
            created_at=_from_iso(row["created_at"]) or datetime.now(),
            expires_at=_from_iso(row["expires_at"]),
            last_used_at=_from_iso(row["last_used_at"]),
            rate_limit=int(row["rate_limit"]),
            is_active=bool(row["is_active"]),
        )

    def create_key(
        self,
        name: str,
        user_id: str,
        permissions: set[Permission] | None = None,
        expires_in_days: int | None = None,
        rate_limit: int = 100,
    ) -> tuple[str, APIKey]:
        """Create and persist a new API key.

        Mirrors :meth:`APIKeyStore.create_key` exactly (same raw-key entropy,
        same hashing, same default permission set) but writes the record to
        SQLite instead of a process-local dict.

        Returns:
            Tuple of ``(raw_key, APIKey)``. The raw key is returned once and
            never stored — only its hash is persisted.
        """
        raw_key = secrets.token_urlsafe(32)
        key_hash = APIKeyStore.hash_key(raw_key)
        key_id = secrets.token_hex(8)

        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            user_id=user_id,
            permissions=permissions or {Permission.READ, Permission.DETECT},
            expires_at=(
                datetime.now() + timedelta(days=expires_in_days) if expires_in_days else None
            ),
            rate_limit=rate_limit,
        )

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO api_keys (
                    key_id, key_hash, name, user_id, permissions,
                    created_at, expires_at, last_used_at, rate_limit, is_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    api_key.key_id,
                    api_key.key_hash,
                    api_key.name,
                    api_key.user_id,
                    _serialize_permissions(api_key.permissions),
                    _iso(api_key.created_at),
                    _iso(api_key.expires_at),
                    _iso(api_key.last_used_at),
                    api_key.rate_limit,
                    int(api_key.is_active),
                ),
            )
            self._conn.commit()
        return raw_key, api_key

    def get_by_key(self, raw_key: str) -> APIKey | None:
        """Return the stored key whose hash matches ``raw_key``, or ``None``."""
        key_hash = APIKeyStore.hash_key(raw_key)
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM api_keys WHERE key_hash = ?", (key_hash,)
            ).fetchone()
        return self._row_to_key(row) if row is not None else None

    def get_by_id(self, key_id: str) -> APIKey | None:
        """Return the stored key with ``key_id``, or ``None``."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM api_keys WHERE key_id = ?", (key_id,)
            ).fetchone()
        return self._row_to_key(row) if row is not None else None

    def revoke(self, key_id: str) -> bool:
        """Deactivate ``key_id``; return whether a row was affected."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE api_keys SET is_active = 0 WHERE key_id = ?", (key_id,)
            )
            self._conn.commit()
            return cur.rowcount > 0

    def update_last_used(self, key_id: str) -> None:
        """Stamp ``key_id``'s last-used time to now (best effort)."""
        with self._lock:
            self._conn.execute(
                "UPDATE api_keys SET last_used_at = ? WHERE key_id = ?",
                (_iso(datetime.now()), key_id),
            )
            self._conn.commit()


def build_key_store() -> KeyStore:
    """Construct the configured API-key backend.

    Selection is environment-driven so no code change is needed to move from
    dev to a durable deployment:

    * ``MERCURY_KEYSTORE_PATH`` unset  → in-memory :class:`APIKeyStore`
      (unchanged default; keys live for the process lifetime).
    * ``MERCURY_KEYSTORE_PATH=<file>`` → :class:`SqliteKeyStore` at that path
      (durable; keys survive restarts and are shared across workers).

    Returns:
        A backend satisfying the :class:`KeyStore` contract.
    """
    path = os.getenv(KEYSTORE_PATH_ENV, "").strip()
    if not path:
        return APIKeyStore()
    return SqliteKeyStore(path)
