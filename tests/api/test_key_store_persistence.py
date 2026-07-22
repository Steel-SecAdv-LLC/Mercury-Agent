# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Contract + durability tests for the API-key storage backends.

Two properties are pinned here:

* **Parity** — the in-memory :class:`APIKeyStore` and the durable
  :class:`SqliteKeyStore` behave identically against the :class:`KeyStore`
  contract, so swapping backends never changes auth behaviour.
* **Durability** — a key written to :class:`SqliteKeyStore` still validates
  after the store is closed and re-opened (a stand-in for a process restart or a
  second worker), which is the exact regression the in-memory store could not
  survive.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from omni_mercury_engine.api.auth import APIKeyStore, Permission
from omni_mercury_engine.api.key_store import (
    KEYSTORE_PATH_ENV,
    KeyStore,
    SqliteKeyStore,
    build_key_store,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[KeyStore]:
    """Yield each backend so every contract test runs against both."""
    if request.param == "memory":
        yield APIKeyStore()
    else:
        sqlite_store = SqliteKeyStore(tmp_path / "keys.db")
        yield sqlite_store
        sqlite_store.close()


# --------------------------------------------------------------------------- #
# Contract parity — identical behaviour across both backends.
# --------------------------------------------------------------------------- #
def test_create_then_validate(store: KeyStore) -> None:
    """A freshly created key validates by its raw value and its id."""
    raw_key, created = store.create_key("ci-key", "user-1")

    by_raw = store.get_by_key(raw_key)
    assert by_raw is not None
    assert by_raw.key_id == created.key_id
    assert by_raw.user_id == "user-1"

    by_id = store.get_by_id(created.key_id)
    assert by_id is not None
    assert by_id.key_id == created.key_id


def test_default_permissions_roundtrip(store: KeyStore) -> None:
    """The default permission set is preserved through storage."""
    raw_key, _ = store.create_key("defaults", "user-2")
    fetched = store.get_by_key(raw_key)
    assert fetched is not None
    assert fetched.permissions == {Permission.READ, Permission.DETECT}


def test_custom_permissions_roundtrip(store: KeyStore) -> None:
    """A custom permission set survives serialization intact and complete."""
    perms = {Permission.READ, Permission.WRITE, Permission.ADMIN}
    _, created = store.create_key("scoped", "user-3", permissions=perms)
    fetched = store.get_by_id(created.key_id)
    assert fetched is not None
    assert fetched.permissions == perms


def test_expiry_persisted(store: KeyStore) -> None:
    """An expiry window is stored and reconstructed; None stays None."""
    _, expiring = store.create_key("temp", "user-4", expires_in_days=30)
    fetched = store.get_by_id(expiring.key_id)
    assert fetched is not None
    assert fetched.expires_at is not None
    assert not fetched.is_expired

    _, permanent = store.create_key("perm", "user-4")
    permanent_fetched = store.get_by_id(permanent.key_id)
    assert permanent_fetched is not None
    assert permanent_fetched.expires_at is None


def test_revoke(store: KeyStore) -> None:
    """Revoking flips is_active; revoking an unknown id is a no-op."""
    _, created = store.create_key("revocable", "user-5")
    assert store.revoke(created.key_id) is True

    fetched = store.get_by_id(created.key_id)
    assert fetched is not None
    assert fetched.is_active is False

    assert store.revoke("does-not-exist") is False


def test_update_last_used(store: KeyStore) -> None:
    """last_used_at starts unset and is stamped on update."""
    _, created = store.create_key("used", "user-6")
    assert created.last_used_at is None

    store.update_last_used(created.key_id)
    fetched = store.get_by_id(created.key_id)
    assert fetched is not None
    assert fetched.last_used_at is not None


def test_unknown_lookups_return_none(store: KeyStore) -> None:
    """Lookups for absent keys return None rather than raising."""
    assert store.get_by_key("not-a-real-key") is None
    assert store.get_by_id("not-a-real-id") is None


def test_both_backends_satisfy_protocol(tmp_path: Path) -> None:
    """Both concrete stores structurally satisfy the KeyStore protocol."""
    assert isinstance(APIKeyStore(), KeyStore)
    sqlite_store = SqliteKeyStore(tmp_path / "proto.db")
    try:
        assert isinstance(sqlite_store, KeyStore)
    finally:
        sqlite_store.close()


# --------------------------------------------------------------------------- #
# Durability — the property the in-memory store could not provide.
# --------------------------------------------------------------------------- #
def test_key_survives_reopen(tmp_path: Path) -> None:
    """A key persists across a close/re-open cycle (restart / second worker)."""
    db_path = tmp_path / "durable.db"

    first = SqliteKeyStore(db_path)
    raw_key, created = first.create_key("persistent", "user-restart")
    first.close()

    # A brand-new store object over the same file stands in for a fresh process.
    second = SqliteKeyStore(db_path)
    try:
        recovered = second.get_by_key(raw_key)
        assert recovered is not None
        assert recovered.key_id == created.key_id
        assert recovered.user_id == "user-restart"
        assert recovered.permissions == {Permission.READ, Permission.DETECT}
    finally:
        second.close()


# --------------------------------------------------------------------------- #
# Backend selection.
# --------------------------------------------------------------------------- #
def test_build_key_store_defaults_to_in_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no path configured, the default backend stays in-memory."""
    monkeypatch.delenv(KEYSTORE_PATH_ENV, raising=False)
    assert isinstance(build_key_store(), APIKeyStore)


def test_build_key_store_selects_sqlite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A configured path selects the durable SQLite backend."""
    monkeypatch.setenv(KEYSTORE_PATH_ENV, str(tmp_path / "configured.db"))
    built = build_key_store()
    assert isinstance(built, SqliteKeyStore)
    try:
        raw_key, created = built.create_key("factory", "user-7")
        fetched = built.get_by_key(raw_key)
        assert fetched is not None
        assert fetched.key_id == created.key_id
    finally:
        built.close()


def test_get_api_key_store_honours_env_and_caches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """get_api_key_store() builds the env-selected backend once and caches it."""
    import omni_mercury_engine.api.auth as auth

    monkeypatch.setattr(auth, "_api_key_store", None)
    monkeypatch.setenv(KEYSTORE_PATH_ENV, str(tmp_path / "singleton.db"))

    store_a = auth.get_api_key_store()
    store_b = auth.get_api_key_store()
    assert isinstance(store_a, SqliteKeyStore)
    assert store_a is store_b  # cached singleton, not rebuilt per call
