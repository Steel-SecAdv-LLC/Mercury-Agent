# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for scripts/reseal_totp_secrets.py (at-rest key rotation for TOTP).

Verifies the rotation is correct and safe: secrets sealed under the OLD key are
re-sealed under the NEW key (and open under it afterwards), plaintext secrets
are sealed, already-current rows are left untouched (idempotent re-runs), and a
value openable under neither key is reported as a failure rather than lost.
Also covers the durable SQLite round-trip and the CLI guards.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from omni_mercury_engine.api.identity_store import (
    Account,
    IdentityStore,
    InMemoryIdentityStore,
    SqliteIdentityStore,
)
from omni_mercury_engine.api.secret_sealer import SecretSealer

if TYPE_CHECKING:
    import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_MOD = REPO_ROOT / "scripts" / "reseal_totp_secrets.py"
_spec = importlib.util.spec_from_file_location("reseal_totp_secrets", _MOD)
assert _spec is not None and _spec.loader is not None
reseal = importlib.util.module_from_spec(_spec)
# Register before exec: the module uses ``from __future__ import annotations``
# with ``@dataclass``; dataclass field resolution looks the module up in
# ``sys.modules`` by name.
sys.modules[_spec.name] = reseal
_spec.loader.exec_module(reseal)

_OLD_KEY = "11" * 32
_NEW_KEY = "22" * 32


def _old() -> SecretSealer:
    return SecretSealer(bytes.fromhex(_OLD_KEY), key_is_stable=True)


def _new() -> SecretSealer:
    return SecretSealer(bytes.fromhex(_NEW_KEY), key_is_stable=True)


def _account(account_id: str, secret: str | None) -> Account:
    acc = Account(id=account_id, email=f"{account_id}@x.io", password_hash="h")  # noqa: S106
    acc.totp_secret = secret
    return acc


def _secret_of(store: IdentityStore, account_id: str) -> str:
    """Fetch an account's TOTP secret, asserting it is present (str, not None)."""
    account = store.get_account_by_id(account_id)
    assert account is not None
    assert account.totp_secret is not None
    return account.totp_secret


class TestResealCore:
    """The framework-free re-seal function over an in-memory store."""

    def test_old_key_secret_rerolls_to_new(self) -> None:
        """A secret sealed under the old key opens under the new one after."""
        old, new = _old(), _new()
        store = InMemoryIdentityStore()
        store.create_account(_account("a1", old.seal("SEEDONE", aad="a1")))

        report = reseal.reseal_totp_secrets(store, new_sealer=new, old_sealer=old)

        assert report.resealed == 1
        assert new.unseal(_secret_of(store, "a1"), aad="a1") == "SEEDONE"

    def test_plaintext_secret_is_sealed(self) -> None:
        """A legacy plaintext secret is sealed under the new key."""
        new = _new()
        store = InMemoryIdentityStore()
        store.create_account(_account("a2", "PLAINSEED"))

        report = reseal.reseal_totp_secrets(store, new_sealer=new, old_sealer=_old())

        assert report.sealed_plaintext == 1
        secret = _secret_of(store, "a2")
        assert SecretSealer.is_sealed(secret)
        assert new.unseal(secret, aad="a2") == "PLAINSEED"

    def test_already_current_is_left_untouched(self) -> None:
        """A row already sealed under the new key is skipped (idempotent)."""
        new = _new()
        store = InMemoryIdentityStore()
        sealed = new.seal("SEEDTHREE", aad="a3")
        store.create_account(_account("a3", sealed))

        report = reseal.reseal_totp_secrets(store, new_sealer=new, old_sealer=_old())

        assert report.already_current == 1 and report.changed == 0
        migrated = store.get_account_by_id("a3")
        assert migrated is not None and migrated.totp_secret == sealed

    def test_rerun_is_idempotent(self) -> None:
        """A second pass changes nothing once every row is current."""
        old, new = _old(), _new()
        store = InMemoryIdentityStore()
        store.create_account(_account("a1", old.seal("S1", aad="a1")))
        store.create_account(_account("a2", "PLAIN"))

        reseal.reseal_totp_secrets(store, new_sealer=new, old_sealer=old)
        second = reseal.reseal_totp_secrets(store, new_sealer=new, old_sealer=old)

        assert second.changed == 0
        assert second.already_current == 2

    def test_no_old_key_reports_failure_without_data_loss(self) -> None:
        """Without the old key, an old-sealed value is a failure, not a loss."""
        old, new = _old(), _new()
        store = InMemoryIdentityStore()
        original = old.seal("SEED", aad="b1")
        store.create_account(_account("b1", original))

        report = reseal.reseal_totp_secrets(store, new_sealer=new, old_sealer=None)

        assert report.failed == ["b1"]
        # The stored value is untouched (never overwritten with garbage).
        preserved = store.get_account_by_id("b1")
        assert preserved is not None and preserved.totp_secret == original

    def test_dry_run_writes_nothing(self) -> None:
        """A dry run classifies every row but leaves the store unchanged."""
        old, new = _old(), _new()
        store = InMemoryIdentityStore()
        original = old.seal("SEED", aad="a1")
        store.create_account(_account("a1", original))

        report = reseal.reseal_totp_secrets(store, new_sealer=new, old_sealer=old, dry_run=True)

        assert report.resealed == 1 and report.dry_run is True
        untouched = store.get_account_by_id("a1")
        assert untouched is not None and untouched.totp_secret == original

    def test_accounts_without_secret_are_ignored(self) -> None:
        """Accounts with no TOTP secret never count toward with_totp."""
        store = InMemoryIdentityStore()
        store.create_account(_account("n1", None))
        report = reseal.reseal_totp_secrets(store, new_sealer=_new(), old_sealer=_old())
        assert report.total_accounts == 1 and report.with_totp == 0


class TestDurableRoundTrip:
    """The rotation persists across a durable SQLite store."""

    def test_sqlite_reseal_persists(self, tmp_path: Path) -> None:
        """An old-sealed secret in SQLite opens under the new key after re-seal."""
        db = tmp_path / "mercury.db"
        old, new = _old(), _new()

        store = SqliteIdentityStore(str(db))
        account = _account("acct-1", old.seal("DURABLESEED", aad="acct-1"))
        account.totp_enabled = True
        store.create_account(account)

        report = reseal.reseal_totp_secrets(store, new_sealer=new, old_sealer=old)
        assert report.resealed == 1
        store.close()

        # Reopen from disk: the persisted value opens under the new key only.
        reopened = SqliteIdentityStore(str(db))
        try:
            assert new.unseal(_secret_of(reopened, "acct-1"), aad="acct-1") == "DURABLESEED"
        finally:
            reopened.close()


class TestCli:
    """The CLI guards refuse unsafe or under-specified runs."""

    def test_refuses_without_durable_store(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An in-memory store (no MERCURY_KEYSTORE_PATH) is refused (exit 2)."""
        monkeypatch.delenv("MERCURY_KEYSTORE_PATH", raising=False)
        monkeypatch.setenv("MERCURY_DATA_ENC_KEY", _NEW_KEY)
        assert reseal.main([]) == 2

    def test_refuses_without_new_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """No new key (arg or env) is refused (exit 2)."""
        monkeypatch.setenv("MERCURY_KEYSTORE_PATH", str(tmp_path / "m.db"))
        monkeypatch.delenv("MERCURY_DATA_ENC_KEY", raising=False)
        monkeypatch.delenv("MERCURY_DATA_ENC_KEY_OLD", raising=False)
        assert reseal.main([]) == 2

    def test_end_to_end_via_main(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """main() reseals a seeded durable store and exits 0."""
        db = tmp_path / "mercury.db"
        monkeypatch.setenv("MERCURY_KEYSTORE_PATH", str(db))
        monkeypatch.setenv("MERCURY_DATA_ENC_KEY", _NEW_KEY)
        monkeypatch.setenv("MERCURY_DATA_ENC_KEY_OLD", _OLD_KEY)

        seed_store = SqliteIdentityStore(str(db))
        seed_store.create_account(_account("a1", _old().seal("VIA-MAIN", aad="a1")))
        seed_store.close()

        assert reseal.main([]) == 0

        verify_store = SqliteIdentityStore(str(db))
        try:
            assert _new().unseal(_secret_of(verify_store, "a1"), aad="a1") == "VIA-MAIN"
        finally:
            verify_store.close()

    def test_failure_exit_code(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unopenable secret makes main() exit non-zero (operator notice)."""
        db = tmp_path / "mercury.db"
        monkeypatch.setenv("MERCURY_KEYSTORE_PATH", str(db))
        monkeypatch.setenv("MERCURY_DATA_ENC_KEY", _NEW_KEY)
        monkeypatch.delenv("MERCURY_DATA_ENC_KEY_OLD", raising=False)

        seed_store = SqliteIdentityStore(str(db))
        # Sealed under the old key, but no old key supplied → cannot open.
        seed_store.create_account(_account("a1", _old().seal("STUCK", aad="a1")))
        seed_store.close()

        assert reseal.main([]) == 1
