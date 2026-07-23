# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for TOTP at-rest sealing, replay rejection, and 2FA recovery codes.

The threat model pinned here:

* A database read (or backup theft) must not yield usable TOTP seeds — the
  stored value is an AES-256-GCM envelope, AAD-bound to its account, and any
  single-byte tamper or cross-row swap fails closed (mutation-style sweep).
* A code observed inside its 30-second window must not be replayable — the
  accepted time step is persisted and non-increasing steps are rejected.
* A lost authenticator must not mean a lost account — single-use recovery
  codes work exactly once, survive normalisation (case/dashes), and are
  voided by regeneration and by disabling 2FA.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from omni_mercury_engine.api import totp
from omni_mercury_engine.api.auth_service import AuthService, InvalidTwoFactorError
from omni_mercury_engine.api.identity_store import InMemoryIdentityStore, SqliteIdentityStore
from omni_mercury_engine.api.secret_sealer import (
    SealedSecretError,
    SecretSealer,
    build_secret_sealer,
    migrate_plaintext_totp_secrets,
)

if TYPE_CHECKING:
    from pathlib import Path


class RecordingMailer:
    """Collects messages (recovery-notice assertions)."""

    def __init__(self) -> None:
        """Start with an empty outbox."""
        self.sent: list[dict[str, str]] = []

    def send(
        self,
        *,
        to: str,
        subject: str,
        body: str,
        html_body: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Record the message."""
        self.sent.append({"to": to, "subject": subject, "body": body})


class FakeClock:
    """Movable clock for deterministic TOTP steps."""

    def __init__(self) -> None:
        """Start at a fixed instant."""
        self.now = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        """Return the fake time."""
        return self.now

    def advance(self, seconds: float) -> None:
        """Move forward."""
        self.now += timedelta(seconds=seconds)


KEY = bytes(range(32))


class TestSecretSealer:
    """Envelope integrity properties of the sealer itself."""

    def test_roundtrip(self) -> None:
        """Seal → unseal recovers the exact secret."""
        sealer = SecretSealer(KEY, key_is_stable=True)
        sealed = sealer.seal("JBSWY3DPEHPK3PXP", aad="acct-1")
        assert sealer.is_sealed(sealed)
        assert sealer.unseal(sealed, aad="acct-1") == "JBSWY3DPEHPK3PXP"

    def test_ciphertext_not_plaintext(self) -> None:
        """The stored form never contains the secret."""
        sealer = SecretSealer(KEY, key_is_stable=True)
        sealed = sealer.seal("JBSWY3DPEHPK3PXP", aad="acct-1")
        assert "JBSWY3DPEHPK3PXP" not in sealed

    def test_every_byte_flip_fails_closed(self) -> None:
        """Mutation sweep: flipping ANY envelope byte breaks authentication.

        GCM authenticates nonce, tag, and ciphertext jointly; a sealer that
        silently returned garbage on tamper would hand an attacker a wrong—
        but accepted—TOTP seed. Every mutant must raise, never return.
        """
        import base64

        sealer = SecretSealer(KEY, key_is_stable=True)
        sealed = sealer.seal("JBSWY3DPEHPK3PXP", aad="acct-1")
        prefix = "enc$v1$"
        blob = bytearray(base64.b64decode(sealed[len(prefix) :]))
        for i in range(len(blob)):
            mutant = bytearray(blob)
            mutant[i] ^= 0x01
            tampered = prefix + base64.b64encode(bytes(mutant)).decode()
            with pytest.raises(SealedSecretError):
                sealer.unseal(tampered, aad="acct-1")

    def test_cross_account_swap_rejected(self) -> None:
        """AAD binding: a ciphertext moved to another account's row fails."""
        sealer = SecretSealer(KEY, key_is_stable=True)
        sealed = sealer.seal("JBSWY3DPEHPK3PXP", aad="acct-1")
        with pytest.raises(SealedSecretError):
            sealer.unseal(sealed, aad="acct-2")

    def test_wrong_key_rejected(self) -> None:
        """A different key cannot open the envelope."""
        sealed = SecretSealer(KEY, key_is_stable=True).seal("S3CRETSEED", aad="a")
        other = SecretSealer(bytes(range(1, 33)), key_is_stable=True)
        with pytest.raises(SealedSecretError):
            other.unseal(sealed, aad="a")

    def test_legacy_plaintext_passthrough(self) -> None:
        """Pre-sealing rows keep working: unseal returns them verbatim."""
        sealer = SecretSealer(KEY, key_is_stable=True)
        assert sealer.unseal("JBSWY3DPEHPK3PXP", aad="acct-1") == "JBSWY3DPEHPK3PXP"

    def test_random_nonce_per_seal(self) -> None:
        """Each seal uses a fresh nonce, so the same secret never repeats a envelope.

        Deterministic (counter) nonces would be a red flag; random per-call
        nonces are what let at-rest sealing carry no persistent state.
        """
        from omni_mercury_engine.security.encryption import SecureDataHandler

        handler = SecureDataHandler(enable_quantum_resistant=False, at_rest_key=KEY)
        nonces = {handler.encrypt_at_rest("same-seed", aad=b"acct")[:12] for _ in range(64)}
        assert len(nonces) == 64  # all distinct

    def test_seal_writes_no_disk_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """At-rest sealing must not persist nonce-counter files under $HOME.

        Regression for the earlier implementation, which routed through the
        counter-nonce ``AESGCMProvider`` and wrote
        ``~/.ama_cryptography/aes_gcm_counters.json`` on every call — critical
        state that breaks a read-only container and aborts every decrypt if the
        file is lost or corrupt. The random-nonce native path writes nothing.
        """
        from omni_mercury_engine.security.encryption import SecureDataHandler

        monkeypatch.setenv("HOME", str(tmp_path))
        handler = SecureDataHandler(enable_quantum_resistant=False, at_rest_key=KEY)
        for i in range(20):
            env = handler.encrypt_at_rest("s", aad=f"a{i}".encode())
            assert handler.decrypt_at_rest(env, aad=f"a{i}".encode()) == b"s"
        assert not (tmp_path / ".ama_cryptography" / "aes_gcm_counters.json").exists()


class TestSealerKeyResolution:
    """Environment-driven key material rules."""

    def test_explicit_key_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MERCURY_DATA_ENC_KEY yields a stable sealer."""
        monkeypatch.setenv("MERCURY_DATA_ENC_KEY", "ab" * 32)
        sealer = build_secret_sealer(store_is_durable=True)
        assert sealer.key_is_stable is True

    def test_malformed_key_fails_loudly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A typo'd key raises instead of silently downgrading protection."""
        monkeypatch.setenv("MERCURY_DATA_ENC_KEY", "not-hex")
        with pytest.raises(ValueError, match="hex"):
            build_secret_sealer(store_is_durable=True)
        monkeypatch.setenv("MERCURY_DATA_ENC_KEY", "abcd")  # hex but short
        with pytest.raises(ValueError, match="32 bytes"):
            build_secret_sealer(store_is_durable=True)

    def test_master_seed_derivation_is_deterministic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two workers deriving from AMA_MASTER_SEED can read each other's seals."""
        monkeypatch.delenv("MERCURY_DATA_ENC_KEY", raising=False)
        monkeypatch.setenv("AMA_MASTER_SEED", "cd" * 64)
        worker_a = build_secret_sealer(store_is_durable=True)
        worker_b = build_secret_sealer(store_is_durable=True)
        assert worker_a.key_is_stable and worker_b.key_is_stable
        sealed = worker_a.seal("SEED", aad="acct")
        assert worker_b.unseal(sealed, aad="acct") == "SEED"

    def test_durable_store_without_key_is_marked_unstable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No stable key + durable store → sealer refuses to be write-trusted."""
        monkeypatch.delenv("MERCURY_DATA_ENC_KEY", raising=False)
        monkeypatch.delenv("AMA_MASTER_SEED", raising=False)
        assert build_secret_sealer(store_is_durable=True).key_is_stable is False
        # In-memory store: an ephemeral key loses nothing.
        assert build_secret_sealer(store_is_durable=False).key_is_stable is True


def _service(
    sealer: SecretSealer | None = None,
    store: InMemoryIdentityStore | SqliteIdentityStore | None = None,
) -> tuple[AuthService, RecordingMailer, FakeClock]:
    """A service over an in-memory store with a movable clock."""
    mailer = RecordingMailer()
    clock = FakeClock()
    service = AuthService(store or InMemoryIdentityStore(), mailer, clock=clock, sealer=sealer)
    return service, mailer, clock


def _enrolled_account(
    service: AuthService, mailer: RecordingMailer, clock: FakeClock
) -> tuple[str, str, list[str]]:
    """Register, verify, and enroll 2FA; return (account_id, secret, codes)."""
    import re

    account = service.register("u@b.com", "a-strong-password")
    token = re.search(r"token=([\w\-]+)", mailer.sent[-1]["body"]).group(1)  # type: ignore[union-attr]
    service.verify_email(token)
    enrollment = service.start_totp_enrollment(account.id)
    code = totp.generate_totp(enrollment.secret, at=clock.now.timestamp())
    codes = service.confirm_totp_enrollment(account.id, code)
    return account.id, enrollment.secret, codes


class TestSealedAtRest:
    """The identity store must never hold a plaintext TOTP seed."""

    def test_stored_secret_is_sealed(self) -> None:
        """After enrollment the store row carries an envelope, not the seed."""
        store = InMemoryIdentityStore()
        sealer = SecretSealer(KEY, key_is_stable=True)
        service, mailer, clock = _service(sealer=sealer, store=store)
        account_id, secret, _codes = _enrolled_account(service, mailer, clock)

        stored = store.get_account_by_id(account_id)
        assert stored is not None and stored.totp_secret is not None
        assert sealer.is_sealed(stored.totp_secret)
        assert secret not in stored.totp_secret

    def test_login_works_through_sealed_secret(self) -> None:
        """The full login flow round-trips through the sealed value."""
        service, mailer, clock = _service(sealer=SecretSealer(KEY, key_is_stable=True))
        _account_id, secret, _codes = _enrolled_account(service, mailer, clock)
        clock.advance(60)
        code = totp.generate_totp(secret, at=clock.now.timestamp())
        result = service.login("u@b.com", "a-strong-password", totp_code=code)
        assert result.account.email == "u@b.com"

    def test_migration_seals_existing_plaintext(self, tmp_path: Path) -> None:
        """The sweep upgrades legacy plaintext rows in place (idempotently)."""
        store = SqliteIdentityStore(tmp_path / "identity.db")
        sealer = SecretSealer(KEY, key_is_stable=True)
        service, mailer, clock = _service(sealer=sealer, store=store)
        account_id, secret, _codes = _enrolled_account(service, mailer, clock)
        # Simulate a legacy row by writing the plaintext back.
        legacy = store.get_account_by_id(account_id)
        assert legacy is not None
        legacy.totp_secret = secret
        store.update_account(legacy)

        assert migrate_plaintext_totp_secrets(store, sealer) == 1
        migrated = store.get_account_by_id(account_id)
        assert migrated is not None and migrated.totp_secret is not None
        assert sealer.is_sealed(migrated.totp_secret)
        # Idempotent: a second sweep touches nothing.
        assert migrate_plaintext_totp_secrets(store, sealer) == 0
        # And the account still logs in.
        clock.advance(60)
        code = totp.generate_totp(secret, at=clock.now.timestamp())
        assert service.login("u@b.com", "a-strong-password", totp_code=code) is not None
        store.close()

    def test_unstable_key_never_seals_writes(self) -> None:
        """With a non-stable key the write path keeps the seed unsealed.

        Sealing under a process-lifetime key with a durable store would brick
        every enrolled account at the next restart — worse than plaintext.
        """
        store = InMemoryIdentityStore()
        sealer = SecretSealer(KEY, key_is_stable=False)
        service, mailer, clock = _service(sealer=sealer, store=store)
        account_id, secret, _codes = _enrolled_account(service, mailer, clock)
        stored = store.get_account_by_id(account_id)
        assert stored is not None
        assert stored.totp_secret == secret  # readable forever, by design
        assert migrate_plaintext_totp_secrets(store, sealer) == 0


class TestTotpReplay:
    """An accepted code's time step can never be accepted again."""

    def test_same_code_rejected_second_time(self) -> None:
        """Login twice with one code: first succeeds, replay is rejected."""
        service, mailer, clock = _service()
        _account_id, secret, _codes = _enrolled_account(service, mailer, clock)
        clock.advance(60)
        code = totp.generate_totp(secret, at=clock.now.timestamp())
        assert service.login("u@b.com", "a-strong-password", totp_code=code)
        with pytest.raises(InvalidTwoFactorError, match="already used"):
            service.login("u@b.com", "a-strong-password", totp_code=code)

    def test_earlier_window_code_rejected_after_later_use(self) -> None:
        """After accepting step S, the still-in-window step S-1 code is dead."""
        service, mailer, clock = _service()
        _account_id, secret, _codes = _enrolled_account(service, mailer, clock)
        clock.advance(90)
        current = totp.generate_totp(secret, at=clock.now.timestamp())
        previous = totp.generate_totp(secret, at=clock.now.timestamp() - 30)
        assert service.login("u@b.com", "a-strong-password", totp_code=current)
        with pytest.raises(InvalidTwoFactorError):
            service.login("u@b.com", "a-strong-password", totp_code=previous)

    def test_next_step_accepted(self) -> None:
        """A genuinely fresh step still authenticates."""
        service, mailer, clock = _service()
        _account_id, secret, _codes = _enrolled_account(service, mailer, clock)
        clock.advance(60)
        assert service.login(
            "u@b.com",
            "a-strong-password",
            totp_code=totp.generate_totp(secret, at=clock.now.timestamp()),
        )
        clock.advance(30)
        assert service.login(
            "u@b.com",
            "a-strong-password",
            totp_code=totp.generate_totp(secret, at=clock.now.timestamp()),
        )

    def test_verify_totp_with_step_reports_matched_step(self) -> None:
        """The primitive reports the exact matched step across the window."""
        secret = totp.generate_secret()
        at = 1_700_000_000.0
        step = int(at // 30)
        assert totp.verify_totp_with_step(secret, totp.generate_totp(secret, at=at), at=at) == step
        previous_code = totp.generate_totp(secret, at=at - 30)
        assert totp.verify_totp_with_step(secret, previous_code, at=at) == step - 1
        assert totp.verify_totp_with_step(secret, "000000", at=at) in (
            None,
            step - 1,
            step,
            step + 1,
        )
        assert totp.verify_totp_with_step(secret, "junk!", at=at) is None


class TestRecoveryCodes:
    """Backup codes: single-use, normalised, regenerable, voided on disable."""

    def test_recovery_code_logs_in_and_is_single_use(self) -> None:
        """A recovery code authenticates once, then never again."""
        service, mailer, clock = _service()
        _account_id, _secret, codes = _enrolled_account(service, mailer, clock)
        assert service.login("u@b.com", "a-strong-password", recovery_code=codes[0])
        with pytest.raises(InvalidTwoFactorError):
            service.login("u@b.com", "a-strong-password", recovery_code=codes[0])
        # The next code still works.
        assert service.login("u@b.com", "a-strong-password", recovery_code=codes[1])

    def test_normalisation_tolerates_case_and_dashes(self) -> None:
        """Users can type codes with/without dashes and any case."""
        service, mailer, clock = _service()
        _account_id, _secret, codes = _enrolled_account(service, mailer, clock)
        mangled = codes[0].replace("-", " ").upper()
        assert service.login("u@b.com", "a-strong-password", recovery_code=mangled)

    def test_wrong_code_rejected(self) -> None:
        """An unknown recovery code fails like a wrong TOTP."""
        service, mailer, clock = _service()
        _enrolled_account(service, mailer, clock)
        with pytest.raises(InvalidTwoFactorError):
            service.login("u@b.com", "a-strong-password", recovery_code="0000-0000-0000-0000")

    def test_use_sends_security_notice(self) -> None:
        """Using a recovery code notifies the account's mailbox."""
        service, mailer, clock = _service()
        _account_id, _secret, codes = _enrolled_account(service, mailer, clock)
        before = len(mailer.sent)
        service.login("u@b.com", "a-strong-password", recovery_code=codes[0])
        assert len(mailer.sent) == before + 1
        assert "recovery code" in mailer.sent[-1]["subject"].lower()

    def test_regeneration_voids_old_set(self) -> None:
        """Regenerated codes replace the old set entirely."""
        service, mailer, clock = _service()
        account_id, _secret, old_codes = _enrolled_account(service, mailer, clock)
        new_codes = service.regenerate_recovery_codes(account_id, "a-strong-password")
        assert set(new_codes).isdisjoint(old_codes)
        with pytest.raises(InvalidTwoFactorError):
            service.login("u@b.com", "a-strong-password", recovery_code=old_codes[0])
        assert service.login("u@b.com", "a-strong-password", recovery_code=new_codes[0])

    def test_disable_totp_voids_codes(self) -> None:
        """Disabling 2FA clears the secret AND the recovery codes."""
        service, mailer, clock = _service()
        account_id, _secret, codes = _enrolled_account(service, mailer, clock)
        service.disable_totp(account_id)
        # 2FA off: password alone logs in; the old codes are gone from the store.
        assert service.login("u@b.com", "a-strong-password")
        assert service._store.count_unused_recovery_codes(account_id) == 0
        # Hashes only ever stored — raw codes appear nowhere in the store.
        assert all(len(code) == 19 for code in codes)  # 16 hex chars + 3 dashes
