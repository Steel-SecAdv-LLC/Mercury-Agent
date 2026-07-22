# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the scrypt password KDF upgrade and legacy verification.

Pins the memory-hard KDF migration: new hashes are scrypt with the configured
cost, legacy PBKDF2 hashes still verify (and are flagged for transparent
rehash), verification fails closed on every malformed/tampered form, and the
work-factor parameters are honoured and bounded (a tampered stored parameter
can't turn verification into a resource bomb).
"""

from __future__ import annotations

import hashlib

import pytest

from omni_mercury_engine.api import passwords


class TestScryptHashing:
    """New hashes use scrypt with the configured, self-describing parameters."""

    def test_new_hash_is_scrypt(self) -> None:
        """A fresh hash carries the scrypt algorithm tag and parameters."""
        stored = passwords.hash_password("correct horse battery staple")
        parts = stored.split("$")
        assert parts[0] == "scrypt"
        assert int(parts[1]) == passwords.SCRYPT_N
        assert int(parts[2]) == passwords.SCRYPT_R
        assert int(parts[3]) == passwords.SCRYPT_P

    def test_roundtrip_and_rejects_wrong(self) -> None:
        """A password verifies against its own hash and rejects a wrong one."""
        stored = passwords.hash_password("s3cure-passphrase")
        assert passwords.verify_password("s3cure-passphrase", stored)
        assert not passwords.verify_password("wrong", stored)

    def test_salted_per_call(self) -> None:
        """The same password hashes differently each time."""
        assert passwords.hash_password("same") != passwords.hash_password("same")

    def test_empty_password_rejected(self) -> None:
        """An empty password cannot be hashed."""
        with pytest.raises(ValueError, match="must not be empty"):
            passwords.hash_password("")

    def test_verify_fails_closed_on_garbage(self) -> None:
        """Malformed stored values return False rather than raising."""
        for bad in ["", "nope", "scrypt$only", "scrypt$a$b$c$d$e", "unknown$1$2$3", 12345]:
            assert not passwords.verify_password("x", bad)  # type: ignore[arg-type]

    def test_tampered_scrypt_params_bounded(self) -> None:
        """An absurd stored ``n`` fails closed instead of exhausting memory."""
        # n far above the verification ceiling must be rejected, not attempted.
        malicious = f"scrypt${2**30}$8$1$" + "00" * 16 + "$" + "00" * 32
        assert not passwords.verify_password("x", malicious)
        # Non-power-of-two n is invalid scrypt and rejected.
        assert not passwords.verify_password("x", "scrypt$1000$8$1$" + "00" * 16 + "$" + "00" * 32)


class TestLegacyPbkdf2:
    """Pre-scrypt PBKDF2 hashes still verify and are flagged for upgrade."""

    def _legacy(self, password: str, iterations: int = 600_000) -> str:
        salt = bytes.fromhex("11" * 16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
        return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"

    def test_legacy_verifies(self) -> None:
        """A legacy PBKDF2 hash still authenticates the right password."""
        stored = self._legacy("legacy-pw")
        assert passwords.verify_password("legacy-pw", stored)
        assert not passwords.verify_password("nope", stored)

    def test_legacy_flagged_for_rehash(self) -> None:
        """Any legacy hash is reported as needing an upgrade to scrypt."""
        assert passwords.needs_rehash(self._legacy("pw"))
        assert passwords.needs_rehash(self._legacy("pw", iterations=100))

    def test_scrypt_not_flagged_when_current(self) -> None:
        """A current-parameter scrypt hash does not need rehashing."""
        assert not passwords.needs_rehash(passwords.hash_password("pw"))

    def test_weak_scrypt_flagged(self) -> None:
        """An scrypt hash below the configured cost is flagged for upgrade."""
        parts = passwords.hash_password("pw").split("$")
        parts[1] = str(passwords.SCRYPT_N // 2)  # halve n
        assert passwords.needs_rehash("$".join(parts))
        parts = passwords.hash_password("pw").split("$")
        parts[3] = str(max(1, passwords.SCRYPT_P - 1))  # lower p
        if passwords.SCRYPT_P > 1:
            assert passwords.needs_rehash("$".join(parts))


class TestEnvironmentTuning:
    """Cost parameters are configurable and validated."""

    def test_env_overrides_and_bounds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MERCURY_SCRYPT_* is honoured; junk and non-powers-of-two fall back."""
        import importlib

        monkeypatch.setenv("MERCURY_SCRYPT_N", str(2**14))
        monkeypatch.setenv("MERCURY_SCRYPT_R", "8")
        monkeypatch.setenv("MERCURY_SCRYPT_P", "2")
        reloaded = importlib.reload(passwords)
        try:
            assert reloaded.SCRYPT_N == 2**14
            assert reloaded.SCRYPT_P == 2
            # A non-power-of-two n falls back to the default.
            monkeypatch.setenv("MERCURY_SCRYPT_N", "12345")
            again = importlib.reload(passwords)
            assert again.SCRYPT_N == 2**15
        finally:
            # Restore the module to its unpatched configured state for other tests.
            monkeypatch.undo()
            importlib.reload(passwords)
