# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The sovereign emergency tier computes without OpenSSL.

``crypto/sovereign.py`` exists so that, when AMA fails at runtime, Mercury
need not "silently borrow a third-party library". It nevertheless built
HKDF on ``hmac.new(..., hashlib.sha256)``, hashed the Ed25519 challenge with
``hashlib.sha512`` and compared AEAD tags with ``hmac.compare_digest``. On an
OpenSSL-linked CPython those resolve to ``openssl_sha256``,
``openssl_sha512``, OpenSSL's HMAC and ``CRYPTO_memcmp``: every sovereign
HKDF, HMAC and challenge hash ran inside OpenSSL.

Two pins:

* **In process**: with ``hashlib``'s and ``hmac``'s entry points replaced by
  tripwires, every sovereign primitive still reproduces its RFC vectors.
  The previous implementation trips them.
* **Out of process**: with OpenSSL's ``_hashlib`` made unimportable before
  anything loads, the tier still reproduces the vectors.
"""

from __future__ import annotations

import hashlib
import hmac
import subprocess
import sys
import textwrap
from typing import Any

import pytest

from omni_mercury_engine.crypto import sovereign

# RFC 5869 appendix A.1.
_IKM = bytes.fromhex("0b" * 22)
_SALT = bytes.fromhex("000102030405060708090a0b0c")
_INFO = bytes.fromhex("f0f1f2f3f4f5f6f7f8f9")
_OKM = bytes.fromhex(
    "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865"
)


def _tripwire(name: str) -> Any:
    def _raise(*_: Any, **__: Any) -> Any:
        raise AssertionError(f"sovereign tier called OpenSSL-backed {name}")

    return _raise


@pytest.fixture
def openssl_entry_points_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the stdlib entry points that dispatch to OpenSSL."""
    for name in ("sha256", "sha512", "new"):
        monkeypatch.setattr(hashlib, name, _tripwire(f"hashlib.{name}"))
    monkeypatch.setattr(hmac, "new", _tripwire("hmac.new"))
    monkeypatch.setattr(hmac, "digest", _tripwire("hmac.digest"))
    monkeypatch.setattr(hmac, "compare_digest", _tripwire("hmac.compare_digest"))


@pytest.mark.usefixtures("openssl_entry_points_trip")
class TestNoStdlibCryptoInProcess:
    """Every primitive runs with the OpenSSL-dispatching entry points tripped."""

    def test_hkdf_rfc5869_a1(self) -> None:
        assert sovereign.hkdf_sha256(_IKM, 42, salt=_SALT, info=_INFO) == _OKM

    def test_hkdf_long_key_hmac_path(self) -> None:
        # A salt longer than the 64-byte block exercises the key-hash branch
        # of RFC 2104; RFC 5869 A.2 uses an 80-byte salt.
        out = sovereign.hkdf_sha256(
            bytes(range(0x50)), 82, salt=bytes(range(0x60, 0xB0)), info=bytes(range(0xB0, 0x100))
        )
        assert out.hex().startswith("b11e398dc80327a1c8e7f78c596a4934")

    def test_ed25519_verify(self) -> None:
        pk = bytes.fromhex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a")
        sig = bytes.fromhex(
            "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
            "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"
        )
        assert sovereign.ed25519_verify(b"", sig, pk) is True
        assert sovereign.ed25519_verify(b"x", sig, pk) is False

    def test_aead_round_trip_and_tag_check(self) -> None:
        key, nonce = bytes(range(32)), bytes(12)
        sealed = sovereign.chacha20poly1305_encrypt(key, nonce, b"payload", b"aad")
        assert sovereign.chacha20poly1305_decrypt(key, nonce, sealed, b"aad") == b"payload"
        tampered = sealed[:-1] + bytes([sealed[-1] ^ 1])
        with pytest.raises(ValueError, match="authentication tag mismatch"):
            sovereign.chacha20poly1305_decrypt(key, nonce, tampered, b"aad")


def test_bound_constructors_are_not_openssl() -> None:
    """The resolved SHA-2 constructors are the interpreter's, not ``_hashlib``'s."""
    for factory in (sovereign._SHA256, sovereign._SHA512):
        assert factory is not None
        assert type(factory(b"")).__module__ != "_hashlib"
        assert "openssl" not in repr(factory)


def test_missing_builtin_hash_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """No built-in SHA-256 means a named refusal, never an OpenSSL fallback."""
    monkeypatch.setattr(sovereign, "_SHA256", None)
    with pytest.raises(RuntimeError, match="refuses to fall back to OpenSSL"):
        sovereign.hkdf_sha256(_IKM, 32)


def test_vectors_hold_with_openssl_unimportable() -> None:
    """A fresh interpreter in which ``_hashlib`` cannot be imported."""
    script = textwrap.dedent(f"""
        import sys
        sys.modules["_hashlib"] = None  # OpenSSL binding unimportable
        from omni_mercury_engine.crypto import sovereign
        okm = sovereign.hkdf_sha256(
            bytes.fromhex({_IKM.hex()!r}), 42,
            salt=bytes.fromhex({_SALT.hex()!r}), info=bytes.fromhex({_INFO.hex()!r}),
        )
        assert okm.hex() == {_OKM.hex()!r}, okm.hex()
        assert "_hashlib" not in {{k for k, v in sys.modules.items() if v is not None}}
        print("ok")
        """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
