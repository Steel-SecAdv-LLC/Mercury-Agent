# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mercury's operational HMAC, HKDF, SHA-3, PBKDF2 and constant-time compare run on AMA.

Four shipped modules computed keyed hashes and key derivations with the
stdlib: the at-rest secret sealer's HKDF-SHA-256 (``api/secret_sealer.py``),
the constant-time helpers' HMAC-SHA-256 (``security/constant_time.py``),
the tamper-evident audit chain's HMAC-SHA3-256 and torn-fragment SHA3-256
(``security/secure_audit_logging.py``), and legacy password-row verification's
PBKDF2-HMAC-SHA-256 (``api/passwords.py``) -- plus ``hmac.compare_digest`` in
all of them. On an OpenSSL-linked CPython, which is what the shipped
``python:3.14-slim`` image is, every one of those calls executed inside
OpenSSL while the repository states that AMA Cryptography owns the
operational crypto path and that third-party crypto libraries are benchmark
and validation targets only.

Two things are pinned here:

* **Compatibility**: the AMA-routed outputs are byte-identical to the stdlib
  formulation on fixed vectors, so secrets sealed, audit chains written and
  password rows stored before the routing still open and verify.
* **Routing**: with ``hashlib``'s and ``hmac``'s entry points replaced by
  tripwires, each module still produces those same outputs. The previous
  implementation trips them.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Any

import pytest

from omni_mercury_engine.api import passwords, secret_sealer
from omni_mercury_engine.security import ama_hmac, constant_time
from omni_mercury_engine.security.encryption import QuantumResistantEncryption
from omni_mercury_engine.security.secure_audit_logging import SecureHashChain

_SEED = bytes(range(64))
_KEY = b"\x42" * 32
_MSG = b"mercury-agent operational crypto routing"
_SALT = b"\x7f" * 16

# Stdlib reference values, computed once at import (before any tripwire).
_REF_PRK = hmac.new(b"mercury-hkdf-salt-v1", _SEED, hashlib.sha256).digest()
_REF_HKDF_SEALER = hmac.new(
    _REF_PRK, b"mercury-agent/totp-at-rest/v1" + b"\x01", hashlib.sha256
).digest()
_REF_HMAC256 = hmac.new(_KEY, _MSG, hashlib.sha256).digest()
_REF_HMAC_SHA3 = hmac.new(_KEY, _MSG, hashlib.sha3_256).hexdigest()
_REF_SHA3 = hashlib.sha3_256(_MSG).digest()
_REF_PBKDF2 = hashlib.pbkdf2_hmac("sha256", b"correct horse", _SALT, 2000)
_LEGACY_ROW = f"pbkdf2_sha256$2000${_SALT.hex()}${_REF_PBKDF2.hex()}"


def _tripwire(name: str) -> Any:
    def _raise(*_: Any, **__: Any) -> Any:
        raise AssertionError(f"operational path called OpenSSL-backed {name}")

    return _raise


@pytest.fixture
def openssl_entry_points_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace every stdlib entry point that dispatches to OpenSSL."""
    for name in ("sha256", "sha512", "sha3_256", "sha3_512", "pbkdf2_hmac", "scrypt", "new"):
        monkeypatch.setattr(hashlib, name, _tripwire(f"hashlib.{name}"))
    for name in ("new", "digest", "compare_digest"):
        monkeypatch.setattr(hmac, name, _tripwire(f"hmac.{name}"))


class TestAdapterMatchesStdlib:
    """Byte-identical to the stdlib formulation: data at rest stays readable."""

    def test_hkdf_sha256_rfc5869_a1(self) -> None:
        ikm = bytes.fromhex("0b" * 22)
        salt = bytes.fromhex("000102030405060708090a0b0c")
        info = bytes.fromhex("f0f1f2f3f4f5f6f7f8f9")
        okm = bytes.fromhex(
            "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865"
        )
        assert ama_hmac.ama_hkdf_sha256(ikm, 42, salt=salt, info=info) == okm

    def test_hmac_sha3_sha3_pbkdf2_equal_stdlib(self) -> None:
        assert ama_hmac.ama_hmac_sha3_256(_KEY, _MSG).hex() == _REF_HMAC_SHA3
        assert ama_hmac.ama_sha3_256(_MSG) == _REF_SHA3
        assert ama_hmac.ama_pbkdf2_hmac_sha256(b"correct horse", _SALT, 2000, 32) == _REF_PBKDF2

    def test_constant_time_compare_semantics(self) -> None:
        assert ama_hmac.ama_constant_time_compare(b"abc", b"abc") is True
        assert ama_hmac.ama_constant_time_compare(b"abc", b"abd") is False
        assert ama_hmac.ama_constant_time_compare(b"ab", b"abc") is False
        assert ama_hmac.ama_constant_time_compare(b"", b"") is True

    def test_available_reports_every_routed_primitive(self) -> None:
        snapshot = ama_hmac.available()
        for name in (
            "ama_hmac_sha256",
            "ama_hkdf_sha256",
            "ama_hmac_sha3_256",
            "ama_sha3_256",
            "ama_pbkdf2_sha256",
        ):
            assert snapshot[name] is True, snapshot


class TestModulesRunWithoutOpenSSL:
    """Each shipped module reproduces the stdlib result with hashlib/hmac tripwired."""

    def test_secret_sealer_hkdf(self, openssl_entry_points_trip: None) -> None:
        derived = secret_sealer._hkdf_sha256(_SEED, b"mercury-agent/totp-at-rest/v1")
        assert derived == _REF_HKDF_SEALER

    def test_constant_time_helpers(self, openssl_entry_points_trip: None) -> None:
        assert constant_time.secure_hash(_MSG, salt=_KEY) == _REF_HMAC256
        assert constant_time.secure_signature_verify(_REF_HMAC256, _MSG, key=_KEY) is True
        assert constant_time.secure_signature_verify(b"\x00" * 32, _MSG, key=_KEY) is False
        assert constant_time.constant_time_compare("same", b"same") is True
        assert constant_time.constant_time_compare("same", "other") is False

    def test_audit_chain_hash(self, openssl_entry_points_trip: None) -> None:
        chain = SecureHashChain(hmac_key=_KEY)
        assert chain._hmac_hash(_MSG) == _REF_HMAC_SHA3
        event_hash, previous, _ = chain.compute_event_hash({"event": "probe"})
        assert len(event_hash) == 64 and isinstance(previous, str)

    def test_encryption_tag_verify(self, openssl_entry_points_trip: None) -> None:
        crypto = QuantumResistantEncryption()
        tag = crypto.sign_data(_MSG)
        assert crypto.verify_signature(_MSG, tag) is True
        assert crypto.verify_signature(_MSG, b"\x00" * len(tag)) is False

    def test_legacy_password_row_verifies(self, openssl_entry_points_trip: None) -> None:
        assert passwords.verify_password("correct horse", _LEGACY_ROW) is True
        assert passwords.verify_password("battery staple", _LEGACY_ROW) is False
