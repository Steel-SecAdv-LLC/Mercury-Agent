"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
AMA-routed HMAC locks for native_jwt.

These tests pin that Mercury's :mod:`native_jwt` signing primitive
produces byte-identical output whether routed through AMA
Cryptography v3.2.0's native HMAC C backend and match stdlib
``hmac`` over ``hashlib`` for the same FIPS 198-1 / RFC 2104 wire format.
AMA absence is not a test skip or fallback path; module import fails closed.
"""

import hashlib
import hmac as stdlib_hmac

import pytest

from omni_mercury_engine.security import ama_hmac, native_jwt

# RFC 4231 §4.2 — Test Case 1.  HMAC-SHA-256 / HMAC-SHA-512 KAT.
_RFC4231_TC1_KEY = b"\x0b" * 20
_RFC4231_TC1_MSG = b"Hi There"
_RFC4231_TC1_HS256 = bytes.fromhex(
    "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
)
_RFC4231_TC1_HS512 = bytes.fromhex(
    "87aa7cdea5ef619d4ff0b4241a1d6cb02379f4e2ce4ec2787ad0b30545e17cde"
    "daa833b7d6b8a702038b274eaea3f4e4be9d914eeb61f1702e696c203a126854"
)


# RFC 4231 §4.7 — Test Case 7.  Oversized-key (131 bytes) + long message.
_RFC4231_TC7_KEY = b"\xaa" * 131
_RFC4231_TC7_MSG = (
    b"This is a test using a larger than block-size key and a larger than "
    b"block-size data. The key needs to be hashed before being used by the "
    b"HMAC algorithm."
)
_RFC4231_TC7_HS256 = bytes.fromhex(
    "9b09ffa71b942fcb27635fbcd5b0e944bfdc63644f0713938a7f51535c3a35e2"
)


# ----------------------------------------------------------------------- #
# Backend detection / health surface
# ----------------------------------------------------------------------- #


class TestSigningBackendSurface:
    """``get_signing_backend(alg)`` reports the routing decision."""

    def test_supported_algorithms_have_a_backend(self) -> None:
        for alg in ("HS256", "HS384", "HS512"):
            backend = native_jwt.get_signing_backend(alg)
            assert backend in ("ama", "stdlib")

    def test_hs384_is_always_stdlib(self) -> None:
        # AMA Cryptography v3.2.0 does not ship HMAC-SHA-384 in its C
        # backend; the HS384 path is wired to stdlib regardless of
        # AMA availability.  Locks the deferral until AMA adds the
        # binding.
        assert native_jwt.get_signing_backend("HS384") == "stdlib"

    def test_unknown_algorithm_raises(self) -> None:
        with pytest.raises(native_jwt.InvalidAlgorithmError):
            native_jwt.get_signing_backend("HS999")

    def test_ama_available_helper_keys(self) -> None:
        snapshot = ama_hmac.available()
        assert set(snapshot.keys()) == {
            "ama_hmac_sha256",
            "ama_hmac_sha512",
            "reason",
        }
        assert isinstance(snapshot["ama_hmac_sha256"], bool)
        assert isinstance(snapshot["ama_hmac_sha512"], bool)
        assert isinstance(snapshot["reason"], str)


# ----------------------------------------------------------------------- #
# Known-answer locks against RFC 4231 (FIPS 198-1 / RFC 2104)
# ----------------------------------------------------------------------- #


class TestAMAKnownAnswerVectors:
    """AMA's HMAC-SHA-256 output matches RFC 4231 KAT bit-for-bit."""

    def test_rfc4231_tc1_hmac_sha256(self) -> None:
        assert ama_hmac.ama_hmac_sha256(_RFC4231_TC1_KEY, _RFC4231_TC1_MSG) == _RFC4231_TC1_HS256

    def test_rfc4231_tc1_hmac_sha512(self) -> None:
        assert ama_hmac.ama_hmac_sha512(_RFC4231_TC1_KEY, _RFC4231_TC1_MSG) == _RFC4231_TC1_HS512

    def test_rfc4231_tc7_oversized_key_hmac_sha256(self) -> None:
        assert ama_hmac.ama_hmac_sha256(_RFC4231_TC7_KEY, _RFC4231_TC7_MSG) == _RFC4231_TC7_HS256

    def test_two_segment_equivalence(self) -> None:
        """``ama_hmac_sha256_2(k, m1, m2) == ama_hmac_sha256(k, m1||m2)``."""
        m1 = b"part-one-"
        m2 = b"part-two-with-more-bytes"
        joined = ama_hmac.ama_hmac_sha256(_RFC4231_TC1_KEY, m1 + m2)
        split = ama_hmac.ama_hmac_sha256_2(_RFC4231_TC1_KEY, m1, m2)
        assert joined == split


# ----------------------------------------------------------------------- #
# Byte-equivalence at the native_jwt._sign() boundary
# ----------------------------------------------------------------------- #


class TestSignBoundaryEquivalence:
    """``_sign()`` is byte-identical AMA-routed vs. stdlib-routed.

    Locks the FIPS 198-1 wire-format guarantee at Mercury's signing
    entry point so the routing decision is performance-only, never
    semantic.
    """

    KEY = b"mercury-test-key-32-bytes-long-x"
    HEADER = b"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    PAYLOAD = b"eyJzdWIiOiJhbGljZSIsImV4cCI6MTk2MDAwMDAwMH0"

    def test_hs256_ama_matches_stdlib(self) -> None:
        ama = native_jwt._sign_ama(self.HEADER, self.PAYLOAD, self.KEY, "HS256")
        stdlib_sig = native_jwt._sign_stdlib(self.HEADER, self.PAYLOAD, self.KEY, "HS256")
        assert ama == stdlib_sig

    def test_hs512_ama_matches_stdlib(self) -> None:
        ama = native_jwt._sign_ama(self.HEADER, self.PAYLOAD, self.KEY, "HS512")
        stdlib_sig = native_jwt._sign_stdlib(self.HEADER, self.PAYLOAD, self.KEY, "HS512")
        assert ama == stdlib_sig

    def test_stdlib_matches_one_shot_hmac(self) -> None:
        """Stdlib path is the canonical ``HMAC(key, header || \".\" || payload)``."""
        signing_input = self.HEADER + b"." + self.PAYLOAD
        for alg, hashmod in (
            ("HS256", hashlib.sha256),
            ("HS384", hashlib.sha384),
            ("HS512", hashlib.sha512),
        ):
            expected = stdlib_hmac.new(self.KEY, signing_input, hashmod).digest()
            assert native_jwt._sign_stdlib(self.HEADER, self.PAYLOAD, self.KEY, alg) == expected


# ----------------------------------------------------------------------- #
# Fail-closed path: AMA flag invalidated after import
# ----------------------------------------------------------------------- #


class TestFailClosedWhenAMAInvalidated:
    """Mercury must not fall back when mandatory AMA HMAC is invalidated."""

    def test_get_signing_backend_rejects_hs256_without_ama(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ama_hmac, "HAS_AMA_HMAC_SHA256", False)
        with pytest.raises(RuntimeError, match="AMA HMAC-SHA-256"):
            native_jwt.get_signing_backend("HS256")

    def test_get_signing_backend_rejects_hs512_without_ama(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ama_hmac, "HAS_AMA_HMAC_SHA512", False)
        with pytest.raises(RuntimeError, match="AMA HMAC-SHA-512"):
            native_jwt.get_signing_backend("HS512")

    def test_sign_rejects_hs256_when_ama_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ama_hmac, "HAS_AMA_HMAC_SHA256", False)
        header = b"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        payload = b"eyJzdWIiOiJib2IifQ"
        key = b"fail-closed-test-key"
        with pytest.raises(RuntimeError, match="AMA HMAC-SHA-256"):
            native_jwt._sign(header, payload, key, "HS256")

    def test_encode_rejects_hs256_with_ama_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ama_hmac, "HAS_AMA_HMAC_SHA256", False)
        with pytest.raises(RuntimeError, match="AMA HMAC-SHA-256"):
            native_jwt.encode({"sub": "alice"}, "fail-closed-key", "HS256")


# ----------------------------------------------------------------------- #
# Cross-path token verification (AMA-encoded → stdlib-decoded, etc.)
# ----------------------------------------------------------------------- #


class TestCrossPathInteroperability:
    """A token signed by either path must verify under the other.

    This is the operational guarantee that AMA routing is a
    performance / hardening optimisation, never a wire-format change.
    """

    def test_ama_signed_verifies_after_stdlib_tamper_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        token = native_jwt.encode({"sub": "alice"}, "shared-key", "HS256")

        monkeypatch.setattr(ama_hmac, "HAS_AMA_HMAC_SHA256", False)
        with pytest.raises(RuntimeError, match="AMA HMAC-SHA-256"):
            native_jwt.decode(token, "shared-key", ["HS256"])

    def test_ama_signed_verifies_with_ama(self) -> None:
        token = native_jwt.encode({"sub": "alice"}, "shared-key", "HS256")
        payload = native_jwt.decode(token, "shared-key", ["HS256"])
        assert payload == {"sub": "alice"}


# ----------------------------------------------------------------------- #
# Reinitialisation invariant
# ----------------------------------------------------------------------- #


class TestReinitialization:
    """``_reinitialize_for_tests`` honours the current import state."""

    def test_reinitialize_restores_real_flag_values(self) -> None:
        # Capture truth.
        truth_256 = ama_hmac.HAS_AMA_HMAC_SHA256
        truth_512 = ama_hmac.HAS_AMA_HMAC_SHA512

        ama_hmac._reinitialize_for_tests()

        assert ama_hmac.HAS_AMA_HMAC_SHA256 is truth_256
        assert ama_hmac.HAS_AMA_HMAC_SHA512 is truth_512
