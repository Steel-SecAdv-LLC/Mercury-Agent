# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mercury's sovereign emergency crypto tier — RFC pins and degrade semantics.

Three contracts under test:

1. **The primitives are correct**, pinned to their RFCs' own vectors
   (RFC 8439 §2.8.2 AEAD, RFC 5869 appendix A, RFC 8032 §7.1) and
   cross-checked against independent implementations (AMA's native backend,
   and pyca ``cryptography`` used here strictly as a *test oracle* — its
   production role ended with the sovereign tier).
2. **The tiers interoperate**: the ``ciphertext || tag`` wire format and the
   deterministic HKDF mean data sealed or keys derived by one tier open
   byte-identically under another — including data sealed by the retired
   PyCA tier.
3. **Degradation is governed**: only an AMA *infrastructure* failure
   (``NativeBackendUnavailableError`` / ``CryptoModuleError``) trips the
   sovereign tier; an authentication failure never does (a forgery does not
   get a second verifier); AES-GCM refuses loudly instead of silently
   switching algorithms; Ed25519 verification degrades while signing stays
   fail-closed.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from omni_mercury_engine.crypto import sovereign

# =============================================================================
# 1. RFC vector pins
# =============================================================================

_RFC8439_KEY = bytes(range(0x80, 0xA0))
_RFC8439_NONCE = bytes.fromhex("070000004041424344454647")
_RFC8439_AAD = bytes.fromhex("50515253c0c1c2c3c4c5c6c7")
_RFC8439_PLAINTEXT = (
    b"Ladies and Gentlemen of the class of '99: If I could offer you "
    b"only one tip for the future, sunscreen would be it."
)
_RFC8439_CIPHERTEXT = bytes.fromhex(
    "d31a8d34648e60db7b86afbc53ef7ec2a4aded51296e08fea9e2b5a736ee62d6"
    "3dbea45e8ca9671282fafb69da92728b1a71de0a9e060b2905d6a5b67ecd3b36"
    "92ddbd7f2d778b8c9803aee328091b58fab324e4fad675945585808b4831d7bc"
    "3ff4def08e4b7a9de576d26586cec64b6116"
)
_RFC8439_TAG = bytes.fromhex("1ae10b594f09e26a7e902ecbd0600691")


class TestRfc8439Aead:
    def test_section_2_8_2_encrypt_vector(self) -> None:
        out = sovereign.chacha20poly1305_encrypt(
            _RFC8439_KEY, _RFC8439_NONCE, _RFC8439_PLAINTEXT, _RFC8439_AAD
        )
        assert out == _RFC8439_CIPHERTEXT + _RFC8439_TAG

    def test_section_2_8_2_decrypt_round_trip(self) -> None:
        out = sovereign.chacha20poly1305_decrypt(
            _RFC8439_KEY, _RFC8439_NONCE, _RFC8439_CIPHERTEXT + _RFC8439_TAG, _RFC8439_AAD
        )
        assert out == _RFC8439_PLAINTEXT

    @pytest.mark.parametrize("flip_at", [0, 60, -17, -1])
    def test_any_tampered_byte_is_rejected_before_plaintext(self, flip_at: int) -> None:
        blob = bytearray(_RFC8439_CIPHERTEXT + _RFC8439_TAG)
        blob[flip_at] ^= 0x01
        with pytest.raises(ValueError, match=r"Decryption failed|tag"):
            sovereign.chacha20poly1305_decrypt(
                _RFC8439_KEY, _RFC8439_NONCE, bytes(blob), _RFC8439_AAD
            )

    def test_aad_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            sovereign.chacha20poly1305_decrypt(
                _RFC8439_KEY, _RFC8439_NONCE, _RFC8439_CIPHERTEXT + _RFC8439_TAG, b"wrong aad"
            )

    def test_input_validation(self) -> None:
        with pytest.raises(ValueError, match="key"):
            sovereign.chacha20poly1305_encrypt(b"short", _RFC8439_NONCE, b"x")
        with pytest.raises(ValueError, match="nonce"):
            sovereign.chacha20poly1305_encrypt(_RFC8439_KEY, b"short", b"x")
        with pytest.raises(ValueError, match="tag"):
            sovereign.chacha20poly1305_decrypt(_RFC8439_KEY, _RFC8439_NONCE, b"too-short")


class TestRfc5869Hkdf:
    def test_appendix_a1_basic(self) -> None:
        okm = sovereign.hkdf_sha256(
            bytes.fromhex("0b" * 22),
            42,
            bytes.fromhex("000102030405060708090a0b0c"),
            bytes.fromhex("f0f1f2f3f4f5f6f7f8f9"),
        )
        assert okm == bytes.fromhex(
            "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865"
        )

    def test_appendix_a2_long_inputs(self) -> None:
        okm = sovereign.hkdf_sha256(
            bytes(range(0x00, 0x50)),
            82,
            bytes(range(0x60, 0xB0)),
            bytes(range(0xB0, 0x100)),
        )
        assert okm == bytes.fromhex(
            "b11e398dc80327a1c8e7f78c596a49344f012eda2d4efad8a050cc4c19afa97c"
            "59045a99cac7827271cb41c65e590e09da3275600c2f09b8367793a9aca3db71"
            "cc30c58179ec3e87c14c01d5c1f3434f1d87"
        )

    def test_appendix_a3_no_salt_no_info(self) -> None:
        okm = sovereign.hkdf_sha256(bytes.fromhex("0b" * 22), 42)
        assert okm == bytes.fromhex(
            "8da4e775a563c18f715f802a063c5a31b8a11f5c5ee1879ec3454e5f3c738d2d9d201395faa4b61a96c8"
        )

    def test_output_length_bounds(self) -> None:
        with pytest.raises(ValueError):
            sovereign.hkdf_sha256(b"ikm", 0)
        with pytest.raises(ValueError):
            sovereign.hkdf_sha256(b"ikm", 255 * 32 + 1)


_RFC8032_VECTORS = [
    # (public_key_hex, message_hex, signature_hex) — RFC 8032 §7.1 TEST 1-3.
    (
        "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
        "",
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
        "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b",
    ),
    (
        "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
        "72",
        "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da"
        "085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00",
    ),
    (
        "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025",
        "af82",
        "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac"
        "18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a",
    ),
]

_ED25519_L = 2**252 + 27742317777372353535851937790883648493


class TestRfc8032Ed25519Verify:
    @pytest.mark.parametrize("pk_hex,msg_hex,sig_hex", _RFC8032_VECTORS)
    def test_section_7_1_vectors_verify(self, pk_hex: str, msg_hex: str, sig_hex: str) -> None:
        assert (
            sovereign.ed25519_verify(
                bytes.fromhex(msg_hex), bytes.fromhex(sig_hex), bytes.fromhex(pk_hex)
            )
            is True
        )

    @pytest.mark.parametrize("pk_hex,msg_hex,sig_hex", _RFC8032_VECTORS)
    def test_tampered_message_and_signature_reject(
        self, pk_hex: str, msg_hex: str, sig_hex: str
    ) -> None:
        pk, msg, sig = bytes.fromhex(pk_hex), bytes.fromhex(msg_hex), bytes.fromhex(sig_hex)
        assert sovereign.ed25519_verify(msg + b"x", sig, pk) is False
        assert sovereign.ed25519_verify(msg, sig[:-1] + bytes([sig[-1] ^ 1]), pk) is False

    def test_malleable_s_above_group_order_rejects(self) -> None:
        pk_hex, msg_hex, sig_hex = _RFC8032_VECTORS[2]
        sig = bytes.fromhex(sig_hex)
        s_int = int.from_bytes(sig[32:], "little") + _ED25519_L
        malleable = sig[:32] + s_int.to_bytes(32, "little")
        assert (
            sovereign.ed25519_verify(bytes.fromhex(msg_hex), malleable, bytes.fromhex(pk_hex))
            is False
        )

    def test_non_canonical_and_malformed_inputs_reject_not_raise(self) -> None:
        pk_hex, msg_hex, sig_hex = _RFC8032_VECTORS[2]
        msg, sig = bytes.fromhex(msg_hex), bytes.fromhex(sig_hex)
        # Non-canonical y (>= p) in the public key.
        non_canonical = (2**255 - 19).to_bytes(32, "little")
        assert sovereign.ed25519_verify(msg, sig, non_canonical) is False
        # x = 0 with the sign bit set (the encoding AMA v5 closed upstream).
        x0_signed = bytes(31) + b"\x81"
        assert sovereign.ed25519_verify(msg, sig, x0_signed) is False
        # Wrong lengths return False, never raise.
        assert sovereign.ed25519_verify(msg, sig[:63], bytes.fromhex(pk_hex)) is False
        assert sovereign.ed25519_verify(msg, sig, bytes.fromhex(pk_hex)[:31]) is False

    def test_sovereign_tier_deliberately_has_no_signer(self) -> None:
        """Fail-safe reads, fail-closed writes: signing must not exist here."""
        assert not hasattr(sovereign, "ed25519_sign")


# =============================================================================
# 2. Cross-tier interop
# =============================================================================


class TestCrossTierInterop:
    def test_sovereign_seals_ama_opens_and_vice_versa(self) -> None:
        from ama_cryptography.pqc_backends import (
            native_chacha20poly1305_decrypt,
            native_chacha20poly1305_encrypt,
        )

        key, nonce, aad = b"k" * 32, b"n" * 12, b"hdr"
        blob = sovereign.chacha20poly1305_encrypt(key, nonce, b"cross-tier", aad)
        assert bytes(native_chacha20poly1305_decrypt(key, nonce, blob[:-16], blob[-16:], aad)) == (
            b"cross-tier"
        )
        ct, tag = native_chacha20poly1305_encrypt(key, nonce, b"cross-tier", aad)
        assert (
            sovereign.chacha20poly1305_decrypt(key, nonce, bytes(ct) + bytes(tag), aad)
            == b"cross-tier"
        )

    def test_legacy_pyca_ciphertext_still_opens(self) -> None:
        """Data sealed by the retired PyCA tier must keep decrypting."""
        aead = pytest.importorskip(
            "cryptography.hazmat.primitives.ciphers.aead",
            reason="pyca unavailable: legacy-continuity oracle absent",
        )
        from omni_mercury_engine import crypto

        key, nonce = b"k" * 32, b"n" * 12
        legacy = aead.ChaCha20Poly1305(key).encrypt(nonce, b"stored record", b"hdr")
        assert crypto.decrypt(legacy, key, nonce, b"hdr", algorithm="chacha20-poly1305") == (
            b"stored record"
        )
        legacy_aes = aead.AESGCM(key).encrypt(nonce, b"stored record", b"hdr")
        assert crypto.decrypt(legacy_aes, key, nonce, b"hdr") == b"stored record"

    def test_hkdf_identical_across_tiers_and_legacy(self) -> None:
        from ama_cryptography.pqc_backends import native_hkdf_sha256

        from omni_mercury_engine import crypto

        ms, salt, info = b"master", b"salt", b"ctx"
        enc_key, auth_key = crypto.derive_key_pair(ms, salt, info)
        assert enc_key == sovereign.hkdf_sha256(ms, 32, salt, info + b"-enc")
        assert auth_key == sovereign.hkdf_sha256(ms, 32, salt, info + b"-auth")
        assert enc_key == bytes(native_hkdf_sha256(ms, 32, salt, info + b"-enc"))

    def test_ama_signature_verifies_under_sovereign(self) -> None:
        from omni_mercury_engine.security.pqc_backends import (
            ed25519_sign,
            generate_ed25519_keypair,
        )

        kp = generate_ed25519_keypair()
        sig = ed25519_sign(b"sovereign cross-check", kp.secret_key)
        assert sovereign.ed25519_verify(b"sovereign cross-check", sig, kp.public_key) is True


# =============================================================================
# 3. Degrade governance in the tiered layer
# =============================================================================


@pytest.fixture()
def fresh_crypto(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """A reloaded crypto layer whose degrade latch starts clean.

    Reload isolates the module-level latch so these tests cannot leak a
    degraded state into (or inherit one from) the rest of the session.
    """
    import omni_mercury_engine.crypto as crypto_pkg

    module = importlib.reload(crypto_pkg)
    yield module
    importlib.reload(crypto_pkg)


class TestDegradeGovernance:
    def test_infrastructure_failure_latches_sovereign_tier(
        self, fresh_crypto: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ama_cryptography.exceptions import NativeBackendUnavailableError

        if fresh_crypto.get_crypto_backend() != "ama-native":
            pytest.skip("AMA tier not active (rust build present)")

        def _boom(*_a: object, **_k: object) -> bytes:
            raise NativeBackendUnavailableError("simulated FIPS lockout")

        monkeypatch.setattr(fresh_crypto, "_ama_chacha_encrypt", _boom)
        key, nonce = b"k" * 32, b"n" * 12
        ct, out_nonce = fresh_crypto.encrypt(
            b"emergency", key, nonce, algorithm="chacha20-poly1305"
        )
        assert out_nonce == nonce
        # Sealed by the sovereign tier after the latch; verify independently.
        assert sovereign.chacha20poly1305_decrypt(key, nonce, ct) == b"emergency"
        assert fresh_crypto.get_crypto_backend() == "sovereign-python"

    def test_auth_failure_is_a_verdict_not_a_degrade(self, fresh_crypto) -> None:  # type: ignore[no-untyped-def]
        if fresh_crypto.get_crypto_backend() != "ama-native":
            pytest.skip("AMA tier not active (rust build present)")
        key, nonce = b"k" * 32, b"n" * 12
        ct, _ = fresh_crypto.encrypt(b"payload", key, nonce)
        tampered = ct[:-1] + bytes([ct[-1] ^ 1])
        with pytest.raises(ValueError, match="Decryption failed"):
            fresh_crypto.decrypt(tampered, key, nonce)
        assert (
            fresh_crypto.get_crypto_backend() == "ama-native"
        ), "a tag mismatch must never trip the sovereign degrade latch"

    def test_aes_gcm_refuses_rather_than_switching_algorithms(
        self, fresh_crypto: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if fresh_crypto.is_rust_available():
            pytest.skip("rust tier serves AES-GCM regardless of AMA state")
        monkeypatch.setattr(fresh_crypto, "_AMA_RUNTIME_FAILED", True)
        with pytest.raises(RuntimeError, match="AES-GCM requires"):
            fresh_crypto.encrypt(b"x", b"k" * 32, b"n" * 12, algorithm="aes-gcm")
        with pytest.raises(RuntimeError, match="AES-GCM requires"):
            fresh_crypto.decrypt(b"c" * 32, b"k" * 32, b"n" * 12, algorithm="aes-gcm")

    def test_degraded_chacha_round_trip_stays_available(
        self, fresh_crypto: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(fresh_crypto, "_AMA_RUNTIME_FAILED", True)
        monkeypatch.setattr(fresh_crypto, "_RUST_AVAILABLE", False)
        key = b"k" * 32
        ct, nonce = fresh_crypto.encrypt(b"still alive", key, algorithm="chacha20-poly1305")
        assert fresh_crypto.decrypt(ct, key, nonce, algorithm="chacha20-poly1305") == b"still alive"
        ek, ak = fresh_crypto.derive_key_pair(b"m", b"s", b"i")
        assert len(ek) == len(ak) == 32
        assert len(fresh_crypto.secure_random(24)) == 24

    def test_ed25519_verify_degrades_but_sign_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ama_cryptography.exceptions import NativeBackendUnavailableError

        from omni_mercury_engine.security import pqc_backends as pb

        kp = pb.generate_ed25519_keypair()
        sig = pb.ed25519_sign(b"degrade", kp.secret_key)

        def _boom(*_a: object, **_k: object) -> bool:
            raise NativeBackendUnavailableError("simulated lockout")

        monkeypatch.setattr(pb, "_ama_native_ed25519_verify", _boom)
        assert pb.ed25519_verify(b"degrade", sig, kp.public_key) is True
        assert pb.ed25519_verify(b"forged", sig, kp.public_key) is False

        monkeypatch.setattr(pb, "_ama_native_ed25519_sign", _boom)
        monkeypatch.setattr(pb, "_ama_native_ed25519_keypair_from_seed", _boom)
        with pytest.raises((RuntimeError, NativeBackendUnavailableError)):
            pb.ed25519_sign(b"must fail closed", kp.secret_key)
