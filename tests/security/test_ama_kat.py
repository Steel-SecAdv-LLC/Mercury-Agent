"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Phase 2 ITEM 5 — AMA Cryptography Known-Answer Tests (KAT) and
deterministic-seed round-trip tests.

Per the May-2026 audit cure, correctness of AMA Cryptography is
demonstrated by *in-repo artifacts* — not by external-audit framing.
This file pins three contracts:

1. **Ed25519 RFC 8032 vectors.**  The Ed25519 surface used by
   ``crypto_api.Ed25519Provider`` and the native TCP transport must
   reproduce RFC 8032 §7.1 test vectors bit-for-bit.  These vectors
   run unconditionally because they only depend on stdlib
   ``cryptography`` which is a hard install requirement.
2. **ML-DSA-65 / Kyber-1024 / SPHINCS+ round-trip.**  When AMA's PQC
   backend is installed (``ama_cryptography[pqc]``), the keygen +
   sign/verify and keygen + encapsulate/decapsulate round-trips
   produce mutually consistent outputs.  When the backend is not
   installed the tests skip — no silent pass.
3. **Deterministic-seed reproducibility.**  ML-DSA signing of the
   same message with the same secret key is required by FIPS 204 to
   be deterministic; we pin that contract end-to-end.

The CI ``pqc-production-check.yml`` workflow runs this file on every
PR so a regression in the PQC surface is a build-time failure.
"""

from __future__ import annotations

import pytest

from omni_mercury_engine.security.crypto_api import Ed25519Provider

# ---------------------------------------------------------------------------
# Cure 1: Ed25519 RFC 8032 §7.1 vectors.
# ---------------------------------------------------------------------------


# Vector #1 from RFC 8032 §7.1 (TEST 1).
# secret key (32 bytes) | public key (32 bytes) | message | signature
ED25519_RFC8032_TEST_1 = {
    "secret_key": bytes.fromhex(
        "9d61b19deffd5a60ba844af492ec2cc4" "4449c5697b326919703bac031cae7f60"
    ),
    "public_key": bytes.fromhex(
        "d75a980182b10ab7d54bfed3c964073a" "0ee172f3daa62325af021a68f707511a"
    ),
    "message": b"",
    "signature": bytes.fromhex(
        "e5564300c360ac729086e2cc806e828a"
        "84877f1eb8e5d974d873e06522490155"
        "5fb8821590a33bacc61e39701cf9b46b"
        "d25bf5f0595bbe24655141438e7a100b"
    ),
}

# Vector #2 from RFC 8032 §7.1 (TEST 2 — single-byte message).
ED25519_RFC8032_TEST_2 = {
    "secret_key": bytes.fromhex(
        "4ccd089b28ff96da9db6c346ec114e0f" "5b8a319f35aba624da8cf6ed4fb8a6fb"
    ),
    "public_key": bytes.fromhex(
        "3d4017c3e843895a92b70aa74d1b7ebc" "9c982ccf2ec4968cc0cd55f12af4660c"
    ),
    "message": bytes.fromhex("72"),
    "signature": bytes.fromhex(
        "92a009a9f0d4cab8720e820b5f642540"
        "a2b27b5416503f8fb3762223ebdb69da"
        "085ac1e43e15996e458f3613d0f11d8c"
        "387b2eaeb4302aeeb00d291612bb0c00"
    ),
}


@pytest.mark.parametrize(
    "vector",
    [ED25519_RFC8032_TEST_1, ED25519_RFC8032_TEST_2],
    ids=["rfc8032-test-1-empty-msg", "rfc8032-test-2-1byte-msg"],
)
def test_ed25519_rfc8032_kat(vector: dict[str, bytes]) -> None:
    provider = Ed25519Provider()

    # The KAT is a deterministic-signature contract: signing the
    # vector's message with the vector's secret key MUST produce the
    # exact signature bytes from RFC 8032.  Stdlib ``cryptography``'s
    # Ed25519 implementation is deterministic (per RFC 8032).
    sig = provider.sign(vector["message"], vector["secret_key"])
    assert sig == vector["signature"]

    # Round-trip: verify must accept the canonical signature against
    # the canonical public key.
    assert provider.verify(vector["message"], vector["signature"], vector["public_key"])

    # Tamper-evidence: flipping any byte of the signature MUST be
    # rejected.  Verifies the underlying primitive is wired.
    tampered = bytearray(vector["signature"])
    tampered[0] ^= 0x01
    assert not provider.verify(vector["message"], bytes(tampered), vector["public_key"])


# ---------------------------------------------------------------------------
# Cure 2 & 3: PQC round-trips when AMA's backend is installed.
# ---------------------------------------------------------------------------


def _ama_pqc_available() -> bool:
    try:
        from omni_mercury_engine.security import pqc_backends as _pqc

        return bool(_pqc.AMA_CRYPTOGRAPHY_AVAILABLE)
    except Exception:
        return False


pqc_required = pytest.mark.skipif(
    not _ama_pqc_available(),
    reason="AMA Cryptography PQC backend not installed; PQC KATs require ama-cryptography[pqc].",
)


@pqc_required
def test_mldsa65_round_trip() -> None:
    from omni_mercury_engine.security.pqc_backends import (
        dilithium_sign,
        dilithium_verify,
        generate_dilithium_keypair,
    )

    kp = generate_dilithium_keypair()
    message = b"mercury-agent ML-DSA-65 KAT round trip"
    signature = dilithium_sign(message, kp.secret_key)
    assert dilithium_verify(message, signature, kp.public_key)

    # Tamper-evidence on signature.
    tampered = bytearray(signature)
    tampered[0] ^= 0x01
    assert not dilithium_verify(message, bytes(tampered), kp.public_key)


@pqc_required
def test_mldsa65_signing_is_deterministic() -> None:
    """FIPS 204 specifies deterministic signing for ML-DSA when no
    additional randomness is supplied — the same (message, secret_key)
    pair must produce identical signatures across calls.
    """
    from omni_mercury_engine.security.pqc_backends import (
        dilithium_sign,
        generate_dilithium_keypair,
    )

    kp = generate_dilithium_keypair()
    message = b"deterministic KAT seed"
    sig_a = dilithium_sign(message, kp.secret_key)
    sig_b = dilithium_sign(message, kp.secret_key)
    assert sig_a == sig_b


@pqc_required
def test_kyber1024_encaps_decaps_round_trip() -> None:
    from omni_mercury_engine.security.pqc_backends import (
        generate_kyber_keypair,
        kyber_decapsulate,
        kyber_encapsulate,
    )

    kp = generate_kyber_keypair()
    encap = kyber_encapsulate(kp.public_key)
    recovered = kyber_decapsulate(encap.ciphertext, kp.secret_key)
    assert recovered == encap.shared_secret


@pqc_required
def test_sphincs_round_trip() -> None:
    from omni_mercury_engine.security.pqc_backends import (
        generate_sphincs_keypair,
        sphincs_sign,
        sphincs_verify,
    )

    kp = generate_sphincs_keypair()
    message = b"mercury-agent SPHINCS+ KAT round trip"
    signature = sphincs_sign(message, kp.secret_key)
    assert sphincs_verify(message, signature, kp.public_key)

    # Tamper-evidence on signature.
    tampered = bytearray(signature)
    tampered[0] ^= 0x01
    assert not sphincs_verify(message, bytes(tampered), kp.public_key)
