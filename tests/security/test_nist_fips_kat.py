# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Phase 2 Deliverable 7 — NIST FIPS KAT vectors for ML-DSA-65, ML-KEM (Kyber-1024), and SLH-DSA (SPHINCS+).

Test vectors are curated from the NIST ACVP-Server canonical test data:
https://github.com/usnistgov/ACVP-Server/tree/master/gen-val/json-files

The curated subset lives in ``tests/security/data/nist_kat/nist_acvp_curated.json``
with 3 vectors per algorithm-operation pair.

These tests go beyond round-trip-only: they perform **bit-for-bit
reproducibility checks** against NIST reference outputs.

- ML-DSA-65 sigGen: verify that the NIST-provided signature is accepted by
  AMA's ``dilithium_verify``, confirming interop with the reference impl.
- ML-KEM-1024 decapsulation: verify that AMA's ``kyber_decapsulate`` with
  the NIST-provided (dk, c) produces the expected shared secret ``k``.
- SLH-DSA sigGen: verify that NIST-provided signatures are accepted by
  AMA's ``sphincs_verify``.
- keyGen, all three: the key pair AMA derives from NIST's seed is NIST's key
  pair, byte for byte. These nine vectors sat in the curated file unused.

What these do NOT cover, so that nothing is read into them: the three ML-KEM
encapsulation vectors need the encapsulation randomness ``m`` as an input, and
AMA exposes no derandomised encapsulation, so they cannot be reproduced; and
there is no vector here for SLH-DSA-SHA2-256f, the parameter set the legacy
``sphincs_sign`` uses.

AMA's PQC backend is mandatory; missing AMA/PQC fails module import rather than
skipping.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from ama_cryptography.pqc_backends import (
    native_dilithium_keypair_from_seed,
    native_kyber_keypair_from_seed,
)

from omni_mercury_engine.security.pqc_backends import (
    dilithium_sign,
    dilithium_sign_ctx,
    dilithium_verify,
    generate_slhdsa_keypair_from_seed,
    kyber_decapsulate,
    slhdsa_sign_deterministic,
    slhdsa_verify,
)

# ---------------------------------------------------------------------------
# Fixture: load curated NIST ACVP vectors
# ---------------------------------------------------------------------------

_KAT_DIR = Path(__file__).resolve().parent / "data" / "nist_kat"
_CURATED_FILE = _KAT_DIR / "nist_acvp_curated.json"


def _load_curated() -> dict[str, Any]:
    with open(_CURATED_FILE) as f:
        data: dict[str, Any] = json.load(f)
        return data


# ---------------------------------------------------------------------------
# ML-DSA-65: sigGen verification
# ---------------------------------------------------------------------------


def _mldsa65_siggen_vectors() -> list[dict[str, Any]]:
    data = _load_curated()
    vectors: list[dict[str, Any]] = data.get("ML-DSA-65", {}).get("sigGen", [])
    return vectors


@pytest.mark.parametrize(
    "vector",
    _mldsa65_siggen_vectors(),
    ids=[f"mldsa65-sigGen-tc{v['tcId']}" for v in _mldsa65_siggen_vectors()],
)
def test_mldsa65_nist_siggen_verify(vector: dict[str, Any]) -> None:
    """NIST ML-DSA-65 sigGen vectors are byte-exact under FIPS 204 §5.2 ctx sign.

    The curated NIST ACVP vectors (tcIds 31–33) all carry a non-empty
    ``context`` field, so the byte-exact reproducibility contract requires
    the ctx-aware signer (``M' = 0x00 || IntegerToBytes(|ctx|, 1) || ctx || M``).
    Signing with the legacy context-blind ``dilithium_sign`` would produce
    a sibling-but-different signature and silently miss the spec.
    """
    sk = bytes.fromhex(vector["sk"])
    message = bytes.fromhex(vector["message"])
    ctx = bytes.fromhex(vector.get("context", ""))
    expected_sig = bytes.fromhex(vector["signature"])

    produced_sig = dilithium_sign_ctx(message, sk, ctx)
    assert produced_sig == expected_sig, (
        f"ML-DSA-65 sigGen tc{vector['tcId']}: AMA signature differs from NIST reference. "
        f"Produced {len(produced_sig)} bytes, expected {len(expected_sig)} bytes."
    )


# ---------------------------------------------------------------------------
# ML-KEM-1024: decapsulation
# ---------------------------------------------------------------------------


def _mlkem1024_decaps_vectors() -> list[dict[str, Any]]:
    data = _load_curated()
    vectors: list[dict[str, Any]] = data.get("ML-KEM-1024", {}).get("decapsulation", [])
    return vectors


@pytest.mark.parametrize(
    "vector",
    _mlkem1024_decaps_vectors(),
    ids=[f"mlkem1024-decaps-tc{v['tcId']}" for v in _mlkem1024_decaps_vectors()],
)
def test_mlkem1024_nist_decapsulation(vector: dict[str, Any]) -> None:
    """Verify NIST ML-KEM-1024 decapsulation produces expected shared secret."""
    dk = bytes.fromhex(vector["dk"])
    c = bytes.fromhex(vector["c"])
    expected_k = bytes.fromhex(vector["k"])

    recovered_k = kyber_decapsulate(c, dk)
    assert (
        recovered_k == expected_k
    ), f"ML-KEM-1024 decaps tc{vector['tcId']}: AMA shared secret differs from NIST reference."


# ---------------------------------------------------------------------------
# SLH-DSA (SPHINCS+): sigGen verification
# ---------------------------------------------------------------------------


def _slhdsa_siggen_vectors() -> list[dict[str, Any]]:
    data = _load_curated()
    vectors: list[dict[str, Any]] = data.get("SLH-DSA-SHAKE-128s", {}).get("sigGen", [])
    return vectors


@pytest.mark.parametrize(
    "vector",
    _slhdsa_siggen_vectors(),
    ids=[f"slhdsa-sigGen-tc{v['tcId']}" for v in _slhdsa_siggen_vectors()],
)
def test_slhdsa_nist_siggen_verify(vector: dict[str, Any]) -> None:
    """NIST SLH-DSA-SHAKE-128s sigGen vectors are byte-exact under FIPS 205 §10.2.

    The curated NIST ACVP-Server vectors (tcIds 214–216) are the
    deterministic external/pure subset — ``additionalRandomness`` is
    absent, so AMA's ``slhdsa_sign_deterministic`` (which sets
    ``addrnd = PK.seed`` per FIPS 205 §10.2) is the only path that
    reproduces them byte-for-byte. Calling ``slhdsa_sign`` (the hedged
    default) instead would mix in fresh randomness and produce a
    sibling-but-different signature — a silent KAT miss.

    The legacy ``sphincs_sign`` symbol still targets SLH-DSA-SHA2-256f
    (NIST L5) and is incompatible with these L1 vectors regardless of
    determinism, hence the explicit ``param_set='SHAKE-128s'`` here.
    """
    sk = bytes.fromhex(vector["sk"])
    message = bytes.fromhex(vector["message"])
    ctx = bytes.fromhex(vector.get("context", ""))
    expected_sig = bytes.fromhex(vector["signature"])

    produced_sig = slhdsa_sign_deterministic(message, sk, ctx, param_set="SHAKE-128s")
    assert produced_sig == expected_sig, (
        f"SLH-DSA-SHAKE-128s sigGen tc{vector['tcId']}: "
        f"AMA signature differs from NIST reference. "
        f"Produced {len(produced_sig)} bytes, expected {len(expected_sig)} bytes."
    )


# ---------------------------------------------------------------------------
# keyGen: the key pair derived from NIST's seed is NIST's key pair
# ---------------------------------------------------------------------------


def _keygen_vectors(algorithm: str) -> list[dict[str, Any]]:
    vectors: list[dict[str, Any]] = _load_curated().get(algorithm, {}).get("keyGen", [])
    return vectors


def _both(pair: Any) -> tuple[bytes, bytes]:
    """A key pair as (public, secret), whichever way the binding returns it."""
    if hasattr(pair, "public_key"):
        return bytes(pair.public_key), bytes(pair.secret_key)
    return bytes(pair[0]), bytes(pair[1])


@pytest.mark.parametrize(
    "vector",
    _keygen_vectors("ML-DSA-65"),
    ids=[f"mldsa65-keyGen-tc{v['tcId']}" for v in _keygen_vectors("ML-DSA-65")],
)
def test_mldsa65_nist_keygen(vector: dict[str, Any]) -> None:
    """FIPS 204 §5.1: the seed ``xi`` gives NIST's public and secret key."""
    public, secret = _both(native_dilithium_keypair_from_seed(bytes.fromhex(vector["seed"])))

    assert public == bytes.fromhex(vector["pk"])
    assert secret == bytes.fromhex(vector["sk"])


@pytest.mark.parametrize(
    "vector",
    _keygen_vectors("ML-KEM-1024"),
    ids=[f"mlkem1024-keyGen-tc{v['tcId']}" for v in _keygen_vectors("ML-KEM-1024")],
)
def test_mlkem1024_nist_keygen(vector: dict[str, Any]) -> None:
    """FIPS 203 §7.1: the seeds ``(d, z)`` give NIST's encapsulation and decapsulation key."""
    public, secret = _both(
        native_kyber_keypair_from_seed(bytes.fromhex(vector["d"]), bytes.fromhex(vector["z"]))
    )

    assert public == bytes.fromhex(vector["ek"])
    assert secret == bytes.fromhex(vector["dk"])


@pytest.mark.parametrize(
    "vector",
    _keygen_vectors("SLH-DSA-SHAKE-128s"),
    ids=[f"slhdsa-keyGen-tc{v['tcId']}" for v in _keygen_vectors("SLH-DSA-SHAKE-128s")],
)
def test_slhdsa_nist_keygen(vector: dict[str, Any]) -> None:
    """FIPS 205 §10.1: the three seeds give NIST's public and secret key."""
    public, secret = _both(
        generate_slhdsa_keypair_from_seed(
            bytes.fromhex(vector["skSeed"]),
            bytes.fromhex(vector["skPrf"]),
            bytes.fromhex(vector["pkSeed"]),
            param_set="SHAKE-128s",
        )
    )

    assert public == bytes.fromhex(vector["pk"])
    assert secret == bytes.fromhex(vector["sk"])


# ---------------------------------------------------------------------------
# The other direction: what is wrong is refused
# ---------------------------------------------------------------------------


def _flip(data: bytes, index: int = 0) -> bytes:
    changed = bytearray(data)
    changed[index] ^= 0x01
    return bytes(changed)


def test_mldsa65_refuses_a_changed_signature_or_message_under_a_nist_key() -> None:
    """A vector that only ever verifies says nothing about what is rejected."""
    vector = _keygen_vectors("ML-DSA-65")[0]
    public, secret = bytes.fromhex(vector["pk"]), bytes.fromhex(vector["sk"])
    message = b"a reading taken at the gauge"
    signature = dilithium_sign(message, secret)

    assert dilithium_verify(message, signature, public) is True
    assert dilithium_verify(message + b".", signature, public) is False
    assert dilithium_verify(message, _flip(signature, 10), public) is False
    other = bytes.fromhex(_keygen_vectors("ML-DSA-65")[1]["pk"])
    assert dilithium_verify(message, signature, other) is False


def test_slhdsa_refuses_a_changed_signature_or_message_under_a_nist_key() -> None:
    vector = _slhdsa_siggen_vectors()[0]
    secret = bytes.fromhex(vector["sk"])
    public = secret[len(secret) // 2 :]  # FIPS 205 §9.1: SK ends with PK.seed || PK.root
    message = bytes.fromhex(vector["message"])
    ctx = bytes.fromhex(vector.get("context", ""))
    signature = bytes.fromhex(vector["signature"])

    assert slhdsa_verify(message, signature, public, ctx, param_set="SHAKE-128s") is True
    assert slhdsa_verify(message + b".", signature, public, ctx, param_set="SHAKE-128s") is False
    assert (
        slhdsa_verify(message, _flip(signature, 40), public, ctx, param_set="SHAKE-128s") is False
    )
    assert slhdsa_verify(message, signature, public, ctx + b"x", param_set="SHAKE-128s") is False


def test_mlkem1024_does_not_return_the_secret_for_a_changed_ciphertext() -> None:
    """FIPS 203 implicit rejection: a tampered ciphertext yields a different, unrelated key."""
    vector = _mlkem1024_decaps_vectors()[0]
    secret_key = bytes.fromhex(vector["dk"])
    ciphertext = bytes.fromhex(vector["c"])
    expected = bytes.fromhex(vector["k"])

    assert kyber_decapsulate(ciphertext, secret_key) == expected
    rejected = kyber_decapsulate(_flip(ciphertext, 7), secret_key)
    assert rejected != expected
    assert len(rejected) == len(expected)


def test_what_these_vectors_cover_is_counted_not_assumed() -> None:
    """Eighteen vectors are reproduced; the three encapsulation ones are not, and why."""
    data = _load_curated()
    used = {
        ("ML-DSA-65", "keyGen"),
        ("ML-DSA-65", "sigGen"),
        ("ML-KEM-1024", "keyGen"),
        ("ML-KEM-1024", "decapsulation"),
        ("SLH-DSA-SHAKE-128s", "keyGen"),
        ("SLH-DSA-SHAKE-128s", "sigGen"),
    }
    present = {
        (algorithm, operation)
        for algorithm, operations in data.items()
        if isinstance(operations, dict)
        for operation in operations
    }

    assert present - used == {("ML-KEM-1024", "encapsulation")}
    assert sum(len(data[algorithm][operation]) for algorithm, operation in used) == 18
    # The encapsulation vectors fix the randomness ``m``; AMA takes none.
    import inspect

    from ama_cryptography.pqc_backends import native_ml_kem_encapsulate

    assert list(inspect.signature(native_ml_kem_encapsulate).parameters) == ["ps", "public_key"]


# ---------------------------------------------------------------------------
# Sanity: curated file exists and is well-formed
# ---------------------------------------------------------------------------


def test_nist_kat_file_exists() -> None:
    """The curated NIST ACVP KAT file must exist."""
    assert _CURATED_FILE.exists(), f"Missing {_CURATED_FILE}"


def test_nist_kat_file_has_all_algorithms() -> None:
    """The curated file must cover ML-DSA-65, ML-KEM-1024, SLH-DSA."""
    data = _load_curated()
    assert "ML-DSA-65" in data
    assert "ML-KEM-1024" in data
    assert "SLH-DSA-SHAKE-128s" in data
    # Each must have vectors
    assert len(data["ML-DSA-65"]["sigGen"]) >= 1
    assert len(data["ML-KEM-1024"]["decapsulation"]) >= 1
    assert len(data["SLH-DSA-SHAKE-128s"]["sigGen"]) >= 1
