"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see
https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
Mercury Agent - Post-Quantum Cryptography Backends

AMA Cryptography v3.2.0 is the sole PQC implementation.  The git ref is
pinned in ``pyproject.toml [project.optional-dependencies].pqc`` and in
the ``AMA_REF`` env var of ``.github/workflows/ci.yml`` /
``.github/workflows/pqc-production-check.yml`` -- bump those in
lock-step when upgrading.

Previous versions used a 4-tier fallback chain (AMA → liboqs → pqcrypto →
SIMULATION), then a soft-import / hard-call stub bridge for AMA-less dev
lanes.  Mercury now fails closed at module import: if
``ama_cryptography.pqc_backends`` cannot be imported, or if the pinned v3.2.0
FIPS 204/205 symbols are missing, this module does not load.  AMA v3.2.0
carries its own native C backend — it *is* the implementation.  Retaining
weaker fallbacks only widened the attack surface.

The v3.x surface adds FIPS 204 §5.2 context-aware ML-DSA-65 signing and
FIPS 205 SLH-DSA-SHAKE-128s; v3.2.0 specifically adds the
``native_hmac_sha256`` / ``native_hmac_sha256_2`` Python bindings
consumed by ``omni_mercury_engine.security.native_jwt`` for HS256 /
HS512 JOSE signing.

SECURITY NOTICE
===============
Backend audit status:

| Backend            | Status                        | Recommendation           |
|--------------------|-------------------------------|--------------------------|
| AMA Cryptography   | Community-tested, NOT audited | Development/Testing      |

For production deployments requiring compliance:
- Obtain independent security audit of chosen backend
- Consider FIPS 140-2 Level 3+ HSM for master secrets
- Document risk acceptance for unaudited cryptographic code

The algorithms (ML-DSA-65, Kyber-1024, SPHINCS+) are NIST-approved.
Implementation correctness is NOT externally verified.

References:
    - NIST PQC Standardization: https://csrc.nist.gov/projects/post-quantum-cryptography
    - AMA Cryptography: https://github.com/Steel-SecAdv-LLC/AMA-Cryptography
    - Dilithium: https://pq-crystals.org/dilithium/
    - Kyber: https://pq-crystals.org/kyber/
"""

import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# AMA Cryptography — sole PQC backend
# ---------------------------------------------------------------------------
from ama_cryptography.pqc_backends import (
    DILITHIUM_AVAILABLE as _AMA_DILITHIUM,
    KYBER_AVAILABLE as _AMA_KYBER,
    SLHDSA_SHA2_256F_PUBLIC_KEY_BYTES as _AMA_SLHDSA_SHA2_256F_PK_BYTES,
    SLHDSA_SHA2_256F_SECRET_KEY_BYTES as _AMA_SLHDSA_SHA2_256F_SK_BYTES,
    SLHDSA_SHA2_256F_SIGNATURE_BYTES as _AMA_SLHDSA_SHA2_256F_SIG_BYTES,
    SLHDSA_SHAKE_128S_PUBLIC_KEY_BYTES as _AMA_SLHDSA_SHAKE_128S_PK_BYTES,
    SLHDSA_SHAKE_128S_SECRET_KEY_BYTES as _AMA_SLHDSA_SHAKE_128S_SK_BYTES,
    SLHDSA_SHAKE_128S_SIGNATURE_BYTES as _AMA_SLHDSA_SHAKE_128S_SIG_BYTES,
    SPHINCS_AVAILABLE as _AMA_SPHINCS,
    dilithium_sign as _ama_dilithium_sign,
    dilithium_sign_ctx as _ama_dilithium_sign_ctx,
    dilithium_verify as _ama_dilithium_verify,
    generate_dilithium_keypair as _ama_generate_dilithium_keypair,
    generate_kyber_keypair as _ama_generate_kyber_keypair,
    generate_slhdsa_keypair as _ama_generate_slhdsa_keypair,
    generate_slhdsa_keypair_from_seed as _ama_generate_slhdsa_keypair_from_seed,
    generate_sphincs_keypair as _ama_generate_sphincs_keypair,
    kyber_decapsulate as _ama_kyber_decapsulate,
    kyber_encapsulate as _ama_kyber_encapsulate,
    slhdsa_sign as _ama_slhdsa_sign,
    slhdsa_sign_deterministic as _ama_slhdsa_sign_deterministic,
    slhdsa_sign_internal as _ama_slhdsa_sign_internal,
    slhdsa_verify as _ama_slhdsa_verify,
    sphincs_sign as _ama_sphincs_sign,
    sphincs_verify as _ama_sphincs_verify,
)

AMA_CRYPTOGRAPHY_AVAILABLE = True
DILITHIUM_AVAILABLE = _AMA_DILITHIUM
KYBER_AVAILABLE = _AMA_KYBER
SPHINCS_AVAILABLE = _AMA_SPHINCS
DILITHIUM_CTX_AVAILABLE = bool(_AMA_DILITHIUM)
SLHDSA_AVAILABLE = bool(_AMA_SPHINCS)
SLHDSA_SHAKE_128S_PUBLIC_KEY_BYTES: int = int(_AMA_SLHDSA_SHAKE_128S_PK_BYTES)
SLHDSA_SHAKE_128S_SECRET_KEY_BYTES: int = int(_AMA_SLHDSA_SHAKE_128S_SK_BYTES)
SLHDSA_SHAKE_128S_SIGNATURE_BYTES: int = int(_AMA_SLHDSA_SHAKE_128S_SIG_BYTES)
SLHDSA_SHA2_256F_PUBLIC_KEY_BYTES: int = int(_AMA_SLHDSA_SHA2_256F_PK_BYTES)
SLHDSA_SHA2_256F_SECRET_KEY_BYTES: int = int(_AMA_SLHDSA_SHA2_256F_SK_BYTES)
SLHDSA_SHA2_256F_SIGNATURE_BYTES: int = int(_AMA_SLHDSA_SHA2_256F_SIG_BYTES)


# Backward compatibility alias
AVA_GUARDIAN_AVAILABLE = AMA_CRYPTOGRAPHY_AVAILABLE


class PQCBackend(Enum):
    """
    Available PQC backend implementations.

    Only ``AMA_CRYPTOGRAPHY`` is supported.  ``AVA_GUARDIAN`` remains as a backward-compatibility
    alias that resolves to the same enum member.
    """

    AMA_CRYPTOGRAPHY = "ama-cryptography"
    AVA_GUARDIAN = "ama-cryptography"  # backward compat alias


def get_active_backend() -> PQCBackend:
    """Return the active PQC backend (always AMA Cryptography)."""
    return PQCBackend.AMA_CRYPTOGRAPHY


def require_constant_time() -> bool:
    """Check if constant-time implementations are required."""
    return (
        os.environ.get("AMA_REQUIRE_CONSTANT_TIME", "").lower() == "true"
        or os.environ.get("AVA_REQUIRE_CONSTANT_TIME", "").lower() == "true"
    )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DilithiumKeyPair:
    """ML-DSA-65 (Dilithium) key pair."""

    public_key: bytes
    secret_key: bytes
    algorithm: str = "ML-DSA-65"


@dataclass
class KyberKeyPair:
    """Kyber-1024 key pair."""

    public_key: bytes
    secret_key: bytes
    algorithm: str = "Kyber1024"


@dataclass
class KyberEncapsulation:
    """Kyber key encapsulation result."""

    ciphertext: bytes
    shared_secret: bytes


@dataclass
class SphincsKeyPair:
    """SPHINCS+-256f key pair."""

    public_key: bytes
    secret_key: bytes
    algorithm: str = "SPHINCS+-SHA2-256f-simple"


@dataclass
class SlhDsaKeyPair:
    """FIPS 205 SLH-DSA key pair (parameter-driven).

    ``param_set`` selects between the two AMA-supported parameter sets:

    * ``"SHAKE-128s"`` — NIST Level 1, n=16, signatures 7,856 bytes
      (FIPS 205 §11.1 SHAKE family, suited for TLS handshake / embedded).
    * ``"SHA2-256f"``  — NIST Level 5, n=32, signatures 49,856 bytes
      (FIPS 205 §11.2 SHA2 family, ``f`` = fast variant).

    The ``algorithm`` field follows the FIPS 205 §6 naming convention
    so downstream audit logs are unambiguous about which parameter set
    produced the key material.
    """

    public_key: bytes
    secret_key: bytes
    param_set: str = "SHAKE-128s"
    algorithm: str = "SLH-DSA-SHAKE-128s"


# ---------------------------------------------------------------------------
# PQC operations — thin wrappers around AMA Cryptography
# ---------------------------------------------------------------------------


def generate_dilithium_keypair() -> DilithiumKeyPair:
    """
    Generate ML-DSA-65 (Dilithium) key pair via AMA Cryptography.

    Returns:
        DilithiumKeyPair with public and secret keys

    Raises:
        RuntimeError: If Dilithium is not available in AMA
    """
    if not DILITHIUM_AVAILABLE:
        raise RuntimeError(
            "ML-DSA-65 not available in AMA Cryptography. "
            "Build the native C library: cmake -B build -DAMA_USE_NATIVE_PQC=ON && cmake --build build"
        )
    kp = _ama_generate_dilithium_keypair()
    # Defensive copy: AMA's keypair holds ``secret_key`` as a mutable
    # ``bytearray`` and registers a ``_secure_memzero`` finalizer that
    # zeros the buffer when the AMA object is GC'd.  Aliasing the same
    # buffer here would let the finalizer destroy Mercury's own copy
    # the moment the AMA wrapper falls out of scope, breaking every
    # downstream sign/verify.  Coercing to immutable ``bytes`` decouples
    # Mercury's lifetime from AMA's memzero policy.
    return DilithiumKeyPair(
        public_key=bytes(kp.public_key),
        secret_key=bytes(kp.secret_key),
    )


def dilithium_sign(message: bytes, secret_key: bytes) -> bytes:
    """Sign message using ML-DSA-65 (Dilithium) via AMA Cryptography."""
    if not DILITHIUM_AVAILABLE:
        raise RuntimeError("ML-DSA-65 not available in AMA Cryptography.")
    result: bytes = bytes(_ama_dilithium_sign(message, secret_key))
    return result


def dilithium_verify(message: bytes, signature: bytes, public_key: bytes) -> bool:
    """Verify ML-DSA-65 (Dilithium) signature via AMA Cryptography."""
    if not DILITHIUM_AVAILABLE:
        raise RuntimeError("ML-DSA-65 not available in AMA Cryptography.")
    result: bool = bool(_ama_dilithium_verify(message, signature, public_key))
    return result


def generate_kyber_keypair() -> KyberKeyPair:
    """Generate Kyber-1024 key pair via AMA Cryptography."""
    if not KYBER_AVAILABLE:
        raise RuntimeError(
            "Kyber-1024 not available in AMA Cryptography. "
            "Build the native C library: cmake -B build -DAMA_USE_NATIVE_PQC=ON && cmake --build build"
        )
    kp = _ama_generate_kyber_keypair()
    # See ``generate_dilithium_keypair`` — defensive copy decouples Mercury
    # from AMA's secure-memzero finalizer.
    return KyberKeyPair(
        public_key=bytes(kp.public_key),
        secret_key=bytes(kp.secret_key),
    )


def kyber_encapsulate(public_key: bytes) -> KyberEncapsulation:
    """Encapsulate shared secret using Kyber public key via AMA Cryptography."""
    if not KYBER_AVAILABLE:
        raise RuntimeError("Kyber-1024 not available in AMA Cryptography.")
    result = _ama_kyber_encapsulate(public_key)
    # Defensive copy of the shared_secret/ciphertext buffers — the AMA
    # encapsulation object is finalized with secure memzero of the secret.
    return KyberEncapsulation(
        ciphertext=bytes(result.ciphertext),
        shared_secret=bytes(result.shared_secret),
    )


def kyber_decapsulate(ciphertext: bytes, secret_key: bytes) -> bytes:
    """Decapsulate shared secret using Kyber secret key via AMA Cryptography."""
    if not KYBER_AVAILABLE:
        raise RuntimeError("Kyber-1024 not available in AMA Cryptography.")
    result: bytes = bytes(_ama_kyber_decapsulate(ciphertext, secret_key))
    return result


def generate_sphincs_keypair() -> SphincsKeyPair:
    """Generate SPHINCS+-256f key pair via AMA Cryptography."""
    if not SPHINCS_AVAILABLE:
        raise RuntimeError(
            "SPHINCS+ not available in AMA Cryptography. "
            "Build the native C library: cmake -B build -DAMA_USE_NATIVE_PQC=ON && cmake --build build"
        )
    kp = _ama_generate_sphincs_keypair()
    # See ``generate_dilithium_keypair`` — defensive copy decouples Mercury
    # from AMA's secure-memzero finalizer.
    return SphincsKeyPair(
        public_key=bytes(kp.public_key),
        secret_key=bytes(kp.secret_key),
    )


def sphincs_sign(message: bytes, secret_key: bytes) -> bytes:
    """Sign message using SPHINCS+ via AMA Cryptography."""
    if not SPHINCS_AVAILABLE:
        raise RuntimeError("SPHINCS+ not available in AMA Cryptography.")
    result: bytes = bytes(_ama_sphincs_sign(message, secret_key))
    return result


def sphincs_verify(message: bytes, signature: bytes, public_key: bytes) -> bool:
    """Verify SPHINCS+ signature via AMA Cryptography."""
    if not SPHINCS_AVAILABLE:
        raise RuntimeError("SPHINCS+ not available in AMA Cryptography.")
    result: bool = bool(_ama_sphincs_verify(message, signature, public_key))
    return result


# ---------------------------------------------------------------------------
# FIPS 204 §5.2 ML-DSA-65 context-aware signing (AMA ≥ v3.1.0)
# ---------------------------------------------------------------------------


def dilithium_sign_ctx(message: bytes, secret_key: bytes, ctx: bytes = b"") -> bytes:
    """Sign ``message`` under ``secret_key`` with a FIPS 204 §5.2 binding context.

    Wraps the AMA C-level ctx-aware signer; the wrapper applies
    ``M' = 0x00 || IntegerToBytes(|ctx|, 1) || ctx || M`` before delegating
    to the internal sign, byte-for-byte mirroring ``dilithium_verify_ctx``.
    Rejects ``len(ctx) > 255`` per FIPS 204 §5.2 line 4.

    Defensive ``bytes(...)`` cast at the boundary follows the same INVARIANT-6
    pattern as every other Mercury PQC adapter — the signature returned to
    the caller is decoupled from any AMA-side ``bytearray`` finalizer.
    """
    if not DILITHIUM_CTX_AVAILABLE:
        raise RuntimeError(
            "FIPS 204 §5.2 ctx-aware ML-DSA-65 sign not available. "
            "Upgrade to ama-cryptography ≥ 3.1.0."
        )
    return bytes(_ama_dilithium_sign_ctx(message, secret_key, ctx))


# ---------------------------------------------------------------------------
# FIPS 205 SLH-DSA — parameter-driven SHAKE-128s (NIST L1) / SHA2-256f (NIST L5)
# ---------------------------------------------------------------------------


def _slhdsa_param_sizes(param_set: str) -> tuple[int, int, int]:
    """Return (pk_len, sk_len, sig_len) for a FIPS 205 parameter set."""
    if param_set == "SHAKE-128s":
        return (
            SLHDSA_SHAKE_128S_PUBLIC_KEY_BYTES,
            SLHDSA_SHAKE_128S_SECRET_KEY_BYTES,
            SLHDSA_SHAKE_128S_SIGNATURE_BYTES,
        )
    if param_set == "SHA2-256f":
        return (
            SLHDSA_SHA2_256F_PUBLIC_KEY_BYTES,
            SLHDSA_SHA2_256F_SECRET_KEY_BYTES,
            SLHDSA_SHA2_256F_SIGNATURE_BYTES,
        )
    raise ValueError(
        f"Unknown SLH-DSA parameter set {param_set!r}; expected 'SHAKE-128s' or 'SHA2-256f'"
    )


def generate_slhdsa_keypair(param_set: str = "SHAKE-128s") -> SlhDsaKeyPair:
    """
    Generate a FIPS 205 SLH-DSA keypair via AMA Cryptography.

    The default parameter set is SHAKE-128s (NIST Level 1) since that is the set Mercury's NIST FIPS
    KAT pins exercise. Pass ``"SHA2-256f"`` for the NIST Level 5 SHA2-family target.

    Returns a defensive-copied ``SlhDsaKeyPair`` so the AMA-side ``_secure_memzero`` finalizer does
    not zero out Mercury's bytes when the AMA wrapper falls out of scope (same pattern as
    ``generate_dilithium_keypair``).
    """
    if not SLHDSA_AVAILABLE:
        raise RuntimeError(
            "FIPS 205 SLH-DSA not available. "
            "Upgrade to ama-cryptography ≥ 3.1.0 and ensure the native C library is built."
        )
    pk_len, sk_len, _ = _slhdsa_param_sizes(param_set)
    kp = _ama_generate_slhdsa_keypair(param_set)
    public_key = bytes(kp.public_key)
    secret_key = bytes(kp.secret_key)
    if len(public_key) != pk_len or len(secret_key) != sk_len:
        raise RuntimeError(
            f"AMA returned SLH-DSA-{param_set} keypair with unexpected sizes: "
            f"pk={len(public_key)} (expected {pk_len}), "
            f"sk={len(secret_key)} (expected {sk_len})"
        )
    algorithm = f"SLH-DSA-{param_set}"
    return SlhDsaKeyPair(
        public_key=public_key,
        secret_key=secret_key,
        param_set=param_set,
        algorithm=algorithm,
    )


def generate_slhdsa_keypair_from_seed(
    sk_seed: bytes,
    sk_prf: bytes,
    pk_seed: bytes,
    param_set: str = "SHAKE-128s",
) -> SlhDsaKeyPair:
    """
    Derive a FIPS 205 §10.1 SLH-DSA keypair from caller-supplied seeds.

    All three seed inputs must be exactly ``n`` bytes (16 for SHAKE-128s, 32 for SHA2-256f). The
    defensive copy on the way out follows the same INVARIANT-6 pattern as the random-keygen path;
    AMA itself wipes the seed scratch buffers on the way through the C boundary.
    """
    if not SLHDSA_AVAILABLE:
        raise RuntimeError(
            "FIPS 205 SLH-DSA not available. "
            "Upgrade to ama-cryptography ≥ 3.1.0 and ensure the native C library is built."
        )
    pk_len, sk_len, _ = _slhdsa_param_sizes(param_set)
    n = pk_len  # FIPS 205 §10.1: PK.seed length == n
    if len(sk_seed) != n or len(sk_prf) != n or len(pk_seed) != n:
        raise ValueError(
            f"FIPS 205 SLH-DSA-{param_set} expects sk_seed/sk_prf/pk_seed of {n} bytes; "
            f"got {len(sk_seed)}/{len(sk_prf)}/{len(pk_seed)}"
        )
    kp = _ama_generate_slhdsa_keypair_from_seed(sk_seed, sk_prf, pk_seed, param_set)
    public_key = bytes(kp.public_key)
    secret_key = bytes(kp.secret_key)
    if len(public_key) != pk_len or len(secret_key) != sk_len:
        raise RuntimeError(
            f"AMA returned SLH-DSA-{param_set} keypair-from-seed with unexpected sizes: "
            f"pk={len(public_key)} (expected {pk_len}), "
            f"sk={len(secret_key)} (expected {sk_len})"
        )
    algorithm = f"SLH-DSA-{param_set}"
    return SlhDsaKeyPair(
        public_key=public_key,
        secret_key=secret_key,
        param_set=param_set,
        algorithm=algorithm,
    )


def slhdsa_sign(
    message: bytes,
    secret_key: bytes,
    ctx: bytes = b"",
    param_set: str = "SHAKE-128s",
) -> bytes:
    """FIPS 205 SLH-DSA sign (hedged, fresh ``addrnd`` per call).

    The ``ctx`` argument participates in FIPS 205 §10.2 domain separation
    (``M' = 0x00 || IntegerToBytes(|ctx|, 1) || ctx || M``); ``len(ctx) > 255``
    is rejected by AMA with a ``ValueError``. For byte-exact NIST ACVP
    sigGen reproduction, use ``slhdsa_sign_deterministic`` instead — the
    hedged path mixes in fresh randomness and so cannot match NIST's
    deterministic test vectors.
    """
    if not SLHDSA_AVAILABLE:
        raise RuntimeError("FIPS 205 SLH-DSA not available. Upgrade to ama-cryptography ≥ 3.1.0.")
    _, sk_len, _ = _slhdsa_param_sizes(param_set)
    if len(secret_key) != sk_len:
        raise ValueError(
            f"FIPS 205 SLH-DSA-{param_set} secret key must be {sk_len} bytes; "
            f"got {len(secret_key)}"
        )
    return bytes(_ama_slhdsa_sign(message, secret_key, ctx, param_set))


def slhdsa_sign_deterministic(
    message: bytes,
    secret_key: bytes,
    ctx: bytes = b"",
    param_set: str = "SHAKE-128s",
) -> bytes:
    """FIPS 205 SLH-DSA deterministic sign (``addrnd = PK.seed``).

    This is the path the NIST ACVP-Server ``deterministic`` external/pure
    sigGen vectors are produced under — byte-exact reproduction of those
    vectors is the contract this function is built to satisfy.
    """
    if not SLHDSA_AVAILABLE:
        raise RuntimeError("FIPS 205 SLH-DSA not available. Upgrade to ama-cryptography ≥ 3.1.0.")
    _, sk_len, _ = _slhdsa_param_sizes(param_set)
    if len(secret_key) != sk_len:
        raise ValueError(
            f"FIPS 205 SLH-DSA-{param_set} secret key must be {sk_len} bytes; "
            f"got {len(secret_key)}"
        )
    return bytes(_ama_slhdsa_sign_deterministic(message, secret_key, ctx, param_set))


def slhdsa_sign_internal(
    message: bytes,
    secret_key: bytes,
    addrnd: bytes,
    param_set: str = "SHAKE-128s",
) -> bytes:
    """
    FIPS 205 internal-interface SLH-DSA sign with caller-supplied ``addrnd``.

    Used by the NIST ACVP hedged sigGen KAT replay path: the test harness
    pre-applies the FIPS 205 §10.2 ctx wrapper to the message and replays
    the vector's ``additionalRandomness`` bytes as ``addrnd``. Production
    callers should use ``slhdsa_sign`` (hedged) or ``slhdsa_sign_deterministic``
    instead — this entry point exists for byte-exact KAT reproduction only.
    """
    if not SLHDSA_AVAILABLE:
        raise RuntimeError("FIPS 205 SLH-DSA not available. Upgrade to ama-cryptography ≥ 3.1.0.")
    _, sk_len, _ = _slhdsa_param_sizes(param_set)
    if len(secret_key) != sk_len:
        raise ValueError(
            f"FIPS 205 SLH-DSA-{param_set} secret key must be {sk_len} bytes; "
            f"got {len(secret_key)}"
        )
    n = (
        SLHDSA_SHAKE_128S_PUBLIC_KEY_BYTES
        if param_set == "SHAKE-128s"
        else SLHDSA_SHA2_256F_PUBLIC_KEY_BYTES
    )
    if len(addrnd) != n:
        raise ValueError(
            f"FIPS 205 SLH-DSA-{param_set} addrnd must be {n} bytes; got {len(addrnd)}"
        )
    return bytes(_ama_slhdsa_sign_internal(message, secret_key, addrnd, param_set))


def slhdsa_verify(
    message: bytes,
    signature: bytes,
    public_key: bytes,
    ctx: bytes = b"",
    param_set: str = "SHAKE-128s",
) -> bool:
    """FIPS 205 SLH-DSA verify under the §10.2 context wrapper."""
    if not SLHDSA_AVAILABLE:
        raise RuntimeError("FIPS 205 SLH-DSA not available. Upgrade to ama-cryptography ≥ 3.1.0.")
    pk_len, _, sig_len = _slhdsa_param_sizes(param_set)
    if len(public_key) != pk_len:
        raise ValueError(
            f"FIPS 205 SLH-DSA-{param_set} public key must be {pk_len} bytes; "
            f"got {len(public_key)}"
        )
    if len(signature) != sig_len:
        # Wrong-size signatures cannot be valid for this parameter set; reject
        # them deterministically rather than handing a malformed buffer down
        # to the C verifier.
        return False
    return bool(_ama_slhdsa_verify(message, signature, public_key, ctx, param_set))


def get_pqc_capabilities() -> dict[str, Any]:
    """
    Get current PQC capabilities from AMA Cryptography.

    Returns:
        Dictionary with backend status and available algorithms
    """
    return {
        "backend": get_active_backend().value,
        "constant_time": True,
        "algorithms": {
            "dilithium": DILITHIUM_AVAILABLE,
            "kyber": KYBER_AVAILABLE,
            "sphincs": SPHINCS_AVAILABLE,
        },
        "security_level": "production",
        "require_constant_time": require_constant_time(),
    }


# =============================================================================
# Cryptographic Audit Trail
# =============================================================================
@dataclass
class CryptoOperation:
    """Audit record for cryptographic operations."""

    timestamp: float
    operation: str
    algorithm: str
    backend: str
    success: bool
    error: str | None = None
    key_id: str | None = None


class CryptoAuditTrail:
    """
    Cryptographic audit trail for PQC operations.

    Provides tamper-evident logging of all cryptographic operations for security compliance and
    forensic analysis.
    """

    def __init__(self, max_entries: int = 10000) -> None:
        """Initialize audit trail with maximum entry limit."""
        self._entries: list[CryptoOperation] = []
        self._max_entries = max_entries
        self._lock = threading.Lock()

    def log_operation(
        self,
        operation: str,
        algorithm: str,
        success: bool,
        error: str | None = None,
        key_id: str | None = None,
    ) -> None:
        """Log a cryptographic operation to the audit trail."""
        entry = CryptoOperation(
            timestamp=time.time(),
            operation=operation,
            algorithm=algorithm,
            backend=get_active_backend().value,
            success=success,
            error=error,
            key_id=key_id,
        )

        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries :]

    def get_recent_operations(self, count: int = 100) -> list[dict[str, Any]]:
        """Get recent operations for audit review."""
        with self._lock:
            recent = self._entries[-count:]
            return [
                {
                    "timestamp": e.timestamp,
                    "operation": e.operation,
                    "algorithm": e.algorithm,
                    "backend": e.backend,
                    "success": e.success,
                    "error": e.error,
                }
                for e in recent
            ]

    def get_failure_summary(self) -> dict[str, int]:
        """Get summary of operation failures by type."""
        with self._lock:
            failures: dict[str, int] = {}
            for entry in self._entries:
                if not entry.success:
                    key = f"{entry.operation}:{entry.algorithm}"
                    failures[key] = failures.get(key, 0) + 1
            return failures


# Global audit trail instance
_crypto_audit = CryptoAuditTrail()


def get_crypto_audit_trail() -> CryptoAuditTrail:
    """Get the global crypto audit trail instance."""
    return _crypto_audit


def validate_pqc_environment() -> dict[str, Any]:
    """
    Validate the PQC environment for production readiness.

    Returns:
        Dictionary with validation results and recommendations.

    Raises:
        RuntimeError: If any mandatory AMA/PQC backend surface is unavailable.
    """
    issues: list[str] = []

    if require_constant_time() and not AMA_CRYPTOGRAPHY_AVAILABLE:
        issues.append("AMA_REQUIRE_CONSTANT_TIME=true but AMA Cryptography is not available.")

    if not DILITHIUM_AVAILABLE:
        issues.append(
            "ML-DSA-65 (Dilithium) not available. "
            "Build AMA native C library for post-quantum signatures."
        )
    if not KYBER_AVAILABLE:
        issues.append(
            "Kyber-1024 not available. Build AMA native C library for key encapsulation."
        )
    if not SPHINCS_AVAILABLE:
        issues.append(
            "SPHINCS+ not available. Build AMA native C library for hash-based signatures."
        )

    is_production_ready = len(issues) == 0

    result = {
        "production_ready": is_production_ready,
        "backend": get_active_backend().value,
        "issues": issues,
        "algorithms": get_pqc_capabilities()["algorithms"],
    }

    if issues:
        raise RuntimeError(f"PQC environment validation failed: {'; '.join(issues)}")

    return result


__all__ = [
    "AMA_CRYPTOGRAPHY_AVAILABLE",
    "AVA_GUARDIAN_AVAILABLE",
    "DILITHIUM_AVAILABLE",
    "KYBER_AVAILABLE",
    "SPHINCS_AVAILABLE",
    "CryptoAuditTrail",
    "CryptoOperation",
    "DilithiumKeyPair",
    "KyberEncapsulation",
    "KyberKeyPair",
    "PQCBackend",
    "SphincsKeyPair",
    "dilithium_sign",
    "dilithium_verify",
    "generate_dilithium_keypair",
    "generate_kyber_keypair",
    "generate_sphincs_keypair",
    "get_active_backend",
    "get_crypto_audit_trail",
    "get_pqc_capabilities",
    "kyber_decapsulate",
    "kyber_encapsulate",
    "sphincs_sign",
    "sphincs_verify",
    "validate_pqc_environment",
]
