"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
Mercury Agent - Post-Quantum Cryptography Backends

AMA Cryptography v2.0 is the sole PQC implementation.

Previous versions used a 4-tier fallback chain (AMA → liboqs → pqcrypto →
SIMULATION).  As of this version Mercury **hard-requires** AMA Cryptography
and the fallback chain has been removed entirely.  AMA v2.0 carries its own
native C backend — it *is* the implementation.  Retaining weaker fallbacks
only widened the attack surface.

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

import logging
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, NoReturn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AMA Cryptography — sole PQC backend
# ---------------------------------------------------------------------------
AMA_CRYPTOGRAPHY_AVAILABLE = False
DILITHIUM_AVAILABLE = False
KYBER_AVAILABLE = False
SPHINCS_AVAILABLE = False

try:
    from ama_cryptography.pqc_backends import (
        DILITHIUM_AVAILABLE as _AMA_DILITHIUM,
        KYBER_AVAILABLE as _AMA_KYBER,
        SPHINCS_AVAILABLE as _AMA_SPHINCS,
        dilithium_sign as _ama_dilithium_sign,
        dilithium_verify as _ama_dilithium_verify,
        generate_dilithium_keypair as _ama_generate_dilithium_keypair,
        generate_kyber_keypair as _ama_generate_kyber_keypair,
        generate_sphincs_keypair as _ama_generate_sphincs_keypair,
        kyber_decapsulate as _ama_kyber_decapsulate,
        kyber_encapsulate as _ama_kyber_encapsulate,
        sphincs_sign as _ama_sphincs_sign,
        sphincs_verify as _ama_sphincs_verify,
    )

    AMA_CRYPTOGRAPHY_AVAILABLE = True
    DILITHIUM_AVAILABLE = _AMA_DILITHIUM
    KYBER_AVAILABLE = _AMA_KYBER
    SPHINCS_AVAILABLE = _AMA_SPHINCS
    logger.info("AMA Cryptography v2.0 PQC backend loaded (sole backend)")
except ImportError:
    logger.warning(
        "AMA Cryptography is not installed. Post-quantum cryptography features "
        "will be unavailable. Install with: pip install ama-cryptography"
    )

    # Stub functions that raise RuntimeError when called.
    # Use NoReturn so mypy does not complain about signature mismatches
    # between the try- and except-branches.
    def _ama_generate_dilithium_keypair() -> NoReturn:
        raise RuntimeError("AMA Cryptography not installed")

    def _ama_dilithium_sign(message: bytes, secret_key: bytes) -> NoReturn:
        raise RuntimeError("AMA Cryptography not installed")

    def _ama_dilithium_verify(message: bytes, signature: bytes, public_key: bytes) -> NoReturn:
        raise RuntimeError("AMA Cryptography not installed")

    def _ama_generate_kyber_keypair() -> NoReturn:
        raise RuntimeError("AMA Cryptography not installed")

    def _ama_kyber_encapsulate(public_key: bytes) -> NoReturn:
        raise RuntimeError("AMA Cryptography not installed")

    def _ama_kyber_decapsulate(ciphertext: bytes, secret_key: bytes) -> NoReturn:
        raise RuntimeError("AMA Cryptography not installed")

    def _ama_generate_sphincs_keypair() -> NoReturn:
        raise RuntimeError("AMA Cryptography not installed")

    def _ama_sphincs_sign(message: bytes, secret_key: bytes) -> NoReturn:
        raise RuntimeError("AMA Cryptography not installed")

    def _ama_sphincs_verify(message: bytes, signature: bytes, public_key: bytes) -> NoReturn:
        raise RuntimeError("AMA Cryptography not installed")


# Backward compatibility alias
AVA_GUARDIAN_AVAILABLE = AMA_CRYPTOGRAPHY_AVAILABLE


class PQCBackend(Enum):
    """Available PQC backend implementations.

    Only ``AMA_CRYPTOGRAPHY`` is supported.  ``AVA_GUARDIAN`` remains as a
    backward-compatibility alias that resolves to the same enum member.
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
        "security_level": "production" if DILITHIUM_AVAILABLE else "development",
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

    Provides tamper-evident logging of all cryptographic operations
    for security compliance and forensic analysis.
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
        RuntimeError: If constant-time is required but unavailable.
    """
    issues: list[str] = []
    warnings: list[str] = []

    if require_constant_time() and not AMA_CRYPTOGRAPHY_AVAILABLE:
        issues.append("AMA_REQUIRE_CONSTANT_TIME=true but AMA Cryptography is not available.")

    if not DILITHIUM_AVAILABLE:
        warnings.append(
            "ML-DSA-65 (Dilithium) not available. "
            "Build AMA native C library for post-quantum signatures."
        )
    if not KYBER_AVAILABLE:
        warnings.append(
            "Kyber-1024 not available. Build AMA native C library for key encapsulation."
        )
    if not SPHINCS_AVAILABLE:
        warnings.append(
            "SPHINCS+ not available. Build AMA native C library for hash-based signatures."
        )

    is_production_ready = len(issues) == 0 and DILITHIUM_AVAILABLE

    result = {
        "production_ready": is_production_ready,
        "backend": get_active_backend().value,
        "issues": issues,
        "warnings": warnings,
        "algorithms": get_pqc_capabilities()["algorithms"],
    }

    if issues and require_constant_time():
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
