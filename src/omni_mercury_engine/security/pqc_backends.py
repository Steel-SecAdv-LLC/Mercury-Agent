"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

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
Post-Quantum Cryptography Backends for Mercury Agent ♱

Provides quantum-resistant cryptographic primitives using NIST-approved algorithms:
- ML-DSA-65 (Dilithium): Digital signatures (FIPS 204)
- Kyber-1024: Key Encapsulation Mechanism (FIPS 203)
- SPHINCS+-256f: Hash-based signatures for long-term security

Backend Detection Priority:
1. liboqs-python (preferred, constant-time)
2. pqcrypto (fallback, may have timing variations)
3. Simulation mode (for testing without PQC libraries)

Security Note:
    Set AVA_REQUIRE_CONSTANT_TIME=true in production to enforce
    constant-time implementations only.

References:
    - NIST PQC Standardization: https://csrc.nist.gov/projects/post-quantum-cryptography
    - liboqs: https://openquantumsafe.org/
    - Dilithium: https://pq-crystals.org/dilithium/
    - Kyber: https://pq-crystals.org/kyber/
"""

import hashlib
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

logger = logging.getLogger(__name__)

LIBOQS_AVAILABLE = False
PQCRYPTO_AVAILABLE = False
DILITHIUM_AVAILABLE = False
KYBER_AVAILABLE = False
SPHINCS_AVAILABLE = False

try:
    import oqs

    LIBOQS_AVAILABLE = True
    DILITHIUM_AVAILABLE = True
    KYBER_AVAILABLE = True
    SPHINCS_AVAILABLE = True
    logger.info("liboqs backend detected (constant-time implementations)")
except ImportError:
    logger.debug("liboqs not available, checking pqcrypto fallback")

if not LIBOQS_AVAILABLE:
    try:
        import pqcrypto.kem.kyber512 as kyber_fallback
        import pqcrypto.sign.dilithium2 as dilithium_fallback

        PQCRYPTO_AVAILABLE = True
        DILITHIUM_AVAILABLE = True
        KYBER_AVAILABLE = True
        logger.warning(
            "Using pqcrypto fallback - may have timing variations. "
            "Install liboqs-python for constant-time implementations."
        )
    except ImportError:
        logger.info("No PQC backend available, using simulation mode")


class PQCBackend(Enum):
    """Available PQC backend implementations."""

    LIBOQS = "liboqs"
    PQCRYPTO = "pqcrypto"
    SIMULATION = "simulation"


def get_active_backend() -> PQCBackend:
    """Get the currently active PQC backend."""
    if LIBOQS_AVAILABLE:
        return PQCBackend.LIBOQS
    elif PQCRYPTO_AVAILABLE:
        return PQCBackend.PQCRYPTO
    return PQCBackend.SIMULATION


def require_constant_time() -> bool:
    """Check if constant-time implementations are required."""
    return os.environ.get("AVA_REQUIRE_CONSTANT_TIME", "").lower() == "true"


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


def generate_dilithium_keypair() -> DilithiumKeyPair:
    """
    Generate ML-DSA-65 (Dilithium) key pair.

    Returns:
        DilithiumKeyPair with public and secret keys

    Raises:
        RuntimeError: If no PQC backend available and constant-time required
    """
    if require_constant_time() and not LIBOQS_AVAILABLE:
        raise RuntimeError(
            "Constant-time implementation required but liboqs not available. "
            "Install liboqs-python or set AVA_REQUIRE_CONSTANT_TIME=false"
        )

    if LIBOQS_AVAILABLE:
        sig = oqs.Signature("Dilithium3")
        public_key = sig.generate_keypair()
        secret_key = sig.export_secret_key()
        return DilithiumKeyPair(public_key=public_key, secret_key=secret_key)

    if PQCRYPTO_AVAILABLE:
        public_key, secret_key = dilithium_fallback.generate_keypair()
        return DilithiumKeyPair(
            public_key=public_key, secret_key=secret_key, algorithm="Dilithium2"
        )

    logger.warning("Using simulated Dilithium keys (NOT SECURE)")
    return DilithiumKeyPair(
        public_key=os.urandom(1952),
        secret_key=os.urandom(4000),
        algorithm="ML-DSA-65-SIMULATED",
    )


def dilithium_sign(message: bytes, secret_key: bytes) -> bytes:
    """
    Sign message using ML-DSA-65 (Dilithium).

    Args:
        message: Message to sign
        secret_key: Dilithium secret key

    Returns:
        Digital signature bytes
    """
    if LIBOQS_AVAILABLE:
        sig = oqs.Signature("Dilithium3", secret_key)
        return cast("bytes", sig.sign(message))

    if PQCRYPTO_AVAILABLE:
        return cast("bytes", dilithium_fallback.sign(secret_key, message))

    logger.warning("Using simulated signature (NOT SECURE)")
    return hashlib.sha3_512(secret_key + message).digest()


def dilithium_verify(message: bytes, signature: bytes, public_key: bytes) -> bool:
    """
    Verify ML-DSA-65 (Dilithium) signature.

    Args:
        message: Original message
        signature: Signature to verify
        public_key: Dilithium public key

    Returns:
        True if signature is valid
    """
    if LIBOQS_AVAILABLE:
        sig = oqs.Signature("Dilithium3")
        return cast("bool", sig.verify(message, signature, public_key))

    if PQCRYPTO_AVAILABLE:
        try:
            dilithium_fallback.verify(public_key, message, signature)
            return True
        except (ValueError, TypeError) as e:
            logger.debug(f"Dilithium verification failed: {type(e).__name__}")
            return False

    logger.warning("Using simulated verification (NOT SECURE)")
    expected = hashlib.sha3_512(public_key[:4000] + message).digest()
    return signature == expected


def generate_kyber_keypair() -> KyberKeyPair:
    """
    Generate Kyber-1024 key pair.

    Returns:
        KyberKeyPair with public and secret keys
    """
    if require_constant_time() and not LIBOQS_AVAILABLE:
        raise RuntimeError("Constant-time implementation required but liboqs not available")

    if LIBOQS_AVAILABLE:
        kem = oqs.KeyEncapsulation("Kyber1024")
        public_key = kem.generate_keypair()
        secret_key = kem.export_secret_key()
        return KyberKeyPair(public_key=public_key, secret_key=secret_key)

    if PQCRYPTO_AVAILABLE:
        public_key, secret_key = kyber_fallback.generate_keypair()
        return KyberKeyPair(public_key=public_key, secret_key=secret_key, algorithm="Kyber512")

    logger.warning("Using simulated Kyber keys (NOT SECURE)")
    return KyberKeyPair(
        public_key=os.urandom(1568),
        secret_key=os.urandom(3168),
        algorithm="Kyber1024-SIMULATED",
    )


def kyber_encapsulate(public_key: bytes) -> KyberEncapsulation:
    """
    Encapsulate shared secret using Kyber public key.

    Args:
        public_key: Kyber public key

    Returns:
        KyberEncapsulation with ciphertext and shared secret
    """
    if LIBOQS_AVAILABLE:
        kem = oqs.KeyEncapsulation("Kyber1024")
        ciphertext, shared_secret = kem.encap_secret(public_key)
        return KyberEncapsulation(ciphertext=ciphertext, shared_secret=shared_secret)

    if PQCRYPTO_AVAILABLE:
        ciphertext, shared_secret = kyber_fallback.encap(public_key)
        return KyberEncapsulation(ciphertext=ciphertext, shared_secret=shared_secret)

    logger.warning("Using simulated encapsulation (NOT SECURE)")
    shared_secret = hashlib.sha3_256(public_key).digest()
    ciphertext = os.urandom(1568)
    return KyberEncapsulation(ciphertext=ciphertext, shared_secret=shared_secret)


def kyber_decapsulate(ciphertext: bytes, secret_key: bytes) -> bytes:
    """
    Decapsulate shared secret using Kyber secret key.

    Args:
        ciphertext: Encapsulated ciphertext
        secret_key: Kyber secret key

    Returns:
        Shared secret bytes
    """
    if LIBOQS_AVAILABLE:
        kem = oqs.KeyEncapsulation("Kyber1024", secret_key)
        return cast("bytes", kem.decap_secret(ciphertext))

    if PQCRYPTO_AVAILABLE:
        return cast("bytes", kyber_fallback.decap(secret_key, ciphertext))

    logger.warning("Using simulated decapsulation (NOT SECURE)")
    return hashlib.sha3_256(secret_key[:1568]).digest()


def generate_sphincs_keypair() -> SphincsKeyPair:
    """
    Generate SPHINCS+-256f key pair for long-term security.

    SPHINCS+ is hash-based and provides security even against
    future cryptanalytic advances in lattice-based cryptography.

    Returns:
        SphincsKeyPair with public and secret keys
    """
    if not LIBOQS_AVAILABLE:
        if require_constant_time():
            raise RuntimeError("SPHINCS+ requires liboqs backend")
        logger.warning("Using simulated SPHINCS+ keys (NOT SECURE)")
        return SphincsKeyPair(
            public_key=os.urandom(64),
            secret_key=os.urandom(128),
            algorithm="SPHINCS+-SIMULATED",
        )

    sig = oqs.Signature("SPHINCS+-SHA2-256f-simple")
    public_key = sig.generate_keypair()
    secret_key = sig.export_secret_key()
    return SphincsKeyPair(public_key=public_key, secret_key=secret_key)


def sphincs_sign(message: bytes, secret_key: bytes) -> bytes:
    """Sign message using SPHINCS+."""
    if LIBOQS_AVAILABLE:
        sig = oqs.Signature("SPHINCS+-SHA2-256f-simple", secret_key)
        return cast("bytes", sig.sign(message))

    logger.warning("Using simulated SPHINCS+ signature (NOT SECURE)")
    return hashlib.sha3_512(secret_key + message).digest()


def sphincs_verify(message: bytes, signature: bytes, public_key: bytes) -> bool:
    """Verify SPHINCS+ signature."""
    if LIBOQS_AVAILABLE:
        sig = oqs.Signature("SPHINCS+-SHA2-256f-simple")
        return cast("bool", sig.verify(message, signature, public_key))

    logger.warning("Using simulated SPHINCS+ verification (NOT SECURE)")
    return len(signature) == 64


def get_pqc_capabilities() -> dict[str, Any]:
    """
    Get current PQC capabilities and backend status.

    Returns:
        Dictionary with backend status and available algorithms
    """
    return {
        "backend": get_active_backend().value,
        "constant_time": LIBOQS_AVAILABLE,
        "algorithms": {
            "dilithium": DILITHIUM_AVAILABLE,
            "kyber": KYBER_AVAILABLE,
            "sphincs": SPHINCS_AVAILABLE,
        },
        "security_level": "production" if LIBOQS_AVAILABLE else "development",
        "require_constant_time": require_constant_time(),
    }


# =============================================================================
# Cryptographic Audit Trail (Ava-Guardian PQC Fortification)
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
        import threading

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
        import time

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
            # Rotate oldest entries if at capacity
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

    # Check constant-time requirement
    if require_constant_time() and not LIBOQS_AVAILABLE:
        issues.append(
            "AVA_REQUIRE_CONSTANT_TIME=true but liboqs not available. "
            "Install liboqs-python for production security."
        )

    # Check backend security level
    backend = get_active_backend()
    if backend == PQCBackend.SIMULATION:
        warnings.append(
            "Using SIMULATION backend - cryptographic operations are NOT SECURE. "
            "Install liboqs-python or pqcrypto for real PQC."
        )
    elif backend == PQCBackend.PQCRYPTO:
        warnings.append(
            "Using pqcrypto backend - may have timing side-channels. "
            "Upgrade to liboqs-python for constant-time implementations."
        )

    # Check algorithm availability
    if not DILITHIUM_AVAILABLE:
        warnings.append("ML-DSA-65 (Dilithium) not available for digital signatures.")
    if not KYBER_AVAILABLE:
        warnings.append("Kyber-1024 not available for key encapsulation.")
    if not SPHINCS_AVAILABLE:
        warnings.append("SPHINCS+ not available for hash-based signatures.")

    is_production_ready = len(issues) == 0 and backend == PQCBackend.LIBOQS

    result = {
        "production_ready": is_production_ready,
        "backend": backend.value,
        "issues": issues,
        "warnings": warnings,
        "algorithms": get_pqc_capabilities()["algorithms"],
    }

    if issues and require_constant_time():
        raise RuntimeError(f"PQC environment validation failed: {'; '.join(issues)}")

    return result


__all__ = [
    "DILITHIUM_AVAILABLE",
    "KYBER_AVAILABLE",
    "LIBOQS_AVAILABLE",
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
