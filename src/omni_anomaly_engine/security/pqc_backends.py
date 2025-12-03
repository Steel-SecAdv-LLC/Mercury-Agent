"""
OMNI ♱ AVA (O♱A)
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

"""
Post-Quantum Cryptography Backends for OMNI ♱ AVA

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

Research Sources:
    - NIST PQC Standardization: https://csrc.nist.gov/projects/post-quantum-cryptography
    - liboqs: https://openquantumsafe.org/
    - Dilithium: https://pq-crystals.org/dilithium/
    - Kyber: https://pq-crystals.org/kyber/

Original Implementation: Ava-Guardian (Steel Security Advisors LLC)
Integrated into OMNI ♱ AVA for quantum-resistant anomaly detection security.
"""

import hashlib
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

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
        return sig.sign(message)

    if PQCRYPTO_AVAILABLE:
        return dilithium_fallback.sign(secret_key, message)

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
        return sig.verify(message, signature, public_key)

    if PQCRYPTO_AVAILABLE:
        try:
            dilithium_fallback.verify(public_key, message, signature)
            return True
        except Exception:
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
        return kem.decap_secret(ciphertext)

    if PQCRYPTO_AVAILABLE:
        return kyber_fallback.decap(secret_key, ciphertext)

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
        return sig.sign(message)

    logger.warning("Using simulated SPHINCS+ signature (NOT SECURE)")
    return hashlib.sha3_512(secret_key + message).digest()


def sphincs_verify(message: bytes, signature: bytes, public_key: bytes) -> bool:
    """Verify SPHINCS+ signature."""
    if LIBOQS_AVAILABLE:
        sig = oqs.Signature("SPHINCS+-SHA2-256f-simple")
        return sig.verify(message, signature, public_key)

    logger.warning("Using simulated SPHINCS+ verification (NOT SECURE)")
    return len(signature) == 64


def get_pqc_capabilities() -> dict:
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


__all__ = [
    "PQCBackend",
    "DilithiumKeyPair",
    "KyberKeyPair",
    "KyberEncapsulation",
    "SphincsKeyPair",
    "get_active_backend",
    "get_pqc_capabilities",
    "generate_dilithium_keypair",
    "dilithium_sign",
    "dilithium_verify",
    "generate_kyber_keypair",
    "kyber_encapsulate",
    "kyber_decapsulate",
    "generate_sphincs_keypair",
    "sphincs_sign",
    "sphincs_verify",
    "DILITHIUM_AVAILABLE",
    "KYBER_AVAILABLE",
    "SPHINCS_AVAILABLE",
    "LIBOQS_AVAILABLE",
]
