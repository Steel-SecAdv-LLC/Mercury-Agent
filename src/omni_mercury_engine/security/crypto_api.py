# Copyright (C) 2025 Steel Security Advisors LLC
"""Algorithm-Agnostic Cryptographic API for Mercury Agent.

MercuryCrypto is a thin facade over AMA Cryptography's ``AmaCryptography``.
It delegates all cryptographic operations to AMA while providing Mercury-
specific ergonomics (security-level selection, audit-trail packaging, GOSNN
scalar integration).

Capabilities gained through AMA v2.0:
- AES-256-GCM authenticated encryption
- 6-layer crypto packages (hash + HMAC + Ed25519 + ML-DSA-65 + HKDF + RFC 3161)
- Ethical HKDF context binding
- Cython-accelerated math (18-37x speedup when native C library is built)

Security Levels:
- CLASSICAL: Ed25519/RSA (fast, widely supported)
- POST_QUANTUM: ML-DSA-65/Kyber-1024 (quantum-resistant)
- HYBRID: Both classical and post-quantum (maximum security)

References:
    - NIST SP 800-208: Recommendation for Stateful Hash-Based Signatures
    - NIST FIPS 204: Module-Lattice-Based Digital Signature Standard
    - NIST FIPS 203: Module-Lattice-Based Key-Encapsulation Mechanism
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ama_cryptography.crypto_api import (
    AESGCMProvider as _AmaAESGCMProvider,
    AlgorithmType as _AmaAlgorithmType,
    AmaCryptography as _AmaCryptography,
    CryptoPackageConfig as _AmaCryptoPackageConfig,
    CryptoPackageResult as _AmaCryptoPackageResult,
    create_crypto_package as _ama_create_crypto_package,
    get_pqc_capabilities as _ama_get_pqc_capabilities,
)

AMA_CRYPTO_API_AVAILABLE = True

# Public aliases — kept for backward compatibility with downstream importers.
AESGCMProvider: Any = _AmaAESGCMProvider
AmaAlgorithmType: Any = _AmaAlgorithmType
AmaCryptography: Any = _AmaCryptography
AmaCryptoPackageConfig: Any = _AmaCryptoPackageConfig
AmaCryptoPackageResult: Any = _AmaCryptoPackageResult
ama_create_crypto_package: Any = _ama_create_crypto_package
ama_get_pqc_capabilities: Any = _ama_get_pqc_capabilities

from omni_mercury_engine.security.pqc_backends import (
    dilithium_sign,
    dilithium_verify,
    generate_dilithium_keypair,
    generate_kyber_keypair,
    generate_sphincs_keypair,
    get_pqc_capabilities,
    kyber_decapsulate,
    kyber_encapsulate,
    sphincs_sign,
    sphincs_verify,
)

logger = logging.getLogger(__name__)

ED25519_AVAILABLE = False
InvalidSignature: type[Exception] | None = None
try:
    from cryptography.exceptions import InvalidSignature as _InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )

    InvalidSignature = _InvalidSignature
    ED25519_AVAILABLE = True
except ImportError:
    logger.debug("cryptography library not available for Ed25519")


class AlgorithmType(Enum):
    """Supported cryptographic algorithm types."""

    ED25519 = "ed25519"
    ML_DSA_65 = "ml-dsa-65"
    KYBER_1024 = "kyber-1024"
    SPHINCS_PLUS = "sphincs+"
    HYBRID = "hybrid"
    AES_256_GCM = "aes-256-gcm"


class SecurityLevel(Enum):
    """Security level for cryptographic operations."""

    CLASSICAL = "classical"
    POST_QUANTUM = "post_quantum"
    HYBRID = "hybrid"


class CryptoBackend(Enum):
    """Cryptographic backend selection."""

    AUTO = "auto"
    CLASSICAL_ONLY = "classical_only"
    PQC_ONLY = "pqc_only"
    HYBRID = "hybrid"


# ---------------------------------------------------------------------------
# Mercury data classes (backward-compatible interface)
# ---------------------------------------------------------------------------


@dataclass
class KeyPair:
    """Generic key pair container."""

    public_key: bytes
    secret_key: bytes
    algorithm: AlgorithmType
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Signature:
    """Digital signature with metadata."""

    signature: bytes
    algorithm: AlgorithmType
    public_key_hash: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HybridSignature:
    """Hybrid signature combining classical and post-quantum."""

    classical_signature: bytes
    pqc_signature: bytes
    classical_algorithm: AlgorithmType
    pqc_algorithm: AlgorithmType
    timestamp: float = field(default_factory=time.time)


@dataclass
class EncapsulatedSecret:
    """Key encapsulation result."""

    ciphertext: bytes
    shared_secret: bytes
    algorithm: AlgorithmType


@dataclass
class CryptoPackageConfig:
    """Configuration for cryptographic package creation."""

    sign_data: bool = True
    include_timestamp: bool = True
    include_metadata: bool = True
    security_level: SecurityLevel = SecurityLevel.POST_QUANTUM
    hash_algorithm: str = "sha3-256"
    use_six_layer: bool = False


@dataclass
class CryptoPackageResult:
    """Result of cryptographic package creation."""

    data_hash: str
    signature: Signature | None = None
    hybrid_signature: HybridSignature | None = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    verified: bool = False
    ama_package: AmaCryptoPackageResult | None = None


# ---------------------------------------------------------------------------
# Provider classes — delegate to AMA Cryptography
# ---------------------------------------------------------------------------


class Ed25519Provider:
    """Ed25519 classical signature provider."""

    def __init__(self) -> None:
        """Initialize the instance."""
        if not ED25519_AVAILABLE:
            raise RuntimeError("cryptography library required for Ed25519")
        self._private_key: Ed25519PrivateKey | None = None
        self._public_key: Ed25519PublicKey | None = None

    def generate_keypair(self) -> KeyPair:
        """Generate Ed25519 key pair."""
        self._private_key = Ed25519PrivateKey.generate()
        self._public_key = self._private_key.public_key()

        public_bytes = self._public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        private_bytes = self._private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )

        return KeyPair(
            public_key=public_bytes,
            secret_key=private_bytes,
            algorithm=AlgorithmType.ED25519,
        )

    def sign(self, message: bytes, secret_key: bytes) -> bytes:
        """Sign message with Ed25519."""
        private_key = Ed25519PrivateKey.from_private_bytes(secret_key)
        signature: bytes = private_key.sign(message)
        return signature

    def verify(self, message: bytes, signature: bytes, public_key: bytes) -> bool:
        """Verify Ed25519 signature."""
        try:
            pub_key = Ed25519PublicKey.from_public_bytes(public_key)
            pub_key.verify(signature, message)
            return True
        except (ValueError, TypeError) as e:
            logger.debug(f"Ed25519 verification failed: {type(e).__name__}")
            return False
        except Exception as e:
            if InvalidSignature is not None and isinstance(e, InvalidSignature):
                logger.debug(f"Ed25519 verification failed: {type(e).__name__}")
                return False
            raise


class MLDSAProvider:
    """ML-DSA-65 (Dilithium) post-quantum signature provider — delegates to AMA."""

    def generate_keypair(self) -> KeyPair:
        """Generate keypair."""
        dilithium_kp = generate_dilithium_keypair()
        return KeyPair(
            public_key=dilithium_kp.public_key,
            secret_key=dilithium_kp.secret_key,
            algorithm=AlgorithmType.ML_DSA_65,
            metadata={"pqc_algorithm": dilithium_kp.algorithm},
        )

    def sign(self, message: bytes, secret_key: bytes) -> bytes:
        """Sign."""
        return dilithium_sign(message, secret_key)

    def verify(self, message: bytes, signature: bytes, public_key: bytes) -> bool:
        """Verify."""
        return dilithium_verify(message, signature, public_key)


class KyberProvider:
    """Kyber-1024 key encapsulation provider — delegates to AMA."""

    def generate_keypair(self) -> KeyPair:
        """Generate keypair."""
        kyber_kp = generate_kyber_keypair()
        return KeyPair(
            public_key=kyber_kp.public_key,
            secret_key=kyber_kp.secret_key,
            algorithm=AlgorithmType.KYBER_1024,
            metadata={"pqc_algorithm": kyber_kp.algorithm},
        )

    def encapsulate(self, public_key: bytes) -> EncapsulatedSecret:
        """Encapsulate."""
        result = kyber_encapsulate(public_key)
        return EncapsulatedSecret(
            ciphertext=result.ciphertext,
            shared_secret=result.shared_secret,
            algorithm=AlgorithmType.KYBER_1024,
        )

    def decapsulate(self, ciphertext: bytes, secret_key: bytes) -> bytes:
        """Decapsulate."""
        return kyber_decapsulate(ciphertext, secret_key)


class SphincsProvider:
    """SPHINCS+ hash-based signature provider — delegates to AMA."""

    def generate_keypair(self) -> KeyPair:
        """Generate keypair."""
        sphincs_kp = generate_sphincs_keypair()
        return KeyPair(
            public_key=sphincs_kp.public_key,
            secret_key=sphincs_kp.secret_key,
            algorithm=AlgorithmType.SPHINCS_PLUS,
            metadata={"pqc_algorithm": sphincs_kp.algorithm},
        )

    def sign(self, message: bytes, secret_key: bytes) -> bytes:
        """Sign."""
        return sphincs_sign(message, secret_key)

    def verify(self, message: bytes, signature: bytes, public_key: bytes) -> bool:
        """Verify."""
        return sphincs_verify(message, signature, public_key)


class HybridSignatureProvider:
    """Hybrid signature provider combining classical and post-quantum."""

    def __init__(self) -> None:
        """Initialize the instance."""
        self.classical_provider: Ed25519Provider | None = None
        self.pqc_provider = MLDSAProvider()

        if ED25519_AVAILABLE:
            self.classical_provider = Ed25519Provider()

    def generate_keypairs(self) -> tuple[KeyPair | None, KeyPair]:
        """Generate both classical and PQC key pairs."""
        classical_kp = None
        if self.classical_provider:
            classical_kp = self.classical_provider.generate_keypair()
        pqc_kp = self.pqc_provider.generate_keypair()
        return classical_kp, pqc_kp

    def sign(
        self,
        message: bytes,
        classical_secret: bytes | None,
        pqc_secret: bytes,
    ) -> HybridSignature:
        """Create hybrid signature."""
        classical_sig = b""
        classical_algo = AlgorithmType.ED25519

        if self.classical_provider and classical_secret:
            classical_sig = self.classical_provider.sign(message, classical_secret)

        pqc_sig = self.pqc_provider.sign(message, pqc_secret)

        return HybridSignature(
            classical_signature=classical_sig,
            pqc_signature=pqc_sig,
            classical_algorithm=classical_algo,
            pqc_algorithm=AlgorithmType.ML_DSA_65,
        )

    def verify(
        self,
        message: bytes,
        hybrid_sig: HybridSignature,
        classical_public: bytes | None,
        pqc_public: bytes,
    ) -> tuple[bool, bool]:
        """Verify hybrid signature.

        Returns (classical_valid, pqc_valid).
        """
        classical_valid = True
        if self.classical_provider and classical_public and hybrid_sig.classical_signature:
            classical_valid = self.classical_provider.verify(
                message, hybrid_sig.classical_signature, classical_public
            )

        pqc_valid = self.pqc_provider.verify(message, hybrid_sig.pqc_signature, pqc_public)

        return classical_valid, pqc_valid


# ---------------------------------------------------------------------------
# MercuryCrypto — thin facade over AmaCryptography
# ---------------------------------------------------------------------------

# Map Mercury SecurityLevel → AMA AlgorithmType
_SECURITY_LEVEL_TO_AMA: dict[SecurityLevel, Any] = {
    SecurityLevel.CLASSICAL: AmaAlgorithmType.ED25519,
    SecurityLevel.POST_QUANTUM: AmaAlgorithmType.ML_DSA_65,
    SecurityLevel.HYBRID: AmaAlgorithmType.HYBRID_SIG,
}


class MercuryCrypto:
    """Unified cryptographic interface for Mercury Agent.

    Thin facade over AMA Cryptography's ``AmaCryptography``, providing:
    - Algorithm-agnostic signing, verification, and key encapsulation
    - AES-256-GCM authenticated encryption (via AMA)
    - 6-layer crypto packages (via AMA's ``create_crypto_package``)
    - Ethical HKDF context binding
    - Backward-compatible Mercury API

    Example:
        crypto = MercuryCrypto(security_level=SecurityLevel.POST_QUANTUM)
        keypair = crypto.generate_signing_keypair()
        signature = crypto.sign(b"anomaly detection result", keypair.secret_key)
        is_valid = crypto.verify(b"anomaly detection result", signature, keypair.public_key)
    """

    def __init__(
        self,
        security_level: SecurityLevel = SecurityLevel.POST_QUANTUM,
        backend: CryptoBackend = CryptoBackend.AUTO,
    ):
        """Initialize the instance."""
        self.security_level = security_level
        self.backend = backend

        ama_algo = _SECURITY_LEVEL_TO_AMA[security_level]
        self._ama = AmaCryptography(algorithm=ama_algo)

        # Mercury provider wrappers for backward compatibility
        self.mldsa_provider = MLDSAProvider()
        self.kyber_provider = KyberProvider()
        self.sphincs_provider = SphincsProvider()
        self.hybrid_provider = HybridSignatureProvider()

        self.ed25519_provider: Ed25519Provider | None = None
        if ED25519_AVAILABLE:
            self.ed25519_provider = Ed25519Provider()

        self._signing_keypair: KeyPair | None = None
        self._kem_keypair: KeyPair | None = None

        logger.info(
            f"MercuryCrypto initialized via AmaCryptography "
            f"(level={security_level.value}, backend={backend.value})"
        )

    def generate_signing_keypair(self, algorithm: AlgorithmType | None = None) -> KeyPair:
        """Generate signing key pair based on security level."""
        if algorithm:
            if algorithm == AlgorithmType.ED25519:
                if not self.ed25519_provider:
                    raise RuntimeError("Ed25519 not available")
                return self.ed25519_provider.generate_keypair()
            elif algorithm == AlgorithmType.ML_DSA_65:
                return self.mldsa_provider.generate_keypair()
            elif algorithm == AlgorithmType.SPHINCS_PLUS:
                return self.sphincs_provider.generate_keypair()

        if self.security_level == SecurityLevel.CLASSICAL:
            if self.ed25519_provider:
                return self.ed25519_provider.generate_keypair()
            return self.mldsa_provider.generate_keypair()

        return self.mldsa_provider.generate_keypair()

    def generate_kem_keypair(self) -> KeyPair:
        """Generate key encapsulation key pair."""
        return self.kyber_provider.generate_keypair()

    def sign(
        self,
        message: bytes,
        secret_key: bytes,
        algorithm: AlgorithmType | None = None,
    ) -> Signature:
        """Sign message with appropriate algorithm."""
        if algorithm is None:
            algorithm = (
                AlgorithmType.ED25519
                if self.security_level == SecurityLevel.CLASSICAL
                else AlgorithmType.ML_DSA_65
            )

        if algorithm == AlgorithmType.ED25519:
            if not self.ed25519_provider:
                raise RuntimeError("Ed25519 not available")
            sig_bytes = self.ed25519_provider.sign(message, secret_key)
        elif algorithm == AlgorithmType.ML_DSA_65:
            sig_bytes = self.mldsa_provider.sign(message, secret_key)
        elif algorithm == AlgorithmType.SPHINCS_PLUS:
            sig_bytes = self.sphincs_provider.sign(message, secret_key)
        else:
            raise ValueError(f"Unsupported signing algorithm: {algorithm}")

        return Signature(
            signature=sig_bytes,
            algorithm=algorithm,
            public_key_hash=hashlib.sha3_256(secret_key[:32]).hexdigest()[:16],
        )

    def verify(
        self,
        message: bytes,
        signature: Signature,
        public_key: bytes,
    ) -> bool:
        """Verify signature."""
        if signature.algorithm == AlgorithmType.ED25519:
            if not self.ed25519_provider:
                return False
            return self.ed25519_provider.verify(message, signature.signature, public_key)
        elif signature.algorithm == AlgorithmType.ML_DSA_65:
            return self.mldsa_provider.verify(message, signature.signature, public_key)
        elif signature.algorithm == AlgorithmType.SPHINCS_PLUS:
            return self.sphincs_provider.verify(message, signature.signature, public_key)
        return False

    def encapsulate(self, public_key: bytes) -> EncapsulatedSecret:
        """Encapsulate shared secret using Kyber."""
        return self.kyber_provider.encapsulate(public_key)

    def decapsulate(self, ciphertext: bytes, secret_key: bytes) -> bytes:
        """Decapsulate shared secret."""
        return self.kyber_provider.decapsulate(ciphertext, secret_key)

    def encrypt(
        self,
        plaintext: bytes,
        key: bytes,
        nonce: bytes | None = None,
        aad: bytes = b"",
    ) -> dict[str, Any]:
        """Encrypt data using AES-256-GCM via AMA Cryptography.

        Args:
            plaintext: Data to encrypt
            key: 32-byte AES-256 key
            nonce: 12-byte nonce (auto-generated if None)
            aad: Additional authenticated data

        Returns:
            Dict with 'ciphertext', 'nonce', 'tag', 'aad' keys
        """
        provider = AESGCMProvider()
        result: dict[str, Any] = provider.encrypt(plaintext, key, nonce=nonce, aad=aad)
        return result

    def decrypt(
        self,
        ciphertext: bytes,
        key: bytes,
        nonce: bytes,
        tag: bytes = b"",
        aad: bytes = b"",
    ) -> bytes:
        """Decrypt data using AES-256-GCM via AMA Cryptography.

        Args:
            ciphertext: Encrypted data
            key: 32-byte AES-256 key
            nonce: 12-byte nonce
            tag: 16-byte auth tag (empty if tag is appended to ciphertext)
            aad: Additional authenticated data

        Returns:
            Decrypted plaintext
        """
        provider = AESGCMProvider()
        decrypted: bytes = provider.decrypt(ciphertext, key, nonce, tag, aad=aad)
        return decrypted

    def create_crypto_package(
        self,
        data: dict[str, Any],
        config: CryptoPackageConfig | None = None,
    ) -> CryptoPackageResult:
        """Create cryptographic package for anomaly detection results.

        When ``config.use_six_layer`` is True, delegates to AMA's 6-layer
        ``create_crypto_package`` for defense-in-depth protection:
          Layer 1: SHA3-256 content hash
          Layer 2: HMAC-SHA3-256 authentication
          Layer 3: Ed25519 classical signature
          Layer 4: ML-DSA-65 quantum-resistant signature
          Layer 5: HKDF key derivation
          Layer 6: RFC 3161 timestamp

        Otherwise uses Mercury's standard hash + sign package.
        """
        if config is None:
            config = CryptoPackageConfig()

        data_bytes = json.dumps(data, sort_keys=True, default=str).encode()

        # 6-layer package via AMA
        if config.use_six_layer:
            ama_config = AmaCryptoPackageConfig(
                signature_algorithm=AmaAlgorithmType.HYBRID_SIG,
            )
            ama_pkg = ama_create_crypto_package(data_bytes, config=ama_config)
            return CryptoPackageResult(
                data_hash=ama_pkg.content_hash,
                metadata={
                    "hash_algorithm": "sha3-256",
                    "six_layer": True,
                    "hmac_tag_present": ama_pkg.hmac_tag is not None,
                    "derived_key_count": len(ama_pkg.derived_keys) if ama_pkg.derived_keys else 0,
                },
                verified=True,
                ama_package=ama_pkg,
            )

        # Standard Mercury package
        if config.hash_algorithm == "sha3-512":
            data_hash = hashlib.sha3_512(data_bytes).hexdigest()
        else:
            data_hash = hashlib.sha3_256(data_bytes).hexdigest()

        result = CryptoPackageResult(
            data_hash=data_hash,
            metadata={"hash_algorithm": config.hash_algorithm},
        )

        if config.sign_data:
            if self._signing_keypair is None:
                self._signing_keypair = self.generate_signing_keypair()

            if config.security_level == SecurityLevel.HYBRID:
                classical_kp, pqc_kp = self.hybrid_provider.generate_keypairs()
                result.hybrid_signature = self.hybrid_provider.sign(
                    data_bytes,
                    classical_kp.secret_key if classical_kp else None,
                    pqc_kp.secret_key,
                )
            else:
                result.signature = self.sign(data_bytes, self._signing_keypair.secret_key)

        if config.include_timestamp:
            result.metadata["signed_at"] = time.time()

        return result

    def get_capabilities(self) -> dict[str, Any]:
        """Get current cryptographic capabilities."""
        pqc_caps = get_pqc_capabilities()
        ama_caps = ama_get_pqc_capabilities()
        return {
            "security_level": self.security_level.value,
            "backend": self.backend.value,
            "classical_available": ED25519_AVAILABLE,
            "pqc_capabilities": pqc_caps,
            "ama_capabilities": ama_caps,
            "aes_256_gcm": True,
            "six_layer_packages": True,
            "supported_algorithms": [
                AlgorithmType.ML_DSA_65.value,
                AlgorithmType.KYBER_1024.value,
                AlgorithmType.SPHINCS_PLUS.value,
                AlgorithmType.AES_256_GCM.value,
            ]
            + ([AlgorithmType.ED25519.value] if ED25519_AVAILABLE else []),
        }


__all__ = [
    "AlgorithmType",
    "CryptoBackend",
    "CryptoPackageConfig",
    "CryptoPackageResult",
    "Ed25519Provider",
    "EncapsulatedSecret",
    "HybridSignature",
    "HybridSignatureProvider",
    "KeyPair",
    "KyberProvider",
    "MLDSAProvider",
    "MercuryCrypto",
    "SecurityLevel",
    "Signature",
    "SphincsProvider",
]
