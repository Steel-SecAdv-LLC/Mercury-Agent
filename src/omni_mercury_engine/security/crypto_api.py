# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Algorithm-Agnostic Cryptographic API for Mercury Agent.

MercuryCrypto is a thin facade over AMA Cryptography's ``AmaCryptography``.
It delegates all cryptographic operations to AMA while providing Mercury-
specific ergonomics (security-level selection, audit-trail packaging, GOSNN
scalar integration).

Capabilities gained through AMA v2.0:
- AES-256-GCM authenticated encryption
- 6-layer crypto packages (hash + HMAC + Ed25519 + ML-DSA-65 + HKDF + RFC 3161)
- Ethical HKDF context binding
- Cython-accelerated math when the native C library is built (speedup measured
  by benchmarks/crypto_backend_benchmark.py; see artifacts/crypto_backend_benchmark.json)

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

# Agent-surface PQC gate: this module is a facade over AMA Cryptography and
# cannot function without it. Enforcing the scoped gate here turns the cryptic
# ``ModuleNotFoundError`` a missing AMA build would raise at the import below
# into the named, fail-closed refusal with recovery steps (see
# omni_mercury_engine._pqc_gate). Pinned by
# tests/test_pqc_startup_gate.py::TestScopedGateContract.
from omni_mercury_engine._pqc_gate import _enforce_pqc_production_gate as _pqc_enforce

_pqc_enforce()
del _pqc_enforce

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from ama_cryptography.crypto_api import (
    AESGCMProvider as _AmaAESGCMProvider,
    AlgorithmType as _AmaAlgorithmType,
    AmaCryptography as _AmaCryptography,
    CryptoPackageConfig as _AmaCryptoPackageConfig,
    CryptoPackageResult as _AmaCryptoPackageResult,
    create_crypto_package as _ama_create_crypto_package,
    get_pqc_capabilities as _ama_get_pqc_capabilities,
    verify_crypto_package as _ama_verify_crypto_package,
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
    ED25519_AVAILABLE,
    dilithium_sign,
    dilithium_verify,
    ed25519_sign,
    ed25519_verify,
    generate_dilithium_keypair,
    generate_ed25519_keypair,
    generate_kyber_keypair,
    generate_sphincs_keypair,
    get_pqc_capabilities,
    kyber_decapsulate,
    kyber_encapsulate,
    sphincs_sign,
    sphincs_verify,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


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


#: Length, in hex characters, of the advisory key fingerprint stored in
#: :attr:`Signature.public_key_hash`.  Single source of truth for every
#: site that formats the fingerprint (signing here, corpus verification
#: in ``sigma_immutable_corpus``) so the persisted record format cannot
#: drift between writers.
KEY_FINGERPRINT_HEX_CHARS = 16


def key_fingerprint(key_material: bytes) -> str:
    """Format the advisory SHA3-256 key fingerprint for ``Signature`` records.

    Args:
        key_material: Key bytes to fingerprint (public key, or a secret-key
            prefix at signing time).

    Returns:
        The first :data:`KEY_FINGERPRINT_HEX_CHARS` hex characters of the
        SHA3-256 digest.
    """
    return hashlib.sha3_256(key_material).hexdigest()[:KEY_FINGERPRINT_HEX_CHARS]


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
    """Ed25519 classical signature provider — delegates to AMA's native backend.

    AMA Cryptography is Mercury's sole production crypto backend; since the
    v5.0.0 pin the classical signature half runs on AMA's hardened native
    Ed25519 (RFC 8032 with canonical-encoding rejection and a FIPS 140-3
    pairwise consistency test at keygen) rather than PyCA ``cryptography``.
    Ed25519 signing is deterministic and verification is RFC 8032
    interoperable, so keys and signatures produced by the previous
    PyCA-backed provider — including the committed σ_Immutable corpus
    manifest — verify unchanged.

    Key format is unchanged: ``KeyPair.secret_key`` is the 32-byte RFC 8032
    seed; ``sign`` also accepts AMA's 64-byte ``seed || public_key`` form.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        if not ED25519_AVAILABLE:
            raise RuntimeError("AMA Cryptography native Ed25519 backend required")

    def generate_keypair(self) -> KeyPair:
        """Generate Ed25519 key pair (secret_key is the 32-byte seed)."""
        kp = generate_ed25519_keypair()
        return KeyPair(
            public_key=kp.public_key,
            secret_key=kp.secret_key,
            algorithm=AlgorithmType.ED25519,
        )

    def sign(self, message: bytes, secret_key: bytes) -> bytes:
        """Sign message with Ed25519 (32-byte seed or 64-byte native key)."""
        return ed25519_sign(message, secret_key)

    def verify(self, message: bytes, signature: bytes, public_key: bytes) -> bool:
        """Verify Ed25519 signature; malformed inputs verify False."""
        return ed25519_verify(message, signature, public_key)


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

        The classical half fails closed: when the verifier supplies a
        classical public key it is demanding both halves, so a hybrid
        signature whose ``classical_signature`` is absent — stripped in
        transit, or produced without the classical secret — verifies
        ``classical_valid=False``, never vacuously ``True``. Reporting a
        stripped signature as classically valid would let a PQC-only forgery
        pass a verifier that asked for the hybrid guarantee, defeating the
        reason a hybrid exists (each half protects against the other's
        algorithm breaking). Passing ``classical_public=None`` is the one
        explicit opt-out: the verifier has no classical key, so only the PQC
        half is meaningful and ``classical_valid`` reports ``True``.
        """
        if classical_public is None:
            # Explicit opt-out: no classical key to verify against.
            classical_valid = True
        elif self.classical_provider is None or not hybrid_sig.classical_signature:
            # Fail closed: a classical key was supplied, but the classical
            # half cannot be verified (unavailable provider, empty key, or a
            # missing/stripped classical signature).
            classical_valid = False
        else:
            classical_valid = self.classical_provider.verify(
                message, hybrid_sig.classical_signature, classical_public
            )

        pqc_valid = self.pqc_provider.verify(message, hybrid_sig.pqc_signature, pqc_public)

        return classical_valid, pqc_valid


# ---------------------------------------------------------------------------
# MercuryCrypto — thin facade over AmaCryptography
# ---------------------------------------------------------------------------

#: The algorithms a single-signature package can be signed with.
_PACKAGE_SIGNATURE_ALGORITHMS = frozenset(
    {AlgorithmType.ED25519, AlgorithmType.ML_DSA_65, AlgorithmType.SPHINCS_PLUS}
)


def _carried_key(encoded: Any) -> bytes | None:
    """A public key from a package's metadata, or ``None`` if it carries no usable one."""
    if not isinstance(encoded, str) or not encoded:
        return None
    try:
        return bytes.fromhex(encoded)
    except ValueError:
        return None


def _holds(check: Callable[[], bool]) -> bool:
    """Whether a signature check passes; a key or signature it cannot use does not."""
    try:
        return bool(check())
    except (ValueError, TypeError):
        return False


def _content_matches(data: dict[str, Any], package: CryptoPackageResult) -> tuple[bytes, bool]:
    """The bytes a standard package was created over, and whether its hash is theirs."""
    data_bytes = json.dumps(data, sort_keys=True, default=str).encode()
    algorithm = str(package.metadata.get("hash_algorithm", "sha3-256"))
    digest = hashlib.sha3_512 if algorithm == "sha3-512" else hashlib.sha3_256
    return data_bytes, digest(data_bytes).hexdigest() == package.data_hash


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
        self._hybrid_keypairs: tuple[KeyPair | None, KeyPair] | None = None
        # Held while a package-signing key is made or replaced: two threads
        # that both found none each made one, and the package signed with the
        # pair that lost was signed by a key this object no longer answers for.
        self._package_keys_lock = threading.Lock()
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
            public_key_hash=key_fingerprint(secret_key[:32]),
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
            if config.security_level == SecurityLevel.HYBRID:
                # One pair of keys for this object, kept, and their public
                # halves published with every package. Each package used to be
                # signed with a pair generated on the spot and dropped: nobody,
                # this object included, held the keys to check the signature.
                classical_kp, pqc_kp = self._hybrid_signing_keypairs()
                result.hybrid_signature = self.hybrid_provider.sign(
                    data_bytes,
                    classical_kp.secret_key if classical_kp else None,
                    pqc_kp.secret_key,
                )
                result.metadata["hybrid_public_keys"] = {
                    "classical": classical_kp.public_key.hex() if classical_kp else None,
                    "pqc": pqc_kp.public_key.hex(),
                }
            else:
                # Signed with the algorithm of the key, and the key's public
                # half published, as for a hybrid package. The key was held in
                # a private attribute and named nowhere: the signature on a
                # default package could be checked by nobody.
                keypair = self._package_signing_keypair()
                result.signature = self.sign(data_bytes, keypair.secret_key, keypair.algorithm)
                result.metadata["public_key"] = keypair.public_key.hex()

        if config.include_timestamp:
            result.metadata["signed_at"] = time.time()

        return result

    def _package_signing_keypair(self) -> KeyPair:
        """The key this object signs single-signature packages with, generated once."""
        with self._package_keys_lock:
            if self._signing_keypair is None:
                self._signing_keypair = self.generate_signing_keypair()
            return self._signing_keypair

    def _hybrid_signing_keypairs(self) -> tuple[KeyPair | None, KeyPair]:
        """This object's hybrid signing keys, generated once."""
        with self._package_keys_lock:
            if self._hybrid_keypairs is None:
                self._hybrid_keypairs = self.hybrid_provider.generate_keypairs()
            return self._hybrid_keypairs

    def use_package_signing_keys(
        self,
        *,
        signing: KeyPair | None = None,
        hybrid: tuple[KeyPair | None, KeyPair] | None = None,
    ) -> None:
        """Sign packages with keys the caller keeps, instead of ones made here.

        A key made here lives as long as this object. A verifier who pinned
        its public half can check nothing signed after a restart, because the
        next process signs with a new one. Keys that are to outlive the
        process are the caller's to store (``security.signing_keys`` keeps an
        ML-DSA-65 pair in an owner-only file) and are handed over here.

        Args:
            signing: The pair for single-signature packages (Ed25519, ML-DSA-65
                or SPHINCS+); ``None`` leaves the current one.
            hybrid: ``(classical, post-quantum)`` for hybrid packages: an
                Ed25519 pair, or ``None`` where Ed25519 is unavailable, and an
                ML-DSA-65 pair. ``None`` leaves the current ones.

        Raises:
            ValueError: If a pair is of an algorithm its place cannot sign with.
        """
        if signing is not None and signing.algorithm not in _PACKAGE_SIGNATURE_ALGORITHMS:
            raise ValueError(f"packages are not signed with {signing.algorithm.value}")
        if hybrid is not None:
            classical, pqc = hybrid
            if classical is not None and classical.algorithm is not AlgorithmType.ED25519:
                raise ValueError("the classical half of a hybrid signature is Ed25519")
            if pqc.algorithm is not AlgorithmType.ML_DSA_65:
                raise ValueError("the post-quantum half of a hybrid signature is ML-DSA-65")
        with self._package_keys_lock:
            if signing is not None:
                self._signing_keypair = signing
            if hybrid is not None:
                self._hybrid_keypairs = hybrid

    def signing_public_key(self) -> bytes:
        """The public half of the key this object signs single-signature packages with.

        Hand it to a verifier out of band, as with :meth:`hybrid_public_keys`:
        the copy in a package's ``metadata["public_key"]`` shows only that the
        package agrees with itself.
        """
        return self._package_signing_keypair().public_key

    def hybrid_public_keys(self) -> tuple[bytes | None, bytes]:
        """The public halves of this object's hybrid signing keys: classical, post-quantum.

        Hand these to a verifier out of band. The copy a package carries in
        ``metadata["hybrid_public_keys"]`` shows only that the package agrees
        with itself; it does not show who signed it.
        """
        classical_kp, pqc_kp = self._hybrid_signing_keypairs()
        return (classical_kp.public_key if classical_kp else None, pqc_kp.public_key)

    def verify_signed_package(
        self,
        data: dict[str, Any],
        package: CryptoPackageResult,
        *,
        public_key: bytes | None = None,
    ) -> dict[str, bool]:
        """Check a standard package that carries one signature (the default levels).

        Args:
            data: The content the package was created over.
            package: A result of :meth:`create_crypto_package` carrying a
                ``signature``.
            public_key: The signer's public key, obtained out of band.
                ``None`` uses the copy inside the package.

        Returns:
            ``hash``: the content matches the package's hash. ``signature``:
            the signature verifies under the key. ``key_pinned``: the key came
            from the caller, not from the package. ``core_valid``: hash and
            signature hold (the package agrees with itself). ``all_valid``:
            ``core_valid`` and ``key_pinned``. A key that is malformed, or
            absent from both the call and the package, fails ``signature``; it
            does not raise.

        Raises:
            ValueError: If the package carries no signature.
        """
        if package.signature is None:
            raise ValueError("this package carries no signature to verify")
        key_pinned = public_key is not None
        if public_key is None:
            public_key = _carried_key(package.metadata.get("public_key"))
        data_bytes, hash_ok = _content_matches(data, package)
        signature = package.signature
        signature_ok = public_key is not None and _holds(
            lambda: self.verify(data_bytes, signature, public_key)
        )
        core_valid = bool(hash_ok and signature_ok)
        return {
            "hash": hash_ok,
            "signature": bool(signature_ok),
            "key_pinned": key_pinned,
            "core_valid": core_valid,
            "all_valid": bool(core_valid and key_pinned),
        }

    def verify_hybrid_package(
        self,
        data: dict[str, Any],
        package: CryptoPackageResult,
        *,
        classical_public: bytes | None = None,
        pqc_public: bytes | None = None,
    ) -> dict[str, bool]:
        """Check a standard package signed at :attr:`SecurityLevel.HYBRID`.

        Args:
            data: The content the package was created over.
            package: A result of :meth:`create_crypto_package` carrying a
                ``hybrid_signature``.
            classical_public: The signer's Ed25519 public key, obtained out of
                band. ``None`` uses the copy inside the package.
            pqc_public: The signer's ML-DSA-65 public key, obtained out of
                band. ``None`` uses the copy inside the package.

        Returns:
            ``hash``: the content matches the package's hash. ``classical`` and
            ``pqc``: each half of the signature verifies. ``key_pinned``: both
            keys came from the caller, not from the package. ``core_valid``:
            hash and both halves hold (the package agrees with itself).
            ``all_valid``: ``core_valid`` and ``key_pinned`` -- the same
            distinction :meth:`verify_crypto_package` draws.

        A key that is malformed, or absent from both the call and the
        package, fails its half; it does not raise.

        Raises:
            ValueError: If the package carries no hybrid signature.
        """
        if package.hybrid_signature is None:
            raise ValueError("this package carries no hybrid signature to verify")
        carried = package.metadata.get("hybrid_public_keys")
        if not isinstance(carried, dict):
            carried = {}
        provider = self.hybrid_provider
        # Pinned means both halves: a caller who supplies only the post-quantum
        # key has not asked for the hybrid guarantee, and a package whose
        # classical half was stripped must not read as fully verified to them.
        key_pinned = pqc_public is not None and (
            classical_public is not None or provider.classical_provider is None
        )
        if pqc_public is None:
            pqc_public = _carried_key(carried.get("pqc"))
        if classical_public is None:
            classical_public = _carried_key(carried.get("classical"))
        data_bytes, hash_ok = _content_matches(data, package)
        signature = package.hybrid_signature
        # Each half on its own, so that a key of the wrong size fails the half
        # it belongs to. It used to raise out of the whole check.
        pqc_ok = pqc_public is not None and _holds(
            lambda: provider.pqc_provider.verify(data_bytes, signature.pqc_signature, pqc_public)
        )
        classical = provider.classical_provider
        if classical is None:
            # No Ed25519 here: there is no classical half to ask for, unless
            # the caller brought a key for one.
            classical_ok = classical_public is None
        else:
            # No classical key, supplied or carried, or no classical
            # signature: that half was not checked, and a package with both
            # stripped must not read as "both halves hold".
            classical_ok = (
                classical_public is not None
                and bool(signature.classical_signature)
                and _holds(
                    lambda: classical.verify(
                        data_bytes, signature.classical_signature, classical_public
                    )
                )
            )
        core_valid = bool(hash_ok and classical_ok and pqc_ok)
        return {
            "hash": hash_ok,
            "classical": bool(classical_ok),
            "pqc": bool(pqc_ok),
            "key_pinned": bool(key_pinned),
            "core_valid": core_valid,
            "all_valid": bool(core_valid and key_pinned),
        }

    def verify_crypto_package(
        self,
        content: bytes,
        package: CryptoPackageResult,
        expected_public_key: bytes | None = None,
    ) -> dict[str, bool]:
        """Verify a 6-layer AMA crypto package produced by :meth:`create_crypto_package`.

        Closes the emit-without-verify gap: Mercury could seal 6-layer packages
        (SHA3-256 hash, HMAC-SHA3-256, Ed25519 + ML-DSA-65 signatures, HKDF key
        independence, RFC 3161 timestamp) via AMA but exposed no way to check
        them. Delegates to AMA's ``verify_crypto_package`` against the
        ``ama_package`` carried on the result.

        **Integrity is not authenticity.** Every key needed to check a package
        travels *inside that package* — the signing public key, the HMAC key and
        the HKDF master secret. Verifying a package against its own material
        proves only that its parts agree with each other and were not corrupted;
        it does not prove who produced it, because anyone can build a package
        whose every layer verifies. This is the same distinction Mercury draws
        for the σ_Immutable corpus (tamper-evident, not authenticated) — see
        ``CAPABILITY_MATRIX.md`` §3.

        To get authenticity, pass ``expected_public_key``: a signing public key
        obtained **out of band** (pinned in config, fetched from a directory you
        trust, or established at enrolment). AMA compares it in constant time
        against the package's embedded signing key; on mismatch the signature is
        not evaluated and ``primary_signature`` is False.

        .. versionchanged:: AMA 4.0
           ``all_valid`` is False unless ``expected_public_key`` was supplied and
           matched. Under the pinned AMA 3.x an *unanchored* call returned
           ``all_valid`` True, which reported success for a check that could not
           tell the expected signer from an attacker who had built their own
           package. Callers that want the old meaning — "these parts agree with
           each other" — must read ``core_valid``, which still covers Layers 1-4.
           AMA provides no flag to restore the old aggregate, and Mercury does
           not add one: that would reintroduce the same silent default under a
           different name.

        Args:
            content: The original signed content.
            package: A :class:`CryptoPackageResult` from
                :meth:`create_crypto_package` with ``use_six_layer=True``.
            expected_public_key: Out-of-band signing public key to pin. When
                ``None`` (default) no authenticity claim is made: ``key_pinned``
                and ``all_valid`` are both False even for a wholly intact
                package, and ``core_valid`` is the meaningful result.

        Returns:
            Per-layer booleans plus two aggregates:

            * ``core_valid`` — Layers 1-4 passed (integrity / self-consistency).
            * ``all_valid`` — every executed check passed **and** the signing key
              was pinned. Requires ``expected_public_key``.

        Raises:
            ValueError: If ``package`` carries no AMA six-layer payload to verify
                (e.g. it was built with ``use_six_layer=False``).
        """
        ama_package = getattr(package, "ama_package", None)
        if ama_package is None:
            raise ValueError(
                "verify_crypto_package requires a six-layer package "
                "(create with CryptoPackageConfig(use_six_layer=True)); "
                "this result carries no AMA package to verify."
            )
        result: dict[str, bool] = _ama_verify_crypto_package(
            content, ama_package, expected_public_key
        )
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
