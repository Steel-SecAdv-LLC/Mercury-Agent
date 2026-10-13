# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cryptographic Operations Module.

Tiered, sovereign crypto for Mercury's lightweight layer. Every tier is
either repository-owned or Mercury's mandatory AMA backend — no third-party
crypto library sits on this path (pyca ``cryptography`` is benchmark-only,
enforced by ``tests/test_pyca_import_boundary.py``):

1. **rust** — the PyO3 ``mercury_crypto`` extension (repository-owned,
   fastest; built with ``cd rust_crypto && maturin develop``).
2. **ama-native** — AMA Cryptography's native C backend. AMA is mandatory
   at engine import (``_pqc_gate``), so inside a running engine this tier
   is always present and FIPS-gated.
3. **sovereign-python** — Mercury's own stdlib-only emergency
   implementations (:mod:`omni_mercury_engine.crypto.sovereign`):
   ChaCha20-Poly1305 (RFC 8439) and HKDF-SHA256 (RFC 5869), RFC-vector
   pinned. AES-GCM deliberately has no pure-Python tier (S-box timing);
   requesting it with both native tiers down fails loudly rather than
   silently switching algorithms.

The AEAD wire format is ``ciphertext || 16-byte tag`` on every tier, so
data sealed by one tier opens under any other (same algorithm).

A runtime AMA infrastructure failure (``NativeBackendUnavailableError`` /
``CryptoModuleError`` — e.g. a FIPS error-state lockout) latches this
module into the sovereign tier with a single CRITICAL log; an
authentication failure (tampered ciphertext) is never a degrade trigger
and always raises ``ValueError``.

Features:
- AEAD encryption (AES-GCM, ChaCha20-Poly1305)
- Cryptographic hashing (BLAKE3, SHA-256, SHA-3)
- Key derivation (Argon2id, HKDF-SHA256)
- Secure random generation (AMA health-tested DRBG when active)
- Constant-time operations

Usage:
    from omni_mercury_engine.crypto import encrypt, decrypt, hash_data

    # Encrypt data — returns (ciphertext, nonce); the nonce is generated
    # when not supplied and is required again for decryption
    ciphertext, nonce = encrypt(plaintext, key)

    # Decrypt data
    plaintext = decrypt(ciphertext, key, nonce)

    # Hash data
    digest = hash_data(data, algorithm='blake3')
"""

from __future__ import annotations

import logging
import secrets
from typing import TYPE_CHECKING, Any, Literal

from omni_mercury_engine._version import get_version as _get_version

logger = logging.getLogger(__name__)

from omni_mercury_engine.crypto import sovereign as _sovereign

# Tier 1: Rust bindings (repository-owned, fastest when built).
_RUST_AVAILABLE = False
_rust_crypto: Any = None

try:
    import mercury_crypto as _rust_crypto_module

    _rust_crypto = _rust_crypto_module
    _RUST_AVAILABLE = True
    logger.info("Rust crypto bindings loaded successfully")
except ImportError:
    logger.info(
        "Rust crypto bindings not built; using the AMA native tier. "
        "For maximum throughput build the Rust extension: "
        "cd rust_crypto && maturin develop"
    )

# Tier 2: AMA Cryptography's native backend. Mandatory at engine import
# (the PQC gate), so inside a running engine these imports succeed; the
# guard exists for exotic standalone imports of this module.
_AMA_AVAILABLE = False
_ama_exceptions: tuple[type[BaseException], ...] = ()
try:
    from ama_cryptography.exceptions import (
        CryptoModuleError as _AmaCryptoModuleError,
        NativeBackendUnavailableError as _AmaNativeUnavailableError,
    )
    from ama_cryptography.pqc_backends import (
        native_aes256_gcm_decrypt as _ama_aes_decrypt,
        native_aes256_gcm_encrypt as _ama_aes_encrypt,
        native_chacha20poly1305_decrypt as _ama_chacha_decrypt,
        native_chacha20poly1305_encrypt as _ama_chacha_encrypt,
        native_hkdf_sha256 as _ama_hkdf_sha256,
    )
    from ama_cryptography.secure_memory import secure_random_bytes as _ama_random_bytes

    _ama_exceptions = (_AmaNativeUnavailableError, _AmaCryptoModuleError)
    _AMA_AVAILABLE = True
except ImportError:  # pragma: no cover - engine import already requires AMA
    logger.warning("AMA Cryptography not importable; sovereign tier is primary")

# Tier 3: Mercury's own stdlib-only emergency implementations are always
# available (see crypto/sovereign.py). This flag latches True on the first
# AMA *infrastructure* failure so the module degrades once, loudly, instead
# of hammering a locked-out backend on every call.
_AMA_RUNTIME_FAILED = False


def _ama_degrade(operation: str, exc: BaseException) -> None:
    """Latch the sovereign tier after an AMA runtime infrastructure failure."""
    global _AMA_RUNTIME_FAILED
    if not _AMA_RUNTIME_FAILED:
        _AMA_RUNTIME_FAILED = True
        logger.critical(
            "AMA Cryptography failed at runtime during %s (%s: %s); this "
            "process has degraded to Mercury's sovereign emergency crypto "
            "tier (ChaCha20-Poly1305 / HKDF-SHA256, verify-only signatures). "
            "AES-GCM and signing are unavailable until AMA is restored.",
            operation,
            type(exc).__name__,
            exc,
        )


def _ama_active() -> bool:
    return _AMA_AVAILABLE and not _AMA_RUNTIME_FAILED


try:
    import hashlib

    _HASHLIB_AVAILABLE = True
except ImportError:
    _HASHLIB_AVAILABLE = False

# BLAKE3 pure-Python binding (optional). Detect once at import time rather than
# re-running the import machinery (and swallowing an ImportError) on every hash
# call -- mirrors the _RUST_AVAILABLE / _PYTHON_CRYPTO_AVAILABLE pattern above.
try:
    import blake3 as _blake3

    _BLAKE3_AVAILABLE = True
except ImportError:
    _blake3 = None
    _BLAKE3_AVAILABLE = False

# =============================================================================
# Version and Feature Detection
# =============================================================================

__version__ = _get_version()


def get_crypto_backend() -> str:
    """Return the active cryptographic backend tier.

    ``"rust"`` (repository-owned extension), ``"ama-native"`` (AMA's C
    backend — the in-engine default), or ``"sovereign-python"`` (Mercury's
    stdlib-only emergency tier; also reported after a runtime AMA
    infrastructure failure latches the degrade).
    """
    if _RUST_AVAILABLE:
        return "rust"
    if _ama_active():
        return "ama-native"
    return "sovereign-python"


def is_rust_available() -> bool:
    """Check if Rust crypto bindings are available."""
    return _RUST_AVAILABLE


# =============================================================================
# Hashing Functions
# =============================================================================

HashAlgorithm = Literal["blake3", "sha256", "sha3-256"]


def hash_data(data: bytes, algorithm: HashAlgorithm = "blake3") -> bytes:
    """Compute cryptographic hash of data.

    Args:
        data: Bytes to hash
        algorithm: Hash algorithm ('blake3', 'sha256', 'sha3-256')

    Returns:
        32-byte hash digest
    """
    if _RUST_AVAILABLE:
        if algorithm == "blake3":
            return bytes(_rust_crypto.py_blake3_hash(data))
        elif algorithm == "sha256":
            return bytes(_rust_crypto.py_sha256_hash(data))
        elif algorithm == "sha3-256":
            return bytes(_rust_crypto.py_sha3_256_hash(data))

    # Python fallback
    if _HASHLIB_AVAILABLE:
        if algorithm == "blake3":
            if _BLAKE3_AVAILABLE:
                return bytes(_blake3.blake3(data).digest())
            # Documented fallback: blake3 binding absent -> SHA-256 digest.
            return hashlib.sha256(data).digest()
        elif algorithm == "sha256":
            return hashlib.sha256(data).digest()
        elif algorithm == "sha3-256":
            return hashlib.sha3_256(data).digest()

    raise RuntimeError("No crypto backend available")


def blake3_hash(data: bytes) -> bytes:
    """Compute BLAKE3 hash of data."""
    return hash_data(data, "blake3")


def sha256_hash(data: bytes) -> bytes:
    """Compute SHA-256 hash of data."""
    return hash_data(data, "sha256")


def sha3_256_hash(data: bytes) -> bytes:
    """Compute SHA3-256 hash of data."""
    return hash_data(data, "sha3-256")


# =============================================================================
# Encryption Functions
# =============================================================================

EncryptionAlgorithm = Literal["aes-gcm", "chacha20-poly1305"]


def encrypt(
    plaintext: bytes,
    key: bytes,
    nonce: bytes | None = None,
    aad: bytes | None = None,
    algorithm: EncryptionAlgorithm = "aes-gcm",
) -> tuple[bytes, bytes]:
    """Encrypt data using AEAD cipher.

    Args:
        plaintext: Data to encrypt
        key: 32-byte encryption key
        nonce: 12-byte nonce (generated if not provided)
        aad: Additional authenticated data
        algorithm: Encryption algorithm ('aes-gcm' or 'chacha20-poly1305')

    Returns:
        Tuple of (ciphertext, nonce)
    """
    if len(key) != 32:
        raise ValueError("Key must be 32 bytes")

    if nonce is None:
        nonce = secrets.token_bytes(12)
    elif len(nonce) != 12:
        raise ValueError("Nonce must be 12 bytes")

    aad = aad or b""

    if _RUST_AVAILABLE:
        if algorithm == "aes-gcm":
            ciphertext = bytes(_rust_crypto.py_aes_gcm_encrypt(plaintext, key, nonce, aad))
        else:
            ciphertext = bytes(_rust_crypto.py_chacha_encrypt(plaintext, key, nonce, aad))
        return ciphertext, nonce

    if _ama_active():
        try:
            if algorithm == "aes-gcm":
                ct, tag = _ama_aes_encrypt(key, nonce, plaintext, aad)
            else:
                ct, tag = _ama_chacha_encrypt(key, nonce, plaintext, aad)
            return bytes(ct) + bytes(tag), nonce
        except _ama_exceptions as exc:
            _ama_degrade("encrypt", exc)

    # Sovereign emergency tier: ChaCha20-Poly1305 only — AES-GCM has no
    # safe pure-Python implementation (S-box cache timing), and silently
    # swapping algorithms would betray the caller's stored-data format.
    if algorithm == "aes-gcm":
        raise RuntimeError(
            "AES-GCM requires the Rust or AMA native backend; the sovereign "
            "emergency tier serves ChaCha20-Poly1305 only. Pass "
            "algorithm='chacha20-poly1305' or restore a native backend."
        )
    return _sovereign.chacha20poly1305_encrypt(key, nonce, plaintext, aad), nonce


def decrypt(
    ciphertext: bytes,
    key: bytes,
    nonce: bytes,
    aad: bytes | None = None,
    algorithm: EncryptionAlgorithm = "aes-gcm",
) -> bytes:
    """Decrypt data using AEAD cipher.

    Args:
        ciphertext: Encrypted data with auth tag
        key: 32-byte decryption key
        nonce: 12-byte nonce used for encryption
        aad: Additional authenticated data
        algorithm: Encryption algorithm

    Returns:
        Decrypted plaintext

    Raises:
        ValueError: If authentication fails
    """
    if len(key) != 32:
        raise ValueError("Key must be 32 bytes")
    if len(nonce) != 12:
        raise ValueError("Nonce must be 12 bytes")

    aad = aad or b""

    if _RUST_AVAILABLE:
        try:
            if algorithm == "aes-gcm":
                return bytes(_rust_crypto.py_aes_gcm_decrypt(ciphertext, key, nonce, aad))
            else:
                return bytes(_rust_crypto.py_chacha_decrypt(ciphertext, key, nonce, aad))
        except RuntimeError as e:
            raise ValueError(f"Decryption failed: {e}") from e

    if _ama_active():
        if len(ciphertext) < 16:
            raise ValueError("Decryption failed: ciphertext shorter than the 16-byte tag")
        ct, tag = ciphertext[:-16], ciphertext[-16:]
        try:
            if algorithm == "aes-gcm":
                return bytes(_ama_aes_decrypt(key, nonce, ct, tag, aad))
            return bytes(_ama_chacha_decrypt(key, nonce, ct, tag, aad))
        except _ama_exceptions as exc:
            # Infrastructure failure, not an authentication verdict: latch
            # the degrade and let the sovereign tier give the real verdict
            # below (ChaCha only).
            _ama_degrade("decrypt", exc)
        except (ValueError, RuntimeError) as e:
            # AMA reports a tag mismatch as ValueError (AES-GCM) or a bare
            # RuntimeError (ChaCha). An authentication failure must NEVER
            # trigger a backend fallback — a forgery does not get a second
            # verifier — so it normalises to the layer's ValueError contract.
            raise ValueError(f"Decryption failed: {e}") from e

    if algorithm == "aes-gcm":
        raise RuntimeError(
            "AES-GCM requires the Rust or AMA native backend; the sovereign "
            "emergency tier serves ChaCha20-Poly1305 only."
        )
    return _sovereign.chacha20poly1305_decrypt(key, nonce, ciphertext, aad)


# =============================================================================
# Key Derivation
# =============================================================================


def derive_key(
    password: bytes,
    salt: bytes,
    key_length: int = 32,
    memory_cost: int = 65536,
    time_cost: int = 3,
    parallelism: int = 4,
) -> bytes:
    """Derive a key from a password using Argon2id.

    Args:
        password: Password bytes
        salt: Salt bytes (at least 16 bytes recommended)
        key_length: Desired key length (default 32)
        memory_cost: Memory cost in KiB (default 64 MiB)
        time_cost: Number of iterations
        parallelism: Degree of parallelism

    Returns:
        Derived key bytes
    """
    if len(salt) < 8:
        raise ValueError("Salt must be at least 8 bytes")

    if _RUST_AVAILABLE:
        return bytes(
            _rust_crypto.py_argon2_derive(
                password, salt, key_length, memory_cost, time_cost, parallelism
            )
        )

    # Python fallback using argon2-cffi
    try:
        from argon2.low_level import Type, hash_secret_raw

        return bytes(
            hash_secret_raw(
                secret=password,
                salt=salt,
                time_cost=time_cost,
                memory_cost=memory_cost,
                parallelism=parallelism,
                hash_len=key_length,
                type=Type.ID,
            )
        )
    except ImportError:
        pass

    raise RuntimeError("No Argon2 backend available")


def derive_key_pair(
    master_secret: bytes,
    salt: bytes,
    info: bytes,
) -> tuple[bytes, bytes]:
    """Derive an encryption and authentication key pair from a master secret.

    Uses HKDF-SHA256 for key expansion.

    Args:
        master_secret: Master secret bytes
        salt: Salt for derivation
        info: Context-specific info

    Returns:
        Tuple of (encryption_key, auth_key), each 32 bytes
    """
    if _RUST_AVAILABLE:
        enc_key, auth_key = _rust_crypto.py_derive_key_pair(master_secret, salt, info)
        return bytes(enc_key), bytes(auth_key)

    # HKDF-SHA256 is deterministic by specification (RFC 5869), so the AMA
    # and sovereign tiers produce byte-identical keys — previously-derived
    # key material keeps working whichever tier serves the call.
    if _ama_active():
        try:
            enc_key = bytes(_ama_hkdf_sha256(master_secret, 32, salt, info + b"-enc"))
            auth_key = bytes(_ama_hkdf_sha256(master_secret, 32, salt, info + b"-auth"))
            return enc_key, auth_key
        except _ama_exceptions as exc:
            _ama_degrade("derive_key_pair", exc)

    return (
        _sovereign.hkdf_sha256(master_secret, 32, salt, info + b"-enc"),
        _sovereign.hkdf_sha256(master_secret, 32, salt, info + b"-auth"),
    )


# =============================================================================
# Random Generation
# =============================================================================


def secure_random(length: int) -> bytes:
    """Generate cryptographically secure random bytes.

    Args:
        length: Number of bytes to generate

    Returns:
        Random bytes
    """
    if _RUST_AVAILABLE:
        return bytes(_rust_crypto.py_secure_random(length))

    if _ama_active():
        try:
            # AMA's draw runs under its FIPS 140-3 health tests (continuous
            # repeated-output check, error-state gating) — strictly stronger
            # than a bare urandom read.
            return bytes(_ama_random_bytes(length))
        except _ama_exceptions as exc:
            _ama_degrade("secure_random", exc)

    return secrets.token_bytes(length)


def generate_key() -> bytes:
    """Generate a random 256-bit encryption key."""
    return secure_random(32)


def generate_nonce() -> bytes:
    """Generate a random 96-bit nonce for AEAD encryption."""
    return secure_random(12)


# =============================================================================
# Utility Functions
# =============================================================================


def constant_time_compare(a: bytes, b: bytes) -> bool:
    """Compare two byte strings in constant time.

    Args:
        a: First byte string
        b: Second byte string

    Returns:
        True if equal, False otherwise
    """
    if _RUST_AVAILABLE:
        return bool(_rust_crypto.py_constant_time_compare(a, b))

    # Python fallback using hmac.compare_digest
    import hmac

    return hmac.compare_digest(a, b)


def secure_zero(data: bytearray) -> None:
    """Securely zero out a byte buffer.

    Note: In Python, this is best-effort due to immutable strings and GC.
    Use bytearray for mutable buffers that can be zeroed.

    Args:
        data: Mutable byte buffer to zero
    """
    for i in range(len(data)):
        data[i] = 0


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "__version__",
    "blake3_hash",
    "constant_time_compare",
    "decrypt",
    "derive_key",
    "derive_key_pair",
    "encrypt",
    "generate_key",
    "generate_nonce",
    "get_crypto_backend",
    "hash_data",
    "is_rust_available",
    "secure_random",
    "secure_zero",
    "sha3_256_hash",
    "sha256_hash",
]
