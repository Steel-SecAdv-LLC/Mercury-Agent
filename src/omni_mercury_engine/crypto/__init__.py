"""
Mercury Agent - Cryptographic Operations Module
Copyright (C) 2025 Steel Security Advisors LLC

P3: PyO3-based high-performance cryptographic operations.

This module provides a Python interface to Rust cryptographic primitives,
offering significant performance improvements over pure Python implementations.

Features:
- AEAD encryption (AES-GCM, ChaCha20-Poly1305)
- Cryptographic hashing (BLAKE3, SHA-256, SHA-3)
- Key derivation (Argon2id)
- Secure random generation
- Constant-time operations

Usage:
    from omni_mercury_engine.crypto import encrypt, decrypt, hash_data

    # Encrypt data
    ciphertext = encrypt(plaintext, key, nonce)

    # Decrypt data
    plaintext = decrypt(ciphertext, key, nonce)

    # Hash data
    digest = hash_data(data, algorithm='blake3')

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

from __future__ import annotations

import logging
import secrets
from typing import TYPE_CHECKING, Any, Literal

logger = logging.getLogger(__name__)


# Try to import Rust bindings, fall back to pure Python
_RUST_AVAILABLE = False
_rust_crypto: Any = None

try:
    import mercury_crypto as _rust_crypto_module

    _rust_crypto = _rust_crypto_module
    _RUST_AVAILABLE = True
    logger.info("Rust crypto bindings loaded successfully")
except ImportError:
    logger.warning(
        "Rust crypto bindings not available, falling back to pure Python. "
        "For better performance, build the Rust extension: cd rust_crypto && maturin develop"
    )

# Pure Python fallback imports
try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    _PYTHON_CRYPTO_AVAILABLE = True
except ImportError:
    _PYTHON_CRYPTO_AVAILABLE = False
    logger.warning("cryptography package not available")

try:
    import hashlib

    _HASHLIB_AVAILABLE = True
except ImportError:
    _HASHLIB_AVAILABLE = False


# =============================================================================
# Version and Feature Detection
# =============================================================================

__version__ = "0.1.0"


def get_crypto_backend() -> str:
    """Return the active cryptographic backend."""
    if _RUST_AVAILABLE:
        return "rust"
    elif _PYTHON_CRYPTO_AVAILABLE:
        return "python-cryptography"
    else:
        return "hashlib-only"


def is_rust_available() -> bool:
    """Check if Rust crypto bindings are available."""
    return _RUST_AVAILABLE


# =============================================================================
# Hashing Functions
# =============================================================================

HashAlgorithm = Literal["blake3", "sha256", "sha3-256"]


def hash_data(data: bytes, algorithm: HashAlgorithm = "blake3") -> bytes:
    """
    Compute cryptographic hash of data.

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
            try:
                import blake3 as blake3_lib

                return bytes(blake3_lib.blake3(data).digest())
            except ImportError:
                # Fall back to SHA-256 if blake3 not available
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
    """
    Encrypt data using AEAD cipher.

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

    # Python fallback
    if _PYTHON_CRYPTO_AVAILABLE:
        cipher: AESGCM | ChaCha20Poly1305
        if algorithm == "aes-gcm":
            cipher = AESGCM(key)
        else:
            cipher = ChaCha20Poly1305(key)
        ciphertext = cipher.encrypt(nonce, plaintext, aad)
        return ciphertext, nonce

    raise RuntimeError("No encryption backend available")


def decrypt(
    ciphertext: bytes,
    key: bytes,
    nonce: bytes,
    aad: bytes | None = None,
    algorithm: EncryptionAlgorithm = "aes-gcm",
) -> bytes:
    """
    Decrypt data using AEAD cipher.

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

    # Python fallback
    if _PYTHON_CRYPTO_AVAILABLE:
        try:
            cipher_dec: AESGCM | ChaCha20Poly1305
            if algorithm == "aes-gcm":
                cipher_dec = AESGCM(key)
            else:
                cipher_dec = ChaCha20Poly1305(key)
            return cipher_dec.decrypt(nonce, ciphertext, aad)
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}") from e

    raise RuntimeError("No decryption backend available")


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
    """
    Derive a key from a password using Argon2id.

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
    """
    Derive an encryption and authentication key pair from a master secret.

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

    # Python fallback
    if _PYTHON_CRYPTO_AVAILABLE:
        enc_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            info=info + b"-enc",
        ).derive(master_secret)

        auth_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            info=info + b"-auth",
        ).derive(master_secret)

        return enc_key, auth_key

    raise RuntimeError("No HKDF backend available")


# =============================================================================
# Random Generation
# =============================================================================


def secure_random(length: int) -> bytes:
    """
    Generate cryptographically secure random bytes.

    Args:
        length: Number of bytes to generate

    Returns:
        Random bytes
    """
    if _RUST_AVAILABLE:
        return bytes(_rust_crypto.py_secure_random(length))

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
    """
    Compare two byte strings in constant time.

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
    """
    Securely zero out a byte buffer.

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
