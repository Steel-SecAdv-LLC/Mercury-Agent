// Copyright (C) 2025 Steel Security Advisors LLC
// SPDX-License-Identifier: GPL-3.0-or-later
//! Mercury Agent Cryptographic Operations
//!
//! High-performance cryptographic primitives implemented in Rust with Python bindings.
//!
//! This module provides:
//! - AEAD encryption (AES-GCM, ChaCha20-Poly1305)
//! - Cryptographic hashing (BLAKE3, SHA-256, SHA-3)
//! - Key derivation (Argon2id)
//! - Constant-time operations for security

use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use zeroize::Zeroize;

mod hashing;
mod encryption;
mod kdf;
mod random;

use hashing::{blake3_hash, sha256_hash, sha3_256_hash};
use encryption::{aes_gcm_encrypt, aes_gcm_decrypt, chacha_encrypt, chacha_decrypt};
use kdf::{argon2_derive_key, derive_key_pair};
use random::secure_random_bytes;


/// Mercury Crypto - High-performance cryptographic operations
///
/// This module provides Rust-based cryptographic primitives for the Mercury Agent
/// framework, offering significant performance improvements over pure Python
/// implementations.
#[pymodule]
fn mercury_crypto(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Version information
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Hashing functions
    m.add_function(wrap_pyfunction!(py_blake3_hash, m)?)?;
    m.add_function(wrap_pyfunction!(py_sha256_hash, m)?)?;
    m.add_function(wrap_pyfunction!(py_sha3_256_hash, m)?)?;

    // Encryption functions
    m.add_function(wrap_pyfunction!(py_aes_gcm_encrypt, m)?)?;
    m.add_function(wrap_pyfunction!(py_aes_gcm_decrypt, m)?)?;
    m.add_function(wrap_pyfunction!(py_chacha_encrypt, m)?)?;
    m.add_function(wrap_pyfunction!(py_chacha_decrypt, m)?)?;

    // Key derivation
    m.add_function(wrap_pyfunction!(py_argon2_derive, m)?)?;
    m.add_function(wrap_pyfunction!(py_derive_key_pair, m)?)?;

    // Random generation
    m.add_function(wrap_pyfunction!(py_secure_random, m)?)?;

    // Utility functions
    m.add_function(wrap_pyfunction!(py_constant_time_compare, m)?)?;
    m.add_function(wrap_pyfunction!(py_secure_zero, m)?)?;

    Ok(())
}


// =============================================================================
// Hashing Functions
// =============================================================================

/// Compute BLAKE3 hash of data.
///
/// BLAKE3 is a modern cryptographic hash function that is faster than SHA-256
/// while providing equivalent security. Ideal for high-throughput applications.
///
/// Args:
///     data: Bytes to hash
///
/// Returns:
///     32-byte BLAKE3 hash
#[pyfunction]
fn py_blake3_hash(data: &[u8]) -> PyResult<Vec<u8>> {
    Ok(blake3_hash(data))
}


/// Compute SHA-256 hash of data.
///
/// Standard SHA-256 cryptographic hash function.
///
/// Args:
///     data: Bytes to hash
///
/// Returns:
///     32-byte SHA-256 hash
#[pyfunction]
fn py_sha256_hash(data: &[u8]) -> PyResult<Vec<u8>> {
    Ok(sha256_hash(data))
}


/// Compute SHA3-256 hash of data.
///
/// SHA-3 (Keccak) provides an alternative to SHA-2 with different internal structure.
///
/// Args:
///     data: Bytes to hash
///
/// Returns:
///     32-byte SHA3-256 hash
#[pyfunction]
fn py_sha3_256_hash(data: &[u8]) -> PyResult<Vec<u8>> {
    Ok(sha3_256_hash(data))
}


// =============================================================================
// Encryption Functions
// =============================================================================

/// Encrypt data using AES-256-GCM.
///
/// AES-GCM provides authenticated encryption, ensuring both confidentiality
/// and integrity of the data.
///
/// Args:
///     plaintext: Data to encrypt
///     key: 32-byte encryption key
///     nonce: 12-byte nonce (must be unique per encryption)
///     aad: Additional authenticated data (optional)
///
/// Returns:
///     Ciphertext with 16-byte authentication tag appended
#[pyfunction]
#[pyo3(signature = (plaintext, key, nonce, aad=None))]
fn py_aes_gcm_encrypt(
    plaintext: &[u8],
    key: &[u8],
    nonce: &[u8],
    aad: Option<&[u8]>,
) -> PyResult<Vec<u8>> {
    if key.len() != 32 {
        return Err(PyValueError::new_err("Key must be 32 bytes for AES-256"));
    }
    if nonce.len() != 12 {
        return Err(PyValueError::new_err("Nonce must be 12 bytes for AES-GCM"));
    }

    aes_gcm_encrypt(plaintext, key, nonce, aad.unwrap_or(&[]))
        .map_err(|e| PyRuntimeError::new_err(format!("Encryption failed: {}", e)))
}


/// Decrypt data using AES-256-GCM.
///
/// Args:
///     ciphertext: Encrypted data with authentication tag
///     key: 32-byte decryption key
///     nonce: 12-byte nonce used during encryption
///     aad: Additional authenticated data (must match encryption)
///
/// Returns:
///     Decrypted plaintext
///
/// Raises:
///     RuntimeError: If authentication fails (data tampered or wrong key)
#[pyfunction]
#[pyo3(signature = (ciphertext, key, nonce, aad=None))]
fn py_aes_gcm_decrypt(
    ciphertext: &[u8],
    key: &[u8],
    nonce: &[u8],
    aad: Option<&[u8]>,
) -> PyResult<Vec<u8>> {
    if key.len() != 32 {
        return Err(PyValueError::new_err("Key must be 32 bytes for AES-256"));
    }
    if nonce.len() != 12 {
        return Err(PyValueError::new_err("Nonce must be 12 bytes for AES-GCM"));
    }

    aes_gcm_decrypt(ciphertext, key, nonce, aad.unwrap_or(&[]))
        .map_err(|e| PyRuntimeError::new_err(format!("Decryption failed: {}", e)))
}


/// Encrypt data using ChaCha20-Poly1305.
///
/// ChaCha20-Poly1305 is an AEAD cipher that performs well on systems without
/// hardware AES support.
///
/// Args:
///     plaintext: Data to encrypt
///     key: 32-byte encryption key
///     nonce: 12-byte nonce
///     aad: Additional authenticated data (optional)
///
/// Returns:
///     Ciphertext with authentication tag
#[pyfunction]
#[pyo3(signature = (plaintext, key, nonce, aad=None))]
fn py_chacha_encrypt(
    plaintext: &[u8],
    key: &[u8],
    nonce: &[u8],
    aad: Option<&[u8]>,
) -> PyResult<Vec<u8>> {
    if key.len() != 32 {
        return Err(PyValueError::new_err("Key must be 32 bytes"));
    }
    if nonce.len() != 12 {
        return Err(PyValueError::new_err("Nonce must be 12 bytes"));
    }

    chacha_encrypt(plaintext, key, nonce, aad.unwrap_or(&[]))
        .map_err(|e| PyRuntimeError::new_err(format!("Encryption failed: {}", e)))
}


/// Decrypt data using ChaCha20-Poly1305.
#[pyfunction]
#[pyo3(signature = (ciphertext, key, nonce, aad=None))]
fn py_chacha_decrypt(
    ciphertext: &[u8],
    key: &[u8],
    nonce: &[u8],
    aad: Option<&[u8]>,
) -> PyResult<Vec<u8>> {
    if key.len() != 32 {
        return Err(PyValueError::new_err("Key must be 32 bytes"));
    }
    if nonce.len() != 12 {
        return Err(PyValueError::new_err("Nonce must be 12 bytes"));
    }

    chacha_decrypt(ciphertext, key, nonce, aad.unwrap_or(&[]))
        .map_err(|e| PyRuntimeError::new_err(format!("Decryption failed: {}", e)))
}


// =============================================================================
// Key Derivation Functions
// =============================================================================

/// Derive a key using Argon2id.
///
/// Argon2id is the recommended password hashing algorithm, providing
/// resistance against both GPU cracking and side-channel attacks.
///
/// Args:
///     password: Password bytes
///     salt: Salt bytes (should be at least 16 bytes)
///     output_len: Desired key length in bytes
///     memory_cost: Memory cost in KiB (default: 65536 = 64 MiB)
///     time_cost: Number of iterations (default: 3)
///     parallelism: Degree of parallelism (default: 4)
///
/// Returns:
///     Derived key bytes
#[pyfunction]
#[pyo3(signature = (password, salt, output_len=32, memory_cost=65536, time_cost=3, parallelism=4))]
fn py_argon2_derive(
    password: &[u8],
    salt: &[u8],
    output_len: usize,
    memory_cost: u32,
    time_cost: u32,
    parallelism: u32,
) -> PyResult<Vec<u8>> {
    if salt.len() < 8 {
        return Err(PyValueError::new_err("Salt must be at least 8 bytes"));
    }
    if output_len < 4 || output_len > 1024 {
        return Err(PyValueError::new_err("Output length must be 4-1024 bytes"));
    }

    argon2_derive_key(password, salt, output_len, memory_cost, time_cost, parallelism)
        .map_err(|e| PyRuntimeError::new_err(format!("Key derivation failed: {}", e)))
}


/// Derive an encryption key pair from a master secret.
///
/// Derives two keys: one for encryption, one for authentication.
/// Uses HKDF-SHA256 for key expansion.
///
/// Args:
///     master_secret: Master secret bytes
///     salt: Salt for key derivation
///     info: Context/application-specific info
///
/// Returns:
///     Tuple of (encryption_key, auth_key), each 32 bytes
#[pyfunction]
fn py_derive_key_pair(
    master_secret: &[u8],
    salt: &[u8],
    info: &[u8],
) -> PyResult<(Vec<u8>, Vec<u8>)> {
    derive_key_pair(master_secret, salt, info)
        .map_err(|e| PyRuntimeError::new_err(format!("Key derivation failed: {}", e)))
}


// =============================================================================
// Random Generation
// =============================================================================

/// Generate cryptographically secure random bytes.
///
/// Uses the operating system's cryptographically secure random number generator.
///
/// Args:
///     length: Number of random bytes to generate
///
/// Returns:
///     Random bytes
#[pyfunction]
fn py_secure_random(length: usize) -> PyResult<Vec<u8>> {
    if length > 1024 * 1024 {
        return Err(PyValueError::new_err("Maximum length is 1 MiB"));
    }

    Ok(secure_random_bytes(length))
}


// =============================================================================
// Utility Functions
// =============================================================================

/// Compare two byte slices in constant time.
///
/// This function takes the same amount of time regardless of where the first
/// difference occurs, preventing timing attacks.
///
/// Args:
///     a: First byte slice
///     b: Second byte slice
///
/// Returns:
///     True if slices are equal, False otherwise
#[pyfunction]
fn py_constant_time_compare(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    // Constant-time comparison
    let mut result: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        result |= x ^ y;
    }
    result == 0
}


/// Securely zero out a byte buffer.
///
/// Ensures the buffer is zeroed in a way that cannot be optimized out
/// by the compiler. Use this for sensitive data like keys.
///
/// Args:
///     data: Mutable byte buffer to zero
#[pyfunction]
fn py_secure_zero(mut data: Vec<u8>) -> PyResult<Vec<u8>> {
    data.zeroize();
    Ok(data)
}
