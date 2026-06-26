// Copyright (C) 2025 Steel Security Advisors LLC
// SPDX-License-Identifier: GPL-3.0-or-later
//! Cryptographic Hashing Functions
//!
//! Provides high-performance implementations of various hash functions.

use blake3::Hasher as Blake3Hasher;
use sha2::{Sha256, Digest};
use sha3::Sha3_256;


/// Compute BLAKE3 hash of data.
///
/// BLAKE3 is extremely fast (up to 7x faster than SHA-256 on modern CPUs)
/// while maintaining security equivalent to SHA-256.
pub fn blake3_hash(data: &[u8]) -> Vec<u8> {
    let mut hasher = Blake3Hasher::new();
    hasher.update(data);
    hasher.finalize().as_bytes().to_vec()
}


/// Compute BLAKE3 hash with a key (keyed hash mode).
///
/// This provides a keyed hash function (MAC) using BLAKE3's built-in support.
pub fn blake3_keyed_hash(key: &[u8; 32], data: &[u8]) -> Vec<u8> {
    let mut hasher = Blake3Hasher::new_keyed(key);
    hasher.update(data);
    hasher.finalize().as_bytes().to_vec()
}


/// Compute SHA-256 hash of data.
pub fn sha256_hash(data: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().to_vec()
}


/// Compute SHA3-256 hash of data.
pub fn sha3_256_hash(data: &[u8]) -> Vec<u8> {
    let mut hasher = Sha3_256::new();
    hasher.update(data);
    hasher.finalize().to_vec()
}


/// Compute HMAC-SHA256.
pub fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    use ring::hmac;

    let key = hmac::Key::new(hmac::HMAC_SHA256, key);
    let tag = hmac::sign(&key, data);
    tag.as_ref().to_vec()
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blake3_hash() {
        let data = b"Hello, World!";
        let hash = blake3_hash(data);
        assert_eq!(hash.len(), 32);
    }

    #[test]
    fn test_sha256_hash() {
        let data = b"Hello, World!";
        let hash = sha256_hash(data);
        assert_eq!(hash.len(), 32);

        // Known test vector
        let expected = hex::decode(
            "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"
        ).unwrap();
        assert_eq!(hash, expected);
    }

    #[test]
    fn test_sha3_256_hash() {
        let data = b"Hello, World!";
        let hash = sha3_256_hash(data);
        assert_eq!(hash.len(), 32);
    }
}
