// Copyright (C) 2025 Steel Security Advisors LLC
// SPDX-License-Identifier: GPL-3.0-or-later
//! Key Derivation Functions
//!
//! Provides secure key derivation primitives.

use argon2::{Argon2, Params, Version};
use ring::hkdf::{self, Salt, HKDF_SHA256};
use thiserror::Error;


/// KDF errors
#[derive(Error, Debug)]
pub enum KdfError {
    #[error("Argon2 error: {0}")]
    Argon2Error(String),

    #[error("HKDF error: {0}")]
    HkdfError(String),

    #[error("Invalid parameters")]
    InvalidParams,
}


/// Derive a key using Argon2id.
///
/// Argon2id is the recommended password hashing algorithm, combining
/// resistance to GPU attacks (from Argon2i) and side-channel attacks (from Argon2d).
///
/// Default parameters:
/// - memory_cost: 65536 KiB (64 MiB)
/// - time_cost: 3 iterations
/// - parallelism: 4 lanes
pub fn argon2_derive_key(
    password: &[u8],
    salt: &[u8],
    output_len: usize,
    memory_cost: u32,
    time_cost: u32,
    parallelism: u32,
) -> Result<Vec<u8>, KdfError> {
    let params = Params::new(memory_cost, time_cost, parallelism, Some(output_len))
        .map_err(|e| KdfError::Argon2Error(e.to_string()))?;

    let argon2 = Argon2::new(argon2::Algorithm::Argon2id, Version::V0x13, params);

    let mut output = vec![0u8; output_len];
    argon2.hash_password_into(password, salt, &mut output)
        .map_err(|e| KdfError::Argon2Error(e.to_string()))?;

    Ok(output)
}


/// Derive an encryption key pair using HKDF-SHA256.
///
/// This function derives two 32-byte keys from a master secret:
/// - An encryption key
/// - An authentication key
///
/// This separation of keys follows cryptographic best practices.
pub fn derive_key_pair(
    master_secret: &[u8],
    salt: &[u8],
    info: &[u8],
) -> Result<(Vec<u8>, Vec<u8>), KdfError> {
    // Use ring's HKDF implementation
    let salt = Salt::new(HKDF_SHA256, salt);
    let prk = salt.extract(master_secret);

    // Derive encryption key.  The info-slice array must outlive the returned
    // ``Okm``, which borrows it (ring's ``Prk::expand`` ties the Okm lifetime
    // to ``info``), so it cannot be a temporary in the call expression.
    let mut enc_key = vec![0u8; 32];
    let enc_info = [info, b"-enc"].concat();
    let enc_info_parts: [&[u8]; 1] = [&enc_info];
    let okm = prk.expand(&enc_info_parts, My32ByteKey)
        .map_err(|_| KdfError::HkdfError("HKDF expand failed".to_string()))?;
    okm.fill(&mut enc_key)
        .map_err(|_| KdfError::HkdfError("HKDF fill failed".to_string()))?;

    // Derive authentication key
    let mut auth_key = vec![0u8; 32];
    let auth_info = [info, b"-auth"].concat();
    let auth_info_parts: [&[u8]; 1] = [&auth_info];
    let okm = prk.expand(&auth_info_parts, My32ByteKey)
        .map_err(|_| KdfError::HkdfError("HKDF expand failed".to_string()))?;
    okm.fill(&mut auth_key)
        .map_err(|_| KdfError::HkdfError("HKDF fill failed".to_string()))?;

    Ok((enc_key, auth_key))
}


/// Helper type for HKDF key length specification
struct My32ByteKey;

impl hkdf::KeyType for My32ByteKey {
    fn len(&self) -> usize {
        32
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argon2_derive() {
        let password = b"password123";
        let salt = b"somesalt12345678";

        let key = argon2_derive_key(password, salt, 32, 4096, 1, 1).unwrap();
        assert_eq!(key.len(), 32);

        // Same inputs should produce same output
        let key2 = argon2_derive_key(password, salt, 32, 4096, 1, 1).unwrap();
        assert_eq!(key, key2);

        // Different password should produce different output
        let key3 = argon2_derive_key(b"different", salt, 32, 4096, 1, 1).unwrap();
        assert_ne!(key, key3);
    }

    #[test]
    fn test_derive_key_pair() {
        let master = b"master_secret_key";
        let salt = b"application_salt";
        let info = b"mercury-agent-v1";

        let (enc_key, auth_key) = derive_key_pair(master, salt, info).unwrap();

        assert_eq!(enc_key.len(), 32);
        assert_eq!(auth_key.len(), 32);

        // Keys should be different
        assert_ne!(enc_key, auth_key);
    }
}
