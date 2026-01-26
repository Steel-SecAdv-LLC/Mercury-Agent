//! Authenticated Encryption Functions
//!
//! Provides AEAD (Authenticated Encryption with Associated Data) implementations.

use aes_gcm::{
    aead::{Aead, KeyInit, Payload},
    Aes256Gcm, Nonce as AesNonce,
};
use chacha20poly1305::{ChaCha20Poly1305, Nonce as ChachaNonce};
use thiserror::Error;


/// Encryption errors
#[derive(Error, Debug)]
pub enum CryptoError {
    #[error("Encryption failed")]
    EncryptionFailed,

    #[error("Decryption failed: authentication failed")]
    DecryptionFailed,

    #[error("Invalid key length")]
    InvalidKeyLength,

    #[error("Invalid nonce length")]
    InvalidNonceLength,
}


/// Encrypt data using AES-256-GCM.
///
/// AES-GCM provides authenticated encryption with associated data (AEAD).
/// The ciphertext includes a 16-byte authentication tag.
pub fn aes_gcm_encrypt(
    plaintext: &[u8],
    key: &[u8],
    nonce: &[u8],
    aad: &[u8],
) -> Result<Vec<u8>, CryptoError> {
    let key_array: [u8; 32] = key.try_into()
        .map_err(|_| CryptoError::InvalidKeyLength)?;

    let cipher = Aes256Gcm::new_from_slice(&key_array)
        .map_err(|_| CryptoError::InvalidKeyLength)?;

    let nonce = AesNonce::from_slice(nonce);

    let payload = Payload {
        msg: plaintext,
        aad,
    };

    cipher.encrypt(nonce, payload)
        .map_err(|_| CryptoError::EncryptionFailed)
}


/// Decrypt data using AES-256-GCM.
///
/// Returns an error if authentication fails (wrong key, tampered data, or wrong AAD).
pub fn aes_gcm_decrypt(
    ciphertext: &[u8],
    key: &[u8],
    nonce: &[u8],
    aad: &[u8],
) -> Result<Vec<u8>, CryptoError> {
    let key_array: [u8; 32] = key.try_into()
        .map_err(|_| CryptoError::InvalidKeyLength)?;

    let cipher = Aes256Gcm::new_from_slice(&key_array)
        .map_err(|_| CryptoError::InvalidKeyLength)?;

    let nonce = AesNonce::from_slice(nonce);

    let payload = Payload {
        msg: ciphertext,
        aad,
    };

    cipher.decrypt(nonce, payload)
        .map_err(|_| CryptoError::DecryptionFailed)
}


/// Encrypt data using ChaCha20-Poly1305.
///
/// ChaCha20-Poly1305 is a high-speed AEAD cipher that performs well
/// on systems without AES hardware acceleration.
pub fn chacha_encrypt(
    plaintext: &[u8],
    key: &[u8],
    nonce: &[u8],
    aad: &[u8],
) -> Result<Vec<u8>, CryptoError> {
    let key_array: [u8; 32] = key.try_into()
        .map_err(|_| CryptoError::InvalidKeyLength)?;

    let cipher = ChaCha20Poly1305::new_from_slice(&key_array)
        .map_err(|_| CryptoError::InvalidKeyLength)?;

    let nonce = ChachaNonce::from_slice(nonce);

    let payload = Payload {
        msg: plaintext,
        aad,
    };

    cipher.encrypt(nonce, payload)
        .map_err(|_| CryptoError::EncryptionFailed)
}


/// Decrypt data using ChaCha20-Poly1305.
pub fn chacha_decrypt(
    ciphertext: &[u8],
    key: &[u8],
    nonce: &[u8],
    aad: &[u8],
) -> Result<Vec<u8>, CryptoError> {
    let key_array: [u8; 32] = key.try_into()
        .map_err(|_| CryptoError::InvalidKeyLength)?;

    let cipher = ChaCha20Poly1305::new_from_slice(&key_array)
        .map_err(|_| CryptoError::InvalidKeyLength)?;

    let nonce = ChachaNonce::from_slice(nonce);

    let payload = Payload {
        msg: ciphertext,
        aad,
    };

    cipher.decrypt(nonce, payload)
        .map_err(|_| CryptoError::DecryptionFailed)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aes_gcm_roundtrip() {
        let key = [0u8; 32];
        let nonce = [0u8; 12];
        let plaintext = b"Hello, World!";
        let aad = b"additional data";

        let ciphertext = aes_gcm_encrypt(plaintext, &key, &nonce, aad).unwrap();
        let decrypted = aes_gcm_decrypt(&ciphertext, &key, &nonce, aad).unwrap();

        assert_eq!(plaintext.as_slice(), decrypted.as_slice());
    }

    #[test]
    fn test_chacha_roundtrip() {
        let key = [0u8; 32];
        let nonce = [0u8; 12];
        let plaintext = b"Hello, World!";
        let aad = b"additional data";

        let ciphertext = chacha_encrypt(plaintext, &key, &nonce, aad).unwrap();
        let decrypted = chacha_decrypt(&ciphertext, &key, &nonce, aad).unwrap();

        assert_eq!(plaintext.as_slice(), decrypted.as_slice());
    }

    #[test]
    fn test_aes_gcm_authentication_failure() {
        let key = [0u8; 32];
        let nonce = [0u8; 12];
        let plaintext = b"Hello, World!";
        let aad = b"additional data";

        let mut ciphertext = aes_gcm_encrypt(plaintext, &key, &nonce, aad).unwrap();

        // Tamper with ciphertext
        ciphertext[0] ^= 1;

        // Decryption should fail
        assert!(aes_gcm_decrypt(&ciphertext, &key, &nonce, aad).is_err());
    }
}
