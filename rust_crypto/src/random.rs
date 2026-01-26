//! Cryptographically Secure Random Number Generation
//!
//! Provides secure random number generation using the OS CSPRNG.

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;


/// Thread-safe secure random number generator.
pub struct SecureRng {
    rng: ChaCha20Rng,
}

impl SecureRng {
    /// Create a new secure RNG seeded from the OS.
    pub fn new() -> Self {
        SecureRng {
            rng: ChaCha20Rng::from_entropy(),
        }
    }

    /// Create a new secure RNG from a seed.
    ///
    /// Warning: Only use this for deterministic testing!
    pub fn from_seed(seed: [u8; 32]) -> Self {
        SecureRng {
            rng: ChaCha20Rng::from_seed(seed),
        }
    }

    /// Generate random bytes.
    pub fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.rng.fill_bytes(dest);
    }

    /// Generate a random u64.
    pub fn next_u64(&mut self) -> u64 {
        self.rng.next_u64()
    }

    /// Generate a random u32.
    pub fn next_u32(&mut self) -> u32 {
        self.rng.next_u32()
    }
}

impl Default for SecureRng {
    fn default() -> Self {
        Self::new()
    }
}


/// Generate cryptographically secure random bytes.
///
/// Uses the operating system's CSPRNG via the `getrandom` crate.
pub fn secure_random_bytes(length: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; length];
    rand::thread_rng().fill_bytes(&mut bytes);
    bytes
}


/// Generate a random 128-bit UUID v4.
pub fn random_uuid_v4() -> String {
    let mut bytes = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut bytes);

    // Set version (4) and variant (10xx) bits per RFC 4122
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;

    format!(
        "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
        u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        u16::from_be_bytes([bytes[4], bytes[5]]),
        u16::from_be_bytes([bytes[6], bytes[7]]),
        u16::from_be_bytes([bytes[8], bytes[9]]),
        u64::from_be_bytes([0, 0, bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]])
    )
}


/// Generate a random nonce for AEAD encryption.
pub fn random_nonce_12() -> [u8; 12] {
    let mut nonce = [0u8; 12];
    rand::thread_rng().fill_bytes(&mut nonce);
    nonce
}


/// Generate a random 256-bit key.
pub fn random_key_32() -> [u8; 32] {
    let mut key = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut key);
    key
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secure_random_bytes() {
        let bytes1 = secure_random_bytes(32);
        let bytes2 = secure_random_bytes(32);

        assert_eq!(bytes1.len(), 32);
        assert_eq!(bytes2.len(), 32);

        // Should be different (with overwhelming probability)
        assert_ne!(bytes1, bytes2);
    }

    #[test]
    fn test_secure_rng() {
        let mut rng = SecureRng::new();

        let mut bytes1 = [0u8; 32];
        let mut bytes2 = [0u8; 32];

        rng.fill_bytes(&mut bytes1);
        rng.fill_bytes(&mut bytes2);

        assert_ne!(bytes1, bytes2);
    }

    #[test]
    fn test_deterministic_rng() {
        let seed = [0u8; 32];

        let mut rng1 = SecureRng::from_seed(seed);
        let mut rng2 = SecureRng::from_seed(seed);

        let mut bytes1 = [0u8; 32];
        let mut bytes2 = [0u8; 32];

        rng1.fill_bytes(&mut bytes1);
        rng2.fill_bytes(&mut bytes2);

        // Same seed should produce same output
        assert_eq!(bytes1, bytes2);
    }

    #[test]
    fn test_random_uuid_v4() {
        let uuid1 = random_uuid_v4();
        let uuid2 = random_uuid_v4();

        // Check format (8-4-4-4-12)
        assert_eq!(uuid1.len(), 36);
        assert!(uuid1.chars().filter(|c| *c == '-').count() == 4);

        // Should be different
        assert_ne!(uuid1, uuid2);
    }
}
