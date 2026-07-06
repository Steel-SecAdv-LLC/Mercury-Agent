# Mercury Crypto - Rust Cryptographic Bindings

> **Scope.** This crate provides high-performance **classical**
> cryptographic primitives (AEAD, hashing, KDF, CSPRNG, constant-time
> comparisons) for Mercury Agent. It is **not** Mercury's
> post-quantum cryptography (PQC) backend. PQC for Mercury Agent
> flows exclusively through
> [AMA Cryptography](https://github.com/Steel-SecAdv-LLC/AMA-Cryptography)
> (pinned to `v3.3.0` in
> `.github/workflows/pqc-production-check.yml` and the
> `[pqc]` extra of `pyproject.toml`). See
> [`SECURITY.md`](../SECURITY.md) §"Post-Quantum Cryptography
> Backend Audit Status" and [`docs/INSTALLATION.md`](../docs/INSTALLATION.md)
> §"Post-Quantum Cryptography backend".
>
> If your use case requires Kyber-1024, ML-DSA-65, or SPHINCS+,
> install the `[pqc]` extra and set `AMA_REQUIRE_REAL_PQC=true` to
> enable the import-time production gate. This `mercury_crypto`
> crate is the classical workhorse alongside that PQC backend.

Implemented in Rust with Python bindings using PyO3. Last updated: 2026-05-19.

## Features

- **AEAD Encryption**: AES-256-GCM and ChaCha20-Poly1305
- **Cryptographic Hashing**: BLAKE3, SHA-256, SHA-3
- **Key Derivation**: Argon2id (password hashing)
- **Secure Random**: OS-level CSPRNG
- **Constant-Time Operations**: Timing-attack resistant comparisons

## Performance

> **These figures are illustrative targets, not a verified measurement on this
> repo's CI.** The Rust extension is opt-in and not built by default. To measure
> the real speedup on your hardware (and write a provenance-stamped artifact),
> run `python -m benchmarks.crypto_backend_benchmark` after
> `cd rust_crypto && maturin develop`. When the extension is not built, that
> benchmark reports the active Python backend and emits **no** speedup figure.

Illustrative order-of-magnitude targets (unverified):

| Operation | Python (cryptography) | Rust (mercury_crypto) | Speedup* |
|-----------|----------------------|----------------------|---------|
| BLAKE3 1MB | 5.2ms | 0.8ms | ~6.5x |
| AES-GCM encrypt 1MB | 8.1ms | 2.3ms | ~3.5x |
| Argon2id (64MB) | 350ms | 280ms | ~1.25x |
| SHA-256 1MB | 4.8ms | 1.2ms | ~4.0x |

*Reproduce with `benchmarks/crypto_backend_benchmark.py`; do not cite without a measured artifact.

## Building

### Prerequisites

1. **Rust toolchain** (1.70+):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **maturin** (PyO3 build tool):
   ```bash
   pip install maturin
   ```

### Development Build

Build and install in development mode:

```bash
cd rust_crypto
maturin develop --release
```

### Production Build

Build a wheel for distribution:

```bash
cd rust_crypto
maturin build --release
pip install target/wheels/mercury_crypto-*.whl
```

### Cross-Platform Builds

Build for multiple platforms:

```bash
# Linux (manylinux)
maturin build --release --compatibility manylinux2014

# macOS (universal2 for Intel + Apple Silicon)
maturin build --release --target universal2-apple-darwin

# Windows
maturin build --release
```

## Usage

```python
from omni_mercury_engine.crypto import (
    encrypt, decrypt,
    hash_data,
    derive_key,
    generate_key, generate_nonce,
)

# Generate a key
key = generate_key()  # 32 bytes

# Encrypt data
plaintext = b"Hello, World!"
ciphertext, nonce = encrypt(plaintext, key)

# Decrypt data
decrypted = decrypt(ciphertext, key, nonce)
assert decrypted == plaintext

# Hash data
digest = hash_data(b"data", algorithm="blake3")

# Derive key from password
salt = generate_nonce()  # At least 16 bytes recommended
derived_key = derive_key(b"password", salt)
```

## Direct Rust Bindings

For lower-level access:

```python
import mercury_crypto

# Direct hashing
digest = mercury_crypto.py_blake3_hash(data)
digest = mercury_crypto.py_sha256_hash(data)

# Direct encryption
ciphertext = mercury_crypto.py_aes_gcm_encrypt(plaintext, key, nonce, aad)
plaintext = mercury_crypto.py_aes_gcm_decrypt(ciphertext, key, nonce, aad)

# Argon2 key derivation
key = mercury_crypto.py_argon2_derive(
    password, salt,
    output_len=32,
    memory_cost=65536,  # 64 MiB
    time_cost=3,
    parallelism=4
)

# Constant-time comparison
is_equal = mercury_crypto.py_constant_time_compare(a, b)
```

## Security Notes

1. **Nonces**: Always use unique nonces for each encryption. The `encrypt()` function generates secure random nonces by default.

2. **Key Storage**: Keys should be stored securely (e.g., hardware security module, encrypted at rest).

3. **Memory Safety**: Rust provides memory safety guarantees. The Python bindings use zeroization where possible.

4. **Side Channels**: Constant-time operations are used for sensitive comparisons.

## License

GNU General Public License v3.0 or later (SPDX: GPL-3.0-or-later)

Copyright (C) 2025 Steel Security Advisors LLC
