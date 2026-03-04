"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

from __future__ import annotations

from typing import Any

"""
Secure data handling utilities with quantum-resistant encryption support.

Quantum-resistant algorithms using NIST-standardized post-quantum primitives:
- ML-KEM-768 / Kyber (lattice-based key encapsulation, FIPS 203)
- ML-DSA-65 / Dilithium (lattice-based digital signatures, FIPS 204)
- SPHINCS+ (hash-based signatures, FIPS 205)

Ava Guardian Fail-Fast Policy:
    When liboqs is unavailable, cryptographic operations RAISE instead of
    silently falling back to insecure implementations. Install a real PQC
    backend for production use:
        pip install ava-guardian    # Primary
        pip install liboqs-python   # Secondary

Reference: NIST Post-Quantum Cryptography Standardization (2024)
https://csrc.nist.gov/projects/post-quantum-cryptography
"""

import base64
import hashlib
import secrets

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_AES_GCM_NONCE_SIZE = 12  # 96-bit nonce per NIST SP 800-38D


class QuantumResistantEncryption:
    """
    Quantum-resistant encryption backed by liboqs (via Ava Guardian or direct).

    Production path: ML-KEM-768 (Kyber) + AES-256-GCM for encryption,
    ML-DSA-65 (Dilithium) for signatures. When no real PQC backend is
    available, all cryptographic operations raise RuntimeError per the
    Ava Guardian fail-fast policy.

    Signing identity is long-lived: a keypair is generated once at init
    and reused for the lifetime of the instance. The public key serves as
    the instance's cryptographic identity. For persistence across restarts,
    export the secret key via ``export_signing_secret_key()`` and restore
    it by passing ``signing_secret_key`` to the constructor.

    KEM is ephemeral by design: each ``encrypt_hybrid`` call generates a
    fresh KEM keypair (standard KEM usage per NIST FIPS 203).

    The LWE lattice key generation is retained only for SecureDataHandler's
    deterministic test scaffolding and is never used in the encryption path.
    """

    # NIST FIPS 203/204 standardized names, with legacy fallbacks for
    # older liboqs builds that still use the CRYSTALS project names.
    _KEM_ALGORITHMS = ("ML-KEM-768", "Kyber768")
    _SIG_ALGORITHMS = ("ML-DSA-65", "Dilithium3")

    def __init__(
        self,
        security_level: int = 256,
        use_liboqs: bool = True,
        signing_secret_key: bytes | None = None,
    ) -> None:
        """
        Initialize quantum-resistant encryption.

        Args:
            security_level: Security parameter (128, 192, or 256 bits)
            use_liboqs: Attempt to use liboqs for production-grade PQC
            signing_secret_key: Optional persisted ML-DSA-65 secret key to
                restore a long-lived signing identity instead of generating
                a fresh keypair. Must have been obtained from a prior call
                to ``export_signing_secret_key()``.
        """
        self.security_level = security_level
        self.n = security_level
        self.q = 3329
        self.seed = secrets.token_bytes(32)
        self.use_liboqs = use_liboqs
        self._oqs_available = False
        self._oqs_kem = None
        self._oqs_signature = None
        self._signing_public_key: bytes | None = None

        if use_liboqs:
            try:
                import oqs

                self._oqs_available = True
                self._init_liboqs(oqs, signing_secret_key)
            except ImportError:
                self._oqs_available = False

    def _init_liboqs(self, oqs: Any, signing_secret_key: bytes | None = None) -> None:
        """
        Initialize liboqs KEM and signature schemes for production use.

        Generates a long-lived signing keypair (or restores one from
        ``signing_secret_key``). The KEM instance is initialized without
        a keypair — each encryption generates an ephemeral one.

        Args:
            oqs: The oqs module imported from liboqs
            signing_secret_key: Optional ML-DSA-65 secret key for identity
                restoration. When provided, the constructor skips keypair
                generation and uses the stored key material.
        """
        try:
            # KEM: try NIST name first, fall back to legacy
            for kem_name in self._KEM_ALGORITHMS:
                try:
                    self._oqs_kem = oqs.KeyEncapsulation(kem_name)
                    break
                except (oqs.MechanismNotEnabledError, Exception):
                    continue
            else:
                raise RuntimeError("No supported KEM algorithm available")

            # Signature: try NIST name first, fall back to legacy
            for sig_name in self._SIG_ALGORITHMS:
                try:
                    if signing_secret_key is not None:
                        self._oqs_signature = oqs.Signature(sig_name, signing_secret_key)
                    else:
                        self._oqs_signature = oqs.Signature(sig_name)
                    break
                except (oqs.MechanismNotEnabledError, Exception):
                    continue
            else:
                raise RuntimeError("No supported signature algorithm available")

            # Long-lived signing identity
            if signing_secret_key is not None:
                # Restored from persisted key — public key must be tracked
                # externally (liboqs cannot derive it from the secret key).
                self._signing_public_key = None
            else:
                assert self._oqs_signature is not None
                self._signing_public_key = self._oqs_signature.generate_keypair()

        except Exception:
            self._oqs_available = False
            self._oqs_kem = None
            self._oqs_signature = None
            self._signing_public_key = None

    @property
    def signing_public_key(self) -> bytes | None:
        """The instance's long-lived ML-DSA-65 public key, or None if unavailable."""
        return self._signing_public_key

    def export_signing_secret_key(self) -> bytes:
        """
        Export the ML-DSA-65 secret key for persistent storage.

        The caller is responsible for protecting this key material at rest
        (e.g., encrypted file, HSM, OS keyring). The corresponding public
        key must also be persisted separately — liboqs cannot derive it
        from the secret key alone.

        Returns:
            Raw secret key bytes

        Raises:
            RuntimeError: If no signing identity is available
        """
        if not self._oqs_available or self._oqs_signature is None:
            raise RuntimeError(
                "No signing identity to export (Ava Guardian fail-fast policy).\n"
                "Install one of:\n"
                "  pip install ava-guardian    # Primary (recommended)\n"
                "  pip install liboqs-python   # Secondary fallback"
            )
        return self._oqs_signature.export_secret_key()

    def _generate_lattice_key(
        self,
    ) -> tuple[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]], np.ndarray[Any, Any]]:
        """
        Generate lattice-based key pair (public, private) using LWE.

        Note: Noise term e set to zero for deterministic testing.
        Production implementation should use proper noise distribution.
        """
        seed_hash = hashlib.sha3_256(self.seed).digest()
        seed_int = int.from_bytes(seed_hash[:4], "big")
        rng = np.random.RandomState(seed_int)

        A = rng.randint(0, self.q, size=(self.n, self.n)).astype(np.int64)
        s = (rng.randint(0, 3, size=self.n) - 1).astype(np.int64)
        e = np.zeros(self.n, dtype=np.int64)

        b = np.mod(A @ s + e, self.q).astype(np.int64)

        public_key: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]] = (A, b)
        private_key: np.ndarray[Any, Any] = s

        return public_key, private_key

    def encrypt_hybrid(
        self,
        data: bytes,
        public_key: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]] | None = None,
    ) -> bytes:
        """
        Encrypt data using Kyber768 KEM + AES-256-GCM via liboqs.

        Raises RuntimeError if no PQC backend is available (Ava Guardian fail-fast).

        Args:
            data: Data to encrypt
            public_key: Unused (retained for API compatibility); liboqs manages keys internally

        Returns:
            KEM ciphertext || nonce || AES-GCM ciphertext with authentication tag
        """
        if self._oqs_available and self._oqs_kem is not None:
            return self._encrypt_with_liboqs(data)

        raise RuntimeError(
            "Encryption requires a real PQC backend (Ava Guardian fail-fast policy).\n"
            "The LWE fallback with XOR stream cipher provides no meaningful security.\n"
            "Install one of:\n"
            "  pip install ava-guardian    # Primary (recommended)\n"
            "  pip install liboqs-python   # Secondary fallback"
        )

    def decrypt_hybrid(self, encrypted_data: bytes, private_key: np.ndarray[Any, Any]) -> bytes:
        """
        Decrypt data using Kyber768 KEM decapsulation + AES-256-GCM via liboqs.

        Raises RuntimeError if no PQC backend is available (Ava Guardian fail-fast).

        Args:
            encrypted_data: KEM ciphertext || nonce || AES-GCM ciphertext with tag
            private_key: Unused (retained for API compatibility); liboqs manages keys internally

        Returns:
            Decrypted data

        Raises:
            RuntimeError: If no PQC backend is available
            cryptography.exceptions.InvalidTag: If ciphertext was tampered with
        """
        if self._oqs_available and self._oqs_kem is not None:
            return self._decrypt_with_liboqs(encrypted_data)

        raise RuntimeError(
            "Decryption requires a real PQC backend (Ava Guardian fail-fast policy).\n"
            "The LWE fallback with XOR stream cipher provides no meaningful security.\n"
            "Install one of:\n"
            "  pip install ava-guardian    # Primary (recommended)\n"
            "  pip install liboqs-python   # Secondary fallback"
        )

    def _encrypt_with_liboqs(self, data: bytes) -> bytes:
        """
        Encrypt using liboqs Kyber768 KEM + AES-256-GCM.

        The KEM-derived shared secret is hashed (SHA3-256) to produce a
        256-bit AES key. A random 96-bit nonce is generated per encryption.
        Output format: kem_ciphertext || nonce || aes_gcm_ciphertext_with_tag

        Args:
            data: Data to encrypt

        Returns:
            Encrypted data with KEM ciphertext header and AES-GCM payload
        """
        assert self._oqs_kem is not None
        public_key_bytes = self._oqs_kem.generate_keypair()

        kem_ciphertext, shared_secret = self._oqs_kem.encap_secret(public_key_bytes)

        aes_key = hashlib.sha3_256(shared_secret).digest()
        nonce = secrets.token_bytes(_AES_GCM_NONCE_SIZE)
        aesgcm = AESGCM(aes_key)
        encrypted = aesgcm.encrypt(nonce, data, None)

        return bytes(kem_ciphertext + nonce + encrypted)

    def _decrypt_with_liboqs(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt using liboqs Kyber768 KEM + AES-256-GCM.

        Parses: kem_ciphertext || nonce || aes_gcm_ciphertext_with_tag
        Decapsulates the shared secret, derives AES key, and decrypts.

        Args:
            encrypted_data: Encrypted data with KEM ciphertext header and AES-GCM payload

        Returns:
            Decrypted data

        Raises:
            cryptography.exceptions.InvalidTag: If ciphertext was tampered with
        """
        assert self._oqs_kem is not None
        kem_ct_size = self._oqs_kem.details["length_ciphertext"]
        min_size = kem_ct_size + _AES_GCM_NONCE_SIZE + 16  # +16 for AES-GCM tag
        if len(encrypted_data) < min_size:
            raise ValueError(
                f"Ciphertext too short: got {len(encrypted_data)} bytes, "
                f"minimum {min_size} bytes required"
            )
        kem_ciphertext = encrypted_data[:kem_ct_size]
        nonce = encrypted_data[kem_ct_size : kem_ct_size + _AES_GCM_NONCE_SIZE]
        aes_ciphertext = encrypted_data[kem_ct_size + _AES_GCM_NONCE_SIZE :]

        shared_secret = self._oqs_kem.decap_secret(kem_ciphertext)

        aes_key = hashlib.sha3_256(shared_secret).digest()
        aesgcm = AESGCM(aes_key)
        return aesgcm.decrypt(nonce, aes_ciphertext, None)

    def sign_data(self, data: bytes) -> bytes:
        """
        Sign data using the instance's long-lived ML-DSA-65 identity.

        The signing keypair is generated once at init (or restored from a
        persisted secret key). Every call produces a signature verifiable
        against ``signing_public_key``.

        Args:
            data: Data to sign

        Returns:
            Signature bytes

        Raises:
            RuntimeError: If no PQC backend is available
        """
        if not self._oqs_available or self._oqs_signature is None:
            raise RuntimeError(
                "Signing requires a real PQC backend (Ava Guardian fail-fast policy).\n"
                "sha3_256(data + seed) is a MAC, not a signature — anyone with the\n"
                "seed can forge it. Install one of:\n"
                "  pip install ava-guardian    # Primary (recommended)\n"
                "  pip install liboqs-python   # Secondary fallback"
            )

        return self._oqs_signature.sign(data)

    def verify_signature(
        self, data: bytes, signature: bytes, public_key: bytes | None = None
    ) -> bool:
        """
        Verify an ML-DSA-65 signature.

        Delegates to liboqs ``OQS_SIG_verify``, which performs constant-time
        verification internally.

        Args:
            data: Original data
            signature: Signature to verify
            public_key: Explicit ML-DSA-65 public key. When None, uses
                this instance's ``signing_public_key`` (for verifying
                signatures produced by the same instance).

        Returns:
            True if signature is valid

        Raises:
            RuntimeError: If no PQC backend is available
            ValueError: If no public key is available for verification
        """
        if not self._oqs_available or self._oqs_signature is None:
            raise RuntimeError(
                "Signature verification requires a real PQC backend "
                "(Ava Guardian fail-fast policy).\n"
                "Install one of:\n"
                "  pip install ava-guardian    # Primary (recommended)\n"
                "  pip install liboqs-python   # Secondary fallback"
            )

        pk = public_key if public_key is not None else self._signing_public_key
        if pk is None:
            raise ValueError(
                "No public key available for verification. Either pass one "
                "explicitly or use an instance that generated its own keypair."
            )

        return self._oqs_signature.verify(data, signature, pk)


class SecureDataHandler:
    """Handle sensitive data securely with quantum-resistant options"""

    qr_encryption: QuantumResistantEncryption | None
    public_key: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]] | None
    private_key: np.ndarray[Any, Any] | None

    def __init__(self, enable_quantum_resistant: bool = True) -> None:
        """
        Initialize secure data handler.

        Args:
            enable_quantum_resistant: Enable quantum-resistant encryption
        """
        self.enable_quantum_resistant = enable_quantum_resistant
        if enable_quantum_resistant:
            self.qr_encryption = QuantumResistantEncryption()
            self.public_key, self.private_key = self.qr_encryption._generate_lattice_key()
        else:
            self.qr_encryption = None
            self.public_key = None
            self.private_key = None

    def sanitize_input(self, data: str) -> str:
        """Sanitize user input"""
        sanitized = data.replace("<", "&lt;").replace(">", "&gt;")
        sanitized = sanitized.replace("'", "&#39;").replace('"', "&quot;")
        return sanitized

    def encode_data(self, data: str | bytes) -> str:
        """Base64 encode data"""
        if isinstance(data, str):
            data = data.encode()
        return base64.b64encode(data).decode()

    def decode_data(self, encoded: str) -> bytes:
        """Base64 decode data"""
        return base64.b64decode(encoded.encode())

    def encrypt_quantum_resistant(self, data: str | bytes) -> bytes:
        """
        Encrypt data with quantum-resistant encryption.

        Args:
            data: Data to encrypt

        Returns:
            Encrypted bytes
        """
        if not self.enable_quantum_resistant or self.qr_encryption is None:
            raise ValueError("Quantum-resistant encryption not enabled")

        if isinstance(data, str):
            data = data.encode()

        return self.qr_encryption.encrypt_hybrid(data, self.public_key)

    def decrypt_quantum_resistant(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt quantum-resistant encrypted data.

        Args:
            encrypted_data: Encrypted data

        Returns:
            Decrypted bytes
        """
        if not self.enable_quantum_resistant or self.qr_encryption is None:
            raise ValueError("Quantum-resistant encryption not enabled")

        assert self.private_key is not None
        return self.qr_encryption.decrypt_hybrid(encrypted_data, self.private_key)
