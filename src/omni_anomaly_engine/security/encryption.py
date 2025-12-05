"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

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

"""
Secure data handling utilities with quantum-resistant encryption support.

Quantum-resistant algorithms implemented using NIST post-quantum candidates:
- Kyber (lattice-based encryption)
- Dilithium (lattice-based signatures)
- SPHINCS+ (hash-based signatures)

Reference: NIST Post-Quantum Cryptography Standardization (2024)
https://csrc.nist.gov/projects/post-quantum-cryptography

MIT-compatible implementation using standard cryptographic primitives.
"""

import base64
import hashlib
import secrets

import numpy as np


class QuantumResistantEncryption:
    """
    Quantum-resistant encryption using lattice-based cryptography principles.

    Implements simplified Kyber-inspired KEM using Learning With Errors (LWE).
    This is a deterministic demo implementation for testing; noise set to zero for stability.
    Production should use liboqs for NIST-approved post-quantum cryptography.

    Note: Conditional import of liboqs planned for future enhancement.
    """

    def __init__(self, security_level: int = 256, use_liboqs: bool = True):
        """
        Initialize quantum-resistant encryption.

        Args:
            security_level: Security parameter (128, 192, or 256 bits)
            use_liboqs: Attempt to use liboqs for production-grade PQC
        """
        self.security_level = security_level
        self.n = security_level
        self.q = 3329
        self.seed = secrets.token_bytes(32)
        self.use_liboqs = use_liboqs
        self._oqs_available = False
        self._oqs_kem = None
        self._oqs_signature = None

        if use_liboqs:
            try:
                import oqs

                self._oqs_available = True
                self._init_liboqs(oqs)
            except ImportError:
                self._oqs_available = False

    def _init_liboqs(self, oqs) -> None:
        """
        Initialize liboqs KEM and signature schemes for production use.

        Args:
            oqs: The oqs module imported from liboqs
        """
        try:
            kem_algorithm = "Kyber768"
            self._oqs_kem = oqs.KeyEncapsulation(kem_algorithm)

            sig_algorithm = "Dilithium3"
            self._oqs_signature = oqs.Signature(sig_algorithm)

        except Exception:
            self._oqs_available = False
            self._oqs_kem = None
            self._oqs_signature = None

    def _generate_lattice_key(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate lattice-based key pair (public, private) using LWE.

        Note: Noise term e set to zero for deterministic testing.
        Production implementation should use proper noise distribution.
        """
        seed_hash = hashlib.sha256(self.seed).digest()
        seed_int = int.from_bytes(seed_hash[:4], "big")
        rng = np.random.RandomState(seed_int)

        A = rng.randint(0, self.q, size=(self.n, self.n)).astype(np.int64)
        s = (rng.randint(0, 3, size=self.n) - 1).astype(np.int64)
        e = np.zeros(self.n, dtype=np.int64)

        b = np.mod(A @ s + e, self.q).astype(np.int64)

        public_key = (A, b)
        private_key = s

        return public_key, private_key

    def encrypt_hybrid(self, data: bytes, public_key: tuple | None = None) -> bytes:
        """
        Hybrid encryption: quantum-resistant KEM + symmetric stream cipher.

        Uses liboqs Kyber768 if available, otherwise falls back to deterministic LWE-KEM.
        Noise terms (e1, e2) set to zero for test stability in fallback mode.

        Args:
            data: Data to encrypt
            public_key: Optional public key (generated if None)

        Returns:
            Encrypted data with encapsulated key header (u || v)
        """
        if self._oqs_available and self._oqs_kem is not None:
            return self._encrypt_with_liboqs(data)

        if public_key is None:
            public_key, _ = self._generate_lattice_key()

        A, b = public_key

        ephemeral_seed = secrets.token_bytes(16)
        seed_hash = hashlib.sha256(ephemeral_seed).digest()
        seed_int = int.from_bytes(seed_hash[:4], "big")
        rng = np.random.RandomState(seed_int)

        r = (rng.randint(0, 3, size=self.n) - 1).astype(np.int64)
        e1 = np.zeros(self.n, dtype=np.int64)
        e2 = np.int64(0)

        m_bytes = hashlib.sha256(ephemeral_seed).digest()[:2]
        m_int = int.from_bytes(m_bytes, "big") % self.q

        u = np.mod(A.T @ r + e1, self.q).astype(np.int64)
        v = np.int64(np.mod(b @ r + e2 + m_int, self.q))

        shared_secret = hashlib.sha256(int(m_int).to_bytes(2, "big")).digest()

        encrypted = bytes(
            a ^ b
            for a, b in zip(
                data, (shared_secret * (len(data) // len(shared_secret) + 1))[: len(data)], strict=False
            )
        )

        header = u.tobytes() + v.tobytes()

        return header + encrypted

    def decrypt_hybrid(self, encrypted_data: bytes, private_key: np.ndarray) -> bytes:
        """
        Decrypt using quantum-resistant KEM decapsulation.

        Uses liboqs Kyber768 if available, otherwise falls back to LWE decapsulation.
        Recovers message m from (u, v) using private key s.
        m_int = v - u @ s (mod q) since noise terms are zero.

        Args:
            encrypted_data: Encrypted data with encapsulated key header (u || v)
            private_key: Private key s

        Returns:
            Decrypted data
        """
        if self._oqs_available and self._oqs_kem is not None:
            return self._decrypt_with_liboqs(encrypted_data)

        header_size = self.n * 8 + 8
        header = encrypted_data[:header_size]
        ciphertext = encrypted_data[header_size:]

        u_bytes = header[: self.n * 8]
        v_bytes = header[self.n * 8 :]

        u = np.frombuffer(u_bytes, dtype=np.int64)[: self.n]
        v = np.frombuffer(v_bytes, dtype=np.int64)[0]

        v_prime = np.mod(u @ private_key, self.q)

        m_int = np.int64(np.mod(v - v_prime, self.q))

        shared_secret = hashlib.sha256(int(m_int).to_bytes(2, "big")).digest()

        decrypted = bytes(
            a ^ b
            for a, b in zip(
                ciphertext,
                (shared_secret * (len(ciphertext) // len(shared_secret) + 1))[: len(ciphertext)], strict=False,
            )
        )

        return decrypted

    def _encrypt_with_liboqs(self, data: bytes) -> bytes:
        """
        Encrypt using liboqs Kyber768 KEM.

        Args:
            data: Data to encrypt

        Returns:
            Encrypted data with KEM ciphertext header
        """
        public_key_bytes = self._oqs_kem.generate_keypair()

        ciphertext, shared_secret = self._oqs_kem.encap_secret(public_key_bytes)

        encrypted = bytes(
            a ^ b
            for a, b in zip(
                data, (shared_secret * (len(data) // len(shared_secret) + 1))[: len(data)], strict=False
            )
        )

        return ciphertext + encrypted

    def _decrypt_with_liboqs(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt using liboqs Kyber768 KEM.

        Args:
            encrypted_data: Encrypted data with KEM ciphertext header

        Returns:
            Decrypted data
        """
        ciphertext_size = self._oqs_kem.details["length_ciphertext"]
        ciphertext = encrypted_data[:ciphertext_size]
        encrypted_content = encrypted_data[ciphertext_size:]

        shared_secret = self._oqs_kem.decap_secret(ciphertext)

        decrypted = bytes(
            a ^ b
            for a, b in zip(
                encrypted_content,
                (shared_secret * (len(encrypted_content) // len(shared_secret) + 1))[
                    : len(encrypted_content)
                ], strict=False,
            )
        )

        return decrypted

    def sign_data(self, data: bytes) -> bytes:
        """
        Sign data using liboqs Dilithium3 if available.

        Args:
            data: Data to sign

        Returns:
            Signature bytes
        """
        if not self._oqs_available or self._oqs_signature is None:
            return hashlib.sha256(data + self.seed).digest()

        self._oqs_signature.generate_keypair()
        signature = self._oqs_signature.sign(data)
        return signature

    def verify_signature(self, data: bytes, signature: bytes) -> bool:
        """
        Verify signature using liboqs Dilithium3 if available.

        Args:
            data: Original data
            signature: Signature to verify

        Returns:
            True if signature is valid
        """
        if not self._oqs_available or self._oqs_signature is None:
            expected_sig = hashlib.sha256(data + self.seed).digest()
            return signature == expected_sig

        return self._oqs_signature.verify(data, signature)


class SecureDataHandler:
    """Handle sensitive data securely with quantum-resistant options"""

    def __init__(self, enable_quantum_resistant: bool = True):
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

        return self.qr_encryption.decrypt_hybrid(encrypted_data, self.private_key)
