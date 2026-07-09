# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Secure data handling backed by AMA Cryptography.

Post-quantum data protection is a KEM-DEM construction over Mercury's sole
crypto backend, AMA Cryptography:

- **KEM:** ML-KEM-1024 (Kyber-1024, FIPS 203) encapsulates a fresh 256-bit
  shared secret to the recipient's public key.
- **DEM:** AES-256-GCM (FIPS 197 + SP 800-38D) encrypts the payload under
  that shared secret, providing confidentiality *and* authentication — any
  bit flip in the ciphertext, nonce, or tag makes decryption fail closed.

This module previously shipped a homebrew "LWE-inspired" routine with the
noise term zeroed (which makes the secret key recoverable by Gaussian
elimination) keying a repeating-XOR stream and an unkeyed digest "signature"
compared non-constant-time. That construction was cryptographically broken
and has been removed; every operation here now routes through AMA via
:class:`~omni_mercury_engine.security.crypto_api.MercuryCrypto`. There is no
fallback backend — AMA is mandatory (enforced at package import by the PQC
gate), so an unavailable backend fails loudly rather than degrading.

Reference: NIST FIPS 203 (ML-KEM), FIPS 197 / SP 800-38D (AES-GCM).
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni_mercury_engine.security.crypto_api import MercuryCrypto

# Wire-format constants for the KEM-DEM envelope produced by
# ``encrypt_hybrid``:  ct_len(4, big-endian) || kyber_ciphertext ||
# nonce(12) || tag(16) || aes_gcm_ciphertext.  Kyber-1024 ciphertexts are a
# fixed 1568 bytes, but the length prefix keeps the envelope self-describing
# and robust to backend parameter changes.
_CT_LEN_BYTES = 4
_NONCE_BYTES = 12
_GCM_TAG_BYTES = 16
_AES_KEY_BYTES = 32  # AES-256; the Kyber-1024 shared secret is exactly 32 bytes.


class QuantumResistantEncryption:
    """Post-quantum hybrid encryption: ML-KEM-1024 + AES-256-GCM (via AMA).

    ``security_level`` is retained for backward compatibility. The AMA backend
    is ML-KEM-1024 (NIST security category 5) regardless of the value passed,
    so a request for a lower level is served at the strongest parameter set —
    never weaker than asked. It is recorded for introspection only.
    """

    def __init__(self, security_level: int = 256) -> None:
        """Initialize the encryptor.

        Args:
            security_level: Advisory NIST-equivalent bit level. Informational
                only — the backend is always ML-KEM-1024 (category 5).
        """
        self.security_level = security_level
        self._crypto = self._new_crypto()
        # Per-instance MAC key for :meth:`sign_data` (native HMAC via AMA).
        import secrets

        self._mac_key = secrets.token_bytes(_AES_KEY_BYTES)

    @staticmethod
    def _new_crypto() -> MercuryCrypto:
        """Construct the AMA-backed crypto facade (fail-loud if unavailable)."""
        from omni_mercury_engine.security.crypto_api import MercuryCrypto

        return MercuryCrypto()

    def generate_keypair(self) -> tuple[bytes, bytes]:
        """Generate an ML-KEM-1024 key pair.

        Returns:
            ``(public_key, secret_key)`` as raw bytes. Encrypt to
            ``public_key`` with :meth:`encrypt_hybrid`; recover the plaintext
            with :meth:`decrypt_hybrid` and ``secret_key``.
        """
        keypair = self._crypto.generate_kem_keypair()
        return keypair.public_key, keypair.secret_key

    def encrypt_hybrid(self, data: bytes, public_key: bytes | None = None) -> bytes:
        """Encapsulate a fresh key to ``public_key`` and AES-256-GCM the data.

        Args:
            data: Plaintext bytes to protect.
            public_key: Recipient ML-KEM-1024 public key. When ``None`` an
                ephemeral key pair is generated and its public half is used;
                as with the historical API the matching secret is not
                returned, so provide a ``public_key`` whenever the ciphertext
                must be decryptable.

        Returns:
            Self-describing envelope: ``ct_len || kyber_ct || nonce || tag ||
            aes_gcm_ciphertext``.
        """
        if public_key is None:
            public_key, _ = self.generate_keypair()

        encapsulated = self._crypto.encapsulate(public_key)
        aes_key = encapsulated.shared_secret[:_AES_KEY_BYTES]
        sealed = self._crypto.encrypt(data, key=aes_key)

        kyber_ct: bytes = encapsulated.ciphertext
        header = len(kyber_ct).to_bytes(_CT_LEN_BYTES, "big")
        nonce: bytes = sealed["nonce"]
        tag: bytes = sealed["tag"]
        ciphertext: bytes = sealed["ciphertext"]
        return header + kyber_ct + nonce + tag + ciphertext

    def decrypt_hybrid(self, encrypted_data: bytes, secret_key: bytes) -> bytes:
        """Decapsulate the shared secret and AES-256-GCM-decrypt the payload.

        Args:
            encrypted_data: Envelope produced by :meth:`encrypt_hybrid`.
            secret_key: ML-KEM-1024 secret key matching the encryption public
                key.

        Returns:
            Recovered plaintext bytes.

        Raises:
            ValueError: If the envelope is truncated/malformed, or if GCM
                authentication fails (tampered ciphertext, nonce, or tag, or
                the wrong key).
        """
        offset = 0
        if len(encrypted_data) < _CT_LEN_BYTES:
            raise ValueError("Ciphertext envelope too short for KEM header")
        ct_len = int.from_bytes(encrypted_data[:_CT_LEN_BYTES], "big")
        offset += _CT_LEN_BYTES

        min_len = _CT_LEN_BYTES + ct_len + _NONCE_BYTES + _GCM_TAG_BYTES
        if len(encrypted_data) < min_len:
            raise ValueError("Ciphertext envelope truncated")

        kyber_ct = encrypted_data[offset : offset + ct_len]
        offset += ct_len
        nonce = encrypted_data[offset : offset + _NONCE_BYTES]
        offset += _NONCE_BYTES
        tag = encrypted_data[offset : offset + _GCM_TAG_BYTES]
        offset += _GCM_TAG_BYTES
        ciphertext = encrypted_data[offset:]

        shared_secret = self._crypto.decapsulate(kyber_ct, secret_key)
        aes_key = shared_secret[:_AES_KEY_BYTES]
        return self._crypto.decrypt(ciphertext, key=aes_key, nonce=nonce, tag=tag)

    def sign_data(self, data: bytes) -> bytes:
        """Compute a keyed HMAC-SHA-256 tag over ``data`` (native, via AMA).

        This is an instance-scoped message authentication tag, not a
        public-key signature. For verifiable PQC signatures use AMA ML-DSA-65
        (``pqc_backends.dilithium_sign``) via
        :class:`~omni_mercury_engine.security.crypto_api.MercuryCrypto`.
        """
        from omni_mercury_engine.security.ama_hmac import ama_hmac_sha256

        return ama_hmac_sha256(self._mac_key, data)

    def verify_signature(self, data: bytes, signature: bytes) -> bool:
        """Constant-time verification of a :meth:`sign_data` tag."""
        import hmac

        from omni_mercury_engine.security.ama_hmac import ama_hmac_sha256

        expected = ama_hmac_sha256(self._mac_key, data)
        return hmac.compare_digest(expected, signature)


class SecureDataHandler:
    """Handle sensitive data securely with post-quantum encryption options."""

    qr_encryption: QuantumResistantEncryption | None
    public_key: bytes | None
    private_key: bytes | None

    def __init__(self, enable_quantum_resistant: bool = True) -> None:
        """Initialize secure data handler.

        Args:
            enable_quantum_resistant: Provision an ML-KEM-1024 key pair for
                :meth:`encrypt_quantum_resistant` / decrypt.
        """
        self.enable_quantum_resistant = enable_quantum_resistant
        if enable_quantum_resistant:
            self.qr_encryption = QuantumResistantEncryption()
            self.public_key, self.private_key = self.qr_encryption.generate_keypair()
        else:
            self.qr_encryption = None
            self.public_key = None
            self.private_key = None

    def sanitize_input(self, data: str) -> str:
        """Sanitize user input."""
        sanitized = data.replace("<", "&lt;").replace(">", "&gt;")
        sanitized = sanitized.replace("'", "&#39;").replace('"', "&quot;")
        return sanitized

    def encode_data(self, data: str | bytes) -> str:
        """Base64 encode data."""
        if isinstance(data, str):
            data = data.encode()
        return base64.b64encode(data).decode()

    def decode_data(self, encoded: str) -> bytes:
        """Base64 decode data."""
        return base64.b64decode(encoded.encode())

    def encrypt_quantum_resistant(self, data: str | bytes) -> bytes:
        """Encrypt data with ML-KEM-1024 + AES-256-GCM.

        Args:
            data: Data to encrypt.

        Returns:
            Encrypted envelope bytes.
        """
        if not self.enable_quantum_resistant or self.qr_encryption is None:
            raise ValueError("Quantum-resistant encryption not enabled")

        if isinstance(data, str):
            data = data.encode()

        return self.qr_encryption.encrypt_hybrid(data, self.public_key)

    def decrypt_quantum_resistant(self, encrypted_data: bytes) -> bytes:
        """Decrypt an :meth:`encrypt_quantum_resistant` envelope.

        Args:
            encrypted_data: Encrypted envelope bytes.

        Returns:
            Decrypted plaintext bytes.
        """
        if not self.enable_quantum_resistant or self.qr_encryption is None:
            raise ValueError("Quantum-resistant encryption not enabled")

        assert self.private_key is not None
        return self.qr_encryption.decrypt_hybrid(encrypted_data, self.private_key)
