# Copyright (C) 2025 Steel Security Advisors LLC
"""Comprehensive tests for security/encryption.py module."""

from __future__ import annotations

import pytest

from omni_mercury_engine.security.encryption import (
    QuantumResistantEncryption,
    SecureDataHandler,
)

# =============================================================================
# QuantumResistantEncryption Tests
# =============================================================================


class TestQuantumResistantEncryption:
    """Tests for QuantumResistantEncryption class."""

    def test_initialization_defaults(self) -> None:
        """Test default initialization parameters."""
        qre = QuantumResistantEncryption()
        assert qre.security_level == 256
        assert qre.n == 256
        assert qre.q == 3329
        assert qre.seed is not None
        assert len(qre.seed) == 32

    def test_initialization_custom_level(self) -> None:
        """Test custom security level."""
        qre = QuantumResistantEncryption(security_level=128)
        assert qre.security_level == 128
        assert qre.n == 128

    def test_generate_lattice_key(self) -> None:
        """Test lattice-based key pair generation."""
        qre = QuantumResistantEncryption(security_level=64)
        public_key, private_key = qre._generate_lattice_key()

        A, b = public_key
        assert A.shape == (64, 64)
        assert b.shape == (64,)
        assert private_key.shape == (64,)

    def test_encrypt_decrypt_roundtrip(self) -> None:
        """Test that encryption followed by decryption recovers the plaintext."""
        qre = QuantumResistantEncryption(security_level=64)
        public_key, private_key = qre._generate_lattice_key()

        plaintext = b"Hello, quantum-resistant world!"
        ciphertext = qre.encrypt_hybrid(plaintext, public_key)

        assert ciphertext != plaintext
        assert len(ciphertext) > len(plaintext)

        decrypted = qre.decrypt_hybrid(ciphertext, private_key)
        assert decrypted == plaintext

    def test_encrypt_decrypt_empty_data(self) -> None:
        """Test encryption/decryption with empty data."""
        qre = QuantumResistantEncryption(security_level=64)
        public_key, private_key = qre._generate_lattice_key()

        plaintext = b""
        ciphertext = qre.encrypt_hybrid(plaintext, public_key)
        decrypted = qre.decrypt_hybrid(ciphertext, private_key)
        assert decrypted == plaintext

    def test_encrypt_generates_key_if_none(self) -> None:
        """Test that encryption auto-generates key if not provided."""
        qre = QuantumResistantEncryption(security_level=64)
        plaintext = b"auto-key test"

        # Should not raise
        ciphertext = qre.encrypt_hybrid(plaintext, public_key=None)
        assert isinstance(ciphertext, bytes)
        assert len(ciphertext) > 0

    def test_sign_data(self) -> None:
        """Test data signing uses SHA3-256 HMAC."""
        qre = QuantumResistantEncryption()
        data = b"sign this message"
        signature = qre.sign_data(data)

        assert isinstance(signature, bytes)
        assert len(signature) == 32  # SHA3-256 digest length

    def test_verify_signature_valid(self) -> None:
        """Test valid signature verification."""
        qre = QuantumResistantEncryption()
        data = b"verify this message"
        signature = qre.sign_data(data)

        assert qre.verify_signature(data, signature) is True

    def test_verify_signature_invalid(self) -> None:
        """Test invalid signature rejection."""
        qre = QuantumResistantEncryption()
        data = b"original message"
        signature = qre.sign_data(data)

        # Tamper with data
        assert qre.verify_signature(b"tampered message", signature) is False

    def test_verify_signature_wrong_signature(self) -> None:
        """Test wrong signature rejection."""
        qre = QuantumResistantEncryption()
        data = b"test message"
        wrong_sig = b"\x00" * 32

        assert qre.verify_signature(data, wrong_sig) is False

    def test_different_seeds_produce_different_keys(self) -> None:
        """Test that different instances produce different keys."""
        qre1 = QuantumResistantEncryption(security_level=64)
        qre2 = QuantumResistantEncryption(security_level=64)

        # Seeds should differ (random)
        assert qre1.seed != qre2.seed

    def test_encrypt_binary_data(self) -> None:
        """Test encryption of arbitrary binary data."""
        qre = QuantumResistantEncryption(security_level=64)
        public_key, private_key = qre._generate_lattice_key()

        # Binary data with all byte values
        plaintext = bytes(range(256))
        ciphertext = qre.encrypt_hybrid(plaintext, public_key)
        decrypted = qre.decrypt_hybrid(ciphertext, private_key)
        assert decrypted == plaintext


# =============================================================================
# SecureDataHandler Tests
# =============================================================================


class TestSecureDataHandler:
    """Tests for SecureDataHandler class."""

    def test_initialization_with_quantum(self) -> None:
        """Test initialization with quantum-resistant enabled."""
        handler = SecureDataHandler(enable_quantum_resistant=True)
        assert handler.enable_quantum_resistant is True
        assert handler.qr_encryption is not None
        assert handler.public_key is not None
        assert handler.private_key is not None

    def test_initialization_without_quantum(self) -> None:
        """Test initialization with quantum-resistant disabled."""
        handler = SecureDataHandler(enable_quantum_resistant=False)
        assert handler.enable_quantum_resistant is False
        assert handler.qr_encryption is None
        assert handler.public_key is None
        assert handler.private_key is None

    def test_sanitize_input_xss_prevention(self) -> None:
        """Test XSS prevention through input sanitization."""
        handler = SecureDataHandler(enable_quantum_resistant=False)

        # Test HTML tag escaping
        assert handler.sanitize_input("<script>alert('xss')</script>") == (
            "&lt;script&gt;alert(&#39;xss&#39;)&lt;/script&gt;"
        )

    def test_sanitize_input_quote_escaping(self) -> None:
        """Test quote escaping."""
        handler = SecureDataHandler(enable_quantum_resistant=False)
        assert handler.sanitize_input('He said "hello"') == "He said &quot;hello&quot;"

    def test_sanitize_input_preserves_safe_text(self) -> None:
        """Test that safe text is preserved."""
        handler = SecureDataHandler(enable_quantum_resistant=False)
        safe = "Hello, world! 123 @#$%"
        assert handler.sanitize_input(safe) == safe

    def test_encode_data_string(self) -> None:
        """Test base64 encoding of string data."""
        handler = SecureDataHandler(enable_quantum_resistant=False)
        encoded = handler.encode_data("Hello")
        assert isinstance(encoded, str)
        # Should be valid base64
        import base64

        decoded = base64.b64decode(encoded)
        assert decoded == b"Hello"

    def test_encode_data_bytes(self) -> None:
        """Test base64 encoding of bytes data."""
        handler = SecureDataHandler(enable_quantum_resistant=False)
        encoded = handler.encode_data(b"\x00\x01\x02")
        decoded = handler.decode_data(encoded)
        assert decoded == b"\x00\x01\x02"

    def test_decode_data(self) -> None:
        """Test base64 decoding."""
        handler = SecureDataHandler(enable_quantum_resistant=False)
        import base64

        original = b"test data"
        encoded = base64.b64encode(original).decode()
        decoded = handler.decode_data(encoded)
        assert decoded == original

    def test_encrypt_quantum_resistant_string(self) -> None:
        """Test quantum-resistant encryption of string."""
        handler = SecureDataHandler(enable_quantum_resistant=True)
        plaintext = "sensitive data"
        encrypted = handler.encrypt_quantum_resistant(plaintext)
        assert isinstance(encrypted, bytes)
        assert encrypted != plaintext.encode()

    def test_encrypt_quantum_resistant_bytes(self) -> None:
        """Test quantum-resistant encryption of bytes."""
        handler = SecureDataHandler(enable_quantum_resistant=True)
        plaintext = b"binary sensitive data"
        encrypted = handler.encrypt_quantum_resistant(plaintext)
        assert isinstance(encrypted, bytes)
        assert encrypted != plaintext

    def test_encrypt_decrypt_quantum_resistant_roundtrip(self) -> None:
        """Test quantum-resistant encryption/decryption round-trip."""
        handler = SecureDataHandler(enable_quantum_resistant=True)
        plaintext = b"round trip test data"
        encrypted = handler.encrypt_quantum_resistant(plaintext)
        decrypted = handler.decrypt_quantum_resistant(encrypted)
        assert decrypted == plaintext

    def test_encrypt_raises_when_disabled(self) -> None:
        """Test that encryption raises when quantum-resistant is disabled."""
        handler = SecureDataHandler(enable_quantum_resistant=False)
        with pytest.raises(ValueError, match="not enabled"):
            handler.encrypt_quantum_resistant("test")

    def test_decrypt_raises_when_disabled(self) -> None:
        """Test that decryption raises when quantum-resistant is disabled."""
        handler = SecureDataHandler(enable_quantum_resistant=False)
        with pytest.raises(ValueError, match="not enabled"):
            handler.decrypt_quantum_resistant(b"test")
