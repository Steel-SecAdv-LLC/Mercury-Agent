"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Comprehensive tests for security/encryption.py module.

Covers:
- QuantumResistantEncryption key generation
- Ava Guardian fail-fast policy (RuntimeError on insecure fallbacks)
- SecureDataHandler sanitization, encoding/decoding
- Quantum-resistant encryption enable/disable
"""

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

    def test_initialization_defaults(self):
        """Test default initialization parameters."""
        qre = QuantumResistantEncryption(use_liboqs=False)
        assert qre.security_level == 256
        assert qre.n == 256
        assert qre.q == 3329
        assert qre.seed is not None
        assert len(qre.seed) == 32

    def test_initialization_custom_level(self):
        """Test custom security level."""
        qre = QuantumResistantEncryption(security_level=128, use_liboqs=False)
        assert qre.security_level == 128
        assert qre.n == 128

    def test_generate_lattice_key(self):
        """Test lattice-based key pair generation (pure math, no crypto)."""
        qre = QuantumResistantEncryption(security_level=64, use_liboqs=False)
        public_key, private_key = qre._generate_lattice_key()

        A, b = public_key
        assert A.shape == (64, 64)
        assert b.shape == (64,)
        assert private_key.shape == (64,)

    def test_encrypt_raises_without_liboqs(self):
        """Ava Guardian fail-fast: encrypt raises RuntimeError without PQC backend."""
        qre = QuantumResistantEncryption(security_level=64, use_liboqs=False)
        public_key, _ = qre._generate_lattice_key()

        with pytest.raises(RuntimeError, match="Ava Guardian fail-fast"):
            qre.encrypt_hybrid(b"test data", public_key)

    def test_decrypt_raises_without_liboqs(self):
        """Ava Guardian fail-fast: decrypt raises RuntimeError without PQC backend."""
        qre = QuantumResistantEncryption(security_level=64, use_liboqs=False)
        _, private_key = qre._generate_lattice_key()

        with pytest.raises(RuntimeError, match="Ava Guardian fail-fast"):
            qre.decrypt_hybrid(b"fake ciphertext", private_key)

    def test_encrypt_auto_key_raises_without_liboqs(self):
        """Fail-fast when no public key provided and liboqs unavailable."""
        qre = QuantumResistantEncryption(security_level=64, use_liboqs=False)

        with pytest.raises(RuntimeError, match="Ava Guardian fail-fast"):
            qre.encrypt_hybrid(b"auto-key test", public_key=None)

    def test_sign_data_raises_without_liboqs(self):
        """Ava Guardian fail-fast: signing raises RuntimeError without PQC backend."""
        qre = QuantumResistantEncryption(use_liboqs=False)

        with pytest.raises(RuntimeError, match="Ava Guardian fail-fast"):
            qre.sign_data(b"sign this message")

    def test_verify_signature_raises_without_liboqs(self):
        """Ava Guardian fail-fast: verification raises RuntimeError without PQC backend."""
        qre = QuantumResistantEncryption(use_liboqs=False)

        with pytest.raises(RuntimeError, match="Ava Guardian fail-fast"):
            qre.verify_signature(b"data", b"\x00" * 32)

    def test_different_seeds_produce_different_keys(self):
        """Test that different instances produce different keys."""
        qre1 = QuantumResistantEncryption(security_level=64, use_liboqs=False)
        qre2 = QuantumResistantEncryption(security_level=64, use_liboqs=False)

        # Seeds should differ (random)
        assert qre1.seed != qre2.seed

    def test_lattice_key_deterministic_from_seed(self):
        """Test that same seed produces same lattice keys."""
        qre = QuantumResistantEncryption(security_level=64, use_liboqs=False)
        pk1, sk1 = qre._generate_lattice_key()
        pk2, sk2 = qre._generate_lattice_key()

        import numpy as np

        np.testing.assert_array_equal(pk1[0], pk2[0])
        np.testing.assert_array_equal(pk1[1], pk2[1])
        np.testing.assert_array_equal(sk1, sk2)


# =============================================================================
# SecureDataHandler Tests
# =============================================================================


class TestSecureDataHandler:
    """Tests for SecureDataHandler class."""

    def test_initialization_with_quantum(self):
        """Test initialization with quantum-resistant enabled."""
        handler = SecureDataHandler(enable_quantum_resistant=True)
        assert handler.enable_quantum_resistant is True
        assert handler.qr_encryption is not None
        assert handler.public_key is not None
        assert handler.private_key is not None

    def test_initialization_without_quantum(self):
        """Test initialization with quantum-resistant disabled."""
        handler = SecureDataHandler(enable_quantum_resistant=False)
        assert handler.enable_quantum_resistant is False
        assert handler.qr_encryption is None
        assert handler.public_key is None
        assert handler.private_key is None

    def test_sanitize_input_xss_prevention(self):
        """Test XSS prevention through input sanitization."""
        handler = SecureDataHandler(enable_quantum_resistant=False)

        # Test HTML tag escaping
        assert handler.sanitize_input("<script>alert('xss')</script>") == (
            "&lt;script&gt;alert(&#39;xss&#39;)&lt;/script&gt;"
        )

    def test_sanitize_input_quote_escaping(self):
        """Test quote escaping."""
        handler = SecureDataHandler(enable_quantum_resistant=False)
        assert handler.sanitize_input('He said "hello"') == "He said &quot;hello&quot;"

    def test_sanitize_input_preserves_safe_text(self):
        """Test that safe text is preserved."""
        handler = SecureDataHandler(enable_quantum_resistant=False)
        safe = "Hello, world! 123 @#$%"
        assert handler.sanitize_input(safe) == safe

    def test_encode_data_string(self):
        """Test base64 encoding of string data."""
        handler = SecureDataHandler(enable_quantum_resistant=False)
        encoded = handler.encode_data("Hello")
        assert isinstance(encoded, str)
        # Should be valid base64
        import base64

        decoded = base64.b64decode(encoded)
        assert decoded == b"Hello"

    def test_encode_data_bytes(self):
        """Test base64 encoding of bytes data."""
        handler = SecureDataHandler(enable_quantum_resistant=False)
        encoded = handler.encode_data(b"\x00\x01\x02")
        decoded = handler.decode_data(encoded)
        assert decoded == b"\x00\x01\x02"

    def test_decode_data(self):
        """Test base64 decoding."""
        handler = SecureDataHandler(enable_quantum_resistant=False)
        import base64

        original = b"test data"
        encoded = base64.b64encode(original).decode()
        decoded = handler.decode_data(encoded)
        assert decoded == original

    def test_encrypt_raises_without_liboqs(self):
        """Ava Guardian fail-fast: encrypt raises when PQC backend unavailable."""
        handler = SecureDataHandler(enable_quantum_resistant=True)
        # Without liboqs installed, this should raise RuntimeError
        if not handler.qr_encryption._oqs_available:
            with pytest.raises(RuntimeError, match="Ava Guardian fail-fast"):
                handler.encrypt_quantum_resistant("sensitive data")
        else:
            # liboqs available — encryption should work
            encrypted = handler.encrypt_quantum_resistant("sensitive data")
            assert isinstance(encrypted, bytes)

    def test_encrypt_raises_when_disabled(self):
        """Test that encryption raises when quantum-resistant is disabled."""
        handler = SecureDataHandler(enable_quantum_resistant=False)
        with pytest.raises(ValueError, match="not enabled"):
            handler.encrypt_quantum_resistant("test")

    def test_decrypt_raises_when_disabled(self):
        """Test that decryption raises when quantum-resistant is disabled."""
        handler = SecureDataHandler(enable_quantum_resistant=False)
        with pytest.raises(ValueError, match="not enabled"):
            handler.decrypt_quantum_resistant(b"test")
