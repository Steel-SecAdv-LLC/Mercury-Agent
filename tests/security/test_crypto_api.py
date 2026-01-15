"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

Tests for the crypto_api module - cryptographic operations.
"""

from __future__ import annotations

import pytest

# Import crypto module components
try:
    from omni_mercury_engine.security.crypto_api import (
        AlgorithmType,
        CryptoPackageConfig,
        Ed25519Provider,
        KeyPair,
        MercuryCrypto,
        SecurityLevel,
        Signature,
    )

    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False


pytestmark = pytest.mark.skipif(not HAS_CRYPTO, reason="crypto_api not available")


class TestEd25519Provider:
    """Tests for Ed25519 signature provider."""

    def test_generate_keypair(self):
        """Test Ed25519 keypair generation."""
        provider = Ed25519Provider()
        keypair = provider.generate_keypair()

        assert keypair is not None
        assert keypair.public_key is not None
        assert keypair.private_key is not None
        assert len(keypair.public_key) == 32  # Ed25519 public key is 32 bytes
        assert len(keypair.private_key) == 32  # Ed25519 seed is 32 bytes

    def test_sign_and_verify_roundtrip(self):
        """Test signature creation and verification."""
        provider = Ed25519Provider()
        keypair = provider.generate_keypair()
        message = b"Hello, Mercury Agent!"

        signature = provider.sign(message, keypair.private_key)
        assert signature is not None
        assert len(signature) == 64  # Ed25519 signature is 64 bytes

        # Verification should succeed
        is_valid = provider.verify(message, signature, keypair.public_key)
        assert is_valid is True

    def test_verify_fails_with_wrong_message(self):
        """Test that verification fails with tampered message."""
        provider = Ed25519Provider()
        keypair = provider.generate_keypair()
        message = b"Original message"
        tampered = b"Tampered message"

        signature = provider.sign(message, keypair.private_key)
        is_valid = provider.verify(tampered, signature, keypair.public_key)
        assert is_valid is False

    def test_verify_fails_with_wrong_key(self):
        """Test that verification fails with wrong public key."""
        provider = Ed25519Provider()
        keypair1 = provider.generate_keypair()
        keypair2 = provider.generate_keypair()
        message = b"Test message"

        signature = provider.sign(message, keypair1.private_key)
        is_valid = provider.verify(message, signature, keypair2.public_key)
        assert is_valid is False

    def test_verify_fails_with_corrupted_signature(self):
        """Test that verification fails with corrupted signature."""
        provider = Ed25519Provider()
        keypair = provider.generate_keypair()
        message = b"Test message"

        signature = provider.sign(message, keypair.private_key)
        # Corrupt the signature
        corrupted = bytes([b ^ 0xFF for b in signature[:10]]) + signature[10:]
        is_valid = provider.verify(message, corrupted, keypair.public_key)
        assert is_valid is False


class TestMercuryCrypto:
    """Tests for the main MercuryCrypto interface."""

    def test_generate_signing_keypair_classical(self):
        """Test keypair generation with classical security level."""
        crypto = MercuryCrypto(security_level=SecurityLevel.CLASSICAL)
        keypair = crypto.generate_signing_keypair()

        assert keypair is not None
        assert keypair.public_key is not None
        assert keypair.private_key is not None

    def test_sign_and_verify_roundtrip(self):
        """Test complete sign/verify cycle through main interface."""
        crypto = MercuryCrypto(security_level=SecurityLevel.CLASSICAL)
        keypair = crypto.generate_signing_keypair()
        message = b"Mercury Agent cryptographic test"

        signature = crypto.sign(message, keypair.private_key)
        assert signature is not None

        is_valid = crypto.verify(message, signature.signature, keypair.public_key)
        assert is_valid is True

    def test_get_capabilities(self):
        """Test capability reporting."""
        crypto = MercuryCrypto()
        capabilities = crypto.get_capabilities()

        assert isinstance(capabilities, dict)
        assert "signing" in capabilities or "algorithms" in capabilities

    def test_create_crypto_package(self):
        """Test crypto package creation with hashing."""
        crypto = MercuryCrypto(security_level=SecurityLevel.CLASSICAL)
        keypair = crypto.generate_signing_keypair()
        data = b"Data to be packaged and signed"

        config = CryptoPackageConfig(
            include_timestamp=True,
            sign_data=True,
            hash_algorithm="sha3-256",
        )

        package = crypto.create_crypto_package(
            data=data,
            private_key=keypair.private_key,
            config=config,
        )

        assert package is not None
        assert package.data_hash is not None
        # Hash should be deterministic
        package2 = crypto.create_crypto_package(
            data=data,
            private_key=keypair.private_key,
            config=config,
        )
        assert package.data_hash == package2.data_hash


class TestKeyPairDataClass:
    """Tests for KeyPair data structure."""

    def test_keypair_creation(self):
        """Test KeyPair can be created with valid data."""
        keypair = KeyPair(
            public_key=b"x" * 32,
            private_key=b"y" * 32,
            algorithm=AlgorithmType.ED25519,
        )

        assert keypair.public_key == b"x" * 32
        assert keypair.private_key == b"y" * 32
        assert keypair.algorithm == AlgorithmType.ED25519


class TestSignatureDataClass:
    """Tests for Signature data structure."""

    def test_signature_creation(self):
        """Test Signature can be created."""
        sig = Signature(
            signature=b"s" * 64,
            algorithm=AlgorithmType.ED25519,
        )

        assert sig.signature == b"s" * 64
        assert sig.algorithm == AlgorithmType.ED25519


class TestSecurityLevelEnum:
    """Tests for SecurityLevel enum values."""

    def test_security_levels_exist(self):
        """Test all expected security levels are defined."""
        assert hasattr(SecurityLevel, "CLASSICAL")
        assert hasattr(SecurityLevel, "POST_QUANTUM")
        assert hasattr(SecurityLevel, "HYBRID")


class TestAlgorithmTypeEnum:
    """Tests for AlgorithmType enum values."""

    def test_algorithm_types_exist(self):
        """Test all expected algorithm types are defined."""
        assert hasattr(AlgorithmType, "ED25519")
        # Post-quantum algorithms
        assert hasattr(AlgorithmType, "ML_DSA_65") or hasattr(AlgorithmType, "DILITHIUM")
