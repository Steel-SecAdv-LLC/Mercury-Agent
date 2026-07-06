# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the crypto_api module - cryptographic operations.

AMA Cryptography v3.3.0 is a mandatory Mercury capability.  There is no
simulation mode and no AMA-less skip path; missing AMA/PQC fails at import.
"""

from __future__ import annotations

import pytest

from omni_mercury_engine.security.crypto_api import (
    AlgorithmType,
    CryptoPackageConfig,
    Ed25519Provider,
    KeyPair,
    MercuryCrypto,
    SecurityLevel,
    Signature,
)


class TestEd25519Provider:
    """Tests for Ed25519 signature provider."""

    def test_generate_keypair(self) -> None:
        """Test Ed25519 keypair generation."""
        provider = Ed25519Provider()
        keypair = provider.generate_keypair()

        assert keypair is not None
        assert keypair.public_key is not None
        assert keypair.secret_key is not None
        assert len(keypair.public_key) == 32  # Ed25519 public key is 32 bytes
        assert len(keypair.secret_key) == 32  # Ed25519 seed is 32 bytes

    def test_sign_and_verify_roundtrip(self) -> None:
        """Test signature creation and verification."""
        provider = Ed25519Provider()
        keypair = provider.generate_keypair()
        message = b"Hello, Mercury Agent!"

        signature = provider.sign(message, keypair.secret_key)
        assert signature is not None
        assert len(signature) == 64  # Ed25519 signature is 64 bytes

        # Verification should succeed
        is_valid = provider.verify(message, signature, keypair.public_key)
        assert is_valid is True

    def test_verify_fails_with_wrong_message(self) -> None:
        """Test that verification fails with tampered message."""
        provider = Ed25519Provider()
        keypair = provider.generate_keypair()
        message = b"Original message"
        tampered = b"Tampered message"

        signature = provider.sign(message, keypair.secret_key)
        is_valid = provider.verify(tampered, signature, keypair.public_key)
        assert is_valid is False

    def test_verify_fails_with_wrong_key(self) -> None:
        """Test that verification fails with wrong public key."""
        provider = Ed25519Provider()
        keypair1 = provider.generate_keypair()
        keypair2 = provider.generate_keypair()
        message = b"Test message"

        signature = provider.sign(message, keypair1.secret_key)
        is_valid = provider.verify(message, signature, keypair2.public_key)
        assert is_valid is False

    def test_verify_fails_with_corrupted_signature(self) -> None:
        """Test that verification fails with corrupted signature."""
        provider = Ed25519Provider()
        keypair = provider.generate_keypair()
        message = b"Test message"

        signature = provider.sign(message, keypair.secret_key)
        # Corrupt the signature
        corrupted = bytes([b ^ 0xFF for b in signature[:10]]) + signature[10:]
        is_valid = provider.verify(message, corrupted, keypair.public_key)
        assert is_valid is False


class TestMercuryCrypto:
    """Tests for the main MercuryCrypto interface."""

    def test_generate_signing_keypair_classical(self) -> None:
        """Test keypair generation with classical security level."""
        crypto = MercuryCrypto(security_level=SecurityLevel.CLASSICAL)
        keypair = crypto.generate_signing_keypair()

        assert keypair is not None
        assert keypair.public_key is not None
        assert keypair.secret_key is not None

    def test_sign_and_verify_roundtrip(self) -> None:
        """Test complete sign/verify cycle through main interface."""
        crypto = MercuryCrypto(security_level=SecurityLevel.CLASSICAL)
        keypair = crypto.generate_signing_keypair()
        message = b"Mercury Agent cryptographic test"

        signature = crypto.sign(message, keypair.secret_key)
        assert signature is not None

        is_valid = crypto.verify(message, signature, keypair.public_key)
        assert is_valid is True

    def test_get_capabilities(self) -> None:
        """Test capability reporting."""
        crypto = MercuryCrypto()
        capabilities = crypto.get_capabilities()

        assert isinstance(capabilities, dict)
        assert "security_level" in capabilities or "supported_algorithms" in capabilities

    def test_create_crypto_package(self) -> None:
        """Test crypto package creation with hashing."""
        crypto = MercuryCrypto(security_level=SecurityLevel.CLASSICAL)
        data = {"message": "Data to be packaged and signed", "value": 42}

        config = CryptoPackageConfig(
            include_timestamp=True,
            sign_data=True,
            hash_algorithm="sha3-256",
        )

        package = crypto.create_crypto_package(
            data=data,
            config=config,
        )

        assert package is not None
        assert package.data_hash is not None
        # Hash should be deterministic
        package2 = crypto.create_crypto_package(
            data=data,
            config=config,
        )
        assert package.data_hash == package2.data_hash


class TestKeyPairDataClass:
    """Tests for KeyPair data structure."""

    def test_keypair_creation(self) -> None:
        """Test KeyPair can be created with valid data."""
        keypair = KeyPair(
            public_key=b"x" * 32,
            secret_key=b"y" * 32,
            algorithm=AlgorithmType.ED25519,
        )

        assert keypair.public_key == b"x" * 32
        assert keypair.secret_key == b"y" * 32
        assert keypair.algorithm == AlgorithmType.ED25519


class TestSignatureDataClass:
    """Tests for Signature data structure."""

    def test_signature_creation(self) -> None:
        """Test Signature can be created."""
        sig = Signature(
            signature=b"s" * 64,
            algorithm=AlgorithmType.ED25519,
            public_key_hash="abcd1234",
        )

        assert sig.signature == b"s" * 64
        assert sig.algorithm == AlgorithmType.ED25519
        assert sig.public_key_hash == "abcd1234"


class TestSecurityLevelEnum:
    """Tests for SecurityLevel enum values."""

    def test_security_levels_exist(self) -> None:
        """Test all expected security levels are defined."""
        assert hasattr(SecurityLevel, "CLASSICAL")
        assert hasattr(SecurityLevel, "POST_QUANTUM")
        assert hasattr(SecurityLevel, "HYBRID")


class TestAlgorithmTypeEnum:
    """Tests for AlgorithmType enum values."""

    def test_algorithm_types_exist(self) -> None:
        """Test all expected algorithm types are defined."""
        assert hasattr(AlgorithmType, "ED25519")
        # Post-quantum algorithms
        assert hasattr(AlgorithmType, "ML_DSA_65") or hasattr(AlgorithmType, "DILITHIUM")


class TestPostQuantumProviders:
    """
    Tests for Post-Quantum Cryptographic Providers via crypto_api.

    Requires AMA Cryptography's native C library to be built.
    """

    def test_mldsa_keypair_generation(self) -> None:
        """Test ML-DSA-65 keypair generation."""
        from omni_mercury_engine.security.crypto_api import MLDSAProvider

        provider = MLDSAProvider()
        keypair = provider.generate_keypair()

        assert keypair is not None
        assert keypair.public_key is not None
        assert keypair.secret_key is not None
        assert keypair.algorithm == AlgorithmType.ML_DSA_65

    def test_mldsa_sign_verify(self) -> None:
        """Test ML-DSA-65 sign and verify roundtrip."""
        from omni_mercury_engine.security.crypto_api import MLDSAProvider

        provider = MLDSAProvider()
        keypair = provider.generate_keypair()
        message = b"Post-quantum secure message"

        signature = provider.sign(message, keypair.secret_key)
        is_valid = provider.verify(message, signature, keypair.public_key)
        assert is_valid is True, "ML-DSA-65 should verify signatures correctly"

        # Wrong message should fail
        is_valid_wrong = provider.verify(b"wrong", signature, keypair.public_key)
        assert is_valid_wrong is False

    def test_kyber_encapsulation(self) -> None:
        """Test Kyber key encapsulation roundtrip."""
        from omni_mercury_engine.security.crypto_api import KyberProvider

        provider = KyberProvider()
        keypair = provider.generate_keypair()

        encapsulated = provider.encapsulate(keypair.public_key)
        assert encapsulated is not None
        assert encapsulated.ciphertext is not None
        assert encapsulated.shared_secret is not None

        recovered = provider.decapsulate(encapsulated.ciphertext, keypair.secret_key)
        assert (
            recovered == encapsulated.shared_secret
        ), "Kyber shared secrets should match after encap/decap"

    def test_sphincs_plus_signatures(self) -> None:
        """Test SPHINCS+ signature roundtrip."""
        from omni_mercury_engine.security.crypto_api import SphincsProvider

        provider = SphincsProvider()
        keypair = provider.generate_keypair()
        message = b"Hash-based signature test"

        signature = provider.sign(message, keypair.secret_key)
        is_valid = provider.verify(message, signature, keypair.public_key)
        assert is_valid is True, "SPHINCS+ should verify signatures correctly"


class TestHybridCryptography:
    """
    Tests for Hybrid Classical+Post-Quantum Operations.

    Requires AMA Cryptography's native C library.
    """

    def test_hybrid_keypair_generation(self) -> None:
        """Test hybrid keypair generation includes both algorithms."""
        from omni_mercury_engine.security.crypto_api import HybridSignatureProvider

        provider = HybridSignatureProvider()
        classical_kp, pqc_kp = provider.generate_keypairs()

        assert pqc_kp is not None
        assert pqc_kp.algorithm == AlgorithmType.ML_DSA_65
        if classical_kp is not None:
            assert classical_kp.algorithm == AlgorithmType.ED25519

    def test_hybrid_signature_verification(self) -> None:
        """Test hybrid signature roundtrip."""
        from omni_mercury_engine.security.crypto_api import HybridSignatureProvider

        provider = HybridSignatureProvider()
        classical_kp, pqc_kp = provider.generate_keypairs()
        message = b"Hybrid security message"

        classical_secret = classical_kp.secret_key if classical_kp else None
        hybrid_sig = provider.sign(message, classical_secret, pqc_kp.secret_key)

        classical_public = classical_kp.public_key if classical_kp else None
        classical_valid, pqc_valid = provider.verify(
            message, hybrid_sig, classical_public, pqc_kp.public_key
        )

        if classical_kp is not None:
            assert classical_valid is True, "Classical Ed25519 should verify correctly"
        assert pqc_valid is True, "ML-DSA-65 should verify signatures correctly"

    def test_hybrid_security_level(self) -> None:
        """Test MercuryCrypto with hybrid security level."""
        crypto = MercuryCrypto(security_level=SecurityLevel.HYBRID)
        caps = crypto.get_capabilities()
        assert caps is not None


class TestCryptoPackageOperations:
    """Tests for Crypto Package Operations with Anomaly Data."""

    def test_package_with_anomaly_detection_data(self) -> None:
        """Test packaging typical anomaly detection output."""
        crypto = MercuryCrypto(security_level=SecurityLevel.CLASSICAL)
        anomaly_data = {
            "detector": "omni_mercury_engine",
            "anomaly_score": 0.87,
            "is_anomaly": True,
            "confidence": 0.92,
            "timestamp": 1704067200,
            "features": [0.1, 0.2, 0.3, 0.4, 0.5],
        }

        config = CryptoPackageConfig(
            include_timestamp=True,
            sign_data=True,
        )

        package = crypto.create_crypto_package(anomaly_data, config)
        assert package is not None
        assert package.data_hash is not None

    def test_package_deterministic_hash(self) -> None:
        """Test that hashing is deterministic."""
        crypto = MercuryCrypto()
        data = {"key": "value", "number": 42}
        config = CryptoPackageConfig(include_timestamp=False, sign_data=False)

        package1 = crypto.create_crypto_package(data, config)
        package2 = crypto.create_crypto_package(data, config)
        assert package1.data_hash == package2.data_hash

    def test_package_different_data_different_hash(self) -> None:
        """Test that different data produces different hash."""
        crypto = MercuryCrypto()
        config = CryptoPackageConfig(include_timestamp=False, sign_data=False)

        package1 = crypto.create_crypto_package({"a": 1}, config)
        package2 = crypto.create_crypto_package({"a": 2}, config)
        assert package1.data_hash != package2.data_hash


class TestCryptoEdgeCases:
    """Edge case tests for cryptographic operations."""

    def test_empty_message_signing(self) -> None:
        """Test signing empty message."""
        crypto = MercuryCrypto(security_level=SecurityLevel.CLASSICAL)
        keypair = crypto.generate_signing_keypair()

        signature = crypto.sign(b"", keypair.secret_key)
        is_valid = crypto.verify(b"", signature, keypair.public_key)
        assert is_valid is True

    def test_large_message_signing(self) -> None:
        """Test signing large message."""
        crypto = MercuryCrypto(security_level=SecurityLevel.CLASSICAL)
        keypair = crypto.generate_signing_keypair()
        large_message = b"A" * (1024 * 1024)  # 1MB

        signature = crypto.sign(large_message, keypair.secret_key)
        is_valid = crypto.verify(large_message, signature, keypair.public_key)
        assert is_valid is True

    def test_binary_data_signing(self) -> None:
        """Test signing binary data with all byte values."""
        crypto = MercuryCrypto(security_level=SecurityLevel.CLASSICAL)
        keypair = crypto.generate_signing_keypair()
        binary_message = bytes(range(256))

        signature = crypto.sign(binary_message, keypair.secret_key)
        is_valid = crypto.verify(binary_message, signature, keypair.public_key)
        assert is_valid is True

    def test_keypair_uniqueness(self) -> None:
        """Test that each keypair is unique."""
        crypto = MercuryCrypto(security_level=SecurityLevel.CLASSICAL)
        keypairs = [crypto.generate_signing_keypair() for _ in range(5)]

        public_keys = [kp.public_key for kp in keypairs]
        assert len(set(public_keys)) == len(public_keys)


@pytest.mark.security
class TestCryptoIntegration:
    """Integration tests for complete cryptographic workflows."""

    def test_key_rotation_scenario(self) -> None:
        """Test key rotation preserves security guarantees."""
        crypto = MercuryCrypto(security_level=SecurityLevel.CLASSICAL)

        # Old keypair signs data
        old_keypair = crypto.generate_signing_keypair()
        data = b"Important data to sign"
        old_sig = crypto.sign(data, old_keypair.secret_key)

        # Rotate to new keypair
        new_keypair = crypto.generate_signing_keypair()
        new_sig = crypto.sign(data, new_keypair.secret_key)

        # Verify old signature with old key still works
        assert crypto.verify(data, old_sig, old_keypair.public_key) is True

        # Verify new signature with new key works
        assert crypto.verify(data, new_sig, new_keypair.public_key) is True

        # Cross-verification fails (security property)
        assert crypto.verify(data, old_sig, new_keypair.public_key) is False
        assert crypto.verify(data, new_sig, old_keypair.public_key) is False

    def test_multiple_algorithm_support(self) -> None:
        """Test that multiple algorithms can be used together."""
        crypto = MercuryCrypto()
        caps = crypto.get_capabilities()

        # Should support at least classical algorithms
        assert caps is not None


class TestPQCRealImplementation:
    """
    Tests for real PQC cryptographic operations via AMA Cryptography.

    These tests require the native C library to be built.
    """

    def test_real_dilithium_sign_verify_succeeds(self) -> None:
        """ML-DSA-65 sign/verify roundtrip."""
        from omni_mercury_engine.security.pqc_backends import (
            dilithium_sign,
            dilithium_verify,
            generate_dilithium_keypair,
        )

        keypair = generate_dilithium_keypair()
        message = b"Test message for real PQC"

        signature = dilithium_sign(message, keypair.secret_key)
        is_valid = dilithium_verify(message, signature, keypair.public_key)

        assert is_valid is True, "ML-DSA-65 should verify signatures correctly"

    def test_real_kyber_encap_decap_matches(self) -> None:
        """Kyber encap/decap shared secret roundtrip."""
        from omni_mercury_engine.security.pqc_backends import (
            generate_kyber_keypair,
            kyber_decapsulate,
            kyber_encapsulate,
        )

        keypair = generate_kyber_keypair()
        encapsulated = kyber_encapsulate(keypair.public_key)
        recovered = kyber_decapsulate(encapsulated.ciphertext, keypair.secret_key)

        assert (
            recovered == encapsulated.shared_secret
        ), "Kyber should produce matching shared secrets"

    def test_real_sphincs_sign_verify_succeeds(self) -> None:
        """SPHINCS+ sign/verify roundtrip."""
        from omni_mercury_engine.security.pqc_backends import (
            generate_sphincs_keypair,
            sphincs_sign,
            sphincs_verify,
        )

        keypair = generate_sphincs_keypair()
        message = b"Test message for real SPHINCS+"

        signature = sphincs_sign(message, keypair.secret_key)
        is_valid = sphincs_verify(message, signature, keypair.public_key)

        assert is_valid is True, "SPHINCS+ should verify signatures correctly"
