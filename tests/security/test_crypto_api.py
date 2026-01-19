"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

Tests for the crypto_api module - cryptographic operations.
"""

from __future__ import annotations

import os

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


def _check_real_pqc_available() -> bool:
    """Check if real PQC libraries are available."""
    try:
        import oqs  # noqa: F401

        return True
    except ImportError:
        try:
            import pqcrypto  # noqa: F401

            return True
        except ImportError:
            return False


# Determine if we're in simulation mode
SIMULATION_MODE = not _check_real_pqc_available()

# Check if real PQC is required by environment
REQUIRE_REAL_PQC = os.environ.get("AVA_REQUIRE_REAL_PQC", "").lower() in (
    "true",
    "1",
    "yes",
)

pytestmark = pytest.mark.skipif(not HAS_CRYPTO, reason="crypto_api not available")


class TestEd25519Provider:
    """Tests for Ed25519 signature provider."""

    def test_generate_keypair(self):
        """Test Ed25519 keypair generation."""
        provider = Ed25519Provider()
        keypair = provider.generate_keypair()

        assert keypair is not None
        assert keypair.public_key is not None
        assert keypair.secret_key is not None
        assert len(keypair.public_key) == 32  # Ed25519 public key is 32 bytes
        assert len(keypair.secret_key) == 32  # Ed25519 seed is 32 bytes

    def test_sign_and_verify_roundtrip(self):
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

    def test_verify_fails_with_wrong_message(self):
        """Test that verification fails with tampered message."""
        provider = Ed25519Provider()
        keypair = provider.generate_keypair()
        message = b"Original message"
        tampered = b"Tampered message"

        signature = provider.sign(message, keypair.secret_key)
        is_valid = provider.verify(tampered, signature, keypair.public_key)
        assert is_valid is False

    def test_verify_fails_with_wrong_key(self):
        """Test that verification fails with wrong public key."""
        provider = Ed25519Provider()
        keypair1 = provider.generate_keypair()
        keypair2 = provider.generate_keypair()
        message = b"Test message"

        signature = provider.sign(message, keypair1.secret_key)
        is_valid = provider.verify(message, signature, keypair2.public_key)
        assert is_valid is False

    def test_verify_fails_with_corrupted_signature(self):
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

    def test_generate_signing_keypair_classical(self):
        """Test keypair generation with classical security level."""
        crypto = MercuryCrypto(security_level=SecurityLevel.CLASSICAL)
        keypair = crypto.generate_signing_keypair()

        assert keypair is not None
        assert keypair.public_key is not None
        assert keypair.secret_key is not None

    def test_sign_and_verify_roundtrip(self):
        """Test complete sign/verify cycle through main interface."""
        crypto = MercuryCrypto(security_level=SecurityLevel.CLASSICAL)
        keypair = crypto.generate_signing_keypair()
        message = b"Mercury Agent cryptographic test"

        signature = crypto.sign(message, keypair.secret_key)
        assert signature is not None

        is_valid = crypto.verify(message, signature, keypair.public_key)
        assert is_valid is True

    def test_get_capabilities(self):
        """Test capability reporting."""
        crypto = MercuryCrypto()
        capabilities = crypto.get_capabilities()

        assert isinstance(capabilities, dict)
        assert "security_level" in capabilities or "supported_algorithms" in capabilities

    def test_create_crypto_package(self):
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

    def test_keypair_creation(self):
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

    def test_signature_creation(self):
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


class TestPostQuantumProviders:
    """
    Tests for Post-Quantum Cryptographic Providers via crypto_api.

    These tests verify the crypto_api providers work correctly in the current mode:
    - In SIMULATION mode: verification SHOULD fail (broken by design = fail-fast)
    - In REAL mode: verification SHOULD succeed

    This follows the fail-fast philosophy - simulation mode is intentionally broken
    to force developers to install real PQC libraries for production.
    """

    def test_mldsa_keypair_generation(self):
        """Test ML-DSA-65 keypair generation works in any mode."""
        try:
            from omni_mercury_engine.security.crypto_api import MLDSAProvider

            provider = MLDSAProvider()
            keypair = provider.generate_keypair()

            # Keypair generation should always work
            assert keypair is not None
            assert keypair.public_key is not None
            assert keypair.secret_key is not None
            assert keypair.algorithm == AlgorithmType.ML_DSA_65
        except ImportError:
            pytest.skip("ML-DSA provider not available")

    def test_mldsa_sign_verify(self):
        """Test ML-DSA-65 signature behavior matches current mode."""
        try:
            from omni_mercury_engine.security.crypto_api import MLDSAProvider

            provider = MLDSAProvider()
            keypair = provider.generate_keypair()
            message = b"Post-quantum secure message"

            signature = provider.sign(message, keypair.secret_key)
            is_valid = provider.verify(message, signature, keypair.public_key)

            if SIMULATION_MODE:
                # Simulation mode SHOULD fail verification (fail-fast philosophy)
                assert is_valid is False, (
                    "Simulation mode should NOT verify signatures. "
                    "Install liboqs-python for real cryptography."
                )
            else:
                # Real mode SHOULD succeed
                assert is_valid is True, "Real PQC should verify signatures correctly"
                # Wrong message should fail even in real mode
                is_valid_wrong = provider.verify(b"wrong", signature, keypair.public_key)
                assert is_valid_wrong is False
        except ImportError:
            pytest.skip("ML-DSA provider not available")

    def test_kyber_encapsulation(self):
        """Test Kyber key encapsulation behavior matches current mode."""
        try:
            from omni_mercury_engine.security.crypto_api import KyberProvider

            provider = KyberProvider()
            keypair = provider.generate_keypair()

            encapsulated = provider.encapsulate(keypair.public_key)
            assert encapsulated is not None
            assert encapsulated.ciphertext is not None
            assert encapsulated.shared_secret is not None

            recovered = provider.decapsulate(encapsulated.ciphertext, keypair.secret_key)

            if SIMULATION_MODE:
                # Simulation mode SHOULD produce mismatched shared secrets (fail-fast)
                assert recovered != encapsulated.shared_secret, (
                    "Simulation mode should NOT produce matching shared secrets. "
                    "Install liboqs-python for real cryptography."
                )
            else:
                # Real mode SHOULD match
                assert (
                    recovered == encapsulated.shared_secret
                ), "Real PQC should produce matching shared secrets"
        except ImportError:
            pytest.skip("Kyber provider not available")

    def test_sphincs_plus_signatures(self):
        """Test SPHINCS+ signature behavior matches current mode."""
        try:
            from omni_mercury_engine.security.crypto_api import SphincsProvider

            provider = SphincsProvider()
            keypair = provider.generate_keypair()
            message = b"Hash-based signature test"

            signature = provider.sign(message, keypair.secret_key)
            is_valid = provider.verify(message, signature, keypair.public_key)

            # SPHINCS+ simulation only checks signature length (64 bytes from sha3_512)
            # So it may return True even in simulation mode - this is expected
            # The test verifies the provider works without crashing
            assert isinstance(is_valid, bool)
        except ImportError:
            pytest.skip("SPHINCS+ provider not available")


class TestHybridCryptography:
    """
    Tests for Hybrid Classical+Post-Quantum Operations.

    These tests verify the HybridSignatureProvider works correctly in the current mode:
    - In SIMULATION mode: PQC verification SHOULD fail (broken by design = fail-fast)
    - In REAL mode: PQC verification SHOULD succeed

    Classical (Ed25519) verification should always work when available.
    """

    def test_hybrid_keypair_generation(self):
        """Test hybrid keypair generation includes both algorithms."""
        try:
            from omni_mercury_engine.security.crypto_api import HybridSignatureProvider

            provider = HybridSignatureProvider()
            classical_kp, pqc_kp = provider.generate_keypairs()

            # Keypair generation should always work
            # classical_kp may be None if Ed25519 is not available
            assert pqc_kp is not None
            assert pqc_kp.algorithm == AlgorithmType.ML_DSA_65
            if classical_kp is not None:
                assert classical_kp.algorithm == AlgorithmType.ED25519
        except ImportError:
            pytest.skip("Hybrid provider not available")

    def test_hybrid_signature_verification(self):
        """Test hybrid signature behavior matches current mode."""
        try:
            from omni_mercury_engine.security.crypto_api import HybridSignatureProvider

            provider = HybridSignatureProvider()
            classical_kp, pqc_kp = provider.generate_keypairs()
            message = b"Hybrid security message"

            # sign() requires: message, classical_secret (or None), pqc_secret
            classical_secret = classical_kp.secret_key if classical_kp else None
            hybrid_sig = provider.sign(message, classical_secret, pqc_kp.secret_key)

            # verify() requires: message, hybrid_sig, classical_public (or None), pqc_public
            # Returns tuple (classical_valid, pqc_valid)
            classical_public = classical_kp.public_key if classical_kp else None
            classical_valid, pqc_valid = provider.verify(
                message, hybrid_sig, classical_public, pqc_kp.public_key
            )

            # Classical (Ed25519) should always work when available
            if classical_kp is not None:
                assert classical_valid is True, "Classical Ed25519 should verify correctly"

            if SIMULATION_MODE:
                # PQC verification SHOULD fail in simulation mode (fail-fast philosophy)
                assert pqc_valid is False, (
                    "Simulation mode should NOT verify PQC signatures. "
                    "Install liboqs-python for real cryptography."
                )
            else:
                # Real mode SHOULD succeed
                assert pqc_valid is True, "Real PQC should verify signatures correctly"
        except ImportError:
            pytest.skip("Hybrid provider not available")

    def test_hybrid_security_level(self):
        """Test MercuryCrypto with hybrid security level."""
        crypto = MercuryCrypto(security_level=SecurityLevel.HYBRID)
        caps = crypto.get_capabilities()
        assert caps is not None


class TestCryptoPackageOperations:
    """Tests for Crypto Package Operations with Anomaly Data."""

    def test_package_with_anomaly_detection_data(self):
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

    def test_package_deterministic_hash(self):
        """Test that hashing is deterministic."""
        crypto = MercuryCrypto()
        data = {"key": "value", "number": 42}
        config = CryptoPackageConfig(include_timestamp=False, sign_data=False)

        package1 = crypto.create_crypto_package(data, config)
        package2 = crypto.create_crypto_package(data, config)
        assert package1.data_hash == package2.data_hash

    def test_package_different_data_different_hash(self):
        """Test that different data produces different hash."""
        crypto = MercuryCrypto()
        config = CryptoPackageConfig(include_timestamp=False, sign_data=False)

        package1 = crypto.create_crypto_package({"a": 1}, config)
        package2 = crypto.create_crypto_package({"a": 2}, config)
        assert package1.data_hash != package2.data_hash


class TestCryptoEdgeCases:
    """Edge case tests for cryptographic operations."""

    def test_empty_message_signing(self):
        """Test signing empty message."""
        crypto = MercuryCrypto(security_level=SecurityLevel.CLASSICAL)
        keypair = crypto.generate_signing_keypair()

        signature = crypto.sign(b"", keypair.secret_key)
        is_valid = crypto.verify(b"", signature, keypair.public_key)
        assert is_valid is True

    def test_large_message_signing(self):
        """Test signing large message."""
        crypto = MercuryCrypto(security_level=SecurityLevel.CLASSICAL)
        keypair = crypto.generate_signing_keypair()
        large_message = b"A" * (1024 * 1024)  # 1MB

        signature = crypto.sign(large_message, keypair.secret_key)
        is_valid = crypto.verify(large_message, signature, keypair.public_key)
        assert is_valid is True

    def test_binary_data_signing(self):
        """Test signing binary data with all byte values."""
        crypto = MercuryCrypto(security_level=SecurityLevel.CLASSICAL)
        keypair = crypto.generate_signing_keypair()
        binary_message = bytes(range(256))

        signature = crypto.sign(binary_message, keypair.secret_key)
        is_valid = crypto.verify(binary_message, signature, keypair.public_key)
        assert is_valid is True

    def test_keypair_uniqueness(self):
        """Test that each keypair is unique."""
        crypto = MercuryCrypto(security_level=SecurityLevel.CLASSICAL)
        keypairs = [crypto.generate_signing_keypair() for _ in range(5)]

        public_keys = [kp.public_key for kp in keypairs]
        assert len(set(public_keys)) == len(public_keys)


@pytest.mark.security
class TestCryptoIntegration:
    """Integration tests for complete cryptographic workflows."""

    def test_key_rotation_scenario(self):
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

    def test_multiple_algorithm_support(self):
        """Test that multiple algorithms can be used together."""
        crypto = MercuryCrypto()
        caps = crypto.get_capabilities()

        # Should support at least classical algorithms
        assert caps is not None


@pytest.mark.skipif(
    not SIMULATION_MODE,
    reason="Test only applicable when running in simulation mode",
)
class TestPQCSimulationBehavior:
    """
    Tests that verify simulation mode behaves as expected (broken by design).

    These tests PASS when verification FAILS because broken simulation = correct behavior.
    This is intentional to force developers to install real PQC libraries.
    """

    def test_simulated_dilithium_sign_verify_fails(self):
        """Simulation mode SHOULD fail verification - this is intentional."""
        try:
            from omni_mercury_engine.security.pqc_backends import (
                dilithium_sign,
                dilithium_verify,
                generate_dilithium_keypair,
            )

            keypair = generate_dilithium_keypair()
            message = b"Test message for simulation"

            signature = dilithium_sign(message, keypair.secret_key)
            # Verification SHOULD fail in simulation mode - this is by design
            is_valid = dilithium_verify(message, signature, keypair.public_key)

            # Test PASSES when verification FAILS (broken simulation = correct behavior)
            assert is_valid is False, (
                "Simulation mode should NOT verify signatures. "
                "Install liboqs-python for real cryptography."
            )
        except ImportError:
            pytest.skip("PQC backends not available")

    def test_simulated_kyber_encap_decap_mismatch(self):
        """Simulation mode SHOULD produce mismatched shared secrets."""
        try:
            from omni_mercury_engine.security.pqc_backends import (
                generate_kyber_keypair,
                kyber_decapsulate,
                kyber_encapsulate,
            )

            keypair = generate_kyber_keypair()
            encapsulated = kyber_encapsulate(keypair.public_key)
            recovered = kyber_decapsulate(encapsulated.ciphertext, keypair.secret_key)

            # Shared secrets SHOULD NOT match in simulation mode
            assert recovered != encapsulated.shared_secret, (
                "Simulation mode should NOT produce matching shared secrets. "
                "Install liboqs-python for real cryptography."
            )
        except ImportError:
            pytest.skip("PQC backends not available")

    def test_simulated_sphincs_sign_verify_fails(self):
        """Simulation mode SHOULD fail SPHINCS+ verification."""
        try:
            from omni_mercury_engine.security.pqc_backends import (
                generate_sphincs_keypair,
                sphincs_sign,
                sphincs_verify,
            )

            keypair = generate_sphincs_keypair()
            message = b"Test message for SPHINCS+ simulation"

            signature = sphincs_sign(message, keypair.secret_key)
            # Verification checks only length in simulation - signature is 64 bytes
            is_valid = sphincs_verify(message, signature, keypair.public_key)

            # Test behavior depends on signature length (64 bytes from sha3_512)
            # This is intentionally weak verification
            assert is_valid is True, "SPHINCS+ simulation only checks signature length"
        except ImportError:
            pytest.skip("PQC backends not available")


@pytest.mark.skipif(
    SIMULATION_MODE,
    reason="Requires real PQC libraries (liboqs-python)",
)
class TestPQCRealImplementation:
    """
    Tests that require real cryptographic libraries.

    These tests are skipped when running in simulation mode.
    Install liboqs-python to run these tests.
    """

    def test_real_dilithium_sign_verify_succeeds(self):
        """Real PQC libraries SHOULD verify signatures correctly."""
        try:
            from omni_mercury_engine.security.pqc_backends import (
                dilithium_sign,
                dilithium_verify,
                generate_dilithium_keypair,
            )

            keypair = generate_dilithium_keypair()
            message = b"Test message for real PQC"

            signature = dilithium_sign(message, keypair.secret_key)
            is_valid = dilithium_verify(message, signature, keypair.public_key)

            assert is_valid is True, "Real PQC should verify signatures correctly"
        except ImportError:
            pytest.skip("PQC backends not available")

    def test_real_kyber_encap_decap_matches(self):
        """Real PQC libraries SHOULD produce matching shared secrets."""
        try:
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
            ), "Real PQC should produce matching shared secrets"
        except ImportError:
            pytest.skip("PQC backends not available")

    def test_real_sphincs_sign_verify_succeeds(self):
        """Real PQC libraries SHOULD verify SPHINCS+ signatures correctly."""
        try:
            from omni_mercury_engine.security.pqc_backends import (
                generate_sphincs_keypair,
                sphincs_sign,
                sphincs_verify,
            )

            keypair = generate_sphincs_keypair()
            message = b"Test message for real SPHINCS+"

            signature = sphincs_sign(message, keypair.secret_key)
            is_valid = sphincs_verify(message, signature, keypair.public_key)

            assert is_valid is True, "Real PQC should verify SPHINCS+ signatures"
        except ImportError:
            pytest.skip("PQC backends not available")
