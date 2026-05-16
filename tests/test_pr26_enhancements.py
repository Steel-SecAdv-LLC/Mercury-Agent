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

"""
Tests for PR #26 enhancements: DB term, N term, RNG utility, LWE encryption
"""

import numpy as np
import pytest

from omni_mercury_engine.detectors.dimensional import DimensionalAnalyzer
from omni_mercury_engine.detectors.directive import SigmaDirectiveDetector
from omni_mercury_engine.security.encryption import QuantumResistantEncryption, SecureDataHandler
from omni_mercury_engine.utils.rng import (
    DeterministicRNG,
    get_global_rng,
    reset_global_rng,
    set_global_seed,
)


class TestDeterministicRNG:
    """Test the internal RNG utility"""

    def test_rng_initialization(self):
        """Test RNG can be initialized"""
        rng = DeterministicRNG(seed=42)
        assert rng.get_seed() == 42

    def test_rng_reproducibility(self):
        """Test that same seed produces same results"""
        rng1 = DeterministicRNG(seed=123)
        data1 = rng1.randn(10, 5)

        rng2 = DeterministicRNG(seed=123)
        data2 = rng2.randn(10, 5)

        np.testing.assert_array_almost_equal(data1, data2)

    def test_rng_temporary_seed(self):
        """Test temporary seed context manager"""
        rng = DeterministicRNG(seed=42)
        original_data = rng.randn(5)

        with rng.temporary_seed(999):
            temp_data = rng.randn(5)

        after_data = rng.randn(5)

        assert not np.allclose(original_data, temp_data)
        assert not np.allclose(temp_data, after_data)

    def test_global_rng(self):
        """Test global RNG management"""
        reset_global_rng()
        set_global_seed(555)

        rng = get_global_rng()
        assert rng.get_seed() == 555

    def test_rng_methods(self):
        """Test various RNG methods"""
        rng = DeterministicRNG(seed=42)

        randn_result = rng.randn(10)
        assert randn_result.shape == (10,)

        rand_result = rng.rand(5, 3)
        assert rand_result.shape == (5, 3)
        assert np.all((rand_result >= 0) & (rand_result < 1))

        randint_result = rng.randint(0, 10, size=20)
        # randint returns int | ndarray; with size given it is always ndarray.
        assert isinstance(randint_result, np.ndarray)
        assert len(randint_result) == 20
        assert np.all((randint_result >= 0) & (randint_result < 10))


class TestDimensionalDBTerm:
    """Test DB term implementation in DimensionalAnalyzer"""

    def test_db_term_initialization(self):
        """Test DB term can be enabled"""
        detector = DimensionalAnalyzer(config={"use_db_term": True})
        assert detector.use_db_term is True

    def test_db_term_spectral_signature(self):
        """Test spectral signature computation"""
        detector = DimensionalAnalyzer(config={"use_db_term": True})

        data = np.random.randn(100, 10)
        detector.fit(data)

        assert detector.baseline_spectral_signature is not None
        assert len(detector.baseline_spectral_signature) > 0

    def test_db_term_detection(self):
        """Test DB term detection"""
        detector = DimensionalAnalyzer(config={"use_db_term": True})

        normal_data = np.random.randn(100, 10)
        detector.fit(normal_data)

        test_data = np.random.randn(10, 10)
        result = detector.detect(test_data)

        assert "db_scores" in result
        if result["db_scores"] is not None:
            assert len(result["db_scores"]) == 10

    def test_db_term_disabled(self):
        """Test DB term can be disabled"""
        detector = DimensionalAnalyzer(config={"use_db_term": False})

        data = np.random.randn(100, 10)
        detector.fit(data)

        result = detector.detect(np.random.randn(10, 10))
        assert result["db_scores"] is None


class TestNanoScaleEnhancements:
    """Test enhanced N term in SigmaDirectiveDetector"""

    def test_nano_detection_initialization(self):
        """Test nano detection can be enabled"""
        detector = SigmaDirectiveDetector(config={"use_nano_detection": True})
        assert detector.use_nano_detection is True

    def test_enhanced_nano_scores(self):
        """Test enhanced nano-scale detection returns new metrics"""
        detector = SigmaDirectiveDetector(config={"use_nano_detection": True})

        normal_data = np.random.randn(50, 10)
        detector.fit(normal_data)

        test_data = np.random.randn(5, 10)
        result = detector.detect(test_data)

        assert "nano_scores" in result
        if result["nano_scores"]:
            assert "micro_anomaly_score" in result["nano_scores"]
            assert "dimensional_micro_score" in result["nano_scores"]

    def test_micro_anomaly_detection(self):
        """Test micro-anomaly detection method"""
        detector = SigmaDirectiveDetector()

        data = np.random.randn(10, 5)
        score = detector._detect_micro_anomalies(data)

        assert isinstance(score, (float, np.floating))
        assert 0 <= score <= 1

    def test_dimensional_downsampling(self):
        """Test dimensional downsampling detection"""
        detector = SigmaDirectiveDetector()

        data = np.random.randn(20, 10)
        score = detector._dimensional_downsampling_detection(data)

        assert isinstance(score, (float, np.floating))
        assert 0 <= score <= 1


class TestLWEEncryption:
    """Test native LWE-KEM encryption (AMA Cryptography is sole PQC backend)"""

    def test_qr_encryption_initialization(self):
        """Test quantum-resistant encryption initializes"""
        qre = QuantumResistantEncryption()
        assert qre.security_level == 256

    def test_encrypt_decrypt_roundtrip(self):
        """Test encryption/decryption round-trip with LWE-KEM"""
        qre = QuantumResistantEncryption()

        public_key, private_key = qre._generate_lattice_key()

        data = b"Test data for encryption"
        encrypted = qre.encrypt_hybrid(data, public_key)
        decrypted = qre.decrypt_hybrid(encrypted, private_key)

        assert decrypted == data

    def test_sign_verify(self):
        """Test SHA3-256 signature"""
        qre = QuantumResistantEncryption()

        data = b"Data to sign"
        signature = qre.sign_data(data)

        assert qre.verify_signature(data, signature) is True
        assert qre.verify_signature(b"Wrong data", signature) is False

    def test_secure_handler_with_qr(self):
        """Test SecureDataHandler with quantum-resistant encryption"""
        handler = SecureDataHandler(enable_quantum_resistant=True)

        data = "Sensitive information"
        encrypted = handler.encrypt_quantum_resistant(data)
        decrypted = handler.decrypt_quantum_resistant(encrypted)

        assert decrypted.decode() == data


class TestIntegration:
    """Integration tests for PR #26 enhancements"""

    def test_dimensional_with_db_term_full_pipeline(self):
        """Test full pipeline with DB term"""
        rng = DeterministicRNG(seed=42)

        detector = DimensionalAnalyzer(config={"use_db_term": True, "n_components": 5})

        normal_data = rng.randn(100, 20)
        detector.fit(normal_data)

        test_data = rng.randn(10, 20)
        result = detector.detect(test_data)

        assert "is_anomaly" in result
        assert "scores" in result
        assert "db_scores" in result

    def test_directive_with_enhanced_nano(self):
        """Test directive detector with enhanced nano detection"""
        rng = DeterministicRNG(seed=123)

        detector = SigmaDirectiveDetector(
            config={
                "use_nano_detection": True,
                "use_quantum_enhanced": True,
                "use_harmonic_detection": True,
            }
        )

        normal_data = rng.randn(50, 15)
        detector.fit(normal_data)

        test_data = rng.randn(10, 15)
        result = detector.detect(test_data)

        assert "nano_scores" in result
        assert "quantum_scores" in result
        assert "harmonic_score" in result

    def test_reproducibility_with_rng(self):
        """Test that RNG provides reproducible results across components"""
        seed = 999

        rng1 = DeterministicRNG(seed=seed)
        data1 = rng1.randn(50, 10)
        detector1 = DimensionalAnalyzer(config={"use_db_term": True})
        detector1.fit(data1)

        rng2 = DeterministicRNG(seed=seed)
        data2 = rng2.randn(50, 10)
        detector2 = DimensionalAnalyzer(config={"use_db_term": True})
        detector2.fit(data2)

        np.testing.assert_array_almost_equal(data1, data2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
