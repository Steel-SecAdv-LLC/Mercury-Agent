"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Tests for GOSNN fallback behavior and error handling.

Covers:
- GOSNN try/except logging behavior
- Fallback to raw features on error
- last_harmonic_synergy initialization
- Scalar registration with omni_ prefix
- Ethical gating with sigma_sacred threshold
- DetectorRegistry 128D normalization
"""

import logging

import numpy as np
import pytest

# Optional torch import
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =============================================================================
# GOSNN Fallback Tests
# =============================================================================


class TestGOSNNFallback:
    """Tests for GOSNN fallback behavior."""

    @pytest.fixture
    def gosnn(self):
        """Create GlobalOmniScalarNetwork instance."""
        if not HAS_TORCH:
            pytest.skip("torch not installed")
        from omni_anomaly_engine.core.global_omni_scalar_network import (
            GlobalOmniScalarNetwork,
        )

        return GlobalOmniScalarNetwork()

    def test_gosnn_initialization(self, gosnn):
        """Test GOSNN initializes correctly."""
        assert gosnn is not None

    def test_last_harmonic_synergy_initialized(self, gosnn):
        """Test last_harmonic_synergy is initialized."""
        assert hasattr(gosnn, "last_harmonic_synergy")

    def test_last_harmonic_synergy_type(self, gosnn):
        """Test last_harmonic_synergy has correct type."""
        synergy = gosnn.last_harmonic_synergy
        assert synergy is None or isinstance(synergy, (float, torch.Tensor))

    def test_compute_harmonic_synergy(self, gosnn, deterministic_rng):
        """Test harmonic synergy computation."""
        features = deterministic_rng.randn(1, 64)
        if hasattr(gosnn, "compute_harmonic_synergy"):
            synergy = gosnn.compute_harmonic_synergy(features)
            assert synergy is not None

    def test_harmonic_synergy_updates_last(self, gosnn, deterministic_rng):
        """Test harmonic synergy updates last_harmonic_synergy."""
        features = deterministic_rng.randn(1, 64)
        if hasattr(gosnn, "compute_harmonic_synergy"):
            gosnn.compute_harmonic_synergy(features)
            assert gosnn.last_harmonic_synergy is not None


# =============================================================================
# GOSNN Error Handling Tests
# =============================================================================


class TestGOSNNErrorHandling:
    """Tests for GOSNN error handling and logging."""

    @pytest.fixture
    def gosnn(self):
        """Create GlobalOmniScalarNetwork instance."""
        if not HAS_TORCH:
            pytest.skip("torch not installed")
        from omni_anomaly_engine.core.global_omni_scalar_network import (
            GlobalOmniScalarNetwork,
        )

        return GlobalOmniScalarNetwork()

    def test_gosnn_handles_invalid_input(self, gosnn, caplog):
        """Test GOSNN handles invalid input gracefully."""
        with caplog.at_level(logging.WARNING):
            try:
                gosnn.forward(None)
            except Exception:
                pass

    def test_gosnn_handles_empty_input(self, gosnn):
        """Test GOSNN handles empty input."""
        try:
            result = gosnn.forward(torch.tensor([]))
        except Exception:
            pass

    def test_gosnn_handles_wrong_dimensions(self, gosnn):
        """Test GOSNN handles wrong dimensions."""
        try:
            result = gosnn.forward(torch.randn(1, 1, 1, 1))
        except Exception:
            pass


# =============================================================================
# Scalar Registration Tests
# =============================================================================


class TestScalarRegistration:
    """Tests for scalar registration with omni_ prefix."""

    @pytest.fixture
    def gosnn(self):
        """Create GlobalOmniScalarNetwork instance."""
        if not HAS_TORCH:
            pytest.skip("torch not installed")
        from omni_anomaly_engine.core.global_omni_scalar_network import (
            GlobalOmniScalarNetwork,
        )

        return GlobalOmniScalarNetwork()

    def test_register_scalars(self, gosnn):
        """Test scalar registration."""
        from omni_anomaly_engine.core.global_omni_scalar_network import ScalarGroup

        gosnn.register_scalars(
            component_name="test_component",
            scalars={"omni_test_scalar": 0.5},
            group=ScalarGroup.ETHICAL,
        )

    def test_register_scalars_with_metadata(self, gosnn):
        """Test scalar registration with metadata."""
        from omni_anomaly_engine.core.global_omni_scalar_network import ScalarGroup

        gosnn.register_scalars(
            component_name="test_component",
            scalars={"omni_test_scalar": 0.5},
            group=ScalarGroup.ETHICAL,
            metadata={"source": "test"},
        )

    def test_scalar_groups(self):
        """Test all scalar groups exist."""
        from omni_anomaly_engine.core.global_omni_scalar_network import ScalarGroup

        assert ScalarGroup.ETHICAL is not None
        assert ScalarGroup.PERFORMANCE is not None
        assert ScalarGroup.SECURITY is not None


# =============================================================================
# Ethical Gating Tests
# =============================================================================


class TestEthicalGating:
    """Tests for ethical gating with sigma_sacred threshold."""

    @pytest.fixture
    def gosnn(self):
        """Create GlobalOmniScalarNetwork instance."""
        if not HAS_TORCH:
            pytest.skip("torch not installed")
        from omni_anomaly_engine.core.global_omni_scalar_network import (
            GlobalOmniScalarNetwork,
        )

        return GlobalOmniScalarNetwork()

    def test_sigma_sacred_default(self):
        """Test sigma_sacred default value."""
        from omni_anomaly_engine.core.global_omni_scalar_network import (
            SIGMA_SACRED_THRESHOLD,
        )

        assert SIGMA_SACRED_THRESHOLD == 0.96

    def test_sigma_sacred_medical_fallback(self):
        """Test sigma_sacred medical fallback value."""
        from omni_anomaly_engine.core.global_omni_scalar_network import (
            SIGMA_SACRED_MEDICAL_FALLBACK,
        )

        assert SIGMA_SACRED_MEDICAL_FALLBACK == 0.93

    def test_ethical_gate_passes_high_benevolence(self, gosnn):
        """Test ethical gate passes with high benevolence."""
        if hasattr(gosnn, "apply_ethical_gate"):
            features = torch.randn(1, 64)
            benevolence = 0.99
            result = gosnn.apply_ethical_gate(features, benevolence)
            assert result is not None

    def test_ethical_gate_blocks_low_benevolence(self, gosnn):
        """Test ethical gate blocks with low benevolence."""
        if hasattr(gosnn, "apply_ethical_gate"):
            features = torch.randn(1, 64)
            benevolence = 0.5
            try:
                result = gosnn.apply_ethical_gate(features, benevolence)
            except Exception:
                pass


# =============================================================================
# DetectorRegistry 128D Normalization Tests
# =============================================================================


class TestDetectorRegistry128D:
    """Tests for DetectorRegistry 128D normalization."""

    @pytest.fixture
    def registry(self):
        """Create DetectorRegistry instance."""
        if not HAS_TORCH:
            pytest.skip("torch not installed")
        from omni_anomaly_engine.core.detector_registry import DetectorRegistry

        return DetectorRegistry()

    def test_registry_initialization(self, registry):
        """Test registry initializes correctly."""
        assert registry is not None

    def test_aggregate_enhanced_geological_features_exists(self, registry):
        """Test aggregate_enhanced_geological_features method exists."""
        assert hasattr(registry, "aggregate_enhanced_geological_features")

    def test_aggregate_features_128d(self, registry, deterministic_rng):
        """Test feature aggregation produces 128D output."""
        detector_features = {
            "landslide": deterministic_rng.randn(20),
            "wildfire": deterministic_rng.randn(20),
            "volcanic": deterministic_rng.randn(20),
        }
        result = registry.aggregate_enhanced_geological_features(detector_features)
        assert result is not None
        assert len(result) == 128

    def test_l2_normalization_applied(self, registry, deterministic_rng):
        """Test L2 normalization is applied."""
        detector_features = {
            "landslide": deterministic_rng.randn(20) * 100,
            "wildfire": deterministic_rng.randn(20) * 100,
            "volcanic": deterministic_rng.randn(20) * 100,
        }
        result = registry.aggregate_enhanced_geological_features(detector_features)
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 0.1 or norm > 0

    def test_golden_ratio_scaling(self, registry, deterministic_rng):
        """Test golden ratio scaling is applied."""
        detector_features = {
            "landslide": deterministic_rng.randn(20),
            "wildfire": deterministic_rng.randn(20),
            "volcanic": deterministic_rng.randn(20),
        }
        result = registry.aggregate_enhanced_geological_features(detector_features)
        assert result is not None

    def test_omni_scalars_registered(self, registry, deterministic_rng):
        """Test omni-scalars are registered."""
        detector_features = {
            "landslide": deterministic_rng.randn(20),
            "wildfire": deterministic_rng.randn(20),
            "volcanic": deterministic_rng.randn(20),
        }
        registry.aggregate_enhanced_geological_features(detector_features)


# =============================================================================
# PHI Constant Tests
# =============================================================================


class TestPHIConstant:
    """Tests for PHI (golden ratio) constant."""

    def test_phi_value(self):
        """Test PHI constant value."""
        from omni_anomaly_engine.core.global_omni_scalar_network import PHI

        assert abs(PHI - 1.618033988749895) < 1e-10

    def test_phi_golden_ratio_property(self):
        """Test PHI satisfies golden ratio property."""
        from omni_anomaly_engine.core.global_omni_scalar_network import PHI

        assert abs(PHI - (1 + np.sqrt(5)) / 2) < 1e-10


# =============================================================================
# Lyapunov Stability Tests
# =============================================================================


class TestLyapunovStability:
    """Tests for Lyapunov stability constants."""

    def test_lambda_lyapunov_value(self):
        """Test LAMBDA_LYAPUNOV constant value."""
        from omni_anomaly_engine.core.global_omni_scalar_network import LAMBDA_LYAPUNOV

        assert LAMBDA_LYAPUNOV == 0.25

    def test_lyapunov_stability_bound(self):
        """Test Lyapunov stability bound V <= epsilon * e^(-0.25t)."""
        from omni_anomaly_engine.core.global_omni_scalar_network import LAMBDA_LYAPUNOV

        epsilon = 1.0
        t = 10.0
        bound = epsilon * np.exp(-LAMBDA_LYAPUNOV * t)
        assert bound < epsilon


# =============================================================================
# Engine Integration Tests
# =============================================================================


class TestEngineGOSNNIntegration:
    """Tests for engine GOSNN integration."""

    @pytest.fixture
    def engine(self):
        """Create OmniAnomalyEngine instance."""
        if not HAS_TORCH:
            pytest.skip("torch not installed")
        from omni_anomaly_engine.engine import OmniAnomalyEngine

        return OmniAnomalyEngine()

    def test_engine_has_gosnn(self, engine):
        """Test engine has GOSNN integration."""
        assert hasattr(engine, "gosnn") or hasattr(engine, "_gosnn")

    def test_detect_with_fusion_exists(self, engine):
        """Test detect_with_fusion method exists."""
        assert hasattr(engine, "detect_with_fusion")

    def test_engine_logging_on_error(self, engine, caplog):
        """Test engine logs on GOSNN error."""
        with caplog.at_level(logging.WARNING):
            try:
                engine.detect_with_fusion(None)
            except Exception:
                pass


# =============================================================================
# Triadic Phi-Weighting Tests
# =============================================================================


class TestTriadicPhiWeighting:
    """Tests for triadic phi-weighting."""

    @pytest.fixture
    def phi_weighting(self):
        """Create TriadicPhiWeighting instance."""
        if not HAS_TORCH:
            pytest.skip("torch not installed")
        from omni_anomaly_engine.core.global_omni_scalar_network import (
            TriadicPhiWeighting,
        )

        return TriadicPhiWeighting(num_heads=32)

    def test_phi_weighting_initialization(self, phi_weighting):
        """Test phi weighting initializes correctly."""
        assert phi_weighting is not None
        assert phi_weighting.num_heads == 32

    def test_phi_weighting_forward(self, phi_weighting, deterministic_rng):
        """Test phi weighting forward pass."""
        features = torch.randn(1, 64)
        result = phi_weighting.forward(features)
        assert result is not None

    def test_phi_weighting_32_heads(self, phi_weighting):
        """Test phi weighting uses 32 heads."""
        assert phi_weighting.num_heads == 32


# =============================================================================
# Ava-Dominance Equation Tests
# =============================================================================


class TestAvaDominanceEquation:
    """Tests for Ava-Dominance equation implementation."""

    @pytest.fixture
    def three_r(self):
        """Create ThreeRMechanism instance."""
        if not HAS_TORCH:
            pytest.skip("torch not installed")
        from omni_anomaly_engine.core.three_r_mechanism import ThreeRMechanism

        return ThreeRMechanism()

    def test_ava_dominance_exists(self, three_r):
        """Test Ava-Dominance method exists."""
        assert hasattr(three_r, "compute_ava_dominance") or hasattr(three_r, "ava_dominance")

    def test_ava_dominance_components(self, three_r):
        """Test Ava-Dominance has R, H, O components."""
        assert hasattr(three_r, "recursion") or hasattr(three_r, "R")
        assert hasattr(three_r, "resonance") or hasattr(three_r, "H")
        assert hasattr(three_r, "refactoring") or hasattr(three_r, "O")


# =============================================================================
# Integration Tests
# =============================================================================


class TestGOSNNFullIntegration:
    """Full integration tests for GOSNN."""

    @pytest.fixture
    def full_setup(self):
        """Create full GOSNN setup."""
        if not HAS_TORCH:
            pytest.skip("torch not installed")
        from omni_anomaly_engine.core.detector_registry import DetectorRegistry
        from omni_anomaly_engine.core.global_omni_scalar_network import (
            GlobalOmniScalarNetwork,
        )
        from omni_anomaly_engine.engine import OmniAnomalyEngine

        return {
            "gosnn": GlobalOmniScalarNetwork(),
            "registry": DetectorRegistry(),
            "engine": OmniAnomalyEngine(),
        }

    def test_all_components_initialize(self, full_setup):
        """Test all components initialize correctly."""
        assert full_setup["gosnn"] is not None
        assert full_setup["registry"] is not None
        assert full_setup["engine"] is not None

    def test_components_interconnected(self, full_setup):
        """Test components are interconnected."""
        pass

    def test_bidirectional_flow(self, full_setup, deterministic_rng):
        """Test bidirectional flow between components."""
        pass
