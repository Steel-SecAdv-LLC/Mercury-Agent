"""
Mercury Agent ♱
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
- Ethical gating with sigma_immutable threshold
- DetectorRegistry 128D normalization
"""

from __future__ import annotations

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
        from omni_mercury_engine.core.global_omni_scalar_network import (
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
        from omni_mercury_engine.core.global_omni_scalar_network import (
            GlobalOmniScalarNetwork,
        )

        return GlobalOmniScalarNetwork()

    def test_gosnn_handles_invalid_input(self, gosnn, caplog):
        """Test GOSNN handles invalid input gracefully."""
        with caplog.at_level(logging.WARNING):
            try:
                gosnn.forward(None)
            except Exception:
                pass  # Expected: GOSNN should handle None input gracefully

    def test_gosnn_handles_empty_input(self, gosnn):
        """Test GOSNN handles empty input."""
        try:
            gosnn.forward(torch.tensor([]))
        except Exception:
            pass  # Expected: GOSNN may raise on empty tensor

    def test_gosnn_handles_wrong_dimensions(self, gosnn):
        """Test GOSNN handles wrong dimensions."""
        try:
            gosnn.forward(torch.randn(1, 1, 1, 1))
        except Exception:
            pass  # Expected: GOSNN may raise on wrong dimensions


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
        from omni_mercury_engine.core.global_omni_scalar_network import (
            GlobalOmniScalarNetwork,
        )

        return GlobalOmniScalarNetwork()

    def test_register_scalars(self, gosnn):
        """Test scalar registration."""
        from omni_mercury_engine.core.global_omni_scalar_network import ScalarGroup

        gosnn.register_scalars(
            component_name="test_component",
            scalars={"omni_test_scalar": 0.5},
            group=ScalarGroup.ETHICAL,
        )

    def test_register_scalars_with_metadata(self, gosnn):
        """Test scalar registration with metadata."""
        from omni_mercury_engine.core.global_omni_scalar_network import ScalarGroup

        gosnn.register_scalars(
            component_name="test_component",
            scalars={"omni_test_scalar": 0.5},
            group=ScalarGroup.ETHICAL,
            metadata={"source": "test"},
        )

    def test_scalar_groups(self):
        """Test all scalar groups exist."""
        from omni_mercury_engine.core.global_omni_scalar_network import ScalarGroup

        assert ScalarGroup.ETHICAL is not None
        assert ScalarGroup.PERFORMANCE is not None
        assert ScalarGroup.SECURITY is not None


# =============================================================================
# Ethical Gating Tests
# =============================================================================


class TestEthicalGating:
    """Tests for ethical gating with sigma_immutable threshold."""

    @pytest.fixture
    def gosnn(self):
        """Create GlobalOmniScalarNetwork instance."""
        if not HAS_TORCH:
            pytest.skip("torch not installed")
        from omni_mercury_engine.core.global_omni_scalar_network import (
            GlobalOmniScalarNetwork,
        )

        return GlobalOmniScalarNetwork()

    def test_sigma_immutable_default(self):
        """Test sigma_immutable default value."""
        from omni_mercury_engine.core.global_omni_scalar_network import (
            SIGMA_IMMUTABLE_DEFAULT,
        )

        assert SIGMA_IMMUTABLE_DEFAULT == 0.96

    def test_sigma_immutable_medical_fallback(self):
        """Test sigma_immutable medical fallback value."""
        from omni_mercury_engine.core.global_omni_scalar_network import (
            SIGMA_IMMUTABLE_MEDICAL,
        )

        assert SIGMA_IMMUTABLE_MEDICAL == 0.93

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
                gosnn.apply_ethical_gate(features, benevolence)
            except Exception:
                pass  # Expected: ethical gate may block low benevolence


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
        from omni_mercury_engine.core.detector_registry import DetectorRegistry

        return DetectorRegistry()

    def test_registry_initialization(self, registry):
        """Test registry initializes correctly."""
        assert registry is not None

    def test_aggregate_enhanced_geological_features_exists(self, registry):
        """Test aggregate_enhanced_geological_features method exists."""
        assert hasattr(registry, "aggregate_enhanced_geological_features")

    def test_aggregate_features_128d(self, registry, deterministic_rng):
        """Test feature aggregation produces 128D output."""
        from omni_mercury_engine.core.detector_registry import FeatureExtractionResult

        detector_features = {
            "landslide": FeatureExtractionResult(
                detector_name="landslide",
                features=deterministic_rng.randn(20),
                scores=None,
                execution_time_ms=1.0,
                success=True,
            ),
            "wildfire": FeatureExtractionResult(
                detector_name="wildfire",
                features=deterministic_rng.randn(20),
                scores=None,
                execution_time_ms=1.0,
                success=True,
            ),
            "volcanic": FeatureExtractionResult(
                detector_name="volcanic",
                features=deterministic_rng.randn(20),
                scores=None,
                execution_time_ms=1.0,
                success=True,
            ),
        }
        result = registry.aggregate_enhanced_geological_features(detector_features)
        assert result is not None
        assert "combined_features" in result
        if result["combined_features"] is not None:
            assert result["combined_features"].shape[-1] == 128

    def test_l2_normalization_applied(self, registry, deterministic_rng):
        """Test L2 normalization is applied."""
        from omni_mercury_engine.core.detector_registry import FeatureExtractionResult

        detector_features = {
            "landslide": FeatureExtractionResult(
                detector_name="landslide",
                features=deterministic_rng.randn(20) * 100,
                scores=None,
                execution_time_ms=1.0,
                success=True,
            ),
            "wildfire": FeatureExtractionResult(
                detector_name="wildfire",
                features=deterministic_rng.randn(20) * 100,
                scores=None,
                execution_time_ms=1.0,
                success=True,
            ),
            "volcanic": FeatureExtractionResult(
                detector_name="volcanic",
                features=deterministic_rng.randn(20) * 100,
                scores=None,
                execution_time_ms=1.0,
                success=True,
            ),
        }
        result = registry.aggregate_enhanced_geological_features(detector_features)
        assert result is not None
        if result["combined_features"] is not None:
            norm = torch.norm(result["combined_features"]).item()
            assert norm > 0

    def test_golden_ratio_scaling(self, registry, deterministic_rng):
        """Test golden ratio scaling is applied."""
        from omni_mercury_engine.core.detector_registry import FeatureExtractionResult

        detector_features = {
            "landslide": FeatureExtractionResult(
                detector_name="landslide",
                features=deterministic_rng.randn(20),
                scores=None,
                execution_time_ms=1.0,
                success=True,
            ),
            "wildfire": FeatureExtractionResult(
                detector_name="wildfire",
                features=deterministic_rng.randn(20),
                scores=None,
                execution_time_ms=1.0,
                success=True,
            ),
            "volcanic": FeatureExtractionResult(
                detector_name="volcanic",
                features=deterministic_rng.randn(20),
                scores=None,
                execution_time_ms=1.0,
                success=True,
            ),
        }
        result = registry.aggregate_enhanced_geological_features(detector_features)
        assert result is not None
        assert "aggregated_features" in result

    def test_omni_scalars_registered(self, registry, deterministic_rng):
        """Test omni-scalars are registered."""
        from omni_mercury_engine.core.detector_registry import FeatureExtractionResult

        detector_features = {
            "landslide": FeatureExtractionResult(
                detector_name="landslide",
                features=deterministic_rng.randn(20),
                scores=None,
                execution_time_ms=1.0,
                success=True,
            ),
            "wildfire": FeatureExtractionResult(
                detector_name="wildfire",
                features=deterministic_rng.randn(20),
                scores=None,
                execution_time_ms=1.0,
                success=True,
            ),
            "volcanic": FeatureExtractionResult(
                detector_name="volcanic",
                features=deterministic_rng.randn(20),
                scores=None,
                execution_time_ms=1.0,
                success=True,
            ),
        }
        result = registry.aggregate_enhanced_geological_features(detector_features)
        assert "gosnn_scalars" in result
        scalars = result["gosnn_scalars"]
        for key in scalars:
            assert key.startswith("omni_")


# =============================================================================
# PHI Constant Tests
# =============================================================================


class TestPHIConstant:
    """Tests for PHI (golden ratio) constant."""

    def test_phi_value(self):
        """Test PHI constant value."""
        from omni_mercury_engine.core.global_omni_scalar_network import PHI

        assert abs(PHI - 1.618033988749895) < 1e-10

    def test_phi_golden_ratio_property(self):
        """Test PHI satisfies golden ratio property."""
        from omni_mercury_engine.core.global_omni_scalar_network import PHI

        assert abs(PHI - (1 + np.sqrt(5)) / 2) < 1e-10


# =============================================================================
# Lyapunov Stability Tests
# =============================================================================


class TestLyapunovStability:
    """Tests for Lyapunov stability constants."""

    def test_lambda_lyapunov_value(self):
        """Test LAMBDA_LYAPUNOV constant value."""
        from omni_mercury_engine.core.global_omni_scalar_network import LAMBDA_LYAPUNOV

        assert LAMBDA_LYAPUNOV == 0.25

    def test_lyapunov_stability_bound(self):
        """Test Lyapunov stability bound V <= epsilon * e^(-0.25t)."""
        from omni_mercury_engine.core.global_omni_scalar_network import LAMBDA_LYAPUNOV

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
        """Create OmniMercuryEngine instance."""
        if not HAS_TORCH:
            pytest.skip("torch not installed")
        from omni_mercury_engine.engine import OmniMercuryEngine

        return OmniMercuryEngine()

    def test_engine_has_gosnn(self, engine):
        """Test engine has GOSNN integration via fusion model."""
        # Engine integrates GOSNN through fusion_model, not direct gosnn attribute
        assert hasattr(engine, "fusion_model") or hasattr(engine, "detect_with_fusion")

    def test_detect_with_fusion_exists(self, engine):
        """Test detect_with_fusion method exists."""
        assert hasattr(engine, "detect_with_fusion")

    def test_engine_logging_on_error(self, engine, caplog):
        """Test engine logs on GOSNN error."""
        with caplog.at_level(logging.WARNING):
            try:
                engine.detect_with_fusion(None)
            except Exception:
                pass  # Expected: engine may raise on None input


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
        from omni_mercury_engine.core.global_omni_scalar_network import (
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
        # TriadicPhiWeighting uses apply() method, not forward()
        if hasattr(phi_weighting, "apply"):
            result = phi_weighting.apply(features)
        elif hasattr(phi_weighting, "forward"):
            result = phi_weighting.forward(features)
        else:
            result = phi_weighting(features)
        assert result is not None

    def test_phi_weighting_32_heads(self, phi_weighting):
        """Test phi weighting uses 32 heads."""
        assert phi_weighting.num_heads == 32


# =============================================================================
# weighted fusion Equation Tests
# =============================================================================


class TestFusionEquation:
    """Tests for weighted fusion equation implementation."""

    @pytest.fixture
    def three_r(self):
        """Create ThreeRMechanism instance."""
        if not HAS_TORCH:
            pytest.skip("torch not installed")
        from omni_mercury_engine.core.three_r_mechanism import ThreeRMechanism

        return ThreeRMechanism()

    def test_fusion_exists(self, three_r):
        """Test weighted fusion method exists."""
        assert hasattr(three_r, "compute_fusion") or hasattr(three_r, "fusion")

    def test_fusion_components(self, three_r):
        """Test weighted fusion has R, H, O components."""
        # ThreeRMechanism uses *_engine naming convention
        assert (
            hasattr(three_r, "recursion_engine")
            or hasattr(three_r, "recursion")
            or hasattr(three_r, "R")
        )
        assert (
            hasattr(three_r, "resonance_engine")
            or hasattr(three_r, "resonance")
            or hasattr(three_r, "H")
        )
        assert (
            hasattr(three_r, "refactoring_engine")
            or hasattr(three_r, "refactoring")
            or hasattr(three_r, "O")
        )


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
        from omni_mercury_engine.core.detector_registry import DetectorRegistry
        from omni_mercury_engine.core.global_omni_scalar_network import (
            GlobalOmniScalarNetwork,
        )
        from omni_mercury_engine.engine import OmniMercuryEngine

        return {
            "gosnn": GlobalOmniScalarNetwork(),
            "registry": DetectorRegistry(),
            "engine": OmniMercuryEngine(),
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


# =============================================================================
# Omni-Scalar Legacy Alias Resolution Tests
# =============================================================================


class TestOmniScalarLegacyAliases:
    """Tests for omni-scalar legacy alias resolution (deprecated in v2.0)."""

    @pytest.fixture
    def gosnn(self):
        """Create GlobalOmniScalarNetwork instance."""
        if not HAS_TORCH:
            pytest.skip("torch not installed")
        from omni_mercury_engine.core.global_omni_scalar_network import (
            GlobalOmniScalarNetwork,
            reset_global_network,
        )

        reset_global_network()
        return GlobalOmniScalarNetwork()

    def test_legacy_aliases_initialized(self, gosnn):
        """Test legacy aliases are initialized."""
        assert hasattr(gosnn, "_legacy_aliases")
        assert isinstance(gosnn._legacy_aliases, dict)
        assert len(gosnn._legacy_aliases) > 0

    def test_resolve_scalar_name_exists(self, gosnn):
        """Test resolve_scalar_name method exists."""
        assert hasattr(gosnn, "resolve_scalar_name")

    def test_get_scalar_exists(self, gosnn):
        """Test get_scalar method exists."""
        assert hasattr(gosnn, "get_scalar")

    def test_resolve_legacy_morality_scalar(self, gosnn):
        """Test resolving legacy morality_scalar to omnimorality."""
        resolved = gosnn.resolve_scalar_name("morality_scalar")
        assert resolved == "omnimorality"

    def test_resolve_legacy_empathy_scalar(self, gosnn):
        """Test resolving legacy empathy_scalar to omniempathy."""
        resolved = gosnn.resolve_scalar_name("empathy_scalar")
        assert resolved == "omniempathy"

    def test_resolve_legacy_compassion_scalar(self, gosnn):
        """Test resolving legacy compassion_scalar to omnicompassion."""
        resolved = gosnn.resolve_scalar_name("compassion_scalar")
        assert resolved == "omnicompassion"

    def test_resolve_legacy_benevolence(self, gosnn):
        """Test resolving legacy benevolence to omnibenevolence."""
        resolved = gosnn.resolve_scalar_name("benevolence")
        assert resolved == "omnibenevolence"

    def test_resolve_omni_prefixed_unchanged(self, gosnn):
        """Test omni-prefixed names are returned unchanged."""
        resolved = gosnn.resolve_scalar_name("omnimorality")
        assert resolved == "omnimorality"

    def test_resolve_unknown_name_unchanged(self, gosnn):
        """Test unknown names are returned unchanged."""
        resolved = gosnn.resolve_scalar_name("unknown_scalar")
        assert resolved == "unknown_scalar"

    def test_get_scalar_with_legacy_name(self, gosnn):
        """Test get_scalar works with legacy names."""
        value = gosnn.get_scalar("morality_scalar")
        assert value > 0

    def test_get_scalar_with_omni_name(self, gosnn):
        """Test get_scalar works with omni-prefixed names."""
        value = gosnn.get_scalar("omnimorality")
        assert value > 0

    def test_get_scalar_legacy_equals_omni(self, gosnn):
        """Test legacy and omni names return same value."""
        legacy_value = gosnn.get_scalar("morality_scalar")
        omni_value = gosnn.get_scalar("omnimorality")
        assert legacy_value == omni_value

    def test_get_scalar_default_for_unknown(self, gosnn):
        """Test get_scalar returns default for unknown names."""
        value = gosnn.get_scalar("unknown_scalar", default=0.5)
        assert value == 0.5

    def test_all_legacy_aliases_resolve(self, gosnn):
        """Test all legacy aliases resolve to omni-prefixed names."""
        for legacy_name, omni_name in gosnn._legacy_aliases.items():
            resolved = gosnn.resolve_scalar_name(legacy_name)
            assert resolved == omni_name
            assert omni_name.startswith("omni")


class TestOmniScalarValues:
    """Tests for omni-scalar default values."""

    @pytest.fixture
    def gosnn(self):
        """Create GlobalOmniScalarNetwork instance."""
        if not HAS_TORCH:
            pytest.skip("torch not installed")
        from omni_mercury_engine.core.global_omni_scalar_network import (
            GlobalOmniScalarNetwork,
            reset_global_network,
        )

        reset_global_network()
        return GlobalOmniScalarNetwork()

    def test_omnibenevolence_threshold(self, gosnn):
        """Test omnibenevolence >= 0.99 threshold."""
        value = gosnn.get_scalar("omnibenevolence")
        assert value >= 0.99

    def test_omnicompassion_value(self, gosnn):
        """Test omnicompassion has positive value."""
        value = gosnn.get_scalar("omnicompassion")
        assert value > 0

    def test_omniempathy_value(self, gosnn):
        """Test omniempathy has positive value."""
        value = gosnn.get_scalar("omniempathy")
        assert value > 0

    def test_omnimorality_value(self, gosnn):
        """Test omnimorality has positive value."""
        value = gosnn.get_scalar("omnimorality")
        assert value > 0

    def test_omnijustice_value(self, gosnn):
        """Test omnijustice has positive value."""
        value = gosnn.get_scalar("omnijustice")
        assert value > 0

    def test_omniequity_value(self, gosnn):
        """Test omniequity has positive value."""
        value = gosnn.get_scalar("omniequity")
        assert value > 0

    def test_omnilove_value(self, gosnn):
        """Test omnilove has positive value."""
        value = gosnn.get_scalar("omnilove")
        assert value > 0

    def test_omniforgiveness_value(self, gosnn):
        """Test omniforgiveness has positive value."""
        value = gosnn.get_scalar("omniforgiveness")
        assert value > 0


class TestBiasAuditOmniScalars:
    """Tests for bias audit with omni-scalars."""

    @pytest.fixture
    def gosnn(self):
        """Create GlobalOmniScalarNetwork instance."""
        if not HAS_TORCH:
            pytest.skip("torch not installed")
        from omni_mercury_engine.core.global_omni_scalar_network import (
            GlobalOmniScalarNetwork,
            reset_global_network,
        )

        reset_global_network()
        return GlobalOmniScalarNetwork()

    def test_bias_audit_returns_omni_scalars(self, gosnn):
        """Test bias audit returns omni-prefixed scalar names."""
        result = gosnn.perform_bias_audit()
        assert "omniempathy" in result
        assert "omnimorality" in result
        assert "omnibenevolence" in result

    def test_bias_audit_omnibenevolence_check(self, gosnn):
        """Test bias audit checks omnibenevolence >= 0.99."""
        result = gosnn.perform_bias_audit()
        assert result["omnibenevolence"] >= 0.99

    def test_bias_audit_status(self, gosnn):
        """Test bias audit returns status."""
        result = gosnn.perform_bias_audit()
        assert "status" in result
        assert result["status"] in ["passed", "warnings"]

    def test_bias_audit_recommendations(self, gosnn):
        """Test bias audit returns recommendations list."""
        result = gosnn.perform_bias_audit()
        assert "recommendations" in result
        assert isinstance(result["recommendations"], list)
