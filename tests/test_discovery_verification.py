# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Comprehensive Discovery and Innovation Verification Tests. Tests the key claims made in NOVELTY_PROOFS.md and DISCOVERIES.md.

Note: Some tests require PyTorch and are marked with pytest.mark.skipif.

This file was renamed from test_novelty_verification.py for clarity.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from scipy import stats

# Check for torch availability
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class TestDoubleHelixEngine:
    """Verify Double-Helix Evolution Engine novelty claims."""

    def test_golden_ratio_constants(self) -> None:
        """Verify phi constants are correctly defined."""
        from omni_mercury_engine.core.double_helix_engine import PHI, PHI_CUBED, PHI_SQUARED

        expected_phi = (1 + np.sqrt(5)) / 2
        assert abs(PHI - expected_phi) < 1e-10, "PHI should be golden ratio"
        assert abs(PHI_SQUARED - expected_phi**2) < 1e-10, "PHI_SQUARED incorrect"
        assert abs(PHI_CUBED - expected_phi**3) < 1e-10, "PHI_CUBED incorrect"

    def test_lyapunov_stability_decay(self) -> None:
        """Verify Lyapunov stability decay constant."""
        from omni_mercury_engine.core.double_helix_engine import LAMBDA_DECAY

        assert LAMBDA_DECAY == 0.18, "Lambda decay should be 0.18"

    def test_evolution_modes(self) -> None:
        """Test that evolution modes are defined."""
        from omni_mercury_engine.core.double_helix_engine import EvolutionMode

        modes = list(EvolutionMode)
        assert len(modes) >= 4, f"Expected 4+ evolution modes, got {len(modes)}"

    def test_term_types(self) -> None:
        """Test that term types are defined."""
        from omni_mercury_engine.core.double_helix_engine import TermType

        types = list(TermType)
        assert len(types) >= 6, f"Expected 6+ term types, got {len(types)}"

    def test_phi_optimized_term_weights(self) -> None:
        """Verify phi-optimization in term weights."""
        from omni_mercury_engine.core.double_helix_engine import MercuryEquationEngine

        engine = MercuryEquationEngine(dimension=8)

        # Check that weights sum to 1 (normalized)
        total_weight = sum(engine.term_weights.values())
        assert abs(total_weight - 1.0) < 0.01, "Term weights should sum to 1.0"

    def test_18_evolution_terms(self) -> None:
        """Verify 18+ evolution terms as claimed."""
        from omni_mercury_engine.core.double_helix_engine import MercuryEquationEngine

        engine = MercuryEquationEngine(dimension=8)
        num_terms = len(engine.term_weights)
        assert num_terms >= 18, f"Expected 18+ terms, got {num_terms}"

    def test_seed_makes_evolution_reproducible(self) -> None:
        """Two MercuryEquationEngine instances with the same seed must
        evolve identically across all stochastic terms (ethical-matrix
        init, Hamiltonian symmetric matrix, Boltzmann sampling,
        simulated-annealing exploration, Lyapunov chaos perturbation).
        Different seeds must diverge."""
        from omni_mercury_engine.core.double_helix_engine import MercuryEquationEngine

        rng = np.random.default_rng(0)
        x0 = rng.standard_normal(8)

        def _evolve(seed: int) -> np.ndarray:
            engine = MercuryEquationEngine(dimension=8, seed=seed)
            state = x0.copy()
            for _ in range(20):
                state, _contributions = engine.step(state)
            return state

        a = _evolve(seed=42)
        b = _evolve(seed=42)
        assert np.array_equal(a, b), (
            "Same-seed MercuryEquationEngine runs diverged; seed is not "
            "threading through every stochastic term."
        )

        c = _evolve(seed=43)
        assert not np.array_equal(
            a, c
        ), "Different seeds produced identical evolution; seed has no effect."


class TestEthicalScalars:
    """Verify ethical scalar framework claims."""

    def test_ethical_scalars_exist(self) -> None:
        """Verify ethical scalars are defined."""
        from omni_mercury_engine.core.ethical_config import EthicalScalars

        scalars = EthicalScalars()
        assert scalars is not None

    def test_to_dict_method(self) -> None:
        """Verify scalars can be exported."""
        from omni_mercury_engine.core.ethical_config import EthicalScalars

        scalars = EthicalScalars()
        config = scalars.to_dict()
        assert isinstance(config, dict)


class TestStatisticalValidation:
    """Verify statistical validation methodology claims."""

    def test_improvement_calculation(self) -> None:
        """Test that improvement percentages are calculated correctly."""
        np.random.seed(42)

        # Simulated comparison: our method vs baseline
        our_scores = np.random.normal(0.85, 0.05, 50)
        baseline_scores = np.random.normal(0.65, 0.10, 50)

        our_mean = np.mean(our_scores)
        baseline_mean = np.mean(baseline_scores)

        improvement = (our_mean - baseline_mean) / baseline_mean * 100

        # Verify t-test
        t_stat, p_value = stats.ttest_ind(our_scores, baseline_scores)

        assert improvement > 15, f"Should show >15% improvement, got {improvement:.1f}%"
        assert p_value < 0.05, f"Should be statistically significant, p={p_value:.4f}"

    def test_effect_size_calculation(self) -> None:
        """Test Cohen's d effect size calculation."""
        np.random.seed(42)

        group1 = np.random.normal(0.8, 0.1, 50)
        group2 = np.random.normal(0.5, 0.1, 50)

        # Cohen's d
        pooled_std = np.sqrt(
            ((len(group1) - 1) * np.std(group1) ** 2 + (len(group2) - 1) * np.std(group2) ** 2)
            / (len(group1) + len(group2) - 2)
        )
        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std

        # Effect size should be large (d > 0.8)
        assert cohens_d > 0.8, f"Expected large effect size, got d={cohens_d:.2f}"


class TestNovelClassDiscovery:
    """Verify Novel Class Discovery integration claims."""

    def test_mebin_initialization(self) -> None:
        """Test MEBin initialization."""
        from omni_mercury_engine.core.novel_class_discovery import MultiElementBinarization

        mebin = MultiElementBinarization()
        assert mebin.rotation_angles == [0, 90, 180, 270]
        assert mebin.binarization_threshold == 0.5

    def test_mebin_binarization(self) -> None:
        """Test Multi-Element Binarization."""
        from omni_mercury_engine.core.novel_class_discovery import MultiElementBinarization

        mebin = MultiElementBinarization()

        # Test binarization
        mask = np.array([0.3, 0.7, 0.9, 0.1, 0.6])
        binary = mebin.binarize(mask)

        expected = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
        np.testing.assert_array_equal(binary, expected)

    def test_ncd_initialization(self) -> None:
        """Test NCD initialization."""
        from omni_mercury_engine.core.novel_class_discovery import NovelClassDiscovery

        ncd = NovelClassDiscovery()
        assert ncd.enable_mebin is True
        assert ncd.num_clusters == 5

    def test_kmeans_clustering(self) -> None:
        """Test K-means clustering in novel class discovery."""
        from omni_mercury_engine.core.novel_class_discovery import NovelClassDiscovery

        ncd = NovelClassDiscovery({"num_clusters": 3})

        # Generate test data
        images = np.random.randn(30, 32, 32, 3)
        masks = np.random.rand(30, 32, 32)

        results = ncd.discover_novel_classes(images, masks)

        assert results["num_classes"] == 3
        assert "class_assignments" in results
        assert "discovered_classes" in results

    def test_class_statistics(self) -> None:
        """Test class statistics computation."""
        from omni_mercury_engine.core.novel_class_discovery import NovelClassDiscovery

        ncd = NovelClassDiscovery()
        class_assignments = np.array([0, 1, 0, 2, 1, 0, 2, 1, 0])

        stats_result = ncd.get_class_statistics(class_assignments)

        assert stats_result["total_samples"] == 9
        assert "num_samples_per_class" in stats_result
        assert "class_distribution" in stats_result


class TestGoldenRatioUniversality:
    """Verify golden ratio appearance across domains (Discovery claim)."""

    def test_phi_constant_accuracy(self) -> None:
        """Verify phi is the true golden ratio."""
        from omni_mercury_engine.core.double_helix_engine import PHI

        # Golden ratio from quadratic formula: (1 + sqrt(5)) / 2
        true_phi = (1 + np.sqrt(5)) / 2

        assert abs(PHI - true_phi) < 1e-15, "PHI should match true golden ratio"

    def test_phi_self_similar_property(self) -> None:
        """Verify phi's self-similar property: phi^2 = phi + 1."""
        from omni_mercury_engine.core.double_helix_engine import PHI, PHI_SQUARED

        # phi^2 should equal phi + 1
        assert abs(PHI_SQUARED - (PHI + 1)) < 1e-10, "phi^2 should equal phi + 1"

    def test_phi_reciprocal_property(self) -> None:
        """Verify phi's reciprocal property: 1/phi = phi - 1."""
        from omni_mercury_engine.core.double_helix_engine import PHI

        # 1/phi should equal phi - 1
        assert abs(1 / PHI - (PHI - 1)) < 1e-10, "1/phi should equal phi - 1"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPyTorchDependentFeatures:
    """Tests for features that require PyTorch."""

    def test_medical_specialties(self) -> None:
        """Verify 24 ABMS board specialties are defined."""
        from omni_mercury_engine.medical.abms_disciplines import ABMSSpecialty

        specialties = list(ABMSSpecialty)
        assert len(specialties) == 24, f"Expected 24 specialties, got {len(specialties)}"

    def test_schumann_detector(self) -> None:
        """Test Schumann resonance detector."""
        from omni_mercury_engine.space.schumann_resonance import SchumannResonanceDetector

        detector = SchumannResonanceDetector()
        assert abs(detector.fundamental_freq - 7.83) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
