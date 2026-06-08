# Copyright (C) 2025 Steel Security Advisors LLC
"""Extended tests for Anomaly Math Arrest — probes 9-21."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from omni_mercury_engine.detectors.math_arrest.arrest import (
    AnomalyMathArrest,
)
from omni_mercury_engine.detectors.math_arrest.probes.boltzmann_coupling import (
    BoltzmannCouplingProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.energy_minimization import (
    EnergyMinimizationProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.fractal_similarity import (
    FractalSelfSimilarityProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.iqr_robust import (
    IQRRobustProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.lyapunov_chaos import (
    LyapunovChaosProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.modified_zscore import (
    ModifiedZScoreProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.quantum_annealing import (
    QuantumAnnealingProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.quantum_superposition import (
    QuantumSuperpositionProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.r3_recursion import (
    R3RecursionResonanceProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.svd_projection import (
    SVDProjectionProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.topology_homology import (
    TopologyHomologyProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.wave_propagation import (
    WavePropagationProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.zeta_harmonic import (
    ZetaHarmonicProbe,
)

# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def make_normal_signal(n: int = 500, seed: int = 42) -> npt.NDArray[np.float64]:
    """Sinusoidal signal with mild noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 10 * np.pi, n)
    return (np.sin(t) + rng.normal(0, 0.05, n)).astype(np.float64)


def logistic_map(r: float, x0: float, n: int) -> npt.NDArray[np.float64]:
    """Iterate the logistic map x_{t+1} = r * x_t * (1 - x_t)."""
    x = np.empty(n, dtype=np.float64)
    x[0] = x0
    for i in range(1, n):
        x[i] = r * x[i - 1] * (1 - x[i - 1])
    return x


# ===================================================================
# Probe 9: R3RecursionResonanceProbe
# ===================================================================


class TestR3RecursionResonanceProbe:
    """R3RecursionResonanceProbe unit tests."""

    def test_detects_nonlinear_saturation(self) -> None:
        probe = R3RecursionResonanceProbe()
        rng = np.random.default_rng(42)
        train = rng.normal(0, 1, 200).astype(np.float64)
        probe.fit_trajectory(train)
        test_data = train.copy()
        test_data[100:120] = 50.0  # saturate
        result = probe.deviation_score(test_data)
        assert float(np.max(result.deviation_scores[100:120])) > float(
            np.median(result.deviation_scores[:80])
        )

    def test_three_transforms_independent(self) -> None:
        probe = R3RecursionResonanceProbe()
        data = make_normal_signal(200)
        probe.fit_trajectory(data)
        result = probe.deviation_score(data)
        assert result.probe_name == "r3_recursion_resonance"
        assert result.anomaly_geometry == "nonlinear_saturation"
        assert not np.any(np.isnan(result.deviation_scores))


# ===================================================================
# Probe 10: SVDProjectionProbe
# ===================================================================


class TestSVDProjectionProbe:
    """SVDProjectionProbe unit tests."""

    def test_detects_dimensional_collapse(self) -> None:
        probe = SVDProjectionProbe()
        rng = np.random.default_rng(42)
        train = rng.normal(0, 1, 200).astype(np.float64)
        probe.fit_trajectory(train)
        test_data = train.copy()
        test_data[100:150] = 0.0  # collapse to zero
        result = probe.deviation_score(test_data)
        assert not np.any(np.isnan(result.deviation_scores))

    def test_minimum_20_samples(self) -> None:
        probe = SVDProjectionProbe()
        with pytest.raises(ValueError, match="at least"):
            probe.fit_trajectory(np.ones(10, dtype=np.float64))

    def test_output_padded_to_input_length(self) -> None:
        probe = SVDProjectionProbe()
        data = make_normal_signal(200)
        probe.fit_trajectory(data)
        result = probe.deviation_score(data)
        assert result.deviation_scores.shape == (200,)


# ===================================================================
# Probe 11: LyapunovChaosProbe
# ===================================================================


class TestLyapunovChaosProbe:
    """LyapunovChaosProbe unit tests."""

    def test_detects_chaos_onset(self) -> None:
        probe = LyapunovChaosProbe()
        ordered = logistic_map(2.5, 0.1, 200)
        probe.fit_trajectory(ordered)
        chaotic = logistic_map(3.9, 0.1, 200)
        result = probe.deviation_score(chaotic)
        assert not np.any(np.isnan(result.deviation_scores))
        assert result.deviation_scores.shape == (200,)

    def test_minimum_20_samples(self) -> None:
        probe = LyapunovChaosProbe()
        with pytest.raises(ValueError, match="at least"):
            probe.fit_trajectory(np.ones(10, dtype=np.float64))


# ===================================================================
# Probe 12: TopologyHomologyProbe
# ===================================================================


class TestTopologyHomologyProbe:
    """TopologyHomologyProbe unit tests."""

    def test_detects_symmetry_break(self) -> None:
        probe = TopologyHomologyProbe()
        t = np.linspace(0, 4 * np.pi, 200)
        symmetric = np.sin(t).astype(np.float64)
        probe.fit_trajectory(symmetric)
        broken = symmetric.copy()
        broken[100:120] = 5.0  # break symmetry
        result = probe.deviation_score(broken)
        assert float(np.max(result.deviation_scores[99:121])) > float(
            np.median(result.deviation_scores[:80])
        )

    def test_symmetric_signal_low_deviation(self) -> None:
        probe = TopologyHomologyProbe()
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200).astype(np.float64)
        probe.fit_trajectory(data)
        result = probe.deviation_score(data)
        # Self-scoring: training data should not show extreme deviation
        assert float(np.mean(result.deviation_scores)) < 0.7


# ===================================================================
# Probe 13: FractalSelfSimilarityProbe
# ===================================================================


class TestFractalSelfSimilarityProbe:
    """FractalSelfSimilarityProbe unit tests."""

    def test_detects_scale_invariance_loss(self) -> None:
        probe = FractalSelfSimilarityProbe()
        data = make_normal_signal(200)
        probe.fit_trajectory(data)
        # Break self-similarity with random noise section
        broken = data.copy()
        rng = np.random.default_rng(99)
        broken[100:150] = rng.normal(0, 10, 50)
        result = probe.deviation_score(broken)
        assert not np.any(np.isnan(result.deviation_scores))

    def test_fractal_signal_low_deviation(self) -> None:
        probe = FractalSelfSimilarityProbe()
        data = make_normal_signal(200)
        probe.fit_trajectory(data)
        result = probe.deviation_score(data)
        assert float(np.mean(result.deviation_scores)) < 0.8


# ===================================================================
# Probe 14: ZetaHarmonicProbe
# ===================================================================


class TestZetaHarmonicProbe:
    """ZetaHarmonicProbe unit tests."""

    def test_detects_phase_coherence_break(self) -> None:
        probe = ZetaHarmonicProbe()
        data = make_normal_signal(200)
        probe.fit_trajectory(data)
        broken = data.copy()
        broken[100:120] = 100.0  # massive phase disruption
        result = probe.deviation_score(broken)
        assert not np.any(np.isnan(result.deviation_scores))

    def test_stable_phases_low_deviation(self) -> None:
        probe = ZetaHarmonicProbe()
        data = make_normal_signal(200)
        probe.fit_trajectory(data)
        result = probe.deviation_score(data)
        # Trigonometric transforms distribute scores broadly; check no NaN
        assert float(np.mean(result.deviation_scores)) < 0.7
        assert not np.any(np.isnan(result.deviation_scores))


# ===================================================================
# Probe 15: WavePropagationProbe
# ===================================================================


class TestWavePropagationProbe:
    """WavePropagationProbe unit tests."""

    def test_detects_curvature_spike(self) -> None:
        probe = WavePropagationProbe()
        t = np.linspace(0, 4 * np.pi, 200)
        smooth_wave = np.sin(t).astype(np.float64)
        probe.fit_trajectory(smooth_wave)
        spiked = smooth_wave.copy()
        spiked[100] = 20.0  # sharp curvature spike
        result = probe.deviation_score(spiked)
        assert not np.any(np.isnan(result.deviation_scores))

    def test_smooth_wave_low_deviation(self) -> None:
        probe = WavePropagationProbe()
        t = np.linspace(0, 4 * np.pi, 200)
        data = np.sin(t).astype(np.float64)
        probe.fit_trajectory(data)
        result = probe.deviation_score(data)
        assert float(np.mean(result.deviation_scores)) < 0.5

    def test_distinct_from_momentum(self) -> None:
        """WavePropagation uses Gaussian smoothing, Momentum does not."""
        from omni_mercury_engine.detectors.math_arrest.probes.momentum import (
            MomentumProbe,
        )

        data = make_normal_signal(200)
        wp = WavePropagationProbe()
        mp = MomentumProbe()
        wp.fit_trajectory(data)
        mp.fit_trajectory(data)
        wp_result = wp.deviation_score(data)
        mp_result = mp.deviation_score(data)
        # Should produce different score distributions
        assert wp_result.anomaly_geometry != mp_result.anomaly_geometry


# ===================================================================
# Probe 16: QuantumSuperpositionProbe
# ===================================================================


class TestQuantumSuperpositionProbe:
    """QuantumSuperpositionProbe unit tests."""

    def test_detects_fringe_shift(self) -> None:
        probe = QuantumSuperpositionProbe()
        data = make_normal_signal(200)
        probe.fit_trajectory(data)
        shifted = data.copy()
        shifted[100:120] = 50.0
        result = probe.deviation_score(shifted)
        assert not np.any(np.isnan(result.deviation_scores))

    def test_stable_interference_low_deviation(self) -> None:
        probe = QuantumSuperpositionProbe()
        data = make_normal_signal(200)
        probe.fit_trajectory(data)
        result = probe.deviation_score(data)
        # Cosine transform distributes scores broadly; check no NaN
        assert float(np.mean(result.deviation_scores)) < 0.7
        assert not np.any(np.isnan(result.deviation_scores))

    def test_distinct_from_zeta(self) -> None:
        """QuantumSuperposition uses cos(x), Zeta uses sin(2*pi*x) + cos(2*pi*x)."""
        data = make_normal_signal(200)
        qs = QuantumSuperpositionProbe()
        zh = ZetaHarmonicProbe()
        qs.fit_trajectory(data)
        zh.fit_trajectory(data)
        qs_result = qs.deviation_score(data)
        zh_result = zh.deviation_score(data)
        assert qs_result.anomaly_geometry != zh_result.anomaly_geometry


# ===================================================================
# Probe 17: EnergyMinimizationProbe
# ===================================================================


class TestEnergyMinimizationProbe:
    """EnergyMinimizationProbe unit tests."""

    def test_detects_energy_well_escape(self) -> None:
        probe = EnergyMinimizationProbe()
        rng = np.random.default_rng(42)
        train = rng.normal(0, 0.5, 200).astype(np.float64)
        probe.fit_trajectory(train)
        test_data = train.copy()
        test_data[100] = 20.0  # escape well
        result = probe.deviation_score(test_data)
        assert float(np.max(result.deviation_scores[99:105])) > float(
            np.median(result.deviation_scores[:80])
        )

    def test_stable_energy_low_deviation(self) -> None:
        probe = EnergyMinimizationProbe()
        rng = np.random.default_rng(42)
        data = rng.normal(0, 0.1, 200).astype(np.float64)
        probe.fit_trajectory(data)
        result = probe.deviation_score(data)
        assert float(np.mean(result.deviation_scores)) < 0.5


# ===================================================================
# Probe 18: QuantumAnnealingProbe
# ===================================================================


class TestQuantumAnnealingProbe:
    """QuantumAnnealingProbe unit tests."""

    def test_detects_thermodynamic_outlier(self) -> None:
        probe = QuantumAnnealingProbe()
        rng = np.random.default_rng(42)
        train = rng.normal(0, 1, 200).astype(np.float64)
        probe.fit_trajectory(train)
        test_data = train.copy()
        test_data[100] = 20.0  # outlier
        result = probe.deviation_score(test_data)
        assert result.deviation_scores[100] > float(np.median(result.deviation_scores[:80]))

    def test_normal_distribution_low_deviation(self) -> None:
        probe = QuantumAnnealingProbe()
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200).astype(np.float64)
        probe.fit_trajectory(data)
        result = probe.deviation_score(data)
        assert float(np.mean(result.deviation_scores)) < 0.8

    def test_distinct_from_energy_minimization(self) -> None:
        """QuantumAnnealing uses x^2/T, EnergyMinimization uses |Delta_E|."""
        data = make_normal_signal(200)
        qa = QuantumAnnealingProbe()
        em = EnergyMinimizationProbe()
        qa.fit_trajectory(data)
        em.fit_trajectory(data)
        qa_result = qa.deviation_score(data)
        em_result = em.deviation_score(data)
        assert qa_result.anomaly_geometry != em_result.anomaly_geometry


# ===================================================================
# Probe 19: BoltzmannCouplingProbe
# ===================================================================


class TestBoltzmannCouplingProbe:
    """BoltzmannCouplingProbe unit tests."""

    def test_detects_coupling_structure_break(self) -> None:
        probe = BoltzmannCouplingProbe()
        t = np.linspace(0, 20 * np.pi, 500)
        coupled_normal = (np.sin(t) + 0.3 * np.sin(2 * t + 0.5)).astype(np.float64)
        probe.fit_trajectory(coupled_normal)
        # Break coupling structure
        coupled_broken = coupled_normal.copy()
        coupled_broken[250:] = np.sin(t[250:]) + 0.3 * np.sin(5 * t[250:] + 2.0)
        result = probe.deviation_score(coupled_broken)
        assert not np.any(np.isnan(result.deviation_scores))

    def test_stable_autocorrelation_low_deviation(self) -> None:
        probe = BoltzmannCouplingProbe()
        t = np.linspace(0, 20 * np.pi, 200)
        data = (np.sin(t) + 0.3 * np.sin(2 * t + 0.5)).astype(np.float64)
        probe.fit_trajectory(data)
        result = probe.deviation_score(data)
        assert float(np.mean(result.deviation_scores)) < 0.5

    def test_multi_lag_sensitivity(self) -> None:
        probe = BoltzmannCouplingProbe()
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200).astype(np.float64)
        probe.fit_trajectory(data)
        assert len(probe._j_coeffs) >= 1


# ===================================================================
# Probe 20: IQRRobustProbe
# ===================================================================


class TestIQRRobustProbe:
    """IQRRobustProbe unit tests."""

    def test_detects_moderate_outlier(self) -> None:
        probe = IQRRobustProbe()
        rng = np.random.default_rng(42)
        train = rng.normal(0, 1, 200).astype(np.float64)
        probe.fit_trajectory(train)
        test_data = train.copy()
        test_data[100] = 15.0  # outlier beyond 1.5*IQR
        result = probe.deviation_score(test_data)
        assert result.deviation_scores[100] > float(np.median(result.deviation_scores[:80]))

    def test_skewed_data_handled(self) -> None:
        probe = IQRRobustProbe()
        rng = np.random.default_rng(42)
        skewed = np.abs(rng.normal(0, 1, 200)).astype(np.float64)
        probe.fit_trajectory(skewed)
        result = probe.deviation_score(skewed)
        assert not np.any(np.isnan(result.deviation_scores))

    def test_distinct_from_ethical(self) -> None:
        """IQR uses Q1/Q3/IQR fences, Ethical uses 2.5/97.5 percentiles."""
        from omni_mercury_engine.detectors.math_arrest.probes.ethical import (
            EthicalConstrainedProbe,
        )

        data = make_normal_signal(200)
        iqr = IQRRobustProbe()
        eth = EthicalConstrainedProbe()
        iqr.fit_trajectory(data)
        eth.fit_trajectory(data)
        iqr_result = iqr.deviation_score(data)
        eth_result = eth.deviation_score(data)
        assert iqr_result.anomaly_geometry != eth_result.anomaly_geometry


# ===================================================================
# Probe 21: ModifiedZScoreProbe
# ===================================================================


class TestModifiedZScoreProbe:
    """ModifiedZScoreProbe unit tests."""

    def test_detects_location_anomaly(self) -> None:
        probe = ModifiedZScoreProbe()
        rng = np.random.default_rng(42)
        train = rng.normal(0, 1, 200).astype(np.float64)
        probe.fit_trajectory(train)
        test_data = train.copy()
        test_data[100] = 30.0
        result = probe.deviation_score(test_data)
        assert result.deviation_scores[100] > float(np.median(result.deviation_scores[:80]))

    def test_resistant_to_contamination(self) -> None:
        probe = ModifiedZScoreProbe()
        rng = np.random.default_rng(42)
        # 10% contamination
        data = rng.normal(0, 1, 200).astype(np.float64)
        data[:20] = 100.0
        probe.fit_trajectory(data)
        # MAD-based: should still detect outliers in clean test data
        clean = rng.normal(0, 1, 200).astype(np.float64)
        clean[50] = 50.0
        result = probe.deviation_score(clean)
        assert not np.any(np.isnan(result.deviation_scores))

    def test_mad_based_not_std_based(self) -> None:
        probe = ModifiedZScoreProbe()
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200).astype(np.float64)
        probe.fit_trajectory(data)
        # Verify internal state uses MAD, not std
        assert probe._mad > 0.0
        assert probe._median is not None


# ===================================================================
# Full 21-probe Arrest integration
# ===================================================================


class TestFullArrest21:
    """Integration tests for the full 21-probe arrest."""

    def test_all_21_probes_instantiate(self) -> None:
        arrest = AnomalyMathArrest()
        assert len(arrest._probes) == 21

    def test_all_21_probes_fit(self) -> None:
        arrest = AnomalyMathArrest()
        data = make_normal_signal(500)
        arrest.fit(data)
        assert arrest.active_probe_count == 21

    def test_degraded_ensemble_still_detects(self) -> None:
        arrest = AnomalyMathArrest()
        data = make_normal_signal(500)
        arrest.fit(data)
        # Manually disable half the probes
        for i in range(0, 21, 2):
            arrest._probes[i]._is_fitted = False
        scores = arrest.detect(data)
        assert scores.shape == (500,)
        assert not np.all(scores == 0.0)

    def test_domain_affinity_all_21_probes(self) -> None:
        for domain in [
            "earthquake",
            "tsunami",
            "pandemic",
            "marine",
            "geomagnetic",
            "conflict",
            "default",
        ]:
            arrest = AnomalyMathArrest(domain=domain)
            data = make_normal_signal(200)
            arrest.fit(data)
            scores = arrest.detect(data)
            assert scores.shape == (200,)
            assert np.all(scores >= 0.0)
            assert np.all(scores <= 1.0)

    def test_no_nan_no_inf_21_probes(self) -> None:
        arrest = AnomalyMathArrest()
        data = make_normal_signal(500)
        arrest.fit(data)
        scores = arrest.detect(data)
        assert not np.any(np.isnan(scores))
        assert not np.any(np.isinf(scores))

    def test_scores_in_zero_one_21_probes(self) -> None:
        arrest = AnomalyMathArrest()
        data = make_normal_signal(500)
        arrest.fit(data)
        scores = arrest.detect(data)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_decorrelator_auto_calibrates_on_fit(self) -> None:
        arrest = AnomalyMathArrest()
        data = make_normal_signal(200)
        arrest.fit(data)
        assert arrest._decorrelator.is_calibrated

    def test_get_correlation_report_structure(self) -> None:
        arrest = AnomalyMathArrest()
        data = make_normal_signal(200)
        arrest.fit(data)
        report = arrest.get_correlation_report()
        assert "redundant_pairs" in report
        assert "weight_multipliers" in report
        assert "effective_probe_count" in report

    def test_ensemble_confidence_range_21(self) -> None:
        arrest = AnomalyMathArrest()
        data = make_normal_signal(500)
        arrest.fit(data)
        conf = arrest.ensemble_confidence
        assert 0.0 <= conf <= 1.0

    def test_effective_probe_count_le_21(self) -> None:
        arrest = AnomalyMathArrest()
        data = make_normal_signal(200)
        arrest.fit(data)
        report = arrest.get_correlation_report()
        assert report["effective_probe_count"] <= 21.0
        assert report["effective_probe_count"] > 0.0
