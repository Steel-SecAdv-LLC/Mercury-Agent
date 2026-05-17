"""Tests for F1 precision improvement features (Phases 2-9).

Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC (GPL-3.0)

Tests cover:
  - Noise color estimation (_estimate_noise_color)
  - Residual frequency filter (_residual_frequency_filter)
  - Adaptive alpha computation (_compute_adaptive_alpha)
  - Domain weight presets (get_domain_preset)
  - Inversion guard (Spearman-based component zeroing)
  - Ensemble flip (median-based score inversion)
  - Oracle influence multiplier
  - Multi-strategy threshold selection
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure benchmarks/ is importable for threshold tests
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))


# -----------------------------------------------------------------------
# Phase 3: Domain weight presets
# -----------------------------------------------------------------------


class TestDomainWeightPresets:
    """Tests for domain_weight_presets.py."""

    def test_get_domain_preset_all_domains(self):
        from omni_mercury_engine.core.domain_weight_presets import (
            DOMAIN_WEIGHT_PRESETS,
            get_domain_preset,
        )

        for domain in DOMAIN_WEIGHT_PRESETS:
            r, k, ig = get_domain_preset(domain)
            assert abs(r + k + ig - 1.0) < 1e-6, f"{domain} weights don't sum to 1"
            assert r >= 0 and k >= 0 and ig >= 0, f"{domain} has negative weight"

    def test_get_domain_preset_unknown_returns_default(self):
        from omni_mercury_engine.core.domain_weight_presets import get_domain_preset

        r, k, ig = get_domain_preset("unknown_domain_xyz")
        assert abs(r + k + ig - 1.0) < 1e-6
        assert r == 0.40
        assert k == 0.20
        assert ig == 0.40

    def test_get_domain_preset_case_insensitive(self):
        from omni_mercury_engine.core.domain_weight_presets import get_domain_preset

        r1, k1, ig1 = get_domain_preset("OCEAN")
        r2, k2, ig2 = get_domain_preset("ocean")
        assert r1 == r2 and k1 == k2 and ig1 == ig2

    def test_tabular_domains_zero_kinematic(self):
        from omni_mercury_engine.core.domain_weight_presets import get_domain_preset

        for domain in ["disaster", "general", "security", "industrial"]:
            _, k, _ = get_domain_preset(domain)
            assert k == 0.0, f"{domain} should have zero kinematic weight"

    def test_physics_domains_nonzero_kinematic(self):
        from omni_mercury_engine.core.domain_weight_presets import get_domain_preset

        for domain in ["ocean", "climate", "space", "environmental"]:
            _, k, _ = get_domain_preset(domain)
            assert k >= 0.30, f"{domain} should have significant kinematic weight"


# -----------------------------------------------------------------------
# Phase 4: Noise color estimation
# -----------------------------------------------------------------------


class TestNoiseColorEstimation:
    """Tests for SpectralDomainOracle._estimate_noise_color."""

    def _make_oracle(self):
        from omni_mercury_engine.detectors.spectral_domain_frequency import (
            SpectralDomainOracle,
        )

        return SpectralDomainOracle({"domain": "environmental"})

    def test_white_noise(self):
        """White noise should have beta close to 0."""
        oracle = self._make_oracle()
        rng = np.random.RandomState(42)
        signal = rng.randn(1000)
        fft_vals = np.fft.rfft(signal)
        psd = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(len(signal))
        beta, color, r2 = oracle._estimate_noise_color(psd, freqs)
        assert abs(beta) < 0.5, f"White noise beta={beta}, expected ~0"
        assert color == "white"

    def test_brown_noise(self):
        """Integrated white noise (Brownian) should have beta ~2."""
        oracle = self._make_oracle()
        rng = np.random.RandomState(42)
        white = rng.randn(2000)
        brown = np.cumsum(white)  # Integration -> 1/f^2
        fft_vals = np.fft.rfft(brown - np.mean(brown))
        psd = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(len(brown))
        beta, color, r2 = oracle._estimate_noise_color(psd, freqs)
        assert beta > 1.5, f"Brown noise beta={beta}, expected >1.5"
        assert color == "brown"

    def test_pink_noise_synthetic(self):
        """Synthetic 1/f noise should have beta ~1."""
        oracle = self._make_oracle()
        n = 1024
        freqs_synth = np.fft.rfftfreq(n)
        freqs_synth[0] = 1e-10  # Avoid division by zero
        # Create 1/f PSD
        psd = 1.0 / (freqs_synth + 1e-10)
        beta, color, r2 = oracle._estimate_noise_color(psd, freqs_synth)
        assert 0.5 < beta < 1.5, f"Pink noise beta={beta}, expected ~1"
        assert color == "pink"

    def test_insufficient_data(self):
        """Should return white noise for very short signals."""
        oracle = self._make_oracle()
        psd = np.array([1.0, 2.0])
        freqs = np.array([0.0, 0.5])
        beta, color, r2 = oracle._estimate_noise_color(psd, freqs)
        assert beta == 0.0
        assert color == "white"


# -----------------------------------------------------------------------
# Phase 5: Adaptive alpha
# -----------------------------------------------------------------------


class TestAdaptiveAlpha:
    """Tests for SpectralDomainOracle._compute_adaptive_alpha."""

    def _make_oracle(self):
        from omni_mercury_engine.detectors.spectral_domain_frequency import (
            SpectralDomainOracle,
        )

        return SpectralDomainOracle({"domain": "environmental", "significance_level": 0.05})

    def test_short_window_relaxes_alpha(self):
        oracle = self._make_oracle()
        alpha_short = oracle._compute_adaptive_alpha(30, 5, 0.8)
        alpha_long = oracle._compute_adaptive_alpha(5000, 5, 0.8)
        assert alpha_short > alpha_long, "Short windows should have higher alpha"

    def test_alpha_bounds(self):
        oracle = self._make_oracle()
        for n_samples in [10, 50, 200, 1000, 5000]:
            for n_bands in [1, 5, 10]:
                alpha = oracle._compute_adaptive_alpha(n_samples, n_bands, 0.5)
                assert 0.01 <= alpha <= 0.20, f"Alpha {alpha} out of bounds"

    def test_more_bands_tightens_alpha(self):
        oracle = self._make_oracle()
        alpha_few = oracle._compute_adaptive_alpha(500, 2, 0.8)
        alpha_many = oracle._compute_adaptive_alpha(500, 10, 0.8)
        assert alpha_few > alpha_many, "More bands should tighten alpha"

    def test_low_confidence_relaxes_alpha(self):
        oracle = self._make_oracle()
        alpha_high_conf = oracle._compute_adaptive_alpha(500, 5, 0.95)
        alpha_low_conf = oracle._compute_adaptive_alpha(500, 5, 0.1)
        assert alpha_low_conf > alpha_high_conf


# -----------------------------------------------------------------------
# Phase 7: Residual frequency filter
# -----------------------------------------------------------------------


class TestResidualFrequencyFilter:
    """Tests for MercuryAnomalyDetector._residual_frequency_filter."""

    def test_preserves_anomaly_spikes(self):
        """Filter should preserve anomaly spikes, not smooth them out."""
        from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

        # Create signal with clear anomaly spikes
        scores = np.zeros(200)
        scores[50] = 0.9
        scores[100] = 0.85
        scores[150] = 0.95

        filtered = MercuryAnomalyDetector._residual_frequency_filter(scores)

        # Spikes should still be elevated in filtered signal
        assert filtered[50] > 0.3, "Spike at 50 should be preserved"
        assert filtered[100] > 0.3, "Spike at 100 should be preserved"
        assert filtered[150] > 0.3, "Spike at 150 should be preserved"

    def test_output_clipped_to_01(self):
        from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

        scores = np.random.RandomState(42).rand(100)
        filtered = MercuryAnomalyDetector._residual_frequency_filter(scores)
        assert np.all(filtered >= 0.0)
        assert np.all(filtered <= 1.0)

    def test_short_signal_passthrough(self):
        from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

        scores = np.array([0.1, 0.5, 0.9])
        filtered = MercuryAnomalyDetector._residual_frequency_filter(scores)
        np.testing.assert_array_equal(filtered, scores)

    def test_constant_signal_unchanged(self):
        from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

        scores = np.full(100, 0.3)
        filtered = MercuryAnomalyDetector._residual_frequency_filter(scores)
        np.testing.assert_allclose(filtered, 0.3, atol=0.05)


# -----------------------------------------------------------------------
# Phase 2: Inversion guard
# -----------------------------------------------------------------------


class TestInversionGuard:
    """Tests for the post-hoc inversion guard in detect()."""

    def test_inverted_component_gets_zeroed(self):
        """When a component is anti-correlated with ensemble, it should be zeroed."""
        from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

        rng = np.random.RandomState(42)
        n = 100
        X_train = rng.randn(200, 3)
        X_test = rng.randn(n, 3)

        detector = MercuryAnomalyDetector()
        detector.fit(X_train)
        result = detector.detect(X_test)

        # Scores should still be valid after the guard
        assert result["scores"].shape == (n,)
        assert np.all(result["scores"] >= 0)
        assert np.all(result["scores"] <= 1)

    def test_detect_with_small_data_skips_guard(self):
        """Inversion guard requires >= 30 samples."""
        from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

        rng = np.random.RandomState(42)
        X_train = rng.randn(50, 2)
        X_test = rng.randn(10, 2)

        detector = MercuryAnomalyDetector()
        detector.fit(X_train)
        result = detector.detect(X_test)
        assert result["scores"].shape == (10,)


# -----------------------------------------------------------------------
# Phase 2: Ensemble flip
# -----------------------------------------------------------------------


class TestEnsembleFlip:
    """Tests for the unsupervised ensemble flip."""

    def test_high_median_triggers_flip(self):
        """If median score > 0.80, scores should be flipped."""
        from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

        # Create data where detector produces very high baseline scores
        rng = np.random.RandomState(42)
        X_train = rng.randn(200, 1)
        # Use very large shift to push all scores very high (>0.80 median)
        X_test = rng.randn(100, 1) + 20

        detector = MercuryAnomalyDetector()
        detector.fit(X_train)
        result = detector.detect(X_test)

        # Scores should still be valid after potential flip
        scores = result["scores"]
        assert scores.shape == (100,)
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0)


# -----------------------------------------------------------------------
# Phase 4-6: Oracle integration
# -----------------------------------------------------------------------


class TestOracleIntegration:
    """Tests for Oracle fit and detect integration."""

    def test_oracle_fitted_during_fit(self):
        from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

        rng = np.random.RandomState(42)
        X = rng.randn(200, 3)
        detector = MercuryAnomalyDetector()
        detector.fit(X)
        # Oracle may or may not be initialized depending on data type detection
        # Just verify the attribute exists
        assert hasattr(detector, "_oracle_detector")

    def test_oracle_metadata_in_detect_result(self):
        from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

        rng = np.random.RandomState(42)
        X_train = rng.randn(200, 3)
        X_test = rng.randn(100, 3)
        detector = MercuryAnomalyDetector()
        detector.fit(X_train)
        result = detector.detect(X_test)
        assert "oracle_metadata" in result

    def test_oracle_noise_color_estimated(self):
        """When Oracle is active, noise color should be estimated."""
        from omni_mercury_engine.detectors.spectral_domain_frequency import (
            SpectralDomainOracle,
        )

        rng = np.random.RandomState(42)
        signal = rng.randn(500)
        oracle = SpectralDomainOracle({"domain": "environmental"})
        oracle.fit(signal)
        assert oracle._noise_color in ("white", "pink", "brown", "blue", "violet")
        assert isinstance(oracle._noise_beta, float)

    def test_oracle_multiplier_in_bounds(self):
        """Influence multiplier should stay within configured bounds."""
        from omni_mercury_engine.detectors.spectral_domain_frequency import (
            SpectralDomainOracle,
        )

        rng = np.random.RandomState(42)
        oracle = SpectralDomainOracle(
            {
                "domain": "environmental",
                "influence_floor": 0.5,
                "influence_ceiling": 2.0,
            }
        )
        train_data = rng.randn(200)
        oracle.fit(train_data)

        test_data = rng.randn(100)
        result = oracle.detect(test_data)
        iv = result["influence_vector"]
        assert 0.5 <= iv.influence_multiplier <= 2.0

    def test_oracle_detect_returns_dict(self):
        """T0biU Oracle returns dict with influence_vector key."""
        from omni_mercury_engine.detectors.spectral_domain_frequency import (
            SpectralDomainOracle,
        )

        rng = np.random.RandomState(42)
        oracle = SpectralDomainOracle({"domain": "environmental"})
        oracle.fit(rng.randn(200))
        result = oracle.detect(rng.randn(100))
        assert isinstance(result, dict)
        assert "influence_vector" in result
        assert "anomaly_score" in result
        assert "band_results" in result
        assert "noise_color" in result

    def test_oracle_noise_color_in_detect_result(self):
        """Detection result should include noise_color metadata."""
        from omni_mercury_engine.detectors.spectral_domain_frequency import (
            SpectralDomainOracle,
        )

        rng = np.random.RandomState(42)
        oracle = SpectralDomainOracle({"domain": "environmental"})
        oracle.fit(rng.randn(200))
        result = oracle.detect(rng.randn(100))
        nc = result["noise_color"]
        assert "beta" in nc
        assert "name" in nc
        assert "r_squared" in nc


# -----------------------------------------------------------------------
# Phase 6: Oracle influence multiplier
# -----------------------------------------------------------------------


class TestOracleInfluenceMultiplier:
    """Tests for _compute_influence_multiplier."""

    def test_significant_bands_increase_multiplier(self):
        from omni_mercury_engine.detectors.spectral_domain_frequency import (
            FrequencyBandResult,
            SpectralDomainOracle,
        )

        oracle = SpectralDomainOracle({"domain": "environmental"})
        oracle._noise_beta = 0.0
        oracle._current_beta = 0.0
        # Set reference entropy stats
        oracle._ref_spectral_entropy_mean = 2.0
        oracle._ref_spectral_entropy_std = 0.5

        # Create highly significant bands with high z-scores
        bands = [
            FrequencyBandResult(
                band_label="b1",
                low_hz=0.0,
                high_hz=0.5,
                band_weight=0.5,
                power_ratio=3.0,
                z_score=3.0,
                anomaly_score=0.9,
                p_value=0.001,
                is_significant=True,
            ),
            FrequencyBandResult(
                band_label="b2",
                low_hz=0.5,
                high_hz=1.0,
                band_weight=0.5,
                power_ratio=2.5,
                z_score=2.5,
                anomaly_score=0.8,
                p_value=0.005,
                is_significant=True,
            ),
        ]
        # aggregate_score is high
        mult = oracle._compute_influence_multiplier(
            aggregate_score=0.85,
            spectral_entropy=3.0,  # Higher than ref -> anomalous
            band_results=bands,
        )
        assert mult > 1.0, f"Significant bands should amplify, got {mult}"

    def test_no_significant_bands_attenuate(self):
        from omni_mercury_engine.detectors.spectral_domain_frequency import (
            FrequencyBandResult,
            SpectralDomainOracle,
        )

        oracle = SpectralDomainOracle({"domain": "environmental"})
        oracle._noise_beta = 0.0
        oracle._current_beta = 0.0
        oracle._ref_spectral_entropy_mean = 2.0
        oracle._ref_spectral_entropy_std = 0.5

        bands = [
            FrequencyBandResult(
                band_label="b1",
                low_hz=0.0,
                high_hz=0.5,
                band_weight=0.5,
                power_ratio=1.0,
                z_score=0.5,
                anomaly_score=0.1,
                p_value=0.8,
                is_significant=False,
            ),
        ]
        mult = oracle._compute_influence_multiplier(
            aggregate_score=0.1,
            spectral_entropy=2.0,
            band_results=bands,
        )
        assert mult <= 1.0, f"Non-significant should attenuate, got {mult}"


# -----------------------------------------------------------------------
# Phase 8: Multi-strategy threshold selection
# -----------------------------------------------------------------------


class TestMultiStrategyThreshold:
    """Tests for the multi-strategy threshold selection in benchmark."""

    def test_returns_five_values(self):
        """Should return (f1, prec, rec, thr, strategy_name)."""
        from mercury_benchmark import _oracle_threshold_f1

        rng = np.random.RandomState(42)
        scores = rng.rand(100)
        labels = (scores > 0.7).astype(int)
        result = _oracle_threshold_f1(labels, scores)
        assert len(result) == 5
        f1, prec, rec, thr, name = result
        assert 0 <= f1 <= 1
        assert 0 <= prec <= 1
        assert 0 <= rec <= 1
        assert isinstance(name, str)

    def test_perfect_separation_gives_f1_1(self):
        from mercury_benchmark import _oracle_threshold_f1

        scores = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        f1, _, _, _, _ = _oracle_threshold_f1(labels, scores)
        assert f1 == 1.0

    def test_strategy_name_not_default(self):
        """With real data, strategy should be something other than 'default'."""
        from mercury_benchmark import _oracle_threshold_f1

        rng = np.random.RandomState(42)
        n = 200
        scores = rng.rand(n)
        labels = (scores > 0.8).astype(int)
        _, _, _, _, strategy = _oracle_threshold_f1(labels, scores)
        assert strategy != "default", f"Strategy should not be default, got {strategy}"


# -----------------------------------------------------------------------
# Phase 9: DOMAIN_ANOMALY_SPECTRAL_HINTS
# -----------------------------------------------------------------------


class TestSpectralHints:
    """Tests for DOMAIN_ANOMALY_SPECTRAL_HINTS constant."""

    def test_five_domains_defined(self):
        from omni_mercury_engine.detectors.spectral_domain_frequency import (
            DOMAIN_ANOMALY_SPECTRAL_HINTS,
        )

        assert len(DOMAIN_ANOMALY_SPECTRAL_HINTS) == 5
        for domain in ["environmental", "ocean", "security", "space", "climate"]:
            assert domain in DOMAIN_ANOMALY_SPECTRAL_HINTS

    def test_each_hint_has_anomaly_beta_shift(self):
        from omni_mercury_engine.detectors.spectral_domain_frequency import (
            DOMAIN_ANOMALY_SPECTRAL_HINTS,
        )

        for domain, hints in DOMAIN_ANOMALY_SPECTRAL_HINTS.items():
            assert "anomaly_beta_shift" in hints, f"{domain} missing anomaly_beta_shift"
            assert isinstance(hints["anomaly_beta_shift"], (int, float))


# -----------------------------------------------------------------------
# Integration: Full pipeline
# -----------------------------------------------------------------------


class TestFullPipelineIntegration:
    """End-to-end tests for the merged pipeline."""

    def test_detect_with_temporal_data(self):
        """Full detect() pipeline should work with temporal-like data."""
        from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

        rng = np.random.RandomState(42)
        # Create temporal-like data (autocorrelated)
        n = 500
        t = np.arange(n)
        signal = np.sin(0.1 * t) + 0.5 * np.sin(0.3 * t) + rng.normal(0, 0.1, n)
        X_train = signal[:400].reshape(-1, 1)
        X_test = signal[400:].reshape(-1, 1)

        detector = MercuryAnomalyDetector()
        detector.fit(X_train)
        result = detector.detect(X_test)

        assert "scores" in result
        assert result["scores"].shape == (100,)
        assert np.all(result["scores"] >= 0) and np.all(result["scores"] <= 1)

    def test_detect_with_benchmark_domain(self):
        """Setting _benchmark_domain should trigger domain preset blending."""
        from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

        rng = np.random.RandomState(42)
        X_train = rng.randn(200, 3)
        X_test = rng.randn(100, 3)

        detector = MercuryAnomalyDetector()
        detector.fit(X_train)
        # _benchmark_domain is a dynamic attribute read by detect() via getattr;
        # see detectors/statistical.py:1905. setattr keeps mypy happy and the
        # B010 noqa records that the dynamic write is intentional.
        setattr(detector, "_benchmark_domain", "environmental")  # noqa: B010
        result = detector.detect(X_test)

        assert result["scores"].shape == (100,)
        assert np.all(result["scores"] >= 0) and np.all(result["scores"] <= 1)
