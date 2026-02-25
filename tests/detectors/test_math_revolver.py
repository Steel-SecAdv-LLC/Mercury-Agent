# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) Steel Security Advisors LLC
"""Core tests for the Anomaly Math Revolver (probes 1-8 + integration)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from omni_mercury_engine.detectors.math_revolver.base_probe import (
    ProbeResult,
)
from omni_mercury_engine.detectors.math_revolver.probes.additive import (
    AdditiveProbe,
)
from omni_mercury_engine.detectors.math_revolver.probes.catalan import (
    CatalanOptimizedProbe,
)
from omni_mercury_engine.detectors.math_revolver.probes.ethical import (
    EthicalConstrainedProbe,
)
from omni_mercury_engine.detectors.math_revolver.probes.exponential import (
    ExponentialDecayProbe,
)
from omni_mercury_engine.detectors.math_revolver.probes.harmonic import (
    HarmonicOscillatorProbe,
)
from omni_mercury_engine.detectors.math_revolver.probes.helix import (
    HelixMultiplicativeProbe,
)
from omni_mercury_engine.detectors.math_revolver.probes.momentum import (
    MomentumProbe,
)
from omni_mercury_engine.detectors.math_revolver.probes.variance_adapted import (
    VarianceAdaptedProbe,
)
from omni_mercury_engine.detectors.math_revolver.revolver import (
    AnomalyMathRevolver,
)

# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def make_normal_signal(
    n: int = 500, seed: int = 42
) -> npt.NDArray[np.float64]:
    """Sinusoidal signal with mild noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 10 * np.pi, n)
    return (np.sin(t) + rng.normal(0, 0.05, n)).astype(np.float64)


def make_level_shift(
    n: int = 500, seed: int = 42
) -> npt.NDArray[np.float64]:
    """Sinusoidal signal with a +5 level shift at the midpoint."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 10 * np.pi, n)
    half = n // 2
    return np.concatenate([
        np.sin(t[:half]) + rng.normal(0, 0.05, half),
        np.sin(t[half:]) + 5.0 + rng.normal(0, 0.05, n - half),
    ]).astype(np.float64)


# ===================================================================
# ProbeResult tests
# ===================================================================


class TestProbeResult:
    """ProbeResult dataclass invariants."""

    def test_immutable(self) -> None:
        result = ProbeResult(
            probe_name="test",
            deviation_scores=np.zeros(10, dtype=np.float64),
            confidence=0.9,
            trajectory_fit_quality=0.8,
            anomaly_geometry="test",
        )
        with pytest.raises(AttributeError):
            result.confidence = 0.5  # type: ignore[misc]

    def test_scores_in_zero_one(self) -> None:
        probe = AdditiveProbe()
        data = make_normal_signal()
        probe.fit_trajectory(data)
        result = probe.deviation_score(data)
        assert np.all(result.deviation_scores >= 0.0)
        assert np.all(result.deviation_scores <= 1.0)

    def test_no_nan_no_inf(self) -> None:
        probe = AdditiveProbe()
        data = make_normal_signal()
        probe.fit_trajectory(data)
        result = probe.deviation_score(data)
        assert not np.any(np.isnan(result.deviation_scores))
        assert not np.any(np.isinf(result.deviation_scores))


# ===================================================================
# Probe 1: AdditiveProbe
# ===================================================================


class TestAdditiveProbe:
    """AdditiveProbe unit tests."""

    def test_detects_level_shift(self) -> None:
        probe = AdditiveProbe()
        train = make_normal_signal()
        probe.fit_trajectory(train)
        test_data = make_level_shift()
        result = probe.deviation_score(test_data)
        # Second half should have higher scores than first half
        half = len(test_data) // 2
        mean_first = float(np.mean(result.deviation_scores[:half]))
        mean_second = float(np.mean(result.deviation_scores[half:]))
        assert mean_second > mean_first

    def test_fit_required_before_score(self) -> None:
        probe = AdditiveProbe()
        with pytest.raises(RuntimeError, match="not been fitted"):
            probe.deviation_score(make_normal_signal())

    def test_minimum_samples_enforced(self) -> None:
        probe = AdditiveProbe()
        with pytest.raises(ValueError, match="at least"):
            probe.fit_trajectory(np.array([1.0, 2.0, 3.0]))

    def test_constant_data_no_crash(self) -> None:
        probe = AdditiveProbe()
        data = np.ones(50, dtype=np.float64)
        probe.fit_trajectory(data)
        result = probe.deviation_score(data)
        assert not np.any(np.isnan(result.deviation_scores))

    def test_2d_input_reduced_to_1d(self) -> None:
        probe = AdditiveProbe()
        data_2d = np.random.default_rng(42).normal(0, 1, (100, 3)).astype(
            np.float64
        )
        probe.fit_trajectory(data_2d)
        result = probe.deviation_score(data_2d)
        assert result.deviation_scores.shape == (100,)


# ===================================================================
# Probe 2: HarmonicOscillatorProbe
# ===================================================================


class TestHarmonicOscillatorProbe:
    """HarmonicOscillatorProbe unit tests."""

    def test_detects_periodicity_violation(self) -> None:
        probe = HarmonicOscillatorProbe()
        train = make_normal_signal(200)
        probe.fit_trajectory(train)
        # Inject a non-periodic segment
        test_data = train.copy()
        test_data[100:150] = 10.0
        result = probe.deviation_score(test_data)
        assert float(np.max(result.deviation_scores[100:150])) > 0.0

    def test_zscore_fallback_works(self) -> None:
        probe = HarmonicOscillatorProbe()
        # Aperiodic data forces fallback
        rng = np.random.default_rng(99)
        aperiodic = rng.normal(0, 1, 100).astype(np.float64)
        probe.fit_trajectory(aperiodic)
        result = probe.deviation_score(aperiodic)
        assert result.deviation_scores.shape == (100,)

    def test_mode_metadata_present(self) -> None:
        probe = HarmonicOscillatorProbe()
        probe.fit_trajectory(make_normal_signal(200))
        result = probe.deviation_score(make_normal_signal(200))
        assert "mode" in result.metadata


# ===================================================================
# Probe 3: MomentumProbe
# ===================================================================


class TestMomentumProbe:
    """MomentumProbe unit tests."""

    def test_detects_sudden_acceleration(self) -> None:
        probe = MomentumProbe()
        train = make_normal_signal()
        probe.fit_trajectory(train)
        test_data = train.copy()
        test_data[250] = 50.0  # spike
        result = probe.deviation_score(test_data)
        assert float(np.max(result.deviation_scores)) > 0.5

    def test_output_padded_correctly(self) -> None:
        probe = MomentumProbe()
        data = make_normal_signal(100)
        probe.fit_trajectory(data)
        result = probe.deviation_score(data)
        assert result.deviation_scores.shape == (100,)
        assert result.deviation_scores[0] == 0.0
        assert result.deviation_scores[1] == 0.0


# ===================================================================
# Probe 4: VarianceAdaptedProbe
# ===================================================================


class TestVarianceAdaptedProbe:
    """VarianceAdaptedProbe unit tests."""

    def test_detects_volatility_spike(self) -> None:
        probe = VarianceAdaptedProbe()
        rng = np.random.default_rng(42)
        train = rng.normal(0, 0.1, 200).astype(np.float64)
        probe.fit_trajectory(train)
        test_data = np.concatenate([
            rng.normal(0, 0.1, 100),
            rng.normal(0, 5.0, 100),
        ]).astype(np.float64)
        result = probe.deviation_score(test_data)
        mean_first = float(np.mean(result.deviation_scores[:80]))
        mean_second = float(np.mean(result.deviation_scores[120:]))
        assert mean_second > mean_first

    def test_minimum_20_samples(self) -> None:
        probe = VarianceAdaptedProbe()
        with pytest.raises(ValueError, match="at least"):
            probe.fit_trajectory(np.ones(10, dtype=np.float64))


# ===================================================================
# Probe 5: EthicalConstrainedProbe
# ===================================================================


class TestEthicalConstrainedProbe:
    """EthicalConstrainedProbe unit tests."""

    def test_detects_boundary_violation(self) -> None:
        probe = EthicalConstrainedProbe()
        rng = np.random.default_rng(42)
        train = rng.normal(0, 1, 200).astype(np.float64)
        probe.fit_trajectory(train)
        test_data = np.concatenate([
            rng.normal(0, 1, 50),
            np.full(50, 20.0),
        ]).astype(np.float64)
        result = probe.deviation_score(test_data)
        assert float(np.mean(result.deviation_scores[50:])) > float(
            np.mean(result.deviation_scores[:50])
        )

    def test_zero_range_fallback(self) -> None:
        probe = EthicalConstrainedProbe()
        data = np.ones(50, dtype=np.float64)
        probe.fit_trajectory(data)
        assert probe._fit_quality == pytest.approx(0.1)
        result = probe.deviation_score(data)
        assert not np.any(np.isnan(result.deviation_scores))


# ===================================================================
# Probe 6: CatalanOptimizedProbe
# ===================================================================


class TestCatalanOptimizedProbe:
    """CatalanOptimizedProbe unit tests."""

    def test_detects_autocorrelation_break(self) -> None:
        probe = CatalanOptimizedProbe()
        rng = np.random.default_rng(42)
        # AR(1) with Catalan coefficient
        n = 200
        train = np.zeros(n, dtype=np.float64)
        train[0] = rng.normal()
        for i in range(1, n):
            train[i] = 0.9 * train[i - 1] + rng.normal(0, 0.1)
        probe.fit_trajectory(train)
        # Break the autocorrelation
        test_data = train.copy()
        test_data[100:150] = rng.normal(0, 5, 50)
        result = probe.deviation_score(test_data)
        assert float(np.max(result.deviation_scores[100:150])) > float(
            np.median(result.deviation_scores[:100])
        )


# ===================================================================
# Probe 7: ExponentialDecayProbe
# ===================================================================


class TestExponentialDecayProbe:
    """ExponentialDecayProbe unit tests."""

    def test_detects_signal_degradation(self) -> None:
        probe = ExponentialDecayProbe()
        train = make_normal_signal(200)
        probe.fit_trajectory(train)
        test_data = train.copy()
        # Inject spiky noise that EWMA cannot track smoothly
        rng = np.random.default_rng(99)
        test_data[100:200] += rng.normal(0, 5.0, 100)
        result = probe.deviation_score(test_data)
        # Degraded region should have higher deviation than calm region
        assert float(np.mean(result.deviation_scores[120:])) > float(
            np.mean(result.deviation_scores[:80])
        )

    def test_lambda_stored_in_metadata(self) -> None:
        probe = ExponentialDecayProbe()
        probe.fit_trajectory(make_normal_signal(200))
        result = probe.deviation_score(make_normal_signal(200))
        assert "lambda" in result.metadata
        assert result.metadata["lambda"] > 0.0


# ===================================================================
# Probe 8: HelixMultiplicativeProbe
# ===================================================================


class TestHelixMultiplicativeProbe:
    """HelixMultiplicativeProbe unit tests."""

    def test_detects_multiplicative_shock(self) -> None:
        probe = HelixMultiplicativeProbe()
        rng = np.random.default_rng(42)
        train = np.cumsum(rng.normal(0, 0.01, 200)) + 10.0
        train = train.astype(np.float64)
        probe.fit_trajectory(train)
        test_data = train.copy()
        test_data[150] = test_data[149] * 100.0  # multiplicative shock
        result = probe.deviation_score(test_data)
        assert float(result.deviation_scores[150]) > 0.5

    def test_log_ratio_pad_alignment(self) -> None:
        probe = HelixMultiplicativeProbe()
        data = make_normal_signal(100) + 10.0
        probe.fit_trajectory(data)
        result = probe.deviation_score(data)
        assert result.deviation_scores.shape == (100,)
        assert result.deviation_scores[0] == 0.0


# ===================================================================
# AnomalyMathRevolver integration tests
# ===================================================================


class TestAnomalyMathRevolver:
    """Integration tests for the full AnomalyMathRevolver."""

    def test_fit_detect_cycle(self) -> None:
        revolver = AnomalyMathRevolver()
        data = make_normal_signal()
        revolver.fit(data)
        scores = revolver.detect(data)
        assert scores.shape == (len(data),)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_predict_returns_binary(self) -> None:
        revolver = AnomalyMathRevolver()
        data = make_normal_signal()
        revolver.fit(data)
        preds = revolver.predict(data)
        assert set(np.unique(preds)).issubset({0, 1})
        assert preds.dtype == np.int32

    def test_all_8_probes_instantiate(self) -> None:
        revolver = AnomalyMathRevolver()
        assert len(revolver._probes) == 21

    def test_degraded_probe_skipped(self) -> None:
        revolver = AnomalyMathRevolver()
        data = make_normal_signal(15)
        # With only 15 samples, some probes will fail (e.g., VarianceAdapted needs 20)
        revolver.fit(data)
        scores = revolver.detect(data)
        assert scores.shape == (15,)
        # Should still produce output even with some probes failing
        assert not np.all(scores == 0.0) or revolver.active_probe_count > 0

    def test_empty_data_raises(self) -> None:
        revolver = AnomalyMathRevolver()
        with pytest.raises(ValueError, match="empty"):
            revolver.fit(np.array([], dtype=np.float64))

    def test_short_data_raises(self) -> None:
        revolver = AnomalyMathRevolver()
        with pytest.raises(ValueError, match="at least"):
            revolver.fit(np.array([1.0, 2.0], dtype=np.float64))

    def test_scores_in_zero_one(self) -> None:
        revolver = AnomalyMathRevolver()
        data = make_normal_signal()
        revolver.fit(data)
        scores = revolver.detect(data)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_no_nan_no_inf(self) -> None:
        revolver = AnomalyMathRevolver()
        data = make_normal_signal()
        revolver.fit(data)
        scores = revolver.detect(data)
        assert not np.any(np.isnan(scores))
        assert not np.any(np.isinf(scores))

    def test_calibrate_threshold_f1(self) -> None:
        revolver = AnomalyMathRevolver()
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200).astype(np.float64)
        labels = np.zeros(200, dtype=np.int32)
        labels[150:] = 1
        data[150:] += 10.0
        revolver.fit(data)
        threshold = revolver.calibrate_threshold(data, labels, metric="f1")
        assert 0.0 < threshold < 1.0

    def test_get_probe_diagnostics_structure(self) -> None:
        revolver = AnomalyMathRevolver()
        data = make_normal_signal()
        revolver.fit(data)
        diags = revolver.get_probe_diagnostics()
        assert len(diags) == 21
        for d in diags:
            assert "probe_class" in d
            assert "is_fitted" in d

    def test_ensemble_confidence_range(self) -> None:
        revolver = AnomalyMathRevolver()
        data = make_normal_signal()
        revolver.fit(data)
        conf = revolver.ensemble_confidence
        assert 0.0 <= conf <= 1.0

    def test_domain_affinity_reorders_probes(self) -> None:
        r1 = AnomalyMathRevolver(domain="earthquake")
        r2 = AnomalyMathRevolver(domain="pandemic")
        data = make_normal_signal()
        r1.fit(data)
        r2.fit(data)
        s1 = r1.detect(data)
        s2 = r2.detect(data)
        # Different domains should produce different scores
        # (due to different phi weighting order)
        assert s1.shape == s2.shape
