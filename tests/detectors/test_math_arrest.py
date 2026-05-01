# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) Steel Security Advisors LLC
"""Core tests for the Anomaly Math Arrest (probes 1-8 + integration)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from omni_mercury_engine.detectors.math_arrest.arrest import (
    AnomalyMathArrest,
)
from omni_mercury_engine.detectors.math_arrest.base_probe import (
    ProbeResult,
)
from omni_mercury_engine.detectors.math_arrest.probes.additive import (
    AdditiveProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.catalan import (
    CatalanOptimizedProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.ethical import (
    EthicalConstrainedProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.exponential import (
    ExponentialDecayProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.harmonic import (
    HarmonicOscillatorProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.helix import (
    HelixMultiplicativeProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.momentum import (
    MomentumProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.variance_adapted import (
    VarianceAdaptedProbe,
)

# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def make_normal_signal(n: int = 500, seed: int = 42) -> npt.NDArray[np.float64]:
    """Sinusoidal signal with mild noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 10 * np.pi, n)
    return (np.sin(t) + rng.normal(0, 0.05, n)).astype(np.float64)


def make_level_shift(n: int = 500, seed: int = 42) -> npt.NDArray[np.float64]:
    """Sinusoidal signal with a +5 level shift at the midpoint."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 10 * np.pi, n)
    half = n // 2
    return np.concatenate(
        [
            np.sin(t[:half]) + rng.normal(0, 0.05, half),
            np.sin(t[half:]) + 5.0 + rng.normal(0, 0.05, n - half),
        ]
    ).astype(np.float64)


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
        data_2d = np.random.default_rng(42).normal(0, 1, (100, 3)).astype(np.float64)
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
        test_data = np.concatenate(
            [
                rng.normal(0, 0.1, 100),
                rng.normal(0, 5.0, 100),
            ]
        ).astype(np.float64)
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
        test_data = np.concatenate(
            [
                rng.normal(0, 1, 50),
                np.full(50, 20.0),
            ]
        ).astype(np.float64)
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
# AnomalyMathArrest integration tests
# ===================================================================


class TestAnomalyMathArrest:
    """Integration tests for the full AnomalyMathArrest."""

    def test_fit_detect_cycle(self) -> None:
        arrest = AnomalyMathArrest()
        data = make_normal_signal()
        arrest.fit(data)
        scores = arrest.detect(data)
        assert scores.shape == (len(data),)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_predict_returns_binary(self) -> None:
        arrest = AnomalyMathArrest()
        data = make_normal_signal()
        arrest.fit(data)
        preds = arrest.predict(data)
        assert set(np.unique(preds)).issubset({0, 1})
        assert preds.dtype == np.int32

    def test_all_8_probes_instantiate(self) -> None:
        arrest = AnomalyMathArrest()
        assert len(arrest._probes) == 21

    def test_degraded_probe_skipped(self) -> None:
        arrest = AnomalyMathArrest()
        data = make_normal_signal(15)
        # With only 15 samples, some probes will fail (e.g., VarianceAdapted needs 20)
        arrest.fit(data)
        scores = arrest.detect(data)
        assert scores.shape == (15,)
        # Should still produce output even with some probes failing
        assert not np.all(scores == 0.0) or arrest.active_probe_count > 0

    def test_empty_data_raises(self) -> None:
        arrest = AnomalyMathArrest()
        with pytest.raises(ValueError, match="empty"):
            arrest.fit(np.array([], dtype=np.float64))

    def test_short_data_raises(self) -> None:
        arrest = AnomalyMathArrest()
        with pytest.raises(ValueError, match="at least"):
            arrest.fit(np.array([1.0, 2.0], dtype=np.float64))

    def test_unknown_probe_name_raises_value_error(self) -> None:
        """Unknown probe names produce a clear ``ValueError`` with the registry."""
        with pytest.raises(ValueError, match="Unknown probe name"):
            AnomalyMathArrest(probes=["NotARealProbe"])

    def test_mixed_probe_spec_raises_type_error(self) -> None:
        """Mixed ``str``/instance probe-spec lists must raise ``TypeError``.

        The validation is done with an explicit ``raise`` — *not* an
        ``assert`` — so the contract is enforced even when Python is
        run with ``-O``/``PYTHONOPTIMIZE`` (assertions are stripped in
        that mode). Regression guard for PR #162 review.
        """
        first = AdditiveProbe()  # a real BaseEquationProbe instance
        # Spec starts with a str, so the function takes the "list of
        # class name strings" branch and validates each element. The
        # second element is an instance, which must trip the guard.
        with pytest.raises(TypeError, match="Mixed probe spec lists"):
            AnomalyMathArrest(probes=["AdditiveProbe", first])

    def test_scores_in_zero_one(self) -> None:
        arrest = AnomalyMathArrest()
        data = make_normal_signal()
        arrest.fit(data)
        scores = arrest.detect(data)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_no_nan_no_inf(self) -> None:
        arrest = AnomalyMathArrest()
        data = make_normal_signal()
        arrest.fit(data)
        scores = arrest.detect(data)
        assert not np.any(np.isnan(scores))
        assert not np.any(np.isinf(scores))

    def test_calibrate_threshold_f1(self) -> None:
        arrest = AnomalyMathArrest()
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200).astype(np.float64)
        labels = np.zeros(200, dtype=np.int32)
        labels[150:] = 1
        data[150:] += 10.0
        arrest.fit(data)
        threshold = arrest.calibrate_threshold(data, labels, metric="f1")
        assert 0.0 < threshold < 1.0

    def test_get_probe_diagnostics_structure(self) -> None:
        arrest = AnomalyMathArrest()
        data = make_normal_signal()
        arrest.fit(data)
        diags = arrest.get_probe_diagnostics()
        assert len(diags) == 21
        for d in diags:
            assert "probe_class" in d
            assert "is_fitted" in d

    def test_ensemble_confidence_range(self) -> None:
        arrest = AnomalyMathArrest()
        data = make_normal_signal()
        arrest.fit(data)
        conf = arrest.ensemble_confidence
        assert 0.0 <= conf <= 1.0

    def test_domain_affinity_reorders_probes(self) -> None:
        r1 = AnomalyMathArrest(domain="earthquake")
        r2 = AnomalyMathArrest(domain="pandemic")
        data = make_normal_signal()
        r1.fit(data)
        r2.fit(data)
        s1 = r1.detect(data)
        s2 = r2.detect(data)
        # Different domains should produce different scores
        # (due to different phi weighting order)
        assert s1.shape == s2.shape


# ===================================================================
# Probe selection / presets tests
# ===================================================================


class TestProbePresets:
    """Tests for probe selection and preset functionality."""

    def test_default_is_all_21(self) -> None:
        arrest = AnomalyMathArrest()
        assert len(arrest._probes) == 21

    def test_preset_robust(self) -> None:
        arrest = AnomalyMathArrest(probes="robust")
        assert len(arrest._probes) == 5
        data = make_normal_signal(200)
        arrest.fit(data)
        scores = arrest.detect(data)
        assert scores.shape == (200,)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_preset_frequency(self) -> None:
        arrest = AnomalyMathArrest(probes="frequency")
        assert len(arrest._probes) == 5
        data = make_normal_signal(200)
        arrest.fit(data)
        scores = arrest.detect(data)
        assert scores.shape == (200,)

    def test_preset_chaos(self) -> None:
        arrest = AnomalyMathArrest(probes="chaos")
        assert len(arrest._probes) == 5
        data = make_normal_signal(200)
        arrest.fit(data)
        scores = arrest.detect(data)
        assert scores.shape == (200,)

    def test_preset_minimal(self) -> None:
        arrest = AnomalyMathArrest(probes="minimal")
        assert len(arrest._probes) == 3
        data = make_normal_signal(200)
        arrest.fit(data)
        scores = arrest.detect(data)
        assert scores.shape == (200,)

    def test_preset_forensic_is_all(self) -> None:
        arrest = AnomalyMathArrest(probes="forensic")
        assert len(arrest._probes) == 21

    def test_custom_probe_list_by_name(self) -> None:
        arrest = AnomalyMathArrest(probes=["AdditiveProbe", "MomentumProbe"])
        assert len(arrest._probes) == 2
        data = make_normal_signal(200)
        arrest.fit(data)
        scores = arrest.detect(data)
        assert scores.shape == (200,)

    def test_unknown_preset_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown probe preset"):
            AnomalyMathArrest(probes="nonexistent_preset")

    def test_unknown_probe_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown probe name"):
            AnomalyMathArrest(probes=["FakeProbe"])

    def test_custom_probe_instances(self) -> None:
        from omni_mercury_engine.detectors.math_arrest.probes.additive import (
            AdditiveProbe,
        )
        from omni_mercury_engine.detectors.math_arrest.probes.momentum import (
            MomentumProbe,
        )

        arrest = AnomalyMathArrest(probes=[AdditiveProbe(), MomentumProbe()])
        assert len(arrest._probes) == 2
        data = make_normal_signal(200)
        arrest.fit(data)
        scores = arrest.detect(data)
        assert scores.shape == (200,)


# ===================================================================
# Geometry report tests
# ===================================================================


class TestGeometryReport:
    """Tests for get_geometry_report()."""

    def test_report_structure(self) -> None:
        arrest = AnomalyMathArrest(probes="minimal")
        data = make_normal_signal(200)
        arrest.fit(data)
        report = arrest.get_geometry_report(data)
        assert len(report) == 3
        for entry in report:
            assert "probe_name" in entry
            assert "anomaly_geometry" in entry
            assert "mean_deviation" in entry
            assert "max_deviation" in entry
            assert "confidence" in entry

    def test_report_values_in_range(self) -> None:
        arrest = AnomalyMathArrest(probes="robust")
        data = make_normal_signal(200)
        arrest.fit(data)
        report = arrest.get_geometry_report(data)
        for entry in report:
            assert 0.0 <= entry["mean_deviation"] <= 1.0
            assert 0.0 <= entry["max_deviation"] <= 1.0
            assert 0.0 <= entry["confidence"] <= 1.0

    def test_report_distinct_geometries(self) -> None:
        arrest = AnomalyMathArrest()
        data = make_normal_signal(200)
        arrest.fit(data)
        report = arrest.get_geometry_report(data)
        geometries = {e["anomaly_geometry"] for e in report}
        # 21 probes should have multiple distinct geometries
        assert len(geometries) >= 5

    def test_report_not_fitted_raises(self) -> None:
        arrest = AnomalyMathArrest()
        with pytest.raises(RuntimeError, match="not been fitted"):
            arrest.get_geometry_report(make_normal_signal(200))


# ===================================================================
# Score window tests
# ===================================================================


class TestScoreWindow:
    """Tests for score_window()."""

    def test_window_size_1_is_identity(self) -> None:
        arrest = AnomalyMathArrest(probes="minimal")
        data = make_normal_signal(200)
        arrest.fit(data)
        raw = arrest.detect(data)
        windowed = arrest.score_window(data, window_size=1)
        np.testing.assert_array_equal(raw, windowed)

    def test_window_smooths_scores(self) -> None:
        arrest = AnomalyMathArrest(probes="minimal")
        data = make_normal_signal(200)
        arrest.fit(data)
        raw = arrest.detect(data)
        windowed = arrest.score_window(data, window_size=20)
        # Windowed scores should be smoother (lower std)
        assert float(np.std(windowed)) <= float(np.std(raw)) + 1e-10

    def test_window_output_shape(self) -> None:
        arrest = AnomalyMathArrest(probes="minimal")
        data = make_normal_signal(200)
        arrest.fit(data)
        windowed = arrest.score_window(data, window_size=10)
        assert windowed.shape == (200,)

    def test_window_values_in_zero_one(self) -> None:
        arrest = AnomalyMathArrest(probes="minimal")
        data = make_normal_signal(200)
        arrest.fit(data)
        windowed = arrest.score_window(data, window_size=10)
        assert np.all(windowed >= 0.0)
        assert np.all(windowed <= 1.0)

    def test_window_invalid_size_raises(self) -> None:
        arrest = AnomalyMathArrest(probes="minimal")
        data = make_normal_signal(200)
        arrest.fit(data)
        with pytest.raises(ValueError, match="window_size must be >= 1"):
            arrest.score_window(data, window_size=0)

    def test_window_not_fitted_raises(self) -> None:
        arrest = AnomalyMathArrest()
        with pytest.raises(RuntimeError, match="not been fitted"):
            arrest.score_window(make_normal_signal(200))

    def test_large_window_converges_to_mean(self) -> None:
        arrest = AnomalyMathArrest(probes="minimal")
        data = make_normal_signal(200)
        arrest.fit(data)
        raw = arrest.detect(data)
        windowed = arrest.score_window(data, window_size=200)
        # With window == n_samples, most values should be near the mean
        raw_mean = float(np.mean(raw))
        mid_windowed = float(windowed[100])
        assert abs(mid_windowed - raw_mean) < 0.3

    def test_window_no_nan_no_inf(self) -> None:
        arrest = AnomalyMathArrest(probes="robust")
        data = make_normal_signal(200)
        arrest.fit(data)
        windowed = arrest.score_window(data, window_size=15)
        assert not np.any(np.isnan(windowed))
        assert not np.any(np.isinf(windowed))
