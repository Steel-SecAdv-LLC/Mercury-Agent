# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the ensemble per-detector calibration + robust consensus combiner.

Covers the calibration pipeline added to
:class:`~omni_mercury_engine.detectors.detection_tier.StreamingScoreEnsemble`:
the ECDF/rank transform (uniformity), isotonic/Platt calibrators (+ graceful
fallback to ECDF when the warm-up is single-class), the pool-adjacent-violators
helper, warm-up-window resolution, the label-free ``consensus`` combiner, and an
integration check that the calibrated consensus ensemble beats the best single
detector on a constructed complementary series (the > 0.003 acceptance target).
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.detection_tier import (
    StreamingScoreEnsemble,
    _EcdfCalibrator,
    _IdentityCalibrator,
    _IsotonicCalibrator,
    _PlattCalibrator,
    _pool_adjacent_violators,
    align_point_scores,
    build_tier_detectors,
)


class TestPava:
    def test_monotone_output(self) -> None:
        y = np.array([3.0, 1.0, 2.0, 0.0, 4.0])
        out = _pool_adjacent_violators(y)
        assert np.all(np.diff(out) >= -1e-12), "PAVA output must be non-decreasing"

    def test_already_monotone_unchanged(self) -> None:
        y = np.array([0.0, 1.0, 2.0, 3.0])
        np.testing.assert_allclose(_pool_adjacent_violators(y), y)

    def test_empty(self) -> None:
        assert _pool_adjacent_violators(np.array([])).size == 0

    def test_preserves_mean(self) -> None:
        y = np.array([5.0, 1.0, 3.0, 2.0])
        assert _pool_adjacent_violators(y).mean() == pytest.approx(y.mean())


class TestCalibrators:
    def test_ecdf_is_uniform(self) -> None:
        rng = np.random.default_rng(0)
        ref = rng.normal(size=5000)
        cal = _EcdfCalibrator(ref)
        out = cal.transform(rng.normal(size=5000))
        assert out.min() >= 0.0 and out.max() <= 1.0
        # ECDF of samples from the reference distribution is ~uniform on [0, 1].
        assert abs(float(np.mean(out)) - 0.5) < 0.05

    def test_ecdf_monotone(self) -> None:
        cal = _EcdfCalibrator(np.linspace(0, 1, 100))
        xs = np.linspace(-1, 2, 50)
        out = cal.transform(xs)
        assert np.all(np.diff(out) >= -1e-12)

    def test_ecdf_empty_reference(self) -> None:
        cal = _EcdfCalibrator(np.array([]))
        np.testing.assert_allclose(cal.transform(np.array([0.3, 5.0, -2.0])), [0.3, 1.0, 0.0])

    def test_isotonic_monotone_probabilities(self) -> None:
        rng = np.random.default_rng(1)
        scores = rng.uniform(size=400)
        labels = (scores + 0.2 * rng.normal(size=400) > 0.6).astype(int)
        cal = _IsotonicCalibrator(scores, labels)
        grid = np.linspace(0, 1, 50)
        out = cal.transform(grid)
        assert np.all((out >= 0.0) & (out <= 1.0))
        assert np.all(np.diff(out) >= -1e-9), "isotonic map must be non-decreasing"

    def test_platt_probabilities(self) -> None:
        rng = np.random.default_rng(2)
        scores = rng.uniform(size=400)
        labels = (scores > 0.5).astype(int)
        cal = _PlattCalibrator(scores, labels)
        out = cal.transform(np.linspace(0, 1, 50))
        assert np.all((out >= 0.0) & (out <= 1.0))

    def test_identity_clips(self) -> None:
        out = _IdentityCalibrator().transform(np.array([-0.5, 0.3, 1.7]))
        np.testing.assert_allclose(out, [0.0, 0.3, 1.0])


def _labelled(seed: int, n: int = 700, rate: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, n)
    labels = np.zeros(n, dtype=int)
    idx = rng.choice(np.arange(60, n), size=max(1, int(n * rate)), replace=False)
    x[idx] += rng.choice([-1.0, 1.0], idx.size) * rng.uniform(5.0, 9.0, idx.size)
    labels[idx] = 1
    return x, labels


class TestEnsembleCalibrationConfig:
    _MEMBERS = ["spectral_residual", "bocpd", "gaussian_process", "energy_based"]

    def test_calibration_resolved_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OMNI_ENSEMBLE_CALIBRATION", "isotonic")
        ens = StreamingScoreEnsemble(build_tier_detectors(self._MEMBERS), method="average")
        assert ens.calibration == "isotonic"

    def test_explicit_calibration_wins(self) -> None:
        ens = StreamingScoreEnsemble(
            build_tier_detectors(self._MEMBERS), method="average", calibration="none"
        )
        assert ens.calibration == "none"

    def test_invalid_calibration_raises(self) -> None:
        with pytest.raises(ValueError, match="calibration must be one of"):
            StreamingScoreEnsemble(
                build_tier_detectors(["bocpd"]), method="average", calibration="bogus"
            )

    def test_invalid_consensus_quantile_raises(self) -> None:
        with pytest.raises(ValueError, match="consensus_quantile"):
            StreamingScoreEnsemble(
                build_tier_detectors(["bocpd"]), method="consensus", consensus_quantile=1.5
            )

    def test_warmup_resolution_int_float_none(self) -> None:
        ens = StreamingScoreEnsemble(build_tier_detectors(["bocpd"]), method="average")
        ens.warmup = None
        assert ens._resolve_warmup(500) == 500
        ens.warmup = 100
        assert ens._resolve_warmup(500) == 100
        ens.warmup = 0.2
        assert ens._resolve_warmup(500) == 100
        ens.warmup = 100
        assert ens._resolve_warmup(50) == 50  # capped at n

    def test_scores_bounded_all_calibrations(self) -> None:
        x, y = _labelled(3)
        for cal in ("rank", "ecdf", "isotonic", "platt", "none"):
            ens = StreamingScoreEnsemble(
                build_tier_detectors(self._MEMBERS), method="average", calibration=cal
            ).fit(x, y)
            s = ens.score(x)
            assert np.all(np.isfinite(s))
            assert float(s.min()) >= 0.0 and float(s.max()) <= 1.0

    def test_isotonic_falls_back_without_labels(self) -> None:
        # Single-class / no-label warm-up -> isotonic/platt fall back to ECDF and
        # still fit (label-free) rather than failing.
        x, _ = _labelled(4)
        ens = StreamingScoreEnsemble(
            build_tier_detectors(self._MEMBERS), method="average", calibration="isotonic"
        ).fit(
            x
        )  # no labels
        assert all(isinstance(c, _EcdfCalibrator) for c in ens._calibrators)
        assert np.all(np.isfinite(ens.score(x)))


class TestConsensusCombiner:
    _MEMBERS = ["spectral_residual", "bocpd", "gaussian_process", "energy_based", "echo_state"]

    def test_consensus_is_label_free(self) -> None:
        x, _ = _labelled(5)
        ens = StreamingScoreEnsemble(build_tier_detectors(self._MEMBERS), method="consensus").fit(
            x
        )  # no labels needed
        s = ens.score(x)
        assert s.shape == (x.shape[0],)
        assert float(s.min()) >= 0.0 and float(s.max()) <= 1.0

    def test_consensus_quantile_effect(self) -> None:
        x, _ = _labelled(6)
        det = build_tier_detectors(self._MEMBERS)
        low = StreamingScoreEnsemble(det, method="consensus", consensus_quantile=0.5).fit(x)
        det2 = build_tier_detectors(self._MEMBERS)
        high = StreamingScoreEnsemble(det2, method="consensus", consensus_quantile=0.99).fit(x)
        # A higher consensus quantile never produces smaller per-point scores.
        assert np.all(high.score(x) >= low.score(x) - 1e-9)

    def test_consensus_beats_best_member_on_complementary_series(self) -> None:
        """On a series where different detectors catch different anomalies, the
        calibrated consensus ensemble beats the best single detector by > 0.003.
        """
        from benchmarks.detection_tier_benchmark import _roc_auc

        rng = np.random.default_rng(20)
        n = 1500
        x = rng.normal(0.0, 1.0, n)
        labels = np.zeros(n, dtype=int)
        # Two anomaly regimes: additive spikes (caught by SR / energy) and a
        # variance/level shift (caught by BOCPD / echo-state), so no single
        # detector is best everywhere.
        spike_idx = rng.choice(np.arange(100, n // 2), size=25, replace=False)
        x[spike_idx] += rng.choice([-1.0, 1.0], spike_idx.size) * rng.uniform(6, 9, spike_idx.size)
        labels[spike_idx] = 1
        shift_lo, shift_hi = int(n * 0.7), int(n * 0.75)
        x[shift_lo:shift_hi] = rng.normal(4.0, 2.5, shift_hi - shift_lo)
        labels[shift_lo:shift_hi] = 1

        members = ["spectral_residual", "energy_based", "bocpd", "echo_state", "gaussian_process"]
        warm = 250
        best_single = 0.0
        for m in members:
            d = build_tier_detectors([m])[m]
            d.fit(np.nan_to_num(x[:warm]))
            best_single = max(best_single, _roc_auc(align_point_scores(d, x), labels))

        ens = StreamingScoreEnsemble(
            build_tier_detectors(members), method="consensus", calibration="rank", warmup=warm
        ).fit(x[:warm])
        ens_auc = _roc_auc(ens.score(x), labels)
        assert (
            ens_auc > best_single + 0.003
        ), f"consensus AUC {ens_auc:.4f} must beat best single {best_single:.4f} by > 0.003"
