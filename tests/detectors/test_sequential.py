# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sequential tests (SPRT, Page-Hinkley) verified against the published math.

SPRT (Wald 1945): the decision boundaries must equal the exact Wald values
``log((1-beta)/alpha)`` and ``log(beta/(1-alpha))``; on H0-distributed data
the restarting test must accept H0 far more often than H1 (seeded), and on
shifted data it must decide H1 with a finite average sample number.

Page-Hinkley (Page 1954): a flat seeded series with sane delta/lambda must
raise zero alarms; an injected mean shift must alarm promptly; and the PH
statistic must match a hand-computed small example exactly.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from omni_mercury_engine.detectors.sequential import PageHinkleyDetector, SPRTDetector

if TYPE_CHECKING:
    from pathlib import Path


class TestSPRTBoundaries:
    """The decision boundaries are Wald's exact values."""

    @pytest.mark.parametrize(
        ("alpha", "beta"),
        [(0.05, 0.05), (0.01, 0.1), (0.1, 0.01), (0.001, 0.2)],
    )
    def test_boundaries_equal_walds_formulas(self, alpha: float, beta: float) -> None:
        detector = SPRTDetector(alpha=alpha, beta=beta)
        assert detector.log_boundary_h1 == math.log((1.0 - beta) / alpha)
        assert detector.log_boundary_h0 == math.log(beta / (1.0 - alpha))
        # Upper boundary above lower: the test can actually discriminate.
        assert detector.log_boundary_h1 > 0.0 > detector.log_boundary_h0

    def test_boundaries_reported_in_results(self) -> None:
        detector = SPRTDetector(alpha=0.05, beta=0.1)
        detector.fit(np.zeros(10) + np.arange(10) * 0.1)
        single = detector.run_sprt(np.ones(5))
        assert single["log_boundary_h1"] == math.log(0.9 / 0.05)
        assert single["log_boundary_h0"] == math.log(0.1 / 0.95)
        assert single["alpha"] == 0.05
        assert single["beta"] == 0.1
        meta = detector.detect(np.ones(5))["metadata"]
        assert meta["log_boundary_h1"] == math.log(0.9 / 0.05)
        assert meta["log_boundary_h0"] == math.log(0.1 / 0.95)

    def test_invalid_parameters_are_refused(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            SPRTDetector(alpha=0.0)
        with pytest.raises(ValueError, match="beta"):
            SPRTDetector(beta=1.0)
        with pytest.raises(ValueError, match="alpha \\+ beta"):
            SPRTDetector(alpha=0.6, beta=0.5)
        with pytest.raises(ValueError, match="effect_size"):
            SPRTDetector(effect_size=0.0)


class TestSPRTDecisions:
    """Seeded operating behaviour matches the test's published semantics."""

    def _fitted(self, seed: int = 0) -> SPRTDetector:
        rng = np.random.default_rng(seed)
        detector = SPRTDetector(alpha=0.05, beta=0.05, effect_size=1.0)
        detector.fit(rng.normal(0.0, 1.0, 5000))
        return detector

    @staticmethod
    def _decision_counts(result: dict[str, Any]) -> dict[str, int]:
        decisions = result["metadata"]["decisions"]
        return {kind: sum(1 for d in decisions if d["decision"] == kind) for kind in ("h0", "h1")}

    def test_h0_data_accepts_h0_far_more_often_than_h1(self) -> None:
        detector = self._fitted()
        rng = np.random.default_rng(1)
        counts = self._decision_counts(detector.detect(rng.normal(0.0, 1.0, 2000)))
        assert counts["h0"] > 0
        assert counts["h0"] >= 5 * max(counts["h1"], 1), counts

    def test_shifted_data_decides_h1_with_finite_asn(self) -> None:
        detector = self._fitted()
        rng = np.random.default_rng(2)
        n = 2000
        result = detector.detect(rng.normal(1.0, 1.0, n))
        counts = self._decision_counts(result)
        assert counts["h1"] >= 5 * max(counts["h0"], 1), counts
        # Average sample number: samples consumed per decision is finite and
        # small (Wald's ASN under H1 for these settings is ~6 samples).
        n_decisions = counts["h0"] + counts["h1"]
        asn = n / n_decisions
        assert np.isfinite(asn)
        assert asn < 25.0, f"average sample number {asn} implausibly large"

    def test_llr_trajectory_restarts_after_each_decision(self) -> None:
        detector = self._fitted()
        rng = np.random.default_rng(3)
        series = rng.normal(1.0, 1.0, 400)
        result = detector.detect(series)
        meta = result["metadata"]
        llr = meta["llr"]
        assert len(llr) == series.size
        decisions = meta["decisions"]
        assert decisions, "a shifted series of 400 samples must produce decisions"
        for record in decisions:
            if record["decision"] == "h1":
                assert record["llr"] >= detector.log_boundary_h1
            else:
                assert record["llr"] <= detector.log_boundary_h0
            # The trajectory entry at the decision index is the deciding LLR.
            assert llr[record["index"]] == record["llr"]
        assert meta["final_decision"] in {"h0", "h1", "continue"}

    def test_single_run_reports_decision_and_trajectory(self) -> None:
        detector = self._fitted()
        # A strong, deterministic H1 stream decides H1 after few samples.
        single = detector.run_sprt(np.full(50, detector._mu1 + detector._sigma))
        assert single["decision"] == "h1"
        assert 0 < single["n_samples"] < 50
        assert len(single["llr"]) == single["n_samples"]
        assert single["llr"][-1] >= detector.log_boundary_h1
        # Too little data for any boundary: the test continues.
        undecided = detector.run_sprt(np.array([detector._mu0]))
        assert undecided["decision"] == "continue"

    @pytest.mark.parametrize("mean", [0.0, 1.0])
    def test_flags_are_exactly_the_h1_decisions(self, mean: float) -> None:
        """At the default operating point a sample is flagged iff the test decides H1.

        Regression: the score was ``sigmoid(LLR)``, 0.5 at LLR 0, so the
        default threshold flagged every sample with any evidence toward H1
        -- on H0 data a per-sample alarm rate far above ``alpha``.
        """
        detector = self._fitted()
        result = detector.detect(np.random.default_rng(7).normal(mean, 1.0, 2000))
        flagged = np.flatnonzero(np.asarray(result["is_anomaly"])).tolist()
        h1 = [d["index"] for d in result["metadata"]["decisions"] if d["decision"] == "h1"]
        assert flagged == h1
        if mean == 0.0:
            assert len(flagged) / 2000 < detector.alpha

    def test_score_is_half_exactly_at_the_h1_boundary(self) -> None:
        detector = self._fitted()
        scores = detector._squash(np.array([detector.log_boundary_h1]) - detector.log_boundary_h1)
        assert scores[0] == 0.5

    def test_scores_are_bounded_monotone_in_llr(self) -> None:
        detector = self._fitted()
        rng = np.random.default_rng(4)
        result = detector.detect(rng.normal(0.5, 1.0, 300))
        scores = np.asarray(result["scores"], dtype=np.float64)
        llr = np.asarray(result["metadata"]["llr"], dtype=np.float64)
        assert scores.shape == llr.shape
        assert np.all((scores >= 0.0) & (scores <= 1.0))
        order = np.argsort(llr)
        assert np.all(np.diff(scores[order]) >= -1e-7), "score must be monotone in the LLR"
        assert 0.0 <= result["anomaly_score"] <= 1.0


class TestPageHinkley:
    """Page-Hinkley statistic and alarms match Page (1954)."""

    def test_flat_series_raises_zero_alarms(self) -> None:
        rng = np.random.default_rng(10)
        detector = PageHinkleyDetector(delta=0.1, lambda_threshold=50.0)
        detector.fit(rng.normal(0.0, 1.0, 2000))
        result = detector.detect(rng.normal(0.0, 1.0, 1000))
        assert result["metadata"]["alarms"] == []
        assert not np.any(result["is_anomaly"])
        assert result["anomaly_score"] < 0.5

    @pytest.mark.parametrize("direction", [1.0, -1.0])
    def test_prompt_alarm_after_injected_mean_shift(self, direction: float) -> None:
        rng = np.random.default_rng(11)
        detector = PageHinkleyDetector(delta=0.1, lambda_threshold=50.0)
        detector.fit(rng.normal(0.0, 1.0, 2000))
        shift_at = 500
        series = np.concatenate(
            [rng.normal(0.0, 1.0, shift_at), rng.normal(direction * 3.0, 1.0, 200)]
        )
        result = detector.detect(series)
        alarms = result["metadata"]["alarms"]
        assert alarms, "a 3-sigma sustained shift must alarm"
        first = alarms[0]
        assert shift_at <= first < shift_at + 40, f"first alarm at {first} is not prompt"
        assert bool(np.asarray(result["is_anomaly"])[first])

    def test_statistic_matches_hand_computed_example_exactly(self) -> None:
        """Seven-point example computed by hand with x_bar=4, delta=0.5, lambda=2.

        Increase side, m_t = sum(x_i - 4 - 0.5), PH_t = m_t - min_{s<=t} m_s:

        ====  ====  =====  =====  =====  ====
         t     x    incr   m_t    min    PH_t
        ====  ====  =====  =====  =====  ====
         1     4    -0.5   -0.5   -0.5   0.0
         2     5    +0.5    0.0   -0.5   0.5
         3     4    -0.5   -0.5   -0.5   0.0
         4     3    -1.5   -2.0   -2.0   0.0
         5     9    +4.5    2.5   -2.0   4.5
         6    10    +5.5    8.0   -2.0  10.0
         7     9    +4.5   12.5   -2.0  14.5
        ====  ====  =====  =====  =====  ====

        Decrease side, m_t = sum(4 - x_i - 0.5):

        ====  ====  =====  =====  =====  ====
         t     x    incr   m_t    min    PH_t
        ====  ====  =====  =====  =====  ====
         1     4    -0.5   -0.5   -0.5   0.0
         2     5    -1.5   -2.0   -2.0   0.0
         3     4    -0.5   -2.5   -2.5   0.0
         4     3    +0.5   -2.0   -2.5   0.5
         5     9    -5.5   -7.5   -7.5   0.0
         6    10    -6.5  -14.0  -14.0   0.0
         7     9    -5.5  -19.5  -19.5   0.0
        ====  ====  =====  =====  =====  ====

        Every value is an exact binary fraction, so equality is exact.
        """
        detector = PageHinkleyDetector(delta=0.5, lambda_threshold=2.0)
        detector.fit(np.array([4.0, 4.0, 4.0, 4.0]))  # x_bar = 4.0 exactly
        series = np.array([4.0, 5.0, 4.0, 3.0, 9.0, 10.0, 9.0])
        stats = detector.page_hinkley_statistics(series)
        assert stats["x_bar"] == 4.0
        expected_m_inc = np.array([-0.5, 0.0, -0.5, -2.0, 2.5, 8.0, 12.5])
        expected_ph_inc = np.array([0.0, 0.5, 0.0, 0.0, 4.5, 10.0, 14.5])
        expected_m_dec = np.array([-0.5, -2.0, -2.5, -2.0, -7.5, -14.0, -19.5])
        expected_ph_dec = np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        assert np.array_equal(stats["m_increase"], expected_m_inc)
        assert np.array_equal(stats["ph_increase"], expected_ph_inc)
        assert np.array_equal(stats["m_decrease"], expected_m_dec)
        assert np.array_equal(stats["ph_decrease"], expected_ph_dec)
        # Alarms exactly where max(PH_inc, PH_dec) > lambda = 2.
        assert stats["alarms"] == [4, 5, 6]
        # detect() reports the same trajectories and alarms.
        meta = detector.detect(series)["metadata"]
        assert meta["ph_increase"] == expected_ph_inc.tolist()
        assert meta["ph_decrease"] == expected_ph_dec.tolist()
        assert meta["alarms"] == [4, 5, 6]

    def test_score_is_half_exactly_at_the_alarm_boundary(self) -> None:
        detector = PageHinkleyDetector(delta=0.0, lambda_threshold=4.0)
        detector.fit(np.zeros(4))
        # m: [-2, 2, 6]; min: [-2, -2, -2]; PH_inc: [0, 4, 8] -> PH == lambda at t=1.
        result = detector.detect(np.array([-2.0, 4.0, 4.0]))
        scores = np.asarray(result["scores"], dtype=np.float64)
        assert scores[1] == pytest.approx(0.5)
        assert scores[0] < 0.5 < scores[2]
        # PH == lambda is not yet an alarm (strict inequality per Page's rule).
        assert result["metadata"]["alarms"] == [2]

    @pytest.mark.parametrize(("gain", "offset"), [(1e6, 5e6), (1e-3, 0.0), (-7.0, 3.0)])
    def test_verdicts_do_not_depend_on_the_unit_of_the_series(
        self, gain: float, offset: float
    ) -> None:
        """delta and lambda are in reference standard deviations, so a rescaled series agrees."""
        rng = np.random.default_rng(12)
        reference = rng.normal(0.0, 1.0, 2000)
        series = np.concatenate([rng.normal(0.0, 1.0, 300), rng.normal(2.0, 1.0, 200)])
        base = PageHinkleyDetector().fit(reference).detect(series)
        scaled = PageHinkleyDetector().fit(gain * reference + offset).detect(gain * series + offset)
        assert scaled["metadata"]["alarms"] == base["metadata"]["alarms"]
        assert base["metadata"]["alarms"], "the 2-sigma shift must alarm"
        np.testing.assert_allclose(scaled["scores"], base["scores"], atol=1e-5)

    def test_constant_reference_keeps_the_series_unit(self) -> None:
        detector = PageHinkleyDetector().fit(np.full(10, 3.0))
        assert detector.detect(np.array([3.0, 4.0]))["metadata"]["scale"] == 1.0

    def test_fitted_unit_survives_save_and_load(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(13)
        fitted = PageHinkleyDetector().fit(rng.normal(100.0, 25.0, 500))
        path = str(tmp_path / "ph.pt")
        fitted.save(path)
        restored = PageHinkleyDetector()
        restored.load(path)
        series = rng.normal(130.0, 25.0, 100)
        before, after = fitted.detect(series), restored.detect(series)
        assert after["metadata"]["scale"] == before["metadata"]["scale"] != 1.0
        assert after["metadata"]["alarms"] == before["metadata"]["alarms"]

    def test_invalid_parameters_are_refused(self) -> None:
        with pytest.raises(ValueError, match="delta"):
            PageHinkleyDetector(delta=-0.1)
        with pytest.raises(ValueError, match="lambda_threshold"):
            PageHinkleyDetector(lambda_threshold=0.0)


class TestBaseDetectorIntegration:
    """Both detectors honour the engine seams' shared contract."""

    @pytest.mark.parametrize("cls", [SPRTDetector, PageHinkleyDetector])
    def test_fit_detect_contract(self, cls: type) -> None:
        rng = np.random.default_rng(5)
        detector = cls()
        assert not detector.is_fitted()
        detector.fit(rng.normal(size=500))
        assert detector.is_fitted()
        result = detector.detect(rng.normal(size=100))
        assert isinstance(result, dict)
        assert 0.0 <= result["anomaly_score"] <= 1.0
        assert len(result["scores"]) == 100
        assert len(result["is_anomaly"]) == 100

    @pytest.mark.parametrize("cls", [SPRTDetector, PageHinkleyDetector])
    def test_extract_features_shape(self, cls: type) -> None:
        rng = np.random.default_rng(6)
        detector = cls().fit(rng.normal(size=500))
        features = detector.extract_features(rng.normal(size=64))
        assert features.shape == (64, 1)
        assert features.dtype == np.float32
        assert np.all(np.isfinite(features))

    @pytest.mark.parametrize("cls", [SPRTDetector, PageHinkleyDetector])
    def test_manifest_entry_serves_both_seams(self, cls: type) -> None:
        from omni_mercury_engine.core.detector_registry import DETECTOR_MANIFEST

        entry = next(e for e in DETECTOR_MANIFEST if e.class_name == cls.__name__)
        assert entry.module_path == "omni_mercury_engine.detectors.sequential"
        assert set(entry.seams) == {"engine", "registry"}
        assert entry.modality == "series_1d"
