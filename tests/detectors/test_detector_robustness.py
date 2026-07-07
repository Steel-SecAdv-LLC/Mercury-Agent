# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Adversarial robustness / contract regression tests for the detector tier.

These lock in the hardening pass that fixed a family of confirmed defects the
streaming / statistical / state-space tier shipped with:

* empty / very-short input crashed ``fit`` and unfitted ``detect`` (``np.quantile``
  on a zero-length array, and delay-embedding indexing);
* a single ``±inf`` / ``NaN`` sample produced non-finite anomaly scores (the
  ``np.nan_to_num`` sentinel ``1.8e308`` overflowed downstream and ``np.clip`` does
  not scrub ``NaN``), and at ``fit`` time permanently poisoned the detector;
* SPOT's ``detect`` mutated its fitted calibration (non-idempotent);
* BOCPD's run-length truncation fold double-counted one bin and dropped the tail;
* digital-twin ``fit`` raised on a constant / large-magnitude series;
* Deep-SVDD / EBM crashed on series shorter than the embedding dimension and
  produced NaN calibration from a single training row;
* spiking / RCA crashed on the ``detect``-before-``fit`` path their own bodies
  branch for; RCA crashed on an empty batch;
* frequent-pattern crashed when a ``detect`` batch was narrower than the training
  vocabulary;
* the ensemble seam (``align_point_scores``) passed a member's ``NaN`` straight
  into the calibrated combiner.

The invariant every detector must satisfy is the ``BaseDetector`` contract: for
*any* input, ``detect`` returns finite per-sample scores in ``[0, 1]`` and a
finite scalar ``anomaly_score`` (or degrades gracefully) — never a crash and
never a non-finite score.
"""

from __future__ import annotations

import warnings
from typing import Any, cast

import numpy as np
import pytest

from omni_mercury_engine.detectors.bocpd import BOCPDDetector
from omni_mercury_engine.detectors.deep_svdd import DeepSVDDDetector
from omni_mercury_engine.detectors.detection_tier import align_point_scores
from omni_mercury_engine.detectors.digital_twin import DigitalTwinResidualDetector
from omni_mercury_engine.detectors.echo_state import EchoStateDetector
from omni_mercury_engine.detectors.energy_based import EnergyBasedDetector
from omni_mercury_engine.detectors.frequent_pattern import FrequentPatternDetector
from omni_mercury_engine.detectors.gaussian_process import GaussianProcessDetector
from omni_mercury_engine.detectors.hawkes import HawkesBurstDetector
from omni_mercury_engine.detectors.imm import IMMDetector
from omni_mercury_engine.detectors.particle_filter import ParticleFilterDetector
from omni_mercury_engine.detectors.rca import RootCauseGraphDetector
from omni_mercury_engine.detectors.spectral_residual import SpectralResidualDetector
from omni_mercury_engine.detectors.spiking import SpikingNetworkDetector
from omni_mercury_engine.detectors.spot_evt import SPOTDetector
from omni_mercury_engine.detectors.survival import SurvivalHazardDetector

# 1-D tier detectors that share the input-coercion / squash / output contract.
ONE_D_DETECTORS = [
    SpectralResidualDetector,
    BOCPDDetector,
    HawkesBurstDetector,
    ParticleFilterDetector,
    IMMDetector,
    DigitalTwinResidualDetector,
    GaussianProcessDetector,
    SurvivalHazardDetector,
    EnergyBasedDetector,
    DeepSVDDDetector,
    EchoStateDetector,
    SpikingNetworkDetector,
]

ADVERSARIAL_1D = {
    "empty": np.array([]),
    "one": np.array([1.0]),
    "two": np.array([1.0, 2.0]),
    "short": np.array([0.3, -0.2, 1.1, 0.5, -0.7]),
    "constant": np.full(64, 3.0),
    "constant_large": np.full(64, 1e6),
    "single_inf": np.concatenate([np.zeros(63), [np.inf]]),
    "nan_and_inf": np.array([1.0, np.nan, 2.0, np.inf, -np.inf] + [0.5] * 60),
    "huge_finite": np.full(64, 1e18),
    "all_negative": -np.abs(np.random.default_rng(1).normal(size=64)) - 1.0,
}


def _assert_contract(result: dict[str, Any]) -> None:
    """Every detect() result must be finite and in [0, 1]."""
    scores = np.asarray(result.get("scores", []), dtype=np.float64)
    if scores.size:
        assert np.all(np.isfinite(scores)), "scores contain NaN/inf"
        assert np.all(scores >= -1e-9) and np.all(scores <= 1 + 1e-9), "scores out of [0,1]"
    anomaly = result.get("anomaly_score", 0.0)
    assert np.isfinite(float(anomaly)), "anomaly_score not finite"


@pytest.mark.parametrize("cls", ONE_D_DETECTORS, ids=lambda c: c.__name__)
class TestOneDContract:
    def test_fitted_detect_on_adversarial_inputs(self, cls: type) -> None:
        rng = np.random.default_rng(0)
        det = cls()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            det.fit(rng.normal(size=200))
            for name, x in ADVERSARIAL_1D.items():
                _assert_contract(det.detect(x))

    def test_unfitted_detect_does_not_crash(self, cls: type) -> None:
        # detect() before fit() must not raise (several detectors branch for it).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for x in ADVERSARIAL_1D.values():
                _assert_contract(cls().detect(x))

    def test_fit_on_empty_does_not_crash(self, cls: type) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            det = cls().fit(np.array([]))  # was IndexError via np.quantile
            _assert_contract(det.detect(np.array([1.0, 2.0, 3.0])))

    def test_fit_poisoning_by_inf_leaves_detector_finite(self, cls: type) -> None:
        # A single inf in TRAINING data must not poison the model into emitting
        # NaN on subsequent CLEAN data.
        rng = np.random.default_rng(2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            det = cls().fit(np.array([1.0, np.inf, 2.0, -np.inf] + [0.4] * 120))
            _assert_contract(det.detect(rng.normal(size=100)))

    def test_fit_on_constant_large_series(self, cls: type) -> None:
        # digital-twin raised LinAlgError('Singular matrix') here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            det = cls().fit(np.full(120, 1e6))
            _assert_contract(det.detect(np.full(64, 1e6)))

    def test_detect_does_not_mutate_input(self, cls: type) -> None:
        rng = np.random.default_rng(3)
        det = cls().fit(rng.normal(size=200))
        x = rng.normal(size=64)
        x_copy = x.copy()
        det.detect(x)
        assert np.array_equal(x, x_copy), "detect mutated its input array"


class TestEmbeddingDetectorsShortInput:
    """EBM / Deep-SVDD crashed on series shorter than the embedding dimension."""

    @pytest.mark.parametrize("cls", [EnergyBasedDetector, DeepSVDDDetector])
    @pytest.mark.parametrize("length", [0, 1, 2, 3, 5, 7])
    def test_short_series_fit_and_detect(self, cls: type, length: int) -> None:
        rng = np.random.default_rng(4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            det = cls().fit(rng.normal(size=200))
            _assert_contract(det.detect(rng.normal(size=length)))
            # Fitting directly on a short series must also stay finite.
            det2 = cls().fit(rng.normal(size=length) if length else np.array([]))
            _assert_contract(det2.detect(rng.normal(size=50)))


class TestSpotIdempotence:
    def test_detect_is_idempotent_and_does_not_drift_calibration(self) -> None:
        rng = np.random.default_rng(5)
        det = SPOTDetector().fit(rng.normal(size=1000))
        zq_fitted = det._zq
        test = rng.normal(size=500)
        s1 = np.asarray(det.detect(test)["scores"], dtype=np.float64)
        s2 = np.asarray(det.detect(test)["scores"], dtype=np.float64)
        assert np.array_equal(s1, s2), "SPOT detect() is not idempotent"
        assert det._zq == zq_fitted, "SPOT detect() drifted the fitted threshold"


class TestBocpdTruncationFold:
    def test_fold_conserves_mass_and_folds_boundary_message(self) -> None:
        # With max_run_length small enough that a long stationary series reaches
        # the cap, the run-length posterior must stay a valid distribution and
        # the change-point score must stay finite in [0, 1]. The pre-fix code
        # double-counted growth[cap-2] and dropped growth[cap-1], inflating the
        # short-run-length (anomaly) mass.
        rng = np.random.default_rng(6)
        det = BOCPDDetector(hazard_lambda=250, change_grace=5, max_run_length=30)
        det.fit(rng.normal(0, 1, 400))
        scores = np.asarray(det.detect(rng.normal(0, 1, 600))["scores"], dtype=np.float64)
        assert np.all(np.isfinite(scores))
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
        # Stationary baseline change-point mass is small; the buggy fold inflated
        # it to ~0.026. The corrected fold keeps it well below the buggy level.
        assert float(np.median(scores[50:])) < 0.022

    def test_default_config_unaffected_by_fold_fix(self) -> None:
        # At the shipped default (max_run_length=500 + hazard 250) the run length
        # rarely reaches the cap, so the fold fix must not perturb normal scoring.
        rng = np.random.default_rng(6)
        det = BOCPDDetector().fit(rng.normal(0, 1, 400))
        scores = np.asarray(det.detect(rng.normal(0, 1, 300))["scores"], dtype=np.float64)
        assert np.all(np.isfinite(scores)) and np.all((scores >= 0) & (scores <= 1))


class TestSpikingUnfitted:
    def test_detect_before_fit_builds_population_lazily(self) -> None:
        det = SpikingNetworkDetector()
        # Previously raised AssertionError (dead unfitted branch).
        _assert_contract(det.detect(np.random.default_rng(7).normal(size=64)))


class TestRcaRobustness:
    def test_empty_batch_returns_empty(self) -> None:
        det = RootCauseGraphDetector().fit(np.random.default_rng(8).normal(size=(100, 5)))
        result = det.detect(np.zeros((0, 5)))
        assert np.asarray(result["scores"]).size == 0
        assert result["anomaly_score"] == 0.0
        assert result["metadata"]["ranked_causes"] == []

    def test_unfitted_detect_self_standardises(self) -> None:
        det = RootCauseGraphDetector()
        _assert_contract(det.detect(np.random.default_rng(9).normal(size=(10, 4))))

    def test_nonfinite_multivariate_input_stays_finite(self) -> None:
        det = RootCauseGraphDetector().fit(np.random.default_rng(10).normal(size=(80, 4)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _assert_contract(det.detect(np.array([[1.0, np.inf, 2.0, -np.inf]])))
            feats = det.extract_features(np.array([[1.0, np.inf, 2.0, -np.inf]]))
        assert np.all(np.isfinite(feats)), "extract_features returned non-finite features"

    def test_node_count_mismatch_raises_clear_error(self) -> None:
        det = RootCauseGraphDetector().fit(np.random.default_rng(11).normal(size=(50, 5)))
        with pytest.raises(ValueError, match="node count must match"):
            det.detect(np.random.default_rng(12).normal(size=(3, 2)))


class TestFrequentPatternColumnMismatch:
    def test_narrower_detect_batch_does_not_crash(self) -> None:
        rng = np.random.default_rng(13)
        det = FrequentPatternDetector(min_support=0.1)
        det.fit((rng.random((60, 12)) > 0.5).astype(float))
        # Narrower than the training vocabulary — rules referencing dropped
        # columns must be skipped, not dereferenced out of bounds.
        _assert_contract(det.detect((rng.random((4, 4)) > 0.5).astype(float)))
        _assert_contract(det.detect(np.array([1.0, 0.0, 1.0])))  # 1-D short path
        _assert_contract(det.detect(np.zeros((0, 12))))  # empty batch

    def test_wider_detect_batch_does_not_crash(self) -> None:
        rng = np.random.default_rng(14)
        det = FrequentPatternDetector(min_support=0.1)
        det.fit((rng.random((60, 8)) > 0.5).astype(float))
        _assert_contract(det.detect((rng.random((4, 20)) > 0.5).astype(float)))


class TestEnsembleSeamSanitisesMembers:
    def test_align_point_scores_scrubs_member_nan(self) -> None:
        # A misbehaving member that emits NaN must not leak into the ensemble;
        # align_point_scores is the defence-in-depth choke point.
        class _NaNDetector:
            def detect(self, data: Any) -> dict[str, Any]:
                n = np.asarray(data).size
                return {"scores": np.full(n, np.nan), "anomaly_score": float("nan")}

        out = align_point_scores(cast("Any", _NaNDetector()), np.arange(20.0))
        assert out.shape == (20,)
        assert np.all(np.isfinite(out)) and np.all((out >= 0) & (out <= 1))
