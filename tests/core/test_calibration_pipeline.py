# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioral tests for the Threshold Auto-Calibration Pipeline (Phase 5).

Covers:
- Dataset fingerprinting: SHA-256 over canonical summary statistics
  (hand-computed digest), 1-D reshaping, NaN handling, permutation
  invariance, serialization.
- KL divergence and symmetric KL: closed-form values, identity of
  indiscernibles, asymmetry, input normalization.
- Two-sample KS statistic: closed-form D and asymptotic p-value for
  identical, disjoint, and partially overlapping samples; empty input.
- Calibration strategies: Youden's J, F1-optimal, and cost-sensitive
  thresholds with hand-computed optima on tiny candidate grids.
- Registry and provenance: system defaults, UNCALIBRATED -> CALIBRATED
  -> STALE lifecycle, manual overrides, provenance introspection.
- calibrate_all_thresholds: guardrail clamping (cap and floor), ethical
  floor verification, confidence-band quantiles (including the
  degenerate constant-score fallback), single-class degradation.
- Drift detection: no-drift and drift directions, KL-only and KS-only
  triggers, per-feature outputs, zero-width inputs, stale marking.
- Error paths: single-class labels, length mismatch, unknown strategy.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from omni_mercury_engine.core.calibration_pipeline import (
    CalibrationStrategy,
    ThresholdCalibrationPipeline,
    ThresholdRecord,
    ThresholdStatus,
    compute_dataset_fingerprint,
    kl_divergence,
    ks_statistic,
    symmetric_kl_divergence,
)
from omni_mercury_engine.core.centralized_constants import ANOMALY, ETHICAL

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: Expected number of system-default thresholds registered at construction:
#: 4 anomaly + 7 ethical + 4 confidence bands.
N_SYSTEM_DEFAULTS = 15


def _separable_data() -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Perfectly separable scores: negatives at 0.0/0.25, positives at 0.75/1.0."""
    scores = np.array([0.0, 0.25, 0.75, 1.0], dtype=np.float64)
    labels = np.array([0, 0, 1, 1], dtype=np.int32)
    return scores, labels


def _interleaved_data() -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Interleaved scores where no threshold achieves a perfect split."""
    scores = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    labels = np.array([0, 1, 0, 1], dtype=np.int32)
    return scores, labels


# ---------------------------------------------------------------------------
# compute_dataset_fingerprint
# ---------------------------------------------------------------------------


class TestComputeDatasetFingerprint:
    """SHA-256 fingerprinting of dataset summary statistics."""

    def test_known_dataset_hash_matches_hand_computed_digest(self) -> None:
        """The digest matches an independently hand-built canonical payload."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        fp = compute_dataset_fingerprint(X)

        # Summary statistics computed by hand for [[1,2],[3,4]].
        assert fp.n_samples == 2
        assert fp.n_features == 2
        assert fp.mean == [2.0, 3.0]
        assert fp.std == [1.0, 1.0]
        assert fp.quantiles == {
            "q25": [1.5, 2.5],
            "q50": [2.0, 3.0],
            "q75": [2.5, 3.5],
        }

        # Reconstruct the documented canonical JSON payload independently.
        payload = {
            "mean": [2.0, 3.0],
            "std": [1.0, 1.0],
            "n_samples": 2,
            "n_features": 2,
            "q25": [1.5, 2.5],
            "q50": [2.0, 3.0],
            "q75": [2.5, 3.5],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert fp.sha256 == expected
        # Pin the digest so any change to the hashing scheme is loud.
        assert fp.sha256 == "f2121c081d6a242b0a0593926f91db287dcf97ed6bcf84793835298333d5c3b8"

    def test_fingerprint_is_deterministic(self) -> None:
        """Two fingerprints of the same data are identical."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(30, 3))
        assert compute_dataset_fingerprint(X).sha256 == compute_dataset_fingerprint(X).sha256

    def test_fingerprint_is_row_permutation_invariant(self) -> None:
        """The hash depends only on summary statistics, not row order."""
        rng = np.random.default_rng(7)
        X = rng.normal(size=(50, 2))
        permuted = X[rng.permutation(50)]
        assert compute_dataset_fingerprint(X).sha256 == compute_dataset_fingerprint(permuted).sha256

    def test_different_data_produces_different_hash(self) -> None:
        """Distinct summary statistics give distinct digests."""
        a = compute_dataset_fingerprint(np.array([1.0, 2.0, 3.0]))
        b = compute_dataset_fingerprint(np.array([1.0, 2.0, 4.0]))
        assert a.sha256 != b.sha256

    def test_one_dimensional_input_reshaped_to_single_feature(self) -> None:
        """A 1-D array is treated as (n, 1)."""
        fp = compute_dataset_fingerprint(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert fp.n_samples == 5
        assert fp.n_features == 1
        assert fp.mean == [3.0]

    def test_nan_values_are_ignored_in_statistics(self) -> None:
        """NaN entries do not poison the summary statistics."""
        fp = compute_dataset_fingerprint(np.array([1.0, 2.0, np.nan, 4.0]))
        assert fp.mean[0] == pytest.approx(7.0 / 3.0)
        assert all(math.isfinite(v) for v in fp.mean + fp.std)
        assert len(fp.sha256) == 64
        assert all(c in "0123456789abcdef" for c in fp.sha256)

    def test_to_dict_is_json_serializable(self) -> None:
        """to_dict output round-trips through json.dumps."""
        fp = compute_dataset_fingerprint(np.array([[0.5, 1.5]] * 4))
        d = fp.to_dict()
        assert set(d) == {
            "sha256",
            "n_samples",
            "n_features",
            "mean",
            "std",
            "quantiles",
            "created_at",
        }
        assert json.loads(json.dumps(d)) == d


# ---------------------------------------------------------------------------
# KL divergence
# ---------------------------------------------------------------------------


class TestKLDivergence:
    """Closed-form checks for the (asymmetric) KL divergence."""

    def test_identical_distributions_give_zero(self) -> None:
        """D_KL(P || P) = 0."""
        p = np.array([0.2, 0.3, 0.5])
        assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-12)

    def test_hand_computed_value(self) -> None:
        """D_KL([.5,.5] || [.25,.75]) = .5 ln 2 + .5 ln(2/3) = .5 ln(4/3)."""
        p = np.array([0.5, 0.5])
        q = np.array([0.25, 0.75])
        expected = 0.5 * math.log(2.0) + 0.5 * math.log(2.0 / 3.0)
        assert kl_divergence(p, q) == pytest.approx(expected, rel=1e-6)

    def test_asymmetry(self) -> None:
        """KL is not symmetric: D(P||Q) != D(Q||P) for these PMFs."""
        p = np.array([0.5, 0.5])
        q = np.array([0.25, 0.75])
        reverse_expected = 0.25 * math.log(0.5) + 0.75 * math.log(1.5)
        assert kl_divergence(q, p) == pytest.approx(reverse_expected, rel=1e-6)
        assert kl_divergence(p, q) != pytest.approx(kl_divergence(q, p), rel=1e-3)

    def test_unnormalized_inputs_are_normalized(self) -> None:
        """Raw counts are normalized to PMFs before the divergence."""
        expected = 0.5 * math.log(2.0) + 0.5 * math.log(2.0 / 3.0)
        assert kl_divergence(np.array([2.0, 2.0]), np.array([1.0, 3.0])) == pytest.approx(
            expected, rel=1e-6
        )

    def test_symmetric_kl_is_jeffreys_average(self) -> None:
        """Symmetric KL equals the mean of the two directed divergences."""
        p = np.array([0.5, 0.5])
        q = np.array([0.25, 0.75])
        forward = 0.5 * math.log(2.0) + 0.5 * math.log(2.0 / 3.0)
        backward = 0.25 * math.log(0.5) + 0.75 * math.log(1.5)
        expected = 0.5 * (forward + backward)
        assert symmetric_kl_divergence(p, q) == pytest.approx(expected, rel=1e-6)
        assert symmetric_kl_divergence(q, p) == pytest.approx(
            symmetric_kl_divergence(p, q), rel=1e-12
        )


# ---------------------------------------------------------------------------
# KS statistic
# ---------------------------------------------------------------------------


class TestKSStatistic:
    """Closed-form checks for the two-sample Kolmogorov-Smirnov test."""

    def test_identical_samples_give_zero_statistic_and_p_one(self) -> None:
        """Identical samples: D = 0, and p = 2 exp(0) clipped to 1."""
        a = np.array([1.0, 2.0, 3.0])
        d, p = ks_statistic(a, a.copy())
        assert d == 0.0
        assert p == 1.0

    def test_disjoint_samples_give_statistic_one(self) -> None:
        """Fully separated samples: D = 1, p = 2 exp(-2 nm/(n+m))."""
        a = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        b = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        d, p = ks_statistic(a, b)
        assert d == 1.0
        # n = m = 5 -> nm/(n+m) = 2.5 -> p = 2 exp(-5).
        assert p == pytest.approx(2.0 * math.exp(-5.0), rel=1e-12)

    def test_partial_overlap_hand_computed(self) -> None:
        """a=[1..4], b=[3..6]: D = 0.5, p = 2 exp(-1) (hand-walked ECDFs)."""
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([3.0, 4.0, 5.0, 6.0])
        d, p = ks_statistic(a, b)
        assert d == 0.5
        # n = m = 4 -> nm/(n+m) = 2 -> p = 2 exp(-2 * 2 * 0.25) = 2 exp(-1).
        assert p == pytest.approx(2.0 * math.exp(-1.0), rel=1e-12)

    def test_empty_input_returns_degenerate_result(self) -> None:
        """Either sample being empty yields (0.0, 1.0)."""
        assert ks_statistic(np.array([]), np.array([1.0, 2.0])) == (0.0, 1.0)
        assert ks_statistic(np.array([1.0, 2.0]), np.array([])) == (0.0, 1.0)


# ---------------------------------------------------------------------------
# Pipeline: system defaults and registry
# ---------------------------------------------------------------------------


class TestPipelineDefaults:
    """Registry construction, defaults, and provenance introspection."""

    def test_system_defaults_registered_uncalibrated(self) -> None:
        """All 15 system thresholds start UNCALIBRATED with known values."""
        pipeline = ThresholdCalibrationPipeline()
        prov = pipeline.get_threshold_provenance()
        assert len(prov) == N_SYSTEM_DEFAULTS

        record = pipeline.get_threshold("anomaly.default_threshold")
        assert record is not None
        assert record.value == ANOMALY.DEFAULT_THRESHOLD
        assert record.status is ThresholdStatus.UNCALIBRATED
        assert record.strategy == "system_default"

        zscore = pipeline.get_threshold("anomaly.zscore_threshold")
        assert zscore is not None
        assert zscore.value == ANOMALY.ZSCORE_DEFAULT_THRESHOLD

        ethical_min = pipeline.get_threshold("ethical.ethical_minimum")
        assert ethical_min is not None
        assert ethical_min.value == ETHICAL.ETHICAL_MINIMUM

        conf_high = pipeline.get_threshold("confidence.high")
        assert conf_high is not None
        assert conf_high.value == 0.9

    def test_uncalibrated_property_lists_everything_initially(self) -> None:
        """Fresh pipeline: everything uncalibrated, nothing stale."""
        pipeline = ThresholdCalibrationPipeline()
        assert len(pipeline.uncalibrated_thresholds) == N_SYSTEM_DEFAULTS
        assert pipeline.stale_thresholds == []

    def test_get_threshold_unknown_name_returns_none(self) -> None:
        """Unregistered names return None instead of raising."""
        assert ThresholdCalibrationPipeline().get_threshold("no.such.threshold") is None

    def test_provenance_entries_have_documented_shape(self) -> None:
        """Provenance dicts expose value/status/strategy/dataset_sha256."""
        prov = ThresholdCalibrationPipeline().get_threshold_provenance()
        entry = prov["anomaly.default_threshold"]
        assert entry["value"] == ANOMALY.DEFAULT_THRESHOLD
        assert entry["status"] == "uncalibrated"
        assert entry["strategy"] == "system_default"
        assert entry["dataset_sha256"] is None
        assert entry["metric_at_threshold"] == 0.0
        assert entry["metadata"] == {}

    def test_set_threshold_manual_override(self) -> None:
        """set_threshold registers a manual UNCALIBRATED record by default."""
        pipeline = ThresholdCalibrationPipeline()
        pipeline.set_threshold("custom.gate", 0.42)
        record = pipeline.get_threshold("custom.gate")
        assert record is not None
        assert record.value == 0.42
        assert record.status is ThresholdStatus.UNCALIBRATED
        assert record.strategy == "manual"

    def test_set_threshold_with_explicit_status(self) -> None:
        """Explicit status and strategy labels are honored."""
        pipeline = ThresholdCalibrationPipeline()
        pipeline.set_threshold(
            "custom.gate", 0.9, status=ThresholdStatus.CALIBRATED, strategy="expert_judgment"
        )
        record = pipeline.get_threshold("custom.gate")
        assert record is not None
        assert record.status is ThresholdStatus.CALIBRATED
        assert record.strategy == "expert_judgment"

    def test_threshold_record_to_dict_without_fingerprint(self) -> None:
        """ThresholdRecord serializes with a None fingerprint."""
        record = ThresholdRecord(name="x", value=0.5)
        d = record.to_dict()
        assert d["name"] == "x"
        assert d["value"] == 0.5
        assert d["status"] == "uncalibrated"
        assert d["dataset_fingerprint"] is None
        assert json.loads(json.dumps(d)) == d


# ---------------------------------------------------------------------------
# calibrate_from_data: Youden's J
# ---------------------------------------------------------------------------


class TestCalibrateYoudenJ:
    """Youden's J = TPR - FPR threshold selection."""

    def test_perfect_separation_closed_form(self) -> None:
        """Candidates [0,.25,.5,.75,1]: first J=1 candidate is t=0.25."""
        scores, labels = _separable_data()
        pipeline = ThresholdCalibrationPipeline()
        result = pipeline.calibrate_from_data(
            scores, labels, method=CalibrationStrategy.YOUDEN_J, n_candidate_thresholds=5
        )
        # t=0.0: pred={.25,.75,1} -> TPR=1, FPR=0.5, J=0.5.
        # t=0.25 (strict >): pred={.75,1} -> TPR=1, FPR=0, J=1.  First maximum wins.
        assert result.threshold == 0.25
        assert result.metric_value == 1.0
        assert result.metric_name == "youden_j"
        assert result.strategy is CalibrationStrategy.YOUDEN_J
        assert result.all_thresholds_evaluated == 5
        assert result.details == {
            "tpr": 1.0,
            "fpr": 0.0,
            "sensitivity": 1.0,
            "specificity": 1.0,
        }

    def test_registry_updated_with_provenance(self) -> None:
        """Calibration flips the record to CALIBRATED with a fingerprint."""
        scores, labels = _separable_data()
        pipeline = ThresholdCalibrationPipeline()
        result = pipeline.calibrate_from_data(scores, labels)
        record = pipeline.get_threshold("anomaly.default_threshold")
        assert record is not None
        assert record.status is ThresholdStatus.CALIBRATED
        assert record.strategy == "youden_j"
        assert record.value == result.threshold
        assert record.dataset_fingerprint is not None
        assert record.dataset_fingerprint.sha256 == compute_dataset_fingerprint(scores).sha256
        assert "anomaly.default_threshold" not in pipeline.uncalibrated_thresholds

    def test_string_method_accepted(self) -> None:
        """method="youden_j" behaves identically to the enum member."""
        scores, labels = _separable_data()
        pipeline = ThresholdCalibrationPipeline()
        result = pipeline.calibrate_from_data(
            scores, labels, method="youden_j", n_candidate_thresholds=5
        )
        assert result.threshold == 0.25
        assert result.strategy is CalibrationStrategy.YOUDEN_J

    def test_custom_threshold_name_registered(self) -> None:
        """threshold_name routes the result to a custom registry key."""
        scores, labels = _separable_data()
        pipeline = ThresholdCalibrationPipeline()
        pipeline.calibrate_from_data(scores, labels, threshold_name="anomaly.custom")
        record = pipeline.get_threshold("anomaly.custom")
        assert record is not None
        assert record.status is ThresholdStatus.CALIBRATED

    def test_multicolumn_features_use_row_mean_scores(self) -> None:
        """For (n, d>1) input the per-row mean acts as the score."""
        # Row means: [0.1, 0.25, 0.8, 1.0] -- separable at the same split.
        X = np.array([[0.0, 0.2], [0.2, 0.3], [0.7, 0.9], [0.9, 1.1]])
        labels = np.array([0, 0, 1, 1], dtype=np.int32)
        pipeline = ThresholdCalibrationPipeline()
        result = pipeline.calibrate_from_data(X, labels, n_candidate_thresholds=10)
        assert result.metric_value == 1.0
        assert 0.25 <= result.threshold < 0.8

    def test_single_column_matrix_equivalent_to_vector(self) -> None:
        """(n, 1) input is ravelled to a score vector."""
        scores, labels = _separable_data()
        pipeline = ThresholdCalibrationPipeline()
        result = pipeline.calibrate_from_data(
            scores.reshape(-1, 1), labels, n_candidate_thresholds=5
        )
        assert result.threshold == 0.25
        assert result.metric_value == 1.0

    def test_result_to_dict_is_json_serializable(self) -> None:
        """ThresholdResult serializes enums and nested fingerprints."""
        scores, labels = _separable_data()
        result = ThresholdCalibrationPipeline().calibrate_from_data(scores, labels)
        d = result.to_dict()
        assert d["strategy"] == "youden_j"
        assert d["dataset_fingerprint"]["n_samples"] == 4
        assert json.loads(json.dumps(d)) == d


# ---------------------------------------------------------------------------
# calibrate_from_data: F1-optimal
# ---------------------------------------------------------------------------


class TestCalibrateF1Optimal:
    """F1 = 2PR/(P+R) threshold selection."""

    def test_f1_closed_form_on_interleaved_scores(self) -> None:
        """Candidates [.1,.2,.3,.4]: best is t=0.1 with F1=0.8."""
        scores, labels = _interleaved_data()
        pipeline = ThresholdCalibrationPipeline()
        result = pipeline.calibrate_from_data(
            scores, labels, method=CalibrationStrategy.F1_OPTIMAL, n_candidate_thresholds=4
        )
        # t=0.1: pred={.2,.3,.4}: TP=2, FP=1, FN=0 -> P=2/3, R=1, F1=0.8.
        # t=0.2: F1=0.5;  t=0.3: F1=2/3;  t=0.4: F1=0.
        assert result.threshold == pytest.approx(0.1)
        assert result.metric_value == pytest.approx(0.8)
        assert result.metric_name == "f1"
        assert result.details["precision"] == pytest.approx(2.0 / 3.0)
        assert result.details["recall"] == pytest.approx(1.0)

    def test_f1_perfect_separation_reaches_one(self) -> None:
        """Separable data yields F1 = 1.0 with perfect precision/recall."""
        scores, labels = _separable_data()
        result = ThresholdCalibrationPipeline().calibrate_from_data(
            scores, labels, method="f1_optimal", n_candidate_thresholds=5
        )
        assert result.metric_value == 1.0
        assert result.details == {"precision": 1.0, "recall": 1.0}


# ---------------------------------------------------------------------------
# calibrate_from_data: cost-sensitive
# ---------------------------------------------------------------------------


class TestCalibrateCostSensitive:
    """Cost = c_FP * FP + c_FN * FN threshold selection (negated metric)."""

    def test_high_fn_cost_prefers_low_threshold(self) -> None:
        """c_FN=10: catching both positives (one FP) is cheapest."""
        scores, labels = _interleaved_data()
        result = ThresholdCalibrationPipeline().calibrate_from_data(
            scores,
            labels,
            method=CalibrationStrategy.COST_SENSITIVE,
            cost_fp=1.0,
            cost_fn=10.0,
            n_candidate_thresholds=4,
        )
        # t=0.1: FP=1, FN=0 -> cost 1.  t=0.2: 11.  t=0.3: 10.  t=0.4: 20.
        assert result.threshold == pytest.approx(0.1)
        assert result.metric_value == -1.0
        assert result.metric_name == "neg_expected_cost"
        assert result.details["n_false_positives"] == 1
        assert result.details["n_false_negatives"] == 0
        assert result.details["total_cost"] == 1.0

    def test_high_fp_cost_prefers_high_threshold(self) -> None:
        """c_FP=10: sacrificing one positive (no FPs) is cheapest."""
        scores, labels = _interleaved_data()
        result = ThresholdCalibrationPipeline().calibrate_from_data(
            scores,
            labels,
            method="cost_sensitive",
            cost_fp=10.0,
            cost_fn=1.0,
            n_candidate_thresholds=4,
        )
        # t=0.1: cost 10.  t=0.2: 11.  t=0.3: FP=0, FN=1 -> cost 1.  t=0.4: 2.
        assert result.threshold == pytest.approx(0.3)
        assert result.metric_value == -1.0
        assert result.details["n_false_positives"] == 0
        assert result.details["n_false_negatives"] == 1

    def test_cost_matrix_recorded_in_details(self) -> None:
        """The supplied cost matrix is echoed back in the details."""
        scores, labels = _separable_data()
        result = ThresholdCalibrationPipeline().calibrate_from_data(
            scores, labels, method="cost_sensitive", cost_fp=2.5, cost_fn=7.5
        )
        assert result.details["cost_fp"] == 2.5
        assert result.details["cost_fn"] == 7.5
        # Separable data: zero misclassification cost is attainable.
        assert result.metric_value == 0.0


# ---------------------------------------------------------------------------
# calibrate_from_data: error paths
# ---------------------------------------------------------------------------


class TestCalibrateErrors:
    """Input validation for single-threshold calibration."""

    def test_single_class_all_negative_raises(self) -> None:
        """y with only class 0 is rejected."""
        pipeline = ThresholdCalibrationPipeline()
        with pytest.raises(ValueError, match="both classes"):
            pipeline.calibrate_from_data(np.array([0.1, 0.2, 0.3]), np.array([0, 0, 0]))

    def test_single_class_all_positive_raises(self) -> None:
        """y with only class 1 is rejected."""
        pipeline = ThresholdCalibrationPipeline()
        with pytest.raises(ValueError, match="both classes"):
            pipeline.calibrate_from_data(np.array([0.1, 0.2, 0.3]), np.array([1, 1, 1]))

    def test_length_mismatch_raises(self) -> None:
        """X and y with different sample counts are rejected."""
        pipeline = ThresholdCalibrationPipeline()
        with pytest.raises(ValueError, match="same number of samples"):
            pipeline.calibrate_from_data(np.array([0.1, 0.2, 0.3]), np.array([0, 1]))

    def test_unknown_strategy_string_raises(self) -> None:
        """An unrecognized strategy string is rejected by the enum."""
        scores, labels = _separable_data()
        with pytest.raises(ValueError):
            ThresholdCalibrationPipeline().calibrate_from_data(scores, labels, method="bogus")

    def test_non_enum_non_string_method_raises(self) -> None:
        """A method that is neither enum nor string hits the guard clause."""
        scores, labels = _separable_data()
        with pytest.raises(ValueError, match="Unknown calibration strategy"):
            ThresholdCalibrationPipeline().calibrate_from_data(
                scores, labels, method=None  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# calibrate_all_thresholds
# ---------------------------------------------------------------------------


class TestCalibrateAllThresholds:
    """System-wide sweep: anomaly, ethical, and confidence thresholds."""

    def test_result_keys_for_two_class_data(self) -> None:
        """2 anomaly + 7 ethical + 4 confidence results; guardrails skipped."""
        scores, labels = _separable_data()
        pipeline = ThresholdCalibrationPipeline()
        results = pipeline.calibrate_all_thresholds(scores, labels)

        assert len(results) == 13
        assert "anomaly.default_threshold" in results
        assert "anomaly.zscore_threshold" in results
        assert "anomaly.max_threshold_cap" not in results
        assert "anomaly.min_threshold_floor" not in results
        assert sum(1 for k in results if k.startswith("ethical.")) == 7
        assert sum(1 for k in results if k.startswith("confidence.")) == 4

        # Guardrail records remain untouched and uncalibrated.
        assert sorted(pipeline.uncalibrated_thresholds) == [
            "anomaly.max_threshold_cap",
            "anomaly.min_threshold_floor",
        ]

    def test_anomaly_result_clamped_to_max_cap(self) -> None:
        """Scores in [5, 10] force the returned threshold onto the cap."""
        scores = np.concatenate([np.linspace(5.0, 7.0, 20), np.linspace(8.0, 10.0, 20)])
        labels = np.array([0] * 20 + [1] * 20, dtype=np.int32)
        results = ThresholdCalibrationPipeline().calibrate_all_thresholds(scores, labels)
        assert results["anomaly.default_threshold"].threshold == ANOMALY.MAX_THRESHOLD_CAP

    def test_anomaly_result_clamped_to_min_floor(self) -> None:
        """Scores in [-10, -5] force the returned threshold onto the floor."""
        scores = np.concatenate([np.linspace(-10.0, -8.0, 20), np.linspace(-7.0, -5.0, 20)])
        labels = np.array([0] * 20 + [1] * 20, dtype=np.int32)
        results = ThresholdCalibrationPipeline().calibrate_all_thresholds(scores, labels)
        assert results["anomaly.default_threshold"].threshold == ANOMALY.MIN_THRESHOLD_FLOOR

    def test_registry_value_respects_guardrail_cap(self) -> None:
        """The registry record must not exceed the system cap after clamping.

        Regression: calibrate_all_thresholds clamped only the *returned*
        ThresholdResult while the registry kept the unclamped optimum, so
        get_threshold() served an operating point violating
        MAX_THRESHOLD_CAP.  The registry now stores the clamped value with
        a ``guardrail_clamped_from`` provenance breadcrumb.
        """
        scores = np.concatenate([np.linspace(5.0, 7.0, 20), np.linspace(8.0, 10.0, 20)])
        labels = np.array([0] * 20 + [1] * 20, dtype=np.int32)
        pipeline = ThresholdCalibrationPipeline()
        results = pipeline.calibrate_all_thresholds(scores, labels)
        record = pipeline.get_threshold("anomaly.default_threshold")
        assert record is not None
        assert record.value <= ANOMALY.MAX_THRESHOLD_CAP
        # Registry and returned result agree on the served operating point.
        assert record.value == results["anomaly.default_threshold"].threshold
        # Provenance: the raw pre-clamp optimum is preserved in metadata.
        raw = record.metadata["guardrail_clamped_from"]
        assert raw > ANOMALY.MAX_THRESHOLD_CAP

    def test_ethical_floors_preserved_with_provenance(self) -> None:
        """Ethical thresholds keep their design values but gain provenance."""
        scores, labels = _separable_data()
        pipeline = ThresholdCalibrationPipeline()
        results = pipeline.calibrate_all_thresholds(scores, labels)

        assert results["ethical.ethical_minimum"].threshold == ETHICAL.ETHICAL_MINIMUM
        assert (
            results["ethical.sigma_immutable_default"].threshold == ETHICAL.SIGMA_IMMUTABLE_DEFAULT
        )
        assert results["ethical.ethical_minimum"].metric_name == "ethical_floor"

        record = pipeline.get_threshold("ethical.ethical_minimum")
        assert record is not None
        assert record.status is ThresholdStatus.CALIBRATED
        assert record.strategy == "ethical_floor_verification"
        assert record.dataset_fingerprint is not None

    def test_confidence_bands_ordered_and_normalized(self) -> None:
        """Bands are quantile-ordered and normalized into [0, 1]."""
        rng = np.random.default_rng(42)
        scores = np.concatenate([rng.uniform(0.0, 0.4, 60), rng.uniform(0.6, 1.0, 40)])
        labels = np.array([0] * 60 + [1] * 40, dtype=np.int32)
        pipeline = ThresholdCalibrationPipeline()
        results = pipeline.calibrate_all_thresholds(scores, labels)

        high = results["confidence.high"].threshold
        medium = results["confidence.medium"].threshold
        low = results["confidence.low"].threshold
        minimum = results["confidence.minimum_actionable"].threshold
        assert 0.0 <= minimum <= low <= medium <= high <= 1.0

        record = pipeline.get_threshold("confidence.high")
        assert record is not None
        assert record.strategy == "quantile_on_correct_predictions"
        assert record.metadata == {"quantile": 90.0}

    def test_constant_scores_confidence_bands_fall_back_to_quantiles(self) -> None:
        """Degenerate constant scores map each band to quantile/100."""
        scores = np.full(40, 0.5)
        labels = np.array([0] * 20 + [1] * 20, dtype=np.int32)
        results = ThresholdCalibrationPipeline().calibrate_all_thresholds(scores, labels)
        assert results["confidence.high"].threshold == 0.9
        assert results["confidence.medium"].threshold == 0.7
        assert results["confidence.low"].threshold == 0.5
        assert results["confidence.minimum_actionable"].threshold == 0.3

    def test_single_class_labels_degrade_gracefully(self) -> None:
        """Single-class y: anomaly keys fail (logged) but the sweep continues."""
        rng = np.random.default_rng(0)
        scores = rng.normal(0.5, 0.1, 50)
        labels = np.zeros(50, dtype=np.int32)
        results = ThresholdCalibrationPipeline().calibrate_all_thresholds(scores, labels)
        assert not any(k.startswith("anomaly.") for k in results)
        assert sum(1 for k in results if k.startswith("ethical.")) == 7
        assert sum(1 for k in results if k.startswith("confidence.")) == 4

    def test_unknown_confidence_key_is_skipped(self) -> None:
        """Custom confidence.* keys without a quantile mapping are ignored."""
        scores, labels = _separable_data()
        pipeline = ThresholdCalibrationPipeline()
        pipeline.set_threshold("confidence.custom_band", 0.42)
        results = pipeline.calibrate_all_thresholds(scores, labels)
        assert "confidence.custom_band" not in results
        record = pipeline.get_threshold("confidence.custom_band")
        assert record is not None
        assert record.value == 0.42
        assert record.status is ThresholdStatus.UNCALIBRATED

    def test_string_method_accepted(self) -> None:
        """The sweep accepts a strategy given as a plain string."""
        scores, labels = _separable_data()
        results = ThresholdCalibrationPipeline().calibrate_all_thresholds(
            scores, labels, method="f1_optimal"
        )
        assert results["anomaly.default_threshold"].metric_name == "f1"


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------


class TestDetectDrift:
    """KL/KS drift detection and the STALE lifecycle."""

    def test_identical_data_reports_no_drift(self) -> None:
        """X_new == X_cal: zero divergence, p = 1, no drift."""
        rng = np.random.default_rng(42)
        X = rng.normal(0.0, 1.0, 200)
        result = ThresholdCalibrationPipeline().detect_drift(X, X)
        assert result.drifted is False
        assert result.kl_divergence == pytest.approx(0.0, abs=1e-9)
        assert result.ks_statistic == 0.0
        assert result.ks_p_value == 1.0
        assert result.message == "No significant drift detected"

    def test_same_distribution_large_sample_no_drift(self) -> None:
        """Independent large samples from one distribution do not drift."""
        rng = np.random.default_rng(42)
        X_cal = rng.normal(0.0, 1.0, 2000)
        X_new = rng.normal(0.0, 1.0, 2000)
        result = ThresholdCalibrationPipeline().detect_drift(X_new, X_cal)
        assert result.drifted is False
        assert result.kl_divergence < 0.1
        assert result.ks_p_value > 0.05

    def test_shifted_distribution_detected_by_both_tests(self) -> None:
        """N(0,1) vs N(8,1): disjoint supports trip both KL and KS."""
        rng = np.random.default_rng(42)
        X_cal = rng.normal(0.0, 1.0, 400)
        X_new = rng.normal(8.0, 1.0, 400)
        result = ThresholdCalibrationPipeline().detect_drift(X_new, X_cal)
        assert result.drifted is True
        assert result.ks_statistic == 1.0
        assert result.ks_p_value < 0.05
        assert result.kl_divergence > 0.1
        assert "KL divergence" in result.message
        assert "KS p-value" in result.message

    def test_kl_only_trigger(self) -> None:
        """ks_alpha=0 disables KS; a tiny kl_threshold still flags drift."""
        rng = np.random.default_rng(7)
        X_cal = rng.normal(0.0, 1.0, 400)
        X_new = rng.normal(0.0, 1.0, 400)
        pipeline = ThresholdCalibrationPipeline(ks_alpha=0.0, kl_threshold=1e-9)
        result = pipeline.detect_drift(X_new, X_cal)
        assert result.drifted is True
        assert "KL divergence" in result.message
        assert "KS p-value" not in result.message

    def test_ks_only_trigger(self) -> None:
        """kl_threshold=inf disables KL; the KS test alone flags drift."""
        rng = np.random.default_rng(42)
        X_cal = rng.normal(0.0, 1.0, 400)
        X_new = rng.normal(8.0, 1.0, 400)
        pipeline = ThresholdCalibrationPipeline(kl_threshold=np.inf)
        result = pipeline.detect_drift(X_new, X_cal)
        assert result.drifted is True
        assert "KS p-value" in result.message
        assert "KL divergence" not in result.message

    def test_drift_marks_calibrated_thresholds_stale(self) -> None:
        """CALIBRATED -> STALE on drift; UNCALIBRATED records untouched."""
        scores, labels = _separable_data()
        pipeline = ThresholdCalibrationPipeline()
        pipeline.calibrate_from_data(scores, labels)

        drifted_data = np.array([100.0, 101.0, 102.0, 103.0] * 20)
        result = pipeline.detect_drift(drifted_data, np.linspace(0.0, 1.0, 80))
        assert result.drifted is True

        record = pipeline.get_threshold("anomaly.default_threshold")
        assert record is not None
        assert record.status is ThresholdStatus.STALE
        assert pipeline.stale_thresholds == ["anomaly.default_threshold"]
        assert "ethical.ethical_minimum" in pipeline.uncalibrated_thresholds

    def test_no_drift_preserves_calibrated_status(self) -> None:
        """A clean drift check leaves CALIBRATED records calibrated."""
        scores, labels = _separable_data()
        pipeline = ThresholdCalibrationPipeline()
        pipeline.calibrate_from_data(scores, labels)
        result = pipeline.detect_drift(scores, scores)
        assert result.drifted is False
        record = pipeline.get_threshold("anomaly.default_threshold")
        assert record is not None
        assert record.status is ThresholdStatus.CALIBRATED
        assert pipeline.stale_thresholds == []

    def test_constant_identical_data_no_drift(self) -> None:
        """Degenerate constant distributions compare as identical."""
        X = np.full(50, 3.0)
        result = ThresholdCalibrationPipeline().detect_drift(X, X.copy())
        assert result.drifted is False
        assert result.kl_divergence == pytest.approx(0.0, abs=1e-12)

    def test_per_feature_outputs_use_common_feature_count(self) -> None:
        """Per-feature lists span min(d_new, d_cal) features."""
        rng = np.random.default_rng(3)
        X_cal = rng.normal(size=(200, 3))
        X_new = rng.normal(size=(200, 2))
        result = ThresholdCalibrationPipeline().detect_drift(X_new, X_cal)
        assert len(result.per_feature_ks) == 2
        assert len(result.per_feature_kl) == 2

    def test_zero_width_inputs_default_to_no_drift(self) -> None:
        """Feature-less arrays yield the documented neutral defaults."""
        result = ThresholdCalibrationPipeline().detect_drift(np.empty((5, 0)), np.empty((7, 0)))
        assert result.drifted is False
        assert result.kl_divergence == 0.0
        assert result.ks_statistic == 0.0
        assert result.ks_p_value == 1.0
        assert result.per_feature_ks == []
        assert result.per_feature_kl == []

    def test_drift_result_to_dict_is_json_serializable(self) -> None:
        """DriftResult serializes to plain JSON types."""
        rng = np.random.default_rng(42)
        X = rng.normal(0.0, 1.0, 100)
        result = ThresholdCalibrationPipeline().detect_drift(X, X)
        d = result.to_dict()
        assert set(d) == {
            "drifted",
            "kl_divergence",
            "ks_statistic",
            "ks_p_value",
            "per_feature_ks",
            "per_feature_kl",
            "message",
        }
        assert json.loads(json.dumps(d)) == d
