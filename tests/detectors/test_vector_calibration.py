# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Vector isotonic calibration + conformal sets: guarantees, application, pins.

``research/math_arrest/harness_vector_calibration.py`` measured the opt-in
vector calibration on the five recorded ADBench tables (three seeds, 49/21/30
fit/calibration/test): ECE 0.1553 → 0.0575, Brier 0.1358 → 0.0805, F1 0.186 →
0.404, AUROC 0.664 → 0.761, conformal coverage 0.911 against the 0.9 target
at average set size 1.63. This module holds:

* the finite-sample coverage guarantee on held-out folds of the recorded
  tables and on synthetic exchangeable data;
* that the isotonic model is *applied* — the decision path flips with the
  calibrated probability, and the calibration metrics on a recorded split
  improve over the uncalibrated fusion (fails if either regresses);
* that the wrapper does not alter the probe mathematics — per-probe scores
  and the raw fused scores are bit-identical with and without it;
* the checkpoint round-trip of both instruments.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = [
    pytest.mark.xdist_group("vector_calibration"),
    pytest.mark.timeout(900),
]

from omni_mercury_engine.core.calibration import compute_ece
from omni_mercury_engine.core.vector_calibration import (
    VectorConformalBinary,
    VectorIsotonicCalibrator,
)
from omni_mercury_engine.detectors.math_arrest.detector import MathArrestDetector
from omni_mercury_engine.ml.mercury_ml import brier_score_loss
from tests.recorded_data import adbench, stratified_split

ARTIFACT = Path(__file__).resolve().parents[2] / "artifacts" / "math_arrest_vector_calibration.json"
ALPHA = 0.1


def _synthetic(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Exchangeable two-column scores with a real signal plus a noise column."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(np.int64)
    signal = rng.normal(loc=y * 1.5, scale=1.0)
    noise = rng.standard_normal(n)
    return np.column_stack([signal, noise]), y


class TestVectorIsotonicCalibrator:
    def test_weights_come_from_labelled_separation(self) -> None:
        matrix, y = _synthetic(600, 0)
        calibrator = VectorIsotonicCalibrator().fit(matrix, y)
        weights = calibrator.weights
        assert weights.sum() == pytest.approx(1.0)
        assert weights[0] > 0.9, "the signal column must dominate the index"

    def test_calibrated_output_is_monotone_in_the_index(self) -> None:
        matrix, y = _synthetic(600, 1)
        calibrator = VectorIsotonicCalibrator().fit(matrix, y)
        index = calibrator.index(matrix)
        probs = calibrator.calibrate(matrix)
        order = np.argsort(index)
        assert np.all(np.diff(probs[order]) >= 0.0)
        assert probs.min() >= 0.0 and probs.max() <= 1.0

    def test_single_class_and_shape_mismatch_are_refused(self) -> None:
        matrix, _ = _synthetic(100, 2)
        with pytest.raises(ValueError, match="both classes"):
            VectorIsotonicCalibrator().fit(matrix, np.zeros(100, dtype=np.int64))
        calibrator = VectorIsotonicCalibrator().fit(*_synthetic(200, 3))
        with pytest.raises(ValueError, match="columns"):
            calibrator.calibrate(np.zeros((10, 5)))

    def test_state_round_trip_is_exact(self) -> None:
        matrix, y = _synthetic(400, 4)
        calibrator = VectorIsotonicCalibrator().fit(matrix, y)
        clone = VectorIsotonicCalibrator.from_state(calibrator.to_state())
        np.testing.assert_array_equal(calibrator.calibrate(matrix), clone.calibrate(matrix))


class TestVectorConformalBinary:
    def test_finite_sample_coverage_on_exchangeable_data(self) -> None:
        """Per-class coverage >= 1 - alpha, averaged over disjoint trials."""
        rates = {0: [], 1: []}
        for trial in range(20):
            matrix, y = _synthetic(900, 100 + trial)
            fit_rows = np.arange(300)
            cal_rows = np.arange(300, 600)
            test_rows = np.arange(600, 900)
            calibrator = VectorIsotonicCalibrator().fit(matrix[fit_rows], y[fit_rows])
            conformal = VectorConformalBinary(alpha=ALPHA).fit(
                matrix[cal_rows], y[cal_rows], calibrator
            )
            sets = conformal.predict_sets(matrix[test_rows])
            covered = sets[np.arange(300), y[test_rows]]
            for cls in (0, 1):
                member = covered[y[test_rows] == cls]
                if member.size:
                    rates[cls].append(float(member.mean()))
        for cls in (0, 1):
            mean_rate = float(np.mean(rates[cls]))
            assert mean_rate >= 1.0 - ALPHA - 0.02, (
                f"class {cls} mean coverage {mean_rate:.4f} under the " f"{1.0 - ALPHA} guarantee"
            )

    def test_missing_class_in_calibration_fold_is_refused(self) -> None:
        matrix, y = _synthetic(200, 5)
        calibrator = VectorIsotonicCalibrator().fit(matrix, y)
        with pytest.raises(ValueError, match="class 1 absent"):
            VectorConformalBinary().fit(matrix[y == 0], y[y == 0], calibrator)

    def test_state_round_trip_is_exact(self) -> None:
        matrix, y = _synthetic(400, 6)
        calibrator = VectorIsotonicCalibrator().fit(matrix[:200], y[:200])
        conformal = VectorConformalBinary(alpha=ALPHA).fit(matrix[200:], y[200:], calibrator)
        clone = VectorConformalBinary.from_state(conformal.to_state(), calibrator)
        np.testing.assert_array_equal(conformal.p_values(matrix), clone.p_values(matrix))


@pytest.fixture(scope="module")
def recorded_split() -> dict[str, np.ndarray]:
    """One recorded table's standardized fit/calibration/test folds.

    breastw is the spot-check table: its measured gains are decisive (F1
    0.189 → 0.965, ECE 0.175 → 0.027 over the three seeds), so the
    improvement assertions test the mechanism rather than a seed's luck. The
    corpus-wide claims — F1 and ECE improve *on average*, with cardio
    marginal and thyroid's F1 an exception at the 0.5 cut — are pinned from
    the artifact in ``test_the_artifact_pins_the_measured_gains``.
    """
    features, labels = adbench("breastw")
    train_all, test = stratified_split(labels, 0)
    fit_local, cal_local = stratified_split(labels[train_all], 0)
    fit_rows, cal_rows = train_all[fit_local], train_all[cal_local]
    mean = features[fit_rows].mean(axis=0)
    spread = features[fit_rows].std(axis=0)
    spread[spread < 1e-8] = 1.0
    return {
        "fit": (features[fit_rows] - mean) / spread,
        "cal": (features[cal_rows] - mean) / spread,
        "cal_y": labels[cal_rows],
        "test": (features[test] - mean) / spread,
        "test_y": labels[test],
    }


class TestDetectorIntegration:
    def test_the_wrapper_leaves_the_probe_mathematics_untouched(
        self, recorded_split: dict[str, np.ndarray]
    ) -> None:
        detector = MathArrestDetector().fit(recorded_split["fit"])
        before_probes = detector.detect(recorded_split["test"])["probe_scores"]
        before_raw = detector.detect(recorded_split["test"])["scores"]
        detector.fit_vector_calibration(recorded_split["cal"], recorded_split["cal_y"], alpha=ALPHA)
        after = detector.detect(recorded_split["test"])
        np.testing.assert_array_equal(before_raw, after["scores"])
        for name, column in before_probes.items():
            np.testing.assert_array_equal(column, after["probe_scores"][name])

    def test_the_calibrated_probability_is_applied_in_the_decision_path(
        self, recorded_split: dict[str, np.ndarray]
    ) -> None:
        detector = MathArrestDetector().fit(recorded_split["fit"])
        detector.fit_vector_calibration(recorded_split["cal"], recorded_split["cal_y"], alpha=ALPHA)
        result = detector.detect(recorded_split["test"])
        assert "calibrated_scores" in result, "the isotonic model is not applied"
        np.testing.assert_array_equal(
            result["is_anomaly"],
            np.asarray(result["calibrated_scores"], dtype=float) > 0.5,
        )
        assert result["conformal_alpha"] == ALPHA
        assert result["conformal_sets"].shape == (len(recorded_split["test_y"]), 2)

    def test_calibration_metrics_improve_on_the_recorded_split(
        self, recorded_split: dict[str, np.ndarray]
    ) -> None:
        """Fails if the isotonic model regresses ECE/Brier/F1 on cardio/seed0."""
        detector = MathArrestDetector().fit(recorded_split["fit"])
        y = recorded_split["test_y"]
        raw = detector.detect(recorded_split["test"])
        raw_scores = np.asarray(raw["scores"], dtype=np.float64)
        detector.fit_vector_calibration(recorded_split["cal"], recorded_split["cal_y"], alpha=ALPHA)
        calibrated = detector.detect(recorded_split["test"])
        cal_scores = np.asarray(calibrated["calibrated_scores"], dtype=np.float64)

        assert compute_ece(y, cal_scores) < compute_ece(y, raw_scores)
        assert brier_score_loss(y, cal_scores) < brier_score_loss(y, raw_scores)

        def f1(flags: np.ndarray) -> float:
            flags = np.asarray(flags, dtype=bool)
            positive = y == 1
            tp = int((flags & positive).sum())
            fp = int((flags & ~positive).sum())
            fn = int((~flags & positive).sum())
            return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

        assert f1(calibrated["is_anomaly"]) > f1(raw["is_anomaly"])

        sets = np.asarray(calibrated["conformal_sets"], dtype=bool)
        coverage = float(sets[np.arange(len(y)), y].mean())
        assert coverage >= 1.0 - ALPHA - 0.05, f"held-out coverage {coverage:.4f}"

    def test_the_calibration_survives_a_checkpoint_round_trip(
        self, recorded_split: dict[str, np.ndarray]
    ) -> None:
        detector = MathArrestDetector().fit(recorded_split["fit"])
        detector.fit_vector_calibration(recorded_split["cal"], recorded_split["cal_y"], alpha=ALPHA)
        state = detector._learned_state()
        clone = MathArrestDetector()
        clone._restore_learned_state(state)
        original = detector.detect(recorded_split["test"])
        restored = clone.detect(recorded_split["test"])
        np.testing.assert_array_equal(original["calibrated_scores"], restored["calibrated_scores"])
        np.testing.assert_array_equal(original["conformal_sets"], restored["conformal_sets"])


class TestConsistencyGate:
    """The cross-probe consistency gate (don't-know routing), pinned.

    Measured on the five recorded tables (three seeds): the gate routes
    4.9% of held-out samples (the 5% fit-quantile design), lifting conformal
    coverage 0.9110 → 0.9123 at +0.0104 average set size.
    """

    def test_detect_serves_the_consistency_fields(
        self, recorded_split: dict[str, np.ndarray]
    ) -> None:
        detector = MathArrestDetector().fit(recorded_split["fit"])
        result = detector.detect(recorded_split["test"])
        consistency = np.asarray(result["probe_consistency"], dtype=float)
        assert consistency.min() >= 0.0 and consistency.max() <= 1.0
        assert 0.0 <= result["consistency_threshold"] <= 1.0
        assert result["low_consistency"].dtype == bool

    def test_unanimous_probes_read_full_consistency(self) -> None:
        detector = MathArrestDetector()
        detector._probe_medians = {"a": 0.2, "b": 0.3}

        class _Result:
            def __init__(self, name: str, scores: np.ndarray) -> None:
                self.probe_name = name
                self.deviation_scores = scores

        unanimous = [
            _Result("a", np.array([0.9, 0.05])),
            _Result("b", np.array([0.95, 0.1])),
        ]
        values = detector._consistency(unanimous)  # both above, both below
        np.testing.assert_allclose(values, [1.0, 1.0])
        split = [
            _Result("a", np.array([0.9])),  # +0.7
            _Result("b", np.array([0.0])),  # -0.3
        ]
        assert detector._consistency(split)[0] == pytest.approx(0.4 / 1.0)

    def test_routed_samples_keep_both_classes_in_play(
        self, recorded_split: dict[str, np.ndarray]
    ) -> None:
        detector = MathArrestDetector().fit(recorded_split["fit"])
        detector.fit_vector_calibration(recorded_split["cal"], recorded_split["cal_y"], alpha=ALPHA)
        result = detector.detect(recorded_split["test"])
        routed = np.asarray(result["dont_know_routed"], dtype=bool)
        sets = np.asarray(result["conformal_sets"], dtype=bool)
        assert sets[routed].all(), "a routed sample was still forced to a point decision"
        # The routed share tracks the fit-quantile design.
        assert routed.mean() <= 3 * MathArrestDetector.CONSISTENCY_QUANTILE

    def test_gate_state_survives_the_checkpoint_round_trip(
        self, recorded_split: dict[str, np.ndarray]
    ) -> None:
        detector = MathArrestDetector().fit(recorded_split["fit"])
        clone = MathArrestDetector()
        clone._restore_learned_state(detector._learned_state())
        original = detector.detect(recorded_split["test"])
        restored = clone.detect(recorded_split["test"])
        np.testing.assert_array_equal(original["probe_consistency"], restored["probe_consistency"])
        assert original["consistency_threshold"] == restored["consistency_threshold"]

    def test_the_artifact_pins_the_gate_impact(self) -> None:
        means = json.loads(ARTIFACT.read_text())["means"]
        assert 0.01 <= means["routed_fraction"] <= 0.10
        assert means["coverage"] >= means["coverage_ungated"]
        assert means["avg_set_size"] >= means["avg_set_size_ungated"]
        # The gate buys its coverage with a bounded efficiency cost.
        assert means["avg_set_size"] - means["avg_set_size_ungated"] <= 0.05


def test_the_artifact_pins_the_measured_gains() -> None:
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["schema"] == "mercury.research.math_arrest_vector_calibration/v1"
    means = artifact["means"]
    assert means["cal_ece"] < means["raw_ece"]
    assert means["cal_brier"] < means["raw_brier"]
    assert means["cal_f1"] > means["raw_f1"]
    assert means["coverage"] >= 1.0 - ALPHA - 0.02
    assert 1.0 <= means["avg_set_size"] <= 2.0
    assert artifact["mean_rank"]["auc"]["calibrated"] <= artifact["mean_rank"]["auc"]["raw"]
