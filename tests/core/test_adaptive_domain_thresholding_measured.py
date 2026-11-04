# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``adaptive_domain_thresholding`` measured against the shipped operating point.

``core/adaptive_domain_thresholding.py`` is 1,289 LOC promising "per-domain
threshold calibration", Platt scaling, isotonic regression and a "domain
ensemble weighting optimizer" to "improve domain competence across Medical,
Financial, and Infrastructure". No path in ``src/`` reaches it (verified by
transitive import closure from the engine, the API, the CLI and every manifest
entry) and it had never been measured against the operating point Mercury
actually ships. This suite measures it on real labelled data and pins the
result, so the module's status rests on evidence instead of its docstring.

What the measurement found, and what these tests hold:

1. **The supervised path is broken, not a tradeoff.** Given a labelled
   calibration set it flags **99.3%** of normal rows on ``pima_real``, 95.7% on
   ``breastw`` and 97.9% on ``wbc`` — mean held-out FPR **0.6335**, and a mean
   F1 (0.3882) *below* the shipped conformal cut's 0.4059 despite flagging
   sixteen times as much. A calibration that flags almost every normal row
   cannot be used, and these tests pin that it is refused rather than reached.
2. **The label-free path is not an improvement either.** Its mean F1 (0.4549 vs
   0.4059) looks higher, but it abandons the false-alarm guarantee — FPR
   0.013-0.160, over the promised alpha=0.05 on three of five tables and up to
   3.2x the budget — and the F1 mean is carried by one table (``breastw``
   0.935 vs 0.253) while it loses 0.36 on ``wbc`` and 0.31 on ``thyroid``. The
   shipped cut never exceeds its alpha on any table or seed. On this
   repository's pre-registered-bar rule that is a refusal, not a promotion.
3. **A real numerical defect, fixed.** ``PlattScalingCalibrator`` evaluated
   ``1 / (1 + np.exp(A * s + B))`` directly, which overflows ``np.exp`` on real
   score distributions. Its comment claimed "clipping for numerical stability",
   but the clip was applied to the *result*, after the overflow had already
   happened inside ``np.exp``; ``calibrate()`` had no guard at all. Both now use
   a stable sigmoid, which is value-preserving: every number above is unchanged.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from omni_mercury_engine.core.adaptive_domain_thresholding import (
    AdaptiveDomainThresholdManager,
    DomainType,
    PlattScalingCalibrator,
)
from omni_mercury_engine.detectors._calibration import conformal_cut
from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector
from tests.recorded_data import ADBENCH_NAMES, adbench, stratified_split

SEEDS = (0, 1, 2)
TOLERANCE = 0.06

#: Mean held-out (F1, FPR) over 5 recorded ADBench tables x 3 seeds, scoring
#: the same detector's scores under three thresholding policies.
MEASURED: dict[str, tuple[float, float]] = {
    "conformal": (0.4059, 0.0231),
    "adt_unsupervised": (0.4549, 0.0873),
    "adt_supervised": (0.3882, 0.6335),
}

#: Tables where the supervised path flags essentially every normal row.
MEASURED_SUPERVISED_COLLAPSE: dict[str, float] = {
    "pima_real": 0.987,
    "breastw": 0.957,
    "wbc": 0.979,
}


def _metrics(flags: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Return ``(f1, fpr)`` for binary flags against binary labels."""
    true_pos = int(((flags == 1) & (labels == 1)).sum())
    false_pos = int(((flags == 1) & (labels == 0)).sum())
    false_neg = int(((flags == 0) & (labels == 1)).sum())
    true_neg = int(((flags == 0) & (labels == 0)).sum())
    precision = true_pos / (true_pos + false_pos) if true_pos + false_pos else 0.0
    recall = true_pos / (true_pos + false_neg) if true_pos + false_neg else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = false_pos / (false_pos + true_neg) if false_pos + true_neg else 0.0
    return f1, fpr


def _threshold_of(manager: AdaptiveDomainThresholdManager) -> float:
    """The manager's current threshold.

    ``get_threshold`` is documented to return a bare ``float`` when called
    with no arguments and an info ``dict`` when given a score or confidence;
    its annotation is the union of both, so the no-arg case is narrowed here.
    """
    threshold = manager.get_threshold()
    assert not isinstance(
        threshold, dict
    ), "get_threshold() with no arguments is documented to return a float"
    return float(threshold)


@pytest.fixture(scope="module")
def operating_points() -> dict[str, dict[str, list[tuple[float, float]]]]:
    """Per-table ``(f1, fpr)`` for each policy, on the recorded tables."""
    out: dict[str, dict[str, list[tuple[float, float]]]] = {policy: {} for policy in MEASURED}
    for table in ADBENCH_NAMES:
        features, labels = adbench(table)
        for per_table in out.values():
            per_table[table] = []
        for seed in SEEDS:
            train, test = stratified_split(labels, seed)
            mean = features[train].mean(axis=0)
            spread = features[train].std(axis=0)
            spread[spread < 1e-8] = 1.0
            detector = MercuryAnomalyDetector()
            detector.fit((features[train] - mean) / spread)
            fit_scores = np.sort(
                np.asarray(
                    detector.detect((features[train] - mean) / spread)["scores"],
                    dtype=float,
                ).ravel()
            )
            test_scores = np.asarray(
                detector.detect((features[test] - mean) / spread)["scores"], dtype=float
            ).ravel()
            held = labels[test]

            cut = conformal_cut(fit_scores, 0.05)
            out["conformal"][table].append(_metrics((test_scores > cut).astype(int), held))
            for policy, supervised in (
                ("adt_unsupervised", None),
                ("adt_supervised", labels[train].astype(np.int32)),
            ):
                manager = AdaptiveDomainThresholdManager(DomainType.FINANCIAL)
                if supervised is None:
                    manager.fit(fit_scores)
                else:
                    manager.fit(fit_scores, supervised)
                threshold = _threshold_of(manager)
                out[policy][table].append(_metrics((test_scores > threshold).astype(int), held))
    return out


def _pooled(per_table: dict[str, list[tuple[float, float]]]) -> np.ndarray:
    return np.asarray([row for rows in per_table.values() for row in rows])


@pytest.mark.parametrize("policy", sorted(MEASURED))
def test_each_policy_reproduces_its_measurement(
    policy: str, operating_points: dict[str, dict[str, list[tuple[float, float]]]]
) -> None:
    """Each thresholding policy reproduces its recorded (F1, FPR)."""
    pinned_f1, pinned_fpr = MEASURED[policy]
    pooled = _pooled(operating_points[policy])
    assert pooled[:, 0].mean() == pytest.approx(
        pinned_f1, abs=TOLERANCE
    ), f"{policy}: F1 {pooled[:, 0].mean():.4f} vs measured {pinned_f1:.4f}"
    assert pooled[:, 1].mean() == pytest.approx(
        pinned_fpr, abs=TOLERANCE
    ), f"{policy}: FPR {pooled[:, 1].mean():.4f} vs measured {pinned_fpr:.4f}"


def test_the_shipped_cut_never_exceeds_its_promised_alpha(
    operating_points: dict[str, dict[str, list[tuple[float, float]]]],
) -> None:
    """The conformal operating point holds alpha=0.05 on every table and seed.

    This is the guarantee the candidate has to beat, and the reason a higher
    mean F1 at a higher false-alarm rate is not an improvement.
    """
    pooled = _pooled(operating_points["conformal"])
    assert (
        pooled[:, 1].max() <= 0.05 + 1e-9
    ), f"the shipped cut exceeded its own alpha: max FPR {pooled[:, 1].max():.4f}"


def test_the_supervised_path_collapses_and_is_therefore_not_wired(
    operating_points: dict[str, dict[str, list[tuple[float, float]]]],
) -> None:
    """Given labels, the module flags almost every normal row on three tables.

    Pinned as a defect, not a tradeoff: the mean F1 is *below* the shipped cut
    while the false-alarm rate is roughly 27x higher.
    """
    per_table = operating_points["adt_supervised"]
    for table, pinned_fpr in MEASURED_SUPERVISED_COLLAPSE.items():
        got = np.asarray(per_table[table])[:, 1].mean()
        assert got == pytest.approx(
            pinned_fpr, abs=0.10
        ), f"{table}: supervised FPR {got:.4f} vs measured {pinned_fpr:.4f}"
        assert got > 0.9, f"{table}: expected the measured collapse, got {got:.4f}"

    pooled = _pooled(per_table)
    conformal = _pooled(operating_points["conformal"])
    assert pooled[:, 0].mean() < conformal[:, 0].mean(), (
        "the supervised path is recorded as worse than the shipped cut on F1 "
        "as well as on false alarms; if that changes, re-measure before wiring"
    )


def test_the_label_free_path_does_not_clear_the_bar(
    operating_points: dict[str, dict[str, list[tuple[float, float]]]],
) -> None:
    """Higher mean F1, but it breaks the budget and loses badly on two tables.

    Pins the evidence for the refusal rather than the refusal itself: the mean
    F1 edge is carried by one table while two others lose by 0.3+, and the
    false-alarm rate goes over the promised alpha on the majority of tables.
    """
    adt = operating_points["adt_unsupervised"]
    conformal = operating_points["conformal"]

    over_budget = [table for table in adt if np.asarray(adt[table])[:, 1].mean() > 0.05]
    assert (
        len(over_budget) >= 3
    ), f"expected the measured budget violation on most tables, got {over_budget}"

    losses = {
        table: np.asarray(conformal[table])[:, 0].mean() - np.asarray(adt[table])[:, 0].mean()
        for table in adt
    }
    assert max(losses.values()) > 0.25, (
        "expected the measured large per-table F1 losses that the mean hides; " f"got {losses}"
    )


class TestPlattNumericalStability:
    """The overflow defect in ``PlattScalingCalibrator``, fixed and pinned."""

    def test_calibrate_does_not_overflow_on_extreme_exponents(self) -> None:
        """``calibrate`` saturates instead of overflowing ``np.exp``.

        The former ``1 / (1 + np.exp(A * s + B))`` raised ``RuntimeWarning:
        overflow encountered in exp`` on real ADBench score distributions, and
        had no guard of any kind.
        """
        calibrator = PlattScalingCalibrator()
        calibrator.A, calibrator.B = -1.0, 0.0
        calibrator._fitted = True
        extreme = np.array([-1e6, -800.0, 0.0, 800.0, 1e6], dtype=np.float64)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            out = calibrator.calibrate(extreme)

        assert np.all(np.isfinite(out))
        assert np.all((out >= 0.0) & (out <= 1.0))
        # Monotone increasing in the score (A is negative).
        assert np.all(np.diff(out) >= 0.0)
        assert out[0] == pytest.approx(0.0)
        assert out[-1] == pytest.approx(1.0)

    def test_fit_does_not_overflow_and_stays_value_preserving(self) -> None:
        """Fitting is warning-free, and the stable form fits the same sigmoid."""
        rng = np.random.default_rng(0)
        scores = np.concatenate([rng.normal(0.3, 0.08, 400), rng.normal(0.8, 0.08, 40)]).clip(
            0.0, 1.0
        )
        labels = np.concatenate([np.zeros(400, dtype=np.int32), np.ones(40, dtype=np.int32)])

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            calibrator = PlattScalingCalibrator().fit(scores, labels)
            probs = calibrator.calibrate(scores)

        # A negative A is Platt's convention for a score where higher means
        # more likely positive; the fit must actually move off its init.
        assert calibrator.A < 0.0
        assert pytest.approx(-1.0) != calibrator.A
        assert np.all(np.isfinite(probs))
        # The calibrated probability separates the two populations.
        assert probs[labels == 1].mean() > probs[labels == 0].mean()
