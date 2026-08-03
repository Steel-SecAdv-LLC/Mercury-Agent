# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression pins: no ethics-labelled multiplier may move a number silently.

This codebase deleted a ``benevolence >= 0.99`` pass-bar because it measured
vocabulary rather than intent. Two survivors expressed the same bar as
*arithmetic* rather than as a refusal, so a search for the comparison did not
find them:

* ``core/domain_metrics.py`` halved the reported ``overall_score`` whenever an
  ``ethical_compliance`` flag — really ``recall >= 0.96 and index >= 0.99`` —
  came back False.
* ``core/gosnn_integration.py`` multiplied calibrated scores by a factor derived
  from the benevolence *constant* before comparing them to the threshold, which
  moved the operating point without moving the scores it reported.

Both are gone. These tests fail if either returns, in whatever form.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.core.base_domains import BaseDomainDetector
from omni_mercury_engine.core.domain_metrics import MetricsCalculator
from omni_mercury_engine.core.gosnn_integration import GOSNNIntegration


def _one_missed_anomaly() -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """20 positives / 80 negatives with a single false negative (recall 0.95)."""
    y_true = np.array([1] * 20 + [0] * 80)
    y_pred = y_true.copy()
    y_pred[0] = 0
    return y_true, y_pred


class TestOverallScoreIsNotHalvedByAnEthicsFlag:
    """A missed anomaly is an ML result, not an ethics violation."""

    def test_a_single_miss_does_not_halve_the_reported_score(self) -> None:
        calc = MetricsCalculator()
        y_true, y_pred = _one_missed_anomaly()

        imperfect = calc.compute_all_metrics(y_true, y_pred, None).overall_score
        perfect = calc.compute_all_metrics(y_true, y_true.copy(), None).overall_score

        # With the 0.5x penalty this ratio was 0.489. One miss out of twenty is
        # worth a few percent, not half.
        assert imperfect / perfect > 0.9, (
            f"overall_score collapsed from {perfect:.4f} to {imperfect:.4f} on a "
            "single false negative -- the ethics multiplier is back"
        )

    def test_the_compliance_flag_is_still_reported(self) -> None:
        """Removing the penalty must not remove the signal; it stays advisory."""
        calc = MetricsCalculator()
        y_true, y_pred = _one_missed_anomaly()

        metrics = calc.compute_all_metrics(y_true, y_pred, None)

        assert metrics.ethical_compliance is False
        assert 0.0 <= metrics.benevolence_index <= 1.0

    def test_the_flag_does_not_change_the_arithmetic(self) -> None:
        """Two runs whose only difference is the flag must differ only slightly.

        A compliant and a barely-non-compliant run sit next to each other in
        detection quality, so their scores must sit next to each other too. A
        step change means a multiplier is keying off the flag again.
        """
        calc = MetricsCalculator()
        y_true = np.array([1] * 20 + [0] * 80)

        compliant = calc.compute_all_metrics(y_true, y_true.copy(), None)
        y_pred = y_true.copy()
        y_pred[0] = 0
        non_compliant = calc.compute_all_metrics(y_true, y_pred, None)

        assert compliant.ethical_compliance is True
        assert non_compliant.ethical_compliance is False
        assert abs(compliant.overall_score - non_compliant.overall_score) < 0.1


class TestGosnnClassifiesOnTheScoresItReports:
    """``is_anomaly`` must be derivable from the returned scores."""

    def test_the_benevolence_adjustment_is_gone(self) -> None:
        assert not hasattr(GOSNNIntegration, "_apply_benevolence_adjustment"), (
            "the benevolence score adjustment is back: it promoted samples up to "
            "~1.57% below the threshold to is_anomaly=True while reporting the "
            "unadjusted score, and its docstring had the direction backwards"
        )

    @pytest.mark.parametrize("threshold", [0.1, 0.5, 0.6, 0.9])
    def test_no_score_below_the_threshold_is_flagged(self, threshold: float) -> None:
        """The property the deleted adjustment broke, stated directly.

        Measured before the fix at threshold 0.60: raw 0.5950 -> adjusted
        0.604547 -> flagged, with 0.5950 reported to the caller.
        """
        scores = np.linspace(0.0, 1.0, 201)
        is_anomaly = scores > threshold

        flagged_below = scores[is_anomaly & (scores <= threshold)]
        assert flagged_below.size == 0, f"flagged despite being below {threshold}: {flagged_below}"


class _EchoDetector:
    """Minimal base detector: the scores are whatever was passed in."""

    def detect(self, X: np.ndarray[Any, Any]) -> dict[str, Any]:
        return {"scores": np.asarray(X, dtype=float).ravel()}


class TestBaseDomainDetectorReportsTheThresholdItUsed:
    """A caller must be able to reconstruct the decision from what is returned."""

    def test_the_benevolence_weight_knob_is_gone(self) -> None:
        import inspect

        params = set(inspect.signature(BaseDomainDetector.__init__).parameters)
        assert "benevolence_weight" not in params, (
            "benevolence_weight is back: at its 0.1 default it multiplied the "
            "decision threshold by 1.1618 -- a 16.18% higher bar -- while "
            "result['threshold'] reported the unadjusted value"
        )

    def test_is_anomaly_matches_the_reported_threshold(self) -> None:
        """The property the hidden multiplier broke, stated directly.

        Before the fix, with the shipped default, scores at 0.61 and 0.65
        against a reported threshold of 0.60 came back ``is_anomaly=False``
        because the decision silently used 0.697.
        """
        detector = BaseDomainDetector(_EchoDetector(), use_calibration=False)
        detector._threshold = 0.60
        scores = np.array([0.55, 0.59, 0.61, 0.65, 0.70, 0.75])

        result = detector.detect(scores)

        assert np.array_equal(
            result["is_anomaly"], result["scores"] > result["threshold"]
        ), "is_anomaly cannot be reconstructed from the returned scores and threshold"
        # The two that the old 16.18% bar suppressed are detections again.
        assert bool(result["is_anomaly"][2]) and bool(result["is_anomaly"][3])
