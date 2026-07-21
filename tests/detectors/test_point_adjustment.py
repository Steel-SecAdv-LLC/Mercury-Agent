# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioral tests for the point-adjustment evaluation protocol.

Covers:
- find_anomaly_segments: contiguous segment extraction with exclusive end
  indices, trailing segments, empty/all-normal/all-anomalous labels, and
  SegmentInfo defaults.
- adjust_predictions: single-hit segment expansion, undetected segments
  left untouched, false positives preserved, input immutability.
- compute_adjusted_metrics: hand-computed confusion matrix, precision /
  recall / F1 / accuracy, segment recall and detection delay, empty
  inputs, single-class labels, ROC-AUC inclusion and omission.
- PointAdjustmentEvaluator: adjusted vs unadjusted metrics with a
  hand-computed F1 improvement, score thresholding, best-threshold
  search, multi-threshold sensitivity sweep, report formatting, and
  error paths for missing labels/predictions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from omni_mercury_engine.detectors.advanced.point_adjustment import (
    PointAdjustmentEvaluator,
    SegmentInfo,
    adjust_predictions,
    compute_adjusted_metrics,
    find_anomaly_segments,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _two_segment_case() -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Hand-analyzed scenario with one detected and one missed segment.

    labels: [0,0,1,1,1,0,1,1,0,0] -> segments (2,5) and (6,8).
    preds:  [1,0,0,1,0,0,0,0,0,0] -> FP at 0; hit inside segment 1 at
    index 3 (delay 1); segment 2 undetected.

    Adjusted predictions: [1,0,1,1,1,0,0,0,0,0]
    -> TP=3, FP=1, FN=2, TN=4
    -> precision=3/4, recall=3/5, F1=2/3, accuracy=7/10.
    """
    labels = np.array([0, 0, 1, 1, 1, 0, 1, 1, 0, 0], dtype=np.int64)
    preds = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    return labels, preds


def _separable_scores() -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """100 points: 90 normals in [0, 0.5), one anomaly segment in [0.8, 1]."""
    rng = np.random.default_rng(42)
    normal = rng.uniform(0.0, 0.5, 90)
    anomalous = rng.uniform(0.8, 1.0, 10)
    scores = np.concatenate([normal, anomalous])
    labels = np.concatenate([np.zeros(90, dtype=np.int64), np.ones(10, dtype=np.int64)])
    return scores, labels


# ---------------------------------------------------------------------------
# find_anomaly_segments
# ---------------------------------------------------------------------------


class TestFindAnomalySegments:
    """Contiguous anomaly segment extraction."""

    def test_empty_labels_yield_no_segments(self) -> None:
        """An empty label array has no segments."""
        assert find_anomaly_segments(np.array([], dtype=np.int64)) == []

    def test_all_normal_labels_yield_no_segments(self) -> None:
        """All-zero labels have no segments."""
        assert find_anomaly_segments(np.zeros(10, dtype=np.int64)) == []

    def test_all_anomalous_labels_yield_single_full_segment(self) -> None:
        """All-one labels form one segment covering the whole series."""
        segments = find_anomaly_segments(np.ones(5, dtype=np.int64))
        assert len(segments) == 1
        assert (segments[0].start, segments[0].end, segments[0].length) == (0, 5, 5)

    def test_multiple_segments_closed_form(self) -> None:
        """[0,1,1,0,0,1,0,1] -> (1,3), (5,6), and trailing (7,8)."""
        labels = np.array([0, 1, 1, 0, 0, 1, 0, 1], dtype=np.int64)
        segments = find_anomaly_segments(labels)
        spans = [(s.start, s.end, s.length) for s in segments]
        assert spans == [(1, 3, 2), (5, 6, 1), (7, 8, 1)]

    def test_end_index_is_exclusive(self) -> None:
        """A segment's end index points one past its last anomalous point."""
        labels = np.array([0, 1, 1, 0], dtype=np.int64)
        segment = find_anomaly_segments(labels)[0]
        assert labels[segment.start] == 1
        assert labels[segment.end] == 0
        assert segment.length == segment.end - segment.start

    def test_segment_at_series_start(self) -> None:
        """A segment beginning at index 0 is captured."""
        segments = find_anomaly_segments(np.array([1, 1, 0], dtype=np.int64))
        assert [(s.start, s.end) for s in segments] == [(0, 2)]

    def test_segment_info_defaults(self) -> None:
        """Fresh segments carry detected=False and detection_delay=-1."""
        segment = find_anomaly_segments(np.array([0, 1, 0], dtype=np.int64))[0]
        assert isinstance(segment, SegmentInfo)
        assert segment.detected is False
        assert segment.detection_delay == -1


# ---------------------------------------------------------------------------
# adjust_predictions
# ---------------------------------------------------------------------------


class TestAdjustPredictions:
    """The point-adjustment expansion rule."""

    def test_single_hit_expands_to_full_segment(self) -> None:
        """One detected point marks the entire ground-truth segment."""
        labels = np.array([0, 1, 1, 1, 0], dtype=np.int64)
        preds = np.array([0, 0, 1, 0, 0], dtype=np.int64)
        np.testing.assert_array_equal(adjust_predictions(preds, labels), np.array([0, 1, 1, 1, 0]))

    def test_undetected_segment_left_untouched(self) -> None:
        """Segments with no predicted point stay at zero."""
        labels = np.array([0, 1, 1, 0, 1, 1], dtype=np.int64)
        preds = np.array([0, 0, 0, 0, 1, 0], dtype=np.int64)
        np.testing.assert_array_equal(
            adjust_predictions(preds, labels), np.array([0, 0, 0, 0, 1, 1])
        )

    def test_false_positives_outside_segments_preserved(self) -> None:
        """Adjustment never clears false positives outside segments."""
        labels = np.array([0, 1, 1, 0], dtype=np.int64)
        preds = np.array([1, 0, 0, 0], dtype=np.int64)
        np.testing.assert_array_equal(adjust_predictions(preds, labels), preds)

    def test_input_predictions_not_mutated(self) -> None:
        """The input array is copied, not modified in place."""
        labels = np.array([0, 1, 1, 1, 0], dtype=np.int64)
        preds = np.array([0, 0, 1, 0, 0], dtype=np.int64)
        original = preds.copy()
        adjust_predictions(preds, labels)
        np.testing.assert_array_equal(preds, original)

    def test_no_anomaly_labels_is_identity(self) -> None:
        """With no ground-truth segments, predictions pass through as-is."""
        labels = np.zeros(6, dtype=np.int64)
        preds = np.array([1, 0, 1, 0, 0, 1], dtype=np.int64)
        np.testing.assert_array_equal(adjust_predictions(preds, labels), preds)

    def test_perfect_predictions_unchanged(self) -> None:
        """Exact predictions are a fixed point of the adjustment."""
        labels = np.array([0, 1, 1, 0, 1], dtype=np.int64)
        np.testing.assert_array_equal(adjust_predictions(labels.copy(), labels), labels)


# ---------------------------------------------------------------------------
# compute_adjusted_metrics
# ---------------------------------------------------------------------------


class TestComputeAdjustedMetrics:
    """Hand-computed adjusted metrics."""

    def test_confusion_matrix_closed_form(self) -> None:
        """Adjusted TP=3, FP=1, FN=2, TN=4 for the two-segment scenario."""
        labels, preds = _two_segment_case()
        m = compute_adjusted_metrics(preds, labels)
        assert m["tp"] == 3
        assert m["fp"] == 1
        assert m["fn"] == 2
        assert m["tn"] == 4
        assert m["precision"] == pytest.approx(0.75)
        assert m["recall"] == pytest.approx(0.6)
        assert m["f1"] == pytest.approx(2.0 / 3.0)
        assert m["accuracy"] == pytest.approx(0.7)

    def test_segment_level_metrics_closed_form(self) -> None:
        """1 of 2 segments detected; first hit at offset 1 -> delay 1.0."""
        labels, preds = _two_segment_case()
        m = compute_adjusted_metrics(preds, labels)
        assert m["n_segments"] == 2
        assert m["detected_segments"] == 1
        assert m["segment_recall"] == pytest.approx(0.5)
        assert m["avg_detection_delay"] == pytest.approx(1.0)

    def test_zero_delay_when_first_segment_point_detected(self) -> None:
        """Hitting each segment's first point gives zero average delay."""
        labels = np.array([0, 1, 1, 0, 1, 1, 0], dtype=np.int64)
        preds = np.array([0, 1, 0, 0, 1, 0, 0], dtype=np.int64)
        m = compute_adjusted_metrics(preds, labels)
        assert m["segment_recall"] == pytest.approx(1.0)
        assert m["avg_detection_delay"] == pytest.approx(0.0)
        assert m["f1"] == pytest.approx(1.0)

    def test_no_detections_yield_zero_scores(self) -> None:
        """All-zero predictions: zero recall, zero delay denominator guard."""
        labels = np.array([0, 1, 1, 0], dtype=np.int64)
        preds = np.zeros(4, dtype=np.int64)
        m = compute_adjusted_metrics(preds, labels)
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0
        assert m["detected_segments"] == 0
        assert m["segment_recall"] == 0.0
        assert m["avg_detection_delay"] == 0.0

    def test_all_normal_labels_omit_segment_metrics(self) -> None:
        """Without ground-truth segments, no segment-level keys appear."""
        labels = np.zeros(8, dtype=np.int64)
        preds = np.array([0, 1, 0, 0, 0, 0, 1, 0], dtype=np.int64)
        m = compute_adjusted_metrics(preds, labels)
        assert "segment_recall" not in m
        assert "n_segments" not in m
        assert m["fp"] == 2
        assert m["tn"] == 6
        assert m["f1"] == 0.0

    def test_all_anomalous_labels_single_hit_gives_perfect_scores(self) -> None:
        """All-anomaly labels plus one hit adjust to a perfect score."""
        labels = np.ones(6, dtype=np.int64)
        preds = np.array([0, 0, 0, 1, 0, 0], dtype=np.int64)
        m = compute_adjusted_metrics(preds, labels)
        assert m["tp"] == 6
        assert m["fp"] == 0
        assert m["fn"] == 0
        assert m["tn"] == 0
        assert m["f1"] == pytest.approx(1.0)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["avg_detection_delay"] == pytest.approx(3.0)

    def test_empty_inputs_return_zeroed_metrics(self) -> None:
        """Zero-length series: all counts zero, accuracy guard hit."""
        empty = np.array([], dtype=np.int64)
        m = compute_adjusted_metrics(empty, empty)
        assert m["tp"] == 0
        assert m["fp"] == 0
        assert m["fn"] == 0
        assert m["tn"] == 0
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0
        assert m["accuracy"] == 0.0
        assert "segment_recall" not in m

    def test_roc_auc_included_for_two_class_labels(self) -> None:
        """Perfectly ranked scores give ROC-AUC of exactly 1.0."""
        labels = np.array([0, 0, 1, 1], dtype=np.int64)
        preds = np.array([0, 0, 1, 1], dtype=np.int64)
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        m = compute_adjusted_metrics(preds, labels, scores)
        assert m["roc_auc"] == pytest.approx(1.0)

    def test_roc_auc_uses_original_labels_not_adjusted(self) -> None:
        """Inversely ranked scores give ROC-AUC of exactly 0.0."""
        labels = np.array([0, 0, 1, 1], dtype=np.int64)
        preds = np.array([0, 0, 1, 0], dtype=np.int64)
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        m = compute_adjusted_metrics(preds, labels, scores)
        assert m["roc_auc"] == pytest.approx(0.0)

    def test_roc_auc_omitted_for_single_class_labels(self) -> None:
        """Single-class labels cannot support AUC; the key is absent."""
        labels = np.zeros(4, dtype=np.int64)
        preds = np.array([0, 1, 0, 0], dtype=np.int64)
        scores = np.array([0.1, 0.9, 0.2, 0.3])
        m = compute_adjusted_metrics(preds, labels, scores)
        assert "roc_auc" not in m


# ---------------------------------------------------------------------------
# PointAdjustmentEvaluator
# ---------------------------------------------------------------------------


class TestPointAdjustmentEvaluator:
    """End-to-end evaluation, threshold search, and reporting."""

    def test_constructor_defaults(self) -> None:
        """Default configuration matches the documented values."""
        evaluator = PointAdjustmentEvaluator()
        assert evaluator.search_best_threshold is True
        assert evaluator.n_thresholds == 100

    def test_search_disabled_without_threshold_fails_loud(self) -> None:
        """search_best_threshold=False + scores-only must raise, not search.

        Regression: the constructor flag was stored but never consulted, so
        evaluate() silently searched anyway.  Anomaly scores have arbitrary
        scale, so with the search opted out and no explicit threshold there
        is no principled operating point — the evaluator now fails loud.
        """
        labels = np.array([0, 0, 1, 1, 0, 0], dtype=np.int64)
        scores = np.array([0.1, 0.2, 0.9, 0.8, 0.1, 0.2])
        evaluator = PointAdjustmentEvaluator(search_best_threshold=False)
        with pytest.raises(ValueError, match="search_best_threshold=False"):
            evaluator.evaluate(labels=labels, scores=scores)

    def test_search_disabled_with_explicit_threshold_works(self) -> None:
        """search_best_threshold=False still evaluates a supplied threshold."""
        labels = np.array([0, 0, 1, 1, 0, 0], dtype=np.int64)
        scores = np.array([0.1, 0.2, 0.9, 0.8, 0.1, 0.2])
        evaluator = PointAdjustmentEvaluator(search_best_threshold=False)
        metrics = evaluator.evaluate(labels=labels, scores=scores, threshold=0.5)
        assert metrics["threshold"] == 0.5
        assert metrics["f1"] == pytest.approx(1.0)

    def test_evaluate_closed_form_with_predictions(self) -> None:
        """Adjusted F1=2/3 vs unadjusted F1=2/7; improvement is their gap.

        Unadjusted on the two-segment case: TP=1, FP=1, FN=4
        -> precision=1/2, recall=1/5, F1=2/7.
        """
        labels, preds = _two_segment_case()
        m = PointAdjustmentEvaluator().evaluate(predictions=preds, labels=labels)
        assert m["f1"] == pytest.approx(2.0 / 3.0)
        assert m["unadjusted_precision"] == pytest.approx(0.5)
        assert m["unadjusted_recall"] == pytest.approx(0.2)
        assert m["unadjusted_f1"] == pytest.approx(2.0 / 7.0)
        assert m["f1_improvement"] == pytest.approx(2.0 / 3.0 - 2.0 / 7.0)
        # No threshold supplied and none searched -> key absent.
        assert "threshold" not in m

    def test_evaluate_with_scores_and_explicit_threshold(self) -> None:
        """scores > threshold defines predictions; threshold is echoed."""
        labels = np.array([0, 0, 1, 1], dtype=np.int64)
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        m = PointAdjustmentEvaluator().evaluate(labels=labels, scores=scores, threshold=0.5)
        assert m["threshold"] == 0.5
        assert m["f1"] == pytest.approx(1.0)
        assert m["unadjusted_f1"] == pytest.approx(1.0)
        assert m["roc_auc"] == pytest.approx(1.0)

    def test_evaluate_searches_best_threshold_when_missing(self) -> None:
        """Without a threshold, the best adjusted-F1 threshold is found."""
        scores, labels = _separable_scores()
        m = PointAdjustmentEvaluator(n_thresholds=25).evaluate(labels=labels, scores=scores)
        assert m["f1"] == pytest.approx(1.0)
        assert m["fp"] == 0
        assert m["recall"] == pytest.approx(1.0)
        assert "threshold" in m

    def test_f1_improvement_zero_for_perfect_predictions(self) -> None:
        """Perfect predictions leave nothing for adjustment to improve."""
        labels = np.array([0, 1, 1, 0, 1], dtype=np.int64)
        m = PointAdjustmentEvaluator().evaluate(predictions=labels.copy(), labels=labels)
        assert m["f1"] == pytest.approx(1.0)
        assert m["f1_improvement"] == pytest.approx(0.0)

    def test_evaluate_requires_labels(self) -> None:
        """labels=None is rejected."""
        with pytest.raises(ValueError, match="labels must be provided"):
            PointAdjustmentEvaluator().evaluate(predictions=np.array([0, 1], dtype=np.int64))

    def test_evaluate_requires_predictions_or_scores(self) -> None:
        """Neither predictions nor scores is an error."""
        with pytest.raises(ValueError, match="predictions or scores"):
            PointAdjustmentEvaluator().evaluate(labels=np.array([0, 1], dtype=np.int64))

    def test_evaluate_multiple_thresholds_sweep(self) -> None:
        """The sweep returns n_thresholds entries over rising percentiles."""
        scores, labels = _separable_scores()
        evaluator = PointAdjustmentEvaluator(n_thresholds=15)
        results = evaluator.evaluate_multiple_thresholds(scores, labels)
        assert len(results) == 15
        thresholds = [r["threshold"] for r in results]
        assert thresholds == sorted(thresholds)
        assert all(isinstance(t, float) for t in thresholds)
        assert all(0.0 <= r["f1"] <= 1.0 for r in results)
        # The separable data admits a perfect adjusted F1 at some threshold.
        assert max(r["f1"] for r in results) == pytest.approx(1.0)

    def test_report_contains_all_sections(self) -> None:
        """Report includes adjusted, segment, ROC-AUC, and comparison blocks."""
        labels, preds = _two_segment_case()
        rng = np.random.default_rng(42)
        scores = labels * 0.8 + rng.uniform(0.0, 0.1, len(labels))
        report = PointAdjustmentEvaluator().report(predictions=preds, labels=labels, scores=scores)
        assert "POINT-ADJUSTED EVALUATION REPORT" in report
        assert "Adjusted Metrics:" in report
        assert "Precision: 0.7500" in report
        assert "Recall:    0.6000" in report
        assert "Segment-Level Metrics:" in report
        assert "Detected:  1" in report
        assert "ROC-AUC" in report
        assert "Comparison (Unadjusted):" in report
        assert "F1 Improvement: +0.3810" in report

    def test_report_without_segments_omits_segment_block(self) -> None:
        """No ground-truth segments -> no segment section, no ROC-AUC line."""
        labels = np.zeros(6, dtype=np.int64)
        preds = np.array([0, 1, 0, 0, 0, 0], dtype=np.int64)
        report = PointAdjustmentEvaluator().report(predictions=preds, labels=labels)
        assert "POINT-ADJUSTED EVALUATION REPORT" in report
        assert "Segment-Level Metrics:" not in report
        assert "ROC-AUC" not in report
