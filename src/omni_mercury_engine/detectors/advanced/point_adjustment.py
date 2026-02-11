"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisors LLC

Point-Adjustment Evaluation Protocol for Time-Series Anomaly Detection

Implements the standard point-adjustment protocol used in time-series
anomaly detection benchmarks (SMD, SMAP, MSL, SWaT, WADI).

Key Concept: If any point within an anomaly segment is detected,
the entire segment is considered correctly detected. This reflects
the practical reality that detecting an anomaly anywhere in its
duration is sufficient for alerting.

This protocol is critical for fair comparison with SOTA methods
that report F1 scores of 0.85+ on these datasets.

Reference:
- Xu et al. (2018) - Unsupervised Anomaly Detection via VAE
- Su et al. (2019) - OmniAnomaly
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


__all__ = [
    "PointAdjustmentEvaluator",
    "adjust_predictions",
    "compute_adjusted_metrics",
    "find_anomaly_segments",
]


@dataclass
class SegmentInfo:
    """Information about an anomaly segment."""

    start: int
    end: int
    length: int
    detected: bool = False
    detection_delay: int = -1  # Time steps until first detection


def find_anomaly_segments(
    labels: NDArray[np.int64],
) -> list[SegmentInfo]:
    """
    Find contiguous anomaly segments in labels.

    Args:
        labels: Binary labels [n_samples]

    Returns:
        List of SegmentInfo objects
    """
    segments = []
    in_segment = False
    start = 0

    for i in range(len(labels)):
        if labels[i] == 1 and not in_segment:
            # Start of segment
            in_segment = True
            start = i
        elif labels[i] == 0 and in_segment:
            # End of segment
            segments.append(
                SegmentInfo(
                    start=start,
                    end=i,
                    length=i - start,
                )
            )
            in_segment = False

    # Handle last segment
    if in_segment:
        segments.append(
            SegmentInfo(
                start=start,
                end=len(labels),
                length=len(labels) - start,
            )
        )

    return segments


def adjust_predictions(
    predictions: NDArray[np.int64],
    labels: NDArray[np.int64],
) -> NDArray[np.int64]:
    """
    Apply point-adjustment to predictions.

    If any point in a ground-truth anomaly segment is predicted as anomaly,
    mark all points in that segment as correctly detected.

    Args:
        predictions: Binary predictions [n_samples]
        labels: Ground truth binary labels [n_samples]

    Returns:
        Adjusted predictions
    """
    adjusted = predictions.copy()
    segments = find_anomaly_segments(labels)

    for segment in segments:
        # Check if any point in segment was predicted as anomaly
        segment_preds = predictions[segment.start : segment.end]
        if segment_preds.sum() > 0:
            # Mark entire segment as detected
            adjusted[segment.start : segment.end] = 1

    return adjusted


def compute_adjusted_metrics(
    predictions: NDArray[np.int64],
    labels: NDArray[np.int64],
    scores: NDArray[np.float64] | None = None,
) -> dict[str, float]:
    """
    Compute metrics with point-adjustment.

    Args:
        predictions: Binary predictions
        labels: Ground truth labels
        scores: Optional continuous anomaly scores

    Returns:
        Dictionary with adjusted metrics
    """
    # Apply point-adjustment
    adjusted_preds = adjust_predictions(predictions, labels)

    # Compute standard metrics on adjusted predictions
    tp = np.sum((adjusted_preds == 1) & (labels == 1))
    fp = np.sum((adjusted_preds == 1) & (labels == 0))
    fn = np.sum((adjusted_preds == 0) & (labels == 1))
    tn = np.sum((adjusted_preds == 0) & (labels == 0))

    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Accuracy
    accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0.0

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }

    # Segment-level metrics
    segments = find_anomaly_segments(labels)
    if segments:
        detected_segments = 0
        total_delay = 0

        for segment in segments:
            segment_preds = predictions[segment.start : segment.end]
            if segment_preds.sum() > 0:
                detected_segments += 1
                # Find first detection
                first_detection = np.argmax(segment_preds)
                total_delay += int(first_detection)

        segment_recall = detected_segments / len(segments)
        avg_delay = total_delay / max(detected_segments, 1)

        metrics["segment_recall"] = segment_recall
        metrics["n_segments"] = len(segments)
        metrics["detected_segments"] = detected_segments
        metrics["avg_detection_delay"] = avg_delay

    # ROC-AUC if scores provided
    if scores is not None:
        try:
            from sklearn.metrics import roc_auc_score

            # Use original labels for AUC (not affected by point-adjustment)
            metrics["roc_auc"] = float(roc_auc_score(labels, scores))
        except (ValueError, ImportError):
            pass

    return metrics


class PointAdjustmentEvaluator:
    """
    Evaluator with point-adjustment for time-series anomaly detection.

    Provides comprehensive evaluation including:
    - Standard metrics (precision, recall, F1)
    - Segment-level metrics (segment recall, detection delay)
    - Multiple threshold evaluation
    - Best F1 threshold search

    Example:
        >>> evaluator = PointAdjustmentEvaluator()
        >>> metrics = evaluator.evaluate(predictions, labels, scores)
        >>> print(f"Adjusted F1: {metrics['f1']:.4f}")
    """

    def __init__(
        self,
        search_best_threshold: bool = True,
        n_thresholds: int = 100,
    ) -> None:
        self.search_best_threshold = search_best_threshold
        self.n_thresholds = n_thresholds

    def evaluate(
        self,
        predictions: NDArray[np.int64] | None = None,
        labels: NDArray[np.int64] | None = None,
        scores: NDArray[np.float64] | None = None,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate with point-adjustment.

        Args:
            predictions: Binary predictions (or None if scores + threshold provided)
            labels: Ground truth labels
            scores: Continuous anomaly scores (optional)
            threshold: Threshold for converting scores to predictions

        Returns:
            Dictionary with evaluation metrics
        """
        if labels is None:
            raise ValueError("labels must be provided")

        # Convert scores to predictions if needed
        if predictions is None:
            if scores is None:
                raise ValueError("Either predictions or scores must be provided")
            if threshold is None:
                # Search for best threshold
                threshold = self._find_best_threshold(scores, labels)
            predictions = (scores > threshold).astype(int)

        # Compute adjusted metrics
        metrics = compute_adjusted_metrics(predictions, labels, scores)

        # Add threshold info
        if threshold is not None:
            metrics["threshold"] = threshold

        # Unadjusted metrics for comparison
        unadjusted = self._compute_unadjusted_metrics(predictions, labels)
        metrics["unadjusted_precision"] = unadjusted["precision"]
        metrics["unadjusted_recall"] = unadjusted["recall"]
        metrics["unadjusted_f1"] = unadjusted["f1"]

        # F1 improvement from adjustment
        metrics["f1_improvement"] = metrics["f1"] - unadjusted["f1"]

        return metrics

    def _find_best_threshold(
        self,
        scores: NDArray[np.float64],
        labels: NDArray[np.int64],
    ) -> float:
        """Find threshold that maximizes adjusted F1."""
        # Generate candidate thresholds
        thresholds = np.percentile(scores, np.linspace(80, 99.9, self.n_thresholds))

        best_f1 = 0.0
        best_threshold = thresholds[0]

        for thresh in thresholds:
            preds = (scores > thresh).astype(int)
            metrics = compute_adjusted_metrics(preds, labels)
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_threshold = thresh

        return float(best_threshold)

    def _compute_unadjusted_metrics(
        self,
        predictions: NDArray[np.int64],
        labels: NDArray[np.int64],
    ) -> dict[str, float]:
        """Compute metrics without point-adjustment."""
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"precision": precision, "recall": recall, "f1": f1}

    def evaluate_multiple_thresholds(
        self,
        scores: NDArray[np.float64],
        labels: NDArray[np.int64],
    ) -> list[dict[str, Any]]:
        """
        Evaluate at multiple thresholds for threshold sensitivity analysis.

        Returns:
            List of metric dictionaries at different thresholds
        """
        thresholds = np.percentile(scores, np.linspace(80, 99.9, self.n_thresholds))
        results = []

        for thresh in thresholds:
            preds = (scores > thresh).astype(int)
            metrics = compute_adjusted_metrics(preds, labels, scores)
            metrics["threshold"] = float(thresh)
            results.append(metrics)

        return results

    def report(
        self,
        predictions: NDArray[np.int64] | None = None,
        labels: NDArray[np.int64] | None = None,
        scores: NDArray[np.float64] | None = None,
        threshold: float | None = None,
    ) -> str:
        """
        Generate a formatted evaluation report.

        Returns:
            Formatted string report
        """
        metrics = self.evaluate(predictions, labels, scores, threshold)

        report_lines = [
            "=" * 50,
            "POINT-ADJUSTED EVALUATION REPORT",
            "=" * 50,
            "",
            "Adjusted Metrics:",
            f"  Precision: {metrics['precision']:.4f}",
            f"  Recall:    {metrics['recall']:.4f}",
            f"  F1 Score:  {metrics['f1']:.4f}",
            f"  Accuracy:  {metrics['accuracy']:.4f}",
            "",
            "Confusion Matrix:",
            f"  TP: {metrics['tp']:,}  FP: {metrics['fp']:,}",
            f"  FN: {metrics['fn']:,}  TN: {metrics['tn']:,}",
            "",
        ]

        if "segment_recall" in metrics:
            report_lines.extend(
                [
                    "Segment-Level Metrics:",
                    f"  Segments:  {metrics['n_segments']}",
                    f"  Detected:  {metrics['detected_segments']}",
                    f"  Seg Recall: {metrics['segment_recall']:.4f}",
                    f"  Avg Delay: {metrics['avg_detection_delay']:.1f} steps",
                    "",
                ]
            )

        if "roc_auc" in metrics:
            report_lines.append(f"ROC-AUC: {metrics['roc_auc']:.4f}")

        report_lines.extend(
            [
                "",
                "Comparison (Unadjusted):",
                f"  Precision: {metrics['unadjusted_precision']:.4f}",
                f"  Recall:    {metrics['unadjusted_recall']:.4f}",
                f"  F1 Score:  {metrics['unadjusted_f1']:.4f}",
                f"  F1 Improvement: +{metrics['f1_improvement']:.4f}",
                "",
                "=" * 50,
            ]
        )

        return "\n".join(report_lines)
