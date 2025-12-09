"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

Standard Evaluation Metrics for Anomaly Detection

Implements industry-standard metrics used in academic research:
- AUC-ROC: Area Under the Receiver Operating Characteristic Curve
- AUC-PR: Area Under the Precision-Recall Curve
- F1-Score: Harmonic mean of precision and recall
- Precision@K: Precision at top-K anomaly predictions
- Best-F1: F1-score at optimal threshold
- Point-Adjusted F1: F1 with anomaly segment adjustment (for time-series)

These metrics are used in papers like:
- OmniAnomaly (KDD 2019)
- MSCRED (AAAI 2019)
- DAGMM (ICLR 2018)
- TranAD (VLDB 2022)
"""
from __future__ import annotations
from typing import Any

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AnomalyMetrics:
    """Container for anomaly detection evaluation metrics."""

    # Core metrics
    auc_roc: float
    auc_pr: float
    best_f1: float
    best_threshold: float

    # At optimal threshold
    precision: float
    recall: float
    f1: float

    # Additional metrics
    accuracy: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    # Optional time-series adjusted metrics
    point_adjusted_f1: float | None = None
    range_based_f1: float | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "auc_roc": self.auc_roc,
            "auc_pr": self.auc_pr,
            "best_f1": self.best_f1,
            "best_threshold": self.best_threshold,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "point_adjusted_f1": self.point_adjusted_f1,
            "range_based_f1": self.range_based_f1,
        }

    def __str__(self) -> str:
        return (
            f"AnomalyMetrics(\n"
            f"  AUC-ROC: {self.auc_roc:.4f}\n"
            f"  AUC-PR: {self.auc_pr:.4f}\n"
            f"  Best F1: {self.best_f1:.4f} @ threshold={self.best_threshold:.4f}\n"
            f"  Precision: {self.precision:.4f}\n"
            f"  Recall: {self.recall:.4f}\n"
            f"  Accuracy: {self.accuracy:.4f}\n"
            f")"
        )


def compute_auc_roc(y_true: np.ndarray[Any, Any], y_score: np.ndarray[Any, Any]) -> float:
    """
    Compute Area Under the ROC Curve.

    Args:
        y_true: Binary ground truth labels (0 = normal, 1 = anomaly)
        y_score: Anomaly scores (higher = more anomalous)

    Returns:
        AUC-ROC score in [0, 1]. Returns 0.5 if all labels are the same class
        (undefined case where ROC curve cannot be computed).
    """
    # Handle edge case: all labels are the same class
    # In this case, AUC-ROC is undefined; we return 0.5 (random classifier baseline)
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, y_score))
    except ImportError:
        # Fallback implementation without sklearn
        return _auc_roc_numpy(y_true, y_score)


def _auc_roc_numpy(y_true: np.ndarray[Any, Any], y_score: np.ndarray[Any, Any]) -> float:
    """Pure numpy AUC-ROC implementation."""
    # Sort by score descending
    desc_score_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_score_indices]

    # Compute TPR and FPR at each threshold
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)

    tpr = tps / n_pos
    fpr = fps / n_neg

    # Compute AUC using trapezoidal rule
    auc = np.trapz(tpr, fpr)
    return float(auc)


def compute_auc_pr(y_true: np.ndarray[Any, Any], y_score: np.ndarray[Any, Any]) -> float:
    """
    Compute Area Under the Precision-Recall Curve.

    More informative than AUC-ROC for imbalanced datasets (common in anomaly detection).

    Args:
        y_true: Binary ground truth labels
        y_score: Anomaly scores

    Returns:
        AUC-PR score in [0, 1]
    """
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(y_true, y_score))
    except ImportError:
        return _auc_pr_numpy(y_true, y_score)


def _auc_pr_numpy(y_true: np.ndarray[Any, Any], y_score: np.ndarray[Any, Any]) -> float:
    """Pure numpy AUC-PR implementation."""
    desc_score_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_score_indices]

    n_pos = np.sum(y_true)
    if n_pos == 0:
        return 0.0

    tps = np.cumsum(y_true_sorted)
    precision = tps / np.arange(1, len(y_true) + 1)
    recall = tps / n_pos

    # Compute AUC using step function
    recall_diff = np.diff(recall, prepend=0)
    auc = np.sum(precision * recall_diff)
    return float(auc)


def compute_best_f1(
    y_true: np.ndarray[Any, Any], y_score: np.ndarray[Any, Any], n_thresholds: int = 100
) -> tuple[float, float]:
    """
    Find the threshold that maximizes F1-score.

    Args:
        y_true: Binary ground truth labels
        y_score: Anomaly scores
        n_thresholds: Number of thresholds to try

    Returns:
        Tuple of (best_f1, best_threshold)
    """
    thresholds = np.percentile(y_score, np.linspace(0, 100, n_thresholds))

    best_f1 = 0.0
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        f1 = compute_f1(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return float(best_f1), float(best_threshold)


def compute_f1(y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any]) -> float:
    """Compute F1-score."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def compute_precision_at_k(y_true: np.ndarray[Any, Any], y_score: np.ndarray[Any, Any], k: int) -> float:
    """
    Compute Precision@K.

    Measures precision among the top-K predicted anomalies.
    Useful when you can only investigate a limited number of alerts.

    Args:
        y_true: Binary ground truth labels
        y_score: Anomaly scores
        k: Number of top predictions to consider

    Returns:
        Precision@K in [0, 1]
    """
    if k <= 0 or k > len(y_true):
        k = len(y_true)

    # Get indices of top-k scores
    top_k_indices = np.argsort(y_score)[-k:]

    # Count true positives in top-k
    tp_at_k = np.sum(y_true[top_k_indices])

    return float(tp_at_k / k)


def compute_point_adjusted_f1(
    y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any], adjust_predicts: bool = True
) -> float:
    """
    Compute Point-Adjusted F1 for time-series anomaly detection.

    In time-series, if any point in an anomaly segment is detected,
    the entire segment is considered detected.

    This is the standard evaluation metric used in:
    - OmniAnomaly (KDD 2019)
    - MSCRED (AAAI 2019)
    - TranAD (VLDB 2022)

    Args:
        y_true: Binary ground truth labels
        y_pred: Binary predictions
        adjust_predicts: Whether to adjust predictions based on ground truth segments

    Returns:
        Point-adjusted F1-score
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    if not adjust_predicts:
        return compute_f1(y_true, y_pred)

    # Find anomaly segments in ground truth
    segments = _find_segments(y_true)

    # Adjust predictions: if any point in a GT segment is predicted,
    # mark all points in that segment as predicted
    adjusted_pred = y_pred.copy()
    for start, end in segments:
        if np.any(y_pred[start:end] == 1):
            adjusted_pred[start:end] = 1

    return compute_f1(y_true, adjusted_pred)


def _find_segments(labels: np.ndarray[Any, Any]) -> list[tuple[int, int]]:
    """Find contiguous segments of 1s in labels."""
    segments = []
    in_segment = False
    start = 0

    for i, val in enumerate(labels):
        if val == 1 and not in_segment:
            start = i
            in_segment = True
        elif val == 0 and in_segment:
            segments.append((start, i))
            in_segment = False

    if in_segment:
        segments.append((start, len(labels)))

    return segments


def compute_range_based_f1(
    y_true: np.ndarray[Any, Any],
    y_pred: np.ndarray[Any, Any],
    alpha: float = 0.0,
    cardinality: str = "reciprocal",
    bias: str = "flat",
) -> float:
    """
    Compute Range-Based F1 Score.

    More sophisticated time-series evaluation that considers:
    - Overlap between predicted and true anomaly ranges
    - Cardinality (how many predicted segments overlap one true segment)
    - Position bias (early/middle/late detection)

    Reference: Tatbul et al., "Precision and Recall for Time Series", NeurIPS 2018

    Args:
        y_true: Binary ground truth labels
        y_pred: Binary predictions
        alpha: Weight for existence reward (0 = strict, 1 = lenient)
        cardinality: How to handle multiple predictions for one GT segment
        bias: Position bias for partial overlap scoring

    Returns:
        Range-based F1-score
    """
    gt_segments = _find_segments(y_true)
    pred_segments = _find_segments(y_pred)

    if len(gt_segments) == 0 and len(pred_segments) == 0:
        return 1.0
    if len(gt_segments) == 0 or len(pred_segments) == 0:
        return 0.0

    # Compute range-based precision and recall
    precision = _range_precision(gt_segments, pred_segments, alpha, cardinality, bias)
    recall = _range_recall(gt_segments, pred_segments, alpha, cardinality, bias)

    if precision + recall == 0:
        return 0.0

    return float(2 * precision * recall / (precision + recall))


def _range_precision(gt_segs, pred_segs, alpha, cardinality, bias) -> float:
    """Compute range-based precision."""
    scores = []
    for pred_start, pred_end in pred_segs:
        overlapping = [(s, e) for s, e in gt_segs if max(s, pred_start) < min(e, pred_end)]
        if not overlapping:
            scores.append(0.0)
        else:
            overlap_score = sum(min(e, pred_end) - max(s, pred_start) for s, e in overlapping) / (
                pred_end - pred_start
            )
            scores.append(min(1.0, overlap_score))

    return sum(scores) / len(scores) if scores else 0.0


def _range_recall(gt_segs, pred_segs, alpha, cardinality, bias) -> float:
    """Compute range-based recall."""
    scores = []
    for gt_start, gt_end in gt_segs:
        overlapping = [(s, e) for s, e in pred_segs if max(s, gt_start) < min(e, gt_end)]
        if not overlapping:
            scores.append(alpha)  # Existence reward only
        else:
            overlap_score = sum(min(e, gt_end) - max(s, gt_start) for s, e in overlapping) / (
                gt_end - gt_start
            )
            scores.append(alpha + (1 - alpha) * min(1.0, overlap_score))

    return sum(scores) / len(scores) if scores else 0.0


def evaluate_anomaly_detection(
    y_true: np.ndarray[Any, Any],
    y_score: np.ndarray[Any, Any],
    threshold: float | None = None,
    is_timeseries: bool = False,
) -> AnomalyMetrics:
    """
    Comprehensive evaluation of anomaly detection results.

    Args:
        y_true: Binary ground truth labels (0 = normal, 1 = anomaly)
        y_score: Anomaly scores (higher = more anomalous)
        threshold: Fixed threshold for binary predictions (if None, finds optimal)
        is_timeseries: Whether to compute time-series adjusted metrics

    Returns:
        AnomalyMetrics object with all evaluation metrics
    """
    y_true = np.array(y_true).flatten().astype(int)
    y_score = np.array(y_score).flatten()

    # Compute AUC metrics
    auc_roc = compute_auc_roc(y_true, y_score)
    auc_pr = compute_auc_pr(y_true, y_score)

    # Find best threshold if not provided
    best_f1, best_threshold = compute_best_f1(y_true, y_score)

    if threshold is None:
        threshold = best_threshold

    # Compute binary predictions
    y_pred = (y_score >= threshold).astype(int)

    # Compute confusion matrix
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

    # Time-series adjusted metrics
    point_adjusted_f1 = None
    range_based_f1 = None
    if is_timeseries:
        point_adjusted_f1 = compute_point_adjusted_f1(y_true, y_pred)
        range_based_f1 = compute_range_based_f1(y_true, y_pred)

    return AnomalyMetrics(
        auc_roc=auc_roc,
        auc_pr=auc_pr,
        best_f1=best_f1,
        best_threshold=best_threshold,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        point_adjusted_f1=point_adjusted_f1,
        range_based_f1=range_based_f1,
    )


def print_metrics_report(metrics: AnomalyMetrics, dataset_name: str = "Unknown") -> str:
    """
    Generate a formatted metrics report.

    Args:
        metrics: Evaluation metrics
        dataset_name: Name of the dataset

    Returns:
        Formatted report string
    """
    lines = [
        f"\n{'='*60}",
        "ANOMALY DETECTION EVALUATION REPORT",
        f"Dataset: {dataset_name}",
        f"{'='*60}",
        "",
        "ROC Analysis:",
        f"  AUC-ROC:        {metrics.auc_roc:.4f}",
        f"  AUC-PR:         {metrics.auc_pr:.4f}",
        "",
        "Optimal Threshold Performance:",
        f"  Best F1:        {metrics.best_f1:.4f}",
        f"  Threshold:      {metrics.best_threshold:.4f}",
        "",
        "Classification Metrics (at optimal threshold):",
        f"  Precision:      {metrics.precision:.4f}",
        f"  Recall:         {metrics.recall:.4f}",
        f"  F1-Score:       {metrics.f1:.4f}",
        f"  Accuracy:       {metrics.accuracy:.4f}",
        "",
        "Confusion Matrix:",
        f"  True Positives:  {metrics.true_positives}",
        f"  False Positives: {metrics.false_positives}",
        f"  True Negatives:  {metrics.true_negatives}",
        f"  False Negatives: {metrics.false_negatives}",
    ]

    if metrics.point_adjusted_f1 is not None:
        lines.extend(
            [
                "",
                "Time-Series Adjusted Metrics:",
                f"  Point-Adjusted F1: {metrics.point_adjusted_f1:.4f}",
            ]
        )
        if metrics.range_based_f1 is not None:
            lines.append(f"  Range-Based F1:    {metrics.range_based_f1:.4f}")

    lines.append(f"{'='*60}\n")

    return "\n".join(lines)
