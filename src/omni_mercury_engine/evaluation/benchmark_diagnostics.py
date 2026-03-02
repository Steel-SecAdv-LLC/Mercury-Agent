"""
Mercury Agent - Benchmark Diagnostics Module
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Comprehensive diagnostic tools for benchmarking anomaly detection:
- Score distribution analysis
- Threshold calibration recommendations
- F1=0 problem diagnosis
- Metric discrepancy detection (ROC-AUC vs F1)

Usage:
    from omni_mercury_engine.evaluation.benchmark_diagnostics import (
        BenchmarkDiagnostics,
        run_diagnostic_benchmark,
    )

    # In your benchmark, add after detection:
    diagnostics = BenchmarkDiagnostics.diagnose(
        scores=result["scores"],
        labels=y_true,
        threshold=detector.threshold,
        detector_name="statistical",
    )
    print(diagnostics.report())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)


@dataclass
class MetricDiscrepancy:
    """Identifies discrepancy between ranking and binary metrics."""

    roc_auc: float
    f1: float
    precision: float
    recall: float

    # Discrepancy analysis
    has_discrepancy: bool
    discrepancy_type: str  # "f1_zero", "low_precision", "low_recall", "balanced"
    root_cause: str
    recommended_action: str

    @classmethod
    def analyze(
        cls,
        roc_auc: float,
        f1: float,
        precision: float,
        recall: float,
        threshold: float,
        score_max: float,
        score_min: float,
    ) -> MetricDiscrepancy:
        """
        Analyze metric discrepancy and identify root cause.

        The classic F1=0 problem occurs when:
        - ROC-AUC is good (0.8+) -> model has discrimination power
        - F1 is 0 -> binary predictions are all False
        - Root cause: threshold > max(scores)
        """
        # Check for classic F1=0 problem
        if roc_auc > 0.7 and f1 == 0:
            if threshold > score_max:
                return cls(
                    roc_auc=roc_auc,
                    f1=f1,
                    precision=precision,
                    recall=recall,
                    has_discrepancy=True,
                    discrepancy_type="f1_zero",
                    root_cause=f"Threshold ({threshold:.4f}) is higher than all scores "
                    f"(max={score_max:.4f}). All predictions are FALSE.",
                    recommended_action="Use auto-calibration: detector.enable_auto_calibration() "
                    "or use percentile-based threshold.",
                )
            else:
                return cls(
                    roc_auc=roc_auc,
                    f1=f1,
                    precision=precision,
                    recall=recall,
                    has_discrepancy=True,
                    discrepancy_type="f1_zero",
                    root_cause=f"Threshold ({threshold:.4f}) is too high for the score "
                    f"distribution (range: [{score_min:.4f}, {score_max:.4f}]).",
                    recommended_action="Lower threshold or use contamination-based calibration.",
                )

        # Check for precision/recall imbalance
        if precision == 0 and recall > 0:
            return cls(
                roc_auc=roc_auc,
                f1=f1,
                precision=precision,
                recall=recall,
                has_discrepancy=True,
                discrepancy_type="low_precision",
                root_cause="All positive predictions are false positives.",
                recommended_action="Increase threshold to reduce false positives.",
            )

        if recall == 0 and precision > 0:
            return cls(
                roc_auc=roc_auc,
                f1=f1,
                precision=precision,
                recall=recall,
                has_discrepancy=True,
                discrepancy_type="low_recall",
                root_cause="No true anomalies are being detected.",
                recommended_action="Lower threshold to catch more anomalies.",
            )

        # No significant discrepancy
        return cls(
            roc_auc=roc_auc,
            f1=f1,
            precision=precision,
            recall=recall,
            has_discrepancy=False,
            discrepancy_type="balanced",
            root_cause="Metrics are consistent.",
            recommended_action="No action needed.",
        )


@dataclass
class DiagnosticResult:
    """Complete diagnostic result for a benchmark run."""

    # Basic info
    detector_name: str
    dataset_name: str

    # Score statistics
    n_samples: int
    n_anomalies_true: int
    n_anomalies_predicted: int
    anomaly_ratio_true: float
    anomaly_ratio_predicted: float

    # Score distribution
    score_min: float
    score_max: float
    score_mean: float
    score_std: float
    score_median: float

    # Threshold info
    threshold_used: float
    threshold_recommended: float

    # Metrics
    roc_auc: float
    f1: float
    f1_at_best_threshold: float
    best_threshold: float
    precision: float
    recall: float

    # Discrepancy analysis
    discrepancy: MetricDiscrepancy

    # Percentiles
    percentiles: dict[int, float] = field(default_factory=dict)

    # Additional metadata
    is_bimodal: bool = False
    calibration_method_recommended: str = "auto"

    def report(self, verbose: bool = True) -> str:
        """Generate formatted diagnostic report."""
        lines = [
            "",
            "=" * 70,
            f"BENCHMARK DIAGNOSTIC REPORT: {self.detector_name}",
            f"Dataset: {self.dataset_name}",
            "=" * 70,
            "",
            "SAMPLE STATISTICS:",
            f"  Total samples:      {self.n_samples}",
            f"  True anomalies:     {self.n_anomalies_true} ({self.anomaly_ratio_true:.2%})",
            f"  Predicted anomalies: {self.n_anomalies_predicted} ({self.anomaly_ratio_predicted:.2%})",
            "",
            "SCORE DISTRIBUTION:",
            f"  Range:  [{self.score_min:.4f}, {self.score_max:.4f}]",
            f"  Mean:   {self.score_mean:.4f}",
            f"  Std:    {self.score_std:.4f}",
            f"  Median: {self.score_median:.4f}",
            f"  Bimodal: {self.is_bimodal}",
            "",
            "THRESHOLD ANALYSIS:",
            f"  Threshold used:        {self.threshold_used:.4f}",
            f"  Best threshold (F1):   {self.best_threshold:.4f}",
            f"  Recommended threshold: {self.threshold_recommended:.4f}",
            "",
            "METRICS:",
            f"  ROC-AUC:          {self.roc_auc:.4f}",
            f"  F1 (used thresh): {self.f1:.4f}",
            f"  F1 (best thresh): {self.f1_at_best_threshold:.4f}",
            f"  Precision:        {self.precision:.4f}",
            f"  Recall:           {self.recall:.4f}",
        ]

        if self.discrepancy.has_discrepancy:
            lines.extend(
                [
                    "",
                    "!" * 70,
                    "METRIC DISCREPANCY DETECTED",
                    "!" * 70,
                    f"Type: {self.discrepancy.discrepancy_type}",
                    "",
                    "ROOT CAUSE:",
                    f"  {self.discrepancy.root_cause}",
                    "",
                    "RECOMMENDED ACTION:",
                    f"  {self.discrepancy.recommended_action}",
                    "!" * 70,
                ]
            )

        if verbose:
            lines.extend(
                [
                    "",
                    "PERCENTILES:",
                ]
            )
            for p, v in sorted(self.percentiles.items()):
                marker = " <-- threshold" if abs(v - self.threshold_used) < 0.001 else ""
                lines.append(f"  P{p:3d}: {v:.4f}{marker}")

        lines.extend(
            [
                "",
                "CALIBRATION RECOMMENDATION:",
                f"  Method: {self.calibration_method_recommended}",
                f"  Code:   detector.enable_auto_calibration(method='{self.calibration_method_recommended}')",
                "",
                "=" * 70,
            ]
        )

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "detector_name": self.detector_name,
            "dataset_name": self.dataset_name,
            "n_samples": self.n_samples,
            "n_anomalies_true": self.n_anomalies_true,
            "n_anomalies_predicted": self.n_anomalies_predicted,
            "anomaly_ratio_true": self.anomaly_ratio_true,
            "anomaly_ratio_predicted": self.anomaly_ratio_predicted,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "score_mean": self.score_mean,
            "score_std": self.score_std,
            "score_median": self.score_median,
            "threshold_used": self.threshold_used,
            "threshold_recommended": self.threshold_recommended,
            "roc_auc": self.roc_auc,
            "f1": self.f1,
            "f1_at_best_threshold": self.f1_at_best_threshold,
            "best_threshold": self.best_threshold,
            "precision": self.precision,
            "recall": self.recall,
            "discrepancy": {
                "has_discrepancy": self.discrepancy.has_discrepancy,
                "type": self.discrepancy.discrepancy_type,
                "root_cause": self.discrepancy.root_cause,
                "recommended_action": self.discrepancy.recommended_action,
            },
            "percentiles": self.percentiles,
            "is_bimodal": self.is_bimodal,
            "calibration_method_recommended": self.calibration_method_recommended,
        }


class BenchmarkDiagnostics:
    """
    Main diagnostic tool for benchmarking.

    Usage:
        from omni_mercury_engine.evaluation.benchmark_diagnostics import BenchmarkDiagnostics

        # After detection
        result = detector.detect(data)
        scores = result["scores"]

        # Run diagnostics
        diagnostics = BenchmarkDiagnostics.diagnose(
            scores=scores,
            labels=y_true,
            threshold=detector.threshold,
            detector_name="MercuryAnomalyDetector",
            dataset_name="covtype",
        )

        # Print report
        print(diagnostics.report())

        # Or quick diagnostic
        BenchmarkDiagnostics.quick_diagnose(scores, y_true, threshold=0.5)
    """

    @staticmethod
    def diagnose(
        scores: NDArray[np.float64],
        labels: NDArray[np.int32],
        threshold: float = 0.5,
        detector_name: str = "Unknown",
        dataset_name: str = "Unknown",
    ) -> DiagnosticResult:
        """
        Run comprehensive diagnostics on detection results.

        Args:
            scores: Anomaly scores from detector
            labels: Ground truth binary labels
            threshold: Threshold used for binary predictions
            detector_name: Name of the detector
            dataset_name: Name of the dataset

        Returns:
            DiagnosticResult with full analysis
        """
        scores = np.asarray(scores).flatten().astype(np.float64)
        labels = np.asarray(labels).flatten().astype(np.int32)

        n = len(scores)

        # Score statistics
        score_min = float(np.min(scores))
        score_max = float(np.max(scores))
        score_mean = float(np.mean(scores))
        score_std = float(np.std(scores))
        score_median = float(np.median(scores))

        # Predictions with given threshold
        predictions = scores > threshold
        n_predicted = int(np.sum(predictions))
        n_true = int(np.sum(labels))

        # Compute metrics
        roc_auc = BenchmarkDiagnostics._compute_roc_auc(labels, scores)
        f1, precision, recall = BenchmarkDiagnostics._compute_f1_pr(labels, predictions)
        best_f1, best_threshold = BenchmarkDiagnostics._find_best_f1(labels, scores)

        # Percentiles
        percentiles = {
            p: float(np.percentile(scores, p)) for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        }

        # Bimodality check
        is_bimodal = BenchmarkDiagnostics._check_bimodal(scores)

        # Recommended threshold
        if n_true > 0:
            contamination = n_true / n
            threshold_recommended = float(np.percentile(scores, 100 * (1 - contamination)))
        else:
            threshold_recommended = float(np.percentile(scores, 95))

        # Recommended calibration method
        if is_bimodal:
            method_recommended = "otsu"
        elif score_std < 0.1:
            method_recommended = "adaptive_iqr"
        else:
            method_recommended = "percentile"

        # Discrepancy analysis
        discrepancy = MetricDiscrepancy.analyze(
            roc_auc=roc_auc,
            f1=f1,
            precision=precision,
            recall=recall,
            threshold=threshold,
            score_max=score_max,
            score_min=score_min,
        )

        return DiagnosticResult(
            detector_name=detector_name,
            dataset_name=dataset_name,
            n_samples=n,
            n_anomalies_true=n_true,
            n_anomalies_predicted=n_predicted,
            anomaly_ratio_true=n_true / n if n > 0 else 0.0,
            anomaly_ratio_predicted=n_predicted / n if n > 0 else 0.0,
            score_min=score_min,
            score_max=score_max,
            score_mean=score_mean,
            score_std=score_std,
            score_median=score_median,
            threshold_used=threshold,
            threshold_recommended=threshold_recommended,
            roc_auc=roc_auc,
            f1=f1,
            f1_at_best_threshold=best_f1,
            best_threshold=best_threshold,
            precision=precision,
            recall=recall,
            discrepancy=discrepancy,
            percentiles=percentiles,
            is_bimodal=is_bimodal,
            calibration_method_recommended=method_recommended,
        )

    @staticmethod
    def quick_diagnose(
        scores: NDArray[np.float64],
        labels: NDArray[np.int32] | None = None,
        threshold: float = 0.5,
        detector_name: str = "Detector",
    ) -> None:
        """
        Print quick diagnostic matching the user's requested format.

        Args:
            scores: Anomaly scores
            labels: Optional ground truth labels
            threshold: Current threshold
            detector_name: Name for display

        This prints exactly what the user requested:
            Score range: [min, max]
            Score mean: mean
            Threshold: threshold
            Predictions above threshold: count/total
        """
        scores = np.asarray(scores).flatten()

        print(f"\n--- {detector_name} Score Diagnostics ---")
        print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"Score mean: {scores.mean():.4f}")
        print(f"Threshold: {threshold}")
        print(f"Predictions above threshold: {(scores > threshold).sum()}/{len(scores)}")

        if labels is not None:
            labels = np.asarray(labels).flatten()
            predictions = scores > threshold
            tp = np.sum((labels == 1) & predictions)
            fp = np.sum((labels == 0) & predictions)
            fn = np.sum((labels == 1) & ~predictions)
            tn = np.sum((labels == 0) & ~predictions)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            print("\nWith ground truth:")
            print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
            print(f"  Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

            if f1 == 0 and threshold > scores.max():
                print(
                    f"\n>>> DIAGNOSIS: F1=0 because threshold ({threshold:.4f}) > max score ({scores.max():.4f})"
                )
                print(">>> SOLUTION: Use auto-calibration or lower threshold")

        print("-" * 40)

    @staticmethod
    def _compute_roc_auc(labels: NDArray, scores: NDArray) -> float:  # type: ignore[type-arg, unused-ignore]
        """Compute ROC-AUC score."""
        from omni_mercury_engine.ml.mercury_ml import roc_auc_score

        try:
            return float(roc_auc_score(labels, scores))
        except ValueError:
            # Edge case (all same class)
            return 0.5

    @staticmethod
    def _compute_f1_pr(labels: NDArray, predictions: NDArray) -> tuple[float, float, float]:  # type: ignore[type-arg, unused-ignore]
        """Compute F1, precision, recall."""
        tp = np.sum((labels == 1) & predictions)
        fp = np.sum((labels == 0) & predictions)
        fn = np.sum((labels == 1) & ~predictions)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return float(f1), float(precision), float(recall)

    @staticmethod
    def _find_best_f1(labels: NDArray, scores: NDArray) -> tuple[float, float]:  # type: ignore[type-arg, unused-ignore]
        """Find threshold that maximizes F1."""
        thresholds = np.percentile(scores, np.linspace(0, 100, 100))

        best_f1 = 0.0
        best_threshold = 0.5

        for threshold in thresholds:
            predictions = scores > threshold
            f1, _, _ = BenchmarkDiagnostics._compute_f1_pr(labels, predictions)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return float(best_f1), float(best_threshold)

    @staticmethod
    def _check_bimodal(scores: NDArray) -> bool:  # type: ignore[type-arg, unused-ignore]
        """Check if score distribution is bimodal."""
        if len(scores) < 20:
            return False

        hist, _ = np.histogram(scores, bins=50)
        kernel = np.array([1, 2, 3, 2, 1]) / 9.0
        smoothed = np.convolve(hist, kernel, mode="same")

        local_maxima = 0
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
                if smoothed[i] > 0.05 * np.max(smoothed):
                    local_maxima += 1

        return local_maxima >= 2


def run_diagnostic_benchmark(
    detector: Any,
    X_train: NDArray[np.float64],
    X_test: NDArray[np.float64],
    y_test: NDArray[np.int32],
    detector_name: str = "Unknown",
    dataset_name: str = "Unknown",
    print_report: bool = True,
) -> DiagnosticResult:
    """
    Run a complete diagnostic benchmark on a detector.

    This is a convenience function that:
    1. Fits the detector
    2. Runs detection
    3. Generates comprehensive diagnostics
    4. Optionally prints the report

    Args:
        detector: Anomaly detector with fit() and detect() methods
        X_train: Training data
        X_test: Test data
        y_test: Ground truth labels
        detector_name: Name for the report
        dataset_name: Dataset name for the report
        print_report: Whether to print the diagnostic report

    Returns:
        DiagnosticResult with full analysis

    Example:
        from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector
        from omni_mercury_engine.evaluation.benchmark_diagnostics import run_diagnostic_benchmark

        detector = MercuryAnomalyDetector()
        result = run_diagnostic_benchmark(
            detector, X_train, X_test, y_test,
            detector_name="MercuryAnomalyDetector",
            dataset_name="covtype",
        )
    """
    # Fit detector
    detector.fit(X_train)

    # Run detection
    result = detector.detect(X_test)

    # Get scores
    scores = result.get("scores")
    if scores is None:
        scores = result.get("anomaly_score")
        if scores is not None:
            scores = np.array([scores])
        else:
            raise ValueError("Detector result must contain 'scores' or 'anomaly_score'")

    scores = np.asarray(scores).flatten()

    # Get threshold
    threshold = result.get("threshold", getattr(detector, "threshold", 0.5))

    # Run diagnostics
    diagnostics = BenchmarkDiagnostics.diagnose(
        scores=scores,
        labels=y_test,
        threshold=threshold,
        detector_name=detector_name,
        dataset_name=dataset_name,
    )

    if print_report:
        print(diagnostics.report())

    return diagnostics


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "BenchmarkDiagnostics",
    "DiagnosticResult",
    "MetricDiscrepancy",
    "run_diagnostic_benchmark",
]
