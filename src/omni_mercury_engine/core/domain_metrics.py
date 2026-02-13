"""
Mercury Agent - Consolidated Domain Metrics Module
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Consolidates all domain-specific metrics:
- Standard ML metrics (AUC, F1, precision, recall)
- Event-based metrics (temporal anomalies)
- Spatial metrics (autocorrelation, clustering)
- Fairness metrics (demographic parity, equalized odds)
- Quantum metrics (entropy, coherence)
- Calibration metrics (Brier, ECE, MCE)
- Benevolence metrics (harm reduction, equity)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from omni_mercury_engine.core.centralized_constants import ETHICAL, MATH

logger = logging.getLogger(__name__)

# Constants from centralized source of truth
PHI = MATH.GOLDEN_RATIO
BENEVOLENCE_THRESHOLD = ETHICAL.BENEVOLENCE_IMMUTABLE
SIGMA_IMMUTABLE_DEFAULT = 0.96


@dataclass
class ComprehensiveMetrics:
    """Complete metrics suite for anomaly detection evaluation."""

    # Standard Classification Metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    roc_auc: float = 0.0
    pr_auc: float = 0.0

    # Point-Adjusted Metrics (for time series)
    pa_precision: float = 0.0
    pa_recall: float = 0.0
    pa_f1: float = 0.0

    # Event-Based Metrics
    event_precision: float = 0.0
    event_recall: float = 0.0
    event_f1: float = 0.0
    time_to_detection: float = 0.0
    detection_delay: float = 0.0

    # Spatial Metrics
    morans_i: float | None = None
    gearys_c: float | None = None
    spatial_clustering: float | None = None

    # Calibration Metrics
    brier_score: float = 0.0
    ece: float = 0.0  # Expected Calibration Error
    mce: float = 0.0  # Maximum Calibration Error
    calibration_improvement: float = 0.0

    # Fairness Metrics
    demographic_parity: float = 1.0
    equalized_odds: float = 1.0
    disparate_impact: float = 1.0
    individual_fairness: float = 1.0

    # Quantum Metrics
    von_neumann_entropy: float | None = None
    purity: float | None = None
    coherence: float | None = None

    # Benevolence Metrics
    harm_reduction_score: float = 1.0
    equity_score: float = 1.0
    benevolence_index: float = 1.0
    ethical_compliance: bool = True

    # Aggregate Scores
    overall_score: float = 0.0
    domain_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "classification": {
                "accuracy": self.accuracy,
                "precision": self.precision,
                "recall": self.recall,
                "f1_score": self.f1_score,
                "roc_auc": self.roc_auc,
                "pr_auc": self.pr_auc,
            },
            "point_adjusted": {
                "precision": self.pa_precision,
                "recall": self.pa_recall,
                "f1": self.pa_f1,
            },
            "event_based": {
                "precision": self.event_precision,
                "recall": self.event_recall,
                "f1": self.event_f1,
                "time_to_detection": self.time_to_detection,
            },
            "calibration": {
                "brier_score": self.brier_score,
                "ece": self.ece,
                "mce": self.mce,
            },
            "fairness": {
                "demographic_parity": self.demographic_parity,
                "equalized_odds": self.equalized_odds,
                "disparate_impact": self.disparate_impact,
            },
            "benevolence": {
                "harm_reduction": self.harm_reduction_score,
                "equity": self.equity_score,
                "index": self.benevolence_index,
                "compliant": self.ethical_compliance,
            },
            "overall_score": self.overall_score,
        }


class MetricsCalculator:
    """
    Unified metrics calculator for all domains.

    Computes comprehensive metrics suite including standard ML metrics,
    domain-specific metrics, and ethical/benevolence metrics.
    """

    def __init__(
        self,
        benevolence_threshold: float = BENEVOLENCE_THRESHOLD,
        sigma_immutable: float = SIGMA_IMMUTABLE_DEFAULT,
        n_calibration_bins: int = 10,
    ):
        """
        Initialize metrics calculator.

        Args:
            benevolence_threshold: Minimum required benevolence
            sigma_immutable: Ethical threshold
            n_calibration_bins: Bins for calibration metrics
        """
        self.benevolence_threshold = benevolence_threshold
        self.sigma_immutable = sigma_immutable
        self.n_calibration_bins = n_calibration_bins

    def compute_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray | None = None,
        protected_attrs: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
        spatial_weights: np.ndarray | None = None,
    ) -> ComprehensiveMetrics:
        """
        Compute comprehensive metrics suite.

        Args:
            y_true: Ground truth labels
            y_pred: Binary predictions
            y_prob: Probability predictions (optional)
            protected_attrs: Protected attribute values for fairness (optional)
            timestamps: Timestamps for temporal metrics (optional)
            spatial_weights: Spatial weight matrix for spatial metrics (optional)

        Returns:
            ComprehensiveMetrics with all computed values
        """
        metrics = ComprehensiveMetrics()

        # Standard classification metrics
        self._compute_classification_metrics(metrics, y_true, y_pred, y_prob)

        # Event-based metrics for time series
        if timestamps is not None or len(y_true) > 100:
            self._compute_event_metrics(metrics, y_true, y_pred)

        # Calibration metrics
        if y_prob is not None:
            self._compute_calibration_metrics(metrics, y_true, y_prob)

        # Fairness metrics
        if protected_attrs is not None:
            self._compute_fairness_metrics(metrics, y_true, y_pred, protected_attrs)

        # Spatial metrics
        if spatial_weights is not None and y_prob is not None:
            self._compute_spatial_metrics(metrics, y_prob, spatial_weights)

        # Benevolence metrics
        self._compute_benevolence_metrics(metrics, y_true, y_pred, y_prob)

        # Compute overall score
        metrics.overall_score = self._compute_overall_score(metrics)

        return metrics

    def _compute_classification_metrics(
        self,
        metrics: ComprehensiveMetrics,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray | None,
    ) -> None:
        """Compute standard classification metrics."""
        try:
            from sklearn.metrics import (
                accuracy_score,
                average_precision_score,
                f1_score,
                precision_score,
                recall_score,
                roc_auc_score,
            )
        except ImportError as e:
            raise ImportError(
                "This feature requires scikit-learn. Install with: pip install mercury-agent[ml]"
            ) from e

        try:
            metrics.accuracy = float(accuracy_score(y_true, y_pred))
            metrics.precision = float(precision_score(y_true, y_pred, zero_division=0))
            metrics.recall = float(recall_score(y_true, y_pred, zero_division=0))
            metrics.f1_score = float(f1_score(y_true, y_pred, zero_division=0))

            if y_prob is not None and len(np.unique(y_true)) > 1:
                metrics.roc_auc = float(roc_auc_score(y_true, y_prob))
                metrics.pr_auc = float(average_precision_score(y_true, y_prob))

        except Exception as e:
            logger.warning(f"Classification metrics computation failed: {e}")

    def _compute_event_metrics(
        self,
        metrics: ComprehensiveMetrics,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        """Compute event-based metrics for time series data."""
        # Extract events
        true_events = self._extract_events(y_true)
        pred_events = self._extract_events(y_pred)

        if not true_events:
            metrics.event_precision = 1.0 if not pred_events else 0.0
            metrics.event_recall = 1.0
            metrics.event_f1 = 1.0 if not pred_events else 0.0
            return

        if not pred_events:
            metrics.event_precision = 1.0
            metrics.event_recall = 0.0
            metrics.event_f1 = 0.0
            return

        # Event matching with tolerance
        tolerance = 3  # Allow 3-point tolerance

        def events_overlap(e1: tuple[int, int], e2: tuple[int, int]) -> bool:
            return not (e1[1] + tolerance < e2[0] or e2[1] + tolerance < e1[0])

        # Event recall
        detected = sum(1 for te in true_events if any(events_overlap(te, pe) for pe in pred_events))
        metrics.event_recall = detected / len(true_events)

        # Event precision
        matched = sum(1 for pe in pred_events if any(events_overlap(pe, te) for te in true_events))
        metrics.event_precision = matched / len(pred_events)

        # Event F1
        if metrics.event_precision + metrics.event_recall > 0:
            metrics.event_f1 = (
                2
                * metrics.event_precision
                * metrics.event_recall
                / (metrics.event_precision + metrics.event_recall)
            )

        # Time to detection
        detection_times = []
        for start, end in true_events:
            event_preds = y_pred[start : end + 1]
            detected_idx = np.where(event_preds == 1)[0]
            if len(detected_idx) > 0:
                detection_times.append(detected_idx[0])
            else:
                detection_times.append(end - start + 1)

        metrics.time_to_detection = float(np.mean(detection_times)) if detection_times else 0.0

        # Point-adjusted metrics
        y_pred_adjusted = self._point_adjust(y_true, y_pred)
        if np.sum(y_pred_adjusted) > 0:
            try:
                from sklearn.metrics import f1_score, precision_score, recall_score
            except ImportError as e:
                raise ImportError(
                    "This feature requires scikit-learn. Install with: pip install mercury-agent[ml]"
                ) from e
            metrics.pa_precision = float(precision_score(y_true, y_pred_adjusted, zero_division=0))
            metrics.pa_recall = float(recall_score(y_true, y_pred_adjusted, zero_division=0))
            metrics.pa_f1 = float(f1_score(y_true, y_pred_adjusted, zero_division=0))

    def _extract_events(self, labels: np.ndarray) -> list[tuple[int, int]]:
        """Extract contiguous events from binary labels."""
        events = []
        in_event = False
        start = 0

        for i, val in enumerate(labels):
            if val == 1 and not in_event:
                start = i
                in_event = True
            elif val == 0 and in_event:
                events.append((start, i - 1))
                in_event = False

        if in_event:
            events.append((start, len(labels) - 1))

        return events

    def _point_adjust(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Apply point-adjustment to predictions."""
        y_adjusted = y_pred.copy()
        true_events = self._extract_events(y_true)

        for start, end in true_events:
            if np.any(y_pred[start : end + 1] == 1):
                y_adjusted[start : end + 1] = 1

        return y_adjusted

    def _compute_calibration_metrics(
        self,
        metrics: ComprehensiveMetrics,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> None:
        """Compute calibration metrics."""
        try:
            from sklearn.metrics import brier_score_loss
        except ImportError as e:
            raise ImportError(
                "This feature requires scikit-learn. Install with: pip install mercury-agent[ml]"
            ) from e

        try:
            # Brier score
            metrics.brier_score = float(brier_score_loss(y_true, y_prob))

            # Expected Calibration Error
            metrics.ece = self._compute_ece(y_true, y_prob)

            # Maximum Calibration Error
            metrics.mce = self._compute_mce(y_true, y_prob)

        except Exception as e:
            logger.warning(f"Calibration metrics computation failed: {e}")

    def _compute_ece(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Compute Expected Calibration Error."""
        bin_edges = np.linspace(0, 1, self.n_calibration_bins + 1)
        ece = 0.0

        for i in range(self.n_calibration_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if i == self.n_calibration_bins - 1:
                mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])

            if np.sum(mask) > 0:
                bin_acc = np.mean(y_true[mask])
                bin_conf = np.mean(y_prob[mask])
                bin_size = np.sum(mask) / len(y_prob)
                ece += bin_size * np.abs(bin_acc - bin_conf)

        return float(ece)

    def _compute_mce(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Compute Maximum Calibration Error."""
        bin_edges = np.linspace(0, 1, self.n_calibration_bins + 1)
        mce = 0.0

        for i in range(self.n_calibration_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if i == self.n_calibration_bins - 1:
                mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])

            if np.sum(mask) > 0:
                bin_acc = np.mean(y_true[mask])
                bin_conf = np.mean(y_prob[mask])
                mce = max(mce, np.abs(bin_acc - bin_conf))

        return float(mce)

    def _compute_fairness_metrics(
        self,
        metrics: ComprehensiveMetrics,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attrs: np.ndarray,
    ) -> None:
        """Compute fairness metrics."""
        try:
            group_0 = protected_attrs == 0
            group_1 = protected_attrs == 1

            if np.sum(group_0) < 5 or np.sum(group_1) < 5:
                return

            # Positive prediction rates
            rate_0 = np.mean(y_pred[group_0])
            rate_1 = np.mean(y_pred[group_1])

            # Demographic Parity Ratio
            metrics.demographic_parity = min(rate_0, rate_1) / (max(rate_0, rate_1) + 1e-10)

            # True/False Positive Rates
            tpr_0 = (
                np.mean(y_pred[group_0 & (y_true == 1)])
                if np.sum(group_0 & (y_true == 1)) > 0
                else 0.5
            )
            tpr_1 = (
                np.mean(y_pred[group_1 & (y_true == 1)])
                if np.sum(group_1 & (y_true == 1)) > 0
                else 0.5
            )
            fpr_0 = (
                np.mean(y_pred[group_0 & (y_true == 0)])
                if np.sum(group_0 & (y_true == 0)) > 0
                else 0.5
            )
            fpr_1 = (
                np.mean(y_pred[group_1 & (y_true == 0)])
                if np.sum(group_1 & (y_true == 0)) > 0
                else 0.5
            )

            # Equalized Odds (average of TPR and FPR parity)
            tpr_parity = min(tpr_0, tpr_1) / (max(tpr_0, tpr_1) + 1e-10)
            fpr_parity = 1 - abs(fpr_0 - fpr_1)  # 1 = equal, 0 = maximally different
            metrics.equalized_odds = (tpr_parity + fpr_parity) / 2

            # Disparate Impact (80% rule)
            metrics.disparate_impact = metrics.demographic_parity

        except Exception as e:
            logger.warning(f"Fairness metrics computation failed: {e}")

    def _compute_spatial_metrics(
        self,
        metrics: ComprehensiveMetrics,
        scores: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        """Compute spatial autocorrelation metrics."""
        try:
            n = len(scores)
            mean = np.mean(scores)
            deviations = scores - mean

            # Normalize weights
            row_sums = weights.sum(axis=1, keepdims=True)
            weights_norm = weights / (row_sums + 1e-10)

            # Moran's I
            numerator = np.sum(weights_norm * np.outer(deviations, deviations))
            denominator = np.sum(deviations**2)
            w_sum = np.sum(weights_norm)

            if denominator > 0 and w_sum > 0:
                metrics.morans_i = float((n / w_sum) * (numerator / denominator))

            # Geary's C
            diff_matrix = (scores[:, None] - scores[None, :]) ** 2
            numerator_c = np.sum(weights * diff_matrix)
            denominator_c = 2 * np.sum(weights) * np.sum((scores - mean) ** 2)

            if denominator_c > 0:
                metrics.gearys_c = float((n - 1) * numerator_c / denominator_c)

        except Exception as e:
            logger.warning(f"Spatial metrics computation failed: {e}")

    def _compute_benevolence_metrics(
        self,
        metrics: ComprehensiveMetrics,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray | None,
    ) -> None:
        """Compute benevolence and ethical metrics."""
        try:
            # Harm reduction: minimize false negatives (missed anomalies)
            fn = np.sum((y_pred == 0) & (y_true == 1))
            total_positive = np.sum(y_true == 1)
            metrics.harm_reduction_score = 1.0 - (fn / (total_positive + 1e-10))

            # Equity: balanced performance across classes
            if np.sum(y_true == 0) > 0 and np.sum(y_true == 1) > 0:
                acc_0 = np.mean(y_pred[y_true == 0] == 0)
                acc_1 = np.mean(y_pred[y_true == 1] == 1)
                metrics.equity_score = min(acc_0, acc_1) / (max(acc_0, acc_1) + 1e-10)

            # Combined benevolence index with golden ratio weighting
            metrics.benevolence_index = (
                PHI * metrics.harm_reduction_score + metrics.equity_score
            ) / (PHI + 1)

            # Ethical compliance check - explicitly cast to Python bool to avoid numpy.bool_
            metrics.ethical_compliance = bool(
                metrics.benevolence_index >= self.benevolence_threshold
                and metrics.harm_reduction_score >= self.sigma_immutable
            )

        except Exception as e:
            logger.warning(f"Benevolence metrics computation failed: {e}")

    def _compute_overall_score(self, metrics: ComprehensiveMetrics) -> float:
        """
        Compute weighted overall score.

        Combines detection performance with ethical compliance.
        """
        # Detection performance (50%)
        detection_score = (
            0.3 * metrics.roc_auc
            + 0.3 * metrics.pr_auc
            + 0.2 * metrics.f1_score
            + 0.2 * (1 - metrics.brier_score)
        )

        # Event performance (20%)
        event_score = metrics.event_f1 if metrics.event_f1 > 0 else metrics.f1_score

        # Fairness (15%)
        fairness_score = (
            metrics.demographic_parity + metrics.equalized_odds + metrics.disparate_impact
        ) / 3

        # Benevolence (15%)
        benevolence_score = metrics.benevolence_index

        # Weighted combination
        overall = (
            0.50 * detection_score
            + 0.20 * event_score
            + 0.15 * fairness_score
            + 0.15 * benevolence_score
        )

        # Apply ethical gate
        if not metrics.ethical_compliance:
            overall *= 0.5  # Penalty for non-compliance

        return float(np.clip(overall, 0, 1))


class DomainSpecificMetrics:
    """
    Domain-specific metrics for specialized detectors.
    """

    @staticmethod
    def compute_temporal_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamps: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Compute temporal domain metrics."""
        calculator = MetricsCalculator()
        metrics = calculator.compute_all_metrics(y_true, y_pred, timestamps=timestamps)

        return {
            "event_precision": metrics.event_precision,
            "event_recall": metrics.event_recall,
            "event_f1": metrics.event_f1,
            "time_to_detection": metrics.time_to_detection,
            "pa_f1": metrics.pa_f1,
        }

    @staticmethod
    def compute_statistical_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Compute statistical domain metrics."""
        calculator = MetricsCalculator()
        metrics = calculator.compute_all_metrics(y_true, y_pred, y_prob)

        return {
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "roc_auc": metrics.roc_auc,
            "pr_auc": metrics.pr_auc,
        }

    @staticmethod
    def compute_spatial_metrics(
        scores: np.ndarray,
        spatial_weights: np.ndarray,
    ) -> dict[str, float]:
        """Compute spatial domain metrics."""
        calculator = MetricsCalculator()

        # Create dummy labels for metric computation
        threshold = np.percentile(scores, 90)
        y_pred = (scores > threshold).astype(int)
        y_true = y_pred  # Self-supervised

        metrics = calculator.compute_all_metrics(
            y_true,
            y_pred,
            scores,
            spatial_weights=spatial_weights,
        )

        return {
            "morans_i": metrics.morans_i if metrics.morans_i is not None else 0.0,
            "gearys_c": metrics.gearys_c if metrics.gearys_c is not None else 0.0,
        }

    @staticmethod
    def compute_graph_metrics(
        node_scores: np.ndarray,
        adjacency_matrix: np.ndarray,
    ) -> dict[str, float]:
        """Compute graph domain metrics."""
        # Use adjacency as spatial weights
        return DomainSpecificMetrics.compute_spatial_metrics(node_scores, adjacency_matrix)


def compute_benchmark_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    domain: str = "general",
    **kwargs: Any,
) -> ComprehensiveMetrics:
    """
    Convenience function to compute metrics for benchmarking.

    Args:
        y_true: Ground truth labels
        y_pred: Binary predictions
        y_prob: Probability predictions
        domain: Domain type for specialized metrics
        **kwargs: Additional arguments

    Returns:
        ComprehensiveMetrics instance
    """
    calculator = MetricsCalculator()
    return calculator.compute_all_metrics(y_true, y_pred, y_prob, **kwargs)
