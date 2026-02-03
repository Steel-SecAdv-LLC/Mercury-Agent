"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

from __future__ import annotations


"""
Fairness and Bias Mitigation Module

Provides Fairlearn-compatible bias detection and mitigation for anomaly detection:
- Demographic parity in anomaly scoring
- Equalized odds across protected groups
- Calibration across subgroups
- Bias auditing and reporting

Aligned with ethical governance requirements (benevolence >= 0.99).
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


class FairnessMetric(str, Enum):
    """Supported fairness metrics."""

    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    PREDICTIVE_PARITY = "predictive_parity"
    CALIBRATION = "calibration"
    DISPARATE_IMPACT = "disparate_impact"


class MitigationStrategy(str, Enum):
    """Bias mitigation strategies."""

    REWEIGHTING = "reweighting"
    THRESHOLD_OPTIMIZATION = "threshold_optimization"
    POST_PROCESSING = "post_processing"
    ADVERSARIAL_DEBIASING = "adversarial_debiasing"


@dataclass
class FairnessReport:
    """Report containing fairness analysis results."""

    overall_fairness_score: float
    metric_scores: dict[str, float]
    group_scores: dict[str, dict[str, float]]
    violations: list[str]
    recommendations: list[str]
    is_fair: bool
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_fairness_score": self.overall_fairness_score,
            "metric_scores": self.metric_scores,
            "group_scores": self.group_scores,
            "violations": self.violations,
            "recommendations": self.recommendations,
            "is_fair": self.is_fair,
            "details": self.details,
        }


@dataclass
class BiasAuditConfig:
    """Configuration for bias auditing."""

    metrics: list[FairnessMetric] = field(
        default_factory=lambda: [
            FairnessMetric.DEMOGRAPHIC_PARITY,
            FairnessMetric.EQUALIZED_ODDS,
        ]
    )
    fairness_threshold: float = 0.8  # Minimum ratio for fairness
    max_disparity: float = 0.2  # Maximum allowed disparity
    protected_features: list[str] = field(default_factory=list)
    reference_group: str | None = None


class FairnessAuditor:
    """
    Fairness auditor for anomaly detection models.

    Computes fairness metrics across protected groups and
    identifies potential bias in anomaly scoring.
    """

    def __init__(self, config: BiasAuditConfig | None = None):
        """
        Initialize fairness auditor.

        Args:
            config: Bias audit configuration
        """
        self.config = config or BiasAuditConfig()

    def compute_demographic_parity(
        self,
        predictions: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> dict[str, Any]:
        """
        Compute demographic parity difference.

        Demographic parity is achieved when the probability of a positive
        prediction is the same across all groups.

        Args:
            predictions: Binary predictions (0/1)
            sensitive_features: Group membership array

        Returns:
            Dictionary with parity scores per group
        """
        groups = np.unique(sensitive_features)
        group_rates = {}

        for group in groups:
            mask = sensitive_features == group
            group_rate = np.mean(predictions[mask])
            group_rates[str(group)] = float(group_rate)

        # Compute overall rate
        overall_rate = np.mean(predictions)

        # Compute disparities
        disparities = {g: abs(r - overall_rate) for g, r in group_rates.items()}

        return {
            "group_rates": group_rates,
            "overall_rate": float(overall_rate),
            "max_disparity": float(max(disparities.values())) if disparities else 0.0,
            "parity_score": float(1.0 - max(disparities.values())) if disparities else 1.0,
        }

    def compute_equalized_odds(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> dict[str, Any]:
        """
        Compute equalized odds difference.

        Equalized odds is achieved when TPR and FPR are equal across groups.

        Args:
            predictions: Binary predictions
            labels: True labels
            sensitive_features: Group membership

        Returns:
            Dictionary with equalized odds scores
        """
        groups = np.unique(sensitive_features)
        group_tpr = {}
        group_fpr = {}

        for group in groups:
            mask = sensitive_features == group

            # True positive rate
            pos_mask = mask & (labels == 1)
            if np.sum(pos_mask) > 0:
                tpr = np.mean(predictions[pos_mask])
            else:
                tpr = 0.0
            group_tpr[str(group)] = float(tpr)

            # False positive rate
            neg_mask = mask & (labels == 0)
            if np.sum(neg_mask) > 0:
                fpr = np.mean(predictions[neg_mask])
            else:
                fpr = 0.0
            group_fpr[str(group)] = float(fpr)

        # Compute max differences
        tpr_values = list(group_tpr.values())
        fpr_values = list(group_fpr.values())

        tpr_diff = max(tpr_values) - min(tpr_values) if tpr_values else 0.0
        fpr_diff = max(fpr_values) - min(fpr_values) if fpr_values else 0.0

        return {
            "group_tpr": group_tpr,
            "group_fpr": group_fpr,
            "tpr_difference": float(tpr_diff),
            "fpr_difference": float(fpr_diff),
            "equalized_odds_score": float(1.0 - max(tpr_diff, fpr_diff)),
        }

    def compute_disparate_impact(
        self,
        predictions: np.ndarray,
        sensitive_features: np.ndarray,
        reference_group: str | None = None,
    ) -> dict[str, Any]:
        """
        Compute disparate impact ratio.

        Disparate impact occurs when the selection rate for one group
        is less than 80% of another group (4/5ths rule).

        Args:
            predictions: Binary predictions
            sensitive_features: Group membership
            reference_group: Reference group for comparison

        Returns:
            Dictionary with disparate impact ratios
        """
        groups = np.unique(sensitive_features)
        group_rates = {}

        for group in groups:
            mask = sensitive_features == group
            rate = np.mean(predictions[mask])
            group_rates[str(group)] = float(rate)

        # Determine reference group
        if reference_group is None:
            # Use group with highest rate as reference
            reference_group = max(group_rates, key=lambda k: group_rates[k])

        reference_rate = group_rates.get(str(reference_group), 1.0)

        # Compute impact ratios
        impact_ratios = {}
        for group, rate in group_rates.items():
            if reference_rate > 0:
                ratio = rate / reference_rate
            else:
                ratio = 1.0
            impact_ratios[group] = float(ratio)

        # Check 4/5ths rule
        min_ratio = min(impact_ratios.values()) if impact_ratios else 1.0
        passes_rule = min_ratio >= 0.8

        return {
            "group_rates": group_rates,
            "impact_ratios": impact_ratios,
            "min_ratio": float(min_ratio),
            "passes_four_fifths": passes_rule,
            "reference_group": str(reference_group),
        }

    def compute_calibration(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        sensitive_features: np.ndarray,
        n_bins: int = 10,
    ) -> dict[str, Any]:
        """
        Compute calibration across groups.

        Checks if predicted probabilities match actual outcomes
        consistently across groups.

        Args:
            predictions: Predicted probabilities
            labels: True labels
            sensitive_features: Group membership
            n_bins: Number of calibration bins

        Returns:
            Dictionary with calibration results
        """
        groups = np.unique(sensitive_features)
        group_calibration = {}

        for group in groups:
            mask = sensitive_features == group
            group_preds = predictions[mask]
            group_labels = labels[mask]

            # Compute calibration error
            bin_edges = np.linspace(0, 1, n_bins + 1)
            calibration_errors = []

            for i in range(n_bins):
                bin_mask = (group_preds >= bin_edges[i]) & (group_preds < bin_edges[i + 1])
                if np.sum(bin_mask) > 0:
                    mean_pred = np.mean(group_preds[bin_mask])
                    mean_actual = np.mean(group_labels[bin_mask])
                    calibration_errors.append(abs(mean_pred - mean_actual))

            if calibration_errors:
                ece = np.mean(calibration_errors)
            else:
                ece = 0.0

            group_calibration[str(group)] = {
                "ece": float(ece),
                "n_samples": int(np.sum(mask)),
            }

        # Compute max calibration gap
        ece_values = [g["ece"] for g in group_calibration.values()]
        max_gap = max(ece_values) - min(ece_values) if ece_values else 0.0

        return {
            "group_calibration": group_calibration,
            "max_calibration_gap": float(max_gap),
            "calibration_score": float(1.0 - max_gap),
        }

    def audit(
        self,
        predictions: np.ndarray,
        labels: np.ndarray | None = None,
        sensitive_features: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> FairnessReport:
        """
        Perform comprehensive fairness audit.

        Args:
            predictions: Model predictions
            labels: True labels (optional for some metrics)
            sensitive_features: Protected group membership
            feature_names: Names of sensitive features

        Returns:
            FairnessReport with audit results
        """
        metric_scores = {}
        group_scores = {}
        violations = []
        recommendations = []

        if sensitive_features is None:
            return FairnessReport(
                overall_fairness_score=1.0,
                metric_scores={},
                group_scores={},
                violations=[],
                recommendations=["No sensitive features provided for audit"],
                is_fair=True,
            )

        # Compute demographic parity
        if FairnessMetric.DEMOGRAPHIC_PARITY in self.config.metrics:
            dp_results = self.compute_demographic_parity(predictions, sensitive_features)
            metric_scores["demographic_parity"] = dp_results["parity_score"]
            group_scores["demographic_parity"] = dp_results["group_rates"]

            if dp_results["max_disparity"] > self.config.max_disparity:
                violations.append(
                    f"Demographic parity violation: max disparity {dp_results['max_disparity']:.3f}"
                )
                recommendations.append(
                    "Consider reweighting training data or adjusting thresholds per group"
                )

        # Compute equalized odds if labels available
        if labels is not None and FairnessMetric.EQUALIZED_ODDS in self.config.metrics:
            eo_results = self.compute_equalized_odds(predictions, labels, sensitive_features)
            metric_scores["equalized_odds"] = eo_results["equalized_odds_score"]
            group_scores["tpr"] = eo_results["group_tpr"]
            group_scores["fpr"] = eo_results["group_fpr"]

            if (
                max(eo_results["tpr_difference"], eo_results["fpr_difference"])
                > self.config.max_disparity
            ):
                violations.append(
                    f"Equalized odds violation: TPR diff {eo_results['tpr_difference']:.3f}, "
                    f"FPR diff {eo_results['fpr_difference']:.3f}"
                )
                recommendations.append(
                    "Consider post-processing to equalize error rates across groups"
                )

        # Compute disparate impact
        if FairnessMetric.DISPARATE_IMPACT in self.config.metrics:
            di_results = self.compute_disparate_impact(
                predictions, sensitive_features, self.config.reference_group
            )
            metric_scores["disparate_impact"] = di_results["min_ratio"]
            group_scores["impact_ratios"] = di_results["impact_ratios"]

            if not di_results["passes_four_fifths"]:
                violations.append(
                    f"Disparate impact violation: min ratio {di_results['min_ratio']:.3f} < 0.8"
                )
                recommendations.append("Review selection criteria for potential discrimination")

        # Compute calibration if labels available
        if labels is not None and FairnessMetric.CALIBRATION in self.config.metrics:
            cal_results = self.compute_calibration(predictions, labels, sensitive_features)
            metric_scores["calibration"] = cal_results["calibration_score"]
            group_scores["calibration"] = {
                g: v["ece"] for g, v in cal_results["group_calibration"].items()
            }

        # Calculate overall fairness score
        if metric_scores:
            overall_score = np.mean(list(metric_scores.values()))
        else:
            overall_score = 1.0

        is_fair = len(violations) == 0 and overall_score >= self.config.fairness_threshold

        return FairnessReport(
            overall_fairness_score=float(overall_score),
            metric_scores=metric_scores,
            group_scores=group_scores,
            violations=violations,
            recommendations=recommendations,
            is_fair=is_fair,
            details={
                "n_samples": len(predictions),
                "n_groups": len(np.unique(sensitive_features)),
                "config": {
                    "fairness_threshold": self.config.fairness_threshold,
                    "max_disparity": self.config.max_disparity,
                },
            },
        )


class BiasmitigationProcessor:
    """
    Post-processing bias mitigation.

    Applies threshold optimization and other post-hoc
    corrections to reduce bias in predictions.
    """

    def __init__(
        self,
        strategy: MitigationStrategy = MitigationStrategy.THRESHOLD_OPTIMIZATION,
        fairness_constraint: FairnessMetric = FairnessMetric.DEMOGRAPHIC_PARITY,
    ):
        """
        Initialize bias mitigation processor.

        Args:
            strategy: Mitigation strategy to use
            fairness_constraint: Fairness metric to optimize
        """
        self.strategy = strategy
        self.fairness_constraint = fairness_constraint
        self.group_thresholds: dict[str, float] = {}

    def fit(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> BiasmitigationProcessor:
        """
        Fit the mitigation processor.

        Args:
            predictions: Predicted probabilities
            labels: True labels
            sensitive_features: Group membership

        Returns:
            Self for method chaining
        """
        if self.strategy == MitigationStrategy.THRESHOLD_OPTIMIZATION:
            self._fit_threshold_optimization(predictions, labels, sensitive_features)
        elif self.strategy == MitigationStrategy.REWEIGHTING:
            self._fit_reweighting(predictions, labels, sensitive_features)

        return self

    def _fit_threshold_optimization(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> None:
        """Fit group-specific thresholds for demographic parity."""
        groups = np.unique(sensitive_features)
        target_rate = np.mean(labels)

        for group in groups:
            mask = sensitive_features == group
            group_preds = predictions[mask]

            # Find threshold that achieves target rate
            sorted_preds = np.sort(group_preds)[::-1]
            n_select = int(len(sorted_preds) * target_rate)
            n_select = max(1, min(n_select, len(sorted_preds) - 1))

            threshold = sorted_preds[n_select]
            self.group_thresholds[str(group)] = float(threshold)

    def _fit_reweighting(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> None:
        """Compute sample weights for reweighting strategy."""
        # Placeholder for reweighting implementation
        groups = np.unique(sensitive_features)
        for group in groups:
            self.group_thresholds[str(group)] = 0.5

    def transform(
        self,
        predictions: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> np.ndarray:
        """
        Apply mitigation to predictions.

        Args:
            predictions: Predicted probabilities
            sensitive_features: Group membership

        Returns:
            Adjusted predictions
        """
        if self.strategy == MitigationStrategy.THRESHOLD_OPTIMIZATION:
            return self._apply_threshold_optimization(predictions, sensitive_features)
        else:
            return predictions

    def _apply_threshold_optimization(
        self,
        predictions: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> np.ndarray:
        """Apply group-specific thresholds."""
        adjusted = np.zeros_like(predictions)

        for group, threshold in self.group_thresholds.items():
            mask = sensitive_features == group
            adjusted[mask] = (predictions[mask] >= threshold).astype(float)

        return adjusted


def compute_fairness_score(
    predictions: np.ndarray,
    labels: np.ndarray | None,
    sensitive_features: np.ndarray,
    metric: FairnessMetric = FairnessMetric.DEMOGRAPHIC_PARITY,
) -> float:
    """
    Quick fairness score computation.

    Args:
        predictions: Model predictions
        labels: True labels (optional)
        sensitive_features: Protected group membership
        metric: Fairness metric to compute

    Returns:
        Fairness score between 0 and 1
    """
    auditor = FairnessAuditor(BiasAuditConfig(metrics=[metric]))

    if metric == FairnessMetric.DEMOGRAPHIC_PARITY:
        result = auditor.compute_demographic_parity(predictions, sensitive_features)
        return float(result["parity_score"])
    elif metric == FairnessMetric.DISPARATE_IMPACT:
        result = auditor.compute_disparate_impact(predictions, sensitive_features)
        return float(result["min_ratio"])
    elif labels is not None and metric == FairnessMetric.EQUALIZED_ODDS:
        result = auditor.compute_equalized_odds(predictions, labels, sensitive_features)
        return float(result["equalized_odds_score"])
    else:
        return 1.0
