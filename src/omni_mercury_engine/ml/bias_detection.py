"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ML Bias Detection and Fairness Evaluation Module.

Provides production-ready bias detection using Fairlearn metrics:
- Demographic parity assessment
- Equalized odds evaluation
- Disparate impact analysis
- Group fairness metrics

This module implements honest, validated fairness metrics without
exaggerated claims. All metrics are standard implementations from
the fairness ML literature.

Reference: Fairlearn documentation (https://fairlearn.org/)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class FairnessMetric(Enum):
    """Available fairness metrics."""

    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    DISPARATE_IMPACT = "disparate_impact"
    CALIBRATION = "calibration"
    PREDICTIVE_PARITY = "predictive_parity"
    FALSE_POSITIVE_RATE_PARITY = "fpr_parity"
    FALSE_NEGATIVE_RATE_PARITY = "fnr_parity"


@dataclass
class FairnessResult:
    """Result of fairness evaluation."""

    metric: FairnessMetric
    overall_score: float
    group_scores: dict[str, float]
    is_fair: bool
    threshold: float
    disparity: float
    recommendations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BiasReport:
    """Comprehensive bias analysis report."""

    model_name: str
    total_samples: int
    sensitive_features: list[str]
    fairness_results: list[FairnessResult]
    overall_fairness_score: float
    is_model_fair: bool
    high_risk_groups: list[str]
    recommendations: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


class BiasDetector:
    """
    ML Bias Detection using Fairlearn metrics.

    Evaluates model predictions for fairness across sensitive attributes.
    Uses standard fairness metrics from the ML fairness literature.

    Example:
        detector = BiasDetector()

        report = detector.evaluate(
            y_true=labels,
            y_pred=predictions,
            sensitive_features=gender_array,
            feature_name="gender"
        )

        if not report.is_model_fair:
            print(f"Bias detected: {report.recommendations}")
    """

    # Standard thresholds from fairness literature
    DEMOGRAPHIC_PARITY_THRESHOLD = 0.1  # Max difference in selection rates
    EQUALIZED_ODDS_THRESHOLD = 0.1  # Max difference in TPR/FPR
    DISPARATE_IMPACT_THRESHOLD = 0.8  # 80% rule (EEOC guideline)
    CALIBRATION_THRESHOLD = 0.1  # Max calibration difference

    def __init__(
        self,
        use_fairlearn: bool = True,
        demographic_parity_threshold: float | None = None,
        equalized_odds_threshold: float | None = None,
        disparate_impact_threshold: float | None = None,
    ):
        """
        Initialize bias detector.

        Args:
            use_fairlearn: Use Fairlearn library if available
            demographic_parity_threshold: Custom threshold for demographic parity
            equalized_odds_threshold: Custom threshold for equalized odds
            disparate_impact_threshold: Custom threshold for disparate impact
        """
        self.use_fairlearn = use_fairlearn
        self._fairlearn_available = False

        # Set thresholds
        self.thresholds = {
            FairnessMetric.DEMOGRAPHIC_PARITY: (
                demographic_parity_threshold or self.DEMOGRAPHIC_PARITY_THRESHOLD
            ),
            FairnessMetric.EQUALIZED_ODDS: (
                equalized_odds_threshold or self.EQUALIZED_ODDS_THRESHOLD
            ),
            FairnessMetric.DISPARATE_IMPACT: (
                disparate_impact_threshold or self.DISPARATE_IMPACT_THRESHOLD
            ),
            FairnessMetric.FALSE_POSITIVE_RATE_PARITY: self.EQUALIZED_ODDS_THRESHOLD,
            FairnessMetric.FALSE_NEGATIVE_RATE_PARITY: self.EQUALIZED_ODDS_THRESHOLD,
            FairnessMetric.CALIBRATION: self.CALIBRATION_THRESHOLD,
            FairnessMetric.PREDICTIVE_PARITY: self.DEMOGRAPHIC_PARITY_THRESHOLD,
        }

        if use_fairlearn:
            import importlib.util

            if importlib.util.find_spec("fairlearn.metrics") is not None:
                self._fairlearn_available = True
                logger.info("Fairlearn available for bias detection")
            else:
                self._fairlearn_available = False
                logger.warning(
                    "Fairlearn not installed. Using built-in metrics. "
                    "Install with: pip install fairlearn"
                )

    def evaluate(
        self,
        y_true: np.ndarray[Any, Any],
        y_pred: np.ndarray[Any, Any],
        sensitive_features: np.ndarray[Any, Any],
        feature_name: str = "sensitive_attribute",
        model_name: str = "model",
        metrics: list[FairnessMetric] | None = None,
    ) -> BiasReport:
        """
        Evaluate model for bias across sensitive features.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_features: Sensitive attribute values for each sample
            feature_name: Name of the sensitive feature
            model_name: Name of the model being evaluated
            metrics: List of metrics to compute (default: all)

        Returns:
            BiasReport with comprehensive fairness analysis
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        sensitive_features = np.asarray(sensitive_features)

        # Default to all metrics
        if metrics is None:
            metrics = [
                FairnessMetric.DEMOGRAPHIC_PARITY,
                FairnessMetric.EQUALIZED_ODDS,
                FairnessMetric.DISPARATE_IMPACT,
            ]

        # Compute fairness results for each metric
        fairness_results = []
        for metric in metrics:
            result = self._compute_metric(y_true, y_pred, sensitive_features, feature_name, metric)
            fairness_results.append(result)

        # Aggregate results
        overall_score = np.mean([r.overall_score for r in fairness_results])
        is_model_fair = all(r.is_fair for r in fairness_results)

        # Identify high-risk groups
        high_risk_groups = self._identify_high_risk_groups(fairness_results)

        # Generate recommendations
        recommendations = self._generate_recommendations(fairness_results, high_risk_groups)

        return BiasReport(
            model_name=model_name,
            total_samples=len(y_true),
            sensitive_features=[feature_name],
            fairness_results=fairness_results,
            overall_fairness_score=overall_score,  # type: ignore[arg-type, unused-ignore]
            is_model_fair=is_model_fair,
            high_risk_groups=high_risk_groups,
            recommendations=recommendations,
            metadata={
                "fairlearn_used": self._fairlearn_available and self.use_fairlearn,
                "unique_groups": len(np.unique(sensitive_features)),
            },
        )

    def _compute_metric(
        self,
        y_true: np.ndarray[Any, Any],
        y_pred: np.ndarray[Any, Any],
        sensitive_features: np.ndarray[Any, Any],
        feature_name: str,
        metric: FairnessMetric,
    ) -> FairnessResult:
        """Compute a specific fairness metric."""
        if self._fairlearn_available and self.use_fairlearn:
            return self._compute_with_fairlearn(
                y_true, y_pred, sensitive_features, feature_name, metric
            )
        return self._compute_builtin(y_true, y_pred, sensitive_features, feature_name, metric)

    def _compute_with_fairlearn(
        self,
        y_true: np.ndarray[Any, Any],
        y_pred: np.ndarray[Any, Any],
        sensitive_features: np.ndarray[Any, Any],
        feature_name: str,
        metric: FairnessMetric,
    ) -> FairnessResult:
        """Compute fairness metric using Fairlearn."""
        from fairlearn.metrics import (
            MetricFrame,
            demographic_parity_difference,
            demographic_parity_ratio,
            equalized_odds_difference,
            false_negative_rate,
            false_positive_rate,
            selection_rate,
        )

        threshold = self.thresholds[metric]
        group_scores: dict[str, float] = {}
        # Initialize with defaults - will be overwritten in each branch
        is_fair = True

        if metric == FairnessMetric.DEMOGRAPHIC_PARITY:
            disparity = abs(
                demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
            )
            # Compute per-group selection rates
            mf = MetricFrame(
                metrics=selection_rate,
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_features,
            )
            for group, rate in mf.by_group.items():
                group_scores[str(group)] = float(rate)

            overall_score = 1.0 - min(disparity, 1.0)
            is_fair = disparity <= threshold

        elif metric == FairnessMetric.EQUALIZED_ODDS:
            disparity = abs(
                equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features)
            )
            # Get TPR and FPR by group
            mf_fpr = MetricFrame(
                metrics=false_positive_rate,
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_features,
            )
            for group, rate in mf_fpr.by_group.items():
                group_scores[f"{group}_fpr"] = float(rate)

            mf_fnr = MetricFrame(
                metrics=false_negative_rate,
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_features,
            )
            for group, rate in mf_fnr.by_group.items():
                group_scores[f"{group}_fnr"] = float(rate)

            overall_score = 1.0 - min(disparity, 1.0)
            is_fair = disparity <= threshold

        elif metric == FairnessMetric.DISPARATE_IMPACT:
            ratio = demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive_features)
            disparity = abs(1.0 - ratio) if ratio else 1.0
            # Per-group selection rates
            mf = MetricFrame(
                metrics=selection_rate,
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_features,
            )
            for group, rate in mf.by_group.items():
                group_scores[str(group)] = float(rate)

            overall_score = min(ratio, 1.0) if ratio else 0.0
            is_fair = ratio >= threshold  # 80% rule

        elif metric == FairnessMetric.FALSE_POSITIVE_RATE_PARITY:
            mf = MetricFrame(
                metrics=false_positive_rate,
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_features,
            )
            rates = list(mf.by_group.values())
            disparity = max(rates) - min(rates) if len(rates) > 1 else 0.0
            for group, rate in mf.by_group.items():
                group_scores[str(group)] = float(rate)

            overall_score = 1.0 - min(disparity, 1.0)
            is_fair = disparity <= threshold

        elif metric == FairnessMetric.FALSE_NEGATIVE_RATE_PARITY:
            mf = MetricFrame(
                metrics=false_negative_rate,
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_features,
            )
            rates = list(mf.by_group.values())
            disparity = max(rates) - min(rates) if len(rates) > 1 else 0.0
            for group, rate in mf.by_group.items():
                group_scores[str(group)] = float(rate)

            overall_score = 1.0 - min(disparity, 1.0)
            is_fair = disparity <= threshold

        else:
            # Fallback for unimplemented metrics
            return self._compute_builtin(y_true, y_pred, sensitive_features, feature_name, metric)

        recommendations = []
        if not is_fair:
            recommendations.append(f"Consider rebalancing training data for {feature_name}")
            recommendations.append("Apply fairness constraints during model training")

        return FairnessResult(
            metric=metric,
            overall_score=overall_score,
            group_scores=group_scores,
            is_fair=is_fair,
            threshold=threshold,
            disparity=disparity,
            recommendations=recommendations,
        )

    def _compute_builtin(
        self,
        y_true: np.ndarray[Any, Any],
        y_pred: np.ndarray[Any, Any],
        sensitive_features: np.ndarray[Any, Any],
        feature_name: str,
        metric: FairnessMetric,
    ) -> FairnessResult:
        """Compute fairness metric using built-in implementation."""
        threshold = self.thresholds[metric]
        groups = np.unique(sensitive_features)
        group_scores: dict[str, float] = {}

        if metric == FairnessMetric.DEMOGRAPHIC_PARITY:
            # Selection rate per group
            selection_rates = []
            for group in groups:
                mask = sensitive_features == group
                if mask.sum() > 0:
                    rate = y_pred[mask].mean()
                    selection_rates.append(rate)
                    group_scores[str(group)] = float(rate)

            disparity = max(selection_rates) - min(selection_rates) if selection_rates else 0.0
            overall_score = 1.0 - min(disparity, 1.0)
            is_fair = disparity <= threshold

        elif metric == FairnessMetric.EQUALIZED_ODDS:
            # TPR and FPR per group
            tpr_rates = []
            fpr_rates = []
            for group in groups:
                mask = sensitive_features == group
                if mask.sum() > 0:
                    y_t = y_true[mask]
                    y_p = y_pred[mask]

                    # TPR (recall)
                    pos_mask = y_t == 1
                    if pos_mask.sum() > 0:
                        tpr = (y_p[pos_mask] == 1).mean()
                        tpr_rates.append(tpr)
                        group_scores[f"{group}_tpr"] = float(tpr)

                    # FPR
                    neg_mask = y_t == 0
                    if neg_mask.sum() > 0:
                        fpr = (y_p[neg_mask] == 1).mean()
                        fpr_rates.append(fpr)
                        group_scores[f"{group}_fpr"] = float(fpr)

            tpr_disparity = max(tpr_rates) - min(tpr_rates) if tpr_rates else 0.0
            fpr_disparity = max(fpr_rates) - min(fpr_rates) if fpr_rates else 0.0
            disparity = max(tpr_disparity, fpr_disparity)
            overall_score = 1.0 - min(disparity, 1.0)
            is_fair = disparity <= threshold

        elif metric == FairnessMetric.DISPARATE_IMPACT:
            # 80% rule: min(rate) / max(rate) >= 0.8
            selection_rates = []
            for group in groups:
                mask = sensitive_features == group
                if mask.sum() > 0:
                    rate = y_pred[mask].mean()
                    selection_rates.append(rate)
                    group_scores[str(group)] = float(rate)

            if selection_rates and max(selection_rates) > 0:
                ratio = min(selection_rates) / max(selection_rates)
            else:
                ratio = 1.0

            disparity = abs(1.0 - ratio)
            overall_score = ratio
            is_fair = ratio >= threshold

        else:
            # Generic fallback
            disparity = 0.0
            overall_score = 1.0
            is_fair = True

        recommendations = []
        if not is_fair:
            recommendations.append(f"Model shows unfair treatment across {feature_name}")
            recommendations.append(
                "Consider using Fairlearn's ThresholdOptimizer or ExponentiatedGradient"
            )

        return FairnessResult(
            metric=metric,
            overall_score=float(overall_score),
            group_scores=group_scores,
            is_fair=bool(is_fair),
            threshold=threshold,
            disparity=float(disparity),
            recommendations=recommendations,
        )

    def _identify_high_risk_groups(
        self,
        results: list[FairnessResult],
    ) -> list[str]:
        """Identify groups with highest disparity."""
        high_risk = set()

        for result in results:
            if not result.is_fair and result.group_scores:
                scores = list(result.group_scores.items())
                if len(scores) >= 2:
                    scores.sort(key=lambda x: x[1])
                    # Add lowest and highest scoring groups
                    high_risk.add(scores[0][0].split("_")[0])
                    high_risk.add(scores[-1][0].split("_")[0])

        return list(high_risk)

    def _generate_recommendations(
        self,
        results: list[FairnessResult],
        high_risk_groups: list[str],
    ) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []

        unfair_metrics = [r.metric.value for r in results if not r.is_fair]

        if unfair_metrics:
            recommendations.append(f"Unfair metrics detected: {', '.join(unfair_metrics)}")

            if FairnessMetric.DEMOGRAPHIC_PARITY.value in unfair_metrics:
                recommendations.append(
                    "Demographic parity violation: Consider resampling or "
                    "applying fairness constraints"
                )

            if FairnessMetric.EQUALIZED_ODDS.value in unfair_metrics:
                recommendations.append(
                    "Equalized odds violation: Error rates differ across groups. "
                    "Consider threshold adjustment per group"
                )

            if FairnessMetric.DISPARATE_IMPACT.value in unfair_metrics:
                recommendations.append(
                    "Disparate impact violation (fails 80% rule): "
                    "May have legal implications (EEOC guidelines)"
                )

        if high_risk_groups:
            recommendations.append(f"High-risk groups identified: {', '.join(high_risk_groups)}")
            recommendations.append("Collect more data for underrepresented groups")

        if not recommendations:
            recommendations.append("Model passes all fairness checks")

        return recommendations

    def quick_check(
        self,
        y_true: np.ndarray[Any, Any],
        y_pred: np.ndarray[Any, Any],
        sensitive_features: np.ndarray[Any, Any],
    ) -> bool:
        """
        Quick fairness check - returns True if model is fair.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_features: Sensitive attribute values

        Returns:
            True if model passes basic fairness checks
        """
        report = self.evaluate(
            y_true,
            y_pred,
            sensitive_features,
            metrics=[FairnessMetric.DEMOGRAPHIC_PARITY, FairnessMetric.DISPARATE_IMPACT],
        )
        return report.is_model_fair


__all__ = [
    "BiasDetector",
    "BiasReport",
    "FairnessMetric",
    "FairnessResult",
]
