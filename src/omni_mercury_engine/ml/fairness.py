# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fairness and Bias Mitigation Module.

Provides Fairlearn-compatible bias detection and mitigation for anomaly detection:
- Demographic parity in anomaly scoring
- Equalized odds across protected groups
- Calibration across subgroups
- Intersectional (joint-subgroup) parity and equalized odds
- Bias auditing and reporting

Marginal metrics measure each protected feature independently; the
intersectional metrics measure the *joint* subgroups formed by crossing
two or more protected features (e.g. ``(race, gender)``), because a model
can satisfy every marginal constraint while still disadvantaging a joint
subgroup (the classic Simpson's-paradox failure mode).  Joint cells are
sparse by construction, so both intersectional metrics exclude cells
below a configurable ``intersectional_min_group_size`` from the disparity
maximum — a one-sample cell would otherwise dominate the gap with pure
noise — and report the excluded cells instead of silently dropping them.

Aligned with ethical governance requirements (benevolence >= 0.99).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: Separator used inside joint-subgroup labels, e.g. ``"race=B|gender=F"``.
#: Feature names are embedded so labels stay self-describing and stable
#: across audits regardless of feature ordering at the call site.
_INTERSECTION_SEPARATOR = "|"


class FairnessMetric(StrEnum):
    """Supported fairness metrics."""

    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    PREDICTIVE_PARITY = "predictive_parity"
    CALIBRATION = "calibration"
    DISPARATE_IMPACT = "disparate_impact"
    INTERSECTIONAL_PARITY = "intersectional_parity"
    INTERSECTIONAL_EQUALIZED_ODDS = "intersectional_equalized_odds"


class MitigationStrategy(StrEnum):
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
            FairnessMetric.INTERSECTIONAL_PARITY,
            FairnessMetric.INTERSECTIONAL_EQUALIZED_ODDS,
        ]
    )
    fairness_threshold: float = 0.8  # Minimum ratio for fairness
    max_disparity: float = 0.2  # Maximum allowed disparity
    protected_features: list[str] = field(default_factory=list)
    reference_group: str | None = None
    # Joint cells with fewer samples than this are excluded from the
    # intersectional disparity maximum (and reported as ``small_groups``)
    # so a near-empty cell cannot dominate the gap with sampling noise.
    intersectional_min_group_size: int = 10


def build_intersectional_groups(
    sensitive_features: Mapping[str, np.ndarray[Any, Any]] | np.ndarray[Any, Any],
    feature_names: Sequence[str] | None = None,
) -> tuple[np.ndarray[Any, Any], list[str]]:
    """Build joint-subgroup labels by crossing protected features.

    Args:
        sensitive_features: Either a mapping of feature name -> per-sample
            values (all the same length), a 2-D array of shape
            ``(n_samples, n_features)``, or a 1-D array (treated as a
            single feature).
        feature_names: Names for array input, one per feature/column.
            Ignored for mapping input (the mapping's keys are used).
            Defaults to ``feature_0..feature_{k-1}``.

    Returns:
        Tuple of (joint labels of shape ``(n_samples,)`` such as
        ``"race=B|gender=F"``, ordered list of feature names used).

    Raises:
        ValueError: If the input is empty, lengths disagree, or
            ``feature_names`` does not match the feature count.
    """
    if isinstance(sensitive_features, Mapping):
        if not sensitive_features:
            raise ValueError("sensitive_features mapping must not be empty")
        names = [str(k) for k in sensitive_features]
        columns = [np.asarray(v) for v in sensitive_features.values()]
    else:
        array = np.asarray(sensitive_features)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            raise ValueError(f"sensitive_features array must be 1-D or 2-D, got ndim={array.ndim}")
        if array.shape[1] == 0:
            raise ValueError("sensitive_features array must have at least one feature column")
        names = (
            [str(n) for n in feature_names]
            if feature_names is not None
            else [f"feature_{i}" for i in range(array.shape[1])]
        )
        if len(names) != array.shape[1]:
            raise ValueError(
                f"feature_names has {len(names)} entries for {array.shape[1]} feature columns"
            )
        columns = [array[:, i] for i in range(array.shape[1])]

    lengths = {len(col) for col in columns}
    if len(lengths) != 1:
        raise ValueError(f"all sensitive features must have the same length, got {sorted(lengths)}")

    n_samples = lengths.pop()
    labels = np.empty(n_samples, dtype=object)
    string_columns = [col.astype(str) for col in columns]
    for i in range(n_samples):
        labels[i] = _INTERSECTION_SEPARATOR.join(
            f"{name}={col[i]}" for name, col in zip(names, string_columns)
        )
    return labels.astype(str), names


class FairnessAuditor:
    """Fairness auditor for anomaly detection models.

    Computes fairness metrics across protected groups and identifies potential bias in anomaly
    scoring.
    """

    def __init__(self, config: BiasAuditConfig | None = None):
        """Initialize fairness auditor.

        Args:
            config: Bias audit configuration
        """
        self.config = config or BiasAuditConfig()

    def compute_demographic_parity(
        self,
        predictions: np.ndarray[Any, Any],
        sensitive_features: np.ndarray[Any, Any],
    ) -> dict[str, Any]:
        """Compute demographic parity difference.

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
        predictions: np.ndarray[Any, Any],
        labels: np.ndarray[Any, Any],
        sensitive_features: np.ndarray[Any, Any],
    ) -> dict[str, Any]:
        """Compute equalized odds difference.

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
        predictions: np.ndarray[Any, Any],
        sensitive_features: np.ndarray[Any, Any],
        reference_group: str | None = None,
    ) -> dict[str, Any]:
        """Compute disparate impact ratio.

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
        predictions: np.ndarray[Any, Any],
        labels: np.ndarray[Any, Any],
        sensitive_features: np.ndarray[Any, Any],
        n_bins: int = 10,
    ) -> dict[str, Any]:
        """Compute calibration across groups.

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

    def compute_intersectional_parity(
        self,
        predictions: np.ndarray[Any, Any],
        sensitive_features: Mapping[str, np.ndarray[Any, Any]] | np.ndarray[Any, Any],
        feature_names: Sequence[str] | None = None,
        min_group_size: int | None = None,
    ) -> dict[str, Any]:
        """Compute demographic parity across joint (intersectional) subgroups.

        Marginal parity per feature can hold while a joint subgroup (e.g.
        one ``(race, gender)`` cell) is still disadvantaged; this metric
        measures the selection-rate gap across the crossed cells directly.

        Args:
            predictions: Binary predictions (0/1).
            sensitive_features: Mapping of feature name -> values, or a
                2-D array of shape ``(n_samples, n_features)``.
            feature_names: Names for 2-D array input (see
                :func:`build_intersectional_groups`).
            min_group_size: Cells smaller than this are excluded from the
                disparity maximum and reported under ``small_groups``.
                Defaults to ``config.intersectional_min_group_size``.

        Returns:
            Dictionary with joint-group rates, the excluded small cells,
            the worst-off evaluated cell, and the parity score.  When no
            cell reaches ``min_group_size`` the result is flagged
            ``insufficient_data=True`` with a neutral score rather than a
            fabricated verdict.
        """
        floor = (
            self.config.intersectional_min_group_size if min_group_size is None else min_group_size
        )
        joint_labels, names = build_intersectional_groups(sensitive_features, feature_names)
        predictions = np.asarray(predictions)
        if len(predictions) != len(joint_labels):
            raise ValueError(
                f"predictions has {len(predictions)} samples but sensitive features have "
                f"{len(joint_labels)}"
            )

        overall_rate = float(np.mean(predictions)) if len(predictions) else 0.0
        group_rates: dict[str, float] = {}
        group_sizes: dict[str, int] = {}
        small_groups: dict[str, int] = {}
        for label in np.unique(joint_labels):
            mask = joint_labels == label
            n = int(np.sum(mask))
            if n < floor:
                small_groups[str(label)] = n
                continue
            group_rates[str(label)] = float(np.mean(predictions[mask]))
            group_sizes[str(label)] = n

        if not group_rates:
            return {
                "feature_names": names,
                "group_rates": {},
                "group_sizes": {},
                "small_groups": small_groups,
                "overall_rate": overall_rate,
                "max_disparity": 0.0,
                "parity_score": 1.0,
                "worst_group": None,
                "min_group_size": floor,
                "insufficient_data": True,
            }

        disparities = {g: abs(r - overall_rate) for g, r in group_rates.items()}
        worst_group = max(disparities, key=lambda g: disparities[g])
        max_disparity = disparities[worst_group]
        return {
            "feature_names": names,
            "group_rates": group_rates,
            "group_sizes": group_sizes,
            "small_groups": small_groups,
            "overall_rate": overall_rate,
            "max_disparity": float(max_disparity),
            "parity_score": float(1.0 - max_disparity),
            "worst_group": worst_group,
            "min_group_size": floor,
            "insufficient_data": False,
        }

    def compute_intersectional_equalized_odds(
        self,
        predictions: np.ndarray[Any, Any],
        labels: np.ndarray[Any, Any],
        sensitive_features: Mapping[str, np.ndarray[Any, Any]] | np.ndarray[Any, Any],
        feature_names: Sequence[str] | None = None,
        min_group_size: int | None = None,
    ) -> dict[str, Any]:
        """Compute equalized odds across joint (intersectional) subgroups.

        TPR gaps are measured only across cells that contain at least one
        positive label and FPR gaps only across cells with at least one
        negative label (a cell with no positives has no defined TPR; the
        marginal metric's ``0.0`` fallback would fabricate a gap).  Cells
        below ``min_group_size`` are excluded entirely and reported.

        Args:
            predictions: Binary predictions (0/1).
            labels: True binary labels (0/1).
            sensitive_features: Mapping of feature name -> values, or a
                2-D array of shape ``(n_samples, n_features)``.
            feature_names: Names for 2-D array input.
            min_group_size: Cell-size floor; defaults to
                ``config.intersectional_min_group_size``.

        Returns:
            Dictionary with per-cell TPR/FPR, the gaps, the worst-off
            cells, and the equalized-odds score (``1 - max(gap)``).
        """
        floor = (
            self.config.intersectional_min_group_size if min_group_size is None else min_group_size
        )
        joint_labels, names = build_intersectional_groups(sensitive_features, feature_names)
        predictions = np.asarray(predictions)
        labels = np.asarray(labels)
        if not (len(predictions) == len(labels) == len(joint_labels)):
            raise ValueError(
                f"length mismatch: predictions={len(predictions)}, labels={len(labels)}, "
                f"sensitive features={len(joint_labels)}"
            )

        group_tpr: dict[str, float] = {}
        group_fpr: dict[str, float] = {}
        small_groups: dict[str, int] = {}
        for label in np.unique(joint_labels):
            mask = joint_labels == label
            n = int(np.sum(mask))
            if n < floor:
                small_groups[str(label)] = n
                continue
            pos_mask = mask & (labels == 1)
            if np.any(pos_mask):
                group_tpr[str(label)] = float(np.mean(predictions[pos_mask]))
            neg_mask = mask & (labels == 0)
            if np.any(neg_mask):
                group_fpr[str(label)] = float(np.mean(predictions[neg_mask]))

        def _gap(rates: dict[str, float]) -> tuple[float, str | None]:
            if len(rates) < 2:
                return 0.0, None
            hi = max(rates, key=lambda g: rates[g])
            lo = min(rates, key=lambda g: rates[g])
            return rates[hi] - rates[lo], lo

        tpr_gap, worst_tpr_group = _gap(group_tpr)
        fpr_gap, _ = _gap(group_fpr)
        worst_fpr_group = (
            max(group_fpr, key=lambda g: group_fpr[g]) if len(group_fpr) >= 2 else None
        )
        insufficient = len(group_tpr) < 2 and len(group_fpr) < 2
        return {
            "feature_names": names,
            "group_tpr": group_tpr,
            "group_fpr": group_fpr,
            "small_groups": small_groups,
            "tpr_difference": float(tpr_gap),
            "fpr_difference": float(fpr_gap),
            "worst_tpr_group": worst_tpr_group,
            "worst_fpr_group": worst_fpr_group,
            "equalized_odds_score": float(1.0 - max(tpr_gap, fpr_gap)),
            "min_group_size": floor,
            "insufficient_data": insufficient,
        }

    def audit(
        self,
        predictions: np.ndarray[Any, Any],
        labels: np.ndarray[Any, Any] | None = None,
        sensitive_features: np.ndarray[Any, Any] | Mapping[str, np.ndarray[Any, Any]] | None = None,
        feature_names: list[str] | None = None,
    ) -> FairnessReport:
        """Perform comprehensive fairness audit.

        Accepts a single protected feature (1-D array — the historical
        surface, unchanged) or multiple named protected features (a
        mapping of name -> values, or a 2-D array plus ``feature_names``).
        With two or more features the marginal metrics run per feature
        (keys suffixed ``":<name>"``) and the intersectional metrics run
        on the crossed joint subgroups.

        Args:
            predictions: Model predictions
            labels: True labels (optional for some metrics)
            sensitive_features: Protected group membership — 1-D array,
                mapping of feature name -> values, or 2-D array of shape
                ``(n_samples, n_features)``
            feature_names: Names of sensitive features (2-D array input)

        Returns:
            FairnessReport with audit results
        """
        metric_scores: dict[str, float] = {}
        group_scores: dict[str, dict[str, float]] = {}
        violations: list[str] = []
        recommendations: list[str] = []

        if sensitive_features is None:
            return FairnessReport(
                overall_fairness_score=1.0,
                metric_scores={},
                group_scores={},
                violations=[],
                recommendations=["No sensitive features provided for audit"],
                is_fair=True,
            )

        # Normalise the input into named 1-D columns.  A single feature
        # keeps the historical un-suffixed metric keys so existing
        # consumers (and the engine's single-feature call sites) see an
        # unchanged report shape.
        named_features: dict[str, np.ndarray[Any, Any]]
        if isinstance(sensitive_features, Mapping):
            named_features = {str(k): np.asarray(v) for k, v in sensitive_features.items()}
            if not named_features:
                raise ValueError("sensitive_features mapping must not be empty")
        else:
            array = np.asarray(sensitive_features)
            if array.ndim == 2 and array.shape[1] > 1:
                names = (
                    [str(n) for n in feature_names]
                    if feature_names is not None
                    else [f"feature_{i}" for i in range(array.shape[1])]
                )
                if len(names) != array.shape[1]:
                    raise ValueError(
                        f"feature_names has {len(names)} entries for "
                        f"{array.shape[1]} feature columns"
                    )
                named_features = {names[i]: array[:, i] for i in range(array.shape[1])}
            else:
                single = array[:, 0] if array.ndim == 2 else array
                single_name = (
                    str(feature_names[0])
                    if feature_names
                    else (
                        self.config.protected_features[0] if self.config.protected_features else ""
                    )
                )
                named_features = {single_name: single}

        intersectional = len(named_features) >= 2

        def _marginal_key(metric: str, name: str) -> str:
            return f"{metric}:{name}" if intersectional else metric

        for name, column in named_features.items():
            # Compute demographic parity
            if FairnessMetric.DEMOGRAPHIC_PARITY in self.config.metrics:
                dp_results = self.compute_demographic_parity(predictions, column)
                key = _marginal_key("demographic_parity", name)
                metric_scores[key] = dp_results["parity_score"]
                group_scores[key] = dp_results["group_rates"]

                if dp_results["max_disparity"] > self.config.max_disparity:
                    violations.append(
                        f"Demographic parity violation"
                        f"{f' ({name})' if intersectional else ''}: "
                        f"max disparity {dp_results['max_disparity']:.3f}"
                    )
                    recommendations.append(
                        "Consider reweighting training data or adjusting thresholds per group"
                    )

            # Compute equalized odds if labels available
            if labels is not None and FairnessMetric.EQUALIZED_ODDS in self.config.metrics:
                eo_results = self.compute_equalized_odds(predictions, labels, column)
                metric_scores[_marginal_key("equalized_odds", name)] = eo_results[
                    "equalized_odds_score"
                ]
                group_scores[_marginal_key("tpr", name)] = eo_results["group_tpr"]
                group_scores[_marginal_key("fpr", name)] = eo_results["group_fpr"]

                if (
                    max(eo_results["tpr_difference"], eo_results["fpr_difference"])
                    > self.config.max_disparity
                ):
                    violations.append(
                        f"Equalized odds violation"
                        f"{f' ({name})' if intersectional else ''}: "
                        f"TPR diff {eo_results['tpr_difference']:.3f}, "
                        f"FPR diff {eo_results['fpr_difference']:.3f}"
                    )
                    recommendations.append(
                        "Consider post-processing to equalize error rates across groups"
                    )

            # Compute disparate impact
            if FairnessMetric.DISPARATE_IMPACT in self.config.metrics:
                di_results = self.compute_disparate_impact(
                    predictions, column, self.config.reference_group
                )
                metric_scores[_marginal_key("disparate_impact", name)] = di_results["min_ratio"]
                group_scores[_marginal_key("impact_ratios", name)] = di_results["impact_ratios"]

                if not di_results["passes_four_fifths"]:
                    violations.append(
                        f"Disparate impact violation"
                        f"{f' ({name})' if intersectional else ''}: "
                        f"min ratio {di_results['min_ratio']:.3f} < 0.8"
                    )
                    recommendations.append("Review selection criteria for potential discrimination")

            # Compute calibration if labels available
            if labels is not None and FairnessMetric.CALIBRATION in self.config.metrics:
                cal_results = self.compute_calibration(predictions, labels, column)
                metric_scores[_marginal_key("calibration", name)] = cal_results["calibration_score"]
                group_scores[_marginal_key("calibration", name)] = {
                    g: v["ece"] for g, v in cal_results["group_calibration"].items()
                }

        details: dict[str, Any] = {
            "n_samples": len(predictions),
            "n_groups": (
                {name: len(np.unique(column)) for name, column in named_features.items()}
                if intersectional
                else len(np.unique(next(iter(named_features.values()))))
            ),
            "config": {
                "fairness_threshold": self.config.fairness_threshold,
                "max_disparity": self.config.max_disparity,
            },
        }

        # Intersectional (joint-subgroup) metrics across crossed features.
        if intersectional and FairnessMetric.INTERSECTIONAL_PARITY in self.config.metrics:
            ip_results = self.compute_intersectional_parity(predictions, named_features)
            metric_scores["intersectional_parity"] = ip_results["parity_score"]
            group_scores["intersectional_parity"] = ip_results["group_rates"]
            details["intersectional_parity"] = {
                "worst_group": ip_results["worst_group"],
                "small_groups": ip_results["small_groups"],
                "min_group_size": ip_results["min_group_size"],
                "insufficient_data": ip_results["insufficient_data"],
            }
            if ip_results["insufficient_data"]:
                recommendations.append(
                    "Intersectional parity indeterminate: every joint subgroup is below "
                    f"min_group_size={ip_results['min_group_size']}; collect more data or "
                    "lower intersectional_min_group_size deliberately"
                )
            elif ip_results["max_disparity"] > self.config.max_disparity:
                violations.append(
                    "Intersectional parity violation: joint subgroup "
                    f"{ip_results['worst_group']!r} deviates by "
                    f"{ip_results['max_disparity']:.3f}"
                )
                recommendations.append(
                    "Audit the worst-off joint subgroup directly; marginal-only "
                    "mitigation can leave joint subgroups disadvantaged"
                )

        if (
            intersectional
            and labels is not None
            and FairnessMetric.INTERSECTIONAL_EQUALIZED_ODDS in self.config.metrics
        ):
            ieo_results = self.compute_intersectional_equalized_odds(
                predictions, labels, named_features
            )
            metric_scores["intersectional_equalized_odds"] = ieo_results["equalized_odds_score"]
            group_scores["intersectional_tpr"] = ieo_results["group_tpr"]
            group_scores["intersectional_fpr"] = ieo_results["group_fpr"]
            details["intersectional_equalized_odds"] = {
                "worst_tpr_group": ieo_results["worst_tpr_group"],
                "worst_fpr_group": ieo_results["worst_fpr_group"],
                "small_groups": ieo_results["small_groups"],
                "min_group_size": ieo_results["min_group_size"],
                "insufficient_data": ieo_results["insufficient_data"],
            }
            if (
                not ieo_results["insufficient_data"]
                and max(ieo_results["tpr_difference"], ieo_results["fpr_difference"])
                > self.config.max_disparity
            ):
                violations.append(
                    "Intersectional equalized odds violation: joint TPR diff "
                    f"{ieo_results['tpr_difference']:.3f}, joint FPR diff "
                    f"{ieo_results['fpr_difference']:.3f}"
                )
                recommendations.append(
                    "Equalize error rates for the worst-off joint subgroups "
                    "(post-processing per joint cell)"
                )

        # Calculate overall fairness score
        if metric_scores:
            overall_score = float(np.mean(list(metric_scores.values())))
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
            details=details,
        )


class BiasmitigationProcessor:
    """Post-processing bias mitigation.

    Applies threshold optimization and other post-hoc corrections to reduce bias in predictions.
    """

    def __init__(
        self,
        strategy: MitigationStrategy = MitigationStrategy.THRESHOLD_OPTIMIZATION,
        fairness_constraint: FairnessMetric = FairnessMetric.DEMOGRAPHIC_PARITY,
    ):
        """Initialize bias mitigation processor.

        Args:
            strategy: Mitigation strategy to use
            fairness_constraint: Fairness metric to optimize
        """
        self.strategy = strategy
        self.fairness_constraint = fairness_constraint
        self.group_thresholds: dict[str, float] = {}
        # Kamiran–Calders reweighing weights keyed by (group, label):
        # w(g, y) = P(g) * P(y) / P(g, y) — >1 for under-represented
        # (group, label) combinations, <1 for over-represented ones.
        self.sample_weight_map: dict[tuple[str, int], float] = {}

    def fit(
        self,
        predictions: np.ndarray[Any, Any],
        labels: np.ndarray[Any, Any],
        sensitive_features: np.ndarray[Any, Any],
    ) -> BiasmitigationProcessor:
        """Fit the mitigation processor.

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
        predictions: np.ndarray[Any, Any],
        labels: np.ndarray[Any, Any],
        sensitive_features: np.ndarray[Any, Any],
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
        predictions: np.ndarray[Any, Any],
        labels: np.ndarray[Any, Any],
        sensitive_features: np.ndarray[Any, Any],
    ) -> None:
        """Compute Kamiran–Calders reweighing weights.

        For every observed ``(group, label)`` combination the weight is
        ``P(group) * P(label) / P(group, label)`` (expected over observed
        joint frequency), which up-weights under-represented combinations
        so a downstream trainer consuming :meth:`get_sample_weights` sees
        a demographically-balanced effective distribution.  Reweighing is
        a *pre-processing* strategy: it produces training weights and
        deliberately leaves predictions untouched in :meth:`transform`.
        """
        labels_int = (np.asarray(labels) > 0).astype(int)
        groups_str = np.asarray(sensitive_features).astype(str)
        n_total = len(labels_int)
        if n_total == 0:
            raise ValueError("cannot fit reweighting on empty data")

        self.sample_weight_map = {}
        for group in np.unique(groups_str):
            group_mask = groups_str == group
            p_group = float(np.mean(group_mask))
            for label_value in (0, 1):
                label_mask = labels_int == label_value
                p_label = float(np.mean(label_mask))
                joint = float(np.mean(group_mask & label_mask))
                if joint > 0:
                    self.sample_weight_map[(str(group), label_value)] = p_group * p_label / joint

    def get_sample_weights(
        self,
        labels: np.ndarray[Any, Any],
        sensitive_features: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Return per-sample Kamiran–Calders training weights.

        Args:
            labels: True binary labels (0/1).
            sensitive_features: Group membership, aligned with ``labels``.

        Returns:
            Array of per-sample weights; ``(group, label)`` combinations
            unseen at fit time receive a neutral weight of 1.0.

        Raises:
            RuntimeError: If called before :meth:`fit` with the
                REWEIGHTING strategy.
        """
        if not self.sample_weight_map:
            raise RuntimeError(
                "get_sample_weights requires fit() with MitigationStrategy.REWEIGHTING first"
            )
        labels_int = (np.asarray(labels) > 0).astype(int)
        groups_str = np.asarray(sensitive_features).astype(str)
        if len(labels_int) != len(groups_str):
            raise ValueError(
                f"length mismatch: labels={len(labels_int)}, "
                f"sensitive_features={len(groups_str)}"
            )
        weights = np.ones(len(labels_int), dtype=float)
        for i in range(len(labels_int)):
            weights[i] = self.sample_weight_map.get((str(groups_str[i]), int(labels_int[i])), 1.0)
        return weights

    def transform(
        self,
        predictions: np.ndarray[Any, Any],
        sensitive_features: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Apply mitigation to predictions.

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
        predictions: np.ndarray[Any, Any],
        sensitive_features: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Apply group-specific thresholds.

        Groups are matched on their string form because ``fit`` stores
        stringified keys — comparing the raw (e.g. integer-typed) array
        against a string key would match nothing and silently zero every
        prediction.  Samples from groups unseen at fit time fall back to
        the default 0.5 threshold instead of being dropped.
        """
        predictions = np.asarray(predictions, dtype=float)
        groups_str = np.asarray(sensitive_features).astype(str)
        adjusted = (predictions >= 0.5).astype(float)

        unseen = set(np.unique(groups_str)) - set(self.group_thresholds)
        if unseen:
            logger.warning(
                "threshold optimization saw unfitted group(s) %s; applying default 0.5 threshold",
                sorted(unseen),
            )

        for group, threshold in self.group_thresholds.items():
            mask = groups_str == group
            adjusted[mask] = (predictions[mask] >= threshold).astype(float)

        return adjusted


def compute_fairness_score(
    predictions: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any] | None,
    sensitive_features: np.ndarray[Any, Any],
    metric: FairnessMetric = FairnessMetric.DEMOGRAPHIC_PARITY,
) -> float:
    """Quick fairness score computation.

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
