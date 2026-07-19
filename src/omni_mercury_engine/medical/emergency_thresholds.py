# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Validate (and advisory-tune) medical emergency-routing thresholds.

Mercury's emergency routing fires on literature-anchored cutoffs -- NIHSS-derived
``stroke_risk >= 0.6`` (NIHSS >= 5, moderate stroke), troponin I ``> 0.4 ng/mL``
(acute MI), and the NEWS2 aggregate ``>= 7`` (high-risk deterioration) -- but
those cutoffs were never *validated against outcomes* or swept. This module
measures each threshold's operating characteristics (sensitivity, specificity,
PPV, NPV, Youden's J, F2) on a seeded outcome cohort, sweeps the full grid, and
reports where an outcome-optimal operating point would sit.

Design stance (identical to the σ_Immutable calibration harness): this is
**measurement + advisory**. The operational cutoffs are NOT repointed here.
Tuning a literature-anchored clinical threshold against a *synthetic* cohort
would be less safe, not more -- a real change requires governed outcome data
(registry / EHR) and clinical validation. The harness exists so that, when such
data is available, the operating point is chosen by measured Fβ / Youden rather
than by eyeballing, and so the current cutoffs' sensitivity/specificity are on
the record today.

For emergencies the cost of a miss dominates, so the default recommendation
criterion is **max F2** (recall weighted 2x precision), reported alongside the
Youden-J optimum and the sensitivity-floor operating point.

Discrimination/confusion metrics reuse the D1 clinical metric engine
(:mod:`omni_mercury_engine.medical.clinical_metrics`), pure numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omni_mercury_engine.medical.clinical_metrics import confusion_at_threshold

__all__ = [
    "EmergencyThresholdReport",
    "ThresholdOperatingPoint",
    "evaluate_threshold",
    "news2_score",
    "recommend_threshold",
    "sweep_threshold",
    "validate_threshold",
]

# NIHSS total -> stroke_risk step function. Mirrors the production mapping in
# ``critical_care.neurocritical_care.NeurocriticalCarePredictor._interpret_nihss``
# (pinned identical by test_emergency_thresholds.test_nihss_mapping_matches_production).
_NIHSS_RISK_BANDS: tuple[tuple[int, float], ...] = (
    (0, 0.0),  # 0: no symptoms
    (4, 0.3),  # 1-4: minor
    (15, 0.6),  # 5-15: moderate (emergency threshold anchor)
    (20, 0.8),  # 16-20: moderate-severe
    (42, 1.0),  # 21-42: severe
)


def nihss_stroke_risk(nihss_total: int) -> float:
    """Map an NIHSS total (0-42) to the production stroke-risk band."""
    for upper, risk in _NIHSS_RISK_BANDS:
        if nihss_total <= upper:
            return risk
    return 1.0


def news2_score(vitals: dict[str, Any]) -> int:
    """Return the NEWS2 aggregate score from routine vitals (public rubric).

    Implements the Royal College of Physicians NEWS2 aggregate (2017): respiratory
    rate, SpO2 (Scale 1), any supplemental O2, temperature, systolic BP, heart
    rate, and level of consciousness (ACVPU). A missing parameter contributes 0
    (it is treated as unassessed, never as deranged) and is the caller's
    responsibility to supply for a valid total.

    Args:
        vitals: Mapping possibly containing ``respiratory_rate``, ``spo2``,
            ``on_oxygen`` (bool), ``temperature_c``, ``systolic_bp``,
            ``heart_rate``, ``consciousness`` ("A" alert, else scores 3).

    Returns:
        The integer NEWS2 aggregate score.
    """

    def band(value: float | None, edges: list[tuple[float, float, int]]) -> int:
        if value is None:
            return 0
        for lo, hi, pts in edges:
            if lo <= value <= hi:
                return pts
        return 0

    total = 0
    total += band(
        vitals.get("respiratory_rate"),
        [(0, 8, 3), (9, 11, 1), (12, 20, 0), (21, 24, 2), (25, 1e9, 3)],
    )
    total += band(
        vitals.get("spo2"),
        [(0, 91, 3), (92, 93, 2), (94, 95, 1), (96, 100, 0)],
    )
    total += 2 if vitals.get("on_oxygen") else 0
    total += band(
        vitals.get("temperature_c"),
        [(0, 35.0, 3), (35.1, 36.0, 1), (36.1, 38.0, 0), (38.1, 39.0, 1), (39.1, 1e9, 2)],
    )
    total += band(
        vitals.get("systolic_bp"),
        [(0, 90, 3), (91, 100, 2), (101, 110, 1), (111, 219, 0), (220, 1e9, 3)],
    )
    total += band(
        vitals.get("heart_rate"),
        [(0, 40, 3), (41, 50, 1), (51, 90, 0), (91, 110, 1), (111, 130, 2), (131, 1e9, 3)],
    )
    consciousness = vitals.get("consciousness", "A")
    total += 0 if consciousness == "A" else 3
    return total


@dataclass
class ThresholdOperatingPoint:
    """Operating characteristics of one decision threshold.

    Attributes:
        threshold: The decision threshold (score ``>= threshold`` routes to emergency).
        n: Number of cases.
        n_positive: Number of true emergency-outcome cases.
        sensitivity: Recall / true-positive rate.
        specificity: True-negative rate.
        ppv: Positive predictive value.
        npv: Negative predictive value.
        youden_j: ``sensitivity + specificity - 1``.
        f2: F-beta with beta=2 (recall weighted 2x precision).
        accuracy: Overall accuracy.
    """

    threshold: float
    n: int
    n_positive: int
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    youden_j: float
    f2: float
    accuracy: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping of the operating point."""
        return {
            "threshold": self.threshold,
            "n": self.n,
            "n_positive": self.n_positive,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "ppv": self.ppv,
            "npv": self.npv,
            "youden_j": self.youden_j,
            "f2": self.f2,
            "accuracy": self.accuracy,
        }


def evaluate_threshold(scores: Any, outcomes: Any, threshold: float) -> ThresholdOperatingPoint:
    """Compute the operating characteristics of ``threshold`` on labelled data.

    Args:
        scores: Instrument scores (higher = more severe).
        outcomes: Binary emergency outcomes (1 = emergency intervention warranted).
        threshold: Decision threshold; ``score >= threshold`` routes to emergency.

    Returns:
        A :class:`ThresholdOperatingPoint`.
    """
    tp, fp, tn, fn = confusion_at_threshold(outcomes, scores, threshold)
    sens = tp / (tp + fn) if (tp + fn) else 1.0
    spec = tn / (tn + fp) if (tn + fp) else 1.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    npv_v = tn / (tn + fn) if (tn + fn) else float("nan")
    f2 = (5 * prec * sens / (4 * prec + sens)) if (4 * prec + sens) > 0 else 0.0
    n = tp + fp + tn + fn
    return ThresholdOperatingPoint(
        threshold=float(threshold),
        n=int(n),
        n_positive=int(tp + fn),
        sensitivity=float(sens),
        specificity=float(spec),
        ppv=float(prec),
        npv=float(npv_v),
        youden_j=float(sens + spec - 1.0),
        f2=float(f2),
        accuracy=float((tp + tn) / n) if n else 0.0,
    )


def sweep_threshold(scores: Any, outcomes: Any, grid: list[float]) -> list[ThresholdOperatingPoint]:
    """Evaluate every threshold in ``grid`` and return the operating points."""
    return [evaluate_threshold(scores, outcomes, t) for t in grid]


def recommend_threshold(
    points: list[ThresholdOperatingPoint],
    *,
    criterion: str = "f2",
    min_sensitivity: float = 0.9,
) -> ThresholdOperatingPoint:
    """Pick the outcome-optimal operating point by a documented criterion.

    Args:
        points: Swept operating points.
        criterion: ``"f2"`` (max F2 — recall-weighted, the emergency default),
            ``"youden"`` (max Youden's J), or ``"sensitivity_floor"`` (highest
            specificity among points meeting ``min_sensitivity``).
        min_sensitivity: Sensitivity floor for the ``sensitivity_floor`` criterion.

    Returns:
        The chosen :class:`ThresholdOperatingPoint`.
    """
    if not points:
        raise ValueError("no operating points to choose from")
    if criterion == "youden":
        return max(points, key=lambda p: p.youden_j)
    if criterion == "sensitivity_floor":
        eligible = [p for p in points if p.sensitivity >= min_sensitivity]
        pool = eligible or points
        return max(pool, key=lambda p: p.specificity)
    if criterion == "f2":
        return max(points, key=lambda p: p.f2)
    raise ValueError(f"unknown criterion {criterion!r}")


@dataclass
class EmergencyThresholdReport:
    """Validation of one emergency threshold: current point + sweep + advisory.

    Attributes:
        instrument: Instrument name (e.g. "NIHSS_stroke_risk").
        literature_anchor: The literature-anchored current threshold + reference.
        current_threshold: The operational threshold in force.
        current: Operating characteristics at ``current_threshold``.
        sweep: Operating points across the threshold grid.
        recommended_f2: Advisory F2-optimal operating point.
        recommended_youden: Advisory Youden-J optimal operating point.
        dgp_doc: The outcome data-generating process (reproducibility).
    """

    instrument: str
    literature_anchor: str
    current_threshold: float
    current: ThresholdOperatingPoint
    sweep: list[ThresholdOperatingPoint]
    recommended_f2: ThresholdOperatingPoint
    recommended_youden: ThresholdOperatingPoint
    dgp_doc: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping of the validation report."""
        return {
            "instrument": self.instrument,
            "literature_anchor": self.literature_anchor,
            "current_threshold": self.current_threshold,
            "current": self.current.to_dict(),
            "sweep": [p.to_dict() for p in self.sweep],
            "recommended_f2": self.recommended_f2.to_dict(),
            "recommended_youden": self.recommended_youden.to_dict(),
            "dgp_doc": self.dgp_doc,
            "advisory_note": (
                "MEASUREMENT + ADVISORY ONLY. The operational threshold is NOT "
                "changed here; tuning a literature-anchored clinical cutoff "
                "requires governed real-outcome data and clinical validation. "
                "Metrics are computed under the stated synthetic DGP."
            ),
        }


def validate_threshold(
    *,
    instrument: str,
    literature_anchor: str,
    scores: Any,
    outcomes: Any,
    current_threshold: float,
    grid: list[float],
    dgp_doc: str = "",
) -> EmergencyThresholdReport:
    """Validate one emergency threshold and produce advisory optima."""
    sweep = sweep_threshold(scores, outcomes, grid)
    return EmergencyThresholdReport(
        instrument=instrument,
        literature_anchor=literature_anchor,
        current_threshold=float(current_threshold),
        current=evaluate_threshold(scores, outcomes, current_threshold),
        sweep=sweep,
        recommended_f2=recommend_threshold(sweep, criterion="f2"),
        recommended_youden=recommend_threshold(sweep, criterion="youden"),
        dgp_doc=dgp_doc,
    )
