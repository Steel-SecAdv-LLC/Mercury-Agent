# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Metric-based signal gate for clinical scores (does the score prove signal?).

A medical ``risk_score`` is only worth surfacing if it *discriminates* outcomes
better than chance with statistical margin, and its magnitude is *reliable*
enough to reason about. This gate turns a :class:`ClinicalMetricReport` into a
hard, auditable verdict -- the metric-based gating that is the core Phase-2
objective.

The gate is deliberately **fail-closed**, matching the medical subsystem's
honesty contract: an untrained network whose AUROC confidence interval straddles
0.5, or a score whose calibration error is large, is reported ``proven == False``
with the exact failing criteria, rather than being trusted by default. It never
*raises* harm or overrides the deterministic clinical instruments; it only
certifies whether a learned/derived score has earned the right to be presented
as signal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from omni_mercury_engine.medical.clinical_metrics import ClinicalMetricReport

__all__ = [
    "ClinicalSignalGate",
    "SignalCriteria",
    "SignalVerdict",
]


@dataclass(frozen=True)
class SignalCriteria:
    """Thresholds a clinical score must clear to be certified as signal.

    Attributes:
        min_auroc: Minimum point AUROC.
        auroc_ci_floor: The AUROC bootstrap *lower* bound must exceed this
            (default 0.5 -> discrimination is statistically above chance).
        max_ece: Maximum tolerated Expected Calibration Error.
        min_sensitivity: Optional sensitivity floor -- set for emergency scores
            where missing a positive is the costly error. ``None`` disables it.
        min_specificity: Optional specificity floor. ``None`` disables it.
        min_n: Minimum number of scored cases for the verdict to be trusted.
        min_positives: Minimum number of positive cases (else AUROC is fragile).
    """

    min_auroc: float = 0.65
    auroc_ci_floor: float = 0.5
    max_ece: float = 0.10
    min_sensitivity: float | None = None
    min_specificity: float | None = None
    min_n: int = 50
    min_positives: int = 10


@dataclass
class SignalVerdict:
    """Outcome of applying a :class:`SignalCriteria` to a metric report.

    Attributes:
        proven: Whether the score cleared every applicable criterion.
        reasons: Human-readable pass/fail lines, one per checked criterion.
        failures: The subset of ``reasons`` that failed (empty iff ``proven``).
        criteria: The criteria that were applied.
        metrics: The report the verdict was computed from.
    """

    proven: bool
    reasons: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    criteria: SignalCriteria | None = None
    metrics: ClinicalMetricReport | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping of the verdict."""
        return {
            "proven": self.proven,
            "reasons": list(self.reasons),
            "failures": list(self.failures),
            "criteria": asdict(self.criteria) if self.criteria else None,
            "metrics": self.metrics.to_dict() if self.metrics else None,
        }


class ClinicalSignalGate:
    """Certify whether a clinical score has proven signal under set criteria."""

    def __init__(self, criteria: SignalCriteria | None = None) -> None:
        """Initialize the gate with default or supplied criteria."""
        self.criteria = criteria or SignalCriteria()

    def evaluate(
        self, report: ClinicalMetricReport, criteria: SignalCriteria | None = None
    ) -> SignalVerdict:
        """Apply the criteria to a metric report and return a verdict.

        Args:
            report: The clinical metric report to judge.
            criteria: Override criteria; falls back to the gate's default.

        Returns:
            A :class:`SignalVerdict` with per-criterion pass/fail reasoning.
        """
        crit = criteria or self.criteria
        reasons: list[str] = []
        failures: list[str] = []

        def check(ok: bool, label: str) -> None:
            line = f"{'PASS' if ok else 'FAIL'}: {label}"
            reasons.append(line)
            if not ok:
                failures.append(line)

        check(
            report.n >= crit.min_n,
            f"n={report.n} >= min_n={crit.min_n}",
        )
        check(
            report.n_positive >= crit.min_positives,
            f"positives={report.n_positive} >= min_positives={crit.min_positives}",
        )
        check(
            report.auroc >= crit.min_auroc,
            f"auroc={report.auroc:.3f} >= min_auroc={crit.min_auroc}",
        )
        check(
            report.auroc_ci_low > crit.auroc_ci_floor,
            f"auroc_ci_low={report.auroc_ci_low:.3f} > floor={crit.auroc_ci_floor}",
        )
        check(
            report.ece <= crit.max_ece,
            f"ece={report.ece:.3f} <= max_ece={crit.max_ece}",
        )
        if crit.min_sensitivity is not None:
            check(
                report.sensitivity >= crit.min_sensitivity,
                f"sensitivity={report.sensitivity:.3f} >= min={crit.min_sensitivity}",
            )
        if crit.min_specificity is not None:
            check(
                report.specificity >= crit.min_specificity,
                f"specificity={report.specificity:.3f} >= min={crit.min_specificity}",
            )

        return SignalVerdict(
            proven=len(failures) == 0,
            reasons=reasons,
            failures=failures,
            criteria=crit,
            metrics=report,
        )
