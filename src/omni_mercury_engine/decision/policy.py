# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Configurable, auditable knobs for the abstention gate.

:class:`DecisionPolicy` collects every threshold the gate consults into one
frozen, serialisable object so a deployment's risk posture is explicit and
reviewable rather than scattered through the control flow.  The defaults are
deliberately conservative -- *when in doubt, abstain* -- in keeping with the
fail-closed, Civilization-First posture of the rest of the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

#: Drift severities strong enough to demote a grounded verdict to a deferral.
#: These are the upper-case ``DriftSeverity.name`` values the engine actually
#: emits via ``detect_with_fusion`` -> ``drift_detection["severity"]`` (the enum
#: is ``NONE / LOW / MEDIUM / HIGH / CRITICAL`` in ``ml/drift.py``), so the set
#: must name ``MEDIUM`` (not ``MODERATE``) and must not list a non-existent
#: ``SEVERE`` -- demote on medium drift and above.
_DEFAULT_DRIFT_DEFER_SEVERITIES: frozenset[str] = frozenset({"MEDIUM", "HIGH", "CRITICAL"})


@dataclass(frozen=True)
class DecisionPolicy:
    """Thresholds governing when the loop decides vs. abstains.

    Attributes:
        indecision_margin: Half-width of the band around the threshold used
            **only** when no conformal certificate is available.  A calibrated
            probability within ``threshold ± indecision_margin`` is treated as
            a resolvable "don't-know" (``DEFER``).  Default 0.05.
        symbolic_agreement_floor: Minimum neuro-symbolic constraint
            satisfaction; a grounded verdict whose symbolic ``satisfaction``
            falls below this is demoted to ``DEFER`` (neural and symbolic
            paths disagree).  Default 0.5.
        defer_on_drift: Whether sufficiently severe distribution drift demotes
            a grounded verdict to ``DEFER`` (calibration may no longer hold).
            Default ``True``.
        drift_defer_severities: The drift severities that trigger that demotion.
        defer_on_gosnn_disagreement: Whether strong disagreement from the
            GOSNN fused-state detection head demotes a grounded verdict to
            ``DEFER``.  The head and its validation-selected thresholds ship
            with the fusion checkpoint only when the detection-metric merit
            gate passed, and ride in the evidence
            (``gosnn_anomaly_prob`` / ``gosnn_demote_act_below`` /
            ``gosnn_demote_clear_above``); with no shipped head the overlay
            never fires regardless of this knob.  Abstention-only: it can
            never ground or upgrade a verdict.  Default ``True``.
        require_calibrated_for_act: When ``True``, an ``ACT`` is only permitted
            with a conformal coverage certificate; an uncalibrated positive is
            demoted to ``DEFER``.  Default ``False`` (a thresholded positive
            may act, but the record is flagged ``calibrated=False``).
        fail_closed_on_atypical: When ``True``, an empty conformal set
            (``set_size == 0`` -- a point no class explains) is ``UNDECIDABLE``
            / ``HOLD`` rather than a resolvable deferral.  Default ``True``.
        fail_closed_on_ethical_block: When ``True``, an explicit ethical-gate
            failure (``ethical_gate_passed is False``) forces ``HOLD``.
            Default ``True``.
    """

    indecision_margin: float = 0.05
    symbolic_agreement_floor: float = 0.5
    defer_on_drift: bool = True
    drift_defer_severities: frozenset[str] = field(
        default_factory=lambda: _DEFAULT_DRIFT_DEFER_SEVERITIES
    )
    defer_on_gosnn_disagreement: bool = True
    require_calibrated_for_act: bool = False
    fail_closed_on_atypical: bool = True
    fail_closed_on_ethical_block: bool = True

    def __post_init__(self) -> None:
        """Validate the knobs eagerly so a misconfiguration fails at build."""
        if not 0.0 <= self.indecision_margin < 0.5:
            raise ValueError(f"indecision_margin must be in [0, 0.5), got {self.indecision_margin}")
        if not 0.0 <= self.symbolic_agreement_floor <= 1.0:
            raise ValueError(
                "symbolic_agreement_floor must be in [0, 1], got "
                f"{self.symbolic_agreement_floor}"
            )
        # Normalise severities to upper-case for case-insensitive matching.
        object.__setattr__(
            self,
            "drift_defer_severities",
            frozenset(s.upper() for s in self.drift_defer_severities),
        )

    def drift_is_deferring(self, severity: str | None) -> bool:
        """Whether a drift ``severity`` name is severe enough to defer."""
        if not self.defer_on_drift or severity is None:
            return False
        return severity.upper() in self.drift_defer_severities

    def with_overrides(self, **changes: Any) -> DecisionPolicy:
        """Return a copy with ``changes`` applied (validated)."""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view of the policy (for record provenance)."""
        return {
            "indecision_margin": self.indecision_margin,
            "symbolic_agreement_floor": self.symbolic_agreement_floor,
            "defer_on_drift": self.defer_on_drift,
            "drift_defer_severities": sorted(self.drift_defer_severities),
            "defer_on_gosnn_disagreement": self.defer_on_gosnn_disagreement,
            "require_calibrated_for_act": self.require_calibrated_for_act,
            "fail_closed_on_atypical": self.fail_closed_on_atypical,
            "fail_closed_on_ethical_block": self.fail_closed_on_ethical_block,
        }


__all__ = ["DecisionPolicy"]
