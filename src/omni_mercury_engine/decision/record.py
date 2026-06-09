# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The auditable artifact the closed loop emits for every event.

A :class:`DecisionRecord` is the loop's "measurement, or honest abstention" --
the operational sibling of the governance layer's ``GovernanceScalar``.  It
carries the grounded label *or* an explicit abstention, the calibrated
confidence (only when a coverage certificate backs it), the bounded response,
and a human-readable trail of *why*.  It is a frozen dataclass with a
JSON-safe :meth:`to_dict` and a narrated :meth:`explain`, so the same record
serves an API payload, an audit log, and a one-line operator explanation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from omni_mercury_engine.verifiers.three_state import ThreeState

if TYPE_CHECKING:
    from omni_mercury_engine.decision.response import ResponsePlan
    from omni_mercury_engine.decision.states import Disposition


@dataclass(frozen=True)
class DecisionRecord:
    """A single closed-loop verdict: a grounded call or an honest abstention.

    Attributes:
        state: The honesty verdict -- ``GROUNDED`` / ``UNAVAILABLE`` /
            ``UNDECIDABLE`` (the engine-wide three-state invariant).
        disposition: The operational stance (act / clear / defer / hold).
        decision_label: The grounded label (``1`` anomaly / ``0`` normal) when
            ``state is GROUNDED``; ``None`` when abstaining (no label decided).
        abstained: ``True`` iff the loop did not produce a grounded label.
        anomaly_prob: The calibrated ``P(anomaly)`` the verdict rests on.
        threshold: The decision threshold in play.
        decision_confidence: Confidence in the grounded label -- the
            distribution-free ``coverage`` when calibrated, a margin heuristic
            when not, and ``None`` when abstaining (honest: no grounded label
            to be confident about).
        coverage: The conformal coverage level, when a certificate was present.
        calibrated: Whether a coverage certificate backed this decision.
        severity: Event severity in ``[0, 1]``.
        response: The bounded, non-destructive :class:`ResponsePlan`.
        reasons: Ordered, human-readable drivers of the verdict.
        caveats: Non-blocking honesty notes (e.g. ethical gate not run).
        signals: Provenance -- the normalised evidence the verdict used.
        domain: Optional domain hint.
    """

    state: ThreeState
    disposition: Disposition
    decision_label: int | None
    abstained: bool
    anomaly_prob: float
    threshold: float
    decision_confidence: float | None
    coverage: float | None
    calibrated: bool
    severity: float
    response: ResponsePlan
    reasons: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()
    signals: dict[str, Any] = field(default_factory=dict)
    domain: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view of the full decision record."""
        return {
            "state": self.state.value,
            "disposition": self.disposition.value,
            "decision_label": self.decision_label,
            "abstained": self.abstained,
            "anomaly_prob": self.anomaly_prob,
            "threshold": self.threshold,
            "decision_confidence": self.decision_confidence,
            "coverage": self.coverage,
            "calibrated": self.calibrated,
            "severity": self.severity,
            "response": self.response.to_dict(),
            "reasons": list(self.reasons),
            "caveats": list(self.caveats),
            "signals": dict(self.signals),
            "domain": self.domain,
        }

    def explain(self) -> str:
        """Return a one-paragraph, operator-readable account of the decision."""
        dom = self.domain or "general"
        if self.state is ThreeState.GROUNDED:
            label = "anomaly" if self.decision_label == 1 else "normal"
            conf = (
                f"{self.decision_confidence:.0%} "
                + ("coverage-certified" if self.calibrated else "margin (uncalibrated)")
                if self.decision_confidence is not None
                else "unspecified"
            )
            head = (
                f"GROUNDED [{dom}]: decided {label} "
                f"(p={self.anomaly_prob:.3f}, threshold={self.threshold:.3f}, "
                f"confidence={conf})."
            )
        elif self.state is ThreeState.UNAVAILABLE:
            head = (
                f"ABSTAIN/UNAVAILABLE [{dom}]: a resolvable don't-know "
                f"(p={self.anomaly_prob:.3f}, threshold={self.threshold:.3f}); "
                "deferring rather than guessing."
            )
        else:  # UNDECIDABLE
            head = (
                f"ABSTAIN/UNDECIDABLE [{dom}]: fail-closed don't-know "
                f"(p={self.anomaly_prob:.3f}); the input is outside certified "
                "scope, so the loop holds."
            )
        why = (" Drivers: " + "; ".join(self.reasons) + ".") if self.reasons else ""
        caveat = (" Caveats: " + "; ".join(self.caveats) + ".") if self.caveats else ""
        resp = (
            f" Response: {self.response.action.value} "
            f"(urgency={self.response.urgency}, "
            f"human={'yes' if self.response.requires_human else 'no'}, "
            f"fail_closed={'yes' if self.response.fail_closed else 'no'})."
        )
        return head + why + caveat + resp


__all__ = ["DecisionRecord"]
