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
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from omni_mercury_engine.verifiers.three_state import ThreeState

if TYPE_CHECKING:
    from collections.abc import Mapping

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
    signals: Mapping[str, Any] = field(default_factory=dict)
    domain: str | None = None

    def __post_init__(self) -> None:
        """Wrap the provenance mapping in a *shallow* read-only view.

        The dataclass is ``frozen``, but a plain ``dict`` ``signals`` field would
        still let a caller add, remove or rebind top-level keys on
        ``record.signals`` after the record is in the ledger.  Wrapping it in a
        :class:`~types.MappingProxyType` over a private copy blocks that.  The
        freeze is deliberately **shallow**: the values stay the plain JSON-safe
        scalars / lists / dicts the record was built with -- so :meth:`to_dict`
        remains JSON-serialisable -- which means a deeply-nested value is not
        itself frozen.  The guarantee is therefore "the top-level provenance keys
        cannot be added, dropped or rebound", not a deep freeze.
        """
        object.__setattr__(self, "signals", MappingProxyType(dict(self.signals)))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DecisionRecord:
        """Reconstruct a record from its :meth:`to_dict` form (the exact inverse).

        Rebuilds the enums and the nested :class:`ResponsePlan` from their wire
        values so a serialised audit trail can be reloaded into live records --
        the deserialisation half of the persistable ledger.  ``from_dict`` is a
        round-trip inverse of :meth:`to_dict`: ``from_dict(r.to_dict()).to_dict()
        == r.to_dict()`` for any record ``r``.

        Args:
            data: A mapping in :meth:`to_dict` shape.

        Returns:
            The reconstructed :class:`DecisionRecord`.
        """
        from omni_mercury_engine.decision.response import ResponsePlan
        from omni_mercury_engine.decision.states import Disposition

        return cls(
            state=ThreeState(data["state"]),
            disposition=Disposition(data["disposition"]),
            decision_label=data["decision_label"],
            abstained=bool(data["abstained"]),
            anomaly_prob=float(data["anomaly_prob"]),
            threshold=float(data["threshold"]),
            decision_confidence=data["decision_confidence"],
            coverage=data["coverage"],
            calibrated=bool(data["calibrated"]),
            severity=float(data["severity"]),
            response=ResponsePlan.from_dict(data["response"]),
            reasons=tuple(data.get("reasons", ())),
            caveats=tuple(data.get("caveats", ())),
            signals=dict(data.get("signals", {})),
            domain=data.get("domain"),
        )

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
