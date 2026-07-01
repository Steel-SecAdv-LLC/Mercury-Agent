# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Human-in-the-loop / bounded-autonomy escalation for gate ESCALATE verdicts.

An ``ESCALATE`` disposition marks a genuine gray-zone request that a *human*
could authorize -- a licensed engineer's production-adjacent query, or an
accretion pattern worth a second look. Before this module, ESCALATE was a
refusal-with-a-note: no reviewer was ever consulted and nothing was recorded
durably. :class:`EscalationBroker` makes it a real control:

* an injectable :data:`HumanReviewCallback` is consulted when wired (a reviewer
  queue, an approval webhook, a SOAR action, an interactive prompt);
* **fail-closed** -- with no reviewer, or on a reviewer error, the escalation is
  *denied* (the gray-zone request is refused), never silently allowed;
* **bounded autonomy** -- at most ``max_approvals`` escalations may be approved
  per session; beyond the ceiling every escalation is denied regardless of the
  reviewer, so a compromised/looping reviewer cannot rubber-stamp without bound;
* every decision (approve/deny) is written to the durable gate audit log.

The broker is deliberately tiny and dependency-light so it can sit on the hot
enforcement boundary (:class:`GeneralAssistant`) without pulling heavy imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from omni_mercury_engine.cognitive.gate_audit import record_gate_decision

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class EscalationRecord:
    """Structured description of a gray-zone request handed to a human reviewer."""

    query: str
    reason: str
    disposition: str = "escalate"
    hazard_domain: str = "none"
    intent: str = "mechanism"
    signals: tuple[str, ...] = ()
    context: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EscalationDecision:
    """Outcome of an escalation review."""

    approved: bool
    reason: str = ""
    reviewer: str = "none"


#: A reviewer: given the record, return True to authorize the gray-zone request.
#: Any exception is treated as a denial (fail-closed).
HumanReviewCallback = "Callable[[EscalationRecord], bool]"


class EscalationBroker:
    """Route ESCALATE verdicts to a human reviewer under a bounded-autonomy cap.

    Args:
        reviewer: Optional ``Callable[[EscalationRecord], bool]``. When ``None``
            every escalation is denied (fail-closed) -- an escalation with no
            human in the loop is a refusal, not an allow.
        max_approvals: Bounded-autonomy ceiling: the maximum number of
            escalations that may be *approved* within this broker's lifetime
            (per session). Beyond it, escalations are denied regardless of the
            reviewer. ``0`` disables approvals entirely (always deny).
        reviewer_name: Provenance label recorded in the audit log.
    """

    def __init__(
        self,
        reviewer: Callable[[EscalationRecord], bool] | None = None,
        *,
        max_approvals: int = 3,
        reviewer_name: str = "wired_reviewer",
    ) -> None:
        """Initialize the broker with an optional reviewer and an approval cap."""
        self._reviewer = reviewer
        self._max_approvals = max(0, int(max_approvals))
        self._reviewer_name = reviewer_name
        self._approvals = 0

    @property
    def approvals_used(self) -> int:
        """How many escalations this broker has approved so far."""
        return self._approvals

    def review(self, record: EscalationRecord) -> EscalationDecision:
        """Adjudicate one escalation, fail-closed, bounded, and durably audited."""
        decision = self._adjudicate(record)
        record_gate_decision(
            decision="approved" if decision.approved else "escalation_denied",
            source="escalation_broker",
            disposition=record.disposition,
            hazard_domain=record.hazard_domain,
            intent=record.intent,
            signals=record.signals,
            reason=decision.reason,
            query=record.query,
            extra={"reviewer": decision.reviewer, "approvals_used": self._approvals},
        )
        return decision

    def _adjudicate(self, record: EscalationRecord) -> EscalationDecision:
        if self._reviewer is None:
            return EscalationDecision(
                approved=False,
                reason="no human reviewer wired; escalation denied (fail-closed)",
                reviewer="none",
            )
        if self._approvals >= self._max_approvals:
            return EscalationDecision(
                approved=False,
                reason=(
                    f"bounded-autonomy ceiling reached ({self._max_approvals} approvals); "
                    "further escalations denied this session"
                ),
                reviewer=self._reviewer_name,
            )
        try:
            approved = bool(self._reviewer(record))
        except Exception as exc:
            return EscalationDecision(
                approved=False,
                reason=f"reviewer raised ({exc}); denied fail-closed",
                reviewer=self._reviewer_name,
            )
        if approved:
            self._approvals += 1
            return EscalationDecision(
                approved=True,
                reason="authorized by human-in-the-loop reviewer",
                reviewer=self._reviewer_name,
            )
        return EscalationDecision(
            approved=False, reason="reviewer declined", reviewer=self._reviewer_name
        )


__all__ = [
    "EscalationBroker",
    "EscalationDecision",
    "EscalationRecord",
    "HumanReviewCallback",
]
