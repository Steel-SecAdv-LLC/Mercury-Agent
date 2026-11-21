# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
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

import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from omni_mercury_engine.cognitive.gate_audit import record_gate_decision


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


#: A reviewer: given the record, return the literal ``True`` to authorize the
#: gray-zone request; any other return value (including truthy non-booleans)
#: and any exception is treated as a denial (fail-closed).
#:
#: "The literal ``True``" means the Python singleton, tested by identity
#: (``verdict is True``), not by truthiness or equality. So ``1``, ``"yes"``,
#: a ``Mock`` and ``numpy.bool_(True)`` are all DENIED: ``numpy.bool_(True)``
#: is truthy and equal to ``True`` and is still a different object. That is
#: intended -- an authorization boundary does not guess what a reviewer meant.
#: A reviewer built on NumPy/pandas comparisons must return ``bool(result)``.
#:
#: A real (implicit) type alias -- not a string -- so consumers that
#: ``from ... import HumanReviewCallback`` get a usable type for annotations
#: and static checking.
#: Bare assignment rather than an explicit ``TypeAlias`` annotation keeps it
#: valid on the project's 3.11 floor (the PEP 695 ``type`` statement ruff's
#: UP040 would suggest is 3.12+ only) while mypy still resolves it as an alias.
HumanReviewCallback = Callable[[EscalationRecord], bool]


class EscalationBroker:
    """Route ESCALATE verdicts to a human reviewer under a bounded-autonomy cap.

    Args:
        reviewer: Optional ``Callable[[EscalationRecord], bool]``. When ``None``
            every escalation is denied (fail-closed) -- an escalation with no
            human in the loop is a refusal, not an allow. Only the Python
            literal ``True`` (by identity) approves; every other return value,
            ``numpy.bool_(True)`` included, is a denial -- see
            :data:`HumanReviewCallback`.
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
        # The bounded-autonomy counter is shared mutable state: concurrent
        # reviews racing an unguarded check-then-increment could approve past
        # the ceiling. Every read/update of ``_approvals`` holds this lock.
        self._approvals_lock = threading.Lock()

    @property
    def approvals_used(self) -> int:
        """How many escalations this broker has approved so far."""
        with self._approvals_lock:
            return self._approvals

    def review(self, record: EscalationRecord) -> EscalationDecision:
        """Adjudicate one escalation, fail-closed, bounded, and durably audited."""
        # ``approvals_used`` comes back from the adjudication itself, read
        # under the approvals lock in the same critical section that made
        # this decision. Reading ``self._approvals`` again here, as this
        # used to, audited whatever count another thread's review had
        # reached in the meantime.
        decision, approvals_used = self._adjudicate(record)
        record_gate_decision(
            decision="approved" if decision.approved else "escalation_denied",
            source="escalation_broker",
            disposition=record.disposition,
            hazard_domain=record.hazard_domain,
            intent=record.intent,
            signals=record.signals,
            reason=decision.reason,
            query=record.query,
            extra={"reviewer": decision.reviewer, "approvals_used": approvals_used},
        )
        return decision

    def _adjudicate(self, record: EscalationRecord) -> tuple[EscalationDecision, int]:
        """Decide one escalation; return it with the approval count at that decision."""
        if self._reviewer is None:
            return (
                EscalationDecision(
                    approved=False,
                    reason="no human reviewer wired; escalation denied (fail-closed)",
                    reviewer="none",
                ),
                self.approvals_used,
            )
        ceiling = EscalationDecision(
            approved=False,
            reason=(
                f"bounded-autonomy ceiling reached ({self._max_approvals} approvals); "
                "further escalations denied this session"
            ),
            reviewer=self._reviewer_name,
        )
        with self._approvals_lock:
            if self._approvals >= self._max_approvals:
                return ceiling, self._approvals
        try:
            # The reviewer runs outside the lock (it may be slow or
            # interactive); the ceiling is re-checked under the lock before
            # the approval is counted, so concurrent reviews can never
            # approve past ``max_approvals``.
            verdict = self._reviewer(record)
        except Exception as exc:
            return (
                EscalationDecision(
                    approved=False,
                    reason=f"reviewer raised ({exc}); denied fail-closed",
                    reviewer=self._reviewer_name,
                ),
                self.approvals_used,
            )
        # A hard authorization boundary: accept only the literal ``True``.
        # Any other return -- a truthy string like "no", a dict, a Mock, a
        # ``numpy.bool_(True)`` -- is a refusal, never coerced via ``bool()``,
        # which would authorize a gray-zone request on garbage (the same
        # identity guard the evidence extractor applies to
        # ``ethical_gate_passed``).
        if verdict is True:
            with self._approvals_lock:
                if self._approvals >= self._max_approvals:
                    return ceiling, self._approvals
                self._approvals += 1
                approvals_used = self._approvals
            return (
                EscalationDecision(
                    approved=True,
                    reason="authorized by human-in-the-loop reviewer",
                    reviewer=self._reviewer_name,
                ),
                approvals_used,
            )
        return (
            EscalationDecision(
                approved=False, reason="reviewer declined", reviewer=self._reviewer_name
            ),
            self.approvals_used,
        )


__all__ = [
    "EscalationBroker",
    "EscalationDecision",
    "EscalationRecord",
    "HumanReviewCallback",
]
