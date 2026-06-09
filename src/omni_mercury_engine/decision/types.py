"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

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

Typed decisions and responses for the Decision / Abstention / Response layer.

This module defines the value objects that close the
``identify -> interpret -> decide -> deter -> verify`` loop:

* :class:`Verdict` -- a three-way detection outcome whose third value,
  :attr:`Verdict.ABSTAIN`, makes "don't-know" a *first-class* answer rather
  than an implicit mid-range of a binary threshold.
* :class:`Decision` -- a verdict bound to a :class:`ThreeState` honesty state,
  the calibrated confidence that drove it, and full provenance.
* :class:`ResponseTier` / :class:`ResponseAction` / :class:`ResponseOutcome`
  -- a *graded, reversible-by-default* response, the proportionate "deter"
  step, plus the record of whether it was applied, deferred, or blocked.
* :class:`LoopResult` -- the single auditable record of one full loop pass.

Every object here is frozen and JSON-serialisable: a decision and its response
are a *verifiable certificate*, not hidden state. The mapping from a detection
abstention to :attr:`ThreeState.UNAVAILABLE` (never ``UNDECIDABLE``) is
deliberate and enforced in :mod:`omni_mercury_engine.decision.abstention`: an
anomaly call is always decidable in principle (more data, a larger calibration
set, or a tighter coverage target could settle it), so the honest state is
"decidable, not produced this run", which is exactly ``UNAVAILABLE``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from omni_mercury_engine.verifiers.three_state import ThreeState

if TYPE_CHECKING:
    from omni_mercury_engine.core.types import ThreatLevel

__all__ = [
    "Decision",
    "LoopResult",
    "ResponseAction",
    "ResponseOutcome",
    "ResponseStatus",
    "ResponseTier",
    "Verdict",
    "verdict_to_three_state",
]


class Verdict(Enum):
    """A three-way detection outcome with an explicit honest abstention.

    Unlike a bare ``anomaly_prob > threshold`` (two outcomes, no deferral),
    this admits :attr:`ABSTAIN` so the layer can decline to commit when the
    calibrated evidence is genuinely ambiguous.
    """

    #: Confident anomaly call -- grounded positive.
    POSITIVE = "positive"

    #: Confident normal call -- grounded negative.
    NEGATIVE = "negative"

    #: Honest "don't-know": the calibrated evidence does not commit either way
    #: at the configured operating point. Maps to :attr:`ThreeState.UNAVAILABLE`.
    ABSTAIN = "abstain"


def verdict_to_three_state(verdict: Verdict) -> ThreeState:
    """Map a :class:`Verdict` onto the cross-repo honesty invariant.

    ``POSITIVE`` / ``NEGATIVE`` are :attr:`ThreeState.GROUNDED` (a decision was
    reached and the value carries it). ``ABSTAIN`` is
    :attr:`ThreeState.UNAVAILABLE` -- decidable in principle, simply not
    committed this run. A detection outcome is *never*
    :attr:`ThreeState.UNDECIDABLE`: that state is reserved by the contract for
    claims with no decision procedure in principle, which an anomaly call is not.
    """
    if verdict is Verdict.ABSTAIN:
        return ThreeState.UNAVAILABLE
    return ThreeState.GROUNDED


@dataclass(frozen=True)
class Decision:
    """A typed detection decision: a verdict, an honesty state, and provenance.

    Attributes:
        verdict: The three-way :class:`Verdict`.
        state: The :class:`ThreeState` the verdict maps to (the honesty contract).
        confidence: Calibrated ``P(anomaly)`` that drove the decision, in ``[0, 1]``.
        margin: Distance from the abstention region in ``[0, 1]`` -- how decisively
            the evidence committed (``0.0`` exactly on a boundary). Interpretable,
            never a second hidden threshold.
        prediction_set: The conformal label set over ``{0, 1}`` when one was
            available (e.g. ``(1,)``, ``(0, 1)``, ``()``), else ``None``.
        coverage: Target conformal coverage when a prediction set drove the call.
        novelty: ``True`` when an empty conformal set flagged an atypical point
            that neither class explains (out-of-distribution / novel).
        reason: Human-readable provenance for the verdict.
        policy: Identifier + version of the policy that produced the decision.
        provenance: JSON-friendly structured context (inputs, thresholds, source).
        timestamp: Unix epoch seconds at construction.
    """

    verdict: Verdict
    state: ThreeState
    confidence: float
    margin: float
    reason: str
    policy: str
    prediction_set: tuple[int, ...] | None = None
    coverage: float | None = None
    novelty: bool = False
    provenance: dict[str, object] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def abstained(self) -> bool:
        """Whether the layer declined to commit (the don't-know gate fired)."""
        return self.verdict is Verdict.ABSTAIN

    @property
    def is_grounded(self) -> bool:
        """Whether a decision was actually reached (GROUNDED, not an abstention)."""
        return self.state is ThreeState.GROUNDED

    def as_metadata(self) -> dict[str, object]:
        """Return a JSON-friendly mapping describing this decision."""
        return {
            "verdict": self.verdict.value,
            "state": self.state.value,
            "confidence": self.confidence,
            "margin": self.margin,
            "reason": self.reason,
            "policy": self.policy,
            "prediction_set": (
                list(self.prediction_set) if self.prediction_set is not None else None
            ),
            "coverage": self.coverage,
            "novelty": self.novelty,
            **self.provenance,
        }


class ResponseTier(Enum):
    """Graded, proportionate response tiers in ascending assertiveness.

    The catalogue is **defensive and reversible-by-default**: every built-in
    action either gathers information, informs a human, or applies a
    *reversible* mitigation. Anything irreversible or escalatory is expressed as
    :attr:`ESCALATE` and carries ``requires_human_authorization=True`` so a
    human, not the loop, commits the hard action.
    """

    #: Nothing to do -- a grounded-negative call on a healthy stream.
    NONE = "none"

    #: Passive watch: keep observing, no external effect.
    MONITOR = "monitor"

    #: The honest response to an abstention -- actively reduce uncertainty
    #: (request more data, widen the calibration set, tighten coverage). Never
    #: a deterrent; an abstention must not actuate against the environment.
    GATHER_EVIDENCE = "gather_evidence"

    #: Inform a human / operator. Reversible, no environmental actuation.
    NOTIFY = "notify"

    #: Apply a reversible mitigation (quarantine-for-review, throttle, shadow).
    SOFT_CONTAIN = "soft_contain"

    #: Hand a hard or irreversible action to a human authority for sign-off.
    #: The loop never executes this tier autonomously.
    ESCALATE = "escalate"


class ResponseStatus(Enum):
    """The disposition of a planned response after the loop tried to act on it."""

    #: No action was warranted (e.g. grounded-negative).
    NOOP = "noop"

    #: Applied autonomously (reversible, ethically gated, grounded).
    APPLIED = "applied"

    #: Deferred to a human (abstention, or an action needing authorization).
    DEFERRED = "deferred"

    #: Blocked by the fail-closed ethical gate before any actuation.
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ResponseAction:
    """A single proportionate response selected for a decision.

    Attributes:
        name: Stable action identifier (e.g. ``"soft_contain.quarantine_review"``).
        tier: The :class:`ResponseTier` this action belongs to.
        severity: Proportionality anchor reused from :class:`ThreatLevel`.
        reversible: Whether the action can be fully undone. Built-ins are always
            ``True``; deployer-supplied irreversible actions must say so.
        requires_human_authorization: Whether a human must sign off before the
            action may actuate. ``True`` for every :attr:`ResponseTier.ESCALATE`.
        rationale: Why this action is proportionate to the decision.
        params: JSON-friendly action parameters (handler-specific).
    """

    name: str
    tier: ResponseTier
    severity: ThreatLevel
    reversible: bool
    requires_human_authorization: bool
    rationale: str
    params: dict[str, object] = field(default_factory=dict)

    def as_metadata(self) -> dict[str, object]:
        """Return a JSON-friendly mapping describing this action."""
        return {
            "name": self.name,
            "tier": self.tier.value,
            "severity": self.severity.name,
            "reversible": self.reversible,
            "requires_human_authorization": self.requires_human_authorization,
            "rationale": self.rationale,
            "params": dict(self.params),
        }


@dataclass(frozen=True)
class ResponseOutcome:
    """The record of attempting one :class:`ResponseAction`.

    Attributes:
        action: The action that was planned.
        status: Its :class:`ResponseStatus` after the loop acted.
        reason: Human-readable explanation of the disposition.
        ethical_gate_passed: ``True``/``False`` when the gate ran, else ``None``
            (e.g. a NOOP that never reached the gate).
        provenance: JSON-friendly structured context (handler result, authority).
        timestamp: Unix epoch seconds at construction.
    """

    action: ResponseAction
    status: ResponseStatus
    reason: str
    ethical_gate_passed: bool | None = None
    provenance: dict[str, object] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def applied(self) -> bool:
        """Whether the action actually actuated."""
        return self.status is ResponseStatus.APPLIED

    def as_metadata(self) -> dict[str, object]:
        """Return a JSON-friendly mapping describing this outcome."""
        return {
            "action": self.action.as_metadata(),
            "status": self.status.value,
            "reason": self.reason,
            "ethical_gate_passed": self.ethical_gate_passed,
            **self.provenance,
        }


@dataclass(frozen=True)
class LoopResult:
    """One auditable pass of the decision/response loop.

    A :class:`LoopResult` is the certificate the depiction layer (pillar *c*) can
    render and the ledger persists. It is fully self-describing: the decision,
    the response disposition, the honesty state, and provenance.

    Attributes:
        decision: The typed :class:`Decision`.
        response: The :class:`ResponseOutcome` (possibly a NOOP).
        domain: The domain hint the loop ran under (context only).
        provenance: JSON-friendly loop-level context.
        timestamp: Unix epoch seconds at construction.
    """

    decision: Decision
    response: ResponseOutcome
    domain: str | None = None
    provenance: dict[str, object] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def three_state(self) -> ThreeState:
        """The honesty state of the underlying decision."""
        return self.decision.state

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly mapping for the whole loop pass."""
        return {
            "decision": self.decision.as_metadata(),
            "response": self.response.as_metadata(),
            "domain": self.domain,
            "three_state": self.three_state.value,
            "timestamp": self.timestamp,
            **self.provenance,
        }
