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

The response ("deter") half of pillar *a*: proportionate, reversible, gated.

A detection that never acts is not autonomous; a system that acts without
guard-rails is not safe. This module is the closed-loop response layer the tree
did not have -- and its design is its safety case:

* **Proportionate.** :class:`ResponsePlanner` maps ``(verdict, severity)`` to a
  single graded :class:`ResponseAction` on an ascending ladder
  (monitor -> notify -> soft-contain -> escalate).
* **Abstention never deters.** An honest "don't-know" can only *gather evidence*
  or *inform a human*; it can never actuate a countermeasure against the
  environment. This is enforced twice -- in the planner and again in the
  actuator (defence in depth).
* **Reversible by default.** Every built-in action is reversible. Anything
  irreversible or escalatory carries ``requires_human_authorization`` and is
  *deferred* to a human, never executed by the loop.
* **Fail-closed ethics.** Every effectful action passes a caller-supplied
  :class:`EthicalGate` *before* actuation. The gate must be chosen explicitly --
  there is no silent default -- and a raising gate blocks the action.

The default handlers are safe, recordable placeholders. A deployment plugs real
effectors (a rate-limiter, a quarantine queue, a CAP alert) via
:meth:`ResponseActuator.register_handler`; the layer's contract -- gating,
reversibility, authorization, provenance -- holds regardless of the effector.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from omni_mercury_engine.core.types import ThreatLevel
from omni_mercury_engine.decision.types import (
    Decision,
    ResponseAction,
    ResponseOutcome,
    ResponseStatus,
    ResponseTier,
    Verdict,
)

__all__ = [
    "Authorization",
    "EthicalGate",
    "ResponseActuator",
    "ResponsePlanner",
    "ResponseVetoError",
    "deny_all_gate",
    "permit_all_gate",
    "threat_level_from_score",
]

# A response handler turns an action + context into a JSON-friendly provenance
# record describing what the effector did. It must not raise for control flow;
# the actuator owns gating and authorization decisions.
ResponseHandler = Callable[[ResponseAction, Mapping[str, object]], dict[str, object]]


class ResponseVetoError(Exception):
    """Raised by an :class:`EthicalGate` to veto a response before actuation."""


@runtime_checkable
class EthicalGate(Protocol):
    """A fail-closed veto consulted before any effectful response actuates.

    Implementations raise to veto (any exception blocks the action) and return
    ``None`` to allow. The engine binds this to its dual hard gate
    (``BenevolenceScorer`` + ``SigmaImmutableGate``) so a response can never
    actuate on input the engine's own boundary would refuse.
    """

    def __call__(
        self,
        *,
        action: ResponseAction,
        decision: Decision,
        domain: str | None,
    ) -> None:
        """Allow (return ``None``) or veto (raise) ``action`` for ``decision``."""
        ...


def permit_all_gate(*, action: ResponseAction, decision: Decision, domain: str | None) -> None:
    """An explicit allow-everything gate. For tests / trusted offline analysis only."""


def deny_all_gate(*, action: ResponseAction, decision: Decision, domain: str | None) -> None:
    """An explicit deny-everything gate (the strictest fail-closed posture)."""
    raise ResponseVetoError(f"deny_all_gate vetoed {action.name!r}")


def threat_level_from_score(severity_score: float) -> ThreatLevel:
    """Map a ``[0, 1]`` severity score onto the shared :class:`ThreatLevel` ladder.

    Args:
        severity_score: Severity in ``[0, 1]`` (e.g. the engine's ``severity``).

    Returns:
        A proportionate :class:`ThreatLevel`.
    """
    score = min(1.0, max(0.0, float(severity_score)))
    if score >= 0.90:
        return ThreatLevel.CRITICAL
    if score >= 0.75:
        return ThreatLevel.SEVERE
    if score >= 0.60:
        return ThreatLevel.HIGH
    if score >= 0.40:
        return ThreatLevel.SUBSTANTIAL
    if score >= 0.25:
        return ThreatLevel.MODERATE
    if score >= 0.10:
        return ThreatLevel.LOW
    return ThreatLevel.NONE


@dataclass(frozen=True)
class Authorization:
    """An explicit human sign-off permitting an escalatory / irreversible action.

    Attributes:
        authority: Who granted it (operator id, role, ticket).
        reason: Why it was granted (audit context).
    """

    authority: str
    reason: str = ""

    def as_metadata(self) -> dict[str, object]:
        """Return a JSON-friendly mapping describing this authorization."""
        return {"authority": self.authority, "reason": self.reason}


class ResponsePlanner:
    """Select one proportionate, reversible-by-default action for a decision.

    The planner is pure: ``(verdict, severity)`` deterministically selects an
    action on the response ladder. It never returns a deterrent for an
    abstention.
    """

    def plan(
        self,
        decision: Decision,
        *,
        severity: ThreatLevel,
        domain: str | None = None,
    ) -> ResponseAction:
        """Return the proportionate :class:`ResponseAction` for ``decision``.

        Args:
            decision: The typed decision to respond to.
            severity: Proportionality anchor (see :func:`threat_level_from_score`).
            domain: Domain hint, recorded for provenance.

        Returns:
            A single graded action. Abstentions yield only
            :attr:`ResponseTier.GATHER_EVIDENCE` or :attr:`ResponseTier.NOTIFY`.
        """
        if decision.verdict is Verdict.NEGATIVE:
            return ResponseAction(
                name="monitor.continue",
                tier=ResponseTier.MONITOR,
                severity=ThreatLevel.NONE,
                reversible=True,
                requires_human_authorization=False,
                rationale="grounded-negative call: keep passively monitoring the stream",
                params={"domain": domain},
            )

        if decision.verdict is Verdict.ABSTAIN:
            if severity.value >= ThreatLevel.HIGH.value:
                return ResponseAction(
                    name="notify.review_abstention",
                    tier=ResponseTier.NOTIFY,
                    severity=severity,
                    reversible=True,
                    requires_human_authorization=False,
                    rationale=(
                        "honest abstention on a high-severity stream: inform a human "
                        "for adjudication -- never deter on uncertainty"
                    ),
                    params={"domain": domain, "novelty": decision.novelty},
                )
            return ResponseAction(
                name="gather_evidence.reduce_uncertainty",
                tier=ResponseTier.GATHER_EVIDENCE,
                severity=severity,
                reversible=True,
                requires_human_authorization=False,
                rationale=(
                    "honest abstention: actively reduce uncertainty (more data / "
                    "tighter calibration) rather than commit"
                ),
                params={"domain": domain, "novelty": decision.novelty},
            )

        # Verdict.POSITIVE -- proportionate ladder by severity.
        if severity.value >= ThreatLevel.SEVERE.value:
            return ResponseAction(
                name="escalate.human_authority",
                tier=ResponseTier.ESCALATE,
                severity=severity,
                reversible=False,
                requires_human_authorization=True,
                rationale=(
                    "grounded anomaly at severe/critical severity: hand the hard, "
                    "potentially irreversible action to a human authority"
                ),
                params={"domain": domain},
            )
        if severity.value >= ThreatLevel.SUBSTANTIAL.value:
            return ResponseAction(
                name="soft_contain.quarantine_review",
                tier=ResponseTier.SOFT_CONTAIN,
                severity=severity,
                reversible=True,
                requires_human_authorization=False,
                rationale=(
                    "grounded anomaly at substantial/high severity: apply a reversible "
                    "mitigation (quarantine-for-review / throttle) pending confirmation"
                ),
                params={"domain": domain, "undoable": True},
            )
        return ResponseAction(
            name="notify.flag_anomaly",
            tier=ResponseTier.NOTIFY,
            severity=severity,
            reversible=True,
            requires_human_authorization=False,
            rationale="grounded anomaly at low/moderate severity: flag for attention",
            params={"domain": domain},
        )


class ResponseActuator:
    """Apply a planned action under the safety contract, recording the outcome.

    The actuator owns the gating decisions; handlers only describe effects. The
    contract, applied in order:

    1. :attr:`ResponseTier.NONE` -> :attr:`ResponseStatus.NOOP`.
    2. An abstention may never actuate a deterrent (``SOFT_CONTAIN`` /
       ``ESCALATE``); such a pairing is *deferred* with an explicit reason.
    3. The :class:`EthicalGate` runs for every effectful action; a veto yields
       :attr:`ResponseStatus.BLOCKED`.
    4. ``requires_human_authorization`` (or a non-reversible action) without a
       matching :class:`Authorization` yields :attr:`ResponseStatus.DEFERRED`.
    5. Otherwise the registered handler runs and the action is
       :attr:`ResponseStatus.APPLIED`.

    Args:
        ethical_gate: The fail-closed veto. Must be chosen explicitly (pass
            :func:`permit_all_gate` only for tests / trusted offline analysis).
        handlers: Optional initial ``{action_name: handler}`` registry; defaults
            to safe, recordable placeholders for every built-in action.
    """

    def __init__(
        self,
        ethical_gate: EthicalGate,
        handlers: Mapping[str, ResponseHandler] | None = None,
    ) -> None:
        self.ethical_gate = ethical_gate
        self._handlers: dict[str, ResponseHandler] = dict(_default_handlers())
        if handlers:
            self._handlers.update(handlers)

    def register_handler(self, action_name: str, handler: ResponseHandler) -> None:
        """Register (or override) the effector for ``action_name``."""
        self._handlers[action_name] = handler

    def actuate(
        self,
        action: ResponseAction,
        decision: Decision,
        *,
        domain: str | None = None,
        authorization: Authorization | None = None,
        context: Mapping[str, object] | None = None,
    ) -> ResponseOutcome:
        """Apply ``action`` for ``decision`` under the safety contract."""
        ctx: dict[str, object] = dict(context or {})

        if action.tier is ResponseTier.NONE:
            return ResponseOutcome(
                action=action,
                status=ResponseStatus.NOOP,
                reason="no action warranted",
            )

        # (2) Defence in depth: an abstention must never deter.
        if decision.abstained and action.tier in (
            ResponseTier.SOFT_CONTAIN,
            ResponseTier.ESCALATE,
        ):
            return ResponseOutcome(
                action=action,
                status=ResponseStatus.DEFERRED,
                reason=(
                    "refusing to actuate a deterrent on an abstention; "
                    "an honest don't-know defers to a human"
                ),
                provenance={"guard": "abstention_no_deter"},
            )

        # (3) Fail-closed ethical gate.
        try:
            self.ethical_gate(action=action, decision=decision, domain=domain)
            gate_passed = True
        except Exception as exc:
            return ResponseOutcome(
                action=action,
                status=ResponseStatus.BLOCKED,
                reason=f"ethical gate vetoed response: {exc}",
                ethical_gate_passed=False,
                provenance={"gate_error": type(exc).__name__},
            )

        # (4) Authorization required for escalatory / irreversible actions.
        needs_authorization = action.requires_human_authorization or not action.reversible
        if needs_authorization and authorization is None:
            return ResponseOutcome(
                action=action,
                status=ResponseStatus.DEFERRED,
                reason=(
                    "action requires explicit human authorization "
                    "(escalatory or irreversible); deferred pending sign-off"
                ),
                ethical_gate_passed=gate_passed,
                provenance={"awaiting": "human_authorization"},
            )

        # (5) Apply via the registered handler.
        handler = self._handlers.get(action.name, _record_only_handler)
        effect = handler(action, ctx)
        provenance: dict[str, object] = {"effect": effect}
        if authorization is not None:
            provenance["authorization"] = authorization.as_metadata()
        return ResponseOutcome(
            action=action,
            status=ResponseStatus.APPLIED,
            reason=f"applied {action.tier.value} response",
            ethical_gate_passed=gate_passed,
            provenance=provenance,
        )


def _record_only_handler(
    action: ResponseAction, context: Mapping[str, object]
) -> dict[str, object]:
    """Default effector: record the action without touching the environment."""
    return {"handler": "record_only", "action": action.name, "reversible": action.reversible}


def _default_handlers() -> dict[str, ResponseHandler]:
    """Safe, recordable default effectors for the built-in actions.

    These intentionally have no environmental side effect -- a deployment swaps
    in real effectors via :meth:`ResponseActuator.register_handler`. The point of
    the layer is the *contract* (gating, reversibility, authorization, audit),
    which holds whatever the effector does.
    """
    return {
        "monitor.continue": _record_only_handler,
        "gather_evidence.reduce_uncertainty": _record_only_handler,
        "notify.review_abstention": _record_only_handler,
        "notify.flag_anomaly": _record_only_handler,
        "soft_contain.quarantine_review": _record_only_handler,
        "escalate.human_authority": _record_only_handler,
    }
