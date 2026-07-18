# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
r"""The deterrence layer: map a disposition onto a bounded response.

A :class:`ResponsePlan` is the loop's *recommendation* -- it is advisory and
non-destructive by construction.  The layer notifies, recommends reversible
countermeasures, and escalates to a human; it never autonomously executes a
destructive or irreversible action.  That boundary is the Civilization-First
posture made concrete, and it is asserted as an invariant in the tests.

:class:`ResponsePolicy` turns a :class:`~omni_mercury_engine.decision.states.\
Disposition` plus the event's severity into a :class:`ResponsePlan`, drawing
advisory steps from an overridable, domain-aware catalogue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from omni_mercury_engine.decision.states import Disposition, ResponseAction

if TYPE_CHECKING:
    from collections.abc import Mapping

#: Severity -> urgency banding.  Ordered high-to-low; first match wins, with an
#: implicit ``routine`` floor (returned by :meth:`ResponsePolicy.urgency_for`
#: when a severity falls below the lowest band).
_URGENCY_BANDS: tuple[tuple[float, str], ...] = (
    (0.90, "critical"),
    (0.70, "urgent"),
    (0.40, "elevated"),
)

#: Urgency levels at or above which a grounded anomaly puts a human in the loop.
_HUMAN_URGENCIES: frozenset[str] = frozenset({"urgent", "critical"})

#: Default advisory, *reversible*, non-destructive countermeasure catalogue,
#: keyed by disposition.  ``{domain}`` is interpolated at build time.  Every
#: entry recommends a manual / human-gated step -- nothing here is auto-applied.
_DEFAULT_CATALOG: dict[Disposition, tuple[str, ...]] = {
    Disposition.ACT: (
        "Notify the on-call operator responsible for the {domain} domain.",
        "Preserve the implicated samples and surrounding context for review.",
        "Increase monitoring cadence on the affected scope.",
        "Prepare a reversible, least-privilege containment option for human approval.",
    ),
    Disposition.CLEAR: (),
    Disposition.DEFER: (
        "Collect additional labelled samples to resolve the ambiguity.",
        "Re-run calibration on the most recent data window.",
        "Request a second opinion from a {domain} domain expert.",
    ),
    Disposition.HOLD: (
        "Withhold autonomous action: input is outside the model's certified scope.",
        "Route to a qualified human for adjudication.",
        "Record the boundary refusal for audit.",
    ),
}

#: Advisory restorative/conversion countermeasures recommended for a grounded
#: anomaly when a :class:`ResponsePolicy` opts into the restorative posture.
#: Non-violent and reversible by construction: every step converts the
#: threat's source toward a benign state or restores what was affected --
#: nothing here contains-by-destroying, and nothing is auto-applied.
_RESTORATIVE_ACT_CATALOG: tuple[str, ...] = (
    "Identify the root cause and prepare a corrective (patch/reconfigure) "
    "path that returns the {domain} source to a benign state.",
    "Draft a staged reintegration plan: remediate, re-validate integrity, "
    "then restore normal access under monitoring.",
    "Where a human actor is implicated, prepare a non-punitive corrective "
    "off-ramp for operator review before any enforcement step.",
    "Restore any affected {domain} state from verified-good backups once "
    "the corrective path is approved.",
)


@dataclass(frozen=True)
class ResponsePlan:
    """A bounded, non-destructive response recommendation.

    Attributes:
        action: The primary recommended :class:`ResponseAction`.
        urgency: ``routine`` / ``elevated`` / ``urgent`` / ``critical``.
        requires_human: Whether a human must approve before any consequential
            step.  Always ``True`` for a fail-closed hold.
        notify: Whether to emit an operator notification (e.g. a CAP alert).
        fail_closed: Whether this is a fail-closed refusal (no autonomous
            action permitted).
        countermeasures: Advisory, reversible steps -- recommendations only,
            never executed by this layer.
        rationale: One-line explanation of the chosen response.
    """

    action: ResponseAction
    urgency: str
    requires_human: bool
    notify: bool
    fail_closed: bool
    countermeasures: tuple[str, ...] = ()
    rationale: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ResponsePlan:
        """Reconstruct a plan from its :meth:`to_dict` form (the exact inverse)."""
        return cls(
            action=ResponseAction(data["action"]),
            urgency=data["urgency"],
            requires_human=bool(data["requires_human"]),
            notify=bool(data["notify"]),
            fail_closed=bool(data["fail_closed"]),
            countermeasures=tuple(data.get("countermeasures", ())),
            rationale=data.get("rationale", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view of the plan."""
        return {
            "action": self.action.value,
            "urgency": self.urgency,
            "requires_human": self.requires_human,
            "notify": self.notify,
            "fail_closed": self.fail_closed,
            "countermeasures": list(self.countermeasures),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class ResponsePolicy:
    """Map a disposition + severity onto a :class:`ResponsePlan`.

    Attributes:
        catalog: Per-disposition advisory countermeasure templates.  Defaults
            to a generic, domain-aware, reversible catalogue; override to
            inject domain playbooks.
        restorative: Opt-in restorative posture (default ``False`` — the
            wire format of existing consumers is unchanged).  When ``True``,
            a grounded anomaly *below* the human-approval urgency bar
            recommends :attr:`ResponseAction.RECOMMEND_CONVERSION` (a
            restorative, non-violent convert-to-benign path) instead of
            :attr:`ResponseAction.RECOMMEND_MITIGATION`, and the plan's
            countermeasures gain the restorative catalogue.  Urgent/critical
            events still escalate to a human, holds still fail closed, and
            every step remains recommend-only — the posture can never
            auto-authorise anything.
    """

    catalog: Mapping[Disposition, tuple[str, ...]] = field(
        default_factory=lambda: dict(_DEFAULT_CATALOG)
    )
    restorative: bool = False

    @staticmethod
    def urgency_for(severity: float) -> str:
        """Band a severity in ``[0, 1]`` into an urgency label."""
        for floor, label in _URGENCY_BANDS:
            if severity >= floor:
                return label
        return "routine"

    def _countermeasures(self, disposition: Disposition, domain: str | None) -> tuple[str, ...]:
        templates = self.catalog.get(disposition, ())
        dom = domain or "general"
        return tuple(step.format(domain=dom) for step in templates)

    def plan(
        self,
        disposition: Disposition,
        *,
        severity: float = 0.0,
        domain: str | None = None,
        resolvable_by_input: bool = False,
    ) -> ResponsePlan:
        """Build the bounded response for a disposition.

        Args:
            disposition: The decided stance.
            severity: Event severity in ``[0, 1]`` (drives urgency).
            domain: Optional domain hint for the countermeasure catalogue.
            resolvable_by_input: For a ``DEFER``, whether the indecision is
                resolvable by gathering a missing signal (selects
                ``REQUEST_INPUT`` over a plain human escalation).

        Returns:
            A non-destructive :class:`ResponsePlan`.
        """
        urgency = self.urgency_for(severity)
        steps = self._countermeasures(disposition, domain)

        if disposition is Disposition.CLEAR:
            return ResponsePlan(
                action=ResponseAction.MONITOR,
                urgency="routine",
                requires_human=False,
                notify=False,
                fail_closed=False,
                countermeasures=steps,
                rationale="Grounded normal: passive monitoring only.",
            )

        if disposition is Disposition.ACT:
            requires_human = urgency in _HUMAN_URGENCIES
            if requires_human:
                action = ResponseAction.ESCALATE_TO_HUMAN
            elif self.restorative:
                action = ResponseAction.RECOMMEND_CONVERSION
            else:
                action = ResponseAction.RECOMMEND_MITIGATION
            if self.restorative:
                dom = domain or "general"
                steps = steps + tuple(
                    step.format(domain=dom) for step in _RESTORATIVE_ACT_CATALOG
                )
            return ResponsePlan(
                action=action,
                urgency=urgency,
                requires_human=requires_human,
                notify=True,
                fail_closed=False,
                countermeasures=steps,
                rationale=(
                    f"Grounded anomaly ({urgency}): notify and recommend "
                    + (
                        "restorative, non-violent conversion steps"
                        if self.restorative
                        else "reversible countermeasures"
                    )
                    + (", human approval required." if requires_human else ".")
                ),
            )

        if disposition is Disposition.DEFER:
            action = (
                ResponseAction.REQUEST_INPUT
                if resolvable_by_input
                else ResponseAction.ESCALATE_TO_HUMAN
            )
            return ResponsePlan(
                action=action,
                urgency=urgency,
                requires_human=True,
                notify=True,
                fail_closed=False,
                countermeasures=steps,
                rationale=(
                    "Resolvable abstention: defer to a human / request the "
                    "missing signal before acting."
                ),
            )

        # Disposition.HOLD -- fail-closed refusal.
        return ResponsePlan(
            action=ResponseAction.HOLD,
            urgency=urgency if urgency != "routine" else "elevated",
            requires_human=True,
            notify=True,
            fail_closed=True,
            countermeasures=steps,
            rationale=(
                "Fail-closed abstention: out-of-scope or ethically blocked; "
                "no autonomous action, route to a human."
            ),
        )


__all__ = ["ResponsePlan", "ResponsePolicy"]
