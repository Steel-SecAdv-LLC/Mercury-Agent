# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Adapters that close the loop into the engine's existing channels.

The decision layer decides; these thin, lazily-bound adapters carry that
decision to the two output channels the engine already ships, so the loop is
genuinely end-to-end rather than a new silo:

* :func:`to_agent_action` -- a :class:`~omni_mercury_engine.agentic.\
agentic_autonomy.AgentAction` for the reinforcement-learning autonomy loop,
  using its existing action vocabulary.
* :func:`to_cap_alert` -- a standards-based CAP 1.2 alert (via the existing
  :class:`~omni_mercury_engine.alerting.cap_generator.CAPAlertGenerator`) for
  every decision whose response asks to notify.

Both imports are deferred so importing :mod:`omni_mercury_engine.decision`
stays cheap and free of the heavier optional dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from omni_mercury_engine.decision.states import Disposition

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omni_mercury_engine.agentic.agentic_autonomy import AgentAction
    from omni_mercury_engine.decision.record import DecisionRecord

#: Disposition -> the existing autonomy ``ACTION_TYPES`` vocabulary, so a
#: decision record drives the RL agent without inventing a second vocabulary.
_DISPOSITION_TO_ACTION_TYPE: dict[Disposition, str] = {
    Disposition.ACT: "flag_anomaly",
    Disposition.CLEAR: "log",
    Disposition.DEFER: "escalate",
    Disposition.HOLD: "escalate",
}


def to_agent_action(record: DecisionRecord) -> AgentAction:
    """Adapt a :class:`DecisionRecord` to an :class:`AgentAction`.

    The agent's confidence is the calibrated decision confidence when the loop
    grounded a label, falling back to the raw anomaly probability when it
    abstained (the agent still needs a scalar to learn from).  The full record
    travels in ``parameters`` so nothing is lost.

    Args:
        record: The decision to adapt.

    Returns:
        An :class:`AgentAction` ready to push into the autonomy loop.
    """
    from omni_mercury_engine.agentic.agentic_autonomy import AgentAction

    action_type = _DISPOSITION_TO_ACTION_TYPE[record.disposition]
    confidence = (
        record.decision_confidence
        if record.decision_confidence is not None
        else record.anomaly_prob
    )
    return AgentAction(
        action_type=action_type,
        parameters={
            "state": record.state.value,
            "disposition": record.disposition.value,
            "response": record.response.to_dict(),
            "domain": record.domain,
            "abstained": record.abstained,
            "decision": record.to_dict(),
        },
        confidence=float(confidence),
        rationale=record.explain(),
    )


def to_cap_alert(
    record: DecisionRecord,
    *,
    domain: str | None = None,
    scores: Any | None = None,
    area_description: str = "Unspecified Area",
    coordinates: tuple[float, float] | None = None,
    geocode: dict[str, str] | None = None,
    rca_causes: list[tuple[int, float]] | None = None,
) -> str | None:
    """Render a CAP 1.2 alert for a decision that asks to notify.

    Returns ``None`` for any decision whose :class:`ResponsePlan` does not set
    ``notify`` (e.g. a grounded-normal ``MONITOR``), so callers can map over a
    stream of records and emit alerts only where one is warranted.

    Args:
        record: The decision to render.
        domain: Domain for the alert (falls back to the record's domain).
        scores: Optional anomaly-score array; defaults to the record's
            ``anomaly_prob``.
        area_description: Human-readable affected area.
        coordinates: Optional ``(lat, lon)``.
        geocode: Optional CAP geocode dict.
        rca_causes: Optional ranked ``(node_index, attribution)`` root causes,
            e.g. from
            :func:`omni_mercury_engine.detectors.detection_tier.rca_localize`.
            When given, the top few are added to the alert's Mercury metadata
            (rendered into the CAP ``<description>`` alongside the other decision
            fields) under ``RootCauses`` so on-call triage sees *where* the
            anomaly originated.

    Returns:
        A CAP 1.2 XML string, or ``None`` when no notification is warranted.
    """
    if not record.response.notify:
        return None

    from omni_mercury_engine.alerting.cap_generator import CAPAlertGenerator

    generator = CAPAlertGenerator()
    alert_domain = domain or record.domain or "general"
    alert_scores = scores if scores is not None else [record.anomaly_prob]
    metadata = {
        "DecisionState": record.state.value,
        "Disposition": record.disposition.value,
        "Urgency": record.response.urgency,
        "RequiresHuman": str(record.response.requires_human),
        "FailClosed": str(record.response.fail_closed),
        "Calibrated": str(record.calibrated),
        "Coverage": "n/a" if record.coverage is None else f"{record.coverage:.2f}",
        "Reasons": " | ".join(record.reasons),
    }
    if rca_causes:
        metadata["RootCauses"] = ", ".join(
            f"node{node}:{attribution:.3f}" for node, attribution in rca_causes[:5]
        )
    return generator.from_detection(
        domain=alert_domain,
        scores=alert_scores,
        metadata=metadata,
        area_description=area_description,
        coordinates=coordinates,
        geocode=geocode,
    )


__all__ = ["to_agent_action", "to_cap_alert"]
