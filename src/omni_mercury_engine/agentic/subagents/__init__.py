# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Internal-only subagent fleet for Mercury Agent — the Greek-pantheon roster.

This package is the orchestration tier through which the main Mercury Agent
*delegates* arbitrary tasks to a fleet of 33 named, full-capability subagents
(the Greek pantheon ``Themis_I`` … ``Rhea_XXXIII``) — singly, across a batch, or
to many replicas in the masses. Each subagent is itself a full
:class:`~omni_mercury_engine.agentic.mercury_a_agent.MercuryAgent`, bound to a
real ``omni_mercury_engine`` subsystem and to exactly one of the Seven Omni-Codes
(its autonomy anchor), and gated by the same ethical, cryptographic, and
governance seams as every other Mercury decision boundary.

Access boundary (deliberate): nothing here is re-exported from the public
``omni_mercury_engine`` package surface. Subagents are reachable only through the
engine-mediated path (``OmniMercuryEngine.enable_subagent_fleet`` /
``MercuryAgent.enable_fleet``), and construction is guarded by an internal access
sentinel (:data:`base._INTERNAL`). Users never instantiate or address a subagent
directly; the main agent calls on them.

Transparency contract (anti-theater): no stage fabricates signal; subagent failures
are surfaced, never silently dropped; the dual hard ethical gates (benevolence
floor + σ-Immutable) run fail-closed at the fleet's commit boundary; and the
autonomy governor enforces a real capability ceiling, ethical floor,
corrigibility kill-switch, recursion bound, and tripwire. Terminology is
Omni-Codes only — no other code system is referenced in this subsystem.
"""

from __future__ import annotations

from omni_mercury_engine.agentic.subagents.base import (
    SubAgent,
    SubAgentAccessError,
    SubAgentCapability,
    SubAgentExecutionError,
    SubAgentResult,
    SubAgentTask,
    anchor_autonomy,
    resolve_anchor,
)
from omni_mercury_engine.agentic.subagents.coordinator import CoordinatorSubAgent
from omni_mercury_engine.agentic.subagents.fleet import (
    AggregateResult,
    FleetResult,
    SubAgentFleet,
)
from omni_mercury_engine.agentic.subagents.governor import (
    AutonomyGovernor,
    CapabilityCeiling,
    GovernorTripped,
)
from omni_mercury_engine.agentic.subagents.registry import (
    SubAgentRegistry,
    default_registry,
)
from omni_mercury_engine.agentic.subagents.roster import (
    ALL_ENTRIES,
    ROSTER,
    RosterEntry,
    code_bearers,
    entry_by_id,
    validate_roster,
)

__all__ = [
    "ALL_ENTRIES",
    "ROSTER",
    "AggregateResult",
    "AutonomyGovernor",
    "CapabilityCeiling",
    "CoordinatorSubAgent",
    "FleetResult",
    "GovernorTripped",
    "RosterEntry",
    "SubAgent",
    "SubAgentAccessError",
    "SubAgentCapability",
    "SubAgentExecutionError",
    "SubAgentFleet",
    "SubAgentRegistry",
    "SubAgentResult",
    "SubAgentTask",
    "anchor_autonomy",
    "code_bearers",
    "default_registry",
    "entry_by_id",
    "resolve_anchor",
    "validate_roster",
]
