# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Agentic AI autonomy modules."""

from __future__ import annotations

from omni_mercury_engine.agentic.agentic_autonomy import AgentAction, AgenticAutonomy, AgentState
from omni_mercury_engine.agentic.bayesian_calibrator import (
    BayesianConfidenceCalibrator,
    CalibrationConfig,
    ContextStats,
)
from omni_mercury_engine.agentic.mercury_a_agent import (
    AgentMemory,
    AgentMode,
    DomainType,
    MercuryAgent,
    MercuryPlanner,
    MercuryReasoner,
    PlanResult,
    ReasoningStep,
    Task,
    TaskPriority,
    create_mercury_agent,
)
from omni_mercury_engine.agentic.orchestration import (
    CoordinationBatch,
    DetectorAgent,
    EpisodeResult,
    MultiAgentOrchestrator,
    OrchestrationError,
    PlanTrace,
    ReflectionRecord,
    default_detector_suite,
)

# Internal subagent fleet (delegation tier). Re-exported here for internal
# consumers; deliberately NOT re-exported from the public ``omni_mercury_engine``
# package surface — subagents are reachable only through the engine-mediated
# path (``OmniMercuryEngine.enable_subagent_fleet`` / ``MercuryAgent.enable_fleet``).
# Importing these symbols does not eagerly load the specializations (those load
# only when ``default_registry`` is called).
from omni_mercury_engine.agentic.subagents import (
    AggregateResult,
    AutonomyGovernor,
    CapabilityCeiling,
    FleetResult,
    GovernorTripped,
    SubAgent,
    SubAgentAccessError,
    SubAgentExecutionError,
    SubAgentFleet,
    SubAgentRegistry,
    SubAgentResult,
    SubAgentTask,
)

__all__ = [
    "AggregateResult",
    "AutonomyGovernor",
    "CapabilityCeiling",
    "FleetResult",
    "GovernorTripped",
    "SubAgent",
    "SubAgentAccessError",
    "SubAgentExecutionError",
    "SubAgentFleet",
    "SubAgentRegistry",
    "SubAgentResult",
    "SubAgentTask",
    "AgentAction",
    "AgentMemory",
    "AgentMode",
    "AgentState",
    "AgenticAutonomy",
    "BayesianConfidenceCalibrator",
    "CalibrationConfig",
    "ContextStats",
    "CoordinationBatch",
    "DetectorAgent",
    "DomainType",
    "EpisodeResult",
    "MercuryAgent",
    "MercuryPlanner",
    "MercuryReasoner",
    "MultiAgentOrchestrator",
    "OrchestrationError",
    "PlanResult",
    "PlanTrace",
    "ReasoningStep",
    "ReflectionRecord",
    "Task",
    "TaskPriority",
    "create_mercury_agent",
    "default_detector_suite",
]
