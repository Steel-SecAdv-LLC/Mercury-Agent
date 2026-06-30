# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Deep subagent specializations — real domain logic, no stubs.

These classes back the ``deep`` members of the pantheon roster (see
:mod:`~omni_mercury_engine.agentic.subagents.roster`); the registry resolves them
lazily by dotted ``impl_path``. Coordinator members use
:class:`~omni_mercury_engine.agentic.subagents.coordinator.CoordinatorSubAgent`.

Provenance: ``compliance``, ``ethics``, and ``guardrail`` capture capabilities
transferred from FINDΩYOU™'s former agent layer (which is being made agent-free);
``detection`` wraps Mercury's own multi-agent detection; ``generalist`` is the
full main-agent pipeline (the internal routing floor).
"""

from __future__ import annotations

from omni_mercury_engine.agentic.subagents.specializations.compliance import (
    ComplianceSubAgent,
)
from omni_mercury_engine.agentic.subagents.specializations.detection import (
    DetectionSubAgent,
)
from omni_mercury_engine.agentic.subagents.specializations.ethics import (
    EthicsEnforcementSubAgent,
)
from omni_mercury_engine.agentic.subagents.specializations.generalist import (
    GeneralistSubAgent,
)
from omni_mercury_engine.agentic.subagents.specializations.guardrail import (
    GuardrailSubAgent,
)

__all__ = [
    "ComplianceSubAgent",
    "DetectionSubAgent",
    "EthicsEnforcementSubAgent",
    "GeneralistSubAgent",
    "GuardrailSubAgent",
]
