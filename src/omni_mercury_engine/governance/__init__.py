"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see
https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""Descriptive (metric-only) governance scalars under a three-state honesty contract.

This package is the governance counterpart to :mod:`omni_mercury_engine.verifiers`.
Verifiers ground *operational* scalars in decidable oracles; governance grounds
*metric-only* scalars in published formulas under the same cross-repo three-state invariant
(:class:`~omni_mercury_engine.governance.contract.ScalarState`: GROUNDED / UNAVAILABLE /
UNDECIDABLE).  Because every scalar here is metric-only it is filtered out of the σ_Immutable
operational vector, so the trained ethical gate is never perturbed.

Each family is kept or dropped by an explicit, codebase-evidenced **signal vet**
(:data:`~omni_mercury_engine.governance.contract.GOVERNANCE_FAMILY_VET`):

* Kept (UNAVAILABLE-capable): clinical SOFA / NEWS2 / MEWS / MELD-Na, medical-device
  ISO 14971, AI-assurance NIST AI RMF (MEASURE) and MITRE ATLAS.
* Dropped (UNDECIDABLE): OWASP LLM Top 10, IMDRF SaMD, ISO 42001/23894, NIST SP 1270,
  IEEE 7000-series -- no runtime signal can exist for them in this engine.
* Tag-only: the EU AI Act risk tier (a gate/tag, never a scalar).
"""

from omni_mercury_engine.governance import ai_safety, clinical, eu_ai_act, medical_device
from omni_mercury_engine.governance.contract import (
    GOVERNANCE_FAMILY_VET,
    FamilyVet,
    GovernanceLedgerEntry,
    GovernanceRegistry,
    GovernanceScalar,
    ScalarState,
    SignalClass,
    grounded,
    unavailable,
    undecidable,
)

__all__ = [
    "GOVERNANCE_FAMILY_VET",
    "FamilyVet",
    "GovernanceLedgerEntry",
    "GovernanceRegistry",
    "GovernanceScalar",
    "ScalarState",
    "SignalClass",
    "ai_safety",
    "clinical",
    "eu_ai_act",
    "grounded",
    "medical_device",
    "unavailable",
    "undecidable",
]
