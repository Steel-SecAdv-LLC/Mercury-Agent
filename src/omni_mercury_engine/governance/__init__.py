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

"""Descriptive (metric-only) governance scalars under an abstention-first contract.

This package is the governance counterpart to :mod:`omni_mercury_engine.verifiers`.
Verifiers ground *operational* scalars in decidable oracles; governance grounds
*metric-only* scalars in published formulas, and -- crucially -- abstains (registers
nothing) whenever the required input signal is absent.  Because every scalar here is
metric-only it is filtered out of the σ_Immutable operational vector, so the trained
ethical gate is never perturbed.

Families: clinical (SOFA, NEWS2), medical-device risk (ISO 14971), and AI assurance
(NIST AI RMF / OWASP LLM Top 10 / MITRE ATLAS, which abstain unless attested).
"""

from omni_mercury_engine.governance import ai_safety, clinical, medical_device
from omni_mercury_engine.governance.contract import (
    GovernanceLedgerEntry,
    GovernanceRegistry,
    GovernanceScalar,
    ScalarStatus,
    available,
    unavailable,
)

__all__ = [
    "GovernanceLedgerEntry",
    "GovernanceRegistry",
    "GovernanceScalar",
    "ScalarStatus",
    "ai_safety",
    "available",
    "clinical",
    "medical_device",
    "unavailable",
]
