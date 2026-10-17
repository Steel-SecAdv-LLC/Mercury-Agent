# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Gated surfaces for the evidence-assessment and human-subject library modules.

Four library modules had no API endpoint, CLI command or MCP tool, and so no
gate in front of them:

* statement verification and the Concealed Information Test
  (:mod:`omni_mercury_engine.truth_decipher`);
* counterintelligence screening and the all-source fusion that consumes it
  (:mod:`omni_mercury_engine.security.counterintelligence`,
  :mod:`omni_mercury_engine.security.intelligence_fusion`);
* the adjudication ledger (:mod:`omni_mercury_engine.security.adjudication`);
* contactless vital signs (:mod:`omni_mercury_engine.biometric.rf_vital_signs`).

:mod:`.service` is the one implementation the three surfaces call: it parses
JSON input, applies the gates in :mod:`.policy`, runs the library and returns
JSON-compatible output. The API routes (``api/routes/assessment.py``), the CLI
groups (``mercury-agent assess`` / ``mercury-agent adjudication``) and the MCP
tools differ only in transport and in how they report a refusal.

Importing this package is cheap; the library modules, the decision gate and
torch are imported on first use.
"""

from __future__ import annotations

from omni_mercury_engine.assessment.policy import (
    HUMAN_SUBJECT_POLICY_ENV,
    CapabilityDisabledError,
    LawfulBasis,
)

__all__ = ["HUMAN_SUBJECT_POLICY_ENV", "CapabilityDisabledError", "LawfulBasis"]
