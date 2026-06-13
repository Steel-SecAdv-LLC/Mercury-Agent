# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Bio-Threat Detection Module (Medical Interdiction).

Detects biological threats using QBM-based pathogen energy modeling
and multi-INT fusion (MASINT bio-signatures, OSINT disease outbreaks).

Part of Medical Interdiction and Intervention framework.
"""

from __future__ import annotations

from omni_mercury_engine.medical.pandemic.bio_threats.pathogen_detector import (
    BioThreatResult,
    PathogenDetector,
)

__all__ = ["BioThreatResult", "PathogenDetector"]
