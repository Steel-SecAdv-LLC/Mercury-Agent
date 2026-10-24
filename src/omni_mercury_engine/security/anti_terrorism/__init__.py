# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Anti-Terrorism Pattern Recognition Module.

Stages reported behaviour against the US Violent Extremist Mobilization
Indicators (NCTC, FBI, DHS) for referral to trained human review.

Part of Mercury Agent Security module.
"""

from __future__ import annotations

from omni_mercury_engine.security.anti_terrorism.pattern_recognition import (
    TerrorismPatternDetector,
    TerrorismThreatResult,
)

__all__ = ["TerrorismPatternDetector", "TerrorismThreatResult"]
