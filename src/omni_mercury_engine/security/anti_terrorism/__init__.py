# Copyright (C) 2025 Steel Security Advisors LLC
"""Anti-Terrorism Pattern Recognition Module.

Detects radicalization patterns using QBM and OSINT anomalies.
Integrates with existing threat knowledge base.

Part of Mercury Agent Security module.
"""

from __future__ import annotations

from omni_mercury_engine.security.anti_terrorism.pattern_recognition import (
    TerrorismPatternDetector,
    TerrorismThreatResult,
)

__all__ = ["TerrorismPatternDetector", "TerrorismThreatResult"]
