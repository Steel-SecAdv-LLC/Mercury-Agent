"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
Anti-Terrorism Pattern Recognition Module

Detects radicalization patterns using QBM and OSINT anomalies.
Integrates with existing threat knowledge base.

Part of Mercury Agent Security module.
"""

from omni_mercury_engine.security.anti_terrorism.pattern_recognition import (
    TerrorismPatternDetector,
    TerrorismThreatResult,
)

__all__ = ["TerrorismPatternDetector", "TerrorismThreatResult"]
