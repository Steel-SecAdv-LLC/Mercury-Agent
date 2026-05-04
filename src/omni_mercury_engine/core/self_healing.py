"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

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

Self-Healing Module - Backwards Compatibility Layer

This module re-exports from resilience.self_healing for backwards compatibility.
New code should import directly from omni_mercury_engine.resilience.self_healing.

Deprecated:
    Import from omni_mercury_engine.resilience.self_healing instead.
"""

from __future__ import annotations

# Re-export everything from canonical location
from omni_mercury_engine.resilience.self_healing import *  # noqa: F401, F403

# Explicit re-exports for type checking
from omni_mercury_engine.resilience.self_healing import (
    AdaptiveDefenseSystem,
    AnomalySignature,
    SelfHealingEngine,
)

# Backward compatibility alias
CRISPRInspiredSelfHealing = AdaptiveDefenseSystem

__all__ = [
    "AdaptiveDefenseSystem",
    "AnomalySignature",
    "CRISPRInspiredSelfHealing",
    "SelfHealingEngine",
]
