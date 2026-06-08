# Copyright (C) 2025 Steel Security Advisors LLC
"""(at your option) any later version."""

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
