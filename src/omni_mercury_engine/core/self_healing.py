# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Self-Healing Module - Backwards Compatibility Layer.

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
