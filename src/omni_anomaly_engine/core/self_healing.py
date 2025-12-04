"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Backward compatibility shim for self-healing module.

Note:
    This module is deprecated. Use the following instead:

    from omni_anomaly_engine.resilience.self_healing import (
        SelfHealingEngine,
        AdaptiveDefenseSystem,
        AnomalySignature,
    )
"""

import warnings

# Re-export from new location
from omni_anomaly_engine.resilience.self_healing import (
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

# Issue deprecation warning on import
warnings.warn(
    "omni_anomaly_engine.core.self_healing is deprecated. "
    "Use omni_anomaly_engine.resilience.self_healing instead.",
    DeprecationWarning,
    stacklevel=2,
)
