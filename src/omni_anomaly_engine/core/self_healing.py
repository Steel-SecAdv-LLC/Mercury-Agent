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

    To suppress this deprecation warning, set the environment variable:
        OMNI_AVA_SUPPRESS_DEPRECATION_WARNINGS=1
"""

from __future__ import annotations

import os
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


class SelfHealingDeprecationWarning(DeprecationWarning):
    """Custom deprecation warning for self_healing module.

    This warning is issued when importing from the deprecated
    omni_anomaly_engine.core.self_healing module.

    To suppress this warning:
        - Set OMNI_AVA_SUPPRESS_DEPRECATION_WARNINGS=1 environment variable
        - Use warnings.filterwarnings('ignore', category=SelfHealingDeprecationWarning)
        - Import from omni_anomaly_engine.resilience.self_healing directly
    """

    pass


def _emit_deprecation_warning() -> None:
    """Emit deprecation warning if not suppressed."""
    if os.environ.get("OMNI_AVA_SUPPRESS_DEPRECATION_WARNINGS", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        return

    warnings.warn(
        "omni_anomaly_engine.core.self_healing is deprecated. "
        "Use omni_anomaly_engine.resilience.self_healing instead. "
        "Set OMNI_AVA_SUPPRESS_DEPRECATION_WARNINGS=1 to suppress this warning.",
        SelfHealingDeprecationWarning,
        stacklevel=3,
    )


_emit_deprecation_warning()
