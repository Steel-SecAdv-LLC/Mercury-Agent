# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Self-Healing Module - Backwards Compatibility Layer.

This module re-exports from resilience.self_healing for backwards compatibility.
New code should import directly from omni_mercury_engine.resilience.self_healing.

Deprecated:
    Import from omni_mercury_engine.resilience.self_healing instead.
"""

from __future__ import annotations

import os
import warnings

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


class SelfHealingDeprecationWarning(DeprecationWarning):
    """Deprecation warning for the ``core.self_healing`` compatibility shim.

    Issued on import of the deprecated
    ``omni_mercury_engine.core.self_healing`` module.

    To suppress this warning:
        - Set MERCURY_AGENT_SUPPRESS_DEPRECATION_WARNINGS=1 environment variable
        - Use warnings.filterwarnings('ignore', category=SelfHealingDeprecationWarning)
        - Import from omni_mercury_engine.resilience.self_healing directly
    """

    pass


def _emit_deprecation_warning() -> None:
    """Emit deprecation warning if not suppressed."""
    if os.environ.get("MERCURY_AGENT_SUPPRESS_DEPRECATION_WARNINGS", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        return

    warnings.warn(
        "omni_mercury_engine.core.self_healing is deprecated. "
        "Import from omni_mercury_engine.resilience.self_healing instead. "
        "Set MERCURY_AGENT_SUPPRESS_DEPRECATION_WARNINGS=1 to suppress this warning.",
        SelfHealingDeprecationWarning,
        stacklevel=3,
    )


_emit_deprecation_warning()
