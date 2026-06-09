# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Circuit Breaker Pattern - Backwards Compatibility Module.

This module re-exports CircuitBreaker from utils.resilience for backwards compatibility.
New code should import directly from omni_mercury_engine.utils.resilience.

Deprecated:
    Import from omni_mercury_engine.utils.resilience instead.
"""

from __future__ import annotations

# Re-export from canonical location
from omni_mercury_engine.utils.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
)

# Import CircuitState from canonical types module
try:
    from omni_mercury_engine.core.types import CircuitState
except ImportError:
    # Fallback for backwards compatibility
    from omni_mercury_engine.utils.resilience import CircuitState

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerOpenError",
    "CircuitState",
]
