# Copyright (C) 2025 Steel Security Advisors LLC
"""(at your option) any later version."""

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
