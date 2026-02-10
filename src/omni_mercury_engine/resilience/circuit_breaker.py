"""
Mercury Agent ♱
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

Circuit Breaker Pattern - Backwards Compatibility Module

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
