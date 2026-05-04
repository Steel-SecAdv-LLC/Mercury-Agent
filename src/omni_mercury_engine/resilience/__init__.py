"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see
https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
Resilience module for Mercury Agent

Provides self-healing, circuit breaker, retry logic, and health monitoring.
"""

from omni_mercury_engine.resilience.api_circuit_breakers import (
    DataLoaderCircuitBreaker,
    DetectorCircuitBreaker,
    ExternalIntegrationCircuitBreaker,
    get_all_breaker_stats,
    get_data_loader_breaker,
    get_detector_breaker,
    get_integration_breaker,
    get_open_breakers,
    reset_all_breakers,
    with_circuit_breaker,
)
from omni_mercury_engine.resilience.circuit_breaker import CircuitBreaker
from omni_mercury_engine.resilience.health_monitoring import HealthMonitor
from omni_mercury_engine.resilience.retry import RetryPolicy
from omni_mercury_engine.resilience.self_healing import (
    AdaptiveDefenseSystem,
    AnomalySignature,
    SelfHealingEngine,
)

__all__ = [
    "AdaptiveDefenseSystem",
    "AnomalySignature",
    "CircuitBreaker",
    "DataLoaderCircuitBreaker",
    "DetectorCircuitBreaker",
    "ExternalIntegrationCircuitBreaker",
    "HealthMonitor",
    "RetryPolicy",
    "SelfHealingEngine",
    "get_all_breaker_stats",
    "get_data_loader_breaker",
    "get_detector_breaker",
    "get_integration_breaker",
    "get_open_breakers",
    "reset_all_breakers",
    "with_circuit_breaker",
]
