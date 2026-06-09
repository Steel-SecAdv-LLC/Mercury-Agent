# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Resilience module for Mercury Agent.

Provides self-healing, circuit breaker, retry logic, and health monitoring.
"""

from __future__ import annotations

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
