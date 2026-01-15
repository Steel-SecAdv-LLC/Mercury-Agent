"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

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
"""

from __future__ import annotations


"""
Pre-configured circuit breakers for external API integrations.

This module provides circuit breaker instances optimized for different
types of external integrations:
- Data loader API calls (NOAA, USGS, etc.)
- Detector invocations in parallel execution
- External integration endpoints

Each circuit breaker is configured with appropriate thresholds and
exponential backoff settings for its use case.
"""

import logging
from functools import wraps
from typing import TYPE_CHECKING, Any

from omni_mercury_engine.resilience.circuit_breaker import CircuitBreaker, CircuitState


if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class DataLoaderCircuitBreaker(CircuitBreaker):
    """
    Circuit breaker optimized for data loader API calls.

    Configured with:
    - 3 failure threshold (APIs may have intermittent issues)
    - 30 second recovery timeout (allow quick retries)
    - Exponential backoff enabled (prevent API overload)
    - Max 5 minute backoff (don't wait too long)
    """

    def __init__(self) -> None:
        super().__init__(
            failure_threshold=3,
            recovery_timeout=30,
            expected_exception=Exception,
            enable_exponential_backoff=True,
            backoff_base=2.0,
            max_backoff_timeout=300.0,
            success_threshold=2,
            enable_jitter=True,
            jitter_factor=0.1,
        )


class DetectorCircuitBreaker(CircuitBreaker):
    """
    Circuit breaker optimized for detector invocations.

    Configured with:
    - 5 failure threshold (detectors should be more stable)
    - 60 second recovery timeout
    - Exponential backoff enabled
    - Max 10 minute backoff
    """

    def __init__(self) -> None:
        super().__init__(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=Exception,
            enable_exponential_backoff=True,
            backoff_base=2.0,
            max_backoff_timeout=600.0,
            success_threshold=1,
            enable_jitter=True,
            jitter_factor=0.05,
        )


class ExternalIntegrationCircuitBreaker(CircuitBreaker):
    """
    Circuit breaker optimized for external integration endpoints.

    Configured with:
    - 3 failure threshold (external services may be unreliable)
    - 45 second recovery timeout
    - Exponential backoff enabled
    - Max 15 minute backoff (external services may need time)
    """

    def __init__(self) -> None:
        super().__init__(
            failure_threshold=3,
            recovery_timeout=45,
            expected_exception=Exception,
            enable_exponential_backoff=True,
            backoff_base=2.0,
            max_backoff_timeout=900.0,
            success_threshold=2,
            enable_jitter=True,
            jitter_factor=0.15,
        )


_data_loader_breakers: dict[str, DataLoaderCircuitBreaker] = {}
_detector_breakers: dict[str, DetectorCircuitBreaker] = {}
_integration_breakers: dict[str, ExternalIntegrationCircuitBreaker] = {}


def get_data_loader_breaker(name: str) -> DataLoaderCircuitBreaker:
    """
    Get or create a circuit breaker for a data loader.

    Args:
        name: Unique identifier for the data loader (e.g., "noaa_space_weather")

    Returns:
        Circuit breaker instance for the data loader
    """
    if name not in _data_loader_breakers:
        _data_loader_breakers[name] = DataLoaderCircuitBreaker()
        logger.debug(f"Created data loader circuit breaker: {name}")
    return _data_loader_breakers[name]


def get_detector_breaker(name: str) -> DetectorCircuitBreaker:
    """
    Get or create a circuit breaker for a detector.

    Args:
        name: Unique identifier for the detector (e.g., "tsunami_detector")

    Returns:
        Circuit breaker instance for the detector
    """
    if name not in _detector_breakers:
        _detector_breakers[name] = DetectorCircuitBreaker()
        logger.debug(f"Created detector circuit breaker: {name}")
    return _detector_breakers[name]


def get_integration_breaker(name: str) -> ExternalIntegrationCircuitBreaker:
    """
    Get or create a circuit breaker for an external integration.

    Args:
        name: Unique identifier for the integration (e.g., "ava_guardian")

    Returns:
        Circuit breaker instance for the integration
    """
    if name not in _integration_breakers:
        _integration_breakers[name] = ExternalIntegrationCircuitBreaker()
        logger.debug(f"Created integration circuit breaker: {name}")
    return _integration_breakers[name]


def with_circuit_breaker(
    breaker_type: str = "data_loader",
    name: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to wrap a function with circuit breaker protection.

    Args:
        breaker_type: Type of circuit breaker ("data_loader", "detector", "integration")
        name: Unique identifier for the circuit breaker (defaults to function name)

    Returns:
        Decorated function with circuit breaker protection

    Example:
        @with_circuit_breaker("data_loader", "noaa_api")
        def fetch_noaa_data():
            ...
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        breaker_name = name or func.__name__

        if breaker_type == "data_loader":
            breaker = get_data_loader_breaker(breaker_name)
        elif breaker_type == "detector":
            breaker = get_detector_breaker(breaker_name)
        elif breaker_type == "integration":
            breaker = get_integration_breaker(breaker_name)
        else:
            raise ValueError(f"Unknown breaker type: {breaker_type}")

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return breaker.call(func, *args, **kwargs)

        wrapper.circuit_breaker = breaker
        return wrapper

    return decorator


def get_all_breaker_stats() -> dict[str, dict[str, Any]]:
    """
    Get statistics for all circuit breakers.

    Returns:
        Dictionary mapping breaker names to their statistics
    """
    stats: dict[str, dict[str, Any]] = {}

    for name, breaker in _data_loader_breakers.items():
        stats[f"data_loader:{name}"] = breaker.get_stats()

    for name, breaker in _detector_breakers.items():
        stats[f"detector:{name}"] = breaker.get_stats()

    for name, breaker in _integration_breakers.items():
        stats[f"integration:{name}"] = breaker.get_stats()

    return stats


def reset_all_breakers() -> None:
    """Reset all circuit breakers to closed state."""
    for breaker in _data_loader_breakers.values():
        breaker.reset()
    for breaker in _detector_breakers.values():
        breaker.reset()
    for breaker in _integration_breakers.values():
        breaker.reset()
    logger.info("All circuit breakers reset")


def get_open_breakers() -> list[str]:
    """
    Get list of circuit breakers that are currently open.

    Returns:
        List of breaker names that are in OPEN state
    """
    open_breakers: list[str] = []

    for name, breaker in _data_loader_breakers.items():
        if breaker.state == CircuitState.OPEN:
            open_breakers.append(f"data_loader:{name}")

    for name, breaker in _detector_breakers.items():
        if breaker.state == CircuitState.OPEN:
            open_breakers.append(f"detector:{name}")

    for name, breaker in _integration_breakers.items():
        if breaker.state == CircuitState.OPEN:
            open_breakers.append(f"integration:{name}")

    return open_breakers
