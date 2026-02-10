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
"""

from __future__ import annotations

"""
Test resilience modules
"""

import time

from omni_mercury_engine.resilience.circuit_breaker import CircuitBreaker
from omni_mercury_engine.resilience.health_monitoring import HealthMonitor
from omni_mercury_engine.resilience.retry import RetryPolicy
from omni_mercury_engine.resilience.self_healing import SelfHealingEngine


def test_circuit_breaker_closed_state():
    """Test circuit breaker in closed state"""
    from omni_mercury_engine.resilience.circuit_breaker import CircuitState

    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1)

    def success_func():
        return "success"

    result = breaker.call(success_func)
    assert result == "success"
    assert breaker.state == CircuitState.CLOSED


def test_circuit_breaker_open_state():
    """Test circuit breaker transitions to open state"""
    from omni_mercury_engine.resilience.circuit_breaker import CircuitState

    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1)

    def fail_func():
        raise Exception("Test failure")

    for _ in range(3):
        try:
            breaker.call(fail_func)
        except Exception:
            pass  # Expected: testing circuit breaker failure counting

    assert breaker.state == CircuitState.OPEN


def test_circuit_breaker_half_open_state():
    """Test circuit breaker half-open state"""
    from omni_mercury_engine.resilience.circuit_breaker import CircuitState

    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

    def fail_func():
        raise Exception("Test failure")

    for _ in range(3):
        try:
            breaker.call(fail_func)
        except Exception:
            pass  # Expected: testing circuit breaker failure counting

    time.sleep(0.2)

    def success_func():
        return "recovered"

    result = breaker.call(success_func)
    assert result == "recovered"
    assert breaker.state == CircuitState.CLOSED


def test_retry_policy_success():
    """Test retry policy on success"""
    counter = {"calls": 0}
    retry = RetryPolicy(max_retries=3)

    @retry
    def eventually_succeeds():
        counter["calls"] += 1
        if counter["calls"] < 2:
            raise Exception("Temporary failure")
        return "success"

    result = eventually_succeeds()
    assert result == "success"
    assert counter["calls"] == 2


def test_retry_policy_failure():
    """Test retry policy exhausts retries"""
    counter = {"calls": 0}
    retry = RetryPolicy(max_retries=2)

    @retry
    def always_fails():
        counter["calls"] += 1
        raise Exception("Permanent failure")

    try:
        always_fails()
        raise AssertionError("Should have raised exception")
    except Exception as e:
        assert "Permanent failure" in str(e)
        assert counter["calls"] == 3


def test_retry_policy_custom_config():
    """Test retry policy with custom configuration"""
    retry = RetryPolicy(
        max_retries=5,
        base_delay=0.5,
        max_delay=30.0,
        exponential_base=3.0,
    )
    assert retry.max_retries == 5
    assert retry.base_delay == 0.5


def test_health_monitor_initialization():
    """Test health monitor initialization"""
    monitor = HealthMonitor()
    assert monitor is not None
    assert hasattr(monitor, "metrics")


def test_health_monitor_record_metrics():
    """Test recording health metrics"""
    from omni_mercury_engine.resilience.health_monitoring import HealthMetrics

    monitor = HealthMonitor()
    metrics = HealthMetrics(cpu_usage=0.5, memory_usage=0.6, response_time=0.1, error_rate=0.01)

    monitor.record_metrics("test_component", metrics)
    assert "test_component" in monitor.metrics
    assert len(monitor.metrics["test_component"]) == 1


def test_health_monitor_get_current_health():
    """Test getting current health status"""
    from omni_mercury_engine.resilience.health_monitoring import HealthMetrics

    monitor = HealthMonitor()
    metrics = HealthMetrics(cpu_usage=0.5, memory_usage=0.6, response_time=0.1, error_rate=0.01)

    monitor.record_metrics("test_component", metrics)
    health = monitor.get_current_health("test_component")

    assert "status" in health
    assert health["status"] in ["healthy", "unhealthy", "unknown"]


def test_self_healing_initialization():
    """Test self-healing engine initialization"""
    healer = SelfHealingEngine()
    assert healer is not None
    assert hasattr(healer, "components")
    assert hasattr(healer, "circuit_breakers")


def test_self_healing_register_component():
    """Test component registration"""
    healer = SelfHealingEngine()

    def health_check():
        return True

    def recovery_action():
        pass

    healer.register_component(
        "test_component", health_check=health_check, recovery_action=recovery_action
    )

    assert "test_component" in healer.components
    assert "test_component" in healer.circuit_breakers


def test_self_healing_check_health():
    """Test health check"""
    healer = SelfHealingEngine()

    def health_check():
        return True

    healer.register_component("test_component", health_check=health_check)
    is_healthy = healer.check_health("test_component")

    assert is_healthy is True


def test_self_healing_attempt_recovery():
    """Test recovery attempt"""
    healer = SelfHealingEngine()
    recovery_called = {"value": False}

    def health_check():
        return recovery_called["value"]

    def recovery_action():
        recovery_called["value"] = True

    healer.register_component(
        "test_component", health_check=health_check, recovery_action=recovery_action
    )

    result = healer.attempt_recovery("test_component")
    assert result is True
    assert recovery_called["value"] is True
