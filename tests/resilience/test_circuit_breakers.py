"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

Tests for the api_circuit_breakers module - circuit breaker pattern.
"""

from __future__ import annotations

import time

import pytest

try:
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

    HAS_CIRCUIT_BREAKER = True
except ImportError:
    HAS_CIRCUIT_BREAKER = False


pytestmark = pytest.mark.skipif(not HAS_CIRCUIT_BREAKER, reason="circuit_breakers not available")


class TestDataLoaderCircuitBreaker:
    """Tests for DataLoaderCircuitBreaker."""

    def test_create_breaker(self):
        """Test breaker instantiation."""
        breaker = DataLoaderCircuitBreaker()
        assert breaker is not None
        assert breaker.failure_threshold == 3

    def test_initial_state_closed(self):
        """Test breaker starts in CLOSED state."""
        breaker = DataLoaderCircuitBreaker()
        from omni_mercury_engine.resilience.circuit_breaker import CircuitState

        assert breaker.state == CircuitState.CLOSED

    def test_successful_call_stays_closed(self):
        """Test successful calls keep breaker closed."""
        breaker = DataLoaderCircuitBreaker()
        from omni_mercury_engine.resilience.circuit_breaker import CircuitState

        def success_fn():
            return "data"

        result = breaker.call(success_fn)
        assert result == "data"
        assert breaker.state == CircuitState.CLOSED

    def test_failures_open_circuit(self):
        """Test that 3 failures open the circuit."""
        breaker = DataLoaderCircuitBreaker()
        breaker.reset()  # Ensure clean state
        from omni_mercury_engine.resilience.circuit_breaker import CircuitState

        def failing_fn():
            raise Exception("API error")

        # DataLoaderCircuitBreaker has failure_threshold of 3
        for _ in range(3):
            try:
                breaker.call(failing_fn)
            except Exception:
                pass

        assert breaker.state == CircuitState.OPEN

    def test_open_circuit_blocks_calls(self):
        """Test that open circuit blocks further calls."""
        breaker = DataLoaderCircuitBreaker()
        breaker.reset()

        def failing_fn():
            raise Exception("API error")

        # Open the circuit
        for _ in range(3):
            try:
                breaker.call(failing_fn)
            except Exception:
                pass

        # Now calls should be blocked
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            breaker.call(lambda: "should not run")


class TestDetectorCircuitBreaker:
    """Tests for DetectorCircuitBreaker."""

    def test_create_breaker(self):
        """Test breaker instantiation."""
        breaker = DetectorCircuitBreaker()
        assert breaker is not None
        assert breaker.failure_threshold == 5

    def test_higher_failure_threshold(self):
        """Test DetectorCircuitBreaker has 5 failure threshold."""
        breaker = DetectorCircuitBreaker()
        breaker.reset()
        from omni_mercury_engine.resilience.circuit_breaker import CircuitState

        def failing_fn():
            raise Exception("Detector error")

        # Should still be closed after 4 failures
        for _ in range(4):
            try:
                breaker.call(failing_fn)
            except Exception:
                pass

        # Should still be closed (threshold is 5)
        assert breaker.state == CircuitState.CLOSED or breaker.failure_count < 5

        # 5th failure should open it
        try:
            breaker.call(failing_fn)
        except Exception:
            pass

        assert breaker.state == CircuitState.OPEN


class TestExternalIntegrationCircuitBreaker:
    """Tests for ExternalIntegrationCircuitBreaker."""

    def test_create_breaker(self):
        """Test breaker instantiation."""
        breaker = ExternalIntegrationCircuitBreaker()
        assert breaker is not None
        assert breaker.failure_threshold == 3

    def test_three_failure_threshold(self):
        """Test ExternalIntegrationCircuitBreaker has 3 failure threshold."""
        breaker = ExternalIntegrationCircuitBreaker()
        breaker.reset()
        from omni_mercury_engine.resilience.circuit_breaker import CircuitState

        def failing_fn():
            raise Exception("External API error")

        # 3 failures should open circuit
        for _ in range(3):
            try:
                breaker.call(failing_fn)
            except Exception:
                pass

        assert breaker.state == CircuitState.OPEN


class TestCircuitBreakerRecovery:
    """Tests for circuit breaker recovery behavior."""

    def test_half_open_after_timeout(self):
        """Test circuit transitions to HALF_OPEN after timeout."""
        breaker = DataLoaderCircuitBreaker()
        breaker.reset()
        # Override timeout for faster testing
        breaker.recovery_timeout = 0.1
        from omni_mercury_engine.resilience.circuit_breaker import CircuitState

        def failing_fn():
            raise Exception("Error")

        # Open the circuit
        for _ in range(3):
            try:
                breaker.call(failing_fn)
            except Exception:
                pass

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.15)

        # Next call attempt should transition to HALF_OPEN
        # (The call itself may succeed or fail, but state should change)
        try:
            breaker.call(lambda: "test")
        except Exception:
            pass

        # State should be HALF_OPEN or CLOSED (if call succeeded)
        assert breaker.state in [CircuitState.HALF_OPEN, CircuitState.CLOSED]

    def test_success_in_half_open_closes_circuit(self):
        """Test successful call in HALF_OPEN closes circuit."""
        breaker = DataLoaderCircuitBreaker()
        breaker.reset()
        breaker.recovery_timeout = 0.1
        from omni_mercury_engine.resilience.circuit_breaker import CircuitState

        def failing_fn():
            raise Exception("Error")

        # Open the circuit
        for _ in range(3):
            try:
                breaker.call(failing_fn)
            except Exception:
                pass

        # Wait for recovery
        time.sleep(0.15)

        # Successful call should close circuit
        result = breaker.call(lambda: "success")
        assert result == "success"

        # May need multiple successes depending on implementation
        # but should eventually close
        for _ in range(3):
            try:
                breaker.call(lambda: "success")
            except Exception:
                pass

        assert breaker.state == CircuitState.CLOSED


class TestBreakerFactoryFunctions:
    """Tests for factory functions."""

    def test_get_data_loader_breaker(self):
        """Test get_data_loader_breaker returns correct type."""
        breaker = get_data_loader_breaker("noaa_api")
        assert breaker is not None
        assert isinstance(breaker, DataLoaderCircuitBreaker)

    def test_get_detector_breaker(self):
        """Test get_detector_breaker returns correct type."""
        breaker = get_detector_breaker("anomaly_detector")
        assert breaker is not None
        assert isinstance(breaker, DetectorCircuitBreaker)

    def test_get_integration_breaker(self):
        """Test get_integration_breaker returns correct type."""
        breaker = get_integration_breaker("external_api")
        assert breaker is not None
        assert isinstance(breaker, ExternalIntegrationCircuitBreaker)

    def test_breaker_caching(self):
        """Test that same name returns same breaker instance."""
        breaker1 = get_data_loader_breaker("cached_breaker")
        breaker2 = get_data_loader_breaker("cached_breaker")
        assert breaker1 is breaker2


class TestBreakerDecorator:
    """Tests for the circuit breaker decorator."""

    def test_decorator_wraps_function(self):
        """Test decorator properly wraps function."""

        @with_circuit_breaker("data_loader", "test_api")
        def api_call():
            return "data"

        result = api_call()
        assert result == "data"

    def test_decorator_attaches_breaker(self):
        """Test decorator attaches breaker to function."""

        @with_circuit_breaker("data_loader", "test_attached")
        def api_call():
            return "data"

        assert hasattr(api_call, "circuit_breaker")
        assert api_call.circuit_breaker is not None


class TestBreakerStatistics:
    """Tests for statistics and monitoring functions."""

    def test_get_all_breaker_stats(self):
        """Test retrieving all breaker statistics."""
        # Create some breakers
        get_data_loader_breaker("stats_test_1")
        get_detector_breaker("stats_test_2")

        stats = get_all_breaker_stats()

        assert isinstance(stats, dict)
        assert len(stats) > 0

    def test_get_open_breakers(self):
        """Test retrieving list of open breakers."""
        reset_all_breakers()

        open_breakers = get_open_breakers()

        # Initially should be empty
        assert isinstance(open_breakers, list)

    def test_reset_all_breakers(self):
        """Test resetting all breakers."""
        from omni_mercury_engine.resilience.circuit_breaker import CircuitState

        # Create and open a breaker
        breaker = get_data_loader_breaker("reset_test")

        def failing_fn():
            raise Exception("Error")

        for _ in range(3):
            try:
                breaker.call(failing_fn)
            except Exception:
                pass

        # Reset all
        reset_all_breakers()

        # Should be back to closed
        breaker_after = get_data_loader_breaker("reset_test")
        assert breaker_after.state == CircuitState.CLOSED
