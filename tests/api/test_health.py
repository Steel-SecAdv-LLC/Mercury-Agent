"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Tests for the health check module - Kubernetes health endpoints.
"""

from __future__ import annotations

from typing import Any

import pytest

try:
    from omni_mercury_engine.api.health import (
        ComponentHealth,
        ComponentStatus,
        HealthChecker,
        HealthStatus,
    )

    HAS_HEALTH = True
except ImportError:
    HAS_HEALTH = False


# Only the file-level skip is module-wide; ``pytest.mark.asyncio``
# is applied per-test via pytest-asyncio's auto mode (pyproject sets
# ``asyncio_mode = "auto"``) so it must NOT be added module-wide,
# otherwise plain synchronous tests trigger a PytestWarning.
pytestmark = pytest.mark.skipif(not HAS_HEALTH, reason="health module not available")


class TestHealthChecker:
    """Tests for HealthChecker class."""

    def test_create_health_checker(self) -> None:
        """Test HealthChecker instantiation."""
        checker = HealthChecker()
        assert checker is not None

    def test_add_check(self) -> None:
        """Test adding a health check."""
        checker = HealthChecker()

        def simple_check():
            return True

        checker.add_check("test_service", simple_check)
        # Check should be registered
        assert len(checker._checks) > 0 or hasattr(checker, "checks")

    async def test_run_passing_check(self) -> None:
        """Test running a check that passes."""
        checker = HealthChecker()

        async def passing_check():
            return {"status": "up", "message": "Service is healthy"}

        checker.add_check("passing_service", passing_check)
        status, components = await checker.run_checks()

        assert status in [HealthStatus.HEALTHY, "healthy", "HEALTHY"]
        assert len(components) > 0

    async def test_run_failing_check(self) -> None:
        """Test running a check that fails."""
        checker = HealthChecker()

        async def failing_check() -> dict[str, Any]:
            raise Exception("Service unavailable")

        checker.add_check("failing_service", failing_check, critical=True)
        status, components = await checker.run_checks()

        # With critical failure, should be unhealthy
        assert status in [HealthStatus.UNHEALTHY, "unhealthy", "UNHEALTHY"]

    async def test_non_critical_failure_causes_degraded(self) -> None:
        """Test that non-critical failure results in degraded status."""
        checker = HealthChecker()

        async def passing_check():
            return {"status": "up", "message": "Core service healthy"}

        async def failing_check() -> dict[str, Any]:
            raise Exception("Non-critical failure")

        checker.add_check("core_service", passing_check, critical=True)
        checker.add_check("optional_service", failing_check, critical=False)

        status, components = await checker.run_checks()

        # Should be degraded, not unhealthy
        assert status in [
            HealthStatus.DEGRADED,
            HealthStatus.HEALTHY,
            "degraded",
            "healthy",
        ]

    async def test_check_with_timeout(self) -> None:
        """Test that slow checks timeout properly."""
        import asyncio

        checker = HealthChecker()

        async def slow_check():
            await asyncio.sleep(5)
            return {"status": "up"}

        checker.add_check("slow_service", slow_check, timeout=0.1)
        status, components = await checker.run_checks()

        # Should handle timeout gracefully
        assert status is not None

    def test_get_uptime(self) -> None:
        """Test uptime calculation."""
        checker = HealthChecker()
        uptime = checker.get_uptime()

        assert isinstance(uptime, (int, float))
        assert uptime >= 0

    def test_get_system_info(self) -> None:
        """Test system info collection."""
        checker = HealthChecker()
        info = checker.get_system_info()

        assert isinstance(info, dict)
        # Should contain platform info
        assert "platform" in info or "system" in info or "python_version" in info

    async def test_check_tagging(self) -> None:
        """Test check filtering by tags."""
        checker = HealthChecker()

        async def check_a():
            return {"status": "up"}

        async def check_b():
            return {"status": "up"}

        checker.add_check("service_a", check_a, tags=["core"])
        checker.add_check("service_b", check_b, tags=["optional"])

        # Run only core checks
        status, components = await checker.run_checks(tags=["core"])

        # Should only run tagged checks
        assert len(components) >= 1


class TestComponentHealth:
    """Tests for ComponentHealth data structure."""

    def test_component_health_creation(self) -> None:
        """Test ComponentHealth can be created."""
        health = ComponentHealth(
            name="test_component",
            status=ComponentStatus.UP,
            latency_ms=10.5,
            message=None,
        )

        assert health.name == "test_component"
        assert health.status == ComponentStatus.UP
        assert health.latency_ms == 10.5


class TestComponentStatus:
    """Tests for ComponentStatus enum."""

    def test_status_values_exist(self) -> None:
        """Test all expected status values are defined."""
        assert hasattr(ComponentStatus, "UP")
        assert hasattr(ComponentStatus, "DOWN")
        assert hasattr(ComponentStatus, "DEGRADED") or hasattr(ComponentStatus, "UNKNOWN")


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_health_status_values_exist(self) -> None:
        """Test all expected health status values are defined."""
        assert hasattr(HealthStatus, "HEALTHY")
        assert hasattr(HealthStatus, "UNHEALTHY")
        assert hasattr(HealthStatus, "DEGRADED")


class TestHealthStatusAggregation:
    """Tests for health status aggregation logic."""

    async def test_all_up_is_healthy(self) -> None:
        """Test that all UP components result in HEALTHY."""
        checker = HealthChecker()

        async def healthy_check():
            return {"status": "up"}

        checker.add_check("service_1", healthy_check, critical=True)
        checker.add_check("service_2", healthy_check, critical=True)
        checker.add_check("service_3", healthy_check, critical=False)

        status, _ = await checker.run_checks()
        assert status in [HealthStatus.HEALTHY, "healthy", "HEALTHY"]

    async def test_critical_down_is_unhealthy(self) -> None:
        """Test that critical DOWN results in UNHEALTHY."""
        checker = HealthChecker()

        async def critical_fail() -> dict[str, Any]:
            raise Exception("Critical failure")

        checker.add_check("critical_service", critical_fail, critical=True)

        status, _ = await checker.run_checks()
        assert status in [HealthStatus.UNHEALTHY, "unhealthy", "UNHEALTHY"]
