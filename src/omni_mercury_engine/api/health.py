"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Enhanced health check endpoints for Kubernetes and monitoring.

Implements standard health check patterns:
- Liveness probe: Is the application alive?
- Readiness probe: Is the application ready to accept traffic?
- Startup probe: Has the application started successfully?

Example:
    Add health routes to FastAPI app::

        from omni_mercury_engine.api.health import health_router

        app.include_router(health_router, prefix="/health")
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Response, status
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

# Track application start time
_start_time = time.time()


class HealthStatus(StrEnum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentStatus(StrEnum):
    """Individual component status."""

    UP = "up"
    DOWN = "down"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Health check definition.

    Attributes:
        name: Check name.
        check_fn: Async function that performs the check.
        timeout: Check timeout in seconds.
        critical: Whether failure marks system unhealthy.
        tags: Tags for categorization.
    """

    name: str
    check_fn: Callable[[], Awaitable[dict[str, Any]]]
    timeout: float = 5.0
    critical: bool = True
    tags: list[str] = field(default_factory=list)


class ComponentHealth(BaseModel):
    """Health status of a single component."""

    name: str = Field(..., description="Component name")
    status: ComponentStatus = Field(..., description="Component status")
    latency_ms: float | None = Field(None, description="Check latency in milliseconds")
    message: str | None = Field(None, description="Status message")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional details")
    last_checked: datetime = Field(default_factory=datetime.now)


class LivenessResponse(BaseModel):
    """Liveness probe response.

    Simple response indicating if the application process is alive.
    """

    status: HealthStatus = Field(..., description="Overall liveness status")
    timestamp: datetime = Field(default_factory=datetime.now)


class ReadinessResponse(BaseModel):
    """Readiness probe response.

    Indicates if the application is ready to accept traffic.
    """

    status: HealthStatus = Field(..., description="Overall readiness status")
    ready: bool = Field(..., description="Whether application is ready for traffic")
    components: list[ComponentHealth] = Field(
        default_factory=list, description="Component health statuses"
    )
    timestamp: datetime = Field(default_factory=datetime.now)


class DetailedHealthResponse(BaseModel):
    """Detailed health check response.

    Comprehensive health information for debugging and monitoring.
    """

    status: HealthStatus = Field(..., description="Overall health status")
    version: str = Field(..., description="Application version")
    uptime_seconds: float = Field(..., description="Application uptime")
    components: list[ComponentHealth] = Field(
        default_factory=list, description="Component health statuses"
    )
    system: dict[str, Any] = Field(default_factory=dict, description="System information")
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "uptime_seconds": 3600.5,
                "components": [
                    {"name": "database", "status": "up", "latency_ms": 5.2},
                    {"name": "cache", "status": "up", "latency_ms": 1.1},
                ],
                "system": {
                    "python_version": "3.11.0",
                    "platform": "linux",
                    "cpu_count": 8,
                },
                "timestamp": "2025-01-01T12:00:00Z",
            }
        }
    }


class HealthChecker:
    """Health check manager.

    Manages registration and execution of health checks.

    Example:
        >>> checker = HealthChecker()
        >>> checker.add_check("database", check_database, critical=True)
        >>> checker.add_check("cache", check_cache, critical=False)
        >>> results = await checker.run_checks()
    """

    def __init__(self, version: str = "1.5.1") -> None:
        """Initialize health checker.

        Args:
            version: Application version string.
        """
        self.version = version
        self._checks: list[HealthCheck] = []
        self._last_results: dict[str, ComponentHealth] = {}

    def add_check(
        self,
        name: str,
        check_fn: Callable[[], Awaitable[dict[str, Any]]],
        timeout: float = 5.0,
        critical: bool = True,
        tags: list[str] | None = None,
    ) -> None:
        """Add a health check.

        Args:
            name: Check name.
            check_fn: Async function that returns health info.
            timeout: Check timeout in seconds.
            critical: Whether failure marks system unhealthy.
            tags: Tags for categorization.
        """
        check = HealthCheck(
            name=name,
            check_fn=check_fn,
            timeout=timeout,
            critical=critical,
            tags=tags or [],
        )
        self._checks.append(check)

    async def run_check(self, check: HealthCheck) -> ComponentHealth:
        """Run a single health check.

        Args:
            check: Health check to run.

        Returns:
            Component health status.
        """
        start_time = time.time()

        try:
            result = await asyncio.wait_for(
                check.check_fn(),
                timeout=check.timeout,
            )

            latency = (time.time() - start_time) * 1000

            status = ComponentStatus.UP
            if result.get("status") == "degraded":
                status = ComponentStatus.DEGRADED
            elif result.get("status") == "down":
                status = ComponentStatus.DOWN

            return ComponentHealth(
                name=check.name,
                status=status,
                latency_ms=round(latency, 2),
                message=result.get("message"),
                details=result.get("details", {}),
            )

        except TimeoutError:
            latency = (time.time() - start_time) * 1000
            return ComponentHealth(
                name=check.name,
                status=ComponentStatus.DOWN,
                latency_ms=round(latency, 2),
                message=f"Check timed out after {check.timeout}s",
            )

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            logger.error("Health check '%s' failed: %s", check.name, e)
            return ComponentHealth(
                name=check.name,
                status=ComponentStatus.DOWN,
                latency_ms=round(latency, 2),
                message="Health check failed",
            )

    async def run_checks(
        self,
        tags: list[str] | None = None,
    ) -> tuple[HealthStatus, list[ComponentHealth]]:
        """Run all health checks.

        Args:
            tags: Filter checks by tags (optional).

        Returns:
            Tuple of (overall status, component statuses).
        """
        checks_to_run = self._checks
        if tags:
            checks_to_run = [c for c in self._checks if any(t in c.tags for t in tags)]

        # Run checks concurrently
        tasks = [self.run_check(check) for check in checks_to_run]
        results = await asyncio.gather(*tasks)

        # Cache results
        for result in results:
            self._last_results[result.name] = result

        # Determine overall status
        has_critical_failure = any(
            result.status == ComponentStatus.DOWN
            for result, check in zip(results, checks_to_run, strict=False)
            if check.critical
        )

        has_degradation = any(
            result.status in (ComponentStatus.DOWN, ComponentStatus.DEGRADED) for result in results
        )

        if has_critical_failure:
            overall_status = HealthStatus.UNHEALTHY
        elif has_degradation:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        return overall_status, list(results)

    def get_system_info(self) -> dict[str, Any]:
        """Get system information.

        Returns:
            Dictionary with system details.
        """
        return {
            "python_version": platform.python_version(),
            "platform": platform.system().lower(),
            "platform_release": platform.release(),
            "architecture": platform.machine(),
            "cpu_count": os.cpu_count(),
            "hostname": platform.node(),
            "pid": os.getpid(),
        }

    def get_uptime(self) -> float:
        """Get application uptime in seconds."""
        return time.time() - _start_time


# Global health checker instance
_health_checker = HealthChecker()


def get_health_checker() -> HealthChecker:
    """Get the health checker instance."""
    return _health_checker


# Default health checks
async def check_self() -> dict[str, Any]:
    """Self-check that always passes."""
    return {
        "status": "up",
        "message": "Application is running",
    }


async def check_memory() -> dict[str, Any]:
    """Check memory usage."""
    try:
        import psutil

        memory = psutil.Process().memory_info()
        memory_mb = memory.rss / (1024 * 1024)

        # Warning if using more than 1GB
        if memory_mb > 1024:
            return {
                "status": "degraded",
                "message": f"High memory usage: {memory_mb:.1f} MB",
                "details": {"memory_mb": memory_mb},
            }

        return {
            "status": "up",
            "details": {"memory_mb": round(memory_mb, 1)},
        }
    except ImportError:
        return {
            "status": "up",
            "message": "psutil not available for memory check",
        }


async def check_disk() -> dict[str, Any]:
    """Check disk space."""
    try:
        import shutil

        total, used, free = shutil.disk_usage("/")
        free_gb = free / (1024**3)
        used_percent = (used / total) * 100

        if free_gb < 1:
            return {
                "status": "degraded",
                "message": f"Low disk space: {free_gb:.1f} GB free",
                "details": {"free_gb": free_gb, "used_percent": used_percent},
            }

        return {
            "status": "up",
            "details": {
                "free_gb": round(free_gb, 1),
                "used_percent": round(used_percent, 1),
            },
        }
    except Exception as e:
        return {
            "status": "unknown",
            "message": f"Could not check disk: {e}",
        }


# Register default checks
_health_checker.add_check("self", check_self, critical=True, tags=["core"])
_health_checker.add_check("memory", check_memory, critical=False, tags=["system"])
_health_checker.add_check("disk", check_disk, critical=False, tags=["system"])


# FastAPI Router
health_router = APIRouter(tags=["Health"])


@health_router.get(
    "/live",
    response_model=LivenessResponse,
    summary="Liveness Probe",
    description="Kubernetes liveness probe. Returns 200 if the application is alive.",
    responses={
        200: {"description": "Application is alive"},
        503: {"description": "Application is not alive"},
    },
)
async def liveness_probe() -> LivenessResponse:
    """Liveness probe endpoint.

    This endpoint is called by Kubernetes to determine if the
    application should be restarted.

    Returns:
        LivenessResponse indicating application is alive.
    """
    return LivenessResponse(status=HealthStatus.HEALTHY)


@health_router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness Probe",
    description="Kubernetes readiness probe. Returns 200 if ready to accept traffic.",
    responses={
        200: {"description": "Application is ready"},
        503: {"description": "Application is not ready"},
    },
)
async def readiness_probe(response: Response) -> ReadinessResponse:
    """Readiness probe endpoint.

    This endpoint is called by Kubernetes to determine if the
    application should receive traffic.

    Returns:
        ReadinessResponse with component statuses.
    """
    checker = get_health_checker()
    overall_status, components = await checker.run_checks(tags=["core"])

    ready = overall_status != HealthStatus.UNHEALTHY

    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        status=overall_status,
        ready=ready,
        components=components,
    )


@health_router.get(
    "/startup",
    response_model=LivenessResponse,
    summary="Startup Probe",
    description="Kubernetes startup probe. Returns 200 if application has started.",
    responses={
        200: {"description": "Application has started"},
        503: {"description": "Application is still starting"},
    },
)
async def startup_probe() -> LivenessResponse:
    """Startup probe endpoint.

    This endpoint is called by Kubernetes during startup to
    give slow-starting applications time to initialize.

    Returns:
        LivenessResponse indicating startup status.
    """
    # Check if minimum startup time has passed
    uptime = get_health_checker().get_uptime()
    if uptime < 1.0:  # Allow at least 1 second for startup
        return LivenessResponse(status=HealthStatus.UNHEALTHY)

    return LivenessResponse(status=HealthStatus.HEALTHY)


@health_router.get(
    "/detailed",
    response_model=DetailedHealthResponse,
    summary="Detailed Health Check",
    description="Comprehensive health check with component statuses and system info.",
    responses={
        200: {"description": "Health check completed"},
        503: {"description": "System is unhealthy"},
    },
)
async def detailed_health(response: Response) -> DetailedHealthResponse:
    """Detailed health check endpoint.

    Provides comprehensive health information including:
    - All component statuses
    - System information
    - Uptime and version

    Returns:
        DetailedHealthResponse with full health details.
    """
    checker = get_health_checker()
    overall_status, components = await checker.run_checks()

    if overall_status == HealthStatus.UNHEALTHY:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return DetailedHealthResponse(
        status=overall_status,
        version=checker.version,
        uptime_seconds=round(checker.get_uptime(), 2),
        components=components,
        system=checker.get_system_info(),
    )


@health_router.get(
    "/metrics",
    summary="Health Metrics",
    description="Prometheus-compatible metrics endpoint.",
    response_class=Response,
)
async def health_metrics() -> Response:
    """Export health metrics in Prometheus format.

    Returns:
        Plain text metrics in Prometheus exposition format.
    """
    checker = get_health_checker()
    overall_status, components = await checker.run_checks()

    lines = [
        "# HELP omni_mercury_up Application up status (1=up, 0=down)",
        "# TYPE omni_mercury_up gauge",
        f"omni_mercury_up {1 if overall_status != HealthStatus.UNHEALTHY else 0}",
        "",
        "# HELP omni_mercury_uptime_seconds Application uptime in seconds",
        "# TYPE omni_mercury_uptime_seconds counter",
        f"omni_mercury_uptime_seconds {checker.get_uptime():.2f}",
        "",
        "# HELP omni_mercury_component_status Component health status (1=up, 0.5=degraded, 0=down)",
        "# TYPE omni_mercury_component_status gauge",
    ]

    for component in components:
        value = (
            1
            if component.status == ComponentStatus.UP
            else (0.5 if component.status == ComponentStatus.DEGRADED else 0)
        )
        lines.append(f'omni_mercury_component_status{{component="{component.name}"}} {value}')

    lines.append("")
    lines.append("# HELP omni_mercury_component_latency_ms Component check latency in milliseconds")
    lines.append("# TYPE omni_mercury_component_latency_ms gauge")

    for component in components:
        if component.latency_ms is not None:
            metric = f'omni_mercury_component_latency_ms{{component="{component.name}"}}'
            lines.append(f"{metric} {component.latency_ms}")

    content = "\n".join(lines) + "\n"
    return Response(
        content=content,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
