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
Health monitoring for components and agents
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class HealthMetrics:
    """Health metrics for a component"""

    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    response_time: float = 0.0
    error_rate: float = 0.0
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now()


class HealthMonitor:
    """Monitor health of components and agents"""

    def __init__(self) -> None:
        self.metrics: dict[str, list[HealthMetrics]] = {}

    def record_metrics(self, component_name: str, metrics: HealthMetrics) -> None:
        """Record health metrics for a component"""
        if component_name not in self.metrics:
            self.metrics[component_name] = []

        self.metrics[component_name].append(metrics)

        if len(self.metrics[component_name]) > 1000:
            self.metrics[component_name] = self.metrics[component_name][-1000:]

    def get_current_health(self, component_name: str) -> dict[str, Any]:
        """Get current health status of a component"""
        if component_name not in self.metrics or not self.metrics[component_name]:
            return {"status": "unknown"}

        latest = self.metrics[component_name][-1]

        is_healthy = (
            latest.cpu_usage < 0.9 and latest.memory_usage < 0.9 and latest.error_rate < 0.1
        )

        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "metrics": {
                "cpu_usage": latest.cpu_usage,
                "memory_usage": latest.memory_usage,
                "response_time": latest.response_time,
                "error_rate": latest.error_rate,
            },
            "timestamp": latest.timestamp.isoformat() if latest.timestamp is not None else None,
        }

    def get_ecosystem_health(self) -> dict[str, Any]:
        """Get overall ecosystem health"""
        component_health = {name: self.get_current_health(name) for name in self.metrics}

        healthy_count = sum(1 for h in component_health.values() if h.get("status") == "healthy")

        total_count = len(component_health)

        return {
            "overall_status": ("healthy" if healthy_count == total_count else "degraded"),
            "healthy_components": healthy_count,
            "total_components": total_count,
            "components": component_health,
        }
