"""
OMNI ♱ AVA (O♱A)
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

"""
Self-healing engine for autonomous error recovery
"""

from typing import Dict, Any, Optional, Callable
import logging
from omni_anomaly_engine.resilience.circuit_breaker import CircuitBreaker


class SelfHealingEngine:
    """
    Self-healing system for autonomous error recovery.

    Features:
    - Automatic error detection
    - Component health monitoring
    - Graceful degradation
    """

    def __init__(self):
        self.components: Dict[str, Dict[str, Any]] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.logger = logging.getLogger(__name__)

    def register_component(
        self,
        name: str,
        health_check: Callable[[], bool],
        recovery_action: Optional[Callable[[], None]] = None,
    ) -> None:
        """Register a component for health monitoring"""
        self.components[name] = {
            "health_check": health_check,
            "recovery_action": recovery_action,
            "status": "healthy",
        }

        self.circuit_breakers[name] = CircuitBreaker()

    def check_health(self, component_name: str) -> bool:
        """Check health of a component"""
        if component_name not in self.components:
            return False

        component = self.components[component_name]

        try:
            is_healthy = component["health_check"]()
            component["status"] = "healthy" if is_healthy else "unhealthy"
            return is_healthy
        except Exception as e:
            self.logger.error(f"Health check failed for {component_name}: {e}")
            component["status"] = "unhealthy"
            return False

    def attempt_recovery(self, component_name: str) -> bool:
        """Attempt to recover a component"""
        if component_name not in self.components:
            return False

        component = self.components[component_name]
        recovery_action = component.get("recovery_action")

        if recovery_action is None:
            return False

        try:
            recovery_action()
            return self.check_health(component_name)
        except Exception as e:
            self.logger.error(f"Recovery failed for {component_name}: {e}")
            return False

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        health_status = {}

        for name, component in self.components.items():
            is_healthy = self.check_health(name)
            health_status[name] = {
                "status": component["status"],
                "is_healthy": is_healthy,
            }

        all_healthy = all(status["is_healthy"] for status in health_status.values())

        return {
            "overall_health": "healthy" if all_healthy else "degraded",
            "components": health_status,
        }
