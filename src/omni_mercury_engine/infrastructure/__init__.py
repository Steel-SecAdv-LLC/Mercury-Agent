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
Critical Infrastructure Anomaly Detection with Multi-Framework Integration

Monitors CISA critical infrastructure sectors, EU Critical Entities,
World Bank economic sectors, and emerging technologies for anomalies.

Includes filtering/selection system to run subsets of modules (1-29+) for
flexible STEM applications.
"""

from typing import Any, TypedDict


class _ModuleInfo(TypedDict):
    """Type definition for module registry entries."""

    class_: type[Any]
    category: str
    priority: str
    description: str


from ..space.space_exploration_analyzer import SpaceExplorationAnalyzer
from .chemical_nuclear import ChemicalNuclearDetector, CISASector
from .communications_it import CommunicationsITDetector
from .cyber.cross_border_intel import CrossBorderIntelligence
from .cyber.space_infrastructure import SpaceInfrastructureMonitor
from .economic.world_bank_sectors import WorldBankSectorsMonitor
from .energy_dams import DamType, EnergyDamsDetector, EnergySubsector
from .healthcare_emergency import EmergencyType, HealthcareEmergencyDetector, PatientStatus
from .humanitarian.crisis_monitoring import CrisisAlert, CrisisMonitor
from .humanitarian.essential_workers import EssentialWorkersMonitor
from .humanitarian.government_facilities import GovernmentFacilitiesMonitor
from .resilience.ncf_monitor import NCFMonitor
from .scientific.emerging_tech_monitor import EmergingTechMonitor
from .streaming import (
    CircuitBreaker,
    CircuitState,
    StreamConfig,
    StreamConsumerFactory,
    StreamingAnomalyPipeline,
    StreamingBackend,
    StreamMessage,
    StreamProducerFactory,
)


class InfrastructureCoordinator:
    """Coordinator for infrastructure monitoring modules with flexible selection.

    Enables filtering/selection to run 1, 2, 5, or 29+ modules simultaneously
    based on priorities, categories, or explicit names. Handles all infrastructure
    frameworks (CISA NCFs, EU Critical Entities, World Bank sectors, etc.)
    individually or as a coordinated whole.

    Example:
        coordinator = InfrastructureCoordinator()

        high_priority = coordinator.instantiate_filtered_modules(priorities=['high'])

        cyber_modules = coordinator.instantiate_filtered_modules(categories=['cyber'])

        specific_modules = coordinator.instantiate_filtered_modules(
            module_names=['ncf_monitor', 'space_infrastructure']
        )
    """

    def __init__(self) -> None:
        """Initialize module registry with all available modules."""
        self.modules: dict[str, _ModuleInfo] = {
            "energy_dams": {
                "class_": EnergyDamsDetector,
                "category": "cisa_sector",
                "priority": "high",
                "description": "Energy and dams infrastructure (CISA Sector 3)",
            },
            "healthcare_emergency": {
                "class_": HealthcareEmergencyDetector,
                "category": "cisa_sector",
                "priority": "high",
                "description": "Healthcare and emergency services (CISA Sector 6)",
            },
            "communications_it": {
                "class_": CommunicationsITDetector,
                "category": "cisa_sector",
                "priority": "high",
                "description": "Communications and IT infrastructure (CISA Sector 5)",
            },
            "chemical_nuclear": {
                "class_": ChemicalNuclearDetector,
                "category": "cisa_sector",
                "priority": "high",
                "description": "Chemical and nuclear facilities (CISA Sector 4)",
            },
            "ncf_monitor": {
                "class_": NCFMonitor,
                "category": "resilience",
                "priority": "high",
                "description": "55 CISA National Critical Functions with cascading analysis",
            },
            "space_infrastructure": {
                "class_": SpaceInfrastructureMonitor,
                "category": "cyber",
                "priority": "high",
                "description": "EU Space sector (satellites, ground stations) - EU unique",
            },
            "cross_border_intel": {
                "class_": CrossBorderIntelligence,
                "category": "cyber",
                "priority": "medium",
                "description": "EU-US cross-border threat intelligence correlation",
            },
            "essential_workers": {
                "class_": EssentialWorkersMonitor,
                "category": "humanitarian",
                "priority": "high",
                "description": "8 essential worker categories with survivor-first ethics",
            },
            "government_facilities": {
                "class_": GovernmentFacilitiesMonitor,
                "category": "humanitarian",
                "priority": "medium",
                "description": "Government facilities (16th CISA sector) with governance",
            },
            "world_bank_sectors": {
                "class_": WorldBankSectorsMonitor,
                "category": "economic",
                "priority": "medium",
                "description": "21 ISIC economic sectors with sustainability focus",
            },
            "emerging_tech_monitor": {
                "class_": EmergingTechMonitor,
                "category": "scientific",
                "priority": "medium",
                "description": "9+ emerging technology categories for future-proofing",
            },
            "space_exploration_analyzer": {
                "class_": SpaceExplorationAnalyzer,
                "category": "scientific",
                "priority": "high",
                "description": "Hubble-inspired cosmic anomaly detection and threat analysis",
            },
        }

    def get_module(self, module_name: str, **kwargs: Any) -> Any:
        """Instantiate a specific module by name.

        Args:
            module_name: Name of module from registry
            **kwargs: Initialization arguments for the module

        Returns:
            Instantiated module
        """
        if module_name not in self.modules:
            raise ValueError(
                f"Unknown module: {module_name}. Available: {list(self.modules.keys())}"
            )

        module_class = self.modules[module_name]["class_"]
        return module_class(**kwargs)

    def get_modules_by_category(self, category: str) -> list[str]:
        """Get all module names in a category.

        Args:
            category: 'cisa_sector', 'resilience', 'cyber', 'humanitarian', 'economic', 'scientific'

        Returns:
            List of module names in the category
        """
        return [name for name, info in self.modules.items() if info["category"] == category]

    def get_modules_by_priority(self, priority: str) -> list[str]:
        """Get all module names with a priority level.

        Args:
            priority: 'high', 'medium', 'low'

        Returns:
            List of module names with the priority
        """
        return [name for name, info in self.modules.items() if info["priority"] == priority]

    def filter_modules(
        self,
        categories: list[str] | None = None,
        priorities: list[str] | None = None,
        module_names: list[str] | None = None,
    ) -> list[str]:
        """Filter modules based on multiple criteria.

        Args:
            categories: Filter by categories (e.g., ['cyber', 'resilience'])
            priorities: Filter by priorities (e.g., ['high'])
            module_names: Explicit list of module names to include

        Returns:
            List of module names matching all filters
        """
        if module_names:
            return [name for name in module_names if name in self.modules]

        filtered = set(self.modules.keys())

        if categories:
            category_modules = set()
            for cat in categories:
                category_modules.update(self.get_modules_by_category(cat))
            filtered &= category_modules

        if priorities:
            priority_modules = set()
            for pri in priorities:
                priority_modules.update(self.get_modules_by_priority(pri))
            filtered &= priority_modules

        return list(filtered)

    def instantiate_filtered_modules(
        self,
        categories: list[str] | None = None,
        priorities: list[str] | None = None,
        module_names: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Instantiate all modules matching filters.

        Args:
            categories: Filter by categories
            priorities: Filter by priorities
            module_names: Explicit list of module names
            **kwargs: Initialization arguments passed to all modules

        Returns:
            Dictionary mapping module names to instantiated modules
        """
        filtered_names = self.filter_modules(categories, priorities, module_names)
        instances = {}

        for name in filtered_names:
            if name == "chemical_nuclear" and "sector" not in kwargs:
                instances[name] = self.get_module(name, sector=CISASector.CHEMICAL)
            elif name == "energy_dams" and "subsector" not in kwargs:
                instances[name] = self.get_module(name, subsector=EnergySubsector.ELECTRICITY)
            else:
                instances[name] = self.get_module(name, **kwargs)

        return instances

    def list_all_modules(self) -> dict[str, dict[str, str]]:
        """List all available modules with metadata.

        Returns:
            Dictionary of module metadata
        """
        return {
            name: {
                "category": info["category"],
                "priority": info["priority"],
                "description": info["description"],
            }
            for name, info in self.modules.items()
        }


__all__ = [
    "CISASector",
    "ChemicalNuclearDetector",
    "CircuitBreaker",
    "CircuitState",
    "CommunicationsITDetector",
    "CrisisAlert",
    "CrisisMonitor",
    "CrossBorderIntelligence",
    "DamType",
    "EmergencyType",
    "EmergingTechMonitor",
    "EnergyDamsDetector",
    "EnergySubsector",
    "EssentialWorkersMonitor",
    "GovernmentFacilitiesMonitor",
    "HealthcareEmergencyDetector",
    "InfrastructureCoordinator",
    "NCFMonitor",
    "PatientStatus",
    "SpaceExplorationAnalyzer",
    "SpaceInfrastructureMonitor",
    "StreamConfig",
    "StreamConsumerFactory",
    "StreamMessage",
    "StreamProducerFactory",
    "StreamingAnomalyPipeline",
    "StreamingBackend",
    "WorldBankSectorsMonitor",
]
