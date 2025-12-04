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
Medical Anomaly Detection Module

Comprehensive medical detection for humanitarian healthcare:
- Pandemic detection, forecasting, and bio-threat analysis
- Critical care (sepsis, neurocritical)
- Cardiology (ECG, arrhythmia, cardiac risk)
- General medical cure prediction

Includes MedicalCoordinator for flexible module selection and filtering.
"""

from typing import Any

# Core medical modules
from omni_anomaly_engine.medical.abms_disciplines import (
    ABMSBoard,
    ABMSDisciplineDetector,
    MedicalAnomalyResult,
)
from omni_anomaly_engine.medical.medical_cure_predictor import (
    MedicalCurePredictor,
    MedicalImagingAnomalyDetector,
    MedicalPredictionResult,
    TemporalVitalSignsDetector,
    TemporalVitalSignsLSTM,
    TreatmentPathwayOptimizer,
)

# Cardiology
from omni_anomaly_engine.medical.cardiology import (
    ArrhythmiaType,
    CardiacBiomarkerAnalyzer,
    CardiologyPredictor,
    CardiologyPredictionResult,
    ECGRhythmAnalyzer,
    FraminghamRiskCalculator,
)

# Critical Care
from omni_anomaly_engine.medical.critical_care import (
    ICPMonitor,
    NeurocriticalCarePredictor,
    NeurocriticalPredictionResult,
    NIHSSCalculator,
    QuickSOFACalculator,
    SeizurePredictor,
    SeizureType,
    SepsisDetector,
    SepsisPredictionResult,
    SepsisProgressionPredictor,
    SepsisStage,
    SOFACalculator,
    StrokeDetector,
    StrokeType,
)

# Pandemic
from omni_anomaly_engine.medical.pandemic import (
    BioThreatResult,
    CaseSurgeDetector,
    EpidemicForecaster,
    MutationTracker,
    OutbreakSeverity,
    PandemicDetector,
    PandemicForecast,
    PandemicPredictionResult,
    PathogenDetector,
    TransmissionNetworkAnalyzer,
    VariantConcern,
)


class MedicalCoordinator:
    """Coordinator for medical detection modules with flexible selection.

    Enables filtering/selection to run 1, 2, 5, or 15+ modules simultaneously
    based on priorities, categories, or explicit names. Handles all medical
    domains (pandemic, critical care, cardiology, etc.) individually or as
    a coordinated whole.

    Example:
        coordinator = MedicalCoordinator()

        # Get only high-priority modules
        high_priority = coordinator.instantiate_filtered_modules(priorities=['high'])

        # Get pandemic-related modules
        pandemic_modules = coordinator.instantiate_filtered_modules(categories=['pandemic'])

        # Get specific modules
        specific_modules = coordinator.instantiate_filtered_modules(
            module_names=['sepsis_detector', 'pandemic_detector']
        )
    """

    def __init__(self):
        """Initialize module registry with all available modules."""
        self.modules = {
            # Pandemic Detection
            "pandemic_detector": {
                "class": PandemicDetector,
                "category": "pandemic",
                "priority": "high",
                "description": "Pandemic detection with case surveillance and mutation tracking",
            },
            "epidemic_forecaster": {
                "class": EpidemicForecaster,
                "category": "pandemic",
                "priority": "high",
                "description": "SEIR epidemiological forecasting with chaos detection",
            },
            "pathogen_detector": {
                "class": PathogenDetector,
                "category": "pandemic",
                "priority": "high",
                "description": "QBM-based pathogen detection with MASINT fusion",
            },
            # Critical Care
            "sepsis_detector": {
                "class": SepsisDetector,
                "category": "critical_care",
                "priority": "high",
                "description": "Sepsis detection with SOFA/qSOFA scoring",
            },
            "neurocritical_care": {
                "class": NeurocriticalCarePredictor,
                "category": "critical_care",
                "priority": "high",
                "description": "Stroke, seizure, TBI, and ICP monitoring",
            },
            # Cardiology
            "cardiology_predictor": {
                "class": CardiologyPredictor,
                "category": "cardiology",
                "priority": "high",
                "description": "ECG analysis, arrhythmia detection, cardiac risk prediction",
            },
            # General Medical
            "medical_cure_predictor": {
                "class": MedicalCurePredictor,
                "category": "general",
                "priority": "medium",
                "description": "Temporal vital signs and medical imaging anomaly detection",
            },
            "abms_discipline_detector": {
                "class": ABMSDisciplineDetector,
                "category": "general",
                "priority": "medium",
                "description": "Multi-specialty medical anomaly detection across ABMS disciplines",
            },
        }

    def get_module(self, module_name: str, **kwargs) -> Any:
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

        module_class = self.modules[module_name]["class"]
        return module_class(**kwargs)

    def get_modules_by_category(self, category: str) -> list[str]:
        """Get all module names in a category.

        Args:
            category: 'pandemic', 'critical_care', 'cardiology', 'general'

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
            categories: Filter by categories (e.g., ['pandemic', 'critical_care'])
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
        **kwargs,
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
    # Coordinator
    "MedicalCoordinator",
    # Core medical
    "ABMSDisciplineDetector",
    "ABMSBoard",
    "MedicalAnomalyResult",
    "MedicalCurePredictor",
    "MedicalPredictionResult",
    "MedicalImagingAnomalyDetector",
    "TemporalVitalSignsDetector",
    "TemporalVitalSignsLSTM",
    "TreatmentPathwayOptimizer",
    # Cardiology
    "CardiologyPredictor",
    "CardiologyPredictionResult",
    "ArrhythmiaType",
    "ECGRhythmAnalyzer",
    "CardiacBiomarkerAnalyzer",
    "FraminghamRiskCalculator",
    # Critical Care - Sepsis
    "SepsisDetector",
    "SepsisPredictionResult",
    "SepsisStage",
    "SOFACalculator",
    "QuickSOFACalculator",
    "SepsisProgressionPredictor",
    # Critical Care - Neurocritical
    "NeurocriticalCarePredictor",
    "NeurocriticalPredictionResult",
    "StrokeDetector",
    "StrokeType",
    "SeizurePredictor",
    "SeizureType",
    "ICPMonitor",
    "NIHSSCalculator",
    # Pandemic
    "PandemicDetector",
    "PandemicPredictionResult",
    "CaseSurgeDetector",
    "MutationTracker",
    "TransmissionNetworkAnalyzer",
    "EpidemicForecaster",
    "PandemicForecast",
    "PathogenDetector",
    "BioThreatResult",
    "OutbreakSeverity",
    "VariantConcern",
]
