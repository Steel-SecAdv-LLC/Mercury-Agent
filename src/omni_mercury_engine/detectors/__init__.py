"""
Mercury Agent
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
Detector module for Mercury Agent

Provides specialized anomaly detectors for different data types and domains.

Includes:
    - Base detectors (statistical, temporal, spatial, dimensional)
    - Visual anomaly detectors (PatchCore, PaDiM, STFPM, Reverse Distillation, CFlow)
    - Vision-Language Model detectors (AnyAnomaly, LAVAD)
    - Advanced SOTA detectors (time-series, industrial, contrastive, copula-based)
"""

import importlib
import logging

# Core detectors — no PyTorch dependency; always available.
from omni_mercury_engine.detectors.statistical import (
    MercuryAnomalyDetector,
    StatisticalAnomalyDetector,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — detectors below require PyTorch (or other optional deps).
# They are loaded on first access via __getattr__ so that importing this
# package never crashes when PyTorch is absent.
# ---------------------------------------------------------------------------
_LAZY_IMPORTS: dict[str, tuple[str, ...]] = {
    "omni_mercury_engine.detectors.acceleration_dynamics": (
        "AccelerationDynamicsDetector",
        "EnergyState",
        "MotionState",
    ),
    "omni_mercury_engine.detectors.advanced": (
        "AdversarialAutoencoderDetector",
        "ContrastiveLearningDetector",
        "COPODDetector",
        "GWOEnsembleDetector",
        "MultiScaleTransformerDetector",
        "PointAdjustmentEvaluator",
        "create_detector",
    ),
    "omni_mercury_engine.detectors.advanced_physics_integration": (
        "AdvancedPhysicsIntegratedDetector",
        "PhysicsDetectorType",
        "PhysicsGOSNNScalars",
        "create_dynamics_detector",
        "create_integrated_detector",
        "create_spectral_detector",
        "create_uiux_detector",
    ),
    "omni_mercury_engine.detectors.dimensional": ("DimensionalAnalyzer",),
    "omni_mercury_engine.detectors.directive": ("SigmaDirectiveDetector",),
    "omni_mercury_engine.detectors.spatial": ("SpatialAnomalyDetector",),
    "omni_mercury_engine.detectors.spectral_vibration": (
        "SpectralAnalysisMode",
        "SpectralVibrationDetector",
        "VibrationSignatureType",
    ),
    "omni_mercury_engine.detectors.temporal": ("TemporalAnomalyDetector",),
    "omni_mercury_engine.detectors.uiux_anomaly": (
        "AnomalyCategory",
        "InteractionType",
        "UIUXAnomalyDetector",
        "UserBehaviorClass",
        "UserInteraction",
    ),
    "omni_mercury_engine.detectors.visual": (
        "BaseVisualDetector",
        "CFlowDetector",
        "PaDiMDetector",
        "PatchCoreDetector",
        "ReverseDistillationDetector",
        "STFPMDetector",
    ),
    "omni_mercury_engine.detectors.vlm": (
        "AnyAnomalyDetector",
        "BaseVLMDetector",
        "LAVADDetector",
    ),
}

_NAME_TO_MODULE: dict[str, str] = {}
for _mod, _names in _LAZY_IMPORTS.items():
    for _name in _names:
        _NAME_TO_MODULE[_name] = _mod


def __getattr__(name: str) -> object:
    if name in _NAME_TO_MODULE:
        mod = importlib.import_module(_NAME_TO_MODULE[name])
        obj = getattr(mod, name)
        # Cache in the module namespace for subsequent access.
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AccelerationDynamicsDetector",
    "AdvancedPhysicsIntegratedDetector",
    "AdversarialAutoencoderDetector",
    "AnomalyCategory",
    "AnyAnomalyDetector",
    "BaseVLMDetector",
    "BaseVisualDetector",
    "CFlowDetector",
    "COPODDetector",
    "ContrastiveLearningDetector",
    "DimensionalAnalyzer",
    "EnergyState",
    "GWOEnsembleDetector",
    "InteractionType",
    "LAVADDetector",
    "MercuryAnomalyDetector",
    "MotionState",
    "MultiScaleTransformerDetector",
    "PaDiMDetector",
    "PatchCoreDetector",
    "PhysicsDetectorType",
    "PhysicsGOSNNScalars",
    "PointAdjustmentEvaluator",
    "ReverseDistillationDetector",
    "STFPMDetector",
    "SigmaDirectiveDetector",
    "SpatialAnomalyDetector",
    "SpectralAnalysisMode",
    "SpectralVibrationDetector",
    "StatisticalAnomalyDetector",
    "TemporalAnomalyDetector",
    "UIUXAnomalyDetector",
    "UserBehaviorClass",
    "UserInteraction",
    "VibrationSignatureType",
    "create_detector",
    "create_dynamics_detector",
    "create_integrated_detector",
    "create_spectral_detector",
    "create_uiux_detector",
]
