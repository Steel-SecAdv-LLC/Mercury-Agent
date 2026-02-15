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
Detector module for Mercury Agent ♱

Provides specialized anomaly detectors for different data types and domains.

Includes:
    - Base detectors (statistical, temporal, spatial, dimensional)
    - Visual anomaly detectors (PatchCore, PaDiM, STFPM, Reverse Distillation, CFlow)
    - Vision-Language Model detectors (AnyAnomaly, LAVAD)
    - Advanced SOTA detectors (time-series, industrial, contrastive, copula-based)
"""

# Advanced SOTA Detectors (v1.4.0)
from omni_mercury_engine.detectors.acceleration_dynamics import (
    AccelerationDynamicsDetector,
    EnergyState,
    MotionState,
)
from omni_mercury_engine.detectors.advanced import (
    AdversarialAutoencoderDetector,
    ContrastiveLearningDetector,
    COPODDetector,
    GWOEnsembleDetector,
    MultiScaleTransformerDetector,
    PointAdjustmentEvaluator,
    create_detector,
)
from omni_mercury_engine.detectors.advanced_physics_integration import (
    AdvancedPhysicsIntegratedDetector,
    PhysicsDetectorType,
    PhysicsGOSNNScalars,
    create_dynamics_detector,
    create_integrated_detector,
    create_spectral_detector,
    create_uiux_detector,
)
from omni_mercury_engine.detectors.dimensional import DimensionalAnalyzer
from omni_mercury_engine.detectors.directive import SigmaDirectiveDetector
from omni_mercury_engine.detectors.spatial import SpatialAnomalyDetector

# Advanced Physics-Inspired Detectors (v1.4.0)
from omni_mercury_engine.detectors.spectral_vibration import (
    SpectralAnalysisMode,
    SpectralVibrationDetector,
    VibrationSignatureType,
)
from omni_mercury_engine.detectors.statistical import (
    MercuryAnomalyDetector,
    StatisticalAnomalyDetector,  # compat alias
)
from omni_mercury_engine.detectors.temporal import TemporalAnomalyDetector
from omni_mercury_engine.detectors.uiux_anomaly import (
    AnomalyCategory,
    InteractionType,
    UIUXAnomalyDetector,
    UserBehaviorClass,
    UserInteraction,
)

# SOTA Visual Anomaly Detection
from omni_mercury_engine.detectors.visual import (
    BaseVisualDetector,
    CFlowDetector,
    PaDiMDetector,
    PatchCoreDetector,
    ReverseDistillationDetector,
    STFPMDetector,
)

# Vision-Language Model Detectors
from omni_mercury_engine.detectors.vlm import AnyAnomalyDetector, BaseVLMDetector, LAVADDetector

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
