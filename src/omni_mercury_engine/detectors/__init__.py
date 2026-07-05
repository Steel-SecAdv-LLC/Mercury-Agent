# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Detector module for Mercury Agent.

Provides specialized anomaly detectors for different data types and domains.

Includes:
    - Base detectors (statistical, temporal, spatial, dimensional)
    - Visual anomaly detectors (PatchCore, PaDiM, STFPM, Reverse Distillation, CFlow)
    - Vision-Language Model detectors (AnyAnomaly, LAVAD)
    - Advanced SOTA detectors (time-series, industrial, contrastive, copula-based)
"""

from __future__ import annotations

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
    "omni_mercury_engine.detectors.spectral_domain_frequency": (
        "SpectralDomainFrequency",
        "SpectralDomainFrequencyConfig",
        "SpectralDomainOracle",
        "SpectralDomainOracleConfig",
        "FrequencyDomainOracle",
        "FrequencyDomainOracleConfig",
        "FrequencyInfluenceVector",
        "FrequencyWeighting",
        "FrequencyBandResult",
        "create_spectral_frequency",
        "create_spectral_oracle",
        "create_frequency_oracle",
        "get_domain_frequency_bands",
    ),
    "omni_mercury_engine.detectors.dimensional": ("DimensionalAnalyzer",),
    "omni_mercury_engine.detectors.directive": ("SigmaDirectiveDetector",),
    "omni_mercury_engine.detectors.geo_movement": (
        "GeoMovementAnomalyDetector",
        "MovementAssessment",
    ),
    "omni_mercury_engine.detectors.kmeans_distance": ("KMeansDistanceDetector",),
    "omni_mercury_engine.detectors.spectral_residual": ("SpectralResidualDetector",),
    "omni_mercury_engine.detectors.bocpd": ("BOCPDDetector",),
    "omni_mercury_engine.detectors.spot_evt": ("SPOTDetector",),
    "omni_mercury_engine.detectors.hawkes": ("HawkesBurstDetector",),
    "omni_mercury_engine.detectors.particle_filter": ("ParticleFilterDetector",),
    "omni_mercury_engine.detectors.imm": ("IMMDetector",),
    "omni_mercury_engine.detectors.gaussian_process": ("GaussianProcessDetector",),
    "omni_mercury_engine.detectors.echo_state": ("EchoStateDetector",),
    "omni_mercury_engine.detectors.digital_twin": ("DigitalTwinResidualDetector",),
    "omni_mercury_engine.detectors.spiking": ("SpikingNetworkDetector",),
    "omni_mercury_engine.detectors.survival": ("SurvivalHazardDetector",),
    "omni_mercury_engine.detectors.energy_based": ("EnergyBasedDetector",),
    "omni_mercury_engine.detectors.deep_svdd": ("DeepSVDDDetector",),
    "omni_mercury_engine.detectors.rca": ("RootCauseGraphDetector",),
    "omni_mercury_engine.detectors.deeplog_sequence": ("DeepLogSequenceDetector",),
    "omni_mercury_engine.detectors.frequent_pattern": ("FrequentPatternDetector",),
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
    "omni_mercury_engine.detectors.math_arrest.arrest": ("AnomalyMathArrest",),
}

_NAME_TO_MODULE: dict[str, str] = {}
for _mod, _names in _LAZY_IMPORTS.items():
    for _name in _names:
        _NAME_TO_MODULE[_name] = _mod


def __getattr__(name: str) -> object:
    """Implement the Python data model method."""
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
    "AnomalyMathArrest",
    "AnyAnomalyDetector",
    "BOCPDDetector",
    "BaseVLMDetector",
    "BaseVisualDetector",
    "CFlowDetector",
    "COPODDetector",
    "ContrastiveLearningDetector",
    "DeepLogSequenceDetector",
    "DeepSVDDDetector",
    "DigitalTwinResidualDetector",
    "DimensionalAnalyzer",
    "EchoStateDetector",
    "EnergyBasedDetector",
    "EnergyState",
    "FrequencyBandResult",
    "FrequencyDomainOracle",
    "FrequencyDomainOracleConfig",
    "FrequencyInfluenceVector",
    "FrequencyWeighting",
    "FrequentPatternDetector",
    "GWOEnsembleDetector",
    "GaussianProcessDetector",
    "GeoMovementAnomalyDetector",
    "HawkesBurstDetector",
    "IMMDetector",
    "InteractionType",
    "KMeansDistanceDetector",
    "LAVADDetector",
    "MercuryAnomalyDetector",
    "MotionState",
    "MovementAssessment",
    "MultiScaleTransformerDetector",
    "PaDiMDetector",
    "ParticleFilterDetector",
    "PatchCoreDetector",
    "PhysicsDetectorType",
    "PhysicsGOSNNScalars",
    "PointAdjustmentEvaluator",
    "ReverseDistillationDetector",
    "RootCauseGraphDetector",
    "SPOTDetector",
    "STFPMDetector",
    "SigmaDirectiveDetector",
    "SpatialAnomalyDetector",
    "SpectralAnalysisMode",
    "SpectralDomainFrequency",
    "SpectralDomainFrequencyConfig",
    "SpectralDomainOracle",
    "SpectralDomainOracleConfig",
    "SpectralResidualDetector",
    "SpectralVibrationDetector",
    "SpikingNetworkDetector",
    "StatisticalAnomalyDetector",
    "SurvivalHazardDetector",
    "TemporalAnomalyDetector",
    "UIUXAnomalyDetector",
    "UserBehaviorClass",
    "UserInteraction",
    "VibrationSignatureType",
    "create_detector",
    "create_dynamics_detector",
    "create_frequency_oracle",
    "create_integrated_detector",
    "create_spectral_detector",
    "create_spectral_frequency",
    "create_spectral_oracle",
    "create_uiux_detector",
    "get_domain_frequency_bands",
]
