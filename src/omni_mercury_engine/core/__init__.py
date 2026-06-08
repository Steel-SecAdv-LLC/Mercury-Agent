# Copyright (C) 2025 Steel Security Advisors LLC
"""Core modules for Mercury Agent."""

from __future__ import annotations

from typing import Any

from omni_mercury_engine.core.adaptive_fusion import (
    AttentionVisualization,
    UncertaintyEstimate,
)
from omni_mercury_engine.core.double_helix_engine import (
    DoubleHelixEvolutionEngine,
    HelixConfig,
    MercuryEquationEngine,
)
from omni_mercury_engine.core.feature_pipeline import (
    FeaturePipeline,
    FeatureStandardizer,
    FeatureStore,
)
from omni_mercury_engine.core.global_omni_scalar_network import (
    EnhancementResult,
    EthicalGate,
    GlobalOmniScalarNetwork,
    MultiHeadAttentionFusion,
    ScalarGroup,
    ScalarRegistration,
    get_global_scalar_network,
    reset_global_network,
)
from omni_mercury_engine.core.learnable_gosnn import (
    LearnableGOSNN,
    ScalarCategory,
    ScalarState,
)
from omni_mercury_engine.core.score_calibration import (
    AutoThresholdOptimizer,
    CalibrationDiagnostics,
    CalibrationMethod,
    CalibrationResult,
    ScoreCalibrationManager,
    ScoreDiagnostics,
    calibrate_scores,
    diagnose_scores,
)
from omni_mercury_engine.core.types import (
    AnomalyType,
    CircuitState,
    ConfidenceLevel,
    DetectorStatus,
    EthicalPrinciple,
    FusionStrategy,
    PrivacyLevel,
    ThreatLevel,
)

__all__ = [
    # Canonical types from core.types
    "AnomalyType",
    "AttentionVisualization",
    # Score calibration (solves F1=0 problem)
    "AutoThresholdOptimizer",
    "CalibrationDiagnostics",
    "CalibrationMethod",
    "CalibrationResult",
    "CircuitState",
    "ConfidenceLevel",
    "DetectorStatus",
    # Original exports
    "DomainType",
    "DoubleHelixEvolutionEngine",
    "EnhancementResult",
    "EthicalConfig",
    "EthicalGate",
    "EthicalPrinciple",
    "FeaturePipeline",
    "FeatureStandardizer",
    "FeatureStore",
    "FusionStrategy",
    "FusionWeightConfig",
    "GlobalOmniScalarNetwork",
    "HelixConfig",
    "LearnableGOSNN",
    "MercuryEngineConfig",
    "MercuryEquationEngine",
    "MultiHeadAttentionFusion",
    "PrivacyLevel",
    "ScalarCategory",
    "ScalarGroup",
    "ScalarRegistration",
    "ScalarState",
    "ScoreCalibrationManager",
    "ScoreDiagnostics",
    "ThreatLevel",
    "ThreeRConfig",
    "UncertaintyEstimate",
    "calibrate_scores",
    "diagnose_scores",
    "get_default_config",
    "get_global_scalar_network",
    "reset_global_network",
]


# Lazy imports for engine configuration (requires pydantic)
def get_default_config() -> MercuryEngineConfig:
    """Get the global default configuration.

    Lazy import to avoid pydantic at module load.
    """
    from omni_mercury_engine.core.engine_config import get_default_config as _get

    return _get()  # type: ignore[return-value, unused-ignore]


class MercuryEngineConfig:
    """Lazy-loaded MercuryEngineConfig wrapper."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        """New."""
        from omni_mercury_engine.core.engine_config import MercuryEngineConfig as _Config

        return _Config(*args, **kwargs)


class EthicalConfig:
    """Lazy-loaded EthicalConfig wrapper."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        """New."""
        from omni_mercury_engine.core.engine_config import EthicalConfig as _Config

        return _Config(*args, **kwargs)


class FusionWeightConfig:
    """Lazy-loaded FusionWeightConfig wrapper."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        """New."""
        from omni_mercury_engine.core.engine_config import FusionWeightConfig as _Config

        return _Config(*args, **kwargs)


class ThreeRConfig:
    """Lazy-loaded ThreeRConfig wrapper."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        """New."""
        from omni_mercury_engine.core.engine_config import ThreeRConfig as _Config

        return _Config(*args, **kwargs)


class DomainType:
    """Lazy-loaded DomainType wrapper."""

    def __new__(cls, value: str) -> Any:
        """New."""
        from omni_mercury_engine.core.engine_config import DomainType as _DomainType

        return _DomainType(value)
