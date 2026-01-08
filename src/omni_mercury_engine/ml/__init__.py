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
Machine Learning module for Mercury Agent ♱

Provides attention mechanisms, feature encoders, fusion networks, training, and inference.

Note: This module requires PyTorch. Imports are lazy to allow core package
to function without torch installed. Access ML components only when torch is available.
"""

from typing import TYPE_CHECKING

# Check if torch is available
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

__all__ = [
    "HAS_TORCH",
    "AffectiveEncoder",
    "AstrophysicalEncoder",
    "AuxiliaryMaxVariance",
    "BiometricEncoder",
    "CheckpointCallback",
    "ConvergenceMonitor",
    "CrossModalAttention",
    "DifferenceTargetPropagation",
    "FairnessAuditor",
    "FusionInference",
    "FusionTrainer",
    "LyapunovAnomalyLoss",
    "MemoryEfficientCache",
    "MultiEnvPPOTrainer",
    "MultiHeadDetectorAttention",
    "OmniFusionModel",
    "PPOConfig",
    "PPOTrainer",
    "ParallelExecutor",
    "QuantumEncoder",
    "SpatialAttention",
    "StatisticalEncoder",
    "SyntheticGradientModule",
    "SyntheticGradientPredictor",
    "TemporalAttention",
    "TemporalEncoder",
    "ThreeRAnomalyTrainer",
    "ThreeRAnomalyTransformer",
    "ThreeRAttentionBlock",
    "TrainingStats",
    "apply_all_optimizations",
    "compute_fairness_score",
    "create_drift_detector",
]

# Lazy imports - only load when torch is available OR during type checking
if HAS_TORCH or TYPE_CHECKING:
    from omni_mercury_engine.ml.advanced_optimizers import (
        AuxiliaryMaxVariance,
        DifferenceTargetPropagation,
        SyntheticGradientModule,
        SyntheticGradientPredictor,
    )
    from omni_mercury_engine.ml.attention import (
        CrossModalAttention,
        MultiHeadDetectorAttention,
        SpatialAttention,
        TemporalAttention,
    )
    from omni_mercury_engine.ml.encoders import (
        AffectiveEncoder,
        AstrophysicalEncoder,
        BiometricEncoder,
        QuantumEncoder,
        StatisticalEncoder,
        TemporalEncoder,
    )
    from omni_mercury_engine.ml.fusion_network import OmniFusionModel
    from omni_mercury_engine.ml.inference import FusionInference
    from omni_mercury_engine.ml.ppo_trainer import (
        CheckpointCallback,
        ConvergenceMonitor,
        MultiEnvPPOTrainer,
        PPOConfig,
        PPOTrainer,
        TrainingStats,
    )
    from omni_mercury_engine.ml.three_r_attention import (
        ThreeRAnomalyTransformer,
        ThreeRAttentionBlock,
    )
    from omni_mercury_engine.ml.training import (
        FusionTrainer,
        LyapunovAnomalyLoss,
        ThreeRAnomalyTrainer,
    )


def _require_torch() -> None:
    """Raise ImportError if torch is not available."""
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for ML components. " "Install with: pip install torch torchvision"
        )


# Lazy imports for drift detection (requires scipy)
def create_drift_detector(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Create a drift detector. Lazy import to avoid scipy dependency at module load."""
    from omni_mercury_engine.ml.drift import create_drift_detector as _create

    return _create(*args, **kwargs)


# Lazy imports for fairness auditing (requires numpy only)
def compute_fairness_score(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Compute fairness score. Lazy import to avoid loading full module."""
    from omni_mercury_engine.ml.fairness import compute_fairness_score as _compute

    return _compute(*args, **kwargs)


class FairnessAuditor:
    """Lazy-loaded FairnessAuditor wrapper."""

    def __new__(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
        from omni_mercury_engine.ml.fairness import FairnessAuditor as _FairnessAuditor

        return _FairnessAuditor(*args, **kwargs)


# Lazy imports for optimization utilities
def apply_all_optimizations(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Apply all optimizations. Lazy import to avoid psutil/joblib dependency."""
    from omni_mercury_engine.ml.optimization import apply_all_optimizations as _apply

    return _apply(*args, **kwargs)


class MemoryEfficientCache:
    """Lazy-loaded MemoryEfficientCache wrapper."""

    def __new__(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
        from omni_mercury_engine.ml.optimization import MemoryEfficientCache as _Cache

        return _Cache(*args, **kwargs)


class ParallelExecutor:
    """Lazy-loaded ParallelExecutor wrapper."""

    def __new__(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
        from omni_mercury_engine.ml.optimization import ParallelExecutor as _Executor

        return _Executor(*args, **kwargs)
