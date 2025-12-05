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
Machine Learning module for OMNI ♱ AVA

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
    "FusionInference",
    "FusionTrainer",
    "MultiEnvPPOTrainer",
    "MultiHeadDetectorAttention",
    "OmniFusionModel",
    "PPOConfig",
    "PPOTrainer",
    "QuantumEncoder",
    "SpatialAttention",
    "StatisticalEncoder",
    "SyntheticGradientModule",
    "SyntheticGradientPredictor",
    "TemporalAttention",
    "TemporalEncoder",
    "TrainingStats",
]

# Lazy imports - only load when torch is available
if HAS_TORCH:
    from omni_anomaly_engine.ml.advanced_optimizers import (
        AuxiliaryMaxVariance,
        DifferenceTargetPropagation,
        SyntheticGradientModule,
        SyntheticGradientPredictor,
    )
    from omni_anomaly_engine.ml.attention import (
        CrossModalAttention,
        MultiHeadDetectorAttention,
        SpatialAttention,
        TemporalAttention,
    )
    from omni_anomaly_engine.ml.encoders import (
        AffectiveEncoder,
        AstrophysicalEncoder,
        BiometricEncoder,
        QuantumEncoder,
        StatisticalEncoder,
        TemporalEncoder,
    )
    from omni_anomaly_engine.ml.fusion_network import OmniFusionModel
    from omni_anomaly_engine.ml.inference import FusionInference
    from omni_anomaly_engine.ml.ppo_trainer import (
        CheckpointCallback,
        ConvergenceMonitor,
        MultiEnvPPOTrainer,
        PPOConfig,
        PPOTrainer,
        TrainingStats,
    )
    from omni_anomaly_engine.ml.training import FusionTrainer

# Type checking imports for IDE support
if TYPE_CHECKING:
    from omni_anomaly_engine.ml.advanced_optimizers import (
        AuxiliaryMaxVariance,
        DifferenceTargetPropagation,
        SyntheticGradientModule,
        SyntheticGradientPredictor,
    )
    from omni_anomaly_engine.ml.attention import (
        CrossModalAttention,
        MultiHeadDetectorAttention,
        SpatialAttention,
        TemporalAttention,
    )
    from omni_anomaly_engine.ml.encoders import (
        AffectiveEncoder,
        AstrophysicalEncoder,
        BiometricEncoder,
        QuantumEncoder,
        StatisticalEncoder,
        TemporalEncoder,
    )
    from omni_anomaly_engine.ml.fusion_network import OmniFusionModel
    from omni_anomaly_engine.ml.inference import FusionInference
    from omni_anomaly_engine.ml.ppo_trainer import (
        CheckpointCallback,
        ConvergenceMonitor,
        MultiEnvPPOTrainer,
        PPOConfig,
        PPOTrainer,
        TrainingStats,
    )
    from omni_anomaly_engine.ml.training import FusionTrainer


def _require_torch() -> None:
    """Raise ImportError if torch is not available."""
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for ML components. "
            "Install with: pip install torch torchvision"
        )
