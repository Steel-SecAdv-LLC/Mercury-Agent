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
"""

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

__all__ = [
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
