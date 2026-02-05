"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisory LLC

Multi-modal fusion module for combining VLM and Visual detector outputs.
"""

from typing import Any
from __future__ import annotations

from .multimodal_fusion import (
    AdaptiveFusion,
    AttentionFusion,
    BaseFusionModule,
    DecisionConfidenceFusion,
    FeatureConcatFusion,
    FusionResult,
    FusionStrategy,
    ModalityInput,
    MultiModalFusionOptimizer,
    ScoreWeightedFusion,
    create_fusion_optimizer,
)


__all__ = [
    "AdaptiveFusion",
    "AttentionFusion",
    # Base class
    "BaseFusionModule",
    "DecisionConfidenceFusion",
    # Fusion modules
    "FeatureConcatFusion",
    "FusionResult",
    # Enums
    "FusionStrategy",
    # Data classes
    "ModalityInput",
    # Optimizer
    "MultiModalFusionOptimizer",
    "ScoreWeightedFusion",
    # Factory
    "create_fusion_optimizer",
]
