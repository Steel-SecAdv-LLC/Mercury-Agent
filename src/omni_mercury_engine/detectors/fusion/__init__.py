"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisory LLC

Multi-modal fusion module for combining VLM and Visual detector outputs.
"""

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
    # Data classes
    "ModalityInput",
    "FusionResult",
    # Enums
    "FusionStrategy",
    # Base class
    "BaseFusionModule",
    # Fusion modules
    "FeatureConcatFusion",
    "AttentionFusion",
    "ScoreWeightedFusion",
    "DecisionConfidenceFusion",
    "AdaptiveFusion",
    # Optimizer
    "MultiModalFusionOptimizer",
    # Factory
    "create_fusion_optimizer",
]
