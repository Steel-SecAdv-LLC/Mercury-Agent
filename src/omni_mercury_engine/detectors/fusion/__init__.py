# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Multi-modal fusion module for combining VLM and Visual detector outputs."""

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
