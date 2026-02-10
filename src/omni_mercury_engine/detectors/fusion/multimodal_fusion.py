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
Multi-Modal Fusion Optimizer for VLM + Visual Detector Ensemble.

Provides intelligent fusion of features from:
- Vision-Language Models (semantic understanding, zero-shot detection)
- Visual Anomaly Detectors (pixel-level precision, trained representations)

Fusion strategies:
- Feature-level: Concatenate and project features
- Score-level: Weighted combination of anomaly scores
- Decision-level: Ensemble voting with confidence weighting
- Attention-based: Learn cross-modal attention weights

Key innovations:
- Adaptive weighting based on input characteristics
- Uncertainty-aware fusion (higher weight to confident predictions)
- Domain-specific calibration
- Complementary fusion (VLM for semantics, Visual for localization)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """Available fusion strategies."""

    FEATURE_CONCAT = "feature_concat"
    FEATURE_ATTENTION = "feature_attention"
    SCORE_AVERAGE = "score_average"
    SCORE_WEIGHTED = "score_weighted"
    SCORE_MAX = "score_max"
    DECISION_VOTING = "decision_voting"
    DECISION_CONFIDENCE = "decision_confidence"
    ADAPTIVE = "adaptive"


@dataclass
class ModalityInput:
    """Input from a single modality (VLM or Visual detector)."""

    modality_type: str  # "vlm" or "visual"
    detector_name: str
    features: torch.Tensor | None = None  # [N, D]
    scores: np.ndarray | None = None  # [N]
    predictions: np.ndarray | None = None  # [N] binary
    confidence: np.ndarray | None = None  # [N] [0, 1]
    anomaly_maps: torch.Tensor | None = None  # [N, H, W] for visual
    explanations: list[str] | None = None  # For VLM
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionResult:
    """Result of multi-modal fusion."""

    fused_scores: np.ndarray  # [N] final anomaly scores
    fused_predictions: np.ndarray  # [N] binary predictions
    fused_features: torch.Tensor | None = None  # [N, D_fused]
    fused_anomaly_maps: torch.Tensor | None = None  # [N, H, W]
    modality_weights: dict[str, float] = field(default_factory=dict)
    confidence: np.ndarray | None = None
    explanation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseFusionModule(ABC):
    """Abstract base class for fusion modules."""

    @abstractmethod
    def fuse(
        self,
        inputs: list[ModalityInput],
        threshold: float = 0.5,
    ) -> FusionResult:
        """
        Fuse inputs from multiple modalities.

        Args:
            inputs: List of modality inputs
            threshold: Decision threshold for binary predictions

        Returns:
            Fused result
        """
        pass


class FeatureConcatFusion(BaseFusionModule):
    """
    Feature concatenation fusion with optional projection.

    Simply concatenates features from all modalities and optionally
    projects to a lower dimension.
    """

    def __init__(
        self,
        output_dim: int | None = None,
        normalize: bool = True,
    ):
        """
        Initialize feature concatenation fusion.

        Args:
            output_dim: Output dimension (None = no projection)
            normalize: Whether to L2 normalize features
        """
        self.output_dim = output_dim
        self.normalize = normalize
        self._projection: nn.Linear | None = None

    def fuse(
        self,
        inputs: list[ModalityInput],
        threshold: float = 0.5,
    ) -> FusionResult:
        """Fuse by concatenating features."""
        # Collect features
        feature_list = []
        score_list = []
        weights = {}

        for inp in inputs:
            if inp.features is not None:
                feature_list.append(inp.features)
                weights[inp.detector_name] = 1.0 / len(inputs)

            if inp.scores is not None:
                score_list.append(inp.scores)

        # Concatenate features
        if feature_list:
            fused_features = torch.cat(feature_list, dim=-1)

            # Project if specified
            if self.output_dim is not None:
                if (
                    self._projection is None
                    or self._projection.in_features != fused_features.shape[-1]
                ):
                    self._projection = nn.Linear(fused_features.shape[-1], self.output_dim)
                fused_features = self._projection(fused_features)

            # Normalize
            if self.normalize:
                fused_features = F.normalize(fused_features, p=2, dim=-1)

            # Detach from computation graph for inference
            fused_features = fused_features.detach()
        else:
            fused_features = None

        # Average scores
        if score_list:
            fused_scores = np.mean(score_list, axis=0)
        else:
            fused_scores = np.zeros(1)

        fused_predictions = (fused_scores >= threshold).astype(np.int32)

        return FusionResult(
            fused_scores=fused_scores,
            fused_predictions=fused_predictions,
            fused_features=fused_features,
            modality_weights=weights,
        )


class AttentionFusion(BaseFusionModule, nn.Module):
    """
    Attention-based fusion that learns cross-modal interactions.

    Uses multi-head attention to weight features from different
    modalities based on their relevance.
    """

    def __init__(
        self,
        feature_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize attention fusion.

        Args:
            feature_dim: Expected feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        nn.Module.__init__(self)

        self.feature_dim = feature_dim
        self.num_heads = num_heads

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

        # Score head
        self.score_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def fuse(
        self,
        inputs: list[ModalityInput],
        threshold: float = 0.5,
    ) -> FusionResult:
        """Fuse using attention mechanism."""
        # Collect and project features
        feature_list = []
        names = []

        for inp in inputs:
            if inp.features is not None:
                feat = inp.features
                # Project to common dimension if needed
                if feat.shape[-1] != self.feature_dim:
                    proj = nn.Linear(feat.shape[-1], self.feature_dim)
                    feat = proj(feat)
                feature_list.append(feat)
                names.append(inp.detector_name)

        if not feature_list:
            return FusionResult(
                fused_scores=np.zeros(1),
                fused_predictions=np.zeros(1, dtype=np.int32),
                modality_weights={},
            )

        # Stack features [B, num_modalities, D]
        stacked = torch.stack(feature_list, dim=1)

        # Self-attention across modalities
        attended, attn_weights = self.attention(
            stacked,
            stacked,
            stacked,
            need_weights=True,
        )

        # Pool across modalities
        pooled = attended.mean(dim=1)  # [B, D]

        # Output projection
        fused_features = self.output_proj(pooled)

        # Score prediction
        scores = self.score_head(fused_features).squeeze(-1)
        fused_scores = scores.detach().cpu().numpy()

        # Detach features from computation graph for inference
        fused_features = fused_features.detach()

        # Compute modality weights from attention
        weights = {}
        mean_attn = attn_weights.mean(dim=(0, 1)).detach().cpu().numpy()
        for i, name in enumerate(names):
            weights[name] = float(mean_attn[i])

        fused_predictions = (fused_scores >= threshold).astype(np.int32)

        return FusionResult(
            fused_scores=fused_scores,
            fused_predictions=fused_predictions,
            fused_features=fused_features,
            modality_weights=weights,
        )


class ScoreWeightedFusion(BaseFusionModule):
    """
    Score-level fusion with learned or fixed weights.

    Combines anomaly scores from multiple detectors using
    weighted averaging.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        normalize_weights: bool = True,
        uncertainty_weighting: bool = True,
    ):
        """
        Initialize score weighted fusion.

        Args:
            weights: Fixed weights per detector (None = equal)
            normalize_weights: Normalize weights to sum to 1
            uncertainty_weighting: Weight by confidence if available
        """
        self.weights = weights or {}
        self.normalize_weights = normalize_weights
        self.uncertainty_weighting = uncertainty_weighting

    def fuse(
        self,
        inputs: list[ModalityInput],
        threshold: float = 0.5,
    ) -> FusionResult:
        """Fuse by weighted score averaging."""
        scores = []
        weights = []
        names = []

        for inp in inputs:
            if inp.scores is None:
                continue

            names.append(inp.detector_name)
            scores.append(inp.scores)

            # Determine weight
            w = self.weights.get(inp.detector_name, 1.0)

            # Adjust by confidence if available
            if self.uncertainty_weighting and inp.confidence is not None:
                mean_conf = inp.confidence.mean()
                w *= mean_conf

            weights.append(w)

        if not scores:
            return FusionResult(
                fused_scores=np.zeros(1),
                fused_predictions=np.zeros(1, dtype=np.int32),
                modality_weights={},
            )

        # Normalize weights
        weights = np.array(weights)  # type: ignore[assignment, unused-ignore]
        if self.normalize_weights:
            weights = weights / weights.sum()  # type: ignore[attr-defined, unused-ignore]

        # Weighted average
        scores = np.array(scores)  # type: ignore[assignment, unused-ignore]
        fused_scores = np.average(scores, axis=0, weights=weights)

        # Build weight dict
        weight_dict = {name: float(w) for name, w in zip(names, weights)}

        fused_predictions = (fused_scores >= threshold).astype(np.int32)

        # Combine features if available
        feature_list = [inp.features for inp in inputs if inp.features is not None]
        fused_features = None
        if feature_list:
            fused_features = torch.cat(feature_list, dim=-1)

        return FusionResult(
            fused_scores=fused_scores,
            fused_predictions=fused_predictions,
            fused_features=fused_features,
            modality_weights=weight_dict,
        )


class DecisionConfidenceFusion(BaseFusionModule):
    """
    Decision-level fusion with confidence weighting.

    Makes final decision based on detector predictions weighted
    by their confidence.
    """

    def __init__(
        self,
        require_consensus: bool = False,
        consensus_threshold: float = 0.5,
    ):
        """
        Initialize decision confidence fusion.

        Args:
            require_consensus: Require majority agreement
            consensus_threshold: Fraction needed for consensus
        """
        self.require_consensus = require_consensus
        self.consensus_threshold = consensus_threshold

    def fuse(
        self,
        inputs: list[ModalityInput],
        threshold: float = 0.5,
    ) -> FusionResult:
        """Fuse at decision level with confidence weighting."""
        predictions = []
        confidences = []
        names = []

        for inp in inputs:
            if inp.scores is None:
                continue

            names.append(inp.detector_name)

            # Get predictions
            if inp.predictions is not None:
                pred = inp.predictions
            else:
                pred = (inp.scores >= threshold).astype(np.int32)
            predictions.append(pred)

            # Get confidence
            if inp.confidence is not None:
                conf = inp.confidence
            else:
                # Derive confidence from score distance to threshold
                conf = np.abs(inp.scores - threshold) * 2
                conf = np.clip(conf, 0, 1)
            confidences.append(conf)

        if not predictions:
            return FusionResult(
                fused_scores=np.zeros(1),
                fused_predictions=np.zeros(1, dtype=np.int32),
                modality_weights={},
            )

        predictions = np.array(predictions)  # type: ignore[assignment, unused-ignore]
        confidences = np.array(confidences)  # type: ignore[assignment, unused-ignore]

        # Weighted voting
        weighted_votes = predictions * confidences  # type: ignore[operator, unused-ignore]
        vote_sum = weighted_votes.sum(axis=0)  # type: ignore[attr-defined, unused-ignore]
        weight_sum = confidences.sum(axis=0) + 1e-8  # type: ignore[attr-defined, unused-ignore]

        fused_scores = vote_sum / weight_sum

        if self.require_consensus:
            # Require majority
            vote_fraction = predictions.sum(axis=0) / len(predictions)  # type: ignore[attr-defined, unused-ignore]
            consensus_mask = vote_fraction >= self.consensus_threshold
            fused_predictions = ((fused_scores >= threshold) & consensus_mask).astype(np.int32)
        else:
            fused_predictions = (fused_scores >= threshold).astype(np.int32)

        # Modality weights based on average confidence
        weight_dict = {name: float(conf.mean()) for name, conf in zip(names, confidences)}

        return FusionResult(
            fused_scores=fused_scores,
            fused_predictions=fused_predictions,
            modality_weights=weight_dict,
            confidence=fused_scores,  # Use fused score as confidence
        )


class AdaptiveFusion(BaseFusionModule):
    """
    Adaptive fusion that selects strategy based on input characteristics.

    Analyzes inputs to determine optimal fusion approach:
    - High VLM confidence + Low Visual confidence → Trust VLM
    - Low VLM confidence + High Visual precision → Trust Visual
    - Agreement between modalities → Confident fusion
    - Disagreement → Conservative fusion with uncertainty
    """

    def __init__(
        self,
        vlm_semantic_weight: float = 0.6,
        visual_localization_weight: float = 0.6,
        agreement_boost: float = 1.2,
        disagreement_penalty: float = 0.8,
    ):
        """
        Initialize adaptive fusion.

        Args:
            vlm_semantic_weight: Weight for VLM on semantic anomalies
            visual_localization_weight: Weight for Visual on localized anomalies
            agreement_boost: Confidence boost when modalities agree
            disagreement_penalty: Confidence penalty when modalities disagree
        """
        self.vlm_semantic_weight = vlm_semantic_weight
        self.visual_localization_weight = visual_localization_weight
        self.agreement_boost = agreement_boost
        self.disagreement_penalty = disagreement_penalty

        # Sub-strategies
        self._score_fusion = ScoreWeightedFusion(uncertainty_weighting=True)
        self._decision_fusion = DecisionConfidenceFusion(require_consensus=False)

    def fuse(
        self,
        inputs: list[ModalityInput],
        threshold: float = 0.5,
    ) -> FusionResult:
        """Adaptively fuse based on input characteristics."""
        # Separate VLM and Visual inputs
        vlm_inputs = [inp for inp in inputs if inp.modality_type == "vlm"]
        visual_inputs = [inp for inp in inputs if inp.modality_type == "visual"]

        # Analyze agreement
        agreement_score = self._compute_agreement(inputs, threshold)

        # Analyze anomaly characteristics
        is_localized = self._check_localization(visual_inputs)
        is_semantic = self._check_semantic(vlm_inputs)

        # Determine weights
        vlm_weight = 0.5
        visual_weight = 0.5

        if is_semantic and not is_localized:
            vlm_weight = self.vlm_semantic_weight
            visual_weight = 1.0 - vlm_weight

        elif is_localized and not is_semantic:
            visual_weight = self.visual_localization_weight
            vlm_weight = 1.0 - visual_weight

        # Adjust by agreement
        confidence_multiplier = 1.0
        if agreement_score > 0.8:
            confidence_multiplier = self.agreement_boost
        elif agreement_score < 0.3:
            confidence_multiplier = self.disagreement_penalty

        # Set weights for score fusion
        weights = {}
        for inp in vlm_inputs:
            weights[inp.detector_name] = vlm_weight / max(len(vlm_inputs), 1)
        for inp in visual_inputs:
            weights[inp.detector_name] = visual_weight / max(len(visual_inputs), 1)

        self._score_fusion.weights = weights

        # Perform fusion
        result = self._score_fusion.fuse(inputs, threshold)

        # Adjust confidence
        if result.confidence is not None:
            result.confidence = result.confidence * confidence_multiplier
        else:
            result.confidence = result.fused_scores * confidence_multiplier

        result.confidence = np.clip(result.confidence, 0, 1)

        # Add metadata
        result.metadata = {
            "agreement_score": agreement_score,
            "is_localized": is_localized,
            "is_semantic": is_semantic,
            "vlm_weight": vlm_weight,
            "visual_weight": visual_weight,
            "confidence_multiplier": confidence_multiplier,
        }

        # Generate explanation
        result.explanation = self._generate_explanation(result, inputs)

        return result

    def _compute_agreement(self, inputs: list[ModalityInput], threshold: float) -> float:
        """Compute agreement between modalities."""
        predictions = []

        for inp in inputs:
            if inp.scores is None:
                continue
            pred = (inp.scores >= threshold).astype(np.float32)
            predictions.append(pred)

        if len(predictions) < 2:
            return 1.0  # Single modality = full agreement

        # Compute pairwise agreement
        agreements = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                agreement = (predictions[i] == predictions[j]).mean()
                agreements.append(agreement)

        return float(np.mean(agreements))

    def _check_localization(self, visual_inputs: list[ModalityInput]) -> bool:
        """Check if anomaly is well-localized in visual detectors."""
        for inp in visual_inputs:
            if inp.anomaly_maps is not None:
                # Check if anomaly map has strong localization
                amap = (
                    inp.anomaly_maps.numpy()
                    if isinstance(inp.anomaly_maps, torch.Tensor)
                    else inp.anomaly_maps
                )
                if amap.max() > 0:
                    # Compute localization score (ratio of high values to total)
                    high_ratio = (amap > amap.max() * 0.5).mean()
                    if high_ratio < 0.3:  # Well-localized
                        return True
        return False

    def _check_semantic(self, vlm_inputs: list[ModalityInput]) -> bool:
        """Check if VLM detected semantic anomaly."""
        for inp in vlm_inputs:
            if inp.explanations:
                # Check for semantic keywords in explanations
                semantic_keywords = [
                    "unusual",
                    "unexpected",
                    "anomal",
                    "strange",
                    "wrong",
                    "incorrect",
                    "missing",
                    "extra",
                ]
                for exp in inp.explanations:
                    exp_lower = exp.lower()
                    if any(kw in exp_lower for kw in semantic_keywords):
                        return True
        return False

    def _generate_explanation(self, result: FusionResult, inputs: list[ModalityInput]) -> str:
        """Generate human-readable explanation of fusion decision."""
        parts = []

        meta = result.metadata
        if meta.get("agreement_score", 0) > 0.8:
            parts.append("Detectors strongly agree on the assessment.")
        elif meta.get("agreement_score", 0) < 0.3:
            parts.append("Detectors show disagreement; result has higher uncertainty.")

        if meta.get("is_semantic"):
            parts.append("Semantic anomaly detected by VLM analysis.")
        if meta.get("is_localized"):
            parts.append("Anomaly is well-localized in visual analysis.")

        # Include VLM explanations
        for inp in inputs:
            if inp.modality_type == "vlm" and inp.explanations:
                parts.append(f"VLM ({inp.detector_name}): {inp.explanations[0]}")

        return " ".join(parts) if parts else "Standard fusion applied."


class MultiModalFusionOptimizer:
    """
    High-level optimizer for multi-modal anomaly detection fusion.

    Manages multiple fusion strategies and selects optimal approach
    based on validation performance or input characteristics.
    """

    def __init__(
        self,
        default_strategy: FusionStrategy = FusionStrategy.ADAPTIVE,
        threshold: float = 0.5,
    ):
        """
        Initialize fusion optimizer.

        Args:
            default_strategy: Default fusion strategy to use
            threshold: Decision threshold
        """
        self.default_strategy = default_strategy
        self.threshold = threshold

        # Initialize fusion modules
        self._modules: dict[FusionStrategy, BaseFusionModule] = {
            FusionStrategy.FEATURE_CONCAT: FeatureConcatFusion(output_dim=256),
            FusionStrategy.FEATURE_ATTENTION: AttentionFusion(feature_dim=128),
            FusionStrategy.SCORE_AVERAGE: ScoreWeightedFusion(uncertainty_weighting=False),
            FusionStrategy.SCORE_WEIGHTED: ScoreWeightedFusion(uncertainty_weighting=True),
            FusionStrategy.SCORE_MAX: ScoreWeightedFusion(),  # Will be configured
            FusionStrategy.DECISION_VOTING: DecisionConfidenceFusion(require_consensus=True),
            FusionStrategy.DECISION_CONFIDENCE: DecisionConfidenceFusion(),
            FusionStrategy.ADAPTIVE: AdaptiveFusion(),
        }

        # Performance tracking
        self._strategy_performance: dict[FusionStrategy, list[float]] = {
            s: [] for s in FusionStrategy
        }

    def fuse(
        self,
        inputs: list[ModalityInput],
        strategy: FusionStrategy | None = None,
    ) -> FusionResult:
        """
        Fuse multi-modal inputs.

        Args:
            inputs: List of modality inputs
            strategy: Fusion strategy (None = use default)

        Returns:
            Fused result
        """
        strategy = strategy or self.default_strategy
        module = self._modules.get(strategy)

        if module is None:
            logger.warning(f"Unknown strategy {strategy}, using adaptive")
            module = self._modules[FusionStrategy.ADAPTIVE]

        result = module.fuse(inputs, self.threshold)
        result.metadata["fusion_strategy"] = strategy.value

        return result

    def fuse_all_strategies(
        self, inputs: list[ModalityInput]
    ) -> dict[FusionStrategy, FusionResult]:
        """
        Apply all fusion strategies and return results.

        Useful for comparison and strategy selection.

        Args:
            inputs: List of modality inputs

        Returns:
            Dict mapping strategy to result
        """
        results = {}
        for strategy, module in self._modules.items():
            try:
                results[strategy] = module.fuse(inputs, self.threshold)
            except Exception as e:
                logger.warning(f"Strategy {strategy} failed: {e}")

        return results

    def update_performance(
        self,
        strategy: FusionStrategy,
        score: float,
    ) -> None:
        """
        Update performance tracking for a strategy.

        Args:
            strategy: Fusion strategy
            score: Performance score (e.g., accuracy, F1)
        """
        self._strategy_performance[strategy].append(score)

        # Keep only recent scores
        max_history = 100
        if len(self._strategy_performance[strategy]) > max_history:
            self._strategy_performance[strategy] = self._strategy_performance[strategy][
                -max_history:
            ]

    def get_best_strategy(self) -> FusionStrategy:
        """
        Get best performing strategy based on history.

        Returns:
            Best fusion strategy
        """
        best_strategy = self.default_strategy
        best_score = 0.0

        for strategy, scores in self._strategy_performance.items():
            if scores:
                avg_score = np.mean(scores)
                if avg_score > best_score:
                    best_score = avg_score  # type: ignore[assignment, unused-ignore]
                    best_strategy = strategy

        return best_strategy

    def set_weights(
        self,
        strategy: FusionStrategy,
        weights: dict[str, float],
    ) -> None:
        """
        Set detector weights for a fusion strategy.

        Args:
            strategy: Fusion strategy
            weights: Detector name to weight mapping
        """
        module = self._modules.get(strategy)
        if module is not None and hasattr(module, "weights"):
            module.weights = weights

    def get_statistics(self) -> dict[str, Any]:
        """Get fusion statistics."""
        stats = {}
        for strategy, scores in self._strategy_performance.items():
            if scores:
                stats[strategy.value] = {
                    "mean_score": float(np.mean(scores)),
                    "std_score": float(np.std(scores)),
                    "num_samples": len(scores),
                }

        return stats


# Factory function
def create_fusion_optimizer(
    strategy: str = "adaptive",
    threshold: float = 0.5,
    **kwargs: Any,
) -> MultiModalFusionOptimizer:
    """
    Create a fusion optimizer with specified configuration.

    Args:
        strategy: Default fusion strategy name
        threshold: Decision threshold
        **kwargs: Additional configuration

    Returns:
        Configured fusion optimizer
    """
    strategy_enum = FusionStrategy(strategy)
    return MultiModalFusionOptimizer(
        default_strategy=strategy_enum,
        threshold=threshold,
    )
