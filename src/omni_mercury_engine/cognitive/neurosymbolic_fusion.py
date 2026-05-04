"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see
https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
Neuro-Symbolic Fusion Engine - Hybrid Anomaly Scoring

Integrates the Neural Memory Layer and Symbolic Logic Layer into a unified
neuro-symbolic architecture for hybrid anomaly detection and scoring.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                  NeurosymbolicFusionEngine                   │
    │                                                              │
    │  ┌──────────────────┐      ┌──────────────────┐            │
    │  │ NeuralMemoryLayer │ ──→ │ SymbolicLogicLayer│            │
    │  │  - Embeddings     │      │  - Logic Graphs   │            │
    │  │  - Clustering     │      │  - Rules          │            │
    │  │  - Patterns       │      │  - Explanations   │            │
    │  └──────────────────┘      └──────────────────┘            │
    │           │                        │                        │
    │           └────────┬───────────────┘                        │
    │                    ▼                                        │
    │           ┌──────────────────┐                             │
    │           │  Hybrid Fusion   │                             │
    │           │  - Weighted Avg  │                             │
    │           │  - Attention     │                             │
    │           │  - Confidence    │                             │
    │           └──────────────────┘                             │
    │                    │                                        │
    │                    ▼                                        │
    │           ┌──────────────────┐                             │
    │           │  Anomaly Score   │                             │
    │           │  + Explanation   │                             │
    │           └──────────────────┘                             │
    └─────────────────────────────────────────────────────────────┘

Research Sources:
- Neuro-Symbolic AI: The 3rd Wave (Garcez & Lamb, 2020)
- Logic Tensor Networks (Serafini & Garcez, 2016)
- Attention-based Fusion (Vaswani et al., 2017)
- Bayesian Deep Learning (Gal & Ghahramani, 2016)

Integration:
    This module is the main interface for the neuro-symbolic architecture,
    combining neural pattern detection with symbolic reasoning for
    explainable anomaly detection.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from omni_mercury_engine.cognitive.neural_memory_layer import (
    MemoryType,
    NeuralMemoryLayer,
)
from omni_mercury_engine.cognitive.symbolic_logic_layer import (
    DecisionType,
    ExplainableDecision,
    RuleType,
    SymbolicLogicLayer,
)

logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """Strategies for combining neural and symbolic outputs."""

    WEIGHTED_AVERAGE = "weighted_average"
    ATTENTION = "attention"
    GATED = "gated"
    HIERARCHICAL = "hierarchical"
    CONFIDENCE_WEIGHTED = "confidence_weighted"


class AnomalyCategory(Enum):
    """Categories of detected anomalies."""

    BEHAVIORAL = "behavioral"
    STRUCTURAL = "structural"
    TEMPORAL = "temporal"
    CONTEXTUAL = "contextual"
    COLLECTIVE = "collective"
    ETHICAL = "ethical"


@dataclass
class HybridAnomalyScore:
    """Hybrid anomaly score combining neural and symbolic components."""

    score_id: str
    anomaly_score: float
    neural_score: float
    symbolic_score: float
    confidence: float
    category: AnomalyCategory
    is_anomaly: bool
    explanation: str
    neural_patterns: list[str]
    symbolic_rules: list[str]
    fusion_strategy: FusionStrategy
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionResult:
    """Complete result from neuro-symbolic fusion."""

    result_id: str
    anomaly_scores: list[HybridAnomalyScore]
    overall_score: float
    overall_confidence: float
    decision: ExplainableDecision
    neural_contribution: float
    symbolic_contribution: float
    patterns_detected: int
    rules_fired: int
    explanation: str
    audit_trail: list[dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class AttentionMechanism:
    """
    Attention mechanism for neural-symbolic fusion.

    Learns to weight neural vs symbolic contributions based on context.
    """

    def __init__(self, hidden_dim: int = 32) -> None:
        """
        Initialize attention mechanism.

        Args:
            hidden_dim: Hidden dimension for attention computation
        """
        self.hidden_dim = hidden_dim
        np.random.seed(42)
        self.W_neural = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.W_symbolic = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.W_attention = np.random.randn(hidden_dim, 1) * 0.1

    def compute_attention(
        self,
        neural_features: np.ndarray[Any, Any],
        symbolic_features: np.ndarray[Any, Any],
    ) -> tuple[float, float]:
        """
        Compute attention weights for neural and symbolic components.

        Args:
            neural_features: Features from neural layer
            symbolic_features: Features from symbolic layer

        Returns:
            Tuple of (neural_weight, symbolic_weight)
        """
        neural_padded = self._pad_or_truncate(neural_features, self.hidden_dim)
        symbolic_padded = self._pad_or_truncate(symbolic_features, self.hidden_dim)

        neural_proj = np.tanh(neural_padded @ self.W_neural)
        symbolic_proj = np.tanh(symbolic_padded @ self.W_symbolic)

        # Use .item() to extract scalar from 1D array result of matrix multiplication
        # float() fails on 1D arrays: "only 0-dimensional arrays can be converted to Python scalars"
        neural_score = (neural_proj @ self.W_attention).item()
        symbolic_score = (symbolic_proj @ self.W_attention).item()

        exp_neural = np.exp(neural_score - max(neural_score, symbolic_score))
        exp_symbolic = np.exp(symbolic_score - max(neural_score, symbolic_score))
        total = exp_neural + exp_symbolic

        neural_weight = exp_neural / total
        symbolic_weight = exp_symbolic / total

        return float(neural_weight), float(symbolic_weight)

    def _pad_or_truncate(self, arr: np.ndarray[Any, Any], target_len: int) -> np.ndarray[Any, Any]:
        """Pad or truncate array to target length."""
        if len(arr) >= target_len:
            return arr[:target_len]
        return np.pad(arr, (0, target_len - len(arr)))


class GatedFusion:
    """
    Gated fusion mechanism for combining neural and symbolic outputs.

    Uses learned gates to control information flow between components.
    """

    def __init__(self) -> None:
        """Initialize gated fusion."""
        self.gate_bias = 0.5

    def fuse(
        self,
        neural_score: float,
        symbolic_score: float,
        neural_confidence: float,
        symbolic_confidence: float,
    ) -> tuple[float, float]:
        """
        Fuse neural and symbolic scores using gating.

        Args:
            neural_score: Score from neural layer
            symbolic_score: Score from symbolic layer
            neural_confidence: Confidence in neural score
            symbolic_confidence: Confidence in symbolic score

        Returns:
            Tuple of (fused_score, fused_confidence)
        """
        gate = 1 / (1 + np.exp(-(neural_confidence - symbolic_confidence)))

        fused_score = gate * neural_score + (1 - gate) * symbolic_score

        fused_confidence = gate * neural_confidence + (1 - gate) * symbolic_confidence

        return float(fused_score), float(fused_confidence)


class NeurosymbolicFusionEngine:
    """
    Neuro-Symbolic Fusion Engine - Main interface for hybrid anomaly detection.

    Combines neural pattern detection with symbolic reasoning to produce
    explainable anomaly scores with full audit trails.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        n_clusters: int = 8,
        confidence_threshold: float = 0.7,
        benevolence_threshold: float = 0.99,
        fusion_strategy: FusionStrategy = FusionStrategy.CONFIDENCE_WEIGHTED,
        neural_weight: float = 0.6,
        symbolic_weight: float = 0.4,
    ):
        """
        Initialize Neuro-Symbolic Fusion Engine.

        Args:
            embedding_dim: Dimension for memory embeddings
            n_clusters: Number of clusters for pattern detection
            confidence_threshold: Minimum confidence for decisions
            benevolence_threshold: Minimum benevolence score for actions
            fusion_strategy: Strategy for combining neural and symbolic
            neural_weight: Default weight for neural component
            symbolic_weight: Default weight for symbolic component
        """
        self.embedding_dim = embedding_dim
        self.n_clusters = n_clusters
        self.confidence_threshold = confidence_threshold
        self.benevolence_threshold = benevolence_threshold
        self.fusion_strategy = fusion_strategy
        self.neural_weight = neural_weight
        self.symbolic_weight = symbolic_weight

        self.neural_layer = NeuralMemoryLayer(
            embedding_dim=embedding_dim,
            n_clusters=n_clusters,
        )

        self.symbolic_layer = SymbolicLogicLayer(
            confidence_threshold=confidence_threshold,
            benevolence_threshold=benevolence_threshold,
        )

        self.attention = AttentionMechanism(hidden_dim=32)
        self.gated_fusion = GatedFusion()

        self._result_counter = 0
        self._score_counter = 0

        self._initialize_fusion_rules()

        logger.info(
            f"NeurosymbolicFusionEngine initialized "
            f"(strategy={fusion_strategy.value}, neural={neural_weight}, symbolic={symbolic_weight})"
        )

    def _initialize_fusion_rules(self) -> None:
        """Initialize rules for fusion-specific reasoning."""
        self.symbolic_layer.add_custom_rule(
            premise="neural_high_confidence",
            conclusion="trust_neural",
            rule_type=RuleType.IMPLICATION,
            confidence=0.9,
            category="fusion",
        )

        self.symbolic_layer.add_custom_rule(
            premise="symbolic_high_confidence",
            conclusion="trust_symbolic",
            rule_type=RuleType.IMPLICATION,
            confidence=0.9,
            category="fusion",
        )

        self.symbolic_layer.add_custom_rule(
            premise="pattern_escalation",
            conclusion="high_priority",
            rule_type=RuleType.IMPLICATION,
            confidence=0.95,
            category="fusion",
        )

        self.symbolic_layer.add_custom_rule(
            premise="multiple_anomaly_patterns",
            conclusion="requires_investigation",
            rule_type=RuleType.IMPLICATION,
            confidence=0.9,
            category="fusion",
        )

    def ingest_data(
        self,
        data: list[dict[str, Any]],
        memory_type: MemoryType = MemoryType.EPISODIC,
    ) -> int:
        """
        Ingest data into the neural memory layer.

        Args:
            data: List of data entries to ingest
            memory_type: Type of memory to create

        Returns:
            Number of entries ingested
        """
        embeddings = self.neural_layer.ingest_memories(data, memory_type)
        return len(embeddings)

    def analyze(
        self,
        context: dict[str, Any] | None = None,
    ) -> FusionResult:
        """
        Perform full neuro-symbolic analysis.

        Args:
            context: Additional context for analysis

        Returns:
            FusionResult with hybrid anomaly scores and explanations
        """
        self._result_counter += 1
        result_id = f"fusion_{self._result_counter:06d}"
        context = context or {}

        audit_trail: list[dict[str, Any]] = []

        audit_trail.append(
            {
                "step": "start",
                "timestamp": time.time(),
                "context": context,
            }
        )

        neural_analysis = self.neural_layer.analyze()
        neural_features = self.neural_layer.get_neural_features()

        audit_trail.append(
            {
                "step": "neural_analysis",
                "timestamp": time.time(),
                "patterns_detected": neural_analysis.get("patterns_detected", 0),
                "predictions_made": neural_analysis.get("predictions_made", 0),
            }
        )

        context_facts = self._extract_facts_from_patterns(neural_analysis)

        neural_values = self._extract_values_from_neural(neural_analysis, neural_features)

        symbolic_decision = self.symbolic_layer.process_neural_output(
            neural_features=neural_values,
            context_facts=context_facts,
        )
        symbolic_features = self.symbolic_layer.get_symbolic_features()

        audit_trail.append(
            {
                "step": "symbolic_analysis",
                "timestamp": time.time(),
                "decision_type": symbolic_decision.decision_type.value,
                "rules_fired": len(symbolic_decision.rules_fired),
            }
        )

        anomaly_scores = self._compute_hybrid_scores(
            neural_analysis,
            symbolic_decision,
            neural_features,
            symbolic_features,
        )

        overall_score, overall_confidence = self._compute_overall_score(
            anomaly_scores,
            neural_features,
            symbolic_features,
        )

        neural_contribution, symbolic_contribution = self._compute_contributions(
            neural_features,
            symbolic_features,
        )

        explanation = self._generate_explanation(
            anomaly_scores,
            symbolic_decision,
            neural_analysis,
            overall_score,
        )

        audit_trail.append(
            {
                "step": "fusion_complete",
                "timestamp": time.time(),
                "overall_score": overall_score,
                "overall_confidence": overall_confidence,
            }
        )

        return FusionResult(
            result_id=result_id,
            anomaly_scores=anomaly_scores,
            overall_score=overall_score,
            overall_confidence=overall_confidence,
            decision=symbolic_decision,
            neural_contribution=neural_contribution,
            symbolic_contribution=symbolic_contribution,
            patterns_detected=neural_analysis.get("patterns_detected", 0),
            rules_fired=len(symbolic_decision.rules_fired),
            explanation=explanation,
            audit_trail=audit_trail,
        )

    def score_single(
        self,
        data: dict[str, Any],
        memory_type: MemoryType = MemoryType.EPISODIC,
    ) -> HybridAnomalyScore:
        """
        Score a single data point for anomalies.

        Args:
            data: Data point to score
            memory_type: Type of memory

        Returns:
            HybridAnomalyScore for the data point
        """
        self._score_counter += 1
        score_id = f"score_{self._score_counter:06d}"

        embeddings = self.neural_layer.ingest_memories([data], memory_type)

        if embeddings:
            neural_score = self.neural_layer.get_anomaly_score(embeddings[0].embedding)
        else:
            neural_score = 0.5

        # Get neural features for symbolic processing
        neural_values = {
            "anomaly_score": neural_score,
            "deviation_score": neural_score * 3,
        }

        symbolic_decision = self.symbolic_layer.process_neural_output(
            neural_features=neural_values,
            context_facts=set(),
        )

        symbolic_score = symbolic_decision.symbolic_contribution

        fused_score, fused_confidence = self._fuse_scores(
            neural_score,
            symbolic_score,
            0.8,
            symbolic_decision.confidence,
        )

        is_anomaly = fused_score > 0.5 or symbolic_decision.decision_type in [
            DecisionType.ANOMALY,
            DecisionType.ESCALATE,
        ]

        category = self._determine_category(
            neural_score,
            symbolic_decision,
        )

        explanation = self._generate_single_explanation(
            neural_score,
            symbolic_score,
            fused_score,
            symbolic_decision,
        )

        return HybridAnomalyScore(
            score_id=score_id,
            anomaly_score=fused_score,
            neural_score=neural_score,
            symbolic_score=symbolic_score,
            confidence=fused_confidence,
            category=category,
            is_anomaly=is_anomaly,
            explanation=explanation,
            neural_patterns=[],
            symbolic_rules=symbolic_decision.rules_fired,
            fusion_strategy=self.fusion_strategy,
        )

    def _extract_facts_from_patterns(
        self,
        neural_analysis: dict[str, Any],
    ) -> set[str]:
        """Extract symbolic facts from neural patterns."""
        facts = set()

        patterns = neural_analysis.get("patterns", [])
        for pattern in patterns:
            pattern_type = pattern.get("type", "")
            if pattern_type == "anomaly":
                facts.add("neural_anomaly_detected")
            elif pattern_type == "escalation":
                facts.add("pattern_escalation")
            elif pattern_type == "trend":
                facts.add("trend_detected")
            elif pattern_type == "novelty":
                facts.add("novelty_detected")

        if len([p for p in patterns if p.get("type") == "anomaly"]) >= 2:
            facts.add("multiple_anomaly_patterns")

        predictions = neural_analysis.get("predictions", [])
        high_prob_predictions = [p for p in predictions if p.get("probability", 0) > 0.7]
        if high_prob_predictions:
            facts.add("high_probability_prediction")

        return facts

    def _extract_values_from_neural(
        self,
        neural_analysis: dict[str, Any],
        neural_features: np.ndarray[Any, Any],
    ) -> dict[str, float]:
        """Extract numeric values from neural analysis."""
        values = {}

        patterns = neural_analysis.get("patterns", [])
        if patterns:
            confidences = [p.get("confidence", 0.5) for p in patterns]
            values["anomaly_score"] = max(confidences) if confidences else 0.0
        else:
            values["anomaly_score"] = 0.0

        escalations = [p for p in patterns if p.get("type") == "escalation"]
        if escalations:
            values["escalation_rate"] = max(p.get("confidence", 0) for p in escalations)
        else:
            values["escalation_rate"] = 0.0

        trends = [p for p in patterns if p.get("type") == "trend"]
        if trends:
            values["deviation_score"] = max(p.get("confidence", 0) for p in trends) * 3
        else:
            values["deviation_score"] = 0.0

        if len(neural_features) > 0:
            values["neural_confidence"] = float(np.mean(np.abs(neural_features[:10])))
        else:
            values["neural_confidence"] = 0.5

        return values

    def _compute_hybrid_scores(
        self,
        neural_analysis: dict[str, Any],
        symbolic_decision: ExplainableDecision,
        neural_features: np.ndarray[Any, Any],
        symbolic_features: np.ndarray[Any, Any],
    ) -> list[HybridAnomalyScore]:
        """Compute hybrid anomaly scores for detected patterns."""
        scores = []

        patterns = neural_analysis.get("patterns", [])

        for pattern in patterns:
            self._score_counter += 1
            score_id = f"score_{self._score_counter:06d}"

            neural_score = pattern.get("confidence", 0.5)

            symbolic_score = symbolic_decision.symbolic_contribution

            fused_score, fused_confidence = self._fuse_scores(
                neural_score,
                symbolic_score,
                neural_score,
                symbolic_decision.confidence,
            )

            pattern_type = pattern.get("type", "unknown")
            category = self._pattern_type_to_category(pattern_type)

            is_anomaly = fused_score > 0.5 or pattern_type == "anomaly"

            explanation = f"Pattern '{pattern.get('id', 'unknown')}': {pattern.get('description', 'No description')}"

            scores.append(
                HybridAnomalyScore(
                    score_id=score_id,
                    anomaly_score=fused_score,
                    neural_score=neural_score,
                    symbolic_score=symbolic_score,
                    confidence=fused_confidence,
                    category=category,
                    is_anomaly=is_anomaly,
                    explanation=explanation,
                    neural_patterns=[pattern.get("id", "")],
                    symbolic_rules=symbolic_decision.rules_fired,
                    fusion_strategy=self.fusion_strategy,
                )
            )

        return scores

    def _fuse_scores(
        self,
        neural_score: float,
        symbolic_score: float,
        neural_confidence: float,
        symbolic_confidence: float,
    ) -> tuple[float, float]:
        """Fuse neural and symbolic scores based on strategy."""
        if self.fusion_strategy == FusionStrategy.WEIGHTED_AVERAGE:
            fused_score = self.neural_weight * neural_score + self.symbolic_weight * symbolic_score
            fused_confidence = (
                self.neural_weight * neural_confidence + self.symbolic_weight * symbolic_confidence
            )

        elif self.fusion_strategy == FusionStrategy.ATTENTION:
            neural_features = np.array([neural_score, neural_confidence])
            symbolic_features = np.array([symbolic_score, symbolic_confidence])
            n_weight, s_weight = self.attention.compute_attention(
                neural_features, symbolic_features
            )
            fused_score = n_weight * neural_score + s_weight * symbolic_score
            fused_confidence = n_weight * neural_confidence + s_weight * symbolic_confidence

        elif self.fusion_strategy == FusionStrategy.GATED:
            fused_score, fused_confidence = self.gated_fusion.fuse(
                neural_score, symbolic_score, neural_confidence, symbolic_confidence
            )

        elif self.fusion_strategy == FusionStrategy.CONFIDENCE_WEIGHTED:
            total_conf = neural_confidence + symbolic_confidence + 1e-10
            n_weight = neural_confidence / total_conf
            s_weight = symbolic_confidence / total_conf
            fused_score = n_weight * neural_score + s_weight * symbolic_score
            fused_confidence = (neural_confidence + symbolic_confidence) / 2

        else:
            fused_score = max(neural_score, symbolic_score)
            fused_confidence = max(neural_confidence, symbolic_confidence)

        return float(fused_score), float(fused_confidence)

    def _compute_overall_score(
        self,
        anomaly_scores: list[HybridAnomalyScore],
        neural_features: np.ndarray[Any, Any],
        symbolic_features: np.ndarray[Any, Any],
    ) -> tuple[float, float]:
        """Compute overall anomaly score and confidence."""
        if not anomaly_scores:
            return 0.0, 0.5

        scores = [s.anomaly_score for s in anomaly_scores]
        confidences = [s.confidence for s in anomaly_scores]

        overall_score = max(scores)

        weights = np.array(confidences) / (sum(confidences) + 1e-10)
        overall_confidence = float(np.sum(weights * confidences))

        return overall_score, overall_confidence

    def _compute_contributions(
        self,
        neural_features: np.ndarray[Any, Any],
        symbolic_features: np.ndarray[Any, Any],
    ) -> tuple[float, float]:
        """Compute neural and symbolic contributions."""
        neural_magnitude = float(np.linalg.norm(neural_features))
        symbolic_magnitude = float(np.linalg.norm(symbolic_features))

        total = neural_magnitude + symbolic_magnitude + 1e-10

        return neural_magnitude / total, symbolic_magnitude / total

    def _pattern_type_to_category(self, pattern_type: str) -> AnomalyCategory:
        """Map pattern type to anomaly category."""
        mapping = {
            "anomaly": AnomalyCategory.BEHAVIORAL,
            "trend": AnomalyCategory.TEMPORAL,
            "escalation": AnomalyCategory.TEMPORAL,
            "novelty": AnomalyCategory.STRUCTURAL,
            "correlation": AnomalyCategory.COLLECTIVE,
        }
        return mapping.get(pattern_type, AnomalyCategory.CONTEXTUAL)

    def _determine_category(
        self,
        neural_score: float,
        symbolic_decision: ExplainableDecision,
    ) -> AnomalyCategory:
        """Determine anomaly category from scores and decision."""
        if symbolic_decision.decision_type == DecisionType.BLOCK:
            return AnomalyCategory.ETHICAL

        if neural_score > 0.8:
            return AnomalyCategory.BEHAVIORAL

        if "escalation" in str(symbolic_decision.rules_fired):
            return AnomalyCategory.TEMPORAL

        return AnomalyCategory.CONTEXTUAL

    def _generate_explanation(
        self,
        anomaly_scores: list[HybridAnomalyScore],
        symbolic_decision: ExplainableDecision,
        neural_analysis: dict[str, Any],
        overall_score: float,
    ) -> str:
        """Generate comprehensive explanation for fusion result."""
        parts = []

        parts.append(f"Overall Anomaly Score: {overall_score:.2%}")
        parts.append(f"Decision: {symbolic_decision.decision_type.value.upper()}")

        if anomaly_scores:
            anomalous = [s for s in anomaly_scores if s.is_anomaly]
            parts.append(f"Anomalies Detected: {len(anomalous)}/{len(anomaly_scores)}")

        patterns_count = neural_analysis.get("patterns_detected", 0)
        if patterns_count > 0:
            parts.append(f"Neural Patterns: {patterns_count}")

        if symbolic_decision.rules_fired:
            parts.append(f"Symbolic Rules Fired: {len(symbolic_decision.rules_fired)}")

        parts.append(f"Fusion Strategy: {self.fusion_strategy.value}")

        return " | ".join(parts)

    def _generate_single_explanation(
        self,
        neural_score: float,
        symbolic_score: float,
        fused_score: float,
        symbolic_decision: ExplainableDecision,
    ) -> str:
        """Generate explanation for single score."""
        parts = [
            f"Fused Score: {fused_score:.2%}",
            f"Neural: {neural_score:.2%}",
            f"Symbolic: {symbolic_score:.2%}",
            f"Decision: {symbolic_decision.decision_type.value}",
        ]

        if symbolic_decision.rules_fired:
            parts.append(f"Rules: {', '.join(symbolic_decision.rules_fired[:3])}")

        return " | ".join(parts)

    def add_rule(
        self,
        premise: str,
        conclusion: str,
        rule_type: RuleType = RuleType.IMPLICATION,
        confidence: float = 0.9,
    ) -> str:
        """Add a custom rule to the symbolic layer."""
        return self.symbolic_layer.add_custom_rule(
            premise=premise,
            conclusion=conclusion,
            rule_type=rule_type,
            confidence=confidence,
        )

    def evaluate_action(
        self,
        action: str,
        context: dict[str, Any],
        benevolence_score: float,
    ) -> tuple[bool, ExplainableDecision]:
        """
        Evaluate whether an action should be allowed.

        Args:
            action: Action to evaluate
            context: Action context
            benevolence_score: Computed benevolence score

        Returns:
            Tuple of (allowed, decision)
        """
        return self.symbolic_layer.evaluate_action(action, context, benevolence_score)

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "fusion_strategy": self.fusion_strategy.value,
            "neural_weight": self.neural_weight,
            "symbolic_weight": self.symbolic_weight,
            "confidence_threshold": self.confidence_threshold,
            "benevolence_threshold": self.benevolence_threshold,
            "results_generated": self._result_counter,
            "scores_generated": self._score_counter,
            "neural_stats": self.neural_layer.get_statistics(),
            "symbolic_stats": self.symbolic_layer.get_statistics(),
        }

    def get_audit_log(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent decision audit log."""
        return self.symbolic_layer.get_decision_history(limit=limit)
