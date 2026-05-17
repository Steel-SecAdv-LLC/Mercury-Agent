"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Tests for Neuro-Symbolic Fusion Engine - Hybrid Anomaly Scoring
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from omni_mercury_engine.cognitive.neural_memory_layer import MemoryType
from omni_mercury_engine.cognitive.neurosymbolic_fusion import (
    AnomalyCategory,
    AttentionMechanism,
    FusionResult,
    FusionStrategy,
    GatedFusion,
    HybridAnomalyScore,
    NeurosymbolicFusionEngine,
)
from omni_mercury_engine.cognitive.symbolic_logic_layer import DecisionType, RuleType


class TestAttentionMechanism:
    """Tests for AttentionMechanism."""

    def test_init(self) -> None:
        """Test attention mechanism initialization."""
        attention = AttentionMechanism(hidden_dim=32)
        assert attention.hidden_dim == 32
        assert attention.W_neural.shape == (32, 32)

    def test_compute_attention_returns_weights(self) -> None:
        """Test attention computation returns valid weights."""
        attention = AttentionMechanism(hidden_dim=16)
        neural_features = np.random.randn(10)
        symbolic_features = np.random.randn(10)

        n_weight, s_weight = attention.compute_attention(neural_features, symbolic_features)

        assert 0 <= n_weight <= 1
        assert 0 <= s_weight <= 1
        assert abs(n_weight + s_weight - 1.0) < 1e-6

    def test_compute_attention_handles_different_sizes(self) -> None:
        """Test attention handles different feature sizes."""
        attention = AttentionMechanism(hidden_dim=32)

        n_weight, s_weight = attention.compute_attention(
            np.random.randn(5),
            np.random.randn(50),
        )

        assert 0 <= n_weight <= 1
        assert 0 <= s_weight <= 1


class TestGatedFusion:
    """Tests for GatedFusion."""

    def test_init(self) -> None:
        """Test gated fusion initialization."""
        fusion = GatedFusion()
        assert fusion.gate_bias == 0.5

    def test_fuse_equal_confidence(self) -> None:
        """Test fusion with equal confidence."""
        fusion = GatedFusion()
        fused_score, fused_conf = fusion.fuse(0.8, 0.6, 0.5, 0.5)

        assert 0 <= fused_score <= 1
        assert 0 <= fused_conf <= 1
        assert fused_score == pytest.approx(0.7, abs=0.01)

    def test_fuse_high_neural_confidence(self) -> None:
        """Test fusion with high neural confidence."""
        fusion = GatedFusion()
        fused_score, fused_conf = fusion.fuse(0.9, 0.3, 0.9, 0.1)

        assert fused_score > 0.7

    def test_fuse_high_symbolic_confidence(self) -> None:
        """Test fusion with high symbolic confidence."""
        fusion = GatedFusion()
        fused_score, fused_conf = fusion.fuse(0.3, 0.9, 0.1, 0.9)

        assert fused_score > 0.7


class TestHybridAnomalyScore:
    """Tests for HybridAnomalyScore dataclass."""

    def test_create_score(self) -> None:
        """Test creating a hybrid anomaly score."""
        score = HybridAnomalyScore(
            score_id="test_001",
            anomaly_score=0.75,
            neural_score=0.8,
            symbolic_score=0.7,
            confidence=0.85,
            category=AnomalyCategory.BEHAVIORAL,
            is_anomaly=True,
            explanation="Test explanation",
            neural_patterns=["pattern_1"],
            symbolic_rules=["rule_1", "rule_2"],
            fusion_strategy=FusionStrategy.WEIGHTED_AVERAGE,
        )

        assert score.score_id == "test_001"
        assert score.anomaly_score == 0.75
        assert score.is_anomaly is True
        assert len(score.symbolic_rules) == 2


class TestFusionResult:
    """Tests for FusionResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating a fusion result."""
        from omni_mercury_engine.cognitive.symbolic_logic_layer import (
            DecisionType,
            ExplainableDecision,
            ExplanationType,
        )

        decision = ExplainableDecision(
            decision_id="dec_001",
            decision_type=DecisionType.ANOMALY,
            confidence=0.9,
            explanation="Test",
            explanation_type=ExplanationType.HYBRID,
            rules_fired=["rule_1"],
        )

        result = FusionResult(
            result_id="result_001",
            anomaly_scores=[],
            overall_score=0.8,
            overall_confidence=0.85,
            decision=decision,
            neural_contribution=0.6,
            symbolic_contribution=0.4,
            patterns_detected=3,
            rules_fired=2,
            explanation="Test explanation",
        )

        assert result.result_id == "result_001"
        assert result.overall_score == 0.8
        assert result.neural_contribution == 0.6


class TestNeurosymbolicFusionEngine:
    """Tests for NeurosymbolicFusionEngine main interface."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        engine = NeurosymbolicFusionEngine()
        assert engine.embedding_dim == 64
        assert engine.n_clusters == 8
        assert engine.fusion_strategy == FusionStrategy.CONFIDENCE_WEIGHTED

    def test_init_custom(self) -> None:
        """Test custom initialization."""
        engine = NeurosymbolicFusionEngine(
            embedding_dim=32,
            n_clusters=4,
            fusion_strategy=FusionStrategy.ATTENTION,
            neural_weight=0.7,
            symbolic_weight=0.3,
        )
        assert engine.embedding_dim == 32
        assert engine.n_clusters == 4
        assert engine.fusion_strategy == FusionStrategy.ATTENTION
        assert engine.neural_weight == 0.7

    def test_ingest_data(self) -> None:
        """Test data ingestion."""
        engine = NeurosymbolicFusionEngine(n_clusters=3)
        data = [
            {"id": "d1", "event": "test1", "timestamp": time.time()},
            {"id": "d2", "event": "test2", "timestamp": time.time()},
            {"id": "d3", "event": "test3", "timestamp": time.time()},
        ]

        count = engine.ingest_data(data, MemoryType.EPISODIC)
        assert count == 3

    def test_analyze_empty(self) -> None:
        """Test analysis with no data."""
        engine = NeurosymbolicFusionEngine(n_clusters=3)
        result = engine.analyze()

        assert result.result_id.startswith("fusion_")
        assert result.overall_score == 0.0
        assert len(result.audit_trail) > 0

    def test_analyze_with_data(self) -> None:
        """Test analysis with data."""
        engine = NeurosymbolicFusionEngine(embedding_dim=16, n_clusters=3)

        data = [
            {"id": f"d{i}", "event": f"event_{i}", "value": i * 10, "timestamp": time.time() + i}
            for i in range(15)
        ]
        engine.ingest_data(data)

        result = engine.analyze()

        assert result.result_id.startswith("fusion_")
        assert "audit_trail" in result.__dict__
        assert result.decision is not None

    def test_score_single(self) -> None:
        """Test scoring a single data point."""
        engine = NeurosymbolicFusionEngine(embedding_dim=16, n_clusters=3)

        data = [{"id": f"d{i}", "event": "baseline", "timestamp": time.time()} for i in range(10)]
        engine.ingest_data(data)

        score = engine.score_single(
            {"id": "test", "event": "anomalous_event", "timestamp": time.time()},
            MemoryType.EPISODIC,
        )

        assert score.score_id.startswith("score_")
        assert 0 <= score.anomaly_score <= 1
        assert 0 <= score.confidence <= 1
        assert score.explanation is not None

    def test_fusion_strategies(self) -> None:
        """Test different fusion strategies."""
        strategies = [
            FusionStrategy.WEIGHTED_AVERAGE,
            FusionStrategy.ATTENTION,
            FusionStrategy.GATED,
            FusionStrategy.CONFIDENCE_WEIGHTED,
        ]

        for strategy in strategies:
            engine = NeurosymbolicFusionEngine(
                embedding_dim=16,
                n_clusters=3,
                fusion_strategy=strategy,
            )

            data = [{"id": f"d{i}", "event": "test"} for i in range(5)]
            engine.ingest_data(data)

            result = engine.analyze()
            assert result is not None

    def test_add_rule(self) -> None:
        """Test adding custom rules."""
        engine = NeurosymbolicFusionEngine()
        initial_rules = len(engine.symbolic_layer.reasoner.logic_graph.rules)

        rule_id = engine.add_rule(
            premise="custom_condition",
            conclusion="custom_result",
            rule_type=RuleType.IMPLICATION,
            confidence=0.95,
        )

        assert rule_id is not None
        assert len(engine.symbolic_layer.reasoner.logic_graph.rules) == initial_rules + 1

    def test_evaluate_action_allowed(self) -> None:
        """Test evaluating allowed action."""
        engine = NeurosymbolicFusionEngine(benevolence_threshold=0.9)

        allowed, decision = engine.evaluate_action(
            action="safe_action",
            context={"humanitarian": True},
            benevolence_score=0.95,
        )

        assert allowed is True

    def test_evaluate_action_blocked(self) -> None:
        """Test evaluating blocked action."""
        engine = NeurosymbolicFusionEngine(benevolence_threshold=0.99)

        allowed, decision = engine.evaluate_action(
            action="risky_action",
            context={"potential_harm": True},
            benevolence_score=0.99,
        )

        assert allowed is False
        assert decision.decision_type == DecisionType.BLOCK

    def test_get_statistics(self) -> None:
        """Test statistics retrieval."""
        engine = NeurosymbolicFusionEngine()
        engine.ingest_data([{"id": "test", "event": "data"}])
        engine.analyze()

        stats = engine.get_statistics()

        assert "fusion_strategy" in stats
        assert "neural_stats" in stats
        assert "symbolic_stats" in stats
        assert stats["results_generated"] == 1

    def test_get_audit_log(self) -> None:
        """Test audit log retrieval."""
        engine = NeurosymbolicFusionEngine()
        engine.ingest_data([{"id": "test", "event": "data"}])
        engine.analyze()

        log = engine.get_audit_log(limit=10)

        assert isinstance(log, list)


class TestAnomalyCategories:
    """Tests for anomaly category enums."""

    def test_anomaly_categories(self) -> None:
        """Test all anomaly categories exist."""
        assert AnomalyCategory.BEHAVIORAL.value == "behavioral"
        assert AnomalyCategory.STRUCTURAL.value == "structural"
        assert AnomalyCategory.TEMPORAL.value == "temporal"
        assert AnomalyCategory.CONTEXTUAL.value == "contextual"
        assert AnomalyCategory.COLLECTIVE.value == "collective"
        assert AnomalyCategory.ETHICAL.value == "ethical"


class TestFusionStrategies:
    """Tests for fusion strategy enums."""

    def test_fusion_strategies(self) -> None:
        """Test all fusion strategies exist."""
        assert FusionStrategy.WEIGHTED_AVERAGE.value == "weighted_average"
        assert FusionStrategy.ATTENTION.value == "attention"
        assert FusionStrategy.GATED.value == "gated"
        assert FusionStrategy.HIERARCHICAL.value == "hierarchical"
        assert FusionStrategy.CONFIDENCE_WEIGHTED.value == "confidence_weighted"


class TestIntegration:
    """Integration tests for the full neuro-symbolic pipeline."""

    def test_full_pipeline(self) -> None:
        """Test complete neuro-symbolic analysis pipeline."""
        engine = NeurosymbolicFusionEngine(
            embedding_dim=32,
            n_clusters=4,
            confidence_threshold=0.7,
            benevolence_threshold=0.99,
        )

        training_data = [
            {
                "id": f"train_{i}",
                "event": "normal_operation",
                "value": i,
                "timestamp": time.time() + i,
            }
            for i in range(20)
        ]
        engine.ingest_data(training_data, MemoryType.SEMANTIC)

        result = engine.analyze(context={"source": "test"})

        assert result.result_id is not None
        assert result.decision is not None
        assert len(result.audit_trail) >= 3

        test_score = engine.score_single(
            {"id": "test_anomaly", "event": "unusual_event", "value": 999},
            MemoryType.EPISODIC,
        )

        assert test_score.score_id is not None
        assert test_score.explanation is not None

        allowed, decision = engine.evaluate_action(
            action="respond_to_anomaly",
            context={"humanitarian": True},
            benevolence_score=0.995,
        )

        assert allowed is True

    def test_ethical_blocking(self) -> None:
        """Test that ethical violations are properly blocked."""
        engine = NeurosymbolicFusionEngine(benevolence_threshold=0.99)

        allowed, decision = engine.evaluate_action(
            action="harmful_action",
            context={"potential_harm": True, "privacy_sensitive": True},
            benevolence_score=0.999,
        )

        assert allowed is False
        assert decision.decision_type == DecisionType.BLOCK

    def test_pattern_to_decision_flow(self) -> None:
        """Test that neural patterns influence symbolic decisions."""
        engine = NeurosymbolicFusionEngine(embedding_dim=16, n_clusters=3)

        data = [
            {
                "id": f"d{i}",
                "event": "escalating",
                "importance": 0.3 + i * 0.07,
                "timestamp": time.time() + i * 100,
            }
            for i in range(15)
        ]
        engine.ingest_data(data)

        result = engine.analyze()

        assert result.patterns_detected >= 0
        assert result.decision is not None
