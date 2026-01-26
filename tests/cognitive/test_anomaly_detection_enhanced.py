"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

Tests for Enhanced Anomaly Detection - Memory Graph and External Integration
"""

from __future__ import annotations

import time

from omni_mercury_engine.cognitive.anomaly_detection_enhanced import (
    BayesianPredictor,
    EnhancedAnomalyDetector,
    ExternalDataIntegrator,
    ExternalSourceCategory,
    HiddenMarkovPredictor,
    MemoryKnowledgeGraph,
    PredictionType,
    SimulatedEnvironmentalSource,
    SimulatedGeologicalSource,
    ValueExtractor,
)


class TestMemoryKnowledgeGraph:
    """Tests for MemoryKnowledgeGraph."""

    def test_init(self):
        """Test graph initialization."""
        graph = MemoryKnowledgeGraph()
        stats = graph.get_statistics()
        assert stats["num_nodes"] == 0
        assert stats["num_edges"] == 0

    def test_add_memory_node(self):
        """Test adding memory nodes."""
        graph = MemoryKnowledgeGraph()
        node_id = graph.add_memory_node(
            memory_id="test_001",
            memory_type="episodic",
            content={"event": "test"},
            importance=0.8,
        )

        assert node_id == "mem_test_001"
        stats = graph.get_statistics()
        assert stats["num_nodes"] == 1

    def test_add_relationship(self):
        """Test adding relationships between nodes."""
        graph = MemoryKnowledgeGraph()
        graph.add_memory_node("m1", "episodic", {"event": "e1"})
        graph.add_memory_node("m2", "episodic", {"event": "e2"})

        edge_id = graph.add_relationship(
            source_id="mem_m1",
            target_id="mem_m2",
            relationship_type="causes",
            weight=0.9,
        )

        assert edge_id.startswith("edge_")
        stats = graph.get_statistics()
        assert stats["num_edges"] == 1

    def test_find_related_memories(self):
        """Test finding related memories."""
        graph = MemoryKnowledgeGraph()
        graph.add_memory_node("m1", "episodic", {"event": "e1"})
        graph.add_memory_node("m2", "episodic", {"event": "e2"})
        graph.add_memory_node("m3", "episodic", {"event": "e3"})

        graph.add_relationship("mem_m1", "mem_m2", "related", 0.9)
        graph.add_relationship("mem_m2", "mem_m3", "related", 0.8)

        related = graph.find_related_memories("mem_m1", max_depth=2)
        assert len(related) >= 1

    def test_compute_centrality(self):
        """Test centrality computation."""
        graph = MemoryKnowledgeGraph()
        graph.add_memory_node("m1", "episodic", {"event": "e1"})
        graph.add_memory_node("m2", "episodic", {"event": "e2"})
        graph.add_relationship("mem_m1", "mem_m2", "related", 0.9)

        centrality = graph.compute_centrality()
        assert len(centrality) == 2


class TestBayesianPredictor:
    """Tests for BayesianPredictor."""

    def test_init(self):
        """Test predictor initialization."""
        predictor = BayesianPredictor(prior_alpha=2.0, prior_beta=2.0)
        assert predictor.prior_alpha == 2.0
        assert predictor.prior_beta == 2.0

    def test_update_success(self):
        """Test updating with success."""
        predictor = BayesianPredictor()
        predictor.update("context_1", success=True)

        assert "context_1" in predictor.contexts
        assert predictor.contexts["context_1"]["alpha"] == 2.0

    def test_update_failure(self):
        """Test updating with failure."""
        predictor = BayesianPredictor()
        predictor.update("context_1", success=False)

        assert predictor.contexts["context_1"]["beta"] == 2.0

    def test_predict_new_context(self):
        """Test prediction for new context."""
        predictor = BayesianPredictor()
        prob, interval = predictor.predict("new_context")

        assert prob == 0.5
        assert interval[0] <= prob <= interval[1]

    def test_predict_after_updates(self):
        """Test prediction after multiple updates."""
        predictor = BayesianPredictor()

        for _ in range(8):
            predictor.update("context_1", success=True)
        for _ in range(2):
            predictor.update("context_1", success=False)

        prob, interval = predictor.predict("context_1")
        assert prob > 0.7

    def test_get_confidence(self):
        """Test confidence calculation."""
        predictor = BayesianPredictor()

        conf_new = predictor.get_confidence("new_context")
        assert conf_new == 0.5

        for _ in range(60):
            predictor.update("context_1", success=True)

        conf_updated = predictor.get_confidence("context_1")
        assert conf_updated >= conf_new


class TestHiddenMarkovPredictor:
    """Tests for HiddenMarkovPredictor."""

    def test_init(self):
        """Test HMM initialization."""
        hmm = HiddenMarkovPredictor(n_states=5)
        assert hmm.n_states == 5
        assert hmm.transition_matrix.shape == (5, 5)

    def test_observe(self):
        """Test observation processing."""
        hmm = HiddenMarkovPredictor(n_states=3)
        state = hmm.observe("event_a")

        assert 0 <= state < 3
        assert len(hmm.state_history) == 1
        assert len(hmm.observation_history) == 1

    def test_observe_sequence(self):
        """Test processing sequence of observations."""
        hmm = HiddenMarkovPredictor(n_states=3)

        for obs in ["a", "b", "a", "c", "b"]:
            hmm.observe(obs)

        assert len(hmm.state_history) == 5
        assert len(hmm.observation_history) == 5

    def test_predict_next_state(self):
        """Test next state prediction."""
        hmm = HiddenMarkovPredictor(n_states=3)

        for obs in ["a", "b", "a"]:
            hmm.observe(obs)

        state, prob = hmm.predict_next_state()
        assert 0 <= state < 3
        assert 0 <= prob <= 1

    def test_detect_anomaly_no_history(self):
        """Test anomaly detection with no history."""
        hmm = HiddenMarkovPredictor(n_states=3)
        assert hmm.detect_anomaly() is False

    def test_detect_anomaly_with_history(self):
        """Test anomaly detection with history."""
        hmm = HiddenMarkovPredictor(n_states=3)

        for obs in ["a", "a", "a", "a", "a"]:
            hmm.observe(obs)

        is_anomaly = hmm.detect_anomaly(threshold=0.1)
        assert isinstance(is_anomaly, bool)


class TestExternalDataSources:
    """Tests for external data sources."""

    def test_simulated_geological_source(self):
        """Test simulated geological source."""
        source = SimulatedGeologicalSource()
        data = source.fetch()

        assert len(data) == 1
        assert data[0].source_type == ExternalSourceCategory.GEOLOGICAL
        assert "magnitude" in data[0].data

    def test_simulated_environmental_source(self):
        """Test simulated environmental source."""
        source = SimulatedEnvironmentalSource()
        data = source.fetch()

        assert len(data) == 1
        assert data[0].source_type == ExternalSourceCategory.ENVIRONMENTAL
        assert "severity" in data[0].data


class TestExternalDataIntegrator:
    """Tests for ExternalDataIntegrator."""

    def test_init(self):
        """Test integrator initialization."""
        integrator = ExternalDataIntegrator()
        assert len(integrator.sources) == 0
        assert len(integrator.data_buffer) == 0

    def test_register_source(self):
        """Test source registration."""
        integrator = ExternalDataIntegrator()
        integrator.register_source("geo", SimulatedGeologicalSource())

        assert "geo" in integrator.sources

    def test_fetch_all(self):
        """Test fetching from all sources."""
        integrator = ExternalDataIntegrator()
        integrator.register_source("geo", SimulatedGeologicalSource())
        integrator.register_source("env", SimulatedEnvironmentalSource())

        data = integrator.fetch_all()
        assert len(data) == 2

    def test_align_with_internal(self):
        """Test alignment with internal patterns."""
        integrator = ExternalDataIntegrator()
        integrator.register_source("geo", SimulatedGeologicalSource())
        integrator.fetch_all()

        internal_patterns = [
            {"type": "geological", "confidence": 0.9, "timestamp": time.time()},
        ]

        alignments = integrator.align_with_internal(internal_patterns)
        assert isinstance(alignments, list)

    def test_get_statistics(self):
        """Test statistics retrieval."""
        integrator = ExternalDataIntegrator()
        integrator.register_source("geo", SimulatedGeologicalSource())

        stats = integrator.get_statistics()
        assert stats["num_sources"] == 1


class TestValueExtractor:
    """Tests for ValueExtractor."""

    def test_init(self):
        """Test extractor initialization."""
        extractor = ValueExtractor(benevolence_threshold=0.95)
        assert extractor.benevolence_threshold == 0.95

    def test_extract_benevolent(self):
        """Test extraction with benevolent score."""
        extractor = ValueExtractor(benevolence_threshold=0.99)
        anomaly = {"id": "a1", "type": "escalation", "confidence": 0.8}

        extraction = extractor.extract(anomaly, ethical_score=0.995)

        assert extraction is not None
        assert extraction.is_benevolent is True
        assert extraction.value_type == "early_warning"

    def test_extract_not_benevolent(self):
        """Test extraction with low ethical score."""
        extractor = ValueExtractor(benevolence_threshold=0.99)
        anomaly = {"id": "a1", "type": "escalation", "confidence": 0.8}

        extraction = extractor.extract(anomaly, ethical_score=0.5)

        assert extraction is None

    def test_extract_different_types(self):
        """Test extraction for different anomaly types."""
        extractor = ValueExtractor(benevolence_threshold=0.9)

        types_and_expected = [
            ("escalation", "early_warning"),
            ("trend", "early_warning"),
            ("novelty", "discovery"),
            ("correlation", "insight"),
            ("unknown", "monitoring"),
        ]

        for anomaly_type, expected_value_type in types_and_expected:
            anomaly = {"id": "test", "type": anomaly_type, "confidence": 0.8}
            extraction = extractor.extract(anomaly, ethical_score=0.95)

            if extraction is not None:
                assert extraction.value_type == expected_value_type


class TestEnhancedAnomalyDetector:
    """Tests for EnhancedAnomalyDetector main interface."""

    def test_init(self):
        """Test detector initialization."""
        detector = EnhancedAnomalyDetector()
        assert detector.benevolence_threshold == 0.99
        assert detector.memory_graph is not None
        assert detector.bayesian_predictor is not None

    def test_add_memory(self):
        """Test adding memory to graph."""
        detector = EnhancedAnomalyDetector()
        node_id = detector.add_memory(
            memory_id="m1",
            memory_type="episodic",
            content={"event": "test"},
            importance=0.8,
        )

        assert node_id == "mem_m1"

    def test_add_memory_with_relations(self):
        """Test adding memory with relations."""
        detector = EnhancedAnomalyDetector()
        detector.add_memory("m1", "episodic", {"event": "e1"})
        node_id = detector.add_memory(
            memory_id="m2",
            memory_type="episodic",
            content={"event": "e2"},
            related_to=["m1"],
        )

        assert node_id == "mem_m2"
        stats = detector.memory_graph.get_statistics()
        assert stats["num_edges"] >= 1

    def test_update_predictor(self):
        """Test updating Bayesian predictor."""
        detector = EnhancedAnomalyDetector()
        detector.update_predictor("context_1", success=True)

        assert "context_1" in detector.bayesian_predictor.contexts

    def test_observe_sequence(self):
        """Test HMM observation."""
        detector = EnhancedAnomalyDetector()
        state = detector.observe_sequence("event_a")

        assert 0 <= state < 3

    def test_predict(self):
        """Test prediction generation."""
        detector = EnhancedAnomalyDetector()

        for _ in range(5):
            detector.update_predictor("test_context", success=True)

        result = detector.predict("test_context", include_external=False)

        assert result.prediction_id.startswith("pred_")
        assert 0 <= result.probability <= 1
        assert result.explanation is not None

    def test_predict_with_external(self):
        """Test prediction with external data."""
        detector = EnhancedAnomalyDetector()
        result = detector.predict("test_context", include_external=True)

        assert "External data points" in str(result.contributing_factors)

    def test_extract_value(self):
        """Test value extraction."""
        detector = EnhancedAnomalyDetector(benevolence_threshold=0.9)
        anomaly = {"id": "a1", "type": "escalation", "confidence": 0.8}

        extraction = detector.extract_value(anomaly, ethical_score=0.95)

        assert extraction is not None
        assert extraction.is_benevolent is True

    def test_analyze_memory_patterns(self):
        """Test memory pattern analysis."""
        detector = EnhancedAnomalyDetector()
        detector.add_memory("m1", "episodic", {"event": "e1"})
        detector.add_memory("m2", "episodic", {"event": "e2"}, related_to=["m1"])

        analysis = detector.analyze_memory_patterns("m1")

        assert analysis["memory_id"] == "m1"
        assert "related_memories" in analysis
        assert "graph_stats" in analysis

    def test_get_statistics(self):
        """Test statistics retrieval."""
        detector = EnhancedAnomalyDetector()
        detector.add_memory("m1", "episodic", {"event": "e1"})
        detector.predict("test", include_external=False)

        stats = detector.get_statistics()

        assert stats["predictions_made"] == 1
        assert "memory_graph" in stats
        assert "external_sources" in stats


class TestPredictionTypes:
    """Tests for prediction type enums."""

    def test_prediction_types(self):
        """Test all prediction types exist."""
        assert PredictionType.ANOMALY.value == "anomaly"
        assert PredictionType.TREND.value == "trend"
        assert PredictionType.ESCALATION.value == "escalation"
        assert PredictionType.OPPORTUNITY.value == "opportunity"
        assert PredictionType.RISK.value == "risk"


class TestExternalSourceCategory:
    """Tests for external source category enums."""

    def test_external_source_categories(self):
        """Test all external source categories exist."""
        assert ExternalSourceCategory.GEOLOGICAL.value == "geological"
        assert ExternalSourceCategory.ENVIRONMENTAL.value == "environmental"
        assert ExternalSourceCategory.NEWS.value == "news"
        assert ExternalSourceCategory.FINANCIAL.value == "financial"
        assert ExternalSourceCategory.SOCIAL.value == "social"
        assert ExternalSourceCategory.SECURITY.value == "security"
        assert ExternalSourceCategory.HEALTH.value == "health"
        assert ExternalSourceCategory.CUSTOM.value == "custom"


class TestIntegration:
    """Integration tests for enhanced anomaly detection."""

    def test_full_pipeline(self):
        """Test complete enhanced detection pipeline."""
        detector = EnhancedAnomalyDetector(benevolence_threshold=0.95)

        for i in range(10):
            detector.add_memory(
                memory_id=f"m{i}",
                memory_type="episodic",
                content={"event": f"event_{i}", "value": i},
                importance=0.5 + i * 0.05,
                related_to=[f"m{i-1}"] if i > 0 else None,
            )

        for i in range(20):
            success = i % 3 != 0
            detector.update_predictor("main_context", success=success)

        for obs in ["normal", "normal", "alert", "normal", "critical"]:
            detector.observe_sequence(obs)

        prediction = detector.predict("main_context", include_external=True)

        assert prediction.prediction_id is not None
        assert len(prediction.contributing_factors) >= 2

        anomaly = {"id": "test_anomaly", "type": "escalation", "confidence": 0.85}
        extraction = detector.extract_value(anomaly, ethical_score=0.98)

        assert extraction is not None
        assert extraction.is_benevolent is True

        analysis = detector.analyze_memory_patterns("m5")
        assert len(analysis["related_memories"]) >= 0

    def test_ethical_filtering(self):
        """Test that unethical extractions are blocked."""
        detector = EnhancedAnomalyDetector(benevolence_threshold=0.99)

        anomaly = {"id": "risky", "type": "opportunity", "confidence": 0.9}

        extraction_low = detector.extract_value(anomaly, ethical_score=0.5)
        assert extraction_low is None

        extraction_high = detector.extract_value(anomaly, ethical_score=0.995)
        assert extraction_high is not None
        assert extraction_high.is_benevolent is True
