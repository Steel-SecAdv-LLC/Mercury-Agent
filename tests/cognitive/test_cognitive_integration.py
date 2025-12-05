"""
Tests for Cognitive Architecture Integration

Verifies that all cognitive components integrate properly:
- PlasticityEngine
- KnowledgeGraph
- MultiHopReasoner
- IPBEngine
- CausalDiscoveryEngine
- UncertaintyQuantifier
- CaseBasedReasoner
- IndicatorDevelopmentSystem
- CognitiveOrchestrator
"""

import numpy as np


class TestPlasticityEngine:
    """Tests for PlasticityEngine."""

    def test_initialization(self):
        from omni_anomaly_engine.cognitive.plasticity_engine import PlasticityEngine

        engine = PlasticityEngine()
        assert engine is not None
        stats = engine.get_statistics()
        assert stats["active_connections"] == 0

    def test_adaptation(self):
        from omni_anomaly_engine.cognitive.plasticity_engine import PlasticityEngine

        engine = PlasticityEngine()
        event = engine.adapt("pattern_a", "pattern_b", strength=0.8)

        assert event is not None
        assert event.source_pattern == "pattern_a"
        assert event.target_pattern == "pattern_b"

        # Check connection was created
        stats = engine.get_statistics()
        assert stats["active_connections"] == 1

    def test_association_query(self):
        from omni_anomaly_engine.cognitive.plasticity_engine import PlasticityEngine

        engine = PlasticityEngine()
        engine.adapt("a", "b", strength=0.9)
        engine.adapt("b", "c", strength=0.8)

        associations = engine.query_association("a", depth=2)
        assert "b" in associations


class TestKnowledgeGraph:
    """Tests for KnowledgeGraph."""

    def test_initialization(self):
        from omni_anomaly_engine.cognitive.knowledge_graph import KnowledgeGraph

        graph = KnowledgeGraph()
        assert graph is not None
        stats = graph.get_statistics()
        assert stats["total_nodes"] == 0

    def test_add_node_and_edge(self):
        from omni_anomaly_engine.cognitive.knowledge_graph import (
            EdgeType,
            KnowledgeGraph,
            NodeType,
        )

        graph = KnowledgeGraph()

        graph.add_node("concept1", NodeType.CONCEPT, "First Concept")
        graph.add_node("concept2", NodeType.CONCEPT, "Second Concept")
        graph.add_edge("concept1", "concept2", EdgeType.CAUSES)

        assert graph.get_node("concept1") is not None
        neighbors = graph.get_neighbors("concept1")
        assert len(neighbors) == 1

    def test_spreading_activation(self):
        from omni_anomaly_engine.cognitive.knowledge_graph import (
            EdgeType,
            KnowledgeGraph,
            NodeType,
        )

        graph = KnowledgeGraph()

        for i in range(5):
            graph.add_node(f"node_{i}", NodeType.CONCEPT, f"Node {i}")

        graph.add_edge("node_0", "node_1", EdgeType.CAUSES)
        graph.add_edge("node_1", "node_2", EdgeType.CAUSES)
        graph.add_edge("node_2", "node_3", EdgeType.CAUSES)

        activations = graph.spreading_activation(["node_0"])
        assert "node_0" in activations
        assert activations["node_0"] > 0


class TestMultiHopReasoner:
    """Tests for MultiHopReasoner."""

    def test_initialization(self):
        from omni_anomaly_engine.cognitive.multi_hop_reasoner import MultiHopReasoner

        reasoner = MultiHopReasoner()
        assert reasoner is not None
        stats = reasoner.get_statistics()
        assert stats["total_rules"] > 0  # Core rules initialized

    def test_deduction(self):
        from omni_anomaly_engine.cognitive.multi_hop_reasoner import (
            MultiHopReasoner,
            Proposition,
        )

        reasoner = MultiHopReasoner()

        premises = [
            Proposition(prop_id="premise_1", content="It is raining", truth_value=1.0),
        ]

        # Even without matching rules, deduction should complete
        chain = reasoner.deduce(premises)
        # May or may not find derivations depending on knowledge base

    def test_abduction(self):
        from omni_anomaly_engine.cognitive.multi_hop_reasoner import (
            MultiHopReasoner,
            Proposition,
        )

        reasoner = MultiHopReasoner()

        observation = Proposition(prop_id="obs", content="Ground is wet", truth_value=1.0)
        hypotheses = [
            Proposition(prop_id="h1", content="It rained", truth_value=0.8),
            Proposition(prop_id="h2", content="Sprinkler was on", truth_value=0.7),
        ]

        chain = reasoner.abduce(observation, hypotheses)
        assert chain is not None
        assert chain.final_conclusion is not None


class TestIPBEngine:
    """Tests for IPBEngine."""

    def test_initialization(self):
        from omni_anomaly_engine.cognitive.ipb_engine import IPBEngine

        ipb = IPBEngine()
        assert ipb is not None

    def test_environment_definition(self):
        from omni_anomaly_engine.cognitive.ipb_engine import (
            EnvironmentDomain,
            IPBEngine,
        )

        ipb = IPBEngine()
        env = ipb.define_environment(
            EnvironmentDomain.CYBER,
            area_of_interest={"network": "corporate"},
            critical_assets=["database", "auth_server"],
        )

        assert env is not None
        assert env.domain == EnvironmentDomain.CYBER


class TestCausalDiscovery:
    """Tests for CausalDiscoveryEngine."""

    def test_initialization(self):
        from omni_anomaly_engine.cognitive.causal_discovery import CausalDiscoveryEngine

        causal = CausalDiscoveryEngine()
        assert causal is not None

    def test_structure_discovery(self):
        from omni_anomaly_engine.cognitive.causal_discovery import CausalDiscoveryEngine

        causal = CausalDiscoveryEngine()

        # Generate correlated data
        n_samples = 100
        x1 = np.random.randn(n_samples)
        x2 = 0.8 * x1 + 0.2 * np.random.randn(n_samples)
        x3 = 0.5 * x2 + 0.5 * np.random.randn(n_samples)
        data = np.column_stack([x1, x2, x3])

        graph = causal.discover_structure(data, variable_names=["X1", "X2", "X3"])
        assert graph is not None
        assert len(graph.nodes) == 3


class TestUncertaintyQuantifier:
    """Tests for UncertaintyQuantifier."""

    def test_initialization(self):
        from omni_anomaly_engine.cognitive.uncertainty import UncertaintyQuantifier

        uq = UncertaintyQuantifier()
        assert uq is not None

    def test_uncertainty_estimation(self):
        from omni_anomaly_engine.cognitive.uncertainty import UncertaintyQuantifier

        uq = UncertaintyQuantifier()

        predictions = np.array([0.7, 0.75, 0.72, 0.68, 0.73])
        estimate = uq.estimate_uncertainty(predictions)

        assert estimate is not None
        assert 0 <= estimate.confidence <= 1
        assert estimate.epistemic >= 0
        assert estimate.aleatoric >= 0


class TestCaseBasedReasoner:
    """Tests for CaseBasedReasoner."""

    def test_initialization(self):
        from omni_anomaly_engine.cognitive.case_based_reasoning import CaseBasedReasoner

        cbr = CaseBasedReasoner()
        assert cbr is not None

    def test_add_and_retrieve_case(self):
        from omni_anomaly_engine.cognitive.case_based_reasoning import (
            Case,
            CaseBasedReasoner,
            CaseOutcome,
        )

        cbr = CaseBasedReasoner()

        case = Case(
            case_id="test_case_1",
            problem_description="Test problem",
            problem_features={"severity": 0.8, "type": "alert"},
            feature_vector=None,
            solution={"action": "investigate"},
            outcome=CaseOutcome.SUCCESS,
            outcome_score=0.9,
            domain="test",
        )

        cbr.add_case(case)

        result = cbr.retrieve({"severity": 0.75, "type": "alert"})
        assert result is not None


class TestIndicatorSystem:
    """Tests for IndicatorDevelopmentSystem."""

    def test_initialization(self):
        from omni_anomaly_engine.cognitive.indicator_system import (
            IndicatorDevelopmentSystem,
        )

        ids = IndicatorDevelopmentSystem()
        assert ids is not None

    def test_develop_indicator(self):
        from omni_anomaly_engine.cognitive.indicator_system import (
            IndicatorDevelopmentSystem,
            IndicatorType,
        )

        ids = IndicatorDevelopmentSystem()

        indicator = ids.develop_indicator(
            pattern={"type": "anomaly", "severity_range": (0.7, 1.0)},
            name="High Severity Anomaly",
            description="Detects high severity anomalies",
            indicator_type=IndicatorType.THRESHOLD,
            domain="test",
        )

        assert indicator is not None
        assert indicator.name == "High Severity Anomaly"


class TestCognitiveOrchestrator:
    """Tests for CognitiveOrchestrator integration."""

    def test_initialization(self):
        from omni_anomaly_engine.cognitive.orchestrator import CognitiveOrchestrator

        orchestrator = CognitiveOrchestrator()
        assert orchestrator is not None
        assert orchestrator.knowledge_graph is not None
        assert orchestrator.reasoner is not None
        assert orchestrator.uncertainty is not None

    def test_analyze_detection_result(self):
        from omni_anomaly_engine.cognitive.orchestrator import CognitiveOrchestrator

        orchestrator = CognitiveOrchestrator()

        detection_result = {
            "is_anomaly": True,
            "anomaly_prob": 0.85,
            "severity": 0.7,
        }
        raw_data = np.random.randn(50, 10)
        context = {"domain": "cyber"}

        result = orchestrator.analyze(detection_result, raw_data, context)

        assert result is not None
        assert result.anomaly_detected == True
        assert result.anomaly_score == 0.85
        assert 0 <= result.confidence <= 1

    def test_statistics(self):
        from omni_anomaly_engine.cognitive.orchestrator import CognitiveOrchestrator

        orchestrator = CognitiveOrchestrator()

        # Perform an analysis
        detection_result = {"is_anomaly": True, "anomaly_prob": 0.7, "severity": 0.5}
        orchestrator.analyze(detection_result, np.random.randn(10, 5), {})

        stats = orchestrator.get_statistics()

        assert stats["analyses_performed"] == 1
        assert "knowledge_graph" in stats
        assert "reasoner" in stats


class TestTruthDecipherIntegration:
    """Tests for Truth Decipher Framework with Cognitive Integration."""

    def test_cognitive_integration_disabled(self):
        """Test that framework works with cognitive disabled."""
        from omni_anomaly_engine.truth_decipher import TruthDecipherFramework

        framework = TruthDecipherFramework(enable_cognitive=False)
        assert framework.cognitive is None

    def test_cognitive_integration_enabled(self):
        """Test that cognitive orchestrator is integrated."""
        from omni_anomaly_engine.truth_decipher import TruthDecipherFramework

        framework = TruthDecipherFramework(enable_cognitive=True)
        assert framework.cognitive is not None

    def test_decipher_truth_with_cognitive(self):
        """Test full pipeline with cognitive layer."""
        from omni_anomaly_engine.truth_decipher import TruthDecipherFramework

        framework = TruthDecipherFramework(
            enable_cognitive=True,
            enable_novel_discovery=False,  # Speed up test
        )

        # Create test data that will trigger an anomaly
        data = np.random.randn(100, 20)
        # Add some outliers
        data[0:5, :] = data[0:5, :] * 10

        result = framework.decipher_truth(data, context={"domain": "cyber"})

        assert result is not None
        # Result should have cognitive fields populated if anomaly was detected
        if result.anomaly_detected:
            # These fields come from cognitive analysis
            assert hasattr(result, "confidence")
            assert hasattr(result, "is_reliable")
            assert hasattr(result, "epistemic_uncertainty")

    def test_statistics_include_cognitive(self):
        """Test that statistics include cognitive stats."""
        from omni_anomaly_engine.truth_decipher import TruthDecipherFramework

        framework = TruthDecipherFramework(enable_cognitive=True)
        stats = framework.get_statistics()

        assert "cognitive" in stats
