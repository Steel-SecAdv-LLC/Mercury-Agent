# Copyright (C) 2025 Steel Security Advisors LLC
"""Tests for Advanced Cognitive Modules (arxiv 2508.11957v1 - AI Agents Survey)."""

from __future__ import annotations

from typing import Any

import numpy as np

# =============================================================================
# Chain-of-Thought Reasoning Tests
# =============================================================================


class TestChainOfThoughtEngine:
    """Tests for ChainOfThoughtEngine."""

    def test_initialization(self) -> None:
        from omni_mercury_engine.cognitive.chain_of_thought import ChainOfThoughtEngine

        engine = ChainOfThoughtEngine()
        assert engine is not None
        stats = engine.get_statistics()
        assert stats["total_reasoning_sessions"] == 0

    def test_standard_reasoning(self) -> None:
        from omni_mercury_engine.cognitive.chain_of_thought import (
            ChainOfThoughtEngine,
            ReasoningStrategy,
        )

        engine = ChainOfThoughtEngine()

        query = "Is this network traffic anomalous?"
        context = {
            "packet_rate": 10000,
            "avg_packet_rate": 1000,
            "source_ip_entropy": 0.95,
            "protocol_distribution": {"TCP": 0.8, "UDP": 0.2},
        }

        result = engine.reason(query, context, strategy=ReasoningStrategy.STANDARD_COT)

        assert result is not None
        assert result.conclusion is not None
        assert result.confidence >= 0 and result.confidence <= 1
        assert len(result.thoughts) > 0

    def test_self_consistency_reasoning(self) -> None:
        from omni_mercury_engine.cognitive.chain_of_thought import (
            ChainOfThoughtEngine,
            ReasoningStrategy,
        )

        engine = ChainOfThoughtEngine()

        query = "What is the root cause of this anomaly?"
        context = {
            "anomaly_type": "spike",
            "affected_metrics": ["cpu", "memory", "network"],
            "temporal_pattern": "sudden_onset",
        }

        result = engine.reason(
            query, context, strategy=ReasoningStrategy.SELF_CONSISTENCY, num_samples=3
        )

        assert result is not None
        # Self-consistency should aggregate multiple reasoning paths
        assert "consistency_score" in result.metadata

    def test_tree_of_thoughts_reasoning(self) -> None:
        from omni_mercury_engine.cognitive.chain_of_thought import (
            ChainOfThoughtEngine,
            ReasoningStrategy,
        )

        engine = ChainOfThoughtEngine()

        query = "What actions should be taken for this threat?"
        context = {
            "threat_level": "high",
            "affected_systems": ["firewall", "dns"],
            "available_actions": ["isolate", "block", "alert", "investigate"],
        }

        result = engine.reason(
            query,
            context,
            strategy=ReasoningStrategy.TREE_OF_THOUGHTS,
            beam_width=2,
            max_depth=3,
        )

        assert result is not None
        # Tree of thoughts should explore multiple branches
        assert "branches_explored" in result.metadata

    def test_anomaly_chain_of_thought(self) -> None:
        from omni_mercury_engine.cognitive.chain_of_thought import AnomalyChainOfThought

        cot = AnomalyChainOfThought()

        detection_result = {
            "is_anomaly": True,
            "score": 0.85,
            "detectors": ["statistical", "temporal"],
        }
        raw_features = np.random.randn(100)

        analysis = cot.analyze_anomaly(detection_result, raw_features)

        assert analysis is not None
        assert "reasoning_chain" in analysis
        assert "conclusion" in analysis

    def test_statistics_tracking(self) -> None:
        from omni_mercury_engine.cognitive.chain_of_thought import ChainOfThoughtEngine

        engine = ChainOfThoughtEngine()

        # Perform multiple reasoning sessions
        for i in range(3):
            engine.reason(f"Query {i}", {"data": i})

        stats = engine.get_statistics()
        assert stats["total_reasoning_sessions"] == 3


# =============================================================================
# Reflexion Framework Tests
# =============================================================================


class TestReflexionEngine:
    """Tests for ReflexionEngine."""

    def test_initialization(self) -> None:
        from omni_mercury_engine.cognitive.reflexion import ReflexionEngine

        engine = ReflexionEngine()
        assert engine is not None
        stats = engine.get_statistics()
        assert stats["total_reflections"] == 0

    def test_execute_with_reflection(self) -> None:
        from omni_mercury_engine.cognitive.reflexion import ReflexionEngine

        engine = ReflexionEngine()

        task = {
            "type": "anomaly_classification",
            "data": np.random.randn(50),
            "possible_classes": ["normal", "network_attack", "system_failure"],
        }

        result = engine.execute_with_reflection(task, max_iterations=3)

        assert result is not None
        assert "decision" in result
        assert "iterations" in result
        assert result["iterations"] <= 3

    def test_experience_memory(self) -> None:
        from omni_mercury_engine.cognitive.reflexion import ExperienceMemory

        memory = ExperienceMemory(max_size=100)

        # Store experiences
        for i in range(5):
            memory.store(
                decision={"action": f"action_{i}"},
                outcome={"success": i % 2 == 0},
                context={"iteration": i},
            )

        # Retrieve similar experiences
        similar = memory.retrieve_similar({"action": "action_0"}, k=3)
        assert len(similar) <= 3

    def test_heuristic_evaluator(self) -> None:
        from omni_mercury_engine.cognitive.reflexion import HeuristicEvaluator

        evaluator = HeuristicEvaluator()

        decision = {"confidence": 0.8, "action": "block_ip"}
        outcome = {"success": True, "false_positive": False}

        evaluation = evaluator.evaluate(decision, outcome)

        # evaluator.evaluate returns dict for simplified dict-based API; see
        # cognitive/reflexion.py:585-625 (early-return branch).
        assert isinstance(evaluation, dict)
        assert "score" in evaluation
        assert "feedback" in evaluation

    def test_anomaly_reflexion_integration(self) -> None:
        from omni_mercury_engine.cognitive.reflexion import AnomalyReflexion

        reflexion = AnomalyReflexion()

        detection = {
            "is_anomaly": True,
            "score": 0.75,
            "features": np.random.randn(20).tolist(),
        }

        result = reflexion.reflect_on_detection(detection)

        assert result is not None
        assert "refined_score" in result or "recommendations" in result

    def test_improvement_planning(self) -> None:
        from omni_mercury_engine.cognitive.reflexion import ReflexionEngine

        engine = ReflexionEngine()

        # Execute some tasks to build experience
        for _ in range(3):
            engine.execute_with_reflection({"type": "test", "data": np.random.randn(10)})

        # Get improvement plan
        plan = engine.generate_improvement_plan()

        assert plan is not None
        assert isinstance(plan, dict)


# =============================================================================
# Chain of Hindsight Tests
# =============================================================================


class TestChainOfHindsightEngine:
    """Tests for ChainOfHindsightEngine."""

    def test_initialization(self) -> None:
        from omni_mercury_engine.cognitive.chain_of_hindsight import ChainOfHindsightEngine

        engine = ChainOfHindsightEngine()
        assert engine is not None
        stats = engine.get_statistics()
        assert stats["total_sequences"] == 0

    def test_record_sequence(self) -> None:
        from omni_mercury_engine.cognitive.chain_of_hindsight import ChainOfHindsightEngine

        engine = ChainOfHindsightEngine()

        # Record a sequence of decisions and outcomes
        sequence_id = engine.start_sequence("test_task")

        for i in range(5):
            engine.record_step(
                sequence_id,
                decision={"threshold": 0.5 + i * 0.1},
                outcome={"correct": i > 2},
                features=np.random.randn(10).tolist(),
            )

        engine.end_sequence(sequence_id, final_outcome={"success": True})

        stats = engine.get_statistics()
        assert stats["total_sequences"] == 1

    def test_credit_assignment(self) -> None:
        from omni_mercury_engine.cognitive.chain_of_hindsight import CreditAssignment

        assigner = CreditAssignment()

        sequence = [
            {"decision": {"a": 1}, "outcome": {"reward": 0.1}},
            {"decision": {"a": 2}, "outcome": {"reward": 0.2}},
            {"decision": {"a": 3}, "outcome": {"reward": 0.8}},
        ]

        credits = assigner.assign_credit(sequence, gamma=0.9)

        assert len(credits) == 3
        # Later decisions should have higher credit with positive outcomes
        assert credits[2] >= credits[0]

    def test_hindsight_relabeling(self) -> None:
        from omni_mercury_engine.cognitive.chain_of_hindsight import HindsightRelabeler

        relabeler = HindsightRelabeler()

        trajectory = [
            {"state": np.random.randn(5).tolist(), "action": "detect", "goal": "find_anomaly"},
            {"state": np.random.randn(5).tolist(), "action": "classify", "goal": "find_anomaly"},
        ]
        achieved_goal = "identified_pattern"

        relabeled = relabeler.relabel(trajectory, achieved_goal)

        # Should create alternative trajectory with achieved goal
        assert len(relabeled) == len(trajectory)
        for step in relabeled:
            assert isinstance(step, dict)
            assert step["goal"] == achieved_goal

    def test_feedback_processor(self) -> None:
        from omni_mercury_engine.cognitive.chain_of_hindsight import FeedbackProcessor

        processor = FeedbackProcessor()

        predictions = [0.3, 0.5, 0.7, 0.9]
        ground_truth = [0, 0, 1, 1]

        feedback = processor.process(predictions, ground_truth)

        assert "linguistic_feedback" in feedback
        assert "improvement_signals" in feedback

    def test_anomaly_chain_of_hindsight(self) -> None:
        from omni_mercury_engine.cognitive.chain_of_hindsight import AnomalyChainOfHindsight

        coh = AnomalyChainOfHindsight()

        # Record detection history
        history = [
            {"timestamp": 1, "detection": {"score": 0.6}, "label": False},
            {"timestamp": 2, "detection": {"score": 0.7}, "label": False},
            {"timestamp": 3, "detection": {"score": 0.8}, "label": True},
        ]

        insights = coh.learn_from_history(history)

        assert insights is not None
        assert "threshold_recommendations" in insights or "pattern_insights" in insights


# =============================================================================
# Hierarchical Planning Tests
# =============================================================================


class TestHierarchicalPlanner:
    """Tests for HierarchicalPlanner."""

    def test_initialization(self) -> None:
        from omni_mercury_engine.cognitive.hierarchical_planning import HierarchicalPlanner

        planner = HierarchicalPlanner()
        assert planner is not None
        stats = planner.get_statistics()
        assert stats["total_plans"] == 0

    def test_goal_decomposition(self) -> None:
        from omni_mercury_engine.cognitive.hierarchical_planning import GoalDecomposer

        decomposer = GoalDecomposer()

        high_level_goal = {
            "type": "detect_and_respond",
            "target": "network_intrusion",
            "constraints": {"max_false_positives": 0.01},
        }

        subgoals = decomposer.decompose(high_level_goal)

        assert len(subgoals) > 0
        # Should have hierarchical structure
        for subgoal in subgoals:
            assert isinstance(subgoal, dict)
            assert "level" in subgoal or "parent" in subgoal or "type" in subgoal

    def test_option_library(self) -> None:
        from omni_mercury_engine.cognitive.hierarchical_planning import OptionLibrary

        library = OptionLibrary()

        # Add an option (temporally extended action)
        library.add_option(
            name="isolate_host",
            initiation_set={"threat_detected": True, "host_identified": True},
            policy={"action": "firewall_block"},
            termination_condition={"host_isolated": True},
        )

        options = library.get_applicable_options({"threat_detected": True, "host_identified": True})

        assert len(options) >= 1
        assert options[0]["name"] == "isolate_host"

    def test_hierarchical_value_function(self) -> None:
        from omni_mercury_engine.cognitive.hierarchical_planning import HierarchicalValueFunction

        hvf = HierarchicalValueFunction(num_levels=3)

        state = {"threat_level": 0.8, "system_health": 0.6}
        option = "investigate"

        value = hvf.compute_value(state, option)

        assert isinstance(value, (int, float))

    def test_create_plan(self) -> None:
        from omni_mercury_engine.cognitive.hierarchical_planning import HierarchicalPlanner

        planner = HierarchicalPlanner()

        goal = {
            "objective": "mitigate_threat",
            "priority": "high",
            "deadline": 3600,
        }
        state = {
            "threat_detected": True,
            "threat_severity": 0.9,
            "available_resources": ["firewall", "ids", "analyst"],
        }

        plan = planner.create_plan(goal, state)

        assert plan is not None
        assert "actions" in plan or "steps" in plan
        assert "estimated_success" in plan or "confidence" in plan

    def test_anomaly_hierarchical_planner(self) -> None:
        from omni_mercury_engine.cognitive.hierarchical_planning import AnomalyHierarchicalPlanner

        planner = AnomalyHierarchicalPlanner()

        anomaly = {
            "type": "network_spike",
            "severity": 0.85,
            "affected_systems": ["web_server", "database"],
        }

        response_plan = planner.plan_response(anomaly)

        assert response_plan is not None
        assert isinstance(response_plan, dict)
        assert "strategic_goals" in response_plan or "tactical_actions" in response_plan


# =============================================================================
# Multi-Agent Coordination Tests
# =============================================================================


class TestMultiAgentCoordination:
    """Tests for MultiAgentDetectionSystem."""

    def test_agent_coordinator_initialization(self) -> None:
        from omni_mercury_engine.cognitive.multi_agent_coordination import AgentCoordinator

        coordinator = AgentCoordinator()
        assert coordinator is not None
        stats = coordinator.get_statistics()
        assert stats["registered_agents"] == 0

    def test_register_agent(self) -> None:
        from omni_mercury_engine.cognitive.multi_agent_coordination import (
            AgentCoordinator,
            DetectionAgent,
        )

        coordinator = AgentCoordinator()

        class TestAgent(DetectionAgent):
            def detect(self, data: Any, context: Any = None) -> Any:
                return {"score": 0.5}

        agent = TestAgent(agent_id="agent_1", capabilities=["statistical"])
        coordinator.register_agent(agent)

        stats = coordinator.get_statistics()
        assert stats["registered_agents"] == 1

    def test_consensus_protocol_majority_vote(self) -> None:
        from omni_mercury_engine.cognitive.multi_agent_coordination import ConsensusProtocol

        protocol = ConsensusProtocol(method="majority_vote")

        votes = [
            {"agent": "a1", "decision": True, "confidence": 0.8},
            {"agent": "a2", "decision": True, "confidence": 0.7},
            {"agent": "a3", "decision": False, "confidence": 0.6},
        ]

        consensus = protocol.reach_consensus(votes)

        assert isinstance(consensus, dict)
        assert consensus["decision"]
        assert "agreement_ratio" in consensus

    def test_consensus_protocol_weighted_vote(self) -> None:
        from omni_mercury_engine.cognitive.multi_agent_coordination import ConsensusProtocol

        protocol = ConsensusProtocol(method="weighted_vote")

        votes = [
            {"agent": "a1", "decision": True, "confidence": 0.9, "weight": 2.0},
            {"agent": "a2", "decision": False, "confidence": 0.8, "weight": 1.0},
            {"agent": "a3", "decision": False, "confidence": 0.7, "weight": 1.0},
        ]

        consensus = protocol.reach_consensus(votes)

        # Weighted: True has 1.8, False has 1.5
        assert isinstance(consensus, dict)
        assert consensus["decision"]

    def test_byzantine_tolerant_consensus(self) -> None:
        from omni_mercury_engine.cognitive.multi_agent_coordination import ConsensusProtocol

        protocol = ConsensusProtocol(method="byzantine_tolerant")

        # 4 agents, 1 faulty (can tolerate f < n/3)
        votes = [
            {"agent": "a1", "decision": True, "confidence": 0.9},
            {"agent": "a2", "decision": True, "confidence": 0.8},
            {"agent": "a3", "decision": True, "confidence": 0.85},
            {"agent": "a4", "decision": False, "confidence": 0.99},  # Potentially faulty
        ]

        consensus = protocol.reach_consensus(votes)

        assert isinstance(consensus, dict)
        assert consensus["decision"]
        assert consensus["is_byzantine_safe"]

    def test_coalition_formation(self) -> None:
        from omni_mercury_engine.cognitive.multi_agent_coordination import (
            Coalition,
            DetectionAgent,
        )

        class SpecializedAgent(DetectionAgent):
            def detect(self, data: Any, context: Any = None) -> Any:
                return {"score": np.random.random()}

        agents = [
            SpecializedAgent(agent_id=f"agent_{i}", capabilities=[f"cap_{i % 3}"]) for i in range(5)
        ]

        coalition = Coalition(
            coalition_id="threat_response",
            members=agents[:3],
            objective="investigate_threat",
        )

        assert len(coalition.members) == 3
        assert coalition.objective == "investigate_threat"

    def test_multi_agent_detection_system(self) -> None:
        from omni_mercury_engine.cognitive.multi_agent_coordination import (
            DetectionAgent,
            MultiAgentDetectionSystem,
        )

        class SimpleAgent(DetectionAgent):
            def __init__(self, agent_id: Any, threshold: Any) -> None:
                super().__init__(agent_id, capabilities=["threshold"])
                self.threshold = threshold

            def detect(self, data: Any, context: Any = None) -> Any:
                score = np.mean(np.abs(data))
                return {"is_anomaly": score > self.threshold, "score": score}

        system = MultiAgentDetectionSystem()

        # Register agents with different thresholds
        for i, thresh in enumerate([0.5, 0.6, 0.7]):
            system.register_agent(SimpleAgent(f"agent_{i}", thresh))

        # Test detection
        test_data = np.random.randn(100) * 0.8

        result = system.detect(test_data)

        assert result is not None
        assert "consensus_decision" in result
        assert "individual_results" in result


# =============================================================================
# Formal Verification Tests
# =============================================================================


class TestFormalVerification:
    """Tests for FormalVerificationEngine."""

    def test_initialization(self) -> None:
        from omni_mercury_engine.cognitive.formal_verification import FormalVerificationEngine

        engine = FormalVerificationEngine()
        assert engine is not None
        stats = engine.get_statistics()
        assert stats["total_verifications"] == 0

    def test_safety_verifier(self) -> None:
        from omni_mercury_engine.cognitive.formal_verification import SafetyVerifier

        verifier = SafetyVerifier()

        # Define a safety property
        safety_property = {
            "name": "no_false_negatives_critical",
            "condition": "critical_threat => alert_triggered",
            "priority": "critical",
        }

        system_state = {
            "critical_threat": True,
            "alert_triggered": True,
        }

        result = verifier.verify(safety_property, system_state)

        assert result["satisfied"]

    def test_constraint_solver(self) -> None:
        from omni_mercury_engine.cognitive.formal_verification import ConstraintSolver

        solver = ConstraintSolver()

        constraints = [
            {"type": "range", "variable": "threshold", "min": 0.0, "max": 1.0},
            {"type": "greater_than", "variable": "threshold", "value": 0.5},
            {"type": "less_than", "variable": "threshold", "value": 0.9},
        ]

        result = solver.solve(constraints)

        assert result["satisfiable"]
        assert 0.5 < result["solution"]["threshold"] < 0.9

    def test_reachability_analyzer(self) -> None:
        from omni_mercury_engine.cognitive.formal_verification import ReachabilityAnalyzer

        analyzer = ReachabilityAnalyzer()

        # Define a simple state machine
        states = {
            "normal": {"transitions": {"anomaly_detected": "alert"}},
            "alert": {"transitions": {"investigate": "investigating", "dismiss": "normal"}},
            "investigating": {"transitions": {"confirm": "mitigate", "false_alarm": "normal"}},
            "mitigate": {"transitions": {"resolved": "normal"}},
        }

        # Check if "mitigate" is reachable from "normal"
        result = analyzer.is_reachable("normal", "mitigate", states)

        assert result["reachable"]
        assert len(result["path"]) > 0

    def test_interval_bound_propagator(self) -> None:
        from omni_mercury_engine.cognitive.formal_verification import IntervalBoundPropagator

        propagator = IntervalBoundPropagator()

        # Input bounds
        input_bounds = {
            "x1": (0.0, 1.0),
            "x2": (-1.0, 1.0),
        }

        # Simple linear transformation
        weights = np.array([[0.5, 0.5], [1.0, -0.5]])
        bias = np.array([0.1, 0.0])

        output_bounds = propagator.propagate_linear(input_bounds, weights, bias)

        assert "y0" in output_bounds or len(output_bounds) == 2

    def test_anomaly_verifier(self) -> None:
        from omni_mercury_engine.cognitive.formal_verification import AnomalyVerifier

        verifier = AnomalyVerifier()

        detection_decision = {
            "is_anomaly": True,
            "score": 0.85,
            "severity": "high",
            "recommended_action": "isolate",
        }

        safety_constraints = [
            {"name": "require_high_confidence", "condition": "score > 0.7"},
            {
                "name": "no_isolate_without_confirmation",
                "condition": "severity == 'high' => score > 0.8",
            },
        ]

        result = verifier.verify_decision(detection_decision, safety_constraints)

        assert result is not None
        assert "all_satisfied" in result
        assert result["all_satisfied"]

    def test_statistics_tracking(self) -> None:
        from omni_mercury_engine.cognitive.formal_verification import FormalVerificationEngine

        engine = FormalVerificationEngine()

        # Perform verifications
        for i in range(3):
            engine.verify_property({"name": f"prop_{i}", "condition": "x > 0"}, {"x": i})

        stats = engine.get_statistics()
        assert stats["total_verifications"] == 3


# =============================================================================
# Predictive Coding Tests
# =============================================================================


class TestPredictiveCoding:
    """Tests for PredictiveCodingDetector."""

    def test_hierarchical_predictive_coder_initialization(self) -> None:
        from omni_mercury_engine.cognitive.predictive_coding import HierarchicalPredictiveCoder

        coder = HierarchicalPredictiveCoder(num_levels=3, input_dim=10)
        assert coder is not None
        assert coder.num_levels == 3

    def test_prediction_and_error(self) -> None:
        from omni_mercury_engine.cognitive.predictive_coding import HierarchicalPredictiveCoder

        coder = HierarchicalPredictiveCoder(num_levels=3, input_dim=10)

        input_data = np.random.randn(10)

        prediction, error = coder.predict_and_compute_error(input_data)

        assert prediction.shape == input_data.shape
        assert error.shape == input_data.shape

    def test_precision_estimator(self) -> None:
        from omni_mercury_engine.cognitive.predictive_coding import PrecisionEstimator

        estimator = PrecisionEstimator()

        errors = np.random.randn(100)
        precisions = estimator.estimate(errors)

        assert len(precisions) == len(errors)
        assert all(p > 0 for p in precisions)  # Precisions should be positive

    def test_active_inference_agent(self) -> None:
        from omni_mercury_engine.cognitive.predictive_coding import ActiveInferenceAgent

        agent = ActiveInferenceAgent(
            state_dim=5,
            action_dim=3,
        )

        state = np.random.randn(5)
        available_actions = [
            {"id": 0, "params": np.zeros(3)},
            {"id": 1, "params": np.ones(3)},
        ]

        action = agent.select_action(state, available_actions)

        assert action is not None
        assert "id" in action

    def test_free_energy_minimization(self) -> None:
        from omni_mercury_engine.cognitive.predictive_coding import HierarchicalPredictiveCoder

        coder = HierarchicalPredictiveCoder(num_levels=2, input_dim=5)

        # Generate sequence
        sequence = [np.random.randn(5) for _ in range(10)]

        # Process sequence and compute free energy
        free_energies = []
        for obs in sequence:
            _, error = coder.predict_and_compute_error(obs)
            fe = coder.compute_free_energy(error)
            free_energies.append(fe)

        assert len(free_energies) == 10
        assert all(isinstance(fe, (int, float)) for fe in free_energies)

    def test_predictive_coding_detector(self) -> None:
        from omni_mercury_engine.cognitive.predictive_coding import PredictiveCodingDetector

        detector = PredictiveCodingDetector(
            input_dim=20,
            num_levels=3,
            anomaly_threshold=2.0,
        )

        # Normal data - low prediction error expected
        normal_data = np.sin(np.linspace(0, 4 * np.pi, 20))

        # Train on normal patterns
        for _ in range(10):
            detector.update(normal_data + np.random.randn(20) * 0.1)

        # Anomalous data - high prediction error expected
        anomaly_data = np.random.randn(20) * 10

        result = detector.detect(anomaly_data)

        assert result is not None
        assert "is_anomaly" in result
        assert "prediction_error" in result

    def test_mercury_predictive_coding(self) -> None:
        from omni_mercury_engine.cognitive.predictive_coding import MercuryPredictiveCoding

        pc = MercuryPredictiveCoding()

        # Simulate detection integration
        detection_result = {
            "scores": [0.7, 0.8, 0.75],
            "features": np.random.randn(50).tolist(),
        }

        enhanced_result = pc.enhance_detection(detection_result)

        assert enhanced_result is not None
        assert "prediction_based_score" in enhanced_result or "enhanced_scores" in enhanced_result

    def test_belief_updating(self) -> None:
        from omni_mercury_engine.cognitive.predictive_coding import HierarchicalPredictiveCoder

        coder = HierarchicalPredictiveCoder(num_levels=2, input_dim=5)

        # Initial belief
        prior = np.zeros(5)

        # New observation
        observation = np.ones(5)

        # Update belief
        posterior = coder.update_beliefs(prior, observation, learning_rate=0.1)

        assert posterior.shape == prior.shape
        # Posterior should move towards observation
        assert np.mean(posterior) > np.mean(prior)

    def test_statistics(self) -> None:
        from omni_mercury_engine.cognitive.predictive_coding import PredictiveCodingDetector

        detector = PredictiveCodingDetector(input_dim=10, num_levels=2)

        # Process some data
        for _ in range(5):
            detector.detect(np.random.randn(10))

        stats = detector.get_statistics()

        assert "total_predictions" in stats
        assert stats["total_predictions"] == 5


# =============================================================================
# Integration Tests
# =============================================================================


class TestCognitiveModulesIntegration:
    """Integration tests for all cognitive modules working together."""

    def test_chain_of_thought_with_reflexion(self) -> None:
        from omni_mercury_engine.cognitive.chain_of_thought import ChainOfThoughtEngine
        from omni_mercury_engine.cognitive.reflexion import ReflexionEngine

        cot = ChainOfThoughtEngine()
        reflexion = ReflexionEngine()

        # Use CoT to reason about a problem
        reasoning_result = cot.reason(
            "Should this be classified as an attack?",
            {"packet_rate": 10000, "source_diversity": 0.95},
        )

        # Use reflexion to evaluate the reasoning
        task = {
            "type": "evaluate_reasoning",
            "reasoning": reasoning_result.thoughts,
            "conclusion": reasoning_result.conclusion,
        }

        reflection_result = reflexion.execute_with_reflection(task)

        assert reflection_result is not None

    def test_hierarchical_planning_with_multi_agent(self) -> None:
        from omni_mercury_engine.cognitive.hierarchical_planning import HierarchicalPlanner
        from omni_mercury_engine.cognitive.multi_agent_coordination import (
            DetectionAgent,
            MultiAgentDetectionSystem,
        )

        class PlanningAgent(DetectionAgent):
            def __init__(self, agent_id: Any, planner: Any) -> None:
                super().__init__(agent_id, capabilities=["planning"])
                self.planner = planner

            def detect(self, data: Any, context: Any = None) -> Any:
                plan = self.planner.create_plan(
                    {"objective": "analyze_data"},
                    {"data": data.tolist() if hasattr(data, "tolist") else data},
                )
                return {"score": plan.get("confidence", 0.5), "plan": plan}

        system = MultiAgentDetectionSystem()

        for i in range(3):
            planner = HierarchicalPlanner()
            system.register_agent(PlanningAgent(f"planning_agent_{i}", planner))

        result = system.detect(np.random.randn(50))

        assert "consensus_decision" in result

    def test_formal_verification_of_predictive_coding(self) -> None:
        from omni_mercury_engine.cognitive.formal_verification import AnomalyVerifier
        from omni_mercury_engine.cognitive.predictive_coding import PredictiveCodingDetector

        detector = PredictiveCodingDetector(input_dim=10, num_levels=2)
        verifier = AnomalyVerifier()

        # Detect anomaly
        result = detector.detect(np.random.randn(10) * 5)

        # Verify the detection meets safety constraints
        safety_constraints = [
            {"name": "bounded_score", "condition": "0 <= prediction_error <= 100"},
        ]

        verification = verifier.verify_decision(result, safety_constraints)

        assert verification is not None

    def test_chain_of_hindsight_learning_loop(self) -> None:
        from omni_mercury_engine.cognitive.chain_of_hindsight import ChainOfHindsightEngine
        from omni_mercury_engine.cognitive.predictive_coding import PredictiveCodingDetector

        coh = ChainOfHindsightEngine()
        detector = PredictiveCodingDetector(input_dim=10, num_levels=2)

        # Run detection sequence
        seq_id = coh.start_sequence("detection_run")

        for i in range(5):
            data = np.random.randn(10) * (1 + i * 0.5)
            result = detector.detect(data)

            coh.record_step(
                seq_id,
                decision=result,
                outcome={"ground_truth": i > 2},
                features=data.tolist(),
            )

        coh.end_sequence(seq_id, {"overall_accuracy": 0.8})

        # Learn from sequence
        stats = coh.get_statistics()
        assert stats["total_sequences"] == 1


# =============================================================================
# Import Tests
# =============================================================================


class TestModuleImports:
    """Test that all new modules can be imported correctly."""

    def test_import_chain_of_thought(self) -> None:
        from omni_mercury_engine.cognitive import (
            AnomalyChainOfThought,
            ChainOfThoughtEngine,
            ReasoningStrategy,
            ThoughtGenerator,
        )

        assert ChainOfThoughtEngine is not None
        assert ThoughtGenerator is not None
        assert AnomalyChainOfThought is not None
        assert ReasoningStrategy is not None

    def test_import_reflexion(self) -> None:
        from omni_mercury_engine.cognitive import (
            AnomalyReflexion,
            ExperienceMemory,
            HeuristicEvaluator,
            ReflexionEngine,
        )

        assert ReflexionEngine is not None
        assert ExperienceMemory is not None
        assert HeuristicEvaluator is not None
        assert AnomalyReflexion is not None

    def test_import_chain_of_hindsight(self) -> None:
        from omni_mercury_engine.cognitive import (
            AnomalyChainOfHindsight,
            ChainOfHindsightEngine,
            CreditAssignment,
            FeedbackProcessor,
            HindsightRelabeler,
        )

        assert ChainOfHindsightEngine is not None
        assert CreditAssignment is not None
        assert HindsightRelabeler is not None
        assert FeedbackProcessor is not None
        assert AnomalyChainOfHindsight is not None

    def test_import_hierarchical_planning(self) -> None:
        from omni_mercury_engine.cognitive import (
            AbstractionLevel,
            AnomalyHierarchicalPlanner,
            GoalDecomposer,
            HierarchicalPlanner,
            HierarchicalValueFunction,
            OptionLibrary,
        )

        assert HierarchicalPlanner is not None
        assert GoalDecomposer is not None
        assert OptionLibrary is not None
        assert HierarchicalValueFunction is not None
        assert AnomalyHierarchicalPlanner is not None
        assert AbstractionLevel is not None

    def test_import_multi_agent_coordination(self) -> None:
        from omni_mercury_engine.cognitive import (
            AgentCoordinator,
            Coalition,
            ConsensusProtocol,
            DetectionAgent,
            MultiAgentDetectionSystem,
        )

        assert AgentCoordinator is not None
        assert ConsensusProtocol is not None
        assert DetectionAgent is not None
        assert Coalition is not None
        assert MultiAgentDetectionSystem is not None

    def test_import_formal_verification(self) -> None:
        from omni_mercury_engine.cognitive import (
            AnomalyVerifier,
            ConstraintSolver,
            FormalVerificationEngine,
            IntervalBoundPropagator,
            ReachabilityAnalyzer,
            SafetyVerifier,
        )

        assert FormalVerificationEngine is not None
        assert SafetyVerifier is not None
        assert ConstraintSolver is not None
        assert ReachabilityAnalyzer is not None
        assert IntervalBoundPropagator is not None
        assert AnomalyVerifier is not None

    def test_import_predictive_coding(self) -> None:
        from omni_mercury_engine.cognitive import (
            ActiveInferenceAgent,
            HierarchicalPredictiveCoder,
            MercuryPredictiveCoding,
            PrecisionEstimator,
            PredictiveCodingDetector,
        )

        assert HierarchicalPredictiveCoder is not None
        assert PrecisionEstimator is not None
        assert ActiveInferenceAgent is not None
        assert PredictiveCodingDetector is not None
        assert MercuryPredictiveCoding is not None
