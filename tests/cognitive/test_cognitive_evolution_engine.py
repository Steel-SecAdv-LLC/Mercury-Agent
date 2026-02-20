"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Tests for Cognitive Evolution Engine module.
"""

from __future__ import annotations

from omni_mercury_engine.cognitive.cognitive_evolution_engine import (
    AgentRole,
    ChainOfThoughtReasoner,
    CognitiveEvolutionEngine,
    Counterfactual,
    CounterfactualSimulator,
    CuriosityEngine,
    ExplorationResult,
    IntentInference,
    MutationResult,
    MutationType,
    ReasoningStep,
    Rule,
    RuleMutator,
    SelfPlaySimulator,
    SimulationAgent,
    SimulationResult,
    TheoryOfMind,
    ThoughtChain,
)


class TestSelfPlaySimulator:
    """Tests for SelfPlaySimulator class."""

    def test_init(self):
        """Test simulator initialization."""
        simulator = SelfPlaySimulator(num_agents=5)
        assert len(simulator.agents) == 5
        assert simulator._simulation_counter == 0

    def test_init_custom_agents(self):
        """Test simulator with custom agent count."""
        simulator = SelfPlaySimulator(num_agents=10)
        assert len(simulator.agents) == 10

    def test_run_simulation(self):
        """Test running a simulation."""
        simulator = SelfPlaySimulator()
        result = simulator.run_simulation(
            scenario={"type": "test", "value": 42},
            rounds=5,
        )

        assert isinstance(result, SimulationResult)
        assert result.simulation_id.startswith("sim_")
        assert result.outcomes["rounds_completed"] == 5
        assert len(result.insights) > 0

    def test_run_simulation_outcomes(self):
        """Test simulation outcomes structure."""
        simulator = SelfPlaySimulator()
        result = simulator.run_simulation(
            scenario={"test": True},
            rounds=3,
        )

        assert "rounds_completed" in result.outcomes
        assert "agent_contributions" in result.outcomes
        assert "consensus_reached" in result.outcomes
        assert "final_score" in result.outcomes

    def test_agent_experience_increases(self):
        """Test that agent experience increases after simulation."""
        simulator = SelfPlaySimulator()
        initial_experience = [a.experience for a in simulator.agents]

        simulator.run_simulation({"test": True}, rounds=2)

        for i, agent in enumerate(simulator.agents):
            assert agent.experience > initial_experience[i]

    def test_get_statistics(self):
        """Test statistics retrieval."""
        simulator = SelfPlaySimulator()
        simulator.run_simulation({"test": True})

        stats = simulator.get_statistics()

        assert stats["simulations_run"] == 1
        assert stats["num_agents"] == 5
        assert "agent_roles" in stats


class TestRuleMutator:
    """Tests for RuleMutator class."""

    def test_init(self):
        """Test mutator initialization."""
        mutator = RuleMutator()
        assert mutator.mutation_rate == 0.1
        assert mutator.crossover_rate == 0.7

    def test_init_custom_rates(self):
        """Test mutator with custom rates."""
        mutator = RuleMutator(mutation_rate=0.2, crossover_rate=0.8)
        assert mutator.mutation_rate == 0.2
        assert mutator.crossover_rate == 0.8

    def test_mutate(self):
        """Test rule mutation."""
        mutator = RuleMutator()
        rule = Rule(
            rule_id="rule_001",
            condition="if anomaly > threshold",
            action="alert",
            confidence=0.8,
            fitness=0.7,
        )

        result = mutator.mutate(rule)

        assert isinstance(result, MutationResult)
        assert result.mutation_id.startswith("mut_")
        assert result.child_rule.rule_id != rule.rule_id

    def test_mutate_specific_type(self):
        """Test mutation with specific type."""
        mutator = RuleMutator()
        rule = Rule(
            rule_id="rule_001",
            condition="test_condition",
            action="test_action",
            confidence=0.8,
            fitness=0.7,
        )

        result = mutator.mutate(rule, MutationType.POINT_MUTATION)

        assert result.mutation_type == MutationType.POINT_MUTATION

    def test_crossover(self):
        """Test rule crossover."""
        mutator = RuleMutator()
        parent1 = Rule(
            rule_id="rule_001",
            condition="condition_a",
            action="action_a",
            confidence=0.8,
            fitness=0.7,
        )
        parent2 = Rule(
            rule_id="rule_002",
            condition="condition_b",
            action="action_b",
            confidence=0.9,
            fitness=0.8,
        )

        result = mutator.crossover(parent1, parent2)

        assert result.mutation_type == MutationType.CROSSOVER
        assert len(result.parent_rules) == 2
        assert result.child_rule.generation > 0

    def test_evolve_population(self):
        """Test population evolution."""
        mutator = RuleMutator()
        initial_rules = [
            Rule(
                rule_id=f"rule_{i}",
                condition=f"condition_{i}",
                action=f"action_{i}",
                confidence=0.7,
                fitness=0.5 + i * 0.1,
            )
            for i in range(3)
        ]

        evolved = mutator.evolve_population(
            initial_rules,
            generations=3,
            population_size=10,
        )

        assert len(evolved) >= 3

    def test_get_statistics(self):
        """Test statistics retrieval."""
        mutator = RuleMutator()
        rule = Rule(
            rule_id="rule_001",
            condition="test",
            action="test",
            confidence=0.8,
            fitness=0.7,
        )
        mutator.mutate(rule)

        stats = mutator.get_statistics()

        assert stats["mutations_performed"] >= 1
        assert "mutation_rate" in stats


class TestChainOfThoughtReasoner:
    """Tests for ChainOfThoughtReasoner class."""

    def test_init(self):
        """Test reasoner initialization."""
        reasoner = ChainOfThoughtReasoner()
        assert reasoner.max_steps == 10
        assert reasoner._chain_counter == 0

    def test_init_custom_steps(self):
        """Test reasoner with custom max steps."""
        reasoner = ChainOfThoughtReasoner(max_steps=20)
        assert reasoner.max_steps == 20

    def test_reason(self):
        """Test chain-of-thought reasoning."""
        reasoner = ChainOfThoughtReasoner()
        result = reasoner.reason(
            query="What causes this anomaly?",
            context={"type": "network", "severity": "high"},
        )

        assert isinstance(result, ThoughtChain)
        assert result.chain_id.startswith("cot_")
        assert len(result.steps) == 5
        assert result.conclusion is not None

    def test_reason_steps_order(self):
        """Test that reasoning steps follow correct order."""
        reasoner = ChainOfThoughtReasoner()
        result = reasoner.reason("Test query")

        expected_steps = [
            ReasoningStep.OBSERVE.value,
            ReasoningStep.ANALYZE.value,
            ReasoningStep.HYPOTHESIZE.value,
            ReasoningStep.EVALUATE.value,
            ReasoningStep.CONCLUDE.value,
        ]

        for i, step in enumerate(result.steps):
            assert step["step"] == expected_steps[i]

    def test_reason_confidence(self):
        """Test reasoning confidence calculation."""
        reasoner = ChainOfThoughtReasoner()
        result = reasoner.reason("Test query")

        assert 0 <= result.confidence <= 1

    def test_get_statistics(self):
        """Test statistics retrieval."""
        reasoner = ChainOfThoughtReasoner()
        reasoner.reason("Query 1")
        reasoner.reason("Query 2")

        stats = reasoner.get_statistics()

        assert stats["chains_generated"] == 2


class TestCounterfactualSimulator:
    """Tests for CounterfactualSimulator class."""

    def test_init(self):
        """Test simulator initialization."""
        simulator = CounterfactualSimulator()
        assert simulator._simulation_counter == 0

    def test_simulate(self):
        """Test counterfactual simulation."""
        simulator = CounterfactualSimulator()
        result = simulator.simulate(
            scenario={"risk_level": "high", "users": 100},
            intervention="reduce_risk",
        )

        assert isinstance(result, Counterfactual)
        assert result.counterfactual_id.startswith("cf_")
        assert result.intervention == "reduce_risk"

    def test_simulate_outcome_modified(self):
        """Test that simulation modifies outcome."""
        simulator = CounterfactualSimulator()
        scenario = {"status": "normal"}
        result = simulator.simulate(scenario, "improve_performance")

        assert result.predicted_outcome["outcome_modified"] is True
        assert result.predicted_outcome["intervention_applied"] == "improve_performance"

    def test_simulate_causal_factors(self):
        """Test causal factor identification."""
        simulator = CounterfactualSimulator()
        result = simulator.simulate(
            scenario={"value": 42, "category": "test"},
            intervention="change_value",
        )

        assert len(result.causal_factors) > 0

    def test_compare_scenarios(self):
        """Test comparing multiple scenarios."""
        simulator = CounterfactualSimulator()
        results = simulator.compare_scenarios(
            scenario={"base": True},
            interventions=["intervention_a", "intervention_b", "intervention_c"],
        )

        assert len(results) == 3

    def test_get_statistics(self):
        """Test statistics retrieval."""
        simulator = CounterfactualSimulator()
        simulator.simulate({"test": True}, "test_intervention")

        stats = simulator.get_statistics()

        assert stats["simulations_run"] == 1


class TestTheoryOfMind:
    """Tests for TheoryOfMind class."""

    def test_init(self):
        """Test engine initialization."""
        tom = TheoryOfMind()
        assert tom._inference_counter == 0

    def test_infer_intent(self):
        """Test intent inference."""
        tom = TheoryOfMind()
        result = tom.infer_intent(
            behaviors=["search for anomalies", "query database"],
            context={"user_type": "analyst"},
        )

        assert isinstance(result, IntentInference)
        assert result.inference_id.startswith("tom_")
        assert result.inferred_intent is not None

    def test_infer_intent_information_seeking(self):
        """Test information seeking intent detection."""
        tom = TheoryOfMind()
        result = tom.infer_intent(
            behaviors=["search", "find", "look up", "query"],
        )

        assert result.inferred_intent == "information_seeking"

    def test_infer_intent_problem_solving(self):
        """Test problem solving intent detection."""
        tom = TheoryOfMind()
        result = tom.infer_intent(
            behaviors=["fix the bug", "solve the issue", "debug"],
        )

        assert result.inferred_intent == "problem_solving"

    def test_infer_intent_alternatives(self):
        """Test alternative intent generation."""
        tom = TheoryOfMind()
        result = tom.infer_intent(["general action"])

        assert len(result.alternative_intents) > 0

    def test_infer_intent_evidence(self):
        """Test evidence gathering."""
        tom = TheoryOfMind()
        result = tom.infer_intent(
            behaviors=["action1", "action2"],
            context={"key": "value"},
        )

        assert len(result.evidence) > 0

    def test_get_statistics(self):
        """Test statistics retrieval."""
        tom = TheoryOfMind()
        tom.infer_intent(["action"])

        stats = tom.get_statistics()

        assert stats["inferences_made"] == 1


class TestCuriosityEngine:
    """Tests for CuriosityEngine class."""

    def test_init(self):
        """Test engine initialization."""
        engine = CuriosityEngine()
        assert engine.novelty_threshold == 0.7
        assert engine._exploration_counter == 0

    def test_init_custom_threshold(self):
        """Test engine with custom threshold."""
        engine = CuriosityEngine(novelty_threshold=0.8)
        assert engine.novelty_threshold == 0.8

    def test_explore(self):
        """Test exploration."""
        engine = CuriosityEngine()
        result = engine.explore(
            target="unusual_pattern_xyz",
            data={"attribute": "value"},
        )

        assert isinstance(result, ExplorationResult)
        assert result.exploration_id.startswith("explore_")
        assert 0 <= result.novelty_score <= 1

    def test_explore_discoveries(self):
        """Test discovery generation."""
        engine = CuriosityEngine()
        result = engine.explore(
            target="novel_anomaly_pattern",
            data={"a": 1, "b": 2},
        )

        assert len(result.discoveries) > 0

    def test_explore_questions(self):
        """Test question generation."""
        engine = CuriosityEngine()
        result = engine.explore("test_target")

        assert len(result.questions_generated) > 0

    def test_explore_follow_ups(self):
        """Test follow-up action generation."""
        engine = CuriosityEngine()
        result = engine.explore("test_target")

        assert len(result.follow_up_actions) > 0

    def test_known_patterns_tracked(self):
        """Test that known patterns are tracked."""
        engine = CuriosityEngine(novelty_threshold=0.5)
        engine.explore("pattern_a")

        result = engine.explore("pattern_a")

        assert result.novelty_score < 0.5

    def test_get_statistics(self):
        """Test statistics retrieval."""
        engine = CuriosityEngine()
        engine.explore("target1")
        engine.explore("target2")

        stats = engine.get_statistics()

        assert stats["explorations_performed"] == 2


class TestCognitiveEvolutionEngine:
    """Tests for CognitiveEvolutionEngine class."""

    def test_init(self):
        """Test engine initialization."""
        engine = CognitiveEvolutionEngine()
        assert engine.safety_threshold == 0.99
        assert engine.max_improvement_cycles == 100

    def test_init_custom_params(self):
        """Test bootstrap with custom parameters."""
        bootstrap = CognitiveEvolutionEngine(
            safety_threshold=0.95,
            max_improvement_cycles=50,
        )
        assert bootstrap.safety_threshold == 0.95
        assert bootstrap.max_improvement_cycles == 50

    def test_run_improvement_cycle(self):
        """Test running improvement cycle."""
        bootstrap = CognitiveEvolutionEngine()
        result = bootstrap.run_improvement_cycle(
            scenario={"type": "anomaly", "severity": "medium"},
        )

        assert "cycle" in result
        assert "improvements" in result
        assert "safety_check" in result
        assert "status" in result

    def test_run_improvement_cycle_with_rules(self):
        """Test improvement cycle with rules."""
        bootstrap = CognitiveEvolutionEngine()
        rules = [
            Rule(
                rule_id="rule_001",
                condition="test",
                action="alert",
                confidence=0.8,
                fitness=0.7,
            )
        ]

        result = bootstrap.run_improvement_cycle(
            scenario={"test": True},
            rules=rules,
        )

        assert "evolved_rules" in result

    def test_max_cycles_limit(self):
        """Test maximum cycles limit."""
        bootstrap = CognitiveEvolutionEngine(max_improvement_cycles=2)

        bootstrap.run_improvement_cycle({"test": True})
        bootstrap.run_improvement_cycle({"test": True})
        result = bootstrap.run_improvement_cycle({"test": True})

        assert result["status"] == "max_cycles_reached"

    def test_infer_user_intent(self):
        """Test user intent inference."""
        bootstrap = CognitiveEvolutionEngine()
        result = bootstrap.infer_user_intent(
            behaviors=["search", "analyze"],
            context={"role": "analyst"},
        )

        assert isinstance(result, IntentInference)

    def test_simulate_counterfactual(self):
        """Test counterfactual simulation."""
        bootstrap = CognitiveEvolutionEngine()
        result = bootstrap.simulate_counterfactual(
            scenario={"risk": "high"},
            intervention="mitigate_risk",
        )

        assert isinstance(result, Counterfactual)

    def test_reason_about(self):
        """Test chain-of-thought reasoning."""
        bootstrap = CognitiveEvolutionEngine()
        result = bootstrap.reason_about(
            query="What is the root cause?",
            context={"symptoms": ["a", "b"]},
        )

        assert isinstance(result, ThoughtChain)

    def test_explore_novelty(self):
        """Test novelty exploration."""
        bootstrap = CognitiveEvolutionEngine()
        result = bootstrap.explore_novelty(
            target="unknown_pattern",
            data={"value": 42},
        )

        assert isinstance(result, ExplorationResult)

    def test_get_statistics(self):
        """Test statistics retrieval."""
        bootstrap = CognitiveEvolutionEngine()
        bootstrap.run_improvement_cycle({"test": True})

        stats = bootstrap.get_statistics()

        assert stats["improvement_cycles"] == 1
        assert "self_play" in stats
        assert "rule_mutator" in stats
        assert "reasoner" in stats


class TestEnums:
    """Tests for enum classes."""

    def test_agent_roles(self):
        """Test all agent roles exist."""
        assert AgentRole.EXPLORER.value == "explorer"
        assert AgentRole.CRITIC.value == "critic"
        assert AgentRole.OPTIMIZER.value == "optimizer"
        assert AgentRole.VALIDATOR.value == "validator"
        assert AgentRole.ADVERSARY.value == "adversary"

    def test_mutation_types(self):
        """Test all mutation types exist."""
        assert MutationType.CROSSOVER.value == "crossover"
        assert MutationType.POINT_MUTATION.value == "point_mutation"
        assert MutationType.INSERTION.value == "insertion"
        assert MutationType.DELETION.value == "deletion"
        assert MutationType.INVERSION.value == "inversion"

    def test_reasoning_steps(self):
        """Test all reasoning steps exist."""
        assert ReasoningStep.OBSERVE.value == "observe"
        assert ReasoningStep.ANALYZE.value == "analyze"
        assert ReasoningStep.HYPOTHESIZE.value == "hypothesize"
        assert ReasoningStep.EVALUATE.value == "evaluate"
        assert ReasoningStep.CONCLUDE.value == "conclude"


class TestDataclasses:
    """Tests for dataclasses."""

    def test_simulation_agent(self):
        """Test SimulationAgent dataclass."""
        agent = SimulationAgent(
            agent_id="agent_001",
            role=AgentRole.EXPLORER,
            confidence=0.85,
        )
        assert agent.agent_id == "agent_001"
        assert agent.role == AgentRole.EXPLORER

    def test_rule(self):
        """Test Rule dataclass."""
        rule = Rule(
            rule_id="rule_001",
            condition="if x > 10",
            action="alert",
            confidence=0.9,
            fitness=0.8,
        )
        assert rule.rule_id == "rule_001"
        assert rule.fitness == 0.8

    def test_thought_chain(self):
        """Test ThoughtChain dataclass."""
        chain = ThoughtChain(
            chain_id="cot_001",
            query="Test query",
            steps=[{"step": "observe", "content": "test"}],
            conclusion="Test conclusion",
            confidence=0.85,
            reasoning_time_ms=100.0,
        )
        assert chain.chain_id == "cot_001"
        assert chain.confidence == 0.85


class TestIntegration:
    """Integration tests for superintelligence bootstrap."""

    def test_full_improvement_pipeline(self):
        """Test complete improvement pipeline."""
        bootstrap = CognitiveEvolutionEngine(
            safety_threshold=0.5,
            max_improvement_cycles=10,
        )

        rules = [
            Rule(
                rule_id=f"rule_{i}",
                condition=f"condition_{i}",
                action=f"action_{i}",
                confidence=0.7,
                fitness=0.6,
            )
            for i in range(3)
        ]

        result = bootstrap.run_improvement_cycle(
            scenario={"type": "complex_anomaly", "data": [1, 2, 3]},
            rules=rules,
        )

        assert result["status"] == "success"
        assert result["safety_check"] is True

    def test_multi_cycle_improvement(self):
        """Test multiple improvement cycles."""
        bootstrap = CognitiveEvolutionEngine(max_improvement_cycles=5)

        for i in range(3):
            result = bootstrap.run_improvement_cycle(
                scenario={"iteration": i},
            )
            assert result["cycle"] == i + 1

    def test_cognitive_capabilities_integration(self):
        """Test integration of all cognitive capabilities."""
        bootstrap = CognitiveEvolutionEngine()

        intent = bootstrap.infer_user_intent(["analyze data"])
        assert intent.inferred_intent is not None

        cf = bootstrap.simulate_counterfactual({"state": "a"}, "change_state")
        assert cf.predicted_outcome is not None

        reasoning = bootstrap.reason_about("Why did this happen?")
        assert reasoning.conclusion is not None

        exploration = bootstrap.explore_novelty("new_pattern")
        assert exploration.novelty_score > 0

    def test_safety_enforcement(self):
        """Test that safety is enforced."""
        bootstrap = CognitiveEvolutionEngine(safety_threshold=1.0)

        result = bootstrap.run_improvement_cycle({"risky": True})

        assert result["safety_score"] < 1.0
