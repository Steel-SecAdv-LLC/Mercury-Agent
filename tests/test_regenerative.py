# Copyright (C) 2025 Steel Security Advisors LLC
"""Tests for Regenerative Architecture module."""

from __future__ import annotations

from typing import Any

import numpy as np

from omni_mercury_engine.core.regenerative import (
    FeedbackLoop,
    PermaculturePrinciple,
    RegenerativeArchitecture,
)


class TestRegenerativeArchitecture:
    """Test suite for RegenerativeArchitecture."""

    def test_initialization(self) -> None:
        """Test basic initialization."""
        arch = RegenerativeArchitecture()
        assert arch.enable_closed_loops is True
        assert len(arch.feedback_loops) == 0
        assert len(arch.knowledge_bank) == 0
        assert len(arch.waste_log) == 0

    def test_initialization_without_closed_loops(self) -> None:
        """Test initialization with closed loops disabled."""
        arch = RegenerativeArchitecture(enable_closed_loops=False)
        assert arch.enable_closed_loops is False

    def test_create_feedback_loop(self) -> None:
        """Test creating a feedback loop."""
        arch = RegenerativeArchitecture()
        loop = arch.create_feedback_loop(
            loop_id="test_loop",
            input_metric="accuracy",
            output_metric="learning_rate",
            gain=0.5,
            delay=1.0,
            is_positive=True,
        )

        assert isinstance(loop, FeedbackLoop)
        assert loop.loop_id == "test_loop"
        assert loop.gain == 0.5
        assert loop.is_positive is True
        assert "test_loop" in arch.feedback_loops

    def test_apply_feedback_loops_positive(self) -> None:
        """Test applying positive feedback loops."""
        arch = RegenerativeArchitecture()
        arch.create_feedback_loop(
            loop_id="positive_loop",
            input_metric="input",
            output_metric="output",
            gain=0.1,
            is_positive=True,
        )

        metrics = {"input": 1.0, "output": 0.5}
        updated = arch.apply_feedback_loops(metrics)

        assert updated["output"] == 0.5 + 0.1 * 1.0

    def test_apply_feedback_loops_negative(self) -> None:
        """Test applying negative feedback loops."""
        arch = RegenerativeArchitecture()
        arch.create_feedback_loop(
            loop_id="negative_loop",
            input_metric="input",
            output_metric="output",
            gain=0.1,
            is_positive=False,
        )

        metrics = {"input": 1.0, "output": 0.5}
        updated = arch.apply_feedback_loops(metrics)

        expected = 0.5 - 0.1 * (0.5 - 1.0)
        assert abs(updated["output"] - expected) < 1e-6

    def test_apply_feedback_loops_disabled(self) -> None:
        """Test that feedback is not applied when disabled."""
        arch = RegenerativeArchitecture(enable_closed_loops=False)
        arch.create_feedback_loop(
            loop_id="loop", input_metric="input", output_metric="output", gain=0.1
        )

        metrics = {"input": 1.0, "output": 0.5}
        updated = arch.apply_feedback_loops(metrics)

        assert updated["output"] == 0.5

    def test_apply_permaculture_observe_interact(self) -> None:
        """Test OBSERVE_INTERACT principle."""
        arch = RegenerativeArchitecture()
        context: dict[str, Any] = {}

        result = arch.apply_permaculture_principle(PermaculturePrinciple.OBSERVE_INTERACT, context)

        assert "observation_data" in result
        assert arch.permaculture_metrics[PermaculturePrinciple.OBSERVE_INTERACT] == 1.0

    def test_apply_permaculture_catch_store_energy(self) -> None:
        """Test CATCH_STORE_ENERGY principle."""
        arch = RegenerativeArchitecture()
        context = {"learned_patterns": ["pattern1", "pattern2"]}

        arch.apply_permaculture_principle(PermaculturePrinciple.CATCH_STORE_ENERGY, context)

        assert len(arch.knowledge_bank) == 1
        assert "patterns" in arch.knowledge_bank[0]

    def test_apply_permaculture_obtain_yield(self) -> None:
        """Test OBTAIN_YIELD principle."""
        arch = RegenerativeArchitecture()
        context = {"accuracy": 0.9, "efficiency": 0.8}

        result = arch.apply_permaculture_principle(PermaculturePrinciple.OBTAIN_YIELD, context)

        assert "value_metrics" in result
        assert "total_value" in result["value_metrics"]

    def test_apply_permaculture_self_regulate(self) -> None:
        """Test SELF_REGULATE principle."""
        arch = RegenerativeArchitecture()
        context: dict[str, Any] = {}

        result = arch.apply_permaculture_principle(PermaculturePrinciple.SELF_REGULATE, context)

        assert result["ethical_constraints_applied"] is True
        assert result["parameters_tuned"] is True

    def test_apply_permaculture_renewable_resources(self) -> None:
        """Test RENEWABLE_RESOURCES principle."""
        arch = RegenerativeArchitecture()
        context = {"dependencies": ["numpy", "proprietary_lib", "scipy"]}

        result = arch.apply_permaculture_principle(
            PermaculturePrinciple.RENEWABLE_RESOURCES, context
        )

        assert "proprietary_lib" not in result["dependencies"]
        assert "numpy" in result["dependencies"]

    def test_apply_permaculture_produce_no_waste(self) -> None:
        """Test PRODUCE_NO_WASTE principle."""
        arch = RegenerativeArchitecture()
        context: dict[str, Any] = {}

        result = arch.apply_permaculture_principle(PermaculturePrinciple.PRODUCE_NO_WASTE, context)

        assert len(arch.waste_log) == 1
        assert "waste_eliminated" in result

    def test_apply_permaculture_patterns_to_details(self) -> None:
        """Test PATTERNS_TO_DETAILS principle."""
        arch = RegenerativeArchitecture()
        context = {"patterns": ["pattern1", "pattern2"]}

        result = arch.apply_permaculture_principle(
            PermaculturePrinciple.PATTERNS_TO_DETAILS, context
        )

        assert "detailed_design" in result

    def test_apply_permaculture_integrate(self) -> None:
        """Test INTEGRATE principle."""
        arch = RegenerativeArchitecture()
        context = {"isolated_components": ["comp1", "comp2", "comp3"]}

        result = arch.apply_permaculture_principle(PermaculturePrinciple.INTEGRATE, context)

        assert "integrated_system" in result
        assert result["integrated_system"]["component_count"] == 3

    def test_apply_permaculture_small_slow(self) -> None:
        """Test SMALL_SLOW principle."""
        arch = RegenerativeArchitecture()
        context = {"change_magnitude": 0.5}

        result = arch.apply_permaculture_principle(PermaculturePrinciple.SMALL_SLOW, context)

        assert result["change_magnitude"] == 0.1

    def test_apply_permaculture_value_diversity(self) -> None:
        """Test VALUE_DIVERSITY principle."""
        arch = RegenerativeArchitecture()
        context = {"models": ["model_a", "model_b", "model_a"]}

        result = arch.apply_permaculture_principle(PermaculturePrinciple.VALUE_DIVERSITY, context)

        assert result["ensemble_size"] == 3
        assert "diversity_score" in result

    def test_apply_permaculture_use_edges(self) -> None:
        """Test USE_EDGES principle."""
        arch = RegenerativeArchitecture()
        data = np.random.randn(100, 5)
        context = {"data": data}

        result = arch.apply_permaculture_principle(PermaculturePrinciple.USE_EDGES, context)

        assert "edge_cases" in result
        assert "anomaly_potential" in result

    def test_apply_permaculture_respond_to_change(self) -> None:
        """Test RESPOND_TO_CHANGE principle."""
        arch = RegenerativeArchitecture()
        context = {"environmental_changes": {"temperature": 2.0, "load": 1.5}}

        result = arch.apply_permaculture_principle(PermaculturePrinciple.RESPOND_TO_CHANGE, context)

        assert "adaptation_strategy" in result
        assert result["adaptation_strategy"]["changes_addressed"] == 2

    def test_calculate_net_positive_score(self) -> None:
        """Test net-positive score calculation."""
        arch = RegenerativeArchitecture()

        context = {
            "value_metrics": {"total_value": 10.0},
            "resource_usage": 5.0,
        }

        score = arch.calculate_net_positive_score(context)

        assert score > 0
        assert isinstance(score, float)

    def test_calculate_net_positive_score_with_knowledge(self) -> None:
        """Test net-positive score with accumulated knowledge."""
        arch = RegenerativeArchitecture()
        arch.knowledge_bank.append({"patterns": ["p1"]})
        arch.knowledge_bank.append({"patterns": ["p2"]})

        context = {
            "value_metrics": {"total_value": 10.0},
            "resource_usage": 5.0,
        }

        score = arch.calculate_net_positive_score(context)

        assert score > 0

    def test_identify_edges_empty_data(self) -> None:
        """Test edge identification with empty data."""
        arch = RegenerativeArchitecture()

        edges = arch._identify_edges(np.array([]))

        assert len(edges) == 0


class TestFeedbackLoop:
    """Test FeedbackLoop dataclass."""

    def test_feedback_loop_creation(self) -> None:
        """Test creating a FeedbackLoop."""
        loop = FeedbackLoop(
            loop_id="test",
            input_metric="in",
            output_metric="out",
            gain=1.0,
            delay=0.5,
            is_positive=True,
        )

        assert loop.loop_id == "test"
        assert loop.gain == 1.0
        assert loop.is_positive is True


class TestPermaculturePrinciple:
    """Test PermaculturePrinciple enum."""

    def test_principle_values(self) -> None:
        """Test that all 12 principles are defined."""
        assert len(PermaculturePrinciple) == 12
        assert PermaculturePrinciple.OBSERVE_INTERACT.value == 1
        assert PermaculturePrinciple.RESPOND_TO_CHANGE.value == 12
