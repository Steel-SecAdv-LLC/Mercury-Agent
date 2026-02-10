"""
Mercury Agent ♱
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

"""Tests for Code Analysis Engine."""

import ast

import numpy as np
import pytest

from omni_mercury_engine.core.code_analysis import (
    CodeAnalysisEngine,
    NeurosymbolicConfig,
    NeurosymbolicEngine,
    ReadinessLevel,
    TrainingMetrics,
    TrainingPhase,
)
from omni_mercury_engine.utils.rng import DeterministicRNG


class TestEnums:
    """Tests for enumerations."""

    def test_readiness_level_values(self):
        """Test ReadinessLevel enum values."""
        assert ReadinessLevel.NOT_READY.value == "not_ready"
        assert ReadinessLevel.NEEDS_IMPROVEMENT.value == "needs_improvement"
        assert ReadinessLevel.READY.value == "ready"
        assert ReadinessLevel.PRODUCTION_READY.value == "production_ready"

    def test_training_phase_values(self):
        """Test TrainingPhase enum values."""
        assert TrainingPhase.FOUNDATION.value == "foundation"
        assert TrainingPhase.SPECIALIZATION.value == "specialization"
        assert TrainingPhase.INTEGRATION.value == "integration"
        assert TrainingPhase.VALIDATION.value == "validation"
        assert TrainingPhase.DEPLOYMENT.value == "deployment"


class TestNeurosymbolicConfig:
    """Tests for NeurosymbolicConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = NeurosymbolicConfig()

        assert config.enable_neural is False
        assert config.enable_symbolic is True
        assert config.training_data_path is None
        assert config.model_path is None
        assert config.bias_check_enabled is True
        assert config.transparency_logging is True
        assert config.enable_backprop_tuning is False
        assert config.backprop_learning_rate == 0.001
        assert config.backprop_quantum_noise == 0.01

    def test_custom_values(self):
        """Test custom configuration values."""
        config = NeurosymbolicConfig(
            enable_neural=True,
            bias_check_enabled=False,
            backprop_learning_rate=0.01,
        )

        assert config.enable_neural is True
        assert config.bias_check_enabled is False
        assert config.backprop_learning_rate == 0.01


class TestTrainingMetrics:
    """Tests for TrainingMetrics dataclass."""

    def test_default_values(self):
        """Test default metric values."""
        metrics = TrainingMetrics()

        assert metrics.epoch == 0
        assert metrics.loss == float("inf")
        assert metrics.accuracy == 0.0
        assert metrics.validation_loss == float("inf")
        assert metrics.validation_accuracy == 0.0
        assert metrics.readiness_level == ReadinessLevel.NOT_READY


class TestNeurosymbolicEngine:
    """Tests for NeurosymbolicEngine."""

    @pytest.fixture
    def engine(self):
        """Create engine with deterministic RNG."""
        return NeurosymbolicEngine(
            config=NeurosymbolicConfig(transparency_logging=False),
            rng=DeterministicRNG(seed=42),
        )

    @pytest.fixture
    def sample_code_ast(self):
        """Create sample code AST."""
        code = """
def example_function(x):
    for i in range(10):
        if i > 5:
            result = x + i
    return result
"""
        return ast.parse(code)

    def test_initialization_default(self, engine):
        """Test default initialization."""
        assert engine.config is not None
        assert engine.training_metrics is not None
        assert engine.current_phase == TrainingPhase.FOUNDATION
        assert engine.neural_model is None
        assert engine.pattern_library == {}

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = NeurosymbolicConfig(enable_neural=True)
        engine = NeurosymbolicEngine(config=config)

        assert engine.config.enable_neural is True

    def test_symbolic_analysis(self, engine, sample_code_ast):
        """Test symbolic code analysis."""
        result = engine.symbolic_analysis(sample_code_ast)

        assert result["method"] == "symbolic"
        assert "patterns" in result
        assert result["confidence"] == 1.0

        patterns = result["patterns"]
        assert patterns["loops"] >= 1
        assert patterns["conditionals"] >= 1
        assert patterns["function_calls"] >= 0
        assert patterns["nesting_depth"] >= 0

    def test_symbolic_analysis_empty_ast(self, engine):
        """Test symbolic analysis with empty AST."""
        empty_ast = ast.parse("")
        result = engine.symbolic_analysis(empty_ast)

        assert result["patterns"]["loops"] == 0
        assert result["patterns"]["conditionals"] == 0

    def test_symbolic_analysis_nested_loops(self, engine):
        """Test symbolic analysis with nested loops."""
        code = """
for i in range(10):
    for j in range(5):
        if i > j:
            print(i + j)
"""
        code_ast = ast.parse(code)
        result = engine.symbolic_analysis(code_ast)

        assert result["patterns"]["loops"] >= 2
        assert result["patterns"]["nesting_depth"] >= 2

    def test_neural_analysis_disabled(self, engine):
        """Test neural analysis when disabled returns statistical fallback."""
        features = np.array([1.0, 2.0, 3.0, 4.0])
        result = engine.neural_analysis(features)

        assert result["method"] == "statistical_fallback"
        assert result["available"] is True
        assert result["neural_model_trained"] is False
        assert "statistics" in result

    def test_neural_analysis_enabled_no_model(self):
        """Test neural analysis when enabled but no model returns statistical fallback."""
        config = NeurosymbolicConfig(enable_neural=True)
        engine = NeurosymbolicEngine(config=config)

        features = np.array([1.0, 2.0, 3.0, 4.0])
        result = engine.neural_analysis(features)

        assert result["available"] is True
        assert result["neural_model_trained"] is False
        assert result["method"] == "statistical_fallback"

    def test_hybrid_analysis(self, engine, sample_code_ast):
        """Test hybrid analysis combining symbolic and neural."""
        result = engine.hybrid_analysis(sample_code_ast)

        assert "symbolic" in result
        assert "neural" in result
        assert "hybrid_confidence" in result

        assert result["symbolic"]["method"] == "symbolic"
        assert result["neural"]["method"] == "statistical_fallback"

    def test_train_model_no_data(self, engine):
        """Test training without data."""
        metrics = engine.train_model(training_data=None)

        assert metrics == engine.training_metrics

    def test_train_model_neural_disabled(self, engine, sample_code_ast):
        """Test training with neural disabled."""
        training_data = [(sample_code_ast, {"refactoring": "extract_method"})]
        metrics = engine.train_model(training_data)

        assert metrics == engine.training_metrics

    def test_train_model_with_data(self, sample_code_ast):
        """Test training with data and neural enabled."""
        config = NeurosymbolicConfig(enable_neural=True, transparency_logging=False)
        engine = NeurosymbolicEngine(config=config)

        # Need at least 2 samples for training
        training_data = [
            (sample_code_ast, {"refactoring": "extract_method"}),
            (sample_code_ast, {"refactoring": "rename_variable"}),
        ]
        engine.train_model(training_data)

        # After training completes, phase should be VALIDATION
        assert engine.current_phase == TrainingPhase.VALIDATION

    def test_check_bias_disabled(self):
        """Test bias check when disabled."""
        config = NeurosymbolicConfig(bias_check_enabled=False)
        engine = NeurosymbolicEngine(config=config)

        predictions = [{"type": "extract_method"}]
        result = engine.check_bias(predictions)

        assert result["bias_check"] == "disabled"

    def test_check_bias_no_predictions(self, engine):
        """Test bias check with no predictions."""
        result = engine.check_bias([])

        assert result["bias_detected"] is False
        assert "No predictions" in result["message"]

    def test_check_bias_diverse_predictions(self, engine):
        """Test bias check with diverse predictions."""
        predictions = [
            {"type": "extract_method"},
            {"type": "rename_variable"},
            {"type": "inline_function"},
            {"type": "extract_class"},
        ]
        result = engine.check_bias(predictions)

        assert result["bias_detected"] is False
        assert result["diversity_ratio"] == 1.0

    def test_check_bias_low_diversity(self, engine):
        """Test bias check with low diversity."""
        predictions = [
            {"type": "extract_method"},
            {"type": "extract_method"},
            {"type": "extract_method"},
            {"type": "extract_method"},
            {"type": "rename_variable"},
        ]
        result = engine.check_bias(predictions)

        assert result["diversity_ratio"] < 0.5

    def test_get_readiness_level_neural_disabled(self, engine):
        """Test readiness level with neural disabled."""
        level = engine.get_readiness_level()

        assert level == ReadinessLevel.PRODUCTION_READY

    def test_get_readiness_level_high_accuracy(self):
        """Test readiness level with high accuracy."""
        config = NeurosymbolicConfig(enable_neural=True)
        engine = NeurosymbolicEngine(config=config)
        engine.training_metrics.accuracy = 0.96

        level = engine.get_readiness_level()

        assert level == ReadinessLevel.PRODUCTION_READY

    def test_get_readiness_level_medium_accuracy(self):
        """Test readiness level with medium accuracy."""
        config = NeurosymbolicConfig(enable_neural=True)
        engine = NeurosymbolicEngine(config=config)
        engine.training_metrics.accuracy = 0.85

        level = engine.get_readiness_level()

        assert level == ReadinessLevel.READY

    def test_get_readiness_level_low_accuracy(self):
        """Test readiness level with low accuracy."""
        config = NeurosymbolicConfig(enable_neural=True)
        engine = NeurosymbolicEngine(config=config)
        engine.training_metrics.accuracy = 0.65

        level = engine.get_readiness_level()

        assert level == ReadinessLevel.NEEDS_IMPROVEMENT

    def test_get_readiness_level_very_low_accuracy(self):
        """Test readiness level with very low accuracy."""
        config = NeurosymbolicConfig(enable_neural=True)
        engine = NeurosymbolicEngine(config=config)
        engine.training_metrics.accuracy = 0.3

        level = engine.get_readiness_level()

        assert level == ReadinessLevel.NOT_READY


class TestBackpropTuning:
    """Tests for backpropagation tuning functionality."""

    @pytest.fixture
    def engine(self):
        """Create engine with backprop enabled."""
        config = NeurosymbolicConfig(
            enable_backprop_tuning=True,
            backprop_learning_rate=0.01,
            backprop_quantum_noise=0.001,
            transparency_logging=False,
        )
        return NeurosymbolicEngine(config=config, rng=DeterministicRNG(seed=42))

    def test_backprop_disabled(self):
        """Test backprop tuning when disabled."""
        config = NeurosymbolicConfig(enable_backprop_tuning=False)
        engine = NeurosymbolicEngine(config=config)

        features = np.array([1.0, 2.0, 3.0, 4.0])
        ground_truth = np.array([0.5])
        result = engine.backprop_tune_patterns(features, ground_truth)

        assert result["enabled"] is False

    def test_backprop_enabled(self, engine):
        """Test backprop tuning when enabled."""
        features = np.array([1.0, 2.0, 3.0, 4.0])
        ground_truth = np.array([0.5])
        result = engine.backprop_tune_patterns(features, ground_truth, iterations=50)

        assert result["enabled"] is True
        assert result["iterations"] == 50
        assert "final_loss" in result
        assert "initial_loss" in result
        assert result["tensor_shape"] == (1, 1, 2, 2)

    def test_backprop_convergence(self, engine):
        """Test that backprop shows convergence."""
        features = np.array([1.0, 2.0, 3.0, 4.0])
        ground_truth = np.array([5.0])
        result = engine.backprop_tune_patterns(features, ground_truth, iterations=100)

        # Loss should decrease
        assert result["final_loss"] <= result["initial_loss"]
        assert result["convergence"] >= 0

    def test_backprop_short_features(self, engine):
        """Test backprop with features shorter than 4."""
        features = np.array([1.0, 2.0])
        ground_truth = np.array([0.5])
        result = engine.backprop_tune_patterns(features, ground_truth, iterations=10)

        assert result["enabled"] is True
        assert result["tensor_shape"] == (1, 1, 2, 2)

    def test_backprop_long_features(self, engine):
        """Test backprop with features longer than 4."""
        features = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        ground_truth = np.array([0.5])
        result = engine.backprop_tune_patterns(features, ground_truth, iterations=10)

        assert result["enabled"] is True


class TestAliases:
    """Tests for module aliases."""

    def test_code_analysis_engine_alias(self):
        """Test CodeAnalysisEngine is alias for NeurosymbolicEngine."""
        assert CodeAnalysisEngine is NeurosymbolicEngine
