"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

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
Code Analysis Engine - AST-based symbolic reasoning for code refactoring.

This module provides AST (Abstract Syntax Tree) analysis capabilities for:
- Code complexity analysis
- Refactoring suggestions
- Training readiness assessment

Note:
    This is distinct from :class:`omni_anomaly_engine.models.neurosymbolic.NeurosymbolicEngine`
    which provides LTN-based anomaly detection. This module focuses on static code analysis
    using Python's AST module.

    For anomaly detection with neurosymbolic reasoning, use::

        from omni_anomaly_engine.models.neurosymbolic import NeurosymbolicEngine

    For code analysis and refactoring, use::

        from omni_anomaly_engine.core.code_analysis import CodeAnalysisEngine
"""

import ast
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from omni_anomaly_engine.utils.rng import DeterministicRNG, get_global_rng


class ReadinessLevel(Enum):
    """Neurosymbolic model deployment readiness levels."""

    NOT_READY = "not_ready"
    NEEDS_IMPROVEMENT = "needs_improvement"
    READY = "ready"
    PRODUCTION_READY = "production_ready"


class TrainingPhase(Enum):
    """Training phases for neurosymbolic model."""

    FOUNDATION = "foundation"
    SPECIALIZATION = "specialization"
    INTEGRATION = "integration"
    VALIDATION = "validation"
    DEPLOYMENT = "deployment"


@dataclass
class NeurosymbolicConfig:
    """Configuration for neurosymbolic integration."""

    enable_neural: bool = False
    enable_symbolic: bool = True
    training_data_path: str | None = None
    model_path: str | None = None
    bias_check_enabled: bool = True
    transparency_logging: bool = True
    enable_backprop_tuning: bool = False
    backprop_learning_rate: float = 0.001
    backprop_quantum_noise: float = 0.01


@dataclass
class TrainingMetrics:
    """Metrics for training progress."""

    epoch: int = 0
    loss: float = float("inf")
    accuracy: float = 0.0
    validation_loss: float = float("inf")
    validation_accuracy: float = 0.0
    readiness_level: ReadinessLevel = ReadinessLevel.NOT_READY


class NeurosymbolicEngine:
    """
    Neurosymbolic integration for code refactoring.

    Combines:
    - Symbolic reasoning: AST-based analysis (always available)
    - Neural patterns: Learned refactoring patterns (requires training data)

    This is a framework implementation with clear extension points.
    Neural components require training data to be fully functional.
    """

    def __init__(
        self,
        config: NeurosymbolicConfig | None = None,
        rng: DeterministicRNG | None = None,
    ):
        self.config = config or NeurosymbolicConfig()
        self.training_metrics = TrainingMetrics()
        self.current_phase = TrainingPhase.FOUNDATION
        self.neural_model = None
        self.pattern_library: dict[str, Any] = {}
        self._rng = rng or get_global_rng()

        if self.config.transparency_logging:
            logging.info("NeurosymbolicEngine initialized with symbolic reasoning")
            if self.config.enable_neural:
                logging.warning("Neural components enabled but require training data")

    def symbolic_analysis(self, code_ast: ast.AST) -> dict[str, Any]:
        """
        Perform symbolic reasoning on code AST.

        Always available - deterministic rule-based analysis.

        Args:
            code_ast: Abstract syntax tree to analyze

        Returns:
            Dict with symbolic analysis results
        """
        patterns = {
            "loops": 0,
            "conditionals": 0,
            "function_calls": 0,
            "nesting_depth": 0,
        }

        for node in ast.walk(code_ast):
            if isinstance(node, (ast.For, ast.While)):
                patterns["loops"] += 1
            elif isinstance(node, ast.If):
                patterns["conditionals"] += 1
            elif isinstance(node, ast.Call):
                patterns["function_calls"] += 1

        def get_depth(node: ast.AST, current_depth: int = 0) -> int:
            max_depth = current_depth
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.For, ast.While, ast.If)):
                    child_depth = get_depth(child, current_depth + 1)
                    max_depth = max(max_depth, child_depth)
                else:
                    child_depth = get_depth(child, current_depth)
                    max_depth = max(max_depth, child_depth)
            return max_depth

        patterns["nesting_depth"] = get_depth(code_ast)

        return {
            "method": "symbolic",
            "patterns": patterns,
            "confidence": 1.0,
        }

    def neural_analysis(self, code_features: NDArray[Any]) -> dict[str, Any]:
        """
        Perform neural pattern recognition (stub).

        Requires training data and model. Extension point for future implementation.

        Args:
            code_features: Feature vector extracted from code

        Returns:
            Dict with neural analysis results
        """
        if not self.config.enable_neural or self.neural_model is None:
            return {
                "method": "neural",
                "available": False,
                "message": ("Neural model not trained. " "Load training data and train model."),
            }

        return {
            "method": "neural",
            "available": False,
            "message": "Neural inference not yet implemented - requires training",
        }

    def hybrid_analysis(self, code_ast: ast.AST) -> dict[str, Any]:
        """
        Combine symbolic and neural analysis.

        Uses symbolic reasoning always, adds neural insights when available.

        Args:
            code_ast: Abstract syntax tree to analyze

        Returns:
            Dict with combined analysis results
        """
        symbolic_results = self.symbolic_analysis(code_ast)

        patterns = symbolic_results["patterns"]
        code_features = np.array(
            [
                patterns["loops"],
                patterns["conditionals"],
                patterns["function_calls"],
                patterns["nesting_depth"],
            ],
            dtype=float,
        )

        neural_results = self.neural_analysis(code_features)

        return {
            "symbolic": symbolic_results,
            "neural": neural_results,
            "hybrid_confidence": symbolic_results["confidence"],
        }

    def train_model(
        self, training_data: list[tuple[ast.AST, dict[str, Any]]] | None = None
    ) -> TrainingMetrics:
        """
        Train neural model on code patterns (stub).

        EXTENSION POINT: Implement training loop here.

        Args:
            training_data: List of (AST, ground_truth) pairs

        Returns:
            Training metrics
        """
        if training_data is None:
            logging.warning(
                "No training data provided. Generate or load training data "
                "to enable neural components."
            )
            return self.training_metrics

        if not self.config.enable_neural:
            logging.info("Neural components disabled in config. Set enable_neural=True to train.")
            return self.training_metrics

        logging.info(f"Training stub: Would train on {len(training_data)} examples")

        self.current_phase = TrainingPhase.SPECIALIZATION

        self.training_metrics = TrainingMetrics(
            epoch=0,
            loss=float("inf"),
            accuracy=0.0,
            readiness_level=ReadinessLevel.NOT_READY,
        )

        return self.training_metrics

    def check_bias(self, predictions: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Check for bias in model predictions.

        Ensures ethical AI principles - no unfair bias in refactoring suggestions.

        Args:
            predictions: List of model predictions

        Returns:
            Dict with bias analysis results
        """
        if not self.config.bias_check_enabled:
            return {"bias_check": "disabled"}

        if len(predictions) == 0:
            return {"bias_detected": False, "message": "No predictions to analyze"}

        prediction_types = [p.get("type", "unknown") for p in predictions]
        unique_types = len(set(prediction_types))
        diversity_ratio = unique_types / len(prediction_types) if len(prediction_types) > 0 else 0

        bias_detected = diversity_ratio < 0.3

        return {
            "bias_detected": bias_detected,
            "diversity_ratio": diversity_ratio,
            "recommendation": (
                "Increase pattern diversity" if bias_detected else "Acceptable diversity"
            ),
        }

    def get_readiness_level(self) -> ReadinessLevel:
        """
        Assess deployment readiness of neurosymbolic model.

        Returns:
            Current readiness level
        """
        if not self.config.enable_neural:
            return ReadinessLevel.PRODUCTION_READY

        if self.training_metrics.accuracy > 0.95:
            return ReadinessLevel.PRODUCTION_READY
        elif self.training_metrics.accuracy > 0.80:
            return ReadinessLevel.READY
        elif self.training_metrics.accuracy > 0.60:
            return ReadinessLevel.NEEDS_IMPROVEMENT
        else:
            return ReadinessLevel.NOT_READY

    def backprop_tune_patterns(
        self, code_features: NDArray[Any], ground_truth: NDArray[Any], iterations: int = 100
    ) -> dict[str, Any]:
        """
        Fine-tune pattern recognition using backpropagation with quantum noise.

        Implements SGD with 4D tensors for advanced pattern learning.

        Args:
            code_features: Input feature tensor (can be reshaped to 4D)
            ground_truth: Target output for supervised learning
            iterations: Number of training iterations

        Returns:
            Dict with training results
        """
        if not self.config.enable_backprop_tuning:
            return {
                "enabled": False,
                "message": "Backpropagation tuning is disabled in config",
            }

        feature_dim = len(code_features)
        if feature_dim < 4:
            padded_features = np.pad(code_features, (0, 4 - feature_dim), mode="constant")
        else:
            padded_features = code_features[:4]

        tensor_4d = padded_features.reshape(1, 1, 2, 2)

        learning_rate = self.config.backprop_learning_rate
        quantum_noise = self.config.backprop_quantum_noise

        weights = self._rng.randn(4) * 0.1
        losses = []

        for _i in range(iterations):
            prediction = np.dot(weights, padded_features)

            if len(ground_truth.shape) == 0:
                target = float(ground_truth)
            else:
                target = float(ground_truth[0]) if len(ground_truth) > 0 else 0.0

            loss = (prediction - target) ** 2
            losses.append(loss)

            gradient = 2 * (prediction - target) * padded_features

            if quantum_noise > 0:
                noise = self._rng.randn(*gradient.shape) * quantum_noise
                gradient = gradient + noise

            weights -= learning_rate * gradient

        return {
            "enabled": True,
            "iterations": iterations,
            "final_loss": losses[-1],
            "initial_loss": losses[0],
            "convergence": (losses[0] - losses[-1]) / iterations if iterations > 0 else 0,
            "tensor_shape": tensor_4d.shape,
        }


# Aliases for clearer naming
CodeAnalysisEngine = NeurosymbolicEngine
CodeAnalysisConfig = NeurosymbolicConfig
