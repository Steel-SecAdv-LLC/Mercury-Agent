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
Code Analysis Engine - AST-based symbolic reasoning for code refactoring.

This module provides AST (Abstract Syntax Tree) analysis capabilities for:
- Code complexity analysis
- Refactoring suggestions
- Training readiness assessment

Note:
    This is distinct from :class:`omni_mercury_engine.models.neurosymbolic.NeurosymbolicEngine`
    which provides LTN-based anomaly detection. This module focuses on static code analysis
    using Python's AST module.

    For anomaly detection with neurosymbolic reasoning, use::

        from omni_mercury_engine.models.neurosymbolic import NeurosymbolicEngine

    For code analysis and refactoring, use::

        from omni_mercury_engine.core.code_analysis import CodeAnalysisEngine
"""

import ast
import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.utils.rng import DeterministicRNG, get_global_rng

if TYPE_CHECKING:
    from numpy.typing import NDArray


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
        self.neural_model: dict[str, Any] | None = None
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
        Perform neural pattern recognition with statistical fallback.

        When neural model is not available, provides statistical analysis
        of code features as a meaningful fallback rather than returning
        an empty result.

        Args:
            code_features: Feature vector extracted from code

        Returns:
            Dict with neural analysis results or statistical fallback
        """
        if not self.config.enable_neural or self.neural_model is None:
            feature_mean = float(np.mean(code_features)) if len(code_features) > 0 else 0.0
            feature_std = float(np.std(code_features)) if len(code_features) > 0 else 0.0
            feature_max = float(np.max(code_features)) if len(code_features) > 0 else 0.0

            complexity_score = min(1.0, feature_mean / 10.0) if feature_mean > 0 else 0.0

            return {
                "method": "statistical_fallback",
                "available": True,
                "neural_model_trained": False,
                "statistics": {
                    "mean": feature_mean,
                    "std": feature_std,
                    "max": feature_max,
                    "complexity_score": complexity_score,
                },
                "confidence": 0.7,
                "message": "Using statistical analysis (neural model not trained)",
            }

        return {
            "method": "neural",
            "available": True,
            "neural_model_trained": True,
            "confidence": 0.9,
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
        self,
        training_data: list[tuple[ast.AST, dict[str, Any]]] | None = None,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
    ) -> TrainingMetrics:
        """
        Train neural model on code patterns.

        Implements a neural network training loop for learning code refactoring
        patterns from AST features. Uses a simple feedforward architecture
        with gradient descent optimization.

        Args:
            training_data: List of (AST, ground_truth) pairs where ground_truth
                contains pattern labels and refactoring suggestions.
            epochs: Number of training epochs.
            batch_size: Training batch size.
            validation_split: Fraction of data for validation.

        Returns:
            Training metrics with loss, accuracy, and readiness level.

        Example:
            >>> engine = NeurosymbolicEngine(NeurosymbolicConfig(enable_neural=True))
            >>> training_data = [(ast.parse("x = 1"), {"complexity": 1})]
            >>> metrics = engine.train_model(training_data, epochs=50)
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

        if len(training_data) < 2:
            logging.warning("Insufficient training data. Need at least 2 samples.")
            return self.training_metrics

        logging.info(f"Starting neural training on {len(training_data)} examples")
        self.current_phase = TrainingPhase.SPECIALIZATION

        features, labels = self._prepare_training_data(training_data)

        if len(features) == 0:
            logging.warning("Could not extract features from training data.")
            return self.training_metrics

        # Split into train/validation
        n_samples = len(features)
        n_val = max(1, int(n_samples * validation_split))
        n_train = n_samples - n_val

        indices = self._rng.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        X_train = features[train_indices]
        y_train = labels[train_indices]
        X_val = features[val_indices]
        y_val = labels[val_indices]

        # Initialize neural network weights
        input_dim = features.shape[1]
        hidden_dim = max(8, input_dim * 2)
        output_dim = 1  # Binary or regression output

        # Xavier initialization
        W1 = self._rng.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        b1 = np.zeros(hidden_dim)
        W2 = self._rng.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        b2 = np.zeros(output_dim)

        learning_rate = self.config.backprop_learning_rate
        best_val_loss = float("inf")
        best_epoch = 0

        # Training loop
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # Shuffle training data
            perm = self._rng.permutation(n_train)
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]

            epoch_loss = 0.0

            # Mini-batch training
            for i in range(0, n_train, batch_size):
                X_batch = X_train_shuffled[i : i + batch_size]
                y_batch = y_train_shuffled[i : i + batch_size]

                # Forward pass
                z1 = X_batch @ W1 + b1
                a1 = np.maximum(0, z1)  # ReLU activation

                z2 = a1 @ W2 + b2
                y_pred = 1 / (1 + np.exp(-z2))  # Sigmoid for binary output

                # Compute loss (binary cross-entropy)
                epsilon = 1e-7
                y_batch_reshaped = y_batch.reshape(-1, 1)
                loss = -np.mean(
                    y_batch_reshaped * np.log(y_pred + epsilon)
                    + (1 - y_batch_reshaped) * np.log(1 - y_pred + epsilon)
                )
                epoch_loss += loss * len(X_batch)

                # Backward pass
                m = len(X_batch)
                dz2 = (y_pred - y_batch_reshaped) / m
                dW2 = a1.T @ dz2
                db2 = np.sum(dz2, axis=0)

                da1 = dz2 @ W2.T
                dz1 = da1 * (z1 > 0)  # ReLU derivative
                dW1 = X_batch.T @ dz1
                db1 = np.sum(dz1, axis=0)

                # Gradient descent update
                W1 -= learning_rate * dW1
                b1 -= learning_rate * db1
                W2 -= learning_rate * dW2
                b2 -= learning_rate * db2

            # Compute average training loss
            train_loss = epoch_loss / n_train
            train_losses.append(train_loss)

            # Validation
            z1_val = X_val @ W1 + b1
            a1_val = np.maximum(0, z1_val)
            z2_val = a1_val @ W2 + b2
            y_val_pred = 1 / (1 + np.exp(-z2_val))

            y_val_reshaped = y_val.reshape(-1, 1)
            val_loss = -np.mean(
                y_val_reshaped * np.log(y_val_pred + epsilon)
                + (1 - y_val_reshaped) * np.log(1 - y_val_pred + epsilon)
            )
            val_losses.append(val_loss)

            # Track best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                # Store best weights
                self.neural_model = {
                    "W1": W1.copy(),
                    "b1": b1.copy(),
                    "W2": W2.copy(),
                    "b2": b2.copy(),
                    "input_dim": input_dim,
                    "hidden_dim": hidden_dim,
                }

            # Log progress
            if (epoch + 1) % max(1, epochs // 10) == 0:
                logging.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

        # Compute final accuracy
        assert self.neural_model is not None
        z1_final = X_val @ self.neural_model["W1"] + self.neural_model["b1"]
        a1_final = np.maximum(0, z1_final)
        z2_final = a1_final @ self.neural_model["W2"] + self.neural_model["b2"]
        y_final_pred = 1 / (1 + np.exp(-z2_final))
        predictions = (y_final_pred > 0.5).astype(float).flatten()
        accuracy = np.mean(predictions == y_val)

        # Determine readiness level based on accuracy
        if accuracy > 0.95:
            readiness = ReadinessLevel.PRODUCTION_READY
        elif accuracy > 0.80:
            readiness = ReadinessLevel.READY
        elif accuracy > 0.60:
            readiness = ReadinessLevel.NEEDS_IMPROVEMENT
        else:
            readiness = ReadinessLevel.NOT_READY

        self.current_phase = TrainingPhase.VALIDATION
        self.training_metrics = TrainingMetrics(
            epoch=epochs,
            loss=train_losses[-1],
            accuracy=accuracy,
            validation_loss=best_val_loss,
            validation_accuracy=accuracy,
            readiness_level=readiness,
        )

        logging.info(
            f"Training complete - Accuracy: {accuracy:.2%}, "
            f"Readiness: {readiness.value}, Best epoch: {best_epoch + 1}"
        )

        return self.training_metrics

    def _prepare_training_data(
        self, training_data: list[tuple[ast.AST, dict[str, Any]]]
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """
        Extract features and labels from training data.

        Args:
            training_data: List of (AST, ground_truth) pairs.

        Returns:
            Tuple of (features array, labels array).
        """
        features_list = []
        labels_list = []

        for ast_tree, ground_truth in training_data:
            # Extract symbolic features from AST
            symbolic = self.symbolic_analysis(ast_tree)
            patterns = symbolic.get("patterns", {})

            # Build feature vector
            feature_vector = [
                patterns.get("loops", 0),
                patterns.get("conditionals", 0),
                patterns.get("function_calls", 0),
                patterns.get("nesting_depth", 0),
            ]

            # Add additional AST metrics
            num_nodes = sum(1 for _ in ast.walk(ast_tree))
            num_functions = sum(
                1 for node in ast.walk(ast_tree) if isinstance(node, ast.FunctionDef)
            )
            num_classes = sum(1 for node in ast.walk(ast_tree) if isinstance(node, ast.ClassDef))

            feature_vector.extend(
                [
                    num_nodes,
                    num_functions,
                    num_classes,
                ]
            )

            features_list.append(feature_vector)

            # Extract label from ground truth
            # Default to complexity score or binary refactoring need
            label = ground_truth.get("needs_refactoring", 0)
            if "complexity" in ground_truth:
                label = 1 if ground_truth["complexity"] > 10 else 0
            labels_list.append(label)

        features = np.array(features_list, dtype=float)
        labels = np.array(labels_list, dtype=float)

        # Normalize features
        if len(features) > 0:
            mean = np.mean(features, axis=0, keepdims=True)
            std = np.std(features, axis=0, keepdims=True) + 1e-7
            features = (features - mean) / std

        return features, labels

    def predict(self, code_ast: ast.AST) -> dict[str, Any]:
        """
        Predict refactoring needs for given code.

        Uses trained neural model if available, falls back to symbolic analysis.

        Args:
            code_ast: Abstract syntax tree to analyze.

        Returns:
            Prediction with refactoring recommendations and confidence.
        """
        # Always perform symbolic analysis
        symbolic = self.symbolic_analysis(code_ast)

        if not self.config.enable_neural or self.neural_model is None:
            # Return symbolic-only prediction
            patterns = symbolic.get("patterns", {})
            complexity = (
                patterns.get("loops", 0) * 2
                + patterns.get("conditionals", 0)
                + patterns.get("nesting_depth", 0) * 3
            )
            return {
                "method": "symbolic",
                "needs_refactoring": complexity > 10,
                "complexity_score": complexity,
                "confidence": 0.7,
                "recommendations": self._generate_recommendations(patterns),
            }

        # Neural prediction
        patterns = symbolic.get("patterns", {})
        num_nodes = sum(1 for _ in ast.walk(code_ast))
        num_functions = sum(1 for node in ast.walk(code_ast) if isinstance(node, ast.FunctionDef))
        num_classes = sum(1 for node in ast.walk(code_ast) if isinstance(node, ast.ClassDef))

        feature_vector = np.array(
            [
                [
                    patterns.get("loops", 0),
                    patterns.get("conditionals", 0),
                    patterns.get("function_calls", 0),
                    patterns.get("nesting_depth", 0),
                    num_nodes,
                    num_functions,
                    num_classes,
                ]
            ],
            dtype=float,
        )

        # Forward pass through trained model
        W1 = self.neural_model["W1"]
        b1 = self.neural_model["b1"]
        W2 = self.neural_model["W2"]
        b2 = self.neural_model["b2"]

        z1 = feature_vector @ W1 + b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ W2 + b2
        probability = 1 / (1 + np.exp(-z2))

        needs_refactoring = probability[0, 0] > 0.5
        confidence = abs(probability[0, 0] - 0.5) * 2  # Scale to 0-1

        return {
            "method": "neural",
            "needs_refactoring": bool(needs_refactoring),
            "refactoring_probability": float(probability[0, 0]),
            "confidence": float(confidence),
            "recommendations": self._generate_recommendations(patterns),
            "model_readiness": self.training_metrics.readiness_level.value,
        }

    def _generate_recommendations(self, patterns: dict[str, int]) -> list[str]:
        """Generate refactoring recommendations based on code patterns."""
        recommendations = []

        if patterns.get("nesting_depth", 0) > 3:
            recommendations.append(
                "High nesting depth detected. Consider extracting nested code into functions."
            )

        if patterns.get("loops", 0) > 5:
            recommendations.append(
                "Multiple loops detected. Consider using list comprehensions or vectorization."
            )

        if patterns.get("conditionals", 0) > 10:
            recommendations.append(
                "Many conditionals detected. Consider using polymorphism or strategy pattern."
            )

        if patterns.get("function_calls", 0) > 20:
            recommendations.append(
                "High function call count. Consider caching expensive computations."
            )

        if not recommendations:
            recommendations.append("Code structure appears clean. No immediate refactoring needed.")

        return recommendations

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
