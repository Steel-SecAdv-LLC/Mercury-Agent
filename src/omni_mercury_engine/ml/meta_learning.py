"""
Mercury Agent ♱
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
Meta-Learning Adapter for Mercury Agent.

Implements fast adaptation to new anomaly types with few examples, inspired by:
- "Model-Agnostic Meta-Learning (MAML)" (Finn et al., 2017)
- "Prototypical Networks for Few-Shot Learning" (Snell et al., 2017)
- "Meta-Learning with Differentiable Convex Optimization" (Lee et al., 2019)
- "Learning to Learn with Compound HD Models" (Zaheer et al., 2017)

Meta-learning enables Mercury Agent to:
1. Quickly adapt to new anomaly types with few examples
2. Transfer knowledge across different domains
3. Maintain robust performance on seen anomaly types
4. Handle distribution shift gracefully

Key Concepts:
- Inner loop: Fast adaptation to specific task
- Outer loop: Learning to learn (meta-optimization)
- Task: A specific anomaly detection scenario
- Support set: Examples for adaptation
- Query set: Examples for evaluation
"""

import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


try:
    import torch
    import torch.nn.functional as F
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None


logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

PHI = (1 + np.sqrt(5)) / 2  # Golden ratio

# Meta-learning parameters
DEFAULT_INNER_LR = 0.01
DEFAULT_OUTER_LR = 0.001
DEFAULT_INNER_STEPS = 5
DEFAULT_N_WAY = 5
DEFAULT_K_SHOT = 5


class MetaLearningAlgorithm(Enum):
    """Available meta-learning algorithms."""

    MAML = "maml"  # Model-Agnostic Meta-Learning
    PROTOTYPICAL = "prototypical"  # Prototypical Networks
    REPTILE = "reptile"  # Simplified MAML variant
    MATCHING = "matching"  # Matching Networks
    RELATION = "relation"  # Relation Networks


class AdaptationStrategy(Enum):
    """Strategies for task adaptation."""

    FINE_TUNE = "fine_tune"  # Standard fine-tuning
    FEATURE_REUSE = "feature_reuse"  # Only adapt final layers
    PROTOTYPE = "prototype"  # Use prototype-based classification
    NEAREST = "nearest"  # Nearest neighbor in embedding space


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class Task:
    """A meta-learning task (episode).

    Represents a specific anomaly detection scenario.

    Attributes:
        task_id: Unique identifier
        task_name: Human-readable name
        support_set: Examples for adaptation
        query_set: Examples for evaluation
        n_way: Number of classes
        k_shot: Examples per class in support
        domain: Task domain
        metadata: Additional metadata
    """

    task_id: str
    task_name: str
    support_set: list[tuple[np.ndarray, int]]  # (features, label) pairs
    query_set: list[tuple[np.ndarray, int]]
    n_way: int = 2  # Normal vs anomaly
    k_shot: int = 5
    domain: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_support_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Get support set as arrays."""
        X = np.array([x for x, _ in self.support_set])
        y = np.array([y for _, y in self.support_set])
        return X, y

    def get_query_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Get query set as arrays."""
        X = np.array([x for x, _ in self.query_set])
        y = np.array([y for _, y in self.query_set])
        return X, y


@dataclass
class AdaptationResult:
    """Result of task adaptation.

    Attributes:
        task_id: Task that was adapted to
        pre_adaptation_loss: Loss before adaptation
        post_adaptation_loss: Loss after adaptation
        pre_adaptation_accuracy: Accuracy before
        post_adaptation_accuracy: Accuracy after
        num_steps: Adaptation steps taken
        adaptation_time_ms: Time for adaptation
        adapted_parameters: Number of parameters adapted
    """

    task_id: str
    pre_adaptation_loss: float
    post_adaptation_loss: float
    pre_adaptation_accuracy: float
    post_adaptation_accuracy: float
    num_steps: int
    adaptation_time_ms: float
    adapted_parameters: int = 0


@dataclass
class MetaTrainingResult:
    """Result of meta-training.

    Attributes:
        epoch: Training epoch
        meta_train_loss: Meta-training loss
        meta_val_loss: Meta-validation loss
        meta_train_acc: Meta-training accuracy
        meta_val_acc: Meta-validation accuracy
        tasks_seen: Number of tasks processed
    """

    epoch: int
    meta_train_loss: float
    meta_val_loss: float
    meta_train_acc: float
    meta_val_acc: float
    tasks_seen: int


@dataclass
class Prototype:
    """A class prototype for prototypical networks.

    Attributes:
        class_id: Class identifier
        embedding: Prototype embedding vector
        support_count: Number of support examples
        variance: Embedding variance
    """

    class_id: int
    embedding: np.ndarray
    support_count: int
    variance: float = 0.0


# =============================================================================
# Base Encoder
# =============================================================================


class FeatureEncoder(ABC):
    """Abstract base class for feature encoders."""

    @abstractmethod
    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode features to embedding space."""
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        pass


class MLPEncoder(FeatureEncoder):
    """Simple MLP-based feature encoder.

    Works without PyTorch for basic scenarios.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        embedding_dim: int = 64,
    ):
        """Initialize MLP encoder.

        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions
            embedding_dim: Output embedding dimension
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [128, 64]
        self.embedding_dim = embedding_dim

        # Initialize weights
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

        dims = [input_dim] + self.hidden_dims + [embedding_dim]
        np.random.seed(42)
        for i in range(len(dims) - 1):
            # Xavier initialization
            scale = np.sqrt(2.0 / (dims[i] + dims[i + 1]))
            self.weights.append(np.random.randn(dims[i], dims[i + 1]) * scale)
            self.biases.append(np.zeros(dims[i + 1]))

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode features to embedding space."""
        h = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ w + b
            # ReLU for all but last layer
            if i < len(self.weights) - 1:
                h = np.maximum(0, h)
        # L2 normalize embeddings
        norm = np.linalg.norm(h, axis=-1, keepdims=True) + 1e-8
        return h / norm

    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim


if TORCH_AVAILABLE:

    class TorchEncoder(nn.Module, FeatureEncoder):
        """PyTorch-based feature encoder for meta-learning."""

        def __init__(
            self,
            input_dim: int,
            hidden_dims: list[int] | None = None,
            embedding_dim: int = 64,
            dropout: float = 0.1,
        ):
            """Initialize PyTorch encoder."""
            nn.Module.__init__(self)

            self.input_dim = input_dim
            self.hidden_dims = hidden_dims or [128, 64]
            self.embedding_dim = embedding_dim

            layers = []
            dims = [input_dim] + self.hidden_dims

            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

            layers.append(nn.Linear(dims[-1], embedding_dim))

            self.network = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass."""
            embeddings = self.network(x)
            # L2 normalize
            return F.normalize(embeddings, p=2, dim=-1)

        def encode(self, x: np.ndarray) -> np.ndarray:
            """Encode numpy array."""
            self.eval()
            with torch.no_grad():
                x_tensor = torch.FloatTensor(x)
                embeddings = self.forward(x_tensor)
                return embeddings.numpy()

        def get_embedding_dim(self) -> int:
            """Get embedding dimension."""
            return self.embedding_dim


# =============================================================================
# Prototypical Networks
# =============================================================================


class PrototypicalNetworks:
    """
    Prototypical Networks for few-shot learning.

    Creates class prototypes from support examples and classifies
    query examples by distance to prototypes.

    Pros:
    - Simple and effective
    - No inner loop optimization
    - Fast inference

    Cons:
    - Fixed distance metric
    - May not handle complex decision boundaries
    """

    def __init__(
        self,
        encoder: FeatureEncoder,
        distance_metric: str = "euclidean",
        temperature: float = 1.0,
    ):
        """Initialize Prototypical Networks.

        Args:
            encoder: Feature encoder
            distance_metric: Distance metric (euclidean, cosine)
            temperature: Temperature for softmax
        """
        self.encoder = encoder
        self.distance_metric = distance_metric
        self.temperature = temperature

        self.prototypes: dict[int, Prototype] = {}

    def compute_prototypes(
        self,
        support_set: list[tuple[np.ndarray, int]],
    ) -> dict[int, Prototype]:
        """Compute class prototypes from support set.

        Args:
            support_set: List of (features, label) pairs

        Returns:
            Dictionary of class prototypes
        """
        # Group by class
        class_embeddings: dict[int, list[np.ndarray]] = defaultdict(list)

        for features, label in support_set:
            embedding = self.encoder.encode(features.reshape(1, -1))[0]
            class_embeddings[label].append(embedding)

        # Compute prototypes
        prototypes = {}
        for class_id, embeddings in class_embeddings.items():
            embeddings_array = np.array(embeddings)
            mean_embedding = embeddings_array.mean(axis=0)
            variance = embeddings_array.var()

            prototypes[class_id] = Prototype(
                class_id=class_id,
                embedding=mean_embedding,
                support_count=len(embeddings),
                variance=float(variance),
            )

        self.prototypes = prototypes
        return prototypes

    def predict(
        self,
        query_features: np.ndarray,
    ) -> tuple[int, np.ndarray]:
        """Predict class for query features.

        Args:
            query_features: Query feature vector

        Returns:
            Tuple of (predicted_class, class_probabilities)
        """
        if not self.prototypes:
            raise ValueError("No prototypes computed. Call compute_prototypes first.")

        # Encode query
        query_embedding = self.encoder.encode(query_features.reshape(1, -1))[0]

        # Compute distances to all prototypes
        distances = {}
        for class_id, prototype in self.prototypes.items():
            if self.distance_metric == "euclidean":
                dist = np.linalg.norm(query_embedding - prototype.embedding)
            elif self.distance_metric == "cosine":
                dist = 1 - np.dot(query_embedding, prototype.embedding)
            else:
                dist = np.linalg.norm(query_embedding - prototype.embedding)
            distances[class_id] = dist

        # Convert to probabilities (softmax over negative distances)
        class_ids = sorted(distances.keys())
        neg_distances = np.array([-distances[c] / self.temperature for c in class_ids])
        exp_distances = np.exp(neg_distances - neg_distances.max())
        probabilities = exp_distances / exp_distances.sum()

        # Predict class with highest probability
        best_idx = np.argmax(probabilities)
        predicted_class = class_ids[best_idx]

        # Full probability array
        prob_array = np.zeros(max(class_ids) + 1)
        for i, c in enumerate(class_ids):
            prob_array[c] = probabilities[i]

        return predicted_class, prob_array

    def adapt(self, task: Task) -> AdaptationResult:
        """Adapt to a task by computing prototypes.

        Args:
            task: Task to adapt to

        Returns:
            Adaptation result
        """
        start_time = time.time()

        # Compute prototypes
        self.compute_prototypes(task.support_set)

        # Evaluate on query set
        X_query, y_query = task.get_query_arrays()

        correct = 0
        total_loss = 0.0

        for features, label in zip(X_query, y_query):
            pred, probs = self.predict(features)
            if pred == label:
                correct += 1
            # Cross-entropy loss
            total_loss += -np.log(probs[label] + 1e-8)

        accuracy = correct / len(y_query)
        avg_loss = total_loss / len(y_query)

        return AdaptationResult(
            task_id=task.task_id,
            pre_adaptation_loss=avg_loss,  # Same since no inner loop
            post_adaptation_loss=avg_loss,
            pre_adaptation_accuracy=accuracy,
            post_adaptation_accuracy=accuracy,
            num_steps=1,  # Single prototype computation
            adaptation_time_ms=(time.time() - start_time) * 1000,
        )


# =============================================================================
# MAML (Model-Agnostic Meta-Learning)
# =============================================================================


class MAML:
    """
    Model-Agnostic Meta-Learning implementation.

    MAML learns an initialization that can quickly adapt to new tasks
    through a few gradient steps.

    Algorithm:
    1. Sample batch of tasks
    2. For each task:
       a. Clone model parameters
       b. Take k gradient steps on support set
       c. Evaluate adapted model on query set
    3. Update original parameters based on query set performance

    This is a NumPy-based simplified implementation.
    For full MAML with autograd, use the PyTorch version.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 2,
        inner_lr: float = DEFAULT_INNER_LR,
        outer_lr: float = DEFAULT_OUTER_LR,
        inner_steps: int = DEFAULT_INNER_STEPS,
    ):
        """Initialize MAML.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (number of classes)
            inner_lr: Learning rate for inner loop
            outer_lr: Learning rate for outer loop
            inner_steps: Number of inner loop steps
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps

        # Initialize parameters
        self._init_parameters()

        # Statistics
        self._stats = {
            "tasks_adapted": 0,
            "meta_updates": 0,
            "avg_adaptation_improvement": 0.0,
        }

    def _init_parameters(self) -> None:
        """Initialize model parameters."""
        np.random.seed(42)

        # Two-layer network
        scale1 = np.sqrt(2.0 / (self.input_dim + self.hidden_dim))
        scale2 = np.sqrt(2.0 / (self.hidden_dim + self.output_dim))

        self.params = {
            "W1": np.random.randn(self.input_dim, self.hidden_dim) * scale1,
            "b1": np.zeros(self.hidden_dim),
            "W2": np.random.randn(self.hidden_dim, self.output_dim) * scale2,
            "b2": np.zeros(self.output_dim),
        }

    def _forward(
        self,
        X: np.ndarray,
        params: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Forward pass through network."""
        # Hidden layer
        h = X @ params["W1"] + params["b1"]
        h = np.maximum(0, h)  # ReLU

        # Output layer
        logits = h @ params["W2"] + params["b2"]

        # Softmax
        exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

        return probs

    def _compute_loss(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: dict[str, np.ndarray],
    ) -> tuple[float, np.ndarray]:
        """Compute cross-entropy loss and predictions."""
        probs = self._forward(X, params)
        n_samples = len(y)

        # Cross-entropy loss
        loss = -np.log(probs[range(n_samples), y] + 1e-8).mean()

        return float(loss), probs

    def _compute_gradients(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Compute gradients via manual backprop."""
        n_samples = len(y)

        # Forward pass
        h1 = X @ params["W1"] + params["b1"]
        h1_relu = np.maximum(0, h1)
        logits = h1_relu @ params["W2"] + params["b2"]

        # Softmax
        exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

        # Backward pass
        # dL/dlogits = probs - one_hot(y)
        dlogits = probs.copy()
        dlogits[range(n_samples), y] -= 1
        dlogits /= n_samples

        # dL/dW2 = h1_relu.T @ dlogits
        dW2 = h1_relu.T @ dlogits
        db2 = dlogits.sum(axis=0)

        # dL/dh1_relu = dlogits @ W2.T
        dh1_relu = dlogits @ params["W2"].T

        # ReLU backward
        dh1 = dh1_relu * (h1 > 0)

        # dL/dW1 = X.T @ dh1
        dW1 = X.T @ dh1
        db1 = dh1.sum(axis=0)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def _inner_loop(
        self,
        task: Task,
        params: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Perform inner loop adaptation.

        Args:
            task: Task to adapt to
            params: Starting parameters

        Returns:
            Adapted parameters
        """
        X_support, y_support = task.get_support_arrays()
        adapted_params = {k: v.copy() for k, v in params.items()}

        for _ in range(self.inner_steps):
            grads = self._compute_gradients(X_support, y_support, adapted_params)

            # SGD update
            for key in adapted_params:
                adapted_params[key] = adapted_params[key] - self.inner_lr * grads[key]

        return adapted_params

    def adapt(self, task: Task) -> AdaptationResult:
        """Adapt to a single task.

        Args:
            task: Task to adapt to

        Returns:
            Adaptation result
        """
        start_time = time.time()

        X_query, y_query = task.get_query_arrays()

        # Pre-adaptation evaluation
        pre_loss, pre_probs = self._compute_loss(X_query, y_query, self.params)
        pre_preds = pre_probs.argmax(axis=1)
        pre_acc = (pre_preds == y_query).mean()

        # Inner loop adaptation
        adapted_params = self._inner_loop(task, self.params)

        # Post-adaptation evaluation
        post_loss, post_probs = self._compute_loss(X_query, y_query, adapted_params)
        post_preds = post_probs.argmax(axis=1)
        post_acc = (post_preds == y_query).mean()

        self._stats["tasks_adapted"] += 1
        improvement = post_acc - pre_acc
        n = self._stats["tasks_adapted"]
        self._stats["avg_adaptation_improvement"] = (
            self._stats["avg_adaptation_improvement"] * (n - 1) + improvement
        ) / n

        return AdaptationResult(
            task_id=task.task_id,
            pre_adaptation_loss=pre_loss,
            post_adaptation_loss=post_loss,
            pre_adaptation_accuracy=float(pre_acc),
            post_adaptation_accuracy=float(post_acc),
            num_steps=self.inner_steps,
            adaptation_time_ms=(time.time() - start_time) * 1000,
            adapted_parameters=sum(p.size for p in self.params.values()),
        )

    def meta_train_step(
        self,
        task_batch: list[Task],
    ) -> float:
        """Perform one meta-training step.

        Args:
            task_batch: Batch of tasks

        Returns:
            Average meta-loss
        """
        meta_gradients: dict[str, np.ndarray] = {
            k: np.zeros_like(v) for k, v in self.params.items()
        }
        total_loss = 0.0

        for task in task_batch:
            # Adapt to task
            adapted_params = self._inner_loop(task, self.params)

            # Compute loss on query set with adapted params
            X_query, y_query = task.get_query_arrays()
            loss, _ = self._compute_loss(X_query, y_query, adapted_params)
            total_loss += loss

            # Compute gradients for meta-update
            # (Simplified: use gradients from adapted model)
            grads = self._compute_gradients(X_query, y_query, adapted_params)
            for key in meta_gradients:
                meta_gradients[key] += grads[key]

        # Average gradients
        for key in meta_gradients:
            meta_gradients[key] /= len(task_batch)

        # Meta-update
        for key in self.params:
            self.params[key] = self.params[key] - self.outer_lr * meta_gradients[key]

        self._stats["meta_updates"] += 1

        return total_loss / len(task_batch)

    def get_statistics(self) -> dict[str, Any]:
        """Get MAML statistics."""
        return {
            **self._stats,
            "inner_lr": self.inner_lr,
            "outer_lr": self.outer_lr,
            "inner_steps": self.inner_steps,
        }


# =============================================================================
# Reptile (Simplified MAML)
# =============================================================================


class Reptile:
    """
    Reptile meta-learning algorithm.

    Simplified version of MAML that doesn't require second-order
    gradients. Updates by interpolating towards task-adapted parameters.

    Algorithm:
    1. Sample task
    2. Take k gradient steps on task
    3. Update meta-parameters towards adapted parameters
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 2,
        inner_lr: float = DEFAULT_INNER_LR,
        outer_lr: float = DEFAULT_OUTER_LR,
        inner_steps: int = DEFAULT_INNER_STEPS,
    ):
        """Initialize Reptile."""
        self.maml = MAML(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            inner_lr=inner_lr,
            outer_lr=outer_lr,
            inner_steps=inner_steps,
        )

    def adapt(self, task: Task) -> AdaptationResult:
        """Adapt to task."""
        return self.maml.adapt(task)

    def meta_train_step(self, task_batch: list[Task]) -> float:
        """Perform Reptile meta-training step."""
        total_loss = 0.0

        for task in task_batch:
            # Adapt to task
            adapted_params = self.maml._inner_loop(task, self.maml.params)

            # Reptile update: interpolate towards adapted params
            for key in self.maml.params:
                self.maml.params[key] = self.maml.params[key] + self.maml.outer_lr * (
                    adapted_params[key] - self.maml.params[key]
                )

            # Compute loss for monitoring
            X_query, y_query = task.get_query_arrays()
            loss, _ = self.maml._compute_loss(X_query, y_query, adapted_params)
            total_loss += loss

        return total_loss / len(task_batch)


# =============================================================================
# Meta-Learning Adapter (Main Interface)
# =============================================================================


class MetaLearningAdapter:
    """
    Main Meta-Learning Adapter for Mercury Agent.

    Provides unified interface to different meta-learning algorithms
    for fast adaptation to new anomaly types.

    Key capabilities:
    1. Few-shot learning for new anomaly types
    2. Rapid adaptation with minimal examples
    3. Transfer learning across domains
    4. Online task adaptation
    """

    def __init__(
        self,
        input_dim: int,
        algorithm: MetaLearningAlgorithm = MetaLearningAlgorithm.PROTOTYPICAL,
        hidden_dim: int = 64,
        embedding_dim: int = 64,
        n_way: int = DEFAULT_N_WAY,
        k_shot: int = DEFAULT_K_SHOT,
    ):
        """Initialize Meta-Learning Adapter.

        Args:
            input_dim: Input feature dimension
            algorithm: Meta-learning algorithm to use
            hidden_dim: Hidden layer dimension
            embedding_dim: Embedding dimension
            n_way: Default number of classes per task
            k_shot: Default number of examples per class
        """
        self.input_dim = input_dim
        self.algorithm = algorithm
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.n_way = n_way
        self.k_shot = k_shot

        # Initialize algorithm
        self._init_algorithm()

        # Task counter
        self._task_counter = 0

        # Statistics
        self._stats = {
            "tasks_created": 0,
            "adaptations_performed": 0,
            "avg_post_adaptation_acc": 0.0,
        }

        logger.info(
            f"MetaLearningAdapter initialized (algorithm={algorithm.value}, "
            f"input_dim={input_dim})"
        )

    def _init_algorithm(self) -> None:
        """Initialize meta-learning algorithm."""
        if self.algorithm == MetaLearningAlgorithm.PROTOTYPICAL:
            encoder = MLPEncoder(
                input_dim=self.input_dim,
                hidden_dims=[self.hidden_dim],
                embedding_dim=self.embedding_dim,
            )
            self._learner = PrototypicalNetworks(encoder)

        elif self.algorithm == MetaLearningAlgorithm.MAML:
            self._learner = MAML(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.n_way,
            )

        elif self.algorithm == MetaLearningAlgorithm.REPTILE:
            self._learner = Reptile(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.n_way,
            )

        else:
            # Default to prototypical
            encoder = MLPEncoder(
                input_dim=self.input_dim,
                hidden_dims=[self.hidden_dim],
                embedding_dim=self.embedding_dim,
            )
            self._learner = PrototypicalNetworks(encoder)

    def create_task(
        self,
        support_data: list[tuple[np.ndarray, int]],
        query_data: list[tuple[np.ndarray, int]],
        task_name: str | None = None,
        domain: str = "general",
    ) -> Task:
        """Create a meta-learning task.

        Args:
            support_data: Support set examples
            query_data: Query set examples
            task_name: Optional task name
            domain: Task domain

        Returns:
            Created Task object
        """
        self._task_counter += 1
        task_id = f"task_{self._task_counter:06d}"
        task_name = task_name or f"Anomaly Detection Task {self._task_counter}"

        # Determine n_way from data
        support_labels = {y for _, y in support_data}
        n_way = len(support_labels)

        # Determine k_shot
        label_counts = defaultdict(int)
        for _, y in support_data:
            label_counts[y] += 1
        k_shot = min(label_counts.values()) if label_counts else self.k_shot

        task = Task(
            task_id=task_id,
            task_name=task_name,
            support_set=support_data,
            query_set=query_data,
            n_way=n_way,
            k_shot=k_shot,
            domain=domain,
        )

        self._stats["tasks_created"] += 1
        return task

    def adapt(self, task: Task) -> AdaptationResult:
        """Adapt to a specific task.

        Args:
            task: Task to adapt to

        Returns:
            Adaptation result
        """
        result = self._learner.adapt(task)

        # Update statistics
        self._stats["adaptations_performed"] += 1
        n = self._stats["adaptations_performed"]
        self._stats["avg_post_adaptation_acc"] = (
            self._stats["avg_post_adaptation_acc"] * (n - 1) + result.post_adaptation_accuracy
        ) / n

        return result

    def predict(
        self,
        features: np.ndarray,
        task: Task | None = None,
    ) -> tuple[int, np.ndarray]:
        """Make prediction for features.

        Args:
            features: Input features
            task: Optional task context (for adaptation)

        Returns:
            Tuple of (predicted_class, probabilities)
        """
        if task is not None:
            # Adapt to task first
            self.adapt(task)

        if isinstance(self._learner, PrototypicalNetworks):
            return self._learner.predict(features)
        elif isinstance(self._learner, (MAML, Reptile)):
            # Use current model
            probs = self._learner.maml._forward(
                features.reshape(1, -1),
                (
                    self._learner.maml.params
                    if isinstance(self._learner, Reptile)
                    else self._learner.params
                ),
            )[0]
            return int(probs.argmax()), probs
        else:
            return 0, np.array([0.5, 0.5])

    def few_shot_detect(
        self,
        support_examples: list[tuple[np.ndarray, bool]],
        query_features: np.ndarray,
    ) -> tuple[bool, float]:
        """Few-shot anomaly detection.

        Args:
            support_examples: Support examples (features, is_anomaly)
            query_features: Query features to classify

        Returns:
            Tuple of (is_anomaly, confidence)
        """
        # Convert to task format
        support_data = [(x, int(label)) for x, label in support_examples]
        query_data = [(query_features, 0)]  # Dummy label

        task = self.create_task(
            support_data=support_data,
            query_data=query_data,
            task_name="Few-shot Detection",
        )

        # Adapt and predict
        self.adapt(task)
        pred, probs = self.predict(query_features)

        is_anomaly = pred == 1
        confidence = probs[pred] if pred < len(probs) else 0.5

        return is_anomaly, float(confidence)

    def get_statistics(self) -> dict[str, Any]:
        """Get adapter statistics."""
        learner_stats = {}
        if hasattr(self._learner, "get_statistics"):
            learner_stats = self._learner.get_statistics()

        return {
            **self._stats,
            "algorithm": self.algorithm.value,
            "input_dim": self.input_dim,
            "learner": learner_stats,
        }


# =============================================================================
# Anomaly Detection Integration
# =============================================================================


class AnomalyMetaLearner:
    """
    Meta-learner specialized for anomaly detection.

    Provides domain-specific adaptation for Mercury Agent's
    anomaly detection tasks.
    """

    def __init__(
        self,
        adapter: MetaLearningAdapter | None = None,
        input_dim: int = 64,
        anomaly_threshold: float = 0.5,
    ):
        """Initialize anomaly meta-learner.

        Args:
            adapter: Base meta-learning adapter
            input_dim: Input feature dimension
            anomaly_threshold: Threshold for anomaly classification
        """
        self.adapter = adapter or MetaLearningAdapter(
            input_dim=input_dim,
            algorithm=MetaLearningAlgorithm.PROTOTYPICAL,
        )
        self.anomaly_threshold = anomaly_threshold

        # Cache for recent adaptations
        self._adaptation_cache: dict[str, AdaptationResult] = {}

    def adapt_to_new_anomaly_type(
        self,
        normal_examples: list[np.ndarray],
        anomaly_examples: list[np.ndarray],
        anomaly_type: str = "unknown",
    ) -> AdaptationResult:
        """Adapt to a new type of anomaly.

        Args:
            normal_examples: Examples of normal behavior
            anomaly_examples: Examples of the new anomaly type
            anomaly_type: Name of the anomaly type

        Returns:
            Adaptation result
        """
        # Create support set
        support_data = [(x, 0) for x in normal_examples]  # Normal = 0
        support_data.extend((x, 1) for x in anomaly_examples)  # Anomaly = 1

        # Create query set (use subset of support for evaluation)
        query_data = support_data[: len(support_data) // 2]

        task = self.adapter.create_task(
            support_data=support_data,
            query_data=query_data,
            task_name=f"Adapt to {anomaly_type}",
            domain="anomaly_detection",
        )

        result = self.adapter.adapt(task)
        self._adaptation_cache[anomaly_type] = result

        return result

    def detect_with_adaptation(
        self,
        features: np.ndarray,
        reference_normal: list[np.ndarray],
        reference_anomaly: list[np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Detect anomaly with few-shot adaptation.

        Args:
            features: Features to classify
            reference_normal: Reference normal examples
            reference_anomaly: Reference anomaly examples (optional)

        Returns:
            Detection result
        """
        if reference_anomaly is None:
            reference_anomaly = []

        # Create support examples
        support_examples = [(x, False) for x in reference_normal]
        support_examples.extend((x, True) for x in reference_anomaly)

        if len(support_examples) < 2:
            # Not enough examples for few-shot
            return {
                "is_anomaly": False,
                "confidence": 0.5,
                "method": "insufficient_data",
            }

        is_anomaly, confidence = self.adapter.few_shot_detect(
            support_examples=support_examples,
            query_features=features,
        )

        return {
            "is_anomaly": is_anomaly,
            "confidence": confidence,
            "method": self.adapter.algorithm.value,
            "support_size": len(support_examples),
        }

    def get_adaptation_history(self) -> dict[str, AdaptationResult]:
        """Get history of adaptations."""
        return self._adaptation_cache
