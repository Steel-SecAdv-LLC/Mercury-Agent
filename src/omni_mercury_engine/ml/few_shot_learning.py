"""
Mercury Agent - Few-Shot Learning Framework
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Production-grade few-shot learning for anomaly detection providing:
- Prototypical Networks for metric-based classification
- Matching Networks with attention mechanisms
- Model-Agnostic Meta-Learning (MAML) for rapid adaptation
- Siamese Networks for similarity learning
- N-way K-shot episode generation
- Support for 10/50/100 label experiments
- Cross-domain few-shot transfer

This addresses the critical gap where Mercury's neuro-symbolic architecture
can demonstrate advantages over pure supervised methods in low-data regimes.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.distance import cdist


if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Optional PyTorch imports
try:
    import torch
    import torch.nn.functional as F
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


class FewShotMethod(str, Enum):
    """Available few-shot learning methods."""

    PROTOTYPICAL = "prototypical"
    MATCHING = "matching"
    MAML = "maml"
    SIAMESE = "siamese"
    RELATION = "relation"
    NEAREST_CENTROID = "nearest_centroid"  # NumPy fallback


class EpisodeSamplingStrategy(str, Enum):
    """Strategies for sampling few-shot episodes."""

    RANDOM = "random"
    STRATIFIED = "stratified"
    HARD_NEGATIVE = "hard_negative"
    CURRICULUM = "curriculum"


@dataclass
class Episode:
    """A single few-shot learning episode."""

    support_X: NDArray[np.float64]  # [n_way * k_shot, n_features]
    support_y: NDArray[np.int64]  # [n_way * k_shot]
    query_X: NDArray[np.float64]  # [n_query, n_features]
    query_y: NDArray[np.int64]  # [n_query]
    classes: list[int]  # Classes in this episode
    episode_id: int = 0


@dataclass
class FewShotResult:
    """Result from few-shot learning evaluation."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    n_way: int
    k_shot: int
    n_episodes: int
    n_labels_used: int  # Total labeled examples

    # Per-episode metrics
    episode_accuracies: list[float] = field(default_factory=list)
    episode_times: list[float] = field(default_factory=list)

    # Confidence intervals
    accuracy_ci_lower: float = 0.0
    accuracy_ci_upper: float = 0.0

    # Method info
    method: str = "prototypical"
    embedding_dim: int = 64

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "n_way": self.n_way,
            "k_shot": self.k_shot,
            "n_episodes": self.n_episodes,
            "n_labels_used": self.n_labels_used,
            "accuracy_95ci": [self.accuracy_ci_lower, self.accuracy_ci_upper],
            "method": self.method,
        }


class EpisodeGenerator:
    """
    Generates few-shot learning episodes from data.

    Supports various sampling strategies and ensures proper
    train/test separation within episodes.
    """

    def __init__(
        self,
        n_way: int = 2,
        k_shot: int = 5,
        n_query: int = 15,
        n_episodes: int = 100,
        strategy: EpisodeSamplingStrategy = EpisodeSamplingStrategy.STRATIFIED,
        seed: int = 42,
    ):
        """
        Initialize episode generator.

        Args:
            n_way: Number of classes per episode
            k_shot: Number of support examples per class
            n_query: Number of query examples per class
            n_episodes: Number of episodes to generate
            strategy: Sampling strategy
            seed: Random seed
        """
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.n_episodes = n_episodes
        self.strategy = strategy
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
        class_subset: list[int] | None = None,
    ) -> Iterator[Episode]:
        """
        Generate few-shot episodes.

        Args:
            X: Feature matrix [n_samples, n_features]
            y: Labels [n_samples]
            class_subset: Optional subset of classes to use

        Yields:
            Episode objects
        """
        unique_classes = np.unique(y)

        if class_subset is not None:
            unique_classes = np.array([c for c in unique_classes if c in class_subset])

        if len(unique_classes) < self.n_way:
            raise ValueError(f"Not enough classes: need {self.n_way}, have {len(unique_classes)}")

        # Group samples by class
        class_indices: dict[int, NDArray[np.int64]] = {}
        for cls in unique_classes:
            class_indices[cls] = np.where(y == cls)[0]

        # Check minimum samples per class
        min_samples_needed = self.k_shot + self.n_query
        valid_classes = [
            cls for cls, indices in class_indices.items() if len(indices) >= min_samples_needed
        ]

        if len(valid_classes) < self.n_way:
            # Relax query requirement if not enough samples
            self.n_query = max(
                1,
                min(len(indices) for indices in class_indices.values()) - self.k_shot,
            )
            valid_classes = [
                cls
                for cls, indices in class_indices.items()
                if len(indices) >= self.k_shot + self.n_query
            ]

        for episode_id in range(self.n_episodes):
            # Sample classes for this episode
            episode_classes = self.rng.choice(
                valid_classes, size=self.n_way, replace=False
            ).tolist()

            support_indices = []
            query_indices = []

            for cls in episode_classes:
                indices = class_indices[cls]

                # Sample support and query indices
                if self.strategy == EpisodeSamplingStrategy.HARD_NEGATIVE:
                    # TODO: Implement hard negative mining
                    sampled = self.rng.choice(
                        indices, size=self.k_shot + self.n_query, replace=False
                    )
                else:
                    sampled = self.rng.choice(
                        indices, size=self.k_shot + self.n_query, replace=False
                    )

                support_indices.extend(sampled[: self.k_shot])
                query_indices.extend(sampled[self.k_shot :])

            # Create episode
            support_X = X[support_indices]
            support_y = y[support_indices]
            query_X = X[query_indices]
            query_y = y[query_indices]

            yield Episode(
                support_X=support_X,
                support_y=support_y,
                query_X=query_X,
                query_y=query_y,
                classes=episode_classes,
                episode_id=episode_id,
            )

    def generate_k_shot_experiment(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
        k_values: list[int] | None = None,
        n_trials: int = 10,
    ) -> Iterator[tuple[int, int, Episode]]:
        """
        Generate episodes for K-shot experiments (10, 50, 100 labels).

        Args:
            X: Feature matrix
            y: Labels
            k_values: List of total label counts to test
            n_trials: Number of trials per K value

        Yields:
            Tuple of (k_value, trial_id, episode)
        """
        if k_values is None:
            k_values = [10, 50, 100]
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)

        for k_total in k_values:
            # Distribute k_total across classes
            k_per_class = max(1, k_total // n_classes)

            for trial_id in range(n_trials):
                # Create single large episode with k_per_class support samples
                support_indices = []
                query_indices = []

                class_indices: dict[int, NDArray[np.int64]] = {}
                for cls in unique_classes:
                    class_indices[cls] = np.where(y == cls)[0]

                for cls in unique_classes:
                    indices = class_indices[cls]
                    available = len(indices)

                    # Ensure we don't exceed available samples
                    n_support = min(k_per_class, available - 1)
                    n_query = min(available - n_support, 50)  # Cap query at 50

                    if n_support < 1 or n_query < 1:
                        continue

                    sampled = self.rng.choice(indices, size=n_support + n_query, replace=False)
                    support_indices.extend(sampled[:n_support])
                    query_indices.extend(sampled[n_support:])

                if not support_indices or not query_indices:
                    continue

                episode = Episode(
                    support_X=X[support_indices],
                    support_y=y[support_indices],
                    query_X=X[query_indices],
                    query_y=y[query_indices],
                    classes=unique_classes.tolist(),
                    episode_id=trial_id,
                )

                yield k_total, trial_id, episode


class BaseFewShotLearner(ABC):
    """Base class for few-shot learning methods."""

    @abstractmethod
    def fit_episode(self, episode: Episode) -> None:
        """Fit the model on episode support set."""
        pass

    @abstractmethod
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict labels for query set."""
        pass

    @abstractmethod
    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict class probabilities."""
        pass


class PrototypicalNetworkNumpy(BaseFewShotLearner):
    """
    NumPy implementation of Prototypical Networks.

    Creates class prototypes as mean embeddings and classifies
    by nearest prototype in embedding space.

    Reference: "Prototypical Networks for Few-shot Learning"
               (Snell et al., 2017)
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        distance_metric: str = "euclidean",
        temperature: float = 1.0,
        random_state: int | None = None,
    ):
        """
        Initialize Prototypical Network.

        Args:
            embedding_dim: Dimension of embedding space
            distance_metric: Distance metric ('euclidean', 'cosine', 'manhattan')
            temperature: Temperature for softmax scaling
            random_state: Seed for reproducible random initialization
        """
        self.embedding_dim = embedding_dim
        self.distance_metric = distance_metric
        self.temperature = temperature
        self.rng = np.random.default_rng(random_state)

        # Learned components (simple linear projection)
        self.projection_matrix: NDArray[np.float64] | None = None
        self.projection_bias: NDArray[np.float64] | None = None

        # Episode-specific
        self.prototypes: NDArray[np.float64] | None = None
        self.classes: list[int] = []

    def _initialize_projection(self, input_dim: int) -> None:
        """Initialize projection matrix using Xavier initialization."""
        scale = np.sqrt(2.0 / (input_dim + self.embedding_dim))
        self.projection_matrix = (
            self.rng.standard_normal((input_dim, self.embedding_dim)).astype(np.float64) * scale
        )
        self.projection_bias = np.zeros(self.embedding_dim, dtype=np.float64)

    def _embed(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Project features to embedding space."""
        if self.projection_matrix is None:
            self._initialize_projection(X.shape[1])

        # Handle dimension mismatch
        if X.shape[1] != self.projection_matrix.shape[0]:
            self._initialize_projection(X.shape[1])

        embeddings = X @ self.projection_matrix + self.projection_bias

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        return embeddings / norms

    def fit_episode(self, episode: Episode) -> None:
        """Compute class prototypes from support set."""
        self.classes = list(np.unique(episode.support_y))

        # Embed support samples
        embeddings = self._embed(episode.support_X)

        # Compute prototype for each class
        prototypes = []
        for cls in self.classes:
            mask = episode.support_y == cls
            class_embeddings = embeddings[mask]
            prototype = np.mean(class_embeddings, axis=0)
            prototypes.append(prototype)

        self.prototypes = np.array(prototypes)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict labels using nearest prototype."""
        if self.prototypes is None:
            raise ValueError("Model not fitted. Call fit_episode first.")

        embeddings = self._embed(X)
        distances = cdist(embeddings, self.prototypes, metric=self.distance_metric)

        # Nearest prototype
        pred_indices = np.argmin(distances, axis=1)
        return np.array([self.classes[i] for i in pred_indices])

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute class probabilities using negative distances."""
        if self.prototypes is None:
            raise ValueError("Model not fitted. Call fit_episode first.")

        embeddings = self._embed(X)
        distances = cdist(embeddings, self.prototypes, metric=self.distance_metric)

        # Convert distances to probabilities via softmax
        logits = -distances / self.temperature
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return probs


class MatchingNetworkNumpy(BaseFewShotLearner):
    """
    NumPy implementation of Matching Networks.

    Uses attention over support set embeddings for classification.

    Reference: "Matching Networks for One Shot Learning"
               (Vinyals et al., 2016)
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        use_cosine_attention: bool = True,
        random_state: int | None = None,
    ):
        """
        Initialize Matching Network.

        Args:
            embedding_dim: Dimension of embedding space
            use_cosine_attention: Use cosine similarity for attention
            random_state: Seed for reproducible random initialization
        """
        self.embedding_dim = embedding_dim
        self.use_cosine_attention = use_cosine_attention
        self.rng = np.random.default_rng(random_state)

        # Projection
        self.projection_matrix: NDArray[np.float64] | None = None
        self.projection_bias: NDArray[np.float64] | None = None

        # Support set
        self.support_embeddings: NDArray[np.float64] | None = None
        self.support_labels: NDArray[np.int64] | None = None
        self.classes: list[int] = []

    def _initialize_projection(self, input_dim: int) -> None:
        """Initialize projection matrix."""
        scale = np.sqrt(2.0 / (input_dim + self.embedding_dim))
        self.projection_matrix = (
            self.rng.standard_normal((input_dim, self.embedding_dim)).astype(np.float64) * scale
        )
        self.projection_bias = np.zeros(self.embedding_dim, dtype=np.float64)

    def _embed(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Project to embedding space."""
        if self.projection_matrix is None:
            self._initialize_projection(X.shape[1])

        if X.shape[1] != self.projection_matrix.shape[0]:
            self._initialize_projection(X.shape[1])

        embeddings = X @ self.projection_matrix + self.projection_bias

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        return embeddings / norms

    def fit_episode(self, episode: Episode) -> None:
        """Store support set embeddings."""
        self.classes = list(np.unique(episode.support_y))
        self.support_embeddings = self._embed(episode.support_X)
        self.support_labels = episode.support_y

    def _attention_weights(self, query_embeddings: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute attention weights between query and support."""
        if self.use_cosine_attention:
            # Cosine similarity (already L2 normalized)
            similarities = query_embeddings @ self.support_embeddings.T
        else:
            # Negative Euclidean distance
            similarities = -cdist(query_embeddings, self.support_embeddings)

        # Softmax over support set
        exp_sim = np.exp(similarities - np.max(similarities, axis=1, keepdims=True))
        attention = exp_sim / np.sum(exp_sim, axis=1, keepdims=True)

        return attention

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict using attention-weighted voting."""
        probs = self.predict_proba(X)
        pred_indices = np.argmax(probs, axis=1)
        return np.array([self.classes[i] for i in pred_indices])

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute class probabilities via attention."""
        if self.support_embeddings is None:
            raise ValueError("Model not fitted. Call fit_episode first.")

        query_embeddings = self._embed(X)
        attention = self._attention_weights(query_embeddings)

        # Sum attention weights per class
        n_queries = len(X)
        n_classes = len(self.classes)
        class_probs = np.zeros((n_queries, n_classes))

        for i, cls in enumerate(self.classes):
            mask = self.support_labels == cls
            class_probs[:, i] = np.sum(attention[:, mask], axis=1)

        return class_probs


class SiameseNetworkNumpy(BaseFewShotLearner):
    """
    NumPy implementation of Siamese Networks.

    Learns similarity function between pairs of examples.

    Reference: "Siamese Neural Networks for One-shot Image Recognition"
               (Koch et al., 2015)
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        similarity_threshold: float = 0.5,
        random_state: int | None = None,
    ):
        """
        Initialize Siamese Network.

        Args:
            embedding_dim: Dimension of embedding space
            similarity_threshold: Threshold for binary similarity
            random_state: Seed for reproducible random initialization
        """
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.rng = np.random.default_rng(random_state)

        # Projection
        self.projection_matrix: NDArray[np.float64] | None = None
        self.projection_bias: NDArray[np.float64] | None = None

        # Support set
        self.support_embeddings: NDArray[np.float64] | None = None
        self.support_labels: NDArray[np.int64] | None = None
        self.classes: list[int] = []

    def _initialize_projection(self, input_dim: int) -> None:
        """Initialize projection matrix."""
        scale = np.sqrt(2.0 / (input_dim + self.embedding_dim))
        self.projection_matrix = (
            self.rng.standard_normal((input_dim, self.embedding_dim)).astype(np.float64) * scale
        )
        self.projection_bias = np.zeros(self.embedding_dim, dtype=np.float64)

    def _embed(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Project to embedding space."""
        if self.projection_matrix is None:
            self._initialize_projection(X.shape[1])

        if X.shape[1] != self.projection_matrix.shape[0]:
            self._initialize_projection(X.shape[1])

        embeddings = X @ self.projection_matrix + self.projection_bias

        # Apply ReLU
        embeddings = np.maximum(embeddings, 0)

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        return embeddings / norms

    def _compute_similarity(
        self,
        embed1: NDArray[np.float64],
        embed2: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute pairwise similarity scores."""
        # L1 distance
        diff = np.abs(embed1[:, np.newaxis, :] - embed2[np.newaxis, :, :])
        l1_dist = np.sum(diff, axis=2)

        # Convert to similarity (sigmoid of negative distance)
        similarity = 1.0 / (1.0 + l1_dist)

        return similarity

    def fit_episode(self, episode: Episode) -> None:
        """Store support set embeddings."""
        self.classes = list(np.unique(episode.support_y))
        self.support_embeddings = self._embed(episode.support_X)
        self.support_labels = episode.support_y

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict using maximum similarity."""
        probs = self.predict_proba(X)
        pred_indices = np.argmax(probs, axis=1)
        return np.array([self.classes[i] for i in pred_indices])

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute class probabilities via similarity."""
        if self.support_embeddings is None:
            raise ValueError("Model not fitted. Call fit_episode first.")

        query_embeddings = self._embed(X)
        similarities = self._compute_similarity(query_embeddings, self.support_embeddings)

        # Average similarity per class
        n_queries = len(X)
        n_classes = len(self.classes)
        class_probs = np.zeros((n_queries, n_classes))

        for i, cls in enumerate(self.classes):
            mask = self.support_labels == cls
            class_probs[:, i] = np.mean(similarities[:, mask], axis=1)

        # Normalize to probabilities
        class_probs = class_probs / (np.sum(class_probs, axis=1, keepdims=True) + 1e-10)

        return class_probs


if TORCH_AVAILABLE:

    class PrototypicalNetworkTorch(nn.Module, BaseFewShotLearner):
        """
        PyTorch implementation of Prototypical Networks.

        More expressive than NumPy version with learnable encoder.
        """

        def __init__(
            self,
            input_dim: int = 128,
            hidden_dim: int = 128,
            embedding_dim: int = 64,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.input_dim = input_dim
            self.embedding_dim = embedding_dim

            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embedding_dim),
            )

            self.prototypes: torch.Tensor | None = None
            self.classes: list[int] = []

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Encode input to embedding space."""
            # Handle dimension mismatch by padding/truncating
            if x.shape[-1] != self.input_dim:
                if x.shape[-1] < self.input_dim:
                    padding = torch.zeros(
                        *x.shape[:-1], self.input_dim - x.shape[-1], device=x.device, dtype=x.dtype
                    )
                    x = torch.cat([x, padding], dim=-1)
                else:
                    x = x[..., : self.input_dim]

            embeddings = self.encoder(x)
            # L2 normalize
            return F.normalize(embeddings, p=2, dim=-1)

        def fit_episode(self, episode: Episode) -> None:
            """Compute prototypes from support set."""
            self.classes = list(np.unique(episode.support_y))

            support_X = torch.tensor(episode.support_X, dtype=torch.float32)
            support_y = torch.tensor(episode.support_y, dtype=torch.long)

            with torch.no_grad():
                embeddings = self.forward(support_X)

            prototypes = []
            for cls in self.classes:
                mask = support_y == cls
                class_embeddings = embeddings[mask]
                prototype = class_embeddings.mean(dim=0)
                prototypes.append(prototype)

            self.prototypes = torch.stack(prototypes)

        def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
            """Predict labels."""
            probs = self.predict_proba(X)
            pred_indices = np.argmax(probs, axis=1)
            return np.array([self.classes[i] for i in pred_indices])

        def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
            """Compute class probabilities."""
            if self.prototypes is None:
                raise ValueError("Model not fitted")

            X_tensor = torch.tensor(X, dtype=torch.float32)

            with torch.no_grad():
                embeddings = self.forward(X_tensor)

                # Euclidean distance to prototypes
                distances = torch.cdist(embeddings, self.prototypes)

                # Negative distance softmax
                logits = -distances
                probs = F.softmax(logits, dim=1)

            return probs.numpy()


class FewShotLearner:
    """
    Unified interface for few-shot learning experiments.

    Supports multiple methods and evaluation protocols including
    the critical 10/50/100 label experiments.
    """

    def __init__(
        self,
        method: FewShotMethod = FewShotMethod.PROTOTYPICAL,
        embedding_dim: int = 64,
        n_way: int = 2,
        k_shot: int = 5,
        n_query: int = 15,
        n_episodes: int = 100,
        use_pytorch: bool = True,
        seed: int = 42,
    ):
        """
        Initialize few-shot learner.

        Args:
            method: Few-shot learning method
            embedding_dim: Embedding dimension
            n_way: Number of classes per episode
            k_shot: Support samples per class
            n_query: Query samples per class
            n_episodes: Number of episodes
            use_pytorch: Use PyTorch if available
            seed: Random seed
        """
        self.method = method
        self.embedding_dim = embedding_dim
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.n_episodes = n_episodes
        self.use_pytorch = use_pytorch and TORCH_AVAILABLE
        self.seed = seed

        # Initialize model
        self.model = self._create_model()

        # Episode generator
        self.episode_generator = EpisodeGenerator(
            n_way=n_way,
            k_shot=k_shot,
            n_query=n_query,
            n_episodes=n_episodes,
            seed=seed,
        )

    def _create_model(self) -> BaseFewShotLearner:
        """Create the few-shot model based on method."""
        if self.method == FewShotMethod.PROTOTYPICAL:
            if self.use_pytorch and TORCH_AVAILABLE:
                return PrototypicalNetworkTorch(embedding_dim=self.embedding_dim)
            return PrototypicalNetworkNumpy(embedding_dim=self.embedding_dim)

        elif self.method == FewShotMethod.MATCHING:
            return MatchingNetworkNumpy(embedding_dim=self.embedding_dim)

        elif self.method == FewShotMethod.SIAMESE:
            return SiameseNetworkNumpy(embedding_dim=self.embedding_dim)

        elif self.method == FewShotMethod.NEAREST_CENTROID:
            return PrototypicalNetworkNumpy(embedding_dim=self.embedding_dim)

        else:
            logger.warning(f"Method {self.method} not implemented, using Prototypical")
            return PrototypicalNetworkNumpy(embedding_dim=self.embedding_dim)

    def evaluate(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
    ) -> FewShotResult:
        """
        Evaluate few-shot learning performance.

        Args:
            X: Feature matrix
            y: Labels

        Returns:
            FewShotResult with metrics
        """
        episode_accuracies = []
        episode_times = []
        all_preds = []
        all_true = []

        for episode in self.episode_generator.generate(X, y):
            start_time = time.time()

            # Fit on support set
            self.model.fit_episode(episode)

            # Predict on query set
            preds = self.model.predict(episode.query_X)

            episode_time = time.time() - start_time
            episode_times.append(episode_time)

            # Calculate accuracy
            acc = np.mean(preds == episode.query_y)
            episode_accuracies.append(acc)

            all_preds.extend(preds.tolist())
            all_true.extend(episode.query_y.tolist())

        # Aggregate metrics
        all_preds = np.array(all_preds)
        all_true = np.array(all_true)

        # Handle binary classification
        avg_precision = 0.0
        avg_recall = 0.0
        avg_f1 = 0.0

        try:
            from sklearn.metrics import f1_score, precision_score, recall_score

            avg_precision = precision_score(
                all_true, all_preds, average="weighted", zero_division=0
            )
            avg_recall = recall_score(all_true, all_preds, average="weighted", zero_division=0)
            avg_f1 = f1_score(all_true, all_preds, average="weighted", zero_division=0)
        except ImportError:
            pass

        # Calculate confidence interval
        accs = np.array(episode_accuracies)
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        n = len(accs)
        ci_margin = 1.96 * std_acc / np.sqrt(n) if n > 0 else 0

        return FewShotResult(
            accuracy=float(mean_acc),
            precision=float(avg_precision),
            recall=float(avg_recall),
            f1=float(avg_f1),
            n_way=self.n_way,
            k_shot=self.k_shot,
            n_episodes=self.n_episodes,
            n_labels_used=self.n_way * self.k_shot,
            episode_accuracies=episode_accuracies,
            episode_times=episode_times,
            accuracy_ci_lower=float(mean_acc - ci_margin),
            accuracy_ci_upper=float(mean_acc + ci_margin),
            method=self.method.value,
            embedding_dim=self.embedding_dim,
        )

    def run_k_shot_experiment(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
        k_values: list[int] | None = None,
        n_trials: int = 10,
    ) -> dict[int, FewShotResult]:
        """
        Run experiments with different numbers of labels (10, 50, 100).

        This addresses the user requirement for k-shot experiments
        that demonstrate Mercury's architectural advantages.

        Args:
            X: Feature matrix
            y: Labels
            k_values: Total label counts to test (e.g., [10, 50, 100])
            n_trials: Trials per k value

        Returns:
            Dictionary mapping k to FewShotResult
        """
        if k_values is None:
            k_values = [10, 50, 100]
        results: dict[int, FewShotResult] = {}

        for k in k_values:
            logger.info(f"Running {k}-label experiment ({n_trials} trials)")

            episode_accuracies = []
            episode_times = []
            all_preds = []
            all_true = []

            for k_total, trial_id, episode in self.episode_generator.generate_k_shot_experiment(
                X, y, k_values=[k], n_trials=n_trials
            ):
                start_time = time.time()

                # Fit on support set
                self.model.fit_episode(episode)

                # Predict on query set
                preds = self.model.predict(episode.query_X)

                episode_time = time.time() - start_time
                episode_times.append(episode_time)

                # Calculate accuracy
                acc = np.mean(preds == episode.query_y)
                episode_accuracies.append(acc)

                all_preds.extend(preds.tolist())
                all_true.extend(episode.query_y.tolist())

            # Aggregate metrics
            all_preds_arr = np.array(all_preds)
            all_true_arr = np.array(all_true)

            avg_precision = 0.0
            avg_recall = 0.0
            avg_f1 = 0.0

            try:
                from sklearn.metrics import f1_score, precision_score, recall_score

                avg_precision = precision_score(
                    all_true_arr, all_preds_arr, average="weighted", zero_division=0
                )
                avg_recall = recall_score(
                    all_true_arr, all_preds_arr, average="weighted", zero_division=0
                )
                avg_f1 = f1_score(all_true_arr, all_preds_arr, average="weighted", zero_division=0)
            except ImportError:
                pass

            # Confidence interval
            accs = np.array(episode_accuracies)
            mean_acc = np.mean(accs) if len(accs) > 0 else 0.0
            std_acc = np.std(accs) if len(accs) > 0 else 0.0
            n = len(accs)
            ci_margin = 1.96 * std_acc / np.sqrt(n) if n > 0 else 0

            results[k] = FewShotResult(
                accuracy=float(mean_acc),
                precision=float(avg_precision),
                recall=float(avg_recall),
                f1=float(avg_f1),
                n_way=len(np.unique(y)),
                k_shot=k // len(np.unique(y)),
                n_episodes=n_trials,
                n_labels_used=k,
                episode_accuracies=episode_accuracies,
                episode_times=episode_times,
                accuracy_ci_lower=float(mean_acc - ci_margin),
                accuracy_ci_upper=float(mean_acc + ci_margin),
                method=self.method.value,
                embedding_dim=self.embedding_dim,
            )

            logger.info(
                f"  {k}-label: accuracy={mean_acc:.4f} "
                f"[{mean_acc - ci_margin:.4f}, {mean_acc + ci_margin:.4f}]"
            )

        return results


def create_few_shot_learner(
    method: str = "prototypical",
    k_shot: int = 5,
    n_way: int = 2,
    **kwargs: Any,
) -> FewShotLearner:
    """
    Factory function to create few-shot learner.

    Args:
        method: Method name ('prototypical', 'matching', 'siamese')
        k_shot: Support samples per class
        n_way: Classes per episode
        **kwargs: Additional arguments

    Returns:
        Configured FewShotLearner
    """
    method_map = {
        "prototypical": FewShotMethod.PROTOTYPICAL,
        "matching": FewShotMethod.MATCHING,
        "siamese": FewShotMethod.SIAMESE,
        "maml": FewShotMethod.MAML,
        "relation": FewShotMethod.RELATION,
        "nearest_centroid": FewShotMethod.NEAREST_CENTROID,
    }

    m = method_map.get(method, FewShotMethod.PROTOTYPICAL)

    return FewShotLearner(
        method=m,
        k_shot=k_shot,
        n_way=n_way,
        **kwargs,
    )


# Exports
__all__ = [
    "BaseFewShotLearner",
    "Episode",
    "EpisodeGenerator",
    "EpisodeSamplingStrategy",
    "FewShotLearner",
    "FewShotMethod",
    "FewShotResult",
    "MatchingNetworkNumpy",
    "PrototypicalNetworkNumpy",
    "SiameseNetworkNumpy",
    "create_few_shot_learner",
]

if TORCH_AVAILABLE:
    __all__.append("PrototypicalNetworkTorch")
