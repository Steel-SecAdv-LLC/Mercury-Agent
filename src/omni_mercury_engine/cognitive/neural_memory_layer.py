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
Neural Memory Layer - Memory Embeddings and Pattern Detection

Implements the neural layer of the neuro-symbolic architecture:
- Memory entry vectorization and embedding
- K-means clustering on semantic/episodic data
- Pattern detection and anomaly prediction from historical trajectories
- Lightweight vector operations (<10M parameters constraint)

Research Sources:
- Memory Networks (Weston et al., 2015)
- Neural Turing Machines (Graves et al., 2014)
- Episodic Memory in AI (Tulving, 1972)
- K-means clustering (Lloyd, 1982)

Integration:
    This module integrates with AgentMemory from mercury_a_agent.py
    and feeds into the SymbolicReasoningLayer for hybrid inference.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory for embedding."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    PROCEDURAL = "procedural"


class PatternType(Enum):
    """Types of detected patterns."""

    ANOMALY = "anomaly"
    TREND = "trend"
    CYCLE = "cycle"
    ESCALATION = "escalation"
    CORRELATION = "correlation"
    NOVELTY = "novelty"


@dataclass
class MemoryEmbedding:
    """Embedded memory entry with vector representation."""

    entry_id: str
    memory_type: MemoryType
    embedding: np.ndarray
    timestamp: float
    importance: float = 0.5
    cluster_id: int = -1
    metadata: dict[str, Any] = field(default_factory=dict)

    def similarity(self, other: MemoryEmbedding) -> float:
        """Compute cosine similarity with another embedding."""
        norm_self = np.linalg.norm(self.embedding)
        norm_other = np.linalg.norm(other.embedding)
        if norm_self == 0 or norm_other == 0:
            return 0.0
        return float(np.dot(self.embedding, other.embedding) / (norm_self * norm_other))


@dataclass
class DetectedPattern:
    """A detected pattern from memory analysis."""

    pattern_id: str
    pattern_type: PatternType
    confidence: float
    description: str
    supporting_memories: list[str]
    centroid: np.ndarray[Any, Any] | None = None
    temporal_span: tuple[float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyPrediction:
    """Prediction of potential anomaly from patterns."""

    prediction_id: str
    predicted_type: str
    probability: float
    time_horizon: float
    contributing_patterns: list[str]
    explanation: str
    confidence_interval: tuple[float, float] = (0.0, 1.0)


class MemoryVectorizer:
    """
    Vectorize memory entries into dense embeddings.

    Uses lightweight feature extraction to convert memory content into fixed-dimensional vectors
    suitable for clustering and similarity.
    """

    def __init__(self, embedding_dim: int = 64, seed: int | None = 42) -> None:
        """
        Initialize memory vectorizer.

        Args:
            embedding_dim: Dimension of output embeddings (default 64)
            seed: Optional seed for the random projection used in
                ``_project_to_dim``.  The projection matrix is rebuilt on
                every call from a fresh ``np.random.default_rng(seed)`` so
                the SAME ``MemoryVectorizer`` instance always produces the
                SAME projection for a given input length — pass a different
                ``seed`` to get a different projection identity.  Defaults
                to ``42`` to preserve the deterministic behavior of the
                previous ``np.random.seed(42)`` global-state call.  Pass
                ``None`` to draw projection bytes from OS entropy on every
                call (NOT recommended — projections become non-reproducible).
        """
        self.embedding_dim = embedding_dim
        self._vocab: dict[str, int] = {}
        self._vocab_size = 0
        self._idf_weights: dict[str, float] = {}
        # Stored as the *projection seed*, not a Generator state, because
        # ``_project_to_dim`` needs to rebuild the projection matrix
        # deterministically on every call (using a stateful per-instance
        # Generator would advance through the RNG stream and produce a
        # different matrix on each call, breaking the projection identity).
        self._projection_seed: int | None = seed

    def fit(self, memory_contents: list[dict[str, Any]]) -> None:
        """
        Fit vectorizer on memory contents to build vocabulary.

        Args:
            memory_contents: List of memory content dictionaries
        """
        doc_freq: dict[str, int] = {}
        n_docs = len(memory_contents)

        for content in memory_contents:
            tokens = self._tokenize(content)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1
                if token not in self._vocab:
                    self._vocab[token] = self._vocab_size
                    self._vocab_size += 1

        for token, freq in doc_freq.items():
            self._idf_weights[token] = np.log((n_docs + 1) / (freq + 1)) + 1

    def transform(self, content: dict[str, Any]) -> np.ndarray[Any, Any]:
        """
        Transform memory content into embedding vector.

        Args:
            content: Memory content dictionary

        Returns:
            Dense embedding vector of shape (embedding_dim,)
        """
        tokens = self._tokenize(content)

        if not tokens:
            return np.zeros(self.embedding_dim)

        tf_idf = np.zeros(max(self._vocab_size, 1))
        token_counts: dict[str, int] = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        for token, count in token_counts.items():
            if token in self._vocab:
                tf = count / len(tokens)
                idf = self._idf_weights.get(token, 1.0)
                tf_idf[self._vocab[token]] = tf * idf

        embedding = self._project_to_dim(tf_idf)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _tokenize(self, content: dict[str, Any]) -> list[str]:
        """Tokenize memory content into string tokens."""
        tokens = []

        def extract_tokens(obj: Any, prefix: str = "") -> None:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    extract_tokens(value, f"{prefix}{key}_")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    extract_tokens(item, f"{prefix}{i}_")
            elif isinstance(obj, str):
                words = obj.lower().split()
                tokens.extend([f"{prefix}{w}" for w in words])
            elif isinstance(obj, (int, float)):
                tokens.append(f"{prefix}num_{int(obj) // 10}")
            elif obj is not None:
                tokens.append(f"{prefix}{type(obj).__name__}")

        extract_tokens(content)
        return tokens

    def _project_to_dim(self, sparse_vec: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Project sparse vector to target embedding dimension."""
        if len(sparse_vec) == 0:
            return np.zeros(self.embedding_dim)

        # Build the projection matrix deterministically from the
        # constructor seed.  Using a fresh ``default_rng(seed)`` here (and
        # not a long-lived per-instance Generator) is intentional: every
        # call must regenerate the SAME projection matrix for a given
        # input length, otherwise the projection identity would drift.
        proj_rng = np.random.default_rng(self._projection_seed)
        projection = proj_rng.standard_normal((len(sparse_vec), self.embedding_dim))
        projection = projection / np.sqrt(self.embedding_dim)

        return sparse_vec @ projection


class KMeansClusterer:
    """
    K-means clustering for memory embeddings.

    Lightweight implementation suitable for real-time clustering of memory entries without external
    dependencies.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: int = 42,
    ):
        """
        Initialize K-means clusterer.

        Args:
            n_clusters: Number of clusters
            max_iter: Maximum iterations
            tol: Convergence tolerance
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids: np.ndarray[Any, Any] | None = None
        self.labels_: np.ndarray[Any, Any] | None = None
        self.inertia_: float = 0.0

    def fit(self, X: np.ndarray[Any, Any]) -> KMeansClusterer:
        """
        Fit K-means on embeddings.

        Args:
            X: Embedding matrix of shape (n_samples, n_features)

        Returns:
            Self for chaining
        """
        n_samples = X.shape[0]
        if n_samples < self.n_clusters:
            self.n_clusters = max(1, n_samples)

        # Per-call ``Generator`` so the centroid initialization is
        # reproducible w.r.t. ``self.random_state`` without touching the
        # global ``np.random`` state.
        kmeans_rng = np.random.default_rng(self.random_state)
        indices = kmeans_rng.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[indices].copy()

        for _ in range(self.max_iter):
            distances = cdist(X, self.centroids, metric="euclidean")
            self.labels_ = np.argmin(distances, axis=1)

            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.n_clusters):
                mask = self.labels_ == k
                if np.any(mask):
                    new_centroids[k] = X[mask].mean(axis=0)
                else:
                    new_centroids[k] = self.centroids[k]

            shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids

            if shift < self.tol:
                break

        distances = cdist(X, self.centroids, metric="euclidean")
        self.inertia_ = float(np.sum(np.min(distances, axis=1) ** 2))

        return self

    def predict(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        Predict cluster labels for new embeddings.

        Args:
            X: Embedding matrix of shape (n_samples, n_features)

        Returns:
            Cluster labels
        """
        if self.centroids is None:
            raise ValueError("Clusterer not fitted. Call fit() first.")

        distances = cdist(X, self.centroids, metric="euclidean")
        return np.argmin(distances, axis=1)

    def get_cluster_distances(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Get distances to all cluster centroids."""
        if self.centroids is None:
            raise ValueError("Clusterer not fitted. Call fit() first.")
        return cdist(X, self.centroids, metric="euclidean")


class PatternDetector:
    """
    Detect patterns from clustered memory embeddings.

    Identifies anomalies, trends, cycles, and escalations from temporal sequences of memory entries.
    """

    def __init__(
        self,
        anomaly_threshold: float = 2.0,
        trend_window: int = 10,
        min_pattern_support: int = 3,
    ):
        """
        Initialize pattern detector.

        Args:
            anomaly_threshold: Standard deviations for anomaly detection
            trend_window: Window size for trend detection
            min_pattern_support: Minimum memories to form a pattern
        """
        self.anomaly_threshold = anomaly_threshold
        self.trend_window = trend_window
        self.min_pattern_support = min_pattern_support
        self._pattern_counter = 0

    def detect_patterns(
        self,
        embeddings: list[MemoryEmbedding],
        clusterer: KMeansClusterer,
    ) -> list[DetectedPattern]:
        """
        Detect patterns from memory embeddings.

        Args:
            embeddings: List of memory embeddings
            clusterer: Fitted K-means clusterer

        Returns:
            List of detected patterns
        """
        patterns: list[DetectedPattern] = []

        if not embeddings or clusterer.centroids is None:
            return patterns

        X = np.array([e.embedding for e in embeddings])
        distances = clusterer.get_cluster_distances(X)
        labels = clusterer.predict(X)

        for i, emb in enumerate(embeddings):
            emb.cluster_id = int(labels[i])

        patterns.extend(self._detect_anomalies(embeddings, distances, labels))
        patterns.extend(self._detect_trends(embeddings))
        patterns.extend(self._detect_escalations(embeddings))
        patterns.extend(self._detect_novelty(embeddings, clusterer))

        return patterns

    def _detect_anomalies(
        self,
        embeddings: list[MemoryEmbedding],
        distances: np.ndarray[Any, Any],
        labels: np.ndarray[Any, Any],
    ) -> list[DetectedPattern]:
        """Detect anomalous memories based on cluster distance."""
        patterns = []

        min_distances = np.min(distances, axis=1)
        mean_dist = np.mean(min_distances)
        std_dist = np.std(min_distances)

        threshold = mean_dist + self.anomaly_threshold * std_dist

        for i, (emb, dist) in enumerate(zip(embeddings, min_distances)):
            if dist > threshold:
                self._pattern_counter += 1
                patterns.append(
                    DetectedPattern(
                        pattern_id=f"anomaly_{self._pattern_counter}",
                        pattern_type=PatternType.ANOMALY,
                        confidence=min(1.0, (dist - mean_dist) / (std_dist + 1e-10) / 5),
                        description=f"Memory {emb.entry_id} deviates {(dist - mean_dist) / (std_dist + 1e-10):.2f} std from cluster",
                        supporting_memories=[emb.entry_id],
                        centroid=None,
                        metadata={"distance": float(dist), "threshold": float(threshold)},
                    )
                )

        return patterns

    def _detect_trends(
        self,
        embeddings: list[MemoryEmbedding],
    ) -> list[DetectedPattern]:
        """Detect temporal trends in memory importance."""
        patterns: list[DetectedPattern] = []

        if len(embeddings) < self.trend_window:
            return patterns

        sorted_embs = sorted(embeddings, key=lambda e: e.timestamp)
        importances = [e.importance for e in sorted_embs]

        for i in range(len(importances) - self.trend_window + 1):
            window = importances[i : i + self.trend_window]
            slope = np.polyfit(range(len(window)), window, 1)[0]

            if abs(slope) > 0.05:
                self._pattern_counter += 1
                direction = "increasing" if slope > 0 else "decreasing"
                patterns.append(
                    DetectedPattern(
                        pattern_id=f"trend_{self._pattern_counter}",
                        pattern_type=PatternType.TREND,
                        confidence=min(1.0, abs(slope) * 10),
                        description=f"{direction.capitalize()} importance trend detected (slope={slope:.4f})",
                        supporting_memories=[
                            e.entry_id for e in sorted_embs[i : i + self.trend_window]
                        ],
                        temporal_span=(
                            sorted_embs[i].timestamp,
                            sorted_embs[i + self.trend_window - 1].timestamp,
                        ),
                        metadata={"slope": float(slope), "direction": direction},
                    )
                )

        return patterns

    def _detect_escalations(
        self,
        embeddings: list[MemoryEmbedding],
    ) -> list[DetectedPattern]:
        """Detect escalation patterns (rapid importance increase)."""
        patterns: list[DetectedPattern] = []

        if len(embeddings) < 3:
            return patterns

        sorted_embs = sorted(embeddings, key=lambda e: e.timestamp)

        for i in range(2, len(sorted_embs)):
            recent = [sorted_embs[j].importance for j in range(max(0, i - 2), i + 1)]
            if len(recent) >= 3:
                acceleration = recent[-1] - 2 * recent[-2] + recent[-3]
                if acceleration > 0.1:
                    self._pattern_counter += 1
                    patterns.append(
                        DetectedPattern(
                            pattern_id=f"escalation_{self._pattern_counter}",
                            pattern_type=PatternType.ESCALATION,
                            confidence=min(1.0, acceleration * 5),
                            description=f"Escalation detected: importance accelerating (acc={acceleration:.4f})",
                            supporting_memories=[
                                e.entry_id for e in sorted_embs[max(0, i - 2) : i + 1]
                            ],
                            temporal_span=(
                                sorted_embs[max(0, i - 2)].timestamp,
                                sorted_embs[i].timestamp,
                            ),
                            metadata={"acceleration": float(acceleration)},
                        )
                    )

        return patterns

    def _detect_novelty(
        self,
        embeddings: list[MemoryEmbedding],
        clusterer: KMeansClusterer,
    ) -> list[DetectedPattern]:
        """Detect novel patterns not fitting existing clusters well."""
        patterns: list[DetectedPattern] = []

        if clusterer.centroids is None or len(embeddings) < self.min_pattern_support:
            return patterns

        cluster_counts: dict[int, list[MemoryEmbedding]] = {}
        for emb in embeddings:
            if emb.cluster_id not in cluster_counts:
                cluster_counts[emb.cluster_id] = []
            cluster_counts[emb.cluster_id].append(emb)

        for cluster_id, members in cluster_counts.items():
            if len(members) < self.min_pattern_support:
                continue

            member_embeddings = np.array([m.embedding for m in members])
            centroid = clusterer.centroids[cluster_id]
            avg_dist = np.mean(np.linalg.norm(member_embeddings - centroid, axis=1))

            if avg_dist > 0.5:
                self._pattern_counter += 1
                patterns.append(
                    DetectedPattern(
                        pattern_id=f"novelty_{self._pattern_counter}",
                        pattern_type=PatternType.NOVELTY,
                        confidence=min(1.0, avg_dist),
                        description=f"Novel cluster {cluster_id} with high internal variance",
                        supporting_memories=[m.entry_id for m in members],
                        centroid=centroid,
                        metadata={"avg_distance": float(avg_dist), "cluster_size": len(members)},
                    )
                )

        return patterns


class AnomalyPredictor:
    """
    Predict future anomalies from detected patterns.

    Uses pattern history and Bayesian-inspired confidence to forecast potential anomalies.
    """

    def __init__(
        self,
        prediction_horizon: float = 3600.0,
        min_confidence: float = 0.5,
    ):
        """
        Initialize anomaly predictor.

        Args:
            prediction_horizon: Time horizon for predictions (seconds)
            min_confidence: Minimum confidence for predictions
        """
        self.prediction_horizon = prediction_horizon
        self.min_confidence = min_confidence
        self._prediction_counter = 0

    def predict(
        self,
        patterns: list[DetectedPattern],
        current_time: float | None = None,
    ) -> list[AnomalyPrediction]:
        """
        Predict future anomalies from patterns.

        Args:
            patterns: List of detected patterns
            current_time: Current timestamp (defaults to now)

        Returns:
            List of anomaly predictions
        """
        predictions: list[AnomalyPrediction] = []
        current_time = current_time or time.time()

        escalations = [p for p in patterns if p.pattern_type == PatternType.ESCALATION]
        if escalations:
            predictions.extend(self._predict_from_escalations(escalations, current_time))

        trends = [p for p in patterns if p.pattern_type == PatternType.TREND]
        if trends:
            predictions.extend(self._predict_from_trends(trends, current_time))

        anomalies = [p for p in patterns if p.pattern_type == PatternType.ANOMALY]
        if len(anomalies) >= 3:
            predictions.extend(self._predict_anomaly_cluster(anomalies, current_time))

        return [p for p in predictions if p.probability >= self.min_confidence]

    def _predict_from_escalations(
        self,
        escalations: list[DetectedPattern],
        current_time: float,
    ) -> list[AnomalyPrediction]:
        """Predict anomalies from escalation patterns."""
        predictions = []

        for esc in escalations:
            acceleration = esc.metadata.get("acceleration", 0.1)
            probability = min(0.95, 0.5 + acceleration * 2)

            self._prediction_counter += 1
            predictions.append(
                AnomalyPrediction(
                    prediction_id=f"pred_esc_{self._prediction_counter}",
                    predicted_type="escalation_anomaly",
                    probability=probability,
                    time_horizon=self.prediction_horizon * (1 - acceleration),
                    contributing_patterns=[esc.pattern_id],
                    explanation=f"Escalation pattern suggests {probability:.0%} chance of anomaly",
                    confidence_interval=(max(0, probability - 0.15), min(1, probability + 0.15)),
                )
            )

        return predictions

    def _predict_from_trends(
        self,
        trends: list[DetectedPattern],
        current_time: float,
    ) -> list[AnomalyPrediction]:
        """Predict anomalies from trend patterns."""
        predictions = []

        increasing_trends = [t for t in trends if t.metadata.get("direction") == "increasing"]

        if increasing_trends:
            avg_slope = np.mean([t.metadata.get("slope", 0) for t in increasing_trends])
            probability = min(0.9, 0.3 + avg_slope * 5)

            self._prediction_counter += 1
            predictions.append(
                AnomalyPrediction(
                    prediction_id=f"pred_trend_{self._prediction_counter}",
                    predicted_type="trend_anomaly",
                    probability=probability,
                    time_horizon=self.prediction_horizon,
                    contributing_patterns=[t.pattern_id for t in increasing_trends],
                    explanation=f"Increasing trend (slope={avg_slope:.4f}) suggests potential anomaly",
                    confidence_interval=(max(0, probability - 0.2), min(1, probability + 0.1)),
                )
            )

        return predictions

    def _predict_anomaly_cluster(
        self,
        anomalies: list[DetectedPattern],
        current_time: float,
    ) -> list[AnomalyPrediction]:
        """Predict from clustering of anomalies."""
        predictions = []

        recent_anomalies = [
            a
            for a in anomalies
            if a.temporal_span and current_time - a.temporal_span[1] < self.prediction_horizon
        ]

        if len(recent_anomalies) >= 2:
            probability = min(0.95, 0.4 + len(recent_anomalies) * 0.1)

            self._prediction_counter += 1
            predictions.append(
                AnomalyPrediction(
                    prediction_id=f"pred_cluster_{self._prediction_counter}",
                    predicted_type="cluster_anomaly",
                    probability=probability,
                    time_horizon=self.prediction_horizon * 0.5,
                    contributing_patterns=[a.pattern_id for a in recent_anomalies],
                    explanation=f"Cluster of {len(recent_anomalies)} recent anomalies suggests elevated risk",
                    confidence_interval=(max(0, probability - 0.1), min(1, probability + 0.1)),
                )
            )

        return predictions


class NeuralMemoryLayer:
    """
    Neural Memory Layer - Main interface for memory-based pattern detection.

    Integrates vectorization, clustering, pattern detection, and prediction
    into a unified interface for the neuro-symbolic architecture.

    This is the neural component that feeds into the symbolic reasoning layer.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        n_clusters: int = 8,
        anomaly_threshold: float = 2.0,
        prediction_horizon: float = 3600.0,
    ):
        """
        Initialize Neural Memory Layer.

        Args:
            embedding_dim: Dimension of memory embeddings
            n_clusters: Number of clusters for K-means
            anomaly_threshold: Threshold for anomaly detection
            prediction_horizon: Time horizon for predictions
        """
        self.embedding_dim = embedding_dim
        self.n_clusters = n_clusters

        self.vectorizer = MemoryVectorizer(embedding_dim=embedding_dim)
        self.clusterer = KMeansClusterer(n_clusters=n_clusters)
        self.pattern_detector = PatternDetector(anomaly_threshold=anomaly_threshold)
        self.predictor = AnomalyPredictor(prediction_horizon=prediction_horizon)

        self.embeddings: list[MemoryEmbedding] = []
        self.patterns: list[DetectedPattern] = []
        self.predictions: list[AnomalyPrediction] = []

        self._fitted = False
        logger.info(f"NeuralMemoryLayer initialized (dim={embedding_dim}, clusters={n_clusters})")

    def ingest_memories(
        self,
        memories: list[dict[str, Any]],
        memory_type: MemoryType = MemoryType.EPISODIC,
    ) -> list[MemoryEmbedding]:
        """
        Ingest memory entries and create embeddings.

        Args:
            memories: List of memory content dictionaries
            memory_type: Type of memories being ingested

        Returns:
            List of created memory embeddings
        """
        if not memories:
            return []

        if not self._fitted:
            self.vectorizer.fit(memories)
            self._fitted = True

        new_embeddings = []
        for i, mem in enumerate(memories):
            embedding = self.vectorizer.transform(mem)
            mem_emb = MemoryEmbedding(
                entry_id=mem.get("id", f"mem_{len(self.embeddings) + i}"),
                memory_type=memory_type,
                embedding=embedding,
                timestamp=mem.get("timestamp", time.time()),
                importance=mem.get("importance", 0.5),
                metadata=mem,
            )
            new_embeddings.append(mem_emb)
            self.embeddings.append(mem_emb)

        if len(self.embeddings) >= self.n_clusters:
            X = np.array([e.embedding for e in self.embeddings])
            self.clusterer.fit(X)

        logger.info(f"Ingested {len(new_embeddings)} memories (total: {len(self.embeddings)})")
        return new_embeddings

    def analyze(self) -> dict[str, Any]:
        """
        Analyze all ingested memories for patterns and predictions.

        Returns:
            Analysis result with patterns, predictions, and statistics
        """
        if len(self.embeddings) < self.n_clusters:
            return {
                "status": "insufficient_data",
                "embeddings_count": len(self.embeddings),
                "required": self.n_clusters,
                "patterns": [],
                "predictions": [],
            }

        self.patterns = self.pattern_detector.detect_patterns(self.embeddings, self.clusterer)
        self.predictions = self.predictor.predict(self.patterns)

        return {
            "status": "analyzed",
            "embeddings_count": len(self.embeddings),
            "patterns_detected": len(self.patterns),
            "predictions_made": len(self.predictions),
            "patterns": [
                {
                    "id": p.pattern_id,
                    "type": p.pattern_type.value,
                    "confidence": p.confidence,
                    "description": p.description,
                }
                for p in self.patterns
            ],
            "predictions": [
                {
                    "id": p.prediction_id,
                    "type": p.predicted_type,
                    "probability": p.probability,
                    "explanation": p.explanation,
                }
                for p in self.predictions
            ],
            "cluster_inertia": self.clusterer.inertia_,
        }

    def get_similar_memories(
        self,
        query_embedding: np.ndarray[Any, Any],
        top_k: int = 5,
    ) -> list[tuple[MemoryEmbedding, float]]:
        """
        Find memories most similar to a query embedding.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return

        Returns:
            List of (embedding, similarity) tuples
        """
        if not self.embeddings:
            return []

        similarities = []
        query_norm = np.linalg.norm(query_embedding)

        for emb in self.embeddings:
            emb_norm = np.linalg.norm(emb.embedding)
            if query_norm > 0 and emb_norm > 0:
                sim = np.dot(query_embedding, emb.embedding) / (query_norm * emb_norm)
            else:
                sim = 0.0
            similarities.append((emb, float(sim)))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def get_anomaly_score(self, embedding: np.ndarray[Any, Any]) -> float:
        """
        Get anomaly score for a single embedding.

        Args:
            embedding: Input embedding vector

        Returns:
            Anomaly score (0-1, higher = more anomalous)
        """
        if self.clusterer.centroids is None:
            return 0.5

        distances = self.clusterer.get_cluster_distances(embedding.reshape(1, -1))
        min_dist = float(np.min(distances))

        all_distances = self.clusterer.get_cluster_distances(
            np.array([e.embedding for e in self.embeddings])
        )
        all_min_dists = np.min(all_distances, axis=1)
        mean_dist = np.mean(all_min_dists)
        std_dist = np.std(all_min_dists)

        if std_dist > 0:
            z_score = (min_dist - mean_dist) / std_dist
            score = 1 / (1 + np.exp(-z_score))
        else:
            score = 0.5

        return float(score)

    def get_neural_features(self) -> np.ndarray[Any, Any]:
        """
        Get aggregated neural features for symbolic layer input.

        Returns:
            Feature vector summarizing neural layer state
        """
        if not self.embeddings:
            return np.zeros(self.embedding_dim + 10)

        all_embeddings = np.array([e.embedding for e in self.embeddings])
        mean_embedding = np.mean(all_embeddings, axis=0)

        pattern_counts = np.zeros(len(PatternType))
        for p in self.patterns:
            pattern_counts[list(PatternType).index(p.pattern_type)] += 1

        prediction_probs = [p.probability for p in self.predictions] or [0.0]

        features = np.concatenate(
            [
                mean_embedding,
                pattern_counts / (len(self.patterns) + 1),
                [np.mean(prediction_probs)],
                [len(self.embeddings) / 1000],
                [self.clusterer.inertia_ / 100 if self.clusterer.inertia_ else 0],
            ]
        )

        return features

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the neural memory layer."""
        return {
            "total_embeddings": len(self.embeddings),
            "embedding_dim": self.embedding_dim,
            "n_clusters": self.n_clusters,
            "fitted": self._fitted,
            "patterns_detected": len(self.patterns),
            "predictions_made": len(self.predictions),
            "cluster_inertia": self.clusterer.inertia_ if self._fitted else None,
            "memory_types": {
                mt.value: sum(1 for e in self.embeddings if e.memory_type == mt)
                for mt in MemoryType
            },
        }
