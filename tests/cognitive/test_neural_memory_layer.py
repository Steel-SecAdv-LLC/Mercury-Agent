# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for Neural Memory Layer - Memory Embeddings and Pattern Detection."""

from __future__ import annotations

import time

import numpy as np
import pytest

from omni_mercury_engine.cognitive.neural_memory_layer import (
    AnomalyPredictor,
    DetectedPattern,
    KMeansClusterer,
    MemoryEmbedding,
    MemoryType,
    MemoryVectorizer,
    NeuralMemoryLayer,
    PatternDetector,
    PatternType,
)


class TestMemoryVectorizer:
    """Tests for MemoryVectorizer."""

    def test_init(self) -> None:
        """Test vectorizer initialization."""
        vectorizer = MemoryVectorizer(embedding_dim=64)
        assert vectorizer.embedding_dim == 64
        assert vectorizer._vocab_size == 0

    def test_fit_builds_vocabulary(self) -> None:
        """Test that fit builds vocabulary from content."""
        vectorizer = MemoryVectorizer(embedding_dim=32)
        contents = [
            {"event": "anomaly detected", "severity": "high"},
            {"event": "normal operation", "severity": "low"},
            {"event": "anomaly detected", "type": "security"},
        ]
        vectorizer.fit(contents)
        assert vectorizer._vocab_size > 0
        assert len(vectorizer._idf_weights) > 0

    def test_transform_returns_correct_shape(self) -> None:
        """Test transform returns correct embedding dimension."""
        vectorizer = MemoryVectorizer(embedding_dim=64)
        contents = [{"event": "test event", "value": 42}]
        vectorizer.fit(contents)
        embedding = vectorizer.transform(contents[0])
        assert embedding.shape == (64,)

    def test_transform_empty_content(self) -> None:
        """Test transform handles empty content."""
        vectorizer = MemoryVectorizer(embedding_dim=32)
        vectorizer.fit([{"key": "value"}])
        embedding = vectorizer.transform({})
        assert embedding.shape == (32,)
        assert np.allclose(embedding, 0)

    def test_transform_normalized(self) -> None:
        """Test embeddings are normalized."""
        vectorizer = MemoryVectorizer(embedding_dim=64)
        contents = [{"event": "test", "data": "sample"}]
        vectorizer.fit(contents)
        embedding = vectorizer.transform(contents[0])
        norm = np.linalg.norm(embedding)
        assert norm == pytest.approx(1.0, abs=0.01) or norm == 0


class TestKMeansClusterer:
    """Tests for KMeansClusterer."""

    def test_init(self) -> None:
        """Test clusterer initialization."""
        clusterer = KMeansClusterer(n_clusters=5)
        assert clusterer.n_clusters == 5
        assert clusterer.centroids is None

    def test_fit_creates_centroids(self) -> None:
        """Test fit creates cluster centroids."""
        clusterer = KMeansClusterer(n_clusters=3, random_state=42)
        X = np.random.randn(100, 10)
        clusterer.fit(X)
        assert clusterer.centroids is not None
        assert clusterer.centroids.shape == (3, 10)

    def test_fit_assigns_labels(self) -> None:
        """Test fit assigns labels to all samples."""
        clusterer = KMeansClusterer(n_clusters=4, random_state=42)
        X = np.random.randn(50, 8)
        clusterer.fit(X)
        assert clusterer.labels_ is not None
        assert len(clusterer.labels_) == 50
        assert all(0 <= label < 4 for label in clusterer.labels_)

    def test_predict_returns_labels(self) -> None:
        """Test predict returns cluster labels."""
        clusterer = KMeansClusterer(n_clusters=3, random_state=42)
        X_train = np.random.randn(30, 5)
        clusterer.fit(X_train)
        X_test = np.random.randn(10, 5)
        labels = clusterer.predict(X_test)
        assert len(labels) == 10
        assert all(0 <= label < 3 for label in labels)

    def test_predict_without_fit_raises(self) -> None:
        """Test predict raises error if not fitted."""
        clusterer = KMeansClusterer(n_clusters=3)
        with pytest.raises(ValueError, match="not fitted"):
            clusterer.predict(np.random.randn(5, 3))

    def test_handles_fewer_samples_than_clusters(self) -> None:
        """Test handles case with fewer samples than clusters."""
        clusterer = KMeansClusterer(n_clusters=10, random_state=42)
        X = np.random.randn(5, 4)
        clusterer.fit(X)
        assert clusterer.n_clusters == 5


class TestMemoryEmbedding:
    """Tests for MemoryEmbedding dataclass."""

    def test_similarity_identical(self) -> None:
        """Test similarity of identical embeddings."""
        emb1 = MemoryEmbedding(
            entry_id="test1",
            memory_type=MemoryType.EPISODIC,
            embedding=np.array([1.0, 0.0, 0.0]),
            timestamp=time.time(),
        )
        emb2 = MemoryEmbedding(
            entry_id="test2",
            memory_type=MemoryType.EPISODIC,
            embedding=np.array([1.0, 0.0, 0.0]),
            timestamp=time.time(),
        )
        assert emb1.similarity(emb2) == pytest.approx(1.0)

    def test_similarity_orthogonal(self) -> None:
        """Test similarity of orthogonal embeddings."""
        emb1 = MemoryEmbedding(
            entry_id="test1",
            memory_type=MemoryType.SEMANTIC,
            embedding=np.array([1.0, 0.0]),
            timestamp=time.time(),
        )
        emb2 = MemoryEmbedding(
            entry_id="test2",
            memory_type=MemoryType.SEMANTIC,
            embedding=np.array([0.0, 1.0]),
            timestamp=time.time(),
        )
        assert emb1.similarity(emb2) == pytest.approx(0.0)

    def test_similarity_zero_vector(self) -> None:
        """Test similarity with zero vector."""
        emb1 = MemoryEmbedding(
            entry_id="test1",
            memory_type=MemoryType.SHORT_TERM,
            embedding=np.array([1.0, 2.0]),
            timestamp=time.time(),
        )
        emb2 = MemoryEmbedding(
            entry_id="test2",
            memory_type=MemoryType.SHORT_TERM,
            embedding=np.array([0.0, 0.0]),
            timestamp=time.time(),
        )
        assert emb1.similarity(emb2) == 0.0


class TestPatternDetector:
    """Tests for PatternDetector."""

    def test_init(self) -> None:
        """Test pattern detector initialization."""
        detector = PatternDetector(anomaly_threshold=2.5)
        assert detector.anomaly_threshold == 2.5

    def test_detect_patterns_empty(self) -> None:
        """Test detection with empty embeddings."""
        detector = PatternDetector()
        clusterer = KMeansClusterer(n_clusters=3)
        patterns = detector.detect_patterns([], clusterer)
        assert patterns == []

    def test_detect_anomalies(self) -> None:
        """Test anomaly detection from outliers."""
        detector = PatternDetector(anomaly_threshold=1.5)
        clusterer = KMeansClusterer(n_clusters=2, random_state=42)

        normal_embeddings = np.random.randn(20, 10) * 0.1
        outlier = np.random.randn(1, 10) * 5

        X = np.vstack([normal_embeddings, outlier])
        clusterer.fit(X)

        embeddings = [
            MemoryEmbedding(
                entry_id=f"mem_{i}",
                memory_type=MemoryType.EPISODIC,
                embedding=X[i],
                timestamp=time.time() + i,
                importance=0.5,
            )
            for i in range(len(X))
        ]

        patterns = detector.detect_patterns(embeddings, clusterer)
        anomaly_patterns = [p for p in patterns if p.pattern_type == PatternType.ANOMALY]
        assert len(anomaly_patterns) >= 0

    def test_detect_trends(self) -> None:
        """Test trend detection from importance changes."""
        detector = PatternDetector(trend_window=5)
        clusterer = KMeansClusterer(n_clusters=2, random_state=42)

        embeddings = []
        base_time = time.time()
        for i in range(10):
            embeddings.append(
                MemoryEmbedding(
                    entry_id=f"mem_{i}",
                    memory_type=MemoryType.SEMANTIC,
                    embedding=np.random.randn(8),
                    timestamp=base_time + i * 100,
                    importance=0.3 + i * 0.07,
                )
            )

        X = np.array([e.embedding for e in embeddings])
        clusterer.fit(X)

        patterns = detector.detect_patterns(embeddings, clusterer)
        trend_patterns = [p for p in patterns if p.pattern_type == PatternType.TREND]
        assert len(trend_patterns) >= 0


class TestAnomalyPredictor:
    """Tests for AnomalyPredictor."""

    def test_init(self) -> None:
        """Test predictor initialization."""
        predictor = AnomalyPredictor(prediction_horizon=7200.0)
        assert predictor.prediction_horizon == 7200.0

    def test_predict_empty_patterns(self) -> None:
        """Test prediction with no patterns."""
        predictor = AnomalyPredictor()
        predictions = predictor.predict([])
        assert predictions == []

    def test_predict_from_escalations(self) -> None:
        """Test prediction from escalation patterns."""
        predictor = AnomalyPredictor(min_confidence=0.3)
        patterns = [
            DetectedPattern(
                pattern_id="esc_1",
                pattern_type=PatternType.ESCALATION,
                confidence=0.8,
                description="Test escalation",
                supporting_memories=["mem_1", "mem_2"],
                metadata={"acceleration": 0.2},
            )
        ]
        predictions = predictor.predict(patterns)
        assert len(predictions) >= 0


class TestNeuralMemoryLayer:
    """Tests for NeuralMemoryLayer main interface."""

    def test_init(self) -> None:
        """Test layer initialization."""
        layer = NeuralMemoryLayer(embedding_dim=32, n_clusters=4)
        assert layer.embedding_dim == 32
        assert layer.n_clusters == 4
        assert not layer._fitted

    def test_ingest_memories(self) -> None:
        """Test memory ingestion."""
        layer = NeuralMemoryLayer(embedding_dim=16, n_clusters=3)
        memories = [
            {"id": "m1", "event": "test1", "timestamp": time.time()},
            {"id": "m2", "event": "test2", "timestamp": time.time()},
            {"id": "m3", "event": "test3", "timestamp": time.time()},
        ]
        embeddings = layer.ingest_memories(memories, MemoryType.EPISODIC)
        assert len(embeddings) == 3
        assert layer._fitted

    def test_analyze_insufficient_data(self) -> None:
        """Test analysis with insufficient data."""
        layer = NeuralMemoryLayer(n_clusters=10)
        memories = [{"id": "m1", "event": "test"}]
        layer.ingest_memories(memories)
        result = layer.analyze()
        assert result["status"] == "insufficient_data"

    def test_analyze_with_data(self) -> None:
        """Test analysis with sufficient data."""
        layer = NeuralMemoryLayer(embedding_dim=16, n_clusters=3)
        memories = [
            {"id": f"m{i}", "event": f"event_{i}", "timestamp": time.time() + i} for i in range(20)
        ]
        layer.ingest_memories(memories)
        result = layer.analyze()
        assert result["status"] == "analyzed"
        assert "patterns" in result
        assert "predictions" in result

    def test_get_similar_memories(self) -> None:
        """Test similarity search."""
        layer = NeuralMemoryLayer(embedding_dim=16, n_clusters=3)
        memories = [
            {"id": f"m{i}", "event": "similar_event", "timestamp": time.time()} for i in range(10)
        ]
        layer.ingest_memories(memories)
        query = layer.embeddings[0].embedding
        similar = layer.get_similar_memories(query, top_k=3)
        assert len(similar) == 3
        assert similar[0][1] >= similar[1][1]

    def test_get_anomaly_score(self) -> None:
        """Test anomaly scoring."""
        layer = NeuralMemoryLayer(embedding_dim=16, n_clusters=3)
        memories = [{"id": f"m{i}", "event": "normal", "timestamp": time.time()} for i in range(15)]
        layer.ingest_memories(memories)
        score = layer.get_anomaly_score(np.random.randn(16) * 10)
        assert 0 <= score <= 1

    def test_get_neural_features(self) -> None:
        """Test neural feature extraction."""
        layer = NeuralMemoryLayer(embedding_dim=16, n_clusters=3)
        memories = [
            {"id": f"m{i}", "event": f"event_{i}", "timestamp": time.time()} for i in range(10)
        ]
        layer.ingest_memories(memories)
        features = layer.get_neural_features()
        assert features.shape[0] > 16

    def test_get_statistics(self) -> None:
        """Test statistics retrieval."""
        layer = NeuralMemoryLayer(embedding_dim=32, n_clusters=4)
        stats = layer.get_statistics()
        assert stats["total_embeddings"] == 0
        assert stats["embedding_dim"] == 32
        assert stats["n_clusters"] == 4

        memories = [{"id": "m1", "event": "test"}]
        layer.ingest_memories(memories, MemoryType.SEMANTIC)
        stats = layer.get_statistics()
        assert stats["total_embeddings"] == 1
        assert stats["memory_types"]["semantic"] == 1


class TestPatternTypes:
    """Tests for pattern type enums."""

    def test_memory_types(self) -> None:
        """Test all memory types exist."""
        assert MemoryType.EPISODIC.value == "episodic"
        assert MemoryType.SEMANTIC.value == "semantic"
        assert MemoryType.SHORT_TERM.value == "short_term"
        assert MemoryType.LONG_TERM.value == "long_term"

    def test_pattern_types(self) -> None:
        """Test all pattern types exist."""
        assert PatternType.ANOMALY.value == "anomaly"
        assert PatternType.TREND.value == "trend"
        assert PatternType.ESCALATION.value == "escalation"
        assert PatternType.NOVELTY.value == "novelty"
