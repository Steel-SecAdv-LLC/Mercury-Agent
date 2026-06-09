# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for GraphAnomalyDetector."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("torch")

import networkx as nx
import numpy as np
import pytest
import torch

from omni_mercury_engine.detectors.graph_based import GraphAnomalyDetector


class TestGraphAnomalyDetector:
    """Tests for GraphAnomalyDetector."""

    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing."""
        G = nx.barabasi_albert_graph(50, 3, seed=42)
        return G

    @pytest.fixture
    def adjacency_matrix(self, sample_graph: Any) -> Any:
        """Get adjacency matrix from sample graph."""
        return nx.to_numpy_array(sample_graph)

    @pytest.fixture
    def empty_graph(self):
        """Create an empty graph."""
        return nx.Graph()

    def test_initialization_default(self) -> None:
        """Test initialization with default config."""
        detector = GraphAnomalyDetector()
        assert detector.threshold == 3.0
        assert not detector.fitted
        assert detector.baseline_metrics == {}

    def test_initialization_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = {"threshold": 2.0}
        detector = GraphAnomalyDetector(config=config)
        assert detector.threshold == 2.0

    def test_fit_with_graph(self, sample_graph: Any) -> None:
        """Test fitting with NetworkX graph."""
        detector = GraphAnomalyDetector()
        result = detector.fit(sample_graph)

        assert result is detector
        assert detector.fitted
        assert detector.baseline_metrics != {}
        assert "avg_degree" in detector.baseline_metrics
        assert "density" in detector.baseline_metrics
        assert "avg_clustering" in detector.baseline_metrics

    def test_fit_with_adjacency_matrix(self, adjacency_matrix: Any) -> None:
        """Test fitting with adjacency matrix."""
        detector = GraphAnomalyDetector()
        result = detector.fit(adjacency_matrix)

        assert result is detector
        assert detector.fitted

    def test_is_fitted(self, sample_graph: Any) -> None:
        """Test is_fitted method."""
        detector = GraphAnomalyDetector()
        assert not detector.is_fitted()

        detector.fit(sample_graph)
        assert detector.is_fitted()

    def test_detect_with_graph(self, sample_graph: Any) -> None:
        """Test detection with NetworkX graph."""
        detector = GraphAnomalyDetector()
        detector.fit(sample_graph)
        result = detector.detect(sample_graph)

        assert "is_anomaly" in result
        assert "anomaly_score" in result
        assert "cascade_failure_risk" in result
        assert "metrics" in result
        assert isinstance(result["is_anomaly"], bool)
        assert isinstance(result["anomaly_score"], float)

    def test_detect_with_adjacency_matrix(self, adjacency_matrix: Any) -> None:
        """Test detection with adjacency matrix."""
        detector = GraphAnomalyDetector()
        detector.fit(adjacency_matrix)
        result = detector.detect(adjacency_matrix)

        assert "is_anomaly" in result
        assert "anomaly_score" in result

    def test_detect_identifies_structural_changes(self, sample_graph: Any) -> None:
        """Test that structural changes are detected."""
        detector = GraphAnomalyDetector(config={"threshold": 0.5})
        detector.fit(sample_graph)

        # Create a significantly different graph
        different_graph = nx.complete_graph(50)
        result = detector.detect(different_graph)

        # Should have a higher anomaly score
        assert result["anomaly_score"] > 0

    def test_detect_no_anomaly_same_graph(self, sample_graph: Any) -> None:
        """Test that same graph gives low anomaly score."""
        detector = GraphAnomalyDetector()
        detector.fit(sample_graph)
        result = detector.detect(sample_graph)

        # Same graph should have very low anomaly score
        assert result["anomaly_score"] < 0.1

    def test_extract_features(self, sample_graph: Any) -> None:
        """Test feature extraction."""
        detector = GraphAnomalyDetector()
        features = detector.extract_features(sample_graph)

        assert isinstance(features, torch.Tensor)
        # Should have 6 features
        assert features.shape == (1, 6)

    def test_extract_features_adjacency_matrix(self, adjacency_matrix: Any) -> None:
        """Test feature extraction from adjacency matrix."""
        detector = GraphAnomalyDetector()
        features = detector.extract_features(adjacency_matrix)

        assert isinstance(features, torch.Tensor)
        assert features.shape[1] == 6

    def test_array_to_graph(self, adjacency_matrix: Any) -> None:
        """Test adjacency matrix to graph conversion."""
        detector = GraphAnomalyDetector()
        graph = detector._array_to_graph(adjacency_matrix)

        assert isinstance(graph, nx.Graph)
        assert len(graph.nodes()) == adjacency_matrix.shape[0]

    def test_array_to_graph_1d(self) -> None:
        """Test 1D array conversion (flattened adjacency matrix)."""
        detector = GraphAnomalyDetector()
        # 4x4 matrix flattened
        flat_adj = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0])
        graph = detector._array_to_graph(flat_adj)

        assert isinstance(graph, nx.Graph)
        assert len(graph.nodes()) == 4

    def test_compute_graph_metrics(self, sample_graph: Any) -> None:
        """Test graph metrics computation."""
        detector = GraphAnomalyDetector()
        metrics = detector._compute_graph_metrics(sample_graph)

        assert "avg_degree" in metrics
        assert "density" in metrics
        assert "avg_clustering" in metrics
        assert "num_components" in metrics
        assert "avg_betweenness" in metrics
        assert "avg_pagerank" in metrics

        assert metrics["avg_degree"] > 0
        assert 0 <= metrics["density"] <= 1
        assert 0 <= metrics["avg_clustering"] <= 1
        assert metrics["num_components"] >= 1

    def test_compute_graph_metrics_empty_graph(self, empty_graph: Any) -> None:
        """Test metrics computation for empty graph."""
        detector = GraphAnomalyDetector()
        metrics = detector._compute_graph_metrics(empty_graph)

        assert metrics["avg_degree"] == 0.0
        assert metrics["density"] == 0.0
        assert metrics["avg_clustering"] == 0.0
        assert metrics["num_components"] == 0

    def test_compute_anomaly_score(self) -> None:
        """Test anomaly score computation from metric differences."""
        detector = GraphAnomalyDetector()

        baseline = {"avg_degree": 4.0, "density": 0.1, "avg_clustering": 0.3}
        current = {"avg_degree": 8.0, "density": 0.2, "avg_clustering": 0.6}

        score = detector._compute_anomaly_score(current, baseline)

        assert score > 0
        assert isinstance(score, float)

    def test_compute_anomaly_score_empty_baseline(self) -> None:
        """Test anomaly score with empty baseline returns 0."""
        detector = GraphAnomalyDetector()

        current = {"avg_degree": 4.0, "density": 0.1, "avg_clustering": 0.3}
        score = detector._compute_anomaly_score(current, {})

        assert score == 0.0

    def test_detect_cascade_failure_risk(self, sample_graph: Any) -> None:
        """Test cascade failure risk assessment."""
        detector = GraphAnomalyDetector()
        risk = detector._detect_cascade_failure_risk(sample_graph)

        assert 0 <= risk <= 1

    def test_detect_cascade_failure_risk_empty_graph(self, empty_graph: Any) -> None:
        """Test cascade risk for empty graph."""
        detector = GraphAnomalyDetector()
        risk = detector._detect_cascade_failure_risk(empty_graph)

        assert risk == 0.0

    def test_detect_cascade_failure_star_graph(self) -> None:
        """Test cascade risk for star graph (high centrality)."""
        # Star graphs have one central node with high betweenness
        star_graph = nx.star_graph(20)
        detector = GraphAnomalyDetector()
        risk = detector._detect_cascade_failure_risk(star_graph)

        # Star graphs should have high cascade risk due to central node
        assert risk > 0.5


class TestGraphMetricsAccuracy:
    """Tests for accuracy of graph metrics calculations."""

    def test_degree_calculation(self) -> None:
        """Test average degree calculation accuracy."""
        # Simple graph: 4 nodes, each connected to 2 others
        G = nx.cycle_graph(4)
        detector = GraphAnomalyDetector()
        metrics = detector._compute_graph_metrics(G)

        # Each node in a cycle has degree 2
        assert abs(metrics["avg_degree"] - 2.0) < 1e-6

    def test_density_calculation(self) -> None:
        """Test density calculation accuracy."""
        # Complete graph of 4 nodes
        G = nx.complete_graph(4)
        detector = GraphAnomalyDetector()
        metrics = detector._compute_graph_metrics(G)

        # Complete graph has density 1.0
        assert abs(metrics["density"] - 1.0) < 1e-6

    def test_clustering_calculation(self) -> None:
        """Test clustering coefficient calculation."""
        # Complete graph has clustering 1.0
        G = nx.complete_graph(5)
        detector = GraphAnomalyDetector()
        metrics = detector._compute_graph_metrics(G)

        assert abs(metrics["avg_clustering"] - 1.0) < 1e-6

    def test_components_calculation(self) -> None:
        """Test connected components calculation."""
        # Create graph with 2 components
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])  # Component 1
        G.add_edges_from([(3, 4), (4, 5)])  # Component 2

        detector = GraphAnomalyDetector()
        metrics = detector._compute_graph_metrics(G)

        assert metrics["num_components"] == 2
