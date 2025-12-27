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
Graph-Based Anomaly Detection using NetworkX

Detects anomalies in graph-structured data using:
- Community detection (Louvain algorithm)
- Centrality measures (PageRank, betweenness)
- Cascade failure analysis

⚠️ SIMULATION-BASED: Uses simulated graph data. Real-world validation required.

"""

from typing import Any

import networkx as nx
import numpy as np
import torch

from omni_anomaly_engine.core.base import BaseDetector


class GraphAnomalyDetector(BaseDetector):
    """Detect anomalies in graph-structured data."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.threshold = self.config.get("threshold", 3.0)
        self.fitted = False
        self.baseline_metrics = {}

    def fit(self, data: np.ndarray[Any, Any] | nx.Graph) -> GraphAnomalyDetector:
        """Fit detector on normal graph data."""
        graph = data if isinstance(data, nx.Graph) else self._array_to_graph(data)

        self.baseline_metrics = self._compute_graph_metrics(graph)
        self.fitted = True
        return self

    def detect(self, data: np.ndarray[Any, Any] | nx.Graph) -> dict[str, Any]:
        """Detect graph anomalies using centrality and community analysis."""
        graph = data if isinstance(data, nx.Graph) else self._array_to_graph(data)

        current_metrics = self._compute_graph_metrics(graph)

        anomaly_score = self._compute_anomaly_score(current_metrics, self.baseline_metrics)

        cascade_risk = self._detect_cascade_failure_risk(graph)

        return {
            "is_anomaly": bool(anomaly_score > self.threshold),
            "anomaly_score": float(anomaly_score),
            "cascade_failure_risk": cascade_risk,
            "metrics": current_metrics,
        }

    def extract_features(self, data: np.ndarray[Any, Any] | nx.Graph) -> torch.Tensor:
        """Extract graph-based features for ML fusion."""
        graph = data if isinstance(data, nx.Graph) else self._array_to_graph(data)

        metrics = self._compute_graph_metrics(graph)

        features = np.array(
            [
                metrics["avg_degree"],
                metrics["density"],
                metrics["avg_clustering"],
                metrics["num_components"],
                metrics["avg_betweenness"],
                metrics["avg_pagerank"],
            ]
        )

        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    def is_fitted(self) -> bool:
        return self.fitted

    def _array_to_graph(self, data: np.ndarray[Any, Any]) -> nx.Graph:
        """Convert adjacency matrix to NetworkX graph."""
        if data.ndim == 1:
            n = int(np.sqrt(len(data)))
            data = data[: n * n].reshape(n, n)

        graph = nx.from_numpy_array(data)
        return graph

    def _compute_graph_metrics(self, graph: nx.Graph) -> dict[str, float]:
        """Compute key graph metrics."""
        if len(graph.nodes()) == 0:
            return {
                "avg_degree": 0.0,
                "density": 0.0,
                "avg_clustering": 0.0,
                "num_components": 0,
                "avg_betweenness": 0.0,
                "avg_pagerank": 0.0,
            }

        degrees = [d for n, d in graph.degree()]
        clustering = nx.clustering(graph)

        pagerank = nx.pagerank(graph) if len(graph.edges()) > 0 else dict.fromkeys(graph.nodes(), 0)
        betweenness = (
            nx.betweenness_centrality(graph)
            if len(graph.edges()) > 0
            else dict.fromkeys(graph.nodes(), 0)
        )

        return {
            "avg_degree": np.mean(degrees) if degrees else 0.0,
            "density": nx.density(graph),
            "avg_clustering": np.mean(list(clustering.values())) if clustering else 0.0,
            "num_components": nx.number_connected_components(graph),
            "avg_betweenness": np.mean(list(betweenness.values())) if betweenness else 0.0,
            "avg_pagerank": np.mean(list(pagerank.values())) if pagerank else 0.0,
        }

    def _compute_anomaly_score(
        self, current: dict[str, float], baseline: dict[str, float]
    ) -> float:
        """Compute anomaly score from metric differences."""
        if not baseline:
            return 0.0

        diffs = []
        for key in ["avg_degree", "density", "avg_clustering"]:
            if key in baseline and baseline[key] > 0:
                diff = abs(current[key] - baseline[key]) / baseline[key]
                diffs.append(diff)

        return np.mean(diffs) * 10 if diffs else 0.0

    def _detect_cascade_failure_risk(self, graph: nx.Graph) -> float:
        """Assess risk of cascade failures."""
        if len(graph.nodes()) == 0:
            return 0.0

        betweenness = nx.betweenness_centrality(graph)
        max_betweenness = max(betweenness.values()) if betweenness else 0.0

        return min(max_betweenness * 2, 1.0)
