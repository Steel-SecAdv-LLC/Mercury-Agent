"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

from __future__ import annotations

"""
Knowledge Graph Engine - Production Implementation

Provides graph-based knowledge storage for neuro-symbolic reasoning:
- Typed nodes with learned embeddings
- GNN-style message passing for representation learning
- Random walk embeddings (DeepWalk/Node2Vec inspired)
- Link prediction with embedding similarity
- PageRank for node importance
- Semantic clustering via spectral methods

Research Sources:
- Perozzi et al. (2014): DeepWalk - Online Learning of Social Representations
- Grover & Leskovec (2016): node2vec - Scalable Feature Learning for Networks
- Kipf & Welling (2017): Semi-Supervised Classification with GCNs
- Page et al. (1999): The PageRank Citation Ranking
"""

import logging
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of knowledge nodes."""

    CONCEPT = "concept"
    ENTITY = "entity"
    EVENT = "event"
    PROPERTY = "property"
    RELATION = "relation"
    RULE = "rule"
    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    EVIDENCE = "evidence"
    INDICATOR = "indicator"
    ANOMALY = "anomaly"
    PATTERN = "pattern"


class EdgeType(Enum):
    """Types of relationships between nodes."""

    IS_A = "is_a"  # Taxonomy
    PART_OF = "part_of"  # Mereology
    CAUSES = "causes"  # Causation
    CORRELATES = "correlates"  # Correlation
    PRECEDES = "precedes"  # Temporal
    IMPLIES = "implies"  # Logical implication
    CONTRADICTS = "contradicts"  # Logical contradiction
    SUPPORTS = "supports"  # Evidence support
    REFUTES = "refutes"  # Evidence refutation
    SIMILAR_TO = "similar_to"  # Similarity
    INSTANCE_OF = "instance_of"  # Class membership
    HAS_PROPERTY = "has_property"  # Attribution
    CO_OCCURS = "co_occurs"  # Co-occurrence
    TRIGGERS = "triggers"  # Trigger relationship


@dataclass
class KnowledgeNode:
    """A node in the knowledge graph."""

    node_id: str
    node_type: NodeType
    label: str
    attributes: dict[str, Any] = field(default_factory=dict)
    embedding: np.ndarray[Any, Any] | None = None
    confidence: float = 1.0
    source: str = "system"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    activation: float = 0.0  # Current activation level
    pagerank: float = 0.0  # PageRank score
    cluster_id: int = -1  # Cluster assignment

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type.value,
            "label": self.label,
            "attributes": self.attributes,
            "confidence": self.confidence,
            "source": self.source,
            "activation": self.activation,
            "pagerank": self.pagerank,
            "cluster": self.cluster_id,
        }


@dataclass
class KnowledgeEdge:
    """An edge in the knowledge graph."""

    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    attributes: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source: str = "system"
    created_at: float = field(default_factory=time.time)
    bidirectional: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "type": self.edge_type.value,
            "weight": self.weight,
            "confidence": self.confidence,
            "bidirectional": self.bidirectional,
        }


@dataclass
class TraversalResult:
    """Result of a graph traversal."""

    path: list[str]
    nodes: list[KnowledgeNode]
    edges: list[KnowledgeEdge]
    total_weight: float
    total_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "weight": self.total_weight,
            "confidence": self.total_confidence,
        }


class RandomWalkEmbedding:
    """
    Learn node embeddings via random walks (DeepWalk/Node2Vec inspired).

    Uses truncated random walks to sample node context,
    then learns embeddings via skip-gram with negative sampling.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        walk_length: int = 10,
        num_walks: int = 20,
        window_size: int = 5,
        p: float = 1.0,  # Node2Vec return parameter
        q: float = 1.0,  # Node2Vec in-out parameter
        learning_rate: float = 0.025,
        negative_samples: int = 5,
    ):
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.p = p
        self.q = q
        self.learning_rate = learning_rate
        self.negative_samples = negative_samples

        self.embeddings: dict[str, np.ndarray[Any, Any]] = {}
        self._node_to_idx: dict[str, int] = {}
        self._idx_to_node: dict[int, str] = {}

    def fit(
        self,
        adjacency: dict[str, list[tuple[str, float]]],
        node_ids: list[str],
    ) -> dict[str, np.ndarray[Any, Any]]:
        """
        Learn embeddings from graph structure.

        Args:
            adjacency: Node -> [(neighbor, weight), ...]
            node_ids: List of all node IDs

        Returns:
            Dictionary of node_id -> embedding
        """
        if not node_ids:
            return {}

        # Build index
        self._node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        self._idx_to_node = {i: nid for i, nid in enumerate(node_ids)}
        n_nodes = len(node_ids)

        # Initialize embeddings randomly
        scale = 0.5 / self.embedding_dim
        input_embeddings = np.random.randn(n_nodes, self.embedding_dim) * scale
        output_embeddings = np.random.randn(n_nodes, self.embedding_dim) * scale

        # Generate random walks
        walks = self._generate_walks(adjacency, node_ids)

        # Train with skip-gram
        for walk in walks:
            self._train_skip_gram(walk, input_embeddings, output_embeddings, n_nodes)

        # Store final embeddings
        for nid in node_ids:
            idx = self._node_to_idx[nid]
            # Combine input and output embeddings
            self.embeddings[nid] = (input_embeddings[idx] + output_embeddings[idx]) / 2

        return self.embeddings

    def _generate_walks(
        self,
        adjacency: dict[str, list[tuple[str, float]]],
        node_ids: list[str],
    ) -> list[list[int]]:
        """Generate random walks from all nodes."""
        walks = []

        for _ in range(self.num_walks):
            np.random.shuffle(node_ids)
            for start in node_ids:
                walk = self._random_walk(adjacency, start)
                if len(walk) > 1:
                    walks.append([self._node_to_idx[n] for n in walk])

        return walks

    def _random_walk(
        self,
        adjacency: dict[str, list[tuple[str, float]]],
        start: str,
    ) -> list[str]:
        """Execute a single random walk."""
        walk = [start]
        prev = None

        for _ in range(self.walk_length - 1):
            current = walk[-1]
            neighbors = adjacency.get(current, [])

            if not neighbors:
                break

            # Compute transition probabilities (Node2Vec biased walk)
            weights = []
            for neighbor, edge_weight in neighbors:
                if neighbor == prev:
                    # Return to previous node
                    weights.append(edge_weight / self.p)
                elif prev and neighbor in dict(adjacency.get(prev, [])):
                    # Neighbor of previous node (BFS-like)
                    weights.append(edge_weight)
                else:
                    # Far from previous (DFS-like)
                    weights.append(edge_weight / self.q)

            if not weights:
                break

            # Normalize and sample
            probs = np.array(weights)
            probs = probs / probs.sum()

            next_idx = np.random.choice(len(neighbors), p=probs)
            next_node = neighbors[next_idx][0]

            prev = current
            walk.append(next_node)

        return walk

    def _train_skip_gram(
        self,
        walk: list[int],
        input_emb: np.ndarray[Any, Any],
        output_emb: np.ndarray[Any, Any],
        n_nodes: int,
    ):
        """Train skip-gram on a single walk."""
        for i, center in enumerate(walk):
            # Context window
            start = max(0, i - self.window_size)
            end = min(len(walk), i + self.window_size + 1)

            for j in range(start, end):
                if i == j:
                    continue

                context = walk[j]

                # Positive sample
                self._sgd_update(center, context, 1, input_emb, output_emb)

                # Negative samples
                for _ in range(self.negative_samples):
                    neg = np.random.randint(n_nodes)
                    if neg != center and neg != context:
                        self._sgd_update(center, neg, 0, input_emb, output_emb)

    def _sgd_update(
        self,
        center: int,
        context: int,
        label: int,
        input_emb: np.ndarray[Any, Any],
        output_emb: np.ndarray[Any, Any],
    ):
        """Single SGD update for skip-gram."""
        # Sigmoid
        z = np.dot(input_emb[center], output_emb[context])
        z = np.clip(z, -10, 10)
        pred = 1 / (1 + np.exp(-z))

        # Gradient
        error = (label - pred) * self.learning_rate

        # Update
        input_emb[center] += error * output_emb[context]
        output_emb[context] += error * input_emb[center]


class GNNMessagePassing:
    """
    Graph Neural Network message passing for representation learning.

    Implements simplified GCN-style aggregation:
    h_v = σ(W * AGGREGATE({h_u : u ∈ N(v)}))
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        aggregation: str = "mean",  # "mean", "sum", "max"
        activation: str = "relu",
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.aggregation = aggregation
        self.activation = activation

        self._weights: list[np.ndarray[Any, Any]] = []

    def forward(
        self,
        node_features: np.ndarray[Any, Any],
        adjacency: sparse.spmatrix,
        normalize: bool = True,
    ) -> np.ndarray[Any, Any]:
        """
        Forward pass through GNN layers.

        Args:
            node_features: Initial node features (n_nodes, feature_dim)
            adjacency: Sparse adjacency matrix
            normalize: Whether to apply symmetric normalization

        Returns:
            Updated node embeddings (n_nodes, hidden_dim)
        """
        feature_dim = node_features.shape[1]

        # Initialize weights if needed
        if not self._weights:
            self._init_weights(feature_dim)

        # Normalize adjacency (GCN-style)
        if normalize:
            adj = self._normalize_adjacency(adjacency)
        else:
            adj = adjacency

        # Message passing layers
        h = node_features

        for layer in range(self.num_layers):
            # Aggregate messages
            if self.aggregation == "mean":
                messages = adj @ h
            elif self.aggregation == "sum":
                messages = adjacency @ h
            else:  # max
                messages = self._max_aggregate(adjacency, h)

            # Transform
            h = messages @ self._weights[layer]

            # Activation (except last layer)
            if layer < self.num_layers - 1:
                if self.activation == "relu":
                    h = np.maximum(0, h)
                elif self.activation == "tanh":
                    h = np.tanh(h)

        return h

    def _init_weights(self, input_dim: int) -> None:
        """Initialize weight matrices with Xavier initialization."""
        dims = [input_dim] + [self.hidden_dim] * self.num_layers

        for i in range(self.num_layers):
            # Xavier initialization
            scale = np.sqrt(2.0 / (dims[i] + dims[i + 1]))
            W = np.random.randn(dims[i], dims[i + 1]) * scale
            self._weights.append(W)

    def _normalize_adjacency(
        self,
        adjacency: sparse.spmatrix,
    ) -> sparse.spmatrix:
        """Symmetric normalization: D^(-1/2) A D^(-1/2)."""
        # Add self-loops
        n = adjacency.shape[0]
        adj = adjacency + sparse.eye(n)

        # Degree matrix
        degrees = np.array(adj.sum(axis=1)).flatten()
        degrees = np.maximum(degrees, 1e-10)  # Avoid division by zero

        # D^(-1/2)
        d_inv_sqrt = 1.0 / np.sqrt(degrees)
        d_inv_sqrt = sparse.diags(d_inv_sqrt)

        # Symmetric normalization
        return d_inv_sqrt @ adj @ d_inv_sqrt

    def _max_aggregate(
        self,
        adjacency: sparse.spmatrix,
        features: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Max aggregation over neighbors."""
        result = np.zeros_like(features)

        adj_coo = adjacency.tocoo()
        for i, j, v in zip(adj_coo.row, adj_coo.col, adj_coo.data):
            result[i] = np.maximum(result[i], features[j] * v)

        return result


class LinkPredictor:
    """
    Predict missing or future links using embeddings.

    Methods:
    - Dot product similarity
    - Hadamard product + classifier
    - Distance-based scoring
    """

    def __init__(self, method: str = "dot") -> None:
        self.method = method

    def score(
        self,
        source_embedding: np.ndarray[Any, Any],
        target_embedding: np.ndarray[Any, Any],
    ) -> float:
        """
        Score a potential link.

        Args:
            source_embedding: Source node embedding
            target_embedding: Target node embedding

        Returns:
            Link probability/score
        """
        if self.method == "dot":
            score = np.dot(source_embedding, target_embedding)
            # Sigmoid to get probability
            return float(1 / (1 + np.exp(-np.clip(score, -10, 10))))

        elif self.method == "cosine":
            norm_s = np.linalg.norm(source_embedding)
            norm_t = np.linalg.norm(target_embedding)
            if norm_s == 0 or norm_t == 0:
                return 0.0
            return float(np.dot(source_embedding, target_embedding) / (norm_s * norm_t))

        elif self.method == "distance":
            dist = np.linalg.norm(source_embedding - target_embedding)
            # Convert distance to similarity
            return float(1 / (1 + dist))

        else:
            return 0.0

    def predict_links(
        self,
        embeddings: dict[str, np.ndarray[Any, Any]],
        candidate_pairs: list[tuple[str, str]],
        threshold: float = 0.5,
    ) -> list[tuple[str, str, float]]:
        """
        Predict links for candidate pairs.

        Args:
            embeddings: Node embeddings
            candidate_pairs: List of (source, target) pairs to evaluate
            threshold: Score threshold for prediction

        Returns:
            List of (source, target, score) for predicted links
        """
        predictions = []

        for source, target in candidate_pairs:
            if source not in embeddings or target not in embeddings:
                continue

            score = self.score(embeddings[source], embeddings[target])
            if score >= threshold:
                predictions.append((source, target, score))

        predictions.sort(key=lambda x: x[2], reverse=True)
        return predictions


class KnowledgeGraph:
    """
    Production Knowledge Graph for neuro-symbolic reasoning.

    Features:
    1. Random Walk Embeddings (DeepWalk/Node2Vec)
       - Learn node representations from graph structure
       - Capture local and global graph properties

    2. GNN Message Passing
       - Aggregate neighborhood information
       - Learn from node features and structure

    3. Link Prediction
       - Predict missing relationships
       - Score potential new edges

    4. PageRank Importance
       - Identify important nodes
       - Weight spreading activation

    5. Spectral Clustering
       - Group similar nodes
       - Identify communities

    Architecture follows Nucleoid's logic graph principles while
    extending for production neuro-symbolic AI.
    """

    def __init__(
        self,
        enable_embeddings: bool = True,
        embedding_dim: int = 64,
        activation_decay: float = 0.1,
        gnn_layers: int = 2,
    ):
        """
        Initialize Knowledge Graph.

        Args:
            enable_embeddings: Enable vector embeddings for nodes
            embedding_dim: Dimension of node embeddings
            activation_decay: Decay rate for spreading activation
            gnn_layers: Number of GNN layers for message passing
        """
        self.enable_embeddings = enable_embeddings
        self.embedding_dim = embedding_dim
        self.activation_decay = activation_decay

        # Core storage
        self._nodes: dict[str, KnowledgeNode] = {}
        self._edges: dict[str, list[KnowledgeEdge]] = defaultdict[str, list[Any]](list)
        self._reverse_edges: dict[str, list[KnowledgeEdge]] = defaultdict[str, list[Any]](list)
        self._type_index: dict[NodeType, set[str]] = defaultdict[str, set[Any]](set)
        self._edge_type_index: dict[EdgeType, list[KnowledgeEdge]] = defaultdict[str, list[Any]](
            list
        )

        # Embedding components
        self._random_walk = RandomWalkEmbedding(embedding_dim=embedding_dim)
        self._gnn = GNNMessagePassing(hidden_dim=embedding_dim, num_layers=gnn_layers)
        self._link_predictor = LinkPredictor(method="dot")

        # Cached computations
        self._pagerank_computed = False
        self._embeddings_computed = False

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self._stats = {
            "nodes_added": 0,
            "edges_added": 0,
            "queries": 0,
            "traversals": 0,
            "embeddings_computed": 0,
        }

        logger.info(
            f"KnowledgeGraph initialized (embeddings={enable_embeddings}, dim={embedding_dim})"
        )

    def add_node(
        self,
        node_id: str,
        node_type: NodeType,
        label: str,
        attributes: dict[str, Any] | None = None,
        embedding: np.ndarray[Any, Any] | None = None,
        confidence: float = 1.0,
        source: str = "system",
    ) -> KnowledgeNode:
        """Add a node to the knowledge graph."""
        with self._lock:
            node = KnowledgeNode(
                node_id=node_id,
                node_type=node_type,
                label=label,
                attributes=attributes or {},
                embedding=embedding,
                confidence=confidence,
                source=source,
            )

            if node_id in self._nodes:
                node.created_at = self._nodes[node_id].created_at
            else:
                self._stats["nodes_added"] += 1

            self._nodes[node_id] = node
            self._type_index[node_type].add(node_id)

            # Invalidate caches
            self._pagerank_computed = False
            self._embeddings_computed = False

            return node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        weight: float = 1.0,
        attributes: dict[str, Any] | None = None,
        confidence: float = 1.0,
        bidirectional: bool = False,
    ) -> KnowledgeEdge | None:
        """Add an edge between nodes."""
        with self._lock:
            if source_id not in self._nodes or target_id not in self._nodes:
                logger.warning("Cannot add edge: node(s) not found")
                return None

            edge = KnowledgeEdge(
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                weight=weight,
                attributes=attributes or {},
                confidence=confidence,
                bidirectional=bidirectional,
            )

            self._edges[source_id].append(edge)
            self._reverse_edges[target_id].append(edge)
            self._edge_type_index[edge_type].append(edge)
            self._stats["edges_added"] += 1

            if bidirectional:
                reverse_edge = KnowledgeEdge(
                    source_id=target_id,
                    target_id=source_id,
                    edge_type=edge_type,
                    weight=weight,
                    attributes=attributes or {},
                    confidence=confidence,
                    bidirectional=True,
                )
                self._edges[target_id].append(reverse_edge)
                self._reverse_edges[source_id].append(reverse_edge)

            # Invalidate caches
            self._pagerank_computed = False
            self._embeddings_computed = False

            return edge

    def get_node(self, node_id: str) -> KnowledgeNode | None:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def get_neighbors(
        self,
        node_id: str,
        edge_types: list[EdgeType] | None = None,
        direction: str = "outgoing",
    ) -> list[tuple[KnowledgeNode, KnowledgeEdge]]:
        """Get neighboring nodes."""
        with self._lock:
            self._stats["queries"] += 1
            neighbors = []

            if direction in ("outgoing", "both"):
                for edge in self._edges.get(node_id, []):
                    if edge_types is None or edge.edge_type in edge_types:
                        neighbor = self._nodes.get(edge.target_id)
                        if neighbor:
                            neighbors.append((neighbor, edge))

            if direction in ("incoming", "both"):
                for edge in self._reverse_edges.get(node_id, []):
                    if edge_types is None or edge.edge_type in edge_types:
                        neighbor = self._nodes.get(edge.source_id)
                        if neighbor:
                            neighbors.append((neighbor, edge))

            return neighbors

    def find_by_type(self, node_type: NodeType) -> list[KnowledgeNode]:
        """Find all nodes of a specific type."""
        with self._lock:
            return [
                self._nodes[nid]
                for nid in self._type_index.get(node_type, set())
                if nid in self._nodes
            ]

    def compute_embeddings(self, method: str = "random_walk") -> dict[str, np.ndarray[Any, Any]]:
        """
        Compute node embeddings.

        Args:
            method: "random_walk" or "gnn"

        Returns:
            Dictionary of node_id -> embedding
        """
        with self._lock:
            if not self._nodes:
                return {}

            node_ids = list(self._nodes.keys())

            if method == "random_walk":
                # Build adjacency for random walk
                adjacency = {}
                for nid in node_ids:
                    neighbors = []
                    for edge in self._edges.get(nid, []):
                        neighbors.append((edge.target_id, edge.weight))
                    adjacency[nid] = neighbors

                embeddings = self._random_walk.fit(adjacency, node_ids)

            else:  # gnn
                # Build sparse adjacency and feature matrix
                n_nodes = len(node_ids)
                node_to_idx = {nid: i for i, nid in enumerate(node_ids)}

                # Build adjacency matrix
                rows, cols, data = [], [], []
                for nid in node_ids:
                    i = node_to_idx[nid]
                    for edge in self._edges.get(nid, []):
                        if edge.target_id in node_to_idx:
                            j = node_to_idx[edge.target_id]
                            rows.append(i)
                            cols.append(j)
                            data.append(edge.weight)

                adj = sparse.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

                # Build feature matrix (use existing embeddings or random)
                features = np.zeros((n_nodes, self.embedding_dim))
                for nid, node in self._nodes.items():
                    i = node_to_idx[nid]
                    if node.embedding is not None:
                        features[i] = node.embedding[: self.embedding_dim]
                    else:
                        features[i] = np.random.randn(self.embedding_dim) * 0.1

                # GNN forward pass
                new_embeddings = self._gnn.forward(features, adj)

                embeddings = {nid: new_embeddings[node_to_idx[nid]] for nid in node_ids}

            # Store in nodes
            for nid, emb in embeddings.items():
                if nid in self._nodes:
                    self._nodes[nid].embedding = emb

            self._embeddings_computed = True
            self._stats["embeddings_computed"] += 1

            return embeddings

    def compute_pagerank(
        self,
        damping: float = 0.85,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> dict[str, float]:
        """
        Compute PageRank for all nodes.

        Args:
            damping: Damping factor (probability of following a link)
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance

        Returns:
            Dictionary of node_id -> pagerank score
        """
        with self._lock:
            if not self._nodes:
                return {}

            node_ids = list(self._nodes.keys())
            n_nodes = len(node_ids)
            node_to_idx = {nid: i for i, nid in enumerate(node_ids)}

            # Initialize PageRank uniformly
            pagerank = np.ones(n_nodes) / n_nodes

            # Build transition matrix
            out_degrees = np.zeros(n_nodes)
            for i, nid in enumerate(node_ids):
                out_degrees[i] = len(self._edges.get(nid, []))

            # Handle dangling nodes
            dangling = out_degrees == 0
            out_degrees[dangling] = 1

            for _ in range(max_iterations):
                new_pagerank = np.zeros(n_nodes)

                # Contribution from dangling nodes
                dangling_contrib = np.sum(pagerank[dangling]) / n_nodes

                # Random surfer contribution
                new_pagerank += (1 - damping) / n_nodes + damping * dangling_contrib

                # Link contributions
                for i, nid in enumerate(node_ids):
                    for edge in self._edges.get(nid, []):
                        if edge.target_id in node_to_idx:
                            j = node_to_idx[edge.target_id]
                            new_pagerank[j] += damping * pagerank[i] / out_degrees[i]

                # Check convergence
                if np.sum(np.abs(new_pagerank - pagerank)) < tolerance:
                    break

                pagerank = new_pagerank

            # Normalize
            pagerank = pagerank / pagerank.sum()

            # Store in nodes
            result = {}
            for i, nid in enumerate(node_ids):
                result[nid] = float(pagerank[i])
                if nid in self._nodes:
                    self._nodes[nid].pagerank = result[nid]

            self._pagerank_computed = True
            return result

    def spectral_clustering(
        self,
        n_clusters: int = 5,
    ) -> dict[str, int]:
        """
        Cluster nodes using spectral clustering.

        Args:
            n_clusters: Number of clusters

        Returns:
            Dictionary of node_id -> cluster_id
        """
        with self._lock:
            if len(self._nodes) < n_clusters:
                return dict.fromkeys(self._nodes, 0)

            node_ids = list(self._nodes.keys())
            n_nodes = len(node_ids)
            node_to_idx = {nid: i for i, nid in enumerate(node_ids)}

            # Build adjacency matrix
            rows, cols, data = [], [], []
            for nid in node_ids:
                i = node_to_idx[nid]
                for edge in self._edges.get(nid, []):
                    if edge.target_id in node_to_idx:
                        j = node_to_idx[edge.target_id]
                        rows.extend([i, j])
                        cols.extend([j, i])
                        data.extend([edge.weight, edge.weight])

            if not data:
                return dict.fromkeys(self._nodes, 0)

            adj = sparse.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

            # Compute Laplacian
            degrees = np.array(adj.sum(axis=1)).flatten()
            degrees = np.maximum(degrees, 1e-10)
            d_inv_sqrt = sparse.diags(1.0 / np.sqrt(degrees))
            laplacian = sparse.eye(n_nodes) - d_inv_sqrt @ adj @ d_inv_sqrt

            # Compute eigenvectors
            try:
                k = min(n_clusters, n_nodes - 1)
                eigenvalues, eigenvectors = sparse.linalg.eigsh(laplacian, k=k, which="SM")

                # K-means on eigenvectors
                from scipy.cluster.vq import kmeans2

                centroids, labels = kmeans2(eigenvectors.real, n_clusters, minit="points")
            except Exception:
                labels = np.zeros(n_nodes, dtype=int)

            # Store in nodes
            result = {}
            for i, nid in enumerate(node_ids):
                cluster_id = int(labels[i])
                result[nid] = cluster_id
                if nid in self._nodes:
                    self._nodes[nid].cluster_id = cluster_id

            return result

    def predict_links(
        self,
        top_k: int = 10,
        threshold: float = 0.5,
    ) -> list[tuple[str, str, float]]:
        """
        Predict missing links.

        Args:
            top_k: Number of top predictions to return
            threshold: Minimum score for prediction

        Returns:
            List of (source, target, score) predictions
        """
        with self._lock:
            # Ensure embeddings are computed
            if not self._embeddings_computed:
                self.compute_embeddings()

            embeddings = {
                nid: node.embedding
                for nid, node in self._nodes.items()
                if node.embedding is not None
            }

            if not embeddings:
                return []

            # Generate candidate pairs (non-existing edges)
            existing_edges = set()
            for nid in self._nodes:
                for edge in self._edges.get(nid, []):
                    existing_edges.add((nid, edge.target_id))

            candidates = []
            node_ids = list(embeddings.keys())
            for i, src in enumerate(node_ids):
                for tgt in node_ids[i + 1 :]:
                    if (src, tgt) not in existing_edges and (tgt, src) not in existing_edges:
                        candidates.append((src, tgt))

            # Limit candidates for efficiency
            if len(candidates) > 10000:
                candidates = [
                    candidates[i] for i in np.random.choice(len(candidates), 10000, replace=False)
                ]

            predictions = self._link_predictor.predict_links(embeddings, candidates, threshold)

            return predictions[:top_k]

    def traverse_bfs(
        self,
        start_id: str,
        max_depth: int = 3,
        edge_types: list[EdgeType] | None = None,
        node_filter: Callable[[KnowledgeNode], bool] | None = None,
    ) -> list[TraversalResult]:
        """Breadth-first traversal from a starting node."""
        with self._lock:
            self._stats["traversals"] += 1
            results = []
            visited = {start_id}
            queue = [(start_id, [start_id], [], 0, 1.0, 1.0)]

            while queue:
                current_id, path, edges, depth, total_weight, total_conf = queue.pop(0)

                if depth > 0:
                    node = self._nodes.get(current_id)
                    if node and (node_filter is None or node_filter(node)):
                        results.append(
                            TraversalResult(
                                path=path.copy(),
                                nodes=[self._nodes[pid] for pid in path if pid in self._nodes],
                                edges=edges.copy(),
                                total_weight=total_weight,
                                total_confidence=total_conf,
                            )
                        )

                if depth < max_depth:
                    for neighbor, edge in self.get_neighbors(current_id, edge_types):
                        if neighbor.node_id not in visited:
                            visited.add(neighbor.node_id)
                            new_path = path + [neighbor.node_id]
                            new_edges = edges + [edge]
                            new_weight = total_weight * edge.weight
                            new_conf = total_conf * edge.confidence
                            queue.append(
                                (
                                    neighbor.node_id,
                                    new_path,
                                    new_edges,
                                    depth + 1,
                                    new_weight,
                                    new_conf,
                                )
                            )

            return results

    def find_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5,
        edge_types: list[EdgeType] | None = None,
    ) -> TraversalResult | None:
        """Find shortest path between two nodes."""
        with self._lock:
            self._stats["traversals"] += 1

            if start_id not in self._nodes or end_id not in self._nodes:
                return None

            visited = {start_id}
            queue = [(start_id, [start_id], [], 1.0, 1.0)]

            while queue:
                current_id, path, edges, total_weight, total_conf = queue.pop(0)

                if current_id == end_id:
                    return TraversalResult(
                        path=path,
                        nodes=[self._nodes[pid] for pid in path],
                        edges=edges,
                        total_weight=total_weight,
                        total_confidence=total_conf,
                    )

                if len(path) <= max_depth:
                    for neighbor, edge in self.get_neighbors(current_id, edge_types, "both"):
                        if neighbor.node_id not in visited:
                            visited.add(neighbor.node_id)
                            queue.append(
                                (
                                    neighbor.node_id,
                                    path + [neighbor.node_id],
                                    edges + [edge],
                                    total_weight * edge.weight,
                                    total_conf * edge.confidence,
                                )
                            )

            return None

    def spreading_activation(
        self,
        source_ids: list[str],
        activation_strength: float = 1.0,
        decay_factor: float | None = None,
        max_iterations: int = 3,
        min_activation: float = 0.01,
        use_pagerank: bool = True,
    ) -> dict[str, float]:
        """
        Spreading activation for associative retrieval.

        Enhanced with PageRank weighting for importance-aware activation.
        """
        with self._lock:
            decay = decay_factor or self.activation_decay

            # Compute PageRank if not cached and requested
            if use_pagerank and not self._pagerank_computed:
                self.compute_pagerank()

            activations: dict[str, float] = {}

            # Initialize source nodes
            for node_id in source_ids:
                if node_id in self._nodes:
                    initial = activation_strength
                    if use_pagerank:
                        # Weight by PageRank
                        initial *= 1 + self._nodes[node_id].pagerank * 10
                    activations[node_id] = initial
                    self._nodes[node_id].activation = initial

            # Spread activation
            for _ in range(max_iterations):
                new_activations: dict[str, float] = {}

                for node_id, current_activation in activations.items():
                    if current_activation < min_activation:
                        continue

                    for neighbor, edge in self.get_neighbors(node_id, direction="both"):
                        spread = current_activation * edge.weight * decay
                        if use_pagerank:
                            spread *= 1 + neighbor.pagerank

                        if neighbor.node_id in new_activations:
                            new_activations[neighbor.node_id] = max(
                                new_activations[neighbor.node_id], spread
                            )
                        else:
                            new_activations[neighbor.node_id] = spread

                for node_id, activation in new_activations.items():
                    if node_id not in activations or activation > activations[node_id]:
                        activations[node_id] = activation
                        if node_id in self._nodes:
                            self._nodes[node_id].activation = activation

            return activations

    def find_similar(
        self,
        query_embedding: np.ndarray[Any, Any],
        top_k: int = 10,
        node_type: NodeType | None = None,
    ) -> list[tuple[KnowledgeNode, float]]:
        """Find nodes similar to a query embedding."""
        with self._lock:
            if not self.enable_embeddings:
                return []

            # Ensure embeddings are computed
            if not self._embeddings_computed:
                self.compute_embeddings()

            similarities = []
            query_norm = np.linalg.norm(query_embedding)
            if query_norm == 0:
                return []

            for node in self._nodes.values():
                if node_type and node.node_type != node_type:
                    continue
                if node.embedding is None:
                    continue

                node_norm = np.linalg.norm(node.embedding)
                if node_norm == 0:
                    continue

                similarity = float(
                    np.dot(query_embedding, node.embedding) / (query_norm * node_norm)
                )
                similarities.append((node, similarity))

            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]

    def infer_edges(
        self,
        inference_rules: list[tuple[EdgeType, EdgeType, EdgeType]] | None = None,
    ) -> list[KnowledgeEdge]:
        """Infer new edges using transitivity rules."""
        with self._lock:
            rules = inference_rules or [
                (EdgeType.IS_A, EdgeType.IS_A, EdgeType.IS_A),
                (EdgeType.CAUSES, EdgeType.CAUSES, EdgeType.CAUSES),
                (EdgeType.PART_OF, EdgeType.PART_OF, EdgeType.PART_OF),
                (EdgeType.IMPLIES, EdgeType.IMPLIES, EdgeType.IMPLIES),
            ]

            inferred = []
            for rule in rules:
                edge_type1, edge_type2, inferred_type = rule

                for edge1 in self._edge_type_index.get(edge_type1, []):
                    for edge2 in self._edge_type_index.get(edge_type2, []):
                        if edge1.target_id == edge2.source_id:
                            existing = any(
                                e.target_id == edge2.target_id and e.edge_type == inferred_type
                                for e in self._edges.get(edge1.source_id, [])
                            )
                            if not existing:
                                new_edge = self.add_edge(
                                    edge1.source_id,
                                    edge2.target_id,
                                    inferred_type,
                                    weight=edge1.weight * edge2.weight,
                                    confidence=edge1.confidence * edge2.confidence * 0.9,
                                    attributes={"inferred": True},
                                )
                                if new_edge:
                                    inferred.append(new_edge)

            logger.info(f"Inferred {len(inferred)} new edges")
            return inferred

    def get_subgraph(
        self,
        center_id: str,
        radius: int = 2,
    ) -> dict[str, Any]:
        """Extract a subgraph around a node."""
        with self._lock:
            results = self.traverse_bfs(center_id, max_depth=radius)

            nodes = {center_id: self._nodes[center_id].to_dict()}
            edges = []

            for result in results:
                for node in result.nodes:
                    if node.node_id not in nodes:
                        nodes[node.node_id] = node.to_dict()
                for edge in result.edges:
                    edges.append(edge.to_dict())

            return {"nodes": nodes, "edges": edges}

    def get_statistics(self) -> dict[str, Any]:
        """Get graph statistics."""
        with self._lock:
            return {
                **self._stats,
                "total_nodes": len(self._nodes),
                "total_edges": sum(len(e) for e in self._edges.values()),
                "node_types": {t.value: len(ids) for t, ids in self._type_index.items() if ids},
                "edge_types": {
                    t.value: len(edges) for t, edges in self._edge_type_index.items() if edges
                },
                "embeddings_computed": self._embeddings_computed,
                "pagerank_computed": self._pagerank_computed,
            }

    def export(self) -> dict[str, Any]:
        """Export entire graph for serialization."""
        with self._lock:
            return {
                "nodes": [n.to_dict() for n in self._nodes.values()],
                "edges": [e.to_dict() for edges in self._edges.values() for e in edges],
                "statistics": self.get_statistics(),
            }
