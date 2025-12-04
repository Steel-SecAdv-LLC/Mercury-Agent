"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

"""
Knowledge Graph Engine - Relationship-Based Knowledge Storage

Provides graph-based knowledge storage for neuro-symbolic reasoning:
- Typed nodes with attributes and embeddings
- Typed edges with relationship semantics
- Efficient traversal and querying
- Integration with symbolic reasoning

Research Sources:
- Nucleoid: Logic graph for relationships
- DARPA ANSR: Knowledge representation
- Knowledge Graphs: Survey and research directions
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Callable
from collections import defaultdict

import numpy as np

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


@dataclass
class KnowledgeNode:
    """A node in the knowledge graph."""

    node_id: str
    node_type: NodeType
    label: str
    attributes: dict[str, Any] = field(default_factory=dict)
    embedding: np.ndarray | None = None
    confidence: float = 1.0
    source: str = "system"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    activation: float = 0.0  # Current activation level

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type.value,
            "label": self.label,
            "attributes": self.attributes,
            "confidence": self.confidence,
            "source": self.source,
            "activation": self.activation,
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


class KnowledgeGraph:
    """
    Knowledge Graph for neuro-symbolic reasoning.

    Provides:
    - Semantic storage of concepts, entities, and relationships
    - Efficient traversal algorithms (BFS, DFS, shortest path)
    - Spreading activation for associative retrieval
    - Integration with symbolic inference rules
    - Embedding-based similarity search

    Architecture follows Nucleoid's logic graph principles while
    extending for neuro-symbolic AI requirements.
    """

    def __init__(
        self,
        enable_embeddings: bool = True,
        embedding_dim: int = 128,
        activation_decay: float = 0.1,
    ):
        """
        Initialize Knowledge Graph.

        Args:
            enable_embeddings: Enable vector embeddings for nodes
            embedding_dim: Dimension of node embeddings
            activation_decay: Decay rate for spreading activation
        """
        self.enable_embeddings = enable_embeddings
        self.embedding_dim = embedding_dim
        self.activation_decay = activation_decay

        # Core storage
        self._nodes: dict[str, KnowledgeNode] = {}
        self._edges: dict[str, list[KnowledgeEdge]] = defaultdict(list)
        self._reverse_edges: dict[str, list[KnowledgeEdge]] = defaultdict(list)
        self._type_index: dict[NodeType, set[str]] = defaultdict(set)
        self._edge_type_index: dict[EdgeType, list[KnowledgeEdge]] = defaultdict(list)

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self._stats = {
            "nodes_added": 0,
            "edges_added": 0,
            "queries": 0,
            "traversals": 0,
        }

        logger.info(f"KnowledgeGraph initialized (embeddings={enable_embeddings})")

    def add_node(
        self,
        node_id: str,
        node_type: NodeType,
        label: str,
        attributes: dict[str, Any] | None = None,
        embedding: np.ndarray | None = None,
        confidence: float = 1.0,
        source: str = "system",
    ) -> KnowledgeNode:
        """
        Add a node to the knowledge graph.

        Args:
            node_id: Unique identifier for the node
            node_type: Type of knowledge node
            label: Human-readable label
            attributes: Additional attributes
            embedding: Optional vector embedding
            confidence: Confidence in this knowledge (0-1)
            source: Source of this knowledge

        Returns:
            The created or updated node
        """
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
                # Update existing node
                node.created_at = self._nodes[node_id].created_at
            else:
                self._stats["nodes_added"] += 1

            self._nodes[node_id] = node
            self._type_index[node_type].add(node_id)

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
        """
        Add an edge between nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Type of relationship
            weight: Edge weight
            attributes: Additional attributes
            confidence: Confidence in this relationship
            bidirectional: If True, edge goes both ways

        Returns:
            The created edge, or None if nodes don't exist
        """
        with self._lock:
            if source_id not in self._nodes or target_id not in self._nodes:
                logger.warning(f"Cannot add edge: node(s) not found")
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
        """
        Get neighboring nodes.

        Args:
            node_id: Node to get neighbors for
            edge_types: Filter by edge types (None = all)
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of (neighbor_node, edge) tuples
        """
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

    def traverse_bfs(
        self,
        start_id: str,
        max_depth: int = 3,
        edge_types: list[EdgeType] | None = None,
        node_filter: Callable[[KnowledgeNode], bool] | None = None,
    ) -> list[TraversalResult]:
        """
        Breadth-first traversal from a starting node.

        Args:
            start_id: Starting node ID
            max_depth: Maximum traversal depth
            edge_types: Filter by edge types
            node_filter: Optional filter function for nodes

        Returns:
            List of traversal results (paths found)
        """
        with self._lock:
            self._stats["traversals"] += 1
            results = []
            visited = {start_id}
            queue = [(start_id, [start_id], [], 0, 1.0, 1.0)]  # id, path, edges, depth, weight, conf

            while queue:
                current_id, path, edges, depth, total_weight, total_conf = queue.pop(0)

                if depth > 0:  # Don't include starting node as a result
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
                                (neighbor.node_id, new_path, new_edges, depth + 1, new_weight, new_conf)
                            )

            return results

    def find_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5,
        edge_types: list[EdgeType] | None = None,
    ) -> TraversalResult | None:
        """
        Find shortest path between two nodes.

        Args:
            start_id: Starting node ID
            end_id: Target node ID
            max_depth: Maximum search depth
            edge_types: Filter by edge types

        Returns:
            TraversalResult if path found, None otherwise
        """
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
                            queue.append((
                                neighbor.node_id,
                                path + [neighbor.node_id],
                                edges + [edge],
                                total_weight * edge.weight,
                                total_conf * edge.confidence,
                            ))

            return None

    def spreading_activation(
        self,
        source_ids: list[str],
        activation_strength: float = 1.0,
        decay_factor: float | None = None,
        max_iterations: int = 3,
        min_activation: float = 0.01,
    ) -> dict[str, float]:
        """
        Spreading activation for associative retrieval.

        Activation spreads from source nodes through edges,
        decaying with distance.

        Args:
            source_ids: Starting nodes for activation
            activation_strength: Initial activation level
            decay_factor: Decay per hop (uses default if None)
            max_iterations: Maximum spreading iterations
            min_activation: Minimum activation to continue spreading

        Returns:
            Dictionary mapping node IDs to activation levels
        """
        with self._lock:
            decay = decay_factor or self.activation_decay
            activations: dict[str, float] = {}

            # Initialize source nodes
            for node_id in source_ids:
                if node_id in self._nodes:
                    activations[node_id] = activation_strength
                    self._nodes[node_id].activation = activation_strength

            # Spread activation
            for _ in range(max_iterations):
                new_activations: dict[str, float] = {}

                for node_id, current_activation in activations.items():
                    if current_activation < min_activation:
                        continue

                    # Spread to neighbors
                    for neighbor, edge in self.get_neighbors(node_id, direction="both"):
                        spread = current_activation * edge.weight * decay
                        if neighbor.node_id in new_activations:
                            new_activations[neighbor.node_id] = max(
                                new_activations[neighbor.node_id], spread
                            )
                        else:
                            new_activations[neighbor.node_id] = spread

                # Update activations
                for node_id, activation in new_activations.items():
                    if node_id not in activations or activation > activations[node_id]:
                        activations[node_id] = activation
                        if node_id in self._nodes:
                            self._nodes[node_id].activation = activation

            return activations

    def find_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        node_type: NodeType | None = None,
    ) -> list[tuple[KnowledgeNode, float]]:
        """
        Find nodes similar to a query embedding.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            node_type: Filter by node type

        Returns:
            List of (node, similarity) tuples
        """
        with self._lock:
            if not self.enable_embeddings:
                return []

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
        """
        Infer new edges using transitivity rules.

        Default rules:
        - IS_A is transitive: A IS_A B, B IS_A C => A IS_A C
        - CAUSES is transitive: A CAUSES B, B CAUSES C => A CAUSES C

        Args:
            inference_rules: Custom rules as (edge1, edge2, inferred)

        Returns:
            List of newly inferred edges
        """
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
                            # Check if edge doesn't already exist
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
                "node_types": {
                    t.value: len(ids) for t, ids in self._type_index.items() if ids
                },
                "edge_types": {
                    t.value: len(edges)
                    for t, edges in self._edge_type_index.items()
                    if edges
                },
            }

    def export(self) -> dict[str, Any]:
        """Export entire graph for serialization."""
        with self._lock:
            return {
                "nodes": [n.to_dict() for n in self._nodes.values()],
                "edges": [
                    e.to_dict()
                    for edges in self._edges.values()
                    for e in edges
                ],
                "statistics": self.get_statistics(),
            }
