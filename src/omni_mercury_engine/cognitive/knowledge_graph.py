# Copyright (C) 2025 Steel Security Advisors LLC
"""Knowledge Graph Engine - Production Implementation.

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

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import sparse

if TYPE_CHECKING:
    from collections.abc import Callable

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
    CONTEXT = "context"
    DOMAIN = "domain"


class OntologyClassType(Enum):
    """Types of ontology classes."""

    ENTITY = "entity"
    EVENT = "event"
    RELATION = "relation"
    ATTRIBUTE = "attribute"
    ANOMALY = "anomaly"
    CONTEXT = "context"


class PropertyType(Enum):
    """Types of ontology properties."""

    OBJECT_PROPERTY = "object_property"
    DATA_PROPERTY = "data_property"
    ANNOTATION_PROPERTY = "annotation_property"
    TRANSITIVE = "transitive"
    SYMMETRIC = "symmetric"
    FUNCTIONAL = "functional"
    INVERSE_FUNCTIONAL = "inverse_functional"


class DataType(Enum):
    """Supported data types for literals."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    URI = "uri"


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
        """To dict."""
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
        """To dict."""
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
        """To dict."""
        return {
            "path": self.path,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "weight": self.total_weight,
            "confidence": self.total_confidence,
        }


@dataclass
class OntologyClass:
    """Ontology class definition with typed predicates."""

    uri: str
    name: str
    class_type: OntologyClassType
    parent_classes: list[str] = field(default_factory=list)
    description: str = ""
    properties: list[str] = field(default_factory=list)
    constraints: list[dict[str, Any]] = field(default_factory=list)
    embedding: np.ndarray[Any, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """To dict."""
        return {
            "uri": self.uri,
            "name": self.name,
            "class_type": self.class_type.value,
            "parent_classes": self.parent_classes,
            "description": self.description,
            "properties": self.properties,
        }


@dataclass
class OntologyProperty:
    """Ontology property definition with domain/range constraints."""

    uri: str
    name: str
    property_type: PropertyType
    domain: list[str] = field(default_factory=list)
    range: list[str] = field(default_factory=list)
    inverse_of: str | None = None
    is_transitive: bool = False
    is_symmetric: bool = False
    is_functional: bool = False
    cardinality_min: int | None = None
    cardinality_max: int | None = None
    description: str = ""
    embedding: np.ndarray[Any, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """To dict."""
        return {
            "uri": self.uri,
            "name": self.name,
            "property_type": self.property_type.value,
            "domain": self.domain,
            "range": self.range,
            "is_transitive": self.is_transitive,
            "is_symmetric": self.is_symmetric,
            "is_functional": self.is_functional,
            "description": self.description,
        }


class Ontology:
    """Ontology management for anomaly detection knowledge graph.

    Provides:
    - Class hierarchies with inheritance
    - Typed properties with domain/range constraints
    - Transitive and symmetric property inference
    - Core anomaly detection ontology
    """

    def __init__(self, base_uri: str = "mercury://ontology/") -> None:
        """Initialize the instance."""
        self.base_uri = base_uri
        self.classes: dict[str, OntologyClass] = {}
        self.properties: dict[str, OntologyProperty] = {}
        self._class_hierarchy: dict[str, set[str]] = defaultdict(set)
        self._property_hierarchy: dict[str, set[str]] = defaultdict(set)

        self._initialize_core_ontology()

    def _initialize_core_ontology(self) -> None:
        """Initialize core anomaly detection ontology."""
        # Root class
        self.add_class("Thing", OntologyClassType.ENTITY, description="Root class")

        # Anomaly classes
        self.add_class(
            "AnomalyPattern",
            OntologyClassType.ANOMALY,
            parent_classes=["Thing"],
            description="Base anomaly pattern",
        )
        self.add_class(
            "BehavioralAnomaly",
            OntologyClassType.ANOMALY,
            parent_classes=["AnomalyPattern"],
            description="Behavioral deviation",
        )
        self.add_class(
            "TemporalAnomaly",
            OntologyClassType.ANOMALY,
            parent_classes=["AnomalyPattern"],
            description="Time-based anomaly",
        )
        self.add_class(
            "StructuralAnomaly",
            OntologyClassType.ANOMALY,
            parent_classes=["AnomalyPattern"],
            description="Structural deviation",
        )
        self.add_class(
            "CollectiveAnomaly",
            OntologyClassType.ANOMALY,
            parent_classes=["AnomalyPattern"],
            description="Collective behavior anomaly",
        )
        self.add_class(
            "ContextualAnomaly",
            OntologyClassType.ANOMALY,
            parent_classes=["AnomalyPattern"],
            description="Context-dependent anomaly",
        )

        # Context classes
        self.add_class(
            "Context",
            OntologyClassType.CONTEXT,
            parent_classes=["Thing"],
            description="Contextual information",
        )
        self.add_class(
            "Domain",
            OntologyClassType.CONTEXT,
            parent_classes=["Context"],
            description="Application domain",
        )
        self.add_class(
            "DataSource",
            OntologyClassType.ENTITY,
            parent_classes=["Thing"],
            description="Data source",
        )

        # Core properties
        self.add_property(
            "hasAnomaly",
            PropertyType.OBJECT_PROPERTY,
            domain=["DataSource"],
            range=["AnomalyPattern"],
            description="Links data source to detected anomaly",
        )
        self.add_property(
            "hasContext",
            PropertyType.OBJECT_PROPERTY,
            domain=["AnomalyPattern"],
            range=["Context"],
            description="Links anomaly to its context",
        )
        self.add_property(
            "isRelatedTo",
            PropertyType.OBJECT_PROPERTY,
            domain=["AnomalyPattern"],
            range=["AnomalyPattern"],
            is_symmetric=True,
            description="Symmetric relation between anomalies",
        )
        self.add_property(
            "causes",
            PropertyType.OBJECT_PROPERTY,
            domain=["AnomalyPattern"],
            range=["AnomalyPattern"],
            is_transitive=True,
            description="Causal relation",
        )
        self.add_property(
            "hasScore",
            PropertyType.DATA_PROPERTY,
            domain=["AnomalyPattern"],
            range=["float"],
            is_functional=True,
            description="Anomaly score",
        )
        self.add_property(
            "hasConfidence",
            PropertyType.DATA_PROPERTY,
            domain=["AnomalyPattern"],
            range=["float"],
            is_functional=True,
            description="Detection confidence",
        )
        self.add_property(
            "hasSeverity",
            PropertyType.DATA_PROPERTY,
            domain=["AnomalyPattern"],
            range=["string"],
            is_functional=True,
            description="Severity level",
        )

    def add_class(
        self,
        name: str,
        class_type: OntologyClassType,
        parent_classes: list[str] | None = None,
        description: str = "",
        properties: list[str] | None = None,
        constraints: list[dict[str, Any]] | None = None,
    ) -> OntologyClass:
        """Add a class to the ontology."""
        uri = f"{self.base_uri}class/{name}"

        parent_uris = []
        if parent_classes:
            for parent in parent_classes:
                parent_uri = f"{self.base_uri}class/{parent}"
                parent_uris.append(parent_uri)

        ontology_class = OntologyClass(
            uri=uri,
            name=name,
            class_type=class_type,
            parent_classes=parent_uris,
            description=description,
            properties=properties or [],
            constraints=constraints or [],
        )

        self.classes[uri] = ontology_class

        for parent_uri in parent_uris:
            self._class_hierarchy[uri].add(parent_uri)

        return ontology_class

    def add_property(
        self,
        name: str,
        property_type: PropertyType,
        domain: list[str] | None = None,
        range: list[str] | None = None,
        inverse_of: str | None = None,
        is_transitive: bool = False,
        is_symmetric: bool = False,
        is_functional: bool = False,
        cardinality_min: int | None = None,
        cardinality_max: int | None = None,
        description: str = "",
    ) -> OntologyProperty:
        """Add a property to the ontology."""
        uri = f"{self.base_uri}property/{name}"

        domain_uris = [f"{self.base_uri}class/{d}" for d in (domain or [])]
        range_uris = [
            (
                f"{self.base_uri}class/{r}"
                if r not in ("string", "integer", "float", "boolean", "datetime")
                else r
            )
            for r in (range or [])
        ]

        ontology_property = OntologyProperty(
            uri=uri,
            name=name,
            property_type=property_type,
            domain=domain_uris,
            range=range_uris,
            inverse_of=f"{self.base_uri}property/{inverse_of}" if inverse_of else None,
            is_transitive=is_transitive,
            is_symmetric=is_symmetric,
            is_functional=is_functional,
            cardinality_min=cardinality_min,
            cardinality_max=cardinality_max,
            description=description,
        )

        self.properties[uri] = ontology_property
        return ontology_property

    def get_class(self, name: str) -> OntologyClass | None:
        """Get a class by name."""
        uri = f"{self.base_uri}class/{name}"
        return self.classes.get(uri)

    def get_property(self, name: str) -> OntologyProperty | None:
        """Get a property by name."""
        uri = f"{self.base_uri}property/{name}"
        return self.properties.get(uri)

    def is_subclass_of(self, child: str, parent: str) -> bool:
        """Check if child is a subclass of parent."""
        child_uri = f"{self.base_uri}class/{child}"
        parent_uri = f"{self.base_uri}class/{parent}"

        if child_uri == parent_uri:
            return True

        visited: set[str] = set()
        queue = [child_uri]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            if current == parent_uri:
                return True

            if current in self.classes:
                queue.extend(self.classes[current].parent_classes)

        return False

    def get_all_superclasses(self, class_name: str) -> set[str]:
        """Get all superclasses of a class."""
        superclasses: set[str] = set()
        class_uri = f"{self.base_uri}class/{class_name}"

        if class_uri not in self.classes:
            return superclasses

        queue = list(self.classes[class_uri].parent_classes)
        while queue:
            parent_uri = queue.pop(0)
            if parent_uri in superclasses:
                continue
            superclasses.add(parent_uri)
            if parent_uri in self.classes:
                queue.extend(self.classes[parent_uri].parent_classes)

        return superclasses

    def infer_symmetric_relations(
        self,
        relations: list[tuple[str, str, str]],
    ) -> list[tuple[str, str, str]]:
        """Infer symmetric relations based on property definitions."""
        inferred = []
        for subject, predicate, obj in relations:
            prop = self.get_property(predicate)
            if prop and prop.is_symmetric:
                inferred.append((obj, predicate, subject))
        return inferred

    def infer_transitive_relations(
        self,
        relations: list[tuple[str, str, str]],
        max_depth: int = 3,
    ) -> list[tuple[str, str, str]]:
        """Infer transitive closure for transitive properties."""
        inferred = []

        transitive_props = [p.name for p in self.properties.values() if p.is_transitive]

        for prop_name in transitive_props:
            prop_relations = [(s, o) for s, p, o in relations if p == prop_name]

            closure = set(prop_relations)
            changed = True
            depth = 0

            while changed and depth < max_depth:
                changed = False
                depth += 1
                new_relations = set()

                for s1, o1 in closure:
                    for s2, o2 in closure:
                        if o1 == s2 and (s1, o2) not in closure:
                            new_relations.add((s1, o2))
                            changed = True

                closure.update(new_relations)

            for s, o in closure - set(prop_relations):
                inferred.append((s, prop_name, o))

        return inferred

    def export(self) -> dict[str, Any]:
        """Export ontology for serialization."""
        return {
            "base_uri": self.base_uri,
            "classes": [c.to_dict() for c in self.classes.values()],
            "properties": [p.to_dict() for p in self.properties.values()],
        }


class RandomWalkEmbedding:
    """Learn node embeddings via random walks (DeepWalk/Node2Vec inspired).

    Uses truncated random walks to sample node context, then learns embeddings via skip-gram with
    negative sampling.
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
        seed: int | None = None,
    ):
        """Initialize the instance."""
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.p = p
        self.q = q
        self.learning_rate = learning_rate
        self.negative_samples = negative_samples
        self._rng: np.random.Generator = np.random.default_rng(seed)

        self.embeddings: dict[str, np.ndarray[Any, Any]] = {}
        self._node_to_idx: dict[str, int] = {}
        self._idx_to_node: dict[int, str] = {}

    def fit(
        self,
        adjacency: dict[str, list[tuple[str, float]]],
        node_ids: list[str],
    ) -> dict[str, np.ndarray[Any, Any]]:
        """Learn embeddings from graph structure.

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
        input_embeddings = self._rng.standard_normal((n_nodes, self.embedding_dim)) * scale
        output_embeddings = self._rng.standard_normal((n_nodes, self.embedding_dim)) * scale

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
            self._rng.shuffle(node_ids)
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

            next_idx = self._rng.choice(len(neighbors), p=probs)
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
    ) -> None:
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
                    neg = int(self._rng.integers(n_nodes))
                    if neg != center and neg != context:
                        self._sgd_update(center, neg, 0, input_emb, output_emb)

    def _sgd_update(
        self,
        center: int,
        context: int,
        label: int,
        input_emb: np.ndarray[Any, Any],
        output_emb: np.ndarray[Any, Any],
    ) -> None:
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
    """Graph Neural Network message passing for representation learning.

    Implements simplified GCN-style aggregation:
    h_v = σ(W * AGGREGATE({h_u : u ∈ N(v)}))
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        aggregation: str = "mean",  # "mean", "sum", "max"
        activation: str = "relu",
        seed: int | None = None,
    ):
        """Initialize the instance."""
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.aggregation = aggregation
        self.activation = activation
        self._rng: np.random.Generator = np.random.default_rng(seed)

        self._weights: list[np.ndarray[Any, Any]] = []

    def forward(
        self,
        node_features: np.ndarray[Any, Any],
        adjacency: sparse.spmatrix,
        normalize: bool = True,
    ) -> np.ndarray[Any, Any]:
        """Forward pass through GNN layers.

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
            W = self._rng.standard_normal((dims[i], dims[i + 1])) * scale
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
    """Predict missing or future links using learned node embeddings.

    Link prediction is a fundamental task in knowledge graphs that enables:
    - Knowledge graph completion (inferring missing facts)
    - Relationship recommendation (suggesting new connections)
    - Temporal prediction (forecasting future relationships)
    - Anomaly detection (identifying unusual link patterns)

    Supported Methods:
    -----------------
    1. **Dot Product** (method="dot"):
       - Score = σ(e_s · e_t) where σ is sigmoid
       - Fast computation, captures linear similarity
       - Best for: Dense graphs with uniform edge semantics
       - Complexity: O(d) per pair where d = embedding dimension

    2. **Cosine Similarity** (method="cosine"):
       - Score = (e_s · e_t) / (||e_s|| × ||e_t||)
       - Normalized scoring, invariant to embedding magnitude
       - Best for: Graphs with varied node importance
       - Complexity: O(d) per pair

    3. **Distance-Based** (method="distance"):
       - Score = 1 / (1 + ||e_s - e_t||₂)
       - TransE-inspired, treats relations as translations
       - Best for: Hierarchical or spatial relationships
       - Complexity: O(d) per pair

    Research References:
    -------------------
    - Bordes et al. (2013): Translating Embeddings for Modeling Multi-relational Data
    - Yang et al. (2015): Embedding Entities and Relations for Learning and Inference
    - Trouillon et al. (2016): Complex Embeddings for Simple Link Prediction
    - Sun et al. (2019): RotatE: Knowledge Graph Embedding by Relational Rotation

    Example Usage:
    -------------
    >>> predictor = LinkPredictor(method="dot")
    >>> score = predictor.score(source_embedding, target_embedding)
    >>> # Score in [0, 1] representing link probability

    >>> predictions = predictor.predict_links(
    ...     embeddings=node_embeddings,
    ...     candidate_pairs=[("node_a", "node_b"), ("node_c", "node_d")],
    ...     threshold=0.7
    ... )
    >>> # Returns: [("node_a", "node_b", 0.85), ...]

    Performance Considerations:
    -------------------------
    - For large candidate sets, consider batching predictions
    - Pre-normalize embeddings when using cosine method frequently
    - Use approximate nearest neighbor search for top-k queries on large graphs
    """

    def __init__(self, method: str = "dot") -> None:
        """Initialize LinkPredictor with specified scoring method.

        Args:
            method: Scoring method - one of:
                - "dot": Dot product with sigmoid activation (default)
                - "cosine": Cosine similarity (normalized)
                - "distance": Inverse Euclidean distance

        Raises:
            ValueError: If method is not one of the supported methods
        """
        valid_methods = {"dot", "cosine", "distance"}
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Must be one of: {valid_methods}")
        self.method = method

    def score(
        self,
        source_embedding: np.ndarray[Any, Any],
        target_embedding: np.ndarray[Any, Any],
    ) -> float:
        """Score a potential link.

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
        """Predict links for candidate pairs.

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
    """Production Knowledge Graph for neuro-symbolic reasoning.

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
        ontology: Ontology | None = None,
        seed: int | None = None,
    ):
        """Initialize Knowledge Graph.

        Args:
            enable_embeddings: Enable vector embeddings for nodes
            embedding_dim: Dimension of node embeddings
            activation_decay: Decay rate for spreading activation
            gnn_layers: Number of GNN layers for message passing
            ontology: Optional ontology for typed predicates and inference
            seed: Optional seed for the per-instance ``Generator`` driving
                missing-feature initialization, candidate down-sampling
                and the embedded ``RandomWalkEmbedding`` /
                ``GNNMessagePassing`` components. ``None`` (default) uses
                an OS-seeded ``Generator`` — same effective behavior as
                before.
        """
        self.enable_embeddings = enable_embeddings
        self.embedding_dim = embedding_dim
        self.activation_decay = activation_decay
        self._rng: np.random.Generator = np.random.default_rng(seed)

        # Ontology support
        self.ontology = ontology or Ontology()

        # Core storage
        self._nodes: dict[str, KnowledgeNode] = {}
        self._edges: defaultdict[str, list[KnowledgeEdge]] = defaultdict(list)
        self._reverse_edges: defaultdict[str, list[KnowledgeEdge]] = defaultdict(list)
        self._type_index: defaultdict[NodeType, set[str]] = defaultdict(set)
        self._edge_type_index: defaultdict[EdgeType, list[KnowledgeEdge]] = defaultdict(list)

        # Triple storage for ontology-based queries
        self._triples: list[tuple[str, str, str, float]] = []

        # Embedding components
        self._random_walk = RandomWalkEmbedding(embedding_dim=embedding_dim, seed=seed)
        self._gnn = GNNMessagePassing(hidden_dim=embedding_dim, num_layers=gnn_layers, seed=seed)
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
            "triples_added": 0,
            "inferences_made": 0,
        }

        logger.info(
            f"KnowledgeGraph initialized (embeddings={enable_embeddings}, dim={embedding_dim}, "
            f"ontology_classes={len(self.ontology.classes)})"
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
        """Compute node embeddings.

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
                        features[i] = self._rng.standard_normal(self.embedding_dim) * 0.1

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
        """Compute PageRank for all nodes.

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
        """Cluster nodes using spectral clustering.

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
            except Exception as e:
                logger.debug("Spectral clustering failed, assigning all nodes to cluster 0: %s", e)
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
        """Predict missing links in the knowledge graph using embedding-based similarity.

        This method performs knowledge graph completion by identifying potential
        relationships that are not yet present in the graph. It leverages the
        learned node embeddings (via random walk or GNN) to score candidate pairs.

        Algorithm:
        ---------
        1. Compute node embeddings if not already cached (auto-triggers compute_embeddings())
        2. Generate candidate pairs from all non-adjacent node combinations
        3. Apply sampling if candidate count exceeds 10,000 (performance optimization)
        4. Score each pair using the configured LinkPredictor method
        5. Filter by threshold and return top-k predictions sorted by score

        Use Cases:
        ---------
        - **Knowledge Graph Completion**: Discover missing facts and relationships
        - **Recommendation Systems**: Suggest connections between entities
        - **Anomaly Detection**: High-scoring absent links may indicate hidden patterns
        - **Data Quality**: Identify potential data entry omissions

        Args:
            top_k: Maximum number of predictions to return. Higher values provide
                more candidates but increase result size. Default: 10.
            threshold: Minimum score (0.0 to 1.0) for a prediction to be included.
                Higher values increase precision but may miss valid predictions.
                Recommended ranges:
                - 0.3-0.5: Exploratory (high recall, lower precision)
                - 0.5-0.7: Balanced (default)
                - 0.7-0.9: Conservative (high precision, lower recall)
                Default: 0.5.

        Returns:
            List of (source_id, target_id, score) tuples, sorted by score descending.
            Empty list if:
            - Graph has no nodes with computed embeddings
            - No candidate pairs exceed the threshold
            - All node pairs already have edges

        Example:
            >>> kg = KnowledgeGraph()
            >>> kg.add_node("alice", NodeType.ENTITY, "Alice")
            >>> kg.add_node("bob", NodeType.ENTITY, "Bob")
            >>> kg.add_node("project_x", NodeType.CONCEPT, "Project X")
            >>> kg.add_edge("alice", "project_x", EdgeType.PART_OF)
            >>> kg.add_edge("bob", "project_x", EdgeType.PART_OF)
            >>> # Predict potential link between Alice and Bob
            >>> predictions = kg.predict_links(top_k=5, threshold=0.4)
            >>> # May return: [("alice", "bob", 0.78)]

        Note:
            - For graphs with >10,000 potential candidate pairs, random sampling
              is applied. Set a random seed for reproducibility if needed.
            - Embeddings are computed lazily on first call and cached thereafter.
            - Thread-safe: uses internal locking for concurrent access.
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
                    candidates[i] for i in self._rng.choice(len(candidates), 10000, replace=False)
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
            queue: list[tuple[str, list[str], list[KnowledgeEdge], int, float, float]] = [
                (start_id, [start_id], [], 0, 1.0, 1.0)
            ]

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
            queue: list[tuple[str, list[str], list[KnowledgeEdge], float, float]] = [
                (start_id, [start_id], [], 1.0, 1.0)
            ]

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
        """Spreading activation for associative retrieval.

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

    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
    ) -> tuple[str, str, str, float]:
        """Add an RDF-style triple with ontology validation.

        Args:
            subject: Subject entity
            predicate: Property name
            obj: Object entity or literal
            confidence: Triple confidence

        Returns:
            The added triple (subject, predicate, object, confidence)
        """
        with self._lock:
            # Validate against ontology if property exists
            prop = self.ontology.get_property(predicate)
            if prop:
                # Check domain constraints
                if prop.domain:
                    subject_node = self._nodes.get(subject)
                    if subject_node:
                        if prop.domain and not any(
                            self.ontology.is_subclass_of(
                                subject_node.node_type.value,
                                d.replace(self.ontology.base_uri + "class/", ""),
                            )
                            for d in prop.domain
                        ):
                            logger.warning(
                                f"Domain constraint violation: {subject} not in {prop.domain}"
                            )

            triple = (subject, predicate, obj, confidence)
            self._triples.append(triple)
            self._stats["triples_added"] += 1

            # Add symmetric inference
            if prop and prop.is_symmetric:
                inverse_triple = (obj, predicate, subject, confidence * 0.99)
                if inverse_triple not in self._triples:
                    self._triples.append(inverse_triple)
                    self._stats["inferences_made"] += 1

            return triple

    def query_triples(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        obj: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[tuple[str, str, str, float]]:
        """Query triples with pattern matching.

        Args:
            subject: Subject filter (None for wildcard)
            predicate: Predicate filter (None for wildcard)
            obj: Object filter (None for wildcard)
            min_confidence: Minimum confidence threshold

        Returns:
            List of matching triples
        """
        with self._lock:
            self._stats["queries"] += 1
            results = []

            for s, p, o, conf in self._triples:
                if conf < min_confidence:
                    continue
                if subject is not None and s != subject:
                    continue
                if predicate is not None and p != predicate:
                    continue
                if obj is not None and o != obj:
                    continue
                results.append((s, p, o, conf))

            return results

    def infer_transitive_closure(
        self,
        predicate: str,
        max_depth: int = 5,
    ) -> list[tuple[str, str, str, float]]:
        """Compute transitive closure for a property.

        Args:
            predicate: Property name
            max_depth: Maximum inference depth

        Returns:
            List of inferred triples
        """
        with self._lock:
            prop = self.ontology.get_property(predicate)
            if not prop or not prop.is_transitive:
                return []

            existing = set()
            for s, p, o, _ in self._triples:
                if p == predicate:
                    existing.add((s, o))

            closure = set(existing)
            changed = True
            depth = 0

            while changed and depth < max_depth:
                changed = False
                depth += 1
                new_relations = set()

                for s1, o1 in closure:
                    for s2, o2 in closure:
                        if o1 == s2 and (s1, o2) not in closure:
                            new_relations.add((s1, o2))
                            changed = True

                closure.update(new_relations)

            inferred = []
            for s, o in closure - existing:
                triple = (s, predicate, o, 0.9**depth)
                self._triples.append(triple)
                inferred.append(triple)
                self._stats["inferences_made"] += 1

            return inferred

    def explain_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
    ) -> dict[str, Any]:
        """Explain how a triple was derived.

        Args:
            subject: Subject entity
            predicate: Property name
            obj: Object entity

        Returns:
            Explanation with provenance information
        """
        with self._lock:
            # Check if triple exists directly
            direct_matches = [
                (s, p, o, conf)
                for s, p, o, conf in self._triples
                if s == subject and p == predicate and o == obj
            ]

            if not direct_matches:
                return {
                    "found": False,
                    "explanation": f"Triple ({subject}, {predicate}, {obj}) not found",
                }

            triple = direct_matches[0]

            # Check property characteristics
            prop = self.ontology.get_property(predicate)
            prop_info = prop.to_dict() if prop else {}

            # Find supporting evidence
            supporting = []

            # For transitive properties, find intermediate nodes
            if prop and prop.is_transitive:
                for s1, p1, o1, c1 in self._triples:
                    if p1 == predicate and s1 == subject and o1 != obj:
                        for s2, p2, o2, c2 in self._triples:
                            if p2 == predicate and s2 == o1 and o2 == obj:
                                supporting.append(
                                    {
                                        "type": "transitive_chain",
                                        "chain": [
                                            (subject, predicate, o1),
                                            (o1, predicate, obj),
                                        ],
                                        "confidence": c1 * c2,
                                    }
                                )

            return {
                "found": True,
                "triple": triple,
                "confidence": triple[3],
                "property": prop_info,
                "supporting_evidence": supporting,
                "explanation": (
                    f"Triple ({subject}, {predicate}, {obj}) exists with "
                    f"confidence {triple[3]:.3f}"
                ),
            }

    def get_entity_context(
        self,
        entity: str,
        max_depth: int = 2,
    ) -> dict[str, Any]:
        """Get full context for an entity including ontology information.

        Args:
            entity: Entity identifier
            max_depth: Maximum traversal depth

        Returns:
            Dictionary with entity context
        """
        with self._lock:
            node = self._nodes.get(entity)
            if not node:
                return {"found": False, "entity": entity}

            # Get ontology class information
            class_info = None
            ont_class = self.ontology.get_class(node.node_type.value)
            if ont_class:
                class_info = ont_class.to_dict()
                class_info["superclasses"] = list(
                    self.ontology.get_all_superclasses(node.node_type.value)
                )

            # Get all triples involving this entity
            outgoing = self.query_triples(subject=entity)
            incoming = self.query_triples(obj=entity)

            # Get neighbors
            neighbors = self.get_neighbors(entity, direction="both")

            return {
                "found": True,
                "entity": entity,
                "node": node.to_dict(),
                "ontology_class": class_info,
                "outgoing_relations": [
                    {"predicate": p, "object": o, "confidence": c} for _, p, o, c in outgoing
                ],
                "incoming_relations": [
                    {"subject": s, "predicate": p, "confidence": c} for s, p, _, c in incoming
                ],
                "neighbor_count": len(neighbors),
                "pagerank": node.pagerank,
                "cluster": node.cluster_id,
            }

    def export(self) -> dict[str, Any]:
        """Export entire graph for serialization."""
        with self._lock:
            return {
                "nodes": [n.to_dict() for n in self._nodes.values()],
                "edges": [e.to_dict() for edges in self._edges.values() for e in edges],
                "triples": self._triples,
                "ontology": self.ontology.export(),
                "statistics": self.get_statistics(),
            }
