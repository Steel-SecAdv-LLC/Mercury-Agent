"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Behavioural tests for ``cognitive/knowledge_graph.py``.

These pin *algorithmic correctness* of the graph methods that were previously
import-clean-only (no behavioural assertions): node-embedding recovery on a
known two-cluster graph, GNN message passing, link-prediction recovery,
transitive-closure inference, and symmetric-relation inference.  They do NOT
claim the graph is a good anomaly *detector* (the DORMANCY_LEDGER measured that
at chance) — they assert the methods compute what they say they compute.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from omni_mercury_engine.cognitive.knowledge_graph import (
    EdgeType,
    GNNMessagePassing,
    KnowledgeGraph,
    LinkPredictor,
    NodeType,
    Ontology,
    PropertyType,
    RandomWalkEmbedding,
)


def _cos(u: np.ndarray, v: np.ndarray) -> float:
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))


def _two_cluster_graph(seed: int, embedding_dim: int = 16) -> KnowledgeGraph:
    """Two triangles (clusters) {a1,a2,a3} and {b1,b2,b3} with no cross-edges."""
    g = KnowledgeGraph(seed=seed, embedding_dim=embedding_dim)
    for n in ["a1", "a2", "a3", "b1", "b2", "b3"]:
        g.add_node(n, NodeType.CONCEPT, n)
    for s, t in [
        ("a1", "a2"),
        ("a2", "a3"),
        ("a1", "a3"),
        ("b1", "b2"),
        ("b2", "b3"),
        ("b1", "b3"),
    ]:
        g.add_edge(s, t, EdgeType.CORRELATES, bidirectional=True)
    return g


class TestRandomWalkEmbeddingRecovery:
    def test_embeddings_recover_cluster_structure(self) -> None:
        """Random-walk embeddings place intra-cluster nodes closer than
        inter-cluster nodes on a known two-cluster graph (embedding recovery).

        Averaged over seeds so the assertion is not a single-seed fluke.
        """
        margins = []
        for seed in (0, 1, 2, 3, 4):
            g = _two_cluster_graph(seed)
            emb = g.compute_embeddings(method="random_walk")
            assert set(emb) == {"a1", "a2", "a3", "b1", "b2", "b3"}
            assert emb["a1"].shape == (16,)
            intra = np.mean(
                [_cos(emb["a1"], emb["a2"]), _cos(emb["a2"], emb["a3"]), _cos(emb["b1"], emb["b2"])]
            )
            inter = np.mean(
                [_cos(emb["a1"], emb["b1"]), _cos(emb["a2"], emb["b2"]), _cos(emb["a3"], emb["b3"])]
            )
            margins.append(intra - inter)
        # Mean intra-cluster similarity must exceed inter-cluster similarity.
        assert np.mean(margins) > 0.1
        # And it must hold for the majority of seeds, not just on average.
        assert sum(m > 0 for m in margins) >= 4

    def test_fit_is_deterministic_under_seed(self) -> None:
        adjacency = {
            "x": [("y", 1.0)],
            "y": [("x", 1.0), ("z", 1.0)],
            "z": [("y", 1.0)],
        }
        # ``fit`` shuffles ``node_ids`` in place via its seeded RNG, so each
        # call must get a fresh copy; with the same seed the embeddings are
        # then byte-for-byte reproducible.
        a = RandomWalkEmbedding(embedding_dim=8, seed=11).fit(adjacency, ["x", "y", "z"])
        b = RandomWalkEmbedding(embedding_dim=8, seed=11).fit(adjacency, ["x", "y", "z"])
        for n in ["x", "y", "z"]:
            np.testing.assert_allclose(a[n], b[n])


class TestGNNMessagePassing:
    def test_forward_shape_and_finiteness(self) -> None:
        gnn = GNNMessagePassing(hidden_dim=8, num_layers=2, seed=0)
        feats = np.eye(4, dtype=np.float64)
        adj = sparse.csr_matrix(
            np.array(
                [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
                dtype=np.float64,
            )
        )
        out = gnn.forward(feats, adj)
        assert out.shape == (4, 8)
        assert np.all(np.isfinite(out))

    def test_message_passing_propagates_neighbour_signal(self) -> None:
        """A node's output must reflect its neighbourhood: nodes 0 and 2 start
        with identical features, but node 0 is connected to a distinct
        neighbour and node 2 is isolated, so they diverge after message
        passing."""
        gnn = GNNMessagePassing(hidden_dim=6, num_layers=1, seed=1)
        # Node 0 and node 2 share the same initial feature; node 1 (node 0's
        # neighbour) has a different one.
        feats = np.array(
            [[1.0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0, 0]],
            dtype=np.float64,
        )
        # Node 0 and 1 are connected; node 2 is isolated.
        adj = sparse.csr_matrix(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float64))
        out = gnn.forward(feats, adj)
        # Same initial feature, different neighbourhood ⇒ different output.
        assert not np.allclose(out[0], out[2])


class TestLinkPrediction:
    def test_link_predictor_ranks_similar_pairs_higher(self) -> None:
        lp = LinkPredictor()
        emb = {
            "a1": np.array([1.0, 0.0]),
            "a2": np.array([0.9, 0.1]),
            "b1": np.array([0.0, 1.0]),
        }
        ranked = lp.predict_links(emb, [("a1", "a2"), ("a1", "b1")], threshold=-1.0)
        # Sorted by score descending; the near-parallel intra pair outranks the
        # near-orthogonal cross pair.
        assert ranked[0][:2] == ("a1", "a2")
        assert ranked[0][2] > ranked[1][2]

    def test_kg_predict_links_recovers_held_out_edge(self) -> None:
        """A missing edge inside a dense cluster scores above a cross-cluster
        non-edge (link recovery on a known graph)."""
        scores_intra: list[float] = []
        scores_cross: list[float] = []
        for seed in (0, 1, 2, 3):
            g = KnowledgeGraph(seed=seed, embedding_dim=16)
            for n in ["a1", "a2", "a3", "a4", "b1", "b2"]:
                g.add_node(n, NodeType.CONCEPT, n)
            # Dense cluster a1..a4 minus the a1-a4 edge (held out); b cluster.
            for s, t in [("a1", "a2"), ("a2", "a3"), ("a3", "a4"), ("a1", "a3"), ("b1", "b2")]:
                g.add_edge(s, t, EdgeType.CORRELATES, bidirectional=True)
            g.compute_embeddings(method="random_walk")
            preds = {(s, t): sc for s, t, sc in g.predict_links(top_k=50, threshold=-1.0)}
            preds.update({(t, s): sc for (s, t), sc in list(preds.items())})
            if ("a1", "a4") in preds:
                scores_intra.append(preds[("a1", "a4")])
            if ("a1", "b1") in preds:
                scores_cross.append(preds[("a1", "b1")])
        # The held-out intra-cluster edge is, on average, scored above the
        # cross-cluster non-edge.
        assert scores_intra and scores_cross
        assert np.mean(scores_intra) > np.mean(scores_cross)


class TestSymbolicInference:
    def test_symmetric_relation_inference(self) -> None:
        ont = Ontology()
        ont.add_property("married_to", PropertyType.OBJECT_PROPERTY, is_symmetric=True)
        out = ont.infer_symmetric_relations([("alice", "married_to", "bob")])
        assert ("bob", "married_to", "alice") in out

    def test_symmetric_inference_ignores_non_symmetric(self) -> None:
        ont = Ontology()
        ont.add_property("parent_of", PropertyType.OBJECT_PROPERTY, is_symmetric=False)
        out = ont.infer_symmetric_relations([("alice", "parent_of", "bob")])
        assert ("bob", "parent_of", "alice") not in out

    def test_transitive_relation_inference(self) -> None:
        ont = Ontology()
        ont.add_property("ancestor", PropertyType.OBJECT_PROPERTY, is_transitive=True)
        out = ont.infer_transitive_relations([("a", "ancestor", "b"), ("b", "ancestor", "c")])
        assert ("a", "ancestor", "c") in out

    def test_kg_transitive_closure_over_chain(self) -> None:
        """The KG-level transitive closure infers every reachable pair on a
        chain a→b→c→d for a transitive predicate."""
        ont = Ontology()
        ont.add_property("ancestor", PropertyType.OBJECT_PROPERTY, is_transitive=True)
        g = KnowledgeGraph(seed=0, ontology=ont)
        g.add_triple("a", "ancestor", "b")
        g.add_triple("b", "ancestor", "c")
        g.add_triple("c", "ancestor", "d")
        inferred = {(s, o) for s, _, o, _ in g.infer_transitive_closure("ancestor", max_depth=5)}
        # Full closure minus the three base edges.
        assert {("a", "c"), ("a", "d"), ("b", "d")} <= inferred

    def test_transitive_closure_empty_for_non_transitive_predicate(self) -> None:
        ont = Ontology()
        ont.add_property("likes", PropertyType.OBJECT_PROPERTY, is_transitive=False)
        g = KnowledgeGraph(seed=0, ontology=ont)
        g.add_triple("a", "likes", "b")
        g.add_triple("b", "likes", "c")
        assert g.infer_transitive_closure("likes") == []


class TestComputeEmbeddingsMethods:
    def test_gnn_method_returns_embeddings(self) -> None:
        g = _two_cluster_graph(seed=0, embedding_dim=8)
        emb = g.compute_embeddings(method="gnn")
        assert set(emb) == {"a1", "a2", "a3", "b1", "b2", "b3"}
        for vec in emb.values():
            assert np.all(np.isfinite(vec))

    def test_both_methods_cover_every_node(self) -> None:
        g = _two_cluster_graph(seed=0, embedding_dim=8)
        rw = g.compute_embeddings(method="random_walk")
        gnn = g.compute_embeddings(method="gnn")
        assert set(rw) == set(gnn)
        assert len(rw) == 6
