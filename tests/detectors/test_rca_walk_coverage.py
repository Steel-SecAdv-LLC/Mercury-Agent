# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Coverage tests for the RCA random-walk degenerate + adjacency-inference paths.

Targets the previously-untested branches in
:class:`RootCauseGraphDetector`: the ``_walk`` degenerate fallback (uniform
attribution when the walk mass collapses to zero) and every branch of the
``fit``-time adjacency inference (supplied / correlation-inferred / too-few-rows /
single-node), so the changed code paths reach full coverage.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.rca import RootCauseGraphDetector


class TestWalkDegenerate:
    def test_zero_residual_row_uniform_fallback(self) -> None:
        # An all-zero residual row makes the seed (and hence the walk mass)
        # collapse to zero -> the degenerate fallback returns a uniform vector.
        det = RootCauseGraphDetector(adjacency=np.zeros((4, 4))).fit(np.zeros((6, 4)))
        walk = det._walk(np.zeros(4))
        assert walk.shape == (4,)
        assert np.allclose(walk, 0.25), "degenerate walk must fall back to uniform"
        assert walk.sum() == pytest.approx(1.0)

    def test_zero_adjacency_keeps_seed(self) -> None:
        # With a zero adjacency the transition matrix is all-zero, so the walk
        # stays at the (teleport) seed -- still a valid normalised attribution.
        det = RootCauseGraphDetector(adjacency=np.zeros((3, 3))).fit(
            np.random.default_rng(0).normal(size=(50, 3))
        )
        attr = det._walk(np.array([1.0, 0.0, 0.0]))
        assert attr.sum() == pytest.approx(1.0)
        assert np.all(np.isfinite(attr))

    def test_rank_root_causes_on_degenerate_input(self) -> None:
        det = RootCauseGraphDetector(adjacency=np.zeros((3, 3))).fit(np.zeros((5, 3)))
        ranked = det.rank_root_causes(np.zeros((2, 3)))
        assert len(ranked) == 3
        assert sum(w for _, w in ranked) == pytest.approx(1.0)


class TestAdjacencyInference:
    def test_inferred_from_correlations(self) -> None:
        rng = np.random.default_rng(1)
        base = rng.normal(size=(300, 4))
        base[:, 1] += 0.9 * base[:, 0]  # nodes 0,1 strongly correlated
        det = RootCauseGraphDetector().fit(base)  # adjacency=None -> inferred
        assert det._adjacency is not None
        assert det._adjacency.shape == (4, 4)
        assert np.all(np.diag(det._adjacency) == 0.0), "self-loops removed"
        assert det._adjacency[0, 1] > 0.0 or det._adjacency[1, 0] > 0.0

    def test_single_node_zero_adjacency(self) -> None:
        # n_nodes == 1 -> the correlation branch is skipped, a zero adjacency used.
        det = RootCauseGraphDetector().fit(np.random.default_rng(2).normal(size=(50, 1)))
        assert det._adjacency is not None
        assert det._adjacency.shape == (1, 1)
        assert det._adjacency[0, 0] == 0.0

    def test_too_few_rows_zero_adjacency(self) -> None:
        # Fewer than 2 rows -> correlation cannot be computed, zero adjacency used.
        det = RootCauseGraphDetector().fit(np.array([[1.0, 2.0, 3.0]]))
        assert det._adjacency is not None
        assert np.all(det._adjacency == 0.0)

    def test_supplied_adjacency_size_mismatch_raises(self) -> None:
        det = RootCauseGraphDetector(adjacency=np.ones((3, 3)))
        with pytest.raises(ValueError, match="adjacency size"):
            det.fit(np.random.default_rng(3).normal(size=(20, 5)))  # 5 nodes != 3

    def test_supplied_adjacency_used_verbatim(self) -> None:
        adj = np.array([[0.0, 1.0], [0.0, 0.0]])
        det = RootCauseGraphDetector(adjacency=adj).fit(
            np.random.default_rng(4).normal(size=(40, 2))
        )
        np.testing.assert_array_equal(det._adjacency, adj)


class TestWalkNonDegenerate:
    def test_walk_flows_child_to_parent(self) -> None:
        # 0 -> {1, 2}. Anomaly evidence on 1 and 2 should attribute mass upstream to 0.
        adj = np.zeros((3, 3))
        adj[0, 1] = adj[0, 2] = 1.0
        det = RootCauseGraphDetector(adjacency=adj).fit(
            np.random.default_rng(5).normal(size=(60, 3))
        )
        attr = det._walk(np.array([0.0, 1.0, 1.0]))
        assert attr.sum() == pytest.approx(1.0)
        assert attr[0] > 0.0, "root cause node must accumulate some upstream mass"
