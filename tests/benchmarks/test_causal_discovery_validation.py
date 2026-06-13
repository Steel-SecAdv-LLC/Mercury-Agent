# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the causal-discovery validation harness (``benchmarks.causal_discovery_validation``): the synthetic-SEM generator, the skeleton metrics, and that the revived ``causal_discovery`` engine recovers a known structure well above chance on a small, deterministic problem."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.causal_discovery_validation import (
    _prf,
    _random_dag,
    _sample_sem,
    _skeleton,
    evaluate,
)


class TestHarnessMath:
    def test_random_dag_is_acyclic_upper_triangular(self) -> None:
        a = _random_dag(8, 0.4, np.random.default_rng(0))
        # Upper-triangular adjacency (i -> j only for i < j) is acyclic by order.
        assert np.all(np.tril(a) == False)  # noqa: E712

    def test_prf_perfect_and_disjoint(self) -> None:
        true = {frozenset((0, 1)), frozenset((1, 2))}
        assert _prf(true, set(true))[2] == pytest.approx(1.0)
        p, r, f1, shd = _prf(true, {frozenset((0, 2))})
        assert f1 == 0.0 and shd == 3

    def test_sem_respects_parents(self) -> None:
        # A single edge 0 -> 1: X1 must correlate with X0, X2 (absent) must not.
        a = np.zeros((3, 3), dtype=bool)
        a[0, 1] = True
        x = _sample_sem(a, 2000, np.random.default_rng(0))
        assert abs(np.corrcoef(x[:, 0], x[:, 1])[0, 1]) > 0.3
        assert abs(np.corrcoef(x[:, 0], x[:, 2])[0, 1]) < 0.2


class TestRecovery:
    def test_engine_recovers_structure_above_chance(self) -> None:
        res = evaluate(n_vars=5, n_samples=800, n_graphs=2)
        assert res["n_graphs"] == 2
        # Recovery must clearly beat the random-graph chance baseline.
        assert res["mean_f1"] > res["chance_f1"] + 0.2
        assert res["mean_f1"] > 0.6

    def test_skeleton_of_known_graph(self) -> None:
        a = np.zeros((3, 3), dtype=bool)
        a[0, 1] = True
        a[1, 2] = True
        assert _skeleton(a) == {frozenset((0, 1)), frozenset((1, 2))}
