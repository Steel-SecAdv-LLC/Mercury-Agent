# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""NSGA-II engine: sorting, crowding, convergence and determinism."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.ml.multi_objective import (
    NSGA2Optimizer,
    crowding_distance,
    dominates,
    fast_non_dominated_sort,
)


def _zdt1(x: np.ndarray[Any, Any]) -> list[float]:
    f1 = float(x[0])
    g = 1.0 + 9.0 * float(np.mean(x[1:]))
    return [f1, g * (1.0 - np.sqrt(f1 / g))]


def test_dominance_and_sorting() -> None:
    assert dominates(np.array([1.0, 1.0]), np.array([1.0, 2.0]))
    assert not dominates(np.array([1.0, 2.0]), np.array([2.0, 1.0]))
    fronts = fast_non_dominated_sort(np.array([[1, 2], [2, 1], [2, 2], [3, 3]], dtype=float))
    assert fronts == [[0, 1], [2], [3]]


def test_crowding_keeps_extremes() -> None:
    d = crowding_distance(np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]))
    assert np.isinf(d[0]) and np.isinf(d[2])
    assert np.isfinite(d[1]) and d[1] > 0


def test_converges_to_zdt1_front() -> None:
    front = NSGA2Optimizer(_zdt1, 2, population_size=40, n_generations=60, seed=0).optimize(
        [(0.0, 1.0)] * 5
    )
    f = front.objective_matrix()
    distance = np.abs(f[:, 1] - (1.0 - np.sqrt(f[:, 0])))
    assert float(np.mean(distance)) < 0.05
    assert f[:, 0].max() - f[:, 0].min() > 0.8  # spread along the front


def test_deterministic_for_a_seed() -> None:
    def run() -> np.ndarray[Any, Any]:
        opt = NSGA2Optimizer(_zdt1, 2, population_size=20, n_generations=10, seed=7)
        return opt.optimize([(0.0, 1.0)] * 3).objective_matrix()

    np.testing.assert_array_equal(run(), run())


def test_non_finite_objectives_are_dominated() -> None:
    def objective(x: np.ndarray[Any, Any]) -> list[float]:
        return [float("nan"), 0.0] if x[0] > 0.5 else [float(x[0]), 1.0 - float(x[0])]

    front = NSGA2Optimizer(objective, 2, population_size=20, n_generations=10, seed=1).optimize(
        [(0.0, 1.0)]
    )
    assert all(np.all(np.isfinite(s.objectives)) for s in front.solutions)
    assert all(float(s.parameters[0]) <= 0.5 for s in front.solutions)


def test_knee_and_best_for() -> None:
    front = NSGA2Optimizer(_zdt1, 2, population_size=20, n_generations=20, seed=2).optimize(
        [(0.0, 1.0)] * 2
    )
    best_f1 = front.best_for(0)
    knee = front.knee()
    assert best_f1 is not None and knee is not None
    assert float(best_f1.objectives[0]) == pytest.approx(
        float(front.objective_matrix()[:, 0].min())
    )
    assert front.get_best_balanced() is knee


def test_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        NSGA2Optimizer(_zdt1, 0)
    with pytest.raises(ValueError):
        NSGA2Optimizer(_zdt1, 2, population_size=2)
