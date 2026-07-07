# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the AutoML optimizer core.

These exercise the genuine Bayesian-optimization machinery (TPE / GP samplers,
scheduler pruning, time budget) against known-answer objectives, and pin the
``MercuryAutoML`` f1-metric regression that previously recursed to a
``RecursionError`` on the default metric.

Pure numpy/scipy -- no torch required.
"""

from __future__ import annotations

import time

import numpy as np

from omni_mercury_engine.automl.optimizer import (
    BayesianOptimizer,
    MercuryAutoML,
    TrialStatus,
)
from omni_mercury_engine.automl.search_space import SearchSpace, UniformParameter


def _bowl_space() -> SearchSpace:
    space = SearchSpace()
    space.add(UniformParameter("x", -10.0, 10.0))
    space.add(UniformParameter("y", -10.0, 10.0))
    return space


def _bowl_objective(config: dict[str, float]) -> float:
    """Convex bowl with a unique minimum at (3, -2), f=0."""
    return (config["x"] - 3.0) ** 2 + (config["y"] + 2.0) ** 2


def test_gp_sampler_converges_and_beats_random() -> None:
    """The GP+EI sampler genuinely optimizes: it finds the bowl minimum.

    This is the load-bearing "not a shell" check -- Gaussian-process Expected
    Improvement lands on the true optimum (3, -2) and crushes undirected random
    search. (TPE is exercised for validity below; on a smooth low-dimensional
    bowl it is not reliably better than random, so we don't over-claim it.)
    """
    gp = BayesianOptimizer(
        _bowl_space(), _bowl_objective, sampler="gp", direction="minimize", n_trials=60, seed=0
    ).optimize()
    rand = BayesianOptimizer(
        _bowl_space(), _bowl_objective, sampler="random", direction="minimize", n_trials=60, seed=0
    ).optimize()

    assert gp.best_metric < 0.1
    assert abs(gp.best_config["x"] - 3.0) < 0.4
    assert abs(gp.best_config["y"] + 2.0) < 0.4
    assert gp.best_metric < rand.best_metric / 5.0

    # convergence_history is the running best -> monotone non-increasing.
    hist = gp.convergence_history
    assert all(hist[i] <= hist[i - 1] + 1e-9 for i in range(1, len(hist)))


def test_seeding_is_reproducible() -> None:
    """A fixed seed yields identical results regardless of global RNG state.

    Regression: the samplers created a seeded ``self._rng`` but sampled via
    ``search_space.sample()`` without passing it, so every draw fell back to a
    fresh OS-entropy generator and ``seed=`` was silently ignored.
    """

    def run() -> float:
        return BayesianOptimizer(
            _bowl_space(), _bowl_objective, sampler="tpe", n_trials=40, seed=7
        ).optimize().best_metric

    np.random.seed(1)
    first = run()
    np.random.seed(999)
    second = run()
    assert first == second


def test_all_samplers_return_valid_bounded_configs() -> None:
    for sampler in ("tpe", "gp", "random"):
        result = BayesianOptimizer(
            _bowl_space(), _bowl_objective, sampler=sampler, n_trials=15, seed=0
        ).optimize()
        assert result.best_config, sampler
        assert -10.0 <= result.best_config["x"] <= 10.0
        assert -10.0 <= result.best_config["y"] <= 10.0
        assert np.isfinite(result.best_metric)


def test_time_budget_stops_before_all_trials() -> None:
    space = SearchSpace()
    space.add(UniformParameter("x", -5.0, 5.0))

    def slow(config: dict[str, float]) -> float:
        time.sleep(0.02)
        return float(config["x"] ** 2)

    opt = BayesianOptimizer(
        space, slow, sampler="random", n_trials=10_000, seed=1, time_budget=0.15
    )
    result = opt.optimize()
    # Budget must cut the run far short of the 10k trial cap.
    assert result.n_trials < 100
    assert result.n_trials >= 1


def test_scheduler_prunes_trials() -> None:
    opt = BayesianOptimizer(
        _bowl_space(),
        _bowl_objective,
        sampler="tpe",
        scheduler="asha",
        direction="minimize",
        n_trials=20,
        seed=0,
    )
    result = opt.optimize()
    assert result.n_pruned > 0
    pruned = [t for t in result.all_trials if t.status == TrialStatus.PRUNED]
    assert len(pruned) == result.n_pruned


def test_failing_objective_is_isolated() -> None:
    space = SearchSpace()
    space.add(UniformParameter("x", -1.0, 1.0))

    def boom(config: dict[str, float]) -> float:
        raise ValueError("objective exploded")

    opt = BayesianOptimizer(
        space, boom, sampler="random", direction="minimize", n_trials=5, seed=0
    )
    result = opt.optimize()
    # Run completes; every trial is marked FAILED with the sentinel metric.
    assert len(result.all_trials) == 5
    assert all(t.status == TrialStatus.FAILED for t in result.all_trials)
    assert all(np.isinf(t.metric) for t in result.all_trials)


def test_mercury_automl_f1_metric_no_recursion() -> None:
    """Regression: the default ``metric="f1"`` used to recurse to RecursionError.

    A prior implementation temporarily mutated ``self._metric`` while recursing
    into ``_compute_metric``; the first recursive call re-entered the ``"f1"``
    branch before the mutation and recursed until the stack blew. This asserts
    the fitted run now completes with every trial COMPLETED (never FAILED) and a
    finite f1 in ``[0, 1]``.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(160, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    automl = MercuryAutoML(metric="f1", n_trials=5, seed=0, scheduler=None)
    automl.add_parameter("contamination", "uniform", 0.05, 0.4)
    result = automl.fit(X, y)

    assert not any(t.status == TrialStatus.FAILED for t in result.all_trials)
    assert all(t.status == TrialStatus.COMPLETED for t in result.all_trials)
    assert 0.0 <= result.best_metric <= 1.0


def test_compute_metric_f1_matches_definition() -> None:
    """Directly pin f1 = 2PR/(P+R) on a hand-checked confusion matrix."""
    automl = MercuryAutoML(metric="f1", seed=0)
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 1, 0, 0])  # tp=2, fp=1, fn=1 -> P=2/3, R=2/3, f1=2/3
    f1 = automl._compute_metric(y_true, y_pred)
    assert abs(f1 - (2 / 3)) < 1e-6
    # And the object's metric was not mutated as a side effect.
    assert automl._metric == "f1"


def test_evaluate_model_uses_continuous_scores_for_auc() -> None:
    """Ranking metrics must consume decision_function, not binary predict."""

    class _RankModel:
        def decision_function(self, X: np.ndarray) -> np.ndarray:
            return X[:, 0]

        def predict(self, X: np.ndarray) -> np.ndarray:
            return (X[:, 0] > 0.5).astype(int)

    # Perfectly separable by the continuous score, ranked in order.
    X_val = np.array([[0.1], [0.4], [0.6], [0.9]])
    y_val = np.array([0, 0, 1, 1])

    automl = MercuryAutoML(task="anomaly_detection", metric="auc", seed=0)
    auc = automl._evaluate_model(_RankModel(), X_val, y_val)
    assert auc == 1.0  # continuous scores rank all positives above negatives
