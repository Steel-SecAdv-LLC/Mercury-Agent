# Copyright (C) 2025 Steel Security Advisors LLC
"""Tests for Grey Wolf Optimizer."""

import numpy as np
import pytest

from omni_mercury_engine.ml.gwo_optimizer import GreyWolfOptimizer
from omni_mercury_engine.ml.mercury_ml import GradientBoostingClassifier
from omni_mercury_engine.utils.rng import DeterministicRNG


class TestGreyWolfOptimizer:
    """Tests for GreyWolfOptimizer class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        gwo = GreyWolfOptimizer()
        assert gwo.n_wolves == 10
        assert gwo.max_iter == 50
        assert gwo.dim is None
        assert gwo.alpha_score == float("inf")
        assert gwo.beta_score == float("inf")
        assert gwo.delta_score == float("inf")

    def test_init_custom(self) -> None:
        """Test custom initialization."""
        gwo = GreyWolfOptimizer(n_wolves=20, max_iter=100, dim=5)
        assert gwo.n_wolves == 20
        assert gwo.max_iter == 100
        assert gwo.dim == 5

    def test_init_with_rng(self) -> None:
        """Test initialization with custom RNG."""
        rng = DeterministicRNG(seed=42)
        gwo = GreyWolfOptimizer(rng=rng)
        assert gwo._rng is rng

    def test_optimize_simple_function(self) -> None:
        """Test optimization of a simple quadratic function."""
        gwo = GreyWolfOptimizer(n_wolves=5, max_iter=20)

        def objective(x: np.ndarray) -> float:
            return float(np.sum(x**2))

        lb = np.array([-5.0, -5.0])
        ub = np.array([5.0, 5.0])

        best_pos, best_score = gwo.optimize(objective, lb, ub)

        assert best_pos is not None
        assert len(best_pos) == 2
        assert best_score < 10.0

    def test_optimize_rosenbrock(self) -> None:
        """Test optimization of Rosenbrock function."""
        gwo = GreyWolfOptimizer(n_wolves=10, max_iter=30)

        def rosenbrock(x: np.ndarray) -> float:
            return float(sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))

        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])

        best_pos, best_score = gwo.optimize(rosenbrock, lb, ub)

        assert best_pos is not None
        assert best_score < 100.0

    def test_optimize_single_dimension(self) -> None:
        """Test optimization in single dimension."""
        gwo = GreyWolfOptimizer(n_wolves=5, max_iter=20)

        def objective(x: np.ndarray) -> float:
            return float((x[0] - 3) ** 2)

        lb = np.array([0.0])
        ub = np.array([10.0])

        best_pos, best_score = gwo.optimize(objective, lb, ub)

        assert best_pos is not None
        assert len(best_pos) == 1
        assert best_score < 5.0

    def test_optimize_high_dimension(self) -> None:
        """Test optimization in high dimensions."""
        gwo = GreyWolfOptimizer(n_wolves=10, max_iter=30)

        def objective(x: np.ndarray) -> float:
            return float(np.sum(x**2))

        dim = 10
        lb = np.ones(dim) * -5.0
        ub = np.ones(dim) * 5.0

        best_pos, best_score = gwo.optimize(objective, lb, ub)

        assert best_pos is not None
        assert len(best_pos) == dim

    def test_optimize_updates_wolf_positions(self) -> None:
        """Test that optimization updates alpha, beta, delta positions."""
        gwo = GreyWolfOptimizer(n_wolves=5, max_iter=10)

        def objective(x: np.ndarray) -> float:
            return float(np.sum(x**2))

        lb = np.array([-5.0, -5.0])
        ub = np.array([5.0, 5.0])

        gwo.optimize(objective, lb, ub)

        assert gwo.alpha_pos is not None
        assert gwo.beta_pos is not None
        assert gwo.delta_pos is not None
        assert gwo.alpha_score <= gwo.beta_score
        assert gwo.beta_score <= gwo.delta_score

    def test_optimize_deterministic_with_seed(self) -> None:
        """Test that optimization is deterministic with same seed."""
        rng1 = DeterministicRNG(seed=42)
        rng2 = DeterministicRNG(seed=42)

        gwo1 = GreyWolfOptimizer(n_wolves=5, max_iter=10, rng=rng1)
        gwo2 = GreyWolfOptimizer(n_wolves=5, max_iter=10, rng=rng2)

        def objective(x: np.ndarray) -> float:
            return float(np.sum(x**2))

        lb = np.array([-5.0, -5.0])
        ub = np.array([5.0, 5.0])

        best_pos1, best_score1 = gwo1.optimize(objective, lb, ub)
        best_pos2, best_score2 = gwo2.optimize(objective, lb, ub)

        np.testing.assert_array_almost_equal(best_pos1, best_pos2)
        assert best_score1 == pytest.approx(best_score2)

    def test_select_features_basic(self) -> None:
        """Test basic feature selection."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        gwo = GreyWolfOptimizer(n_wolves=5, max_iter=5)
        clf = GradientBoostingClassifier(n_estimators=5, max_depth=3, random_state=42)

        mask = gwo.select_features(X, y, clf, n_features=3)

        assert mask.dtype == bool
        assert len(mask) == 10
        assert np.sum(mask) == 3

    def test_select_features_all_features(self) -> None:
        """Test feature selection requesting all features."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)

        gwo = GreyWolfOptimizer(n_wolves=3, max_iter=3)
        clf = GradientBoostingClassifier(n_estimators=5, max_depth=2, random_state=42)

        mask = gwo.select_features(X, y, clf, n_features=5)

        assert np.sum(mask) == 5

    def test_select_features_single_feature(self) -> None:
        """Test feature selection with single feature."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)

        gwo = GreyWolfOptimizer(n_wolves=3, max_iter=3)
        clf = GradientBoostingClassifier(n_estimators=5, max_depth=2, random_state=42)

        mask = gwo.select_features(X, y, clf, n_features=1)

        assert np.sum(mask) == 1

    def test_optimize_bounds_respected(self) -> None:
        """Test that optimization respects bounds."""
        gwo = GreyWolfOptimizer(n_wolves=5, max_iter=20)

        def objective(x: np.ndarray) -> float:
            return float(np.sum(x**2))

        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        best_pos, _ = gwo.optimize(objective, lb, ub)

        assert np.all(best_pos >= lb)
        assert np.all(best_pos <= ub)

    def test_optimize_asymmetric_bounds(self) -> None:
        """Test optimization with asymmetric bounds."""
        gwo = GreyWolfOptimizer(n_wolves=5, max_iter=20)

        def objective(x: np.ndarray) -> float:
            return float((x[0] - 2) ** 2 + (x[1] + 3) ** 2)

        lb = np.array([0.0, -5.0])
        ub = np.array([5.0, 0.0])

        best_pos, _ = gwo.optimize(objective, lb, ub)

        assert best_pos[0] >= 0.0 and best_pos[0] <= 5.0
        assert best_pos[1] >= -5.0 and best_pos[1] <= 0.0


class TestGreyWolfOptimizerEdgeCases:
    """Edge case tests for GreyWolfOptimizer."""

    def test_single_wolf(self) -> None:
        """Test optimization with single wolf."""
        gwo = GreyWolfOptimizer(n_wolves=1, max_iter=10)

        def objective(x: np.ndarray) -> float:
            return float(np.sum(x**2))

        lb = np.array([-5.0, -5.0])
        ub = np.array([5.0, 5.0])

        best_pos, best_score = gwo.optimize(objective, lb, ub)

        assert best_pos is not None

    def test_single_iteration(self) -> None:
        """Test optimization with single iteration."""
        gwo = GreyWolfOptimizer(n_wolves=5, max_iter=1)

        def objective(x: np.ndarray) -> float:
            return float(np.sum(x**2))

        lb = np.array([-5.0, -5.0])
        ub = np.array([5.0, 5.0])

        best_pos, best_score = gwo.optimize(objective, lb, ub)

        assert best_pos is not None

    def test_constant_objective(self) -> None:
        """Test optimization with constant objective function."""
        gwo = GreyWolfOptimizer(n_wolves=5, max_iter=10)

        def objective(x: np.ndarray) -> float:
            return 1.0

        lb = np.array([-5.0, -5.0])
        ub = np.array([5.0, 5.0])

        best_pos, best_score = gwo.optimize(objective, lb, ub)

        assert best_score == 1.0

    def test_narrow_bounds(self) -> None:
        """Test optimization with very narrow bounds."""
        gwo = GreyWolfOptimizer(n_wolves=5, max_iter=10)

        def objective(x: np.ndarray) -> float:
            return float(np.sum(x**2))

        lb = np.array([0.0, 0.0])
        ub = np.array([0.01, 0.01])

        best_pos, best_score = gwo.optimize(objective, lb, ub)

        assert np.all(best_pos >= lb)
        assert np.all(best_pos <= ub)
