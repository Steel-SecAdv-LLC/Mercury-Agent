# Copyright (C) 2025 Steel Security Advisors LLC
"""Grey Wolf Optimizer for Feature Selection.

Bio-inspired optimization algorithm that mimics grey wolf hunting behavior
for optimal feature subset selection in anomaly detection.

⚠️ SIMULATION-BASED: Optimization on simulated data. Real-world validation required.

Reference: Mirjalili et al. (2014) - Grey Wolf Optimizer
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

logger = logging.getLogger(__name__)

from omni_mercury_engine.utils.rng import DeterministicRNG, get_global_rng

if TYPE_CHECKING:
    from collections.abc import Callable


class GreyWolfOptimizer:
    """GWO for feature selection and hyperparameter optimization."""

    def __init__(
        self,
        n_wolves: int = 10,
        max_iter: int = 50,
        dim: int | None = None,
        rng: DeterministicRNG | None = None,
    ):
        """Initialize the instance."""
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.dim = dim
        self._rng = rng or get_global_rng()

        self.alpha_pos: np.ndarray[Any, Any] | None = None
        self.beta_pos: np.ndarray[Any, Any] | None = None
        self.delta_pos: np.ndarray[Any, Any] | None = None

        self.alpha_score = float("inf")
        self.beta_score = float("inf")
        self.delta_score = float("inf")

    def optimize(
        self,
        objective_func: Callable[[np.ndarray[Any, Any]], float],
        lb: np.ndarray[Any, Any],
        ub: np.ndarray[Any, Any],
    ) -> tuple[np.ndarray[Any, Any], float]:
        """Optimize using GWO algorithm.

        Args:
            objective_func: Function to minimize (e.g., 1 - accuracy)
            lb: Lower bounds for each dimension
            ub: Upper bounds for each dimension

        Returns:
            Best position (feature subset) and best score
        """
        dim = len(lb)

        # Use numpy random for array-based uniform sampling
        positions = np.random.default_rng(self._rng.randint(0, 2**31)).uniform(
            lb, ub, (self.n_wolves, dim)
        )

        for iteration in range(self.max_iter):
            for i in range(self.n_wolves):
                fitness = objective_func(positions[i])

                if fitness < self.alpha_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy() if self.beta_pos is not None else None

                    self.beta_score = self.alpha_score
                    self.beta_pos = self.alpha_pos.copy() if self.alpha_pos is not None else None

                    self.alpha_score = fitness
                    self.alpha_pos = positions[i].copy()
                elif fitness < self.beta_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy() if self.beta_pos is not None else None

                    self.beta_score = fitness
                    self.beta_pos = positions[i].copy()
                elif fitness < self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = positions[i].copy()

            a = 2 - iteration * (2 / self.max_iter)

            # Ensure wolf positions are initialized before updating
            if self.alpha_pos is None or self.beta_pos is None or self.delta_pos is None:
                continue

            for i in range(self.n_wolves):
                for j in range(dim):
                    r1 = self._rng.random()
                    r2 = self._rng.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.alpha_pos[j] - positions[i, j])
                    X1 = self.alpha_pos[j] - A1 * D_alpha

                    r1 = self._rng.random()
                    r2 = self._rng.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.beta_pos[j] - positions[i, j])
                    X2 = self.beta_pos[j] - A2 * D_beta

                    r1 = self._rng.random()
                    r2 = self._rng.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.delta_pos[j] - positions[i, j])
                    X3 = self.delta_pos[j] - A3 * D_delta

                    positions[i, j] = (X1 + X2 + X3) / 3

                    positions[i, j] = np.clip(positions[i, j], lb[j], ub[j])

        assert self.alpha_pos is not None, "Alpha position must be set after optimization"
        return self.alpha_pos, self.alpha_score

    def select_features(
        self, X: np.ndarray[Any, Any], y: np.ndarray[Any, Any], clf: Any, n_features: int
    ) -> np.ndarray[Any, Any]:
        """Select optimal feature subset using GWO.

        Args:
            X: Feature matrix
            y: Labels
            clf: Sklearn-compatible classifier
            n_features: Number of features to select

        Returns:
            Boolean mask of selected features
        """
        n_total_features = X.shape[1]

        def objective(mask_real: np.ndarray[Any, Any]) -> float:
            mask = (mask_real > 0.5).astype(bool)

            if np.sum(mask) < n_features:
                return 1.0

            X_selected = X[:, mask]

            try:
                from omni_mercury_engine.ml.mercury_ml import cross_val_score

                scores = cross_val_score(clf, X_selected, y, cv=3)
                return 1.0 - float(np.mean(np.asarray(scores, dtype=np.float64)))
            except Exception as e:
                logger.debug("Cross-validation failed for feature selection: %s", e)
                return 1.0

        lb = np.zeros(n_total_features)
        ub = np.ones(n_total_features)

        best_pos, _best_score = self.optimize(objective, lb, ub)

        indices = np.argsort(best_pos)[-n_features:]
        mask = np.zeros(n_total_features, dtype=bool)
        mask[indices] = True

        return mask
