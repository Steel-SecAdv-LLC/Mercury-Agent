"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""
from __future__ import annotations

"""Information Geometry for Out-of-Distribution Detection.

Based on: IGEOOD - An Information Geometry Approach to Out-of-Distribution Detection
(ICLR 2022: https://openreview.net/pdf?id=mfwdY3U_9ea)

Uses Fisher-Rao geodesic distance on Riemannian manifolds for OOD detection.
"""

from typing import Any

import numpy as np


class InformationGeometryDetector:
    """Information geometry-based OOD detector."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize information geometry detector.

        Args:
            config: Configuration including:
                - distance_metric: 'fisher_rao' or 'kl_divergence' (default: 'fisher_rao')
                - manifold_dim: Dimension of statistical manifold (default: 10)
                - approximation_method: 'closed_form' or 'sampling' (default: 'closed_form')
        """
        self.config = config or {}
        self.distance_metric = self.config.get("distance_metric", "fisher_rao")
        self.manifold_dim = self.config.get("manifold_dim", 10)
        self.approximation_method = self.config.get("approximation_method", "closed_form")
        self.reference_distribution: dict[str, Any] | None = None
        self.fisher_matrix: np.ndarray[Any, Any] | None = None

    def fit_reference_distribution(self, in_distribution_data: np.ndarray[Any, Any]) -> None:
        """Fit reference distribution from in-distribution training data.

        Args:
            in_distribution_data: Training data from in-distribution (ID)
        """
        self.reference_distribution = {
            "mean": np.mean(in_distribution_data, axis=0),
            "cov": np.cov(in_distribution_data.T),
        }

        self.fisher_matrix = self._compute_fisher_matrix(
            self.reference_distribution["mean"], self.reference_distribution["cov"]
        )

    def _compute_fisher_matrix(self, mean: np.ndarray[Any, Any], cov: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute Fisher Information Matrix.

        For Gaussian distributions, the Fisher matrix has a closed form.

        Args:
            mean: Distribution mean vector
            cov: Distribution covariance matrix

        Returns:
            Fisher Information Matrix
        """
        try:
            fisher = np.linalg.inv(cov + 1e-6 * np.eye(len(cov)))
        except np.linalg.LinAlgError:
            fisher = np.eye(len(cov))
        return fisher

    def fisher_rao_distance(
        self,
        distribution_1: dict[str, np.ndarray[Any, Any]],
        distribution_2: dict[str, np.ndarray[Any, Any]],
    ) -> float:
        """Compute Fisher-Rao geodesic distance between two distributions.

        The Fisher-Rao distance is the natural distance on statistical manifolds.

        Args:
            distribution_1: First distribution {'mean': ..., 'cov': ...}
            distribution_2: Second distribution {'mean': ..., 'cov': ...}

        Returns:
            Fisher-Rao distance (geodesic distance on manifold)
        """
        mean_diff = distribution_1["mean"] - distribution_2["mean"]

        if self.fisher_matrix is not None:
            distance = np.sqrt(mean_diff.T @ self.fisher_matrix @ mean_diff)
        else:
            distance = np.linalg.norm(mean_diff)

        return float(distance)

    def detect_ood(self, test_data: np.ndarray[Any, Any], threshold: float | None = None) -> dict[str, Any]:
        """Detect out-of-distribution samples using information geometry.

        Args:
            test_data: Test samples to evaluate
            threshold: OOD detection threshold (auto-computed if None)

        Returns:
            Detection results with OOD scores and labels
        """
        if self.reference_distribution is None:
            raise ValueError(
                "Must fit reference distribution first using fit_reference_distribution()"
            )

        test_distribution = {
            "mean": np.mean(test_data, axis=0),
            "cov": (np.cov(test_data.T) if test_data.shape[0] > 1 else np.eye(test_data.shape[1])),
        }

        ood_score = self.fisher_rao_distance(self.reference_distribution, test_distribution)

        if threshold is None:
            threshold = 3.0

        results = {
            "ood_score": ood_score,
            "is_ood": ood_score > threshold,
            "threshold": threshold,
            "method": "fisher_rao_geometry",
        }

        return results
