# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Spatial anomaly detector for geographic data.

Optimized with Numba JIT compilation for performance-critical paths.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from scipy.spatial import KDTree

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.core.exceptions import DetectorException

# Numba optimization for distance computation (hot path)
try:
    from numba import jit

    @jit(nopython=True, cache=True)
    def _compute_distances_jit(
        data: np.ndarray[Any, Any], center: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """JIT-compiled Euclidean distance computation.

        Optimized for large datasets to achieve <1s/sample inference.
        """
        n_samples = data.shape[0]
        distances = np.empty(n_samples, dtype=np.float64)
        for i in range(n_samples):
            diff = data[i] - center
            distances[i] = np.sqrt(np.sum(diff * diff))
        return distances

    @jit(nopython=True, cache=True)
    def _compute_distance_scores_jit(
        distances: np.ndarray[Any, Any], radius_threshold: float
    ) -> np.ndarray[Any, Any]:
        """JIT-compiled distance-based anomaly scoring."""
        n_samples = len(distances)
        scores = np.empty(n_samples, dtype=np.float64)
        for i in range(n_samples):
            excess = distances[i] - radius_threshold
            if excess > 0:
                scores[i] = excess / (radius_threshold + 1e-6)
            else:
                scores[i] = 0.0
        return scores

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


class _NativeLOF:
    """Local Outlier Factor via scipy KDTree (no sklearn dependency).

    Implements LOF as defined by Breunig et al. (2000).  Only the ``fit`` / ``decision_function``
    surface used by SpatialAnomalyDetector is provided.
    """

    def __init__(self, n_neighbors: int = 20) -> None:
        """Initialize the instance."""
        self.k = n_neighbors
        self._tree: KDTree | None = None
        self._lrd: np.ndarray[Any, Any] | None = None  # local reachability densities of training

    # Reachability floor used at both fit and inference time. Using one
    # symbol (and the same numeric value at both sites) is load-bearing for
    # correctness: a duplicate-cluster query point must produce
    # ``lrd_query == mean(lrd_neighbors)`` so the LOF ratio evaluates to 1
    # (decision == 0, the inlier point). An asymmetric floor (e.g. eps in
    # fit, 1e-10 in inference) breaks the ratio by orders of magnitude and
    # mis-classifies duplicate-cluster queries as massive outliers. The
    # value matches ``sklearn.neighbors.LocalOutlierFactor``'s internal
    # epsilon for the same reason.
    _REACH_FLOOR: float = 1e-10

    def fit(self, X: np.ndarray[Any, Any]) -> _NativeLOF:
        self._tree = KDTree(X)
        k = min(self.k, len(X) - 1)
        if k < 1:
            self._lrd = np.ones(len(X))
            return self
        dists, idx = self._tree.query(X, k=k + 1)  # +1 because query includes self
        dists, idx = dists[:, 1:], idx[:, 1:]  # drop self-neighbor
        # k-distance of each neighbor (Breunig et al. 2000, eq. 1).
        kdist_neighbors = dists[idx, -1] if dists.ndim > 1 else dists
        # reachability distance = max(k-dist of neighbor, actual dist).
        reach = np.maximum(dists, kdist_neighbors)
        mean_reach = reach.mean(axis=1)
        # local reachability density = 1 / mean(reach-dist). The floor on
        # ``mean_reach`` is the same constant the inference path uses on
        # ``reach`` (and therefore on ``mean_reach``), so a duplicate-cluster
        # training point and a duplicate-cluster query point end up with
        # comparable LRDs and the LOF ratio collapses to ~1 (decision ~0,
        # inlier) instead of blowing up by orders of magnitude.
        self._lrd = 1.0 / np.maximum(mean_reach, self._REACH_FLOOR)
        return self

    def decision_function(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Return LOF-style decision scores (negative = more anomalous)."""
        assert self._tree is not None and self._lrd is not None
        k = min(self.k, len(self._lrd))
        if k < 1:
            return np.zeros(len(X))
        dists, idx = self._tree.query(X, k=k)
        if dists.ndim == 1:
            # scipy squeezes the neighbor axis for k == 1 (single training
            # sample, or n_neighbors=1); restore it so the axis-1 reductions
            # below see the documented (n, k) shape instead of raising.
            dists = dists[:, np.newaxis]
            idx = idx[:, np.newaxis]
        # Same floor as fit; see _REACH_FLOOR docstring for why this must
        # match the fit-time floor exactly.
        reach = np.maximum(dists, self._REACH_FLOOR)
        mean_reach = reach.mean(axis=1)
        lrd_query = 1.0 / mean_reach
        neighbor_lrd = self._lrd[idx]
        mean_neighbor_lrd = neighbor_lrd.mean(axis=1)
        # Both LRDs are on the same scale by construction (matched floors at
        # fit and inference), so the LOF ratio is well-defined for every
        # input and equals 1.0 for a query inside a duplicate cluster.
        lof = mean_neighbor_lrd / lrd_query
        # sklearn convention: negative = more anomalous, ~0 = inlier.
        return -(lof - 1.0)


class SpatialAnomalyDetector(BaseDetector):
    """Spatial anomaly detection for geographic data using:.

    - Distance-based outliers
    - Density-based outliers (LOF via scipy KDTree)
    - Spatial clustering
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the instance."""
        super().__init__(config)
        self.n_neighbors = self.config.get("n_neighbors", 20)
        self.contamination = self.config.get("contamination", 0.1)

        self.lof = _NativeLOF(n_neighbors=self.n_neighbors)

        self.center: np.ndarray[Any, Any] | None = None
        self.radius_threshold: float | None = None

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> SpatialAnomalyDetector:
        """Fit detector to normal spatial data."""
        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        assert isinstance(data, np.ndarray)

        if data.shape[1] < 2:
            raise DetectorException("Spatial data must have at least 2 dimensions")

        self.center = np.mean(data, axis=0)

        distances = np.linalg.norm(data - self.center, axis=1)
        self.radius_threshold = np.percentile(distances, 95)

        self.lof.fit(data)

        self._is_fitted = True
        return self

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Detect spatial anomalies with optional auto-calibration.

        Uses distance-based outliers and Local Outlier Factor (LOF)
        to compute anomaly scores for geographic/spatial data.

        Auto-Calibration:
            When auto_calibrate=True (via enable_auto_calibration()), the
            threshold is automatically calibrated based on the score
            distribution, solving the F1=0 problem.

        Returns:
            Dictionary containing:
                - is_anomaly: Boolean array of anomaly predictions
                - scores: Combined normalized scores [0, 1]
                - distance_scores: Raw distance-based scores
                - lof_scores: Local Outlier Factor scores
                - detector_type: "spatial"
                - threshold: Effective threshold (may be calibrated)
                - calibration_diagnostics: Diagnostics if auto-calibrated
        """
        if not self._is_fitted:
            raise DetectorException("Detector must be fitted before detection")

        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        assert isinstance(data, np.ndarray)
        data_np: np.ndarray[Any, Any] = data

        # Sanitise non-finite inputs at the boundary (mirrors the temporal
        # detector): scipy's KDTree refuses NaN/Inf queries outright, which
        # crashed detection before the NaN guards below could ever run.
        if data_np.dtype.kind == "f" and not np.all(np.isfinite(data_np)):
            data_np = np.nan_to_num(data_np, nan=0.0, posinf=0.0, neginf=0.0)

        distance_scores = self._compute_distance_scores(data_np)
        lof_scores = self.lof.decision_function(data_np)

        # Safe normalization with NaN/constant array handling
        # Fix for P0: Division by near-zero in min/max normalization
        distance_scores_norm = self._safe_normalize(distance_scores)
        lof_scores_norm = self._safe_normalize(-lof_scores)

        # Validate for NaN propagation before combining
        if np.any(~np.isfinite(distance_scores_norm)):
            distance_scores_norm = np.nan_to_num(
                distance_scores_norm, nan=0.5, posinf=1.0, neginf=0.0
            )
        if np.any(~np.isfinite(lof_scores_norm)):
            lof_scores_norm = np.nan_to_num(lof_scores_norm, nan=0.5, posinf=1.0, neginf=0.0)

        combined_scores = (distance_scores_norm + lof_scores_norm) / 2.0

        # Auto-calibration: compute optimal threshold from score distribution
        effective_threshold = self.threshold
        calibration_diagnostics = None

        if self._auto_calibrate:
            effective_threshold = self.calibrate_threshold(combined_scores)
            calibration_diagnostics = self._last_diagnostics

        is_anomaly = combined_scores > effective_threshold

        return {
            "is_anomaly": is_anomaly,
            "scores": combined_scores,
            "distance_scores": distance_scores,
            "lof_scores": -lof_scores,
            "detector_type": "spatial",
            "threshold": effective_threshold,
            "calibration_diagnostics": calibration_diagnostics,
        }

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> torch.Tensor:
        """Extract spatial features for ML fusion."""
        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        if not self._is_fitted:
            self.fit(data)

        if self.center is None:
            raise DetectorException("Detector center not initialized")

        distances = np.linalg.norm(data - self.center, axis=1, keepdims=True)

        angles = np.arctan2(
            data[:, 1] - self.center[1],
            data[:, 0] - self.center[0],
        ).reshape(-1, 1)

        features = np.column_stack(
            [
                data[:, : min(2, data.shape[1])],
                distances,
                angles,
            ]
        )

        if features.shape[1] < 32:
            padding = np.zeros((features.shape[0], 32 - features.shape[1]))
            features = np.column_stack([features, padding])

        return torch.tensor(features, dtype=torch.float32)

    def _safe_normalize(self, scores: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Safely normalize scores to [0, 1] range.

        Handles edge cases:
        - Constant arrays (max == min): returns 0.5 for all
        - NaN values: replaced with 0.5 (neutral score)
        - Inf values: clipped to valid range

        Fix for P0: Division by near-zero in min/max normalization.

        Args:
            scores: Raw anomaly scores.

        Returns:
            Normalized scores in [0, 1] range.
        """
        # Handle NaN/Inf first
        if np.any(~np.isfinite(scores)):
            scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)

        score_min = scores.min()
        score_max = scores.max()
        score_range = score_max - score_min

        # Check for constant array (all same value)
        if score_range < 1e-10:
            # Return neutral 0.5 for constant arrays
            return np.full_like(scores, 0.5)

        # Standard min-max normalization with safe denominator
        normalized = (scores - score_min) / score_range
        return np.clip(normalized, 0.0, 1.0)

    def _compute_distance_scores(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute distance-based anomaly scores.

        Uses Numba JIT compilation when available for ~10x speedup on large datasets, targeting
        <1s/sample inference requirement.
        """
        if self.center is None or self.radius_threshold is None:
            raise DetectorException("Detector center or radius_threshold not initialized")

        if NUMBA_AVAILABLE:
            # Use JIT-compiled version for performance
            distances = _compute_distances_jit(
                data.astype(np.float64), self.center.astype(np.float64)
            )
            scores = _compute_distance_scores_jit(distances, float(self.radius_threshold))
        else:
            # Fallback to numpy
            distances = np.linalg.norm(data - self.center, axis=1)
            scores = np.maximum(distances - self.radius_threshold, 0) / (
                self.radius_threshold + 1e-6
            )
        return scores

    def get_fitted_state(self) -> dict[str, Any] | None:
        """Export the fitted state for checkpoint round-tripping.

        The KDTree is represented by its training points (``tree.data``);
        :meth:`set_fitted_state` rebuilds it, which is deterministic for the
        same points, so reloaded LOF queries reproduce the saving engine's
        scores exactly (ROADMAP row 16).

        Returns:
            JSON/tensor-safe mapping, or ``None`` when unfitted.
        """
        if not self._is_fitted or self.center is None or self.radius_threshold is None:
            return None
        if self.lof._tree is None or self.lof._lrd is None:
            return None
        return {
            "center": np.asarray(self.center, dtype=np.float64),
            "radius_threshold": float(self.radius_threshold),
            "lof_k": int(self.lof.k),
            "lof_points": np.asarray(self.lof._tree.data, dtype=np.float64),
            "lof_lrd": np.asarray(self.lof._lrd, dtype=np.float64),
        }

    def set_fitted_state(self, state: dict[str, Any]) -> None:
        """Restore a state produced by :meth:`get_fitted_state`."""
        self.center = np.asarray(state["center"], dtype=np.float64)
        self.radius_threshold = float(state["radius_threshold"])
        self.lof = _NativeLOF(n_neighbors=int(state["lof_k"]))
        self.lof._tree = KDTree(np.asarray(state["lof_points"], dtype=np.float64))
        self.lof._lrd = np.asarray(state["lof_lrd"], dtype=np.float64)
        self._is_fitted = True
