# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Spatial anomaly detector for geographic data.

Scores points against the training distribution's geometry: distance from
the fitted centroid past a quantile radius (computed with a BLAS expansion,
the fastest formulation measured) and a KDTree-backed Local Outlier Factor,
with spatial autocorrelation summaries on detection output.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from scipy.spatial import KDTree

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.core.exceptions import DetectorException, InputModalityError
from omni_mercury_engine.detectors._calibration import batch_evidence, conformal_cut
from omni_mercury_engine.detectors.provenance import DecisionSource

logger = logging.getLogger(__name__)


def _euclidean_distances(
    data: np.ndarray[Any, Any], center: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """Distances to ``center`` via ``|x - c|^2 = |x|^2 - 2 x.c + |c|^2``.

    The expansion runs on BLAS without materialising the ``data - center``
    temporary, which makes it the fastest formulation on every host (262144
    x 64 rows: 23 ms, against 46 ms for a numba JIT loop and 324 ms for
    ``np.linalg.norm(data - center, axis=1)``); the previous numba lane was
    both slower and optional. The subtraction can go negative by rounding
    for points near the centre, hence the clip.
    """
    squared = np.einsum("ij,ij->i", data, data) - 2.0 * (data @ center) + float(center @ center)
    return np.asarray(np.sqrt(np.maximum(squared, 0.0)), dtype=np.float64)


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
        # k-distance of each training point (Breunig et al. 2000, def. 3): the
        # reachability distance from a query p to a training point o is
        # max(k-distance(o), d(p, o)) at inference exactly as at fit.
        self._kdist: np.ndarray[Any, Any] | None = None
        # Leave-one-out LOF of each training point (its own neighbourhood
        # excludes itself), the reference an unseen normal query compares to.
        self._train_lof: np.ndarray[Any, Any] | None = None

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
            self._kdist = np.zeros(len(X))
            self._train_lof = np.ones(len(X))
            return self
        dists, idx = self._tree.query(X, k=k + 1)  # +1 because query includes self
        dists, idx = dists[:, 1:], idx[:, 1:]  # drop self-neighbor
        self._kdist = dists[:, -1].copy()
        # k-distance of each neighbor (Breunig et al. 2000, eq. 1).
        kdist_neighbors = self._kdist[idx]
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
        self._train_lof = self._lrd[idx].mean(axis=1) / self._lrd
        return self

    def restore_neighbourhoods(self) -> None:
        """Recompute ``_kdist`` / ``_train_lof`` from the tree (checkpoint restore).

        Both are deterministic functions of the training points and ``k``,
        so a state saved without them restores to the values ``fit`` gives.
        """
        if self._tree is None or self._lrd is None:
            return
        points = np.asarray(self._tree.data, dtype=np.float64)
        k = min(self.k, len(points) - 1)
        if k < 1:
            self._kdist = np.zeros(len(points))
            self._train_lof = np.ones(len(points))
            return
        dists, idx = self._tree.query(points, k=k + 1)
        dists, idx = dists[:, 1:], idx[:, 1:]
        self._kdist = dists[:, -1].copy()
        self._train_lof = self._lrd[idx].mean(axis=1) / self._lrd

    def training_decision(self) -> np.ndarray[Any, Any]:
        """Leave-one-out decision scores of the training points (negative = anomalous)."""
        assert self._train_lof is not None
        return -(self._train_lof - 1.0)

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
        # Reachability as Breunig et al. define it and as fit computes it:
        # max(k-distance(o), d(p, o)). Plain distances made a query's density
        # systematically higher than a training point's at the same place, so
        # LOF ratios of queries and of training points were not comparable.
        kdist = self._kdist[idx] if self._kdist is not None else 0.0
        reach = np.maximum(dists, kdist)
        # Same floor as fit; see _REACH_FLOOR docstring for why this must
        # match the fit-time floor exactly.
        mean_reach = np.maximum(reach.mean(axis=1), self._REACH_FLOOR)
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

    decision_source = DecisionSource(
        kind="published",
        description=(
            "The score is the mean of a centroid-distance channel and a Local "
            "Outlier Factor channel computed as published (reachability "
            "max(k-distance(o), d(p, o)) at fit and at query), each min-max "
            "normalised against the training rows' range widened by the row, so "
            "a row's score does not depend on its batch. The decision threshold "
            "is the configured or auto-calibrated operating point, else the "
            "one-class split-conformal 0.95 quantile of the training rows' "
            "leave-one-out combined scores (held-out Gaussian rows flagged "
            "0.051-0.053 at batch sizes 10 to 2,000, against 0.40 in 10-row "
            "batches under the batch-relative normalisation and fixed 0.5 cut "
            "this replaced). The LOF channel is published; the centroid channel "
            "and the equal-weight fusion are in-house."
        ),
        citation=(
            "Breunig, Kriegel, Ng & Sander (2000), 'LOF: Identifying "
            "Density-Based Local Outliers', ACM SIGMOD 2000 (the LOF channel)"
        ),
    )

    #: Verified free of process-global mutable state at inference (bit-identical
    #: concurrent vs sequential: tests/core/test_parallel_detectors.py).
    parallel_safe = True

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the instance."""
        super().__init__(config)
        self.n_neighbors = self.config.get("n_neighbors", 20)
        self.contamination = self.config.get("contamination", 0.1)
        # Spatial autocorrelation of the scores (Moran's I / Geary's C with a
        # permutation test): are the anomalies a regional cluster or
        # scattered points? Above _AUTOCORRELATION_SAMPLE_LIMIT points the
        # statistics and their permutation test run on an evenly spaced subsample.
        self.spatial_autocorrelation_enabled = bool(
            self.config.get("spatial_autocorrelation", True)
        )
        self.autocorrelation_k = int(self.config.get("autocorrelation_k", 8))
        self.autocorrelation_permutations = int(self.config.get("autocorrelation_permutations", 99))

        self.lof = _NativeLOF(n_neighbors=self.n_neighbors)

        self.center: np.ndarray[Any, Any] | None = None
        self.radius_threshold: float | None = None
        # Fit-time anchors for the two raw scores: (min, max) over the
        # training rows (leave-one-out for LOF). A row is normalised against
        # them, so its score does not depend on the rows it is scored with.
        self._distance_range: tuple[float, float] | None = None
        self._neg_lof_range: tuple[float, float] | None = None
        self._threshold_is_configured = isinstance(config, dict) and "threshold" in config
        # Sorted leave-one-out combined scores of the training rows: the
        # reference for the conformal p-values and the batch-level test.
        self._reference_scores: np.ndarray[Any, Any] | None = None

    @staticmethod
    def _require_spatial_matrix(data: np.ndarray[Any, Any]) -> None:
        """Refuse input that is not a spatial coordinate matrix.

        This detector places points in a metric space: the centroid radius and
        the LOF neighbourhood are only defined over two or more coordinate
        axes. A 1-D series, or a single-column matrix, carries no such space,
        so it is declined as the wrong modality rather than scored.

        ``InputModalityError`` is the convention for a component declining a
        modality it does not process (see
        :class:`~omni_mercury_engine.core.exceptions.InputModalityError`): the
        engine skips the detector and the metrics layer counts it as
        *declined* rather than as an inference error. Checking ``ndim`` first
        keeps a 1-D array from indexing ``shape[1]`` and surfacing an opaque
        ``IndexError`` in place of this refusal.

        Raises:
            InputModalityError: when ``data`` has fewer than two columns.
        """
        if data.ndim < 2 or data.shape[1] < 2:
            raise InputModalityError(
                "Spatial data must have at least 2 dimensions; got shape "
                f"{tuple(data.shape)}. This detector scores points in a metric "
                "space and declines univariate input."
            )

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> SpatialAnomalyDetector:
        """Fit detector to normal spatial data."""
        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        assert isinstance(data, np.ndarray)

        self._require_spatial_matrix(data)

        self.center = np.mean(data, axis=0)

        distances = np.linalg.norm(data - self.center, axis=1)
        self.radius_threshold = np.percentile(distances, 95)

        self.lof.fit(data)
        self._fit_anchors(np.asarray(data, dtype=np.float64))

        self._is_fitted = True
        return self

    #: Share of training rows above the installed cut (one-class split-conformal
    #: quantile of their leave-one-out combined scores).
    CALIBRATION_ALPHA = 0.05

    def _fit_anchors(self, data: np.ndarray[Any, Any]) -> None:
        """Anchor normalisation to the training rows and cut the threshold on them.

        The two raw scores used to be min-max normalised over the batch being
        scored, so a batch's most extreme row scored about 1 whatever its
        value: held-out Gaussian rows were flagged at 0.40 in 10-row batches
        and 0.012 in 2,000-row ones. They are normalised against the training
        rows' range instead -- the ``[reference; row]`` semantics the agentic
        serving path already used -- and, unless a threshold was configured,
        the cut is the ``ceil((n + 1)(1 - alpha))``-th smallest leave-one-out
        combined training score.
        """
        ref_distance = np.asarray(self._compute_distance_scores(data), dtype=np.float64)
        ref_neg_lof = -np.asarray(self.lof.training_decision(), dtype=np.float64)
        self._distance_range = (float(ref_distance.min()), float(ref_distance.max()))
        self._neg_lof_range = (float(ref_neg_lof.min()), float(ref_neg_lof.max()))
        self._reference_scores = np.sort(self._combine(ref_distance, ref_neg_lof))
        if self._threshold_is_configured or self._auto_calibrate:
            return
        cut = conformal_cut(self._reference_scores, self.CALIBRATION_ALPHA)
        if cut is not None:
            self.threshold = cut

    @staticmethod
    def _anchored(
        values: np.ndarray[Any, Any], bounds: tuple[float, float]
    ) -> np.ndarray[Any, Any]:
        """Min-max against ``[reference; row]`` per row: ``bounds`` widened by the row."""
        v = np.asarray(values, dtype=np.float64)
        v = np.where(np.isfinite(v), v, np.nan)
        lo = np.minimum(bounds[0], v)
        hi = np.maximum(bounds[1], v)
        span = hi - lo
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(span < 1e-10, 0.5, (v - lo) / np.where(span < 1e-10, 1.0, span))
        return np.asarray(np.clip(np.nan_to_num(out, nan=0.5), 0.0, 1.0), dtype=np.float64)

    def _combine(
        self, distance_scores: np.ndarray[Any, Any], neg_lof_scores: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """Mean of the two anchored scores (batch min-max for a legacy state)."""
        if self._distance_range is None or self._neg_lof_range is None:
            d_norm = self._safe_normalize(distance_scores)
            l_norm = self._safe_normalize(neg_lof_scores)
        else:
            d_norm = self._anchored(distance_scores, self._distance_range)
            l_norm = self._anchored(neg_lof_scores, self._neg_lof_range)
        d_norm = np.nan_to_num(d_norm, nan=0.5, posinf=1.0, neginf=0.0)
        l_norm = np.nan_to_num(l_norm, nan=0.5, posinf=1.0, neginf=0.0)
        return np.asarray((d_norm + l_norm) / 2.0, dtype=np.float64)

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
                - spatial_autocorrelation: Moran's I / Geary's C of the
                  scores with permutation p-values and a
                  clustered / dispersed / random verdict (``None`` when
                  undefined or disabled)
        """
        if not self._is_fitted:
            raise DetectorException("Detector must be fitted before detection")

        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        assert isinstance(data, np.ndarray)
        self._require_spatial_matrix(data)
        data_np: np.ndarray[Any, Any] = data

        # Sanitise non-finite inputs at the boundary (mirrors the temporal
        # detector): scipy's KDTree refuses NaN/Inf queries outright, which
        # crashed detection before the NaN guards below could ever run.
        if data_np.dtype.kind == "f" and not np.all(np.isfinite(data_np)):
            data_np = np.nan_to_num(data_np, nan=0.0, posinf=0.0, neginf=0.0)

        distance_scores = self._compute_distance_scores(data_np)
        lof_scores = self.lof.decision_function(data_np)

        # Each row against the fit-time anchors (see _fit_anchors): its score
        # is the same whichever rows it is scored with.
        combined_scores = self._combine(distance_scores, -lof_scores)

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
            "spatial_autocorrelation": self._score_autocorrelation(data_np, combined_scores),
            **batch_evidence(self._reference_scores, combined_scores, self.CALIBRATION_ALPHA),
        }

    #: Above this many points the statistics are estimated on an evenly spaced
    #: subsample of this size. The kNN graph of the whole batch costs close to
    #: O(n^2 * d) once ``d`` is past a few dimensions (a KD-tree degrades to
    #: brute force): measured 4.8 s of the detector's 5.7 s at 16,384 x 16 and
    #: about 19 s at 32,768, for a diagnostic that never enters the scores.
    _AUTOCORRELATION_SAMPLE_LIMIT = 4096

    def _score_autocorrelation(
        self, data: np.ndarray[Any, Any], scores: np.ndarray[Any, Any]
    ) -> dict[str, Any] | None:
        """Moran's I / Geary's C of the scores over the points' kNN graph.

        Larger batches are estimated on :attr:`_AUTOCORRELATION_SAMPLE_LIMIT`
        evenly spaced points (``sampled_from`` then gives the batch size): a
        global statistic estimated at a coarser neighbourhood scale, not the
        batch's exact value. ``None`` when disabled, with fewer than 10
        points, or when the scores are constant (the statistics are undefined).
        """
        if not self.spatial_autocorrelation_enabled or len(scores) < 10:
            return None
        from omni_mercury_engine.ml.spatial_autocorrelation import spatial_autocorrelation

        sampled_from = None
        if len(scores) > self._AUTOCORRELATION_SAMPLE_LIMIT:
            sampled_from = len(scores)
            picked = np.linspace(0, len(scores) - 1, self._AUTOCORRELATION_SAMPLE_LIMIT)
            index = np.unique(picked.round().astype(np.intp))
            data, scores = data[index], scores[index]
        try:
            result = spatial_autocorrelation(
                scores,
                data,
                k=self.autocorrelation_k,
                n_permutations=self.autocorrelation_permutations,
            )
        except ValueError as exc:
            logger.debug("spatial autocorrelation skipped: %s", exc)
            return None
        if result is None:
            return None
        return {**result.to_dict(), "sampled_from": sampled_from}

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
        """Compute distance-based anomaly scores."""
        if self.center is None or self.radius_threshold is None:
            raise DetectorException("Detector center or radius_threshold not initialized")

        distances = _euclidean_distances(data.astype(np.float64), self.center.astype(np.float64))
        return np.maximum(distances - self.radius_threshold, 0) / (self.radius_threshold + 1e-6)

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
        self.lof.restore_neighbourhoods()
        self._fit_anchors(np.asarray(state["lof_points"], dtype=np.float64))
        self._is_fitted = True

    def _learned_state(self) -> dict[str, Any]:
        """Bridge the shared checkpoint to :meth:`get_fitted_state`.

        The fitted ``_NativeLOF`` holds a scipy KDTree, which the shared
        primitives-only payload cannot carry; the exported state carries the
        training points instead and :meth:`set_fitted_state` rebuilds the
        tree deterministically, so restored scores match exactly.
        """
        from omni_mercury_engine.core.base import _serializable_state

        fitted = self.get_fitted_state()
        if fitted is None:
            return {}
        return {
            key: _serializable_state(value, owner=type(self).__name__, attr=key)
            for key, value in fitted.items()
        }

    def _restore_learned_state(self, state: dict[str, Any]) -> None:
        """Reinstate the state captured by :meth:`_learned_state`."""
        from omni_mercury_engine.core.base import _deserialized_state

        if state:
            self.set_fitted_state({key: _deserialized_state(value) for key, value in state.items()})
