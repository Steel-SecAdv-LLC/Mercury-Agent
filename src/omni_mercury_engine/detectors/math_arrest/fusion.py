# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) Steel Security Advisors LLC
"""Phi-weighted fusion engine with correlation-aware decorrelation.

Combines probe scores using golden-ratio-derived weights, modulated by
probe confidence and optional decorrelation multipliers to prevent
redundant probe clusters from dominating the ensemble signal.
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.detectors.math_arrest.base_probe import (
    EPSILON,
    PHI,
    ProbeResult,
)

logger = logging.getLogger(__name__)

REDUNDANCY_THRESHOLD: float = 0.85
MIN_SAMPLES_FOR_DECORRELATION: int = 50


class CorrelationAwareDecorrelator:
    """Detect redundant probe clusters and reduce their weight contributions.

    A 21-probe ensemble with 6 correlated pairs is not a 21-D signal.
    This class quantifies effective dimensionality and corrects for it
    by computing pairwise Pearson correlations, identifying connected
    components of highly-correlated probes via BFS, and reducing
    weight multipliers for non-dominant members of each cluster.
    """

    def __init__(self, threshold: float = REDUNDANCY_THRESHOLD) -> None:
        self._threshold = threshold
        self._weight_multipliers: dict[str, float] = {}
        self._redundant_pairs: list[tuple[str, str, float]] = []
        self._is_calibrated: bool = False

    @property
    def is_calibrated(self) -> bool:
        """Whether :meth:`calibrate` has been called."""
        return self._is_calibrated

    @property
    def weight_multipliers(self) -> dict[str, float]:
        """Copy of weight multipliers; empty if not calibrated."""
        return dict(self._weight_multipliers)

    @property
    def redundant_pairs(self) -> list[tuple[str, str, float]]:
        """List of ``(probe_a, probe_b, r_value)`` for high-correlation pairs."""
        return list(self._redundant_pairs)

    @property
    def effective_probe_count(self) -> float:
        """Sum of all weight multipliers (effective independent dimensions).

        A fully independent 21-probe ensemble scores 21.0.
        Returns 0.0 if not calibrated.
        """
        if not self._is_calibrated:
            return 0.0
        return sum(self._weight_multipliers.values())

    def calibrate(
        self,
        score_matrix: npt.NDArray[np.float64],
        probe_names: list[str],
        fit_qualities: dict[str, float],
    ) -> dict[str, float]:
        """Compute pairwise correlations and set weight multipliers.

        Args:
            score_matrix: Shape ``(n_samples, n_probes)``.
            probe_names: Probe names corresponding to matrix columns.
            fit_qualities: Map of probe_name to trajectory_fit_quality.

        Returns:
            Weight multipliers dict (probe_name to float in ``(0, 1]``).
        """
        n_samples, n_probes = score_matrix.shape

        # Guard: insufficient data or trivial ensemble
        if n_samples < MIN_SAMPLES_FOR_DECORRELATION or n_probes <= 1:
            self._weight_multipliers = dict.fromkeys(probe_names, 1.0)
            self._redundant_pairs = []
            self._is_calibrated = True
            return dict(self._weight_multipliers)

        # Initialize all multipliers to 1.0
        multipliers: dict[str, float] = dict.fromkeys(probe_names, 1.0)

        # Identify zero-variance columns (exclude from correlation)
        col_stds = np.std(score_matrix, axis=0)
        valid_mask = col_stds >= EPSILON
        valid_indices = [i for i in range(n_probes) if valid_mask[i]]

        if len(valid_indices) <= 1:
            self._weight_multipliers = multipliers
            self._redundant_pairs = []
            self._is_calibrated = True
            return dict(self._weight_multipliers)

        # Compute correlation matrix for valid columns only
        valid_matrix = score_matrix[:, valid_indices]
        corr = np.corrcoef(valid_matrix.T)
        corr = np.nan_to_num(corr, nan=0.0)

        # Build adjacency for redundant pairs
        adjacency: dict[int, set[int]] = {i: set() for i in range(len(valid_indices))}
        redundant_pairs: list[tuple[str, str, float]] = []

        for i in range(len(valid_indices)):
            for j in range(i + 1, len(valid_indices)):
                r_val = float(abs(corr[i, j]))
                if r_val >= self._threshold:
                    adjacency[i].add(j)
                    adjacency[j].add(i)
                    name_i = probe_names[valid_indices[i]]
                    name_j = probe_names[valid_indices[j]]
                    redundant_pairs.append((name_i, name_j, r_val))
                    logger.warning(
                        "Redundant probe pair: %s <-> %s (r=%.4f)",
                        name_i,
                        name_j,
                        r_val,
                    )

        self._redundant_pairs = redundant_pairs

        # BFS connected component detection
        visited: set[int] = set()
        clusters: list[list[int]] = []

        for start in range(len(valid_indices)):
            if start in visited or not adjacency[start]:
                continue
            cluster: list[int] = []
            queue: deque[int] = deque([start])
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                cluster.append(node)
                for neighbor in adjacency[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            if len(cluster) >= 2:
                clusters.append(cluster)

        # Assign multipliers per cluster
        for cluster in clusters:
            cluster_size = len(cluster)
            # Find the probe with highest fit quality
            best_local_idx = max(
                cluster,
                key=lambda idx: fit_qualities.get(probe_names[valid_indices[idx]], 0.0),
            )
            for local_idx in cluster:
                probe_name = probe_names[valid_indices[local_idx]]
                if local_idx == best_local_idx:
                    multipliers[probe_name] = 1.0
                else:
                    multipliers[probe_name] = 1.0 / cluster_size

        self._weight_multipliers = multipliers
        self._is_calibrated = True
        return dict(self._weight_multipliers)


class PhiWeightedFusion:
    """Phi-weighted score fusion with confidence modulation and decorrelation.

    Probe scores are combined using golden-ratio-derived weights:
    weight[rank] = PHI^(-rank), normalized to sum to 1.
    """

    def __init__(self, n_probes: int = 21) -> None:
        self._n_probes = n_probes
        ranks = np.arange(n_probes, dtype=np.float64)
        raw_weights = PHI ** (-ranks)
        self._base_weights: npt.NDArray[np.float64] = raw_weights / raw_weights.sum()

    @property
    def base_weights(self) -> npt.NDArray[np.float64]:
        """Normalized Phi^-rank weights, shape ``(n_probes,)``."""
        return self._base_weights.copy()

    def fuse(
        self,
        probe_results: list[ProbeResult],
        affinity_order: list[int] | None = None,
        decorrelator: CorrelationAwareDecorrelator | None = None,
    ) -> npt.NDArray[np.float64]:
        """Fuse probe scores into a single anomaly score per sample.

        Args:
            probe_results: Active probe results.
            affinity_order: Optional reordering indices by domain affinity.
            decorrelator: Optional calibrated decorrelator.

        Returns:
            Array of shape ``(n_samples,)`` with scores in ``[0, 1]``.
        """
        if not probe_results:
            return np.array([], dtype=np.float64)

        n_active = len(probe_results)

        # Reorder by affinity
        if affinity_order is not None:
            reordered: list[ProbeResult] = []
            for idx in affinity_order:
                if idx < n_active:
                    reordered.append(probe_results[idx])
            # Add any not covered by affinity_order
            covered = set(affinity_order)
            for i in range(n_active):
                if i not in covered:
                    reordered.append(probe_results[i])
            probe_results = reordered

        # Slice base weights to active count
        weights = self._base_weights[:n_active].copy()
        if len(weights) < n_active:
            # More probes than pre-computed weights: extend
            extra = n_active - len(weights)
            extra_w = PHI ** (-(np.arange(extra) + self._n_probes))
            weights = np.concatenate([weights, extra_w])
            weights = weights / weights.sum()

        # Multiply by probe confidence
        confidences = np.array([r.confidence for r in probe_results], dtype=np.float64)
        weights = weights * confidences

        # Apply decorrelation multipliers if available
        if decorrelator is not None and decorrelator.is_calibrated:
            multiplier_map = decorrelator.weight_multipliers
            for i, result in enumerate(probe_results):
                if result.probe_name in multiplier_map:
                    weights[i] *= multiplier_map[result.probe_name]
        elif decorrelator is not None:
            logger.debug(
                "Decorrelator not calibrated; proceeding with unmodified confidence weights."
            )

        # Normalize to sum to 1
        w_sum = float(weights.sum())
        if w_sum > EPSILON:
            weights = weights / w_sum
        else:
            weights = np.ones(n_active, dtype=np.float64) / n_active

        # Align score arrays to minimum length
        min_len = min(len(r.deviation_scores) for r in probe_results)
        score_matrix = np.column_stack([r.deviation_scores[:min_len] for r in probe_results])

        # Weighted sum
        result = score_matrix @ weights
        result = np.clip(
            np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0),
            0.0,
            1.0,
        )
        return result
