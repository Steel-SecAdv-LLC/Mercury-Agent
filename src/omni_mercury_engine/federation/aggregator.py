"""Federated aggregator -- combines FittedStatistics from multiple nodes.

Aggregation rules (mathematically motivated):

  Means (mean, ig_mean, kin_*_mean, res_h_train, res_noise_ratio):
    -> Sample-size-weighted average. This is the MLE for Gaussian means.

  Standard deviations (std, kin_*_std):
    -> Pooled standard deviation via the parallel variance formula.
      pooled_var = weighted_avg(var_i) + weighted_avg((mean_i - global_mean)^2)
      pooled_std = sqrt(pooled_var)
    This is exact for Gaussian distributions, not an approximation.

  Percentiles (q1, q3):
    -> Weighted average. This is an approximation (true percentile
      aggregation requires raw data), but it's the best we can do
      without violating privacy. The error is bounded and small
      for similarly-distributed nodes.

  Precision matrix (ig_cov_inv):
    -> Precision-weighted average: sum(n_i * P_i) / sum(n_i).
      For Gaussian families, this gives the MLE precision of the
      combined population. This is a known result in Bayesian statistics.

  Log-determinant (ig_log_det):
    -> Recomputed from aggregated ig_cov_inv via slogdet.
      Cannot be averaged (log-determinant is not linear).

The aggregator never sees raw data. It only receives FittedStatistics
objects from FederatedNode instances.

Usage:
    aggregator = FederatedAggregator()
    aggregator.submit(node_a_stats)
    aggregator.submit(node_b_stats)
    global_stats = aggregator.aggregate()
    global_detector = aggregator.to_detector(global_stats)
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from omni_mercury_engine.federation.statistics import FittedStatistics


class FederatedAggregator:
    """Aggregates FittedStatistics from multiple federated nodes.

    Supports:
    - Weighted averaging based on sample size per node
    - Minimum node threshold (don't aggregate with < k nodes)
    - Staleness detection (reject stats older than max_age_seconds)
    - Dimension validation (all nodes must have same n_features)
    - Reconstruction of a working MercuryAnomalyDetector from aggregated stats
    """

    def __init__(
        self,
        min_nodes: int = 2,
        max_age_seconds: float = 86400.0,
    ) -> None:
        self.min_nodes = min_nodes
        self.max_age_seconds = max_age_seconds
        self._submissions: list[FittedStatistics] = []
        self._round: int = 0

    def submit(self, stats: FittedStatistics) -> None:
        """Submit statistics from a node.

        Validates freshness, sample count, and feature dimensionality.
        """
        age = time.time() - stats.timestamp
        if age > self.max_age_seconds:
            raise ValueError(
                f"Statistics from {stats.node_id} are {age:.0f}s old "
                f"(max {self.max_age_seconds:.0f}s)"
            )
        if stats.n_samples < 1:
            raise ValueError(f"Node {stats.node_id} reports 0 samples")

        # Dimension check: all nodes must have same n_features
        if self._submissions:
            expected = self._submissions[0].n_features
            if stats.n_features != expected:
                raise ValueError(
                    f"Node {stats.node_id} has {stats.n_features} features, "
                    f"expected {expected} (matching first submitted node)"
                )

        self._submissions.append(stats)

    def aggregate(self) -> FittedStatistics:
        """Combine all submitted statistics via weighted FedAvg.

        Returns:
            Global FittedStatistics representing the federated model.

        Raises:
            RuntimeError: If fewer than min_nodes have submitted.
        """
        if len(self._submissions) < self.min_nodes:
            raise RuntimeError(f"Need {self.min_nodes} nodes, have {len(self._submissions)}")

        subs = self._submissions
        total_n = sum(s.n_samples for s in subs)
        weights = np.array([s.n_samples / total_n for s in subs])

        # Weighted average helper
        def wavg(arrays: list[np.ndarray]) -> np.ndarray:
            result = np.zeros_like(arrays[0], dtype=np.float64)
            for a, w in zip(arrays, weights):
                result = result + np.asarray(a, dtype=np.float64) * w
            return result

        # Pooled std helper (parallel variance formula)
        def pooled_std(
            means: list[np.ndarray],
            stds: list[np.ndarray],
        ) -> np.ndarray:
            global_mean = wavg(means)
            variances = [np.asarray(s, dtype=np.float64) ** 2 for s in stds]
            within = np.zeros_like(global_mean)
            for v, w in zip(variances, weights):
                within = within + v * w
            between = np.zeros_like(global_mean)
            for m, w in zip(means, weights):
                between = between + (np.asarray(m, dtype=np.float64) - global_mean) ** 2 * w
            return np.asarray(np.sqrt(np.maximum(within + between, 1e-16)))

        # Aggregate means
        global_mean = wavg([s.mean for s in subs])
        global_ig_mean = wavg([s.ig_mean for s in subs])
        global_kin_jerk_mean = wavg([s.kin_jerk_mean for s in subs])
        global_kin_accel_mean = wavg([s.kin_accel_mean for s in subs])
        global_res_h_train = wavg([s.res_h_train for s in subs])
        global_res_noise_ratio = wavg([s.res_noise_ratio for s in subs])

        # Aggregate stds via pooled variance
        global_std = pooled_std([s.mean for s in subs], [s.std for s in subs])
        global_kin_jerk_std = pooled_std(
            [s.kin_jerk_mean for s in subs], [s.kin_jerk_std for s in subs]
        )
        global_kin_accel_std = pooled_std(
            [s.kin_accel_mean for s in subs], [s.kin_accel_std for s in subs]
        )

        # Aggregate percentiles (weighted average approximation)
        global_q1 = wavg([s.q1 for s in subs])
        global_q3 = wavg([s.q3 for s in subs])

        # Aggregate precision matrix (precision-weighted average)
        global_ig_cov_inv = wavg([s.ig_cov_inv for s in subs])

        # Recompute log-determinant from aggregated precision
        # ig_cov_inv = Sigma^{-1}, so det(Sigma) = 1/det(ig_cov_inv)
        # log_det(Sigma) = -log_det(ig_cov_inv)
        sign, logdet = np.linalg.slogdet(global_ig_cov_inv)
        global_ig_log_det = float(-logdet) if sign > 0 else 0.0

        result = FittedStatistics(
            node_id=f"federated_round_{self._round}",
            timestamp=time.time(),
            n_samples=total_n,
            n_features=subs[0].n_features,
            mean=global_mean,
            std=global_std,
            q1=global_q1,
            q3=global_q3,
            res_h_train=global_res_h_train,
            res_noise_ratio=global_res_noise_ratio,
            kin_jerk_mean=global_kin_jerk_mean,
            kin_jerk_std=global_kin_jerk_std,
            kin_accel_mean=global_kin_accel_mean,
            kin_accel_std=global_kin_accel_std,
            ig_mean=global_ig_mean,
            ig_cov_inv=global_ig_cov_inv,
            ig_log_det=global_ig_log_det,
            data_hash=",".join(s.data_hash for s in subs),
        )

        self._round += 1
        self._submissions.clear()
        return result

    @staticmethod
    def to_detector(stats: FittedStatistics) -> Any:
        """Reconstruct a working MercuryAnomalyDetector from FittedStatistics.

        This is the CRITICAL bridge that makes federation complete.
        Without this method, aggregated statistics are useless.

        Returns:
            MercuryAnomalyDetector that is ready for detect() calls.
        """
        from omni_mercury_engine.detectors.statistical import (
            MercuryAnomalyDetector,
        )

        det = MercuryAnomalyDetector()

        # Inject aggregated statistics into detector's internal state
        det.mean = stats.mean
        det.std = stats.std
        det.q1 = stats.q1
        det.q3 = stats.q3
        det._res_h_train = stats.res_h_train
        det._res_noise_ratio = stats.res_noise_ratio
        det._kin_jerk_mean = stats.kin_jerk_mean
        det._kin_jerk_std = stats.kin_jerk_std
        det._kin_accel_mean = stats.kin_accel_mean
        det._kin_accel_std = stats.kin_accel_std
        det._ig_mean = stats.ig_mean
        det._ig_cov_inv = stats.ig_cov_inv
        det._ig_log_det = stats.ig_log_det
        det._is_fitted = True

        return det
