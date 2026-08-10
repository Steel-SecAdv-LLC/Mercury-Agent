# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
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
    -> Pooled in COVARIANCE space, the matrix analogue of the pooled-std
      rule above:
        Sigma_i    = P_i^-1
        Sigma_pool = sum_i w_i * (Sigma_i + (mu_i - mu)(mu_i - mu)^T)
        P          = Sigma_pool^-1
      The within-group term carries each node's own spread; the rank-one
      between-group term carries the spread of the node means around the
      global mean. Both are required -- exactly as for the scalar variances,
      and for the same reason.

      Averaging the *precisions* instead (the previous rule, sum(n_i P_i)/n)
      is not the combined-population MLE and is not a result in Bayesian
      statistics. It is the posterior-precision update for combining
      independent Gaussian *likelihoods about one shared parameter*, which is
      a different problem: there the precisions add because each observation
      constrains the same mean, whereas here each node describes a different
      slice of one population. Because the map P -> P^-1 is convex, averaging
      precisions systematically *understates* the pooled covariance whenever
      the nodes differ, and it drops the between-group term entirely -- so the
      global model reports a population tighter than the one it was built
      from, and every Mahalanobis score computed from it is inflated.

  Log-determinant (ig_log_det):
    -> Recomputed from aggregated ig_cov_inv via slogdet.
      Cannot be averaged (log-determinant is not linear).

  Provenance (data_hash):
    -> A digest over the participating node IDs and the round index. Each
      node's own ``data_hash`` is a fingerprint of its raw training bytes;
      concatenating them into the global model (the previous rule)
      republished every node's fingerprint to every consumer of the global
      model, which is precisely the information federation exists to keep on
      the node. Cohort provenance stays verifiable; data fingerprints do not
      leave.

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

import hashlib
import time
from typing import Any

import numpy as np

from omni_mercury_engine.cognitive.ethical_bounding import (
    MINIMUM_BENEVOLENCE_FLOOR,
    BenevolenceScorer,
    sanitize_domain,
)
from omni_mercury_engine.federation.statistics import FittedStatistics
from omni_mercury_engine.security.sigma_immutable_gate import (
    SigmaImmutableGate,
    enforce_dual_ethical_gate,
    get_sigma_immutable_gate,
)

#: Relative floor applied to the eigenvalues of a symmetric matrix before it
#: is inverted, expressed as a fraction of that matrix's largest eigenvalue.
#: Submitted precision matrices are already Tikhonov-regularised by the
#: detector, but a differentially private release adds unbounded Gaussian
#: noise to every entry and can push the spectrum negative; flooring keeps the
#: pooled result positive-definite so the reconstructed detector's Mahalanobis
#: form stays a metric instead of silently going indefinite.
_EIGENVALUE_FLOOR_RATIO = 1e-10

#: Absolute floor for a matrix whose whole spectrum is at or below zero.
_EIGENVALUE_FLOOR_MIN = 1e-12


def _spd_inverse(matrix: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Invert a symmetric matrix, flooring its spectrum first.

    Args:
        matrix: A square matrix, symmetric up to floating-point error.

    Returns:
        The symmetric positive-definite inverse.
    """
    symmetric = 0.5 * (
        np.asarray(matrix, dtype=np.float64) + np.asarray(matrix, dtype=np.float64).T
    )
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    floor = max(
        float(np.max(eigenvalues)) * _EIGENVALUE_FLOOR_RATIO,
        _EIGENVALUE_FLOOR_MIN,
    )
    eigenvalues = np.maximum(eigenvalues, floor)
    inverse = (eigenvectors / eigenvalues) @ eigenvectors.T
    return np.asarray(0.5 * (inverse + inverse.T))


def _pool_precision(
    precisions: list[np.ndarray[Any, Any]],
    means: list[np.ndarray[Any, Any]],
    global_mean: np.ndarray[Any, Any],
    weights: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Pool per-node precision matrices through covariance space.

    Implements the matrix law of total covariance for the combined
    population::

        Sigma_pool = E[Sigma_i] + Cov[mu_i]
                   = sum_i w_i * Sigma_i
                     + sum_i w_i * (mu_i - mu)(mu_i - mu)^T

    with ``Sigma_i = P_i^-1`` and ``w_i = n_i / n``. The first term is the
    within-node spread; the second is the between-node spread that a plain
    average of precisions discards entirely.

    Args:
        precisions: Per-node precision matrices (``ig_cov_inv``).
        means: Per-node Gaussian centres (``ig_mean``), aligned with
            ``precisions``.
        global_mean: The sample-size-weighted mean of ``means``.
        weights: Per-node sample-size weights summing to 1.

    Returns:
        ``(pooled_precision, pooled_covariance)``.
    """
    global_mean_64 = np.asarray(global_mean, dtype=np.float64)
    n_features = int(np.asarray(precisions[0]).shape[0])
    pooled_cov = np.zeros((n_features, n_features), dtype=np.float64)
    for precision, mean, weight in zip(precisions, means, weights):
        within = _spd_inverse(precision)
        deviation = np.asarray(mean, dtype=np.float64) - global_mean_64
        between = np.outer(deviation, deviation)
        pooled_cov = pooled_cov + weight * (within + between)
    pooled_cov = 0.5 * (pooled_cov + pooled_cov.T)
    return _spd_inverse(pooled_cov), pooled_cov


def _cohort_provenance(node_ids: list[str], round_index: int) -> str:
    """Digest the participating cohort, carrying no data-derived input.

    Args:
        node_ids: Identifiers of the nodes whose statistics were merged.
        round_index: The aggregation round the merge belongs to.

    Returns:
        A 16-hex-character digest over the sorted node IDs and the round,
        reproducible by anyone who knows the cohort and round -- and
        therefore useful for verifying provenance, while revealing nothing
        about any node's training data.
    """
    payload = f"round={round_index};nodes={','.join(sorted(node_ids))}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


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
        domain: str | None = None,
    ) -> None:
        """Initialize the instance."""
        self.min_nodes = min_nodes
        self.max_age_seconds = max_age_seconds
        self._submissions: list[FittedStatistics] = []
        self._round: int = 0
        # σ_Immutable Wave C: both ethical gates are constructed eagerly so
        # the first concurrent ``submit`` / ``aggregate`` cannot race the
        # gate into existence.  ``sanitize_domain`` collapses a hostile or
        # typo'd domain hint to the whitelisted alphabet before it can ride
        # into the scorer action text or the audit surface.
        self._domain = sanitize_domain(domain)
        # Mirror the engine/orchestrator boundary contract: clamp the
        # benevolence floor to ``MINIMUM_BENEVOLENCE_FLOOR`` (0.70) so a
        # legitimate, positive-keyword aggregation action clears the first
        # gate while a harm-laden one does not.
        self._benevolence_scorer = BenevolenceScorer(
            benevolence_threshold=MINIMUM_BENEVOLENCE_FLOOR
        )
        self._sigma_immutable_gate: SigmaImmutableGate = get_sigma_immutable_gate()

    def _enforce_ethics(self, boundary: str, extra_details: dict[str, Any] | None = None) -> None:
        """Run the benevolence + σ_Immutable dual hard gate for federation.

        The gated decision is the real one — this boundary carries no caller
        free text, so the subject is the operation plus the node/round
        provenance in ``extra_details``. It used to be a hand-written
        positive-keyword string asserting the merge was benign, which told the
        gate nothing it could check. Severity / anomaly_prob stay at their
        benign defaults. Both gates fail closed; a violation raises
        :class:`EthicalConstraintViolationError` and the federation operation
        halts (the call is *not* wrapped in a swallowing ``try/except``).
        """
        from omni_mercury_engine.cognitive.decision_gate import DecisionSubject

        enforce_dual_ethical_gate(
            subject=DecisionSubject(
                surface=boundary,
                operation="merge fitted per-node statistics into a federated model",
                domain=self._domain,
                payload=extra_details,
            ),
            sigma_gate=self._sigma_immutable_gate,
            advisory_scorer=self._benevolence_scorer,
            boundary=boundary,
            domain=self._domain,
            extra_details=extra_details,
        )

    def submit(self, stats: FittedStatistics) -> None:
        """Submit statistics from a node.

        Validates freshness, sample count, and feature dimensionality, then
        runs the benevolence + σ_Immutable dual hard ethical gate before the
        node's statistics are admitted to the round.
        """
        self._enforce_ethics(
            "FederatedAggregator.submit",
            extra_details={"node_id": stats.node_id, "n_samples": int(stats.n_samples)},
        )
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

        # σ_Immutable Wave C: round-level dual hard gate before a global
        # model is produced from the pooled statistics.  Fails closed.
        self._enforce_ethics(
            "FederatedAggregator.aggregate",
            extra_details={"round": self._round, "n_nodes": len(self._submissions)},
        )

        subs = self._submissions
        total_n = sum(s.n_samples for s in subs)
        weights = np.array([s.n_samples / total_n for s in subs])

        # Weighted average helper
        def wavg(arrays: list[np.ndarray[Any, Any]]) -> np.ndarray[Any, Any]:
            result = np.zeros_like(arrays[0], dtype=np.float64)
            for a, w in zip(arrays, weights):
                result = result + np.asarray(a, dtype=np.float64) * w
            return result

        # Pooled std helper (parallel variance formula)
        def pooled_std(
            means: list[np.ndarray[Any, Any]],
            stds: list[np.ndarray[Any, Any]],
        ) -> np.ndarray[Any, Any]:
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

        # Aggregate precision by pooling in covariance space (within-group
        # spread + between-group spread of the node means), then inverting.
        # This is the matrix form of the pooled_std rule above.
        global_ig_cov_inv, global_pooled_cov = _pool_precision(
            [s.ig_cov_inv for s in subs],
            [s.ig_mean for s in subs],
            global_ig_mean,
            weights,
        )

        # log_det(Sigma) is read straight off the pooled covariance rather
        # than negating slogdet of its inverse: the pooled covariance is the
        # quantity that was actually constructed, and taking the determinant
        # of the inverse doubles the conditioning error on a near-singular
        # matrix. ``_pool_precision`` floors the spectrum, so the sign is
        # positive by construction.
        sign, logdet = np.linalg.slogdet(global_pooled_cov)
        global_ig_log_det = float(logdet) if sign > 0 else 0.0

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
            data_hash=_cohort_provenance([s.node_id for s in subs], self._round),
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
