# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Federated aggregation must reconstruct the combined population, not a fiction.

Two defects are pinned here:

* **Precision matrices were plain-averaged.** ``sum(n_i P_i) / sum(n_i)`` is
  the posterior-precision update for independent likelihoods about one shared
  parameter, not the combined-population covariance. It drops the
  between-node spread entirely, and because ``P -> P^-1`` is convex it
  understates the pooled covariance whenever nodes differ -- so the global
  model claims a tighter population than it was built from and inflates every
  Mahalanobis score. The scalar path twelve lines above it already pooled
  correctly (within + between); the matrix path did not.
* **Every node's raw-data fingerprint was republished.** The global model's
  ``data_hash`` was the comma-join of each node's SHA-256 over its training
  bytes, handing every consumer of the global model a membership oracle for
  every node's private data.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector
from omni_mercury_engine.federation.aggregator import (
    FederatedAggregator,
    _cohort_provenance,
    _pool_precision,
    _spd_inverse,
)
from omni_mercury_engine.federation.node import FederatedNode

_N_FEATURES = 4
_PER_NODE = 400


def _split_population(gap: float, seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two node shards separated by ``gap``, plus their concatenation."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((_PER_NODE, _N_FEATURES))
    b = rng.standard_normal((_PER_NODE, _N_FEATURES)) + gap
    return a, b, np.vstack([a, b])


def _aggregate(a: np.ndarray, b: np.ndarray) -> tuple[object, object, object]:
    node_a = FederatedNode("node_a")
    node_a.fit(a)
    node_b = FederatedNode("node_b")
    node_b.fit(b)
    stats_a = node_a.export_statistics()
    stats_b = node_b.export_statistics()
    agg = FederatedAggregator(min_nodes=2)
    agg.submit(stats_a)
    agg.submit(stats_b)
    return agg.aggregate(), stats_a, stats_b


def _relative_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(estimate - truth) / np.linalg.norm(truth))


class TestPooledPrecision:
    """The pooled precision must track a centralized fit on the same data."""

    @pytest.mark.parametrize("gap", (0.0, 1.0, 4.0))
    def test_beats_the_plain_average_against_a_centralized_fit(self, gap: float) -> None:
        """Measured, not asserted by construction.

        Reference numbers for this fixture (relative Frobenius error against a
        centralized ``MercuryAnomalyDetector.fit`` on the concatenated data)::

            gap   pooled    plain average
            0.0   0.0012    0.0055
            1.0   0.0012    0.2756
            4.0   0.0013    0.5587

        The pooled error is flat in the gap; the plain average degrades with
        it, because the term it omits is exactly the between-node spread.
        """
        node_a, node_b, full = _split_population(gap)
        central = MercuryAnomalyDetector()
        central.fit(full)

        aggregated, stats_a, stats_b = _aggregate(node_a, node_b)
        plain_average = 0.5 * stats_a.ig_cov_inv + 0.5 * stats_b.ig_cov_inv

        pooled_error = _relative_error(aggregated.ig_cov_inv, central._ig_cov_inv)
        average_error = _relative_error(plain_average, central._ig_cov_inv)

        assert pooled_error < 0.01, f"gap={gap}: pooled error {pooled_error:.4f}"
        assert pooled_error <= average_error

    def test_the_plain_average_really_does_degrade_with_separation(self) -> None:
        """Guards the comparison above from being vacuous."""
        errors = []
        for gap in (0.0, 1.0, 4.0):
            node_a, node_b, full = _split_population(gap)
            central = MercuryAnomalyDetector()
            central.fit(full)
            _, stats_a, stats_b = _aggregate(node_a, node_b)
            plain = 0.5 * stats_a.ig_cov_inv + 0.5 * stats_b.ig_cov_inv
            errors.append(_relative_error(plain, central._ig_cov_inv))
        assert errors == sorted(errors)
        assert errors[-1] > 20 * errors[0]

    @pytest.mark.parametrize("gap", (0.0, 1.0, 4.0))
    def test_log_determinant_tracks_the_centralized_value(self, gap: float) -> None:
        """``ig_log_det`` is log det(Sigma); it must match the pooled spread."""
        node_a, node_b, full = _split_population(gap)
        central = MercuryAnomalyDetector()
        central.fit(full)
        aggregated, _, _ = _aggregate(node_a, node_b)
        assert aggregated.ig_log_det == pytest.approx(central._ig_log_det, abs=0.05)

    def test_between_group_term_is_present(self) -> None:
        """Two identical-covariance nodes with separated means pool wider.

        With the plain-average rule the result would be identical to a single
        node's precision, because the mean separation never enters it.
        """
        identity = np.eye(3)
        precision, covariance = _pool_precision(
            [identity, identity],
            [np.zeros(3), np.array([6.0, 0.0, 0.0])],
            np.array([3.0, 0.0, 0.0]),
            np.array([0.5, 0.5]),
        )
        # Within = I; between = diag(9, 0, 0) => pooled cov diag = (10, 1, 1).
        np.testing.assert_allclose(np.diag(covariance), [10.0, 1.0, 1.0], rtol=1e-12)
        np.testing.assert_allclose(np.diag(precision), [0.1, 1.0, 1.0], rtol=1e-9)

    def test_result_is_symmetric_positive_definite(self) -> None:
        node_a, node_b, _ = _split_population(2.0)
        aggregated, _, _ = _aggregate(node_a, node_b)
        matrix = aggregated.ig_cov_inv
        np.testing.assert_allclose(matrix, matrix.T, rtol=0, atol=1e-15)
        assert float(np.min(np.linalg.eigvalsh(matrix))) > 0.0

    def test_survives_an_indefinite_submission(self) -> None:
        """A DP-noised precision can go indefinite; pooling must not blow up.

        The plain average happily produced an indefinite global precision,
        which turns the reconstructed detector's Mahalanobis form into
        something that can return negative squared distances.
        """
        indefinite = np.diag([1.0, -0.5, 2.0])
        precision, covariance = _pool_precision(
            [indefinite, np.eye(3)],
            [np.zeros(3), np.zeros(3)],
            np.zeros(3),
            np.array([0.5, 0.5]),
        )
        assert float(np.min(np.linalg.eigvalsh(precision))) > 0.0
        assert float(np.min(np.linalg.eigvalsh(covariance))) > 0.0

    def test_spd_inverse_round_trips_a_well_conditioned_matrix(self) -> None:
        rng = np.random.default_rng(3)
        factor = rng.standard_normal((5, 5))
        spd = factor @ factor.T + 5.0 * np.eye(5)
        np.testing.assert_allclose(_spd_inverse(_spd_inverse(spd)), spd, rtol=1e-8)


class TestCohortProvenance:
    """The global model must not carry any node's data fingerprint."""

    def test_node_fingerprints_do_not_survive_aggregation(self) -> None:
        node_a, node_b, _ = _split_population(1.0)
        aggregated, stats_a, stats_b = _aggregate(node_a, node_b)
        assert stats_a.data_hash
        assert stats_b.data_hash
        assert stats_a.data_hash not in aggregated.data_hash
        assert stats_b.data_hash not in aggregated.data_hash

    def test_provenance_is_reproducible_from_cohort_and_round(self) -> None:
        first = _cohort_provenance(["node_b", "node_a"], 0)
        assert first == _cohort_provenance(["node_a", "node_b"], 0)
        assert first != _cohort_provenance(["node_a", "node_b"], 1)
        assert first != _cohort_provenance(["node_a", "node_c"], 0)

    def test_provenance_ignores_the_data_entirely(self) -> None:
        """Same nodes, different data => same provenance token."""
        first, _, _ = _aggregate(*_split_population(0.0, seed=1)[:2])
        second, _, _ = _aggregate(*_split_population(9.0, seed=2)[:2])
        assert first.data_hash == second.data_hash


class TestEndToEnd:
    """The reconstructed global detector behaves like a centralized one."""

    def test_scores_correlate_with_a_centralized_fit(self) -> None:
        node_a, node_b, full = _split_population(1.0)
        rng = np.random.default_rng(99)
        probe = rng.standard_normal((200, _N_FEATURES))

        central = MercuryAnomalyDetector()
        central.fit(full)
        central_scores = central.detect(probe)["scores"]

        aggregated, _, _ = _aggregate(node_a, node_b)
        federated_scores = FederatedAggregator.to_detector(aggregated).detect(probe)["scores"]

        correlation = float(np.corrcoef(central_scores, federated_scores)[0, 1])
        assert correlation > 0.85, f"score correlation {correlation:.3f}"
