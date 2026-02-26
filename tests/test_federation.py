"""Tests for Mercury-native federated anomaly detection.

Covers: FederatedNode, FederatedAggregator, FittedStatistics,
and end-to-end federation with real domain data.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector
from omni_mercury_engine.federation.aggregator import FederatedAggregator
from omni_mercury_engine.federation.node import FederatedNode
from omni_mercury_engine.federation.statistics import FittedStatistics

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)
_N_FEATURES = 5
_N_SAMPLES_A = 200
_N_SAMPLES_B = 300


@pytest.fixture()
def data_a() -> np.ndarray:
    """Synthetic data for node A."""
    return _RNG.standard_normal((_N_SAMPLES_A, _N_FEATURES))


@pytest.fixture()
def data_b() -> np.ndarray:
    """Synthetic data for node B (shifted distribution)."""
    return _RNG.standard_normal((_N_SAMPLES_B, _N_FEATURES)) + 1.0


@pytest.fixture()
def node_a(data_a: np.ndarray) -> FederatedNode:
    node = FederatedNode("node_a")
    node.fit(data_a)
    return node


@pytest.fixture()
def node_b(data_b: np.ndarray) -> FederatedNode:
    node = FederatedNode("node_b")
    node.fit(data_b)
    return node


@pytest.fixture()
def stats_a(node_a: FederatedNode) -> FittedStatistics:
    return node_a.export_statistics()


@pytest.fixture()
def stats_b(node_b: FederatedNode) -> FittedStatistics:
    return node_b.export_statistics()


# ---------------------------------------------------------------------------
# Test 1: node fit and export
# ---------------------------------------------------------------------------


def test_node_fit_export(node_a: FederatedNode) -> None:
    """Fit a node on synthetic data, export statistics, verify all 13
    fields populated and shapes match."""
    stats = node_a.export_statistics()

    assert stats.node_id == "node_a"
    assert stats.n_samples == _N_SAMPLES_A
    assert stats.n_features == _N_FEATURES

    # All 13 fitted fields must be populated
    for attr in ["mean", "std", "q1", "q3"]:
        arr = getattr(stats, attr)
        assert arr.shape == (_N_FEATURES,), f"{attr} shape mismatch"

    for attr in [
        "res_h_train",
        "res_noise_ratio",
        "kin_jerk_mean",
        "kin_jerk_std",
        "kin_accel_mean",
        "kin_accel_std",
        "ig_mean",
    ]:
        arr = getattr(stats, attr)
        assert arr.shape == (_N_FEATURES,), f"{attr} shape mismatch"

    assert stats.ig_cov_inv.shape == (_N_FEATURES, _N_FEATURES)
    assert isinstance(stats.ig_log_det, float)

    # Data hash must be populated
    assert len(stats.data_hash) > 0


# ---------------------------------------------------------------------------
# Test 2: statistics match detector
# ---------------------------------------------------------------------------


def test_statistics_match_detector(data_a: np.ndarray) -> None:
    """Fit a node, export stats, compare each exported field against
    the detector's internal attribute (must be identical before DP)."""
    node = FederatedNode("test")
    node.fit(data_a)
    stats = node.export_statistics()  # no DP

    det = node._detector

    np.testing.assert_array_equal(stats.mean, det.mean)
    np.testing.assert_array_equal(stats.std, det.std)
    np.testing.assert_array_equal(stats.q1, det.q1)
    np.testing.assert_array_equal(stats.q3, det.q3)
    np.testing.assert_array_equal(stats.res_h_train, det._res_h_train)
    np.testing.assert_array_equal(stats.res_noise_ratio, det._res_noise_ratio)
    np.testing.assert_array_equal(stats.kin_jerk_mean, det._kin_jerk_mean)
    np.testing.assert_array_equal(stats.kin_jerk_std, det._kin_jerk_std)
    np.testing.assert_array_equal(stats.kin_accel_mean, det._kin_accel_mean)
    np.testing.assert_array_equal(stats.kin_accel_std, det._kin_accel_std)
    np.testing.assert_array_equal(stats.ig_mean, det._ig_mean)
    np.testing.assert_array_equal(stats.ig_cov_inv, det._ig_cov_inv)
    assert stats.ig_log_det == det._ig_log_det


# ---------------------------------------------------------------------------
# Test 3: aggregator two nodes
# ---------------------------------------------------------------------------


def test_aggregator_two_nodes(
    stats_a: FittedStatistics,
    stats_b: FittedStatistics,
) -> None:
    """Two nodes with different data distributions, verify aggregated
    stats are between the two."""
    agg = FederatedAggregator(min_nodes=2)
    agg.submit(stats_a)
    agg.submit(stats_b)
    global_stats = agg.aggregate()

    # Aggregated mean should be between node A and node B means
    for i in range(_N_FEATURES):
        lo = min(stats_a.mean[i], stats_b.mean[i])
        hi = max(stats_a.mean[i], stats_b.mean[i])
        assert lo <= global_stats.mean[i] <= hi, (
            f"Feature {i}: aggregated mean {global_stats.mean[i]} " f"not between {lo} and {hi}"
        )

    assert global_stats.n_samples == _N_SAMPLES_A + _N_SAMPLES_B
    assert global_stats.n_features == _N_FEATURES


# ---------------------------------------------------------------------------
# Test 4: aggregator weighted by sample size
# ---------------------------------------------------------------------------


def test_aggregator_weighted_by_sample_size() -> None:
    """Node with 1000 samples should dominate over node with 10 samples."""
    rng = np.random.default_rng(123)
    data_large = rng.standard_normal((1000, 3))
    data_small = rng.standard_normal((10, 3)) + 100.0

    node_large = FederatedNode("large")
    node_large.fit(data_large)
    stats_large = node_large.export_statistics()

    node_small = FederatedNode("small")
    node_small.fit(data_small)
    stats_small = node_small.export_statistics()

    agg = FederatedAggregator(min_nodes=2)
    agg.submit(stats_large)
    agg.submit(stats_small)
    global_stats = agg.aggregate()

    # Aggregated mean should be much closer to large node's mean
    dist_to_large = np.linalg.norm(global_stats.mean - stats_large.mean)
    dist_to_small = np.linalg.norm(global_stats.mean - stats_small.mean)
    assert dist_to_large < dist_to_small, (
        f"Aggregated mean should be closer to large node "
        f"(dist_large={dist_to_large:.4f}, dist_small={dist_to_small:.4f})"
    )


# ---------------------------------------------------------------------------
# Test 5: to_detector produces working detector
# ---------------------------------------------------------------------------


def test_to_detector_produces_working_detector(
    stats_a: FittedStatistics,
    stats_b: FittedStatistics,
) -> None:
    """Aggregate stats, call to_detector(), call detect() on new data,
    verify scores are in [0, 1] and is_anomaly is boolean."""
    agg = FederatedAggregator(min_nodes=2)
    agg.submit(stats_a)
    agg.submit(stats_b)
    global_stats = agg.aggregate()

    det = FederatedAggregator.to_detector(global_stats)
    test_data = np.random.default_rng(99).standard_normal((20, _N_FEATURES))
    result = det.detect(test_data)

    assert result["scores"].shape == (20,)
    assert result["is_anomaly"].shape == (20,)
    assert np.all(result["scores"] >= 0.0)
    assert np.all(result["scores"] <= 1.0)
    assert result["is_anomaly"].dtype == bool or np.issubdtype(result["is_anomaly"].dtype, np.bool_)


# ---------------------------------------------------------------------------
# Test 6: federated detector matches centralized (approximately)
# ---------------------------------------------------------------------------


def test_federated_detector_matches_centralized() -> None:
    """Split data in half, fit two nodes, aggregate, reconstruct
    detector. Compare scores against centralized detector fit on
    full data. Scores should be correlated."""
    rng = np.random.default_rng(777)
    full_data = rng.standard_normal((400, 4))
    test_data = rng.standard_normal((50, 4))

    # Centralized
    centralized = MercuryAnomalyDetector()
    centralized.fit(full_data)
    central_scores = centralized.detect(test_data)["scores"]

    # Federated
    node1 = FederatedNode("half_1")
    node1.fit(full_data[:200])
    node2 = FederatedNode("half_2")
    node2.fit(full_data[200:])

    agg = FederatedAggregator(min_nodes=2)
    agg.submit(node1.export_statistics())
    agg.submit(node2.export_statistics())
    fed_det = FederatedAggregator.to_detector(agg.aggregate())
    fed_scores = fed_det.detect(test_data)["scores"]

    # Scores should be positively correlated
    correlation = np.corrcoef(central_scores, fed_scores)[0, 1]
    assert (
        correlation > 0.5
    ), f"Federated and centralized scores poorly correlated: r={correlation:.3f}"


# ---------------------------------------------------------------------------
# Test 7: differential privacy adds noise
# ---------------------------------------------------------------------------


def test_differential_privacy_adds_noise(
    node_a: FederatedNode,
) -> None:
    """Export with epsilon=1.0, compare to raw export. Arrays must differ."""
    raw = node_a.export_statistics()
    noised = node_a.export_statistics(epsilon=1.0)

    # At least some fields should differ
    diffs = 0
    for attr in ["mean", "std", "q1", "q3", "ig_mean"]:
        if not np.array_equal(getattr(raw, attr), getattr(noised, attr)):
            diffs += 1

    assert diffs > 0, "DP should have modified at least some statistics"
    assert noised.epsilon == 1.0


# ---------------------------------------------------------------------------
# Test 8: stronger epsilon => more noise
# ---------------------------------------------------------------------------


def test_dp_stronger_epsilon_more_noise(data_a: np.ndarray) -> None:
    """epsilon=0.1 should produce more noise than epsilon=10.0."""
    node = FederatedNode("test")
    node.fit(data_a)
    raw = node.export_statistics()

    # Run multiple trials and average L2 distance
    n_trials = 5
    dist_strong = 0.0
    dist_weak = 0.0

    for _ in range(n_trials):
        strong = node.export_statistics(epsilon=0.1)
        weak = node.export_statistics(epsilon=10.0)

        dist_strong += np.linalg.norm(strong.mean - raw.mean)
        dist_weak += np.linalg.norm(weak.mean - raw.mean)

    dist_strong /= n_trials
    dist_weak /= n_trials

    assert dist_strong > dist_weak, (
        f"Stronger privacy (eps=0.1, dist={dist_strong:.6f}) should "
        f"produce more noise than weaker (eps=10.0, dist={dist_weak:.6f})"
    )


# ---------------------------------------------------------------------------
# Test 9: DP preserves non-negative
# ---------------------------------------------------------------------------


def test_dp_preserves_non_negative(node_a: FederatedNode) -> None:
    """Noised std, kin_*_std, res_noise_ratio must remain positive."""
    noised = node_a.export_statistics(epsilon=1.0)

    assert np.all(noised.std > 0), "std must remain positive after DP"
    assert np.all(noised.kin_jerk_std > 0), "kin_jerk_std must remain positive"
    assert np.all(noised.kin_accel_std > 0), "kin_accel_std must remain positive"
    assert np.all(noised.res_noise_ratio > 0), "res_noise_ratio must remain positive"


# ---------------------------------------------------------------------------
# Test 10: serialization roundtrip
# ---------------------------------------------------------------------------


def test_serialization_roundtrip(stats_a: FittedStatistics) -> None:
    """FittedStatistics.to_dict() -> from_dict() preserves all values."""
    d = stats_a.to_dict()
    restored = FittedStatistics.from_dict(d)

    assert restored.node_id == stats_a.node_id
    assert restored.n_samples == stats_a.n_samples
    assert restored.n_features == stats_a.n_features
    assert restored.ig_log_det == pytest.approx(stats_a.ig_log_det)

    for attr in [
        "mean",
        "std",
        "q1",
        "q3",
        "res_h_train",
        "res_noise_ratio",
        "kin_jerk_mean",
        "kin_jerk_std",
        "kin_accel_mean",
        "kin_accel_std",
        "ig_mean",
        "ig_cov_inv",
    ]:
        np.testing.assert_allclose(
            getattr(restored, attr),
            getattr(stats_a, attr),
            rtol=1e-10,
            err_msg=f"{attr} roundtrip mismatch",
        )


# ---------------------------------------------------------------------------
# Test 11: aggregator rejects stale stats
# ---------------------------------------------------------------------------


def test_aggregator_rejects_stale_stats(stats_a: FittedStatistics) -> None:
    """Stats older than max_age_seconds should raise ValueError."""
    stale = FittedStatistics(
        node_id="stale",
        timestamp=time.time() - 200000,  # ~2.3 days old
        n_samples=stats_a.n_samples,
        n_features=stats_a.n_features,
        mean=stats_a.mean,
        std=stats_a.std,
        q1=stats_a.q1,
        q3=stats_a.q3,
        res_h_train=stats_a.res_h_train,
        res_noise_ratio=stats_a.res_noise_ratio,
        kin_jerk_mean=stats_a.kin_jerk_mean,
        kin_jerk_std=stats_a.kin_jerk_std,
        kin_accel_mean=stats_a.kin_accel_mean,
        kin_accel_std=stats_a.kin_accel_std,
        ig_mean=stats_a.ig_mean,
        ig_cov_inv=stats_a.ig_cov_inv,
        ig_log_det=stats_a.ig_log_det,
    )

    agg = FederatedAggregator(max_age_seconds=86400)
    with pytest.raises(ValueError, match="old"):
        agg.submit(stale)


# ---------------------------------------------------------------------------
# Test 12: aggregator min nodes
# ---------------------------------------------------------------------------


def test_aggregator_min_nodes(stats_a: FittedStatistics) -> None:
    """Aggregation with fewer than min_nodes should raise RuntimeError."""
    agg = FederatedAggregator(min_nodes=3)
    agg.submit(stats_a)

    with pytest.raises(RuntimeError, match="Need 3 nodes"):
        agg.aggregate()


# ---------------------------------------------------------------------------
# Test 13: aggregator rejects dimension mismatch
# ---------------------------------------------------------------------------


def test_aggregator_rejects_dimension_mismatch() -> None:
    """Node A with 5 features, Node B with 8 features should raise."""
    rng = np.random.default_rng(99)

    node_5 = FederatedNode("five")
    node_5.fit(rng.standard_normal((100, 5)))
    stats_5 = node_5.export_statistics()

    node_8 = FederatedNode("eight")
    node_8.fit(rng.standard_normal((100, 8)))
    stats_8 = node_8.export_statistics()

    agg = FederatedAggregator(min_nodes=2)
    agg.submit(stats_5)

    with pytest.raises(ValueError, match="features"):
        agg.submit(stats_8)


# ---------------------------------------------------------------------------
# Test 14: from_statistics classmethod
# ---------------------------------------------------------------------------


def test_from_statistics_classmethod(data_a: np.ndarray) -> None:
    """Verify from_statistics produces a working detector."""
    det = MercuryAnomalyDetector(enable_ama=False)
    det.fit(data_a)

    reconstructed = MercuryAnomalyDetector.from_statistics(
        mean=det.mean,
        std=det.std,
        q1=det.q1,
        q3=det.q3,
        res_h_train=det._res_h_train,
        res_noise_ratio=det._res_noise_ratio,
        kin_jerk_mean=det._kin_jerk_mean,
        kin_jerk_std=det._kin_jerk_std,
        kin_accel_mean=det._kin_accel_mean,
        kin_accel_std=det._kin_accel_std,
        ig_mean=det._ig_mean,
        ig_cov_inv=det._ig_cov_inv,
        ig_log_det=det._ig_log_det,
        adaptive_weights=getattr(det, "_adaptive_weights", None),
        data_type=det._data_type.value if hasattr(det, "_data_type") else None,
        oracle_ref_stats=det.get_oracle_statistics(),
    )

    test = np.random.default_rng(42).standard_normal((10, _N_FEATURES))
    orig_result = det.detect(test)
    recon_result = reconstructed.detect(test)

    np.testing.assert_allclose(
        orig_result["scores"],
        recon_result["scores"],
        rtol=1e-10,
        err_msg="from_statistics detector should produce identical scores",
    )


# ---------------------------------------------------------------------------
# Test 15: federation end-to-end with real earthquake data
# ---------------------------------------------------------------------------


def test_federation_end_to_end_with_real_domain() -> None:
    """Federation with real EarthquakeLoader data — NOT synthetic."""
    try:
        from omni_mercury_engine.loaders.earthquake_loader import EarthquakeLoader

        loader = EarthquakeLoader()
        events = loader.list_events()
        if not events:
            pytest.skip("No earthquake events available (network issue)")

        eid = events[0]["event_id"]
        raw_data = loader.fetch_historical(eid)
        X = loader.engineer_features(raw_data)

        if X is None or len(X) < 30:
            pytest.skip(f"Insufficient earthquake data: " f"N={len(X) if X is not None else 0}")

        # Split into 3 partitions
        n = len(X)
        idx = np.random.default_rng(42).permutation(n)
        splits = np.array_split(idx, 3)

        nodes = []
        for i, split_idx in enumerate(splits):
            node = FederatedNode(node_id=f"eq_node_{i}")
            node.fit(X[split_idx])
            nodes.append(node)

        agg = FederatedAggregator(min_nodes=2)
        for node in nodes:
            stats = node.export_statistics()
            agg.submit(stats)

        global_stats = agg.aggregate()
        det = FederatedAggregator.to_detector(global_stats)
        result = det.detect(X)

        scores = np.asarray(result["scores"])
        is_anomaly = np.asarray(result["is_anomaly"])

        assert len(scores) == n
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)
        assert is_anomaly.dtype == bool or np.issubdtype(is_anomaly.dtype, np.bool_)

    except ImportError:
        pytest.skip("EarthquakeLoader not available")
    except ConnectionError as e:
        pytest.skip(f"Network issue: {e}")
