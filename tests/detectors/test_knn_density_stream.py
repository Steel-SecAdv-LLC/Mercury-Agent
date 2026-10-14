# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""kNN density stream fused into the statistical ensemble."""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.statistical import StatisticalAnomalyDetector
from omni_mercury_engine.ml.mercury_ml import roc_auc_score


def _two_cluster_normals(n: int, seed: int) -> np.ndarray:
    """Bimodal normal data: the gap between the modes is anomalous."""
    rng = np.random.default_rng(seed)
    half = n // 2
    return np.vstack([rng.normal(-4.0, 0.5, (half, 2)), rng.normal(4.0, 0.5, (n - half, 2))])


@pytest.fixture(scope="module")
def fitted() -> StatisticalAnomalyDetector:
    det = StatisticalAnomalyDetector()
    det.fit(np.random.default_rng(0).normal(size=(400, 5)))
    return det


def test_stream_is_fitted_by_default(fitted: StatisticalAnomalyDetector) -> None:
    assert fitted._knn_tree is not None
    assert fitted._knn_k == StatisticalAnomalyDetector._KNN_K
    result = fitted.detect(np.random.default_rng(1).normal(size=(20, 5)))
    assert result["knn_density_scores"] is not None
    assert np.asarray(result["knn_density_scores"]).shape == (20,)


def test_disabled_stream_reproduces_three_component_ensemble() -> None:
    X = np.random.default_rng(2).normal(size=(300, 4))
    test = np.random.default_rng(3).normal(size=(50, 4))
    det = StatisticalAnomalyDetector({"knn_stream": False}).fit(X)
    result = det.detect(test)
    assert det._knn_tree is None
    assert result["knn_density_scores"] is None
    base, _ = det._ensemble_base_scores(test)
    np.testing.assert_array_equal(result["scores"], base)


def test_single_row_scores_match_batch(fitted: StatisticalAnomalyDetector) -> None:
    batch = np.random.default_rng(4).normal(size=(12, 5))
    batch_knn = np.asarray(fitted.detect(batch)["knn_density_scores"])
    single = [float(fitted.detect(row[None, :])["knn_density_scores"][0]) for row in batch]
    np.testing.assert_allclose(single, batch_knn, rtol=0, atol=1e-12)


def test_training_row_is_not_its_own_neighbour() -> None:
    X = np.random.default_rng(5).normal(size=(200, 3))
    det = StatisticalAnomalyDetector().fit(X)
    assert det._knn_center is not None and det._knn_scale is not None
    z = (X - det._knn_center) / det._knn_scale
    k = det._knn_k
    dist, _ = det._knn_tree.query(z, k=k + 1)
    leave_one_out = np.asarray(dist)[:, 1:].mean(axis=1)
    np.testing.assert_allclose(det._knn_distances(X), leave_one_out, rtol=1e-12)


def test_gap_between_modes_is_detected_only_with_the_stream() -> None:
    X = _two_cluster_normals(600, seed=6)
    rng = np.random.default_rng(7)
    normals = _two_cluster_normals(200, seed=8)
    gap = rng.normal(0.0, 0.5, (40, 2))
    test = np.vstack([normals, gap])
    y = np.r_[np.zeros(len(normals)), np.ones(len(gap))]

    off = StatisticalAnomalyDetector({"knn_stream": False}).fit(X).detect(test)
    on = StatisticalAnomalyDetector().fit(X).detect(test)
    auc_off = roc_auc_score(y, np.asarray(off["scores"]))
    auc_on = roc_auc_score(y, np.asarray(on["scores"]))
    # A single Gaussian manifold sees the gap as its centre (not anomalous);
    # local density does not.
    assert auc_on > 0.95
    assert auc_on > auc_off + 0.2


def test_upper_tail_is_strictly_increasing() -> None:
    v = np.array([0.0, 0.5, 0.8, 0.9, 2.0, 50.0, 5e3])
    out = StatisticalAnomalyDetector._saturate_upper(v)
    assert np.all(np.diff(out) > 0)
    assert np.all(out < 1.0)
    np.testing.assert_array_equal(out[:3], v[:3])


def test_far_shift_is_flagged_and_ranked(fitted: StatisticalAnomalyDetector) -> None:
    rng = np.random.default_rng(9)
    batch = np.vstack([rng.normal(size=(30, 5)), rng.normal(8.0, 1.0, (30, 5))])
    result = fitted.detect(batch)
    scores = np.asarray(result["scores"])
    flags = np.asarray(result["is_anomaly"], dtype=bool)
    assert scores[30:].min() > scores[:30].max()
    assert flags[30:].all()
    assert flags[:30].sum() <= 3


def test_live_reference_tracks_weight_changes() -> None:
    det = StatisticalAnomalyDetector().fit(np.random.default_rng(10).normal(size=(300, 4)))
    test = np.random.default_rng(11).normal(size=(10, 4))
    det.detect(test)
    assert det._knn_live_stats is not None
    key_before = det._knn_live_stats[0]
    det._adaptive_weights = np.array([0.2, 0.2, 0.6])
    det.detect(test)
    assert det._knn_live_stats[0] != key_before
    expected = det._robust_location_scale(det._ensemble_base_scores(det._train_data)[0])
    assert expected is not None
    assert det._knn_live_stats[1:] == pytest.approx(expected)


def test_fitted_state_round_trip_restores_the_stream(
    fitted: StatisticalAnomalyDetector,
) -> None:
    state = fitted.get_fitted_state()
    assert state is not None and state["knn_stream"] is not None
    restored = StatisticalAnomalyDetector()
    restored.set_fitted_state(state)
    test = np.random.default_rng(12).normal(size=(25, 5))
    np.testing.assert_allclose(
        restored.detect(test)["scores"], fitted.detect(test)["scores"], rtol=1e-12
    )


def test_legacy_state_without_stream_scores_as_before(
    fitted: StatisticalAnomalyDetector,
) -> None:
    state = fitted.get_fitted_state()
    assert state is not None
    state.pop("knn_stream")
    restored = StatisticalAnomalyDetector()
    restored.set_fitted_state(state)
    test = np.random.default_rng(13).normal(size=(25, 5))
    result = restored.detect(test)
    assert result["knn_density_scores"] is None
    base, _ = restored._ensemble_base_scores(test)
    np.testing.assert_array_equal(result["scores"], base)


def test_reference_tree_is_capped() -> None:
    X = np.random.default_rng(14).normal(size=(5000, 2))
    det = StatisticalAnomalyDetector().fit(X)
    assert det._knn_tree_index is not None
    assert len(det._knn_tree_index) == StatisticalAnomalyDetector._KNN_MAX_REFERENCE
