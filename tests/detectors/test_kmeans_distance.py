# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the revived k-means-distance detector (``omni_mercury_engine.detectors.kmeans_distance``), which promotes the previously-dormant ``cognitive.neural_memory_layer.KMeansClusterer`` to a first-class fusion detector. These verify the base-detector contract and that the distance signal actually separates anomalies on a clearly-separable fixture (``benchmarks/dormant_module_revival.py`` measures it on real ADBench labels)."""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.kmeans_distance import KMeansDistanceDetector
from omni_mercury_engine.ml.mercury_ml import roc_auc_score


def _separable(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    normal = rng.normal(0.0, 1.0, (300, 8))
    anomaly = rng.normal(4.0, 1.0, (30, 8))
    X = np.vstack([normal, anomaly]).astype(np.float32)
    y = np.concatenate([np.zeros(300), np.ones(30)]).astype(int)
    order = rng.permutation(len(X))
    return X[order], y[order]


class TestContract:
    def test_unfitted_then_fitted(self) -> None:
        det = KMeansDistanceDetector(n_clusters=4)
        assert det.is_fitted() is False
        det.fit(_separable()[0])
        assert det.is_fitted() is True

    def test_extract_features_shape(self) -> None:
        X, _ = _separable()
        det = KMeansDistanceDetector(n_clusters=8).fit(X)
        feats = det.extract_features(X)
        # one distance per centroid plus the nearest-centroid distance.
        assert feats.shape == (len(X), 8 + 1)
        assert np.isfinite(feats).all()

    def test_extract_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="must be fit"):
            KMeansDistanceDetector().extract_features(_separable()[0])

    def test_more_clusters_than_samples_clamped(self) -> None:
        X = np.random.RandomState(0).normal(size=(3, 8)).astype(np.float32)
        det = KMeansDistanceDetector(n_clusters=8).fit(X)
        assert det.extract_features(X).shape[0] == 3

    def test_invalid_n_clusters(self) -> None:
        with pytest.raises(ValueError):
            KMeansDistanceDetector(n_clusters=0)

    def test_single_sample_1d_input(self) -> None:
        # A 1-D vector is one sample of n_features (not n_features samples):
        # extract_features -> (1, k+1) and detect -> (1,), not transposed.
        det = KMeansDistanceDetector(n_clusters=4).fit(_separable()[0])
        assert det.extract_features(np.zeros(8, dtype=np.float32)).shape == (1, 4 + 1)
        assert det.detect(np.zeros(8, dtype=np.float32))["scores"].shape == (1,)


class TestSignal:
    def test_nearest_distance_separates_anomalies(self) -> None:
        # Fit the clusterer on normal structure, then score a held-out mix by
        # nearest-centroid distance; points far from every learned centroid (the
        # anomalies) must rank above normals -- the property the revival measured
        # on real ADBench labels. (Fitting on normal structure isolates the
        # distance signal from the separate question of contaminated training.)
        rng = np.random.RandomState(0)
        normal_train = rng.normal(0.0, 1.0, (300, 8)).astype(np.float32)
        Xte, yte = _separable(1)
        det = KMeansDistanceDetector(n_clusters=8).fit(normal_train)
        nearest = det.extract_features(Xte)[:, -1]
        assert roc_auc_score(yte, nearest) > 0.9

    def test_detect_returns_unit_interval_scores(self) -> None:
        # detect() is the live-inference contract: per-sample scores in [0, 1].
        X, _ = _separable()
        out = KMeansDistanceDetector(n_clusters=8).fit(X).detect(X)
        scores = np.asarray(out["scores"])
        assert scores.shape == (len(X),)
        assert float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0

    def test_detect_scores_separate_anomalies(self) -> None:
        # detect() must carry the same signal as the distance feature.
        normal_train = np.random.RandomState(0).normal(0.0, 1.0, (300, 8)).astype(np.float32)
        Xte, yte = _separable(1)
        det = KMeansDistanceDetector(n_clusters=8).fit(normal_train)
        assert roc_auc_score(yte, det.detect(Xte)["scores"]) > 0.9
