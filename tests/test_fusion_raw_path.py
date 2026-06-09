# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Reproducibility tests for the raw fusion training path (Issue: expose train -> fit_fusion(X, y) on raw input with no manual glue).

These verify that:
  * ``fit_fusion`` trained on raw features reaches the headline detection
    quality (ROC-AUC >= 0.90) on a deterministic, network-free fixture, and
  * the offline ``build_feature_npz`` builder yields an archive that
    ``train_fusion_model`` consumes, producing the same detector feature set
    that ``fit_fusion`` extracts internally.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

pytestmark = pytest.mark.xdist_group("fusion_raw_path")

from omni_mercury_engine.ml.mercury_ml import roc_auc_score


def _separable_fixture(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic, clearly-separable anomaly fixture (no network).

    A compact normal Gaussian cluster plus a shifted anomaly cluster. The
    signal is strong enough that a correctly-wired raw training path reaches
    ROC-AUC >= 0.90 reliably, while still exercising the full detector
    fit -> feature-extraction -> fusion-train pipeline end to end.
    """
    rng = np.random.RandomState(seed)
    n_normal, n_anom, dim = 500, 60, 12
    normal = rng.normal(0.0, 1.0, (n_normal, dim))
    anomaly = rng.normal(3.0, 1.0, (n_anom, dim))
    X = np.vstack([normal, anomaly]).astype(np.float32)
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anom)]).astype(np.int64)
    order = rng.permutation(len(X))
    return X[order], y[order]


@pytest.fixture
def engine() -> Any:
    from omni_mercury_engine.engine import OmniMercuryEngine

    return OmniMercuryEngine(mode="fusion", device="cpu")


class TestRawTrainingPath:
    """fit_fusion(X, y) reproduces headline AUC from raw input."""

    def test_fit_fusion_reaches_headline_auc(self, engine: Any) -> None:
        torch.manual_seed(0)
        np.random.seed(0)

        X, y = _separable_fixture()
        n_train = int(len(X) * 0.7)
        X_tr, y_tr = X[:n_train], y[:n_train]
        X_te, y_te = X[n_train:], y[n_train:]

        engine.fit_fusion(X_tr, y_tr, epochs=40, batch_size=32, early_stopping_patience=12)

        probs = engine.score_fusion(X_te)
        assert probs.shape == (len(X_te),)
        assert np.all((probs >= 0.0) & (probs <= 1.0))

        auc = roc_auc_score(y_te, probs)
        assert auc >= 0.90, f"Raw fit_fusion path AUC {auc:.4f} below 0.90 headline target"

    def test_score_fusion_requires_fusion_mode(self) -> None:
        from omni_mercury_engine.engine import OmniMercuryEngine

        eng = OmniMercuryEngine(mode="statistical", device="cpu")
        with pytest.raises(ValueError, match="requires mode='fusion'"):
            eng.score_fusion(np.zeros((4, 5), dtype=np.float32))


class TestFeatureNpzBuilder:
    """build_feature_npz bridges raw input to the .npz training path."""

    def test_build_then_train_roundtrip(self, engine: Any, tmp_path: Any) -> None:
        X, y = _separable_fixture()
        out = str(tmp_path / "feats.npz")

        path = engine.build_feature_npz(X, out, y=y)
        archive = np.load(path)

        assert "labels" in archive.files
        assert np.array_equal(archive["labels"], y)
        feature_keys = [k for k in archive.files if k != "labels"]
        assert feature_keys, "archive should contain detector feature groups"
        for key in feature_keys:
            assert archive[key].shape[0] == len(X)

        # The archive trains directly via the pre-extracted-feature path.
        from omni_mercury_engine.engine import OmniMercuryEngine

        trainer = OmniMercuryEngine(mode="fusion", device="cpu")
        metrics = trainer.train_fusion_model(path, epochs=5, batch_size=32)
        assert metrics["epochs_trained"] > 0

    def test_builder_feature_groups_match_fit_fusion(self, engine: Any, tmp_path: Any) -> None:
        """Archive exposes the same detector feature groups (keys + shapes) that
        fit_fusion extracts internally.

        Note: a bit-exact value comparison is intentionally avoided — a few base
        detectors have stochastic feature extraction, so the contract the
        builder guarantees is structural (it runs the identical shared
        extraction code path), not byte-for-byte reproducible values.
        """
        X, y = _separable_fixture()
        out = str(tmp_path / "feats.npz")
        engine.build_feature_npz(X, out, y=y)
        archive = np.load(out)

        direct = engine._extract_fusion_features(X, fit_detectors=False)
        archive_feature_keys = {k for k in archive.files if k != "labels"}
        assert archive_feature_keys == set(direct.keys())
        for name, tensor in direct.items():
            assert archive[name].shape == tuple(tensor.shape)
            assert np.all(np.isfinite(archive[name]))

    def test_build_feature_npz_pseudo_labels(self, engine: Any, tmp_path: Any) -> None:
        """Without labels the builder emits detector-consensus pseudo-labels."""
        X, _ = _separable_fixture()
        out = str(tmp_path / "feats_unlabeled.npz")
        engine.build_feature_npz(X, out, y=None, contamination=0.1)
        archive = np.load(out)
        labels = archive["labels"]
        assert len(labels) == len(X)
        assert 0 < labels.sum() < len(X)
