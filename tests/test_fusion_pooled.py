# Copyright (C) 2025 Steel Security Advisors LLC
"""Offline tests for pooled fusion training and the shared training tail.

``fit_fusion`` and the new ``fit_fusion_pooled`` both funnel through
``_fit_fusion_on_features``; these tests pin the contract that makes the shipped
checkpoint trustworthy regardless of how it was trained:

* the network reports trained,
* a temperature calibrator is fitted (not stored-but-ignored),
* the trained feature groups are recorded, and
* calibration + groups + provenance round-trip through ``save_model`` /
  ``load_model`` — exactly what ``detect_with_fusion`` relies on.

Everything here is synthetic and offline (no network); ``torch`` is required.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.xdist_group("fusion_pooled")

pytest.importorskip("torch")

from omni_mercury_engine.engine import OmniMercuryEngine


def _blob(
    n_normal: int, n_anom: int, n_features: int, seed: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """A small labelled Gaussian-cluster + outlier set of ``n_features`` dims."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 4.0, size=(3, n_features))
    assign = rng.integers(0, 3, size=n_normal)
    normal = centers[assign] + rng.normal(0.0, 1.0, size=(n_normal, n_features))
    anom = rng.normal(0.0, 1.0, size=(n_anom, n_features)) + rng.choice(
        [-9.0, 9.0], size=(n_anom, n_features)
    )
    x = np.vstack([normal, anom]).astype(np.float32)
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anom)]).astype(np.int64)
    perm = rng.permutation(len(x))
    return x[perm], y[perm]


def test_fit_fusion_smoke_after_refactor():
    """fit_fusion still trains + calibrates after the tail was extracted."""
    engine = OmniMercuryEngine(mode="fusion", device="cpu")
    x, y = _blob(360, 40, 16, seed=1)
    metrics = engine.fit_fusion(x, y, epochs=20, batch_size=64, early_stopping_patience=8)

    assert engine._fusion_trained is True
    assert engine._fusion_feature_groups, "trained feature groups must be recorded"
    assert "best_loss" in metrics
    # Calibration is wired through the shared tail, not stored-and-ignored.
    assert engine._fusion_calibrator is not None
    assert metrics.get("temperature") is not None


def test_fit_fusion_pooled_wires_calibration_groups_and_provenance(tmp_path):
    """Pooled training wires the same contract and round-trips through a checkpoint."""
    engine = OmniMercuryEngine(mode="fusion", device="cpu")
    datasets = [
        _blob(300, 40, 16, seed=11),
        _blob(260, 30, 16, seed=12),
        _blob(280, 35, 16, seed=13),
    ]
    metrics = engine.fit_fusion_pooled(
        datasets, epochs=20, batch_size=64, early_stopping_patience=8
    )

    assert engine._fusion_trained is True
    assert engine._fusion_calibrator is not None, "pooled training must fit the calibrator"
    assert engine._fusion_feature_groups, "pooled training must record trained groups"
    assert metrics["pooled_datasets"] == 3
    assert metrics["pooled_samples"] == sum(len(y) for _, y in datasets)
    assert metrics["pooled_groups"] == engine._fusion_feature_groups

    # Provenance is recorded by the caller and must survive save -> load.
    engine._fusion_provenance = {"source": "adbench", "datasets": ["a", "b"], "seed": 7}
    ckpt = tmp_path / "pooled.pt"
    engine.save_model(str(ckpt))

    fresh = OmniMercuryEngine(mode="fusion", device="cpu")
    fresh.load_model(str(ckpt))
    assert fresh._fusion_trained is True
    assert fresh._fusion_calibrator is not None
    assert fresh._fusion_calibrator.temperature == pytest.approx(
        engine._fusion_calibrator.temperature, abs=1e-6
    )
    assert fresh._fusion_feature_groups == engine._fusion_feature_groups
    assert fresh._fusion_provenance == {"source": "adbench", "datasets": ["a", "b"], "seed": 7}


def test_fit_fusion_pooled_handles_heterogeneous_dimensionality():
    """Datasets of differing input width pool on their common, consistent groups."""
    engine = OmniMercuryEngine(mode="fusion", device="cpu")
    datasets = [
        _blob(280, 35, 14, seed=21),
        _blob(300, 40, 20, seed=22),
    ]
    metrics = engine.fit_fusion_pooled(
        datasets, epochs=15, batch_size=64, early_stopping_patience=6
    )
    # Pooling must still yield a usable, group-consistent prior (or fail loudly,
    # never silently train on misaligned features).
    assert engine._fusion_trained is True
    assert metrics["pooled_groups"], "at least one consistent feature group must survive pooling"
    assert engine._fusion_feature_groups == metrics["pooled_groups"]


def test_fit_fusion_pooled_rejects_empty_datasets():
    engine = OmniMercuryEngine(mode="fusion", device="cpu")
    with pytest.raises(ValueError, match="at least one"):
        engine.fit_fusion_pooled([])
