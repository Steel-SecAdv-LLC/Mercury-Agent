"""
Tests for FocalLoss training + post-hoc temperature calibration (Issue #3).

Verifies the two-part contract:
  * temperature scaling is monotonic, so ROC-AUC/ranking is preserved exactly
    (calibrated scores rank identically to raw scores), and
  * calibration improves ECE (reliability) on data where focal-loss training
    flattens probabilities — checked on a real imbalanced dataset.

Both must hold: improving calibration must not cost discrimination.

Mercury Agent - Copyright (C) 2025 Steel Security Advisors LLC
Licensed under GNU GPL v3
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from omni_mercury_engine.core.calibration import compute_ece
from omni_mercury_engine.ml.mercury_ml import roc_auc_score


def _fixture(seed: int = 7, sep: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    normal = rng.normal(0.0, 1.0, (500, 12))
    anomaly = rng.normal(sep, 1.0, (60, 12))
    X = np.vstack([normal, anomaly]).astype(np.float32)
    y = np.concatenate([np.zeros(500), np.ones(60)]).astype(np.int64)
    order = rng.permutation(len(X))
    return X[order], y[order]


@pytest.fixture
def engine() -> Any:
    from omni_mercury_engine.engine import OmniMercuryEngine

    return OmniMercuryEngine(mode="fusion", device="cpu")


class TestCalibrationContract:
    def test_focal_training_reports_calibration_metrics(self, engine: Any) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        X, y = _fixture()
        metrics = engine.fit_fusion(X, y, epochs=30, early_stopping_patience=10)

        assert "temperature" in metrics
        assert "ece_before" in metrics and "ece_after" in metrics
        assert metrics["temperature"] > 0.0
        assert engine._fusion_calibrator is not None

    def test_temperature_scaling_preserves_auc(self, engine: Any) -> None:
        """Monotonic calibration must not change ranking/AUC at all."""
        torch.manual_seed(0)
        np.random.seed(0)
        X, y = _fixture()
        n = int(len(X) * 0.7)
        engine.fit_fusion(X[:n], y[:n], epochs=30, early_stopping_patience=10)

        calibrated = engine.score_fusion(X[n:])
        saved = engine._fusion_calibrator
        engine._fusion_calibrator = None
        raw = engine.score_fusion(X[n:])
        engine._fusion_calibrator = saved

        auc_raw = roc_auc_score(y[n:], raw)
        auc_cal = roc_auc_score(y[n:], calibrated)
        assert (
            abs(auc_raw - auc_cal) < 1e-9
        ), f"AUC changed under calibration: {auc_raw} vs {auc_cal}"

    def test_bce_fallback_still_trains(self, engine: Any) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        X, y = _fixture()
        metrics = engine.fit_fusion(X, y, epochs=10, use_focal_loss=False, calibrate=False)
        assert metrics["epochs_trained"] > 0
        assert engine._fusion_calibrator is None

    def test_calibrate_false_skips_calibrator(self, engine: Any) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        X, y = _fixture()
        engine.fit_fusion(X, y, epochs=10, calibrate=False)
        assert engine._fusion_calibrator is None


class TestCalibrationOnRealData:
    """The headline #3 claim: ECE improves with AUC held flat on the real
    imbalanced datasets that previously ranked well but had flattened
    probabilities (thyroid / WBC). Network-gated."""

    @pytest.mark.parametrize("dataset", ["thyroid", "WBC"])
    def test_ece_improves_auc_flat(self, dataset: str) -> None:
        from omni_mercury_engine.datasets.adbench import ADBenchLoader
        from omni_mercury_engine.datasets.base import DatasetConfig
        from omni_mercury_engine.engine import OmniMercuryEngine

        try:
            loader = ADBenchLoader(
                DatasetConfig(name="adbench", preprocessing={"dataset": dataset})
            )
            loader.download()
            data = loader.load()
        except Exception as exc:
            pytest.skip(f"ADBench {dataset} unavailable (network?): {exc}")

        X = np.asarray(data[0], dtype=np.float32)
        y = np.asarray(data[1]).astype(int).ravel()

        rng = np.random.RandomState(0)
        tr, te = [], []
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            rng.shuffle(idx)
            cut = max(1, int(len(idx) * 0.7))
            tr += idx[:cut].tolist()
            te += idx[cut:].tolist()

        torch.manual_seed(0)
        np.random.seed(0)
        engine = OmniMercuryEngine(mode="fusion", device="cpu")
        engine.fit_fusion(X[tr], y[tr], epochs=40, batch_size=64, early_stopping_patience=15)

        calibrated = engine.score_fusion(X[te])
        saved = engine._fusion_calibrator
        engine._fusion_calibrator = None
        raw = engine.score_fusion(X[te])
        engine._fusion_calibrator = saved

        auc_raw = roc_auc_score(y[te], raw)
        auc_cal = roc_auc_score(y[te], calibrated)
        ece_raw = compute_ece(y[te], raw)
        ece_cal = compute_ece(y[te], calibrated)

        # AUC held flat (monotonic), calibration strictly not worse.
        assert abs(auc_raw - auc_cal) < 1e-9
        assert ece_cal <= ece_raw + 1e-6, f"ECE worsened: {ece_raw:.4f} -> {ece_cal:.4f}"
