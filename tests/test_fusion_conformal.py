"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Tests for conformal uncertainty in the fusion serving path
(``engine.calibrate_fusion_conformal`` / ``engine.score_fusion_conformal``).

* Network-free: wiring + contract on a deterministic separable fixture.
* Network-gated: the *true* coverage guarantee on real ADBench labels --
  the empirical fraction of prediction sets containing the true label meets
  the target on a held-out test split.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

pytestmark = pytest.mark.xdist_group("fusion_conformal")


def _separable_fixture(seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    n_normal, n_anom, dim = 600, 90, 12
    normal = rng.normal(0.0, 1.0, (n_normal, dim))
    anomaly = rng.normal(2.6, 1.0, (n_anom, dim))
    X = np.vstack([normal, anomaly]).astype(np.float32)
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anom)]).astype(np.int64)
    order = rng.permutation(len(X))
    return X[order], y[order]


def _engine() -> Any:
    from omni_mercury_engine.engine import OmniMercuryEngine

    return OmniMercuryEngine(mode="fusion", device="cpu")


def _three_way(X: np.ndarray, y: np.ndarray) -> tuple[Any, ...]:
    n = len(X)
    a, b = int(n * 0.5), int(n * 0.75)
    return X[:a], y[:a], X[a:b], y[a:b], X[b:], y[b:]


class TestConformalContract:
    """Wiring and guard rails, network-free."""

    def test_score_before_calibrate_raises(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        X, y = _separable_fixture()
        X_tr, y_tr, _, _, X_te, _ = _three_way(X, y)
        engine = _engine()
        engine.fit_fusion(X_tr, y_tr, epochs=8, batch_size=32)
        with pytest.raises(RuntimeError, match="Conformal calibrator not fitted"):
            engine.score_fusion_conformal(X_te)

    def test_calibrate_requires_trained_model(self) -> None:
        engine = _engine()
        X, y = _separable_fixture()
        with pytest.raises(RuntimeError, match="untrained"):
            engine.calibrate_fusion_conformal(X[:50], y[:50])

    def test_conformal_output_shapes_and_sets(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        X, y = _separable_fixture()
        X_tr, y_tr, X_cal, y_cal, X_te, _ = _three_way(X, y)
        engine = _engine()
        engine.fit_fusion(X_tr, y_tr, epochs=15, batch_size=32, early_stopping_patience=10)
        diag = engine.calibrate_fusion_conformal(X_cal, y_cal, coverage=0.9)
        assert diag["coverage"] == pytest.approx(0.9)
        assert set(diag["thresholds"]) == {0, 1}

        out = engine.score_fusion_conformal(X_te)
        n = len(X_te)
        assert out["probabilities"].shape == (n,)
        assert out["set_sizes"].shape == (n,)
        assert len(out["prediction_sets"]) == n
        assert set(np.unique(out["set_sizes"])).issubset({0, 1, 2})
        for labels in out["prediction_sets"]:
            assert set(labels).issubset({0, 1})


class TestConformalCoverageRealData:
    """The coverage guarantee on real ADBench labels (network-gated)."""

    @pytest.mark.parametrize("dataset", ["cardio", "breastw"])
    def test_empirical_coverage_meets_target(self, dataset: str) -> None:
        from omni_mercury_engine.datasets.adbench import ADBenchLoader
        from omni_mercury_engine.datasets.base import DatasetConfig

        try:
            loader = ADBenchLoader(
                DatasetConfig(name="adbench", preprocessing={"dataset": dataset})
            )
            loader.download()
            data = loader.load()
        except Exception as exc:  # noqa: BLE001 - network/data unavailable
            pytest.skip(f"ADBench {dataset} unavailable: {exc}")

        X = np.asarray(data[0], dtype=np.float32)
        y = np.asarray(data[1]).astype(int).ravel()

        rng = np.random.RandomState(0)
        idx = rng.permutation(len(X))
        a, b = int(len(X) * 0.5), int(len(X) * 0.75)
        tr, cal, te = idx[:a], idx[a:b], idx[b:]
        if len(np.unique(y[cal])) < 2 or len(np.unique(y[te])) < 2:
            pytest.skip("split is single-class; reseed/raise sample count")

        torch.manual_seed(0)
        engine = _engine()
        engine.fit_fusion(X[tr], y[tr], epochs=20, batch_size=64, early_stopping_patience=15)
        coverage = 0.9
        engine.calibrate_fusion_conformal(X[cal], y[cal], coverage=coverage)

        assert engine._fusion_conformal is not None
        probs = engine.score_fusion(X[te])
        report = engine._fusion_conformal.coverage_report(probs, y[te])
        # Distribution-free marginal coverage with finite-sample tolerance.
        assert report["empirical_coverage"] >= coverage - 0.05
        # Sets are informative (not trivially always both labels).
        assert report["average_set_size"] < 2.0
