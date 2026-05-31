"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Integration tests for neuro-symbolic co-training in ``fit_fusion``.

These verify the *contract* of the symbolic-constraint path, deterministically
and without network access:

* ``symbolic_weight == 0`` is byte-for-byte the purely-neural path (no symbolic
  state, identical metrics keys).
* ``symbolic_weight > 0`` co-trains a ``SymbolicConstraintModule`` (retained on
  the engine), reports satisfaction/loss, and does not break detection on a
  clearly-separable fixture.

Whether the constraint *improves* held-out detection is a separate, empirical
question settled by ``benchmarks/neurosymbolic_ablation.py`` on real labels --
deliberately NOT asserted here, where a synthetic pass would be meaningless.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from omni_mercury_engine.ml.mercury_ml import roc_auc_score
from omni_mercury_engine.ml.symbolic_constraint import SymbolicConstraintModule

pytestmark = pytest.mark.xdist_group("fusion_symbolic_cotraining")


def _separable_fixture(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic, clearly-separable anomaly fixture (no network)."""
    rng = np.random.RandomState(seed)
    n_normal, n_anom, dim = 400, 50, 12
    normal = rng.normal(0.0, 1.0, (n_normal, dim))
    anomaly = rng.normal(3.0, 1.0, (n_anom, dim))
    X = np.vstack([normal, anomaly]).astype(np.float32)
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anom)]).astype(np.int64)
    order = rng.permutation(len(X))
    return X[order], y[order]


def _engine() -> Any:
    from omni_mercury_engine.engine import OmniMercuryEngine

    return OmniMercuryEngine(mode="fusion", device="cpu")


def _split(X: np.ndarray, y: np.ndarray) -> tuple[Any, Any, Any, Any]:
    n_train = int(len(X) * 0.7)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


class TestNeuralPathUnchanged:
    """symbolic_weight == 0 must not alter the neural training contract."""

    def test_zero_weight_has_no_symbolic_state(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        X, y = _separable_fixture()
        X_tr, y_tr, _, _ = _split(X, y)
        engine = _engine()
        metrics = engine.fit_fusion(X_tr, y_tr, epochs=10, batch_size=32, symbolic_weight=0.0)
        assert "symbolic_satisfaction" not in metrics
        assert "symbolic_loss" not in metrics
        assert engine._symbolic_module is None


class TestCoTrainingContract:
    """symbolic_weight > 0 co-trains and reports a valid constraint."""

    def test_cotraining_reports_satisfaction_and_keeps_detecting(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        X, y = _separable_fixture()
        X_tr, y_tr, X_te, y_te = _split(X, y)
        engine = _engine()
        metrics = engine.fit_fusion(
            X_tr, y_tr, epochs=30, batch_size=32, early_stopping_patience=12, symbolic_weight=0.1
        )

        # Constraint diagnostics are present and well-formed.
        assert metrics["symbolic_weight"] == pytest.approx(0.1)
        assert 0.0 <= metrics["symbolic_satisfaction"] <= 1.0
        assert metrics["symbolic_loss"] == pytest.approx(
            1.0 - metrics["symbolic_satisfaction"], abs=1e-5
        )

        # The constraint module is retained and learned over real channels.
        module = engine._symbolic_module
        assert isinstance(module, SymbolicConstraintModule)
        assert module.num_detectors > 0

        # Co-training must not break detection on a clearly-separable fixture.
        probs = engine.score_fusion(X_te)
        assert np.all((probs >= 0.0) & (probs <= 1.0))
        assert roc_auc_score(y_te, probs) >= 0.9

    def test_explainability_round_trips_after_training(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        X, y = _separable_fixture()
        X_tr, y_tr, _, _ = _split(X, y)
        engine = _engine()
        engine.fit_fusion(X_tr, y_tr, epochs=8, batch_size=32, symbolic_weight=0.1)
        module = engine._symbolic_module
        assert module is not None
        scores = torch.rand(16, module.num_detectors)
        explanation = module.explain(torch.rand(16, 1), scores)
        assert explanation["graph"] == "detector_consensus"
        assert set(explanation["rules"]) == {"R1_evidence", "R2_precision"}
        assert len(explanation["detector_weights"]) == module.num_detectors


class TestStability:
    """Co-trained satisfaction is stable under a fixed seed.

    The full ``fit_fusion`` stack (base-detector fitting + feature extraction)
    is not bit-exact reproducible -- the existing raw-path test asserts only
    ``AUC >= 0.90`` for the same reason -- so this checks *stability* rather
    than bit-identity. The constraint module itself is deterministic on fixed
    inputs (see ``tests/ml/test_symbolic_constraint.py``).
    """

    def test_satisfaction_stable_under_fixed_seed(self) -> None:
        def _run() -> float:
            torch.manual_seed(123)
            np.random.seed(123)
            X, y = _separable_fixture()
            X_tr, y_tr, _, _ = _split(X, y)
            engine = _engine()
            metrics = engine.fit_fusion(X_tr, y_tr, epochs=8, batch_size=32, symbolic_weight=0.1)
            return float(metrics["symbolic_satisfaction"])

        assert _run() == pytest.approx(_run(), abs=0.05)
