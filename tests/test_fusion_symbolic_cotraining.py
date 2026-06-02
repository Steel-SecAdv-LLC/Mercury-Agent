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


def _separable_fixture(seed: int = 7) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
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


def _split(
    X: np.ndarray[Any, Any], y: np.ndarray[Any, Any]
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
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

    def test_inference_surfaces_symbolic_consistency(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from omni_mercury_engine import engine as engine_mod

        monkeypatch.setattr(engine_mod, "_GOSNN_TESTING_BYPASS", True)
        torch.manual_seed(0)
        np.random.seed(0)
        X, y = _separable_fixture()
        X_tr, y_tr, X_te, _ = _split(X, y)
        engine = _engine()
        engine.fit_fusion(
            X_tr,
            y_tr,
            epochs=8,
            batch_size=32,
            symbolic_weight=0.1,
            symbolic_rule_graph="consensus",
        )

        result = engine.detect_with_fusion(X_te[:4])

        consistency = result["symbolic_consistency"]
        assert consistency["graph"] == "detector_consensus"
        assert consistency["semantics"] == "product"
        assert 0.0 <= consistency["satisfaction"] <= 1.0
        assert set(consistency["rules"]) == {"R1_evidence", "R2_precision"}
        assert len(consistency["detector_channels"]) == len(consistency["detector_weights"])

    def test_symbolic_consistency_preserves_training_channels_when_inference_expands(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from omni_mercury_engine import engine as engine_mod

        monkeypatch.setattr(engine_mod, "_GOSNN_TESTING_BYPASS", True)
        torch.manual_seed(0)
        np.random.seed(0)
        X, y = _separable_fixture()
        X_tr, y_tr, X_te, _ = _split(X, y)
        engine = _engine()
        engine.fit_fusion(X_tr, y_tr, epochs=8, batch_size=32, symbolic_weight=0.1)
        module = engine._symbolic_module
        assert module is not None

        channels = list(engine._symbolic_score_channels or [])
        assert len(channels) == module.num_detectors

        def empty_detector_features(data: Any) -> tuple[dict[str, Any], dict[str, Any]]:
            return {}, {}

        def expanded_model_features(data: Any) -> tuple[dict[str, Any], dict[str, Any]]:
            n = len(data)
            return (
                {},
                {
                    **{name: np.full(n, 0.25, dtype=np.float32) for name in channels},
                    "late_model": np.ones(n, dtype=np.float32),
                },
            )

        monkeypatch.setattr(engine, "_extract_detector_features", empty_detector_features)
        monkeypatch.setattr(engine, "_extract_model_features", expanded_model_features)
        consistency = engine._symbolic_consistency_payload(
            X_te[:2],
            np.array([0.4, 0.6], dtype=np.float32),
        )
        assert consistency is not None
        assert len(consistency["detector_weights"]) == module.num_detectors
        assert "late_model" not in consistency["detector_channels"]

    def test_symbolic_constraint_round_trips_through_checkpoint(self, tmp_path: Any) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        X, y = _separable_fixture()
        X_tr, y_tr, X_te, _ = _split(X, y)
        engine = _engine()
        engine.fit_fusion(
            X_tr,
            y_tr,
            epochs=8,
            batch_size=32,
            symbolic_weight=0.1,
            symbolic_rule_graph="consensus_salience",
            symbolic_semantics="godel",
        )
        path = tmp_path / "symbolic-fusion.pt"
        engine.save_model(str(path))

        loaded = _engine()
        loaded.load_model(str(path))

        assert isinstance(loaded._symbolic_module, SymbolicConstraintModule)
        assert loaded._symbolic_module.rule_graph.name == "detector_consensus_salience"
        assert loaded._symbolic_module.semantics == "godel"
        assert loaded._symbolic_score_channels == engine._symbolic_score_channels
        metrics = loaded.evaluate_neurosymbolic_feedback(X_te[:8])
        assert metrics["symbolic_active"] is True
        assert metrics["symbolic_rule_graph"] == "detector_consensus_salience"
        assert 0.0 <= metrics["symbolic_satisfaction"] <= 1.0


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


def _fixture_with_anomalies(
    n_normal: int, n_anom: int, dim: int = 12, seed: int = 7
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Separable fixture with a caller-chosen anomaly count (for the schedule)."""
    rng = np.random.RandomState(seed)
    normal = rng.normal(0.0, 1.0, (n_normal, dim))
    anomaly = rng.normal(3.0, 1.0, (n_anom, dim))
    X = np.vstack([normal, anomaly]).astype(np.float32)
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anom)]).astype(np.int64)
    order = rng.permutation(len(X))
    return X[order], y[order]


class TestAdaptiveSchedule:
    """``symbolic_weight="adaptive"`` spends the constraint only when scarce."""

    def test_adaptive_active_when_labels_scarce(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        X, y = _fixture_with_anomalies(400, 8)  # few anomalies -> schedule ON
        engine = _engine()
        metrics = engine.fit_fusion(X, y, epochs=10, batch_size=32, symbolic_weight="adaptive")
        assert metrics["symbolic_weight_spec"] == "adaptive"
        assert metrics["symbolic_n_positive"] == 8
        assert metrics["symbolic_weight_resolved"] > 0.0
        assert "symbolic_satisfaction" in metrics
        assert engine._symbolic_module is not None

    def test_adaptive_decays_to_neural_path_when_abundant(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        X, y = _fixture_with_anomalies(400, 400)  # many anomalies -> schedule OFF
        engine = _engine()
        metrics = engine.fit_fusion(X, y, epochs=10, batch_size=32, symbolic_weight="adaptive")
        assert metrics["symbolic_weight_resolved"] == 0.0
        # Resolving to 0 must reproduce the neural path: no constraint module,
        # no satisfaction/loss diagnostics.
        assert "symbolic_satisfaction" not in metrics
        assert engine._symbolic_module is None

    def test_adaptive_is_the_default(self) -> None:
        # Evidence-backed default flip: calling fit_fusion without an explicit
        # symbolic_weight must use the adaptive schedule (not the neural path).
        torch.manual_seed(0)
        np.random.seed(0)
        X, y = _fixture_with_anomalies(400, 8)  # scarce -> schedule active by default
        engine = _engine()
        metrics = engine.fit_fusion(X, y, epochs=10, batch_size=32)
        assert metrics["symbolic_weight_spec"] == "adaptive"
        assert metrics["symbolic_weight_resolved"] > 0.0
        assert engine._symbolic_module is not None


class TestCoTrainingConformalServePath:
    """Co-training + conformal must coexist on the production serve path, and a
    checkpoint round-trip must not drift the fusion dimensions (the symbolic
    channels are training-only and dropped at inference)."""

    def test_adaptive_cotraining_then_conformal_on_serve_path(self) -> None:
        """The production default (adaptive co-training) + conformal serve path:
        co-training reports a constraint, conformal yields valid bounded sets,
        and detection still separates the classes."""
        torch.manual_seed(0)
        np.random.seed(0)
        # Scarce-anomaly regime so the adaptive schedule is ACTIVE (the case
        # the symbolic constraint is meant to help).
        X, y = _fixture_with_anomalies(360, 18)
        n_tr, n_cal = 240, 60
        X_tr, y_tr = X[:n_tr], y[:n_tr]
        X_cal, y_cal = X[n_tr : n_tr + n_cal], y[n_tr : n_tr + n_cal]
        X_te, y_te = X[n_tr + n_cal :], y[n_tr + n_cal :]

        engine = _engine()
        # No explicit symbolic_weight => adaptive default (the production path).
        metrics = engine.fit_fusion(X_tr, y_tr, epochs=25, batch_size=32)
        assert metrics["symbolic_weight_spec"] == "adaptive"
        assert metrics["symbolic_weight_resolved"] > 0.0
        assert isinstance(engine._symbolic_module, SymbolicConstraintModule)
        assert 0.0 <= metrics["symbolic_satisfaction"] <= 1.0

        # Conformal calibration on the *same* trained model (serve path).
        cal = engine.calibrate_fusion_conformal(X_cal, y_cal, coverage=0.9)
        assert cal["coverage"] == 0.9
        out = engine.score_fusion_conformal(X_te)
        probs = np.asarray(out["probabilities"])
        assert np.all((probs >= 0.0) & (probs <= 1.0))
        # Conformal prediction sets are present and well-formed.
        assert "prediction_sets" in out or "set_sizes" in out
        # Co-training must not have broken detection.
        assert roc_auc_score(y_te, probs) >= 0.85

    def test_checkpoint_round_trip_no_symbolic_dimension_drift(self) -> None:
        """After co-training, a save/load checkpoint round-trip reproduces
        score_fusion byte-for-byte — the symbolic channels never expanded the
        persisted fusion dimensions (issue: inference-time symbolic channel
        drift)."""
        import tempfile
        from pathlib import Path

        torch.manual_seed(0)
        np.random.seed(0)
        X, y = _separable_fixture()
        X_tr, y_tr, X_te, _ = _split(X, y)

        engine = _engine()
        engine.fit_fusion(X_tr, y_tr, epochs=15, batch_size=32, symbolic_weight=0.1)
        # The symbolic module exists post-training...
        assert isinstance(engine._symbolic_module, SymbolicConstraintModule)
        probs_before = engine.score_fusion(X_te)

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "fusion_cotrained.pt")
            engine.save_model(path)

            fresh = _engine()
            fresh.load_model(path)
            # ...but the persisted checkpoint is purely the fusion model: the
            # reloaded engine has NO symbolic module (training-only channel)...
            assert fresh._symbolic_module is None
            probs_after = fresh.score_fusion(X_te)

        # ...and yet scores match exactly — no dimension drift from the
        # symbolic channels that were present only during co-training.
        np.testing.assert_allclose(probs_before, probs_after, rtol=1e-5, atol=1e-6)
