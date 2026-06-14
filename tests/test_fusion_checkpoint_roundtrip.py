# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fusion checkpoint round-trip fidelity (ROADMAP row 16).

Pins the save → load probability-equivalence contract: an engine that loads
a checkpoint must reproduce the saving engine's calibrated per-sample
probabilities (``max|Δ| < 1e-3``; measured ≈6e-8) and conformal prediction
sets — not merely its AUC. Before 2026-06-11 the checkpoint persisted only
the network and temperature, so a loaded engine auto-fit its base detectors
on the first inference batch (leaking it as the reference distribution) and
re-randomized the torch-module / population domain models, drifting
per-sample probabilities by up to ≈0.76. The serve path was additionally
nondeterministic call-to-call (two stub models emitted fresh RNG output per
call; two streaming buffers leaked across calls), so even one engine could
not reproduce itself.

Everything here runs on the deterministic offline synthetic corpus the
fusion regression gate uses — no network access.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omni_mercury_engine.engine import OmniMercuryEngine
from scripts.train_default_fusion import SEED, _stratified_split, build_dataset

EPOCHS = 4  # enough to move every fitted component off init; keeps the lane fast


@pytest.fixture(scope="module")
def trained_engine_and_data() -> dict[str, Any]:
    """One trained + conformal-calibrated engine over the offline gate corpus."""
    x, y = build_dataset(SEED)
    train_idx, rest_idx = _stratified_split(y, train_frac=0.6, seed=SEED)
    cal_local, test_local = _stratified_split(y[rest_idx], train_frac=0.5, seed=SEED)
    cal_idx, test_idx = rest_idx[cal_local], rest_idx[test_local]

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    engine = OmniMercuryEngine(mode="fusion", device="cpu")
    engine.fit_fusion(
        x[train_idx],
        y[train_idx],
        epochs=EPOCHS,
        batch_size=64,
        symbolic_weight=0.0,
    )
    engine.calibrate_fusion_conformal(x[cal_idx], y[cal_idx], coverage=0.9)
    return {"engine": engine, "x_test": x[test_idx][:160], "x": x, "y": y}


@pytest.fixture()
def loaded_engine(trained_engine_and_data: dict[str, Any], tmp_path: Path) -> OmniMercuryEngine:
    """A fresh engine, constructed under scrambled RNG, loading the checkpoint."""
    path = str(tmp_path / "roundtrip.pt")
    trained_engine_and_data["engine"].save_model(path)
    # Scrambled ambient RNG = a different process's reality: every
    # randomly-initialized component would differ unless persisted.
    torch.manual_seed(987_654)
    np.random.seed(424_242)
    engine = OmniMercuryEngine(mode="fusion", device="cpu")
    engine.load_model(path)
    return engine


class TestServeDeterminism:
    def test_repeated_scoring_is_identical(self, trained_engine_and_data: dict[str, Any]) -> None:
        """Same engine, same batch, twice — bit-identical probabilities.

        Pins the serve-path purity fixes: streaming buffers reset at the
        fusion feature boundary, and no model emits fresh RNG output per
        call (measured drift before the fix: ±0.08 per repeat).
        """
        engine = trained_engine_and_data["engine"]
        x_test = trained_engine_and_data["x_test"]
        first = np.asarray(engine.score_fusion(x_test))
        second = np.asarray(engine.score_fusion(x_test))
        np.testing.assert_array_equal(first, second)

    def test_feature_extraction_is_pure(self, trained_engine_and_data: dict[str, Any]) -> None:
        engine = trained_engine_and_data["engine"]
        x_test = trained_engine_and_data["x_test"]
        first = engine._extract_fusion_features(x_test, fit_detectors=False)
        second = engine._extract_fusion_features(x_test, fit_detectors=False)
        assert sorted(first) == sorted(second)
        for name in first:
            assert bool((first[name] == second[name]).all()), f"group {name!r} drifted on repeat"


class TestCheckpointRoundTrip:
    def test_probability_equivalence(
        self, trained_engine_and_data: dict[str, Any], loaded_engine: OmniMercuryEngine
    ) -> None:
        """The ROADMAP row 16 bar: save→load max |Δ probability| < 1e-3."""
        x_test = trained_engine_and_data["x_test"]
        probs_a = np.asarray(trained_engine_and_data["engine"].score_fusion(x_test))
        probs_b = np.asarray(loaded_engine.score_fusion(x_test))
        max_delta = float(np.max(np.abs(probs_a - probs_b)))
        assert max_delta < 1e-3, f"save->load probability drift {max_delta} >= 1e-3"

    def test_no_inference_auto_fit_after_load(
        self, trained_engine_and_data: dict[str, Any], loaded_engine: OmniMercuryEngine
    ) -> None:
        """Loaded detectors are fitted: the first-batch leak cannot happen."""
        x_test = trained_engine_and_data["x_test"]
        for name, detector in loaded_engine.detectors.items():
            assert detector.is_fitted(), f"detector {name!r} not restored as fitted"
        loaded_engine.score_fusion(x_test)
        assert not loaded_engine._inference_auto_fit_detectors

    def test_detector_state_round_trips_exactly(
        self, trained_engine_and_data: dict[str, Any], loaded_engine: OmniMercuryEngine
    ) -> None:
        engine = trained_engine_and_data["engine"]
        for name, detector in engine.detectors.items():
            exporter = getattr(detector, "get_fitted_state", None)
            if not callable(exporter) or exporter() is None:
                continue
            saved = exporter()
            loaded_exporter = getattr(loaded_engine.detectors[name], "get_fitted_state", None)
            assert callable(loaded_exporter), f"{name} lost its exporter after load"
            restored = loaded_exporter()
            assert restored is not None, f"{name} state missing after load"
            assert sorted(saved) == sorted(restored)
            for key, value in saved.items():
                other = restored[key]
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(value, other, err_msg=f"{name}.{key}")
                elif isinstance(value, dict) or value is None:
                    continue  # nested module states are covered by the probability bar
                else:
                    assert value == other, f"{name}.{key}: {value!r} != {other!r}"

    def test_conformal_calibrator_round_trips(
        self, trained_engine_and_data: dict[str, Any], loaded_engine: OmniMercuryEngine
    ) -> None:
        """Thresholds equal; prediction sets identical on a probability grid."""
        original = trained_engine_and_data["engine"]._fusion_conformal
        restored = loaded_engine._fusion_conformal
        assert original is not None and restored is not None
        assert restored._thresholds == original._thresholds
        assert restored.coverage == original.coverage
        grid = np.linspace(0.0, 1.0, 101)
        sets_a, sets_b = original.predict(grid), restored.predict(grid)
        np.testing.assert_array_equal(sets_a.contains_anomaly, sets_b.contains_anomaly)
        np.testing.assert_array_equal(sets_a.contains_normal, sets_b.contains_normal)

    def test_checkpoint_without_conformal_resets_it(
        self, trained_engine_and_data: dict[str, Any], tmp_path: Path
    ) -> None:
        """Loading a conformal-free checkpoint drops this engine's own fit.

        A conformal surface calibrated against a previous fusion stack
        misapplies to the loaded one — loaded state must equal saved state,
        including absent state.
        """
        full_path = str(tmp_path / "with_conformal.pt")
        trained_engine_and_data["engine"].save_model(full_path)
        stripped = torch.load(full_path, map_location="cpu", weights_only=True)
        stripped["conformal_state"] = None
        stripped_path = str(tmp_path / "no_conformal.pt")
        torch.save(stripped, stripped_path)

        loaded = OmniMercuryEngine(mode="fusion", device="cpu")
        loaded.load_model(full_path)
        assert loaded._fusion_conformal is not None
        loaded.load_model(stripped_path)
        assert loaded._fusion_conformal is None

    def test_checkpoint_without_temperature_resets_calibrator(
        self, trained_engine_and_data: dict[str, Any], tmp_path: Path
    ) -> None:
        """Loading a temperature-free checkpoint drops a stale calibrator."""
        full_path = str(tmp_path / "with_temperature.pt")
        trained_engine_and_data["engine"].save_model(full_path)
        stripped = torch.load(full_path, map_location="cpu", weights_only=True)
        stripped["temperature"] = None
        stripped_path = str(tmp_path / "no_temperature.pt")
        torch.save(stripped, stripped_path)

        loaded = OmniMercuryEngine(mode="fusion", device="cpu")
        loaded.load_model(full_path)
        assert loaded._fusion_calibrator is not None
        loaded.load_model(stripped_path)
        assert loaded._fusion_calibrator is None

    def test_legacy_checkpoint_without_new_keys_still_loads(
        self, trained_engine_and_data: dict[str, Any], tmp_path: Path
    ) -> None:
        """Backward compatibility: pre-row-16 checkpoints load (legacy behavior)."""
        path = str(tmp_path / "legacy.pt")
        engine = trained_engine_and_data["engine"]
        engine.save_model(path)
        checkpoint = torch.load(path, weights_only=True)
        for key in (
            "detector_fitted_state",
            "model_state_dicts",
            "model_fitted_state",
            "conformal_state",
        ):
            checkpoint.pop(key, None)
        torch.save(checkpoint, path)
        fresh = OmniMercuryEngine(mode="fusion", device="cpu")
        fresh.load_model(path)
        assert fresh._fusion_trained

    def test_load_clears_superseded_auto_fit_record(
        self, trained_engine_and_data: dict[str, Any], tmp_path: Path
    ) -> None:
        """Restoring detector state from a checkpoint supersedes a prior
        first-batch auto-fit, so the leak audit record is dropped; a legacy
        load that restores nothing keeps the (still accurate) record.
        """
        full_path = str(tmp_path / "full.pt")
        trained_engine_and_data["engine"].save_model(full_path)
        legacy_path = str(tmp_path / "legacy_audit.pt")
        checkpoint = torch.load(full_path, weights_only=True)
        for key in ("detector_fitted_state", "model_state_dicts", "model_fitted_state"):
            checkpoint.pop(key, None)
        torch.save(checkpoint, legacy_path)

        engine = OmniMercuryEngine(mode="fusion", device="cpu")
        engine.load_model(legacy_path)
        # The production serve path auto-fits unfitted detectors on the first
        # inference batch and records the leak for audit.
        engine.detect_with_fusion(trained_engine_and_data["x_test"])
        contaminated = set(engine._inference_auto_fit_detectors)
        assert contaminated

        engine.load_model(legacy_path)  # restores no detector state
        assert engine._inference_auto_fit_detectors == contaminated

        engine.load_model(full_path)  # checkpoint state replaces the leak
        assert not engine._inference_auto_fit_detectors


class TestQuarantinedAffectiveModel:
    def test_affective_is_deterministic_and_neutral(self) -> None:
        """The stub emits neutral constants, never fabricated noise."""
        from omni_mercury_engine.models.affective import AffectiveAnomalyModel

        model = AffectiveAnomalyModel()
        data = np.random.default_rng(0).normal(size=(12, 9))
        first = model.extract_features(data)
        np.testing.assert_array_equal(first, model.extract_features(data))
        np.testing.assert_array_equal(first, np.zeros((12, 64), dtype=np.float32))
        prediction = model.predict(data)
        np.testing.assert_array_equal(
            prediction["anomaly_scores"], np.full(12, 0.5, dtype=np.float32)
        )
