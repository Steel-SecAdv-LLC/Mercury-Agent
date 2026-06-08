# Copyright (C) 2025 Steel Security Advisors LLC
"""Tests for the shipped default fusion checkpoint (Issue #2)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

import omni_mercury_engine.engine as engine_mod
from omni_mercury_engine.engine import OmniMercuryEngine, default_fusion_checkpoint_path


def _engine() -> Any:
    return OmniMercuryEngine(mode="fusion", device="cpu")


class TestDefaultCheckpoint:
    def test_checkpoint_is_packaged(self) -> None:
        path = default_fusion_checkpoint_path()
        assert path.exists(), (
            f"default fusion checkpoint missing at {path}; "
            "regenerate with scripts/train_default_fusion.py"
        )

    def test_fresh_engine_starts_untrained(self) -> None:
        # The default-checkpoint feature must NOT change the construction
        # contract: a plain engine is untrained until asked.
        assert not _engine()._fusion_trained

    def test_load_default_checkpoint_marks_trained(self) -> None:
        eng = _engine()
        assert eng.load_default_fusion_checkpoint() is True
        assert eng._fusion_trained is True

    def test_auto_load_constructor_flag(self) -> None:
        eng = OmniMercuryEngine(mode="fusion", device="cpu", auto_load_checkpoint=True)
        assert eng._fusion_trained is True

    def test_loaded_checkpoint_has_calibrator(self) -> None:
        eng = _engine()
        eng.load_default_fusion_checkpoint()
        # The shipped checkpoint was trained with temperature calibration.
        assert eng._fusion_calibrator is not None
        assert eng._fusion_calibrator.temperature > 0.0

    def test_detect_end_to_end_no_training(self, monkeypatch: Any) -> None:
        # σ_Immutable gate is exercised separately; bypass it here so the test
        # targets the fusion-checkpoint path deterministically.
        monkeypatch.setattr(engine_mod, "_GOSNN_TESTING_BYPASS", True)
        eng = OmniMercuryEngine(mode="fusion", device="cpu", auto_load_checkpoint=True)

        rng = np.random.RandomState(3)
        X = np.vstack([rng.normal(0, 1, (60, 12)), rng.normal(3, 1, (8, 12))]).astype(np.float32)
        result = eng.detect_with_fusion(X)
        assert "anomaly_prob" in result
        assert 0.0 <= result["anomaly_prob"] <= 1.0
        assert isinstance(result["is_anomaly"], bool)


class TestCheckpointRoundTrip:
    def test_rich_checkpoint_roundtrip(self, tmp_path: Any) -> None:
        import torch

        from omni_mercury_engine.ml.mercury_ml import make_classification

        X, y = make_classification(
            n_samples=150, n_features=12, weights=[0.85, 0.15], random_state=0
        )
        eng = _engine()
        eng.fit_fusion(X.astype(np.float32), y, epochs=8, early_stopping_patience=5)
        path = str(tmp_path / "ckpt.pt")
        eng.save_model(path)

        blob = torch.load(path, map_location="cpu", weights_only=True)
        assert blob["format_version"] == engine_mod.FUSION_CHECKPOINT_FORMAT_VERSION
        assert "model_state_dict" in blob and "mercury_version" in blob
        assert "feature_dims" in blob and "projection_registry" in blob

        fresh = _engine()
        fresh.load_model(path)
        assert fresh._fusion_trained is True
        if eng._fusion_calibrator is not None:
            assert fresh._fusion_calibrator is not None
            assert (
                abs(fresh._fusion_calibrator.temperature - eng._fusion_calibrator.temperature)
                < 1e-6
            )

    def test_bare_state_dict_backward_compat(self, tmp_path: Any) -> None:
        import torch

        eng = _engine()
        path = str(tmp_path / "bare.pt")
        torch.save(eng.fusion_model.state_dict(), path)  # legacy format

        fresh = _engine()
        fresh.load_model(path)  # must not raise
        assert fresh._fusion_trained is True
