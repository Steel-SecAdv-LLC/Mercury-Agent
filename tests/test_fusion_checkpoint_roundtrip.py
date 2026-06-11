# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fusion checkpoint round-trip fidelity locks (ROADMAP v1.7.x item #16).

``save_model`` -> ``load_model`` must reproduce the saving engine's serve
path exactly: per-sample calibrated probabilities (the historical defect was
drift up to ~0.76), the conformal label sets, the temperature calibrator,
and the fitted base-detector state (the historical defect was an auto-fit on
the first inference batch, which both leaked the batch and shifted every
feature the fusion network consumes).

The contract bound asserted here is ``max|delta_prob| < 1e-3`` (the ROADMAP
remediation bar); the measured value on the v2 format is ~1e-13. Two root
causes are locked alongside the end-to-end bound: every feature extractor
must be instance-independent (fixed-seed init for the untrained LSTM /
autoencoder projectors, content-seeded draws for the placeholder models) and
call-stateless on the fusion path (the neural model's memory accrual is
opt-in via ``predict``, not a side effect of ``extract_features``).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

import omni_mercury_engine.engine as engine_mod
from omni_mercury_engine.engine import OmniMercuryEngine

# Small but non-trivial corpus: multi-modal normals plus displaced anomalies,
# split train / conformal-calibration / test. Sized so the whole module
# trains once and stays CI-cheap.
_SEED = 20260611
_N_NORMAL, _N_ANOM, _N_FEATURES = 300, 60, 8
_TRAIN_END, _CAL_END = 220, 300
_EPOCHS = 4


def _build_corpus() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(_SEED)
    normal = rng.normal(0.0, 1.0, size=(_N_NORMAL, _N_FEATURES))
    anomalies = rng.normal(0.0, 1.0, size=(_N_ANOM, _N_FEATURES)) + rng.choice(
        [-6.0, 6.0], size=(_N_ANOM, _N_FEATURES)
    )
    x = np.vstack([normal, anomalies]).astype(np.float32)
    y = np.concatenate([np.zeros(_N_NORMAL), np.ones(_N_ANOM)]).astype(np.int64)
    perm = rng.permutation(len(x))
    return x[perm], y[perm]


@pytest.fixture(scope="module")
def trained() -> dict[str, object]:
    """One trained + conformal-calibrated engine and its saved checkpoints."""
    x, y = _build_corpus()
    x_train, y_train = x[:_TRAIN_END], y[:_TRAIN_END]
    x_cal, y_cal = x[_TRAIN_END:_CAL_END], y[_TRAIN_END:_CAL_END]
    x_test = x[_CAL_END:]

    torch.manual_seed(3)
    np.random.seed(3)
    engine = OmniMercuryEngine(mode="fusion", device="cpu")
    engine.fit_fusion(x_train, y_train, epochs=_EPOCHS, symbolic_weight=0.0)
    engine.calibrate_fusion_conformal(x_cal, y_cal)
    return {"engine": engine, "x_test": x_test}


@pytest.fixture(scope="module")
def checkpoint_path(trained: dict[str, object], tmp_path_factory: pytest.TempPathFactory) -> str:
    path = str(tmp_path_factory.mktemp("ckpt") / "fusion_roundtrip.pt")
    engine = trained["engine"]
    assert isinstance(engine, OmniMercuryEngine)
    engine.save_model(path)
    return path


def _fresh_engine() -> OmniMercuryEngine:
    """An engine constructed under a deliberately different global RNG state.

    The round-trip contract must not depend on the loader process sharing the
    trainer process's RNG stream.
    """
    torch.manual_seed(987_654)
    np.random.seed(987_654)
    return OmniMercuryEngine(mode="fusion", device="cpu")


class TestProbabilityEquivalence:
    """The headline lock: load_model reproduces the saved serve path."""

    def test_save_load_probability_equivalence(
        self, trained: dict[str, object], checkpoint_path: str
    ) -> None:
        engine = trained["engine"]
        x_test = trained["x_test"]
        assert isinstance(engine, OmniMercuryEngine)
        assert isinstance(x_test, np.ndarray)
        probs_a = engine.score_fusion(x_test)

        loaded = _fresh_engine()
        loaded.load_model(checkpoint_path)

        # The detectors must be fitted by the load itself -- before any
        # inference -- so the first serve batch is never leaked into a fit.
        assert all(det.is_fitted() for det in loaded.detectors.values())

        probs_b = loaded.score_fusion(x_test)
        assert float(np.max(np.abs(probs_a - probs_b))) < 1e-3
        assert loaded._inference_auto_fit_detectors == set()

    def test_same_engine_rescore_is_bitstable(self, trained: dict[str, object]) -> None:
        """Scoring the same batch twice is identical (no call-stateful features)."""
        engine = trained["engine"]
        x_test = trained["x_test"]
        assert isinstance(engine, OmniMercuryEngine)
        assert isinstance(x_test, np.ndarray)
        first = engine.score_fusion(x_test)
        second = engine.score_fusion(x_test)
        np.testing.assert_array_equal(first, second)


class TestConformalRoundTrip:
    """The fitted conformal serving surface persists -- and stale state dies."""

    def test_conformal_state_and_sets_round_trip(
        self, trained: dict[str, object], checkpoint_path: str
    ) -> None:
        engine = trained["engine"]
        x_test = trained["x_test"]
        assert isinstance(engine, OmniMercuryEngine)
        assert isinstance(x_test, np.ndarray)
        assert engine._fusion_conformal is not None

        loaded = _fresh_engine()
        loaded.load_model(checkpoint_path)
        assert loaded._fusion_conformal is not None
        assert loaded._fusion_conformal._thresholds == engine._fusion_conformal._thresholds
        assert loaded._fusion_conformal.coverage == engine._fusion_conformal.coverage

        # Restore lock: on identical probabilities the restored classifier's
        # sets are exactly the original's (pure threshold comparison).
        probs = engine.score_fusion(x_test)
        sets_a = engine._fusion_conformal.predict(probs)
        sets_restored = loaded._fusion_conformal.predict(probs)
        np.testing.assert_array_equal(sets_a.contains_anomaly, sets_restored.contains_anomaly)
        np.testing.assert_array_equal(sets_a.contains_normal, sets_restored.contains_normal)

        # End-to-end lock: with the loaded engine's own (~1e-13-equal)
        # probabilities, set membership may flip only for samples sitting
        # within epsilon of the conformal threshold itself -- the irreducible
        # knife-edge of any hard threshold. A real regression moves
        # probabilities by far more than the 1e-9 boundary band.
        probs_loaded = loaded.score_fusion(x_test)
        sets_b = loaded._fusion_conformal.predict(probs_loaded)
        thresholds = engine._fusion_conformal._thresholds
        for label, a_contains, b_contains in (
            (1, sets_a.contains_anomaly, sets_b.contains_anomaly),
            (0, sets_a.contains_normal, sets_b.contains_normal),
        ):
            disagree = np.nonzero(a_contains != b_contains)[0]
            p_label = probs if label == 1 else 1.0 - probs
            boundary_distance = np.abs((1.0 - p_label[disagree]) - thresholds[label])
            assert np.all(boundary_distance < 1e-9), (
                f"label {label}: set membership flipped away from the threshold "
                f"boundary (distances {boundary_distance})"
            )

    def test_checkpoint_without_conformal_resets_it(
        self, trained: dict[str, object], checkpoint_path: str, tmp_path: object
    ) -> None:
        """Loading a conformal-free checkpoint drops this engine's own fit."""
        stripped = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        stripped["conformal_state"] = None
        stripped_path = str(tmp_path / "no_conformal.pt")  # type: ignore[operator]
        torch.save(stripped, stripped_path)

        loaded = _fresh_engine()
        loaded.load_model(checkpoint_path)
        assert loaded._fusion_conformal is not None
        loaded.load_model(stripped_path)
        assert loaded._fusion_conformal is None

    def test_checkpoint_without_temperature_resets_calibrator(
        self, checkpoint_path: str, tmp_path: object
    ) -> None:
        stripped = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        stripped["temperature"] = None
        stripped_path = str(tmp_path / "no_temperature.pt")  # type: ignore[operator]
        torch.save(stripped, stripped_path)

        loaded = _fresh_engine()
        loaded.load_model(checkpoint_path)
        assert loaded._fusion_calibrator is not None
        loaded.load_model(stripped_path)
        assert loaded._fusion_calibrator is None


class TestBackwardCompatibility:
    """v1 checkpoints (no refit reference) keep the documented degraded path."""

    def test_v1_checkpoint_loads_with_degraded_path(
        self, checkpoint_path: str, tmp_path: object
    ) -> None:
        legacy = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        legacy.pop("detector_refit_reference", None)
        legacy.pop("conformal_state", None)
        legacy["format_version"] = 1
        legacy_path = str(tmp_path / "legacy_v1.pt")  # type: ignore[operator]
        torch.save(legacy, legacy_path)

        loaded = _fresh_engine()
        loaded.load_model(legacy_path)
        assert loaded._fusion_trained
        assert loaded._detector_fit_reference is None
        # No reference -> detectors stay unfitted until first inference,
        # exactly the pre-v2 contract (auto-fit, warned).
        assert not any(det.is_fitted() for det in loaded.detectors.values())


class TestReferenceSizeGuard:
    """Oversized training data is not embedded; behaviour degrades loudly."""

    def test_capture_respects_size_bound(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine_mod, "DETECTOR_REFIT_REFERENCE_MAX_BYTES", 64)
        x, y = _build_corpus()
        torch.manual_seed(5)
        np.random.seed(5)
        engine = OmniMercuryEngine(mode="fusion", device="cpu")
        engine.fit_fusion(x[:80], y[:80], epochs=1, symbolic_weight=0.0)
        assert engine._detector_fit_reference is None


class TestFeatureExtractorDeterminism:
    """Root-cause locks: the fusion feature path is a pure function of input.

    Historically four groups depended on construction RNG (``temporal``
    LSTM init, ``dimensional`` autoencoder init, ``affective`` and
    ``multiverse`` placeholder draws) and one mutated per call (``neural``
    memory accrual), which is what made checkpoints unreproducible.
    """

    def test_features_are_instance_independent(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.normal(size=(32, _N_FEATURES)).astype(np.float32)

        torch.manual_seed(0)
        np.random.seed(0)
        first = OmniMercuryEngine(mode="fusion", device="cpu")
        torch.manual_seed(999)
        np.random.seed(999)
        second = OmniMercuryEngine(mode="fusion", device="cpu")

        features_a = first._extract_fusion_features(x, fit_detectors=True)
        features_b = second._extract_fusion_features(x, fit_detectors=True)
        assert set(features_a) == set(features_b)
        for group in sorted(features_a):
            delta = float((features_a[group] - features_b[group]).abs().max())
            assert (
                delta == 0.0
            ), f"feature group {group!r} is instance-dependent (max delta {delta})"
