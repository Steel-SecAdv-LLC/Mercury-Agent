# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Engine detector-registration seam and inference-path robustness.

These tests pin two related contracts introduced to close a real wiring gap:

1.  Manifest detectors (e.g. ``geo_movement``) are declared in
    ``DETECTOR_MANIFEST`` but were unreachable through the engine — nothing
    consumed the manifest, so such a detector never participated in
    detect/fuse/decide. ``register_detector`` / ``enable_detector`` /
    ``available_detectors`` are the supported, opt-in bridge, and they are
    additive: the default five-detector path is unchanged until a caller
    opts in.

2.  The inference feature extractors (``_extract_detector_features`` /
    ``_extract_model_features``) must skip a detector that fail-louds with a
    Mercury ``OmniAnomalyException`` (e.g. ``geo_movement`` on
    non-trajectory data) exactly as the training-time extractor already
    does — one incompatible detector must never crash ``detect_with_fusion``.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("torch")

import logging

import numpy as np

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.detectors.geo_movement import GeoMovementAnomalyDetector
from omni_mercury_engine.engine import OmniMercuryEngine

_BASE_FIVE = {"statistical", "temporal", "spatial", "dimensional", "directive"}


class _ConstantDetector(BaseDetector):
    """Minimal BaseDetector returning per-sample features, for register tests."""

    def __init__(self, dim: int = 2) -> None:
        super().__init__({"threshold": 0.5})
        self._dim = dim

    def fit(self, data: Any) -> _ConstantDetector:
        self._is_fitted = True
        return self

    def detect(self, data: Any) -> dict[str, Any]:
        n = int(np.asarray(data).shape[0])
        return {"anomaly_score": 0.0, "is_anomaly": False, "scores": [0.0] * n}

    def extract_features(self, data: Any) -> np.ndarray[Any, Any]:
        n = int(np.asarray(data).shape[0])
        return np.zeros((n, self._dim), dtype=np.float32)


def _trajectory(n: int = 40) -> np.ndarray[Any, Any]:
    """A valid [n, 3] (lat, lon, epoch_s) trajectory: monotone time, in-range."""
    lat = 40.0 + np.cumsum(np.full(n, 0.001))
    lon = -105.0 + np.cumsum(np.full(n, 0.001))
    t = np.arange(n, dtype=float) * 3600.0
    return np.column_stack([lat, lon, t]).astype(np.float32)


def _engine() -> OmniMercuryEngine:
    return OmniMercuryEngine(mode="fusion", auto_load_checkpoint=False)


# ---------------------------------------------------------------------------
# Default path is unchanged (additive contract)
# ---------------------------------------------------------------------------
def test_default_detector_set_is_exactly_the_five_base() -> None:
    """A fresh engine exposes only the five general-purpose base detectors."""
    assert set(_engine().detectors) == _BASE_FIVE


# ---------------------------------------------------------------------------
# register_detector
# ---------------------------------------------------------------------------
def test_register_detector_adds_and_returns_self() -> None:
    eng = _engine()
    returned = eng.register_detector("constant", _ConstantDetector())
    assert returned is eng
    assert "constant" in eng.detectors
    assert isinstance(eng.detectors["constant"], _ConstantDetector)


def test_register_detector_rejects_non_basedetector() -> None:
    with pytest.raises(TypeError):
        _engine().register_detector("bad", object())  # type: ignore[arg-type]


def test_register_detector_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        _engine().register_detector("", _ConstantDetector())


def test_register_detector_refuses_clobber_without_replace() -> None:
    eng = _engine()
    with pytest.raises(ValueError):
        eng.register_detector("statistical", _ConstantDetector())
    # original built-in untouched
    assert not isinstance(eng.detectors["statistical"], _ConstantDetector)


def test_register_detector_replace_true_overrides() -> None:
    eng = _engine()
    eng.register_detector("statistical", _ConstantDetector(), replace=True)
    assert isinstance(eng.detectors["statistical"], _ConstantDetector)


def test_register_after_training_warns_and_records(caplog: pytest.LogCaptureFixture) -> None:
    """Registering after fusion training warns: ignored until fit_fusion re-runs."""
    eng = _engine()
    eng._fusion_trained = True
    eng._fusion_feature_groups = ["statistical", "temporal"]
    with caplog.at_level(logging.WARNING):
        eng.register_detector("constant", _ConstantDetector())
    assert "constant" in eng.detectors  # recorded
    assert any("until fit_fusion()" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# enable_detector (manifest bridge)
# ---------------------------------------------------------------------------
def test_enable_detector_wires_manifest_detector() -> None:
    eng = _engine()
    det = eng.enable_detector("geo_movement")
    assert isinstance(det, GeoMovementAnomalyDetector)
    assert isinstance(eng.detectors["geo_movement"], GeoMovementAnomalyDetector)


def test_enable_detector_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown detector"):
        _engine().enable_detector("no_such_detector")


def test_enable_detector_duplicate_raises() -> None:
    eng = _engine()
    eng.enable_detector("geo_movement")
    with pytest.raises(ValueError, match="already registered"):
        eng.enable_detector("geo_movement")


# ---------------------------------------------------------------------------
# available_detectors
# ---------------------------------------------------------------------------
def test_available_detectors_reflects_state() -> None:
    eng = _engine()
    avail = eng.available_detectors()
    assert avail["geo_movement"] is False
    assert avail["statistical"] is True
    eng.enable_detector("geo_movement")
    assert eng.available_detectors()["geo_movement"] is True


def test_available_detectors_includes_custom_active_detector() -> None:
    eng = _engine()
    eng.register_detector("custom_not_in_manifest", _ConstantDetector())
    assert eng.available_detectors()["custom_not_in_manifest"] is True


# ---------------------------------------------------------------------------
# Wiring: an enabled detector participates in the fusion feature set
# ---------------------------------------------------------------------------
def test_enabled_detector_contributes_fusion_feature_group() -> None:
    """geo_movement, once enabled, contributes its (n, 8) group to fusion features.

    Exercises ``_extract_fusion_features`` — the single source of truth for the
    training/inference fusion feature set — proving the detector is genuinely in
    the fuse path, not merely stored.
    """
    eng = _engine()
    eng.enable_detector("geo_movement")
    track = _trajectory(40)
    feats = eng._extract_fusion_features(track, fit_detectors=True)
    assert "geo_movement" in feats
    assert tuple(feats["geo_movement"].shape) == (40, 8)
    # The five base groups are still present — geo_movement is additive.
    assert _BASE_FIVE.issubset(set(feats))


# ---------------------------------------------------------------------------
# Robustness: a fail-loud detector is skipped, never crashes inference
# ---------------------------------------------------------------------------
def test_failloud_detector_skipped_on_inference_path() -> None:
    """Non-trajectory input makes geo_movement raise DetectorException.

    The inference extractor must skip it gracefully (regression lock for the
    train/inference symmetry fix), leaving the base detectors intact.
    """
    eng = _engine()
    eng.enable_detector("geo_movement")
    non_trajectory = np.random.RandomState(0).randn(20, 7).astype(np.float32)
    det_features, _scores, _certs = eng._extract_detector_features(non_trajectory)
    assert "geo_movement" not in det_features
    assert _BASE_FIVE & set(det_features)  # base detectors still produced features


def test_detect_with_fusion_does_not_crash_with_failloud_detector() -> None:
    """End-to-end public path: an incompatible enabled detector cannot crash it."""
    eng = _engine()
    eng.enable_detector("geo_movement")
    result = eng.detect_with_fusion(np.random.RandomState(1).randn(16, 5).astype(np.float32))
    assert isinstance(result, dict)
    assert "is_anomaly" in result
