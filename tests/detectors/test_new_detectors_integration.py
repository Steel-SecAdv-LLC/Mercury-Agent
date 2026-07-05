# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Integration checks for the new streaming / statistical / state-space detectors.

These lock the wiring added on ``steel/detection-mechanisms``: every new manifest
entry must import, subclass :class:`~omni_mercury_engine.core.base.BaseDetector`,
expose the engine's four-method contract, register through
``DetectorRegistry.auto_discover_detectors`` with the declared ``feature_dim``,
and round-trip ``fit`` -> ``detect`` -> ``extract_features`` producing calibrated
``[0, 1]`` scores. This runs on the lightweight (torch-free) core so it guards
the seam even in the non-ML CI lane; the torch-gated
``test_detector_manifest_integrity`` covers full ``OmniMercuryEngine``
reachability.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.core.detector_registry import (
    DETECTOR_MANIFEST,
    DetectorCategory,
    DetectorRegistry,
)

# Detectors introduced on this branch, with a representative input generator.
_NEW_DETECTORS = ("spectral_residual", "bocpd", "spot_evt", "hawkes", "particle_filter")
_CONTRACT_METHODS = ("fit", "detect", "extract_features", "is_fitted")


def _manifest_entry(name: str):  # type: ignore[no-untyped-def]
    return next(e for e in DETECTOR_MANIFEST if e.name == name)


def _sample_series() -> np.ndarray:
    # Non-negative so it doubles as a valid count stream for the Hawkes detector.
    rng = np.random.default_rng(0)
    return np.abs(rng.normal(3.0, 1.0, 400)).astype(np.float64)


@pytest.mark.parametrize("name", _NEW_DETECTORS)
def test_new_entry_is_registered_and_base(name: str) -> None:
    entry = _manifest_entry(name)
    assert entry.category is DetectorCategory.BASE
    cls = getattr(importlib.import_module(entry.module_path), entry.class_name)
    assert issubclass(cls, BaseDetector)
    for method in _CONTRACT_METHODS:
        assert callable(getattr(cls, method, None)), f"{entry.class_name}.{method} missing"


@pytest.mark.parametrize("name", _NEW_DETECTORS)
def test_new_entry_round_trips(name: str) -> None:
    entry = _manifest_entry(name)
    cls = getattr(importlib.import_module(entry.module_path), entry.class_name)
    series = _sample_series()
    det = cls().fit(series)
    assert det.is_fitted() is True

    out = det.detect(series)
    scores = np.asarray(out["scores"])
    assert scores.shape == (len(series),)
    assert float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0
    assert np.isfinite(scores).all()

    feats = np.asarray(det.extract_features(series))
    assert feats.shape[0] == len(series)
    if entry.feature_dim is not None:
        assert feats.shape[1] == entry.feature_dim


def test_auto_discovery_registers_new_detectors() -> None:
    registry = DetectorRegistry()
    registry.auto_discover_detectors()
    for name in _NEW_DETECTORS:
        info = registry.get(name)
        assert info is not None, f"{name} not auto-discovered"
        assert info.feature_dim == _manifest_entry(name).feature_dim
