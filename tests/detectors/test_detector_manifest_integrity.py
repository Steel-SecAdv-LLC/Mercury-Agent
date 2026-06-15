# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Detector-manifest integrity and engine reachability (drift locks).

Mercury keeps two detector catalogs: the engine's hardcoded default set
(``OmniMercuryEngine._init_detectors``) and the declarative
``DETECTOR_MANIFEST``. They were maintained independently, which let a
detector be registered in the manifest yet unreachable through the engine
(geo_movement), and let a genuine first-class detector exist in neither
(kmeans_distance). These tests are the regression net that keeps the two
catalogs consistent and every registered BASE detector usable through the
engine seam (``register_detector`` / ``enable_detector`` / ``available_detectors``).
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

pytest.importorskip("torch")

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.core.detector_registry import DETECTOR_MANIFEST, DetectorCategory
from omni_mercury_engine.engine import OmniMercuryEngine

_BASE_ENTRIES = [e for e in DETECTOR_MANIFEST if e.category is DetectorCategory.BASE]
_CONTRACT_METHODS = ("fit", "detect", "extract_features", "is_fitted")


@pytest.fixture(scope="module")
def engine() -> OmniMercuryEngine:
    """One engine for the read-only catalog checks (construction is not cheap)."""
    return OmniMercuryEngine(mode="fusion", auto_load_checkpoint=False)


def test_every_manifest_entry_class_resolves() -> None:
    """No dead manifest entry: every module imports and exposes its class."""
    for entry in DETECTOR_MANIFEST:
        module = importlib.import_module(entry.module_path)
        assert hasattr(
            module, entry.class_name
        ), f"{entry.name}: {entry.module_path}.{entry.class_name} is missing"


@pytest.mark.parametrize("entry", _BASE_ENTRIES, ids=lambda e: e.name)
def test_base_detector_is_basedetector_with_contract(entry: Any) -> None:
    """Every BASE entry is a BaseDetector subclass exposing the engine's contract.

    This is the lock that would have caught ``kmeans_distance`` (a first-class
    detector that was not a ``BaseDetector`` and was registered nowhere).
    Checked at the class level — no instantiation — so it stays fast and never
    triggers a heavy/networked model build.
    """
    cls = getattr(importlib.import_module(entry.module_path), entry.class_name)
    assert issubclass(cls, BaseDetector), f"{entry.class_name} is not a BaseDetector"
    for method in _CONTRACT_METHODS:
        assert callable(getattr(cls, method, None)), f"{entry.class_name}.{method} missing"


def test_engine_defaults_are_a_subset_of_manifest(engine: OmniMercuryEngine) -> None:
    """The hardcoded default detector set cannot drift away from the manifest."""
    manifest_names = {entry.name for entry in DETECTOR_MANIFEST}
    missing = set(engine.detectors) - manifest_names
    assert not missing, f"engine default detectors absent from DETECTOR_MANIFEST: {missing}"


def test_available_detectors_covers_every_manifest_entry(engine: OmniMercuryEngine) -> None:
    available = engine.available_detectors()
    uncovered = {entry.name for entry in DETECTOR_MANIFEST} - set(available)
    assert not uncovered, f"available_detectors() omits manifest entries: {uncovered}"


def test_nondefault_base_detectors_reachable_via_engine() -> None:
    """BASE detectors outside the default five are reachable through the seam.

    geo_movement / graph_based / kmeans_distance are registered BASE detectors
    that are intentionally not in the default fusion set; the engine must still
    be able to enable them.
    """
    eng = OmniMercuryEngine(mode="fusion", auto_load_checkpoint=False)
    for name in ("geo_movement", "graph_based", "kmeans_distance"):
        assert eng.available_detectors()[name] is False
        det = eng.enable_detector(name)
        assert isinstance(det, BaseDetector)
        assert name in eng.detectors
        assert eng.available_detectors()[name] is True
