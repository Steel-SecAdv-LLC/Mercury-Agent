# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Provenance guard for the governed-fusion live suite.

The suite labels provenance statically (``RECONSTRUCTED_DOMAINS`` /
``RECONSTRUCTED_EVENTS``).  The one live-labelled loader that can *silently* fall
back to synthesis is ``marine`` (an empty OBIS response), and it tags every
synthesised row with ``dataset_id="synthetic"``.  These tests pin the read-only
guard that turns that marker into a refusal to label silent-fallback data as
live -- so a future OBIS outage can never inflate the 23-event live headline with
synthesised marine data.  All tests are offline (no network, no real OBIS call).
"""

from __future__ import annotations

import types
from typing import Any

import numpy as np
import pandas as pd
import pytest

from research.governed_fusion import suite


def _synthetic_marine_frame() -> pd.DataFrame:
    """Return a real marine synthesis frame (the OBIS-empty fallback output)."""
    from omni_mercury_engine.loaders.marine_loader import _EVENT_CATALOG, MarineLoader

    event = next(iter(_EVENT_CATALOG.values()))
    return MarineLoader._synthesize_event(event)


def _live_like_frame() -> pd.DataFrame:
    """Return a frame shaped like a live OBIS payload (real dataset IDs)."""
    return pd.DataFrame(
        {
            "scientificName": ["Acropora", "Porites"],
            "dataset_id": ["urn:lsid:obis.org:dataset:abc", "urn:lsid:obis.org:dataset:def"],
            "period": ["event", "baseline"],
        }
    )


def _fake_loader_module(frame: pd.DataFrame, class_name: str) -> types.SimpleNamespace:
    """Build a stand-in loader module whose ``fetch_historical`` returns *frame*."""

    class _FakeLoader:
        def fetch_historical(self, event_id: str) -> pd.DataFrame:
            return frame

        def engineer_features(self, raw: Any) -> np.ndarray:
            return np.zeros((10, 2), dtype=np.float64)

        def get_ground_truth(self, event_id: str) -> np.ndarray:
            return np.array([0] * 8 + [1] * 2, dtype=int)

    return types.SimpleNamespace(**{class_name: _FakeLoader})


def test_synthesis_marker_detected_on_real_marine_fallback() -> None:
    """The real marine fallback frame trips ``_looks_synthesized``."""
    frame = _synthetic_marine_frame()
    assert "dataset_id" in frame.columns
    assert (frame["dataset_id"] == "synthetic").all()
    assert suite._looks_synthesized(frame) is True


def test_marker_absent_on_live_like_and_unmarked_frames() -> None:
    """Live-like frames (real IDs) and frames without the column read as not-synthetic."""
    assert suite._looks_synthesized(_live_like_frame()) is False
    assert suite._looks_synthesized(pd.DataFrame({"a": [1, 2]})) is False
    assert suite._looks_synthesized(None) is False


def _patch_loader_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, fake: types.SimpleNamespace
) -> None:
    """Force a cache miss and make ``suite``'s dynamic loader import return *fake*."""
    monkeypatch.setattr(suite, "_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(suite, "importlib", types.SimpleNamespace(import_module=lambda name: fake))


def test_live_labelled_synthesized_event_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """A LIVE-labelled event whose loader synthesised raises ``ProvenanceError``."""
    fake = _fake_loader_module(_synthetic_marine_frame(), "MarineLoader")
    _patch_loader_import(monkeypatch, tmp_path, fake)

    assert suite.is_reconstructed("marine", "marine_heatwave_2023") is False
    with pytest.raises(suite.ProvenanceError):
        suite._load_event("marine", "marine_loader", "MarineLoader", "marine_heatwave_2023")


def test_reconstructed_event_is_allowed_to_synthesize(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """A RECONSTRUCTED-labelled event may synthesise -- the guard must not fire."""
    fake = _fake_loader_module(_synthetic_marine_frame(), "EnergyLoader")
    _patch_loader_import(monkeypatch, tmp_path, fake)

    assert suite.is_reconstructed("energy", "quebec_1989") is True
    event = suite._load_event("energy", "energy_loader", "EnergyLoader", "quebec_1989")
    assert event is not None
    assert event.reconstructed is True
