# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Offline-deterministic tests for SpaceWeatherLoader and MeteorLoader.

Meteor archives replay the recorded 2026-07-09 fixtures under
``tests/fixtures/space_weather/`` (real JPL CNEOS fireball + NASA NeoWs
responses). Space-weather payloads are constructed in-test to the exact
USGS geomag / DONKI GST response shapes the parsers consume (documented in
``_geomag_to_dataframe`` / ``_storm_windows``) — small, labelled as
constructed, and never presented as recorded data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.loaders import label_provenance
from omni_mercury_engine.loaders.meteor_loader import (  # type: ignore[import-not-found,unused-ignore]
    FIREBALL_ANOMALY_THRESHOLD_KT,
    MeteorLoader,
)
from omni_mercury_engine.loaders.space_weather_loader import (  # type: ignore[import-not-found,unused-ignore]
    KP_STORM_THRESHOLD,
    SpaceWeatherLoader,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "space_weather"


def _fixture(name: str) -> Any:
    """Unwrap a recorded fixture: raw payload lives under ["data"], with
    ["_provenance"] documenting source URL + fetch time alongside it."""
    return json.loads((FIXTURES / name).read_text())["data"]


def _geomag_payload(times: list[str], x: list[float], y: list[float]) -> dict[str, Any]:
    """Minimal USGS geomag web-service response shape."""
    return {
        "times": times,
        "values": [
            {"id": "X", "values": x},
            {"id": "Y", "values": y},
            {"id": "Z", "values": [0.0] * len(times)},
            {"id": "F", "values": [50000.0] * len(times)},
        ],
    }


@pytest.fixture()
def sw_loader(tmp_path: Path) -> SpaceWeatherLoader:
    loader = SpaceWeatherLoader(cache_dir=tmp_path)

    event = loader.list_events()[0]
    event_id = event["event_id"]
    meta = loader._event(event_id)

    # Six minute-cadence samples starting at the cataloged storm start; the
    # first three sit inside a Kp>=5 window ending +3 h from the first stamp.
    from datetime import datetime, timedelta

    t0 = datetime.fromisoformat(meta["start"].replace("Z", "+00:00"))
    times = [(t0 + timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(6)]
    gst_window_end = (t0 + timedelta(minutes=3)).strftime("%Y-%m-%dT%H:%MZ")

    def fake_fetch_json(url: str, params: dict[str, Any] | None = None) -> Any:
        if "DONKI" in url:
            return [
                {
                    "gstID": "TEST-GST-001",
                    "allKpIndex": [
                        {"observedTime": gst_window_end, "kpIndex": KP_STORM_THRESHOLD + 2}
                    ],
                }
            ]
        return _geomag_payload(
            times,
            x=[100.0, 105.0, 400.0, 90.0, 92.0, 91.0],
            y=[50.0, 51.0, 300.0, 49.0, 50.0, 50.5],
        )

    loader._fetch_json = fake_fetch_json  # type: ignore[method-assign, unused-ignore]
    return loader


class TestSpaceWeatherLoader:
    def test_catalog_and_contract(self, sw_loader: SpaceWeatherLoader) -> None:
        events = sw_loader.list_events()
        assert events, "catalog must list documented storms"
        assert {"event_id", "name", "date", "description"} <= set(events[0])
        assert sw_loader.LABEL_SOURCE == "statistical"
        assert sw_loader.DOMAIN == "space_weather"

    def test_fetch_historical_parses_geomag_shape(self, sw_loader: SpaceWeatherLoader) -> None:
        event_id = sw_loader.list_events()[0]["event_id"]
        df = sw_loader.fetch_historical(event_id)
        assert list(df.columns) == ["time", "x", "y", "z", "f"]
        assert len(df) == 6
        assert (np.diff(df["time"].to_numpy()) > 0).all()

    def test_ground_truth_marks_kp_window_rows(self, sw_loader: SpaceWeatherLoader) -> None:
        event_id = sw_loader.list_events()[0]["event_id"]
        labels = sw_loader.get_ground_truth(event_id)
        assert labels.shape == (6,)
        assert set(np.unique(labels)) <= {0, 1}
        # The Kp window [end-3h, end) covers the first samples only.
        assert labels[:3].sum() >= 1
        assert labels.sum() < 6, "a storm catalog must not label everything anomalous"

    def test_engineer_features_finite(self, sw_loader: SpaceWeatherLoader) -> None:
        event_id = sw_loader.list_events()[0]["event_id"]
        feats = sw_loader.engineer_features(sw_loader.fetch_historical(event_id))
        assert feats.ndim == 2 and feats.shape[0] == 6
        assert np.isfinite(feats).all()

    def test_empty_geomag_fails_loud(self, tmp_path: Path) -> None:
        loader = SpaceWeatherLoader(cache_dir=tmp_path)
        loader._fetch_json = lambda url, params=None: (  # type: ignore[method-assign, unused-ignore]
            [] if "DONKI" in url else {"times": [], "values": []}
        )
        with pytest.raises(ValueError, match="refusing to fabricate"):
            loader.fetch_historical(loader.list_events()[0]["event_id"])

    def test_provenance_registry_entry(self) -> None:
        registry = label_provenance.LABEL_PROVENANCE_REGISTRY
        key = "space_weather_loader.SpaceWeatherLoader"
        assert key in registry
        assert registry[key][0] == "statistical"


class TestMeteorLoader:
    def _loader(self, tmp_path: Path) -> MeteorLoader:
        loader = MeteorLoader(cache_dir=tmp_path)

        def fake_fetch_json(url: str, params: dict[str, Any] | None = None) -> Any:
            if "fireball" in url:
                return _fixture("jpl_fireball_2012_2013.json")
            return _fixture("neows_feed_2019_07_20.json")

        loader._fetch_json = fake_fetch_json  # type: ignore[method-assign, unused-ignore]
        return loader

    def test_catalog_and_contract(self, tmp_path: Path) -> None:
        loader = self._loader(tmp_path)
        events = loader.list_events()
        assert events
        assert loader.LABEL_SOURCE == "statistical"
        assert loader.DOMAIN == "meteor"
        types = {loader._event(e["event_id"])["type"] for e in events}
        assert "fireball" in types

    def test_fireball_event_replays_recorded_archive(self, tmp_path: Path) -> None:
        loader = self._loader(tmp_path)
        fireball_ids = [
            e["event_id"]
            for e in loader.list_events()
            if loader._event(e["event_id"])["type"] == "fireball"
        ]
        df = loader.fetch_historical(fireball_ids[0])
        assert not df.empty
        labels = loader.get_ground_truth(fireball_ids[0])
        assert labels.shape[0] == len(df)
        # The recorded 2012-2013 window contains Chelyabinsk (440 kt) — the
        # >= 1 kt threshold must flag it while leaving the common <1 kt
        # events unflagged.
        assert 0 < labels.sum() < len(df)

    def test_threshold_is_the_documented_rare_class_boundary(self) -> None:
        assert FIREBALL_ANOMALY_THRESHOLD_KT == 1.0

    def test_engineer_features_finite(self, tmp_path: Path) -> None:
        loader = self._loader(tmp_path)
        event_id = loader.list_events()[0]["event_id"]
        feats = loader.engineer_features(loader.fetch_historical(event_id))
        assert feats.ndim == 2
        assert np.isfinite(feats).all()

    def test_empty_archive_fails_loud(self, tmp_path: Path) -> None:
        loader = MeteorLoader(cache_dir=tmp_path)
        loader._fetch_json = lambda url, params=None: {  # type: ignore[method-assign, unused-ignore]
            "signature": {},
            "count": 0,
            "data": [],
            "near_earth_objects": {},
        }
        with pytest.raises(ValueError, match="refusing to fabricate"):
            loader.fetch_historical(loader.list_events()[0]["event_id"])

    def test_provenance_registry_entry(self) -> None:
        registry = label_provenance.LABEL_PROVENANCE_REGISTRY
        key = "meteor_loader.MeteorLoader"
        assert key in registry
        assert registry[key][0] == "statistical"
