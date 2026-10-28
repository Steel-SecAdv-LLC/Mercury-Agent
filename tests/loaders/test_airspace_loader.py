# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""OpenSky airspace loader: parsing, bbox pinning, and honest-label contract.

The fixture mirrors the live API's 17-element state vectors (verified
2026-10-27 against the anonymous endpoint).
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from omni_mercury_engine.loaders.airspace_loader import (
    AirspaceLoader,
    parse_states,
)


def _state(
    icao24: str = "ada5f1",
    lon: float | None = 10.4574,
    lat: float | None = 46.2216,
    baro_alt: float | None = 12496.8,
    velocity: float | None = 252.15,
    track: float | None = 131.69,
    vert_rate: float | None = 0.33,
    geo_alt: float | None = 12938.76,
    squawk: str | None = "1173",
    on_ground: bool = False,
) -> list[Any]:
    return [
        icao24,
        "N979JW  ",
        "United States",
        1793072219,
        1793072220,
        lon,
        lat,
        baro_alt,
        on_ground,
        velocity,
        track,
        vert_rate,
        None,
        geo_alt,
        squawk,
        False,
        0,
    ]


def _payload(*states: list[Any]) -> dict[str, Any]:
    return {"time": 1793072221, "states": list(states)}


class TestParseStates:
    def test_parses_live_schema(self) -> None:
        df = parse_states(_payload(_state()))
        assert len(df) == 1
        row = df.iloc[0]
        assert row["icao24"] == "ada5f1"
        assert row["snapshot_time"] == 1793072221
        assert not row["squawk_emergency"]

    def test_emergency_squawks_are_flagged(self) -> None:
        df = parse_states(
            _payload(
                _state(icao24="a", squawk="7700"),
                _state(icao24="b", squawk="7600"),
                _state(icao24="c", squawk="7500"),
                _state(icao24="d", squawk="1200"),
            )
        )
        assert df["squawk_emergency"].tolist() == [True, True, True, False]

    def test_rows_without_position_or_velocity_are_dropped(self) -> None:
        df = parse_states(
            _payload(
                _state(icao24="ok"),
                _state(icao24="nolon", lon=None),
                _state(icao24="novel", velocity=None),
                _state(icao24="noalt", baro_alt=None),
            )
        )
        assert df["icao24"].tolist() == ["ok"]

    def test_null_states_mean_empty_snapshot(self) -> None:
        assert parse_states({"time": 1, "states": None}).empty

    def test_schema_drift_raises(self) -> None:
        with pytest.raises(ValueError, match="schema drifted"):
            parse_states({"aircraft": []})


class TestAirspaceLoader:
    def test_bbox_is_pinned_into_the_url(self, tmp_path: Any) -> None:
        loader = AirspaceLoader(
            lamin=45.8389, lomin=5.9962, lamax=47.8229, lomax=10.5226, cache_dir=tmp_path
        )
        assert loader._states_url().endswith(
            "/api/states/all?lamin=45.8389&lomin=5.9962&lamax=47.8229&lomax=10.5226"
        )
        assert AirspaceLoader(cache_dir=tmp_path)._states_url().endswith("/api/states/all")

    def test_partial_bbox_rejected(self, tmp_path: Any) -> None:
        with pytest.raises(ValueError, match="bounding box"):
            AirspaceLoader(lamin=1.0, cache_dir=tmp_path)

    def test_fetch_realtime_parses_mocked_transport(self, tmp_path: Any) -> None:
        body = json.dumps(_payload(_state(), _state(icao24="b", squawk="7700"))).encode()
        loader = AirspaceLoader(cache_dir=tmp_path)
        with patch(
            "omni_mercury_engine.loaders.base.BaseDomainLoader._fetch_url",
            return_value=body,
        ):
            df = loader.fetch_realtime()
        assert len(df) == 2
        assert int(df["squawk_emergency"].sum()) == 1

    def test_engineer_features_excludes_squawk_and_stays_finite(self, tmp_path: Any) -> None:
        df = parse_states(
            _payload(
                _state(),
                _state(icao24="nogeo", geo_alt=None),
                _state(icao24="notrack", track=None),
            )
        )
        loader = AirspaceLoader(cache_dir=tmp_path)
        features = loader.engineer_features(df)
        # notrack is dropped (non-finite); nogeo falls back to baro.
        assert features.shape == (2, len(loader.FEATURE_COLUMNS))
        assert np.isfinite(features).all()
        assert "squawk" not in loader.FEATURE_COLUMNS
        assert "squawk_emergency" not in loader.FEATURE_COLUMNS
        nogeo = features[1]
        assert nogeo[loader.FEATURE_COLUMNS.index("alt_disagreement_m")] == 0.0

    def test_empty_frame_gives_empty_matrix(self, tmp_path: Any) -> None:
        loader = AirspaceLoader(cache_dir=tmp_path)
        features = loader.engineer_features(parse_states({"time": 1, "states": []}))
        assert features.shape == (0, len(loader.FEATURE_COLUMNS))

    def test_history_is_refused_not_fabricated(self, tmp_path: Any) -> None:
        loader = AirspaceLoader(cache_dir=tmp_path)
        assert loader.list_events() == []
        with pytest.raises(NotImplementedError, match="does not fabricate"):
            loader.fetch_historical("some_event")
        with pytest.raises(NotImplementedError):
            loader.get_ground_truth("some_event")
