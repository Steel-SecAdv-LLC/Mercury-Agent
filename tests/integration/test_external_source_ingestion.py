# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""External data-source ingestion integration (network mocked).

Mercury talks to live external feeds (USGS, NOAA, NASA FIRMS, DART buoys,
financial sources). This test exercises the ingestion seam end-to-end —
HTTP fetch → GeoJSON parse → feature matrix → detector consumption — with the
network call mocked at ``http_get_with_retry`` so it is deterministic and runs
by default (no ``@pytest.mark.network``, no live endpoint dependency, per the
convention in ``tests/conftest.py``).

It proves the contract between the dataset layer and the detection layer: the
matrix a loader emits is directly consumable by ``MercuryAnomalyDetector`` and
the strongest event surfaces as the top-scored sample.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from omni_mercury_engine.datasets.base import DatasetConfig
from omni_mercury_engine.datasets.environmental import USGSEarthquakeLoader
from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

pytestmark = pytest.mark.integration

_PATCH_TARGET = "omni_mercury_engine.datasets.environmental.http_get_with_retry"


def _geojson_quakes(magnitudes: list[float]) -> bytes:
    """A minimal USGS FeatureCollection with one feature per magnitude."""
    features = [
        {
            "geometry": {"coordinates": [-118.5 + i * 0.1, 34.1, 12.5]},
            "properties": {
                "mag": mag,
                "gap": 80,
                "dmin": 0.4,
                "rms": 0.3,
                "nst": 22,
                "horizontalError": 1.5,
                "depthError": 3.0,
                "magError": 0.1,
            },
        }
        for i, mag in enumerate(magnitudes)
    ]
    return json.dumps({"features": features}).encode()


def _loader(tmp_path: Any) -> USGSEarthquakeLoader:
    config = DatasetConfig(
        name="earthquake",
        data_dir=str(tmp_path / "data"),
        cache_dir=str(tmp_path / "cache"),
        preprocessing={"min_magnitude": 4.0, "days_back": 7},
        max_samples=50,
    )
    return USGSEarthquakeLoader(config)


class TestUsgsIngestionToFeatureMatrix:
    """The loader fetches, parses, and persists a consumable matrix."""

    def test_download_builds_request_and_parses_matrix(self, tmp_path: Any) -> None:
        loader = _loader(tmp_path)
        body = _geojson_quakes([4.2, 4.4, 4.3, 4.5, 7.9])

        captured: list[str] = []

        def fake_get(url: str, **_: Any) -> bytes:
            captured.append(url)
            return body

        with patch(_PATCH_TARGET, side_effect=fake_get):
            ok = loader._download_from_usgs()

        # Fetch happened exactly once, against a request built from config.
        assert ok is True
        assert loader.is_real_data is True
        assert len(captured) == 1
        assert "minmagnitude=4.0" in captured[0]
        assert "limit=50" in captured[0]

        # Parsed, persisted, and shaped to the documented 11-feature schema.
        cached = loader.data_path / "usgs_earthquake_real.npz"
        assert cached.exists()
        with np.load(cached) as payload:
            features = payload["features"]
            labels = payload["labels"]
        assert features.shape == (5, len(USGSEarthquakeLoader.FEATURE_NAMES))
        assert features.shape[1] == 11
        assert labels.shape == (5,)


class TestIngestedDataFeedsDetection:
    """The loader's output is directly detectable — the cross-layer contract."""

    def test_strongest_event_is_top_scored(self, tmp_path: Any) -> None:
        loader = _loader(tmp_path)
        # One clearly dominant event (M7.9) among moderate ones.
        body = _geojson_quakes([4.2, 4.4, 4.3, 4.5, 7.9])

        with patch(_PATCH_TARGET, return_value=body):
            assert loader._download_from_usgs() is True

        with np.load(loader.data_path / "usgs_earthquake_real.npz") as payload:
            features = payload["features"]

        detector = MercuryAnomalyDetector()
        detector.fit(features)
        result = detector.detect(features)

        scores = np.asarray(result["scores"], dtype=float)
        assert scores.shape == (5,)
        # The M7.9 (last feature) is the highest-scoring sample.
        assert int(np.argmax(scores)) == 4
