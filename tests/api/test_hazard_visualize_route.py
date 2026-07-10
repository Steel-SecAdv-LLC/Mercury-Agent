# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for POST /api/v1/hazard/visualize.

Validates the actual artifacts over the HTTP surface: PNG magic bytes and
content type, RFC 7946 GeoJSON with provenance properties, auth parity with
the sibling detection routes (optional), and fail-loud 400/422 behavior.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")
pytest.importorskip("matplotlib")

# Disable rate limiting BEFORE importing the server module (it reads the env
# var at module-load time), mirroring tests/api/test_routes_comprehensive.py.
os.environ["OMNI_RATE_LIMIT_ENABLED"] = "false"

import numpy as np
from fastapi.testclient import TestClient

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def client() -> TestClient:
    """Create FastAPI test client with server app."""
    from omni_mercury_engine.api.server import app

    return TestClient(app)


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Create valid API key auth headers (sibling-route pattern)."""
    from omni_mercury_engine.api.auth import get_api_key_store

    store = get_api_key_store()
    raw_key, _ = store.create_key(name="hazard_test_key", user_id="hazard_test_user")
    return {"X-API-Key": raw_key}


def _thermal_arrays() -> dict[str, Any]:
    rng = np.random.default_rng(2)
    thermal = rng.normal(300.0, 5.0, size=(3, 24, 24))
    thermal[0, 5:8, 10:14] = 410.0
    return {"thermal_image": thermal.tolist()}


class TestHazardVisualizePng:
    def test_earthquake_png_from_raw_series(self, client: Any) -> None:
        rng = np.random.default_rng(0)
        series = (0.02 * rng.normal(size=2048)).tolist()
        response = client.post(
            "/api/v1/hazard/visualize",
            json={"request": {"hazard": "earthquake", "arrays": {"series": series}}},
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "image/png"
        assert response.content[:8] == PNG_MAGIC
        assert len(response.content) > 5_000

    def test_wildfire_png_from_raw_thermal(self, client: Any) -> None:
        response = client.post(
            "/api/v1/hazard/visualize",
            json={"request": {"hazard": "wildfire", "arrays": _thermal_arrays()}},
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "image/png"
        assert response.content[:8] == PNG_MAGIC

    def test_png_from_prior_diagnostics_payload(self, client: Any) -> None:
        rng = np.random.default_rng(4)
        payload = {
            "hazard": "meteor",
            "arrays": {"doppler_shift_profile": rng.normal(size=100).tolist()},
            "context": {"n_radar_samples": 101},
        }
        response = client.post(
            "/api/v1/hazard/visualize",
            json={"request": {"diagnostics": payload}},
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "image/png"
        assert response.content[:8] == PNG_MAGIC

    def test_works_with_auth_headers_too(self, client: Any, auth_headers: Any) -> None:
        rng = np.random.default_rng(5)
        payload = {
            "hazard": "meteor",
            "arrays": {"doppler_shift_profile": rng.normal(size=50).tolist()},
            "context": {},
        }
        response = client.post(
            "/api/v1/hazard/visualize",
            json={"request": {"diagnostics": payload}},
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert response.content[:8] == PNG_MAGIC


class TestHazardVisualizeGeoJson:
    GEOTRANSFORM = {
        "origin_lon": -120.0,
        "origin_lat": 40.0,
        "deg_per_pixel_lon": 0.01,
        "deg_per_pixel_lat": -0.01,
    }

    def test_wildfire_geojson(self, client: Any) -> None:
        response = client.post(
            "/api/v1/hazard/visualize",
            json={
                "request": {
                    "hazard": "wildfire",
                    "arrays": _thermal_arrays(),
                    "format": "geojson",
                    "geotransform": self.GEOTRANSFORM,
                }
            },
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "application/geo+json"
        fc = response.json()
        assert fc["type"] == "FeatureCollection"
        assert len(fc["features"]) == 1
        feature = fc["features"][0]
        assert feature["geometry"]["type"] == "Point"
        lon, lat = feature["geometry"]["coordinates"]
        assert -121.0 < lon < -119.0 and 39.0 < lat < 41.0
        assert feature["properties"]["source"] == "wildfire_ignition_hotspot"
        assert feature["properties"]["component_pixels"] == 12

    def test_geojson_without_geotransform_is_400(self, client: Any) -> None:
        response = client.post(
            "/api/v1/hazard/visualize",
            json={
                "request": {
                    "hazard": "wildfire",
                    "arrays": _thermal_arrays(),
                    "format": "geojson",
                }
            },
        )
        assert response.status_code == 400
        assert "geotransform" in response.json()["detail"]

    def test_landslide_geojson_is_400_no_zonal_output(self, client: Any) -> None:
        payload = {
            "hazard": "landslide",
            "arrays": {"failure_type_probs": [0.4, 0.2, 0.15, 0.1, 0.1, 0.05]},
            "context": {},
        }
        response = client.post(
            "/api/v1/hazard/visualize",
            json={
                "request": {
                    "diagnostics": payload,
                    "format": "geojson",
                    "geotransform": self.GEOTRANSFORM,
                }
            },
        )
        assert response.status_code == 400
        assert "no zonal/geographic output" in response.json()["detail"]


class TestHazardVisualizeBadInput:
    def test_missing_body_is_422(self, client: Any) -> None:
        response = client.post("/api/v1/hazard/visualize", json={})
        assert response.status_code == 422

    def test_bad_format_is_422(self, client: Any) -> None:
        response = client.post(
            "/api/v1/hazard/visualize",
            json={"request": {"hazard": "meteor", "arrays": {}, "format": "svg"}},
        )
        assert response.status_code == 422

    def test_neither_input_mode_is_400(self, client: Any) -> None:
        response = client.post("/api/v1/hazard/visualize", json={"request": {}})
        assert response.status_code == 400
        assert "diagnostics" in response.json()["detail"]

    def test_both_input_modes_is_400(self, client: Any) -> None:
        response = client.post(
            "/api/v1/hazard/visualize",
            json={
                "request": {
                    "hazard": "meteor",
                    "arrays": {"radar_series": [1.0, 2.0, 3.0]},
                    "diagnostics": {
                        "hazard": "meteor",
                        "arrays": {"doppler_shift_profile": [1.0]},
                    },
                }
            },
        )
        assert response.status_code == 400
        assert "not both" in response.json()["detail"]

    def test_unknown_hazard_is_400(self, client: Any) -> None:
        response = client.post(
            "/api/v1/hazard/visualize",
            json={"request": {"hazard": "sharknado", "arrays": {"series": [1.0, 2.0]}}},
        )
        assert response.status_code == 400
        assert "unknown hazard" in response.json()["detail"]

    def test_wrong_shape_is_400(self, client: Any) -> None:
        response = client.post(
            "/api/v1/hazard/visualize",
            json={
                "request": {
                    "hazard": "tornado",
                    "arrays": {"radar_sequence": [[1.0, 2.0], [3.0, 4.0]]},
                }
            },
        )
        assert response.status_code == 400
        assert "(sweeps, 64)" in response.json()["detail"]
