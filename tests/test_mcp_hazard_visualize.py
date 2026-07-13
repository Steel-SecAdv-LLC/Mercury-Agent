# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the mercury_hazard_visualize MCP tool.

Drives the server with JSON-RPC 2.0 messages (the same wire an MCP client
speaks) and validates the returned artifacts: base64 PNG that decodes to real
PNG bytes, RFC 7946 GeoJSON with provenance, and transparent isError results for
bad input -- the same rendering behind 'mercury-agent hazard-viz' and
'POST /api/v1/hazard/visualize'.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("matplotlib")

from omni_mercury_engine.mcp_server import MercuryMCPServer

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _call(server: MercuryMCPServer, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    resp = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
    )
    assert resp is not None
    result = resp["result"]
    assert isinstance(result, dict)
    return result


def _payload(result: dict[str, Any]) -> dict[str, Any]:
    assert result["isError"] is False, result["content"][0]["text"]
    parsed = json.loads(result["content"][0]["text"])
    assert isinstance(parsed, dict)
    return parsed


def _thermal_arrays() -> dict[str, Any]:
    rng = np.random.default_rng(2)
    thermal = rng.normal(300.0, 5.0, size=(3, 24, 24))
    thermal[0, 5:8, 10:14] = 410.0
    return {"thermal_image": thermal.tolist()}


class TestManifest:
    def test_hazard_visualize_is_advertised(self) -> None:
        names = {t["name"] for t in MercuryMCPServer().manifest()}
        assert "mercury_hazard_visualize" in names


class TestPng:
    def test_earthquake_png_from_raw_series(self) -> None:
        server = MercuryMCPServer()
        rng = np.random.default_rng(0)
        series = (0.02 * rng.normal(size=2048)).tolist()
        payload = _payload(
            _call(
                server,
                "mercury_hazard_visualize",
                {"hazard": "earthquake", "arrays": {"series": series}},
            )
        )
        assert payload["hazard"] == "earthquake"
        assert payload["format"] == "png"
        png = base64.b64decode(payload["png_base64"])
        assert png[:8] == PNG_MAGIC
        assert len(png) == payload["size_bytes"] > 5_000

    def test_png_from_prior_diagnostics_payload(self) -> None:
        server = MercuryMCPServer()
        rng = np.random.default_rng(4)
        diagnostics = {
            "hazard": "meteor",
            "arrays": {"doppler_shift_profile": rng.normal(size=100).tolist()},
            "context": {"n_radar_samples": 101},
        }
        payload = _payload(_call(server, "mercury_hazard_visualize", {"diagnostics": diagnostics}))
        assert payload["hazard"] == "meteor"
        png = base64.b64decode(payload["png_base64"])
        assert png[:8] == PNG_MAGIC


class TestGeoJson:
    GEOTRANSFORM = {
        "origin_lon": -120.0,
        "origin_lat": 40.0,
        "deg_per_pixel_lon": 0.01,
        "deg_per_pixel_lat": -0.01,
    }

    def test_wildfire_geojson(self) -> None:
        server = MercuryMCPServer()
        payload = _payload(
            _call(
                server,
                "mercury_hazard_visualize",
                {
                    "hazard": "wildfire",
                    "arrays": _thermal_arrays(),
                    "format": "geojson",
                    "geotransform": self.GEOTRANSFORM,
                },
            )
        )
        assert payload["format"] == "geojson"
        fc = payload["geojson"]
        assert fc["type"] == "FeatureCollection"
        assert payload["n_features"] == len(fc["features"]) == 1
        feature = fc["features"][0]
        assert feature["type"] == "Feature"
        assert feature["geometry"]["type"] == "Point"
        assert len(feature["geometry"]["coordinates"]) == 2
        assert feature["properties"]["source"] == "wildfire_ignition_hotspot"

    def test_geojson_without_geotransform_is_error(self) -> None:
        server = MercuryMCPServer()
        result = _call(
            server,
            "mercury_hazard_visualize",
            {"hazard": "wildfire", "arrays": _thermal_arrays(), "format": "geojson"},
        )
        assert result["isError"] is True
        assert "geotransform" in result["content"][0]["text"]


class TestBadInput:
    def test_no_input_mode_is_error(self) -> None:
        result = _call(MercuryMCPServer(), "mercury_hazard_visualize", {})
        assert result["isError"] is True
        assert "diagnostics" in result["content"][0]["text"]

    def test_both_input_modes_is_error(self) -> None:
        result = _call(
            MercuryMCPServer(),
            "mercury_hazard_visualize",
            {
                "hazard": "meteor",
                "arrays": {"radar_series": [1.0, 2.0, 3.0]},
                "diagnostics": {
                    "hazard": "meteor",
                    "arrays": {"doppler_shift_profile": [1.0]},
                },
            },
        )
        assert result["isError"] is True
        assert "not both" in result["content"][0]["text"]

    def test_unknown_hazard_is_error(self) -> None:
        result = _call(
            MercuryMCPServer(),
            "mercury_hazard_visualize",
            {"hazard": "sharknado", "arrays": {"series": [1.0, 2.0]}},
        )
        assert result["isError"] is True
        assert "unknown hazard" in result["content"][0]["text"]

    def test_bad_format_is_error(self) -> None:
        result = _call(
            MercuryMCPServer(),
            "mercury_hazard_visualize",
            {"hazard": "meteor", "arrays": {"radar_series": [1.0, 2.0]}, "format": "svg"},
        )
        assert result["isError"] is True
        assert "format" in result["content"][0]["text"]
