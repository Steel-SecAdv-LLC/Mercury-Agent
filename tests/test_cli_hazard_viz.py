# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for `mercury-agent hazard-viz`.

CliRunner-driven artifact smoke tests: renders from a persisted diagnostics
payload (.npz) and by running a detector on raw input, validating the PNG
magic bytes / GeoJSON structure of what lands on disk, plus fail-loud usage
errors.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from click.testing import CliRunner

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("torch")
pytest.importorskip("matplotlib")

from omni_mercury_engine.cli import main

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_hazard_viz_help() -> None:
    result = CliRunner().invoke(main, ["hazard-viz", "--help"])
    assert result.exit_code == 0
    assert "--detector" in result.output
    assert "--geotransform" in result.output


def test_render_png_from_npz_payload(tmp_path: Path) -> None:
    """A persisted diagnostics payload renders to a real PNG on disk."""
    from omni_mercury_engine.detectors.geological.disaster_detectors import EarthquakeDetector

    rng = np.random.default_rng(0)
    series = 0.02 * rng.normal(size=2048)
    result = EarthquakeDetector(keep_diagnostics=True).predict_earthquake(series)
    assert result.diagnostics is not None
    payload_path = tmp_path / "quake.npz"
    result.diagnostics.to_npz(payload_path)

    out = tmp_path / "quake.png"
    cli_result = CliRunner().invoke(main, ["hazard-viz", "-i", str(payload_path), "-o", str(out)])
    assert cli_result.exit_code == 0, cli_result.output
    data = out.read_bytes()
    assert data[:8] == PNG_MAGIC
    assert len(data) > 5_000


def test_run_detector_and_render_png(tmp_path: Path) -> None:
    """--detector runs the hazard detector on raw input and renders its payload."""
    rng = np.random.default_rng(1)
    radar = rng.normal(size=128)
    data_path = tmp_path / "radar.csv"
    np.savetxt(data_path, radar, delimiter=",")

    out = tmp_path / "meteor.png"
    cli_result = CliRunner().invoke(
        main,
        ["hazard-viz", "-d", "meteor", "--data", str(data_path), "-o", str(out)],
    )
    assert cli_result.exit_code == 0, cli_result.output
    assert out.read_bytes()[:8] == PNG_MAGIC


def test_run_wildfire_and_render_geojson(tmp_path: Path) -> None:
    """Wildfire GeoJSON needs a geotransform and carries hotspot provenance."""
    rng = np.random.default_rng(2)
    thermal = rng.normal(300.0, 5.0, size=(3, 24, 24))
    thermal[0, 5:8, 10:14] = 410.0
    data_path = tmp_path / "thermal.npz"
    np.savez(data_path, thermal_image=thermal)

    gt_path = tmp_path / "gt.json"
    gt_path.write_text(
        json.dumps(
            {
                "origin_lon": -120.0,
                "origin_lat": 40.0,
                "deg_per_pixel_lon": 0.01,
                "deg_per_pixel_lat": -0.01,
            }
        )
    )

    out = tmp_path / "fire.geojson"
    cli_result = CliRunner().invoke(
        main,
        [
            "hazard-viz",
            "-d",
            "wildfire",
            "--data",
            str(data_path),
            "-f",
            "geojson",
            "--geotransform",
            str(gt_path),
            "-o",
            str(out),
        ],
    )
    assert cli_result.exit_code == 0, cli_result.output
    fc = json.loads(out.read_text())
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 1
    assert fc["features"][0]["properties"]["source"] == "wildfire_ignition_hotspot"


def test_geojson_without_geotransform_fails_loud(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    thermal = rng.normal(300.0, 5.0, size=(3, 16, 16))
    thermal[0, 4:6, 4:6] = 410.0
    data_path = tmp_path / "thermal.npz"
    np.savez(data_path, thermal_image=thermal)

    cli_result = CliRunner().invoke(
        main,
        [
            "hazard-viz",
            "-d",
            "wildfire",
            "--data",
            str(data_path),
            "-f",
            "geojson",
            "-o",
            str(tmp_path / "fire.geojson"),
        ],
    )
    assert cli_result.exit_code != 0
    assert "geotransform" in cli_result.output


def test_requires_exactly_one_input_mode(tmp_path: Path) -> None:
    cli_result = CliRunner().invoke(main, ["hazard-viz", "-o", str(tmp_path / "x.png")])
    assert cli_result.exit_code != 0
    assert "exactly one" in cli_result.output


def test_wrong_input_shape_fails_loud(tmp_path: Path) -> None:
    """A tornado field with the wrong gate count is refused with the reason."""
    data_path = tmp_path / "radar.npy"
    np.save(data_path, np.zeros((10, 8)))
    cli_result = CliRunner().invoke(
        main,
        [
            "hazard-viz",
            "-d",
            "tornado",
            "--data",
            str(data_path),
            "-o",
            str(tmp_path / "t.png"),
        ],
    )
    assert cli_result.exit_code != 0
    assert "(sweeps, 64)" in cli_result.output
