# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Artifact smoke tests for the hazard diagnostics renderers.

Validates the artifacts themselves:
- PNG: magic bytes, non-trivial size, and byte-identical repeated rendering
  (the renderers are deterministic: fixed figsize/dpi/colormaps, no timestamps).
- GeoJSON: RFC 7946 structure (types, geometry, coordinate arity) and property
  provenance from the real detector outputs.
- Fail-loud paths: wrong hazard, missing arrays, missing geotransform, and the
  detectors that genuinely compute no zonal output.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from omni_mercury_engine.detectors.hazard_diagnostics import (  # type: ignore[import-not-found,unused-ignore]
    HazardDiagnostics,
)
from omni_mercury_engine.detectors.hazard_visuals import (  # type: ignore[import-not-found,unused-ignore]
    build_hazard_geojson,
    render_doppler_field,
    render_hazard_png,
    render_power_spectrum,
    render_score_series,
    render_spectrogram,
    render_thermal_map,
    render_wind_field,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
MIN_PNG_BYTES = 5_000  # a real plot, not a blank stub


def _assert_png(data: bytes) -> None:
    assert data[:8] == PNG_MAGIC
    assert len(data) > MIN_PNG_BYTES


@pytest.fixture
def earthquake_diag() -> HazardDiagnostics:
    rng = np.random.default_rng(0)
    f = np.linspace(0, 50, 40)
    t = np.linspace(0, 20, 60)
    return HazardDiagnostics(
        hazard="earthquake",
        arrays={
            "spectrogram_freqs_hz": f,
            "spectrogram_times_s": t,
            "spectrogram_norm": rng.normal(size=(40, 60)),
            "sta_lta_ratio": np.abs(rng.normal(size=2000)),
        },
        context={"sampling_rate_hz": 100.0, "p_arrival_index": 700, "s_arrival_index": 1100},
    )


@pytest.fixture
def tornado_diag() -> HazardDiagnostics:
    rng = np.random.default_rng(1)
    field = rng.normal(size=(12, 48))
    field[6, 20] = -30.0
    field[6, 21] = 30.0
    return HazardDiagnostics(
        hazard="tornado",
        arrays={
            "doppler_velocity_field": field,
            "radar_attention": np.full(12, 1 / 12),
        },
        context={"couplet_row": 6, "couplet_col": 20, "couplet_shear": 60.0},
    )


@pytest.fixture
def wildfire_diag() -> HazardDiagnostics:
    rng = np.random.default_rng(2)
    thermal = rng.normal(300.0, 5.0, size=(24, 24))
    thermal[5:8, 10:14] = 410.0
    mask = thermal > 350.0
    pixels = np.argwhere(mask)
    return HazardDiagnostics(
        hazard="wildfire",
        arrays={
            "thermal_image_k": thermal,
            "hotspot_mask": mask,
            "ignition_pixels": pixels,
            "ignition_centroids": np.array([[6.0, 11.5]]),
            "ignition_component_sizes": np.array([12]),
        },
        context={
            "hotspot_threshold_k": 350.0,
            "hotspot_count": 12,
            "coordinate_space": "pixel",
            "pixel_size_km": 0.375,
        },
    )


@pytest.fixture
def hurricane_diag() -> HazardDiagnostics:
    n = 16
    omega = 2e-4
    y, x = np.mgrid[0:n, 0:n].astype(float)
    c = (n - 1) / 2.0
    u = -omega * (y - c) * 500.0
    v = omega * (x - c) * 500.0
    return HazardDiagnostics(
        hazard="hurricane",
        arrays={
            "wind_speed_field": np.hypot(u, v),
            "wind_u": u,
            "wind_v": v,
            "vorticity_field": np.full((n, n), 2 * omega),
        },
        context={"grid_spacing_m": 500.0, "max_abs_vorticity": 4e-4},
    )


@pytest.fixture
def tsunami_diag() -> HazardDiagnostics:
    n = 256
    freqs = np.fft.fftfreq(n, d=1.0)
    rng = np.random.default_rng(3)
    return HazardDiagnostics(
        hazard="tsunami",
        arrays={"fft_freqs_hz": freqs, "fft_power": np.abs(rng.normal(size=n)) ** 2},
        context={"sampling_rate_hz": 1.0},
    )


@pytest.fixture
def schumann_diag() -> HazardDiagnostics:
    freqs = np.linspace(0, 50, 500)
    power = np.exp(-((freqs - 7.83) ** 2)) + 0.3 * np.exp(-((freqs - 14.3) ** 2))
    return HazardDiagnostics(
        hazard="schumann",
        arrays={"frequencies_hz": freqs, "power_spectrum": power / power.max()},
        context={"schumann_harmonics_hz": [7.83, 14.3, 20.8, 27.3, 33.8]},
    )


@pytest.fixture
def meteor_diag() -> HazardDiagnostics:
    rng = np.random.default_rng(4)
    return HazardDiagnostics(
        hazard="meteor",
        arrays={"doppler_shift_profile": rng.normal(size=127)},
        context={"n_radar_samples": 128},
    )


@pytest.fixture
def volcanic_diag() -> HazardDiagnostics:
    attention = np.linspace(1, 2, 20)
    attention /= attention.sum()
    return HazardDiagnostics(
        hazard="volcanic",
        arrays={
            "seismic_attention": attention,
            "hmm_state_belief": np.array([0.6, 0.25, 0.1, 0.03, 0.02]),
        },
        context={
            "hmm_state_names": ["QUIESCENT", "UNREST", "PRE_ERUPTIVE", "ERUPTIVE", "POST"],
        },
    )


@pytest.fixture
def landslide_diag() -> HazardDiagnostics:
    return HazardDiagnostics(
        hazard="landslide",
        arrays={"failure_type_probs": np.array([0.4, 0.2, 0.15, 0.1, 0.1, 0.05])},
        context={
            "failure_type_labels": [
                "debris_flow",
                "rock_slide",
                "earth_flow",
                "snow_avalanche",
                "mud_flow",
                "rotational_slide",
            ]
        },
    )


class TestPngRenderers:
    def test_spectrogram_png(self, earthquake_diag: HazardDiagnostics) -> None:
        _assert_png(render_spectrogram(earthquake_diag))

    def test_spectrogram_without_sta_lta(self, earthquake_diag: HazardDiagnostics) -> None:
        del earthquake_diag.arrays["sta_lta_ratio"]
        _assert_png(render_spectrogram(earthquake_diag))

    def test_doppler_png(self, tornado_diag: HazardDiagnostics) -> None:
        _assert_png(render_doppler_field(tornado_diag))

    def test_thermal_png(self, wildfire_diag: HazardDiagnostics) -> None:
        _assert_png(render_thermal_map(wildfire_diag))

    def test_wind_field_png(self, hurricane_diag: HazardDiagnostics) -> None:
        _assert_png(render_wind_field(hurricane_diag))

    def test_wind_field_speed_only(self) -> None:
        diag = HazardDiagnostics(
            hazard="hurricane",
            arrays={"wind_speed_field": np.full((8, 8), 10.0)},
            context={},
        )
        _assert_png(render_wind_field(diag))

    def test_power_spectrum_pngs(
        self,
        tsunami_diag: HazardDiagnostics,
        schumann_diag: HazardDiagnostics,
        meteor_diag: HazardDiagnostics,
    ) -> None:
        for diag in (tsunami_diag, schumann_diag, meteor_diag):
            _assert_png(render_power_spectrum(diag))

    def test_score_series_pngs(
        self, volcanic_diag: HazardDiagnostics, landslide_diag: HazardDiagnostics
    ) -> None:
        _assert_png(render_score_series(volcanic_diag))
        _assert_png(render_score_series(landslide_diag))

    def test_dispatcher_covers_every_hazard(
        self,
        earthquake_diag: HazardDiagnostics,
        tornado_diag: HazardDiagnostics,
        wildfire_diag: HazardDiagnostics,
        hurricane_diag: HazardDiagnostics,
        tsunami_diag: HazardDiagnostics,
        schumann_diag: HazardDiagnostics,
        meteor_diag: HazardDiagnostics,
        volcanic_diag: HazardDiagnostics,
        landslide_diag: HazardDiagnostics,
    ) -> None:
        for diag in (
            earthquake_diag,
            tornado_diag,
            wildfire_diag,
            hurricane_diag,
            tsunami_diag,
            schumann_diag,
            meteor_diag,
            volcanic_diag,
            landslide_diag,
        ):
            _assert_png(render_hazard_png(diag))

    def test_rendering_is_deterministic(
        self,
        earthquake_diag: HazardDiagnostics,
        tornado_diag: HazardDiagnostics,
        wildfire_diag: HazardDiagnostics,
        hurricane_diag: HazardDiagnostics,
    ) -> None:
        for diag in (earthquake_diag, tornado_diag, wildfire_diag, hurricane_diag):
            first = render_hazard_png(diag)
            second = render_hazard_png(diag)
            assert first == second, f"non-deterministic render for {diag.hazard}"

    def test_accepts_jsonable_payload(self, tornado_diag: HazardDiagnostics) -> None:
        as_dict = tornado_diag.to_jsonable()
        assert render_doppler_field(as_dict) == render_doppler_field(tornado_diag)

    def test_wrong_hazard_fails_loud(
        self, tornado_diag: HazardDiagnostics, earthquake_diag: HazardDiagnostics
    ) -> None:
        with pytest.raises(ValueError, match="earthquake"):
            render_spectrogram(tornado_diag)
        with pytest.raises(ValueError, match="tornado"):
            render_doppler_field(earthquake_diag)

    def test_missing_array_fails_loud(self) -> None:
        diag = HazardDiagnostics(
            hazard="earthquake",
            arrays={"spectrogram_freqs_hz": np.ones(4)},
            context={},
        )
        with pytest.raises(ValueError, match="missing arrays"):
            render_spectrogram(diag)


class TestGeoJson:
    GEOTRANSFORM = {
        "origin_lon": -120.0,
        "origin_lat": 40.0,
        "deg_per_pixel_lon": 0.01,
        "deg_per_pixel_lat": -0.01,
    }

    def test_structure_and_provenance(self, wildfire_diag: HazardDiagnostics) -> None:
        fc = build_hazard_geojson(wildfire_diag, geotransform=self.GEOTRANSFORM)
        assert fc["type"] == "FeatureCollection"
        assert isinstance(fc["features"], list) and len(fc["features"]) == 1

        feature: dict[str, Any] = fc["features"][0]
        assert feature["type"] == "Feature"
        geometry = feature["geometry"]
        assert geometry["type"] == "Point"
        coords = geometry["coordinates"]
        assert isinstance(coords, list) and len(coords) == 2
        lon, lat = coords
        assert isinstance(lon, float) and isinstance(lat, float)
        # Pixel (6.0, 11.5) mapped through the supplied geotransform.
        assert lon == pytest.approx(-120.0 + 11.5 * 0.01)
        assert lat == pytest.approx(40.0 - 6.0 * 0.01)

        props = feature["properties"]
        assert props["source"] == "wildfire_ignition_hotspot"
        assert props["pixel_row"] == 6.0 and props["pixel_col"] == 11.5
        assert props["component_pixels"] == 12
        assert props["hotspot_threshold_k"] == 350.0
        assert props["component_area_km2"] == pytest.approx(12 * 0.375**2)

    def test_geojson_is_json_serializable(self, wildfire_diag: HazardDiagnostics) -> None:
        import json

        fc = build_hazard_geojson(wildfire_diag, geotransform=self.GEOTRANSFORM)
        rebuilt = json.loads(json.dumps(fc))
        assert rebuilt == fc

    def test_missing_geotransform_fails_loud(self, wildfire_diag: HazardDiagnostics) -> None:
        with pytest.raises(ValueError, match="geotransform"):
            build_hazard_geojson(wildfire_diag)

    def test_incomplete_geotransform_fails_loud(self, wildfire_diag: HazardDiagnostics) -> None:
        with pytest.raises(ValueError, match="missing keys"):
            build_hazard_geojson(wildfire_diag, geotransform={"origin_lon": 0.0})

    def test_landslide_has_no_zonal_output(self, landslide_diag: HazardDiagnostics) -> None:
        with pytest.raises(ValueError, match="no zonal/geographic output"):
            build_hazard_geojson(landslide_diag, geotransform=self.GEOTRANSFORM)

    def test_other_hazards_emit_no_coordinates(self, tornado_diag: HazardDiagnostics) -> None:
        with pytest.raises(ValueError, match="no coordinates"):
            build_hazard_geojson(tornado_diag, geotransform=self.GEOTRANSFORM)
