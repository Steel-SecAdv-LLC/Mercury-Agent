# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the hazard diagnostics dashboard panels.

The GUI dashboard renders the SAME persisted diagnostics payloads as the
PNG/GeoJSON artifact path -- interactive Plotly heatmaps for the earthquake
spectrogram, tornado Doppler field, wildfire thermal map, hurricane
wind/vorticity fields, and line/bar panels for the 1-D payloads. These tests
validate that each panel builds from a real payload, carries the payload's own
arrays (not synthesized data), and fails loud on the wrong hazard.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("plotly")

# The dev venv's editable install may point at a sibling worktree that
# predates ``hazard_diagnostics`` and the hazard dashboard panels;
# ``unused-ignore`` keeps a correctly installed tree (CI) clean.
from omni_mercury_engine.detectors.hazard_diagnostics import (  # type: ignore[import-not-found,unused-ignore]
    HazardDiagnostics,
)
from omni_mercury_engine.gui.visualization_dashboard import (  # type: ignore[attr-defined,unused-ignore]
    DashboardBuilder,
    HazardDiagnosticsVisualizer,
)


@pytest.fixture
def visualizer() -> HazardDiagnosticsVisualizer:
    return HazardDiagnosticsVisualizer()


@pytest.fixture
def earthquake_diag() -> HazardDiagnostics:
    rng = np.random.default_rng(0)
    return HazardDiagnostics(
        hazard="earthquake",
        arrays={
            "spectrogram_freqs_hz": np.linspace(0, 50, 20),
            "spectrogram_times_s": np.linspace(0, 10, 30),
            "spectrogram_norm": rng.normal(size=(20, 30)),
            "sta_lta_ratio": np.abs(rng.normal(size=1000)),
        },
        context={"sampling_rate_hz": 100.0, "p_arrival_index": 400, "s_arrival_index": 600},
    )


@pytest.fixture
def tornado_diag() -> HazardDiagnostics:
    rng = np.random.default_rng(1)
    field = rng.normal(size=(8, 32))
    return HazardDiagnostics(
        hazard="tornado",
        arrays={"doppler_velocity_field": field, "radar_attention": np.full(8, 1 / 8)},
        context={"couplet_row": 3, "couplet_col": 10, "couplet_shear": 12.0},
    )


@pytest.fixture
def wildfire_diag() -> HazardDiagnostics:
    rng = np.random.default_rng(2)
    thermal = rng.normal(300.0, 5.0, size=(16, 16))
    thermal[4:6, 8:10] = 410.0
    return HazardDiagnostics(
        hazard="wildfire",
        arrays={
            "thermal_image_k": thermal,
            "hotspot_mask": thermal > 350.0,
            "ignition_pixels": np.argwhere(thermal > 350.0),
            "ignition_centroids": np.array([[4.5, 8.5]]),
            "ignition_component_sizes": np.array([4]),
        },
        context={"hotspot_threshold_k": 350.0, "coordinate_space": "pixel"},
    )


@pytest.fixture
def hurricane_diag() -> HazardDiagnostics:
    n = 12
    return HazardDiagnostics(
        hazard="hurricane",
        arrays={
            "wind_speed_field": np.full((n, n), 20.0),
            "wind_u": np.zeros((n, n)),
            "wind_v": np.full((n, n), 20.0),
            "vorticity_field": np.zeros((n, n)),
        },
        context={"grid_spacing_m": 1000.0},
    )


class TestHazardPanels:
    def test_spectrogram_heatmap(
        self, visualizer: HazardDiagnosticsVisualizer, earthquake_diag: HazardDiagnostics
    ) -> None:
        fig = visualizer.spectrogram_heatmap(earthquake_diag)
        heatmaps = [t for t in fig.data if t.type == "heatmap"]
        assert len(heatmaps) == 1
        # The panel plots the payload's own spectrogram, not synthesized data.
        np.testing.assert_array_equal(
            np.asarray(heatmaps[0].z), earthquake_diag.arrays["spectrogram_norm"]
        )

    def test_doppler_heatmap_marks_couplet(
        self, visualizer: HazardDiagnosticsVisualizer, tornado_diag: HazardDiagnostics
    ) -> None:
        fig = visualizer.doppler_field_heatmap(tornado_diag)
        types = {t.type for t in fig.data}
        assert types == {"heatmap", "scatter"}
        scatter = next(t for t in fig.data if t.type == "scatter")
        assert list(scatter.x) == [10, 11] and list(scatter.y) == [3, 3]

    def test_thermal_heatmap_marks_centroids(
        self, visualizer: HazardDiagnosticsVisualizer, wildfire_diag: HazardDiagnostics
    ) -> None:
        fig = visualizer.thermal_map_heatmap(wildfire_diag)
        scatter = next(t for t in fig.data if t.type == "scatter")
        assert list(scatter.x) == [8.5] and list(scatter.y) == [4.5]

    def test_wind_field_with_vorticity_has_two_heatmaps(
        self, visualizer: HazardDiagnosticsVisualizer, hurricane_diag: HazardDiagnostics
    ) -> None:
        fig = visualizer.wind_field_heatmap(hurricane_diag)
        heatmaps = [t for t in fig.data if t.type == "heatmap"]
        assert len(heatmaps) == 2

    def test_wind_field_speed_only_has_one_heatmap(
        self, visualizer: HazardDiagnosticsVisualizer
    ) -> None:
        diag = HazardDiagnostics(
            hazard="hurricane",
            arrays={"wind_speed_field": np.full((6, 6), 5.0)},
            context={},
        )
        fig = visualizer.wind_field_heatmap(diag)
        heatmaps = [t for t in fig.data if t.type == "heatmap"]
        assert len(heatmaps) == 1

    @pytest.mark.parametrize(
        ("hazard", "arrays", "context"),
        [
            (
                "tsunami",
                {
                    "fft_freqs_hz": np.fft.fftfreq(64, d=1.0),
                    "fft_power": np.abs(np.random.default_rng(3).normal(size=64)) ** 2,
                },
                {},
            ),
            (
                "schumann",
                {
                    "frequencies_hz": np.linspace(0, 50, 100),
                    "power_spectrum": np.linspace(1, 0, 100),
                },
                {"schumann_harmonics_hz": [7.83, 14.3]},
            ),
            (
                "meteor",
                {"doppler_shift_profile": np.random.default_rng(4).normal(size=50)},
                {},
            ),
            (
                "volcanic",
                {"seismic_attention": np.full(10, 0.1)},
                {},
            ),
            (
                "landslide",
                {"failure_type_probs": np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.05])},
                {"failure_type_labels": ["a", "b", "c", "d", "e", "f"]},
            ),
        ],
    )
    def test_spectrum_panel_covers_1d_hazards(
        self,
        visualizer: HazardDiagnosticsVisualizer,
        hazard: str,
        arrays: dict[str, np.ndarray],
        context: dict[str, object],
    ) -> None:
        diag = HazardDiagnostics(hazard=hazard, arrays=arrays, context=context)
        fig = visualizer.spectrum_panel(diag)
        assert len(fig.data) >= 1

    def test_hazard_panel_dispatches_and_accepts_dict(
        self, visualizer: HazardDiagnosticsVisualizer, tornado_diag: HazardDiagnostics
    ) -> None:
        fig_obj = visualizer.hazard_panel(tornado_diag)
        fig_dict = visualizer.hazard_panel(tornado_diag.to_jsonable())
        assert {t.type for t in fig_obj.data} == {t.type for t in fig_dict.data}

    def test_wrong_hazard_fails_loud(
        self, visualizer: HazardDiagnosticsVisualizer, tornado_diag: HazardDiagnostics
    ) -> None:
        with pytest.raises(ValueError, match="earthquake"):
            visualizer.spectrogram_heatmap(tornado_diag)


class TestDashboardIntegration:
    def test_add_hazard_panel_to_builder(self, earthquake_diag: HazardDiagnostics) -> None:
        builder = DashboardBuilder()
        # ``add_hazard_panel`` postdates the sibling-worktree editable install.
        builder.add_hazard_panel(  # type: ignore[attr-defined, unused-ignore]
            "Seismic Spectrogram", earthquake_diag
        )
        assert "Seismic Spectrogram" in builder._figures
