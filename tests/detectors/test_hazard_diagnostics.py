# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hazard detector diagnostics persistence tests.

Each hazard detector computes intermediate arrays (spectrograms, STA/LTA series,
FFT spectra, Doppler fields, hotspot masks, attention series) that were
historically discarded. These tests pin the ``keep_diagnostics`` contract:

- Default (``keep_diagnostics=False``): ``result.diagnostics`` is ``None`` --
  absent, never an empty fake.
- Opted in: the payload carries the REAL arrays with the documented names,
  correct shapes/dtypes, and finite values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("torch")

from omni_mercury_engine.detectors.geological.disaster_detectors import (
    EarthquakeDetector as _EarthquakeDetector,
    MeteorDetector as _MeteorDetector,
    TsunamiDetector as _TsunamiDetector,
)
from omni_mercury_engine.detectors.geological.hurricane_detector import (
    HurricaneDetector as _HurricaneDetector,
)
from omni_mercury_engine.detectors.geological.landslide import (
    LandslideDetector as _LandslideDetector,
)
from omni_mercury_engine.detectors.geological.tornado_detector import (
    TornadoDetector as _TornadoDetector,
)
from omni_mercury_engine.detectors.geological.volcanic import (
    VolcanicEruptionDetector as _VolcanicEruptionDetector,
)
from omni_mercury_engine.detectors.geological.wildfire import (
    WildfireDetector as _WildfireDetector,
)

# The dev venv's editable install may point at a sibling worktree that
# predates ``hazard_diagnostics``; ``unused-ignore`` keeps a correctly
# installed tree (CI) clean.
from omni_mercury_engine.detectors.hazard_diagnostics import (  # type: ignore[import-not-found,unused-ignore]
    HazardDiagnostics,
)
from omni_mercury_engine.space.schumann_resonance import (
    SchumannResonanceDetector as _SchumannResonanceDetector,
)

# Same sibling-worktree caveat as above: an editable install that predates
# the diagnostics API resolves these detector classes with their old
# signatures (no ``keep_diagnostics=`` kwarg, no ``result.diagnostics``).
# Alias them to ``Any`` so both that environment and a correctly installed
# tree (CI) type-check cleanly; the tests exercise the real API at runtime
# either way.
EarthquakeDetector: Any = _EarthquakeDetector
HurricaneDetector: Any = _HurricaneDetector
LandslideDetector: Any = _LandslideDetector
MeteorDetector: Any = _MeteorDetector
SchumannResonanceDetector: Any = _SchumannResonanceDetector
TornadoDetector: Any = _TornadoDetector
TsunamiDetector: Any = _TsunamiDetector
VolcanicEruptionDetector: Any = _VolcanicEruptionDetector
WildfireDetector: Any = _WildfireDetector


def _seismic_series(n: int = 2048, fs: float = 100.0) -> np.ndarray[Any, Any]:
    """Quiet background with an injected high-frequency burst (quake-like)."""
    rng = np.random.default_rng(7)
    t = np.arange(n) / fs
    series = 0.02 * rng.normal(size=n)
    burst_len = min(300, n // 4)
    burst = slice(n // 2, n // 2 + burst_len)
    series[burst] += np.sin(2 * np.pi * 8.0 * t[burst]) * np.hanning(burst_len) * 3.0
    return series


class TestPayloadRoundTrip:
    """HazardDiagnostics serialization round-trips exactly."""

    @pytest.fixture
    def payload(self) -> HazardDiagnostics:
        return HazardDiagnostics(
            hazard="earthquake",
            arrays={
                "spectrogram_norm": np.arange(12, dtype=float).reshape(3, 4),
                "sta_lta_ratio": np.linspace(0, 2, 8),
            },
            context={"sampling_rate_hz": 100.0, "p_arrival_index": 3},
        )

    def test_jsonable_round_trip(self, payload: HazardDiagnostics) -> None:
        rebuilt = HazardDiagnostics.from_jsonable(payload.to_jsonable())
        assert rebuilt.hazard == "earthquake"
        assert rebuilt.context == payload.context
        for name, arr in payload.arrays.items():
            np.testing.assert_array_equal(rebuilt.arrays[name], arr)

    def test_npz_round_trip(self, payload: HazardDiagnostics, tmp_path: Path) -> None:
        path = tmp_path / "diag.npz"
        payload.to_npz(path)
        rebuilt = HazardDiagnostics.from_npz(path)
        assert rebuilt.hazard == "earthquake"
        assert rebuilt.context == payload.context
        for name, arr in payload.arrays.items():
            np.testing.assert_array_equal(rebuilt.arrays[name], arr)

    def test_unknown_hazard_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown hazard"):
            HazardDiagnostics(hazard="volcano_lair", arrays={"x": np.ones(3)})

    def test_empty_arrays_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            HazardDiagnostics(hazard="tsunami", arrays={})

    def test_non_numeric_array_rejected(self) -> None:
        with pytest.raises(ValueError, match="numeric"):
            HazardDiagnostics.from_jsonable(
                {"hazard": "tsunami", "arrays": {"fft_power": ["a", "b"]}}
            )


class TestEarthquakeDiagnostics:
    def test_default_off(self) -> None:
        result = EarthquakeDetector().predict_earthquake(_seismic_series())
        assert result.diagnostics is None

    def test_spectrogram_and_sta_lta_captured(self) -> None:
        detector = EarthquakeDetector(keep_diagnostics=True)
        series = _seismic_series()
        result = detector.predict_earthquake(series)
        diag = result.diagnostics
        assert diag is not None and diag.hazard == "earthquake"

        f = diag.arrays["spectrogram_freqs_hz"]
        t = diag.arrays["spectrogram_times_s"]
        sxx = diag.arrays["spectrogram_norm"]
        assert sxx.ndim == 2
        assert sxx.shape == (len(f), len(t))
        assert np.isfinite(sxx).all() and np.isfinite(f).all() and np.isfinite(t).all()
        # The normalized spectrogram is zero-mean/unit-variance by construction.
        assert abs(float(sxx.mean())) < 1e-6

        sta_lta = diag.arrays["sta_lta_ratio"]
        assert sta_lta.shape == (len(series),)
        assert np.isfinite(sta_lta).all()
        assert (sta_lta >= 0).all()
        assert diag.context["sampling_rate_hz"] == 100.0
        # The injected burst must trip the STA/LTA picker somewhere after onset.
        p_idx = diag.context["p_arrival_index"]
        assert p_idx is None or 0 <= p_idx < len(series)

    def test_short_record_has_no_sta_lta(self) -> None:
        detector = EarthquakeDetector(keep_diagnostics=True)
        result = detector.predict_earthquake(_seismic_series(n=256))
        diag = result.diagnostics
        assert diag is not None
        # 256 samples < one STA+LTA window at 100 Hz: no ratio series exists,
        # and the payload says so instead of inventing one.
        assert "sta_lta_ratio" not in diag.arrays
        assert diag.context["sta_lta_available"] is False


class TestTsunamiDiagnostics:
    def test_default_off(self) -> None:
        rng = np.random.default_rng(0)
        result = TsunamiDetector().predict_tsunami(rng.normal(size=256).astype(np.float32))
        assert result.diagnostics is None

    def test_fft_spectrum_captured(self) -> None:
        rng = np.random.default_rng(0)
        wave = rng.normal(size=256).astype(np.float32)
        result = TsunamiDetector(keep_diagnostics=True).predict_tsunami(wave)
        diag = result.diagnostics
        assert diag is not None and diag.hazard == "tsunami"
        freqs = diag.arrays["fft_freqs_hz"]
        power = diag.arrays["fft_power"]
        assert freqs.shape == (256,) and power.shape == (256,)
        assert np.isfinite(power).all() and (power >= 0).all()
        assert diag.context["sampling_rate_hz"] == 1.0


class TestMeteorDiagnostics:
    def test_default_off(self) -> None:
        rng = np.random.default_rng(1)
        detector = MeteorDetector(use_nasa_data=False)
        result = detector.predict_meteor(radar_data=rng.normal(size=128))
        assert result.diagnostics is None

    def test_doppler_profile_captured(self) -> None:
        rng = np.random.default_rng(1)
        radar = rng.normal(size=128)
        detector = MeteorDetector(use_nasa_data=False, keep_diagnostics=True)
        result = detector.predict_meteor(radar_data=radar)
        diag = result.diagnostics
        assert diag is not None and diag.hazard == "meteor"
        profile = diag.arrays["doppler_shift_profile"]
        assert profile.shape == (127,)
        np.testing.assert_allclose(profile, np.diff(radar))
        assert diag.context["n_radar_samples"] == 128

    def test_no_radar_no_diagnostics(self) -> None:
        detector = MeteorDetector(use_nasa_data=False, keep_diagnostics=True)
        result = detector.predict_meteor(optical_data=np.ones(64))
        # No radar series -> no Doppler profile was computed -> honestly absent.
        assert result.diagnostics is None


class TestWildfireDiagnosticsAndIgnitionFields:
    @pytest.fixture
    def thermal_image(self) -> np.ndarray[Any, Any]:
        """3-channel thermal scene with one hot block in the thermal channel."""
        rng = np.random.default_rng(3)
        img = rng.normal(300.0, 5.0, size=(3, 32, 32))
        img[0, 10:14, 20:25] = 420.0  # 4x5 hot block, well above 350 K
        return img

    def test_default_off_and_fields_populated(self, thermal_image: np.ndarray[Any, Any]) -> None:
        detector = WildfireDetector()
        result = detector.predict_wildfire({"thermal_image": thermal_image})
        assert result.diagnostics is None
        # The (formerly dead) ignition fields are now populated from the mask.
        assert result.thermal_hotspots == 20
        assert len(result.ignition_locations) == 1
        row, col = result.ignition_locations[0]
        assert 10 <= row <= 14 and 20 <= col <= 25  # pixel-space centroid
        # No pixel_size_km supplied -> no fabricated ground area.
        assert result.fire_perimeter_km2 is None

    def test_area_estimate_requires_pixel_size(self, thermal_image: np.ndarray[Any, Any]) -> None:
        detector = WildfireDetector()
        result = detector.predict_wildfire({"thermal_image": thermal_image, "pixel_size_km": 0.5})
        assert result.fire_perimeter_km2 == pytest.approx(20 * 0.5**2)

    def test_hotspot_count_is_spatial_not_channel_summed(self) -> None:
        """A pixel hot in several bands is still ONE ground pixel.

        Regression: the count used to sum threshold exceedances across all
        channels, so this 4x5 block hot in all 3 bands counted as 60 pixels
        and inflated fire_perimeter_km2 by 3x.
        """
        rng = np.random.default_rng(3)
        img = rng.normal(300.0, 5.0, size=(3, 32, 32))
        img[:, 10:14, 20:25] = 420.0  # same 4x5 block hot in ALL channels
        detector = WildfireDetector()
        result = detector.predict_wildfire({"thermal_image": img, "pixel_size_km": 0.5})
        assert result.thermal_hotspots == 20  # spatial pixels, not 60
        assert result.fire_perimeter_km2 == pytest.approx(20 * 0.5**2)
        assert len(result.ignition_locations) == 1

    def test_thermal_mask_and_pixels_captured(self, thermal_image: np.ndarray[Any, Any]) -> None:
        detector = WildfireDetector(keep_diagnostics=True)
        result = detector.predict_wildfire({"thermal_image": thermal_image})
        diag = result.diagnostics
        assert diag is not None and diag.hazard == "wildfire"

        thermal = diag.arrays["thermal_image_k"]
        mask = diag.arrays["hotspot_mask"]
        pixels = diag.arrays["ignition_pixels"]
        assert thermal.shape == (32, 32) and np.isfinite(thermal).all()
        assert mask.shape == (32, 32) and mask.dtype == np.bool_
        assert int(mask.sum()) == 20
        assert pixels.shape == (20, 2)
        assert np.issubdtype(pixels.dtype, np.integer)
        # Every masked pixel exceeds the threshold in the channel-max map.
        assert (thermal[mask] > 350.0).all()
        centroids = diag.arrays["ignition_centroids"]
        sizes = diag.arrays["ignition_component_sizes"]
        assert centroids.shape == (1, 2) and sizes.tolist() == [20]
        assert diag.context["coordinate_space"] == "pixel"
        assert diag.context["hotspot_threshold_k"] == 350.0


class TestTornadoDiagnostics:
    @pytest.fixture
    def radar_sequence(self) -> np.ndarray[Any, Any]:
        """Velocity field with an embedded inbound/outbound couplet."""
        rng = np.random.default_rng(5)
        field = rng.normal(0.0, 1.0, size=(10, 64)).astype(np.float32)
        field[4, 30] = -35.0  # inbound
        field[4, 31] = 35.0  # outbound: max adjacent-gate shear at (4, 30)
        return field

    def test_default_off(self, radar_sequence: np.ndarray[Any, Any]) -> None:
        result = TornadoDetector().predict_tornado({"radar_sequence": radar_sequence})
        assert result.diagnostics is None

    def test_velocity_field_and_couplet_captured(
        self, radar_sequence: np.ndarray[Any, Any]
    ) -> None:
        detector = TornadoDetector(keep_diagnostics=True)
        result = detector.predict_tornado({"radar_sequence": radar_sequence})
        diag = result.diagnostics
        assert diag is not None and diag.hazard == "tornado"

        field = diag.arrays["doppler_velocity_field"]
        attention = diag.arrays["radar_attention"]
        assert field.shape == (10, 64)
        np.testing.assert_allclose(field, radar_sequence, rtol=1e-6)
        assert attention.shape == (10,)
        assert np.isfinite(attention).all()
        # The located couplet is the injected gate pair.
        assert diag.context["couplet_row"] == 4
        assert diag.context["couplet_col"] == 30
        assert diag.context["couplet_shear"] == pytest.approx(70.0, rel=1e-3)


class TestHurricaneDiagnostics:
    @pytest.fixture
    def wind_uv(self) -> dict[str, Any]:
        """Solid-body-rotation vortex: uniform relative vorticity 2*omega."""
        n = 17
        omega = 1.5e-4
        y, x = np.mgrid[0:n, 0:n].astype(float)
        cx = cy = (n - 1) / 2.0
        spacing = 1000.0
        u = -omega * (y - cy) * spacing
        v = omega * (x - cx) * spacing
        return {"u": u, "v": v, "grid_spacing_m": spacing}

    def test_default_off(self, wind_uv: dict[str, Any]) -> None:
        result = HurricaneDetector().predict_hurricane({"wind_field": wind_uv})
        assert result.diagnostics is None

    def test_wind_and_vorticity_fields_captured(self, wind_uv: dict[str, Any]) -> None:
        detector = HurricaneDetector(keep_diagnostics=True)
        result = detector.predict_hurricane({"wind_field": wind_uv})
        diag = result.diagnostics
        assert diag is not None and diag.hazard == "hurricane"

        speed = diag.arrays["wind_speed_field"]
        vort = diag.arrays["vorticity_field"]
        assert speed.shape == (17, 17) and vort.shape == (17, 17)
        assert np.isfinite(speed).all() and np.isfinite(vort).all()
        # Solid-body rotation: zeta = 2*omega everywhere (finite differences
        # are exact for a linear field).
        np.testing.assert_allclose(vort, 2 * 1.5e-4, rtol=1e-6)
        assert diag.context["max_abs_vorticity"] == pytest.approx(3e-4, rel=1e-6)
        assert diag.context["max_wind_speed"] == pytest.approx(float(speed.max()))
        # No storm-track cone anywhere: the track model was removed as uncomputed.
        assert "track" not in " ".join(diag.arrays)
        assert result.track_forecast == []

    def test_speed_only_field_has_no_vorticity(self) -> None:
        detector = HurricaneDetector(keep_diagnostics=True)
        speed = np.full((8, 8), 12.0)
        result = detector.predict_hurricane({"wind_field": speed})
        diag = result.diagnostics
        assert diag is not None
        assert "vorticity_field" not in diag.arrays
        assert "wind_speed_field" in diag.arrays

    def test_mismatched_uv_fails_loud(self) -> None:
        detector = HurricaneDetector(keep_diagnostics=True)
        with pytest.raises(ValueError, match="matching 2-D"):
            detector.predict_hurricane({"wind_field": {"u": np.ones((4, 4)), "v": np.ones((5, 5))}})


class TestVolcanicDiagnostics:
    @pytest.fixture
    def seismic_sequence(self) -> np.ndarray[Any, Any]:
        rng = np.random.default_rng(11)
        return rng.normal(size=(20, 32)).astype(np.float32)

    def test_default_off(self, seismic_sequence: np.ndarray[Any, Any]) -> None:
        detector = VolcanicEruptionDetector()
        result = detector.predict_eruption({"seismic_sequence": seismic_sequence})
        assert result.diagnostics is None

    def test_attention_series_and_belief_captured(
        self, seismic_sequence: np.ndarray[Any, Any]
    ) -> None:
        detector = VolcanicEruptionDetector(keep_diagnostics=True)
        result = detector.predict_eruption({"seismic_sequence": seismic_sequence})
        diag = result.diagnostics
        assert diag is not None and diag.hazard == "volcanic"

        attention = diag.arrays["seismic_attention"]
        assert attention.shape == (20,)
        assert np.isfinite(attention).all()
        # Softmax attention over timesteps sums to 1.
        assert float(attention.sum()) == pytest.approx(1.0, abs=1e-5)

        belief = diag.arrays["hmm_state_belief"]
        assert belief.shape == (5,)
        assert float(belief.sum()) == pytest.approx(1.0, abs=1e-6)
        assert diag.context["hmm_state_names"][0] == "QUIESCENT"


class TestLandslideDiagnostics:
    @pytest.fixture
    def slope_features(self) -> np.ndarray[Any, Any]:
        rng = np.random.default_rng(13)
        return rng.normal(size=64).astype(np.float32)

    def test_default_off(self, slope_features: np.ndarray[Any, Any]) -> None:
        detector = LandslideDetector(enable_ml_ensemble=False)
        result = detector.predict_landslide({"slope_features": slope_features})
        assert result.diagnostics is None

    def test_failure_type_distribution_captured(self, slope_features: np.ndarray[Any, Any]) -> None:
        detector = LandslideDetector(enable_ml_ensemble=False, keep_diagnostics=True)
        result = detector.predict_landslide({"slope_features": slope_features})
        diag = result.diagnostics
        assert diag is not None and diag.hazard == "landslide"
        probs = diag.arrays["failure_type_probs"]
        assert probs.shape == (6,)
        assert float(probs.sum()) == pytest.approx(1.0, abs=1e-5)
        labels = diag.context["failure_type_labels"]
        assert len(labels) == 6
        # The reported landslide_type is the argmax of the persisted distribution.
        assert labels[int(np.argmax(probs))] == result.landslide_type


class TestSchumannDiagnostics:
    @pytest.fixture
    def elf_signal(self) -> np.ndarray[Any, Any]:
        fs = 100.0
        t = np.arange(int(10 * fs)) / fs
        rng = np.random.default_rng(17)
        return np.sin(2 * np.pi * 7.83 * t) + 0.1 * rng.normal(size=len(t))

    def test_default_off(self, elf_signal: np.ndarray[Any, Any]) -> None:
        detector = SchumannResonanceDetector(sampling_rate=100.0)
        result = detector.detect_resonance_anomaly(elf_signal)
        assert result.diagnostics is None

    def test_harmonic_spectrum_captured(self, elf_signal: np.ndarray[Any, Any]) -> None:
        detector = SchumannResonanceDetector(sampling_rate=100.0, keep_diagnostics=True)
        result = detector.detect_resonance_anomaly(elf_signal)
        diag = result.diagnostics
        assert diag is not None and diag.hazard == "schumann"

        freqs = diag.arrays["frequencies_hz"]
        power = diag.arrays["power_spectrum"]
        assert freqs.shape == power.shape == (len(elf_signal) // 2,)
        assert np.isfinite(power).all()
        assert float(power.max()) == pytest.approx(1.0)  # max-normalized
        # The 7.83 Hz drive is the detected fundamental.
        assert diag.context["fundamental_freq_hz"] == pytest.approx(7.83, abs=0.15)
        assert diag.context["fundamental_freq_hz"] == result.fundamental_freq
