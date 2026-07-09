# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Offline-deterministic tests for the space-cluster live wiring.

Covers the canonical (deduplicated) SolarFlareDetector, the SolarStormDetector
live path and the SchumannResonanceDetector live path. All HTTP is replaced by
recorded fixtures under ``tests/fixtures/live_wiring/`` (captured from the
real NOAA SWPC API on 2026-07-09); live smoke tests live in
``tests/test_live_wiring_network.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.data_sources.base import DataSourceError
from omni_mercury_engine.data_sources.geomagnetic import BGSELFStationSource
from omni_mercury_engine.data_sources.live_ingestion import (
    LiveDataError,
    SimulatedDataError,
)
from omni_mercury_engine.data_sources.space_weather import NOAASWPCSource
from omni_mercury_engine.space.schumann_resonance import SchumannResonanceDetector
from omni_mercury_engine.space.solar_storm_detector import (
    SolarFlareDetector,
    SolarStormDetector,
)

FIXTURES = Path(__file__).parent / "fixtures" / "live_wiring"


def load_fixture(name: str) -> Any:
    """Load a recorded API response fixture."""
    return json.loads((FIXTURES / name).read_text())


class _FakeResponse:
    """Minimal httpx.Response stand-in."""

    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def json(self) -> Any:
        return self._payload


def make_donki_source() -> Any:
    """NASA DONKI client whose HTTP layer replays recorded FLR/GST payloads.

    Fixtures are REAL 30-day DONKI responses recorded 2026-07-09 through the
    live api.nasa.gov endpoint (71 solar flares incl. the 2026-06 M-class
    cluster, 1 geomagnetic storm).
    """
    from omni_mercury_engine.data_sources.space_weather import NASADONKISource

    source = NASADONKISource()
    flr = load_fixture("donki_flr.json")
    gst = load_fixture("donki_gst.json")

    async def _fake_http_get(endpoint: str, params: dict[str, Any] | None = None) -> _FakeResponse:
        if endpoint == "FLR":
            return _FakeResponse(flr)
        if endpoint == "GST":
            return _FakeResponse(gst)
        raise AssertionError(f"unexpected DONKI endpoint: {endpoint}")

    source._http_get = _fake_http_get  # type: ignore[assignment, method-assign, unused-ignore]
    return source


def make_swpc_source() -> NOAASWPCSource:
    """NOAA SWPC client whose HTTP layer replays recorded product payloads."""
    source = NOAASWPCSource()
    kp = load_fixture("swpc_kp.json")
    xray = load_fixture("swpc_xray.json")
    wind = load_fixture("swpc_solar_wind.json")

    async def _fake_http_get(endpoint: str, params: dict[str, Any] | None = None) -> _FakeResponse:
        if "k-index" in endpoint:
            return _FakeResponse(kp)
        if "xrays" in endpoint:
            return _FakeResponse(xray)
        if "propagated-solar-wind" in endpoint:
            return _FakeResponse(wind)
        raise AssertionError(f"unexpected SWPC endpoint: {endpoint}")

    source._http_get = _fake_http_get  # type: ignore[assignment, method-assign, unused-ignore]
    return source


class TestCanonicalSolarFlareDetector:
    """The single, deduplicated SolarFlareDetector."""

    def test_flux_classification_matches_goes_standard(self) -> None:
        detector = SolarFlareDetector()
        assert detector.predict_solar_flare(5e-9).flare_class == "A"
        assert detector.predict_solar_flare(5e-7).flare_class == "B"
        assert detector.predict_solar_flare(5e-6).flare_class == "C"
        assert detector.predict_solar_flare(5e-5).flare_class == "M"
        assert detector.predict_solar_flare(5e-4).flare_class == "X"

    def test_offline_storm_fields_are_none_not_fabricated(self) -> None:
        """No observed Kp -> NO Kp/Dst/storm-probability (honesty-wave rule)."""
        detector = SolarFlareDetector()
        result = detector.predict_solar_flare(2e-4)
        assert result.kp_index_predicted is None
        assert result.dst_index_predicted is None
        assert result.geomagnetic_storm_probability is None
        assert result.storm_forecast_source is None
        # The flare classification itself is honestly computable offline.
        assert result.flare_class == "X"
        assert result.flare_detected

    def test_observed_kp_populates_storm_fields(self) -> None:
        detector = SolarFlareDetector()
        result = detector.predict_solar_flare(2e-5, observed_kp=7.0, kp_source="test_kp")
        assert result.kp_index_predicted == pytest.approx(7.0)
        # G3/Kp7 anchor: Loewe & Proelss strong-storm midpoint.
        assert result.dst_index_predicted == pytest.approx(-150.0)
        assert result.geomagnetic_storm_probability == pytest.approx(0.75)
        assert result.storm_forecast_source == "test_kp"

    def test_dst_anchors_follow_documented_mapping(self) -> None:
        assert SolarFlareDetector._dst_from_kp(5.0) == pytest.approx(-40.0)
        assert SolarFlareDetector._dst_from_kp(6.0) == pytest.approx(-75.0)
        assert SolarFlareDetector._dst_from_kp(8.0) == pytest.approx(-275.0)
        assert SolarFlareDetector._dst_from_kp(9.0) == pytest.approx(-400.0)
        # Monotone decreasing in Kp, clamped at the ends.
        assert SolarFlareDetector._dst_from_kp(12.0) == pytest.approx(-400.0)
        assert SolarFlareDetector._dst_from_kp(0.0) == pytest.approx(0.0)

    def test_extract_features_survives_absent_storm_fields(self) -> None:
        detector = SolarFlareDetector()
        features = detector.extract_features(np.array([1e-6, 2e-5]))
        assert features.shape == (20,)
        assert np.all(np.isfinite(features))
        assert features[7] == 0.0  # storm probability absent -> encoded as 0
        assert features[8] == 0.0  # Kp absent -> encoded as 0

    def test_deterministic(self) -> None:
        detector = SolarFlareDetector()
        a = detector.predict_solar_flare(3e-6, observed_kp=5.0)
        b = detector.predict_solar_flare(3e-6, observed_kp=5.0)
        assert a == b

    def test_dict_api_unchanged(self) -> None:
        detector = SolarFlareDetector()
        result = detector.detect_solar_flare({"flux_short_wm2": 2e-5, "flux_long_wm2": 1e-5})
        assert result["flare_detected"] is True
        assert result["flare_class"] == "M"

    def test_no_client_fails_loud(self) -> None:
        with pytest.raises(LiveDataError, match="no NOAA SWPC client"):
            SolarFlareDetector().detect_live()

    def test_detect_live_from_recorded_swpc(self) -> None:
        detector = SolarFlareDetector(swpc_source=make_swpc_source())
        result = detector.detect_live()
        # Recorded fixture: latest long-channel flux ~5.7e-7 W/m^2 -> B/C band.
        assert result.flare_class in ("B", "C")
        assert result.data_provenance == "live"
        assert result.source_id == "noaa_swpc"
        # The OBSERVED planetary Kp populates the storm fields.
        assert result.kp_index_predicted == pytest.approx(2.67)
        assert result.storm_forecast_source == "noaa_swpc_planetary_k_index"
        assert result.dst_index_predicted is not None
        assert result.live_context is not None
        assert result.live_context["xray_points"] > 0

    def test_donki_corroboration_from_recorded_events(self) -> None:
        """The DONKI context path counts real recorded FLR/GST events.

        Regression: this corroboration branch previously had zero test
        coverage; a parse change could silently zero the counts.
        """
        detector = SolarFlareDetector(
            swpc_source=make_swpc_source(), donki_source=make_donki_source()
        )
        result = detector.detect_live()
        assert result.live_context is not None
        assert result.live_context["donki_recent_flares"] == 71
        assert result.live_context["donki_recent_storms"] == 1
        assert result.source_id is not None and "nasa_donki" in result.source_id

    def test_donki_failure_is_context_not_detection_failure(self) -> None:
        """A DONKI outage must be surfaced in context, never faked or fatal."""
        from omni_mercury_engine.data_sources.space_weather import NASADONKISource

        donki = NASADONKISource()

        async def _down(endpoint: str, params: Any | None = None) -> Any:
            raise DataSourceError("DONKI down", source_id="nasa_donki")

        donki._http_get = _down  # type: ignore[method-assign]
        detector = SolarFlareDetector(swpc_source=make_swpc_source(), donki_source=donki)
        result = detector.detect_live()
        assert result.data_provenance == "live"  # SWPC measurement unaffected
        assert result.live_context is not None
        assert "donki_error" in result.live_context
        assert "donki_recent_flares" not in result.live_context


class TestSolarStormDetectorLive:
    """SolarStormDetector live path against recorded SWPC products."""

    def test_no_client_fails_loud(self) -> None:
        detector = SolarStormDetector()
        with pytest.raises(LiveDataError, match="no NOAA SWPC client"):
            detector.predict_live()

    def test_predict_live_from_recorded_swpc(self) -> None:
        detector = SolarStormDetector(data_source=make_swpc_source())
        result = detector.predict_live()
        assert result.source_id == "noaa_swpc"
        assert result.data_provenance == "live"
        # Observed Kp (2.67 in the recorded fixture) outranks the estimate.
        assert result.kp_index == pytest.approx(2.67)
        assert result.geomagnetic_storm_level == "none"  # G0 at Kp 2.67
        assert result.live_context is not None
        assert result.live_context["solar_wind_speed_km_s"] is not None
        assert result.live_context["observed_kp"] == pytest.approx(2.67)

    def test_offline_path_unchanged(self) -> None:
        """The existing dict-driven offline API is untouched by the wiring."""
        detector = SolarStormDetector()
        result = detector.predict_solar_storm(
            {
                "xray_data": {"flux_short_wm2": 2e-6, "flux_long_wm2": 1e-6},
                "magnetosphere_data": {"solar_wind_speed_km_s": 400.0, "bz_imf_nt": -2.0},
            }
        )
        assert result.source_id is None
        assert result.data_provenance is None


class TestSchumannDetectorLive:
    """SchumannResonanceDetector live path against the honest BGS client."""

    def test_no_client_fails_loud(self) -> None:
        detector = SchumannResonanceDetector()
        with pytest.raises(LiveDataError, match="no BGS ELF client"):
            detector.detect_live()

    def test_simulated_feed_requires_explicit_opt_in(self) -> None:
        detector = SchumannResonanceDetector(data_source=BGSELFStationSource())
        with pytest.raises(SimulatedDataError):
            detector.detect_live()

    def test_simulated_feed_with_opt_in_is_labelled(self) -> None:
        detector = SchumannResonanceDetector(data_source=BGSELFStationSource())
        result = detector.detect_live(allow_simulated=True)
        assert result.data_provenance == "simulated"
        assert result.source_id == "bgs_elf"
        assert result.live_context is not None
        assert result.live_context["record_provenance"] == "simulated"
        # The simulated record is Schumann-shaped, so the fundamental is found.
        assert 6.0 <= result.fundamental_freq <= 10.0

    def test_instrument_record_is_live_provenance(self) -> None:
        detector = SchumannResonanceDetector(data_source=BGSELFStationSource())
        fs = 100.0
        t = np.arange(4096) / fs
        record = np.sin(2 * np.pi * 7.83 * t) + 0.05 * np.sin(2 * np.pi * 14.3 * t)
        result = detector.detect_live(raw_samples=record)
        assert result.data_provenance == "live"
        assert result.live_context is not None
        assert result.live_context["record_provenance"] == "instrument"
        assert result.fundamental_freq == pytest.approx(7.83, abs=0.15)

    def test_live_runs_same_pipeline_as_offline(self) -> None:
        """detect_live must run the detector's own FFT physics on the record."""
        detector = SchumannResonanceDetector(data_source=BGSELFStationSource())
        fs = 100.0
        t = np.arange(2048) / fs
        record = np.sin(2 * np.pi * 7.83 * t)
        live = detector.detect_live(raw_samples=record)
        offline = detector.detect_resonance_anomaly(record)
        assert live.fundamental_freq == offline.fundamental_freq
        assert live.anomaly_type == offline.anomaly_type
        assert live.confidence == offline.confidence
