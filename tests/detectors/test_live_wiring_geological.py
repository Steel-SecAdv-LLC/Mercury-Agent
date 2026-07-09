# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Offline-deterministic tests for the geological-cluster live wiring.

Earthquake / tsunami / meteor detectors against recorded USGS + NASA/JPL
fixtures (``tests/fixtures/live_wiring/``, captured 2026-07-09), plus the
import-compatibility contract for the deduplicated SolarFlareDetector and the
consolidated NASA loaders. Live smoke tests live in
``tests/test_live_wiring_network.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.data_sources.earth_science import USGSEarthquakeSource
from omni_mercury_engine.data_sources.jpl_ssd import (
    JPLFireballSource,
    JPLSentrySource,
)
from omni_mercury_engine.data_sources.live_ingestion import LiveDataError
from omni_mercury_engine.data_sources.space_weather import NASANeoWsSource
from omni_mercury_engine.detectors.geological.disaster_detectors import (
    EarthquakeDetector,
    MeteorDetector,
    TsunamiDetector,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "live_wiring"


def load_fixture(name: str) -> Any:
    """Load a recorded API response fixture."""
    return json.loads((FIXTURES / name).read_text())


class _FakeResponse:
    """Minimal httpx.Response stand-in."""

    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def json(self) -> Any:
        return self._payload


def patch_http_get(source: Any, payload: Any, fail: bool = False) -> None:
    """Replace a source's async _http_get with a recorded-fixture stub."""

    async def _fake_http_get(endpoint: str, params: dict[str, Any] | None = None) -> _FakeResponse:
        if fail:
            raise RuntimeError("backend down")
        return _FakeResponse(payload)

    source._http_get = _fake_http_get  # type: ignore[method-assign]


def make_catalog_source() -> USGSEarthquakeSource:
    """USGS earthquake client replaying the recorded catalog response."""
    source = USGSEarthquakeSource(min_magnitude=4.5)
    patch_http_get(source, load_fixture("usgs_earthquakes.json"))
    return source


def _fixture_max_magnitude() -> float:
    payload = load_fixture("usgs_earthquakes.json")
    return max(float(f["properties"]["mag"]) for f in payload["features"])


class TestDedupeImportCompatibility:
    """The geological module re-exports the canonical flare surfaces."""

    def test_solar_flare_detector_is_the_canonical_class(self) -> None:
        from omni_mercury_engine.detectors.geological import (
            SolarFlareDetector as GeoExport,
        )
        from omni_mercury_engine.detectors.geological.disaster_detectors import (
            SolarFlareDetector as ModuleExport,
        )
        from omni_mercury_engine.space.solar_storm_detector import SolarFlareDetector

        assert GeoExport is SolarFlareDetector
        assert ModuleExport is SolarFlareDetector

    def test_solar_flare_result_is_the_canonical_dataclass(self) -> None:
        from omni_mercury_engine.detectors.geological import (
            SolarFlarePredictionResult as GeoExport,
        )
        from omni_mercury_engine.space.solar_storm_detector import (
            SolarFlarePredictionResult,
        )

        assert GeoExport is SolarFlarePredictionResult

    def test_legacy_flare_class_enum_still_importable(self) -> None:
        from omni_mercury_engine.detectors.geological import SolarFlareClass

        assert SolarFlareClass.X.value == "x_class"

    def test_nasa_dataclasses_reexported_from_disaster_detectors(self) -> None:
        from omni_mercury_engine.data_sources import jpl_ssd
        from omni_mercury_engine.detectors.geological import disaster_detectors as dd

        assert dd.FireballEvent is jpl_ssd.FireballEvent
        assert dd.CloseApproachEvent is jpl_ssd.CloseApproachEvent
        assert dd.SentryImpactRisk is jpl_ssd.SentryImpactRisk

    def test_private_loaders_are_gone(self) -> None:
        from omni_mercury_engine.detectors.geological import disaster_detectors as dd

        for name in (
            "load_nasa_fireball_data",
            "load_nasa_close_approach_data",
            "load_nasa_sentry_data",
        ):
            assert not hasattr(dd, name), f"{name} must not survive the consolidation"


class TestEarthquakeDetectorLive:
    """Event-stream assessment from the recorded USGS catalog."""

    def test_no_client_fails_loud(self) -> None:
        with pytest.raises(LiveDataError, match="no USGS earthquake client"):
            EarthquakeDetector().detect_live()

    def test_payload_without_features_fails_loud(self) -> None:
        """A payload missing the FDSN 'features' array is drift, not quiet.

        Regression: this used to return an empty success, which downstream
        reads as "no earthquakes this week".
        """
        source = USGSEarthquakeSource(min_magnitude=4.5)
        patch_http_get(source, {"metadata": {"status": 200}})
        result = source.fetch_sync(use_cache=False)
        assert result.success is False
        assert "features" in (result.error or "")
        assert result.unreachable is False  # drift, not an outage

    def test_all_features_unparseable_fails_loud(self) -> None:
        """If every feature fails to parse, refuse the empty-success lie."""
        source = USGSEarthquakeSource(min_magnitude=4.5)
        patch_http_get(
            source,
            {"features": [{"properties": {"mag": "not-a-number", "time": None}}] * 3},
        )
        result = source.fetch_sync(use_cache=False)
        assert result.success is False
        assert "schema drift" in (result.error or "")

    def test_detect_live_reports_observed_catalog_magnitude(self) -> None:
        detector = EarthquakeDetector(data_source=make_catalog_source())
        result = detector.detect_live()
        assert result.earthquake_detected is True
        # estimated_magnitude is the OBSERVED max catalog magnitude -- a real
        # USGS measurement, not a model output.
        assert result.estimated_magnitude == pytest.approx(_fixture_max_magnitude())
        assert result.magnitude_class != "undetermined"
        assert result.source_id == "usgs_earthquake"
        assert result.data_provenance == "live"
        # No waveform was consumed, so no waveform physics is claimed.
        assert result.p_wave_detected is False
        assert result.s_wave_detected is False
        assert result.resonance_score == 0.0
        # And no aftershock forecast is fabricated.
        assert result.aftershock_probability == 0.0

    def test_detect_live_rate_and_clustering_features(self) -> None:
        detector = EarthquakeDetector(data_source=make_catalog_source())
        result = detector.detect_live()
        context = result.live_context
        assert context is not None
        assert context["event_count"] == 12
        assert context["events_per_day"] > 0
        assert context["b_value"] is not None and 0.1 < context["b_value"] < 5.0
        assert 0.0 <= context["clustered_fraction"] <= 1.0
        assert context["strongest_event"]["magnitude"] == result.estimated_magnitude

    def test_detect_live_epicentral_distance_from_station(self) -> None:
        detector = EarthquakeDetector(data_source=make_catalog_source())
        result = detector.detect_live(station_lat=35.0, station_lon=140.0)
        assert result.epicenter_distance_km is not None
        assert result.epicenter_distance_km > 0

    def test_detect_live_empty_catalog_is_honest(self) -> None:
        source = USGSEarthquakeSource()
        patch_http_get(source, {"features": []})
        detector = EarthquakeDetector(data_source=source)
        result = detector.detect_live()
        assert result.earthquake_detected is False
        assert result.estimated_magnitude is None
        assert result.magnitude_class == "undetermined"
        assert result.live_context == {"event_count": 0}

    def test_detect_live_fetch_failure_raises(self) -> None:
        source = USGSEarthquakeSource()
        patch_http_get(source, None, fail=True)
        detector = EarthquakeDetector(data_source=source)
        with pytest.raises(LiveDataError, match="live fetch failed"):
            detector.detect_live()

    def test_waveform_physics_path_unchanged(self) -> None:
        """The offline STA/LTA waveform path is untouched by the wiring."""
        detector = EarthquakeDetector(data_source=make_catalog_source())
        quiet = 0.01 * np.sin(2 * np.pi * 0.5 * np.arange(2000) / 100.0)
        result = detector.predict_earthquake(quiet)
        assert result.source_id is None
        assert result.data_provenance is None


class TestTsunamiDetectorLive:
    """Waveform physics + live catalog source-event enrichment."""

    def test_no_client_fails_loud(self) -> None:
        detector = TsunamiDetector()
        with pytest.raises(LiveDataError, match="no USGS earthquake client"):
            detector.predict_tsunami_live(np.zeros(256), 38.0, 142.0)

    def test_live_source_info_feeds_arrival_physics(self) -> None:
        detector = TsunamiDetector(data_source=make_catalog_source())
        rng = np.random.default_rng(7)
        quiet_sea = 0.05 * rng.standard_normal(512)
        result = detector.predict_tsunami_live(quiet_sea, station_lat=38.0, station_lon=142.0)
        assert result.source_id == "usgs_earthquake"
        assert result.data_provenance == "live"
        context = result.live_context
        assert context is not None
        candidate = context["candidate_source_event"]
        assert candidate is not None
        assert candidate["magnitude"] == pytest.approx(_fixture_max_magnitude())
        assert candidate["distance_km"] > 0
        # source_info drives the arrival-time physics (700 km/h shallow-water
        # speed over the epicentral distance).
        assert result.source_magnitude == pytest.approx(_fixture_max_magnitude())
        assert result.source_distance_km == pytest.approx(candidate["distance_km"])
        assert result.arrival_time_minutes == pytest.approx(candidate["distance_km"] / 700.0 * 60.0)

    def test_quiet_sea_is_not_a_tsunami(self) -> None:
        detector = TsunamiDetector(data_source=make_catalog_source())
        rng = np.random.default_rng(11)
        quiet_sea = 0.05 * rng.standard_normal(512)
        result = detector.predict_tsunami_live(quiet_sea, station_lat=38.0, station_lon=142.0)
        assert result.tsunami_detected is False

    def test_offline_waveform_path_unchanged(self) -> None:
        detector = TsunamiDetector(data_source=make_catalog_source())
        rng = np.random.default_rng(13)
        record = 0.05 * rng.standard_normal(512)
        result = detector.predict_tsunami(record)
        assert result.source_id is None
        assert result.data_provenance is None


def make_meteor_detector(fireball_fail: bool = False) -> MeteorDetector:
    """MeteorDetector with all three NASA/JPL clients fixture-backed."""
    fireball = JPLFireballSource()
    patch_http_get(fireball, load_fixture("jpl_fireball.json"), fail=fireball_fail)
    neo = NASANeoWsSource()
    patch_http_get(neo, load_fixture("neows_feed.json"))
    sentry = JPLSentrySource()
    patch_http_get(sentry, load_fixture("jpl_sentry.json"))
    return MeteorDetector(
        fireball_source=fireball,
        neo_source=neo,
        sentry_source=sentry,
    )


class TestMeteorDetectorConsolidated:
    """MeteorDetector consumes ONLY data_sources clients now."""

    def test_offline_detector_makes_no_live_claims(self) -> None:
        detector = MeteorDetector(use_nasa_data=False)
        result = detector.predict_meteor()
        assert result.source_id is None
        assert result.data_provenance is None
        assert detector.get_recent_fireballs() == []
        assert detector.get_upcoming_close_approaches() == []
        assert detector.get_impact_risks() == []

    def test_predict_meteor_stamps_provenance_from_all_clients(self) -> None:
        detector = make_meteor_detector()
        result = detector.predict_meteor()
        assert result.data_provenance == "live"
        assert result.source_id is not None
        for source_id in ("jpl_fireball", "nasa_neows", "jpl_sentry"):
            assert source_id in result.source_id
        context = result.live_context
        assert context is not None
        assert "upcoming_close_approaches" in context
        assert "sentry_high_risk_objects" in context

    def test_sentry_risk_feeds_impact_probability(self) -> None:
        detector = make_meteor_detector()
        result = detector.predict_meteor()
        # Recorded fixture holds objects with cumulative Palermo > -3, whose
        # real impact probability must flow into the result.
        risks = detector.get_impact_risks()
        high = [r for r in risks if r.palermo_scale > -3]
        assert high
        assert result.impact_probability >= max(r.impact_probability for r in high)

    def test_getters_fail_loud_on_fetch_failure(self) -> None:
        detector = make_meteor_detector(fireball_fail=True)
        with pytest.raises(LiveDataError, match="live fetch failed"):
            detector.get_recent_fireballs()

    def test_predict_meteor_degrades_loudly_not_silently(self) -> None:
        """A failed corroboration feed is recorded, not silently faked."""
        detector = make_meteor_detector(fireball_fail=True)
        result = detector.predict_meteor()
        context = result.live_context
        assert context is not None
        assert "fireball_error" in context
        assert result.source_id is not None
        assert "jpl_fireball" not in result.source_id
        assert "jpl_sentry" in result.source_id

    def test_impact_risks_sorted_by_palermo(self) -> None:
        detector = make_meteor_detector()
        risks = detector.get_impact_risks()
        scales = [r.palermo_scale for r in risks]
        assert scales == sorted(scales, reverse=True)

    def test_injected_clients_imply_nasa_usage(self) -> None:
        fireball = JPLFireballSource()
        patch_http_get(fireball, load_fixture("jpl_fireball.json"))
        detector = MeteorDetector(use_nasa_data=False, fireball_source=fireball)
        assert detector.use_nasa_data is True


# =============================================================================
# Meteorological / volcanic cluster wiring (NWS alerts, NWPS gauges, USGS HANS)
# =============================================================================

from omni_mercury_engine.data_sources.earth_science import (
    NOAANWPSSource,
    NWSWeatherAlertsSource,
    USGSVolcanoSource,
)
from omni_mercury_engine.detectors.geological.flood_detector import FloodDetector
from omni_mercury_engine.detectors.geological.tornado_detector import (
    TornadoDetector,
)
from omni_mercury_engine.detectors.geological.volcanic import (
    VolcanicEruptionDetector,
)


def make_alerts_source(fixture: str) -> NWSWeatherAlertsSource:
    """NWS alerts client replaying a recorded active-alerts response."""
    source = NWSWeatherAlertsSource()
    patch_http_get(source, load_fixture(fixture))
    return source


class TestTornadoDetectorLiveWiring:
    """TornadoDetector consumes real NWS alert state, never synthesizes radar."""

    def test_offline_default_has_no_live_path(self) -> None:
        detector = TornadoDetector()
        with pytest.raises(LiveDataError, match="no NWS weather-alerts client"):
            detector.fetch_live_data()

    def test_detect_live_reports_recorded_warning_state(self) -> None:
        detector = TornadoDetector(data_source=make_alerts_source("nws_alerts_tornado.json"))
        result = detector.detect_live()
        assert result.data_provenance == "live"
        assert result.source_id is not None and result.source_id.startswith("nws_alerts")
        context = result.live_context
        assert context is not None
        assert context["tornado_warnings"] >= 1
        assert result.threat_level == "high"
        # Radar/CAPE fields must stay absent -- never fabricated from alerts.
        assert result.rotation_velocity_ms == 0.0
        assert result.cape_value == 0.0

    def test_detect_live_confidence_follows_cap_certainty(self) -> None:
        payload = load_fixture("nws_alerts_tornado.json")
        certainties = {
            str(f["properties"].get("certainty", "Unknown")).lower()
            for f in payload["features"]
            if str(f["properties"].get("event", "")).lower() == "tornado warning"
        }
        detector = TornadoDetector(data_source=make_alerts_source("nws_alerts_tornado.json"))
        result = detector.detect_live()
        expected = max(
            {"observed": 0.95, "likely": 0.75, "possible": 0.45, "unlikely": 0.1, "unknown": 0.3}[c]
            for c in certainties
        )
        assert result.confidence == pytest.approx(expected)

    def test_quiet_alert_feed_reports_none(self) -> None:
        source = NWSWeatherAlertsSource()
        patch_http_get(source, {"features": []})
        detector = TornadoDetector(data_source=source)
        result = detector.detect_live()
        assert result.tornado_likely is False
        assert result.threat_level == "none"
        assert result.confidence == 0.0

    def test_failed_fetch_is_loud(self) -> None:
        source = NWSWeatherAlertsSource()
        patch_http_get(source, None, fail=True)
        detector = TornadoDetector(data_source=source)
        with pytest.raises(LiveDataError, match="live fetch failed"):
            detector.detect_live()


class TestFloodDetectorLiveWiring:
    """FloodDetector consumes real gauge stages + flood alerts."""

    def test_offline_default_has_no_live_path(self) -> None:
        detector = FloodDetector()
        with pytest.raises(LiveDataError, match="no gauge_source or alerts_source"):
            detector.detect_live()

    def test_detect_live_reports_worst_gauge_category(self) -> None:
        gauges = NOAANWPSSource()
        patch_http_get(gauges, load_fixture("nwps_gauges.json"))
        detector = FloodDetector(gauge_source=gauges)
        result = detector.detect_live()
        assert result.data_provenance == "live"
        context = result.live_context
        assert context is not None
        assert context["gauge_count"] >= 1
        # Severity vocabulary matches _determine_severity's.
        assert result.severity in ("no_flood", "minor", "moderate", "major", "record")
        # Stage fields are real gauge readings or absent, never invented.
        if context.get("worst_gauge") is not None and result.river_stage_ft:
            assert result.river_stage_ft > 0.0

    def test_alert_corroboration_without_gauges(self) -> None:
        detector = FloodDetector(alerts_source=make_alerts_source("nws_alerts_flood.json"))
        result = detector.detect_live()
        context = result.live_context
        assert context is not None
        assert context["flood_alerts"] >= 1
        assert result.flood_likely is True
        assert result.confidence == pytest.approx(0.7)

    def test_dead_runoff_network_is_gone(self) -> None:
        """The untrained TopographicRunoffPredictor surface was removed."""
        import omni_mercury_engine.detectors.geological.flood_detector as fd

        assert not hasattr(fd, "TopographicRunoffPredictor")
        assert not hasattr(FloodDetector(), "runoff_predictor")


class TestVolcanicDetectorLiveWiring:
    """VolcanicEruptionDetector reports the real USGS HANS alert state."""

    def test_offline_default_has_no_live_path(self) -> None:
        detector = VolcanicEruptionDetector()
        with pytest.raises(LiveDataError, match="no USGS HANS volcano client"):
            detector.fetch_live_data()

    def test_detect_live_reports_highest_official_alert(self) -> None:
        source = USGSVolcanoSource()
        patch_http_get(source, load_fixture("hans_elevated.json"))
        detector = VolcanicEruptionDetector(data_source=source)
        result = detector.detect_live(elevated_only=True)
        assert result.data_provenance == "live"
        context = result.live_context
        assert context is not None
        assert context["volcanoes_reported"] >= 1
        assert result.alert_level in ("advisory", "watch", "warning")
        assert result.confidence == pytest.approx(0.98)
        # Instrument fields stay absent -- alerts are never turned into
        # synthetic seismic/gas/thermal measurements.
        assert result.seismic_swarm_detected is False
        assert result.vei_estimate is None

    def test_named_volcano_filter(self) -> None:
        payload = load_fixture("hans_elevated.json")
        name = payload[0]["volcano_name"]
        source = USGSVolcanoSource()
        patch_http_get(source, payload)
        detector = VolcanicEruptionDetector(data_source=source)
        result = detector.detect_live(volcano_name=name, elevated_only=True)
        context = result.live_context
        assert context is not None
        assert context["volcanoes_reported"] >= 1
        assert context["highest_alert_volcano"]["name"] == name

    def test_empty_feed_raises_instead_of_reporting_normal(self) -> None:
        """No data must never become an all-clear.

        Regression: an empty HANS feed used to yield ``alert_level="normal"``
        with confidence 0.0 — a fabricated all-clear (the real HANS
        monitored-volcano list is never empty, so an empty feed means drift
        or an outage, not calm volcanoes).
        """
        source = USGSVolcanoSource()
        patch_http_get(source, [])
        detector = VolcanicEruptionDetector(data_source=source)
        with pytest.raises(LiveDataError, match="refusing to fabricate"):
            detector.detect_live()

    def test_unmatched_volcano_name_raises_instead_of_reporting_normal(self) -> None:
        """Asking about a volcano absent from the feed must fail loud."""
        source = USGSVolcanoSource()
        patch_http_get(source, load_fixture("hans_elevated.json"))
        detector = VolcanicEruptionDetector(data_source=source)
        with pytest.raises(LiveDataError, match="no volcano named"):
            detector.detect_live(volcano_name="Definitely Not A Volcano")
