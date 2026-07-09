# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Offline-deterministic tests for the live-ingestion seam and its data sources.

Covers:

- the :mod:`omni_mercury_engine.data_sources.live_ingestion` seam (fail-loud,
  simulated-source gate, provenance resolution);
- the honest :class:`BGSELFStationSource` (explicit simulation labelling +
  real Welch DSP on caller-supplied instrument records);
- the real-API :class:`USGSVolcanoSource` (HANS) parser;
- the fixed :class:`NOAASWPCSource` parsers (dict-row Kp / X-ray products and
  the propagated solar-wind product);
- the fixed :class:`NOAANWPSSource` parser (real NWPS v1 gauge shape);
- the new :class:`JPLFireballSource` / :class:`JPLSentrySource` clients.

All HTTP is replaced by recorded fixtures under
``tests/fixtures/live_wiring/`` (captured from the real APIs on 2026-07-09);
no test in this module touches the network. Live reachability smoke tests
live in ``tests/test_live_wiring_network.py`` (``@pytest.mark.network``).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.data_sources.base import (
    AlertLevel,
    DataPoint,
    DataSourceBase,
    DataSourceType,
)
from omni_mercury_engine.data_sources.earth_science import (
    NOAANWPSSource,
    USGSVolcanoSource,
)
from omni_mercury_engine.data_sources.geomagnetic import BGSELFStationSource
from omni_mercury_engine.data_sources.jpl_ssd import (
    JPLFireballSource,
    JPLSentrySource,
    close_approaches_from_neows_datapoints,
    fireball_events_from_datapoints,
    sentry_risks_from_datapoints,
)
from omni_mercury_engine.data_sources.live_ingestion import (
    LiveDataError,
    SimulatedDataError,
    fetch_live_datapoints,
    haversine_km,
)
from omni_mercury_engine.data_sources.space_weather import (
    NASANeoWsSource,
    NOAASWPCSource,
    SWPCProduct,
)

FIXTURES = Path(__file__).parent / "fixtures" / "live_wiring"


def load_fixture(name: str) -> Any:
    """Load a recorded API response fixture."""
    return json.loads((FIXTURES / name).read_text())


class _FakeResponse:
    """Minimal httpx.Response stand-in for parser tests."""

    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def json(self) -> Any:
        return self._payload


def patch_http_get(source: DataSourceBase, payload: Any) -> None:
    """Replace a source's async _http_get with a recorded-fixture stub."""

    async def _fake_http_get(endpoint: str, params: dict[str, Any] | None = None) -> _FakeResponse:
        return _FakeResponse(payload)

    source._http_get = _fake_http_get  # type: ignore[method-assign]


class _InMemorySource(DataSourceBase):
    """Tiny in-memory source for exercising the seam without HTTP."""

    def __init__(self, points: list[DataPoint], fail: bool = False) -> None:
        super().__init__()
        self._points = points
        self._fail = fail

    @property
    def source_id(self) -> str:
        return "in_memory"

    @property
    def default_source_types(self) -> list[DataSourceType]:
        return [DataSourceType.CUSTOM]

    async def _fetch_impl(self, start_time=None, end_time=None, **kwargs):  # type: ignore[no-untyped-def]
        if self._fail:
            raise RuntimeError("backend down")
        return self._points


def _make_point(
    simulated: bool = False, source_type: DataSourceType = DataSourceType.CUSTOM
) -> DataPoint:
    return DataPoint(
        source_id="in_memory",
        source_type=source_type,
        event_id="e1",
        timestamp=datetime.now(UTC),
        data={"value": 1.0},
        metadata={"simulated": simulated} if simulated else {},
    )


class TestLiveIngestionSeam:
    """The uniform fetch seam: fail-loud, simulated gate, provenance."""

    def test_live_provenance_for_real_points(self) -> None:
        fetch = fetch_live_datapoints(_InMemorySource([_make_point()]), use_cache=False)
        assert fetch.data_provenance == "live"
        assert fetch.source_id == "in_memory"
        assert len(fetch.data_points) == 1

    def test_simulated_source_refused_without_opt_in(self) -> None:
        source = _InMemorySource([_make_point(simulated=True)])
        with pytest.raises(SimulatedDataError, match="SIMULATED"):
            fetch_live_datapoints(source, use_cache=False)

    def test_simulated_source_allowed_with_explicit_opt_in(self) -> None:
        source = _InMemorySource([_make_point(simulated=True)])
        fetch = fetch_live_datapoints(source, allow_simulated=True, use_cache=False)
        assert fetch.data_provenance == "simulated"

    def test_fetch_failure_raises_instead_of_returning_empty(self) -> None:
        source = _InMemorySource([], fail=True)
        with pytest.raises(LiveDataError, match="live fetch failed"):
            fetch_live_datapoints(source, use_cache=False)

    def test_source_type_filter(self) -> None:
        points = [
            _make_point(source_type=DataSourceType.CUSTOM),
            _make_point(source_type=DataSourceType.EARTHQUAKE),
        ]
        fetch = fetch_live_datapoints(
            _InMemorySource(points),
            source_types=[DataSourceType.EARTHQUAKE],
            use_cache=False,
        )
        assert [dp.source_type for dp in fetch.data_points] == [DataSourceType.EARTHQUAKE]

    def test_haversine_known_distance(self) -> None:
        # London -> Paris is ~343 km great-circle.
        d = haversine_km(51.5074, -0.1278, 48.8566, 2.3522)
        assert 330.0 < d < 355.0


class TestBGSELFStationHonesty:
    """BGS ELF: explicit simulation + real Welch DSP on instrument records."""

    def test_simulated_fetch_is_labelled(self) -> None:
        source = BGSELFStationSource()
        result = source.fetch_sync(use_cache=False)
        assert result.success
        point = result.data_points[0]
        assert point.metadata["simulated"] is True
        assert point.metadata["data_provenance"] == "simulated"
        assert point.confidence == 0.0

    def test_seam_refuses_simulated_without_opt_in(self) -> None:
        with pytest.raises(SimulatedDataError):
            fetch_live_datapoints(BGSELFStationSource(), use_cache=False)

    def test_instrument_record_runs_real_welch_dsp(self) -> None:
        """A pure 7.83 Hz tone must dominate SR1 -- proof the DSP is real."""
        source = BGSELFStationSource()
        fs = 100.0
        t = np.arange(4096) / fs
        record = np.sin(2 * np.pi * 7.83 * t)
        fetch = fetch_live_datapoints(
            source, raw_samples=record, sampling_rate_hz=fs, use_cache=False
        )
        assert fetch.data_provenance == "live"
        point = fetch.data_points[0]
        assert point.metadata["simulated"] is False
        assert point.metadata["data_provenance"] == "instrument"
        powers = point.data["power_spectrum"]
        assert powers["SR1"] > 10 * max(powers[k] for k in powers if k != "SR1")

    def test_simulated_record_is_deterministic(self) -> None:
        source = BGSELFStationSource()
        a = source.fetch_sync(use_cache=False).data_points[0].data["power_spectrum"]
        b = source.fetch_sync(use_cache=False).data_points[0].data["power_spectrum"]
        assert a == b

    def test_short_instrument_record_fails_loud(self) -> None:
        source = BGSELFStationSource()
        with pytest.raises(LiveDataError):
            fetch_live_datapoints(
                source, raw_samples=np.zeros(16), sampling_rate_hz=100.0, use_cache=False
            )

    def test_nonfinite_instrument_record_fails_loud(self) -> None:
        source = BGSELFStationSource()
        record = np.full(512, np.nan)
        with pytest.raises(LiveDataError):
            fetch_live_datapoints(
                source, raw_samples=record, sampling_rate_hz=100.0, use_cache=False
            )


class TestUSGSVolcanoHANS:
    """USGS Volcano source against recorded HANS payloads."""

    def test_parse_monitored_fixture(self) -> None:
        source = USGSVolcanoSource()
        patch_http_get(source, load_fixture("hans_monitored.json"))
        result = source.fetch_sync(use_cache=False)
        assert result.success
        points = result.data_points
        assert len(points) == 10
        by_name = {p.data["name"]: p for p in points}
        assert by_name["Great Sitkin"].data["alert_level"] == "watch"
        assert by_name["Great Sitkin"].data["aviation_color_code"] == "orange"
        assert by_name["Great Sitkin"].alert_level == AlertLevel.STRONG
        assert by_name["Akutan"].alert_level == AlertLevel.NONE
        # Real feed: nothing is labelled simulated.
        assert all(not p.metadata.get("simulated", False) for p in points)

    def test_parse_elevated_fixture(self) -> None:
        source = USGSVolcanoSource()
        patch_http_get(source, load_fixture("hans_elevated.json"))
        result = source.fetch_sync(use_cache=False, elevated_only=True)
        assert result.success
        assert len(result.data_points) == 4
        assert all(p.alert_level != AlertLevel.NONE for p in result.data_points)

    def test_non_list_payload_fails_loud(self) -> None:
        source = USGSVolcanoSource()
        patch_http_get(source, {"error": "Did not find volcano/getFoo"})
        result = source.fetch_sync(use_cache=False)
        assert not result.success
        assert "unexpected HANS payload" in (result.error or "")


class TestNOAASWPCParsers:
    """SWPC parsers against the current (2026-07) product serialisations."""

    def test_kp_index_dict_rows(self) -> None:
        source = NOAASWPCSource()
        points = source._parse_product_data(SWPCProduct.KP_INDEX, load_fixture("swpc_kp.json"))
        assert len(points) == 57
        latest = max(points, key=lambda p: p.timestamp)
        assert latest.data["kp_index"] == pytest.approx(2.67)
        assert latest.alert_level == AlertLevel.NONE

    def test_kp_index_legacy_list_rows(self) -> None:
        source = NOAASWPCSource()
        legacy = [
            ["time_tag", "Kp", "estimated_kp", "a_running", "station_count"],
            ["2026-07-08T00:00:00", "6.33", "6.2", "45", "8"],
        ]
        points = source._parse_product_data(SWPCProduct.KP_INDEX, legacy)
        assert len(points) == 1
        assert points[0].data["kp_index"] == pytest.approx(6.33)
        assert points[0].alert_level == AlertLevel.MODERATE  # G2

    def test_xray_dict_rows_pair_channels(self) -> None:
        source = NOAASWPCSource()
        points = source._parse_product_data(SWPCProduct.XRAY_FLUX, load_fixture("swpc_xray.json"))
        assert points, "recorded X-ray fixture must yield data points"
        for point in points:
            assert point.data["long_flux"] > 0
            assert point.data["short_flux"] >= 0
            # Long channel is the flare-classification channel.
            assert point.data["long_flux"] != point.data["short_flux"]

    def test_propagated_solar_wind(self) -> None:
        source = NOAASWPCSource()
        points = source._parse_product_data(
            SWPCProduct.PROPAGATED_SOLAR_WIND, load_fixture("swpc_solar_wind.json")
        )
        assert len(points) == 24
        latest = max(points, key=lambda p: p.timestamp)
        assert latest.data["speed"] and 200 < latest.data["speed"] < 1200
        assert latest.data["bz"] is not None
        assert latest.data["by"] is not None

    def test_default_products_exclude_dead_dscovr_paths(self) -> None:
        source = NOAASWPCSource()
        assert SWPCProduct.PROPAGATED_SOLAR_WIND in source._products
        assert SWPCProduct.SOLAR_WIND_PLASMA not in source._products


class TestNOAANWPSRealShape:
    """NWPS parser against the real /gauges v1 payload."""

    def test_parse_real_gauge_shape(self) -> None:
        source = NOAANWPSSource(bbox=(-96.0, 28.0, -93.0, 31.0))
        patch_http_get(source, load_fixture("nwps_gauges.json"))
        result = source.fetch_sync(use_cache=False)
        assert result.success
        assert result.data_points
        point = result.data_points[0]
        assert point.data["gauge_id"]
        assert point.data["observed_value"] is not None
        assert point.data["flood_category"] is not None
        assert point.location is not None

    def test_no_flooding_category_maps_to_none(self) -> None:
        source = NOAANWPSSource()
        assert source._flood_category_to_alert("no_flooding") == AlertLevel.NONE
        assert source._flood_category_to_alert("action") == AlertLevel.MINOR
        assert source._flood_category_to_alert("minor") == AlertLevel.MODERATE
        assert source._flood_category_to_alert("moderate") == AlertLevel.STRONG
        assert source._flood_category_to_alert("major") == AlertLevel.SEVERE

    def test_missing_sentinel_becomes_none(self) -> None:
        assert NOAANWPSSource._numeric_or_none(-999) is None
        assert NOAANWPSSource._numeric_or_none(None) is None
        assert NOAANWPSSource._numeric_or_none(72.68) == pytest.approx(72.68)


class TestJPLFireballSource:
    """JPL Fireball client against a recorded response."""

    def test_parse_recorded_fireballs(self) -> None:
        source = JPLFireballSource()
        patch_http_get(source, load_fixture("jpl_fireball.json"))
        result = source.fetch_sync(use_cache=False)
        assert result.success
        points = result.data_points
        assert len(points) == 20
        assert all(p.source_type == DataSourceType.NEAR_EARTH_OBJECT for p in points)
        assert all(p.data["impact_energy_kt"] is not None for p in points)

    def test_size_estimate_matches_brown_2002_relation(self) -> None:
        source = JPLFireballSource()
        patch_http_get(source, load_fixture("jpl_fireball.json"))
        points = source.fetch_sync(use_cache=False).data_points
        events = fireball_events_from_datapoints(points)
        event = events[0]
        assert event.calculated_total_impact_energy_kt is not None
        expected = (event.calculated_total_impact_energy_kt * 4.184e12 / 4.185e10) ** (1 / 3)
        assert event.estimated_size_m == pytest.approx(expected)

    def test_energy_floor_filter(self) -> None:
        source = JPLFireballSource(min_energy_kt=1.0)
        patch_http_get(source, load_fixture("jpl_fireball.json"))
        points = source.fetch_sync(use_cache=False).data_points
        assert all(p.data["impact_energy_kt"] >= 1.0 for p in points)


class TestJPLSentrySource:
    """JPL Sentry client against a recorded response."""

    def test_parse_recorded_summary(self) -> None:
        source = JPLSentrySource()
        patch_http_get(source, load_fixture("jpl_sentry.json"))
        result = source.fetch_sync(use_cache=False)
        assert result.success
        points = result.data_points
        assert points
        # Palermo scale comes from the real summary-mode field ps_cum -- the
        # legacy loader read a nonexistent "ps" key and always fell back to -10.
        assert any(p.data["palermo_scale_cumulative"] > -5 for p in points)

    def test_min_palermo_filter(self) -> None:
        source = JPLSentrySource()
        patch_http_get(source, load_fixture("jpl_sentry.json"))
        points = source.fetch_sync(use_cache=False, min_palermo=-2.0).data_points
        assert points
        assert all(p.data["palermo_scale_cumulative"] >= -2.0 for p in points)

    def test_risk_records_sortable_by_palermo(self) -> None:
        source = JPLSentrySource()
        patch_http_get(source, load_fixture("jpl_sentry.json"))
        risks = sentry_risks_from_datapoints(source.fetch_sync(use_cache=False).data_points)
        assert risks
        ordered = sorted(risks, key=lambda r: r.palermo_scale, reverse=True)
        assert ordered[0].palermo_scale >= ordered[-1].palermo_scale


class TestNeoWsCloseApproachConversion:
    """NeoWs data points convert into CloseApproachEvent records."""

    def test_convert_recorded_feed(self) -> None:
        source = NASANeoWsSource()
        patch_http_get(source, load_fixture("neows_feed.json"))
        start = datetime(2026, 7, 8, tzinfo=UTC)
        end = datetime(2026, 7, 9, tzinfo=UTC)
        result = source.fetch_sync(start_time=start, end_time=end, use_cache=False)
        assert result.success
        events = close_approaches_from_neows_datapoints(result.data_points)
        assert events
        for event in events:
            assert event.nominal_distance_km > 0
            assert event.nominal_distance_au == pytest.approx(
                event.nominal_distance_km / 149597870.7, rel=0.01
            )
            assert event.relative_velocity_km_s > 0
