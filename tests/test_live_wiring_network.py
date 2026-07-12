# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Live network smoke tests for the hazard live-ingestion wiring (T1e).

Each REAL data-source client behind the uniform live-ingestion seam gets a
:func:`pytest.mark.network` smoke test that exercises the actual HTTP path
end-to-end, plus one ``detect_live`` round-trip per wired detector cluster.

Two outcomes count as a pass: a successful fetch (asserting non-empty,
well-formed data points) or a **loud** failure
(:class:`~omni_mercury_engine.data_sources.base.DataSourceError` /
:class:`~omni_mercury_engine.data_sources.live_ingestion.LiveDataError`) when
the upstream service is down. A silent empty success for a service that is
documented to always return data is a failure — that is exactly the kind of
silent bitrot this module exists to catch.

Skipped unless ``MERCURY_NETWORK_TESTS=1`` (weekly ``network-tests.yml`` lane
and on-demand local runs)::

    MERCURY_NETWORK_TESTS=1 pytest tests/test_live_wiring_network.py -m network
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from omni_mercury_engine.data_sources.base import DataSourceError, SourceUnreachableError
from omni_mercury_engine.data_sources.earth_science import (
    NOAANWPSSource,
    NWSWeatherAlertsSource,
    USGSEarthquakeSource,
    USGSVolcanoSource,
)
from omni_mercury_engine.data_sources.jpl_ssd import JPLFireballSource, JPLSentrySource
from omni_mercury_engine.data_sources.live_ingestion import LiveDataError
from omni_mercury_engine.data_sources.space_weather import NASANeoWsSource, NOAASWPCSource

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        os.environ.get("MERCURY_NETWORK_TESTS") != "1",
        reason="live network smoke; set MERCURY_NETWORK_TESTS=1 to run",
    ),
]

_LOUD_FAILURES = (DataSourceError, LiveDataError)


def _fetch_or_loud(source: Any) -> Any:
    """Fetch synchronously; skip ONLY on genuine unreachability.

    A transport-level failure (service down, DNS, timeout, throttle) is the
    one acceptable skip. Any other error from a *reachable* service --
    unexpected payload, contract drift, parse failure -- FAILS the test:
    masking drift as a skip is exactly the silent bitrot this lane exists
    to catch.
    """
    result = source.fetch_sync()
    if not result.success:
        if result.unreachable:
            pytest.skip(f"{source.source_id}: upstream unreachable ({result.error})")
        pytest.fail(
            f"{source.source_id}: service responded but the fetch failed -- "
            f"probable schema/endpoint drift, not unavailability: {result.error}"
        )
    return result


def _skip_only_if_unreachable(exc: BaseException) -> None:
    """Skip for transport-level unavailability; re-raise anything else."""
    if getattr(exc, "unreachable", False):
        pytest.skip(f"upstream unreachable: {exc}")
    if isinstance(exc, DataSourceError) and not isinstance(exc, SourceUnreachableError):
        pytest.fail(f"reachable service returned an error (drift, not outage): {exc}")
    if isinstance(exc, SourceUnreachableError):
        pytest.skip(f"upstream unreachable: {exc}")
    pytest.fail(f"live round-trip failed loudly (not an outage): {exc}")


class TestRealSourcesLive:
    """Every REAL client fetches genuine data through the proxy."""

    def test_usgs_earthquake_catalog(self) -> None:
        result = _fetch_or_loud(USGSEarthquakeSource(min_magnitude=2.5, days_back=7))
        assert result.data_points, "USGS FDSN always has M2.5+ events in a 7-day window"
        assert all(dp.data.get("magnitude") is not None for dp in result.data_points)

    def test_usgs_volcano_hans(self) -> None:
        result = _fetch_or_loud(USGSVolcanoSource())
        assert result.data_points, "HANS monitored-volcano list is never empty"
        levels = {str(dp.data.get("alert_level")) for dp in result.data_points}
        assert levels & {"normal", "advisory", "watch", "warning", "unassigned"}

    def test_nwps_river_gauges(self) -> None:
        # The /gauges endpoint requires a bounding box (documented on the
        # client); without one it returns an empty list even when healthy,
        # which made this smoke unable to pass. Houston-area box from the
        # client's own docstring example.
        result = _fetch_or_loud(NOAANWPSSource(bbox=(-96.0, 28.0, -93.0, 31.0)))
        assert result.data_points

    def test_nws_active_alerts(self) -> None:
        # An empty alert list is legitimate (quiet weather) — shape only.
        result = _fetch_or_loud(NWSWeatherAlertsSource())
        for dp in result.data_points[:5]:
            assert dp.data.get("event")

    def test_swpc_products(self) -> None:
        result = _fetch_or_loud(NOAASWPCSource())
        assert result.data_points, "SWPC Kp/solar-wind products always report"

    def test_nasa_neows(self) -> None:
        result = _fetch_or_loud(NASANeoWsSource())
        assert result.data_points, "NeoWs 7-day feed always has close approaches"

    def test_jpl_fireball(self) -> None:
        # CNEOS publishes the fireball table in lagging batches, so a short
        # recent window can be legitimately empty on a perfectly healthy
        # upstream -- the default 30-day window therefore spuriously red-X'd
        # the weekly lane. Query a decade so "never empty" is a genuine
        # invariant: a real outage still skips via result.unreachable inside
        # _fetch_or_loud, schema drift is raised as DataSourceError by the
        # source (and fails loudly there), and an empty, reachable 200-OK over
        # ten years is real breakage this assert is right to catch.
        result = _fetch_or_loud(JPLFireballSource(days_back=3650))
        assert result.data_points, (
            "CNEOS fireball archive over a 10-year window is never empty; an "
            "empty but reachable response indicates upstream/schema breakage"
        )

    def test_jpl_sentry(self) -> None:
        result = _fetch_or_loud(JPLSentrySource())
        assert result.data_points, "Sentry risk table is never empty"


class TestDetectorsLiveRoundTrip:
    """One end-to-end detect_live per wired cluster against the real feeds."""

    def test_earthquake_detect_live(self) -> None:
        pytest.importorskip("torch")
        from omni_mercury_engine.detectors.geological.disaster_detectors import (
            EarthquakeDetector,
        )

        detector = EarthquakeDetector(data_source=USGSEarthquakeSource(min_magnitude=4.0))
        try:
            result = detector.detect_live()
        except _LOUD_FAILURES as exc:
            _skip_only_if_unreachable(exc)
        assert result.data_provenance == "live"
        assert result.live_context is not None

    def test_tornado_detect_live(self) -> None:
        pytest.importorskip("torch")
        from omni_mercury_engine.detectors.geological.tornado_detector import TornadoDetector

        detector = TornadoDetector(data_source=NWSWeatherAlertsSource())
        try:
            result = detector.detect_live()
        except _LOUD_FAILURES as exc:
            _skip_only_if_unreachable(exc)
        assert result.data_provenance == "live"
        assert result.threat_level in ("none", "marginal", "slight", "moderate", "high")

    def test_flood_detect_live(self) -> None:
        pytest.importorskip("torch")
        from omni_mercury_engine.detectors.geological.flood_detector import FloodDetector

        detector = FloodDetector(gauge_source=NOAANWPSSource())
        try:
            result = detector.detect_live()
        except _LOUD_FAILURES as exc:
            _skip_only_if_unreachable(exc)
        assert result.data_provenance == "live"
        assert result.severity in ("no_flood", "minor", "moderate", "major", "record")

    def test_volcano_detect_live(self) -> None:
        pytest.importorskip("torch")
        from omni_mercury_engine.detectors.geological.volcanic import VolcanicEruptionDetector

        detector = VolcanicEruptionDetector(data_source=USGSVolcanoSource())
        try:
            result = detector.detect_live()
        except _LOUD_FAILURES as exc:
            _skip_only_if_unreachable(exc)
        assert result.data_provenance == "live"
        assert result.live_context is not None
        assert result.live_context["volcanoes_reported"] >= 1

    def test_meteor_predict_live(self) -> None:
        pytest.importorskip("torch")
        from omni_mercury_engine.detectors.geological.disaster_detectors import MeteorDetector

        detector = MeteorDetector(use_nasa_data=True)
        try:
            result = detector.predict_meteor()
        except _LOUD_FAILURES as exc:
            _skip_only_if_unreachable(exc)
        assert result.data_provenance in ("live", None)
