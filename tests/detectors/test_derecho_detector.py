# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Derecho detector: Johns & Hirt (1987) / Corfidi et al. (2016) criteria.

Canonical regression case: the historic 29 June 2012 Ohio Valley /
mid-Atlantic derecho, evaluated over the **recorded real** SPC filtered
storm reports for that convective day
(``tests/fixtures/meteorological/spc_storm_reports_20120629.csv``;
provenance in PROVENANCE.json).  The report chain is selected from the
unmodified fixture using the corridor and timing published in the NOAA/NWS
Service Assessment "The Historic Derecho of June 29, 2012" (states
IA/IL/IN/MI/OH/KY/WV/VA/MD/DC/DE/NJ/PA/NC, 16:00 UTC Jun 29 through
07:00 UTC Jun 30); the selection is test logic, the reports themselves are
real and unaltered (constructed-from-published-event chain).

Geometry sanity checks are verified against direct haversine values.
Constructed unit inputs (clearly marked) exercise the individual criterion
branches that the historical positive case cannot falsify.
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

# The dev venv's editable install may point at a sibling worktree that
# predates ``derecho_detector``; ``unused-ignore`` keeps a correctly
# installed tree (CI) clean.
from omni_mercury_engine.detectors.meteorological.derecho_detector import (  # type: ignore[import-not-found,unused-ignore]
    DerechoDetector,
    WindReport,
)
from omni_mercury_engine.utils.geo import haversine_km

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "meteorological"

_MPH_TO_MS = 0.44704

#: Corridor of the June 2012 derecho per the NWS Service Assessment.
_CORRIDOR_STATES = frozenset(
    {"IA", "IL", "IN", "MI", "OH", "KY", "WV", "VA", "MD", "DC", "DE", "NJ", "PA", "NC"}
)
_WINDOW_START = dt.datetime(2012, 6, 29, 16, 0, tzinfo=dt.UTC)
_WINDOW_END = dt.datetime(2012, 6, 30, 7, 0, tzinfo=dt.UTC)


def _load_june2012_wind_reports() -> list[dict[str, Any]]:
    """Parse the recorded SPC daily-report CSV into the derecho wind chain.

    The SPC file concatenates tornado / wind / hail sections, each headed
    by ``Time,...``; wind rows are ``Time,Speed,Location,County,State,Lat,
    Lon,Comments`` with Speed in mph or ``UNK`` and times HHMM UTC (values
    before 1200 belong to the following calendar day).
    """
    text = (FIXTURE_DIR / "spc_storm_reports_20120629.csv").read_text()
    base = dt.datetime(2012, 6, 29, tzinfo=dt.UTC)
    section = None
    reports: list[dict[str, Any]] = []
    for line in text.strip().split("\n"):
        if line.startswith("Time,"):
            section = "wind" if ",Speed," in line else "other"
            continue
        if section != "wind":
            continue
        row = next(csv.reader([line]))
        time_raw, speed, state = row[0], row[1], row[4]
        lat, lon = float(row[5]), float(row[6])
        hour, minute = int(time_raw[:2]), int(time_raw[2:])
        day = base if hour >= 12 else base + dt.timedelta(days=1)
        stamp = day.replace(hour=hour, minute=minute)
        if state not in _CORRIDOR_STATES or not _WINDOW_START <= stamp <= _WINDOW_END:
            continue
        gust_ms = None if speed in ("UNK", "") else float(speed) * _MPH_TO_MS
        reports.append({"time_s": stamp.timestamp(), "lat": lat, "lon": lon, "gust_ms": gust_ms})
    return reports


@pytest.fixture(scope="module")
def detector() -> DerechoDetector:
    """One detector instance for the module (stateless)."""
    return DerechoDetector()


@pytest.fixture(scope="module")
def june2012_reports() -> list[dict[str, Any]]:
    """The real June 29, 2012 derecho wind-report chain."""
    reports = _load_june2012_wind_reports()
    assert len(reports) > 500, "fixture parse should yield the full corridor chain"
    return reports


class TestJune2012Derecho:
    """The canonical progressive derecho must satisfy every criterion."""

    def test_classified_as_progressive_derecho(
        self, detector: DerechoDetector, june2012_reports: list[dict[str, Any]]
    ) -> None:
        result = detector.evaluate(june2012_reports)
        assert result.is_derecho
        assert result.classification == "progressive"
        assert all(result.criteria.values()), result.criteria

    def test_swath_dimensions_match_published_scale(
        self, detector: DerechoDetector, june2012_reports: list[dict[str, Any]]
    ) -> None:
        """Published: ~700 mi (1130 km) swath crossed in ~12 h."""
        result = detector.evaluate(june2012_reports)
        assert result.geometry.length_km > 1000.0
        assert result.geometry.width_km >= 100.0
        assert 10.0 <= result.geometry.duration_h <= 16.0
        # West-to-east track: axis bearing roughly eastward.
        assert 60.0 <= result.geometry.axis_bearing_deg <= 140.0

    def test_intensity_anchors_from_measured_gusts(
        self, detector: DerechoDetector, june2012_reports: list[dict[str, Any]]
    ) -> None:
        """The chain carries dozens of measured gusts >= 33 m/s (74 mph),
        including the published 91 mph Fort Wayne IN gust, mutually
        separated far beyond 64 km."""
        result = detector.evaluate(june2012_reports)
        assert result.n_significant >= 30
        assert result.n_significant_separated >= 3
        gusts = {r["gust_ms"] for r in june2012_reports if r["gust_ms"] is not None}
        assert 91.0 * _MPH_TO_MS in gusts  # Fort Wayne, IN (1854 UTC)
        assert max(gusts) <= 45.0  # sanity: strongest report that day was 93 mph

    def test_continuity_and_progression(
        self, detector: DerechoDetector, june2012_reports: list[dict[str, Any]]
    ) -> None:
        result = detector.evaluate(june2012_reports)
        assert result.max_report_gap_h <= 3.0
        assert result.progression_correlation >= 0.6

    def test_short_segment_is_not_a_derecho(self, detector: DerechoDetector) -> None:
        """A single-state slice of the same real event (Ohio, 19:00-21:30
        UTC) fails the 650 km length criterion: the criteria distinguish a
        derecho from an ordinary severe MCS window.  (The full corridor
        window cannot be used for this negative control because concurrent
        separate storms in Iowa that evening stretch its extent.)"""
        text = (FIXTURE_DIR / "spc_storm_reports_20120629.csv").read_text()
        base = dt.datetime(2012, 6, 29, tzinfo=dt.UTC)
        section = None
        segment: list[dict[str, Any]] = []
        for line in text.strip().split("\n"):
            if line.startswith("Time,"):
                section = "wind" if ",Speed," in line else "other"
                continue
            if section != "wind":
                continue
            row = next(csv.reader([line]))
            if row[4] != "OH":
                continue
            hour, minute = int(row[0][:2]), int(row[0][2:])
            if hour < 12:
                continue
            stamp = base.replace(hour=hour, minute=minute)
            if not (
                dt.datetime(2012, 6, 29, 19, 0, tzinfo=dt.UTC)
                <= stamp
                <= dt.datetime(2012, 6, 29, 21, 30, tzinfo=dt.UTC)
            ):
                continue
            gust = None if row[1] in ("UNK", "") else float(row[1]) * _MPH_TO_MS
            segment.append(
                {
                    "time_s": stamp.timestamp(),
                    "lat": float(row[5]),
                    "lon": float(row[6]),
                    "gust_ms": gust,
                }
            )
        assert len(segment) > 50
        result = detector.evaluate(segment)
        assert not result.criteria["length"]
        assert not result.is_derecho
        assert result.classification == "none"


class TestGeometry:
    """Great-circle swath geometry against direct haversine values."""

    def test_two_point_axis_length(self, detector: DerechoDetector) -> None:
        reports = [
            WindReport(time_s=0.0, lat=40.0, lon=-90.0, gust_ms=35.0),
            WindReport(time_s=3600.0, lat=41.0, lon=-90.0, gust_ms=35.0),
        ]
        geometry = detector.compute_swath_geometry(reports)
        assert geometry.length_km == pytest.approx(haversine_km(40.0, -90.0, 41.0, -90.0), rel=1e-9)
        assert geometry.width_km == pytest.approx(0.0, abs=1e-6)
        # Earlier report is the axis start endpoint.
        assert geometry.axis_start == (40.0, -90.0)
        assert geometry.axis_end == (41.0, -90.0)

    def test_cross_track_width(self, detector: DerechoDetector) -> None:
        """A point 0.5 deg latitude off a west-east axis at the axis midpoint
        sits ~55.6 km off track; width = max(cross) - min(cross)."""
        reports = [
            WindReport(time_s=0.0, lat=40.0, lon=-95.0, gust_ms=35.0),
            WindReport(time_s=1800.0, lat=40.5, lon=-92.5, gust_ms=35.0),
            WindReport(time_s=3600.0, lat=40.0, lon=-90.0, gust_ms=35.0),
        ]
        geometry = detector.compute_swath_geometry(reports)
        expected_offset = haversine_km(40.0, -92.5, 40.5, -92.5)
        # The great circle between two points at latitude 40 bulges ~0.09
        # deg poleward at mid-longitude, so the true cross-track distance
        # sits a few km below the naive meridian arc (~5-6 % here).
        assert geometry.width_km == pytest.approx(expected_offset, rel=0.07)
        assert geometry.width_km < expected_offset


class TestCriteriaBranches:
    """Constructed unit inputs falsifying one criterion at a time."""

    @staticmethod
    def _chain(
        n: int = 30,
        length_deg: float = 9.0,
        lat: float = 40.0,
        gust_ms: float = 35.0,
        total_hours: float = 10.0,
        cross_offsets_deg: tuple[float, ...] = (0.0, 0.6, -0.6, 0.9, -0.9),
    ) -> list[dict[str, Any]]:
        """Eastward-progressing report chain (constructed unit input)."""
        reports = []
        for i in range(n):
            frac = i / (n - 1)
            reports.append(
                {
                    "time_s": frac * total_hours * 3600.0,
                    "lat": lat + cross_offsets_deg[i % len(cross_offsets_deg)],
                    "lon": -95.0 + frac * length_deg,
                    "gust_ms": gust_ms,
                }
            )
        return reports

    def test_constructed_progressive_chain_qualifies(self, detector: DerechoDetector) -> None:
        result = detector.evaluate(self._chain())
        assert result.is_derecho
        assert result.classification == "progressive"

    def test_broad_swath_classified_serial(self, detector: DerechoDetector) -> None:
        """Width/length >= 0.4 -> serial (documented geometric cut)."""
        result = detector.evaluate(
            self._chain(length_deg=8.0, cross_offsets_deg=(0.0, 1.6, -1.6, 1.3, -1.3))
        )
        assert result.is_derecho
        assert result.geometry.width_km / result.geometry.length_km >= 0.4
        assert result.classification == "serial"

    def test_three_hour_gap_fails_continuity(self, detector: DerechoDetector) -> None:
        chain = self._chain()
        late = [r for r in chain if r["time_s"] > 5.0 * 3600.0]
        early = [r for r in chain if r["time_s"] <= 1.5 * 3600.0]
        result = detector.evaluate(early + late)
        assert not result.criteria["continuity"]
        assert not result.is_derecho

    def test_weak_gusts_fail_intensity_anchors(self, detector: DerechoDetector) -> None:
        result = detector.evaluate(self._chain(gust_ms=28.0))
        assert not result.criteria["intensity_anchors"]
        assert not result.is_derecho

    def test_f1_damage_counts_as_anchor(self, detector: DerechoDetector) -> None:
        """J&H: F1 wind damage substitutes for a measured 33 m/s gust."""
        chain = self._chain(gust_ms=28.0)
        for i in (0, 10, 20):
            chain[i] = {**chain[i], "gust_ms": None, "f_scale": 1}
        result = detector.evaluate(chain)
        assert result.criteria["intensity_anchors"]
        assert result.n_significant == 3

    def test_reversed_chronology_fails_progression(self, detector: DerechoDetector) -> None:
        """Reports scattered in time (no along-axis progression) fail J&H
        criterion 5."""
        chain = self._chain()
        times = [r["time_s"] for r in chain]
        rng = np.random.default_rng(42)
        rng.shuffle(times)
        shuffled = [{**r, "time_s": t} for r, t in zip(chain, times)]
        result = detector.evaluate(shuffled)
        assert not result.criteria["progression"]
        assert not result.is_derecho

    def test_narrow_swath_fails_width(self, detector: DerechoDetector) -> None:
        result = detector.evaluate(self._chain(cross_offsets_deg=(0.0, 0.1, -0.1)))
        assert not result.criteria["width"]
        assert not result.is_derecho


class TestFailLoudInputs:
    """The criteria are undefined without a real report series."""

    def test_none_and_short_series(self, detector: DerechoDetector) -> None:
        with pytest.raises(ValueError):
            detector.evaluate(None)
        with pytest.raises(ValueError):
            detector.evaluate([{"time_s": 0.0, "lat": 40.0, "lon": -90.0}])

    def test_missing_keys(self, detector: DerechoDetector) -> None:
        with pytest.raises(ValueError, match="missing required keys"):
            detector.evaluate([{"time_s": 0.0, "lat": 40.0}, {"time_s": 1.0, "lat": 41.0}])

    def test_bad_coordinates_and_gusts(self, detector: DerechoDetector) -> None:
        good = {"time_s": 0.0, "lat": 40.0, "lon": -90.0, "gust_ms": 35.0}
        with pytest.raises(ValueError):
            detector.evaluate([good, {**good, "time_s": 1.0, "lat": 95.0}])
        with pytest.raises(ValueError):
            detector.evaluate([good, {**good, "time_s": 1.0, "gust_ms": -3.0}])
        with pytest.raises(ValueError):
            detector.evaluate([good, {**good, "time_s": float("nan")}])


class TestBowEchoPrecursor:
    """Forward-propagation heuristic from real motion/wind inputs only."""

    def test_forward_propagating_mcs(self, detector: DerechoDetector) -> None:
        result = detector.bow_echo_precursor(20.0, 0.0, 12.0, 0.0)
        assert result["forward_propagating"]
        assert result["propagation_speed_ms"] == pytest.approx(8.0)
        assert result["propagation_angle_deg"] == pytest.approx(0.0, abs=1e-9)

    def test_backbuilding_mcs_not_flagged(self, detector: DerechoDetector) -> None:
        """Propagation opposed to the mean wind (back-building) is not a
        bow-echo signature."""
        result = detector.bow_echo_precursor(8.0, 0.0, 12.0, 0.0)
        assert not result["forward_propagating"]
        assert result["propagation_angle_deg"] == pytest.approx(180.0)

    def test_calm_mean_wind_fails_loud(self, detector: DerechoDetector) -> None:
        with pytest.raises(ValueError, match="calm"):
            detector.bow_echo_precursor(10.0, 0.0, 0.0, 0.0)

    def test_missing_inputs_fail_loud(self, detector: DerechoDetector) -> None:
        with pytest.raises(ValueError):
            detector.bow_echo_precursor(float("nan"), 0.0, 10.0, 0.0)
        with pytest.raises(ValueError):
            # None deliberately violates the float annotation (fail-loud check).
            missing_u: Any = None
            detector.bow_echo_precursor(missing_u, 0.0, 10.0, 0.0)


class TestExtractFeatures:
    """Fusion feature interface."""

    def test_report_series_path(
        self, detector: DerechoDetector, june2012_reports: list[dict[str, Any]]
    ) -> None:
        features = detector.extract_features(june2012_reports)
        assert isinstance(features, torch.Tensor)
        assert features.shape == (DerechoDetector.FEATURE_DIM,)
        assert features[0].item() == 1.0  # is_derecho
        assert features[9].item() == 1.0  # progressive one-hot

    def test_dict_wrapper_path(self, detector: DerechoDetector) -> None:
        chain = TestCriteriaBranches._chain()
        wrapped = detector.extract_features({"reports": chain})
        direct = detector.extract_features(chain)
        assert torch.equal(wrapped, direct)

    def test_array_path(self, detector: DerechoDetector) -> None:
        features = detector.extract_features(np.linspace(0.0, 1.0, 11))
        assert features.shape == (DerechoDetector.FEATURE_DIM,)

    def test_empty_array_fails_loud(self, detector: DerechoDetector) -> None:
        with pytest.raises(ValueError):
            detector.extract_features(np.array([]))
