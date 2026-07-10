# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the GIC-to-grid detector.

dB/dt anchors are hand-computed finite differences; the plane-wave
geoelectric proxy is verified against the analytic single-tone solution
|E| = B0 * sqrt(omega / (mu0 * sigma)); ingestion tests use the recorded
real USGS Boulder minute-data fixture from the 2024-05-10 Gannon G5 storm.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from omni_mercury_engine.data_sources.base import DataPoint, DataSourceType

# The dev venv's editable install may point at a sibling worktree that
# predates ``gic_detector``; ``unused-ignore`` keeps a correctly
# installed tree (CI) clean.
from omni_mercury_engine.space.gic_detector import (  # type: ignore[import-not-found,unused-ignore]
    MU0,
    GICDetector,
    classify_dbdt_risk,
)

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "space_weather"


def _minute_times(n: int) -> list[datetime]:
    start = datetime(2024, 5, 10, 16, 0, tzinfo=UTC)
    return [start + timedelta(minutes=i) for i in range(n)]


def _load_gannon_fixture() -> tuple[list[datetime], np.ndarray, np.ndarray]:
    with open(FIXTURE_DIR / "usgs_geomag_bou_2024_05_10.json") as fh:
        payload = json.load(fh)
    data = payload["data"]
    times = [datetime.fromisoformat(t.replace("Z", "+00:00")) for t in data["times"]]
    series = {v["id"]: v["values"] for v in data["values"]}
    to_arr = lambda vals: np.array(  # noqa: E731 - local shim
        [np.nan if v is None else float(v) for v in vals], dtype=np.float64
    )
    return times, to_arr(series["X"]), to_arr(series["Y"])


# ---------------------------------------------------------------------------
# Risk tiers
# ---------------------------------------------------------------------------


class TestRiskTiers:
    @pytest.mark.parametrize(
        ("peak", "expected"),
        [
            (0.0, "low"),
            (99.9, "low"),
            (100.0, "moderate"),
            (299.9, "moderate"),
            (300.0, "high"),
            (400.0, "high"),  # task anchor: 400 nT/min -> high tier
            (480.0, "high"),  # March 1989 Hydro-Québec-scale disturbance
            (500.0, "severe"),
            (5000.0, "severe"),  # Carrington-class benchmark context
        ],
    )
    def test_operational_tiers(self, peak: float, expected: str) -> None:
        assert classify_dbdt_risk(peak) == expected

    @pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
    def test_invalid_peak_fails_loud(self, bad: float) -> None:
        with pytest.raises(ValueError):
            classify_dbdt_risk(bad)


# ---------------------------------------------------------------------------
# dB/dt finite differences
# ---------------------------------------------------------------------------


class TestDbdt:
    def test_hand_computed_ramp(self) -> None:
        """B = [0, 60, 180, 180] nT at minute cadence -> dB/dt = [60, 120, 0]."""
        times = _minute_times(4)
        dbdt = GICDetector.compute_dbdt_nt_per_min(times, np.array([0.0, 60.0, 180.0, 180.0]))
        assert dbdt == pytest.approx([60.0, 120.0, 0.0])

    def test_respects_actual_dt(self) -> None:
        """A 30 nT step over 30 s is 60 nT/min."""
        start = datetime(2024, 5, 10, tzinfo=UTC)
        times = [start, start + timedelta(seconds=30)]
        dbdt = GICDetector.compute_dbdt_nt_per_min(times, np.array([0.0, 30.0]))
        assert dbdt == pytest.approx([60.0])

    def test_gap_yields_nan_step(self) -> None:
        dbdt = GICDetector.compute_dbdt_nt_per_min(_minute_times(3), np.array([0.0, np.nan, 10.0]))
        assert np.isnan(dbdt).all()

    def test_too_short_fails_loud(self) -> None:
        with pytest.raises(ValueError, match=r">= 2 samples"):
            GICDetector.compute_dbdt_nt_per_min(_minute_times(1), np.array([1.0]))

    def test_naive_time_fails_loud(self) -> None:
        with pytest.raises(ValueError, match=r"timezone-aware"):
            GICDetector.compute_dbdt_nt_per_min(
                [datetime(2024, 5, 10), datetime(2024, 5, 10, 0, 1)],
                np.array([0.0, 1.0]),
            )

    def test_non_ascending_time_fails_loud(self) -> None:
        times = _minute_times(3)
        times[2] = times[0]
        with pytest.raises(ValueError, match=r"ascending"):
            GICDetector.compute_dbdt_nt_per_min(times, np.array([0.0, 1.0, 2.0]))


# ---------------------------------------------------------------------------
# Plane-wave geoelectric proxy (Cagniard 1953)
# ---------------------------------------------------------------------------


class TestGeoelectric:
    def test_single_tone_matches_analytic_solution(self) -> None:
        """100 nT tone at 600 s period over sigma = 1e-3 S/m ground.

        |Z| = sqrt(omega * mu0 / sigma), |E| = |Z| * B0 / mu0
            = B0 * sqrt(omega / (mu0 * sigma))
            = 1e-7 T * sqrt(0.0104720 / 1.2566371e-9) = 2.8868e-4 V/m
            = 0.28868 V/km.
        """
        detector = GICDetector(conductivity_model="resistive_shield")
        n, dt = 720, 60.0
        t = np.arange(n) * dt
        period = 600.0
        b0_nt = 100.0
        bx = b0_nt * np.sin(2 * math.pi * t / period)
        by = np.zeros(n)

        e_x, e_y = detector.geoelectric_plane_wave_v_per_km(bx, by, dt)
        omega = 2 * math.pi / period
        expected_v_per_km = b0_nt * 1e-9 * math.sqrt(omega / (MU0 * 1e-3)) * 1000.0
        assert expected_v_per_km == pytest.approx(0.28868, abs=2e-4)
        # bx drives e_y only.
        assert np.max(np.abs(e_x)) < 1e-9
        # Discard edge samples (finite-series edge effects), compare bulk peak.
        core = np.abs(e_y[50:-50])
        assert np.max(core) == pytest.approx(expected_v_per_km, rel=0.02)

    def test_more_conductive_ground_gives_smaller_field(self) -> None:
        n, dt = 512, 60.0
        bx = 50.0 * np.sin(2 * math.pi * np.arange(n) / 32.0)
        by = np.zeros(n)
        e_shield = GICDetector("resistive_shield").geoelectric_plane_wave_v_per_km(bx, by, dt)
        e_sediment = GICDetector("conductive_sediment").geoelectric_plane_wave_v_per_km(bx, by, dt)
        ratio = np.max(np.abs(e_shield[1])) / np.max(np.abs(e_sediment[1]))
        # |E| ~ 1/sqrt(sigma): 1e-3 vs 1e-1 -> ratio = 10.
        assert ratio == pytest.approx(10.0, rel=1e-6)

    def test_gap_in_series_fails_loud(self) -> None:
        detector = GICDetector()
        bx = np.zeros(64)
        bx[10] = np.nan
        with pytest.raises(ValueError, match=r"gap-free"):
            detector.geoelectric_plane_wave_v_per_km(bx, np.zeros(64), 60.0)

    def test_custom_conductivity(self) -> None:
        detector = GICDetector(sigma_s_per_m=3.0e-4)
        assert detector.conductivity_model == "custom"
        assert detector.sigma_s_per_m == 3.0e-4

    def test_unknown_model_fails_loud(self) -> None:
        with pytest.raises(ValueError, match=r"Unknown conductivity model"):
            GICDetector(conductivity_model="mars_regolith")
        with pytest.raises(ValueError, match=r"positive"):
            GICDetector(sigma_s_per_m=-1.0)


# ---------------------------------------------------------------------------
# Full assessment
# ---------------------------------------------------------------------------


class TestAssess:
    def test_synthetic_step_assessment(self) -> None:
        """One 400 nT/min horizontal step lands in the 'high' tier."""
        detector = GICDetector()
        n = 32
        times = _minute_times(n)
        bx = np.zeros(n)
        bx[16:] = 400.0  # single 400 nT jump on the step ending at times[16]
        by = np.zeros(n)
        result = detector.assess(times, bx, by, observatory="TEST")
        assert result.peak_dbdt_nt_per_min == pytest.approx(400.0)
        assert result.risk_level == "high"
        assert result.peak_dbdt_time == times[16]
        assert result.geoelectric_peak_v_per_km is not None
        assert result.sustained_dbdt_nt_per_min == pytest.approx(40.0)  # 400/10

    def test_all_gaps_fail_loud(self) -> None:
        detector = GICDetector()
        with pytest.raises(ValueError, match=r"no usable data"):
            detector.assess(_minute_times(4), np.full(4, np.nan), np.full(4, np.nan))

    def test_gappy_series_skips_geoelectric_with_note(self) -> None:
        detector = GICDetector()
        n = 16
        bx = np.arange(n, dtype=float)
        bx[5] = np.nan
        result = detector.assess(_minute_times(n), bx, np.zeros(n))
        assert result.geoelectric_peak_v_per_km is None
        assert any("data gaps" in note for note in result.notes)
        assert result.n_gaps == 1

    def test_single_component_h_only(self) -> None:
        detector = GICDetector()
        n = 16
        h = np.linspace(20000.0, 20150.0, n)  # 10 nT/min ramp
        result = detector.assess(_minute_times(n), h, None, observatory="HON")
        assert result.single_component
        assert result.peak_dbdt_nt_per_min == pytest.approx(10.0)
        assert result.risk_level == "low"
        assert any("H-only" in note for note in result.notes)

    def test_gannon_storm_fixture_real_data(self) -> None:
        """Recorded BOU minute data, 2024-05-10 storm arrival + main phase.

        Peak horizontal dB/dt at Boulder during this window is ~254 nT/min
        (computed from the recorded fixture) — 'moderate' on the
        operational tiers, consistent with a mid-latitude G5 response.
        """
        times, bx, by = _load_gannon_fixture()
        detector = GICDetector()
        result = detector.assess(times, bx, by, observatory="BOU")
        assert result.n_samples == 721
        assert result.peak_dbdt_nt_per_min == pytest.approx(253.9, abs=1.0)
        assert result.risk_level == "moderate"
        # Storm sudden commencement / main phase within the window.
        assert result.peak_dbdt_time.day in (10, 11)
        assert result.geoelectric_peak_v_per_km is not None
        assert result.geoelectric_peak_v_per_km > 0.1  # V/km-scale storm field
        assert result.sustained_dbdt_nt_per_min is not None
        assert result.sustained_dbdt_nt_per_min < result.peak_dbdt_nt_per_min


# ---------------------------------------------------------------------------
# DataPoint ingestion (USGSGeomagnetismSource shape)
# ---------------------------------------------------------------------------


def _make_points(observatory: str, elements_list: list[dict[str, float]]) -> list[DataPoint]:
    start = datetime(2024, 5, 10, 16, 0, tzinfo=UTC)
    return [
        DataPoint(
            source_id="usgs_geomagnetism",
            source_type=DataSourceType.MAGNETOMETER,
            event_id=f"usgs_{observatory}_{i}",
            timestamp=start + timedelta(minutes=i),
            data={"observatory": observatory, "elements": elements, "sampling": "minute"},
        )
        for i, elements in enumerate(elements_list)
    ]


class TestDataPointIngestion:
    def test_xy_datapoints(self) -> None:
        points = _make_points(
            "BOU",
            [{"X": 20000.0, "Y": 100.0}, {"X": 20150.0, "Y": 100.0}, {"X": 20150.0, "Y": 250.0}],
        )
        results = GICDetector().assess_from_datapoints(points)
        assert set(results) == {"BOU"}
        assert results["BOU"].peak_dbdt_nt_per_min == pytest.approx(150.0)
        assert results["BOU"].risk_level == "moderate"

    def test_h_only_datapoints(self) -> None:
        points = _make_points("HON", [{"H": 21000.0}, {"H": 21030.0}])
        results = GICDetector().assess_from_datapoints(points)
        assert results["HON"].single_component

    def test_unusable_elements_fail_loud(self) -> None:
        points = _make_points("GUA", [{"Z": 1.0}, {"Z": 2.0}])
        with pytest.raises(ValueError, match=r"neither X/Y nor H"):
            GICDetector().assess_from_datapoints(points)

    def test_no_points_fail_loud(self) -> None:
        with pytest.raises(ValueError, match=r"No magnetometer DataPoints"):
            GICDetector().assess_from_datapoints([])
