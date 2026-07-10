# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the solar-to-GIC cascade escalation state machine.

Stage-transition tests replay the recorded real May 2024 (Gannon G5 storm)
fixtures — DONKI flares + CME analyses, DONKI GST observed Kp, and USGS
Boulder magnetometer minute data — at different evaluation times, walking
the machine through QUIET -> WATCH -> WARNING -> STORM_IN_PROGRESS exactly
as the real event unfolded.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# The dev venv's editable install may point at a sibling worktree that
# predates ``solar_gic_cascade``; ``unused-ignore`` keeps a correctly
# installed tree (CI) clean.
from omni_mercury_engine.space.solar_gic_cascade import (  # type: ignore[import-not-found,unused-ignore]
    CascadeInputs,
    CascadeStage,
    SolarGICCascadeDetector,
)

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "space_weather"


def _load(name: str) -> Any:
    with open(FIXTURE_DIR / name) as fh:
        return json.load(fh)["data"]


@pytest.fixture(scope="module")
def gannon_inputs() -> dict[str, Any]:
    """Real May 2024 storm observations from the recorded fixtures."""
    flares = _load("donki_flr_2024_05.json")
    cmes = _load("donki_cme_analysis_2024_05.json")

    gst = _load("donki_gst_2024_05.json")
    kp_series: list[tuple[datetime, float]] = []
    for storm in gst:
        for entry in storm["allKpIndex"]:
            kp_series.append(
                (
                    datetime.fromisoformat(entry["observedTime"].replace("Z", "+00:00")),
                    float(entry["kpIndex"]),
                )
            )
    kp_series.sort()

    geomag = _load("usgs_geomag_bou_2024_05_10.json")
    mag_times = [datetime.fromisoformat(t.replace("Z", "+00:00")) for t in geomag["times"]]
    series = {v["id"]: v["values"] for v in geomag["values"]}
    to_arr = lambda vals: np.array(  # noqa: E731 - local shim
        [np.nan if v is None else float(v) for v in vals], dtype=np.float64
    )
    return {
        "flares": flares,
        "cmes": cmes,
        "kp_series": kp_series,
        "mag_times": mag_times,
        "mag_bx": to_arr(series["X"]),
        "mag_by": to_arr(series["Y"]),
    }


def _inputs(gannon: dict[str, Any], now: datetime, with_mag: bool = True) -> CascadeInputs:
    return CascadeInputs(
        now=now,
        flares=gannon["flares"],
        cme_analyses=gannon["cmes"],
        kp_series=gannon["kp_series"],
        mag_times=gannon["mag_times"] if with_mag else None,
        mag_bx_nt=gannon["mag_bx"] if with_mag else None,
        mag_by_nt=gannon["mag_by"] if with_mag else None,
        observatory="BOU",
    )


class TestStageTransitionsOnRealEvent:
    """Walk the machine through the real Gannon-storm timeline."""

    def test_quiet_before_the_flare_sequence(self, gannon_inputs: dict[str, Any]) -> None:
        """2024-05-07: no flares/CMEs observed yet, stale Kp -> QUIET."""
        cascade = SolarGICCascadeDetector()
        result = cascade.evaluate(_inputs(gannon_inputs, datetime(2024, 5, 7, 0, 0, tzinfo=UTC)))
        assert result.stage is CascadeStage.QUIET
        watch = result.evidence_chain[0]
        assert not watch.satisfied
        assert watch.datapoints == []
        assert result.active_arrival_window is None

    def test_watch_after_flares_before_arrival(self, gannon_inputs: dict[str, Any]) -> None:
        """2024-05-09T00Z: X/M flares + Earth-directed CMEs observed, but no
        arrival window open and no storm-level Kp -> WATCH."""
        cascade = SolarGICCascadeDetector()
        result = cascade.evaluate(_inputs(gannon_inputs, datetime(2024, 5, 9, 0, 0, tzinfo=UTC)))
        assert result.stage is CascadeStage.WATCH
        watch, warning, storm = result.evidence_chain
        assert watch.satisfied
        # Real evidence: the May 8 X1.0 flare and Earth-directed CMEs.
        flare_ids = {d.get("flare_id") for d in watch.datapoints if d["type"] == "flare"}
        assert any(fid is not None and "2024-05-08" in fid for fid in flare_ids)
        cme_ids = {d.get("cme_id") for d in watch.datapoints if d["type"] == "cme"}
        assert "2024-05-08T05:36:00-CME-001" in cme_ids
        assert not warning.satisfied
        assert "no predicted CME arrival window is open" in warning.reason
        assert not storm.satisfied

    def test_warning_at_arrival_with_elevated_kp(self, gannon_inputs: dict[str, Any]) -> None:
        """2024-05-10T18Z: windows open, observed Kp 7.67, but BOU dB/dt has
        only reached ~70 nT/min -> WARNING (not yet storm-in-progress)."""
        cascade = SolarGICCascadeDetector()
        result = cascade.evaluate(_inputs(gannon_inputs, datetime(2024, 5, 10, 18, 0, tzinfo=UTC)))
        assert result.stage is CascadeStage.WARNING
        watch, warning, storm = result.evidence_chain
        assert watch.satisfied and warning.satisfied and not storm.satisfied
        kp_points = [d for d in warning.datapoints if d["type"] == "kp_observation"]
        assert kp_points and kp_points[0]["kp"] == pytest.approx(7.67)
        assert result.active_arrival_window is not None
        early, late = result.active_arrival_window
        assert early <= datetime(2024, 5, 10, 18, 0, tzinfo=UTC) <= late
        assert result.gic_assessment is not None
        assert result.gic_assessment.peak_dbdt_nt_per_min < 100.0
        assert "below the 100 nT/min storm tier" in storm.reason

    def test_storm_in_progress_at_main_phase(self, gannon_inputs: dict[str, Any]) -> None:
        """2024-05-11T02Z: Kp 9.0 observed and BOU measured 253.9 nT/min ->
        STORM_IN_PROGRESS with the full evidence chain."""
        cascade = SolarGICCascadeDetector()
        result = cascade.evaluate(_inputs(gannon_inputs, datetime(2024, 5, 11, 2, 0, tzinfo=UTC)))
        assert result.stage is CascadeStage.STORM_IN_PROGRESS
        watch, warning, storm = result.evidence_chain
        assert watch.satisfied and warning.satisfied and storm.satisfied
        dbdt_points = [d for d in storm.datapoints if d["type"] == "dbdt_measurement"]
        assert dbdt_points
        assert dbdt_points[0]["observatory"] == "BOU"
        assert dbdt_points[0]["peak_dbdt_nt_per_min"] == pytest.approx(253.9, abs=1.0)
        assert dbdt_points[0]["risk_level"] == "moderate"
        # Every stage carries concrete triggering datapoints.
        assert all(e.datapoints for e in result.evidence_chain)

    def test_warning_without_magnetometer_data(self, gannon_inputs: dict[str, Any]) -> None:
        """Without ground magnetometer data the machine can never claim a
        storm is in progress — measured dB/dt is a hard requirement."""
        cascade = SolarGICCascadeDetector()
        result = cascade.evaluate(
            _inputs(gannon_inputs, datetime(2024, 5, 11, 2, 0, tzinfo=UTC), with_mag=False)
        )
        assert result.stage is CascadeStage.WARNING
        storm = result.evidence_chain[2]
        assert "no magnetometer data" in storm.reason
        assert result.gic_assessment is None


class TestStrictEscalation:
    def test_unattributed_dbdt_does_not_escalate(self) -> None:
        """High measured dB/dt with no solar driver stays QUIET, loudly."""
        cascade = SolarGICCascadeDetector()
        start = datetime(2024, 5, 10, 16, 0, tzinfo=UTC)
        times = [start + timedelta(minutes=i) for i in range(16)]
        bx = np.zeros(16)
        bx[8:] = 400.0
        result = cascade.evaluate(
            CascadeInputs(
                now=times[-1],
                mag_times=times,
                mag_bx_nt=bx,
                mag_by_nt=np.zeros(16),
                observatory="TEST",
            )
        )
        assert result.stage is CascadeStage.QUIET
        assert any("unattributed_dbdt" in note for note in result.notes)
        assert result.gic_assessment is not None
        assert result.gic_assessment.peak_dbdt_nt_per_min == pytest.approx(400.0)

    def test_unattributed_kp_does_not_escalate(self) -> None:
        now = datetime(2024, 5, 10, 18, 0, tzinfo=UTC)
        result = SolarGICCascadeDetector().evaluate(
            CascadeInputs(now=now, kp_series=[(now - timedelta(hours=1), 8.0)])
        )
        assert result.stage is CascadeStage.QUIET
        assert any("unattributed_kp" in note for note in result.notes)

    def test_stale_kp_is_not_current(self, gannon_inputs: dict[str, Any]) -> None:
        """A Kp observation older than 6 h cannot arm the WARNING stage."""
        cascade = SolarGICCascadeDetector()
        now = datetime(2024, 5, 10, 12, 0, tzinfo=UTC)  # windows open, Kp stale
        stale_kp = [(datetime(2024, 5, 3, 0, 0, tzinfo=UTC), 6.67)]
        result = cascade.evaluate(
            CascadeInputs(
                now=now,
                flares=gannon_inputs["flares"],
                cme_analyses=gannon_inputs["cmes"],
                kp_series=stale_kp,
            )
        )
        assert result.stage is CascadeStage.WATCH
        assert "no Kp observation within 6 h" in result.evidence_chain[1].reason

    def test_malformed_records_noted_not_fatal(self, gannon_inputs: dict[str, Any]) -> None:
        """Broken upstream records are excluded loudly, not silently."""
        cascade = SolarGICCascadeDetector()
        broken_flare = {"flrID": "broken-flr", "classType": None, "beginTime": None}
        broken_cme = {"associatedCMEID": "broken-cme", "speed": 900.0}
        result = cascade.evaluate(
            CascadeInputs(
                now=datetime(2024, 5, 9, 0, 0, tzinfo=UTC),
                flares=[*gannon_inputs["flares"], broken_flare],
                cme_analyses=[*gannon_inputs["cmes"], broken_cme],
            )
        )
        assert result.stage is CascadeStage.WATCH  # real records still count
        assert any("malformed_flare" in n for n in result.notes)
        assert any("malformed_cme" in n for n in result.notes)


class TestFailLoud:
    def test_naive_now_fails(self) -> None:
        with pytest.raises(ValueError, match=r"timezone-aware"):
            SolarGICCascadeDetector().evaluate(CascadeInputs(now=datetime(2024, 5, 10)))

    def test_naive_kp_timestamp_fails(self) -> None:
        with pytest.raises(ValueError, match=r"timezone-aware"):
            SolarGICCascadeDetector().evaluate(
                CascadeInputs(
                    now=datetime(2024, 5, 10, tzinfo=UTC),
                    kp_series=[(datetime(2024, 5, 10), 5.0)],
                )
            )

    def test_out_of_range_kp_fails(self) -> None:
        with pytest.raises(ValueError, match=r"outside \[0, 9\]"):
            SolarGICCascadeDetector().evaluate(
                CascadeInputs(
                    now=datetime(2024, 5, 10, tzinfo=UTC),
                    kp_series=[(datetime(2024, 5, 10, tzinfo=UTC), 12.0)],
                )
            )

    def test_bad_thresholds_fail(self) -> None:
        with pytest.raises(ValueError, match=r"kp_watch_threshold"):
            SolarGICCascadeDetector(kp_watch_threshold=11.0)
        with pytest.raises(ValueError, match=r"dbdt_storm_threshold"):
            SolarGICCascadeDetector(dbdt_storm_threshold_nt_per_min=-5.0)

    def test_empty_inputs_are_quiet_with_reasons(self) -> None:
        result = SolarGICCascadeDetector().evaluate(
            CascadeInputs(now=datetime(2024, 5, 10, tzinfo=UTC))
        )
        assert result.stage is CascadeStage.QUIET
        assert [e.satisfied for e in result.evidence_chain] == [False, False, False]
        assert all(e.reason for e in result.evidence_chain)
