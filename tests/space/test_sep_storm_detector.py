# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the SEP / radiation-storm detector.

S-scale anchors follow the NOAA Space Weather Scales ("Solar Radiation
Storms"); ingestion tests run against a recorded real SWPC integral-proton
fixture (quiet conditions) with provenance headers.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from omni_mercury_engine.data_sources.space_weather import SWPCProduct
from omni_mercury_engine.space.sep_storm_detector import (
    SEPStormDetector,
    assess_flare_connectivity,
    classify_s_scale,
    parse_flare_longitude,
)

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "space_weather"


def _times(n: int, step_minutes: int = 5) -> list[datetime]:
    start = datetime(2024, 5, 10, 0, 0, tzinfo=UTC)
    return [start + timedelta(minutes=step_minutes * i) for i in range(n)]


# ---------------------------------------------------------------------------
# NOAA S-scale classification
# ---------------------------------------------------------------------------


class TestSScale:
    @pytest.mark.parametrize(
        ("flux", "expected"),
        [
            (0.2, "S0"),
            (9.99, "S0"),
            (10.0, "S1"),
            (99.0, "S1"),
            (100.0, "S2"),
            (1500.0, "S3"),  # task anchor: 1500 pfu -> S3
            (1.0e4, "S4"),
            (99999.0, "S4"),
            (1.0e5, "S5"),
            (4.3e5, "S5"),
        ],
    )
    def test_noaa_thresholds(self, flux: float, expected: str) -> None:
        assert classify_s_scale(flux) == expected

    @pytest.mark.parametrize("flux", [-1.0, float("nan"), float("inf")])
    def test_invalid_flux_fails_loud(self, flux: float) -> None:
        with pytest.raises(ValueError, match=r"finite and non-negative"):
            classify_s_scale(flux)


# ---------------------------------------------------------------------------
# Onset detection (threshold + persistence)
# ---------------------------------------------------------------------------


class TestOnset:
    def test_onset_requires_persistence(self) -> None:
        detector = SEPStormDetector()  # 10 pfu, 3 consecutive samples
        # A single spike above threshold is not an onset (NOAA needs 3
        # consecutive 5-minute readings).
        flux = [1.0, 2.0, 50.0, 3.0, 2.0, 1.0]
        result = detector.assess(_times(len(flux)), flux)
        assert not result.onset_detected
        assert result.onset_time is None
        assert result.s_scale == "S1"  # peak still classifies

    def test_onset_time_is_first_sample_of_run(self) -> None:
        detector = SEPStormDetector()
        flux = [1.0, 2.0, 15.0, 20.0, 30.0, 12.0, 5.0]
        times = _times(len(flux))
        result = detector.assess(times, flux)
        assert result.onset_detected
        assert result.onset_time == times[2]
        assert not result.event_active  # last sample below threshold
        assert result.peak_flux_10mev_pfu == 30.0
        assert result.peak_time_10mev == times[4]

    def test_event_active_when_still_elevated(self) -> None:
        detector = SEPStormDetector()
        flux = [1.0, 15.0, 1500.0, 2000.0, 1200.0]
        result = detector.assess(_times(len(flux)), flux)
        assert result.event_active
        assert result.s_scale == "S3"

    def test_hundred_mev_channel(self) -> None:
        detector = SEPStormDetector()
        flux10 = [5.0, 20.0, 40.0, 35.0]
        flux100 = [0.1, 0.5, 2.0, 1.5]
        result = detector.assess(_times(4), flux10, flux_ge100mev_pfu=flux100)
        assert result.peak_flux_100mev_pfu == 2.0
        assert result.hundred_mev_event is True

    def test_no_hundred_mev_channel_reports_none(self) -> None:
        result = SEPStormDetector().assess(_times(2), [1.0, 2.0])
        assert result.peak_flux_100mev_pfu is None
        assert result.hundred_mev_event is None


# ---------------------------------------------------------------------------
# Fail-loud contract
# ---------------------------------------------------------------------------


class TestFailLoud:
    def test_empty_series(self) -> None:
        with pytest.raises(ValueError, match=r"Empty proton-flux series"):
            SEPStormDetector().assess([], [])

    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match=r"lengths differ"):
            SEPStormDetector().assess(_times(3), [1.0, 2.0])

    def test_negative_flux(self) -> None:
        with pytest.raises(ValueError, match=r"Non-finite or negative"):
            SEPStormDetector().assess(_times(2), [1.0, -0.5])

    def test_nan_flux(self) -> None:
        with pytest.raises(ValueError, match=r"Non-finite or negative"):
            SEPStormDetector().assess(_times(2), [1.0, float("nan")])

    def test_naive_timestamps(self) -> None:
        with pytest.raises(ValueError, match=r"timezone-aware"):
            SEPStormDetector().assess([datetime(2024, 5, 10)], [1.0])

    def test_unsorted_timestamps(self) -> None:
        times = _times(3)
        times[1], times[2] = times[2], times[1]
        with pytest.raises(ValueError, match=r"ascending"):
            SEPStormDetector().assess(times, [1.0, 2.0, 3.0])

    def test_bad_configuration(self) -> None:
        with pytest.raises(ValueError, match=r"must be positive"):
            SEPStormDetector(event_threshold_pfu=0.0)
        with pytest.raises(ValueError, match=r">= 1"):
            SEPStormDetector(persistence_samples=0)

    def test_incomplete_flare_record(self) -> None:
        with pytest.raises(ValueError, match=r"classType/sourceLocation"):
            SEPStormDetector().assess(_times(2), [1.0, 2.0], flare={"classType": "X1.0"})


# ---------------------------------------------------------------------------
# Well-connected flare precursor heuristic
# ---------------------------------------------------------------------------


class TestConnectivity:
    def test_parse_longitudes(self) -> None:
        assert parse_flare_longitude("S15W45") == 45.0
        assert parse_flare_longitude("N25W60") == 60.0
        assert parse_flare_longitude("S20E35") == -35.0

    def test_parse_invalid(self) -> None:
        with pytest.raises(ValueError, match=r"Cannot parse"):
            parse_flare_longitude("somewhere-on-the-sun")

    def test_well_connected_western_x_flare(self) -> None:
        result = assess_flare_connectivity("X1.0", "N25W60")
        assert result["well_connected"]
        assert "well-connected" in result["reason"]

    def test_eastern_flare_not_connected(self) -> None:
        result = assess_flare_connectivity("X5.0", "S20E35")
        assert not result["well_connected"]
        assert "E35" in result["reason"]

    def test_far_western_flare_beyond_band(self) -> None:
        result = assess_flare_connectivity("M5.0", "N10W89")
        assert not result["well_connected"]

    def test_weak_flare_not_a_precursor(self) -> None:
        result = assess_flare_connectivity("C5.0", "N10W50")
        assert not result["well_connected"]
        assert "below" in result["reason"]

    def test_advisory_attached_when_no_onset(self) -> None:
        detector = SEPStormDetector()
        result = detector.assess(
            _times(3),
            [0.5, 0.6, 0.4],
            flare={"classType": "X1.0", "sourceLocation": "N25W60"},
        )
        assert result.precursor is not None
        assert result.precursor["well_connected"]
        assert "Advisory only" in result.precursor_advisory
        assert result.s_scale == "S0"  # never fabricates flux


# ---------------------------------------------------------------------------
# SWPC product ingestion (recorded real fixture)
# ---------------------------------------------------------------------------


class TestSWPCIngestion:
    @pytest.fixture(scope="class")
    def fixture_payload(self) -> dict:
        with open(FIXTURE_DIR / "swpc_integral_protons_recent.json") as fh:
            return json.load(fh)

    def test_fixture_provenance(self, fixture_payload: dict) -> None:
        prov = fixture_payload["_provenance"]
        assert "integral-protons-7-day.json" in prov["source_url"]
        assert "SUBSET" in prov["note"]

    def test_assess_real_quiet_fixture(self, fixture_payload: dict) -> None:
        """Recorded quiet-sun fixture: sub-pfu flux, no onset, S0."""
        detector = SEPStormDetector()
        result = detector.assess_from_swpc(fixture_payload["data"])
        assert result.s_scale == "S0"
        assert not result.onset_detected
        assert not result.event_active
        assert result.n_samples >= 200
        assert result.peak_flux_10mev_pfu < 10.0
        assert result.peak_flux_100mev_pfu is not None

    def test_swpc_rows_without_ten_mev_fail_loud(self) -> None:
        rows = [
            {"time_tag": "2024-05-10T00:00:00Z", "flux": 1.0, "energy": ">=1 MeV"},
        ]
        with pytest.raises(ValueError, match=r">=10 MeV"):
            SEPStormDetector().assess_from_swpc(rows)

    def test_swpc_malformed_row_fails_loud(self) -> None:
        rows = [{"energy": ">=10 MeV", "flux": None, "time_tag": None}]
        with pytest.raises(ValueError, match=r"Malformed"):
            SEPStormDetector().assess_from_swpc(rows)

    def test_swpc_product_enum_and_routing(self) -> None:
        """The verified real product path is wired into SWPCProduct."""
        assert SWPCProduct.INTEGRAL_PROTONS.value == "primary/integral-protons-7-day.json"

    def test_swpc_source_parses_fixture_rows(self, fixture_payload: dict) -> None:
        """NOAASWPCSource parses the recorded product into DataPoints."""
        from omni_mercury_engine.data_sources.base import DataSourceType
        from omni_mercury_engine.data_sources.space_weather import NOAASWPCSource

        source = NOAASWPCSource(products=[SWPCProduct.INTEGRAL_PROTONS])
        points = source._parse_product_data(SWPCProduct.INTEGRAL_PROTONS, fixture_payload["data"])
        assert len(points) == len(fixture_payload["data"])
        assert all(p.source_type is DataSourceType.SOLAR_ENERGETIC_PARTICLE for p in points)
        energies = {p.data["energy"] for p in points}
        assert energies == {">=10 MeV", ">=100 MeV"}
