# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Case-surge detection on recorded national surveillance counts.

Italy's confirmed COVID-19 cases, 2020-02-15 to 2020-07-15, from the JHU CSSE
time series (CC BY 4.0; see ``tests/fixtures/medical/jhu_csse_covid19``).
Every 14-day window is scored and its week-on-week ratio, doubling time and
growth-implied R_e (Wallinga and Lipsitch 2007, delta serial interval) are
recomputed here from the raw cumulative counts.  The recorded series also
carries what synthetic counts did not: a post-lockdown decline (R_e < 1,
formerly reported as 1.0) and a reporting correction (a negative daily
count, formerly averaged in as cases).
"""

from __future__ import annotations

import csv
import itertools
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.core.exceptions import InvalidObservationError
from omni_mercury_engine.detectors.adapters.medical import CaseSurgeWindowAdapter
from omni_mercury_engine.medical.pandemic.pandemic_detector import (
    CaseSurgeDetector,
    PandemicDetector,
)

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "medical"
    / "jhu_csse_covid19"
    / "italy_confirmed_2020.csv"
)
SERIAL_INTERVAL = 5.0


def _daily() -> tuple[list[str], list[int]]:
    """Dates and daily new cases (cumulative differences) from the fixture."""
    with FIXTURE.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    cumulative = [int(row["cumulative_confirmed"]) for row in rows]
    dates = [row["date"] for row in rows]
    return dates[1:], [b - a for a, b in itertools.pairwise(cumulative)]


def _window(end: str) -> list[int]:
    dates, daily = _daily()
    stop = dates.index(end) + 1
    return daily[stop - 14 : stop]


@pytest.fixture(scope="module")
def surge() -> CaseSurgeDetector:
    return CaseSurgeDetector()


class TestRecordedItaly:
    def test_every_clean_window_matches_the_formula(self, surge: CaseSurgeDetector) -> None:
        dates, daily = _daily()
        checked = 0
        for stop in range(14, len(daily) + 1):
            window = daily[stop - 14 : stop]
            if min(window) < 0:
                continue  # the correction window has its own test
            baseline, recent = sum(window[:7]) / 7, sum(window[7:]) / 7
            result = surge.detect_case_surge(
                {"daily_cases": window, "serial_interval_days": SERIAL_INTERVAL}
            )
            # (The flag's type has its own test; the value is checked here.)
            assert bool(result["surge_detected"]) == (recent > 1.5 * baseline), dates[stop - 1]
            assert "partial" not in result
            if baseline == 0 or recent == 0:
                assert result["re_estimate"] is None
                continue
            rate = math.log(recent / baseline) / 7
            assert result["re_estimate"] == pytest.approx(math.exp(rate * SERIAL_INTERVAL))
            assert result["week_on_week_ratio"] == pytest.approx(recent / baseline)
            if rate > 0:
                assert result["doubling_time_days"] == pytest.approx(math.log(2) / rate)
            else:
                assert result["doubling_time_days"] is None
            checked += 1
        assert checked > 100

    def test_early_march_growth_is_a_surge(self, surge: CaseSurgeDetector) -> None:
        """Week ending 2020-03-08: mean daily cases 3.69x the week before."""
        result = surge.detect_case_surge({"daily_cases": _window("2020-03-08")})
        assert bool(result["surge_detected"]) is True
        assert result["doubling_time_days"] == pytest.approx(3.72, abs=0.01)
        assert result["re_estimate"] == pytest.approx(2.54, abs=0.01)

    def test_post_lockdown_decline_reports_re_below_one(self, surge: CaseSurgeDetector) -> None:
        """Week ending 2020-05-17: 0.76x the week before; R_e was reported as 1.0."""
        result = surge.detect_case_surge({"daily_cases": _window("2020-05-17")})
        assert result["re_estimate"] == pytest.approx(0.82, abs=0.01)
        assert bool(result["surge_detected"]) is False
        assert result["doubling_time_days"] is None

    def test_reporting_correction_is_a_gap_not_a_count(self, surge: CaseSurgeDetector) -> None:
        """2020-06-19 is -148 (a correction); it was averaged in as cases."""
        window = _window("2020-06-21")
        assert window[11] == -148
        result = surge.detect_case_surge({"daily_cases": window})
        reasons = " ".join(entry["reason"] for entry in result["set_aside"])
        assert "1 negative count(s) are reporting corrections" in reasons
        # The longest clean run is 11 days: too short for two weekly windows.
        assert result["re_method"] == "insufficient_history"
        assert result["re_estimate"] is None
        assert result["partial"] is True


class TestSurgeDefects:
    """Regressions: each failed on the former detector."""

    def test_under_two_weeks_is_not_assessed_rather_than_negative(
        self, surge: CaseSurgeDetector
    ) -> None:
        """Under 14 days the week was compared with itself: never a surge, R_e 1.0."""
        result = surge.detect_case_surge({"daily_cases": [100, 150, 225, 340, 510, 765, 1150]})
        assert result["re_method"] == "insufficient_history"
        assert result["re_estimate"] is None and result["r0_estimate"] is None
        assert result["partial"] is True

    def test_gap_is_set_aside_and_the_run_graded(self, surge: CaseSurgeDetector) -> None:
        """A NaN day made both means NaN: a silent 'no surge'."""
        series = [100 * 2 ** (day / 3) for day in range(16)]
        series[1] = float("nan")
        result = surge.detect_case_surge({"daily_cases": series})
        assert result["surge_detected"] is True
        assert result["re_estimate"] == pytest.approx(2 ** (5 / 3))
        assert result["partial"] is True

    def test_no_usable_count_is_refused(self, surge: CaseSurgeDetector) -> None:
        with pytest.raises(InvalidObservationError, match="no finite sample"):
            surge.detect_case_surge({"daily_cases": [float("nan")] * 14})

    def test_emergence_from_zero_has_no_re(self, surge: CaseSurgeDetector) -> None:
        """A zero baseline has no growth rate; R_e was reported as 1.0."""
        result = surge.detect_case_surge({"daily_cases": [0] * 7 + [40] * 7})
        assert result["surge_detected"] is True
        assert result["re_estimate"] is None
        assert result["re_method"] == "zero_counts"

    @pytest.mark.parametrize("interval", [0.0, float("nan"), -2.0])
    def test_unusable_serial_interval_is_set_aside(
        self, surge: CaseSurgeDetector, interval: float
    ) -> None:
        series = [100 * 1.3**day for day in range(14)]
        result = surge.detect_case_surge({"daily_cases": series, "serial_interval_days": interval})
        assert result["re_estimate"] is None
        assert result["re_method"] == "no_serial_interval"
        assert result["doubling_time_days"] == pytest.approx(math.log(2) / math.log(1.3))
        assert result["partial"] is True

    def test_flag_is_a_python_bool(self, surge: CaseSurgeDetector) -> None:
        result = surge.detect_case_surge({"daily_cases": [100] * 14})
        assert type(result["surge_detected"]) is bool


class TestPandemicDefects:
    """Regressions on the integrated detector."""

    @pytest.fixture
    def detector(self) -> PandemicDetector:
        return PandemicDetector()

    def test_empty_record_is_refused(self, detector: PandemicDetector) -> None:
        """``{}`` graded as 'sporadic': an unassessed population read as quiet."""
        with pytest.raises(InvalidObservationError, match="nothing usable"):
            detector.predict_pandemic({})

    def test_untrained_network_is_set_aside_not_an_all_clear(
        self, detector: PandemicDetector
    ) -> None:
        """The untrained network's noise (0.45-0.56) read as 'no hotspot'."""
        result = detector.predict_pandemic(
            {
                "network_data": {"contact_density": 0.9},
                "geographic_spread": {"countries_affected": 1, "continents_affected": 1},
            }
        )
        assert result.transmission_hotspots == []
        assert result.partial is True
        assert [e["where"] for e in result.set_aside] == ["network_data"]

    def test_trained_network_is_consulted(self) -> None:
        detector = PandemicDetector()
        assert detector.network_analyzer is not None
        detector.network_analyzer.load_trained_weights(detector.network_analyzer.state_dict())
        result = detector.predict_pandemic({"network_data": {"contact_density": 0.9}})
        assert result.partial is False

    def test_unmeasured_antigenic_distance_is_not_zero(self, detector: PandemicDetector) -> None:
        result = detector.predict_pandemic({"genomic_data": {"mutation_count": 12}})
        assert result.antigenic_distance is None
        assert result.mutation_detected is True

    def test_nan_inputs_are_set_aside(self, detector: PandemicDetector) -> None:
        """NaN distance and country counts compared False: wild type, sporadic."""
        result = detector.predict_pandemic(
            {
                "genomic_data": {"antigenic_distance": float("nan"), "spike_mutations": ["N501Y"]},
                "geographic_spread": {"countries_affected": float("nan"), "continents_affected": 1},
                "demographic_data": {"median_age": 45},
            }
        )
        where = {entry["where"] for entry in result.set_aside}
        assert where == {
            "genomic_data.antigenic_distance",
            "geographic_spread.countries_affected",
            "demographic_data",
        }
        assert result.antigenic_distance is None
        assert result.partial is True

    def test_disabled_stream_is_set_aside_not_ignored(self) -> None:
        """Supplied data for a disabled analysis was silently dropped."""
        detector = PandemicDetector(enable_mutation_tracking=False)
        result = detector.predict_pandemic(
            {
                "genomic_data": {"antigenic_distance": 3.0},
                "geographic_spread": {"countries_affected": 4, "continents_affected": 1},
            }
        )
        assert result.antigenic_distance is None
        assert [e["where"] for e in result.set_aside] == ["genomic_data"]
        assert result.partial is True

    def test_case_series_entries_carry_their_section(self, detector: PandemicDetector) -> None:
        series = [100 * 1.5 ** (day - 7) for day in range(15)]
        series[0] = -3
        result = detector.predict_pandemic({"case_data": {"daily_cases": series}})
        assert result.case_surge_detected is True
        assert {entry["where"] for entry in result.set_aside} == {"case_data.daily_cases"}
        assert np.isclose(result.r0_estimate or 0.0, 1.5**5)


class TestWindowAdapter:
    def test_recorded_surge_window_through_the_adapter(self) -> None:
        window = np.asarray(_window("2020-03-08"), dtype=float).reshape(-1, 1)
        out = CaseSurgeWindowAdapter().detect(window)
        ratio = sum(window[7:, 0]) / sum(window[:7, 0])
        assert out["flags"].tolist() == [True]
        assert out["scores"][0] == pytest.approx(ratio / (ratio + 1.5))
        assert "partial" not in out

    def test_decline_scores_below_the_threshold(self) -> None:
        window = np.asarray(_window("2020-05-17"), dtype=float).reshape(-1, 1)
        out = CaseSurgeWindowAdapter().detect(window)
        assert out["flags"].tolist() == [False]
        assert out["scores"][0] < 0.5

    def test_gap_reaches_the_served_verdict(self) -> None:
        window = np.asarray([100 * 1.5 ** (day - 7) for day in range(15)]).reshape(-1, 1)
        window[0, 0] = np.nan
        out = CaseSurgeWindowAdapter().detect(window)
        assert out["flags"].tolist() == [True]
        assert out["partial"] is True
        assert out["set_aside"][0][0]["where"] == "daily_cases"

    def test_too_short_a_clean_run_is_refused(self) -> None:
        window = np.asarray(_window("2020-06-21"), dtype=float).reshape(-1, 1)
        with pytest.raises(InvalidObservationError, match="nothing assessable"):
            CaseSurgeWindowAdapter().detect(window)
