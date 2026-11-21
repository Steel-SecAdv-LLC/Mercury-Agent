# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The watch's time thresholds answer to the records of what they describe.

Two of them were set by judgment: "a condition that returns within three
hours of clearing is the same episode" and "an earthquake stands for a day".
Measured (``research/watch_thresholds/measure.py``; the figures are in
``evidence.json`` beside it):

* Three hours was wrong for floods. Most gauges that pass flood stage more
  than once are tidal and come back at the next high tide, six to
  twenty-four hours later: 57% of all returns. A three-hour window told 658
  "floods" where a twenty-six-hour one tells 138, and one tide station
  alone would have sent 69 messages in a month.
* Three hours was wrong for geomagnetic storms in a different way: Kp is one
  value per three hours, so the commonest return (one interval below G1) is
  exactly three hours, and whether it was merged depended on when the read
  happened to fall.
* A day for an earthquake was right, and the reason is now known.

These tests fail if a value is moved away from where the records put it.
"""

from __future__ import annotations

import json
import statistics
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from omni_mercury_engine.watch import FEEDS
from omni_mercury_engine.watch.feeds import EarthquakeFeed, RiverGaugeFeed, SpaceWeatherFeed
from omni_mercury_engine.watch.store import WatchStore
from research.watch_thresholds.measure import episodes_told, report, share_with_reports

EVIDENCE: dict[str, Any] = json.loads(
    (
        Path(__file__).resolve().parents[2] / "research" / "watch_thresholds" / "evidence.json"
    ).read_text(encoding="utf-8")
)


def _hours(window: timedelta | None) -> float:
    assert window is not None
    return window.total_seconds() / 3600.0


class TestRiverGauges:
    river: dict[str, Any] = EVIDENCE["river_gauges"]

    def test_the_record_is_large_enough_to_say(self) -> None:
        assert self.river["gauges"] > 1000 and self.river["readings"] > 5_000_000
        assert self.river["gauges_that_flooded"] >= 90 and len(self.river["returns"]) >= 700

    def test_the_window_sits_in_the_gap_the_record_shows(self) -> None:
        """No gauge came back between 26 and 30 hours; the tides are all before it."""
        hours = [entry[0] for entry in self.river["returns"]]
        window = _hours(RiverGaugeFeed.same_episode)

        assert window == 26.0
        assert not [h for h in hours if window < h <= 30.0]
        tidal = sum(1 for h in hours if 6.0 < h <= window)
        assert tidal / len(hours) > 0.55  # the two tidal rhythms are most of the returns
        assert sum(1 for h in hours if h <= window) / len(hours) > 0.94
        later = [h for h in hours if h > window]
        assert statistics.median(later) > 72.0  # what is left is another flood, days on

    def test_three_hours_would_have_told_the_same_flood_again_and_again(self) -> None:
        returns, runs = self.river["returns"], self.river["flood_runs"]

        assert episodes_told(returns, runs, 3.0) == 658
        assert episodes_told(returns, runs, _hours(RiverGaugeFeed.same_episode)) == 138
        assert episodes_told(returns, runs, 48.0) >= 130  # a longer window gains almost nothing

    def test_a_return_within_three_hours_is_the_river_still_up(self) -> None:
        graded = [entry for entry in self.river["returns"] if entry[1] is not None]
        soon = [entry for entry in graded if entry[0] <= 3.0]
        tide = [entry for entry in graded if 6.0 < entry[0] <= 26.0]

        assert sum(entry[1] for entry in soon) / len(soon) > 0.75
        assert sum(entry[1] for entry in tide) / len(tide) < 0.05  # the water went right out


class TestGeomagneticStorms:
    kp: dict[str, Any] = EVIDENCE["geomagnetic"]

    def test_the_record_is_the_whole_index(self) -> None:
        assert self.kp["first"] == "1932-01-01" and self.kp["intervals"] > 270_000
        assert self.kp["storm_runs"] == 8339  # by the library's own G scale: G1 from Kp 5-

    def test_the_window_covers_the_dips_that_are_more_often_than_not_the_same_storm(
        self,
    ) -> None:
        returns = self.kp["returns_by_intervals_below"]

        def same(intervals: int) -> float:
            count, stayed = returns[str(intervals)]
            return float(stayed) / float(count)

        assert same(1) > 0.8 and same(2) > 0.5  # activity stayed at Kp 4
        assert same(3) < 0.5 and same(4) < 0.2
        window = _hours(SpaceWeatherFeed.same_episode)
        # Two three-hour intervals, and clear of the grain on both sides: a
        # return after two intervals is always inside it, one after three never.
        assert 6.0 + 1.0 <= window <= 9.0 - 1.0

    def test_the_commonest_return_is_exactly_the_index_grain(self) -> None:
        """Why three hours could not be right: it is the size of one interval."""
        returns = self.kp["returns_by_intervals_below"]
        commonest = max(returns, key=lambda gap: returns[gap][0])

        assert commonest == "1" and returns["1"] == [1638, 1376]
        assert SpaceWeatherFeed.same_episode != timedelta(hours=3)


class TestEarthquakes:
    quakes: dict[str, Any] = EVIDENCE["earthquakes"]

    def test_the_record_is_large_enough_to_say(self) -> None:
        assert self.quakes["with_a_product"] > 300
        assert len(self.quakes["first_shakemap_hours"]) > 150
        assert len(self.quakes["report_shares"]) > 50

    def test_a_day_is_where_the_felt_reports_level_off(self) -> None:
        shares, hours = self.quakes["report_shares"], self.quakes["report_hours"]
        stands = int(_hours(WatchStore.OCCURRENCE_STANDS))

        assert stands == 24
        by_half_a_day = share_with_reports(shares, hours, 12, 0.9)
        by_a_day = share_with_reports(shares, hours, stands, 0.9)
        by_two_days = share_with_reports(shares, hours, 48, 0.9)
        # 57.6% by twelve hours, 84.7% by twenty-four, 86.4% by forty-eight.
        assert by_half_a_day < 0.6 and by_a_day > 0.84  # half a day would be too short
        assert by_two_days - by_a_day < 0.03  # a second day adds almost nothing

    def test_the_felt_extent_is_read_often_while_it_is_arriving_and_seldom_after(self) -> None:
        maps = self.quakes["first_shakemap_hours"]
        shares, hours = self.quakes["report_shares"], self.quakes["report_hours"]
        soon = _hours(EarthquakeFeed.REACH_SOON)

        assert soon == 3.0
        assert sum(1 for m in maps if m <= soon) / len(maps) > 0.9  # the first ShakeMaps are in
        assert statistics.median(row[hours.index(3)] for row in shares) > 0.7
        assert EarthquakeFeed.REACH_EVERY < EarthquakeFeed.REACH_LATER_EVERY
        day = _hours(WatchStore.OCCURRENCE_STANDS)
        reads = soon / _hours(EarthquakeFeed.REACH_EVERY) + (day - soon) / _hours(
            EarthquakeFeed.REACH_LATER_EVERY
        )
        assert reads == 39.0 and day / _hours(EarthquakeFeed.REACH_EVERY) == 144.0


def test_the_storm_record_was_counted_by_the_rule_the_feed_uses() -> None:
    """The first count took G1 from Kp 5.0; NOAA's scale, and the feed, take it from 5-."""
    from omni_mercury_engine.evaluation.hazard_metrics import noaa_g_scale, parse_kp

    assert noaa_g_scale(parse_kp("5-")) == 1 and noaa_g_scale(parse_kp("4+")) == 0
    # Counted from 5.0 the record has 6,107 runs; from 5- it has 8,339.
    assert EVIDENCE["geomagnetic"]["storm_runs"] != 6107


def test_a_feed_that_has_measured_nothing_merges_one_missed_read_and_no_more() -> None:
    measured = {name for name, feed in FEEDS.items() if feed.same_episode is not None}

    assert measured == {"river_gauges", "space_weather"}
    for name, feed in FEEDS.items():
        if name not in measured:
            assert feed().episode_window() == 2 * feed.cadence, name


def test_the_tables_are_printed_from_the_committed_evidence() -> None:
    printed = report(EVIDENCE)

    assert "window 26 h: 884 runs are 138 episodes." in printed
    assert "window 7.5 h covers two intervals." in printed
    assert "an earthquake stands 24 h." in printed


@pytest.mark.parametrize(
    ("window", "told"), [(0.0, 884), (3.0, 658), (12.0, 287), (26.0, 138), (72.0, 126)]
)
def test_episodes_told_under_a_window(window: float, told: int) -> None:
    river = EVIDENCE["river_gauges"]

    assert episodes_told(river["returns"], river["flood_runs"], window) == told
