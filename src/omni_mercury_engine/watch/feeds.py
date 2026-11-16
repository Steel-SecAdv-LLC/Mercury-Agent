# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The feeds the watch reads, and how each decides what is worth saying.

A feed reads one live source through its loader and judges the result. The
judgment is deliberately not "score the batch and flag the top rows": a batch
scored against itself flags about one row in twenty whatever it holds. Each
feed instead uses the rule its field already uses, where one exists:

* **Earthquakes** by USGS magnitude class.
* **Geomagnetic activity** by the NOAA G scale.
* **Tsunami buoys** by the buoy's own event mode, which its tsunameter enters
  when it detects a disturbance.

and, where the source carries no such scale (a river gauge's flood stage is
not in the USGS reading), by the plainest true statement available: this
series has exceeded everything the watch has recorded for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd

from omni_mercury_engine.evaluation.hazard_metrics import noaa_g_scale
from omni_mercury_engine.watch.model import Finding, Severity

if TYPE_CHECKING:
    from omni_mercury_engine.watch.store import WatchStore

__all__ = ["FEEDS", "Feed", "Judgment"]


@dataclass
class Judgment:
    """What a feed made of one read.

    Attributes:
        findings: What is worth saying. Usually empty.
        summary: One line on what was seen, including "nothing notable".
        latest_reading: Time of the newest reading (UTC).
        readings: Rows read.
        subjects: Distinct stations, gauges or events among them.
    """

    findings: list[Finding] = field(default_factory=list)
    summary: str = "nothing notable"
    latest_reading: datetime | None = None
    readings: int = 0
    subjects: int = 0


class Feed:
    """One live source and its judgment.

    Class attributes:
        name: Short identifier.
        title: What the feed is, for a person.
        source: Who measures it, as a person would cite it.
        cadence: How often it is worth reading.
        stale_after: How old its newest reading may be before the feed is
            called stale. A stale feed has answered; it has nothing recent.
    """

    name: ClassVar[str]
    title: ClassVar[str]
    source: ClassVar[str]
    cadence: ClassVar[timedelta]
    stale_after: ClassVar[timedelta]

    def read(self) -> pd.DataFrame:
        """Read the live source. Raises when it cannot be read."""
        raise NotImplementedError

    def judge(self, frame: pd.DataFrame, store: WatchStore, now: datetime) -> Judgment:
        """Decide what in ``frame`` is worth saying, given what is remembered."""
        raise NotImplementedError

    def _loader(self, name: str) -> Any:
        """Build the feed's loader so that every read is a fresh one.

        The loaders cache a "real-time" read for an hour by default, and the
        DART loader for a day, which suits a benchmark and would have the
        watch re-reading yesterday. The watch gives each loader its own cache
        directory and a lifetime shorter than the feed's cadence, so a read at
        the cadence always reaches the source. It also fails fast: the next
        read is minutes away.
        """
        from omni_mercury_engine import loaders
        from omni_mercury_engine.watch.store import default_state_dir

        loader = getattr(loaders, name)(
            cache_dir=default_state_dir() / "cache" / self.name, max_retries=1
        )
        loader.CACHE_TTL = int(self.cadence.total_seconds() // 2)
        return loader


def _utc(value: Any) -> datetime:
    """A pandas or datetime value as an aware UTC ``datetime``."""
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    moment: datetime = stamp.to_pydatetime()
    return moment


def _clock(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%H:%M UTC on %d %B %Y").replace(" 0", " ")


# --------------------------------------------------------------------------- #
# Earthquakes
# --------------------------------------------------------------------------- #


class EarthquakeFeed(Feed):
    """USGS earthquakes of the past hour, by magnitude class."""

    name = "earthquakes"
    title = "Earthquakes worldwide (USGS)"
    source = "U.S. Geological Survey, Earthquake Hazards Program"
    cadence = timedelta(minutes=5)
    stale_after = timedelta(hours=6)

    #: USGS magnitude classes: light 4-4.9, moderate 5-5.9, strong 6-6.9,
    #: major 7-7.9, great 8 and above. 4.5 is the floor of USGS's own
    #: "significant" worldwide feed.
    NOTABLE: ClassVar[float] = 4.5
    SIGNIFICANT: ClassVar[float] = 6.0
    SEVERE: ClassVar[float] = 7.0
    _CLASSES: ClassVar[tuple[tuple[float, str], ...]] = (
        (8.0, "great"),
        (7.0, "major"),
        (6.0, "strong"),
        (5.0, "moderate"),
        (4.0, "light"),
    )

    def read(self) -> pd.DataFrame:
        return pd.DataFrame(self._loader("EarthquakeLoader").fetch_realtime())

    def judge(self, frame: pd.DataFrame, store: WatchStore, now: datetime) -> Judgment:
        if frame.empty:
            return Judgment(summary="USGS lists no earthquake in the past hour")
        times = pd.to_datetime(frame["time"], unit="ms", utc=True)
        magnitudes = pd.to_numeric(frame["magnitude"], errors="coerce")
        findings = []
        for index in np.flatnonzero((magnitudes >= self.NOTABLE).to_numpy()):
            row = frame.iloc[int(index)]
            magnitude = float(magnitudes.iloc[int(index)])
            severity = (
                Severity.SEVERE
                if magnitude >= self.SEVERE
                else Severity.SIGNIFICANT if magnitude >= self.SIGNIFICANT else Severity.NOTABLE
            )
            name = next(label for floor, label in self._CLASSES if magnitude >= floor)
            when = _utc(times.iloc[int(index)])
            place = str(row.get("place") or "location not yet named")
            depth = pd.to_numeric(row.get("depth"), errors="coerce")
            depth_text = f", {depth:.0f} km deep" if np.isfinite(depth) else ""
            findings.append(
                Finding(
                    feed=self.name,
                    kind="earthquake",
                    subject=str(row.get("event_id") or f"{when.isoformat()}"),
                    severity=severity,
                    headline=f"Magnitude {magnitude:.1f} earthquake, {place}",
                    detail=(
                        f"USGS located a magnitude {magnitude:.1f} ({name}) earthquake "
                        f"{place} at {_clock(when)}{depth_text}. The magnitude is "
                        "preliminary: USGS revises it, sometimes by several tenths, as more "
                        "stations report. Mercury reports the size and place; it does not "
                        "estimate shaking or damage."
                    ),
                    observed_at=when,
                    source=self.source,
                    basis=f"USGS magnitude class: {name} (magnitude {self.NOTABLE} and above is reported)",
                    value=magnitude,
                    unit="magnitude",
                    latitude=float(row["latitude"]) if pd.notna(row.get("latitude")) else None,
                    longitude=float(row["longitude"]) if pd.notna(row.get("longitude")) else None,
                )
            )
        largest = float(magnitudes.max()) if magnitudes.notna().any() else float("nan")
        summary = (
            f"{len(frame)} earthquakes in the past hour, the largest magnitude {largest:.1f}; "
            + (
                f"{len(findings)} at magnitude {self.NOTABLE} or above"
                if findings
                else f"none at magnitude {self.NOTABLE} or above"
            )
        )
        return Judgment(findings, summary, _utc(times.max()), len(frame), len(frame))


# --------------------------------------------------------------------------- #
# Space weather
# --------------------------------------------------------------------------- #


class SpaceWeatherFeed(Feed):
    """Planetary Kp and the solar wind (NOAA SWPC), by the NOAA G scale."""

    name = "space_weather"
    title = "Geomagnetic activity (NOAA SWPC)"
    source = "NOAA Space Weather Prediction Center"
    cadence = timedelta(minutes=15)
    stale_after = timedelta(hours=9)

    _G_NAMES: ClassVar[dict[int, str]] = {
        1: "minor", 2: "moderate", 3: "strong", 4: "severe", 5: "extreme",
    }  # fmt: skip
    _G_EFFECTS: ClassVar[dict[int, str]] = {
        1: "weak power-grid fluctuations and aurora at high latitudes",
        2: "voltage alarms on high-latitude power systems and aurora to about 55 degrees",
        3: "voltage corrections on power systems, satellite drag and navigation errors",
        4: "widespread voltage-control problems and degraded satellite navigation for hours",
        5: "grid collapse or blackouts in places and satellite navigation out for days",
    }

    def read(self) -> pd.DataFrame:
        return pd.DataFrame(self._loader("EnergyLoader").fetch_realtime())

    def judge(self, frame: pd.DataFrame, store: WatchStore, now: datetime) -> Judgment:
        if frame.empty or "kp" not in frame:
            return Judgment(summary="NOAA returned no Kp readings")
        frame = frame.sort_values("timestamp")
        measured = frame.dropna(subset=["kp"])
        if measured.empty:
            return Judgment(summary="NOAA returned no Kp readings", readings=len(frame))
        latest = measured.iloc[-1]
        kp, when = float(latest["kp"]), _utc(latest["timestamp"])
        level = noaa_g_scale(min(max(kp, 0.0), 9.0))
        wind = pd.to_numeric(latest.get("solar_wind_speed"), errors="coerce")
        wind_text = f" Solar wind speed is {wind:.0f} km/s." if np.isfinite(wind) else ""
        findings = []
        if level >= 1:
            severity = (
                Severity.SEVERE
                if level >= 4
                else Severity.SIGNIFICANT if level >= 3 else Severity.NOTABLE
            )
            findings.append(
                Finding(
                    feed=self.name,
                    kind="geomagnetic_storm",
                    subject="planetary",
                    severity=severity,
                    headline=f"G{level} ({self._G_NAMES[level]}) geomagnetic storm, Kp {kp:.1f}",
                    detail=(
                        f"The planetary Kp index is {kp:.1f} for the three hours to "
                        f"{_clock(when)}, which is G{level} ({self._G_NAMES[level]}) on NOAA's "
                        f"scale. At this level NOAA describes {self._G_EFFECTS[level]}."
                        f"{wind_text} This Kp is NOAA's estimate from the stations reporting "
                        "now; the definitive value is published later and can differ by a third "
                        "of a unit."
                    ),
                    observed_at=when,
                    source=self.source,
                    basis=f"NOAA G scale: Kp {kp:.1f} is G{level} (G1 begins at Kp 5)",
                    value=kp,
                    unit="Kp",
                    ongoing=True,
                )
            )
        summary = f"Kp {kp:.1f}" + (
            f", G{level}" if level else ", below storm level (G1 begins at Kp 5)"
        )
        if np.isfinite(wind):
            summary += f"; solar wind {wind:.0f} km/s"
        return Judgment(findings, summary, when, len(frame), 1)


# --------------------------------------------------------------------------- #
# Tsunami buoys
# --------------------------------------------------------------------------- #


class TsunamiBuoyFeed(Feed):
    """NOAA DART buoys: a buoy in event mode has itself detected a disturbance."""

    name = "tsunami_buoys"
    title = "Deep-ocean tsunami buoys (NOAA DART)"
    source = "NOAA National Data Buoy Center, DART network"
    cadence = timedelta(minutes=15)
    stale_after = timedelta(hours=6)

    #: A buoy reports every 15 minutes (type 1) and switches to 15-second and
    #: 1-minute samples (types 2 and 3) when its tsunameter detects an event.
    #: Event mode within this period of the read is "now".
    RECENT: ClassVar[timedelta] = timedelta(hours=3)

    def read(self) -> pd.DataFrame:
        return pd.DataFrame(self._loader("TsunamiLoader").fetch_realtime())

    def judge(self, frame: pd.DataFrame, store: WatchStore, now: datetime) -> Judgment:
        if frame.empty:
            return Judgment(summary="no DART buoy returned readings")
        times = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.assign(timestamp=times)
        latest = _utc(times.max())
        stations = sorted(frame["station_id"].astype(str).unique())
        recent = frame[(frame["sample_type"] != 1) & (times >= pd.Timestamp(now) - self.RECENT)]
        triggered = sorted(recent["station_id"].astype(str).unique())
        findings = []
        for station in triggered:
            samples = recent[recent["station_id"].astype(str) == station]
            began, last = _utc(samples["timestamp"].min()), _utc(samples["timestamp"].max())
            swing = float(samples["bpr"].max() - samples["bpr"].min())
            findings.append(
                Finding(
                    feed=self.name,
                    kind="buoy_event_mode",
                    subject=station,
                    severity=Severity.SIGNIFICANT if len(triggered) >= 2 else Severity.NOTABLE,
                    headline=f"DART buoy {station} is in event mode",
                    detail=(
                        f"DART station {station} switched to rapid sampling at {_clock(began)} "
                        f"and was still reporting that way at {_clock(last)}. A buoy does this "
                        "when its seafloor pressure sensor detects a disturbance, which can be a "
                        "tsunami, the seismic wave of a distant earthquake, or a test. The water "
                        f"column varied by {swing:.2f} m over those samples, tide included. "
                        + (
                            f"{len(triggered)} buoys are in event mode at once. "
                            if len(triggered) >= 2
                            else ""
                        )
                        + "This is not a tsunami warning: warnings come from the NOAA tsunami "
                        "warning centers."
                    ),
                    observed_at=last,
                    source=self.source,
                    basis="the buoy's own event mode (sample types 2 and 3 in the NDBC record)",
                    value=swing,
                    unit="m",
                    ongoing=True,
                )
            )
        summary = f"{len(stations)} buoys reporting; " + (
            f"in event mode: {', '.join(triggered)}" if triggered else "none in event mode"
        )
        return Judgment(findings, summary, latest, len(frame), len(stations))


# --------------------------------------------------------------------------- #
# River gauges
# --------------------------------------------------------------------------- #


class RiverGaugeFeed(Feed):
    """USGS river gauges: a gauge above everything recorded for it."""

    name = "river_gauges"
    title = "River gauges (USGS Water Services)"
    source = "U.S. Geological Survey, National Water Information System"
    cadence = timedelta(minutes=15)
    stale_after = timedelta(hours=6)

    #: Readings a gauge needs on record before a new high means anything: two
    #: days at the 15-minute cadence.
    MIN_RECORD: ClassVar[int] = 192
    #: How far above the recorded maximum, in robust standard deviations of
    #: the record, the latest reading must be. Below this a "record" is noise.
    MARGIN: ClassVar[float] = 3.0

    def read(self) -> pd.DataFrame:
        return pd.DataFrame(self._loader("FloodLoader").fetch_realtime())

    def judge(self, frame: pd.DataFrame, store: WatchStore, now: datetime) -> Judgment:
        if frame.empty:
            return Judgment(summary="no gauge returned readings")
        times = pd.to_datetime(frame["datetime"], utc=True)
        frame = frame.assign(_time=times).sort_values("_time")
        findings, sites = [], 0
        for site, readings in frame.groupby(frame["site_id"].astype(str)):
            sites += 1
            heights = pd.to_numeric(readings["gauge_height_ft"], errors="coerce")
            usable = readings[heights.notna()]
            if usable.empty:
                continue
            record = store.remember(
                self.name,
                site,
                "gauge_height_ft",
                [_utc(t) for t in usable["_time"]],
                heights[heights.notna()].to_numpy(),
            )
            if record.latest is None:
                # First sight of this gauge: the read itself becomes its record.
                continue
            new = usable[usable["_time"] > pd.Timestamp(record.latest)]
            if new.empty or len(record) < self.MIN_RECORD:
                continue
            level = float(pd.to_numeric(new["gauge_height_ft"]).max())
            margin = self.MARGIN * max(record.spread, 0.01)
            if level <= record.maximum + margin:
                continue
            when = _utc(new["_time"].iloc[-1])
            name = str(readings["site_name"].iloc[-1]).title()
            findings.append(
                Finding(
                    feed=self.name,
                    kind="gauge_record_high",
                    subject=site,
                    severity=Severity.NOTABLE,
                    headline=f"{name}: highest level Mercury has recorded",
                    detail=(
                        f"The gauge at {name} (USGS {site}) read {level:.2f} ft at "
                        f"{_clock(when)}, above the {record.maximum:.2f} ft maximum of the "
                        f"{len(record)} readings Mercury holds for it (median "
                        f"{record.median:.2f} ft). Mercury does not have this gauge's flood "
                        "stage, so it cannot say whether the river is in flood: the National "
                        "Weather Service sets that level. These are provisional USGS readings."
                    ),
                    observed_at=when,
                    source=self.source,
                    basis=(
                        f"above the gauge's own recorded maximum by more than {self.MARGIN:g} "
                        "robust standard deviations"
                    ),
                    value=level,
                    unit="ft",
                    ongoing=True,
                )
            )
        summary = f"{sites} gauges reporting; " + (
            f"{len(findings)} above their recorded maximum" if findings else "none above its record"
        )
        return Judgment(findings, summary, _utc(times.max()), len(frame), sites)


# --------------------------------------------------------------------------- #
# FEMA declarations
# --------------------------------------------------------------------------- #


class DeclarationFeed(Feed):
    """FEMA tropical-cyclone declarations: new ones, and the year's busiest week."""

    name = "fema_declarations"
    title = "Disaster declarations for tropical cyclones (FEMA)"
    source = "Federal Emergency Management Agency, OpenFEMA"
    cadence = timedelta(hours=1)
    stale_after = timedelta(days=2)

    #: A declaration is news for this long after it is made.
    RECENT_DAYS: ClassVar[int] = 3

    def read(self) -> pd.DataFrame:
        return pd.DataFrame(self._loader("FEMAActivityLoader").fetch_realtime())

    def judge(self, frame: pd.DataFrame, store: WatchStore, now: datetime) -> Judgment:
        if frame.empty:
            return Judgment(summary="OpenFEMA returned no record for this year")
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        days = pd.to_datetime(frame["timestamp"], utc=True)
        counts = frame["designations"].astype(float)
        week = counts.rolling(7, min_periods=1).sum()
        recent = frame[days >= pd.Timestamp(now).floor("D") - pd.Timedelta(days=self.RECENT_DAYS)]
        declared = recent[recent["designations"] > 0]
        findings = []
        if not declared.empty:
            total = int(declared["designations"].sum())
            states = int(declared["states"].max())
            when = _utc(pd.to_datetime(declared["timestamp"], utc=True).max())
            this_week = float(week.iloc[-1])
            busiest = this_week >= float(week.iloc[:-7].max()) if len(week) > 7 else False
            findings.append(
                Finding(
                    feed=self.name,
                    kind="declarations",
                    subject=when.date().isoformat(),
                    severity=Severity.SIGNIFICANT if busiest else Severity.NOTABLE,
                    headline=(
                        f"FEMA declared {total} county designations for tropical cyclones"
                        + (", the busiest week of the year" if busiest else "")
                    ),
                    detail=(
                        f"In the {self.RECENT_DAYS} days to {when.date():%d %B %Y} FEMA declared "
                        f"{total} county designations for hurricanes, tropical storms or "
                        f"typhoons, across up to {states} states or territories in a day. "
                        + (
                            f"The past seven days hold {this_week:.0f} designations, more than "
                            "any earlier week this year. "
                            if busiest
                            else ""
                        )
                        + "A declaration follows a storm by days to weeks: it records the "
                        "federal response, not the storm's arrival."
                    ),
                    observed_at=when,
                    source=self.source,
                    basis="new declarations in the OpenFEMA record"
                    + ("; seven-day count above every earlier week of the year" if busiest else ""),
                    value=float(total),
                    unit="county designations",
                    ongoing=True,
                )
            )
        year_total = int(counts.sum())
        summary = f"{year_total} tropical-cyclone designations this year; " + (
            f"{int(declared['designations'].sum())} in the past {self.RECENT_DAYS} days"
            if not declared.empty
            else f"none in the past {self.RECENT_DAYS} days"
        )
        return Judgment(findings, summary, _utc(days.max()), len(frame), 1)


#: The feeds the watch reads, by name.
FEEDS: dict[str, type[Feed]] = {
    feed.name: feed
    for feed in (EarthquakeFeed, SpaceWeatherFeed, TsunamiBuoyFeed, RiverGaugeFeed, DeclarationFeed)
}
