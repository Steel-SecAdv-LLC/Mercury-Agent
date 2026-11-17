# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The feeds the watch reads, and how each decides what is worth saying.

A feed reads one live source through its loader and judges the result. The
judgment is deliberately not "score the batch and flag the top rows": a batch
scored against itself flags about one row in twenty whatever it holds. Each
feed instead uses the rule its field already uses:

* **Earthquakes** by USGS magnitude class.
* **Geomagnetic activity** by the NOAA G scale.
* **Tsunami buoys** by the buoy's own event mode, which its tsunameter enters
  when it detects a disturbance.
* **River gauges** by the flood category the National Weather Service sets
  for each gauge.
* **Volcanoes** by the USGS volcano alert levels.
* **Warnings** are the National Weather Service's own alerts, relayed.
* **Disaster declarations** by their count against the year's own weeks.

Each feed also knows the later record that settles one of its findings
(``review``): the reviewed catalogue entry, the final Kp, the tsunami event
database, the gauge's own record.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd

from omni_mercury_engine.evaluation.hazard_metrics import noaa_g_scale
from omni_mercury_engine.watch.model import Finding, Outcome, Severity

if TYPE_CHECKING:
    from omni_mercury_engine.watch.store import WatchStore

logger = logging.getLogger(__name__)

__all__ = ["FEEDS", "Feed", "Judgment"]

#: USGS event lookup: the catalogue's current entry for one event id.
_USGS_EVENT_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
#: NOAA NCEI Global Historical Tsunami Database, event list.
_NCEI_TSUNAMI_URL = "https://www.ngdc.noaa.gov/hazel/hazard-service/api/v1/tsunamis/events"
#: NCEI validity codes that mean a tsunami did occur.
_TSUNAMI_VALID = {3: "probable", 4: "definite"}


@dataclass
class Judgment:
    """What a feed made of one read.

    Attributes:
        findings: What is worth saying. Usually empty.
        summary: One line on what was seen, including "nothing notable".
        latest_reading: Time of the newest reading (UTC).
        readings: Rows read.
        subjects: Distinct stations, gauges or events among them.
        partial: The read could not look at everything it covers (a lookup
            failed). What it does not show may only be what it could not
            see, so no standing finding is ended on its account.
    """

    findings: list[Finding] = field(default_factory=list)
    summary: str = "nothing notable"
    latest_reading: datetime | None = None
    readings: int = 0
    subjects: int = 0
    partial: bool = False


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
    #: How long after a finding is first seen a later record is worth consulting.
    review_after: ClassVar[timedelta] = timedelta(hours=6)
    #: How often to consult it again while the record is not settled.
    review_every: ClassVar[timedelta] = timedelta(hours=12)
    #: After this long the last answer stands and the finding is not checked again.
    review_until: ClassVar[timedelta] = timedelta(days=7)

    def read(self) -> pd.DataFrame:
        """Read the live source. Raises when it cannot be read."""
        raise NotImplementedError

    def review(self, finding: dict[str, Any], now: datetime) -> Outcome | None:
        """Say what a later, authoritative record makes of a finding.

        Args:
            finding: The finding as the store holds it.
            now: Time of the check (UTC).

        Returns:
            The outcome, or ``None`` when the record cannot say yet.
        """
        return None

    def judge(self, frame: pd.DataFrame, now: datetime) -> Judgment:
        """Decide what in ``frame`` is worth saying."""
        raise NotImplementedError

    def bind(self, store: WatchStore) -> None:
        """Give the feed the watch's store. Most feeds need nothing from it."""
        return

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


def _out_of_bounds(exc: Exception) -> bool:
    """Whether ``exc`` is the National Weather Service saying a point is outside its area.

    For a point outside the United States its alert query answers HTTP 400
    with the detail ``Parameter "point" is invalid: out of bounds`` (read
    2026-11-17 for Tokyo and London). Any other refusal is a failure, not an
    answer.
    """
    response = getattr(exc, "response", None)
    if response is None or getattr(response, "status_code", None) != 400:
        return False
    try:
        detail = str(response.json().get("detail", ""))
    except Exception:
        return False
    return "out of bounds" in detail


def _count(number: int, noun: str) -> str:
    """``number`` with ``noun`` in the singular or plural: "1 gauge", "11,435 gauges"."""
    return f"{number:,} {noun}{'' if number == 1 else 's'}"


def _shorten(text: str, limit: int) -> str:
    """``text`` cut at a word to ``limit`` characters, with an ellipsis when cut."""
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",;") + "…"


def _finite(value: Any) -> float | None:
    """``value`` as a float, or ``None`` when it is missing or not a number."""
    number = pd.to_numeric(value, errors="coerce")
    return float(number) if np.isfinite(number) else None


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

    #: How often an earthquake's felt extent is read again while it stands.
    #: USGS revises its shaking map and gathers reports for hours.
    REACH_EVERY: ClassVar[timedelta] = timedelta(minutes=10)
    #: Earthquakes whose felt extent is read in one pass.
    REACH_PER_READ: ClassVar[int] = 12

    def __init__(self) -> None:
        """Start with no store bound."""
        self._store: WatchStore | None = None

    def bind(self, store: WatchStore) -> None:
        """Take the store, to keep each earthquake's felt extent beside its finding."""
        self._store = store

    def read(self) -> pd.DataFrame:
        frame = pd.DataFrame(self._loader("EarthquakeLoader").fetch_realtime())
        try:
            self._refresh_reach(frame, datetime.now(UTC))
        except Exception as exc:  # where it was felt is an addition; the read stands without it
            logger.warning("watch: could not refresh where earthquakes were felt: %s", exc)
        return frame

    def _refresh_reach(self, frame: pd.DataFrame, now: datetime) -> int:
        """Read where USGS says each reportable earthquake was felt. Returns how many were read.

        The earthquakes in this read at the reporting magnitude, and those
        already standing (USGS publishes its shaking map some minutes after
        the earthquake and gathers reports for hours, after it has left the
        hourly feed).
        """
        from omni_mercury_engine.watch.shaking import read_reach

        if self._store is None:
            return 0
        subjects: list[str] = []
        if not frame.empty and "event_id" in frame:
            strong = pd.to_numeric(frame["magnitude"], errors="coerce") >= self.NOTABLE
            subjects += [str(event) for event in frame.loc[strong, "event_id"] if event]
        subjects += [
            str(finding["subject"])
            for finding in self._store.standing(now=now, feed=self.name)
            if finding.get("kind") == "earthquake"
        ]
        read = 0
        for subject in dict.fromkeys(subjects):
            if read >= self.REACH_PER_READ:
                break
            last = self._store.reach_read_at(subject)
            if last is not None and now - last < self.REACH_EVERY:
                continue
            try:
                self._store.save_reach(subject, read_reach(subject), now)
                read += 1
            except Exception as exc:  # one earthquake's products failing costs no other
                logger.warning("watch: no felt extent for %s: %s", subject, exc)
        return read

    def judge(self, frame: pd.DataFrame, now: datetime) -> Judgment:
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
            f"{_count(len(frame), 'earthquake')} in the past hour, "
            f"the largest magnitude {largest:.1f}; "
            + (
                f"{len(findings)} at magnitude {self.NOTABLE} or above"
                if findings
                else f"none at magnitude {self.NOTABLE} or above"
            )
        )
        return Judgment(findings, summary, _utc(times.max()), len(frame), len(frame))

    review_after = timedelta(hours=2)
    review_until = timedelta(days=14)

    def review(self, finding: dict[str, Any], now: datetime) -> Outcome | None:
        """Compare the reported magnitude with the USGS catalogue's current entry."""
        from omni_mercury_engine.loaders.base import FetchHTTPError

        record = "USGS earthquake catalogue"
        try:
            entry = self._loader("EarthquakeLoader")._fetch_json(
                _USGS_EVENT_URL, {"format": "geojson", "eventid": str(finding["subject"])}
            )
        except FetchHTTPError as exc:
            if exc.status_code in (404, 409):
                return Outcome(
                    "withdrawn",
                    "USGS no longer lists this event: an automatic solution that did not "
                    "survive review.",
                    record,
                    final=True,
                )
            raise
        properties = entry.get("properties", {}) if isinstance(entry, dict) else {}
        magnitude = pd.to_numeric(properties.get("mag"), errors="coerce")
        status = str(properties.get("status") or "automatic")
        if status == "deleted":
            return Outcome("withdrawn", "USGS deleted this event after review.", record, final=True)
        if not np.isfinite(magnitude):
            return None
        reported = float(finding["value"])
        settled = status == "reviewed"
        standing = "reviewed by a seismologist" if settled else "still an automatic solution"
        if abs(float(magnitude) - reported) < 0.05:
            return Outcome(
                "confirmed",
                f"USGS lists it at magnitude {magnitude:.1f}, as reported ({standing}).",
                record,
                final=settled,
            )
        below = float(magnitude) < self.NOTABLE
        return Outcome(
            "revised",
            f"USGS now lists it at magnitude {magnitude:.1f}; it was reported at "
            f"{reported:.1f} ({standing})."
            + (f" That is below the {self.NOTABLE} at which the watch reports." if below else ""),
            record,
            final=settled,
        )


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

    def judge(self, frame: pd.DataFrame, now: datetime) -> Judgment:
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

    review_until = timedelta(days=3)

    def review(self, finding: dict[str, Any], now: datetime) -> Outcome | None:
        """Compare the reported Kp with what NOAA lists for that interval now."""
        frame = self.read()
        if frame.empty or "kp" not in frame:
            return None
        observed = pd.Timestamp(finding["observed_at"])
        rows = frame[pd.to_datetime(frame["timestamp"], utc=True) == observed].dropna(subset=["kp"])
        if rows.empty:
            return None
        kp, reported = float(rows["kp"].iloc[-1]), float(finding["value"])
        level, was = noaa_g_scale(min(max(kp, 0.0), 9.0)), noaa_g_scale(
            min(max(reported, 0.0), 9.0)
        )
        record = "NOAA SWPC planetary Kp record"
        settled = now - observed.to_pydatetime() >= timedelta(days=2)
        if level == was:
            return Outcome(
                "confirmed",
                f"NOAA lists Kp {kp:.1f} for that interval: G{level}, as reported.",
                record,
                final=settled,
            )
        return Outcome(
            "revised",
            f"NOAA now lists Kp {kp:.1f} for that interval, which is "
            + (f"G{level}" if level else "below storm level")
            + f"; it was reported as G{was} at Kp {reported:.1f}.",
            record,
            final=settled,
        )


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

    def judge(self, frame: pd.DataFrame, now: datetime) -> Judgment:
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
        summary = f"{_count(len(stations), 'buoy')} reporting; " + (
            f"in event mode: {', '.join(triggered)}" if triggered else "none in event mode"
        )
        return Judgment(findings, summary, latest, len(frame), len(stations))

    review_after = timedelta(hours=12)
    review_every = timedelta(days=3)
    review_until = timedelta(days=45)

    #: A tsunami reaches a deep-ocean buoy within a day of its source.
    TRAVEL: ClassVar[timedelta] = timedelta(hours=24)

    def review(self, finding: dict[str, Any], now: datetime) -> Outcome | None:
        """Look for a tsunami in NOAA NCEI's event database around the trigger."""
        seen = pd.Timestamp(finding["observed_at"])
        earliest = seen - self.TRAVEL - self.RECENT
        listing = self._loader("TsunamiLoader")._fetch_json(
            _NCEI_TSUNAMI_URL, {"minYear": str(earliest.year), "maxYear": str(seen.year)}
        )
        items = listing.get("items", []) if isinstance(listing, dict) else []
        record = "NOAA NCEI Global Historical Tsunami Database"
        for item in items:
            if item.get("eventValidity") not in _TSUNAMI_VALID:
                continue
            try:
                day = pd.Timestamp(
                    year=int(item["year"]), month=int(item["month"]), day=int(item["day"]), tz="UTC"
                )
            except (KeyError, TypeError, ValueError):
                continue
            hour = item.get("hour")
            began = (
                day + pd.Timedelta(hours=int(hour), minutes=int(item.get("minute") or 0))
                if hour is not None
                else day
            )
            ended = began if hour is not None else day + pd.Timedelta(days=1)
            if ended < earliest or began > seen:
                continue
            place = str(item.get("locationName") or "location not given").title()
            height = pd.to_numeric(item.get("maxWaterHeight"), errors="coerce")
            return Outcome(
                "confirmed",
                f"NCEI lists a {_TSUNAMI_VALID[item['eventValidity']]} tsunami at {place} on "
                f"{began:%d %B %Y}"
                + (f", {began:%H:%M} UTC" if hour is not None else "")
                + (
                    f", with a largest recorded water height of {height:.2f} m"
                    if np.isfinite(height)
                    else ""
                )
                + ". The buoy's trigger falls within a day of it.",
                record,
                final=True,
            )
        return Outcome(
            "unconfirmed",
            "NCEI lists no tsunami in the day before the buoy triggered. Its database is "
            "updated weeks after an event, and a buoy also triggers on seismic shaking and "
            "on tests.",
            record,
        )


# --------------------------------------------------------------------------- #
# River gauges
# --------------------------------------------------------------------------- #


class RiverGaugeFeed(Feed):
    """NWS river gauges: a gauge the National Weather Service classes as in flood."""

    name = "river_gauges"
    title = "River gauges in flood (NOAA National Water Prediction Service)"
    source = "NOAA National Weather Service, National Water Prediction Service"
    cadence = timedelta(minutes=30)
    stale_after = timedelta(hours=6)

    #: NWS flood category to severity. "action" stage is below flood and is counted, not reported.
    SEVERITY: ClassVar[dict[str, Severity]] = {
        "minor": Severity.NOTABLE,
        "moderate": Severity.SIGNIFICANT,
        "major": Severity.SEVERE,
    }
    #: A reading older than this is the gauge's last word, not its present state.
    CURRENT: ClassVar[timedelta] = timedelta(hours=6)
    #: Two readings of a stage this close (in the gauge's unit) are the same reading.
    SAME: ClassVar[float] = 0.02

    def read(self) -> pd.DataFrame:
        return pd.DataFrame(self._loader("FloodLoader").fetch_flood_status())

    def judge(self, frame: pd.DataFrame, now: datetime) -> Judgment:
        if frame.empty:
            return Judgment(summary="no gauge returned a reading")
        times = pd.to_datetime(frame["observed_at"], utc=True, errors="coerce")
        # Stamp before filtering: assigning a series to an empty selection would bring its rows back.
        frame = frame.assign(read_at=times)
        current = frame[times >= pd.Timestamp(now - self.CURRENT)]
        if current.empty:
            latest = _utc(times.max()) if times.notna().any() else None
            return Judgment([], "no gauge has a current reading", latest, len(frame), 0)
        findings = []
        for row in current[current["category"].isin(list(self.SEVERITY))].itertuples(index=False):
            category, when = str(row.category), _utc(row.read_at)
            place = f"{row.name}, {row.state}" if row.state else str(row.name)
            level = f"{row.stage:g} {row.unit}".strip() if np.isfinite(row.stage) else "a level"
            forecast = str(row.forecast_category)
            ahead = (
                f" Its forecast reaches {forecast} flooding."
                if forecast in self.SEVERITY and forecast != category
                else ""
            )
            findings.append(
                Finding(
                    feed=self.name,
                    kind="river_flood",
                    subject=str(row.gauge_id),
                    severity=self.SEVERITY[category],
                    headline=f"{category.capitalize()} flooding: {place}",
                    detail=(
                        f"The gauge at {place} (NWS {row.gauge_id}) read {level} at "
                        f"{_clock(when)}, which the National Weather Service classes as "
                        f"{category} flooding at this gauge.{ahead} Flood stages are set gauge "
                        "by gauge by the local forecast office. Mercury reports the reading and "
                        "its category; flood warnings come from the National Weather Service."
                    ),
                    observed_at=when,
                    source=self.source,
                    basis=(
                        f"NWS flood category at this gauge: {category} "
                        "(minor flooding and above is reported)"
                    ),
                    value=float(row.stage) if np.isfinite(row.stage) else None,
                    unit=str(row.unit),
                    latitude=_finite(row.latitude),
                    longitude=_finite(row.longitude),
                    ongoing=True,
                )
            )
        counts = current["category"].value_counts()
        in_flood = [f"{int(counts[c])} {c}" for c in ("major", "moderate", "minor") if c in counts]
        summary = f"{_count(len(current), 'gauge')} reporting; " + (
            f"{len(findings)} in flood ({', '.join(in_flood)})" if findings else "none in flood"
        )
        if int(counts.get("action", 0)):
            summary += f"; {int(counts['action'])} at action stage, below flood"
        return Judgment(findings, summary, _utc(current["read_at"].max()), len(frame), len(current))

    review_until = timedelta(days=7)

    def review(self, finding: dict[str, Any], now: datetime) -> Outcome | None:
        """Re-read the gauge's record: does it still show that level, and how far did it go."""
        reported = finding.get("value")
        if reported is None:
            return None
        stages, record = self._loader("FloodLoader").fetch_gauge_record(str(finding["subject"]))
        observed = pd.Timestamp(finding["observed_at"])
        if record.empty or record["observed_at"].max() < observed:
            return None
        unit = str(finding.get("unit") or "")
        source = "NWS record of the same gauge"
        settled = finding.get("ended") is not None

        def category(level: float) -> str:
            reached = [
                name for name in ("minor", "moderate", "major") if level >= stages.get(name, np.inf)
            ]
            return f"{reached[-1]} flooding" if reached else "below flood stage"

        since = record[record["observed_at"] >= observed]
        peak_row = since.loc[since["stage"].idxmax()]
        peak, latest = float(peak_row["stage"]), float(since["stage"].iloc[-1])
        course = (
            f"It went on to {peak:g} {unit} at {_clock(_utc(peak_row['observed_at']))}"
            + (f" ({category(peak)})" if stages else "")
            + f" and stands at {latest:g} {unit}"
            if peak > float(reported) + self.SAME
            else f"That is the highest since; it stands at {latest:g} {unit}"
        ) + (f" ({category(latest)})." if stages else ".")
        at_the_time = record[record["observed_at"] == observed]
        if at_the_time.empty:
            return Outcome(
                "withdrawn",
                "The NWS record no longer holds a reading for that time: the provisional value "
                "was removed.",
                source,
                final=settled,
            )
        current = float(at_the_time["stage"].iloc[-1])
        if abs(current - float(reported)) <= self.SAME:
            return Outcome(
                "confirmed",
                f"The NWS record still shows {current:g} {unit} for that time. {course}",
                source,
                final=settled,
            )
        if "minor" in stages and current < stages["minor"]:
            return Outcome(
                "withdrawn",
                f"The NWS record now shows {current:g} {unit} for that time, below this gauge's "
                f"minor flood stage of {stages['minor']:g} {unit}; {float(reported):g} {unit} was "
                "reported.",
                source,
                final=settled,
            )
        return Outcome(
            "revised",
            f"The NWS record now shows {current:g} {unit} for that time; {float(reported):g} "
            f"{unit} was reported. {course}",
            source,
            final=settled,
        )


# --------------------------------------------------------------------------- #
# Volcanoes
# --------------------------------------------------------------------------- #


class VolcanoFeed(Feed):
    """USGS-monitored volcanoes: one above NORMAL on the USGS volcano alert levels."""

    name = "volcanoes"
    title = "U.S. volcano alert levels (USGS Volcano Hazards Program)"
    source = "U.S. Geological Survey, Volcano Hazards Program"
    cadence = timedelta(minutes=30)
    # A quiet volcano's last notice can be years old; the list itself is current when read.
    stale_after = timedelta(days=36500)

    SEVERITY: ClassVar[dict[str, Severity]] = {
        "ADVISORY": Severity.NOTABLE,
        "WATCH": Severity.SIGNIFICANT,
        "WARNING": Severity.SEVERE,
    }
    #: What each level means, after the USGS definitions of its volcano alert levels.
    MEANING: ClassVar[dict[str, str]] = {
        "ADVISORY": "elevated unrest above its known background, or activity that has "
        "decreased from a higher level and is still closely watched",
        "WATCH": "heightened or escalating unrest with increased potential of eruption, or an "
        "eruption under way that poses limited hazards",
        "WARNING": "a hazardous eruption that is imminent, under way or suspected",
    }

    def read(self) -> pd.DataFrame:
        return pd.DataFrame(self._loader("VolcanicLoader").fetch_realtime())

    def judge(self, frame: pd.DataFrame, now: datetime) -> Judgment:
        if frame.empty:
            return Judgment(summary="USGS listed no monitored volcano")
        levels = frame["alert_level"].astype(str).str.upper()
        findings = []
        for row in frame[levels.isin(list(self.SEVERITY))].itertuples(index=False):
            level, colour = str(row.alert_level).upper(), str(row.color_code).upper()
            issued = pd.to_datetime(row.alert_date, utc=True, errors="coerce")
            when = _utc(issued) if pd.notna(issued) else now
            observatory = f" ({row.observatory})" if str(row.observatory) else ""
            findings.append(
                Finding(
                    feed=self.name,
                    kind="volcano_alert",
                    subject=str(row.volcano_number) or str(row.volcano_name),
                    severity=self.SEVERITY[level],
                    headline=f"{row.volcano_name} volcano: USGS alert level {level}",
                    detail=(
                        f"USGS has {row.volcano_name} at alert level {level}"
                        + (f", aviation colour code {colour}" if colour else "")
                        + (
                            f", by the notice of {_clock(when)}{observatory}"
                            if pd.notna(issued)
                            else ""
                        )
                        + f". USGS uses {level} for {self.MEANING[level]}. Mercury reports the "
                        "level USGS set; it does not assess the volcano."
                    ),
                    observed_at=when,
                    source=self.source,
                    basis=f"USGS volcano alert level {level} (ADVISORY and above is reported)",
                    value=float(row.alert_level_numeric),
                    latitude=_finite(getattr(row, "latitude", None)),
                    longitude=_finite(getattr(row, "longitude", None)),
                    ongoing=True,
                    reference=str(row.notice_id),
                )
            )
        counts = levels.value_counts()
        above = [
            f"{int(counts[lv])} {lv}" for lv in ("WARNING", "WATCH", "ADVISORY") if lv in counts
        ]
        summary = f"{len(frame)} monitored volcano{'' if len(frame) == 1 else 'es'}; " + (
            f"{len(findings)} above NORMAL ({', '.join(above)})" if findings else "all at NORMAL"
        )
        return Judgment(findings, summary, None, len(frame), len(frame))

    review_after = timedelta(hours=1)
    review_until = timedelta(days=3)

    def review(self, finding: dict[str, Any], now: datetime) -> Outcome | None:
        """Read the observatory's notice: does it give this volcano that level."""
        from omni_mercury_engine.loaders.base import FetchHTTPError

        notice_id = str(finding.get("reference") or "")
        if not notice_id:
            return None
        record = "observatory's notice in the USGS Hazard Notification System"
        try:
            notice = self._loader("VolcanicLoader").fetch_notice(notice_id)
        except FetchHTTPError as exc:
            if exc.status_code in (400, 404):
                return Outcome(
                    "withdrawn",
                    "USGS no longer holds the notice the level came from.",
                    record,
                    True,
                )
            raise
        sections = notice.get("notice_sections")
        about = [
            section
            for section in (sections if isinstance(sections, list) else [])
            if isinstance(section, dict) and str(section.get("vnum")) == str(finding["subject"])
        ]
        if not about:
            return None
        stated = str(about[0].get("alertLevel") or "").upper()
        reported = next(
            (lv for lv in self.SEVERITY if f"alert level {lv}" in str(finding.get("basis"))), ""
        )
        synopsis = " ".join(str(about[0].get("synopsis") or "").split())
        said = f' It reads: "{synopsis}"' if synopsis else ""
        title = str(notice.get("notice_type_title") or "notice").lower()
        if stated == reported:
            return Outcome(
                "confirmed",
                f"The observatory's {title} gives {stated}, as reported.{said}",
                record,
                True,
            )
        return Outcome(
            "revised",
            f"The observatory's {title} gives {stated or 'no level'}; {reported} was reported.{said}",
            record,
            True,
        )


# --------------------------------------------------------------------------- #
# Official warnings
# --------------------------------------------------------------------------- #

_NWS_ALERTS_URL = "https://api.weather.gov/alerts/active"
_NWS_ALERT_URL = "https://api.weather.gov/alerts/"
_NWS_HEADERS = {"Accept": "application/geo+json"}


class WarningsFeed(Feed):
    """National Weather Service alerts in effect: the official warnings themselves.

    Every other feed reports a measurement and says "warnings come from the
    National Weather Service". This one relays them: nationwide, every alert
    the Service rates Extreme (its word for an extraordinary threat to life
    or property: a tornado warning, a flash flood emergency), and, for each
    place a subscriber has given, every alert in effect at that point.
    """

    name = "warnings"
    title = "Warnings in effect (National Weather Service)"
    source = "NOAA National Weather Service"
    cadence = timedelta(minutes=5)
    stale_after = timedelta(days=36500)  # a quiet day has no recent alert; the list is current

    SEVERITY: ClassVar[dict[str, Severity]] = {
        "Extreme": Severity.SEVERE,
        "Severe": Severity.SIGNIFICANT,
    }
    #: Distinct places looked up per read. Each is one request to the Service.
    MAX_PLACES: ClassVar[int] = 250
    _COLUMNS: ClassVar[list[str]] = [
        "id", "event", "severity", "urgency", "certainty", "area", "headline",
        "instruction", "sender", "sent", "until", "place_ids",
    ]  # fmt: skip

    def __init__(self) -> None:
        self._store: WatchStore | None = None
        #: Places the Service has refused as outside its area, learned as they are asked.
        self._outside: set[tuple[float, float]] = set()

    def bind(self, store: WatchStore) -> None:
        self._store = store

    def _alerts(self, params: dict[str, str]) -> list[dict[str, Any]]:
        from omni_mercury_engine.security.safe_http import SafeHTTPClient

        payload = SafeHTTPClient.get_json(
            _NWS_ALERTS_URL, params=params, headers=_NWS_HEADERS, timeout=30.0
        )
        features = payload.get("features") if isinstance(payload, dict) else None
        if not isinstance(features, list):
            raise ValueError("NWS alerts are not the documented shape: no 'features' list")
        return [
            feature["properties"]
            for feature in features
            if isinstance(feature, dict) and isinstance(feature.get("properties"), dict)
        ]

    def read(self) -> pd.DataFrame:
        rows: dict[str, dict[str, Any]] = {}

        def take(alert: dict[str, Any], place: int | None) -> None:
            identifier = str(alert.get("id") or "")
            if not identifier or alert.get("messageType") == "Cancel":
                return
            row = rows.setdefault(
                identifier,
                {
                    "id": identifier,
                    "event": str(alert.get("event") or "Alert"),
                    "severity": str(alert.get("severity") or "Unknown"),
                    "urgency": str(alert.get("urgency") or ""),
                    "certainty": str(alert.get("certainty") or ""),
                    "area": str(alert.get("areaDesc") or ""),
                    "headline": str(alert.get("headline") or ""),
                    "instruction": " ".join(str(alert.get("instruction") or "").split()),
                    "sender": str(alert.get("senderName") or "National Weather Service"),
                    "sent": alert.get("sent"),
                    "until": alert.get("ends") or alert.get("expires"),
                    "place_ids": [],
                },
            )
            if place is not None and place not in row["place_ids"]:
                row["place_ids"].append(place)

        for alert in self._alerts({"status": "actual"}):
            take(alert, None)
        subscribed = self._store.places() if self._store is not None else {}
        # A place the Service has said is outside its area is not asked about
        # again, and takes none of the places one read may look up.
        points = {point: ids for point, ids in subscribed.items() if point not in self._outside}
        # The nationwide relay never depends on the places. More places than
        # one read may look up, or a place the Service will not answer for,
        # used to raise here: every subscriber then lost every warning, the
        # Extreme ones included, for as long as that place was subscribed.
        beyond = max(0, len(points) - self.MAX_PLACES)
        asked = unanswered = 0
        for point, ids in list(points.items())[: self.MAX_PLACES]:
            try:
                here = self._alerts({"status": "actual", "point": f"{point[0]},{point[1]}"})
            except Exception as exc:
                if _out_of_bounds(exc):
                    self._outside.add(point)
                    continue
                asked += 1
                unanswered += 1
                logger.warning("watch: no alert lookup for a subscribed place: %s", exc)
                continue
            asked += 1
            for alert in here:
                for place in ids:
                    take(alert, place)
        if asked and unanswered == asked:
            # Every lookup failed: that is the Service's point query being
            # down, and saying "no alert in effect" there would be a guess.
            raise ConnectionError(
                f"the alert lookup failed for all {unanswered} subscribed places checked"
            )
        frame = pd.DataFrame(list(rows.values()), columns=self._COLUMNS)
        frame.attrs["places"] = asked - unanswered
        frame.attrs["places_unanswered"] = unanswered
        frame.attrs["places_beyond"] = beyond
        frame.attrs["places_outside"] = sum(1 for point in subscribed if point in self._outside)
        return frame

    def judge(self, frame: pd.DataFrame, now: datetime) -> Judgment:
        places = int(frame.attrs.get("places", 0))
        watched = f"; {_count(places, 'subscribed place')} checked" if places else ""
        unanswered = int(frame.attrs.get("places_unanswered", 0))
        if unanswered:
            watched += f"; {_count(unanswered, 'place')} could not be checked this time"
        outside = int(frame.attrs.get("places_outside", 0))
        if outside:
            watched += (
                f"; {_count(outside, 'subscribed place')} outside the Service's area "
                "(it covers the United States)"
            )
        beyond = int(frame.attrs.get("places_beyond", 0))
        if beyond:
            watched += (
                f"; {_count(beyond, 'place')} NOT checked: more are subscribed than the "
                f"{self.MAX_PLACES} one read looks up"
            )
        if frame.empty:
            return Judgment([], f"no alert in effect{watched}", None, 0, 0, partial=unanswered > 0)
        findings = []
        for row in frame.itertuples(index=False):
            here = tuple(int(place) for place in row.place_ids)
            if row.severity != "Extreme" and not here:
                continue
            sent = pd.to_datetime(row.sent, utc=True, errors="coerce")
            until = pd.to_datetime(row.until, utc=True, errors="coerce")
            rated = ", ".join(
                part for part in (row.severity, row.urgency, row.certainty) if part
            ).lower()
            findings.append(
                Finding(
                    feed=self.name,
                    kind="nws_alert",
                    subject=str(row.id),
                    severity=self.SEVERITY.get(str(row.severity), Severity.NOTABLE),
                    headline=f"{row.event}: {_shorten(str(row.area), 90)}",
                    detail=(
                        (
                            f"{row.headline}. "
                            if row.headline
                            else f"{row.event} from {row.sender}. "
                        )
                        + (f"In effect until {_clock(_utc(until))}. " if pd.notna(until) else "")
                        + f"The National Weather Service rates it {rated}. This is the "
                        "Service's own alert, relayed as issued."
                    ),
                    observed_at=_utc(sent) if pd.notna(sent) else now,
                    source=str(row.sender),
                    basis=(
                        "an NWS alert in effect at a subscribed place"
                        if here
                        else "an NWS alert rated Extreme (every Extreme alert is reported)"
                    ),
                    ongoing=True,
                    reference=str(row.id),
                    action=str(row.instruction),
                    action_source=str(row.sender) if row.instruction else "",
                    area=str(row.area),
                    place_ids=here,
                )
            )
        rated_counts = frame["severity"].value_counts()
        summary = (
            f"{_count(len(frame), 'alert')} in effect nationwide ("
            + ", ".join(
                f"{int(rated_counts.get(level, 0))} {level.lower()}"
                for level in ("Extreme", "Severe")
            )
            + f"){watched}"
        )
        at_places = sum(1 for finding in findings if finding.place_ids)
        if at_places:
            summary += f"; {at_places} in effect at subscribed places"
        return Judgment(findings, summary, None, len(frame), len(frame), partial=unanswered > 0)

    review_after = timedelta(minutes=30)
    review_until = timedelta(days=2)

    def review(self, finding: dict[str, Any], now: datetime) -> Outcome | None:
        """Read the Service's own record of the alert: was it issued as relayed."""
        from omni_mercury_engine.security.safe_http import SafeHTTPClient

        record = "National Weather Service record of the alert"
        identifier = str(finding.get("reference") or "")
        if not identifier.startswith("urn:oid:") or not all(
            part.isalnum() for part in identifier[8:].split(".")
        ):
            return None
        try:
            payload = SafeHTTPClient.get_json(
                _NWS_ALERT_URL + identifier, headers=_NWS_HEADERS, timeout=30.0
            )
        except Exception as exc:
            if getattr(getattr(exc, "response", None), "status_code", None) == 404:
                return Outcome(
                    "withdrawn", "The Service holds no alert by that identifier.", record, True
                )
            raise
        held = payload.get("properties") if isinstance(payload, dict) else None
        if not isinstance(held, dict):
            return None
        event, sender = str(held.get("event") or ""), str(held.get("senderName") or "the Service")
        if held.get("status") != "Actual":
            return Outcome(
                "withdrawn",
                f"The Service's record marks it {str(held.get('status')).lower()}, not an actual alert.",
                record,
                True,
            )
        if event and not str(finding.get("headline", "")).startswith(event):
            return Outcome("revised", f"The Service's record holds it as {event}.", record, True)
        return Outcome(
            "confirmed",
            f"The Service's record holds this {event or 'alert'} as issued by {sender}. Mercury "
            "relayed it; whether the hazard occurred is the Service's to verify.",
            record,
            True,
        )


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

    def judge(self, frame: pd.DataFrame, now: datetime) -> Judgment:
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

    review_after = timedelta(days=1)
    review_every = timedelta(days=3)
    review_until = timedelta(days=14)

    def review(self, finding: dict[str, Any], now: datetime) -> Outcome | None:
        """Re-read OpenFEMA: are the same days still declared as reported."""
        frame = self.read()
        if frame.empty:
            return None
        days = pd.to_datetime(frame["timestamp"], utc=True)
        last = pd.Timestamp(finding["observed_at"]).floor("D")
        window = frame[(days <= last) & (days >= last - pd.Timedelta(days=self.RECENT_DAYS))]
        if window.empty:
            return None
        total, reported = int(window["designations"].sum()), int(finding["value"])
        record = "OpenFEMA Disaster Declarations Summaries, re-read"
        settled = now - last.to_pydatetime() >= timedelta(days=10)
        if total == reported:
            return Outcome(
                "confirmed",
                f"OpenFEMA still lists {total} designations for those days.",
                record,
                final=settled,
            )
        return Outcome(
            "revised",
            f"OpenFEMA now lists {total} designations for those days; {reported} were reported. "
            "FEMA amends a declaration as counties are added.",
            record,
            final=settled,
        )


def _feeds() -> dict[str, type[Feed]]:
    """Every feed the watch reads, by name. The world feeds are built on this module."""
    from omni_mercury_engine.watch.world import GdacsFeed, WorldWarningsFeed

    return {
        feed.name: feed
        for feed in (
            EarthquakeFeed,
            SpaceWeatherFeed,
            TsunamiBuoyFeed,
            RiverGaugeFeed,
            VolcanoFeed,
            WarningsFeed,
            WorldWarningsFeed,
            GdacsFeed,
            DeclarationFeed,
        )
    }


#: The feeds the watch reads, by name.
FEEDS: dict[str, type[Feed]] = _feeds()
