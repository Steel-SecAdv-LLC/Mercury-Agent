# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Train the TsunamiDetector's WaveformFFTAnalyzer on real DART bottom-pressure data.

Data sources (hook ``tsunami_waveform``,
``TsunamiDetector.load_neural_weights``):

* **NOAA NDBC DART historical bottom-pressure archive**
  (``https://www.ndbc.noaa.gov/data/historical/dart/<station>t<year>.txt.gz``)
  -- real water-column-height records from deep-ocean DART bottom-pressure
  recorders. The station-year file list is parsed from the live directory
  index (some station-years are missing upstream).
* **NOAA NCEI HazEL tsunami event database**
  (``https://www.ngdc.noaa.gov/hazel/hazard-service/api/v1/tsunamis/...``) --
  the authoritative tsunami event catalog. Per-event runup records with
  ``typeMeasurementId == 3`` are deep-ocean DART BPR observations carrying the
  observed arrival time (``arrDay``/``arrHour``/``arrMin``, UTC, day-of-month
  relative to the event date) and the measured maximum wave amplitude at the
  recorder (``runupHt``, metres). The DART station id is parsed from
  ``locationName`` (``D(\d{5})``) and cross-checked against the NDBC station
  table coordinates so a mislabeled record cannot mislabel a waveform.
* **NDBC station table**
  (``https://www.ndbc.noaa.gov/data/stations/station_table.txt``) -- station
  coordinates for that cross-check.

Sample construction (uniform 15-minute grid, 96 samples = 24 h per window):

* The grid holds the standard-mode (``T == 1``) 15-minute samples;
  ``9999.000`` sentinels are missing data. **Documented design change forced
  by the data**: while a DART buoy is in event mode, the archive blanks the
  standard-mode stream to sentinels (only 17 of 143 candidate arrival windows
  survive on pure standard-mode data), so a grid slot whose standard sample
  is missing is filled with the event-mode (``T == 2/3``) measurement taken
  at *exactly* that grid timestamp when one exists. These are real
  measurements from the same instrument at the same instant -- only the
  telemetry rate differs. Event-mode rows at off-grid timestamps are never
  used, event-mode presence is never exposed as a feature, and windows are
  never created *because* event mode ran; this keeps the original
  anti-leakage intent (event-mode rows are not training samples) while
  retaining the large events the hook exists to detect.
* Windows require every slot present after bridging interior gaps of at most
  2 consecutive slots (30 min) by linear interpolation; longer gaps discard
  the window.
* Each window is deterministically detided by ordinary least squares on
  [constant, linear, cos/sin of M2 (12.4206 h), S2 (12.0000 h), K1
  (23.9345 h), O1 (25.8193 h)] and the fit is subtracted. **The residual
  (metres) is the model input**, and this preprocessing is part of the
  dataset builder: at evaluation both the physics and the learned paths
  receive the identical detrended window through the public
  ``TsunamiDetector.predict_tsunami`` API. The build stage asserts the
  residual RMS is cm-scale, i.e. detiding actually worked.
* Positives: for each (station, event) DART arrival at ``t_a``, windows
  starting at ``t_a - 6/4/2/1 h`` (each covers ``[t_a, t_a + 4 h]``), label
  1. The wave-height target is the event's ``runupHt`` at that station when
  HazEL provides one (it is the measured peak amplitude at the same
  recorder), else the window's peak absolute residual.
* Negatives: the same station-years, deterministic starts every 9 days
  (season-stratified), excluding any window within +/-3 days of ANY HazEL
  tsunami event (basin-agnostic, conservative), capped at 40 per
  station-year, label 0.

Temporal split (never random -- tides, seasons and instrumentation eras
autocorrelate): train 2005-2014, validation 2015-2019, test 2020-2025. The
validation span was widened from the design default (2016-2019) because only
one event (2018 Alaska) yields usable validation windows there; including
2015 adds the Illapel (Chile) M8.3 event so early stopping is judged on two
distinct events. ``train < val < test`` ordering is preserved.
"""

from __future__ import annotations

import datetime as _dt
import gzip
import json
import logging
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from pathlib import Path

from omni_mercury_engine.ml.hazard_training.common import (
    EvaluationOutcome,
    PipelineContext,
    TemporalSplit,
    binary_auc,
    cached_fetch,
    candidate_paths,
    save_candidate,
    save_evaluation,
    seed_everything,
    sha256_file,
    ship_checkpoint,
)

logger = logging.getLogger(__name__)

HOOK_NAME = "tsunami_dart"
CHECKPOINT_NAME = "tsunami_dart"
FEATURE_SPEC_VERSION = "tsunami-dart-v1"
DETIDE_METHOD = "ols-m2s2k1o1+linear"

DART_INDEX_URL = "https://www.ndbc.noaa.gov/data/historical/dart/"
DART_FILE_URL = "https://www.ndbc.noaa.gov/data/historical/dart/{station}t{year}.txt.gz"
STATION_TABLE_URL = "https://www.ndbc.noaa.gov/data/stations/station_table.txt"
HAZEL_EVENTS_URL = (
    "https://www.ngdc.noaa.gov/hazel/hazard-service/api/v1/tsunamis/events"
    "?minYear={min_year}&maxYear={max_year}&itemsPerPage=200&page={page}"
)
HAZEL_RUNUPS_URL = (
    "https://www.ngdc.noaa.gov/hazel/hazard-service/api/v1/tsunamis/events/"
    "{event_id}/runups?itemsPerPage=200&page={page}"
)
HAZEL_PAGE_SIZE = 200

EVENTS_MIN_YEAR = 2005
EVENTS_MAX_YEAR = 2025

SPLIT = TemporalSplit(
    train_years=tuple(range(2005, 2015)),
    val_years=(2015, 2016, 2017, 2018, 2019),
    test_years=tuple(range(2020, 2026)),
)

#: 96 x 15-minute samples = 24 hours per window.
WINDOW_SAMPLES = 96
SAMPLE_PERIOD_S = 900.0
SLOTS_PER_DAY = 96
SLOTS_PER_HOUR = 4
#: Missing-data sentinel used by the DART archive.
DART_SENTINEL = 9999.0
#: Longest run of missing slots a window may bridge by linear interpolation.
MAX_GAP_SLOTS = 2
#: Window starts relative to the observed arrival (hours before t_a).
POSITIVE_OFFSETS_H = (6, 4, 2, 1)
#: Negative-window sampling: one candidate start every 9 days, cap 40/year.
NEGATIVE_STRIDE_DAYS = 9
NEGATIVES_PER_STATION_YEAR = 40
#: Exclusion half-width around ANY HazEL event for negative sampling.
EVENT_EXCLUSION_DAYS = 3.0
#: Reject a runup whose coordinates sit farther than this from the NDBC
#: station-table position for the station id parsed from its locationName
#: (DART moorings redeploy within tens of km; 150 km catches id mixups).
STATION_COORD_TOLERANCE_KM = 150.0
#: Tidal constituent periods (hours): M2, S2, K1, O1.
TIDE_PERIODS_H = (12.4206012, 12.0, 23.9344697, 25.8193417)

_STATION_RE = re.compile(r"D(\d{5})")
_INDEX_FILE_RE = re.compile(r'href="(\d{5})t(\d{4})\.txt\.gz"')
_COORD_RE = re.compile(r"([0-9.]+)\s*([NS])\s+([0-9.]+)\s*([EW])")


def _tsunami_dir(ctx: PipelineContext) -> Path:
    """Cache directory for this hook's fetched data."""
    return ctx.data_dir / "tsunami_dart"


def _require(path: Path) -> Path:
    """Return ``path`` or fail loud with the stage that produces it."""
    if not path.exists():
        raise FileNotFoundError(f"missing cached file {path}; run the --fetch stage first")
    return path


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km (local copy; keeps this module a leaf)."""
    rlat1, rlon1, rlat2, rlon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    a = (
        math.sin((rlat2 - rlat1) / 2.0) ** 2
        + math.cos(rlat1) * math.cos(rlat2) * math.sin((rlon2 - rlon1) / 2.0) ** 2
    )
    return 2.0 * 6371.0 * math.asin(min(1.0, math.sqrt(a)))


# ---------------------------------------------------------------------------
# Raw-source parsing (shared by fetch and build; every parse fails loud).
# ---------------------------------------------------------------------------


def parse_station_table(path: Path) -> dict[str, tuple[float, float]]:
    """Parse NDBC ``station_table.txt`` into ``{station_id: (lat, lon)}``.

    Args:
        path: Cached pipe-delimited station table.

    Returns:
        Mapping of 5-character station ids to (latitude, longitude) degrees.

    Raises:
        RuntimeError: If no station coordinates parse at all (format drift).
    """
    coords: dict[str, tuple[float, float]] = {}
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("#") or "|" not in line:
            continue
        fields = line.split("|")
        if len(fields) < 7:
            continue
        m = _COORD_RE.search(fields[6])
        if m is None:
            continue
        lat = float(m.group(1)) * (1.0 if m.group(2) == "N" else -1.0)
        lon = float(m.group(3)) * (1.0 if m.group(4) == "E" else -1.0)
        coords[fields[0].strip()] = (lat, lon)
    if not coords:
        raise RuntimeError(f"{path}: no station coordinates parsed; format changed -- refusing")
    return coords


def parse_dart_index(path: Path) -> set[tuple[str, int]]:
    """Parse the DART directory index into available (station, year) pairs.

    Raises:
        RuntimeError: If the index yields no files (layout changed).
    """
    listing = {
        (m.group(1), int(m.group(2)))
        for m in _INDEX_FILE_RE.finditer(path.read_text(errors="replace"))
    }
    if not listing:
        raise RuntimeError(f"{path}: no DART station-year files parsed from the directory index")
    return listing


@dataclass(frozen=True)
class DartYearGrid:
    """One station-year of DART data on the uniform 15-minute grid.

    Attributes:
        station: 5-character DART station id.
        year: Calendar year of the grid.
        values: Water-column height (m), NaN where no measurement exists,
            shape ``(96 * days_in_year,)``.
        from_event_mode: Boolean mask marking slots whose value came from an
            event-mode (``T == 2/3``) measurement at exactly the grid
            timestamp because the standard sample was missing. Diagnostic
            only -- never a feature.
    """

    station: str
    year: int
    values: np.ndarray
    from_event_mode: np.ndarray


def parse_dart_file(path: Path, station: str, year: int) -> DartYearGrid:
    """Parse a DART archive file onto the uniform 15-minute grid.

    Standard-mode (``T == 1``) samples populate the grid; ``9999.000``
    sentinels stay missing. Slots still missing afterwards are filled from an
    event-mode (``T == 2/3``) row timestamped exactly on the grid, if any
    (see the module docstring for why). Event-mode rows at off-grid times are
    ignored entirely.

    Args:
        path: Cached ``<station>t<year>.txt.gz`` (gzip or already-inflated).
        station: Station id the file is expected to contain.
        year: Calendar year the file covers; rows outside it are dropped.

    Returns:
        The parsed :class:`DartYearGrid`.

    Raises:
        RuntimeError: If the file contains no parseable measurement rows.
    """
    n_days = 366 if _dt.date(year, 12, 31).timetuple().tm_yday == 366 else 365
    n_slots = n_days * SLOTS_PER_DAY
    standard = np.full(n_slots, np.nan)
    event = np.full(n_slots, np.nan)
    jan1 = _dt.datetime(year, 1, 1, tzinfo=_dt.UTC)

    try:
        with gzip.open(path, "rt", errors="replace") as fh:
            lines = fh.readlines()
    except (OSError, gzip.BadGzipFile):
        lines = path.read_text(errors="replace").splitlines()

    n_rows = 0
    for line in lines:
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            ts = _dt.datetime(
                int(parts[0]),
                int(parts[1]),
                int(parts[2]),
                int(parts[3]),
                int(parts[4]),
                int(parts[5]),
                tzinfo=_dt.UTC,
            )
            mode = int(parts[6])
            height = float(parts[7])
        except ValueError:
            continue
        if ts.year != year or height == DART_SENTINEL:
            continue
        n_rows += 1
        offset_s = (ts - jan1).total_seconds()
        if offset_s % SAMPLE_PERIOD_S != 0.0:
            continue  # off-grid (event-mode) timestamp: never used
        slot = int(offset_s // SAMPLE_PERIOD_S)
        if not 0 <= slot < n_slots:
            continue
        if mode == 1:
            standard[slot] = height
        else:
            event[slot] = height
    if n_rows == 0:
        raise RuntimeError(f"{path}: no parseable DART measurement rows -- refusing to continue")

    fill_mask = np.isnan(standard) & ~np.isnan(event)
    values = np.where(fill_mask, event, standard)
    return DartYearGrid(station=station, year=year, values=values, from_event_mode=fill_mask)


# ---------------------------------------------------------------------------
# HazEL events / runups -> labeled DART arrivals.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DartArrival:
    """One observed tsunami arrival at a DART bottom-pressure recorder.

    Attributes:
        event_id: HazEL tsunami event id.
        station: DART station id parsed from the runup ``locationName``.
        arrival: Observed arrival time (UTC).
        runup_ht_m: HazEL ``runupHt`` (max amplitude at the BPR, m) or None.
    """

    event_id: int
    station: str
    arrival: _dt.datetime
    runup_ht_m: float | None


def _event_origin(event: dict[str, Any]) -> _dt.datetime | None:
    """Best-precision UTC origin time of a HazEL event (None if no date)."""
    try:
        return _dt.datetime(
            int(event["year"]),
            int(event.get("month") or 1),
            int(event.get("day") or 1),
            int(event.get("hour") or 0),
            int(event.get("minute") or 0),
            tzinfo=_dt.UTC,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _events_pages(ddir: Path) -> list[Path]:
    """Paths of every cached HazEL events page, driven by page 1's totalPages."""
    first = _require(ddir / "hazel_events_p1.json")
    total_pages = int(json.loads(first.read_text())["totalPages"])
    return [_require(ddir / f"hazel_events_p{p}.json") for p in range(1, total_pages + 1)]


def load_events(ddir: Path) -> list[dict[str, Any]]:
    """Load all cached HazEL events (fail loud if the cache is incomplete)."""
    events: list[dict[str, Any]] = []
    for path in _events_pages(ddir):
        events.extend(json.loads(path.read_text())["items"])
    if not events:
        raise RuntimeError("HazEL events cache is empty; cannot label anything")
    return events


def _runup_pages(event: dict[str, Any]) -> int:
    """Number of runup pages a HazEL event needs (200 items per page)."""
    n = int(event.get("numRunups") or 0)
    return -(-n // HAZEL_PAGE_SIZE) if n > 0 else 0


def load_runups(ddir: Path, events: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """Load cached per-event runup records for every event that has any."""
    runups: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        eid = int(event["id"])
        items: list[dict[str, Any]] = []
        for page in range(1, _runup_pages(event) + 1):
            path = _require(ddir / f"hazel_runups_ev{eid}_p{page}.json")
            items.extend(json.loads(path.read_text())["items"])
        if items:
            runups[eid] = items
    return runups


def resolve_arrivals(
    events: list[dict[str, Any]],
    runups: dict[int, list[dict[str, Any]]],
    station_coords: dict[str, tuple[float, float]],
) -> tuple[list[DartArrival], dict[str, int]]:
    """Turn HazEL DART runup records into time-resolved station arrivals.

    A runup qualifies when ``typeMeasurementId == 3`` (deep-ocean BPR), a
    station id parses from ``locationName``, and ``arrDay/arrHour/arrMin``
    are all present. ``arrDay`` is a UTC day-of-month relative to the event
    date; a day before the event's day rolls into the next month. Arrivals
    outside ``[-2 h, +60 h]`` of the event origin are dropped (bad record),
    as are runups whose own coordinates sit farther than
    :data:`STATION_COORD_TOLERANCE_KM` from the NDBC station-table position
    (station-id mixup). Nothing is imputed.

    Args:
        events: HazEL event dicts.
        runups: Per-event runup dicts from :func:`load_runups`.
        station_coords: NDBC station table coordinates.

    Returns:
        Tuple of (arrivals, drop-statistics dict).

    Raises:
        RuntimeError: If more than 30% of time-resolved arrivals fail the
            coordinate cross-check -- that would mean the station-id parsing
            is systematically wrong, not a few bad records.
    """
    by_id = {int(e["id"]): e for e in events}
    arrivals: list[DartArrival] = []
    stats = {
        "dart_runups": 0,
        "no_station_id": 0,
        "no_arrival_time": 0,
        "bad_arrival_time": 0,
        "coord_mismatch": 0,
        "coord_checked": 0,
    }
    for eid, items in runups.items():
        event = by_id.get(eid)
        origin = _event_origin(event) if event is not None else None
        if event is None or origin is None:
            continue
        for runup in items:
            if runup.get("typeMeasurementId") != 3:
                continue
            stats["dart_runups"] += 1
            match = _STATION_RE.search(str(runup.get("locationName") or ""))
            if match is None:
                stats["no_station_id"] += 1
                continue
            station = match.group(1)
            if any(runup.get(k) is None for k in ("arrDay", "arrHour", "arrMin")):
                stats["no_arrival_time"] += 1
                continue
            arrival = _arrival_datetime(origin, runup)
            if arrival is None:
                stats["bad_arrival_time"] += 1
                continue
            lat, lon = runup.get("latitude"), runup.get("longitude")
            table = station_coords.get(station)
            if lat is not None and lon is not None and table is not None:
                stats["coord_checked"] += 1
                distance = _haversine_km(float(lat), float(lon), table[0], table[1])
                if distance > STATION_COORD_TOLERANCE_KM:
                    stats["coord_mismatch"] += 1
                    logger.warning(
                        "dropping runup for event %d: locationName station D%s is %.0f km "
                        "from the NDBC station-table position",
                        eid,
                        station,
                        distance,
                    )
                    continue
            height = runup.get("runupHt")
            arrivals.append(
                DartArrival(
                    event_id=eid,
                    station=station,
                    arrival=arrival,
                    runup_ht_m=float(height) if height is not None else None,
                )
            )
    if stats["coord_checked"] > 0 and stats["coord_mismatch"] > 0.3 * stats["coord_checked"]:
        raise RuntimeError(
            f"{stats['coord_mismatch']}/{stats['coord_checked']} DART runups fail the "
            "station-coordinate cross-check; the station-id parsing is wrong -- refusing "
            "to train on mislabeled waveforms"
        )
    return arrivals, stats


def _arrival_datetime(origin: _dt.datetime, runup: dict[str, Any]) -> _dt.datetime | None:
    """Resolve arrDay/arrHour/arrMin against the event origin (see caller)."""
    day, hour, minute = int(runup["arrDay"]), int(runup["arrHour"]), int(runup["arrMin"])
    for month_offset in (0, 1):
        month = origin.month + month_offset
        year = origin.year + (1 if month > 12 else 0)
        month = (month - 1) % 12 + 1
        try:
            candidate = _dt.datetime(year, month, day, hour, minute, tzinfo=_dt.UTC)
        except ValueError:
            continue
        delta = candidate - origin
        if _dt.timedelta(hours=-2) <= delta <= _dt.timedelta(hours=60):
            return candidate
    return None


# ---------------------------------------------------------------------------
# Stage 1: fetch.
# ---------------------------------------------------------------------------


def fetch(ctx: PipelineContext) -> dict[str, Any]:
    """Download every raw source: HazEL labels, station table, DART archives.

    The set of DART station-year files to download is derived from the data
    itself: every (station, arrival-year) pair with a time-resolved DART
    arrival whose file exists in the live directory index.

    Returns:
        Manifest with per-file URLs and SHA-256 digests plus label counts.
    """
    ddir = _tsunami_dir(ctx)
    sources: list[dict[str, Any]] = []

    index_path = cached_fetch(DART_INDEX_URL, ddir / "dart_index.html")
    sources.append(
        {
            "url": DART_INDEX_URL,
            "sha256": sha256_file(index_path),
            "description": "NDBC DART historical archive directory index",
        }
    )
    table_path = cached_fetch(STATION_TABLE_URL, ddir / "station_table.txt")
    sources.append(
        {
            "url": STATION_TABLE_URL,
            "sha256": sha256_file(table_path),
            "description": "NDBC station table (station coordinates cross-check)",
        }
    )

    first_url = HAZEL_EVENTS_URL.format(min_year=EVENTS_MIN_YEAR, max_year=EVENTS_MAX_YEAR, page=1)
    first_path = cached_fetch(first_url, ddir / "hazel_events_p1.json")
    sources.append(
        {
            "url": first_url,
            "sha256": sha256_file(first_path),
            "description": f"NCEI HazEL tsunami events {EVENTS_MIN_YEAR}-{EVENTS_MAX_YEAR}, p1",
        }
    )
    total_pages = int(json.loads(first_path.read_text())["totalPages"])
    for page in range(2, total_pages + 1):
        url = HAZEL_EVENTS_URL.format(min_year=EVENTS_MIN_YEAR, max_year=EVENTS_MAX_YEAR, page=page)
        path = cached_fetch(url, ddir / f"hazel_events_p{page}.json")
        sources.append(
            {
                "url": url,
                "sha256": sha256_file(path),
                "description": (
                    f"NCEI HazEL tsunami events {EVENTS_MIN_YEAR}-{EVENTS_MAX_YEAR}, p{page}"
                ),
            }
        )

    events = load_events(ddir)
    for event in events:
        eid = int(event["id"])
        for page in range(1, _runup_pages(event) + 1):
            url = HAZEL_RUNUPS_URL.format(event_id=eid, page=page)
            path = cached_fetch(url, ddir / f"hazel_runups_ev{eid}_p{page}.json")
            sources.append(
                {
                    "url": url,
                    "sha256": sha256_file(path),
                    "description": f"NCEI HazEL runups for tsunami event {eid}, p{page}",
                }
            )

    station_coords = parse_station_table(table_path)
    runups = load_runups(ddir, events)
    arrivals, stats = resolve_arrivals(events, runups, station_coords)
    available = parse_dart_index(index_path)
    needed = sorted(
        {(a.station, a.arrival.year) for a in arrivals if (a.station, a.arrival.year) in available}
    )
    if not needed:
        raise RuntimeError(
            "no DART station-year archive files match any time-resolved HazEL arrival; "
            "the archive layout or the label API must have changed -- refusing to continue"
        )
    for station, year in needed:
        url = DART_FILE_URL.format(station=station, year=year)
        path = cached_fetch(url, ddir / "dart" / f"{station}t{year}.txt.gz")
        sources.append(
            {
                "url": url,
                "sha256": sha256_file(path),
                "description": f"NDBC DART bottom-pressure archive, station {station}, {year}",
            }
        )

    manifest = {
        "hook": HOOK_NAME,
        "sources": sources,
        "label_stats": stats,
        "n_events": len(events),
        "n_arrivals": len(arrivals),
        "n_station_years": len(needed),
    }
    (ddir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    logger.info(
        "fetch complete: %d sources (%d DART station-year files, %d arrivals) under %s",
        len(sources),
        len(needed),
        len(arrivals),
        ddir,
    )
    return manifest


# ---------------------------------------------------------------------------
# Stage 2: build dataset.
# ---------------------------------------------------------------------------


def detide_window(values: np.ndarray) -> np.ndarray:
    """Deterministically detide one gap-free window by OLS harmonic fit.

    Fits [constant, linear, cos/sin of M2, S2, K1, O1] over the window and
    subtracts the fit (:data:`DETIDE_METHOD`).

    Args:
        values: Water-column heights (m), shape ``(WINDOW_SAMPLES,)``, finite.

    Returns:
        Residual (m), same shape, ``float64``.
    """
    t = np.arange(WINDOW_SAMPLES, dtype=np.float64) * SAMPLE_PERIOD_S
    columns = [np.ones_like(t), (t - t.mean()) / t.std()]
    for period_h in TIDE_PERIODS_H:
        omega = 2.0 * np.pi / (period_h * 3600.0)
        columns.append(np.cos(omega * t))
        columns.append(np.sin(omega * t))
    design = np.stack(columns, axis=1)
    coef, *_ = np.linalg.lstsq(design, values, rcond=None)
    return np.asarray(values - design @ coef, dtype=np.float64)


def extract_residual_window(grid_values: np.ndarray, start_slot: int) -> np.ndarray | None:
    """Extract one detided 96-sample window from a station-year grid.

    Applies the gap rule: the first and last samples must be present and no
    interior run of missing slots may exceed :data:`MAX_GAP_SLOTS`; runs of
    1-2 missing slots are bridged by linear interpolation. Anything else
    returns None (the window is discarded, never imputed wholesale).

    Args:
        grid_values: Full-year grid values (m, NaN = missing).
        start_slot: Index of the window's first 15-minute slot.

    Returns:
        Detided residual (m) as ``float32`` of shape ``(96,)``, or None.
    """
    if start_slot < 0 or start_slot + WINDOW_SAMPLES > grid_values.shape[0]:
        return None
    window = grid_values[start_slot : start_slot + WINDOW_SAMPLES].astype(np.float64, copy=True)
    missing = np.isnan(window)
    if missing[0] or missing[-1]:
        return None
    if missing.any():
        longest = 0
        run = 0
        for gap in missing:
            run = run + 1 if gap else 0
            longest = max(longest, run)
        if longest > MAX_GAP_SLOTS:
            return None
        idx = np.arange(WINDOW_SAMPLES)
        window[missing] = np.interp(idx[missing], idx[~missing], window[~missing])
    return detide_window(window).astype(np.float32)


@dataclass
class TsunamiDataset:
    """Windowed DART dataset with labels and split metadata.

    Attributes:
        windows: Detided residuals (m), ``float32 [n, 96]``.
        labels: 1.0 tsunami-arrival window / 0.0 quiet window.
        heights: Wave-height targets (m); 0 for negatives (never used).
        height_is_runup: True where the target is HazEL ``runupHt`` (else the
            window's peak absolute residual).
        years: Window-start calendar year per sample.
        stations: DART station id per sample.
        event_ids: HazEL event id per sample (0 for negatives).
        summary: Build statistics recorded into evaluation extras.
    """

    windows: np.ndarray
    labels: np.ndarray
    heights: np.ndarray
    height_is_runup: np.ndarray
    years: np.ndarray
    stations: np.ndarray
    event_ids: np.ndarray
    summary: dict[str, Any]


def _slot_of(ts: _dt.datetime, year: int) -> int:
    """15-minute slot index of ``ts`` within ``year`` (floor)."""
    jan1 = _dt.datetime(year, 1, 1, tzinfo=_dt.UTC)
    return int((ts - jan1).total_seconds() // SAMPLE_PERIOD_S)


def build_dataset(ctx: PipelineContext) -> TsunamiDataset:
    """Assemble labeled windows from the cached raw sources.

    Everything derives from the cached fetch stage; any missing file fails
    loud. Sanity gates: >= 10 stations and >= 12 distinct events must yield
    positive windows, every split must contain both classes, and the median
    residual RMS must be cm-scale (detiding worked).

    Returns:
        The assembled :class:`TsunamiDataset`.

    Raises:
        RuntimeError: On any sanity-gate failure (never silently degraded).
    """
    ddir = _tsunami_dir(ctx)
    station_coords = parse_station_table(_require(ddir / "station_table.txt"))
    available = parse_dart_index(_require(ddir / "dart_index.html"))
    events = load_events(ddir)
    runups = load_runups(ddir, events)
    arrivals, label_stats = resolve_arrivals(events, runups, station_coords)

    usable = [a for a in arrivals if (a.station, a.arrival.year) in available]
    station_years = sorted({(a.station, a.arrival.year) for a in usable})
    grids: dict[tuple[str, int], DartYearGrid] = {}
    for station, year in station_years:
        path = _require(ddir / "dart" / f"{station}t{year}.txt.gz")
        grids[(station, year)] = parse_dart_file(path, station, year)

    origins = [o for o in (_event_origin(e) for e in events) if o is not None]

    windows: list[np.ndarray] = []
    labels: list[float] = []
    heights: list[float] = []
    height_is_runup: list[bool] = []
    years: list[int] = []
    stations: list[str] = []
    event_ids: list[int] = []

    n_pos_discarded = 0
    for a in usable:
        grid = grids[(a.station, a.arrival.year)]
        anchor = _slot_of(a.arrival, a.arrival.year)
        for offset_h in POSITIVE_OFFSETS_H:
            residual = extract_residual_window(grid.values, anchor - offset_h * SLOTS_PER_HOUR)
            if residual is None:
                n_pos_discarded += 1
                continue
            windows.append(residual)
            labels.append(1.0)
            peak = float(np.max(np.abs(residual)))
            heights.append(a.runup_ht_m if a.runup_ht_m is not None else peak)
            height_is_runup.append(a.runup_ht_m is not None)
            years.append(a.arrival.year)
            stations.append(a.station)
            event_ids.append(a.event_id)
    n_positive = len(windows)

    window_span = _dt.timedelta(seconds=WINDOW_SAMPLES * SAMPLE_PERIOD_S)
    exclusion = _dt.timedelta(days=EVENT_EXCLUSION_DAYS)
    for (station, year), grid in sorted(grids.items()):
        kept = 0
        for day in range(2, 367, NEGATIVE_STRIDE_DAYS):
            if kept >= NEGATIVES_PER_STATION_YEAR:
                break
            start_slot = day * SLOTS_PER_DAY
            start_dt = _dt.datetime(year, 1, 1, tzinfo=_dt.UTC) + _dt.timedelta(
                seconds=start_slot * SAMPLE_PERIOD_S
            )
            end_dt = start_dt + window_span
            if any(
                start_dt <= origin + exclusion and end_dt >= origin - exclusion
                for origin in origins
            ):
                continue
            residual = extract_residual_window(grid.values, start_slot)
            if residual is None:
                continue
            windows.append(residual)
            labels.append(0.0)
            heights.append(0.0)
            height_is_runup.append(False)
            years.append(year)
            stations.append(station)
            event_ids.append(0)
            kept += 1

    if not windows:
        raise RuntimeError("no windows were built from the cached DART data; cannot proceed")

    ds = TsunamiDataset(
        windows=np.stack(windows).astype(np.float32),
        labels=np.asarray(labels, dtype=np.float32),
        heights=np.asarray(heights, dtype=np.float32),
        height_is_runup=np.asarray(height_is_runup, dtype=bool),
        years=np.asarray(years, dtype=np.int64),
        stations=np.asarray(stations),
        event_ids=np.asarray(event_ids, dtype=np.int64),
        summary={},
    )

    pos = ds.labels == 1.0
    n_stations = len(set(ds.stations[pos]))
    n_events = len(set(ds.event_ids[pos].tolist()))
    if n_stations < 10 or n_events < 12:
        raise RuntimeError(
            f"insufficient real label coverage: {n_stations} stations / {n_events} events with "
            "positive windows (need >= 10 stations and >= 12 events); refusing to train on a "
            "corpus this thin"
        )
    rms = np.sqrt(np.mean(ds.windows.astype(np.float64) ** 2, axis=1))
    median_rms = float(np.median(rms))
    if median_rms > 0.05:
        raise RuntimeError(
            f"median residual RMS {median_rms:.4f} m is not cm-scale; the detiding failed -- "
            "refusing to train on tide-contaminated windows"
        )
    train_mask, val_mask, test_mask = SPLIT.masks(ds.years)
    for name, mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
        if not (ds.labels[mask] == 1.0).any() or not (ds.labels[mask] == 0.0).any():
            raise RuntimeError(f"{name} split lacks a class; the corpus cannot support this split")

    def _split_events(mask: np.ndarray) -> int:
        return len(set(ds.event_ids[mask & pos].tolist()))

    ds.summary = {
        "label_stats": label_stats,
        "n_arrivals_usable": len(usable),
        "n_station_years": len(station_years),
        "n_stations": n_stations,
        "n_events": n_events,
        "n_positive_windows": n_positive,
        "n_negative_windows": int(len(ds.labels) - n_positive),
        "n_positive_discarded": n_pos_discarded,
        "median_residual_rms_m": median_rms,
        "events_train": _split_events(train_mask),
        "events_val": _split_events(val_mask),
        "events_test": _split_events(test_mask),
        "windows_train": int(train_mask.sum()),
        "windows_val": int(val_mask.sum()),
        "windows_test": int(test_mask.sum()),
        "detide": DETIDE_METHOD,
    }
    logger.info("build complete: %s", json.dumps(ds.summary, sort_keys=True))

    if ctx.limit_samples is not None and len(ds.labels) > ctx.limit_samples:
        n_pos_keep = min(int(pos.sum()), max(1, ctx.limit_samples // 2))
        keep = np.concatenate(
            [
                np.flatnonzero(pos)[:n_pos_keep],
                np.flatnonzero(~pos)[: ctx.limit_samples - n_pos_keep],
            ]
        )
        ds = TsunamiDataset(
            windows=ds.windows[keep],
            labels=ds.labels[keep],
            heights=ds.heights[keep],
            height_is_runup=ds.height_is_runup[keep],
            years=ds.years[keep],
            stations=ds.stations[keep],
            event_ids=ds.event_ids[keep],
            summary=ds.summary,
        )
    return ds


# ---------------------------------------------------------------------------
# Stage 3: train.
# ---------------------------------------------------------------------------


def train(ctx: PipelineContext) -> dict[str, Any]:
    """Train the WaveformFFTAnalyzer with early stopping on validation AUC.

    Weighted BCE on the tsunami-probability head (pos_weight from the train
    imbalance) plus MSE on the wave-height head over positives only. The
    early-stopping metric is the network's own probability AUC on validation
    windows (the full-API comparison belongs to :func:`evaluate`).

    Two deliberate, documented departures from a vanilla loop:

    * **Input standardization folded into the weights.** The public input
      contract is the detided residual in metres (train std is centimetres),
      so per-feature scaling at inference is impossible. Instead the first
      conv layer's default initialization -- which assumes unit-variance
      inputs -- is rescaled once by 1/std of the TRAIN windows. This is
      ordinary input standardization expressed inside the shipped weights.
    * **Validation-calibrated operating point.** The detector's deployed
      decision is ``confidence > 0.96``. After early stopping the classifier
      logit is affinely recalibrated (monotone -- ranking/AUC unchanged) so
      that the 0.96 crossing sits at a threshold chosen on the VALIDATION
      years only: zero validation false alarms (physics' validation FAR is
      0) with the best achievable validation recall. Mirrors the ratified
      solar-storm operating-point policy; test years are never consulted.

    Returns:
        Training record (epochs run, best validation AUC, sample counts,
        operating-point record).
    """
    from omni_mercury_engine.detectors.geological.disaster_detectors import WaveformFFTAnalyzer

    rng = seed_everything(ctx.seed)
    ds = build_dataset(ctx)
    train_mask, val_mask, _ = SPLIT.masks(ds.years)

    x_train = torch.from_numpy(ds.windows[train_mask])
    y_train = torch.from_numpy(ds.labels[train_mask])
    h_train = torch.from_numpy(ds.heights[train_mask])
    x_val = torch.from_numpy(ds.windows[val_mask])
    y_val = ds.labels[val_mask]

    n_pos = float(y_train.sum().item())
    n_neg = float(len(y_train) - n_pos)
    pos_weight = n_neg / max(n_pos, 1.0)
    logger.info(
        "training on %d windows (%d positive, pos_weight %.2f), validating on %d",
        len(y_train),
        int(n_pos),
        pos_weight,
        len(y_val),
    )

    model = WaveformFFTAnalyzer()
    train_input_std = float(ds.windows[train_mask].std())
    if not np.isfinite(train_input_std) or train_input_std <= 0:
        raise RuntimeError("degenerate train-window std; refusing to standardize")
    with torch.no_grad():
        model.conv1d.weight.mul_(1.0 / train_input_std)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 128
    best_val_auc = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    patience, bad_epochs = 8, 0
    epochs_run = 0

    for epoch in range(ctx.max_epochs):
        epochs_run = epoch + 1
        model.train()
        perm = torch.from_numpy(rng.permutation(len(y_train)))
        epoch_loss = 0.0
        for start in range(0, len(y_train), batch_size):
            idx = perm[start : start + batch_size]
            xb, yb, hb = x_train[idx], y_train[idx], h_train[idx]
            prob, height = model(xb)
            weights = torch.where(yb > 0.5, torch.full_like(yb, pos_weight), torch.ones_like(yb))
            loss = torch.nn.functional.binary_cross_entropy(
                prob.clamp(1e-6, 1 - 1e-6), yb, weight=weights
            )
            pos_mask = yb > 0.5
            if pos_mask.any():
                loss = loss + torch.nn.functional.mse_loss(height[pos_mask], hb[pos_mask])
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call, unused-ignore]
            optimizer.step()
            epoch_loss += float(loss.item()) * len(idx)

        model.eval()
        with torch.no_grad():
            val_prob = torch.cat(
                [model(x_val[i : i + 512])[0] for i in range(0, len(x_val), 512)]
            ).numpy()
        val_auc = binary_auc(y_val, val_prob)
        logger.info(
            "epoch %d: train loss %.4f, val AUC %.4f",
            epoch + 1,
            epoch_loss / len(y_train),
            val_auc,
        )
        if np.isfinite(val_auc) and val_auc > best_val_auc + 1e-4:
            best_val_auc = float(val_auc)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.info("early stop at epoch %d (patience %d)", epoch + 1, patience)
                break

    if best_state is None:
        raise RuntimeError("training produced no finite validation AUC; refusing to save")
    model.load_state_dict(best_state)
    operating_point = _calibrate_operating_point(model, ds, val_mask)

    record = {
        "seed": ctx.seed,
        "epochs_run": epochs_run,
        "best_val_auc": best_val_auc,
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "train_pos_weight": pos_weight,
        "train_input_std": train_input_std,
        "operating_point": operating_point,
        "train_years": list(SPLIT.train_years),
        "val_years": list(SPLIT.val_years),
        "build_summary": ds.summary,
    }
    payload: dict[str, Any] = {
        "waveform_analyzer": model.state_dict(),
        "feature_spec": FEATURE_SPEC_VERSION,
        "window_samples": WINDOW_SAMPLES,
        "sampling_period_s": SAMPLE_PERIOD_S,
        "detide": DETIDE_METHOD,
    }
    save_candidate(ctx.data_dir, HOOK_NAME, payload, record)
    return record


def _calibrate_operating_point(
    model: Any, ds: TsunamiDataset, val_mask: np.ndarray
) -> dict[str, Any]:
    """Recalibrate the classifier so the deployed 0.96 point is val-chosen.

    Policy (documented for owner ratification, mirroring the ratified
    solar-storm dual-rule machinery): on the VALIDATION years only, run the
    physics detector through the public API to get its recall / false-alarm
    rate at the deployed ``confidence > 0.96`` decision and the per-window
    resonance score (identical for both paths -- it is a deterministic
    function of the window). Candidate raw-logit thresholds are the
    midpoints between consecutive sorted validation logits; a threshold is
    feasible when the calibrated learned decision -- ``min(1, sigmoid(a *
    (logit - t) + logit(0.96)) + 0.3 * resonance) > 0.96`` -- yields ZERO
    validation false alarms (physics' validation FAR is the ceiling and it
    is 0 here; an unmeasurable ceiling must not pass silently). Among
    feasible thresholds the one with the highest validation recall wins
    (ties -> larger threshold, i.e. more conservative). The affine map is
    then folded into the final linear layer: monotone in the logit, so the
    ranking and AUC the merit gate's primary metric measures are unchanged.

    The steepness ``a`` maps the median above-threshold validation positive
    to probability 0.995, keeping the calibrated probabilities spread (a
    saturated 0/1 output would destroy the confidence ranking).

    Args:
        model: Trained WaveformFFTAnalyzer (modified in place).
        ds: The built dataset.
        val_mask: Boolean validation mask over ``ds`` rows.

    Returns:
        Operating-point record (policy, threshold, steepness, validation
        recall/FAR for learned and physics).

    Raises:
        RuntimeError: If validation lacks a class or no calibration achieves
            zero validation false alarms.
    """
    from omni_mercury_engine.detectors.geological.disaster_detectors import TsunamiDetector

    val_idx = np.flatnonzero(val_mask)
    y_val = ds.labels[val_idx].astype(bool)
    if not y_val.any() or y_val.all():
        raise RuntimeError("validation years contain a single class; cannot calibrate honestly")

    physics_det = TsunamiDetector(sampling_rate=1.0 / SAMPLE_PERIOD_S)
    resonance = np.zeros(val_idx.size)
    physics_detected = np.zeros(val_idx.size, dtype=bool)
    for row, i in enumerate(val_idx):
        out = physics_det.predict_tsunami(ds.windows[i])
        resonance[row] = float(out.resonance_score)
        physics_detected[row] = bool(out.tsunami_detected)
    physics_recall = float(physics_detected[y_val].mean())
    physics_far = float(physics_detected[~y_val].mean())

    model.eval()
    with torch.no_grad():
        prob = torch.cat(
            [
                model(torch.from_numpy(ds.windows[val_idx][i : i + 512]))[0]
                for i in range(0, val_idx.size, 512)
            ]
        ).numpy()
    prob = np.clip(prob.astype(np.float64), 1e-9, 1 - 1e-9)
    logit = np.log(prob) - np.log1p(-prob)
    logit_96 = float(np.log(0.96 / 0.04))
    logit_995 = float(np.log(0.995 / 0.005))

    ordered = np.sort(np.unique(logit))
    candidates = [(ordered[i] + ordered[i + 1]) / 2.0 for i in range(len(ordered) - 1)]
    candidates.append(float(ordered[-1] + 1.0))
    best: dict[str, Any] | None = None
    for t in candidates:
        above = logit[y_val] > t
        if not above.any():
            continue
        spread = float(np.median(logit[y_val][above]) - t)
        a = float(np.clip((logit_995 - logit_96) / max(spread, 1e-9), 1.0, 1e6))
        conf = np.minimum(
            1.0, 1.0 / (1.0 + np.exp(-(a * (logit - t) + logit_96))) + 0.3 * resonance
        )
        detected = conf > 0.96
        far = float(detected[~y_val].mean())
        # Solar's ratified 20% FAR headroom against val->test shift; with
        # physics' validation FAR of 0 this demands zero validation alarms.
        if far > 0.8 * physics_far + 1e-12:
            continue
        recall = float(detected[y_val].mean())
        if (
            best is None
            or recall > best["val_recall"]
            or (recall == best["val_recall"] and t > best["logit_threshold"])
        ):
            best = {
                "logit_threshold": float(t),
                "steepness": a,
                "val_recall": recall,
                "val_far": far,
            }
    if best is None:
        raise RuntimeError(
            "no calibration achieves zero validation false alarms at the deployed 0.96 "
            "point; refusing to ship an operating point physics already beats"
        )

    final: torch.nn.Linear = model.classifier[3]
    with torch.no_grad():
        final.weight.mul_(best["steepness"])
        final.bias.mul_(best["steepness"])
        final.bias.add_(-best["steepness"] * best["logit_threshold"] + logit_96)

    record = {
        "policy": (
            "validation-years-only affine logit recalibration; zero validation false "
            "alarms at confidence>0.96 (physics validation FAR is the ceiling), best "
            "validation recall among feasible thresholds"
        ),
        "physics_val_recall": physics_recall,
        "physics_val_far": physics_far,
        **best,
    }
    logger.info("operating point calibrated: %s", json.dumps(record, sort_keys=True))
    return record


# ---------------------------------------------------------------------------
# Stage 4: evaluate (public-API comparison on identical held-out windows).
# ---------------------------------------------------------------------------


def evaluate(ctx: PipelineContext) -> EvaluationOutcome:
    """Compare learned vs physics through ``TsunamiDetector.predict_tsunami``.

    Both paths see the *identical* held-out test windows (detided residuals,
    exactly what the dataset builder produces): physics is a fresh detector
    with no weights loaded (deterministic amplitude + resonance fallback),
    learned is a fresh detector after ``load_neural_weights`` on the
    candidate checkpoint. Primary metric: window-ranking AUC from
    ``result.confidence`` (higher is better). Secondary non-regression
    constraints at the detector's deployed ``confidence > 0.96`` decision:
    detection recall, false-alarm rate, and per-(station, event) recall must
    not regress below physics (parity allowed).

    ``wave_height_mae_m`` is reported but deliberately NOT a constraint: the
    physics path's "estimate" is the window's own peak detided deviation --
    i.e. the direct DART measurement of the amplitude, which HazEL's
    ``runupHt`` is itself derived from -- so parity is structurally
    unreachable for a regression head and the amplitude measurement remains
    available to operators regardless of which path scored the window.

    Returns:
        The evaluation outcome, persisted next to the candidate.
    """
    from omni_mercury_engine.detectors.geological.disaster_detectors import TsunamiDetector

    ds = build_dataset(ctx)
    _, _, test_mask = SPLIT.masks(ds.years)
    test_idx = np.flatnonzero(test_mask)
    if test_idx.size == 0:
        raise RuntimeError("no test windows found; cannot evaluate")

    cand_path, _ = candidate_paths(ctx.data_dir, HOOK_NAME)
    if not cand_path.exists():
        raise FileNotFoundError(f"no candidate checkpoint at {cand_path}; run --train first")

    physics_det = TsunamiDetector(sampling_rate=1.0 / SAMPLE_PERIOD_S)
    learned_det = TsunamiDetector(sampling_rate=1.0 / SAMPLE_PERIOD_S)
    learned_det.load_neural_weights(str(cand_path))

    labels = ds.labels[test_idx]
    results: dict[str, dict[str, list[float]]] = {
        "physics": {"conf": [], "det": [], "height": []},
        "learned": {"conf": [], "det": [], "height": []},
    }
    for i in test_idx:
        window = ds.windows[i]
        for name, det in (("physics", physics_det), ("learned", learned_det)):
            out = det.predict_tsunami(window)
            if not np.isfinite(out.confidence):
                raise RuntimeError(f"{name} path returned non-finite confidence for window {i}")
            results[name]["conf"].append(float(out.confidence))
            results[name]["det"].append(float(out.tsunami_detected))
            results[name]["height"].append(float(out.estimated_wave_height_m))

    pos = labels == 1.0
    neg = ~pos
    heights_true = ds.heights[test_idx]
    pos_keys = {(str(ds.stations[i]), int(ds.event_ids[i])) for i in test_idx[pos]}

    def _metrics(name: str) -> dict[str, float]:
        conf = np.asarray(results[name]["conf"])
        detected = np.asarray(results[name]["det"]) > 0.5
        est_height = np.asarray(results[name]["height"])
        hit_events = set()
        for row, i in enumerate(test_idx):
            if pos[row] and detected[row]:
                hit_events.add((str(ds.stations[i]), int(ds.event_ids[i])))
        return {
            "auc": binary_auc(labels, conf),
            "detection_recall_at_0.96": float(detected[pos].mean()),
            "false_alarm_rate_at_0.96": float(detected[neg].mean()),
            "wave_height_mae_m": float(np.mean(np.abs(est_height[pos] - heights_true[pos]))),
            "event_recall": float(len(hit_events) / len(pos_keys)) if pos_keys else float("nan"),
        }

    outcome = EvaluationOutcome(
        hook=HOOK_NAME,
        primary_metric="auc",
        higher_is_better=True,
        learned=_metrics("learned"),
        physics=_metrics("physics"),
        n_test_samples=int(test_idx.size),
        test_years=SPLIT.test_years,
        constraints=[
            {
                "metric": "detection_recall_at_0.96",
                "higher_is_better": True,
                "description": "detection recall at the detector's deployed 0.96 "
                "confidence threshold must not regress below physics",
            },
            {
                "metric": "false_alarm_rate_at_0.96",
                "higher_is_better": False,
                "description": "false-alarm rate at the deployed 0.96 threshold must "
                "not exceed physics",
            },
            {
                "metric": "event_recall",
                "higher_is_better": True,
                "description": "fraction of held-out (station, event) arrivals with any "
                "window detected at 0.96 must not regress below physics",
            },
        ],
        extras={
            **ds.summary,
            "n_test_positive": int(pos.sum()),
            "n_test_negative": int(neg.sum()),
            "n_test_station_events": len(pos_keys),
            "comparison": (
                "identical held-out detided DART windows through "
                "TsunamiDetector.predict_tsunami (sampling_rate=1/900 Hz), physics fallback "
                "vs loaded candidate checkpoint"
            ),
        },
    )
    save_evaluation(ctx.data_dir, outcome)
    logger.info(
        "evaluation: learned AUC %.4f vs physics %.4f on %d held-out windows (%s)",
        outcome.learned["auc"],
        outcome.physics["auc"],
        outcome.n_test_samples,
        "LEARNED WINS" if outcome.learned_beats_physics else "PHYSICS WINS",
    )
    return outcome


# ---------------------------------------------------------------------------
# Stage 5: ship (merit-gated).
# ---------------------------------------------------------------------------


def ship(ctx: PipelineContext) -> tuple[Any, Any]:
    """Promote the candidate through the merit gate (may refuse loudly)."""
    from omni_mercury_engine.ml.hazard_training.common import load_evaluation

    outcome = load_evaluation(ctx.data_dir, HOOK_NAME)
    manifest_path = _tsunami_dir(ctx) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing fetch manifest {manifest_path}; run --fetch first")
    manifest = json.loads(manifest_path.read_text())
    return ship_checkpoint(
        hook=HOOK_NAME,
        checkpoint_name=CHECKPOINT_NAME,
        data_dir=ctx.data_dir,
        outcome=outcome,
        data_sources=manifest["sources"],
        seed=ctx.seed,
        out_dir=ctx.ship_dir,
    )
