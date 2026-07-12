# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Train the SlopeStabilityModel on real NASA GLC landslides + CHIRPS rainfall.

Data sources (hook ``landslide_stability``,
``LandslideDetector.load_neural_weights``):

* **NASA Global Landslide Catalog (GLC / COOLR)** via the public ArcGIS
  Online mirror (``services1.arcgis.com``, layer
  ``nasa_global_landslide_catalog_point``) -- 14,750 citizen/media-reported
  landslide events with dates, coordinates, trigger, category, size and a
  location-accuracy radius, through May 2025. Pages are fetched with
  ``orderByFields=FID ASC`` so offset paging is deterministic. Field names
  are content-introspected at load time (the AGOL mirror truncates names to
  10 characters, e.g. ``landslide_`` = category, ``landslide1`` = trigger;
  the HDX shapefile mirror shifts them again), never hardcoded blindly.
  Fallback: the HDX shapefile zip (``data.humdata.org``); note the HDX
  filestore 302-redirects to a signed S3 URL that the project transport
  (SafeHTTPClient, no-redirect policy) refuses, so the fallback only works
  from an operator-supplied local copy of the archive -- the loud redirect
  error documents exactly that.
* **CHIRPS v2.0 global daily precipitation, 0.25 deg** (UCSB Climate
  Hazards Center, ``data.chc.ucsb.edu``, one netCDF file per year,
  50S-50N, 1981-present) -- the rainfall covariates. Every feature is
  computed from the event's own 0.25 deg grid cell using days STRICTLY
  before or on the event day (no lookahead), and every percentile is taken
  against that cell's own fixed 1981-2006 climatology -- a pre-train-era
  baseline, so no statistic ever leaks from validation/test years.

Task: rainfall-triggered landslide occurrence classification. Positives are
GLC events with rain-family triggers (downpour / rain / continuous_rain /
monsoon / tropical_cyclone / flooding, case-normalized), 2007-2024,
|lat| <= 50 (CHIRPS coverage), location accuracy <= 25 km. Negatives
(3x positives, seeded) come from two schemes: (a) the same cells at random
dates >= 60 days from any GLC event at that cell (controls for location
bias) and (b) random CHIRPS land cells at random dates with no GLC event
within 100 km and +/-60 days. The type head is trained on the GLC category
mapped honestly onto the detector's six classes (unmappable categories are
excluded from the type loss, never guessed).

Temporal split (never random -- landslide reporting and rainfall both
autocorrelate across years): train 2007-2015, validation 2016-2017, test
2018-2024, split on each sample's own date year. The task sketch suggested
test 2021-2024, but GLC ingestion largely stopped in 2019 (2021-2024 hold
only ~53 filtered positives), so the boundary moved to 2018 (656 test
positives); the ordering constraint train < val < test is unchanged.

Feature spec ``landslide-coolr-v1`` (64 dims, matching
``SlopeStabilityModel(input_dim=64)``). This module IS the canonical
definition of the ``slope_features`` vector the detector's neural path
consumes; the shipped checkpoint's first encoder layer has the train-year
standardization folded in, so callers pass this vector RAW:

======  ==============================================================
 index  meaning
======  ==============================================================
     0  log1p(day-of-event rain, mm)
     1  log1p(3-day rain sum ending on event day, mm)
     2  log1p(7-day rain sum ending on event day, mm)
     3  log1p(14-day rain sum ending on event day, mm)
     4  log1p(30-day rain sum ending on event day, mm)
     5  log1p(60-day rain sum ending on event day, mm)
     6  log1p(max 1-day rain in the 30 days ending on event day, mm)
     7  log1p(max 3-day rain sum in the 30 days ending on event day, mm)
     8  wet-day (>= 1 mm) fraction of the 30 days ending on event day
  9-17  the same nine quantities as PERCENTILES (midrank, in [0, 1])
        against the cell's own 1981-2006 CHIRPS climatology of the
        identical rolling quantity (fixed pre-train-era baseline)
    18  latitude / 50
    19  |latitude| / 50
    20  sin(2 pi day_of_year / 365.25)
    21  cos(2 pi day_of_year / 365.25)
    22  sin(4 pi day_of_year / 365.25)
    23  cos(4 pi day_of_year / 365.25)
    24  presence flag: full 61-day rain window observed
    25  presence flag: cell climatology available
    26  slope angle (deg) / 60          [site dim -- see note below]
    27  presence flag: slope angle observed
    28  soil saturation fraction        [site dim]
    29  presence flag: soil saturation observed
    30  displacement rate (mm/day) / 50 [site dim]
    31  presence flag: displacement observed
    32  seismic PGA (g)                 [site dim]
    33  presence flag: PGA observed
    34  snowmelt (mm/day) / 50          [site dim]
    35  presence flag: snowmelt observed
 36-63  reserved (0.0)
======  ==============================================================

Honesty note on dims 26-35: the GLC/COOLR corpus carries NO slope,
geotechnical, hydrological-sensor or seismic observations, so this pipeline
always sets those dims (and their presence flags) to zero. The spec
reserves them for operators who have real site instrumentation; the shipped
model was trained with them absent and therefore learns nothing about them
-- it must not be presented as using slope data it never saw.

The detector's own physics path cannot grade slope failure from rainfall
alone (without slope/saturation/displacement observations its failure
probability is identically 0 -- it abstains), so the merit-gate physics
baseline is the stronger of (1) the public-API physics path fed the honest
rainfall-only fields it defines, and (2) a documented Caine-1980-style
antecedent-rainfall percentile threshold fitted on TRAIN years only. The
evaluate stage records both, verbatim, in ``extras``.
"""

from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
import logging
import re
import zipfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from omni_mercury_engine.ml.hazard_training.common import (
    EvaluationOutcome,
    PipelineContext,
    TemporalSplit,
    binary_auc,
    brier_score,
    cached_fetch,
    candidate_paths,
    save_candidate,
    save_evaluation,
    seed_everything,
    sha256_file,
    ship_checkpoint,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

logger = logging.getLogger(__name__)

HOOK_NAME = "landslide_stability"
CHECKPOINT_NAME = "landslide_coolr"
FEATURE_SPEC_VERSION = "landslide-coolr-v1"
FEATURE_DIM = 64

AGOL_QUERY_URL = (
    "https://services1.arcgis.com/yFGHRCyBneULM8ci/arcgis/rest/services/"
    "nasa_global_landslide_catalog_point/FeatureServer/0/query"
    "?where=1%3D1&outFields=*&f=json&orderByFields=FID%20ASC"
    "&resultOffset={offset}&resultRecordCount={count}"
)
AGOL_PAGE_SIZE = 2000
AGOL_MAX_PAGES = 20

HDX_ZIP_URL = (
    "https://data.humdata.org/dataset/1eb911ba-3681-4a96-b025-ae0c33b80a12/"
    "resource/ed703c45-2001-4286-ba16-8248c17fec80/download/"
    "global_landslide_catalog_nasa.zip"
)

CHIRPS_URL_TEMPLATE = (
    "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p25/"
    "chirps-v2.0.{year}.days_p25.nc"
)

SPLIT = TemporalSplit(
    train_years=tuple(range(2007, 2016)),
    val_years=(2016, 2017),
    test_years=tuple(range(2018, 2025)),
)

#: Fixed pre-train-era climatology years (percentile baseline; never touched
#: by the temporal split, so percentiles cannot leak future information).
CLIMATOLOGY_YEARS = tuple(range(1981, 2007))
CLIMATOLOGY_ERA = "1981-2006"

#: CHIRPS years needed for sample windows: 2006 buffers the 60-day
#: antecedent window of early-2007 samples.
ERA_YEARS = tuple(range(2006, 2025))

MIN_EVENT_YEAR, MAX_EVENT_YEAR = 2007, 2024
MAX_ABS_LAT = 50.0
MAX_LOCATION_ACCURACY_KM = 25.0

#: Rain-family trigger vocabulary (GLC ``landslide1``-style values after
#: lowercasing/stripping; the raw catalog mixes cases and a few synonyms).
RAIN_TRIGGERS = frozenset(
    {
        "downpour",
        "rain",
        "rainfall",
        "heavy rain",
        "heavy rainfall",
        "continuous_rain",
        "monsoon",
        "tropical_cyclone",
        "flooding",
    }
)

#: Non-rain trigger values used to content-introspect the trigger column.
_OTHER_TRIGGERS = frozenset(
    {
        "unknown",
        "earthquake",
        "snowfall_snowmelt",
        "construction",
        "mining",
        "no_apparent_trigger",
        "freeze_thaw",
        "volcano",
        "other",
        "dam_embankment_collapse",
        "leaking_pipe",
        "vibration",
    }
)

#: Category vocabulary used to content-introspect the category column.
_CATEGORY_VOCAB = frozenset(
    {
        "landslide",
        "mudslide",
        "rock_fall",
        "rock fall",
        "rockfall",
        "rock_slide",
        "rock slide",
        "debris_flow",
        "debris flow",
        "complex",
        "rotational_slide",
        "rotational slide",
        "translational_slide",
        "translational slide",
        "riverbank_collapse",
        "riverbank collapse",
        "creep",
        "snow_avalanche",
        "earth_flow",
        "lahar",
        "topple",
        "other",
        "unknown",
    }
)

#: The detector's six type-classifier classes, in ``LandslideDetector``'s
#: exact output order (index = class id used by the type head).
LANDSLIDE_TYPE_LABELS = (
    "debris_flow",
    "rock_slide",
    "earth_flow",
    "snow_avalanche",
    "mud_flow",
    "rotational_slide",
)

#: Honest GLC-category -> detector-class mapping. Categories with no
#: defensible counterpart among the six classes (generic "landslide",
#: "unknown", "complex", "other", "translational_slide", "riverbank
#: collapse", "topple") map to -1 and are EXCLUDED from the type loss --
#: guessing a class for them would corrupt the head.
GLC_CATEGORY_TO_TYPE: dict[str, int] = {
    "debris_flow": 0,
    "debris flow": 0,
    "lahar": 0,  # volcanic debris flow
    "rock_fall": 1,
    "rock fall": 1,
    "rockfall": 1,
    "rock_slide": 1,
    "rock slide": 1,
    "earth_flow": 2,
    "earth flow": 2,
    "creep": 2,  # slow earth flow
    "snow_avalanche": 3,
    "mudslide": 4,
    "mud_flow": 4,
    "mud flow": 4,
    "rotational_slide": 5,
    "rotational slide": 5,
}

#: Negative-sampling policy (seeded; both schemes documented in the module
#: docstring and reported in ``extras``).
NEG_SAME_CELL_PER_POS = 2
NEG_FAR_CELL_PER_POS = 1
NEG_MIN_DAYS_FROM_EVENT = 60
NEG_FAR_MIN_KM = 100.0

ANTECEDENT_DAYS = 60  # window length is ANTECEDENT_DAYS + 1 (day-of included)
WET_DAY_MM = 1.0
_EARTH_RADIUS_KM = 6371.0
#: The detector's fixed neural-path alert threshold: ``predict_landslide``
#: declares imminence only above this slope-failure probability.
DEPLOYED_PROB_THRESHOLD = 0.6

RAIN_QUANTITY_KEYS = (
    "day0",
    "sum3",
    "sum7",
    "sum14",
    "sum30",
    "sum60",
    "max1d_30",
    "max3d_30",
    "wet30",
)

FEATURE_NAMES: tuple[str, ...] = (
    "rain_day0_log1p_mm",
    "rain_sum3d_log1p_mm",
    "rain_sum7d_log1p_mm",
    "rain_sum14d_log1p_mm",
    "rain_sum30d_log1p_mm",
    "rain_sum60d_log1p_mm",
    "rain_max1d_30d_log1p_mm",
    "rain_max3d_30d_log1p_mm",
    "wet_days_30d_frac",
    "pct_rain_day0",
    "pct_rain_sum3d",
    "pct_rain_sum7d",
    "pct_rain_sum14d",
    "pct_rain_sum30d",
    "pct_rain_sum60d",
    "pct_rain_max1d_30d",
    "pct_rain_max3d_30d",
    "pct_wet_days_30d",
    "lat_over_50",
    "abs_lat_over_50",
    "doy_sin_annual",
    "doy_cos_annual",
    "doy_sin_semiannual",
    "doy_cos_semiannual",
    "flag_rain_window",
    "flag_climatology",
    "slope_angle_deg_over_60",
    "flag_slope_angle",
    "soil_saturation_frac",
    "flag_soil_saturation",
    "displacement_mm_day_over_50",
    "flag_displacement",
    "seismic_pga_g",
    "flag_seismic_pga",
    "snowmelt_mm_day_over_50",
    "flag_snowmelt",
) + tuple(f"reserved_{i}" for i in range(36, FEATURE_DIM))

# CHIRPS p25 grid definition (validated against every opened file).
_GRID_LAT0, _GRID_LON0, _GRID_STEP = -49.875, -179.875, 0.25
_GRID_NLAT, _GRID_NLON = 400, 1440


# ---------------------------------------------------------------------------
# GLC catalog: fetch, field introspection, normalization
# ---------------------------------------------------------------------------


def _agol_page_path(land_dir: Path, page: int) -> Path:
    """Cache path of one AGOL query page."""
    return land_dir / f"glc_agol_p{page}.json"


def _fetch_glc_agol(land_dir: Path) -> list[dict[str, Any]]:
    """Fetch all AGOL pages (deterministic FID order) and return sources.

    Args:
        land_dir: Pipeline cache directory for this hook.

    Returns:
        Provenance entries (url/sha256/description) for every page fetched.

    Raises:
        RuntimeError: If a page is structurally not an AGOL feature response
            or the pagination never terminates within ``AGOL_MAX_PAGES``.
    """
    sources: list[dict[str, Any]] = []
    for page in range(AGOL_MAX_PAGES):
        url = AGOL_QUERY_URL.format(offset=page * AGOL_PAGE_SIZE, count=AGOL_PAGE_SIZE)
        path = cached_fetch(url, _agol_page_path(land_dir, page))
        raw = json.loads(path.read_text())
        if "features" not in raw or "fields" not in raw:
            raise RuntimeError(
                f"AGOL page {page} at {path} is not a feature response "
                f"(keys: {sorted(raw)}); refusing to treat it as catalog data"
            )
        sources.append(
            {
                "url": url,
                "sha256": sha256_file(path),
                "description": f"NASA Global Landslide Catalog (AGOL mirror), page {page}",
            }
        )
        if not raw.get("exceededTransferLimit"):
            return sources
    raise RuntimeError(
        f"AGOL pagination did not terminate within {AGOL_MAX_PAGES} pages; "
        "the layer grew unexpectedly -- refusing to fetch unbounded data"
    )


def _parse_event_date(raw: Any) -> _dt.date | None:
    """Parse a GLC event date from epoch-ms, ``YYYYMMDD``, or ISO-ish text.

    Numeric values are treated as epoch milliseconds only when their
    magnitude is plausible for one (>= 1e11, i.e. outside ~1967-1973);
    integral values in the ``YYYYMMDD`` range parse as such. Small numbers
    (fatality counts, IDs) therefore can NOT masquerade as 1970 dates --
    that guard keeps content-based column introspection honest.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        v = float(raw)
        if abs(v) >= 1e11:
            try:
                return _dt.datetime.fromtimestamp(v / 1000.0, tz=_dt.UTC).date()
            except (OverflowError, OSError, ValueError):
                return None
        if v == int(v) and 19000101 <= v <= 20351231:
            raw = str(int(v))
        else:
            return None
    text = str(raw).strip()
    if not text:
        return None
    for fmt in ("%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return _dt.datetime.strptime(text[: len(fmt) + 2], fmt).date()
        except ValueError:
            continue
    return None


_ACCURACY_RE = re.compile(r"^(?:known within )?(\d+)\s*km$")


def parse_accuracy_km(raw: Any) -> float | None:
    """Parse a GLC ``location_a`` accuracy value into a radius in km.

    ``"exact"``/``"known exactly"`` parse to 0.0; ``"25km"`` / ``"known
    within 25 km"`` parse to 25.0; unknown/blank/unparseable values return
    None (the caller excludes those events rather than guessing).

    Args:
        raw: Raw accuracy field value.

    Returns:
        Accuracy radius in km, or None when the accuracy is not stated.
    """
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text in ("exact", "known exactly"):
        return 0.0
    match = _ACCURACY_RE.match(text)
    if match:
        return float(match.group(1))
    return None


def _norm(value: Any) -> str:
    """Lowercase/strip a raw string field value ('' when absent)."""
    return str(value).strip().lower() if value is not None else ""


def introspect_glc_columns(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Identify the GLC logical columns by CONTENT, not by hardcoded names.

    Both mirrors truncate/shift field names (AGOL: 10-char DBF-style names
    like ``landslide_``/``landslide1``; the HDX shapefile shifts them
    again), so each logical column is located by validating values:

    * ``date``: most values parse to dates in 1900-2035.
    * ``latitude``/``longitude``: numeric columns in +/-90 / +/-180 with
      the pair chosen so latitude has the smaller value spread.
    * ``trigger``: string column with the largest overlap with the known
      trigger vocabulary (requires >= 3 distinct known values).
    * ``category``: same, against the landslide-category vocabulary.
    * ``accuracy``: column where the most values parse via
      :func:`parse_accuracy_km` or equal ``unknown``.

    Args:
        rows: Raw attribute dicts (AGOL ``attributes`` or DBF records).

    Returns:
        Mapping of logical name -> actual field name.

    Raises:
        RuntimeError: When any required logical column cannot be identified
            with confidence -- failing loud beats training on garbage.
    """
    if not rows:
        raise RuntimeError("cannot introspect GLC columns from zero rows")
    keys = list(rows[0].keys())
    sample = rows[: min(len(rows), 2000)]

    def _score_vocab(key: str, vocab: frozenset[str]) -> int:
        values = {_norm(r.get(key)) for r in sample}
        return len(values & vocab)

    def _frac(key: str, pred: Any) -> float:
        vals = [r.get(key) for r in sample if r.get(key) not in (None, "")]
        if not vals:
            return 0.0
        return sum(1 for v in vals if pred(v)) / len(vals)

    resolved: dict[str, str] = {}

    def _date_ok(v: Any) -> bool:
        d = _parse_event_date(v)
        return d is not None and 1900 <= d.year <= 2035

    date_scores = {k: _frac(k, _date_ok) for k in keys}
    # Prefer the event date over audit timestamps (submitted/edited) when
    # several date-like columns qualify: audit columns cluster after 2007
    # while event dates span the catalog; break ties by earliest minimum.
    date_candidates = [k for k, s in date_scores.items() if s >= 0.9]
    if not date_candidates:
        raise RuntimeError(f"no GLC date column found (scores: {date_scores})")

    def _min_year(key: str) -> int:
        years = [d.year for r in sample if (d := _parse_event_date(r.get(key))) is not None]
        return min(years) if years else 9999

    resolved["date"] = min(date_candidates, key=_min_year)

    def _is_num(v: Any) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    lat_cands = [
        k
        for k in keys
        if _frac(k, lambda v: _is_num(v) and -90.0 <= float(v) <= 90.0) >= 0.99
        and _frac(k, _is_num) >= 0.99
    ]
    lon_cands = [
        k
        for k in keys
        if _frac(k, lambda v: _is_num(v) and -180.0 <= float(v) <= 180.0) >= 0.99
        and _frac(k, _is_num) >= 0.99
    ]

    def _spread(key: str) -> float:
        vals = [float(r[key]) for r in sample if _is_num(r.get(key))]
        return float(np.ptp(vals)) if vals else 0.0

    # Longitude must show spread beyond +/-90 or be the wider-spread member
    # of a lat/lon-named pair; use name hints only to break content ties.
    lat_named = [k for k in lat_cands if _norm(k).startswith("lat")]
    lon_named = [k for k in lon_cands if _norm(k).startswith("lon")]
    if len(lat_named) == 1 and len(lon_named) == 1:
        resolved["latitude"], resolved["longitude"] = lat_named[0], lon_named[0]
    else:
        wide = [k for k in lon_cands if _spread(k) > 185.0]
        if not wide or not lat_cands:
            raise RuntimeError(
                f"cannot identify lat/lon columns (lat candidates {lat_cands}, "
                f"lon candidates {lon_cands})"
            )
        resolved["longitude"] = max(wide, key=_spread)
        lat_only = [k for k in lat_cands if k != resolved["longitude"]]
        resolved["latitude"] = min(lat_only, key=_spread)

    trigger_vocab = RAIN_TRIGGERS | _OTHER_TRIGGERS
    trig_scores = {k: _score_vocab(k, trigger_vocab) for k in keys}
    best_trig = max(trig_scores, key=lambda k: trig_scores[k])
    if trig_scores[best_trig] < 3:
        raise RuntimeError(f"no GLC trigger column found (vocab overlap: {trig_scores})")
    resolved["trigger"] = best_trig

    cat_scores = {k: _score_vocab(k, _CATEGORY_VOCAB) for k in keys if k != resolved["trigger"]}
    best_cat = max(cat_scores, key=lambda k: cat_scores[k])
    if cat_scores[best_cat] < 3:
        raise RuntimeError(f"no GLC category column found (vocab overlap: {cat_scores})")
    resolved["category"] = best_cat

    def _acc_ok(v: Any) -> bool:
        return parse_accuracy_km(v) is not None or _norm(v) == "unknown"

    acc_scores = {k: _frac(k, _acc_ok) for k in keys if not _is_num(sample[0].get(k))}
    best_acc = max(acc_scores, key=lambda k: acc_scores[k])
    if acc_scores[best_acc] < 0.5:
        raise RuntimeError(f"no GLC accuracy column found (scores: {acc_scores})")
    resolved["accuracy"] = best_acc
    logger.info("GLC columns introspected: %s", resolved)
    return resolved


@dataclass(frozen=True)
class GlcEvent:
    """One normalized GLC event (only the fields this pipeline consumes)."""

    date: _dt.date
    lat: float
    lon: float
    trigger: str
    category: str
    accuracy_km: float | None


def normalize_glc_rows(rows: list[dict[str, Any]]) -> list[GlcEvent]:
    """Normalize raw catalog rows via content-introspected columns.

    Rows without a parseable date or coordinates are dropped (counted by the
    caller via length difference); trigger/category are case-normalized.

    Args:
        rows: Raw attribute dicts from either mirror.

    Returns:
        Normalized events (order preserved).
    """
    cols = introspect_glc_columns(rows)
    events: list[GlcEvent] = []
    for row in rows:
        date = _parse_event_date(row.get(cols["date"]))
        lat_raw, lon_raw = row.get(cols["latitude"]), row.get(cols["longitude"])
        if date is None or lat_raw is None or lon_raw is None:
            continue
        lat, lon = float(lat_raw), float(lon_raw)
        if not (np.isfinite(lat) and np.isfinite(lon)) or abs(lat) > 90 or abs(lon) > 180:
            continue
        events.append(
            GlcEvent(
                date=date,
                lat=lat,
                lon=lon,
                trigger=_norm(row.get(cols["trigger"])),
                category=_norm(row.get(cols["category"])),
                accuracy_km=parse_accuracy_km(row.get(cols["accuracy"])),
            )
        )
    if not events:
        raise RuntimeError("GLC normalization produced zero events; catalog format changed?")
    return events


def _parse_dbf(data: bytes) -> list[dict[str, Any]]:
    """Parse a dBase-III .dbf table into row dicts (HDX shapefile fallback).

    Only the record data is read (no shapes needed -- the GLC table carries
    latitude/longitude columns). Field values are converted by DBF type:
    ``N``/``F`` to float, ``D`` to the raw ``YYYYMMDD`` string (the shared
    date parser handles it), everything else to stripped text.

    Args:
        data: Raw .dbf file bytes.

    Returns:
        One dict per non-deleted record.

    Raises:
        RuntimeError: On a structurally invalid header.
    """
    if len(data) < 33:
        raise RuntimeError("DBF too short to contain a header")
    n_records = int.from_bytes(data[4:8], "little")
    header_len = int.from_bytes(data[8:10], "little")
    record_len = int.from_bytes(data[10:12], "little")
    fields: list[tuple[str, str, int]] = []
    off = 32
    while off < header_len - 1 and data[off] != 0x0D:
        name = data[off : off + 11].split(b"\x00")[0].decode("ascii", "replace")
        ftype = chr(data[off + 11])
        flen = data[off + 16]
        fields.append((name, ftype, flen))
        off += 32
    if not fields or 1 + sum(f[2] for f in fields) != record_len:
        raise RuntimeError(
            f"DBF field descriptors ({len(fields)} fields) do not add up to the "
            f"declared record length {record_len}; refusing to parse"
        )
    rows: list[dict[str, Any]] = []
    pos = header_len
    for _ in range(n_records):
        rec = data[pos : pos + record_len]
        pos += record_len
        if len(rec) < record_len or rec[:1] == b"*":
            continue
        row: dict[str, Any] = {}
        o = 1
        for name, ftype, flen in fields:
            text = rec[o : o + flen].decode("latin-1").strip()
            o += flen
            if ftype in ("N", "F"):
                try:
                    row[name] = float(text) if text else None
                except ValueError:
                    row[name] = None
            else:
                row[name] = text or None
        rows.append(row)
    return rows


def _load_glc_rows(land_dir: Path) -> list[dict[str, Any]]:
    """Load raw GLC rows from cached AGOL pages, else the local HDX zip.

    Raises:
        FileNotFoundError: When neither source is cached (run --fetch).
    """
    rows: list[dict[str, Any]] = []
    page = 0
    while _agol_page_path(land_dir, page).exists():
        raw = json.loads(_agol_page_path(land_dir, page).read_text())
        rows.extend(f["attributes"] for f in raw["features"])
        page += 1
    if rows:
        logger.info("loaded %d GLC rows from %d cached AGOL pages", len(rows), page)
        return rows
    hdx_zip = land_dir / "global_landslide_catalog_nasa.zip"
    if hdx_zip.exists():
        with zipfile.ZipFile(hdx_zip) as zf:
            dbf_names = [n for n in zf.namelist() if n.lower().endswith(".dbf")]
            if not dbf_names:
                raise RuntimeError(f"{hdx_zip} contains no .dbf table")
            rows = _parse_dbf(zf.read(dbf_names[0]))
        logger.info("loaded %d GLC rows from HDX shapefile %s", len(rows), hdx_zip)
        return rows
    raise FileNotFoundError(
        f"no cached GLC catalog under {land_dir} (AGOL pages or HDX zip); "
        "run the --fetch stage first"
    )


# ---------------------------------------------------------------------------
# CHIRPS: grid math, cell-series extraction, climatology
# ---------------------------------------------------------------------------


def _chirps_path(land_dir: Path, year: int) -> Path:
    """Cache path of one CHIRPS yearly netCDF file."""
    return land_dir / "chirps" / f"chirps-v2.0.{year}.days_p25.nc"


def cell_of(lat: float, lon: float) -> tuple[int, int]:
    """Map a coordinate to its 0.25-deg CHIRPS p25 cell (lat_idx, lon_idx).

    Args:
        lat: Latitude in degrees (must satisfy ``|lat| <= 50``).
        lon: Longitude in degrees ([-180, 180]; 180 wraps to -180).

    Returns:
        Tuple of (latitude index 0..399, longitude index 0..1439).

    Raises:
        ValueError: If the latitude is outside CHIRPS coverage.
    """
    if abs(lat) > MAX_ABS_LAT:
        raise ValueError(f"latitude {lat} outside CHIRPS coverage (|lat| <= 50)")
    lat_idx = int(np.clip(round((lat - _GRID_LAT0) / _GRID_STEP), 0, _GRID_NLAT - 1))
    lon_idx = round((lon - _GRID_LON0) / _GRID_STEP) % _GRID_NLON
    return lat_idx, lon_idx


def _cells_hash(cells: list[tuple[int, int]], years: tuple[int, ...]) -> str:
    """Deterministic cache key for a (cell set, year range) extraction."""
    payload = json.dumps({"cells": sorted(cells), "years": list(years)}).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _extract_cell_series(
    land_dir: Path, years: tuple[int, ...], cells: list[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    """Extract daily rainfall series for a fixed cell set from CHIRPS files.

    Reads each yearly netCDF in ~61-day time slabs (verified to slice lazily
    without loading the whole file) and keeps only the requested cells, so
    memory stays bounded. Results are cached as one npz per (cell set,
    years) key. Ocean/missing cells hold NaN.

    Args:
        land_dir: Pipeline cache directory.
        years: Yearly files to read (must all be cached by --fetch).
        cells: (lat_idx, lon_idx) cells to keep, in output column order.

    Returns:
        Tuple of (dates, matrix): proleptic-Gregorian ordinal day numbers
        ``[n_days]`` and rainfall mm/day ``[n_days, n_cells]`` (float32,
        NaN = missing).

    Raises:
        FileNotFoundError: When a yearly file is not cached.
        RuntimeError: When a file's grid does not match the p25 definition.
    """
    cache = land_dir / "cells" / f"cells_{_cells_hash(cells, years)}.npz"
    if cache.exists():
        with np.load(cache) as npz:
            return npz["dates"], npz["matrix"]

    import netCDF4  # lazy: only the pipeline needs it, not the detector

    lat_idx = np.array([c[0] for c in cells], dtype=np.int64)
    lon_idx = np.array([c[1] for c in cells], dtype=np.int64)
    all_dates: list[np.ndarray] = []
    all_rows: list[np.ndarray] = []
    base_ordinal = _dt.date(1980, 1, 1).toordinal()
    for year in years:
        path = _chirps_path(land_dir, year)
        if not path.exists():
            raise FileNotFoundError(f"missing CHIRPS cache file {path}; run --fetch first")
        with netCDF4.Dataset(path) as ds:
            lats = np.asarray(ds.variables["latitude"][:], dtype=np.float64)
            lons = np.asarray(ds.variables["longitude"][:], dtype=np.float64)
            expected_lat = _GRID_LAT0 + _GRID_STEP * np.arange(_GRID_NLAT)
            expected_lon = _GRID_LON0 + _GRID_STEP * np.arange(_GRID_NLON)
            if lats.shape != (_GRID_NLAT,) or lons.shape != (_GRID_NLON,):
                raise RuntimeError(f"{path}: unexpected grid shape {lats.shape}/{lons.shape}")
            if not (
                np.allclose(lats, expected_lat, atol=1e-4)
                and np.allclose(lons, expected_lon, atol=1e-4)
            ):
                raise RuntimeError(f"{path}: grid coordinates deviate from the p25 definition")
            times = np.asarray(ds.variables["time"][:], dtype=np.float64)
            units = str(ds.variables["time"].units)
            if not units.startswith("days since 1980-1-1"):
                raise RuntimeError(f"{path}: unexpected time units {units!r}")
            dates = base_ordinal + np.round(times).astype(np.int64)
            precip = ds.variables["precip"]
            n_days = precip.shape[0]
            year_mat = np.empty((n_days, len(cells)), dtype=np.float32)
            for start in range(0, n_days, 61):
                stop = min(start + 61, n_days)
                slab = precip[start:stop, :, :]
                filled = np.ma.filled(slab.astype(np.float32), np.nan)
                year_mat[start:stop] = filled[:, lat_idx, lon_idx]
            all_dates.append(dates)
            all_rows.append(year_mat)
        logger.info("extracted %d cells from CHIRPS %d", len(cells), year)
    dates_arr = np.concatenate(all_dates)
    matrix = np.concatenate(all_rows, axis=0)
    order = np.argsort(dates_arr, kind="mergesort")
    dates_arr, matrix = dates_arr[order], matrix[order]
    if np.any(np.diff(dates_arr) != 1):
        raise RuntimeError("CHIRPS day sequence has gaps/duplicates after concatenation")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, dates=dates_arr, matrix=matrix)
    return dates_arr, matrix


def _land_mask(land_dir: Path) -> np.ndarray:
    """CHIRPS land mask [400, 1440] from day 0 of the first climatology year."""
    import netCDF4  # lazy

    path = _chirps_path(land_dir, CLIMATOLOGY_YEARS[0])
    if not path.exists():
        raise FileNotFoundError(f"missing CHIRPS cache file {path}; run --fetch first")
    with netCDF4.Dataset(path) as ds:
        day0 = ds.variables["precip"][0, :, :]
    return ~np.ma.getmaskarray(day0)


def _rolling_sum(values: np.ndarray, k: int) -> np.ndarray:
    """Rolling k-day sum ending at each index (NaN when incomplete/missing)."""
    x = np.asarray(values, dtype=np.float64)
    out = np.full(x.size, np.nan)
    if x.size < k:
        return out
    cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x, nan=0.0))])
    bad = np.concatenate([[0], np.cumsum(np.isnan(x).astype(np.int64))])
    sums = cs[k:] - cs[:-k]
    nan_counts = bad[k:] - bad[:-k]
    sums[nan_counts > 0] = np.nan
    out[k - 1 :] = sums
    return out


def _rolling_max(values: np.ndarray, k: int) -> np.ndarray:
    """Rolling k-day max ending at each index (NaN when incomplete/missing)."""
    from numpy.lib.stride_tricks import sliding_window_view

    x = np.asarray(values, dtype=np.float64)
    out = np.full(x.size, np.nan)
    if x.size >= k:
        out[k - 1 :] = np.max(sliding_window_view(x, k), axis=1)
    return out


def _derived_series(daily_mm: np.ndarray) -> dict[str, np.ndarray]:
    """All nine rain quantities as day-aligned rolling series.

    The value at index ``i`` uses ONLY days ``<= i`` (window ending at
    ``i``), which is the no-lookahead property the feature builder and the
    climatology tables both inherit from this single implementation.

    Args:
        daily_mm: Daily rainfall series (mm/day, NaN = missing).

    Returns:
        Mapping of quantity key -> aligned series (NaN where the trailing
        window is incomplete or contains missing days).
    """
    x = np.asarray(daily_mm, dtype=np.float64)
    sum3 = _rolling_sum(x, 3)
    wet = np.where(np.isnan(x), np.nan, (x >= WET_DAY_MM).astype(np.float64))
    return {
        "day0": x.copy(),
        "sum3": sum3,
        "sum7": _rolling_sum(x, 7),
        "sum14": _rolling_sum(x, 14),
        "sum30": _rolling_sum(x, 30),
        "sum60": _rolling_sum(x, 60),
        "max1d_30": _rolling_max(x, 30),
        "max3d_30": _rolling_max(sum3, 30),
        "wet30": _rolling_sum(wet, 30),
    }


def compute_rain_quantities(daily_mm: np.ndarray, day_index: int) -> dict[str, float]:
    """Rain quantities for the 61-day window ending AT ``day_index``.

    Only ``daily_mm[day_index - 60 : day_index + 1]`` is ever touched, so
    the result provably cannot depend on data after the event day (the
    no-lookahead unit test perturbs later days and asserts equality).

    Args:
        daily_mm: Daily rainfall series for one cell (mm/day).
        day_index: Index of the event day within ``daily_mm``.

    Returns:
        Mapping with the nine :data:`RAIN_QUANTITY_KEYS` (NaN where the
        window is incomplete) plus ``antecedent_7day_mm`` (sum of the seven
        days strictly BEFORE the event day, for the physics path).

    Raises:
        ValueError: If fewer than 60 days precede ``day_index``.
    """
    if day_index < ANTECEDENT_DAYS or day_index >= len(daily_mm):
        raise ValueError(
            f"day_index {day_index} needs {ANTECEDENT_DAYS} antecedent days within "
            f"a series of length {len(daily_mm)}"
        )
    window = np.asarray(daily_mm[day_index - ANTECEDENT_DAYS : day_index + 1], dtype=np.float64)
    derived = _derived_series(window)
    out = {key: float(derived[key][-1]) for key in RAIN_QUANTITY_KEYS}
    out["antecedent_7day_mm"] = float(np.sum(window[-8:-1]))
    return out


def climatology_tables(clim_daily_mm: np.ndarray) -> dict[str, np.ndarray] | None:
    """Sorted climatology distributions of every rain quantity for one cell.

    Args:
        clim_daily_mm: The cell's full daily series over the fixed
            climatology era (1981-2006).

    Returns:
        Mapping of quantity key -> ascending-sorted finite climatology
        values, or None when fewer than 365 finite window values exist
        (ocean/degenerate cells) -- the feature builder then sets the
        climatology presence flag to 0 instead of fabricating percentiles.
    """
    derived = _derived_series(np.asarray(clim_daily_mm, dtype=np.float64))
    tables: dict[str, np.ndarray] = {}
    for key in RAIN_QUANTITY_KEYS:
        finite = derived[key][np.isfinite(derived[key])]
        if finite.size < 365:
            return None
        tables[key] = np.sort(finite)
    return tables


def percentile_of(sorted_values: np.ndarray, value: float) -> float:
    """Midrank percentile of ``value`` within an ascending-sorted sample.

    Deterministic and tie-aware: equal values contribute the average of
    their left/right ranks (a degenerate all-equal distribution yields 0.5).

    Args:
        sorted_values: Ascending-sorted finite sample.
        value: Query value (NaN returns NaN).

    Returns:
        Percentile in [0, 1] (NaN for a NaN query).
    """
    if not np.isfinite(value):
        return float("nan")
    n = sorted_values.size
    left = int(np.searchsorted(sorted_values, value, side="left"))
    right = int(np.searchsorted(sorted_values, value, side="right"))
    return float((left + right) / (2.0 * n))


def build_feature_vector(
    quantities: Mapping[str, float],
    percentiles: Mapping[str, float] | None,
    lat: float,
    day_of_year: int,
) -> np.ndarray:
    """Build the canonical 64-dim ``landslide-coolr-v1`` feature vector.

    Args:
        quantities: The nine rain quantities from
            :func:`compute_rain_quantities` (NaN = unobserved; the rain
            presence flag drops to 0 and the dims are zero-filled).
        percentiles: Climatology percentiles for the same keys, or None
            when the cell has no usable climatology (flag 0, dims 0.5 --
            the uninformative midrank, never a fabricated extreme).
        lat: Sample latitude in degrees.
        day_of_year: 1-based day of year of the sample date.

    Returns:
        ``float32`` array of shape ``(64,)``. Site dims 26-35 are always
        zero with presence flags 0: this corpus has no slope/geotechnical
        observations (see module docstring).
    """
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)
    raw = np.array([float(quantities[k]) for k in RAIN_QUANTITY_KEYS], dtype=np.float64)
    window_complete = bool(np.all(np.isfinite(raw)))
    filled = np.nan_to_num(raw, nan=0.0)
    vec[0:8] = np.log1p(np.maximum(filled[0:8], 0.0))
    vec[8] = filled[8] / 30.0
    if percentiles is not None:
        pct = np.array([float(percentiles[k]) for k in RAIN_QUANTITY_KEYS], dtype=np.float64)
        vec[9:18] = np.nan_to_num(pct, nan=0.5)
        vec[25] = 1.0
    else:
        vec[9:18] = 0.5
    angle = 2.0 * np.pi * float(day_of_year) / 365.25
    vec[18] = float(lat) / MAX_ABS_LAT
    vec[19] = abs(float(lat)) / MAX_ABS_LAT
    vec[20] = np.sin(angle)
    vec[21] = np.cos(angle)
    vec[22] = np.sin(2.0 * angle)
    vec[23] = np.cos(2.0 * angle)
    vec[24] = 1.0 if window_complete else 0.0
    return vec


# ---------------------------------------------------------------------------
# Stage 1: fetch
# ---------------------------------------------------------------------------


def fetch(ctx: PipelineContext) -> dict[str, Any]:
    """Download the GLC catalog pages and every CHIRPS year the split needs.

    Args:
        ctx: Pipeline context (data dir).

    Returns:
        Manifest with per-file URLs and SHA-256 digests, also written to
        ``<data_dir>/landslide/manifest.json``.
    """
    land_dir = ctx.data_dir / "landslide"
    try:
        sources = _fetch_glc_agol(land_dir)
    except Exception as exc:
        logger.warning(
            "AGOL GLC fetch failed (%s); trying the HDX shapefile fallback. NOTE: "
            "the HDX filestore 302-redirects to a signed S3 URL that the "
            "no-redirect transport refuses -- an operator must place the zip at "
            "%s by hand if this fails.",
            exc,
            land_dir / "global_landslide_catalog_nasa.zip",
        )
        path = cached_fetch(HDX_ZIP_URL, land_dir / "global_landslide_catalog_nasa.zip")
        sources = [
            {
                "url": HDX_ZIP_URL,
                "sha256": sha256_file(path),
                "description": "NASA Global Landslide Catalog (HDX shapefile mirror)",
            }
        ]
    for year in sorted({*CLIMATOLOGY_YEARS, *ERA_YEARS}):
        url = CHIRPS_URL_TEMPLATE.format(year=year)
        path = cached_fetch(url, _chirps_path(land_dir, year), timeout=560.0)
        sources.append(
            {
                "url": url,
                "sha256": sha256_file(path),
                "description": f"CHIRPS v2.0 global daily 0.25deg precipitation, {year}",
            }
        )
    rows = _load_glc_rows(land_dir)
    events = normalize_glc_rows(rows)
    manifest = {
        "hook": HOOK_NAME,
        "sources": sources,
        "glc_rows": len(rows),
        "glc_events_with_date_and_coords": len(events),
    }
    manifest_path = land_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    logger.info(
        "fetch complete: %d sources, %d GLC rows (%d dated+located)",
        len(sources),
        len(rows),
        len(events),
    )
    return manifest


# ---------------------------------------------------------------------------
# Stage 2: build dataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Sample:
    """One (cell, date) candidate before feature extraction."""

    cell: tuple[int, int]
    date: _dt.date
    lat: float
    label: int
    type_idx: int
    scheme: str  # "positive" | "neg_same_cell" | "neg_far_cell"


@dataclass
class LandslideDataset:
    """Assembled feature/label matrices plus per-sample physics inputs."""

    features: np.ndarray
    labels: np.ndarray
    type_idx: np.ndarray
    years: np.ndarray
    raw_fields: list[dict[str, float]]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    extras: dict[str, Any] = field(default_factory=dict)


def _filter_positives(
    events: list[GlcEvent],
) -> tuple[list[GlcEvent], dict[str, Any]]:
    """Apply the positive-event filters, reporting every cutoff's effect.

    Returns:
        Tuple of (positives, report) where the report carries the trigger
        histogram, the accuracy distribution, and per-filter drop counts.
    """
    trigger_hist: dict[str, int] = {}
    accuracy_hist: dict[str, int] = {}
    report: dict[str, Any] = {}
    in_era = [e for e in events if MIN_EVENT_YEAR <= e.date.year <= MAX_EVENT_YEAR]
    report["events_in_2007_2024"] = len(in_era)
    rain = []
    for e in in_era:
        trigger_hist[e.trigger or "(blank)"] = trigger_hist.get(e.trigger or "(blank)", 0) + 1
        if e.trigger in RAIN_TRIGGERS:
            rain.append(e)
    report["rain_trigger_histogram"] = dict(sorted(trigger_hist.items(), key=lambda kv: -kv[1]))
    report["rain_triggered"] = len(rain)
    in_lat = [e for e in rain if abs(e.lat) <= MAX_ABS_LAT]
    report["dropped_outside_lat50"] = len(rain) - len(in_lat)
    for e in in_lat:
        key = "unknown" if e.accuracy_km is None else f"{e.accuracy_km:g}km"
        accuracy_hist[key] = accuracy_hist.get(key, 0) + 1
    positives = [
        e for e in in_lat if e.accuracy_km is not None and e.accuracy_km <= MAX_LOCATION_ACCURACY_KM
    ]
    report["accuracy_histogram"] = dict(sorted(accuracy_hist.items(), key=lambda kv: -kv[1]))
    report["accuracy_cutoff_km"] = MAX_LOCATION_ACCURACY_KM
    report["dropped_by_accuracy_cutoff"] = len(in_lat) - len(positives)
    report["positives_after_filters"] = len(positives)
    return positives, report


def _haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    """Great-circle distance (km) from arrays of points to one point."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2.0) ** 2
    return 2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _sample_negatives(
    rng: np.random.Generator,
    positives: list[GlcEvent],
    all_events: list[GlcEvent],
    land: np.ndarray,
) -> tuple[list[_Sample], dict[str, int]]:
    """Draw the two seeded negative schemes (see module docstring).

    Args:
        rng: Seeded generator (determinism).
        positives: Filtered positive events (negatives are year-matched to
            them so the class balance is stable across split years).
        all_events: The FULL normalized catalog (any trigger/accuracy) --
            exclusion zones use everything known, not just the positives.
        land: CHIRPS land mask [400, 1440].

    Returns:
        Tuple of (negative samples, per-scheme counts including skips).
    """
    events_by_cell: dict[tuple[int, int], list[int]] = {}
    ev_lat = np.array([e.lat for e in all_events])
    ev_lon = np.array([e.lon for e in all_events])
    ev_ord = np.array([e.date.toordinal() for e in all_events], dtype=np.int64)
    for e in all_events:
        if abs(e.lat) <= MAX_ABS_LAT:
            events_by_cell.setdefault(cell_of(e.lat, e.lon), []).append(e.date.toordinal())

    land_cells = np.argwhere(land)
    negatives: list[_Sample] = []
    counts = {
        "neg_same_cell": 0,
        "neg_far_cell": 0,
        "neg_same_cell_skipped": 0,
        "neg_far_cell_skipped": 0,
    }

    def _random_ordinal(year: int) -> int:
        start = _dt.date(year, 1, 1).toordinal()
        end = _dt.date(year, 12, 31).toordinal()
        return int(rng.integers(start, end + 1))

    for pos in positives:
        cell = cell_of(pos.lat, pos.lon)
        cell_events = np.asarray(events_by_cell.get(cell, []), dtype=np.int64)
        drawn = 0
        for _ in range(200):
            if drawn >= NEG_SAME_CELL_PER_POS:
                break
            cand = _random_ordinal(pos.date.year)
            if cell_events.size and np.min(np.abs(cell_events - cand)) < NEG_MIN_DAYS_FROM_EVENT:
                continue
            negatives.append(
                _Sample(
                    cell=cell,
                    date=_dt.date.fromordinal(cand),
                    lat=pos.lat,
                    label=0,
                    type_idx=-1,
                    scheme="neg_same_cell",
                )
            )
            drawn += 1
        counts["neg_same_cell"] += drawn
        counts["neg_same_cell_skipped"] += NEG_SAME_CELL_PER_POS - drawn

        drawn = 0
        for _ in range(200):
            if drawn >= NEG_FAR_CELL_PER_POS:
                break
            li, lo = land_cells[int(rng.integers(0, len(land_cells)))]
            cell_lat = _GRID_LAT0 + _GRID_STEP * float(li)
            cell_lon = _GRID_LON0 + _GRID_STEP * float(lo)
            cand = _random_ordinal(pos.date.year)
            near_time = np.abs(ev_ord - cand) <= NEG_MIN_DAYS_FROM_EVENT
            if near_time.any():
                dists = _haversine_km(ev_lat[near_time], ev_lon[near_time], cell_lat, cell_lon)
                if float(np.min(dists)) < NEG_FAR_MIN_KM:
                    continue
            negatives.append(
                _Sample(
                    cell=(int(li), int(lo)),
                    date=_dt.date.fromordinal(cand),
                    lat=cell_lat,
                    label=0,
                    type_idx=-1,
                    scheme="neg_far_cell",
                )
            )
            drawn += 1
        counts["neg_far_cell"] += drawn
        counts["neg_far_cell_skipped"] += NEG_FAR_CELL_PER_POS - drawn
    return negatives, counts


def build_dataset(ctx: PipelineContext) -> LandslideDataset:
    """Assemble the feature/label dataset from cached GLC + CHIRPS files.

    Standardization statistics come from TRAIN-year rows only, and every
    climatology percentile uses the fixed 1981-2006 era, so nothing from
    validation/test years influences any statistic.

    Args:
        ctx: Pipeline context (data dir, seed, optional sample cap).

    Returns:
        The assembled dataset.

    Raises:
        RuntimeError: On empty splits or a corrupted CHIRPS sequence.
    """
    land_dir = ctx.data_dir / "landslide"
    rng = np.random.default_rng(ctx.seed)
    events = normalize_glc_rows(_load_glc_rows(land_dir))
    positives, report = _filter_positives(events)
    if not positives:
        raise RuntimeError("zero positive events after filters; cannot build a dataset")

    land = _land_mask(land_dir)
    kept_pos: list[_Sample] = []
    dropped_ocean = 0
    for e in positives:
        cell = cell_of(e.lat, e.lon)
        if not land[cell[0], cell[1]]:
            dropped_ocean += 1
            continue
        kept_pos.append(
            _Sample(
                cell=cell,
                date=e.date,
                lat=e.lat,
                label=1,
                type_idx=GLC_CATEGORY_TO_TYPE.get(e.category, -1),
                scheme="positive",
            )
        )
    report["dropped_positives_on_nonland_cells"] = dropped_ocean

    negatives, neg_counts = _sample_negatives(
        rng, [e for e in positives if land[cell_of(e.lat, e.lon)]], events, land
    )
    report.update(neg_counts)

    samples = sorted(
        kept_pos + negatives, key=lambda s: (s.date.toordinal(), s.cell, s.label, s.scheme)
    )
    if ctx.limit_samples is not None:
        samples = samples[: ctx.limit_samples]

    cells = sorted({s.cell for s in samples})
    cell_col = {c: j for j, c in enumerate(cells)}
    era_dates, era_mat = _extract_cell_series(land_dir, ERA_YEARS, cells)
    clim_mat = _extract_cell_series(land_dir, CLIMATOLOGY_YEARS, cells)[1]
    date_row = {int(d): i for i, d in enumerate(era_dates)}

    by_cell: dict[tuple[int, int], list[int]] = {}
    for i, s in enumerate(samples):
        by_cell.setdefault(s.cell, []).append(i)

    features = np.zeros((len(samples), FEATURE_DIM), dtype=np.float32)
    raw_fields: list[dict[str, float] | None] = [None] * len(samples)
    dropped_no_rain = 0
    keep = np.zeros(len(samples), dtype=bool)
    cells_without_climatology = 0
    for cell, idxs in by_cell.items():
        col = cell_col[cell]
        tables = climatology_tables(clim_mat[:, col])
        if tables is None:
            cells_without_climatology += 1
        series = era_mat[:, col].astype(np.float64)
        for i in idxs:
            s = samples[i]
            row = date_row.get(s.date.toordinal())
            if row is None or row < ANTECEDENT_DAYS:
                dropped_no_rain += 1
                continue
            q = compute_rain_quantities(series, row)
            if not (np.isfinite(q["day0"]) and np.isfinite(q["antecedent_7day_mm"])):
                dropped_no_rain += 1
                continue
            pct = (
                {k: percentile_of(tables[k], q[k]) for k in RAIN_QUANTITY_KEYS}
                if tables is not None
                else None
            )
            features[i] = build_feature_vector(q, pct, s.lat, s.date.timetuple().tm_yday)
            raw_fields[i] = {
                "intensity_mm_hr": q["day0"] / 24.0,
                "duration_hours": 24.0,
                "antecedent_7day_mm": q["antecedent_7day_mm"],
            }
            keep[i] = True
    report["dropped_samples_without_chirps_rain"] = dropped_no_rain
    report["cells_without_climatology"] = cells_without_climatology

    kept_idx = np.flatnonzero(keep)
    if kept_idx.size == 0:
        raise RuntimeError("no samples survived CHIRPS extraction; cannot build a dataset")
    features = features[kept_idx]
    labels = np.array([samples[i].label for i in kept_idx], dtype=np.float32)
    type_idx = np.array([samples[i].type_idx for i in kept_idx], dtype=np.int64)
    years = np.array([samples[i].date.year for i in kept_idx], dtype=np.int64)
    fields_list = [f for i in kept_idx if (f := raw_fields[i]) is not None]
    if len(fields_list) != kept_idx.size:
        raise RuntimeError("internal invariant broken: kept sample without physics fields")

    train_mask, val_mask, test_mask = SPLIT.masks(years)
    if not (train_mask.any() and val_mask.any() and test_mask.any()):
        raise RuntimeError("a split has zero samples; check the catalog density per year")
    mean = features[train_mask].mean(axis=0)
    std = features[train_mask].std(axis=0)
    std[std < 1e-6] = 1.0

    def _split_counts(mask: np.ndarray) -> dict[str, int]:
        return {
            "total": int(mask.sum()),
            "positives": int(labels[mask].sum()),
            "negatives": int(mask.sum() - labels[mask].sum()),
        }

    report["split_counts"] = {
        "train_2007_2015": _split_counts(train_mask),
        "val_2016_2017": _split_counts(val_mask),
        "test_2018_2024": _split_counts(test_mask),
    }
    report["type_class_counts"] = {
        label: int(np.sum(type_idx == k)) for k, label in enumerate(LANDSLIDE_TYPE_LABELS)
    }
    report["type_unmapped_positives"] = int(np.sum((type_idx < 0) & (labels == 1)))
    logger.info("dataset assembled: %s", report["split_counts"])

    return LandslideDataset(
        features=features,
        labels=labels,
        type_idx=type_idx,
        years=years,
        raw_fields=[dict(f) for f in fields_list],
        feature_mean=mean.astype(np.float32),
        feature_std=std.astype(np.float32),
        extras=report,
    )


# ---------------------------------------------------------------------------
# Stage 3: train
# ---------------------------------------------------------------------------


def _fold_standardization(model: Any, mean: np.ndarray, std: np.ndarray) -> None:
    """Fold ``(x - mean) / std`` into the first encoder Linear layer.

    After folding, the model consumes RAW canonical feature vectors --
    exactly what ``LandslideDetector._assess_slope_stability`` passes at
    inference -- while remaining numerically identical to the network that
    was trained on standardized inputs (pure linear algebra on layer 0).

    Args:
        model: A ``SlopeStabilityModel`` whose encoder starts with Linear.
        mean: Train-year feature means (canonical space).
        std: Train-year feature stds (floored, canonical space).
    """
    first = model.feature_encoder[0]
    if not isinstance(first, torch.nn.Linear):
        raise TypeError("SlopeStabilityModel.feature_encoder[0] is not Linear; cannot fold")
    with torch.no_grad():
        std_t = torch.from_numpy(std.astype(np.float32))
        mean_t = torch.from_numpy(mean.astype(np.float32))
        folded_w = first.weight / std_t  # broadcast over input columns
        folded_b = first.bias - folded_w @ mean_t
        first.weight.copy_(folded_w)
        first.bias.copy_(folded_b)


def train(ctx: PipelineContext) -> dict[str, Any]:
    """Train the SlopeStabilityModel with early stopping on validation AUC.

    The occurrence head is optimized with positive-weighted BCE (the seeded
    3:1 negative scheme fixes the base rate at 25%); the type head with
    cross-entropy over the honestly mapped GLC categories only (unmapped
    categories and all negatives are excluded via mask, never guessed).
    After restoring the best-validation-AUC weights, the train-year
    standardization is folded into the first encoder layer (verified
    numerically) so the shipped checkpoint consumes raw canonical vectors.

    Args:
        ctx: Pipeline context.

    Returns:
        Training record (epochs, best validation AUC, sample counts).
    """
    from omni_mercury_engine.detectors.geological.landslide import SlopeStabilityModel

    rng = seed_everything(ctx.seed)
    ds = build_dataset(ctx)
    train_mask, val_mask, _ = SPLIT.masks(ds.years)
    x = (ds.features - ds.feature_mean) / ds.feature_std
    x_train = torch.from_numpy(x[train_mask])
    y_train = torch.from_numpy(ds.labels[train_mask])
    t_train = torch.from_numpy(ds.type_idx[train_mask])
    x_val = torch.from_numpy(x[val_mask])
    y_val = ds.labels[val_mask]

    n_pos = float(y_train.sum().item())
    n_neg = float(len(y_train) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise RuntimeError("training years contain a single class; cannot train honestly")
    pos_weight = n_neg / n_pos

    model = SlopeStabilityModel(input_dim=FEATURE_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    logger.info(
        "training on %d rows (%.1f%% positive, pos_weight %.2f), validating on %d rows",
        x_train.shape[0],
        100.0 * n_pos / len(y_train),
        pos_weight,
        x_val.shape[0],
    )

    batch_size = 128
    type_loss_weight = 0.3
    best_val_auc = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience, bad_epochs = 8, 0
    epochs_run = 0

    for epoch in range(ctx.max_epochs):
        epochs_run = epoch + 1
        model.train()
        perm = torch.from_numpy(rng.permutation(x_train.shape[0]))
        epoch_loss = 0.0
        for start in range(0, x_train.shape[0], batch_size):
            batch_idx = perm[start : start + batch_size]
            if batch_idx.shape[0] < 2:
                continue  # BatchNorm needs >1 sample
            xb, yb, tb = x_train[batch_idx], y_train[batch_idx], t_train[batch_idx]
            failure_prob, type_logits = model(xb)
            prob = failure_prob.squeeze(-1).clamp(1e-6, 1 - 1e-6)
            weights = 1.0 + (pos_weight - 1.0) * yb  # yb in {0,1}: pos rows get pos_weight
            loss = torch.nn.functional.binary_cross_entropy(prob, yb, weight=weights)
            mapped = tb >= 0
            if bool(mapped.any()):
                loss = loss + type_loss_weight * torch.nn.functional.cross_entropy(
                    type_logits[mapped], tb[mapped]
                )
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            epoch_loss += float(loss.item()) * batch_idx.shape[0]

        model.eval()
        with torch.no_grad():
            val_prob = model(x_val)[0].squeeze(-1).numpy()
        val_auc = binary_auc(y_val, val_prob)
        logger.info(
            "epoch %d: train loss %.4f, val AUC %.4f",
            epoch + 1,
            epoch_loss / x_train.shape[0],
            val_auc,
        )
        if np.isfinite(val_auc) and val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
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
    model.eval()

    # Fold train-year standardization into layer 0 and verify numerically:
    # the folded model on RAW vectors must reproduce the trained model on
    # standardized vectors (this is the train/serve parity guarantee).
    folded = copy.deepcopy(model)
    _fold_standardization(folded, ds.feature_mean, ds.feature_std)
    folded.eval()
    check = min(512, x_val.shape[0])
    with torch.no_grad():
        p_std, l_std = model(x_val[:check])
        p_raw, l_raw = folded(torch.from_numpy(ds.features[val_mask][:check]))
    prob_diff = float((p_std - p_raw).abs().max().item())
    logit_diff = float((l_std - l_raw).abs().max().item())
    if prob_diff > 1e-4 or logit_diff > 1e-3:
        raise RuntimeError(
            f"standardization folding verification failed (prob diff {prob_diff:.2e}, "
            f"logit diff {logit_diff:.2e}); refusing to ship a non-equivalent model"
        )

    with torch.no_grad():
        val_prob = model(x_val)[0].squeeze(-1).numpy()
    val_deployed = val_prob > DEPLOYED_PROB_THRESHOLD
    is_pos = y_val == 1.0
    tp = float(np.sum(val_deployed & is_pos))
    fp = float(np.sum(val_deployed & ~is_pos))
    fn = float(np.sum(~val_deployed & is_pos))
    t_val = ds.type_idx[val_mask]
    mapped_val = (t_val >= 0) & is_pos
    type_acc = float("nan")
    if mapped_val.any():
        with torch.no_grad():
            type_pred = model(x_val)[1].argmax(dim=1).numpy()
        type_acc = float(np.mean(type_pred[mapped_val] == t_val[mapped_val]))

    operating_point = _select_operating_point(
        model, ds, x_val=x_val, val_mask=val_mask, train_mask=train_mask
    )

    record = {
        "seed": ctx.seed,
        "epochs_run": epochs_run,
        "best_val_auc": best_val_auc,
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "train_years": list(SPLIT.train_years),
        "val_years": list(SPLIT.val_years),
        "pos_weight": pos_weight,
        "type_loss_weight": type_loss_weight,
        "val_recall_at_0p6": tp / max(tp + fn, 1.0),
        "val_far_at_0p6": fp / max(float(np.sum(~is_pos)), 1.0),
        "val_type_accuracy_mapped": type_acc,
        "standardization_fold_prob_diff": prob_diff,
        "operating_point": operating_point,
    }
    payload: dict[str, Any] = {
        "stability_model": folded.state_dict(),
        "feature_spec": FEATURE_SPEC_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": ds.feature_mean.tolist(),
        "feature_std": ds.feature_std.tolist(),
        "climatology_era": CLIMATOLOGY_ERA,
        "climatology_years": list(CLIMATOLOGY_YEARS),
        "type_labels": list(LANDSLIDE_TYPE_LABELS),
        "standardization": "folded into feature_encoder[0]; pass RAW landslide-coolr-v1 vectors",
        "operating_point": operating_point,
    }
    save_candidate(ctx.data_dir, HOOK_NAME, payload, record)
    return record


def _rainfall_trigger_confidence(fields: Mapping[str, float]) -> float:
    """Rainfall-trigger probability for one case (physics parity mirror).

    Mirrors ``RainfallTriggerModel.assess_rainfall_trigger`` exactly
    (Caine-style intensity-duration threshold with an antecedent-rainfall
    saturation boost); used only to score the API physics path on the
    validation years during operating-point selection. The evaluate stage
    still measures physics through the public detector API.
    """
    intensity = float(fields.get("intensity_mm_hr", 0.0))
    duration = float(fields.get("duration_hours", 0.0))
    antecedent = float(fields.get("antecedent_7day_mm", 0.0))
    id_threshold = intensity * duration**0.5
    critical_id = 10.0 * 6.0**0.5
    return float(min((id_threshold / critical_id) * (1.0 + antecedent / 100.0), 1.0))


def _select_operating_point(
    model: Any,
    ds: LandslideDataset,
    *,
    x_val: torch.Tensor,
    val_mask: np.ndarray,
    train_mask: np.ndarray,
) -> dict[str, Any]:
    """Choose the learned path's slope-failure alert threshold (tau).

    The stability head is trained with positive-weighted BCE on a
    25%-positive dataset, so its probability scale need not align with the
    detector's fixed neural-path alert bar (slope failure probability >
    0.6). Policy (documented for owner ratification, mirroring the
    solar-storm/tsunami machinery): on the VALIDATION years only, require
    the learned decision ``prob > tau`` to reach a recall of at least
    ``max(physics validation recall, 0.5)`` AND a false-alarm rate of at
    most ``0.8 * physics validation FAR`` (the 20% headroom guards the
    val->test distribution shift the ship gate's hard FAR constraint does
    not forgive); among feasible taus pick the one maximizing CSI (ties ->
    higher tau, i.e. fewer false alarms). "Physics" mirrors the merit
    gate's own baseline selection, computed on validation years: the
    stronger (by validation AUC, ties -> baseline) of (1) the API physics
    path, which abstains on rainfall-only inputs (slope failure
    probability identically 0, so its deployed recall/FAR are both 0),
    and (2) the train-years-fitted antecedent-rainfall percentile
    baseline at its train-fitted threshold.

    Args:
        model: The trained (standardized-input) SlopeStabilityModel.
        ds: Assembled dataset.
        x_val: Standardized validation feature tensor.
        val_mask: Validation-year sample mask.
        train_mask: Train-year sample mask (baseline fitting only).

    Returns:
        Operating-point record stored in the checkpoint payload and the
        provenance sidecar: the threshold (``detection_threshold``,
        consumed decision-only by ``LandslideDetector.load_neural_weights``),
        the policy text, the validation recall/FAR/CSI at tau, and the
        physics floors it was selected against.

    Raises:
        RuntimeError: When validation holds a single class or no tau
            satisfies even the FAR ceiling -- a doomed operating point must
            refuse loudly, not ship quietly.
    """
    is_pos = ds.labels[val_mask] == 1.0
    if not is_pos.any() or is_pos.all():
        raise RuntimeError(
            "validation years contain a single class; cannot select an operating point honestly"
        )
    model.eval()
    with torch.no_grad():
        prob = model(x_val)[0].squeeze(-1).numpy().astype(np.float64)

    val_idx = np.flatnonzero(val_mask)
    api_conf = np.array([_rainfall_trigger_confidence(ds.raw_fields[i]) for i in val_idx])
    api_auc = binary_auc(ds.labels[val_mask], api_conf)
    fit = _fit_rain_threshold(ds, train_mask)
    base_scores = ds.features[val_mask, fit["column"]].astype(np.float64)
    base_auc = binary_auc(ds.labels[val_mask], base_scores)
    base_alert = base_scores >= fit["threshold"]
    tp = float(np.sum(base_alert & is_pos))
    fp = float(np.sum(base_alert & ~is_pos))
    fn = float(np.sum(~base_alert & is_pos))
    if np.isfinite(base_auc) and (not np.isfinite(api_auc) or base_auc >= api_auc):
        physics_used = "fitted_rain_threshold_baseline"
        physics_recall = tp / max(tp + fn, 1.0)
        physics_far = fp / max(float(np.sum(~is_pos)), 1.0)
    else:
        # The API physics path abstains on rainfall-only inputs (slope
        # failure probability identically 0): its deployed rule never fires.
        physics_used = "detector_api_physics_path"
        physics_recall, physics_far = 0.0, 0.0

    recall_floor = max(physics_recall, 0.5)
    far_ceiling = 0.8 * physics_far

    def _metrics_at(tau: float) -> tuple[float, float, float]:
        alert = prob > tau
        tp = float(np.sum(alert & is_pos))
        fn = float(np.sum(~alert & is_pos))
        fp = float(np.sum(alert & ~is_pos))
        recall = tp / max(tp + fn, 1.0)
        far = fp / max(float(np.sum(~is_pos)), 1.0)
        csi = tp / max(tp + fn + fp, 1.0)
        return recall, far, csi

    best: dict[str, Any] | None = None
    fallback: dict[str, Any] | None = None
    for tau in np.unique(np.quantile(prob, np.linspace(0.0, 1.0, 513))):
        if not 0.0 < float(tau) < 1.0:
            continue
        recall, far, csi = _metrics_at(float(tau))
        entry = {
            "detection_threshold": float(tau),
            "val_recall": recall,
            "val_far": far,
            "val_csi": csi,
        }
        # Fallback if no tau satisfies both floors: the best-recall point
        # among FAR-feasible taus (a recall-maximizing fallback that blows
        # the FAR ceiling would be selecting a point the ship gate is
        # guaranteed to refuse).
        if far <= far_ceiling and (fallback is None or recall > fallback["val_recall"]):
            fallback = entry
        if (
            recall >= recall_floor
            and far <= far_ceiling
            and (
                best is None
                or csi > best["val_csi"]
                or (csi == best["val_csi"] and tau > best["detection_threshold"])
            )
        ):
            best = entry
    floor_met = best is not None
    chosen = best if best is not None else fallback
    if chosen is None:
        raise RuntimeError(
            "no operating point satisfies even the FAR ceiling on validation; "
            "the stability head is not usable for alert decisions -- refusing "
            "to record a doomed operating point"
        )
    return {
        **chosen,
        "policy": "single-rule (slope failure probability > tau, decision-only); tau "
        "maximizes val CSI subject to val recall >= max(physics val recall, 0.5) AND "
        "val FAR <= 0.8 * physics val FAR; physics mirrors the merit gate's baseline "
        "selection (stronger of the API physics path and the train-fitted "
        "antecedent-rainfall percentile threshold, by validation AUC)",
        "recall_floor": recall_floor,
        "recall_floor_met": floor_met,
        "far_ceiling": far_ceiling,
        "val_recall_physics": physics_recall,
        "val_far_physics": physics_far,
        "val_physics_baseline_used": physics_used,
        "val_auc_api_physics": api_auc,
        "val_auc_fitted_baseline": base_auc,
        "fixed_bar_without_operating_point": DEPLOYED_PROB_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Stage 4: evaluate (public detector API, identical held-out cases)
# ---------------------------------------------------------------------------

_BASELINE_CANDIDATES = {
    "pct_rain_day0": 9,
    "pct_rain_sum3d": 10,
    "pct_rain_sum7d": 11,
    "pct_rain_sum14d": 12,
    "pct_rain_sum30d": 13,
    "pct_rain_sum60d": 14,
    "pct_rain_max1d_30d": 15,
    "pct_rain_max3d_30d": 16,
}


def _threshold_metrics(
    labels: np.ndarray, scores: np.ndarray, probs: np.ndarray, deployed: np.ndarray
) -> dict[str, float]:
    """Shared metric dict: AUC, Brier, and recall/FAR/CSI at the deployment."""
    is_pos = labels == 1.0
    tp = float(np.sum(deployed & is_pos))
    fp = float(np.sum(deployed & ~is_pos))
    fn = float(np.sum(~deployed & is_pos))
    return {
        "auc": binary_auc(labels, scores),
        "brier": brier_score(labels, np.clip(probs, 0.0, 1.0)),
        "recall_deployed": tp / max(tp + fn, 1.0),
        "far_deployed": fp / max(float(np.sum(~is_pos)), 1.0),
        "csi_deployed": tp / max(tp + fn + fp, 1.0),
    }


def _fit_rain_threshold(ds: LandslideDataset, train_mask: np.ndarray) -> dict[str, Any]:
    """Fit the Caine-1980-style percentile baseline on TRAIN years only.

    Selects the single climatology-percentile feature with the best train
    AUC, then the alert threshold on it maximizing train CSI. Nothing from
    validation/test years touches the fit.

    Args:
        ds: Assembled dataset (features are raw canonical vectors).
        train_mask: Train-year sample mask.

    Returns:
        Fit record: feature name, canonical column index, train AUC, alert
        threshold, and the train CSI at that threshold.

    Raises:
        RuntimeError: When no candidate feature has a finite train AUC.
    """
    y_train = ds.labels[train_mask]
    best_name, best_col, best_auc = "", -1, float("-inf")
    for name, col in _BASELINE_CANDIDATES.items():
        auc = binary_auc(y_train, ds.features[train_mask, col])
        if np.isfinite(auc) and auc > best_auc:
            best_name, best_col, best_auc = name, col, auc
    if best_col < 0:
        raise RuntimeError("no baseline feature produced a finite train AUC")
    train_scores = ds.features[train_mask, best_col].astype(np.float64)
    is_pos = y_train == 1.0
    best_tau, best_csi = 0.5, -1.0
    for tau in np.unique(np.quantile(train_scores, np.linspace(0.0, 1.0, 513))):
        alert = train_scores >= tau
        tp = float(np.sum(alert & is_pos))
        fp = float(np.sum(alert & ~is_pos))
        fn = float(np.sum(~alert & is_pos))
        csi = tp / max(tp + fn + fp, 1.0)
        if csi > best_csi:
            best_tau, best_csi = float(tau), csi
    return {
        "feature": best_name,
        "column": best_col,
        "train_auc": best_auc,
        "threshold": best_tau,
        "train_csi_at_threshold": best_csi,
    }


def _fit_rain_threshold_baseline(
    ds: LandslideDataset,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> tuple[dict[str, float], dict[str, Any], np.ndarray]:
    """Caine-1980-style antecedent-rainfall percentile baseline (train-fitted).

    Selects, on TRAIN years only, the single climatology-percentile feature
    with the best train AUC, then the alert threshold on it maximizing
    train CSI. Nothing from validation/test years touches the fit.

    Args:
        ds: Assembled dataset (features are raw canonical vectors).
        train_mask: Train-year sample mask.
        test_mask: Test-year sample mask.

    Returns:
        Tuple of (test metrics, baseline definition record, test scores).
    """
    fit = _fit_rain_threshold(ds, train_mask)
    test_scores = ds.features[test_mask, fit["column"]].astype(np.float64)
    metrics = _threshold_metrics(
        ds.labels[test_mask], test_scores, test_scores, test_scores >= fit["threshold"]
    )
    definition = {
        "kind": "antecedent-rainfall percentile threshold (Caine-1980-style, "
        "fitted on TRAIN years only)",
        "feature": fit["feature"],
        "train_auc": fit["train_auc"],
        "threshold": fit["threshold"],
        "train_csi_at_threshold": fit["train_csi_at_threshold"],
    }
    return metrics, definition, test_scores


def evaluate(ctx: PipelineContext) -> EvaluationOutcome:
    """Compare learned vs physics through the public detector API.

    Both paths receive the IDENTICAL held-out cases, constructed from the
    same underlying real observations (mapping recorded in ``extras``):

    * physics path: ``{"rainfall_data": {intensity_mm_hr = day-of CHIRPS
      rain / 24 (daily-mean intensity), duration_hours = 24 (the daily
      resolution floor), antecedent_7day_mm = observed prior-7-day sum}}``
      -- every rainfall field ``RainfallTriggerModel`` defines, honestly
      derived from measured rain; no slope/saturation/displacement is
      fabricated, so ``_assess_slope_stability_physics`` abstains (failure
      probability 0) and the path's graded output is its rainfall-trigger
      confidence.
    * learned path: the same dict plus ``"slope_features"`` = the canonical
      64-dim vector built from the same CHIRPS series.

    Because the API physics path near-abstains on rainfall-only inputs, the
    merit-gate ``physics`` metrics are the STRONGER (by held-out AUC) of
    that path and the documented train-years-fitted antecedent-rainfall
    percentile baseline; both are recorded verbatim in ``extras``.

    Each path is scored on its OWN deployed decision rule: the learned
    path thresholds its slope-failure probability at the candidate's
    validation-selected operating point (``payload["operating_point"]``,
    the same rule ``LandslideDetector.load_neural_weights`` consumes
    decision-only; the fixed 0.6 bar when absent), the API physics path at
    the detector's fixed 0.6 neural-path alert bar, and the fitted
    baseline at its train-fitted threshold. Extras additionally carry a
    seeded 1000-resample bootstrap 95% CI on the AUC difference over the
    test cases, the model parameter count, the median single-case
    inference latency (100 runs), the trigger histogram, and per-year test
    counts.

    Args:
        ctx: Pipeline context.

    Returns:
        The evaluation outcome (primary metric: AUC, higher is better).
    """
    import time

    from omni_mercury_engine.detectors.geological.landslide import LandslideDetector

    ds = build_dataset(ctx)
    train_mask, _, test_mask = SPLIT.masks(ds.years)
    test_idx = np.flatnonzero(test_mask)
    if test_idx.size == 0:
        raise RuntimeError("no test rows found; cannot evaluate")

    cand_path, _ = candidate_paths(ctx.data_dir, HOOK_NAME)
    if not cand_path.exists():
        raise FileNotFoundError(f"no candidate checkpoint at {cand_path}; run --train first")

    physics_det = LandslideDetector(enable_ml_ensemble=False, enable_recursion=False)
    learned_det = LandslideDetector(enable_ml_ensemble=False, enable_recursion=False)
    learned_det.load_neural_weights(str(cand_path))
    payload = torch.load(cand_path, map_location="cpu", weights_only=True)
    operating_point = payload.get("operating_point")
    learned_bar = DEPLOYED_PROB_THRESHOLD
    if operating_point is not None:
        learned_bar = float(operating_point["detection_threshold"])

    labels = ds.labels[test_idx]
    scores: dict[str, list[float]] = {"physics": [], "learned": []}
    deployed: dict[str, list[bool]] = {"physics": [], "learned": []}
    rainfall_trigger_fired = 0
    for i in test_idx:
        rainfall_case = {"rainfall_data": dict(ds.raw_fields[i])}
        phys_out = physics_det.predict_landslide(dict(rainfall_case))
        learned_out = learned_det.predict_landslide(
            {**rainfall_case, "slope_features": ds.features[i]}
        )
        if not np.isfinite(learned_out.slope_failure_probability):
            raise RuntimeError(f"learned path returned non-finite probability for case {i}")
        scores["physics"].append(float(phys_out.confidence))
        scores["learned"].append(float(learned_out.slope_failure_probability))
        # Deployed decision, each path under its own rule: physics keeps the
        # detector's fixed neural-path alert bar (slope failure probability
        # > 0.6); the learned path applies the checkpoint's ratified
        # operating point. The physics path's probability is identically 0
        # on rainfall-only inputs (it abstains) -- that near-abstention is
        # the recorded result, not a scoring artifact; its trigger rate is
        # reported alongside.
        deployed["physics"].append(phys_out.slope_failure_probability > DEPLOYED_PROB_THRESHOLD)
        deployed["learned"].append(learned_out.slope_failure_probability > learned_bar)
        rainfall_trigger_fired += int(phys_out.rainfall_trigger)

    learned_metrics = _threshold_metrics(
        labels,
        np.asarray(scores["learned"]),
        np.asarray(scores["learned"]),
        np.asarray(deployed["learned"], dtype=bool),
    )
    physics_api_metrics = _threshold_metrics(
        labels,
        np.asarray(scores["physics"]),
        np.asarray(scores["physics"]),
        np.asarray(deployed["physics"], dtype=bool),
    )
    baseline_metrics, baseline_definition, baseline_test_scores = _fit_rain_threshold_baseline(
        ds, train_mask, test_mask
    )

    api_auc = physics_api_metrics["auc"]
    base_auc = baseline_metrics["auc"]
    if np.isfinite(base_auc) and (not np.isfinite(api_auc) or base_auc >= api_auc):
        physics_metrics = baseline_metrics
        physics_used = "fitted_rain_threshold_baseline"
        physics_gate_scores = baseline_test_scores
    else:
        physics_metrics = physics_api_metrics
        physics_used = "detector_api_physics_path"
        physics_gate_scores = np.asarray(scores["physics"])

    # Seeded bootstrap 95% CI on the AUC difference (learned - gate physics)
    # over the identical test cases: 1000 resamples with replacement;
    # resamples that lose a class (astronomically unlikely at this
    # prevalence) are skipped.
    rng = np.random.default_rng(ctx.seed)
    learned_scores_arr = np.asarray(scores["learned"])
    n_resamples = 1000
    diffs: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, test_idx.size, size=test_idx.size)
        auc_l = binary_auc(labels[idx], learned_scores_arr[idx])
        auc_p = binary_auc(labels[idx], physics_gate_scores[idx])
        if np.isfinite(auc_l) and np.isfinite(auc_p):
            diffs.append(float(auc_l - auc_p))
    diffs_arr = np.asarray(diffs)
    auc_diff_ci = {
        "n_resamples": n_resamples,
        "n_valid": int(diffs_arr.size),
        "seed": ctx.seed,
        "mean": float(diffs_arr.mean()),
        "ci95_low": float(np.percentile(diffs_arr, 2.5)),
        "ci95_high": float(np.percentile(diffs_arr, 97.5)),
        "ci_excludes_zero": bool(
            float(np.percentile(diffs_arr, 2.5)) > 0.0
            or float(np.percentile(diffs_arr, 97.5)) < 0.0
        ),
        "note": "AUC difference (learned minus gate physics), case-resampling "
        "bootstrap over the identical held-out (cell, day) cases",
    }

    # Median single-case inference latency through the public API (both
    # detectors are warm from the evaluation loop above).
    latency_case = {"rainfall_data": dict(ds.raw_fields[test_idx[0]])}
    latency_features = ds.features[test_idx[0]]

    def _median_latency_ms(det: LandslideDetector, with_features: bool) -> float:
        times = []
        for _ in range(100):
            case: dict[str, Any] = {"rainfall_data": dict(latency_case["rainfall_data"])}
            if with_features:
                case["slope_features"] = latency_features
            t0 = time.perf_counter()
            det.predict_landslide(case)
            times.append((time.perf_counter() - t0) * 1000.0)
        return float(np.median(times))

    test_years_arr = ds.years[test_idx]
    per_year_test_counts = {
        str(year): {
            "total": int(np.sum(test_years_arr == year)),
            "positives": int(labels[test_years_arr == year].sum()),
            "negatives": int(np.sum(test_years_arr == year) - labels[test_years_arr == year].sum()),
        }
        for year in SPLIT.test_years
    }

    outcome = EvaluationOutcome(
        hook=HOOK_NAME,
        primary_metric="auc",
        higher_is_better=True,
        learned=learned_metrics,
        physics=physics_metrics,
        n_test_samples=int(test_idx.size),
        test_years=SPLIT.test_years,
        extras={
            "comparison": "identical held-out (cell, day) cases through "
            "LandslideDetector.predict_landslide: physics fallback (no checkpoint, no "
            "slope_features) vs loaded candidate checkpoint (+slope_features), both fed "
            "the same honest rainfall_data derived from the same CHIRPS observations",
            "physics_input_mapping": "intensity_mm_hr = CHIRPS day-of rain / 24 h "
            "(daily-mean intensity); duration_hours = 24 (daily resolution floor); "
            "antecedent_7day_mm = observed sum of the 7 days before the event day. "
            "These are ALL the rainfall fields RainfallTriggerModel defines; the "
            "physics slope-stability term abstains (probability 0) because no "
            "slope/saturation/displacement observation exists in this corpus and "
            "none was fabricated",
            "physics_metrics_used_for_gate": physics_used,
            "physics_api_metrics": physics_api_metrics,
            "fitted_baseline_metrics": baseline_metrics,
            "fitted_baseline_definition": baseline_definition,
            "physics_rainfall_trigger_rate": rainfall_trigger_fired / float(test_idx.size),
            "operating_point": operating_point,
            "deployed_rule": f"learned: slope failure probability > {learned_bar:.6g} (the "
            "checkpoint's validation-selected operating point, consumed decision-only by "
            "LandslideDetector.load_neural_weights; the fixed "
            f"{DEPLOYED_PROB_THRESHOLD} bar when absent); physics API path: probability > "
            f"{DEPLOYED_PROB_THRESHOLD} (the detector's fixed neural-path alert bar); "
            "fitted baseline: its train-fitted threshold -- each path is scored on its "
            "own deployed decision rule",
            "test_positive_fraction": float(labels.mean()),
            "auc_diff_bootstrap_ci95": auc_diff_ci,
            "model_parameter_count": (
                int(sum(p.numel() for p in learned_det.stability_model.parameters()))
                if learned_det.stability_model is not None
                else 0
            ),
            "median_inference_latency_ms": {
                "learned": _median_latency_ms(learned_det, with_features=True),
                "physics": _median_latency_ms(physics_det, with_features=False),
                "n_runs": 100,
            },
            "trigger_histogram": ds.extras.get("rain_trigger_histogram"),
            "per_year_test_counts": per_year_test_counts,
            "dataset_report": ds.extras,
            "type_mapping": {k: LANDSLIDE_TYPE_LABELS[v] for k, v in GLC_CATEGORY_TO_TYPE.items()},
        },
        constraints=[
            {
                "metric": "recall_deployed",
                "higher_is_better": True,
                "description": "recall at each path's deployed alert rule must not "
                "regress below the physics baseline",
            },
            {
                "metric": "far_deployed",
                "higher_is_better": False,
                "description": "false-alarm rate at each path's deployed alert rule "
                "must not exceed the physics baseline",
            },
            {
                "metric": "brier",
                "higher_is_better": False,
                "description": "probability calibration (Brier) must not regress below "
                "the physics baseline",
            },
        ],
    )
    save_evaluation(ctx.data_dir, outcome)
    logger.info(
        "evaluation: learned AUC %.4f vs physics(%s) AUC %.4f on %d held-out cases (%s)",
        outcome.learned["auc"],
        physics_used,
        outcome.physics["auc"],
        outcome.n_test_samples,
        "LEARNED WINS" if outcome.learned_beats_physics else "PHYSICS WINS",
    )
    return outcome


# ---------------------------------------------------------------------------
# Stage 5: ship
# ---------------------------------------------------------------------------


def ship(ctx: PipelineContext) -> tuple[Any, Any]:
    """Promote the candidate through the merit gate (may refuse loudly).

    Args:
        ctx: Pipeline context.

    Returns:
        Tuple of (shipped checkpoint path, provenance sidecar path).
    """
    from omni_mercury_engine.ml.hazard_training.common import load_evaluation

    outcome = load_evaluation(ctx.data_dir, HOOK_NAME)
    manifest_path = ctx.data_dir / "landslide" / "manifest.json"
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
