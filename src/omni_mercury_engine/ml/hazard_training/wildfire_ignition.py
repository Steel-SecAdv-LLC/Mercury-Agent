# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Train the FireIgnitionDetector CNN on real NASA FIRMS VIIRS detections.

Data source (hook ``wildfire_ignition``, ``WildfireDetector.load_neural_weights``):

* **NASA FIRMS keyless country archive**
  (``https://firms.modaps.eosdis.nasa.gov/data/country/viirs-snpp/{YYYY}/
  viirs-snpp_{YYYY}_United_States.csv``) -- the science-quality VIIRS
  Suomi-NPP active-fire detection record for the United States, 2012-2024.
  Each row is one satellite-confirmed thermal-anomaly detection with
  latitude/longitude, I4-band brightness temperature (Kelvin), fire radiative
  power (MW), acquisition date, detection confidence (l/n/h) and type
  (0 = presumed vegetation fire). No MAP_KEY is used or required.

Task -- next-day fire activity forecasting on a grid. This is the honest task
FIRMS supports: the archive is a *census* of satellite-confirmed fires, so the
absence of a detection means "no confirmed fire", not a fabricated background
class. Samples are 32x32-cell (0.04-degree, ~4 km) raster patches over
California whose three channels are built ONLY from days ``[t-6 .. t]``:

* ch0 ``bt_max_t_kelvin`` -- per-cell max brightness temperature on day t
  (0 where no detection);
* ch1 ``log1p_frp_3d`` -- log1p of the per-cell FRP sum over days t-2..t;
* ch2 ``log1p_count_7d`` -- log1p of the per-cell detection count over days
  t-6..t.

The label is "any type-0 detection in the CENTER 2x2 cells on day t+1".
IMPORTANT HONESTY NOTE: the channels are detection-derived brightness/FRP
fields rasterized from the FIRMS point census -- they are NOT full thermal
imagery. The feature spec is named ``wildfire-firms-v1`` accordingly, and the
checkpoint carries the channel names so no caller can mistake the contract.

The physics baseline is the detector's deterministic VIIRS-style
brightness-temperature path applied to the same patches: because ch0 is
*today's* max brightness temperature, thresholding it is effectively a
PERSISTENCE forecaster of tomorrow ("fire today near here => fire tomorrow").
Persistence is a genuinely strong baseline for next-day fire activity; the
learned model's edge must come from decay/growth morphology and multi-day
dynamics (channels 1-2).

Temporal split (never random -- fire seasons autocorrelate within a year):
train 2012-2019, validation 2020-2021, test 2022-2024. Note that the record
2020 California season (August Complex etc.) lands in the VALIDATION years,
so early stopping sees an extreme season but the held-out test never does.
Days without US-wide VIIRS coverage (instrument outages, e.g. the Suomi-NPP
safe-mode gap of 2022-07/08) are excluded from sampling windows: absence of
data must never be read as absence of fire.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch

if TYPE_CHECKING:
    from pathlib import Path

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

logger = logging.getLogger(__name__)

HOOK_NAME = "wildfire_ignition"
CHECKPOINT_NAME = "wildfire_firms"
FEATURE_SPEC_VERSION = "wildfire-firms-v1"

FIRMS_URL_TEMPLATE = (
    "https://firms.modaps.eosdis.nasa.gov/data/country/viirs-snpp/"
    "{year}/viirs-snpp_{year}_United_States.csv"
)

SPLIT = TemporalSplit(
    train_years=tuple(range(2012, 2020)),
    val_years=(2020, 2021),
    test_years=(2022, 2023, 2024),
)

# California bounding box (degrees); detections outside are dropped.
LON_MIN, LON_MAX = -124.5, -114.0
LAT_MIN, LAT_MAX = 32.4, 42.1

GRID_DEG = 0.04  # ~4 km cells
N_ROWS = int(np.ceil((LAT_MAX - LAT_MIN) / GRID_DEG))  # 243 (latitude cells)
N_COLS = int(np.ceil((LON_MAX - LON_MIN) / GRID_DEG))  # 263 (longitude cells)

PATCH_CELLS = 32
# Patch anchored at cell (cy, cx): rows [cy-15, cy+16], cols [cx-15, cx+16];
# the CENTER 2x2 label region is rows {cy, cy+1} x cols {cx, cx+1}.
_PATCH_LO = PATCH_CELLS // 2 - 1  # 15
_PATCH_HI = PATCH_CELLS // 2  # 16

CHANNEL_NAMES = ("bt_max_t_kelvin", "log1p_frp_3d", "log1p_count_7d")
LABEL_SPEC = "center 2x2 any type-0 detection day t+1"

FRP_WINDOW_DAYS = 3  # ch1 uses days t-2..t
COUNT_WINDOW_DAYS = 7  # ch2 uses days t-6..t
QUIET_WINDOW_DAYS = 14  # easy negatives require a patch quiet over t-13..t

# Seeded sampling quotas (per year); negatives ~= 2.5x positives.
POS_PER_YEAR = 350
HARD_NEG_RATIO = 1.5
EASY_NEG_RATIO = 1.0

SAMPLE_KINDS = ("positive", "hard_negative", "easy_negative")

# The detector's deployed decision threshold on the CNN fire probability
# (WildfireDetector._detect_ignition: fire_detected = prob > 0.6).
DEPLOYED_PROB_THRESHOLD = 0.6

_REQUIRED_COLUMNS = (
    "latitude",
    "longitude",
    "bright_ti4",
    "acq_date",
    "confidence",
    "frp",
    "type",
)


@dataclass
class FirmsYearData:
    """One year of parsed, California-filtered VIIRS detections.

    Attributes:
        lat: Detection latitudes (degrees), filtered rows only.
        lon: Detection longitudes (degrees).
        bt: I4-band brightness temperature (Kelvin).
        frp: Fire radiative power (MW).
        day: Proleptic-Gregorian day ordinal of the acquisition date.
        covered_days: Unique day ordinals with ANY US-wide row (pre-filter) --
            the instrument-coverage record. A day absent here is a data gap,
            never a fire-free day.
        rows_total: Total rows in the source file.
        rows_filtered: Rows surviving the CA bbox + type==0 + confidence
            n/h filter.
    """

    lat: np.ndarray
    lon: np.ndarray
    bt: np.ndarray
    frp: np.ndarray
    day: np.ndarray
    covered_days: np.ndarray
    rows_total: int
    rows_filtered: int


def parse_firms_csv(path: Path | str) -> FirmsYearData:
    """Parse one FIRMS country CSV into California-filtered detection arrays.

    Filter: California bbox (lon [-124.5, -114.0), lat [32.4, 42.1)),
    ``type == 0`` (presumed vegetation fire) and ``confidence in ('n', 'h')``.
    Low-confidence and non-vegetation detections are excluded from both
    features and labels.

    Args:
        path: A cached ``viirs-snpp_{YYYY}_United_States.csv`` file.

    Returns:
        Parsed arrays plus the US-wide coverage-day record.

    Raises:
        RuntimeError: On missing columns, unexpected confidence/type
            vocabulary, or non-finite values in required fields -- a format
            drift must fail loud, not silently corrupt training data.
    """
    df = pd.read_csv(path, usecols=list(_REQUIRED_COLUMNS))
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"{path}: missing FIRMS columns {missing}; format changed")
    conf_vocab = set(df["confidence"].unique())
    if not conf_vocab <= {"l", "n", "h"}:
        raise RuntimeError(
            f"{path}: unexpected confidence values {sorted(conf_vocab - {'l', 'n', 'h'})}; "
            "the l/n/h vocabulary changed -- refusing to guess"
        )
    type_vocab = {int(t) for t in df["type"].unique()}
    if not type_vocab <= {0, 1, 2, 3}:
        raise RuntimeError(f"{path}: unexpected type values {sorted(type_vocab)}")

    covered = np.unique(
        np.array(
            [_dt.date.fromisoformat(d).toordinal() for d in df["acq_date"].unique()],
            dtype=np.int64,
        )
    )
    rows_total = len(df)

    keep = (
        (df["longitude"].to_numpy() >= LON_MIN)
        & (df["longitude"].to_numpy() < LON_MAX)
        & (df["latitude"].to_numpy() >= LAT_MIN)
        & (df["latitude"].to_numpy() < LAT_MAX)
        & (df["type"].to_numpy() == 0)
        & df["confidence"].isin(["n", "h"]).to_numpy()
    )
    sub = df.loc[keep]
    lat = sub["latitude"].to_numpy(dtype=np.float64)
    lon = sub["longitude"].to_numpy(dtype=np.float64)
    bt = sub["bright_ti4"].to_numpy(dtype=np.float64)
    frp = sub["frp"].to_numpy(dtype=np.float64)
    if not (
        np.all(np.isfinite(lat))
        and np.all(np.isfinite(lon))
        and np.all(np.isfinite(bt))
        and np.all(np.isfinite(frp))
    ):
        raise RuntimeError(f"{path}: non-finite values in required numeric columns")
    date_map = {d: _dt.date.fromisoformat(d).toordinal() for d in sub["acq_date"].unique()}
    day = sub["acq_date"].map(date_map).to_numpy(dtype=np.int64)
    return FirmsYearData(
        lat=lat,
        lon=lon,
        bt=bt,
        frp=frp,
        day=day,
        covered_days=covered,
        rows_total=rows_total,
        rows_filtered=len(sub),
    )


@dataclass
class DayCells:
    """Per-cell aggregates for one day: unique active cells only."""

    rows: np.ndarray  # int32 grid row per active cell
    cols: np.ndarray  # int32 grid col per active cell
    bt_max: np.ndarray  # float32 max brightness temperature (K)
    frp_sum: np.ndarray  # float32 FRP sum (MW)
    count: np.ndarray  # int32 detection count


@dataclass
class DailyGrids:
    """Daily per-cell rasters of the CA detection census plus coverage.

    ``days`` maps day OFFSET (relative to ``day0``) to that day's active
    cells; days with no CA detection are simply absent. ``covered`` marks
    days with US-wide instrument coverage -- sampling windows must lie
    entirely inside covered days so a data gap is never read as quiet.
    """

    day0: int
    n_days: int
    covered: np.ndarray  # bool (n_days,)
    days: dict[int, DayCells]
    n_detections: int


def rasterize_daily(years: list[FirmsYearData]) -> DailyGrids:
    """Aggregate detections into per-day, per-cell (bt max, frp sum, count).

    Args:
        years: Parsed year files (any order).

    Returns:
        Daily grids on the module's fixed CA 0.04-degree grid.

    Raises:
        RuntimeError: If no detections survive filtering, or a detection maps
            outside the grid (bbox/grid mismatch bug -- fail loud).
    """
    if not years:
        raise RuntimeError("no parsed FIRMS years supplied")
    lat = np.concatenate([y.lat for y in years])
    lon = np.concatenate([y.lon for y in years])
    bt = np.concatenate([y.bt for y in years])
    frp = np.concatenate([y.frp for y in years])
    day = np.concatenate([y.day for y in years])
    covered_days = np.unique(np.concatenate([y.covered_days for y in years]))
    if lat.size == 0:
        raise RuntimeError("zero California detections after filtering; cannot rasterize")

    rows = np.floor((lat - LAT_MIN) / GRID_DEG).astype(np.int64)
    cols = np.floor((lon - LON_MIN) / GRID_DEG).astype(np.int64)
    if rows.min() < 0 or rows.max() >= N_ROWS or cols.min() < 0 or cols.max() >= N_COLS:
        raise RuntimeError("detection mapped outside the CA grid; bbox/grid mismatch")

    day0 = int(covered_days.min())
    n_days = int(covered_days.max()) - day0 + 1
    covered = np.zeros(n_days, dtype=bool)
    covered[covered_days - day0] = True

    day_off = day - day0
    key = day_off * (N_ROWS * N_COLS) + rows * N_COLS + cols
    order = np.argsort(key, kind="mergesort")
    key_sorted = key[order]
    uniq_keys, starts = np.unique(key_sorted, return_index=True)
    ends = np.append(starts[1:], key_sorted.size)

    bt_sorted = bt[order]
    frp_sorted = frp[order]
    bt_max = np.maximum.reduceat(bt_sorted, starts).astype(np.float32)
    frp_sum = np.add.reduceat(frp_sorted, starts).astype(np.float32)
    count = (ends - starts).astype(np.int32)

    uniq_day = (uniq_keys // (N_ROWS * N_COLS)).astype(np.int64)
    rem = uniq_keys % (N_ROWS * N_COLS)
    uniq_rows = (rem // N_COLS).astype(np.int32)
    uniq_cols = (rem % N_COLS).astype(np.int32)

    days: dict[int, DayCells] = {}
    d_starts = np.flatnonzero(np.diff(uniq_day, prepend=uniq_day[0] - 1))
    d_ends = np.append(d_starts[1:], uniq_day.size)
    for s, e in zip(d_starts, d_ends, strict=True):
        d = int(uniq_day[s])
        days[d] = DayCells(
            rows=uniq_rows[s:e],
            cols=uniq_cols[s:e],
            bt_max=bt_max[s:e],
            frp_sum=frp_sum[s:e],
            count=count[s:e],
        )
    return DailyGrids(
        day0=day0,
        n_days=n_days,
        covered=covered,
        days=days,
        n_detections=int(lat.size),
    )


def build_patch(grids: DailyGrids, t: int, cy: int, cx: int) -> np.ndarray:
    """Build one 3-channel 32x32 patch from days <= t ONLY (no lookahead).

    Args:
        grids: Daily rasters.
        t: Day offset of "today".
        cy: Anchor grid row (patch rows [cy-15, cy+16]).
        cx: Anchor grid col.

    Returns:
        ``(3, 32, 32)`` float32 array: today's max brightness temperature
        (Kelvin, 0 where no detection), log1p 3-day FRP sum, log1p 7-day
        detection count.
    """
    r0, c0 = cy - _PATCH_LO, cx - _PATCH_LO
    patch = np.zeros((3, PATCH_CELLS, PATCH_CELLS), dtype=np.float32)
    frp = np.zeros((PATCH_CELLS, PATCH_CELLS), dtype=np.float32)
    cnt = np.zeros((PATCH_CELLS, PATCH_CELLS), dtype=np.float32)

    def _in_patch(cells: DayCells) -> np.ndarray:
        return (
            (cells.rows >= r0)
            & (cells.rows < r0 + PATCH_CELLS)
            & (cells.cols >= c0)
            & (cells.cols < c0 + PATCH_CELLS)
        )

    today = grids.days.get(t)
    if today is not None:
        m = _in_patch(today)
        patch[0, today.rows[m] - r0, today.cols[m] - c0] = today.bt_max[m]
    for d in range(t - COUNT_WINDOW_DAYS + 1, t + 1):
        cells = grids.days.get(d)
        if cells is None:
            continue
        m = _in_patch(cells)
        rr, cc = cells.rows[m] - r0, cells.cols[m] - c0
        np.add.at(cnt, (rr, cc), cells.count[m].astype(np.float32))
        if d >= t - FRP_WINDOW_DAYS + 1:
            np.add.at(frp, (rr, cc), cells.frp_sum[m])
    patch[1] = np.log1p(frp)
    patch[2] = np.log1p(cnt)
    return patch


def center_label(grids: DailyGrids, t: int, cy: int, cx: int) -> float:
    """Label for anchor (cy, cx) at day t: any detection in the center 2x2 on t+1."""
    nxt = grids.days.get(t + 1)
    if nxt is None:
        return 0.0
    hit = (nxt.rows >= cy) & (nxt.rows <= cy + 1) & (nxt.cols >= cx) & (nxt.cols <= cx + 1)
    return 1.0 if bool(hit.any()) else 0.0


@dataclass
class WildfireDataset:
    """Sampled patches with labels, per-sample year/month/kind, and counts."""

    patches: np.ndarray  # (N, 3, 32, 32) float32
    labels: np.ndarray  # (N,) float32
    years: np.ndarray  # (N,) int64 (year of day t)
    months: np.ndarray  # (N,) int64 (month of day t)
    kinds: np.ndarray  # (N,) int8 index into SAMPLE_KINDS
    per_year_counts: dict[int, dict[str, int]]


def _encode(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Encode (row, col) cells as single int64 codes."""
    return rows.astype(np.int64) * N_COLS + cols.astype(np.int64)


def _window_covered(cum: np.ndarray, lo: int, hi: int) -> bool:
    """True when day offsets [lo, hi] are all instrument-covered."""
    return bool(cum[hi + 1] - cum[lo] == hi - lo + 1)


def _stratified_by_month(months: np.ndarray, quota: int, rng: np.random.Generator) -> np.ndarray:
    """Pick ``quota`` pool indices stratified proportionally by month.

    Largest-remainder allocation over the months actually present, so fire
    season dominates proportionally but shoulder months keep representation.
    """
    n = months.size
    if quota >= n:
        return np.arange(n)
    uniq, counts = np.unique(months, return_counts=True)
    exact = counts * (quota / n)
    alloc = np.floor(exact).astype(np.int64)
    remainder = exact - alloc
    short = quota - int(alloc.sum())
    if short > 0:
        alloc[np.argsort(-remainder, kind="mergesort")[:short]] += 1
    picks: list[np.ndarray] = []
    for m, k in zip(uniq, alloc, strict=True):
        if k <= 0:
            continue
        idx = np.flatnonzero(months == m)
        picks.append(rng.choice(idx, size=min(int(k), idx.size), replace=False))
    return np.sort(np.concatenate(picks)) if picks else np.zeros(0, dtype=np.int64)


def _year_day_bounds(year: int, grids: DailyGrids) -> tuple[int, int]:
    """Inclusive (first, last) day offsets of ``year`` clipped to the grids."""
    first = _dt.date(year, 1, 1).toordinal() - grids.day0
    last = _dt.date(year, 12, 31).toordinal() - grids.day0
    return max(first, 0), min(last, grids.n_days - 1)


def assemble_samples(
    grids: DailyGrids,
    rng: np.random.Generator,
    *,
    years: tuple[int, ...] | None = None,
    pos_per_year: int = POS_PER_YEAR,
) -> WildfireDataset:
    """Seeded sample assembly: positives, hard negatives, easy negatives.

    Per year (windows never cross year boundaries, and every window day must
    be instrument-covered):

    * positives -- anchors whose center 2x2 is active on day t+1, stratified
      by month;
    * hard negatives -- anchors on/next to cells active during [t-6, t]
      whose center 2x2 is quiet on t+1 (fire decay / fire-edge cases),
      stratified by month;
    * easy negatives -- seeded random anchors whose whole 32x32 patch had no
      detection over [t-13, t] (labels still computed honestly from t+1).

    Negatives are capped at ~2.5x positives (1.5x hard + 1.0x easy).

    Args:
        grids: Daily rasters (features AND labels come from these).
        rng: Seeded generator; the only randomness source.
        years: Years to sample (default: every year the grids cover).
        pos_per_year: Positive quota per year.

    Returns:
        The assembled dataset with per-year sampling counts.

    Raises:
        RuntimeError: If no positive or no negative samples exist at all.
    """
    cum = np.concatenate([[0], np.cumsum(grids.covered.astype(np.int64))])
    if years is None:
        first_year = _dt.date.fromordinal(grids.day0).year
        last_year = _dt.date.fromordinal(grids.day0 + grids.n_days - 1).year
        years = tuple(range(first_year, last_year + 1))

    offsets_2x2 = ((-1, -1), (-1, 0), (0, -1), (0, 0))
    neighborhood = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1))

    all_t: list[np.ndarray] = []
    all_cy: list[np.ndarray] = []
    all_cx: list[np.ndarray] = []
    all_kind: list[np.ndarray] = []
    per_year_counts: dict[int, dict[str, int]] = {}

    for year in years:
        lo, hi = _year_day_bounds(year, grids)
        pos_t: list[int] = []
        pos_code: list[int] = []
        hard_t: list[int] = []
        hard_code: list[int] = []
        valid_days: list[int] = []
        for t in range(lo + COUNT_WINDOW_DAYS - 1, hi):
            if not _window_covered(cum, t - COUNT_WINDOW_DAYS + 1, t + 1):
                continue
            valid_days.append(t)
            nxt = grids.days.get(t + 1)
            pos_codes = np.zeros(0, dtype=np.int64)
            if nxt is not None and nxt.rows.size:
                cand = [_encode(nxt.rows + dr, nxt.cols + dc) for dr, dc in offsets_2x2]
                anchors = np.unique(np.concatenate(cand))
                ar, ac = anchors // N_COLS, anchors % N_COLS
                ok = (
                    (ar >= _PATCH_LO)
                    & (ar < N_ROWS - _PATCH_HI)
                    & (ac >= _PATCH_LO)
                    & (ac < N_COLS - _PATCH_HI)
                )
                pos_codes = anchors[ok]
                pos_t.extend([t] * pos_codes.size)
                pos_code.extend(pos_codes.tolist())
            window_cells = [
                _encode(grids.days[d].rows, grids.days[d].cols)
                for d in range(t - COUNT_WINDOW_DAYS + 1, t + 1)
                if d in grids.days
            ]
            if window_cells:
                active = np.unique(np.concatenate(window_cells))
                wr, wc = active // N_COLS, active % N_COLS
                edge = np.unique(
                    np.concatenate([_encode(wr + dr, wc + dc) for dr, dc in neighborhood])
                )
                er, ec = edge // N_COLS, edge % N_COLS
                ok = (
                    (er >= _PATCH_LO)
                    & (er < N_ROWS - _PATCH_HI)
                    & (ec >= _PATCH_LO)
                    & (ec < N_COLS - _PATCH_HI)
                )
                edge = edge[ok]
                edge = edge[~np.isin(edge, pos_codes)]
                hard_t.extend([t] * edge.size)
                hard_code.extend(edge.tolist())

        pos_t_arr = np.asarray(pos_t, dtype=np.int64)
        pos_code_arr = np.asarray(pos_code, dtype=np.int64)
        hard_t_arr = np.asarray(hard_t, dtype=np.int64)
        hard_code_arr = np.asarray(hard_code, dtype=np.int64)

        def _months_of(t_arr: np.ndarray) -> np.ndarray:
            return np.asarray(
                [_dt.date.fromordinal(grids.day0 + int(t)).month for t in t_arr],
                dtype=np.int64,
            )

        pos_pick = _stratified_by_month(_months_of(pos_t_arr), pos_per_year, rng)
        n_pos = int(pos_pick.size)
        hard_quota = round(HARD_NEG_RATIO * n_pos)
        hard_pick = _stratified_by_month(_months_of(hard_t_arr), hard_quota, rng)

        easy_quota = round(EASY_NEG_RATIO * n_pos)
        easy_t: list[int] = []
        easy_code: list[int] = []
        easy_valid = [
            t
            for t in valid_days
            if t - QUIET_WINDOW_DAYS + 1 >= lo
            and _window_covered(cum, t - QUIET_WINDOW_DAYS + 1, t + 1)
        ]
        if easy_valid and easy_quota > 0:
            attempts = 0
            max_attempts = easy_quota * 60
            while len(easy_t) < easy_quota and attempts < max_attempts:
                attempts += 1
                t = int(rng.choice(np.asarray(easy_valid, dtype=np.int64)))
                cy = int(rng.integers(_PATCH_LO, N_ROWS - _PATCH_HI))
                cx = int(rng.integers(_PATCH_LO, N_COLS - _PATCH_HI))
                r0, c0 = cy - _PATCH_LO, cx - _PATCH_LO
                quiet = True
                for d in range(t - QUIET_WINDOW_DAYS + 1, t + 1):
                    cells = grids.days.get(d)
                    if cells is None:
                        continue
                    m = (
                        (cells.rows >= r0)
                        & (cells.rows < r0 + PATCH_CELLS)
                        & (cells.cols >= c0)
                        & (cells.cols < c0 + PATCH_CELLS)
                    )
                    if bool(m.any()):
                        quiet = False
                        break
                if quiet:
                    easy_t.append(t)
                    easy_code.append(cy * N_COLS + cx)
            if len(easy_t) < easy_quota:
                logger.warning(
                    "year %d: only %d/%d easy negatives found", year, len(easy_t), easy_quota
                )

        year_t = np.concatenate(
            [pos_t_arr[pos_pick], hard_t_arr[hard_pick], np.asarray(easy_t, dtype=np.int64)]
        )
        year_codes = np.concatenate(
            [
                pos_code_arr[pos_pick],
                hard_code_arr[hard_pick],
                np.asarray(easy_code, dtype=np.int64),
            ]
        )
        year_kind = np.concatenate(
            [
                np.zeros(n_pos, dtype=np.int8),
                np.ones(hard_pick.size, dtype=np.int8),
                np.full(len(easy_t), 2, dtype=np.int8),
            ]
        )
        all_t.append(year_t)
        all_cy.append((year_codes // N_COLS).astype(np.int64))
        all_cx.append((year_codes % N_COLS).astype(np.int64))
        all_kind.append(year_kind)
        per_year_counts[year] = {
            "positives": n_pos,
            "hard_negatives": int(hard_pick.size),
            "easy_negatives": len(easy_t),
            "positive_pool": int(pos_t_arr.size),
            "hard_negative_pool": int(hard_t_arr.size),
        }
        logger.info("year %d: %s", year, per_year_counts[year])

    t_all = np.concatenate(all_t)
    cy_all = np.concatenate(all_cy)
    cx_all = np.concatenate(all_cx)
    kind_all = np.concatenate(all_kind)
    if t_all.size == 0:
        raise RuntimeError("sample assembly produced zero samples")

    patches = np.zeros((t_all.size, 3, PATCH_CELLS, PATCH_CELLS), dtype=np.float32)
    labels = np.zeros(t_all.size, dtype=np.float32)
    years_arr = np.zeros(t_all.size, dtype=np.int64)
    months_arr = np.zeros(t_all.size, dtype=np.int64)
    for i in range(t_all.size):
        t, cy, cx = int(t_all[i]), int(cy_all[i]), int(cx_all[i])
        patches[i] = build_patch(grids, t, cy, cx)
        labels[i] = center_label(grids, t, cy, cx)
        date = _dt.date.fromordinal(grids.day0 + t)
        years_arr[i] = date.year
        months_arr[i] = date.month

    n_pos_total = int((labels == 1.0).sum())
    n_neg_total = int((labels == 0.0).sum())
    if n_pos_total == 0 or n_neg_total == 0:
        raise RuntimeError(f"degenerate dataset: {n_pos_total} positives / {n_neg_total} negatives")
    return WildfireDataset(
        patches=patches,
        labels=labels,
        years=years_arr,
        months=months_arr,
        kinds=kind_all,
        per_year_counts=per_year_counts,
    )


def _firms_dir(ctx: PipelineContext) -> Path:
    """On-disk cache directory for FIRMS year files."""
    return ctx.data_dir / "firms"


def _year_path(ctx: PipelineContext, year: int) -> Path:
    """Cache path of one FIRMS year file."""
    return _firms_dir(ctx) / f"viirs-snpp_{year}_United_States.csv"


def fetch(ctx: PipelineContext) -> dict[str, Any]:
    """Download and pin all FIRMS year files needed by the split.

    Reports the per-year California-filtered detection counts (type 0,
    confidence n/h) in the manifest so the training population is auditable.

    Returns:
        Manifest with per-file URLs, SHA-256 digests, and per-year row counts.
    """
    sources: list[dict[str, Any]] = []
    per_year_rows: dict[str, dict[str, int]] = {}
    for year in SPLIT.all_years:
        url = FIRMS_URL_TEMPLATE.format(year=year)
        path = cached_fetch(url, _year_path(ctx, year), timeout=600.0)
        parsed = parse_firms_csv(path)
        sources.append(
            {
                "url": url,
                "sha256": sha256_file(path),
                "description": f"NASA FIRMS VIIRS-SNPP US active-fire detections, {year}",
            }
        )
        per_year_rows[str(year)] = {
            "rows_total_us": parsed.rows_total,
            "rows_ca_type0_conf_nh": parsed.rows_filtered,
            "covered_days": int(parsed.covered_days.size),
        }
        logger.info("%d: %s", year, per_year_rows[str(year)])
    manifest = {"hook": HOOK_NAME, "sources": sources, "per_year_rows": per_year_rows}
    manifest_path = _firms_dir(ctx) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    logger.info("fetch complete: %d FIRMS years cached under %s", len(sources), _firms_dir(ctx))
    return manifest


def _load_grids(ctx: PipelineContext) -> DailyGrids:
    """Parse every cached FIRMS year of the split, failing loud on gaps."""
    parsed: list[FirmsYearData] = []
    for year in SPLIT.all_years:
        path = _year_path(ctx, year)
        if not path.exists():
            raise FileNotFoundError(f"missing FIRMS cache file {path}; run the --fetch stage first")
        parsed.append(parse_firms_csv(path))
    return rasterize_daily(parsed)


def build_dataset(ctx: PipelineContext) -> WildfireDataset:
    """Assemble the sampled patch dataset from cached FIRMS files.

    Deterministic given ``ctx.seed``. There is no train-statistics
    standardization stage: the public detector API feeds raw patches to the
    CNN, so the checkpoint must consume raw channel values (Kelvin / log1p
    fields) and the network's BatchNorm layers learn scale from the TRAIN
    years via SGD -- validation/test years never contribute statistics.
    """
    grids = _load_grids(ctx)
    rng = np.random.default_rng(ctx.seed)
    ds = assemble_samples(grids, rng, years=SPLIT.all_years)
    if ctx.limit_samples is not None and ctx.limit_samples < ds.labels.size:
        keep = np.sort(
            np.random.default_rng(ctx.seed).permutation(ds.labels.size)[: ctx.limit_samples]
        )
        ds = WildfireDataset(
            patches=ds.patches[keep],
            labels=ds.labels[keep],
            years=ds.years[keep],
            months=ds.months[keep],
            kinds=ds.kinds[keep],
            per_year_counts=ds.per_year_counts,
        )
    return ds


def _model_probs(model: Any, x: torch.Tensor, batch: int = 256) -> np.ndarray:
    """Fire probabilities for a tensor of patches, batched, no grad."""
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch):
            p = model(x[start : start + batch]).squeeze(-1)
            out.append(p.numpy().astype(np.float64))
    return np.concatenate(out)


def train(ctx: PipelineContext) -> dict[str, Any]:
    """Train FireIgnitionDetector with early stopping on validation AUC.

    Pos-weighted BCE on the ``fire_classifier`` sigmoid output, Adam 1e-3,
    batch 64, patience 6 on validation AUC (higher is better), seeded.

    Returns:
        Training record (epochs run, best validation AUC, sample counts).
    """
    from omni_mercury_engine.detectors.geological.wildfire import FireIgnitionDetector

    rng = seed_everything(ctx.seed)
    ds = build_dataset(ctx)
    train_mask, val_mask, _ = SPLIT.masks(ds.years)
    if not train_mask.any() or not val_mask.any():
        raise RuntimeError("empty train or validation split; cannot train")
    x_train = torch.from_numpy(ds.patches[train_mask])
    y_train = torch.from_numpy(ds.labels[train_mask])
    x_val = torch.from_numpy(ds.patches[val_mask])
    y_val = ds.labels[val_mask]

    n_pos = float(y_train.sum().item())
    n_neg = float(y_train.shape[0] - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise RuntimeError("training split has a single class; cannot train")
    pos_weight = n_neg / n_pos

    model = FireIgnitionDetector(input_channels=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    logger.info(
        "training on %d patches (%.1f%% positive, pos_weight %.2f), validating on %d",
        x_train.shape[0],
        100.0 * n_pos / x_train.shape[0],
        pos_weight,
        x_val.shape[0],
    )

    batch_size = 64
    best_val_auc = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience, bad_epochs = 6, 0
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
            yb = y_train[batch_idx]
            prob = model(x_train[batch_idx]).squeeze(-1).clamp(1e-6, 1 - 1e-6)
            weights = 1.0 + (pos_weight - 1.0) * yb
            loss = torch.nn.functional.binary_cross_entropy(prob, yb, weight=weights)
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call, unused-ignore]
            optimizer.step()
            epoch_loss += float(loss.item()) * batch_idx.shape[0]

        val_probs = _model_probs(model, x_val)
        val_auc = binary_auc(y_val, val_probs)
        logger.info(
            "epoch %d: train loss %.4f, val AUC %.4f",
            epoch + 1,
            epoch_loss / x_train.shape[0],
            val_auc,
        )
        if np.isfinite(val_auc) and val_auc > best_val_auc + 1e-5:
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

    record = {
        "seed": ctx.seed,
        "epochs_run": epochs_run,
        "best_val_auc": best_val_auc,
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "train_years": list(SPLIT.train_years),
        "val_years": list(SPLIT.val_years),
        "train_positive_fraction": n_pos / x_train.shape[0],
        "pos_weight": pos_weight,
    }
    payload: dict[str, Any] = {
        "ignition_detector": model.state_dict(),
        "feature_spec": FEATURE_SPEC_VERSION,
        "grid_deg": GRID_DEG,
        "patch_cells": PATCH_CELLS,
        "channels": list(CHANNEL_NAMES),
        "label": LABEL_SPEC,
    }
    save_candidate(ctx.data_dir, HOOK_NAME, payload, record)
    return record


def evaluate(ctx: PipelineContext) -> EvaluationOutcome:
    """Compare learned vs physics through the public detector API.

    Both detectors receive the IDENTICAL held-out test patches via
    ``WildfireDetector.predict_wildfire({"thermal_image": patch})``. Physics
    is the deterministic brightness-temperature path (no weights loaded);
    learned is the same detector after ``load_neural_weights(candidate)``.

    The physics path thresholds ch0 -- TODAY'S max brightness temperature --
    so on this forecasting task it is effectively a persistence forecaster
    of tomorrow ("fire burning today near the center => alarm"). That is the
    honest baseline to beat, and it is recorded as such in ``extras``.

    Returns:
        The evaluation outcome (primary metric: AUC of the deployed
        confidence output, higher is better).
    """
    from omni_mercury_engine.detectors.geological.wildfire import WildfireDetector

    ds = build_dataset(ctx)
    _, _, test_mask = SPLIT.masks(ds.years)
    test_idx = np.flatnonzero(test_mask)
    if test_idx.size == 0:
        raise RuntimeError("no test samples found; cannot evaluate")

    cand_path, _ = candidate_paths(ctx.data_dir, HOOK_NAME)
    if not cand_path.exists():
        raise FileNotFoundError(f"no candidate checkpoint at {cand_path}; run --train first")

    def _detector() -> WildfireDetector:
        return WildfireDetector(
            enable_spread_modeling=False,
            enable_ndvi_processing=False,
            enable_resonance=False,
            enable_enhanced_cnn=False,
        )

    physics_det = _detector()
    learned_det = _detector()
    learned_det.load_neural_weights(str(cand_path))

    labels = ds.labels[test_idx]
    conf: dict[str, list[float]] = {"physics": [], "learned": []}
    detected: dict[str, list[bool]] = {"physics": [], "learned": []}
    for i in test_idx:
        case = {"thermal_image": ds.patches[i]}
        for name, det in (("physics", physics_det), ("learned", learned_det)):
            out = det.predict_wildfire(case)
            c = float(out.confidence)
            if not np.isfinite(c):
                raise RuntimeError(f"{name} path returned non-finite confidence for case {i}")
            conf[name].append(c)
            detected[name].append(bool(out.fire_detected))

    def _metrics(name: str) -> dict[str, float]:
        c = np.asarray(conf[name])
        d = np.asarray(detected[name], dtype=bool)
        is_pos = labels == 1.0
        tp = float(np.sum(d & is_pos))
        fn = float(np.sum(~d & is_pos))
        fp = float(np.sum(d & ~is_pos))
        return {
            "auc": binary_auc(labels, c),
            "recall_deployed": float(tp / max(tp + fn, 1.0)),
            "false_alarm_deployed": float(fp / max(float(np.sum(~is_pos)), 1.0)),
            "csi_deployed": float(tp / max(tp + fn + fp, 1.0)),
            "brier": brier_score(labels, c),
        }

    test_year_counts = {
        int(y): {
            **ds.per_year_counts.get(int(y), {}),
            "n_test_samples": int(np.sum(ds.years[test_idx] == y)),
        }
        for y in np.unique(ds.years[test_idx])
    }
    outcome = EvaluationOutcome(
        hook=HOOK_NAME,
        primary_metric="auc",
        higher_is_better=True,
        learned=_metrics("learned"),
        physics=_metrics("physics"),
        n_test_samples=int(test_idx.size),
        test_years=SPLIT.test_years,
        extras={
            "test_base_rate": float(labels.mean()),
            "per_year_counts": {int(y): c for y, c in ds.per_year_counts.items()},
            "test_year_counts": test_year_counts,
            "channels": list(CHANNEL_NAMES),
            "feature_spec": FEATURE_SPEC_VERSION,
            "label": LABEL_SPEC,
            "comparison": "identical held-out FIRMS-derived patches through "
            "WildfireDetector.predict_wildfire, physics fallback vs loaded checkpoint",
            "physics_baseline_interpretation": "the physics brightness-threshold path "
            "reads ch0 = TODAY's max brightness temperature, so on this next-day task "
            "it acts as a persistence forecaster of tomorrow -- the honest baseline",
            "deployed_thresholds": "learned: fire probability > "
            f"{DEPLOYED_PROB_THRESHOLD} (the detector's fixed decision rule); physics: "
            "its own absolute+contextual brightness-temperature decision",
        },
        constraints=[
            {
                "metric": "brier",
                "higher_is_better": False,
                "description": "probability quality of the deployed confidence output "
                "must not regress below the physics confidence",
            },
            {
                "metric": "csi_deployed",
                "higher_is_better": True,
                "description": "critical success index at each path's deployed decision "
                "rule must not regress -- recall/false-alarm are reported individually "
                "but sit at different fixed operating rules, so the gate constrains "
                "their operational aggregate rather than demanding pointwise dominance",
            },
        ],
    )
    save_evaluation(ctx.data_dir, outcome)
    logger.info(
        "evaluation: learned AUC %.4f vs physics %.4f on %d held-out patches (%s)",
        outcome.learned["auc"],
        outcome.physics["auc"],
        outcome.n_test_samples,
        "LEARNED WINS" if outcome.learned_beats_physics else "PHYSICS WINS",
    )
    return outcome


def ship(ctx: PipelineContext) -> tuple[Any, Any]:
    """Promote the candidate through the merit gate (may refuse loudly)."""
    from omni_mercury_engine.ml.hazard_training.common import load_evaluation

    outcome = load_evaluation(ctx.data_dir, HOOK_NAME)
    manifest_path = _firms_dir(ctx) / "manifest.json"
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
