# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Train the EarthquakePrecursorAnalyzer as a catalog-based seismicity-rate forecaster.

Spec: ``docs/research/EARTHQUAKE_PRECURSOR_LITERATURE_REVIEW.md`` (binding).
That review REJECTED the hook's original electromagnetic/Schumann framing
(no validated EM precursor exists; Jordan et al. 2011, Conti/Picozza 2021)
and approved exactly one honest reinterpretation: a **probabilistic regional
seismicity-rate forecast** trained on the real USGS ComCat catalog.

Task
    Label: ``y = 1`` iff >= 1 catalog earthquake with M >= 5.0 occurs inside a
    0.5-degree California cell within ``(t, t + 30 d]``. Forecast epochs every
    10 days; grid 32..42 N, -125..-114 E (20 x 22 = 440 cells); catalog
    1980-2024, Mc floor 2.5, ``eventtype=earthquake`` (excludes quarry blasts
    and the Nevada Test Site nuclear shots inside the box). M >= 6 remains an
    evaluation-only threshold — never a shipped claim of M6 prediction.

Mandatory model-card statement (review section 4.4)
    Most positive labels sit inside aftershock sequences of prior M >= 5
    events. A model trained on this label chiefly learns Omori/ETAS-style
    clustering: "large earthquake recently -> elevated probability now."
    That is genuine, honest forecasting skill — it is the mechanism behind
    every operational system (USGS/Reasenberg-Jones, INGV OEF, GNS) — but it
    is NOT a novel precursor capability, and this model has NOT demonstrated
    precursory skill beyond clustering unless the shipped provenance shows it
    beating the Reasenberg-Jones baseline below on held-out years. The
    evaluation reports the aftershock-dominance fraction explicitly.

Prohibited claims (review section 7)
    No deterministic prediction of individual earthquakes; no EM/Schumann/
    geomagnetic/ionospheric precursor detection; no magnitude or
    time-to-event estimates for specific future events; not an early-warning
    system — always defer to USGS and official agencies.

Feature spec ``seismicity-catalog-v1`` (128-dim contract, dims 32-127 zero)
    All features are computed STRICTLY from catalog events with
    ``event_time < t`` (no lookahead; property-tested). "nbhd" is the
    1.5-degree neighborhood: the 3 x 3 block of cells centred on the target
    cell. Invalid estimates are emitted as 0.0 with a presence flag of 0
    (mirroring the geomag spec) — never fabricated.

    ======  =============================================================
     index  meaning
    ======  =============================================================
       0-3  log1p count of M>=2.5 events in cell, trailing 7/30/90/365 d
       4-7  same, 1.5-degree neighborhood
         8  log1p Reasenberg-Jones triggered rate (per day) of M>=5 at t,
            summed over all prior nbhd events, generic California params
            a=-1.67, b=0.91, c=0.05 d, p=1.08 (Reasenberg & Jones 1989)
         9  log1p RJ expected triggered count of M>=5 in (t, t+30 d], nbhd
     10-11  max magnitude in cell, trailing 30/365 d (0 if none)
     12-13  max magnitude in nbhd, trailing 30/365 d (0 if none)
        14  Aki-Utsu MLE b-value, nbhd trailing 365 d, m >= Mc(MAXC)+0.2,
            binning-corrected; 0 unless flag 18 (Aki 1965; Utsu 1965)
        15  Shi & Bolt (1982) standard error of dim 14; 0 unless flag 18
        16  b-positive estimate (van der Elst 2021); 0 unless flag 19
        17  Mc via maximum curvature + 0.2 (Woessner & Wiemer 2005);
            0 unless flag 20
        18  flag: >= 50 events at/above Mc in nbhd w365 (dims 14-15 valid)
        19  flag: >= 50 positive successive-magnitude differences (dim 16)
        20  flag: >= 50 events in nbhd w365 (dim 17 computable)
        21  flag: MAXC bin <= 2.55, i.e. window consistent with the 2.5
            completeness floor (validity mask, review F4)
        22  log1p days since last M>=5.5 in nbhd, capped at 3650; the cap
            value when none seen. Clustering covariate ONLY — the "overdue"/
            seismic-gap reading is forbidden (Kagan & Jackson 1991)
        23  flag: an M>=5.5 occurred in nbhd within the 3650-d lookback
        24  log1p days since the largest-magnitude nbhd event of the
            trailing 365 d (0 when the window is empty)
        25  log10 sum of moment proxy 10^(1.5 m) over nbhd w365 (0 if none)
        26  median hypocentral depth km / 10, nbhd w365 (0 when unknown)
        27  IQR of hypocentral depth km / 10, nbhd w365
        28  Zaliapin-Ben-Zion: fraction of nbhd-w365 events with
            log10(eta) < -5 (eta = dt_yr * r_km^1.6 * 10^(-1.0 m_parent);
            Zaliapin & Ben-Zion 2013); 0 unless flag 30
        29  mean log10(eta) over the same events; 0 unless flag 30
        30  flag: >= 20 events with a computable nearest-neighbor eta
            (capped at the 400 most recent window events, deterministic)
        31  (epoch - 1981-01-01) / 44 yr — network-era drift covariate
    32-127  zero padding (reserved by the fixed input_dim=128 contract)
    ======  =============================================================

    Deliberately absent (review): every EM/Schumann/geomagnetic/ionospheric
    input, any "overdue"/gap/moment-deficit feature, astronomical features.

Merit-gate baseline (review section 5(b))
    A deterministic Reasenberg-Jones/ETAS-lite clustering model computed from
    the identical catalog: ``lambda(M>=5, 30 d) = mu_cell + sum_i
    10^(a + b (m_i - 5)) * integral_t^{t+30} (s - t_i + c)^-p ds`` over prior
    in-cell events, ``P = 1 - exp(-lambda)``; ``mu_cell`` is the train-years
    in-cell M>=5 rate with +0.5 Laplace smoothing. The detector's non-neural
    fallback abstains from event probabilities, so this documented physics
    baseline stands in for it (allowed by ``EvaluationOutcome.physics``). A
    bare time-independent Poisson (``P = 1 - exp(-mu_cell)``) is recorded in
    extras as a sanity floor only — beating Poisson merely rediscovers Omori
    (1894) and does not justify shipping.

Splits (by year, never random): train 1985-2009, val 2010-2016, test
2017-2024 — the 2019 Ridgecrest sequence (M6.4 + M7.1) lands in the held-out
test years. Training rows keep every positive plus a seeded, stratified
(year x train-years cell-activity tercile) negative sample with
inverse-sampling-probability weights, so the weighted BCE estimates the
full-population log-loss; test years keep the FULL (cell, epoch) grid.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import itertools
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

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

logger = logging.getLogger(__name__)

HOOK_NAME = "earthquake_precursor"
CHECKPOINT_NAME = "earthquake_precursor_ca"
FEATURE_SPEC_VERSION = "seismicity-catalog-v1"
EQ_FEATURE_DIM = 128
_N_INFORMATIVE = 32

EQ_FEATURE_NAMES: tuple[str, ...] = (
    "rate_cell_w7",
    "rate_cell_w30",
    "rate_cell_w90",
    "rate_cell_w365",
    "rate_nbhd_w7",
    "rate_nbhd_w30",
    "rate_nbhd_w90",
    "rate_nbhd_w365",
    "rj_rate_instant",
    "rj_rate_30d",
    "max_mag_cell_w30",
    "max_mag_cell_w365",
    "max_mag_nbhd_w30",
    "max_mag_nbhd_w365",
    "b_aki_w365",
    "b_aki_stderr_w365",
    "b_positive_w365",
    "mc_maxc_w365",
    "flag_b_aki",
    "flag_b_positive",
    "flag_mc",
    "flag_mc_ok",
    "t_since_m55",
    "flag_m55_seen",
    "t_since_largest_w365",
    "moment_w365",
    "depth_median_w365",
    "depth_iqr_w365",
    "nn_frac_clustered_w365",
    "nn_mean_log_eta_w365",
    "flag_nn",
    "years_since_1981",
) + tuple(f"reserved_{i}" for i in range(_N_INFORMATIVE, EQ_FEATURE_DIM))

# --- region / grid / label constants (documented in the module docstring) ---
LAT_MIN, LAT_MAX = 32.0, 42.0
LON_MIN, LON_MAX = -125.0, -114.0
CELL_DEG = 0.5
N_LAT = round((LAT_MAX - LAT_MIN) / CELL_DEG)  # 20
N_LON = round((LON_MAX - LON_MIN) / CELL_DEG)  # 22
CATALOG_MIN_MAG = 2.5
LABEL_MIN_MAG = 5.0
LABEL_WINDOW_DAYS = 30.0
EPOCH_STRIDE_DAYS = 10.0
NEG_RATIO = 30

CATALOG_YEARS = tuple(range(1980, 2025))
SPLIT = TemporalSplit(
    train_years=tuple(range(1985, 2010)),
    val_years=tuple(range(2010, 2017)),
    test_years=tuple(range(2017, 2025)),
)

# Generic California Reasenberg-Jones parameters (Reasenberg & Jones 1989;
# review section 6, slot F2/rj_triggered_rate). Recorded in provenance.
RJ_A = -1.67
RJ_B = 0.91
RJ_C = 0.05  # days
RJ_P = 1.08

# b-value estimation (review F3/F4): Aki-Utsu with 0.1 magnitude binning,
# Mc = MAXC + 0.2, minimum sample 50; b-positive per van der Elst (2021).
MAG_BIN = 0.1
BVAL_MIN_N = 50
BPOS_DIFF_CUT = 0.1

# Zaliapin-Ben-Zion nearest-neighbor parameters for California (2013 paper):
# b = 1.0, fractal dimension df = 1.6, clustered threshold log10(eta) < -5.
ETA_B = 1.0
ETA_DF = 1.6
ETA_LOG10_CLUSTERED = -5.0
ETA_MIN_EVENTS = 20
ETA_MAX_EVENTS = 400

# Time origin for the day axis; epochs start 1985-01-01 so every feature
# window has >= 5 years of catalog history behind it.
DAY0 = _dt.datetime(1980, 1, 1, tzinfo=_dt.UTC)
EPOCH_START_DAY = float((_dt.datetime(1985, 1, 1, tzinfo=_dt.UTC) - DAY0).days)
CATALOG_END_DAY = float((_dt.datetime(2025, 1, 1, tzinfo=_dt.UTC) - DAY0).days)
_DAYS_1981 = float((_dt.datetime(1981, 1, 1, tzinfo=_dt.UTC) - DAY0).days)

_QUERY_BASE = (
    "https://earthquake.usgs.gov/fdsnws/event/1/{endpoint}"
    "?starttime={start}&endtime={end}"
    f"&minmagnitude={CATALOG_MIN_MAG}"
    f"&minlatitude={LAT_MIN:.0f}&maxlatitude={LAT_MAX:.0f}"
    f"&minlongitude={LON_MIN:.0f}&maxlongitude={LON_MAX:.0f}"
    "&eventtype=earthquake"
)
#: FDSN row cap; a yearly page at/above this means the paging must shrink.
FDSN_ROW_CAP = 20000

# Probability clamp applied identically to every model before log-loss.
_P_CLAMP = 1e-6


def _count_url(year: int) -> str:
    """FDSN ``count`` endpoint URL for one catalog year."""
    return _QUERY_BASE.format(endpoint="count", start=f"{year}-01-01", end=f"{year + 1}-01-01")


def _query_url(year: int) -> str:
    """FDSN ``query`` (CSV) endpoint URL for one catalog year, time-ascending."""
    return (
        _QUERY_BASE.format(endpoint="query", start=f"{year}-01-01", end=f"{year + 1}-01-01")
        + "&format=csv&orderby=time-asc"
    )


def fetch(ctx: PipelineContext) -> dict[str, Any]:
    """Download the 1980-2024 USGS ComCat California catalog, one CSV per year.

    For every year the FDSN ``count`` endpoint is probed (and cached) first;
    a yearly page at or above the 20k row cap fails loud instead of silently
    truncating. All fetches go through :func:`cached_fetch` (allowlisted
    host, sha256-pinned files).

    Returns:
        Manifest with per-file URLs, sha256 digests, and event totals.
    """
    cat_dir = ctx.data_dir / "usgs_catalog"
    sources: list[dict[str, Any]] = []
    total_expected = 0
    for year in CATALOG_YEARS:
        count_path = cached_fetch(_count_url(year), cat_dir / f"usgs_count_{year}.txt")
        n = int(count_path.read_text().strip())
        if n >= FDSN_ROW_CAP:
            raise RuntimeError(
                f"USGS count for {year} is {n} >= the {FDSN_ROW_CAP} FDSN row cap; "
                "yearly paging is no longer safe -- switch this fetch to monthly pages."
            )
        total_expected += n
        csv_path = cached_fetch(_query_url(year), cat_dir / f"usgs_ca_{year}.csv", timeout=180.0)
        n_rows = sum(1 for _ in csv_path.open()) - 1  # minus header
        if n_rows >= FDSN_ROW_CAP:
            raise RuntimeError(f"{csv_path} holds {n_rows} rows -- at the FDSN cap; refetch")
        if abs(n_rows - n) > max(5, n // 100):
            logger.warning(
                "year %d: CSV rows (%d) differ from count probe (%d); catalog was "
                "revised between requests -- the cached CSV is authoritative",
                year,
                n_rows,
                n,
            )
        sources.append(
            {
                "url": _query_url(year),
                "sha256": sha256_file(csv_path),
                "description": f"USGS ComCat California M>=2.5 earthquakes, {year} ({n_rows} rows)",
            }
        )
    manifest = {
        "hook": HOOK_NAME,
        "sources": sources,
        "total_events_count_probe": total_expected,
        "rj_parameters": {"a": RJ_A, "b": RJ_B, "c_days": RJ_C, "p": RJ_P},
    }
    manifest_path = cat_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    logger.info(
        "fetch complete: %d yearly CSVs, %d events (count probe) under %s",
        len(CATALOG_YEARS),
        total_expected,
        cat_dir,
    )
    return manifest


@dataclass
class Catalog:
    """Parsed earthquake catalog, time-sorted (days since 1980-01-01 UTC)."""

    t_days: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    depth: np.ndarray  # km; NaN when the catalog row omits it
    mag: np.ndarray


def parse_catalog(paths: list[Path]) -> Catalog:
    """Parse USGS ComCat CSV files into time-sorted arrays.

    Accepts full ComCat exports or reduced fixture excerpts; only the
    ``time``, ``latitude``, ``longitude``, ``depth`` and ``mag`` columns are
    read (by header name).

    Args:
        paths: CSV paths (any order; output is globally time-sorted).

    Returns:
        The concatenated, time-sorted catalog.

    Raises:
        ValueError: If a file lacks the required columns -- the format
            assumption is wrong and guessing would corrupt every label.
    """
    import csv

    t_list: list[float] = []
    lat_list: list[float] = []
    lon_list: list[float] = []
    depth_list: list[float] = []
    mag_list: list[float] = []
    required = {"time", "latitude", "longitude", "depth", "mag"}
    n_skipped = 0
    for path in paths:
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
                raise ValueError(
                    f"{path}: missing required columns {sorted(required)}; "
                    f"got {reader.fieldnames} -- refusing to guess the format"
                )
            for row in reader:
                mag_raw = row["mag"].strip()
                if not mag_raw:
                    n_skipped += 1
                    continue
                stamp = row["time"].strip().replace("Z", "+00:00")
                dt = _dt.datetime.fromisoformat(stamp)
                t_list.append((dt - DAY0).total_seconds() / 86400.0)
                lat_list.append(float(row["latitude"]))
                lon_list.append(float(row["longitude"]))
                depth_raw = row["depth"].strip()
                depth_list.append(float(depth_raw) if depth_raw else float("nan"))
                mag_list.append(float(mag_raw))
    if n_skipped:
        logger.info("parse_catalog: skipped %d rows without a magnitude", n_skipped)
    order = np.argsort(np.asarray(t_list), kind="mergesort")
    return Catalog(
        t_days=np.asarray(t_list, dtype=np.float64)[order],
        lat=np.asarray(lat_list, dtype=np.float64)[order],
        lon=np.asarray(lon_list, dtype=np.float64)[order],
        depth=np.asarray(depth_list, dtype=np.float64)[order],
        mag=np.asarray(mag_list, dtype=np.float64)[order],
    )


def cell_of(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map coordinates to 0.5-degree grid indices (edge events clipped in)."""
    ix = np.clip(((np.asarray(lat) - LAT_MIN) / CELL_DEG).astype(np.int64), 0, N_LAT - 1)
    iy = np.clip(((np.asarray(lon) - LON_MIN) / CELL_DEG).astype(np.int64), 0, N_LON - 1)
    return ix, iy


@dataclass
class _CellArrays:
    """Time-sorted event arrays for one cell or one 3x3 neighborhood."""

    t: np.ndarray
    mag: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    depth: np.ndarray


_EMPTY = _CellArrays(*(np.empty(0, dtype=np.float64) for _ in range(5)))


class CatalogIndex:
    """Per-cell and per-neighborhood views of a catalog for fast windowing.

    Everything is precomputed once; feature/label/baseline queries then run
    on sorted arrays via ``searchsorted``.
    """

    def __init__(self, catalog: Catalog) -> None:
        """Build the per-cell buckets and derived per-neighborhood arrays."""
        self.catalog = catalog
        ix, iy = cell_of(catalog.lat, catalog.lon)
        flat = ix * N_LON + iy
        self._cell: dict[int, _CellArrays] = {}
        for cid in np.unique(flat):
            m = flat == cid
            self._cell[int(cid)] = _CellArrays(
                catalog.t_days[m], catalog.mag[m], catalog.lat[m], catalog.lon[m], catalog.depth[m]
            )
        self._nbhd: dict[int, _CellArrays] = {}
        self._cell_m5_t: dict[int, np.ndarray] = {}
        self._cell_m4_t: dict[int, np.ndarray] = {}
        self._nbhd_m55_t: dict[int, np.ndarray] = {}

    def cell(self, ix: int, iy: int) -> _CellArrays:
        """Events inside one cell, time-sorted (empty view when none)."""
        return self._cell.get(ix * N_LON + iy, _EMPTY)

    def nbhd(self, ix: int, iy: int) -> _CellArrays:
        """Events inside the 3x3 neighborhood of a cell, time-sorted."""
        cid = ix * N_LON + iy
        cached = self._nbhd.get(cid)
        if cached is not None:
            return cached
        parts = [
            self._cell[jx * N_LON + jy]
            for jx in range(max(0, ix - 1), min(N_LAT, ix + 2))
            for jy in range(max(0, iy - 1), min(N_LON, iy + 2))
            if jx * N_LON + jy in self._cell
        ]
        if not parts:
            out = _EMPTY
        else:
            t = np.concatenate([p.t for p in parts])
            order = np.argsort(t, kind="mergesort")
            out = _CellArrays(
                t[order],
                np.concatenate([p.mag for p in parts])[order],
                np.concatenate([p.lat for p in parts])[order],
                np.concatenate([p.lon for p in parts])[order],
                np.concatenate([p.depth for p in parts])[order],
            )
        self._nbhd[cid] = out
        return out

    def cell_m5_times(self, ix: int, iy: int) -> np.ndarray:
        """Times of in-cell M>=5.0 label events (sorted)."""
        cid = ix * N_LON + iy
        if cid not in self._cell_m5_t:
            c = self.cell(ix, iy)
            self._cell_m5_t[cid] = c.t[c.mag >= LABEL_MIN_MAG]
        return self._cell_m5_t[cid]

    def cell_m4_times(self, ix: int, iy: int) -> np.ndarray:
        """Times of in-cell M>=4.0 events (diagnostic time-head target)."""
        cid = ix * N_LON + iy
        if cid not in self._cell_m4_t:
            c = self.cell(ix, iy)
            self._cell_m4_t[cid] = c.t[c.mag >= 4.0]
        return self._cell_m4_t[cid]

    def nbhd_m55_times(self, ix: int, iy: int) -> np.ndarray:
        """Times of neighborhood M>=5.5 events (sorted)."""
        cid = ix * N_LON + iy
        if cid not in self._nbhd_m55_t:
            n = self.nbhd(ix, iy)
            self._nbhd_m55_t[cid] = n.t[n.mag >= 5.5]
        return self._nbhd_m55_t[cid]

    def label(self, ix: int, iy: int, t_days: float) -> int:
        """1 iff an in-cell M>=5.0 event occurs within ``(t, t + 30 d]``."""
        m5 = self.cell_m5_times(ix, iy)
        lo = int(np.searchsorted(m5, t_days, side="right"))
        hi = int(np.searchsorted(m5, t_days + LABEL_WINDOW_DAYS, side="right"))
        return int(hi > lo)


def epoch_days_and_years() -> tuple[np.ndarray, np.ndarray]:
    """Forecast epochs (day offsets from 1980-01-01) and their calendar years.

    Epochs run every 10 days from 1985-01-01 while the full 30-day label
    window still fits inside the fetched catalog span.
    """
    days = np.arange(EPOCH_START_DAY, CATALOG_END_DAY - LABEL_WINDOW_DAYS + 1e-9, EPOCH_STRIDE_DAYS)
    years = np.array(
        [(DAY0 + _dt.timedelta(days=float(d))).year for d in days],
        dtype=np.int64,
    )
    return days, years


# ---------------------------------------------------------------------------
# estimators (unit-tested against hand computations on real fixture excerpts)
# ---------------------------------------------------------------------------


def mc_maxc(mags: np.ndarray) -> float:
    """Magnitude of completeness: maximum-curvature mode + 0.2 correction.

    Args:
        mags: Window magnitudes (any order).

    Returns:
        Mc estimate (MAXC bin + 0.2), per Wiemer & Wyss (2000) with the
        Woessner & Wiemer (2005) correction. Caller enforces sample-size
        validity; this function only needs a non-empty array.
    """
    bins = np.round(np.asarray(mags, dtype=np.float64) / MAG_BIN) * MAG_BIN
    values, counts = np.unique(bins, return_counts=True)
    return float(values[int(np.argmax(counts))] + 0.2)


def aki_utsu_b(mags: np.ndarray, mc: float) -> tuple[float, float, int]:
    """Aki-Utsu maximum-likelihood b-value with Shi & Bolt standard error.

    ``b = log10(e) / (mean(m) - (Mc - bin/2))`` over events with ``m >= Mc``
    (Aki 1965; Utsu's binned-magnitude correction), with the Shi & Bolt
    (1982) standard error ``2.3026 b^2 sqrt(sum (m - mean)^2 / (n (n-1)))``.

    Args:
        mags: Window magnitudes.
        mc: Completeness cut applied before estimation.

    Returns:
        Tuple ``(b, stderr, n_used)``; ``(0.0, 0.0, n)`` when ``n < 2`` or
        the mean does not exceed the corrected cut (degenerate sample).
    """
    m = np.asarray(mags, dtype=np.float64)
    m = m[m >= mc - 1e-9]
    n = int(m.size)
    if n < 2:
        return 0.0, 0.0, n
    denom = float(np.mean(m)) - (mc - MAG_BIN / 2.0)
    if denom <= 0:
        return 0.0, 0.0, n
    b = float(np.log10(np.e) / denom)
    var_m = float(np.sum((m - np.mean(m)) ** 2) / (n * (n - 1)))
    stderr = float(2.3026 * b * b * np.sqrt(var_m))
    return b, stderr, n


def b_positive(mags_time_ordered: np.ndarray) -> tuple[float, int]:
    """b-positive estimate from successive magnitude differences.

    Applies the Aki-Utsu form to the positive differences
    ``dm = m_i - m_(i-1) >= 0.1`` (van der Elst 2021), which is robust to the
    transient incompleteness that biases classical b right after mainshocks.

    Args:
        mags_time_ordered: Window magnitudes in time order.

    Returns:
        Tuple ``(b_positive, n_differences_used)``; ``(0.0, n)`` when
        degenerate.
    """
    m = np.asarray(mags_time_ordered, dtype=np.float64)
    if m.size < 2:
        return 0.0, 0
    dm = np.diff(m)
    dm = dm[dm >= BPOS_DIFF_CUT - 1e-9]
    n = int(dm.size)
    if n < 2:
        return 0.0, n
    denom = float(np.mean(dm)) - (BPOS_DIFF_CUT - MAG_BIN / 2.0)
    if denom <= 0:
        return 0.0, n
    return float(np.log10(np.e) / denom), n


def _flat_distances_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Equirectangular distance in km (adequate at neighborhood scale)."""
    mean_lat = np.deg2rad((lat1 + lat2) / 2.0)
    dx = (np.asarray(lon2) - np.asarray(lon1)) * 111.320 * np.cos(mean_lat)
    dy = (np.asarray(lat2) - np.asarray(lat1)) * 110.574
    return np.sqrt(dx * dx + dy * dy)


def zaliapin_eta_stats(window: _CellArrays) -> tuple[float, float, int]:
    """Zaliapin-Ben-Zion nearest-neighbor statistics for a time-sorted window.

    For each event j the parent is the preceding event minimizing
    ``eta = dt_years * r_km^df * 10^(-b m_parent)`` with the California
    parameters b=1.0, df=1.6 (Zaliapin & Ben-Zion 2013). Deterministically
    capped at the 400 most recent window events.

    Args:
        window: Time-sorted events (the trailing-365 d neighborhood window).

    Returns:
        ``(fraction with log10 eta < -5, mean log10 eta, n_children)``;
        zeros with ``n_children`` when fewer than 2 events exist.
    """
    k = int(window.t.size)
    if k < 2:
        return 0.0, 0.0, 0
    if k > ETA_MAX_EVENTS:
        sl = slice(k - ETA_MAX_EVENTS, k)
        window = _CellArrays(
            window.t[sl], window.mag[sl], window.lat[sl], window.lon[sl], window.depth[sl]
        )
        k = ETA_MAX_EVENTS
    dt_years = (window.t[None, :] - window.t[:, None]) / 365.25  # [parent, child]
    r_km = np.maximum(
        _flat_distances_km(
            window.lat[:, None], window.lon[:, None], window.lat[None, :], window.lon[None, :]
        ),
        0.1,
    )
    with np.errstate(over="ignore"):
        eta = np.where(
            dt_years > 0,
            np.maximum(dt_years, 1e-8) * r_km**ETA_DF * 10.0 ** (-ETA_B * window.mag[:, None]),
            np.inf,
        )
    min_eta = eta.min(axis=0)[1:]  # children: every event but the first
    finite = np.isfinite(min_eta)
    if not finite.any():
        return 0.0, 0.0, 0
    log_eta = np.log10(np.maximum(min_eta[finite], 1e-15))
    return (
        float(np.mean(log_eta < ETA_LOG10_CLUSTERED)),
        float(np.mean(log_eta)),
        int(log_eta.size),
    )


def _rj_kernel_weight(mags: np.ndarray) -> np.ndarray:
    """Reasenberg-Jones productivity term ``10^(a + b (m - 5))`` per event."""
    return 10.0 ** (RJ_A + RJ_B * (np.asarray(mags, dtype=np.float64) - LABEL_MIN_MAG))


def rj_triggered_rate(times: np.ndarray, mags: np.ndarray, t_days: float) -> float:
    """Instantaneous RJ triggered rate (events/day of M>=5) at ``t_days``."""
    dt = t_days - np.asarray(times, dtype=np.float64)
    m = dt > 0
    if not m.any():
        return 0.0
    return float(np.sum(_rj_kernel_weight(mags[m]) * (dt[m] + RJ_C) ** (-RJ_P)))


def rj_triggered_count_30d(times: np.ndarray, mags: np.ndarray, t_days: float) -> float:
    """Expected RJ-triggered count of M>=5 events in ``(t, t + 30 d]``.

    Uses the closed-form Omori integral
    ``[(dt + 30 + c)^(1-p) - (dt + c)^(1-p)] / (1 - p)`` per prior event.
    """
    dt = t_days - np.asarray(times, dtype=np.float64)
    m = dt > 0
    if not m.any():
        return 0.0
    upper = (dt[m] + LABEL_WINDOW_DAYS + RJ_C) ** (1.0 - RJ_P)
    lower = (dt[m] + RJ_C) ** (1.0 - RJ_P)
    return float(np.sum(_rj_kernel_weight(mags[m]) * (upper - lower) / (1.0 - RJ_P)))


def rj_probability(cell_events: _CellArrays, t_days: float, mu_30d: float) -> float:
    """Documented physics baseline: P(>=1 in-cell M>=5 event in 30 days).

    ``P = 1 - exp(-(mu_30d + triggered))`` -- the Reasenberg-Jones/ETAS-lite
    clustering baseline mandated by the literature review's merit gate.
    Deterministic: same catalog, epoch and background rate always give the
    same probability.

    Args:
        cell_events: In-cell catalog view (only events before ``t_days``
            contribute; later events are excluded, never consulted).
        t_days: Forecast epoch (days since 1980-01-01).
        mu_30d: Train-years background expectation for the cell per 30 days.

    Returns:
        Probability in ``(0, 1)``.
    """
    lam = mu_30d + rj_triggered_count_30d(cell_events.t, cell_events.mag, t_days)
    return float(1.0 - np.exp(-lam))


def poisson_probability(mu_30d: float) -> float:
    """Sanity-floor time-independent Poisson probability for one cell."""
    return float(1.0 - np.exp(-mu_30d))


def background_mu(index: CatalogIndex) -> np.ndarray:
    """Train-years in-cell M>=5 background expectation per 30 days, per cell.

    Laplace-smoothed (+0.5 events) so cells with zero train-years M>=5
    activity keep a finite rate; computed STRICTLY from train years.

    Returns:
        Array of shape ``(N_LAT, N_LON)``.
    """
    t_lo = float((_dt.datetime(min(SPLIT.train_years), 1, 1, tzinfo=_dt.UTC) - DAY0).days)
    t_hi = float((_dt.datetime(max(SPLIT.train_years) + 1, 1, 1, tzinfo=_dt.UTC) - DAY0).days)
    n_days = t_hi - t_lo
    mu = np.empty((N_LAT, N_LON), dtype=np.float64)
    for ix in range(N_LAT):
        for iy in range(N_LON):
            m5 = index.cell_m5_times(ix, iy)
            n = int(np.searchsorted(m5, t_hi) - np.searchsorted(m5, t_lo))
            mu[ix, iy] = (n + 0.5) / n_days * LABEL_WINDOW_DAYS
    return mu


# ---------------------------------------------------------------------------
# feature builder
# ---------------------------------------------------------------------------


def _window_slice(arrs: _CellArrays, t_days: float, window: float) -> _CellArrays:
    """Events with ``t - window <= time < t`` (strictly before the epoch)."""
    lo = int(np.searchsorted(arrs.t, t_days - window, side="left"))
    hi = int(np.searchsorted(arrs.t, t_days, side="left"))
    sl = slice(lo, hi)
    return _CellArrays(arrs.t[sl], arrs.mag[sl], arrs.lat[sl], arrs.lon[sl], arrs.depth[sl])


def build_feature_vector(index: CatalogIndex, ix: int, iy: int, t_days: float) -> np.ndarray:
    """Build the 128-dim ``seismicity-catalog-v1`` feature vector for a case.

    Every value derives from catalog events strictly before ``t_days``
    (asserted); the label window ``(t, t + 30 d]`` is never touched.

    Args:
        index: Prebuilt catalog index.
        ix: Cell latitude index.
        iy: Cell longitude index.
        t_days: Forecast epoch, days since 1980-01-01 UTC.

    Returns:
        ``float32`` array of shape ``(128,)`` per the module-docstring spec.
    """
    vec = np.zeros(EQ_FEATURE_DIM, dtype=np.float32)
    cell = index.cell(ix, iy)
    nbhd = index.nbhd(ix, iy)

    for base, arrs in ((0, cell), (4, nbhd)):
        for k, window in enumerate((7.0, 30.0, 90.0, 365.0)):
            w = _window_slice(arrs, t_days, window)
            if w.t.size:
                assert float(w.t[-1]) < t_days, "lookahead: feature window crossed the epoch"
            vec[base + k] = np.log1p(float(w.t.size))

    hist_hi = int(np.searchsorted(nbhd.t, t_days, side="left"))
    vec[8] = np.log1p(rj_triggered_rate(nbhd.t[:hist_hi], nbhd.mag[:hist_hi], t_days))
    vec[9] = np.log1p(rj_triggered_count_30d(nbhd.t[:hist_hi], nbhd.mag[:hist_hi], t_days))

    cell_w30 = _window_slice(cell, t_days, 30.0)
    cell_w365 = _window_slice(cell, t_days, 365.0)
    nbhd_w30 = _window_slice(nbhd, t_days, 30.0)
    nbhd_w365 = _window_slice(nbhd, t_days, 365.0)
    vec[10] = float(cell_w30.mag.max()) if cell_w30.mag.size else 0.0
    vec[11] = float(cell_w365.mag.max()) if cell_w365.mag.size else 0.0
    vec[12] = float(nbhd_w30.mag.max()) if nbhd_w30.mag.size else 0.0
    vec[13] = float(nbhd_w365.mag.max()) if nbhd_w365.mag.size else 0.0

    # b-value block with presence flags (never fabricated when under-sampled).
    if nbhd_w365.mag.size >= BVAL_MIN_N:
        mc = mc_maxc(nbhd_w365.mag)
        vec[17] = mc
        vec[20] = 1.0
        vec[21] = 1.0 if mc - 0.2 <= CATALOG_MIN_MAG + MAG_BIN / 2.0 else 0.0
        b, stderr, n_used = aki_utsu_b(nbhd_w365.mag, mc)
        if n_used >= BVAL_MIN_N and b > 0:
            vec[14] = b
            vec[15] = stderr
            vec[18] = 1.0
    bpos, n_pos = b_positive(nbhd_w365.mag)
    if n_pos >= BVAL_MIN_N and bpos > 0:
        vec[16] = bpos
        vec[19] = 1.0

    m55 = index.nbhd_m55_times(ix, iy)
    hi55 = int(np.searchsorted(m55, t_days, side="left"))
    if hi55 > 0 and (t_days - float(m55[hi55 - 1])) <= 3650.0:
        vec[22] = np.log1p(t_days - float(m55[hi55 - 1]))
        vec[23] = 1.0
    else:
        vec[22] = np.log1p(3650.0)  # capped; flag 23 stays 0 ("none seen")

    if nbhd_w365.mag.size:
        t_largest = float(nbhd_w365.t[int(np.argmax(nbhd_w365.mag))])
        vec[24] = np.log1p(t_days - t_largest)
        vec[25] = float(np.log10(np.sum(10.0 ** (1.5 * nbhd_w365.mag))))
        depths = nbhd_w365.depth[np.isfinite(nbhd_w365.depth)]
        if depths.size:
            vec[26] = float(np.median(depths)) / 10.0
            q75, q25 = np.percentile(depths, [75, 25])
            vec[27] = float(q75 - q25) / 10.0

    frac, mean_log_eta, n_eta = zaliapin_eta_stats(nbhd_w365)
    if n_eta >= ETA_MIN_EVENTS:
        vec[28] = frac
        vec[29] = mean_log_eta
        vec[30] = 1.0

    vec[31] = (t_days - _DAYS_1981) / (44.0 * 365.25)
    return vec


# ---------------------------------------------------------------------------
# dataset assembly
# ---------------------------------------------------------------------------


@dataclass
class EarthquakeDataset:
    """Feature/label matrices plus sampling weights and diagnostics targets.

    ``weight`` is the inverse-sampling-probability weight (1.0 for every
    positive and every test-year row); the weighted BCE / log-loss over the
    sampled rows is therefore an unbiased estimate of the full-grid value.
    """

    features: np.ndarray
    label: np.ndarray
    weight: np.ndarray
    years: np.ndarray
    cell_ix: np.ndarray
    cell_iy: np.ndarray
    epoch_day: np.ndarray
    mag_target: np.ndarray  # observed max M in window / 9 (positives only)
    time_target: np.ndarray  # days to first in-cell M>=4 / 30, clamped [0,1]
    time_mask: np.ndarray  # rows where an in-cell M>=4 occurs in the window
    feature_mean: np.ndarray
    feature_std: np.ndarray
    meta: dict[str, Any]


def _load_catalog(ctx: PipelineContext) -> tuple[Catalog, list[Path]]:
    """Load every cached catalog year, failing loud on gaps."""
    cat_dir = ctx.data_dir / "usgs_catalog"
    paths = []
    for year in CATALOG_YEARS:
        path = cat_dir / f"usgs_ca_{year}.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing catalog cache {path}; run the --fetch stage first")
        paths.append(path)
    return parse_catalog(paths), paths


def _dataset_cache_key(paths: list[Path], seed: int) -> str:
    """Cache key pinning the spec, split, sampling config and input files."""
    h = hashlib.sha256()
    h.update(
        json.dumps(
            {
                "spec": FEATURE_SPEC_VERSION,
                "seed": seed,
                "neg_ratio": NEG_RATIO,
                "split": [list(SPLIT.train_years), list(SPLIT.val_years), list(SPLIT.test_years)],
                "files": [sha256_file(p) for p in paths],
            },
            sort_keys=True,
        ).encode()
    )
    return h.hexdigest()[:16]


def build_dataset(ctx: PipelineContext) -> EarthquakeDataset:
    """Assemble the (cell, epoch) dataset from the cached real catalog.

    Keeps ALL positive cases; train/val negatives are a seeded stratified
    sample (year x train-years cell-activity tercile) of ``NEG_RATIO`` times
    the train+val positive count, carrying inverse-sampling weights; test
    years keep the complete grid so held-out evaluation is unbiased.
    Standardization statistics come from TRAIN rows only.
    """
    catalog, paths = _load_catalog(ctx)
    cache_path = (
        ctx.data_dir / "usgs_catalog" / f"dataset_{_dataset_cache_key(paths, ctx.seed)}.npz"
    )
    if cache_path.exists() and ctx.limit_samples is None:
        z = np.load(cache_path, allow_pickle=False)
        meta = json.loads(str(z["meta_json"]))
        logger.info("dataset cache hit: %s (%d rows)", cache_path, z["label"].size)
        return EarthquakeDataset(
            features=z["features"],
            label=z["label"],
            weight=z["weight"],
            years=z["years"],
            cell_ix=z["cell_ix"],
            cell_iy=z["cell_iy"],
            epoch_day=z["epoch_day"],
            mag_target=z["mag_target"],
            time_target=z["time_target"],
            time_mask=z["time_mask"],
            feature_mean=z["feature_mean"],
            feature_std=z["feature_std"],
            meta=meta,
        )

    index = CatalogIndex(catalog)
    epochs, epoch_years = epoch_days_and_years()
    n_epochs = epochs.size

    # Full label grid: labels[cell, epoch] via per-cell searchsorted.
    labels_grid = np.zeros((N_LAT, N_LON, n_epochs), dtype=np.int8)
    for ix in range(N_LAT):
        for iy in range(N_LON):
            m5 = index.cell_m5_times(ix, iy)
            if m5.size:
                lo = np.searchsorted(m5, epochs, side="right")
                hi = np.searchsorted(m5, epochs + LABEL_WINDOW_DAYS, side="right")
                labels_grid[ix, iy] = (hi > lo).astype(np.int8)

    train_mask_e, val_mask_e, test_mask_e = SPLIT.masks(epoch_years)
    trainval_e = train_mask_e | val_mask_e

    # Train-years cell activity terciles (stratification variable).
    t_lo = float((_dt.datetime(min(SPLIT.train_years), 1, 1, tzinfo=_dt.UTC) - DAY0).days)
    t_hi = float((_dt.datetime(max(SPLIT.train_years) + 1, 1, 1, tzinfo=_dt.UTC) - DAY0).days)
    activity = np.zeros((N_LAT, N_LON), dtype=np.int64)
    for ix in range(N_LAT):
        for iy in range(N_LON):
            c = index.cell(ix, iy)
            activity[ix, iy] = int(np.searchsorted(c.t, t_hi) - np.searchsorted(c.t, t_lo))
    q1, q2 = (
        np.quantile(activity[activity > 0], [1.0 / 3.0, 2.0 / 3.0])
        if (activity > 0).any()
        else (1, 2)
    )
    tercile = np.digitize(activity, [max(q1, 1), max(q2, q1 + 1)])  # 0/1/2

    rows: list[tuple[int, int, int]] = []  # (ix, iy, epoch index)
    weights: list[float] = []

    # 1) all positives, every split.
    pos_ix, pos_iy, pos_ie = np.nonzero(labels_grid)
    for pix, piy, pie in zip(pos_ix, pos_iy, pos_ie, strict=True):
        rows.append((int(pix), int(piy), int(pie)))
        weights.append(1.0)
    n_pos_trainval = int(np.sum(labels_grid[:, :, trainval_e]))

    # 2) full negative grid for test years.
    neg_test = np.nonzero(labels_grid[:, :, test_mask_e] == 0)
    test_epoch_indices = np.flatnonzero(test_mask_e)
    for nix, niy, k in zip(*neg_test, strict=True):
        rows.append((int(nix), int(niy), int(test_epoch_indices[k])))
        weights.append(1.0)

    # 3) seeded stratified negative sample for train+val years.
    rng = np.random.default_rng(ctx.seed)
    trainval_epoch_indices = np.flatnonzero(trainval_e)
    strata: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for ie in trainval_epoch_indices:
        year = int(epoch_years[ie])
        neg_cells = np.nonzero(labels_grid[:, :, ie] == 0)
        for nix, niy in zip(*neg_cells, strict=True):
            strata.setdefault((year, int(tercile[nix, niy])), []).append(
                (int(nix), int(niy), int(ie))
            )
    n_neg_candidates = sum(len(v) for v in strata.values())
    target_total = NEG_RATIO * max(n_pos_trainval, 1)
    for key in sorted(strata):
        members = strata[key]
        n_take = min(len(members), max(1, round(target_total * len(members) / n_neg_candidates)))
        take = rng.choice(len(members), size=n_take, replace=False)
        w = len(members) / n_take
        for j in sorted(int(i) for i in take):
            rows.append(members[j])
            weights.append(w)

    order = np.lexsort(
        (
            np.array([r[2] for r in rows]),
            np.array([r[1] for r in rows]),
            np.array([r[0] for r in rows]),
        )
    )
    rows = [rows[i] for i in order]
    weight = np.asarray(weights, dtype=np.float64)[order]
    if ctx.limit_samples is not None:
        rows = rows[: ctx.limit_samples]
        weight = weight[: ctx.limit_samples]

    n = len(rows)
    features = np.zeros((n, EQ_FEATURE_DIM), dtype=np.float32)
    label = np.zeros(n, dtype=np.float32)
    years = np.zeros(n, dtype=np.int64)
    cell_ix = np.zeros(n, dtype=np.int64)
    cell_iy = np.zeros(n, dtype=np.int64)
    epoch_day = np.zeros(n, dtype=np.float64)
    mag_target = np.zeros(n, dtype=np.float32)
    time_target = np.zeros(n, dtype=np.float32)
    time_mask = np.zeros(n, dtype=np.float32)
    for r, (cix, ciy, cie) in enumerate(rows):
        t = float(epochs[cie])
        features[r] = build_feature_vector(index, cix, ciy, t)
        label[r] = float(labels_grid[cix, ciy, cie])
        years[r] = epoch_years[cie]
        cell_ix[r], cell_iy[r], epoch_day[r] = cix, ciy, t
        if label[r] > 0:
            c = index.cell(cix, ciy)
            j0 = int(np.searchsorted(c.t, t, side="right"))
            j1 = int(np.searchsorted(c.t, t + LABEL_WINDOW_DAYS, side="right"))
            mag_target[r] = float(c.mag[j0:j1].max()) / 9.0 if j1 > j0 else 0.0
        m4 = index.cell_m4_times(cix, ciy)
        lo4 = int(np.searchsorted(m4, t, side="right"))
        hi4 = int(np.searchsorted(m4, t + LABEL_WINDOW_DAYS, side="right"))
        if hi4 > lo4:
            time_mask[r] = 1.0
            time_target[r] = float(np.clip((m4[lo4] - t) / LABEL_WINDOW_DAYS, 0.0, 1.0))
        if (r + 1) % 20000 == 0:
            logger.info("feature build: %d / %d rows", r + 1, n)

    tr_mask, _, _ = SPLIT.masks(years)
    if not tr_mask.any():
        raise RuntimeError("no training rows in dataset; cannot standardize honestly")
    mean = features[tr_mask].mean(axis=0)
    std = features[tr_mask].std(axis=0)
    std[std < 1e-6] = 1.0

    n_grid_trainval = int(trainval_e.sum()) * N_LAT * N_LON
    n_grid_test = int(test_mask_e.sum()) * N_LAT * N_LON
    meta = {
        "n_rows": n,
        "n_events_catalog": int(catalog.t_days.size),
        "n_epochs": int(n_epochs),
        "n_cells": N_LAT * N_LON,
        "n_positives_total": int(labels_grid.sum()),
        "n_positives_trainval": n_pos_trainval,
        "n_positives_test": int(np.sum(labels_grid[:, :, test_mask_e])),
        "base_rate_trainval_true": float(np.sum(labels_grid[:, :, trainval_e]) / n_grid_trainval),
        "base_rate_test_true": float(np.sum(labels_grid[:, :, test_mask_e]) / n_grid_test),
        "neg_sample_fraction_trainval": float(
            (target_total if n_neg_candidates else 0) / max(n_neg_candidates, 1)
        ),
    }
    logger.info("dataset meta: %s", meta)

    ds = EarthquakeDataset(
        features=features,
        label=label,
        weight=weight.astype(np.float64),
        years=years,
        cell_ix=cell_ix,
        cell_iy=cell_iy,
        epoch_day=epoch_day,
        mag_target=mag_target,
        time_target=time_target,
        time_mask=time_mask,
        feature_mean=mean.astype(np.float32),
        feature_std=std.astype(np.float32),
        meta=meta,
    )
    if ctx.limit_samples is None:
        np.savez_compressed(
            cache_path,
            features=features,
            label=label,
            weight=ds.weight,
            years=years,
            cell_ix=cell_ix,
            cell_iy=cell_iy,
            epoch_day=epoch_day,
            mag_target=mag_target,
            time_target=time_target,
            time_mask=time_mask,
            feature_mean=ds.feature_mean,
            feature_std=ds.feature_std,
            meta_json=json.dumps(meta),
        )
        logger.info("dataset cached: %s", cache_path)
    return ds


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def _weighted_log_loss(labels: np.ndarray, probs: np.ndarray, weights: np.ndarray) -> float:
    """Weighted binary log-loss (nats) with the shared probability clamp."""
    p = np.clip(np.asarray(probs, dtype=np.float64), _P_CLAMP, 1.0 - _P_CLAMP)
    y = np.asarray(labels, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    ll = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(np.sum(w * ll) / np.sum(w))


def train(ctx: PipelineContext) -> dict[str, Any]:
    """Train the analyzer; early stopping on weighted validation log-loss.

    The confidence head is the PRIMARY output -- P(M>=5.0 in-cell within
    30 days) under weighted BCE (inverse-sampling weights restore full-grid
    calibration). The magnitude and time heads are trained as DIAGNOSTIC
    regressions of observables (observed window max magnitude / 9 on
    positives; days to first in-cell M>=4 / 30 where one occurs) and are
    never event predictions -- the review forbids that claim.

    Returns:
        Training record (epochs run, best val log-loss, sample counts).
    """
    from omni_mercury_engine.space.disaster_precursor_detector import EarthquakePrecursorAnalyzer

    rng = seed_everything(ctx.seed)
    ds = build_dataset(ctx)
    train_mask, val_mask, _ = SPLIT.masks(ds.years)
    x = (ds.features - ds.feature_mean) / ds.feature_std
    xt = torch.from_numpy(x[train_mask])
    yt = torch.from_numpy(ds.label[train_mask])
    wt = torch.from_numpy(ds.weight[train_mask].astype(np.float32))
    mag_t = torch.from_numpy(ds.mag_target[train_mask])
    time_t = torch.from_numpy(ds.time_target[train_mask])
    tmask_t = torch.from_numpy(ds.time_mask[train_mask])
    xv = torch.from_numpy(x[val_mask])

    model = EarthquakePrecursorAnalyzer(input_dim=EQ_FEATURE_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    logger.info(
        "training on %d rows (%d positives), validating on %d rows (%d positives)",
        int(train_mask.sum()),
        int(ds.label[train_mask].sum()),
        int(val_mask.sum()),
        int(ds.label[val_mask].sum()),
    )

    batch_size = 256
    best_val_ll = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience, bad_epochs = 8, 0
    epochs_run = 0

    for epoch in range(ctx.max_epochs):
        epochs_run = epoch + 1
        model.train()
        perm = torch.from_numpy(rng.permutation(xt.shape[0]))
        epoch_loss = 0.0
        for start in range(0, xt.shape[0], batch_size):
            bidx = perm[start : start + batch_size]
            if bidx.shape[0] < 2:
                continue  # BatchNorm needs >1 sample
            mag_p, time_p, conf_p = model(xt[bidx])
            conf_p = conf_p.squeeze(-1).clamp(1e-6, 1 - 1e-6)
            yb, wb = yt[bidx], wt[bidx]
            bce = -(yb * conf_p.log() + (1 - yb) * (1 - conf_p).log())
            loss = (wb * bce).sum() / wb.sum()
            pos = yb > 0.5
            if pos.any():  # diagnostic magnitude head, positives only
                loss = loss + 0.25 * torch.nn.functional.mse_loss(
                    mag_p.squeeze(-1)[pos], mag_t[bidx][pos]
                )
            tm = tmask_t[bidx] > 0.5
            if tm.any():  # diagnostic time head, only where an M>=4 occurs
                loss = loss + 0.25 * torch.nn.functional.mse_loss(
                    time_p.squeeze(-1)[tm], time_t[bidx][tm]
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * bidx.shape[0]

        model.eval()
        with torch.no_grad():
            _, _, conf_v = model(xv)
        val_ll = _weighted_log_loss(
            ds.label[val_mask], conf_v.squeeze(-1).numpy(), ds.weight[val_mask]
        )
        logger.info(
            "epoch %d: train loss %.5f, weighted val log-loss %.6f",
            epoch + 1,
            epoch_loss / xt.shape[0],
            val_ll,
        )
        if val_ll < best_val_ll - 1e-7:
            best_val_ll = val_ll
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.info("early stop at epoch %d (patience %d)", epoch + 1, patience)
                break

    if best_state is None:
        raise RuntimeError("training produced no finite validation log-loss; refusing to save")
    model.load_state_dict(best_state)

    record = {
        "seed": ctx.seed,
        "epochs_run": epochs_run,
        "best_val_log_loss_weighted": best_val_ll,
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "train_years": list(SPLIT.train_years),
        "val_years": list(SPLIT.val_years),
        "train_positives": int(ds.label[train_mask].sum()),
        "val_positives": int(ds.label[val_mask].sum()),
    }
    payload: dict[str, Any] = {
        "earthquake_analyzer": model.state_dict(),
        "feature_spec": FEATURE_SPEC_VERSION,
        "feature_names": list(EQ_FEATURE_NAMES),
        "grid": "0.5deg CA",
        "label": "P(M>=5.0, 30d)",
        "feature_mean": ds.feature_mean.tolist(),
        "feature_std": ds.feature_std.tolist(),
    }
    save_candidate(ctx.data_dir, HOOK_NAME, payload, record)
    return record


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def _reliability(labels: np.ndarray, probs: np.ndarray) -> tuple[float, list[dict[str, float]]]:
    """Calibration summary: expected calibration error + per-bin table."""
    edges = np.array([0.0, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0])
    table: list[dict[str, float]] = []
    ece = 0.0
    n_total = labels.size
    for lo, hi in itertools.pairwise(edges):
        m = (probs >= lo) & (probs < hi) if hi < 1.0 else (probs >= lo) & (probs <= hi)
        if not m.any():
            continue
        mean_p = float(probs[m].mean())
        frac = float(labels[m].mean())
        table.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "n": float(m.sum()),
                "mean_predicted": mean_p,
                "observed_frequency": frac,
            }
        )
        ece += (m.sum() / n_total) * abs(mean_p - frac)
    return float(ece), table


def _aftershock_dominance(index: CatalogIndex, ds: EarthquakeDataset, mask: np.ndarray) -> float:
    """Fraction of positive cases whose labeling event follows a recent M>=5.

    A positive (cell, epoch) counts as aftershock-driven when its first
    in-window M>=5.0 event has a prior M>=5.0 catalog event within 45 days
    and 50 km -- the review's mandatory honesty statistic (section 4.4).
    """
    cat = index.catalog
    m5_mask = cat.mag >= LABEL_MIN_MAG
    m5_t, m5_lat, m5_lon = cat.t_days[m5_mask], cat.lat[m5_mask], cat.lon[m5_mask]
    pos_rows = np.flatnonzero(mask & (ds.label > 0.5))
    if pos_rows.size == 0:
        return float("nan")
    n_after = 0
    for r in pos_rows:
        c = index.cell(int(ds.cell_ix[r]), int(ds.cell_iy[r]))
        t = float(ds.epoch_day[r])
        lo = int(np.searchsorted(c.t, t, side="right"))
        hi = int(np.searchsorted(c.t, t + LABEL_WINDOW_DAYS, side="right"))
        win = slice(lo, hi)
        big = np.flatnonzero(c.mag[win] >= LABEL_MIN_MAG)
        if big.size == 0:
            continue
        j = lo + int(big[0])
        te, la, lo_ = float(c.t[j]), float(c.lat[j]), float(c.lon[j])
        prior = (m5_t < te) & (m5_t >= te - 45.0)
        if prior.any():
            d = _flat_distances_km(
                np.full(prior.sum(), la), np.full(prior.sum(), lo_), m5_lat[prior], m5_lon[prior]
            )
            if (d <= 50.0).any():
                n_after += 1
    return float(n_after / pos_rows.size)


def evaluate(ctx: PipelineContext) -> EvaluationOutcome:
    """Compare learned vs the Reasenberg-Jones baseline on identical cases.

    Held-out cases are the COMPLETE test-years (cell, epoch) grid (weight 1,
    no sampling). The learned path runs through the public detector API
    (``DisasterPrecursorDetector.detect_disaster_precursor`` with
    ``seismicity_features``) after loading the candidate checkpoint; the
    physics side is the documented deterministic Reasenberg-Jones/ETAS-lite
    baseline from this module -- the detector's own non-neural fallback
    abstains from event probabilities, which ``EvaluationOutcome.physics``
    explicitly permits substituting. A bare-Poisson floor is reported in
    extras for context only.

    Returns:
        The evaluation outcome (primary metric: log_loss, lower is better).
    """
    from omni_mercury_engine.space.disaster_precursor_detector import DisasterPrecursorDetector

    ds = build_dataset(ctx)
    catalog, _paths = _load_catalog(ctx)
    index = CatalogIndex(catalog)
    mu = background_mu(index)

    _, _, test_mask = SPLIT.masks(ds.years)
    test_idx = np.flatnonzero(test_mask)
    if test_idx.size == 0:
        raise RuntimeError("no test rows found; cannot evaluate")

    cand_path, _ = candidate_paths(ctx.data_dir, HOOK_NAME)
    if not cand_path.exists():
        raise FileNotFoundError(f"no candidate checkpoint at {cand_path}; run --train first")

    detector = DisasterPrecursorDetector(enable_tsunami=False, enable_geomagnetic=False)
    detector.load_neural_weights(str(cand_path))
    det_logger = logging.getLogger("omni_mercury_engine.space.disaster_precursor_detector")
    old_level = det_logger.level
    det_logger.setLevel(logging.WARNING)  # 100k+ per-case INFO lines otherwise
    try:
        learned_p = np.empty(test_idx.size, dtype=np.float64)
        for k, i in enumerate(test_idx):
            result = detector.detect_disaster_precursor({"seismicity_features": ds.features[i]})
            learned_p[k] = float(result.confidence)
    finally:
        det_logger.setLevel(old_level)

    physics_p = np.empty(test_idx.size, dtype=np.float64)
    poisson_p = np.empty(test_idx.size, dtype=np.float64)
    for k, i in enumerate(test_idx):
        ix, iy = int(ds.cell_ix[i]), int(ds.cell_iy[i])
        cell = index.cell(ix, iy)
        physics_p[k] = rj_probability(cell, float(ds.epoch_day[i]), float(mu[ix, iy]))
        poisson_p[k] = poisson_probability(float(mu[ix, iy]))

    y = ds.label[test_idx]
    w = np.ones_like(y, dtype=np.float64)  # full grid: no sampling weights
    n_pos = int(y.sum())

    def _ll_sum(probs: np.ndarray) -> float:
        p = np.clip(probs, _P_CLAMP, 1.0 - _P_CLAMP)
        return float(np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    poisson_ll_sum = _ll_sum(poisson_p)
    poisson_metrics = {
        "log_loss": _weighted_log_loss(y, poisson_p, w),
        "auc": binary_auc(y, poisson_p),
        "brier": brier_score(y, poisson_p),
    }

    def _metrics(probs: np.ndarray) -> dict[str, float]:
        ece, _table = _reliability(y, probs)
        return {
            "log_loss": _weighted_log_loss(y, probs, w),
            "auc": binary_auc(y, probs),
            "brier": brier_score(y, probs),
            "information_gain_per_active_cell": float(
                (_ll_sum(probs) - poisson_ll_sum) / max(n_pos, 1)
            ),
            "reliability_ece": ece,
        }

    learned_metrics = _metrics(learned_p)
    physics_metrics = _metrics(physics_p)
    _, learned_table = _reliability(y, learned_p)
    _, physics_table = _reliability(y, physics_p)
    aftershock_frac = _aftershock_dominance(index, ds, test_mask)

    outcome = EvaluationOutcome(
        hook=HOOK_NAME,
        primary_metric="log_loss",
        higher_is_better=False,
        learned=learned_metrics,
        physics=physics_metrics,
        n_test_samples=int(test_idx.size),
        test_years=SPLIT.test_years,
        extras={
            "comparison": (
                "identical held-out (cell, epoch) cases -- the complete test-years grid. "
                "Learned path runs through the public "
                "DisasterPrecursorDetector.detect_disaster_precursor API with the candidate "
                "checkpoint loaded. The 'physics' side is the documented deterministic "
                "Reasenberg-Jones/ETAS-lite clustering baseline (generic California "
                "parameters a=-1.67, b=0.91, c=0.05 d, p=1.08 + train-years Laplace-smoothed "
                "Poisson background), standing in for the detector's non-neural fallback, "
                "which abstains from event probabilities (allowed by EvaluationOutcome.physics)."
            ),
            "aftershock_dominance_fraction_test_positives": aftershock_frac,
            "aftershock_dominance_note": (
                "Fraction of held-out positives whose labeling M>=5.0 event has a prior "
                "M>=5.0 within 45 days / 50 km. Skill on these cases is honest ETAS-style "
                "clustering (Omori aftershock triggering), NOT novel precursor detection; "
                "the review (section 4.4) requires this statement wherever results are shown."
            ),
            "test_base_rate": float(y.mean()),
            "n_test_positives": n_pos,
            "test_m5_events": int(
                np.sum(
                    (catalog.mag >= LABEL_MIN_MAG)
                    & (
                        catalog.t_days
                        >= float(
                            (_dt.datetime(min(SPLIT.test_years), 1, 1, tzinfo=_dt.UTC) - DAY0).days
                        )
                    )
                )
            ),
            "poisson_floor": poisson_metrics,
            "poisson_floor_note": (
                "Bare time-independent Poisson from train-years cell rates; sanity floor "
                "only -- beating it merely rediscovers Omori-law clustering and does NOT "
                "satisfy the merit gate."
            ),
            "reliability_learned": learned_table,
            "reliability_physics": physics_table,
            "probability_clamp": _P_CLAMP,
            "ridgecrest_note": (
                "The 2019 Ridgecrest sequence (M6.4 + M7.1) falls in the held-out test "
                "years (2017-2024)."
            ),
            "dataset_meta": ds.meta,
        },
    )
    save_evaluation(ctx.data_dir, outcome)
    logger.info(
        "evaluation: learned log_loss %.6f vs Reasenberg-Jones %.6f (Poisson floor %.6f) "
        "on %d held-out cases (%s)",
        learned_metrics["log_loss"],
        physics_metrics["log_loss"],
        poisson_metrics["log_loss"],
        outcome.n_test_samples,
        "LEARNED WINS" if outcome.learned_beats_physics else "PHYSICS BASELINE WINS",
    )
    return outcome


def ship(ctx: PipelineContext) -> tuple[Any, Any]:
    """Promote the candidate through the merit gate (may refuse loudly)."""
    from omni_mercury_engine.ml.hazard_training.common import load_evaluation

    outcome = load_evaluation(ctx.data_dir, HOOK_NAME)
    manifest_path = ctx.data_dir / "usgs_catalog" / "manifest.json"
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
