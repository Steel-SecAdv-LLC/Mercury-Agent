# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Train the hurricane WindPatternAnalyzer on real ERA5 winds + IBTrACS labels.

Data sources (hook ``hurricane_wind``,
``HurricaneDetector.load_neural_weights``):

* **IBTrACS v04r01 best-track archive** (NOAA NCEI, public) -- the official
  WMO consolidated tropical-cyclone record. Provides storm positions and the
  USA-agency maximum sustained wind (``USA_WIND``, knots) plus the
  Saffir-Simpson class (``USA_SSHS``) at every 6-hourly synoptic time. These
  are the intensity/category labels.
* **ARCO-ERA5** (``gcp-public-data-arco-era5`` on ``storage.googleapis.com``,
  public) -- the ECMWF ERA5 reanalysis mirrored as a zarr-format-2 store
  (the path says ``.zarr-v3``; the actual on-disk format, verified from the
  per-array ``.zarray`` metadata at fetch time, is zarr format 2). The
  ``10m_u_component_of_wind`` / ``10m_v_component_of_wind`` arrays hold one
  global 721x1440 float32 field per hour (chunks ``[1, 721, 1440]``,
  blosc/lz4, ~3.3 MB each), indexed by hours since 1900-01-01T00. Raw chunk
  GETs are decoded with :mod:`numcodecs` -- no zarr client dependency.

Dataset design (chunk-sharing bounds the network budget): a seeded,
year+month-stratified selection of ~595 distinct synoptic hours spanning
1990-2024 (one hour per calendar month plus one extra in the peak months
1, 2, 8, 9, 10 so both hemispheres' seasons and the off-season are covered).
Each selected hour costs exactly 4 chunk GETs (u and v at t-6h and t); every
sample at that hour -- every active IBTrACS main-track point (positives) and
3 seeded random far-from-storm locations (negatives) -- is a 33x33-point
(8.25 deg) patch cut from those SAME four global fields. Only the extracted
patches are cached (npz per year, sha256-pinned in the manifest); raw chunks
never touch disk.

Honesty notes, recorded here because they are label-defining:

* Negatives are drawn at |lat| <= 60 deg, >= 1000 km from every active
  best-track system at that hour. There is no land mask in this environment
  and none is fabricated, so some negatives fall on land; that is acceptable
  because the negative labels do not assume ocean: the intensity label is the
  patch's own observed ERA5 maximum 10 m wind (true by construction anywhere)
  and the category label is ``no_cyclone`` (true anywhere >= 1000 km from
  every tracked system). Winter storms and monsoon winds captured this way
  are honest hard negatives, not noise.
* The intensity target is in **knots**, the detector's native
  ``max_wind_speed_kt`` unit: ``USA_WIND`` for positives; for negatives the
  patch's own ERA5 max 10 m wind converted to kt -- that IS the observed
  maximum wind for a no-cyclone patch. The network's intensity head is
  therefore trained and served in the same unit with no rescaling
  (train/serve parity; feature spec ``hurricane-era5-v1``).
* ERA5 0.25-deg patches systematically under-resolve tropical-cyclone peak
  winds, so the physics fallback (observed max wind in the patch) is a
  structural under-estimate for strong storms. Beating it requires the
  network to infer unresolved intensity from the resolved wind structure --
  that is the merit-gate task, evaluated through the public
  :meth:`HurricaneDetector.predict_hurricane` API on identical held-out
  cases.

Temporal split (never random -- storm seasons autocorrelate and reanalysis
quality drifts across years): train 1990-2015, validation 2016-2019, test
2020-2024, split on the sample's own synoptic-time year.
"""

from __future__ import annotations

import calendar
import datetime as _dt
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from omni_mercury_engine.datasets.base import http_get_with_retry
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

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

HOOK_NAME = "hurricane_wind"
CHECKPOINT_NAME = "hurricane_era5"
FEATURE_SPEC_VERSION = "hurricane-era5-v1"

IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-"
    "climate-stewardship-ibtracs/v04r01/access/csv/"
    "ibtracs.since1980.list.v04r01.csv"
)
ARCO_ERA5_BASE = (
    "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/"
    "full_37-1h-0p25deg-chunk-1.zarr-v3"
)
ERA5_U_VAR = "10m_u_component_of_wind"
ERA5_V_VAR = "10m_v_component_of_wind"
ERA5_GRID_SHAPE = (721, 1440)

#: Same m/s -> kt factor the detector's physics fallback uses.
MS_TO_KT = 1.9438

PATCH_HALF = 16  # 33x33 grid points = 8.25 deg at 0.25 deg spacing
PATCH_SIZE = 2 * PATCH_HALF + 1
PATCH_DEG = (PATCH_SIZE - 1) * 0.25
#: Meridional 0.25-deg spacing in metres (constant; zonal spacing shrinks
#: with latitude -- both evaluation paths receive this same value, and the
#: patch max wind, the quantity under test, does not depend on it).
GRID_SPACING_M = 27750.0
SEQ_OFFSET_HOURS = (-6, 0)
CHANNELS = ("u10", "v10", "speed")

SYNOPTIC_HOURS = (0, 6, 12, 18)
#: Months that get one extra sampled hour: Aug-Oct covers the N-hemisphere
#: peak, Jan-Feb the S-hemisphere peak; the remaining months keep the
#: off-season represented at one hour each.
PEAK_MONTHS = (1, 2, 8, 9, 10)
NEGATIVES_PER_HOUR = 3
MIN_NEG_DISTANCE_KM = 1000.0
NEG_MAX_ABS_LAT = 60.0

SPLIT = TemporalSplit(
    train_years=tuple(range(1990, 2016)),
    val_years=(2016, 2017, 2018, 2019),
    test_years=(2020, 2021, 2022, 2023, 2024),
)

_EPOCH = _dt.datetime(1900, 1, 1)
#: kt threshold for the hurricane-detection AUC (Saffir-Simpson category 1).
HURRICANE_WIND_KT = 64.0

#: USA_SSHS -> index into the detector's NEURAL_CATEGORY_ORDER. SSHS -1 is a
#: tropical depression, 0 a tropical storm, 1..5 the hurricane categories.
#: Points whose SSHS is outside this table (subtropical -2, disturbances -3,
#: extratropical -4, unknown -5) are not classified tropical systems at that
#: time and are excluded from the positives rather than mislabeled.
SSHS_TO_CATEGORY_INDEX: dict[int, int] = {-1: 1, 0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7}

#: Scale (kt) that normalizes the intensity MSE so it is commensurate with
#: the category cross-entropy at the start of training.
INTENSITY_SCALE_KT = 50.0

_IBTRACS_COLUMNS = [
    "SID",
    "SEASON",
    "BASIN",
    "ISO_TIME",
    "LAT",
    "LON",
    "USA_WIND",
    "USA_PRES",
    "USA_SSHS",
    "WMO_WIND",
    "TRACK_TYPE",
]

#: IBTrACS basin code -> the detector's ``basin`` vocabulary.
_BASIN_TO_DETECTOR = {
    "NA": "atlantic",
    "SA": "atlantic",
    "EP": "eastern_pacific",
    "CP": "central_pacific",
    "WP": "western_pacific",
    "NI": "indian",
    "SI": "indian",
    "SP": "south_pacific",
}


def _category_order() -> tuple[str, ...]:
    """The 8-class category vocabulary, from its single source of truth.

    The detector owns ``NEURAL_CATEGORY_ORDER`` (its learned path decodes the
    category head with it); training imports it lazily so importing this
    pipeline module stays light.
    """
    from omni_mercury_engine.detectors.geological.hurricane_detector import (
        NEURAL_CATEGORY_ORDER,
    )

    order: tuple[str, ...] = NEURAL_CATEGORY_ORDER
    return order


def _hours_since_epoch(t: _dt.datetime) -> int:
    """Whole hours between ``t`` and the store's 1900-01-01T00 epoch."""
    return int((t - _EPOCH).total_seconds() // 3600)


def _chunk_url(var: str, time_index: int) -> str:
    """URL of the single global-field chunk for ``var`` at ``time_index``."""
    return f"{ARCO_ERA5_BASE}/{var}/{time_index}.0.0"


def _decode_chunk(raw: bytes) -> np.ndarray[Any, Any]:
    """Blosc-decode one raw chunk into a (721, 1440) float32 field."""
    import numcodecs  # type: ignore[import-untyped]

    payload = numcodecs.Blosc().decode(raw)
    expected = ERA5_GRID_SHAPE[0] * ERA5_GRID_SHAPE[1] * 4
    if len(payload) != expected:
        raise RuntimeError(
            f"ERA5 chunk decoded to {len(payload)} bytes, expected {expected}; "
            "the store layout changed -- refusing to guess"
        )
    return np.frombuffer(payload, dtype="<f4").reshape(ERA5_GRID_SHAPE)


def _fetch_chunk(var: str, time_index: int) -> tuple[np.ndarray[Any, Any], int]:
    """Fetch + decode one hourly global field; returns (field, bytes fetched)."""
    raw = http_get_with_retry(_chunk_url(var, time_index), timeout=180.0)
    field = _decode_chunk(raw)
    if not np.isfinite(field).all():
        raise RuntimeError(
            f"ERA5 chunk {var}/{time_index} contains non-finite values; "
            "refusing to train on a corrupted field"
        )
    return field, len(raw)


def _verify_store() -> dict[str, Any]:
    """Verify the ARCO-ERA5 store layout and grid orientation, fail loud.

    Checks the per-array ``.zarray``/``.zattrs`` metadata (equivalent to the
    consolidated ``.zmetadata``, which spans hundreds of arrays and is not
    needed) and decodes the actual latitude/longitude coordinate chunks so
    the grid orientation is read from the store itself, never assumed.

    Returns:
        A manifest-ready description of the verified store.
    """
    import numcodecs

    blosc = numcodecs.Blosc()
    arrays: dict[str, Any] = {}
    for var in (ERA5_U_VAR, ERA5_V_VAR):
        meta = json.loads(http_get_with_retry(f"{ARCO_ERA5_BASE}/{var}/.zarray", timeout=60.0))
        if (
            meta.get("zarr_format") != 2
            or meta.get("dtype") != "<f4"
            or meta.get("chunks") != [1, *ERA5_GRID_SHAPE]
            or meta.get("shape", [None, None, None])[1:] != list(ERA5_GRID_SHAPE)
            or (meta.get("compressor") or {}).get("id") != "blosc"
        ):
            raise RuntimeError(
                f"ARCO-ERA5 array {var} no longer matches the verified layout "
                f"(got {meta}); refusing to decode chunks blind"
            )
        arrays[var] = {"shape": meta["shape"], "chunks": meta["chunks"], "dtype": meta["dtype"]}

    time_attrs = json.loads(http_get_with_retry(f"{ARCO_ERA5_BASE}/time/.zattrs", timeout=60.0))
    if time_attrs.get("units") != "hours since 1900-01-01 00:00:00":
        raise RuntimeError(
            f"ARCO-ERA5 time units changed to {time_attrs.get('units')!r}; "
            "chunk-index arithmetic would be wrong -- refusing"
        )

    lat = np.frombuffer(
        blosc.decode(http_get_with_retry(f"{ARCO_ERA5_BASE}/latitude/0", timeout=60.0)),
        dtype="<f4",
    )
    lon = np.frombuffer(
        blosc.decode(http_get_with_retry(f"{ARCO_ERA5_BASE}/longitude/0", timeout=60.0)),
        dtype="<f4",
    )
    if lat.shape != (721,) or lat[0] != 90.0 or lat[-1] != -90.0:
        raise RuntimeError(f"unexpected latitude coordinate ({lat[:3]}..{lat[-3:]}); refusing")
    if lon.shape != (1440,) or lon[0] != 0.0 or float(lon[-1]) != 359.75:
        raise RuntimeError(f"unexpected longitude coordinate ({lon[:3]}..{lon[-3:]}); refusing")

    return {
        "base": ARCO_ERA5_BASE,
        "zarr_format": 2,
        "arrays": arrays,
        "time_units": time_attrs["units"],
        "latitude": "90.0 .. -90.0 step -0.25 (descending, verified from store)",
        "longitude": "0.0 .. 359.75 step 0.25 (verified from store)",
    }


def _probe_availability() -> dict[str, Any]:
    """Probe the store's usable time range around this pipeline's needs.

    The declared array shape extends decades past the present; chunks past
    the ERA5 production frontier simply 404. The earliest and latest hours
    this pipeline reads must exist (fail loud otherwise); a far-future chunk
    is probed and *expected* missing, which is recorded, not raised.
    """
    import requests

    def _exists(t: _dt.datetime) -> bool:
        try:
            http_get_with_retry(_chunk_url(ERA5_U_VAR, _hours_since_epoch(t)), timeout=60.0)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return False
            raise
        return True

    earliest = _dt.datetime(1990, 1, 1, 0) + _dt.timedelta(hours=min(SEQ_OFFSET_HOURS))
    latest = _dt.datetime(max(SPLIT.test_years), 12, 31, 18)
    future = _dt.datetime.now(tz=_dt.UTC).replace(tzinfo=None) + _dt.timedelta(days=60)
    result = {
        "earliest_needed": {"time": earliest.isoformat(), "available": _exists(earliest)},
        "latest_needed": {"time": latest.isoformat(), "available": _exists(latest)},
        "future_probe": {"time": future.isoformat(), "available": _exists(future)},
    }
    if not (result["earliest_needed"]["available"] and result["latest_needed"]["available"]):
        raise RuntimeError(f"ERA5 store does not cover the split years: {result}")
    return result


@dataclass
class _SynopticPoints:
    """IBTrACS main-track synoptic points, parsed into flat arrays."""

    sid: np.ndarray[Any, Any]
    basin: np.ndarray[Any, Any]
    time_h: np.ndarray[Any, Any]
    lat: np.ndarray[Any, Any]
    lon: np.ndarray[Any, Any]
    usa_wind: np.ndarray[Any, Any]
    usa_sshs: np.ndarray[Any, Any]


def _load_ibtracs(path: Path) -> _SynopticPoints:
    """Parse the IBTrACS CSV into synoptic main-track points for the split.

    IBTrACS ships two header rows (names + units; the units row is skipped)
    and blank-string missing values. Only ``TRACK_TYPE == 'main'`` rows on
    the 00/06/12/18Z synoptic hours within the split years are kept.

    Raises:
        RuntimeError: If the parsed archive is implausibly small or carries
            out-of-range coordinates (a format change -- fail loud).
    """
    import pandas as pd

    df = pd.read_csv(
        path,
        usecols=_IBTRACS_COLUMNS,
        skiprows=[1],
        dtype=str,
        keep_default_na=False,
        na_values=["", " "],
        low_memory=False,
    )
    df = df[df["TRACK_TYPE"] == "main"]
    times = pd.to_datetime(df["ISO_TIME"], format="%Y-%m-%d %H:%M:%S")
    synoptic = times.dt.hour.isin(SYNOPTIC_HOURS) & (times.dt.minute == 0)
    in_years = times.dt.year.isin(SPLIT.all_years)
    df = df[synoptic & in_years]
    times = times[synoptic & in_years]

    lat = pd.to_numeric(df["LAT"], errors="coerce").to_numpy(dtype=np.float64)
    lon = pd.to_numeric(df["LON"], errors="coerce").to_numpy(dtype=np.float64)
    usa_wind = pd.to_numeric(df["USA_WIND"], errors="coerce").to_numpy(dtype=np.float64)
    usa_sshs = pd.to_numeric(df["USA_SSHS"], errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(lat) & np.isfinite(lon)
    if int(ok.sum()) < 100_000:
        raise RuntimeError(
            f"IBTrACS parse yielded only {int(ok.sum())} usable synoptic points; "
            "expected >100k for 1990-2024 -- the format assumption is wrong"
        )
    # IBTrACS keeps tracks continuous across the antimeridian, so LON runs
    # past +180 (observed max ~267 deg) for Pacific crossers; anything in
    # (-180, 360) maps onto the grid via ``lon % 360``.
    if (
        np.nanmax(np.abs(lat[ok])) > 90.0
        or np.nanmin(lon[ok]) < -180.0
        or np.nanmax(lon[ok]) >= 360.0
    ):
        raise RuntimeError("IBTrACS LAT/LON out of documented range; refusing to continue")

    epoch = pd.Timestamp(_EPOCH)
    time_h = ((times - epoch) // pd.Timedelta(hours=1)).to_numpy(dtype=np.int64)
    return _SynopticPoints(
        sid=df["SID"].to_numpy(dtype=object)[ok],
        basin=df["BASIN"].to_numpy(dtype=object)[ok],
        time_h=time_h[ok],
        lat=lat[ok],
        lon=lon[ok],
        usa_wind=usa_wind[ok],
        usa_sshs=usa_sshs[ok],
    )


def _select_hours_for_year(year: int, seed: int) -> list[int]:
    """Seeded, month-stratified synoptic-hour selection for one year.

    One hour per calendar month plus one extra in :data:`PEAK_MONTHS`. The
    per-year RNG stream makes the selection independent of processing order
    (resumable fetch).
    """
    rng = np.random.default_rng([seed, year])
    chosen: list[int] = []
    for month in range(1, 13):
        n_days = calendar.monthrange(year, month)[1]
        candidates = [
            _hours_since_epoch(_dt.datetime(year, month, day, hour))
            for day in range(1, n_days + 1)
            for hour in SYNOPTIC_HOURS
        ]
        n_pick = 2 if month in PEAK_MONTHS else 1
        picks = rng.choice(len(candidates), size=n_pick, replace=False)
        chosen.extend(candidates[int(i)] for i in picks)
    return sorted(set(chosen))


def _haversine_km(
    lat1: float, lon1: float, lat2: np.ndarray[Any, Any], lon2: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """Great-circle distances (km) from one point to arrays of points."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(((lon2 - lon1) + 180.0) % 360.0 - 180.0)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2.0) ** 2
    out: np.ndarray[Any, Any] = 6371.0 * 2.0 * np.arcsin(np.sqrt(a))
    return out


def _extract_patch(
    fields: dict[str, dict[int, np.ndarray[Any, Any]]],
    time_indices: list[int],
    lat: float,
    lon: float,
) -> np.ndarray[Any, Any] | None:
    """Cut one [T, 3, 33, 33] (u, v, speed) patch from cached global fields.

    Returns None when the requested centre is within 4.25 deg of a pole (the
    patch would leave the grid; such samples are discarded, never padded).
    Longitude wraps; latitude row 0 is +90 deg (verified in fetch).
    """
    lat_idx = round((90.0 - lat) / 0.25)
    lon_idx = round((lon % 360.0) / 0.25) % ERA5_GRID_SHAPE[1]
    if lat_idx - PATCH_HALF < 0 or lat_idx + PATCH_HALF >= ERA5_GRID_SHAPE[0]:
        return None
    rows = np.arange(lat_idx - PATCH_HALF, lat_idx + PATCH_HALF + 1)
    cols = np.arange(lon_idx - PATCH_HALF, lon_idx + PATCH_HALF + 1) % ERA5_GRID_SHAPE[1]
    patch = np.empty((len(time_indices), len(CHANNELS), PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    for t_i, th in enumerate(time_indices):
        u = fields[ERA5_U_VAR][th][np.ix_(rows, cols)]
        v = fields[ERA5_V_VAR][th][np.ix_(rows, cols)]
        patch[t_i, 0] = u
        patch[t_i, 1] = v
        patch[t_i, 2] = np.hypot(u, v)
    return patch


def _extract_hour(
    time_h: int, points: _SynopticPoints, seed: int
) -> tuple[list[dict[str, Any]], int, int]:
    """Fetch one selected hour's 4 chunks and cut every sample patch from them.

    Returns:
        (samples, chunk_gets, bytes_fetched). Each sample dict carries the
        patch tensor plus its label/metadata fields.
    """
    time_indices = [time_h + off for off in SEQ_OFFSET_HOURS]
    fields: dict[str, dict[int, np.ndarray[Any, Any]]] = {ERA5_U_VAR: {}, ERA5_V_VAR: {}}
    gets = 0
    nbytes = 0
    for var in (ERA5_U_VAR, ERA5_V_VAR):
        for th in time_indices:
            fields[var][th], b = _fetch_chunk(var, th)
            gets += 1
            nbytes += b

    at_t = points.time_h == time_h
    samples: list[dict[str, Any]] = []

    # Positives: every active main-track point with a finite USA wind and a
    # defined tropical-system class at hour t.
    pos_mask = at_t & np.isfinite(points.usa_wind) & (points.usa_wind > 0)
    for i in np.flatnonzero(pos_mask):
        sshs = points.usa_sshs[i]
        if not np.isfinite(sshs) or int(sshs) not in SSHS_TO_CATEGORY_INDEX:
            continue
        patch = _extract_patch(fields, time_indices, float(points.lat[i]), float(points.lon[i]))
        if patch is None:
            continue
        samples.append(
            {
                "x": patch,
                "intensity_kt": float(points.usa_wind[i]),
                "category_idx": SSHS_TO_CATEGORY_INDEX[int(sshs)],
                "is_positive": True,
                "lat": float(points.lat[i]),
                "lon": float(points.lon[i]),
                "time_h": time_h,
                "basin": str(points.basin[i]),
                "sid": str(points.sid[i]),
                "usa_sshs": float(sshs),
            }
        )

    # Negatives: seeded random locations >= MIN_NEG_DISTANCE_KM from every
    # active system (any main-track point at t, tropical-classified or not).
    active_lat = points.lat[at_t]
    active_lon = points.lon[at_t]
    rng = np.random.default_rng([seed, time_h])
    accepted = 0
    for _ in range(300):
        if accepted >= NEGATIVES_PER_HOUR:
            break
        cand_lat = float(rng.uniform(-NEG_MAX_ABS_LAT, NEG_MAX_ABS_LAT))
        cand_lon = float(rng.uniform(0.0, 360.0))
        if (
            active_lat.size
            and float(_haversine_km(cand_lat, cand_lon, active_lat, active_lon).min())
            < MIN_NEG_DISTANCE_KM
        ):
            continue
        patch = _extract_patch(fields, time_indices, cand_lat, cand_lon)
        if patch is None:
            continue
        obs_max_kt = float(patch[:, 2].max() * MS_TO_KT)
        samples.append(
            {
                "x": patch,
                # The honest no-cyclone intensity: the patch's own observed
                # ERA5 maximum 10 m wind (kt) -- exactly what the physics
                # fallback reports for the same field.
                "intensity_kt": obs_max_kt,
                "category_idx": 0,
                "is_positive": False,
                "lat": cand_lat,
                "lon": cand_lon,
                "time_h": time_h,
                "basin": "",
                "sid": "",
                "usa_sshs": float("nan"),
            }
        )
        accepted += 1
    if accepted < NEGATIVES_PER_HOUR:
        logger.warning("hour %d: only %d negatives found", time_h, accepted)
    return samples, gets, nbytes


def _patch_paths(ctx: PipelineContext, year: int) -> tuple[Path, Path]:
    """(npz, meta-json) cache paths for one year of extracted patches."""
    patch_dir = ctx.data_dir / "hurricane" / "era5_patches"
    return patch_dir / f"patches_{year}.npz", patch_dir / f"patches_{year}.meta.json"


def _save_year(
    ctx: PipelineContext, year: int, samples: list[dict[str, Any]], gets: int, nbytes: int
) -> dict[str, Any]:
    """Write one year's patches npz + meta sidecar; returns the meta dict."""
    npz_path, meta_path = _patch_paths(ctx, year)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    samples = sorted(samples, key=lambda s: (s["time_h"], not s["is_positive"], s["sid"]))
    x = np.stack([s["x"] for s in samples]).astype(np.float32)
    obs_max_kt = x[:, :, 2].reshape(x.shape[0], -1).max(axis=1) * MS_TO_KT
    np.savez_compressed(
        npz_path,
        x=x,
        intensity_kt=np.array([s["intensity_kt"] for s in samples], dtype=np.float64),
        category_idx=np.array([s["category_idx"] for s in samples], dtype=np.int64),
        is_positive=np.array([s["is_positive"] for s in samples], dtype=bool),
        obs_max_kt=obs_max_kt.astype(np.float64),
        lat=np.array([s["lat"] for s in samples], dtype=np.float64),
        lon=np.array([s["lon"] for s in samples], dtype=np.float64),
        time_h=np.array([s["time_h"] for s in samples], dtype=np.int64),
        basin=np.array([s["basin"] for s in samples], dtype=np.str_),
        sid=np.array([s["sid"] for s in samples], dtype=np.str_),
        usa_sshs=np.array([s["usa_sshs"] for s in samples], dtype=np.float64),
    )
    meta = {
        "year": year,
        "n_samples": len(samples),
        "n_positive": int(sum(1 for s in samples if s["is_positive"])),
        "n_negative": int(sum(1 for s in samples if not s["is_positive"])),
        "chunk_gets": gets,
        "bytes_fetched": nbytes,
        "sha256": sha256_file(npz_path),
        "seed": ctx.seed,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return meta


def fetch(ctx: PipelineContext) -> dict[str, Any]:
    """Fetch IBTrACS labels + stream ERA5 chunks into per-year patch caches.

    Verifies the ARCO-ERA5 store layout/orientation and its usable time
    range first, then walks the seeded hour selection year by year. Years
    whose patch cache already exists are skipped (resumable); raw chunks are
    held only in memory. Returns (and writes) the provenance manifest.
    """
    hur_dir = ctx.data_dir / "hurricane"
    ibtracs_path = cached_fetch(IBTRACS_URL, hur_dir / "ibtracs.since1980.list.v04r01.csv")
    ibtracs_sha = sha256_file(ibtracs_path)

    store = _verify_store()
    availability = _probe_availability()
    points = _load_ibtracs(ibtracs_path)
    logger.info("IBTrACS parsed: %d synoptic main-track points in split years", points.lat.size)

    per_year: dict[str, dict[str, Any]] = {}
    for year in SPLIT.all_years:
        npz_path, meta_path = _patch_paths(ctx, year)
        if npz_path.exists() and meta_path.exists():
            per_year[str(year)] = json.loads(meta_path.read_text())
            logger.info("year %d: patch cache hit (%s)", year, npz_path)
            continue
        hours = _select_hours_for_year(year, ctx.seed)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda h: _extract_hour(h, points, ctx.seed), hours))
        samples = [s for r in results for s in r[0]]
        gets = sum(r[1] for r in results)
        nbytes = sum(r[2] for r in results)
        if not samples:
            raise RuntimeError(f"year {year}: zero samples extracted; refusing to cache nothing")
        per_year[str(year)] = _save_year(ctx, year, samples, gets, nbytes)
        logger.info(
            "year %d: %d hours -> %d samples (%d pos / %d neg), %d GETs, %.1f MiB",
            year,
            len(hours),
            per_year[str(year)]["n_samples"],
            per_year[str(year)]["n_positive"],
            per_year[str(year)]["n_negative"],
            gets,
            nbytes / 2**20,
        )

    sources: list[dict[str, Any]] = [
        {
            "url": IBTRACS_URL,
            "sha256": ibtracs_sha,
            "description": "IBTrACS v04r01 since-1980 best-track CSV (labels)",
        },
        {
            "url": ARCO_ERA5_BASE,
            "sha256": "streamed-chunks (integrity: per-chunk blosc decode + shape/finite checks;"
            " derived patch caches pinned below)",
            "description": "ARCO-ERA5 10m u/v wind, raw zarr-2 chunk GETs (features)",
        },
    ]
    sources.extend(
        {
            "url": f"file:era5_patches/patches_{year}.npz",
            "sha256": meta["sha256"],
            "description": f"extracted 33x33 u/v/speed patch tensors, {year}",
        }
        for year, meta in sorted(per_year.items())
    )
    manifest: dict[str, Any] = {
        "hook": HOOK_NAME,
        "sources": sources,
        "era5_store": store,
        "era5_availability": availability,
        "per_year": per_year,
        "chunk_gets_total": int(sum(m["chunk_gets"] for m in per_year.values())),
        "bytes_fetched_total": int(sum(m["bytes_fetched"] for m in per_year.values())),
        "n_samples_total": int(sum(m["n_samples"] for m in per_year.values())),
        "seed": ctx.seed,
    }
    (hur_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    logger.info(
        "fetch complete: %d samples, %d chunk GETs, %.2f GiB streamed",
        manifest["n_samples_total"],
        manifest["chunk_gets_total"],
        manifest["bytes_fetched_total"] / 2**30,
    )
    return manifest


@dataclass
class HurricaneWindDataset:
    """Patch tensors + labels with per-sample year for temporal splitting.

    ``x`` holds raw m/s channels (u10, v10, speed) -- deliberately NOT
    standardized: the detector feeds the caller's raw wind field straight to
    the network, so training on raw fields is what guarantees train/serve
    parity. Train-year channel statistics are recorded for provenance only.
    """

    x: np.ndarray[Any, Any]
    intensity_kt: np.ndarray[Any, Any]
    category_idx: np.ndarray[Any, Any]
    is_positive: np.ndarray[Any, Any]
    obs_max_kt: np.ndarray[Any, Any]
    years: np.ndarray[Any, Any]
    basins: np.ndarray[Any, Any]
    channel_mean: np.ndarray[Any, Any]
    channel_std: np.ndarray[Any, Any]


def build_dataset(ctx: PipelineContext) -> HurricaneWindDataset:
    """Assemble the dataset from the per-year patch caches (fetch first)."""
    parts: list[dict[str, np.ndarray[Any, Any]]] = []
    for year in SPLIT.all_years:
        npz_path, _ = _patch_paths(ctx, year)
        if not npz_path.exists():
            raise FileNotFoundError(f"missing patch cache {npz_path}; run the --fetch stage first")
        with np.load(npz_path, allow_pickle=False) as z:
            part = {k: z[k] for k in z.files}
        part["years"] = np.full(part["x"].shape[0], year, dtype=np.int64)
        parts.append(part)

    def _cat(key: str) -> np.ndarray[Any, Any]:
        return np.concatenate([p[key] for p in parts])

    x = _cat("x")
    years = _cat("years")
    if ctx.limit_samples is not None:
        x, years = x[: ctx.limit_samples], years[: ctx.limit_samples]

    def _lim(a: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return a[: ctx.limit_samples] if ctx.limit_samples is not None else a

    train_mask, _, _ = SPLIT.masks(years)
    if not train_mask.any():
        raise RuntimeError("no training samples in the patch caches; cannot proceed")
    train_x = x[train_mask]
    return HurricaneWindDataset(
        x=x,
        intensity_kt=_lim(_cat("intensity_kt")),
        category_idx=_lim(_cat("category_idx")),
        is_positive=_lim(_cat("is_positive")),
        obs_max_kt=_lim(_cat("obs_max_kt")),
        years=years,
        basins=_lim(_cat("basin")),
        channel_mean=train_x.mean(axis=(0, 1, 3, 4)),
        channel_std=train_x.std(axis=(0, 1, 3, 4)),
    )


def train(ctx: PipelineContext) -> dict[str, Any]:
    """Train the WindPatternAnalyzer with early stopping on val intensity MAE.

    The intensity head is optimized on its pre-ReLU linear output (the final
    ReLU would zero gradients whenever the head starts negative); inference
    through ``forward`` is unchanged. The validation MAE mirrors serving
    exactly: the prediction is floored at the patch's observed max wind, as
    the detector's learned path does.

    Returns:
        Training record (epochs run, best validation MAE, sample counts).
    """
    from omni_mercury_engine.detectors.geological.hurricane_detector import WindPatternAnalyzer

    rng = seed_everything(ctx.seed)
    ds = build_dataset(ctx)
    train_mask, val_mask, _ = SPLIT.masks(ds.years)

    x_train = torch.from_numpy(ds.x[train_mask])
    y_kt_train = torch.from_numpy(ds.intensity_kt[train_mask].astype(np.float32))
    y_cat_train = torch.from_numpy(ds.category_idx[train_mask])
    x_val = torch.from_numpy(ds.x[val_mask])
    y_kt_val = ds.intensity_kt[val_mask]
    obs_val = ds.obs_max_kt[val_mask]

    n_classes = len(_category_order())
    counts = np.bincount(ds.category_idx[train_mask], minlength=n_classes).astype(np.float64)
    present = counts > 0
    weights = np.zeros(n_classes, dtype=np.float32)
    weights[present] = counts.sum() / (present.sum() * counts[present])
    class_weights = torch.from_numpy(weights)

    model = WindPatternAnalyzer()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce = torch.nn.CrossEntropyLoss(weight=class_weights)

    logger.info(
        "training on %d samples (%.1f%% positive), validating on %d",
        int(train_mask.sum()),
        100.0 * float(ds.is_positive[train_mask].mean()),
        int(val_mask.sum()),
    )

    def _heads(xb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared encoder pass -> (pre-ReLU intensity kt, category logits)."""
        batch, seq_len = xb.shape[:2]
        encoded = [model.conv_encoder(xb[:, t]).view(batch, -1) for t in range(seq_len)]
        lstm_out, _ = model.lstm(torch.stack(encoded, dim=1))
        final_state = lstm_out[:, -1]
        intensity_pre = model.intensity_predictor[:3](final_state).squeeze(-1)
        return intensity_pre, model.category_classifier(final_state)

    batch_size = 32
    best_val_mae = float("inf")
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
            intensity_pre, logits = _heads(x_train[batch_idx])
            loss = torch.nn.functional.mse_loss(
                intensity_pre / INTENSITY_SCALE_KT, y_kt_train[batch_idx] / INTENSITY_SCALE_KT
            ) + ce(logits, y_cat_train[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * batch_idx.shape[0]

        model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, x_val.shape[0], 128):
                max_wind, _ = model(x_val[start : start + 128])
                preds.append(max_wind.squeeze(-1).numpy())
        val_pred = np.maximum(np.concatenate(preds), obs_val)  # serve-parity floor
        val_mae = float(np.mean(np.abs(val_pred - y_kt_val)))
        logger.info(
            "epoch %d: train loss %.4f, val intensity MAE %.3f kt",
            epoch + 1,
            epoch_loss / x_train.shape[0],
            val_mae,
        )
        if val_mae < best_val_mae - 1e-3:
            best_val_mae = val_mae
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.info("early stop at epoch %d (patience %d)", epoch + 1, patience)
                break

    if best_state is None:
        raise RuntimeError("training produced no finite validation MAE; refusing to save")
    model.load_state_dict(best_state)

    record = {
        "seed": ctx.seed,
        "epochs_run": epochs_run,
        "best_val_intensity_mae_kt": best_val_mae,
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "train_years": list(SPLIT.train_years),
        "val_years": list(SPLIT.val_years),
        "train_positive_fraction": float(ds.is_positive[train_mask].mean()),
        "train_category_counts": counts.astype(int).tolist(),
        "class_weights": weights.tolist(),
        "train_channel_mean_ms": ds.channel_mean.tolist(),
        "train_channel_std_ms": ds.channel_std.tolist(),
    }
    payload: dict[str, Any] = {
        "wind_analyzer": model.state_dict(),
        "feature_spec": FEATURE_SPEC_VERSION,
        "patch_deg": PATCH_DEG,
        "grid": "0.25deg",
        "seq_hours": list(SEQ_OFFSET_HOURS),
        "channels": list(CHANNELS),
    }
    save_candidate(ctx.data_dir, HOOK_NAME, payload, record)
    return record


def evaluate(ctx: PipelineContext) -> EvaluationOutcome:
    """Compare learned vs physics through the public detector API.

    Both detectors receive the *identical* held-out cases: the same
    ``wind_field`` dict a real caller would pass (u/v patch sequences in
    m/s). Physics is the detector's deterministic observed-kinematics
    fallback; learned is the same detector after ``load_neural_weights`` on
    the candidate checkpoint. Primary metric: intensity MAE in knots (lower
    is better).
    """
    from omni_mercury_engine.detectors.geological.hurricane_detector import HurricaneDetector

    ds = build_dataset(ctx)
    _, _, test_mask = SPLIT.masks(ds.years)
    test_idx = np.flatnonzero(test_mask)
    if test_idx.size == 0:
        raise RuntimeError("no test samples found; cannot evaluate")

    cand_path, _ = candidate_paths(ctx.data_dir, HOOK_NAME)
    if not cand_path.exists():
        raise FileNotFoundError(f"no candidate checkpoint at {cand_path}; run --train first")

    physics_det = HurricaneDetector()
    learned_det = HurricaneDetector()
    learned_det.load_neural_weights(str(cand_path))

    truth_kt = ds.intensity_kt[test_idx]
    is_pos = ds.is_positive[test_idx]
    cat_true = np.array([_category_order()[int(c)] for c in ds.category_idx[test_idx]])

    results: dict[str, dict[str, list[Any]]] = {
        "physics": {"kt": [], "conf": [], "cat": []},
        "learned": {"kt": [], "conf": [], "cat": []},
    }
    for i in test_idx:
        case = {
            "wind_field": {
                "u": ds.x[i, :, 0].astype(np.float64),
                "v": ds.x[i, :, 1].astype(np.float64),
                "grid_spacing_m": GRID_SPACING_M,
            },
            "basin": _BASIN_TO_DETECTOR.get(str(ds.basins[i]), "atlantic"),
        }
        for label, det in (("physics", physics_det), ("learned", learned_det)):
            out = det.predict_hurricane(case)
            if not np.isfinite(out.max_wind_speed_kt):
                raise RuntimeError(f"{label} path returned non-finite wind for case {i}")
            results[label]["kt"].append(float(out.max_wind_speed_kt))
            results[label]["conf"].append(float(out.confidence))
            results[label]["cat"].append(out.category)

    # Hurricane-detection AUC: hurricane-strength positives vs negatives
    # (sub-hurricane positives are excluded from this binary task only).
    auc_mask = (~is_pos) | (is_pos & (truth_kt >= HURRICANE_WIND_KT))

    def _metrics(label: str) -> dict[str, float]:
        kt = np.asarray(results[label]["kt"])
        conf = np.asarray(results[label]["conf"])
        cat = np.asarray(results[label]["cat"])
        return {
            "intensity_mae_kt": float(np.mean(np.abs(kt - truth_kt))),
            "intensity_rmse_kt": float(np.sqrt(np.mean((kt - truth_kt) ** 2))),
            "hurricane_detection_auc": binary_auc(is_pos[auc_mask], conf[auc_mask]),
            "hurricane_detection_auc_wind_score": binary_auc(is_pos[auc_mask], kt[auc_mask]),
            "category_accuracy": float(np.mean(cat == cat_true)),
            "intensity_mae_kt_positives": float(np.mean(np.abs(kt[is_pos] - truth_kt[is_pos]))),
            "intensity_mae_kt_negatives": float(np.mean(np.abs(kt[~is_pos] - truth_kt[~is_pos]))),
        }

    basins, basin_counts = np.unique(
        np.where(ds.basins[test_idx] == "", "(negative)", ds.basins[test_idx]),
        return_counts=True,
    )
    manifest_path = ctx.data_dir / "hurricane" / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    outcome = EvaluationOutcome(
        hook=HOOK_NAME,
        primary_metric="intensity_mae_kt",
        higher_is_better=False,
        learned=_metrics("learned"),
        physics=_metrics("physics"),
        n_test_samples=int(test_idx.size),
        test_years=SPLIT.test_years,
        extras={
            "comparison": "identical held-out ERA5 patch sequences through "
            "HurricaneDetector.predict_hurricane, physics fallback vs loaded checkpoint",
            "n_test_positive": int(is_pos.sum()),
            "n_test_negative": int((~is_pos).sum()),
            "n_auc_cases": int(auc_mask.sum()),
            "per_basin_test_counts": dict(zip(basins.tolist(), basin_counts.tolist(), strict=True)),
            "chunk_gets_total": manifest.get("chunk_gets_total"),
            "bytes_fetched_total": manifest.get("bytes_fetched_total"),
            "era5_availability": manifest.get("era5_availability"),
            "notes": [
                "negative-sample intensity truth equals the observed patch max wind, so the "
                "physics fallback is exact on negatives by construction; the learned path "
                "must win on positives without losing the negatives",
                "physics confidence is quantized (vorticity needs a single-time 2-D field, "
                "and sequence input yields zero vorticity), so its confidence-based AUC is "
                "uninformative; the wind-score AUC gives physics its best detection shot",
            ],
        },
        constraints=[
            {
                "metric": "intensity_mae_kt_positives",
                "higher_is_better": False,
                "description": "the win must come from real storms: positive-sample "
                "intensity error must not regress below the physics under-estimate",
            },
            {
                "metric": "category_accuracy",
                "higher_is_better": True,
                "description": "Saffir-Simpson bucket accuracy through the public API "
                "must not regress",
            },
            {
                "metric": "hurricane_detection_auc_wind_score",
                "higher_is_better": True,
                "description": "hurricane-vs-negative ranking must not regress against "
                "physics' best detection score (patch max wind)",
            },
        ],
    )
    save_evaluation(ctx.data_dir, outcome)
    logger.info(
        "evaluation: learned intensity MAE %.3f kt vs physics %.3f kt on %d held-out samples (%s)",
        outcome.learned["intensity_mae_kt"],
        outcome.physics["intensity_mae_kt"],
        outcome.n_test_samples,
        "LEARNED WINS" if outcome.learned_beats_physics else "PHYSICS WINS",
    )
    return outcome


def ship(ctx: PipelineContext) -> tuple[Any, Any]:
    """Promote the candidate through the merit gate (may refuse loudly)."""
    from omni_mercury_engine.ml.hazard_training.common import load_evaluation

    outcome = load_evaluation(ctx.data_dir, HOOK_NAME)
    manifest_path = ctx.data_dir / "hurricane" / "manifest.json"
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
