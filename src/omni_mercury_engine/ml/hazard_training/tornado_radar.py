# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Train the DopplerRadarAnalyzer on real NEXRAD Level-II velocity + SPC reports.

Data sources (hook ``tornado_radar``, ``TornadoDetector.load_neural_weights``):

* **NEXRAD Level-II volume scans** -- the Unidata public S3 mirror
  (``https://unidata-nexrad-level2.s3.amazonaws.com``, anonymous, per-scan
  objects, verified for 2011-2023). The GCS mirror
  (``gcp-public-data-nexrad-l2``) is probed at fetch time and recorded in the
  manifest, but it stores hourly ``.tar`` bundles rather than per-scan
  objects, so the Unidata mirror is what this pipeline downloads from.
  Volumes are decoded with :class:`metpy.io.Level2File` (lazy import); the
  training input is the LOWEST-elevation sweep that carries radial velocity
  (``VEL``) -- on split-cut VCPs that is the ~0.5 deg Doppler cut.
* **SPC tornado reports** (``www.spc.noaa.gov/wcm/data``,
  ``1950-2023_actual_tornadoes.csv``) -- the positives. Report times are CST
  (``tz == 3``); they are converted to UTC by adding 6 hours, rows with any
  other timezone code are excluded and counted. EF0 reports are excluded
  (``mag >= 1`` only): EF0 timing/location is too noisy to align a +/-20-min
  radar scan window against. Only full-track records (``sg == 1``) are used
  so multi-county segment rows cannot duplicate one tornado.
* **SPC wind/hail reports** (``1955-2023_wind.csv.zip`` / ``_hail.csv.zip``)
  -- used ONLY to select honest hard-negative days: convective days with
  severe storms near a radar but no tornado report within 300 km. They are
  never used as tornado labels.
* **WSR-88D site table** (``www.ncei.noaa.gov/access/homr/file/
  nexrad-stations.txt``) -- radar latitude/longitude for the range/azimuth
  geometry; fetched with sha256 provenance like every other source.

Sample construction (the (rays, gates) array IS ``weather_data
["radar_sequence"]``, exactly what the deployed detector consumes):

* **Positives**: for each sampled EF1+ report, the nearest WSR-88D within
  150 km; the volume scan closest to the report time within
  [-20 min, +5 min]; from the lowest VEL sweep, the 64 contiguous range
  gates centered on the report's range and a +/-30-ray sector around the
  report's azimuth -> a ``(61, 64)`` float32 m/s window. Gates the radar
  masked (below SNR / range-folded) decode as NaN and are stored as NaN in
  the sector cache; :func:`build_dataset` maps NaN -> 0.0 because that is
  precisely what ``TornadoDetector._analyze_radar_physics`` does and what
  the LSTM must therefore also see. Velocities are NOT dealiased -- both
  the physics fallback and the learned model see the same raw archive data.
* **Quiet-day negatives**: same radars, days with at least two SPC wind/hail
  reports within 120 km (storms genuinely present) but NO tornado report of
  any rating within 300 km that UTC day; the scan nearest the (seeded,
  jittered) median severe-report time; up to five random-azimuth /
  random-range sectors per volume.
* **Same-day marginal negatives**: sectors re-extracted from the POSITIVE
  volumes at seeded random azimuth/range whose center lies >= 100 km from
  every tornado report (any rating) of that UTC day. Two documented
  deviations from the original plan: (1) the plan called for separate scans
  >= 2 h from every report, but reusing the already-downloaded tornadic
  volume is both the stronger hard negative (an ongoing outbreak elsewhere
  on the radar) and what keeps total downloads under the 6-GiB cap; (2) the
  planned 150-km exclusion radius rejects essentially every draw once the
  sector range is capped at 140 km (a report near the radar excludes the
  whole disk), so the radius is 100 km -- still more than five sector
  lengths (a sector spans ~16 km of range by +/-15 deg of azimuth) and far
  outside the report's parent storm at the same scan time. The honesty of
  the 0 label rests on that spatial exclusion against every same-day
  report. Residual label noise (unreported circulations, EF0-only areas)
  is inherent to SPC ground truth and is documented rather than filtered.

All samples (both classes) must have >= 30% finite gates so the task is
echo-vs-echo discrimination, not echo-vs-void; reports beyond usable echo
coverage at the lowest sweep are skipped and counted in the manifest.

Training targets: ``mesocyclone_classifier`` BCE (tornado-report-associated
sector = 1); ``rotation_estimator`` MSE on positives with target = the
observed velocity-couplet rotational velocity of the SAME window divided by
50 (the detector multiplies the head's output by 50 at inference). That
target is a measured kinematic quantity computed by the detector's own
physics formula (per-ray ``(Vmax - Vmin)/2``, median over rays, NaN -> 0),
not a human label. There are no fitted normalization statistics: the
deployed detector feeds raw m/s to the LSTM, so training does too.

Temporal split (never random -- outbreaks autocorrelate within seasons and
years): train 2011-2019, validation 2020-2021, test 2022-2023.

Evaluation runs BOTH models through the public
:meth:`TornadoDetector.predict_tornado` API on identical held-out windows.
The composite ``result.confidence`` quantizes to {0, 0.5} for radar-only
input (the mesocyclone indicator is the only contributor), so the primary
AUC is computed on a deployed-outputs-only score documented in
``extras['metric_choice']``: ``rotation_velocity_ms + 100 *
mesocyclone_detected``. For the physics path this is a strictly monotone
transform of its rotational velocity (detection IS ``v_rot >= 15``), so
physics AUC equals the AUC of its continuous ranking -- no handicap.

The learned decision threshold is NOT the architecture's built-in 0.5: the
class-weighted BCE shifts the head's probability scale, so the deployed
``meso_prob >= tau`` rule is selected on the VALIDATION years
(:func:`_select_operating_point`, mirroring the solar-storm hook), carried
in the checkpoint payload as ``payload['operating_point']``, and
consumed/validated by ``TornadoDetector.load_neural_weights``.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import re
import struct
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from omni_mercury_engine.security.safe_torch import safe_torch_load

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import torch

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

HOOK_NAME = "tornado_radar"
CHECKPOINT_NAME = "tornado_nexrad"
FEATURE_SPEC = "tornado-nexrad-v1"

SPC_TORNADO_URL = "https://www.spc.noaa.gov/wcm/data/1950-2023_actual_tornadoes.csv"
SPC_WIND_URL = "https://www.spc.noaa.gov/wcm/data/1955-2023_wind.csv.zip"
SPC_HAIL_URL = "https://www.spc.noaa.gov/wcm/data/1955-2023_hail.csv.zip"
NEXRAD_STATIONS_URL = "https://www.ncei.noaa.gov/access/homr/file/nexrad-stations.txt"
UNIDATA_BASE = "https://unidata-nexrad-level2.s3.amazonaws.com"
GCS_PROBE_URL = (
    "https://storage.googleapis.com/storage/v1/b/gcp-public-data-nexrad-l2/o"
    "?prefix=2013/05/20/KTLX/&maxResults=2"
)

SPLIT = TemporalSplit(
    train_years=tuple(range(2011, 2020)),
    val_years=(2020, 2021),
    test_years=(2022, 2023),
)

#: Deployed input contract: +/-30 rays around the report azimuth ...
SECTOR_RAYS = 61
#: ... by 64 contiguous range gates centered on the report range.
SECTOR_GATES = 64
_HALF_RAYS = SECTOR_RAYS // 2
_HALF_GATES = SECTOR_GATES // 2

#: SPC reports are CST when ``tz == 3``; UTC = CST + 6 h (documented in the
#: SPC WCM format notes; rows with any other tz code are excluded, counted).
_CST_TO_UTC = _dt.timedelta(hours=6)

MIN_EF = 1
MAX_RADAR_RANGE_KM = 150.0
MIN_RADAR_RANGE_KM = 12.0
SCAN_BEFORE_S = 20 * 60
SCAN_AFTER_S = 5 * 60
MIN_FINITE_FRACTION = 0.30

#: Originally planned at 28/year; shrunk to 20/year after measuring the
#: mirror (modern storm-day volumes run 10-26 MB) so the whole 2011-2023
#: fetch fits the 4-GiB download budget. A smaller real dataset beats a
#: padded or truncated one; the manifest records the final counts.
POSITIVES_PER_YEAR = 20
#: Quiet-day volumes are the only negatives that cost fresh downloads, so
#: the byte budget is spent on more sectors per volume rather than more
#: volumes (2016+ uncompressed SAILS volumes run 15-25 MB each). Shrunk
#: from 6x4 to 4x5 sectors/year under the same 4-GiB budget.
QUIET_VOLUMES_PER_YEAR = 4
QUIET_SECTORS_PER_VOLUME = 5
MARGINAL_SECTORS_PER_POSITIVE = 1
#: Seeded draws per volume when hunting an acceptable random sector. Outbreak
#: days reject most draws on the report-exclusion rule, so the hunt needs
#: headroom; sector range stays inside the positive band so range alone
#: cannot become a class shortcut.
MARGINAL_ATTEMPTS = 24
QUIET_ATTEMPTS = 12
QUIET_TORNADO_EXCLUSION_KM = 300.0
#: Same-day marginal negatives: minimum distance from the sector CENTER to
#: every tornado report of that UTC day. See the module docstring for why
#: this is 100 km rather than the planned 150 km (a 150-km exclusion is
#: geometrically infeasible against the <=140-km sector-range band).
MARGINAL_EXCLUSION_KM = 100.0
STORM_PROXIMITY_KM = 120.0
MIN_STORM_REPORTS = 2
MAX_DOWNLOAD_BYTES = 4 * 2**30

_EARTH_RADIUS_KM = 6371.0
_KEY_TIME_RE = re.compile(r"([A-Z]{4})(\d{8})_(\d{6})")


# ---------------------------------------------------------------------------
# Geometry (spherical Earth; adequate at < 300 km against a 250-m gate grid)
# ---------------------------------------------------------------------------


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2.0) ** 2
    return float(2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)))


def _haversine_vec(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Vectorized great-circle distances from one point to many, km."""
    p1 = np.radians(lat)
    p2 = np.radians(np.asarray(lats, dtype=np.float64))
    dphi = p2 - p1
    dlmb = np.radians(np.asarray(lons, dtype=np.float64) - lon)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2.0) ** 2
    return 2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from point 1 to point 2, degrees [0, 360)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlmb = np.radians(lon2 - lon1)
    y = np.sin(dlmb) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dlmb)
    return float(np.degrees(np.arctan2(y, x)) % 360.0)


def destination_point(
    lat: float, lon: float, bearing_deg: float, dist_km: float
) -> tuple[float, float]:
    """Point reached from (lat, lon) along ``bearing_deg`` for ``dist_km``."""
    p1 = np.radians(lat)
    l1 = np.radians(lon)
    brg = np.radians(bearing_deg)
    dr = dist_km / _EARTH_RADIUS_KM
    p2 = np.arcsin(np.sin(p1) * np.cos(dr) + np.cos(p1) * np.sin(dr) * np.cos(brg))
    l2 = l1 + np.arctan2(
        np.sin(brg) * np.sin(dr) * np.cos(p1), np.cos(dr) - np.sin(p1) * np.sin(p2)
    )
    return float(np.degrees(p2)), float((np.degrees(l2) + 540.0) % 360.0 - 180.0)


# ---------------------------------------------------------------------------
# Deployed-parity physics observable
# ---------------------------------------------------------------------------


def couplet_v_rot(window: np.ndarray) -> float:
    """Observed rotational velocity of a window, detector-physics parity.

    Mirrors ``TornadoDetector._analyze_radar_physics`` exactly: NaN -> 0,
    per-ray ``(Vmax - Vmin) / 2``, median over rays. Used as the
    rotation-estimator training target (a measured kinematic quantity on the
    same window, not a human label) and as the reference for rotation MAE.

    Args:
        window: 2-D ``(rays, gates)`` velocity array in m/s (NaN allowed).

    Returns:
        The median per-ray couplet velocity in m/s.
    """
    arr = np.asarray(window, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    finite = np.where(np.isfinite(arr), arr, 0.0)
    per_frame = [(np.max(frame) - np.min(frame)) / 2.0 for frame in finite]
    return float(np.median(per_frame)) if per_frame else 0.0


# ---------------------------------------------------------------------------
# SPC report parsing
# ---------------------------------------------------------------------------


def spc_time_to_utc(date: str, time: str) -> _dt.datetime:
    """Convert an SPC CST date/time pair to a UTC datetime.

    SPC WCM files store times in CST (``tz == 3``); UTC = CST + 6 h. The
    conversion handles the day rollover for evening reports.

    Args:
        date: ``YYYY-MM-DD``.
        time: ``HH:MM:SS`` (CST).

    Returns:
        Timezone-aware UTC datetime.
    """
    local = _dt.datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M:%S")
    return (local + _CST_TO_UTC).replace(tzinfo=_dt.UTC)


def load_spc_reports(
    csv_path: Path,
    *,
    years: tuple[int, ...],
    min_mag: int | None = None,
    full_track_only: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Load SPC report rows with CST -> UTC conversion and honest filtering.

    Args:
        csv_path: Cached SPC CSV (tornado, wind, or hail; may be ``.zip``).
        years: Keep only reports whose UTC year is in this tuple.
        min_mag: Optional minimum magnitude (EF scale for tornadoes). Rows
            with unknown magnitude (``-9``) are excluded when this is set.
        full_track_only: Keep only ``sg == 1`` rows. Applies to the TORNADO
            file, where ``sg == 1`` marks the whole-track record and county
            segments would duplicate one tornado; wind/hail files use
            ``sg == 0`` for every row, so the filter must stay off there.

    Returns:
        Tuple of (report dicts with ``utc`` / ``slat`` / ``slon`` / ``mag`` /
        ``om`` / ``yr`` / ``st`` keys, exclusion counters).
    """
    import pandas as pd

    df = pd.read_csv(csv_path, low_memory=False)
    stats = {"rows_total": len(df)}
    # Year prefilter on the file's own (CST) year: +/-1 year margin so the
    # UTC conversion at New Year cannot drop a row before it is converted.
    df = df[(df["yr"] >= min(years) - 1) & (df["yr"] <= max(years) + 1)]
    if full_track_only:
        n = len(df)
        df = df[df["sg"] == 1]
        stats["excluded_segment_rows"] = n - len(df)
    n = len(df)
    df = df[df["tz"] == 3]
    stats["excluded_non_cst_tz"] = n - len(df)
    if min_mag is not None:
        n = len(df)
        df = df[df["mag"] >= min_mag]
        stats["excluded_below_min_mag"] = n - len(df)
    reports: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        utc = spc_time_to_utc(str(row.date), str(row.time))
        if utc.year not in years:
            continue
        mag = float(row.mag)
        reports.append(
            {
                "om": int(row.om),
                "yr": utc.year,
                "utc": utc,
                "slat": float(row.slat),
                "slon": float(row.slon),
                # Wind reports carry gust knots (float, possibly missing);
                # tornado rows are integral EF. -9 = unknown, SPC convention.
                "mag": int(mag) if np.isfinite(mag) else -9,
                "st": str(row.st),
            }
        )
    stats["rows_used"] = len(reports)
    return reports, stats


def parse_nexrad_stations(text: str) -> dict[str, tuple[float, float]]:
    """Parse the NCEI HOMR ``nexrad-stations.txt`` fixed-width site table.

    Args:
        text: File contents.

    Returns:
        Mapping ICAO -> (lat, lon) for NEXRAD (WSR-88D) sites.

    Raises:
        ValueError: If the header layout is not the documented one.
    """
    lines = text.splitlines()
    if len(lines) < 3 or "ICAO" not in lines[0]:
        raise ValueError("nexrad-stations.txt: unexpected header; refusing to guess columns")
    header, dashes = lines[0], lines[1]
    spans = [(m.start(), m.end()) for m in re.finditer(r"-+", dashes)]
    names = [header[a:b].strip() for a, b in spans]
    idx = {name: i for i, name in enumerate(names)}
    for required in ("ICAO", "LAT", "LON", "STNTYPE"):
        if required not in idx:
            raise ValueError(f"nexrad-stations.txt: missing column {required}")
    sites: dict[str, tuple[float, float]] = {}
    for line in lines[2:]:
        if not line.strip():
            continue
        fields = [line[a:b].strip() for a, b in spans]
        if "NEXRAD" not in fields[idx["STNTYPE"]].upper():
            continue
        icao = fields[idx["ICAO"]]
        if not icao:
            continue
        sites[icao] = (float(fields[idx["LAT"]]), float(fields[idx["LON"]]))
    if len(sites) < 100:
        raise ValueError(f"nexrad-stations.txt: parsed only {len(sites)} NEXRAD sites")
    return sites


# ---------------------------------------------------------------------------
# NEXRAD listing / decode / sector extraction
# ---------------------------------------------------------------------------


def _paths(ctx: PipelineContext) -> dict[str, Path]:
    """Standard on-disk layout under the pipeline data directory."""
    root = ctx.data_dir / "nexrad_tornado"
    return {
        "root": root,
        "labels": root / "labels",
        "listings": root / "listings",
        "volumes": root / "volumes",
        "sectors": root / "sectors",
        "samples": root / "samples.json",
        "manifest": root / "manifest.json",
    }


def list_day_scans(
    ctx: PipelineContext, site: str, day: _dt.date
) -> list[tuple[_dt.datetime, str]]:
    """List a site's volume-scan keys for one UTC day (cached S3 listing).

    Args:
        ctx: Pipeline context (cache directory).
        site: Radar ICAO (e.g. ``KTLX``).
        day: UTC date.

    Returns:
        Sorted (scan UTC datetime, S3 key) tuples; ``_MDM`` metadata objects
        are excluded.

    Raises:
        RuntimeError: If the listing is truncated (would silently hide scans).
    """
    prefix = f"{day:%Y/%m/%d}/{site}/"
    url = f"{UNIDATA_BASE}/?list-type=2&prefix={prefix}&max-keys=1000"
    path = cached_fetch(url, _paths(ctx)["listings"] / f"{site}_{day:%Y%m%d}.xml")
    text = path.read_text()
    if "<IsTruncated>true</IsTruncated>" in text:
        raise RuntimeError(f"S3 listing truncated for {prefix}; raise max-keys")
    scans: list[tuple[_dt.datetime, str]] = []
    for key in re.findall(r"<Key>([^<]+)</Key>", text):
        if key.endswith("MDM"):
            continue
        m = _KEY_TIME_RE.search(key.rsplit("/", 1)[-1])
        if m is None:
            continue
        stamp = _dt.datetime.strptime(m.group(2) + m.group(3), "%Y%m%d%H%M%S").replace(
            tzinfo=_dt.UTC
        )
        scans.append((stamp, key))
    scans.sort()
    return scans


def fetch_volume(ctx: PipelineContext, key: str) -> Path:
    """Download (or reuse) one Level-II volume object by S3 key."""
    dest = _paths(ctx)["volumes"] / key.replace("/", "_")
    return cached_fetch(f"{UNIDATA_BASE}/{key}", dest, timeout=300.0)


@dataclass
class VelSweep:
    """Lowest-elevation radial-velocity sweep decoded from a Level-II volume.

    Attributes:
        az_deg: Per-ray azimuths, degrees.
        vel: ``(n_rays, n_gates)`` velocity in m/s; NaN where the radar
            reported no valid estimate (below SNR / censored).
        first_gate_km: Range of the first gate center, km.
        gate_width_km: Gate spacing, km.
        elevation_deg: Sweep elevation angle, degrees.
    """

    az_deg: np.ndarray
    vel: np.ndarray
    first_gate_km: float
    gate_width_km: float
    elevation_deg: float


def decode_lowest_vel_sweep(volume_path: Path) -> VelSweep:
    """Decode the lowest-elevation VEL sweep from a Level-II archive file.

    Args:
        volume_path: Cached volume file (plain or gzip; metpy sniffs).

    Returns:
        The decoded sweep.

    Raises:
        ValueError: If no sweep in the volume carries VEL (fail loud; the
            caller counts and re-selects rather than fabricating data).
    """
    from metpy.io import Level2File  # type: ignore[import-untyped]

    f = Level2File(str(volume_path))
    best_idx: int | None = None
    best_el = float("inf")
    for i, sweep in enumerate(f.sweeps):
        if not sweep:  # metpy emits empty sweeps for missed elevations
            continue
        ray0 = sweep[0]
        if not isinstance(ray0[4], dict) or b"VEL" not in ray0[4]:
            continue
        el = float(ray0[0].el_angle)
        if el < best_el:
            best_el = el
            best_idx = i
    if best_idx is None:
        raise ValueError(f"{volume_path.name}: no VEL sweep in volume")
    sweep = f.sweeps[best_idx]
    hdr = sweep[0][4][b"VEL"][0]
    n_gates = int(hdr.num_gates)
    az = np.array([float(ray[0].az_angle) for ray in sweep], dtype=np.float64)
    vel = np.full((len(sweep), n_gates), np.nan, dtype=np.float32)
    for j, ray in enumerate(sweep):
        if not isinstance(ray[4], dict) or b"VEL" not in ray[4]:
            continue
        data = np.asarray(ray[4][b"VEL"][1], dtype=np.float32)
        vel[j, : min(len(data), n_gates)] = data[:n_gates]
    return VelSweep(
        az_deg=az,
        vel=vel,
        first_gate_km=float(hdr.first_gate),
        gate_width_km=float(hdr.gate_width),
        elevation_deg=best_el,
    )


def extract_sector(
    sweep_az_deg: np.ndarray,
    sweep_vel: np.ndarray,
    first_gate_km: float,
    gate_width_km: float,
    bearing_deg: float,
    range_km: float,
) -> np.ndarray | None:
    """Cut the deployed ``(SECTOR_RAYS, SECTOR_GATES)`` window from a sweep.

    The window is the 64 contiguous range gates centered on ``range_km``
    across the +/-30 rays (circular) nearest ``bearing_deg``. NaN gates are
    preserved (mapped to 0.0 later, at dataset-build time, to match the
    deployed detector's NaN handling).

    Args:
        sweep_az_deg: Per-ray azimuths, degrees.
        sweep_vel: ``(n_rays, n_gates)`` velocity, m/s.
        first_gate_km: Range of the first gate center, km.
        gate_width_km: Gate spacing, km.
        bearing_deg: Target azimuth from the radar, degrees.
        range_km: Target range from the radar, km.

    Returns:
        ``(61, 64)`` float32 array, or None when the gate window falls
        outside the sweep's range coverage (never zero-padded -- padding
        would fabricate calm air).
    """
    n_rays, n_gates = sweep_vel.shape
    if n_rays < SECTOR_RAYS:
        return None
    gate_center = round((range_km - first_gate_km) / gate_width_km)
    g0 = gate_center - _HALF_GATES
    g1 = gate_center + _HALF_GATES
    if g0 < 0 or g1 > n_gates:
        return None
    ang = np.abs((sweep_az_deg - bearing_deg + 180.0) % 360.0 - 180.0)
    center_ray = int(np.argmin(ang))
    ray_idx = [(center_ray + k) % n_rays for k in range(-_HALF_RAYS, _HALF_RAYS + 1)]
    return sweep_vel[ray_idx, g0:g1].astype(np.float32)


def _finite_fraction(window: np.ndarray) -> float:
    """Fraction of finite gates in a window."""
    return float(np.mean(np.isfinite(window)))


# ---------------------------------------------------------------------------
# fetch stage
# ---------------------------------------------------------------------------


def _candidate_sites(
    sites: dict[str, tuple[float, float]], lat: float, lon: float, k: int = 3
) -> list[tuple[str, float]]:
    """Up to ``k`` WSR-88D sites within the usable range band, nearest first.

    More than one candidate matters in practice: the nearest radar can be
    absent from the archive for that day (e.g. KOUN, the Norman research
    radar, on 2013-05-20), and skipping such reports entirely would bias the
    positive sample toward well-archived sites.
    """
    names = list(sites)
    lats = np.array([sites[n][0] for n in names], dtype=np.float64)
    lons = np.array([sites[n][1] for n in names], dtype=np.float64)
    dists = _haversine_vec(lat, lon, lats, lons)
    in_band = (dists >= MIN_RADAR_RANGE_KM) & (dists <= MAX_RADAR_RANGE_KM)
    order = np.argsort(dists)
    picked = [(names[int(i)], float(dists[int(i)])) for i in order if in_band[int(i)]]
    return picked[:k]


def _scan_in_window(
    scans: list[tuple[_dt.datetime, str]], when: _dt.datetime
) -> tuple[_dt.datetime, str] | None:
    """The scan closest to ``when`` within [-SCAN_BEFORE_S, +SCAN_AFTER_S]."""
    best: tuple[_dt.datetime, str] | None = None
    best_abs = float("inf")
    for stamp, key in scans:
        delta = (stamp - when).total_seconds()
        if -SCAN_BEFORE_S <= delta <= SCAN_AFTER_S and abs(delta) < best_abs:
            best_abs = abs(delta)
            best = (stamp, key)
    return best


def _reports_by_utc_day(
    reports: list[dict[str, Any]],
) -> dict[_dt.date, list[dict[str, Any]]]:
    """Group reports by their UTC calendar date."""
    by_day: dict[_dt.date, list[dict[str, Any]]] = {}
    for rep in reports:
        by_day.setdefault(rep["utc"].date(), []).append(rep)
    return by_day


def _save_sector(
    ctx: PipelineContext, sample_id: str, window: np.ndarray, meta: dict[str, Any]
) -> None:
    """Persist one extracted sector (NaN preserved) with its metadata."""
    sectors = _paths(ctx)["sectors"]
    sectors.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        sectors / f"{sample_id}.npz",
        window=window,
        meta=np.array(json.dumps(meta, default=str)),
    )


class _DownloadBudget:
    """Loud accounting against the volume-download byte cap.

    Crossing the cap flips :attr:`exhausted` instead of raising: the
    collectors stop SELECTING further volumes, everything already extracted
    is kept, and the manifest records the truncation. Raising here would
    throw away hours of already-downloaded real data over the last volume;
    a smaller real dataset (reported honestly) beats a dead pipeline. The
    volume that crossed the cap is still used -- its bytes are already
    spent and counted.
    """

    def __init__(self, cap_bytes: int) -> None:
        self.cap = cap_bytes
        self.bytes = 0
        self.volumes = 0
        self.exhausted = False

    def add(self, path: Path) -> None:
        self.bytes += path.stat().st_size
        self.volumes += 1
        if self.bytes > self.cap and not self.exhausted:
            self.exhausted = True
            logger.warning(
                "volume download budget exhausted (%d > %d bytes); "
                "stopping further volume selection -- the manifest records "
                "the shrunken counts",
                self.bytes,
                self.cap,
            )


def _collect_positive_samples(
    ctx: PipelineContext,
    rng: np.random.Generator,
    tornado_reports: list[dict[str, Any]],
    tornado_any_mag: list[dict[str, Any]],
    sites: dict[str, tuple[float, float]],
    budget: _DownloadBudget,
    stats: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Sample EF1+ reports per year; extract tornadic + marginal sectors.

    The same-day marginal negatives are drawn here, from the SAME decoded
    sweep as the positive they share a volume with (decoding a Level-II
    volume costs seconds; re-decoding hundreds of them would double the
    build time for zero data benefit). Their 0 label is enforced against
    every tornado report of ANY rating on that UTC day.

    Args:
        ctx: Pipeline context.
        rng: Seeded generator (selection order + marginal sector draws).
        tornado_reports: EF1+ full-track reports (the positive pool).
        tornado_any_mag: All-ratings full-track reports (marginal exclusion).
        sites: WSR-88D ICAO -> (lat, lon).
        budget: Download-byte accounting.
        stats: Mutable exclusion counters.

    Returns:
        Tuple of (positive sample metadata, marginal-negative metadata).
    """
    by_year: dict[int, list[dict[str, Any]]] = {}
    for rep in tornado_reports:
        by_year.setdefault(rep["yr"], []).append(rep)
    all_by_day = _reports_by_utc_day(tornado_any_mag)

    samples: list[dict[str, Any]] = []
    marginal: list[dict[str, Any]] = []
    for year in SPLIT.all_years:
        if budget.exhausted:
            logger.warning("positives for %d+ skipped: download budget exhausted", year)
            break
        pool = sorted(by_year.get(year, []), key=lambda r: r["om"])
        order = rng.permutation(len(pool))
        taken = 0
        for pi in order:
            if taken >= POSITIVES_PER_YEAR or budget.exhausted:
                break
            rep = pool[int(pi)]
            cands = _candidate_sites(sites, rep["slat"], rep["slon"])
            if not cands:
                stats["pos_skipped_no_site_in_band"] += 1
                continue
            when = rep["utc"]
            extracted = False
            for site, dist_km in cands:
                slat, slon = sites[site]
                bearing = initial_bearing_deg(slat, slon, rep["slat"], rep["slon"])
                days = {(when - _dt.timedelta(seconds=SCAN_BEFORE_S)).date(), when.date()}
                scans: list[tuple[_dt.datetime, str]] = []
                try:
                    for day in sorted(days):
                        scans.extend(list_day_scans(ctx, site, day))
                except RuntimeError:
                    raise
                except Exception:
                    stats["pos_site_attempt_listing_error"] += 1
                    continue
                hit = _scan_in_window(scans, when)
                if hit is None:
                    stats["pos_site_attempt_no_scan_in_window"] += 1
                    continue
                stamp, key = hit
                try:
                    vol_path = fetch_volume(ctx, key)
                    budget.add(vol_path)
                    sweep = decode_lowest_vel_sweep(vol_path)
                except (ValueError, OSError, EOFError, IndexError, KeyError, struct.error) as exc:
                    stats["pos_site_attempt_decode_error"] += 1
                    logger.warning("skipping undecodable volume %s: %s", key, exc)
                    continue
                window = extract_sector(
                    sweep.az_deg,
                    sweep.vel,
                    sweep.first_gate_km,
                    sweep.gate_width_km,
                    bearing,
                    dist_km,
                )
                if window is None or _finite_fraction(window) < MIN_FINITE_FRACTION:
                    stats["pos_site_attempt_void_sector"] += 1
                    continue
                sample_id = f"pos_{year}_{rep['om']}"
                meta = {
                    "id": sample_id,
                    "kind": "positive",
                    "label": 1,
                    "year": year,
                    "om": rep["om"],
                    "ef": rep["mag"],
                    "state": rep["st"],
                    "site": site,
                    "volume_key": key,
                    "volume_sha256": sha256_file(vol_path),
                    "scan_utc": stamp.isoformat(),
                    "report_utc": when.isoformat(),
                    "bearing_deg": round(bearing, 3),
                    "range_km": round(dist_km, 3),
                    "elevation_deg": round(sweep.elevation_deg, 3),
                    "finite_fraction": round(_finite_fraction(window), 4),
                }
                _save_sector(ctx, sample_id, window, meta)
                samples.append(meta)
                marginal.extend(
                    _draw_marginal_negatives(
                        ctx,
                        rng,
                        sweep=sweep,
                        sample_id=sample_id,
                        year=year,
                        site=site,
                        site_latlon=(slat, slon),
                        volume_key=key,
                        volume_sha256=str(meta["volume_sha256"]),
                        day_reports=[
                            (r["slat"], r["slon"]) for r in all_by_day.get(when.date(), [])
                        ],
                        stats=stats,
                    )
                )
                extracted = True
                break
            if extracted:
                taken += 1
            else:
                stats["pos_skipped_all_candidate_sites"] += 1
        logger.info("year %d: %d positive sectors extracted", year, taken)
    return samples, marginal


def _draw_marginal_negatives(
    ctx: PipelineContext,
    rng: np.random.Generator,
    *,
    sweep: VelSweep,
    sample_id: str,
    year: int,
    site: str,
    site_latlon: tuple[float, float],
    volume_key: str,
    volume_sha256: str,
    day_reports: list[tuple[float, float]],
    stats: dict[str, int],
) -> list[dict[str, Any]]:
    """Cut far-away sectors from an already-decoded positive sweep."""
    samples: list[dict[str, Any]] = []
    slat, slon = site_latlon
    rep_lats = np.array([r[0] for r in day_reports], dtype=np.float64)
    rep_lons = np.array([r[1] for r in day_reports], dtype=np.float64)
    accepted = 0
    for attempt in range(MARGINAL_ATTEMPTS):
        if accepted >= MARGINAL_SECTORS_PER_POSITIVE:
            break
        bearing = float(rng.uniform(0.0, 360.0))
        rng_km = float(rng.uniform(20.0, 140.0))
        clat, clon = destination_point(slat, slon, bearing, rng_km)
        if (
            rep_lats.size
            and float(_haversine_vec(clat, clon, rep_lats, rep_lons).min()) < MARGINAL_EXCLUSION_KM
        ):
            stats["neg_marginal_rejected_near_report"] += 1
            continue
        window = extract_sector(
            sweep.az_deg,
            sweep.vel,
            sweep.first_gate_km,
            sweep.gate_width_km,
            bearing,
            rng_km,
        )
        if window is None or _finite_fraction(window) < MIN_FINITE_FRACTION:
            stats["neg_marginal_rejected_void"] += 1
            continue
        neg_id = f"negm_{sample_id}_{attempt}"
        meta = {
            "id": neg_id,
            "kind": "marginal_negative",
            "label": 0,
            "year": year,
            "om": None,
            "ef": -1,
            "site": site,
            "volume_key": volume_key,
            "volume_sha256": volume_sha256,
            "bearing_deg": round(bearing, 3),
            "range_km": round(rng_km, 3),
            "finite_fraction": round(_finite_fraction(window), 4),
            "exclusion_km": MARGINAL_EXCLUSION_KM,
        }
        _save_sector(ctx, neg_id, window, meta)
        samples.append(meta)
        accepted += 1
    return samples


def _collect_quiet_negatives(
    ctx: PipelineContext,
    rng: np.random.Generator,
    storm_reports: list[dict[str, Any]],
    tornado_any_mag: list[dict[str, Any]],
    positive_sites: dict[int, list[str]],
    sites: dict[str, tuple[float, float]],
    budget: _DownloadBudget,
    stats: dict[str, int],
) -> list[dict[str, Any]]:
    """Storm-day-but-tornado-free negatives from fresh volumes."""
    storms_by_day = _reports_by_utc_day(storm_reports)
    tor_by_day = _reports_by_utc_day(tornado_any_mag)
    # Vectorized per-day coordinate arrays: the candidate scan is
    # (days x sites) with hundreds of reports per day -- scalar haversine
    # loops would add minutes per year for no accuracy gain.
    storm_coords = {
        day: (
            np.array([r["slat"] for r in reps], dtype=np.float64),
            np.array([r["slon"] for r in reps], dtype=np.float64),
        )
        for day, reps in storms_by_day.items()
    }
    tor_coords = {
        day: (
            np.array([t["slat"] for t in reps], dtype=np.float64),
            np.array([t["slon"] for t in reps], dtype=np.float64),
        )
        for day, reps in tor_by_day.items()
    }

    samples: list[dict[str, Any]] = []
    for year in SPLIT.all_years:
        if budget.exhausted:
            logger.warning("quiet negatives for %d+ skipped: download budget exhausted", year)
            break
        year_sites = sorted(set(positive_sites.get(year, [])))
        if not year_sites:
            continue
        candidates: list[tuple[str, _dt.date, list[_dt.datetime]]] = []
        for day, reps in storms_by_day.items():
            if day.year != year:
                continue
            s_lats, s_lons = storm_coords[day]
            for site in year_sites:
                slat, slon = sites[site]
                near_mask = _haversine_vec(slat, slon, s_lats, s_lons) <= STORM_PROXIMITY_KM
                if int(near_mask.sum()) < MIN_STORM_REPORTS:
                    continue
                if day in tor_coords:
                    t_lats, t_lons = tor_coords[day]
                    if bool(
                        (
                            _haversine_vec(slat, slon, t_lats, t_lons) <= QUIET_TORNADO_EXCLUSION_KM
                        ).any()
                    ):
                        continue
                near = [r for r, keep in zip(reps, near_mask, strict=True) if keep]
                candidates.append((site, day, sorted(r["utc"] for r in near)))
        order = rng.permutation(len(candidates))
        taken_volumes = 0
        for ci in order:
            if taken_volumes >= QUIET_VOLUMES_PER_YEAR or budget.exhausted:
                break
            site, day, times = candidates[int(ci)]
            target = times[len(times) // 2] + _dt.timedelta(minutes=float(rng.uniform(-30.0, 30.0)))
            try:
                scans = list_day_scans(ctx, site, day)
            except RuntimeError:
                raise
            except Exception:
                stats["neg_quiet_listing_error"] += 1
                continue
            if not scans:
                stats["neg_quiet_no_scans"] += 1
                continue
            stamp, key = min(scans, key=lambda s: abs((s[0] - target).total_seconds()))
            if abs((stamp - target).total_seconds()) > 45 * 60:
                stats["neg_quiet_no_scan_near_storms"] += 1
                continue
            try:
                vol_path = fetch_volume(ctx, key)
                budget.add(vol_path)
                sweep = decode_lowest_vel_sweep(vol_path)
            except (ValueError, OSError, EOFError, IndexError, KeyError, struct.error) as exc:
                stats["neg_quiet_decode_error"] += 1
                logger.warning("skipping undecodable volume %s: %s", key, exc)
                continue
            vol_sha = sha256_file(vol_path)
            accepted = 0
            for attempt in range(QUIET_ATTEMPTS):
                if accepted >= QUIET_SECTORS_PER_VOLUME:
                    break
                bearing = float(rng.uniform(0.0, 360.0))
                rng_km = float(rng.uniform(20.0, 140.0))
                window = extract_sector(
                    sweep.az_deg,
                    sweep.vel,
                    sweep.first_gate_km,
                    sweep.gate_width_km,
                    bearing,
                    rng_km,
                )
                if window is None or _finite_fraction(window) < MIN_FINITE_FRACTION:
                    stats["neg_quiet_rejected_void"] += 1
                    continue
                sample_id = f"negq_{site}_{day:%Y%m%d}_{attempt}"
                meta = {
                    "id": sample_id,
                    "kind": "quiet_negative",
                    "label": 0,
                    "year": year,
                    "om": None,
                    "ef": -1,
                    "site": site,
                    "volume_key": key,
                    "volume_sha256": vol_sha,
                    "scan_utc": stamp.isoformat(),
                    "bearing_deg": round(bearing, 3),
                    "range_km": round(rng_km, 3),
                    "finite_fraction": round(_finite_fraction(window), 4),
                    "tornado_exclusion_km": QUIET_TORNADO_EXCLUSION_KM,
                }
                _save_sector(ctx, sample_id, window, meta)
                samples.append(meta)
                accepted += 1
            if accepted:
                taken_volumes += 1
        logger.info("year %d: %d quiet-day volumes used", year, taken_volumes)
    return samples


def fetch(ctx: PipelineContext) -> dict[str, Any]:
    """Download labels, site table and volumes; extract all sector windows.

    Selection is seeded from ``ctx.seed`` and every skip is counted, so the
    manifest states exactly what was excluded and why. Sector extraction
    happens here (not in :func:`build_dataset`) because usability of a
    report (scan in window, echo coverage) is only knowable after the
    download, and the re-selection loop belongs with the network stage.

    Returns:
        Manifest with per-file sha256 provenance and selection statistics.
    """
    p = _paths(ctx)
    for d in ("labels", "listings", "volumes", "sectors"):
        p[d].mkdir(parents=True, exist_ok=True)

    sources: list[dict[str, Any]] = []
    spc_path = cached_fetch(SPC_TORNADO_URL, p["labels"] / "1950-2023_actual_tornadoes.csv")
    sources.append(
        {
            "url": SPC_TORNADO_URL,
            "sha256": sha256_file(spc_path),
            "description": "SPC WCM tornado reports 1950-2023 (labels; CST times)",
        }
    )
    wind_path = cached_fetch(SPC_WIND_URL, p["labels"] / "1955-2023_wind.csv.zip")
    hail_path = cached_fetch(SPC_HAIL_URL, p["labels"] / "1955-2023_hail.csv.zip")
    for url, path, kind in (
        (SPC_WIND_URL, wind_path, "wind"),
        (SPC_HAIL_URL, hail_path, "hail"),
    ):
        sources.append(
            {
                "url": url,
                "sha256": sha256_file(path),
                "description": f"SPC WCM {kind} reports (negative-day storm-presence check only)",
            }
        )
    stations_path = cached_fetch(NEXRAD_STATIONS_URL, p["labels"] / "nexrad-stations.txt")
    sources.append(
        {
            "url": NEXRAD_STATIONS_URL,
            "sha256": sha256_file(stations_path),
            "description": "NCEI HOMR WSR-88D site table (radar lat/lon geometry)",
        }
    )
    gcs_probe: dict[str, Any] = {"url": GCS_PROBE_URL}
    try:
        probe_path = cached_fetch(GCS_PROBE_URL, p["labels"] / "gcs_mirror_probe.json")
        payload = json.loads(probe_path.read_text())
        gcs_probe["reachable"] = True
        gcs_probe["object_layout"] = (
            "hourly .tar bundles"
            if any("tar" in i.get("name", "") for i in payload.get("items", []))
            else "per-scan objects"
        )
    except Exception as exc:
        gcs_probe["reachable"] = False
        gcs_probe["error"] = str(exc)[:200]
    gcs_probe["decision"] = (
        "Unidata mirror used for all volume downloads (per-scan objects, verified 2011-2023); "
        "GCS mirror stores hourly tar bundles"
    )

    sites = parse_nexrad_stations(stations_path.read_text())
    tornado_ef1, tor_stats = load_spc_reports(
        spc_path, years=SPLIT.all_years, min_mag=MIN_EF, full_track_only=True
    )
    tornado_any, tor_any_stats = load_spc_reports(
        spc_path, years=SPLIT.all_years, min_mag=None, full_track_only=True
    )
    wind_reports, wind_stats = load_spc_reports(wind_path, years=SPLIT.all_years, min_mag=None)
    hail_reports, hail_stats = load_spc_reports(hail_path, years=SPLIT.all_years, min_mag=None)

    rng = np.random.default_rng(ctx.seed)
    budget = _DownloadBudget(MAX_DOWNLOAD_BYTES)
    stats: dict[str, int] = defaultdict(int)

    positives, marginal = _collect_positive_samples(
        ctx, rng, tornado_ef1, tornado_any, sites, budget, stats
    )
    positive_sites: dict[int, list[str]] = {}
    for s in positives:
        positive_sites.setdefault(int(s["year"]), []).append(str(s["site"]))
    quiet = _collect_quiet_negatives(
        ctx,
        rng,
        wind_reports + hail_reports,
        tornado_any,
        positive_sites,
        sites,
        budget,
        stats,
    )

    samples = positives + marginal + quiet
    p["samples"].write_text(json.dumps(samples, indent=1, sort_keys=True))

    # One provenance entry per distinct volume (positives share volumes with
    # their same-day marginal negatives).
    volume_sources_by_url: dict[str, dict[str, Any]] = {}
    for s in samples:
        if "volume_sha256" not in s:
            continue
        url = f"{UNIDATA_BASE}/{s['volume_key']}"
        entry = volume_sources_by_url.setdefault(
            url,
            {"url": url, "sha256": s["volume_sha256"], "description": "NEXRAD Level-II volume;"},
        )
        entry["description"] += f" {s['id']}"
    volume_sources = [volume_sources_by_url[u] for u in sorted(volume_sources_by_url)]
    manifest = {
        "hook": HOOK_NAME,
        "sources": sources + volume_sources,
        "gcs_mirror_probe": gcs_probe,
        "selection_stats": dict(stats),
        "label_parse_stats": {
            "tornado_ef1_plus": tor_stats,
            "tornado_any_mag": tor_any_stats,
            "wind": wind_stats,
            "hail": hail_stats,
        },
        "counts": {
            "positives": len(positives),
            "marginal_negatives": len(marginal),
            "quiet_negatives": len(quiet),
            "volumes_downloaded": budget.volumes,
            "volume_bytes_downloaded": budget.bytes,
            "budget_exhausted": budget.exhausted,
        },
        "quotas": {
            "positives_per_year": POSITIVES_PER_YEAR,
            "quiet_volumes_per_year": QUIET_VOLUMES_PER_YEAR,
            "quiet_sectors_per_volume": QUIET_SECTORS_PER_VOLUME,
            "marginal_sectors_per_positive": MARGINAL_SECTORS_PER_POSITIVE,
            "download_cap_bytes": MAX_DOWNLOAD_BYTES,
        },
        "sites_used": sorted({str(s["site"]) for s in samples}),
    }
    p["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True))
    logger.info(
        "fetch complete: %d positives, %d marginal negatives, %d quiet negatives, "
        "%d volumes (%.2f GiB)",
        len(positives),
        len(marginal),
        len(quiet),
        budget.volumes,
        budget.bytes / 2**30,
    )
    return manifest


# ---------------------------------------------------------------------------
# build stage
# ---------------------------------------------------------------------------


@dataclass
class TornadoRadarDataset:
    """Sector windows with labels and physics observables for training.

    Attributes:
        sequences: ``(N, 61, 64)`` float32 m/s, NaN already mapped to 0.0
            (deployed-detector parity).
        labels: ``(N,)`` float32 in {0, 1}.
        v_rot: ``(N,)`` float32 observed couplet velocity (m/s) of each
            window via :func:`couplet_v_rot` -- rotation-head target and
            rotation-MAE reference.
        years: ``(N,)`` int64 UTC year per sample (temporal split key).
        kinds: Per-sample kind strings (positive / marginal_negative /
            quiet_negative).
        ef: ``(N,)`` int64 EF rating (positives) or -1.
        ids: Sample identifiers.
        sites: Radar ICAO per sample.
    """

    sequences: np.ndarray
    labels: np.ndarray
    v_rot: np.ndarray
    years: np.ndarray
    kinds: list[str]
    ef: np.ndarray
    ids: list[str]
    sites: list[str]


def build_dataset(ctx: PipelineContext) -> TornadoRadarDataset:
    """Assemble the training arrays from the cached sector windows.

    NaN gates become 0.0 here -- byte-for-byte what the deployed
    ``TornadoDetector`` does to the same input. There are no fitted
    normalization statistics (the detector consumes raw m/s), so there is
    nothing to leak across the temporal split.

    Raises:
        FileNotFoundError: If the fetch stage has not been run.
        RuntimeError: If any split ends up without both classes.
    """
    p = _paths(ctx)
    if not p["samples"].exists():
        raise FileNotFoundError(f"missing {p['samples']}; run the --fetch stage first")
    samples: list[dict[str, Any]] = json.loads(p["samples"].read_text())
    samples.sort(key=lambda s: str(s["id"]))
    if ctx.limit_samples is not None:
        order = np.random.default_rng(ctx.seed).permutation(len(samples))
        samples = [samples[int(i)] for i in order[: ctx.limit_samples]]

    seqs = np.zeros((len(samples), SECTOR_RAYS, SECTOR_GATES), dtype=np.float32)
    labels = np.zeros(len(samples), dtype=np.float32)
    v_rot = np.zeros(len(samples), dtype=np.float32)
    years = np.zeros(len(samples), dtype=np.int64)
    ef = np.full(len(samples), -1, dtype=np.int64)
    kinds: list[str] = []
    ids: list[str] = []
    sites: list[str] = []
    for i, s in enumerate(samples):
        with np.load(p["sectors"] / f"{s['id']}.npz") as npz:
            window = npz["window"]
        if window.shape != (SECTOR_RAYS, SECTOR_GATES):
            raise RuntimeError(f"sector {s['id']} has shape {window.shape}; cache is corrupt")
        window = np.where(np.isfinite(window), window, 0.0).astype(np.float32)
        seqs[i] = window
        labels[i] = float(s["label"])
        v_rot[i] = couplet_v_rot(window)
        years[i] = int(s["year"])
        ef[i] = int(s.get("ef", -1))
        kinds.append(str(s["kind"]))
        ids.append(str(s["id"]))
        sites.append(str(s["site"]))

    train_mask, val_mask, test_mask = SPLIT.masks(years)
    for name, mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
        if ctx.limit_samples is None and (
            not np.any(labels[mask] == 1.0) or not np.any(labels[mask] == 0.0)
        ):
            raise RuntimeError(
                f"{name} split lacks a class (n={int(mask.sum())}); the fetch stage did not "
                "produce a usable dataset -- refusing to train/evaluate on it"
            )
    return TornadoRadarDataset(
        sequences=seqs,
        labels=labels,
        v_rot=v_rot,
        years=years,
        kinds=kinds,
        ef=ef,
        ids=ids,
        sites=sites,
    )


# ---------------------------------------------------------------------------
# train stage
# ---------------------------------------------------------------------------


def train(ctx: PipelineContext) -> dict[str, Any]:
    """Train the DopplerRadarAnalyzer with early stopping on validation AUC.

    Losses: BCE on the mesocyclone head with a positive-class weight equal
    to the train-year neg/pos ratio (pushes the deployed 0.5 threshold
    toward operational recall), plus MSE on the rotation head for POSITIVE
    samples against the observed couplet velocity / 50 (the detector
    multiplies by 50 at inference).

    Returns:
        Training record (epochs, best validation AUC, counts, class balance).
    """
    from omni_mercury_engine.detectors.geological.tornado_detector import DopplerRadarAnalyzer

    rng = seed_everything(ctx.seed)
    ds = build_dataset(ctx)
    train_mask, val_mask, _ = SPLIT.masks(ds.years)

    x_train = torch.from_numpy(ds.sequences[train_mask])
    y_train = torch.from_numpy(ds.labels[train_mask])
    r_train = torch.from_numpy(ds.v_rot[train_mask] / 50.0)
    x_val = torch.from_numpy(ds.sequences[val_mask])
    y_val_np = ds.labels[val_mask]

    n_pos = float(y_train.sum().item())
    n_neg = float(len(y_train) - n_pos)
    if n_pos < 1 or n_neg < 1:
        raise RuntimeError("training years lack a class; cannot train honestly")
    pos_weight = n_neg / n_pos

    model = DopplerRadarAnalyzer(input_dim=SECTOR_GATES)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    logger.info(
        "training on %d windows (%.1f%% positive), validating on %d",
        x_train.shape[0],
        100.0 * n_pos / max(x_train.shape[0], 1),
        x_val.shape[0],
    )

    batch_size = 32
    best_val_auc = -float("inf")
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
                continue
            xb = x_train[batch_idx]
            yb = y_train[batch_idx]
            rb = r_train[batch_idx]
            meso_prob, rotation, _ = model(xb)
            meso_prob = meso_prob.squeeze(-1).clamp(1e-6, 1 - 1e-6)
            rotation = rotation.squeeze(-1)
            weights = torch.where(yb > 0.5, torch.full_like(yb, pos_weight), torch.ones_like(yb))
            bce = torch.nn.functional.binary_cross_entropy(meso_prob, yb, weight=weights)
            pos_sel = yb > 0.5
            if bool(pos_sel.any()):
                rot_mse = torch.nn.functional.mse_loss(rotation[pos_sel], rb[pos_sel])
            else:
                rot_mse = torch.zeros((), dtype=xb.dtype)
            loss = bce + rot_mse
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += float(loss.item()) * batch_idx.shape[0]

        model.eval()
        with torch.no_grad():
            val_prob, _, _ = model(x_val)
        val_auc = binary_auc(y_val_np, val_prob.squeeze(-1).numpy())
        logger.info(
            "epoch %d: train loss %.4f, val meso AUC %.4f",
            epoch + 1,
            epoch_loss / max(x_train.shape[0], 1),
            val_auc,
        )
        if np.isfinite(val_auc) and val_auc > best_val_auc + 1e-5:
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

    operating_point = _select_operating_point(model, ds, x_val=x_val, val_mask=val_mask)

    record = {
        "seed": ctx.seed,
        "epochs_run": epochs_run,
        "best_val_meso_auc": best_val_auc,
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "train_years": list(SPLIT.train_years),
        "val_years": list(SPLIT.val_years),
        "train_positive_fraction": n_pos / max(float(x_train.shape[0]), 1.0),
        "bce_pos_weight": pos_weight,
        "rotation_target": "couplet_v_rot(window)/50 on positives (observed kinematics)",
        "operating_point": operating_point,
    }
    payload: dict[str, Any] = {
        "radar_analyzer": model.state_dict(),
        "feature_spec": FEATURE_SPEC,
        "gates": SECTOR_GATES,
        "sector_rays": SECTOR_RAYS,
        "sweep": "lowest VEL",
        "units": "m/s",
        "operating_point": operating_point,
    }
    save_candidate(ctx.data_dir, HOOK_NAME, payload, record)
    return record


def _select_operating_point(
    model: Any,
    ds: TornadoRadarDataset,
    *,
    x_val: torch.Tensor,
    val_mask: np.ndarray,
) -> dict[str, Any]:
    """Choose the deployed mesocyclone-probability threshold on validation.

    Policy (mirrors ``solar_storm._select_operating_point``, documented for
    owner ratification): the deployed learned decision is
    ``meso_prob >= tau``. The detector's built-in default is 0.5, but the
    class-weighted BCE (pos_weight = train neg/pos ratio) deliberately
    shifts the head's probability scale, so 0.5 is a training artifact,
    not a calibrated decision rule. On the VALIDATION years only, require
    recall of at least ``max(physics validation recall, 0.5)`` AND a
    false-alarm rate of at most ``0.8 * physics validation FAR`` (the 20%
    headroom guards the val->test distribution shift the ship gate's hard
    FAR constraint does not forgive); among feasible thresholds pick the
    one maximizing CSI (ties -> higher tau, i.e. fewer false alarms).
    Physics on the same validation windows is the deployed
    velocity-couplet rule ``couplet_v_rot(window) >= 15 m/s`` --
    byte-identical to ``TornadoDetector._analyze_radar_physics`` on these
    NaN->0 windows (that observable is ``ds.v_rot``).

    Returns:
        Operating-point record stored in the checkpoint payload and the
        provenance sidecar (threshold, policy, and the validation-year
        recall/FAR/CSI for both the learned rule and physics).

    Raises:
        RuntimeError: If validation lacks a class, or no threshold
            satisfies even the FAR ceiling (a doomed operating point must
            not be recorded).
    """
    labels_val = ds.labels[val_mask].astype(bool)
    if not labels_val.any() or labels_val.all():
        raise RuntimeError(
            "validation years contain a single class; cannot select an operating point honestly"
        )

    model.eval()
    with torch.no_grad():
        meso_prob_t, _, _ = model(x_val)
    meso_prob = meso_prob_t.squeeze(-1).numpy().astype(np.float64)

    phys_detect = ds.v_rot[val_mask] >= 15.0
    physics_recall = float(np.mean(phys_detect[labels_val]))
    physics_far = float(np.mean(phys_detect[~labels_val]))
    recall_floor = max(physics_recall, 0.5)
    far_ceiling = 0.8 * physics_far

    def _rule_metrics(tau: float) -> tuple[float, float, float]:
        detect = meso_prob >= tau
        tp = float(np.sum(detect & labels_val))
        fn = float(np.sum(~detect & labels_val))
        fp = float(np.sum(detect & ~labels_val))
        recall = tp / max(tp + fn, 1.0)
        far = fp / max(float(np.sum(~labels_val)), 1.0)
        csi = tp / max(tp + fn + fp, 1.0)
        return recall, far, csi

    # Candidate grid: the head's own validation quantiles, plus the 0.5
    # default (so "keep the default" is always considered) and 1.0 (sigmoid
    # output is strictly < 1, so tau=1 means "never detect" -- the honest
    # last-resort point when even one detection breaks the FAR ceiling).
    taus = np.unique(
        np.concatenate([np.quantile(meso_prob, np.linspace(0.0, 1.0, 513)), [0.5, 1.0]])
    )
    best: dict[str, Any] | None = None
    fallback: dict[str, Any] | None = None
    for tau in taus:
        recall, far, csi = _rule_metrics(float(tau))
        entry = {
            "meso_prob_threshold": float(tau),
            "val_recall": recall,
            "val_far": far,
            "val_csi": csi,
        }
        # Fallback if no threshold satisfies both floors: the most
        # conservative feasible-on-FAR point with the best recall (a
        # recall-maximizing fallback that blows the FAR ceiling would be
        # selecting a point the ship gate is guaranteed to refuse).
        if far <= far_ceiling and (fallback is None or recall > fallback["val_recall"]):
            fallback = entry
        if (
            recall >= recall_floor
            and far <= far_ceiling
            and (
                best is None
                or csi > best["val_csi"]
                or (csi == best["val_csi"] and tau > best["meso_prob_threshold"])
            )
        ):
            best = entry
    floor_met = best is not None
    chosen = best if best is not None else fallback
    if chosen is None:
        raise RuntimeError(
            "no operating point satisfies even the FAR ceiling on validation; "
            "the mesocyclone head is not usable for deployed decisions -- "
            "refusing to record a doomed operating point"
        )
    return {
        **chosen,
        "policy": "mesocyclone_detected = meso_prob >= tau; tau maximizes val CSI "
        "subject to val recall >= max(physics val recall, 0.5) AND "
        "val FAR <= 0.8 * physics val FAR",
        "recall_floor": recall_floor,
        "recall_floor_met": floor_met,
        "far_ceiling": far_ceiling,
        "val_recall_physics": physics_recall,
        "val_far_physics": physics_far,
    }


# ---------------------------------------------------------------------------
# evaluate stage
# ---------------------------------------------------------------------------

#: Deployed-score decision bonus. |v| in Level-II archives is Nyquist-bounded
#: well under 100 m/s, so adding 100 on detection keeps the physics score a
#: strictly monotone transform of its rotational velocity (identical AUC).
_DECISION_BONUS = 100.0

#: Seeded bootstrap resamples for the 95% CI on the AUC difference.
_BOOTSTRAP_RESAMPLES = 1000


def _bootstrap_auc_delta_ci(
    labels: np.ndarray,
    learned_score: np.ndarray,
    physics_score: np.ndarray,
    *,
    seed: int,
    n_resamples: int = _BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Seeded stratified paired bootstrap 95% CI on AUC(learned) - AUC(physics).

    Paired: each resample scores BOTH models on the same resampled windows,
    so the interval is on the difference, not two marginal intervals.
    Stratified: positives and negatives are resampled separately with their
    class counts preserved, so every resampled AUC is defined (no
    single-class resamples to silently drop).

    Args:
        labels: 0/1 test labels.
        learned_score: Learned deployed score per test window.
        physics_score: Physics deployed score per test window.
        seed: Bootstrap RNG seed (recorded in the returned record).
        n_resamples: Number of resamples.

    Returns:
        Record with the observed delta, 95% CI bounds, and method notes.
    """
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(labels == 1.0)
    neg_idx = np.flatnonzero(labels != 1.0)
    deltas = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        pi = rng.choice(pos_idx, size=pos_idx.size, replace=True)
        ni = rng.choice(neg_idx, size=neg_idx.size, replace=True)
        idx = np.concatenate([pi, ni])
        deltas[b] = binary_auc(labels[idx], learned_score[idx]) - binary_auc(
            labels[idx], physics_score[idx]
        )
    return {
        "observed_delta": binary_auc(labels, learned_score) - binary_auc(labels, physics_score),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "method": "stratified paired bootstrap over test windows "
        "(class counts preserved; percentile interval)",
    }


def evaluate(ctx: PipelineContext) -> EvaluationOutcome:
    """Compare learned vs physics through the public detector API.

    Both detectors receive the identical held-out ``(61, 64)`` windows as
    ``weather_data["radar_sequence"]``. Physics is the detector with no
    checkpoint loaded (velocity-couplet fallback); learned is the same
    detector after ``load_neural_weights(candidate)``.

    Returns:
        The evaluation outcome (primary metric: AUC of the deployed score,
        higher is better), with deployed-decision recall/false-alarm
        non-regression constraints.
    """
    from omni_mercury_engine.detectors.geological.tornado_detector import TornadoDetector

    ds = build_dataset(ctx)
    _, _, test_mask = SPLIT.masks(ds.years)
    test_idx = np.flatnonzero(test_mask)
    if test_idx.size == 0:
        raise RuntimeError("no test windows found; cannot evaluate")

    cand_path, _ = candidate_paths(ctx.data_dir, HOOK_NAME)
    if not cand_path.exists():
        raise FileNotFoundError(f"no candidate checkpoint at {cand_path}; run --train first")

    def _detector() -> TornadoDetector:
        return TornadoDetector(
            enable_radar=True,
            enable_atmospheric=False,
            enable_pressure=False,
            enable_resonance=False,
            enable_recursion=False,
            enable_refactoring=False,
        )

    physics_det = _detector()
    learned_det = _detector()
    learned_det.load_neural_weights(str(cand_path))

    labels = ds.labels[test_idx]
    det_logger = logging.getLogger("omni_mercury_engine.detectors.geological.tornado_detector")
    previous_level = det_logger.level
    det_logger.setLevel(logging.WARNING)  # silence per-call INFO; restored below
    results: dict[str, dict[str, list[float]]] = {
        "physics": {"rotation": [], "detected": [], "confidence": [], "latency_s": []},
        "learned": {"rotation": [], "detected": [], "confidence": [], "latency_s": []},
    }
    try:
        for i in test_idx:
            case = {"radar_sequence": ds.sequences[i]}
            for label, det in (("physics", physics_det), ("learned", learned_det)):
                t0 = time.perf_counter()
                out = det.predict_tornado(case)
                results[label]["latency_s"].append(time.perf_counter() - t0)
                if not np.isfinite(out.rotation_velocity_ms):
                    raise RuntimeError(f"{label} path returned non-finite rotation for case {i}")
                results[label]["rotation"].append(float(out.rotation_velocity_ms))
                results[label]["detected"].append(float(out.mesocyclone_detected))
                results[label]["confidence"].append(float(out.confidence))
    finally:
        det_logger.setLevel(previous_level)

    pos = labels == 1.0
    v_rot_ref = ds.v_rot[test_idx]
    detected_arr = {
        label: np.asarray(results[label]["detected"], dtype=bool)
        for label in ("physics", "learned")
    }
    score_arr = {
        label: np.asarray(results[label]["rotation"])
        + _DECISION_BONUS * detected_arr[label].astype(np.float64)
        for label in ("physics", "learned")
    }

    def _metrics(label: str) -> dict[str, float]:
        rotation = np.asarray(results[label]["rotation"])
        detected = detected_arr[label]
        confidence = np.asarray(results[label]["confidence"])
        tp = float(np.sum(detected & pos))
        fn = float(np.sum(~detected & pos))
        fp = float(np.sum(detected & ~pos))
        return {
            "auc": binary_auc(labels, score_arr[label]),
            "auc_rotation_only": binary_auc(labels, rotation),
            "auc_composite_confidence": binary_auc(labels, confidence),
            "meso_recall_deployed": float(tp / max(tp + fn, 1.0)),
            "meso_far_deployed": float(fp / max(float(np.sum(~pos)), 1.0)),
            "meso_csi_deployed": float(tp / max(tp + fn + fp, 1.0)),
            "rotation_mae_pos_ms": float(np.mean(np.abs(rotation[pos] - v_rot_ref[pos]))),
        }

    def _ef_stratified_recall() -> dict[str, dict[str, Any]]:
        """Deployed-rule recall stratified by EF rating (EF1 vs EF2+)."""
        ef_pos = ds.ef[test_idx]
        strata = {"ef1": pos & (ef_pos == 1), "ef2_plus": pos & (ef_pos >= 2)}
        table: dict[str, dict[str, Any]] = {}
        for name, mask in strata.items():
            n = int(mask.sum())
            row: dict[str, Any] = {"n": n}
            for label in ("physics", "learned"):
                row[label] = float(np.mean(detected_arr[label][mask])) if n else float("nan")
            table[name] = row
        return table

    kind_counts: dict[str, int] = {}
    for i in test_idx:
        kind_counts[ds.kinds[int(i)]] = kind_counts.get(ds.kinds[int(i)], 0) + 1
    ef_test = ds.ef[test_idx]
    manifest_path = _paths(ctx)["manifest"]
    manifest_counts: dict[str, Any] = {}
    if manifest_path.exists():
        manifest_counts = json.loads(manifest_path.read_text()).get("counts", {})

    year_counts = {int(y): int(np.sum(ds.years == y)) for y in np.unique(ds.years)}
    cand_payload = safe_torch_load(cand_path, map_location="cpu")
    assert learned_det.radar_analyzer is not None  # constructed with enable_radar=True
    learned_param_count = int(sum(p.numel() for p in learned_det.radar_analyzer.parameters()))
    outcome = EvaluationOutcome(
        hook=HOOK_NAME,
        primary_metric="auc",
        higher_is_better=True,
        learned=_metrics("learned"),
        physics=_metrics("physics"),
        n_test_samples=int(test_idx.size),
        test_years=SPLIT.test_years,
        extras={
            "comparison": "identical held-out (61, 64) m/s windows through "
            "TornadoDetector.predict_tornado, physics fallback vs loaded checkpoint",
            "metric_choice": "primary auc is over the deployed-outputs score "
            "rotation_velocity_ms + 100*mesocyclone_detected: the composite "
            "result.confidence quantizes to {0, 0.5} for radar-only input (see "
            "auc_composite_confidence), and for the physics path the combined score "
            "is a strictly monotone transform of its rotational velocity (detection "
            "IS v_rot >= 15), so physics AUC equals its continuous-ranking AUC",
            "rotation_mae_note": "physics rotation_mae_pos_ms is 0 by construction "
            "(its deployed output IS the reference observable); reported for the "
            "learned model as a regression-quality measure, not gated",
            "operating_point": "learned mesocyclone decision uses the "
            "validation-selected threshold carried by the checkpoint (see "
            "payload['operating_point']); physics decision is its fixed "
            "v_rot >= 15 m/s rule -- each path is scored on its own deployed rule",
            "operating_point_record": cand_payload.get("operating_point"),
            "auc_delta_bootstrap": _bootstrap_auc_delta_ci(
                labels, score_arr["learned"], score_arr["physics"], seed=ctx.seed
            ),
            "learned_parameter_count": learned_param_count,
            "median_inference_latency_ms": {
                label: float(np.median(results[label]["latency_s"]) * 1e3)
                for label in ("physics", "learned")
            },
            "ef_stratified_recall_deployed": _ef_stratified_recall(),
            "test_kind_counts": kind_counts,
            "test_ef_distribution": {
                str(int(v)): int(np.sum(ef_test == v)) for v in np.unique(ef_test)
            },
            "dataset_year_counts": year_counts,
            "fetch_counts": manifest_counts,
            "volumes_fetched": manifest_counts.get("volumes_downloaded"),
            "volume_bytes_fetched": manifest_counts.get("volume_bytes_downloaded"),
            "sites_used_test": sorted({ds.sites[int(i)] for i in test_idx}),
        },
        constraints=[
            {
                "metric": "meso_recall_deployed",
                "higher_is_better": True,
                "description": "mesocyclone recall at the deployed decision "
                "(meso_prob >= checkpoint-carried tau vs v_rot >= 15 m/s) must not "
                "regress below physics",
            },
            {
                "metric": "meso_far_deployed",
                "higher_is_better": False,
                "description": "false-alarm rate at the deployed decision must not "
                "exceed physics",
            },
        ],
    )
    save_evaluation(ctx.data_dir, outcome)
    logger.info(
        "evaluation: learned auc %.4f vs physics %.4f on %d held-out windows (%s)",
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
    manifest_path = _paths(ctx)["manifest"]
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
