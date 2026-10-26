# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""NOAA NHC HURDAT2 Hurricane Loader - the full best-track record, parsed honestly.

Canonical port of ``validation.data_loaders.NOAAHurricaneLoader``, rebuilt on
the fields HURDAT2 actually carries. The validation-era loader read four real
columns (lat, lon, wind, pressure) and padded fourteen constants named
``storm_speed_mph``, ``radius_34kt_ne``, ``sst_celsius`` and so on — while
ignoring the twelve real wind-radii columns present in every post-2004
best-track line — and labelled "major hurricane" by comparing HURDAT2's
knots against a 111 *mph* threshold, so only storms at roughly Category 4
strength were flagged. This loader:

- parses the documented HURDAT2 layout (header lines name the storm and the
  number of best-track lines that follow; track lines carry date, time,
  record identifier, system status, lat, lon, max wind [kt], min pressure
  [mb], and the 34/50/64-kt wind radii [nmi] in NE/SE/SW/NW quadrants,
  with ``-999`` marking missing values);
- derives storm translation speed and heading from consecutive fixes of the
  same storm (great-circle distance over the real time delta) — measured
  kinematics, not constants — dropping each storm's first fix;
- labels major hurricanes at ``wind >= 96 kt`` (Saffir-Simpson Category 3
  in HURDAT2's own unit), a threshold on a feature, so ``LABEL_SOURCE`` is
  ``"statistical"``;
- keeps only seasons from ``since_year`` (default 2004), the era in which
  the radii columns are populated, instead of imputing them.

Source: https://www.nhc.noaa.gov/data/#hurdat (Landsea & Franklin, 2013).
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from typing import Any

import numpy as np

from omni_mercury_engine.datasets.base import (
    DatasetConfig,
    DatasetLoader,
    http_get_with_retry,
)
from omni_mercury_engine.datasets.exceptions import (
    ALLOW_SYNTHETIC,
    DataSourceUnavailableError,
    check_synthetic_allowed,
)
from omni_mercury_engine.security.input_validation import TrustedEndpoints

logger = logging.getLogger(__name__)

_EARTH_RADIUS_NM = 3440.065  # nautical miles
_MISSING = -999.0


def _parse_coordinate(token: str) -> float:
    """``'28.0N'`` -> 28.0, ``'94.8W'`` -> -94.8 (south/west negative)."""
    text = token.strip()
    hemisphere = text[-1].upper()
    value = float(text[:-1])
    return -value if hemisphere in ("S", "W") else value


def _great_circle_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine great-circle distance in nautical miles."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * _EARTH_RADIUS_NM * math.asin(min(1.0, math.sqrt(a)))


def _initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from fix 1 to fix 2, degrees [0, 360)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def parse_hurdat2(text: str, since_year: int) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Parse a HURDAT2 best-track file into features over real fields only.

    Args:
        text: The raw HURDAT2 file contents.
        since_year: Keep only storms whose header season is at least this
            (the wind-radii columns are populated from 2004 onward).

    Returns:
        ``(features, wind_kt_labels)`` where features follow
        :attr:`NOAAHurricaneLoader.FEATURE_NAMES` and rows with any missing
        (``-999``) retained field, or without a preceding fix to derive
        motion from, are dropped rather than filled.
    """
    rows: list[list[float]] = []
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    index = 0
    while index < len(lines):
        header = lines[index].split(",")
        index += 1
        if len(header) < 3 or len(header[0].strip()) != 8:
            continue  # not a storm header; skip defensively
        storm_id = header[0].strip()
        try:
            season = int(storm_id[4:8])
            n_track_lines = int(header[2])
        except ValueError:
            continue
        track = lines[index : index + n_track_lines]
        index += n_track_lines
        if season < since_year:
            continue
        previous: tuple[datetime, float, float] | None = None
        for line in track:
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 20:
                continue
            try:
                moment = datetime.strptime(parts[0] + parts[1], "%Y%m%d%H%M").replace(tzinfo=UTC)
                lat = _parse_coordinate(parts[4])
                lon = _parse_coordinate(parts[5])
                wind_kt = float(parts[6])
                pressure_mb = float(parts[7])
                radii = [float(p) for p in parts[8:20]]
            except (ValueError, IndexError):
                continue
            fix = (moment, lat, lon)
            if wind_kt <= 0 or pressure_mb == _MISSING or any(r == _MISSING for r in radii):
                previous = fix
                continue
            if previous is None:
                previous = fix
                continue
            prev_moment, prev_lat, prev_lon = previous
            hours = (moment - prev_moment).total_seconds() / 3600.0
            previous = fix
            if hours <= 0:
                continue
            speed_kt = _great_circle_nm(prev_lat, prev_lon, lat, lon) / hours
            heading_deg = _initial_bearing_deg(prev_lat, prev_lon, lat, lon)
            rows.append(
                [
                    lat,
                    lon,
                    wind_kt,
                    pressure_mb,
                    *radii,
                    speed_kt,
                    heading_deg,
                    float(moment.hour),
                    float(moment.timetuple().tm_yday),
                ]
            )
    if not rows:
        return (
            np.empty((0, len(NOAAHurricaneLoader.FEATURE_NAMES)), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )
    features = np.asarray(rows, dtype=np.float64)
    return features, features[:, 2].copy()


class NOAAHurricaneLoader(DatasetLoader):
    """NOAA NHC HURDAT2 best-track loader over the fields the record carries.

    Note:
        Major-hurricane labels are a threshold on the wind feature itself
        (``LABEL_SOURCE = "statistical"``): use for unsupervised evaluation,
        never for headline supervised AUC.
    """

    DATASET_NAME = "noaa_hurricane"
    LABEL_SOURCE = "statistical"  # labels = wind_kt >= Cat-3 threshold (threshold on a feature)
    DATASET_URL = "https://www.nhc.noaa.gov/data/#hurdat"
    LICENSE = "Public Domain (U.S. Government Work)"
    CITATION = """Landsea, C. W., and J. L. Franklin (2013): Atlantic Hurricane Database
    Uncertainty and Presentation of a New Database Format. Mon. Wea. Rev., 141, 3576-3592.
    NOAA National Hurricane Center. https://www.nhc.noaa.gov/"""
    REQUIRES_CREDENTIALS = False

    HURDAT2_INDEX_URL = TrustedEndpoints.NOAA_NHC_HURDAT2_INDEX
    HURDAT2_URL = TrustedEndpoints.NOAA_NHC_HURDAT2

    #: Saffir-Simpson Category 3 floor in HURDAT2's own unit (knots).
    MAJOR_HURRICANE_KT = 96.0

    FEATURE_NAMES = [
        "latitude",
        "longitude",
        "max_wind_kt",
        "min_pressure_mb",
        "radius_34kt_ne_nm",
        "radius_34kt_se_nm",
        "radius_34kt_sw_nm",
        "radius_34kt_nw_nm",
        "radius_50kt_ne_nm",
        "radius_50kt_se_nm",
        "radius_50kt_sw_nm",
        "radius_50kt_nw_nm",
        "radius_64kt_ne_nm",
        "radius_64kt_se_nm",
        "radius_64kt_sw_nm",
        "radius_64kt_nw_nm",
        "storm_speed_kt",
        "storm_heading_deg",
        "hour_of_day",
        "day_of_year",
    ]

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize the instance."""
        super().__init__(config)
        self.major_threshold_kt = float(
            config.preprocessing.get("major_threshold_kt", self.MAJOR_HURRICANE_KT)
        )
        self.since_year = int(config.preprocessing.get("since_year", 2004))
        self._is_real_data = False

    @property
    def is_real_data(self) -> bool:
        """Return True if real data was loaded."""
        return self._is_real_data

    def _real_cache_name(self) -> str:
        """Cache name carries the era and threshold: labels are baked in."""
        return f"noaa_hurricane_real_{self.since_year}_kt{self.major_threshold_kt:g}.npz"

    def _synthetic_cache_name(self) -> str:
        return f"noaa_hurricane_synthetic_kt{self.major_threshold_kt:g}.npz"

    def download(self) -> bool:
        """Fetch and parse HURDAT2, caching the feature matrix."""
        if self._download_hurdat2():
            return True
        if ALLOW_SYNTHETIC:
            check_synthetic_allowed("NOAAHurricane", "NHC HURDAT2 fetch failed")
            return self._create_synthetic()
        raise DataSourceUnavailableError(
            loader_name="NOAAHurricane",
            source_url=self.HURDAT2_URL,
            reason="NHC HURDAT2 fetch failed",
        )

    def _resolve_latest_url(self) -> str:
        """Newest ``hurdat2-1851-*.txt`` from the NHC index, else the pin.

        HURDAT2 filenames are date-stamped and churn with each reanalysis
        release (the pre-2026 ``gis/forecast/archive`` copy now 404s), so the
        index page is authoritative and the pinned file is the fallback.
        """
        import re

        try:
            TrustedEndpoints.validate_url(self.HURDAT2_INDEX_URL)
            page = http_get_with_retry(
                self.HURDAT2_INDEX_URL, timeout=60, timeout_per_attempt=30, retries=1
            ).decode("utf-8", errors="replace")
            names = sorted(
                set(re.findall(r"hurdat2-1851-(\d{4})-(\d{6,8})\.txt", page)),
                key=lambda pair: (int(pair[0]), pair[1][-2:], pair[1][:-2]),
            )
            if names:
                end_year, stamp = names[-1]
                return f"{self.HURDAT2_INDEX_URL}hurdat2-1851-{end_year}-{stamp}.txt"
        except Exception as e:
            logger.warning(f"HURDAT2 index lookup failed ({e}); using the pinned file")
        return self.HURDAT2_URL

    def _download_hurdat2(self) -> bool:
        cache_file = self.data_path / self._real_cache_name()
        if cache_file.exists():
            logger.info(f"HURDAT2 data already cached at {cache_file}")
            self._is_real_data = True
            return True
        try:
            url = self._resolve_latest_url()
            TrustedEndpoints.validate_url(url)
            content = http_get_with_retry(url, timeout=180, timeout_per_attempt=60, retries=1)
            features, wind_kt = parse_hurdat2(content.decode("utf-8"), self.since_year)
            if len(features) == 0:
                logger.warning("HURDAT2 fetched but no fully-observed track fixes parsed")
                return False
            labels = (wind_kt >= self.major_threshold_kt).astype(np.int64)
            np.savez_compressed(cache_file, features=features, labels=labels)
            self._is_real_data = True
            logger.info(
                f"HURDAT2 loaded: {len(features)} track fixes since {self.since_year}, "
                f"{int(labels.sum())} at Category 3+ (is_real_data=True)"
            )
            return True
        except Exception as e:
            logger.warning(f"HURDAT2 download failed: {e}")
            return False

    def _create_synthetic(self) -> bool:
        """Create a labelled synthetic track matrix (policy-gated callers only)."""
        rng = np.random.default_rng(self.config.random_seed)
        n = self.config.max_samples or 3000
        lat = rng.uniform(10, 45, size=n)
        lon = rng.uniform(-100, -20, size=n)
        wind = np.clip(rng.gamma(shape=2.5, scale=22.0, size=n), 15, 160)
        pressure = np.clip(1013 - (wind - 15) * 0.9 + rng.normal(0, 6, size=n), 880, 1015)
        r34 = np.clip(rng.normal(120, 40, size=(n, 4)) * (wind[:, None] / 64), 0, 400)
        r50 = np.clip(r34 * rng.uniform(0.4, 0.7, size=(n, 4)), 0, 300)
        r64 = np.clip(r50 * rng.uniform(0.3, 0.6, size=(n, 4)), 0, 200)
        speed = np.clip(rng.gamma(shape=3.0, scale=4.0, size=n), 0, 50)
        heading = rng.uniform(0, 360, size=n)
        hour = rng.choice([0.0, 6.0, 12.0, 18.0], size=n)
        day = rng.integers(152, 335, size=n).astype(np.float64)
        features = np.column_stack(
            [lat, lon, wind, pressure, r34, r50, r64, speed, heading, hour, day]
        ).astype(np.float64)
        labels = (wind >= self.major_threshold_kt).astype(np.int64)
        cache_file = self.data_path / self._synthetic_cache_name()
        np.savez_compressed(cache_file, features=features, labels=labels)
        self._is_real_data = False
        return True

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load the cached matrix (real preferred over synthetic)."""
        for name, is_real in (
            (self._real_cache_name(), True),
            (self._synthetic_cache_name(), False),
        ):
            cache_file = self.data_path / name
            if cache_file.exists():
                data = np.load(cache_file)
                self._is_real_data = is_real
                return data["features"], data["labels"]
        raise FileNotFoundError(
            f"No cached HURDAT2 data under {self.data_path}; call download() first."
        )

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """No-op: rows with any missing retained field were dropped at parse."""
        return data


# Register at import so DatasetRegistry-driven callers can construct by name.
from omni_mercury_engine.datasets.base import DatasetRegistry

DatasetRegistry.register("noaa_hurricane", NOAAHurricaneLoader)
