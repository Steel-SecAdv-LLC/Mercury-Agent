# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""NOAA SWPC Space-Weather Loader - real multi-instrument geomagnetic data.

Canonical port of ``validation.data_loaders.NOAASpaceWeatherLoader``, rebuilt
on real products. The validation-era loader fetched one real channel (the
planetary Kp index) and synthesised the other thirteen features as
deterministic functions of Kp while naming them ``dst_index``,
``solar_wind_speed`` and so on; this loader reads every channel from the
instrument that measures it:

- Planetary Kp (3-hour synoptic, ~7 days):
  ``products/noaa-planetary-k-index.json``
- Solar-wind plasma + IMF (propagated to the bow shock, ~7 days at 1-minute
  cadence): ``products/geospace/propagated-solar-wind.json``
  (speed, density, temperature, bx, by, bz, bt)
- GOES X-ray flux: ``json/goes/primary/xrays-7-day.json``
  (0.05-0.4 nm and 0.1-0.8 nm bands)
- GOES integral proton flux: ``json/goes/primary/integral-protons-7-day.json``
  (>=10 MeV and >=100 MeV channels)

All products are public, unauthenticated, and schema-verified live on
2026-10-26. Channels are averaged onto a shared hourly UTC grid and only
hours observed by every instrument become rows, so no value is imputed
across instruments. One physics-derived feature is included and documented:
the Shue et al. (1998) magnetopause standoff distance, a closed-form
function of the measured solar-wind dynamic pressure and IMF Bz.

Labels mark geomagnetic storm hours (``Kp >= storm_threshold``; G1 = 5),
i.e. a threshold on a feature, so ``LABEL_SOURCE`` is ``"statistical"`` and
headline supervised AUC excludes the set as circular.
"""

from __future__ import annotations

import json
import logging
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

#: Hours are the shared grid every instrument is averaged onto.
_HOUR_SECONDS = 3600.0


def _parse_time_tag(value: str) -> datetime:
    """Parse an SWPC ``time_tag`` (with or without a trailing ``Z``)."""
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1]
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def _hour_bucket(moment: datetime) -> int:
    """The UTC hour this timestamp falls in, as whole hours since the epoch."""
    return int(moment.timestamp() // _HOUR_SECONDS)


def _finite(value: Any) -> float | None:
    """A finite float, or None for missing/unparseable/non-finite input."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def parse_kp_rows(rows: list[dict[str, Any]]) -> dict[int, list[float]]:
    """Hourly-bucketed Kp samples from ``products/noaa-planetary-k-index``.

    The product carries one row per 3-hour synoptic interval (field ``Kp``;
    the 1-minute product's ``estimated_kp``/``kp_index`` spellings are also
    accepted). Kp is *defined* per 3-hour interval, so a 3-hourly row's value
    is assigned to each of its three hours — the index's native resolution,
    not an interpolation.
    """
    buckets: dict[int, list[float]] = {}
    for row in rows:
        time_tag = row.get("time_tag")
        if not isinstance(time_tag, str):
            continue
        kp = _finite(row.get("Kp"))
        is_synoptic = kp is not None
        if kp is None:
            kp = _finite(row.get("estimated_kp"))
        if kp is None:
            kp = _finite(row.get("kp_index"))
        if kp is None:
            continue
        bucket = _hour_bucket(_parse_time_tag(time_tag))
        for offset in range(3 if is_synoptic else 1):
            buckets.setdefault(bucket + offset, []).append(kp)
    return buckets


#: Column order of the propagated-solar-wind product's value rows, after its
#: header row. Pinned to the live schema (2026-10-26); the parser checks the
#: header and refuses a drifted product instead of misreading columns.
_SOLAR_WIND_HEADER_PREFIX = [
    "time_tag",
    "speed",
    "density",
    "temperature",
    "bx",
    "by",
    "bz",
    "bt",
]


def parse_solar_wind_rows(table: list[list[Any]]) -> dict[int, list[list[float]]]:
    """Hourly-bucketed [speed, density, temperature, bx, by, bz, bt] samples.

    ``table`` is the raw ``products/geospace/propagated-solar-wind-1-hour``
    payload: a header row followed by value rows.

    Raises:
        ValueError: If the header does not start with the pinned columns —
            the product schema drifted and silent misreads are refused.
    """
    if not table:
        return {}
    header = [str(name) for name in table[0]]
    if header[: len(_SOLAR_WIND_HEADER_PREFIX)] != _SOLAR_WIND_HEADER_PREFIX:
        raise ValueError(
            "SWPC propagated-solar-wind product schema drifted: expected the "
            f"header to start with {_SOLAR_WIND_HEADER_PREFIX}, got {header[:8]}"
        )
    buckets: dict[int, list[list[float]]] = {}
    for row in table[1:]:
        if len(row) < len(_SOLAR_WIND_HEADER_PREFIX) or not isinstance(row[0], str):
            continue
        values = [_finite(v) for v in row[1 : len(_SOLAR_WIND_HEADER_PREFIX)]]
        if any(v is None for v in values):
            continue
        bucket = _hour_bucket(_parse_time_tag(row[0]))
        buckets.setdefault(bucket, []).append([v for v in values if v is not None])
    return buckets


def parse_xray_rows(rows: list[dict[str, Any]]) -> dict[int, list[list[float]]]:
    """Hourly-bucketed [short-band, long-band] GOES X-ray flux samples.

    The product interleaves one row per band per minute; samples enter a
    bucket only when both bands are present for the same timestamp.
    """
    by_time: dict[str, dict[str, float]] = {}
    for row in rows:
        time_tag = row.get("time_tag")
        energy = row.get("energy")
        flux = _finite(row.get("flux"))
        if not isinstance(time_tag, str) or flux is None:
            continue
        if energy in ("0.05-0.4nm", "0.1-0.8nm"):
            by_time.setdefault(time_tag, {})[str(energy)] = flux
    buckets: dict[int, list[list[float]]] = {}
    for time_tag, bands in by_time.items():
        if "0.05-0.4nm" in bands and "0.1-0.8nm" in bands:
            bucket = _hour_bucket(_parse_time_tag(time_tag))
            buckets.setdefault(bucket, []).append([bands["0.05-0.4nm"], bands["0.1-0.8nm"]])
    return buckets


def parse_proton_rows(rows: list[dict[str, Any]]) -> dict[int, list[list[float]]]:
    """Hourly-bucketed [>=10 MeV, >=100 MeV] integral proton flux samples."""
    by_time: dict[str, dict[str, float]] = {}
    for row in rows:
        time_tag = row.get("time_tag")
        energy = row.get("energy")
        flux = _finite(row.get("flux"))
        if not isinstance(time_tag, str) or flux is None:
            continue
        if energy in (">=10 MeV", ">=100 MeV"):
            by_time.setdefault(time_tag, {})[str(energy)] = flux
    buckets: dict[int, list[list[float]]] = {}
    for time_tag, bands in by_time.items():
        if ">=10 MeV" in bands and ">=100 MeV" in bands:
            bucket = _hour_bucket(_parse_time_tag(time_tag))
            buckets.setdefault(bucket, []).append([bands[">=10 MeV"], bands[">=100 MeV"]])
    return buckets


def shue_magnetopause_standoff_re(density_cm3: float, speed_km_s: float, bz_nt: float) -> float:
    """Shue et al. (1998) subsolar magnetopause standoff distance in Earth radii.

    ``r0 = (10.22 + 1.29 * tanh(0.184 * (Bz + 8.14))) * Dp ** (-1 / 6.6)``
    with the solar-wind dynamic pressure ``Dp = 1.6726e-6 * n * v**2`` in nPa
    (n in protons/cm^3, v in km/s). A closed-form function of measured
    quantities — derived physics, not a synthesised observation.
    """
    dynamic_pressure_npa = 1.6726e-6 * density_cm3 * speed_km_s**2
    dynamic_pressure_npa = max(dynamic_pressure_npa, 1e-6)
    return float(
        (10.22 + 1.29 * np.tanh(0.184 * (bz_nt + 8.14))) * dynamic_pressure_npa ** (-1.0 / 6.6)
    )


def build_hourly_matrix(
    kp: dict[int, list[float]],
    solar_wind: dict[int, list[list[float]]],
    xray: dict[int, list[list[float]]],
    protons: dict[int, list[list[float]]],
    storm_threshold: float,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Assemble the feature matrix over hours observed by every instrument.

    Returns:
        ``(features, labels)`` with one row per fully-observed hour, feature
        order matching :attr:`NOAASpaceWeatherLoader.FEATURE_NAMES`, and
        ``labels = (mean hourly Kp >= storm_threshold)``.
    """
    shared_hours = sorted(set(kp) & set(solar_wind) & set(xray) & set(protons))
    rows: list[list[float]] = []
    labels: list[int] = []
    for bucket in shared_hours:
        kp_mean = float(np.mean(kp[bucket]))
        sw_mean = np.mean(np.asarray(solar_wind[bucket], dtype=np.float64), axis=0)
        xray_mean = np.mean(np.asarray(xray[bucket], dtype=np.float64), axis=0)
        proton_mean = np.mean(np.asarray(protons[bucket], dtype=np.float64), axis=0)
        moment = datetime.fromtimestamp(bucket * _HOUR_SECONDS, tz=UTC)
        speed, density, temperature, bx, by, bz, bt = (float(v) for v in sw_mean)
        rows.append(
            [
                kp_mean,
                speed,
                density,
                temperature,
                bx,
                by,
                bz,
                bt,
                float(xray_mean[0]),
                float(xray_mean[1]),
                float(proton_mean[0]),
                float(proton_mean[1]),
                shue_magnetopause_standoff_re(density, speed, bz),
                float(moment.hour),
                float(moment.timetuple().tm_yday),
            ]
        )
        labels.append(int(kp_mean >= storm_threshold))
    if not rows:
        return (
            np.empty((0, len(NOAASpaceWeatherLoader.FEATURE_NAMES)), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
        )
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64)


class NOAASpaceWeatherLoader(DatasetLoader):
    """NOAA SWPC space-weather loader over real multi-instrument channels.

    Every feature is measured by the instrument named in the module
    docstring (or, for the magnetopause standoff, derived from measured
    quantities by a cited closed form). Hours not observed by every
    instrument are dropped rather than filled.

    Note:
        Storm labels are a threshold on the Kp feature itself
        (``LABEL_SOURCE = "statistical"``): use for unsupervised evaluation,
        never for headline supervised AUC.
    """

    DATASET_NAME = "noaa_space_weather"
    LABEL_SOURCE = "statistical"  # labels = Kp >= storm_threshold (threshold on a feature)
    DATASET_URL = "https://www.swpc.noaa.gov/"
    LICENSE = "Public Domain (U.S. Government Work)"
    CITATION = """NOAA Space Weather Prediction Center. https://www.swpc.noaa.gov/"""
    REQUIRES_CREDENTIALS = False

    KP_URL = TrustedEndpoints.NOAA_SWPC_KP_PRODUCTS
    SOLAR_WIND_URL = TrustedEndpoints.NOAA_SWPC_SOLAR_WIND
    XRAY_URL = TrustedEndpoints.NOAA_SWPC_XRAYS
    PROTON_URL = TrustedEndpoints.NOAA_SWPC_PROTONS_7DAY

    FEATURE_NAMES = [
        "kp_estimated",
        "solar_wind_speed_km_s",
        "solar_wind_density_p_cm3",
        "solar_wind_temperature_k",
        "imf_bx_nt",
        "imf_by_nt",
        "imf_bz_nt",
        "imf_bt_nt",
        "xray_flux_short_w_m2",
        "xray_flux_long_w_m2",
        "proton_flux_10mev_pfu",
        "proton_flux_100mev_pfu",
        "magnetopause_standoff_re",
        "hour_of_day",
        "day_of_year",
    ]

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize the instance."""
        super().__init__(config)
        self.storm_threshold = float(config.preprocessing.get("storm_threshold", 5.0))
        self._is_real_data = False

    @property
    def is_real_data(self) -> bool:
        """Return True if real data was loaded."""
        return self._is_real_data

    def _real_cache_name(self) -> str:
        """Cache name carries the threshold: labels are baked into the matrix."""
        return f"noaa_space_weather_real_kp{self.storm_threshold:g}.npz"

    def _synthetic_cache_name(self) -> str:
        return f"noaa_space_weather_synthetic_kp{self.storm_threshold:g}.npz"

    def download(self) -> bool:
        """Fetch the four SWPC products and cache the aligned hourly matrix."""
        if self._download_from_swpc():
            return True
        if ALLOW_SYNTHETIC:
            check_synthetic_allowed("NOAASpaceWeather", "SWPC API failed")
            return self._create_synthetic()
        raise DataSourceUnavailableError(
            loader_name="NOAASpaceWeather",
            source_url=self.KP_URL,
            reason="SWPC API failed",
        )

    def _fetch_json(self, url: str) -> Any:
        TrustedEndpoints.validate_url(url)
        content = http_get_with_retry(url, timeout=120, timeout_per_attempt=30, retries=1)
        return json.loads(content.decode("utf-8"))

    def _download_from_swpc(self) -> bool:
        cache_file = self.data_path / self._real_cache_name()
        if cache_file.exists():
            logger.info(f"NOAA space-weather data already cached at {cache_file}")
            self._is_real_data = True
            return True
        try:
            kp = parse_kp_rows(self._fetch_json(self.KP_URL))
            solar_wind = parse_solar_wind_rows(self._fetch_json(self.SOLAR_WIND_URL))
            xray = parse_xray_rows(self._fetch_json(self.XRAY_URL))
            protons = parse_proton_rows(self._fetch_json(self.PROTON_URL))
            features, labels = build_hourly_matrix(
                kp, solar_wind, xray, protons, self.storm_threshold
            )
            if len(features) == 0:
                logger.warning("SWPC products fetched but no hour was observed by every instrument")
                return False
            np.savez_compressed(cache_file, features=features, labels=labels)
            self._is_real_data = True
            logger.info(
                f"NOAA space weather loaded: {len(features)} fully-observed hours, "
                f"{int(labels.sum())} storm hours (is_real_data=True)"
            )
            return True
        except Exception as e:
            logger.warning(f"SWPC download failed: {e}")
            return False

    def _create_synthetic(self) -> bool:
        """Create a labelled synthetic matrix (policy-gated callers only)."""
        rng = np.random.default_rng(self.config.random_seed)
        n = self.config.max_samples or 5000
        kp = np.clip(rng.exponential(scale=1.5, size=n), 0, 9)
        speed = np.clip(rng.normal(450, 100, size=n) + kp * 30, 250, 1200)
        density = np.clip(rng.exponential(scale=5, size=n), 0.1, 100)
        temperature = np.clip(rng.normal(1e5, 4e4, size=n), 1e4, 1e6)
        bx = rng.normal(0, 4, size=n)
        by = rng.normal(0, 4, size=n)
        bz = np.clip(rng.normal(0, 5, size=n) - kp * 0.5, -30, 30)
        bt = np.sqrt(bx**2 + by**2 + bz**2)
        xray_short = rng.exponential(scale=1e-9, size=n) * (1 + kp)
        xray_long = rng.exponential(scale=1e-7, size=n) * (1 + kp)
        p10 = rng.exponential(scale=1.0, size=n) * (1 + kp * 0.5)
        p100 = rng.exponential(scale=0.05, size=n) * (1 + kp * 0.2)
        standoff = np.array(
            [
                shue_magnetopause_standoff_re(float(d), float(v), float(z))
                for d, v, z in zip(density, speed, bz, strict=True)
            ]
        )
        hour = rng.integers(0, 24, size=n).astype(np.float64)
        day = rng.integers(1, 366, size=n).astype(np.float64)
        features = np.column_stack(
            [
                kp,
                speed,
                density,
                temperature,
                bx,
                by,
                bz,
                bt,
                xray_short,
                xray_long,
                p10,
                p100,
                standoff,
                hour,
                day,
            ]
        ).astype(np.float64)
        labels = (kp >= self.storm_threshold).astype(np.int64)
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
            f"No cached NOAA space-weather data under {self.data_path}; call download() first."
        )

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """No-op: rows are already aligned, fully observed and finite."""
        return data


# Register at import so DatasetRegistry-driven callers can construct by name.
from omni_mercury_engine.datasets.base import DatasetRegistry

DatasetRegistry.register("noaa_space_weather", NOAASpaceWeatherLoader)
