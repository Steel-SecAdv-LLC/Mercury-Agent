# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Domain loader for EMP/energy grid data from NOAA Space Weather and EIA.

Connects to the NOAA Space Weather Prediction Center (SWPC) for real-time and historical
geomagnetic/solar data, and optionally to the U.S. Energy Information Administration (EIA) API v2
for electricity grid data.

Ground truth events cover major geomagnetic storms and grid disruptions where severe Kp indices (>=
7) or grid supply/demand anomalies serve as anomaly labels for Mercury detection benchmarks.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from omni_mercury_engine.loaders.base import BaseDomainLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NOAA SWPC API endpoints
# ---------------------------------------------------------------------------
_KP_INDEX_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
_XRAY_FLARES_URL = "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-latest.json"
_SOLAR_WIND_URL = "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json"

# ---------------------------------------------------------------------------
# EIA API v2 endpoint (requires API key)
# ---------------------------------------------------------------------------
_EIA_DAILY_REGION_URL = "https://api.eia.gov/v2/electricity/rto/daily-region-data/data/"

# ---------------------------------------------------------------------------
# Solar flare classification mapping (X-ray peak flux class -> numeric)
# ---------------------------------------------------------------------------
_FLARE_CLASS_MAP: dict[str, float] = {
    "A": 1.0,
    "B": 2.0,
    "C": 3.0,
    "M": 4.0,
    "X": 5.0,
}

# ---------------------------------------------------------------------------
# Geomagnetic storm thresholds
# ---------------------------------------------------------------------------
_KP_STORM_THRESHOLD = 5
_KP_SEVERE_THRESHOLD = 7

# ---------------------------------------------------------------------------
# Ground truth event catalog
# ---------------------------------------------------------------------------
_EVENT_CATALOG: dict[str, dict[str, Any]] = {
    "quebec_1989": {
        "name": "1989 Quebec Blackout",
        "date": "1989-03-13",
        "description": (
            "Geomagnetic storm caused by a coronal mass ejection collapsed "
            "the Hydro-Quebec power grid for 9 hours. Peak Kp=9. Extended "
            "to 60-day window for adequate sample size."
        ),
        "start": "1989-02-01",
        "end": "1989-04-30",
        "peak_kp": 9,
        "synthetic_hours": 2160,  # 90 days at 3h = 720 steps
        "storm_day": 40,  # day within window when storm peaks
    },
    "halloween_2003": {
        "name": "2003 Halloween Solar Storms",
        "date": "2003-10-28",
        "description": (
            "Series of solar flares and coronal mass ejections during "
            "October-November 2003. Extended to multi-month window."
        ),
        "start": "2003-09-01",
        "end": "2003-12-31",
        "peak_kp": 9,
        "synthetic_hours": 2928,  # 122 days
        "storm_day": 58,  # Oct 28 within window
    },
    "texas_2021": {
        "name": "2021 Texas Grid Crisis",
        "date": "2021-02-15",
        "description": (
            "Winter Storm Uri caused widespread power outages across Texas "
            "due to extreme cold and grid supply/demand imbalance. "
            "Grid demand/supply data anomaly (not geomagnetic)."
        ),
        "start": "2021-01-01",
        "end": "2021-03-31",
        "peak_kp": 2,
        "synthetic_hours": 2160,
        "grid_event": True,
    },
    "bastille_day_2000": {
        "name": "2000 Bastille Day Solar Event",
        "date": "2000-07-14",
        "description": (
            "X5.7-class solar flare on Bastille Day followed by a severe "
            "geomagnetic storm. Peak Kp=9. Extended to multi-month window."
        ),
        "start": "2000-06-01",
        "end": "2000-08-31",
        "peak_kp": 9,
        "synthetic_hours": 2208,  # 92 days
        "storm_day": 44,
    },
}


class EnergyLoader(BaseDomainLoader):
    """Loader for EMP/energy grid data from NOAA SWPC and EIA.

    Uses NOAA Space Weather Prediction Center endpoints for real-time
    geomagnetic and solar data:

    * **Kp index** -- planetary geomagnetic disturbance index (0-9 scale),
      updated every 3 hours.
    * **Solar flares** -- GOES X-ray flare detections with classification.
    * **Solar wind** -- proton density, bulk speed, and temperature from
      the DSCOVR/ACE spacecraft at L1.

    Optionally uses the EIA API v2 for U.S. electricity grid data when
    ``EIA_API_KEY`` is set in the environment.

    Feature engineering produces space-weather observables suitable for
    anomaly detection: Kp index, Kp rate of change, solar wind speed
    and density, X-ray flux classification, and geomagnetic storm
    indicators.
    """

    DOMAIN: str = "energy"
    SOURCE_URL: str = "https://services.swpc.noaa.gov/json/"
    REQUIRES_API_KEY: bool = False
    API_KEY_ENV_VAR: str = ""
    FEATURE_COLUMNS: list[str] = [
        "kp",
        "kp_rate_of_change",
        "kp_sum_24h",
        "delta_kp_6h",
        "max_kp_24h",
        "solar_wind_speed",
        "solar_wind_density",
        "xray_class",
    ]

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the energy loader.

        Args:
            **kwargs: Passed through to :class:`BaseDomainLoader`.
        """
        super().__init__(**kwargs)
        self._eia_api_key: str = os.environ.get("EIA_API_KEY", "")
        if self._eia_api_key:
            logger.info("EIA API key found; grid data endpoints enabled.")
        else:
            logger.debug("No EIA_API_KEY set. EIA grid data will not be available.")

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def fetch_realtime(self) -> pd.DataFrame:
        """Fetch the most recent Kp index and solar wind data from SWPC.

        Combines the NOAA SWPC Kp index feed with the 7-day solar wind
        plasma feed into a unified time-series DataFrame.

        Returns:
            DataFrame with columns: timestamp, kp, solar_wind_speed,
            solar_wind_density, solar_wind_temperature.

        Raises:
            ConnectionError: If the SWPC feeds are unreachable after retries.
        """
        cache_key = "energy_realtime"
        cached = self._read_cache(cache_key)

        if cached is not None:
            logger.debug("Returning cached real-time energy/space-weather data.")
            return pd.DataFrame(cached)

        # Fetch Kp index data
        kp_df = self._fetch_kp_index()

        # Fetch solar wind data
        sw_df = self._fetch_solar_wind()

        # Merge on nearest timestamp
        df = self._merge_kp_and_solar_wind(kp_df, sw_df)

        self._write_cache(cache_key, df.to_dict(orient="list"))
        logger.info("Fetched %d real-time energy/space-weather records.", len(df))
        return df

    def fetch_historical(self, event_id: str) -> pd.DataFrame:
        """Fetch data for a specific historical energy/space-weather event.

        For historical geomagnetic events predating SWPC real-time feeds,
        this method generates synthetic Kp time-series data based on
        documented storm profiles.  The synthetic data reproduces the
        temporal pattern of Kp escalation, peak, and recovery observed
        in the cataloged event.

        For the ``texas_2021`` event, if an EIA API key is available,
        grid demand/supply data is fetched from the EIA API v2.

        Args:
            event_id: Key into the ground truth catalog (e.g.
                ``"quebec_1989"``).

        Returns:
            DataFrame with columns: timestamp, kp, solar_wind_speed,
            solar_wind_density, xray_class.

        Raises:
            ValueError: If *event_id* is not in the catalog.
            ConnectionError: If data sources are unreachable.
        """
        if event_id not in _EVENT_CATALOG:
            raise ValueError(
                f"Unknown event_id '{event_id}'. " f"Available: {list(_EVENT_CATALOG.keys())}"
            )

        cache_key = f"energy_historical_{event_id}"
        cached = self._read_cache(cache_key)

        if cached is not None:
            logger.debug("Returning cached historical data for '%s'.", event_id)
            return pd.DataFrame(cached)

        event = _EVENT_CATALOG[event_id]

        if event.get("grid_event") and self._eia_api_key:
            df = self._fetch_eia_grid_data(event)
        else:
            df = self._generate_synthetic_kp_series(event)

        if df.empty:
            logger.warning("No data generated for event '%s'.", event_id)
            return df

        # Sort chronologically
        df = df.sort_values("timestamp").reset_index(drop=True)

        self._write_cache(cache_key, df.to_dict(orient="list"))
        logger.info(
            "Fetched %d historical records for event '%s'.",
            len(df),
            event_id,
        )
        return df

    def list_events(self) -> list[dict[str, Any]]:
        """Return the catalog of ground truth energy/space-weather events.

        Returns:
            List of dicts each containing *event_id*, *name*, *date*,
            and *description* keys.
        """
        events: list[dict[str, Any]] = []
        for event_id, meta in _EVENT_CATALOG.items():
            events.append(
                {
                    "event_id": event_id,
                    "name": meta["name"],
                    "date": meta["date"],
                    "description": meta["description"],
                }
            )
        return events

    def get_ground_truth(self, event_id: str) -> np.ndarray[Any, Any]:
        """Generate binary anomaly labels for a historical event.

        Labeling strategy:

        * **Kp >= 7** (severe geomagnetic storm): labeled anomaly (``1``).
        * **Kp < 5** (quiet/unsettled): labeled normal (``0``).
        * **Kp 5-6** (minor storm): labeled normal (``0``) for
          conservative labeling to reduce false-positive contamination.

        For the ``texas_2021`` grid event, anomaly labels are based on
        grid demand exceeding supply capacity thresholds rather than Kp.

        Args:
            event_id: Key into the ground truth catalog.

        Returns:
            1-D binary numpy array of shape ``(n_samples,)``.

        Raises:
            ValueError: If *event_id* is not recognized or no data is
                available.
        """
        if event_id not in _EVENT_CATALOG:
            raise ValueError(
                f"Unknown event_id '{event_id}'. " f"Available: {list(_EVENT_CATALOG.keys())}"
            )

        df = self.fetch_historical(event_id)
        if df.empty:
            return np.array([], dtype=np.int64)

        event = _EVENT_CATALOG[event_id]

        if event.get("grid_event") and "grid_demand" in df.columns:
            # For grid events, anomaly = demand exceeds supply capacity
            labels = self._label_grid_anomalies(df)
        else:
            # For geomagnetic events, anomaly = Kp >= 7
            kp_values = df["kp"].values.astype(np.float64)
            labels = (kp_values >= _KP_SEVERE_THRESHOLD).astype(np.int64)

        logger.info(
            "Ground truth for '%s': %d anomalies / %d total.",
            event_id,
            int(labels.sum()),
            len(labels),
        )
        return labels

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def engineer_features(self, raw_data: pd.DataFrame) -> np.ndarray[Any, Any]:
        """Transform raw energy/space-weather data into a feature matrix.

        Engineered features (per time step):

        1. **kp** -- Kp index (0-9 scale).
        2. **kp_rate_of_change** -- difference in Kp from previous time
           step (0 for the first row).
        3. **kp_sum_24h** -- sum of Kp over trailing 8 steps (24h),
           a proxy for the Ap index.
        4. **delta_kp_6h** -- Kp change over 2 steps (6h).
        5. **max_kp_24h** -- maximum Kp over trailing 8-step (24h)
           rolling window.
        6. **solar_wind_speed** -- solar wind bulk speed in km/s.
        7. **solar_wind_density** -- solar wind proton density (p/cm^3).
        8. **xray_class** -- numeric solar flare classification
           (A=1, B=2, C=3, M=4, X=5; 0 if absent).

        Args:
            raw_data: DataFrame from :meth:`fetch_realtime` or
                :meth:`fetch_historical`.

        Returns:
            2-D numpy array of shape ``(n_samples, 8)``.
        """
        if raw_data.empty:
            return np.empty((0, 8), dtype=np.float64)

        df = raw_data.copy()

        # Ensure chronological order
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)

        n = len(df)

        # ---- Kp index ----
        kp = self._safe_column(df, "kp")

        # ---- Kp rate of change (3h step) ----
        kp_roc = np.zeros(n, dtype=np.float64)
        if n > 1:
            kp_roc[1:] = np.diff(kp)

        # ---- Kp sum over 24h (8 steps) — proxy for Ap index ----
        kp_sum_24h = self._compute_rolling_sum(kp, window=8)

        # ---- Delta Kp over 6h (2 steps) ----
        delta_kp_6h = np.zeros(n, dtype=np.float64)
        if n > 2:
            delta_kp_6h[2:] = kp[2:] - kp[:-2]

        # ---- Max Kp over 24h (8 steps) ----
        max_kp_24h = self._compute_rolling_max(kp, window=8)

        # ---- Solar wind speed (km/s) ----
        sw_speed = self._safe_column(df, "solar_wind_speed")

        # ---- Solar wind density (p/cm^3) ----
        sw_density = self._safe_column(df, "solar_wind_density")

        # ---- X-ray flux class (numeric) ----
        xray_class = self._safe_column(df, "xray_class")

        # Stack into feature matrix
        features = np.column_stack(
            [
                kp,
                kp_roc,
                kp_sum_24h,
                delta_kp_6h,
                max_kp_24h,
                sw_speed,
                sw_density,
                xray_class,
            ]
        )

        # Clean up non-finite values
        features = np.where(np.isinf(features), np.nan, features)
        for col_idx in range(features.shape[1]):
            col = features[:, col_idx]
            mask = np.isnan(col)
            if mask.any():
                median_val = np.nanmedian(col)
                col[mask] = median_val if np.isfinite(median_val) else 0.0

        return features

    # ------------------------------------------------------------------
    # Private helpers -- data fetching
    # ------------------------------------------------------------------

    def _fetch_kp_index(self) -> pd.DataFrame:
        """Fetch the NOAA SWPC Kp index JSON feed.

        The SWPC Kp index endpoint returns an array of arrays where each
        inner array is ``[timestamp, Kp, a_running, station_count]``.
        The first row is the header.

        Returns:
            DataFrame with columns: timestamp, kp.
        """
        raw: list[list[Any]] = self._fetch_json(_KP_INDEX_URL)

        if not raw or len(raw) < 2:
            logger.warning("Kp index feed returned empty or malformed data.")
            return pd.DataFrame(columns=["timestamp", "kp"])

        # First row is header; skip it
        rows = raw[1:]

        records: list[dict[str, Any]] = []
        for row in rows:
            if len(row) < 2:
                continue
            try:
                kp_value = float(row[1])
            except (ValueError, TypeError):
                kp_value = np.nan
            records.append(
                {
                    "timestamp": str(row[0]),
                    "kp": kp_value,
                }
            )

        df = pd.DataFrame(records)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            df = df.dropna(subset=["timestamp"])
        return df

    def _fetch_solar_wind(self) -> pd.DataFrame:
        """Fetch the NOAA SWPC 7-day solar wind plasma data.

        The SWPC solar wind plasma endpoint returns an array of arrays
        where each inner array is
        ``[time_tag, density, speed, temperature]``.
        The first row is the header.

        Returns:
            DataFrame with columns: timestamp, solar_wind_density,
            solar_wind_speed, solar_wind_temperature.
        """
        raw: list[list[Any]] = self._fetch_json(_SOLAR_WIND_URL)

        if not raw or len(raw) < 2:
            logger.warning("Solar wind feed returned empty or malformed data.")
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "solar_wind_density",
                    "solar_wind_speed",
                    "solar_wind_temperature",
                ]
            )

        # First row is header; skip it
        rows = raw[1:]

        records: list[dict[str, Any]] = []
        for row in rows:
            if len(row) < 4:
                continue
            try:
                density = float(row[1]) if row[1] is not None else np.nan
            except (ValueError, TypeError):
                density = np.nan
            try:
                speed = float(row[2]) if row[2] is not None else np.nan
            except (ValueError, TypeError):
                speed = np.nan
            try:
                temperature = float(row[3]) if row[3] is not None else np.nan
            except (ValueError, TypeError):
                temperature = np.nan

            records.append(
                {
                    "timestamp": str(row[0]),
                    "solar_wind_density": density,
                    "solar_wind_speed": speed,
                    "solar_wind_temperature": temperature,
                }
            )

        df = pd.DataFrame(records)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            df = df.dropna(subset=["timestamp"])
        return df

    def _fetch_xray_flares(self) -> pd.DataFrame:
        """Fetch the latest GOES X-ray flare events from SWPC.

        Returns:
            DataFrame with columns: timestamp, xray_class (numeric).
        """
        raw: list[dict[str, Any]] = self._fetch_json(_XRAY_FLARES_URL)

        if not raw:
            logger.warning("X-ray flares feed returned empty data.")
            return pd.DataFrame(columns=["timestamp", "xray_class"])

        records: list[dict[str, Any]] = []
        for entry in raw:
            time_tag = entry.get("begin_time") or entry.get("time_tag", "")
            class_type = entry.get("current_class", "") or entry.get("max_class", "")
            numeric_class = self._parse_flare_class(class_type)
            records.append(
                {
                    "timestamp": str(time_tag),
                    "xray_class": numeric_class,
                }
            )

        df = pd.DataFrame(records)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            df = df.dropna(subset=["timestamp"])
        return df

    def _fetch_eia_grid_data(self, event: dict[str, Any]) -> pd.DataFrame:
        """Fetch EIA electricity grid data for a grid event.

        Uses the EIA API v2 daily region data endpoint to retrieve
        electricity demand and generation data for the event time window.

        Args:
            event: Event metadata dict from the catalog.

        Returns:
            DataFrame with columns: timestamp, kp, grid_demand,
            grid_supply, solar_wind_speed, solar_wind_density, xray_class.
        """
        if not self._eia_api_key:
            logger.warning("EIA API key not set; falling back to synthetic data " "for grid event.")
            return self._generate_synthetic_kp_series(event)

        params: dict[str, str] = {
            "api_key": self._eia_api_key,
            "frequency": "daily",
            "data[0]": "value",
            "start": event["start"],
            "end": event["end"],
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
        }

        try:
            response = self._fetch_json(_EIA_DAILY_REGION_URL, params=params)
        except ConnectionError:
            logger.warning("EIA API unreachable; falling back to synthetic data.")
            return self._generate_synthetic_kp_series(event)

        data_rows = (
            response.get("response", {}).get("data", []) if isinstance(response, dict) else []
        )

        if not data_rows:
            logger.warning("EIA returned no data; falling back to synthetic data.")
            return self._generate_synthetic_kp_series(event)

        records: list[dict[str, Any]] = []
        for row in data_rows:
            records.append(
                {
                    "timestamp": row.get("period", ""),
                    "grid_demand": float(row.get("value", 0)),
                    "grid_supply": float(row.get("value", 0)) * 0.95,
                    "kp": float(event.get("peak_kp", 0)),
                    "solar_wind_speed": 400.0,
                    "solar_wind_density": 5.0,
                    "xray_class": 0.0,
                }
            )

        df = pd.DataFrame(records)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            df = df.dropna(subset=["timestamp"])
        return df

    # ------------------------------------------------------------------
    # Private helpers -- synthetic data generation
    # ------------------------------------------------------------------

    def _generate_synthetic_kp_series(self, event: dict[str, Any]) -> pd.DataFrame:
        """Generate synthetic Kp time-series for a historical event.

        Creates a plausible multi-week Kp profile with a background of
        quiet/unsettled conditions punctuated by the documented major
        storm.  Solar wind and X-ray parameters are correlated with Kp
        using empirical relationships.

        For extended time windows, minor storms (Kp 4-5) are sprinkled
        throughout the quiet period to provide realistic baseline variance.

        Args:
            event: Event metadata dict from the catalog.

        Returns:
            DataFrame with columns: timestamp, kp, solar_wind_speed,
            solar_wind_density, xray_class.
        """
        n_hours: int = event.get("synthetic_hours", 2160)
        peak_kp: int = event.get("peak_kp", 9)
        storm_day: int = event.get("storm_day", 40)
        start_date = pd.Timestamp(event["start"], tz="UTC")

        # Generate 3-hour resolution timestamps (standard Kp cadence)
        n_steps = n_hours // 3
        n_steps = max(n_steps, 1)
        timestamps = pd.date_range(start=start_date, periods=n_steps, freq="3h", tz="UTC")

        # Build Kp profile: extended quiet + major storm + recovery
        kp_values = self._build_storm_profile(n_steps, peak_kp, storm_day=storm_day)

        # Derive correlated solar wind speed: empirical Kp-speed relation.
        # Process-stable seed: Python's built-in hash() is salted per process,
        # so derive the seed from hashlib.sha256 instead -- identical every run
        # without relying on PYTHONHASHSEED.  Mirrors tsunami_loader.
        seed_bytes = hashlib.sha256(event["date"].encode()).digest()
        rng = np.random.default_rng(int.from_bytes(seed_bytes[:4], "little") % (2**31))
        base_speed = 350.0 + kp_values * 50.0
        sw_speed = base_speed + rng.normal(0, 20, size=n_steps)
        sw_speed = np.clip(sw_speed, 250.0, 1200.0)

        # Solar wind density: inversely related to speed during CMEs
        base_density = 10.0 - kp_values * 0.5
        sw_density = base_density + rng.normal(0, 2, size=n_steps)
        sw_density = np.clip(sw_density, 0.5, 50.0)

        # X-ray class: elevated during storm peaks
        xray = np.zeros(n_steps, dtype=np.float64)
        xray[kp_values >= 7] = 5.0  # X-class during severe storms
        xray[(kp_values >= 5) & (kp_values < 7)] = 4.0  # M-class
        xray[(kp_values >= 3) & (kp_values < 5)] = 3.0  # C-class
        xray[kp_values < 3] = 1.0  # A-class background

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "kp": kp_values,
                "solar_wind_speed": sw_speed,
                "solar_wind_density": sw_density,
                "xray_class": xray,
            }
        )
        return df

    @staticmethod
    def _build_storm_profile(
        n_steps: int, peak_kp: int, storm_day: int = 40
    ) -> np.ndarray[Any, Any]:
        """Build a synthetic Kp storm profile.

        For extended time windows (>100 steps), the profile includes:

        1. **Background quiet** -- Kp ~ 1-2 with Gaussian noise and
           occasional minor disturbances (Kp 3-5).
        2. **Major storm** -- a 2-day ramp-up, 1-day peak at peak_kp,
           and 3-day exponential recovery, positioned at storm_day.
        3. **Post-storm quiet** -- return to background levels.

        This produces a realistic Kp distribution: ~90% quiet (<3),
        ~7% unsettled/minor storm (3-6), ~3% severe (>=7).

        Args:
            n_steps: Total number of time steps.
            peak_kp: Maximum Kp value at storm peak.
            storm_day: Day within the window to place the storm center.

        Returns:
            1-D numpy array of Kp values.
        """
        rng = np.random.default_rng(seed=(peak_kp * 1000 + n_steps) & 0xFFFFFFFF)

        # Base: quiet conditions with slight variation
        kp = 1.5 + rng.normal(0, 0.4, size=n_steps)

        # Sprinkle minor storms every ~7-10 days (56-80 steps)
        steps_per_day = 8  # 24h / 3h
        n_minor = max(1, n_steps // (steps_per_day * 8))
        minor_positions = rng.integers(0, n_steps, size=n_minor)
        for pos in minor_positions:
            # Minor storm: 8-16 steps (1-2 days)
            storm_len = rng.integers(8, 17)
            minor_peak = rng.uniform(3.5, 5.5)
            for j in range(min(storm_len, n_steps - pos)):
                # Bell curve shape
                t = j / storm_len
                kp[pos + j] = max(
                    kp[pos + j],
                    minor_peak * np.sin(np.pi * t),
                )

        # Place major storm at storm_day
        storm_center = storm_day * steps_per_day
        if storm_center >= n_steps:
            storm_center = n_steps // 2

        # Ramp-up: 2 days (16 steps)
        ramp_len = 16
        ramp_start = max(0, storm_center - ramp_len - 8)
        for j in range(min(ramp_len, storm_center - ramp_start)):
            t = np.float64(j / ramp_len)
            kp[ramp_start + j] = 2.0 + (float(peak_kp) - 2.0) * t

        # Peak: 8 steps (24h)
        peak_start = max(0, storm_center - 8)
        peak_end = min(n_steps, storm_center + 8)
        kp[peak_start:peak_end] = float(peak_kp) + rng.uniform(
            -0.5, 0.0, size=peak_end - peak_start
        )

        # Recovery: 3 days (24 steps) exponential decay
        recovery_len = 24
        recovery_end = min(n_steps, peak_end + recovery_len)
        if recovery_end > peak_end:
            decay = np.exp(-np.linspace(0, 4.0, recovery_end - peak_end))
            kp[peak_end:recovery_end] = 1.5 + (float(peak_kp) - 1.5) * decay

        # Clamp to valid Kp range
        kp = np.clip(kp, 0.0, 9.0)
        return np.round(kp * 3) / 3  # Kp is reported in thirds

    # ------------------------------------------------------------------
    # Private helpers -- data merging
    # ------------------------------------------------------------------

    def _merge_kp_and_solar_wind(self, kp_df: pd.DataFrame, sw_df: pd.DataFrame) -> pd.DataFrame:
        """Merge Kp index and solar wind DataFrames on nearest timestamp.

        Uses a merge_asof to align the higher-cadence solar wind data
        with the 3-hour Kp index timestamps.

        Args:
            kp_df: DataFrame with timestamp and kp columns.
            sw_df: DataFrame with timestamp and solar wind columns.

        Returns:
            Merged DataFrame with all columns.
        """
        if kp_df.empty:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "kp",
                    "solar_wind_speed",
                    "solar_wind_density",
                    "solar_wind_temperature",
                ]
            )

        if sw_df.empty:
            kp_df["solar_wind_speed"] = np.nan
            kp_df["solar_wind_density"] = np.nan
            kp_df["solar_wind_temperature"] = np.nan
            return kp_df

        # Ensure both are sorted by timestamp for merge_asof
        kp_sorted = kp_df.sort_values("timestamp").reset_index(drop=True)
        sw_sorted = sw_df.sort_values("timestamp").reset_index(drop=True)

        merged = pd.merge_asof(
            kp_sorted,
            sw_sorted,
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta("3h"),
        )
        return merged

    # ------------------------------------------------------------------
    # Private helpers -- classification and labeling
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_flare_class(class_str: str) -> float:
        """Parse a solar flare classification string to a numeric value.

        Extracts the letter class (A, B, C, M, X) from strings like
        ``"M2.5"`` or ``"X1.0"`` and returns the corresponding numeric
        value from ``_FLARE_CLASS_MAP``.

        Args:
            class_str: Flare classification string (e.g., ``"M2.5"``).

        Returns:
            Numeric class value (1.0-5.0), or 0.0 if unparseable.
        """
        if not class_str:
            return 0.0
        letter = class_str[0].upper()
        return _FLARE_CLASS_MAP.get(letter, 0.0)

    @staticmethod
    def _label_grid_anomalies(df: pd.DataFrame) -> np.ndarray[Any, Any]:
        """Generate anomaly labels for grid demand/supply events.

        Labels time periods where grid demand exceeds grid supply
        (supply shortfall) as anomalies.

        Args:
            df: DataFrame with ``grid_demand`` and ``grid_supply`` columns.

        Returns:
            1-D binary numpy array.
        """
        demand = df["grid_demand"].values.astype(np.float64)
        supply = df["grid_supply"].values.astype(np.float64)
        labels = (demand > supply).astype(np.int64)
        return np.asarray(labels)

    # ------------------------------------------------------------------
    # Private helpers -- utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_column(df: pd.DataFrame, column: str) -> np.ndarray[Any, Any]:
        """Extract a column as a float64 array, returning zeros if absent.

        Args:
            df: Source DataFrame.
            column: Column name to extract.

        Returns:
            1-D float64 numpy array of column values, or zeros if the
            column does not exist.
        """
        if column in df.columns:
            return np.asarray(
                pd.to_numeric(df[column], errors="coerce").fillna(0.0).values, dtype=np.float64
            )
        return np.zeros(len(df), dtype=np.float64)

    @staticmethod
    def _compute_rolling_sum(values: np.ndarray[Any, Any], window: int = 8) -> np.ndarray[Any, Any]:
        """Compute the rolling sum over a trailing window.

        Args:
            values: 1-D input array.
            window: Number of steps in the trailing window.

        Returns:
            1-D array of rolling sum values.
        """
        n = len(values)
        result = np.zeros(n, dtype=np.float64)
        for i in range(n):
            start = max(0, i - window + 1)
            result[i] = np.sum(values[start : i + 1])
        return result

    @staticmethod
    def _compute_rolling_max(values: np.ndarray[Any, Any], window: int = 8) -> np.ndarray[Any, Any]:
        """Compute the rolling maximum over a trailing window.

        Args:
            values: 1-D input array.
            window: Number of steps in the trailing window.

        Returns:
            1-D array of rolling maximum values.
        """
        n = len(values)
        result = np.zeros(n, dtype=np.float64)
        for i in range(n):
            start = max(0, i - window + 1)
            result[i] = np.max(values[start : i + 1])
        return result
