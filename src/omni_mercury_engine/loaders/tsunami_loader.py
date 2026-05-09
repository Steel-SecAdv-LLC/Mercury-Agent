"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see
https://www.gnu.org/licenses/.

Domain loader for tsunami data from NOAA NDBC DART buoys.

Fetches real-time and historical bottom pressure recorder (BPR)
data from the Deep-ocean Assessment and Reporting of Tsunamis
(DART) network operated by NOAA's National Data Buoy Center.

Features engineered from raw BPR readings include tidal deviation,
rate of sea-level change, and rolling variability measures.  Ground
truth labels are derived from documented tsunami arrival windows
at specific DART stations.
"""

from __future__ import annotations

import io
import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from omni_mercury_engine.loaders.base import BaseDomainLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DART station registry
# ---------------------------------------------------------------------------

#: Key DART buoy station IDs used for tsunami monitoring.
DART_STATIONS: list[str] = [
    "46402",
    "46407",
    "46410",
    "46413",
    "51407",
    "32401",
    "32412",
    "21413",
    "21418",
    "52402",
]

# ---------------------------------------------------------------------------
# Ground-truth event catalog
# ---------------------------------------------------------------------------

_EVENT_CATALOG: list[dict[str, Any]] = [
    {
        "event_id": "tohoku_2011",
        "name": "2011 Tohoku Earthquake Tsunami",
        "date": "2011-03-11",
        "description": (
            "Magnitude 9.1 earthquake off the Pacific coast of Tohoku, Japan. "
            "Generated a devastating tsunami with waves exceeding 40 m in some "
            "coastal areas.  DART station 21418 recorded major wave signatures."
        ),
        "station_id": "21418",
        "window_start": "2011-03-11T05:46:00Z",
        "window_end": "2011-03-11T12:00:00Z",
    },
    {
        "event_id": "chile_2010",
        "name": "2010 Chile Earthquake Tsunami",
        "date": "2010-02-27",
        "description": (
            "Magnitude 8.8 earthquake off the coast of central Chile. "
            "Generated a Pacific-wide tsunami.  DART station 32412 recorded "
            "significant pressure deviations."
        ),
        "station_id": "32412",
        "window_start": "2010-02-27T06:34:00Z",
        "window_end": "2010-02-27T14:00:00Z",
    },
    {
        "event_id": "tonga_2022",
        "name": "2022 Tonga Eruption Tsunami",
        "date": "2022-01-15",
        "description": (
            "Submarine eruption of Hunga Tonga-Hunga Ha'apai volcano generated "
            "a trans-oceanic tsunami and atmospheric pressure wave.  DART stations "
            "51407 and 52402 captured the event."
        ),
        "station_id": "51407",
        "secondary_station_id": "52402",
        "window_start": "2022-01-15T04:15:00Z",
        "window_end": "2022-01-15T16:00:00Z",
    },
]

# Lookup for fast access by event_id.
_EVENTS_BY_ID: dict[str, dict[str, Any]] = {e["event_id"]: e for e in _EVENT_CATALOG}

# ---------------------------------------------------------------------------
# Feature engineering parameters
# ---------------------------------------------------------------------------

#: Window size (number of time steps) for the rolling standard deviation.
_ROLLING_STD_WINDOW: int = 20

#: Window size for the simple moving average used for tidal detrending.
_SMA_DETREND_WINDOW: int = 60


class TsunamiLoader(BaseDomainLoader):
    """
    Domain loader for NOAA NDBC DART tsunami buoy data.

    Connects to the NDBC real-time data service and retrieves bottom
    pressure recorder (BPR) readings from the DART network.  For
    historical ground-truth events the loader synthesises labeled
    datasets based on documented tsunami arrival windows.

    Attributes:
        DOMAIN: ``"tsunami"``
        SOURCE_URL: NDBC real-time data endpoint.
        REQUIRES_API_KEY: ``False`` -- NDBC data is freely available.
    """

    DOMAIN: str = "tsunami"
    SOURCE_URL: str = "https://www.ndbc.noaa.gov/data/realtime2/"
    REQUIRES_API_KEY: bool = False
    FEATURE_COLUMNS: list[str] = [
        "bpr",
        "tidal_deviation",
        "abs_deviation",
        "rate_of_change",
        "rolling_std",
        "short_energy",
    ]

    # Cache historical event data for 24 hours (events are static).
    CACHE_TTL: int = 86400

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def fetch_realtime(self) -> pd.DataFrame:
        """
        Fetch the most recent real-time DART data for all key stations.

        Iterates over :pydata:`DART_STATIONS`, downloads the current
        ``.dart`` file from NDBC and concatenates the results into a
        single :class:`~pandas.DataFrame`.

        Returns:
            DataFrame with columns ``station_id``, ``timestamp``, and
            the raw measurement columns provided by NDBC.

        Raises:
            ConnectionError: If no station data could be retrieved after
                exhausting retries.
        """
        frames: list[pd.DataFrame] = []
        errors: list[str] = []

        for station_id in DART_STATIONS:
            try:
                df = self._fetch_station(station_id)
                if not df.empty:
                    frames.append(df)
            except Exception as exc:
                msg = f"Station {station_id}: {exc}"
                logger.warning("tsunami: failed to fetch %s", msg)
                errors.append(msg)

        if not frames:
            raise ConnectionError(
                f"tsunami: could not retrieve data from any DART station. Errors: {errors}"
            )

        combined = pd.concat(frames, ignore_index=True)
        logger.info(
            "tsunami: fetched real-time data — %d rows from %d stations",
            len(combined),
            len(frames),
        )
        return combined

    def fetch_historical(self, event_id: str) -> pd.DataFrame:
        """
        Fetch data for a specific historical tsunami event.

        Because NDBC real-time files rotate frequently, historical event
        data is synthesised from characteristic BPR patterns observed
        during well-documented tsunamis.  The synthesised data mirrors
        the statistical properties (amplitude, period, noise floor) of
        the original recordings.

        Args:
            event_id: One of ``"tohoku_2011"``, ``"chile_2010"``, or
                ``"tonga_2022"``.

        Returns:
            DataFrame with columns ``timestamp``, ``bpr``, and
            ``station_id``.

        Raises:
            ValueError: If *event_id* is not recognised.
        """
        if event_id not in _EVENTS_BY_ID:
            raise ValueError(f"Unknown event_id {event_id!r}. Available: {list(_EVENTS_BY_ID)}")

        # Try the cache first.
        cache_key = f"tsunami_historical_{event_id}"
        cached = self._read_cache(cache_key)
        if cached is not None:
            return pd.DataFrame(cached)

        event = _EVENTS_BY_ID[event_id]

        # NDBC real-time files only cover the most recent ~45 days, so
        # historical events (all of which predate that window) must be
        # reconstructed from characteristic BPR patterns.
        df = self._synthesize_event(event)
        self._write_cache(cache_key, df.to_dict(orient="list"))
        return df

    def list_events(self) -> list[dict[str, Any]]:
        """
        Return the catalog of ground-truth tsunami events.

        Returns:
            List of dicts, each containing at least ``event_id``,
            ``name``, ``date``, and ``description``.
        """
        return [
            {
                "event_id": e["event_id"],
                "name": e["name"],
                "date": e["date"],
                "description": e["description"],
            }
            for e in _EVENT_CATALOG
        ]

    def get_ground_truth(self, event_id: str) -> np.ndarray:
        """
        Return binary anomaly labels for a historical event.

        Time steps falling within the documented tsunami arrival window
        are labeled ``1`` (anomaly); all other time steps are ``0``
        (normal tide).

        Args:
            event_id: Identifier for the event.

        Returns:
            1-D numpy array of shape ``(n_samples,)`` with binary labels.

        Raises:
            ValueError: If *event_id* is not recognised.
        """
        if event_id not in _EVENTS_BY_ID:
            raise ValueError(f"Unknown event_id {event_id!r}. Available: {list(_EVENTS_BY_ID)}")

        event = _EVENTS_BY_ID[event_id]
        df = self.fetch_historical(event_id)

        window_start = datetime.fromisoformat(event["window_start"].replace("Z", "+00:00"))
        window_end = datetime.fromisoformat(event["window_end"].replace("Z", "+00:00"))

        timestamps = pd.to_datetime(df["timestamp"], utc=True)
        labels = np.where(
            (timestamps >= window_start) & (timestamps <= window_end),
            1,
            0,
        ).astype(np.intp)

        logger.info(
            "tsunami: ground truth for %s — %d anomaly / %d total",
            event_id,
            int(labels.sum()),
            len(labels),
        )
        return labels

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def engineer_features(self, raw_data: pd.DataFrame) -> np.ndarray:
        """
        Transform raw DART data into a feature matrix.

        Engineered features (per time step):

        1. **BPR reading** -- raw bottom pressure recorder value.
        2. **Tidal deviation** -- residual after removing a simple
           moving average (proxy for predicted tide).
        3. **Absolute deviation** -- magnitude of tidal deviation,
           emphasises departure regardless of sign.
        4. **Rate of change** -- first-order difference of BPR between
           consecutive time steps.
        5. **Rolling standard deviation** -- local variability measure
           over a 20-sample sliding window.
        6. **Short-window energy** -- rolling standard deviation over
           a 5-sample window, capturing rapid oscillations typical of
           tsunami arrivals.

        Args:
            raw_data: DataFrame returned by :meth:`fetch_realtime` or
                :meth:`fetch_historical`.  Must contain a ``bpr``
                column with numeric pressure values.

        Returns:
            2-D numpy array of shape ``(n_samples, 6)``.
        """
        if "bpr" not in raw_data.columns:
            # Fall back to base class behaviour for non-standard data.
            return super().engineer_features(raw_data)

        bpr = raw_data["bpr"].astype(np.float64).values.copy()

        # Replace non-finite values with forward-fill then backward-fill.
        bpr = _fill_non_finite(bpr)

        # Feature 1: raw BPR
        feat_bpr = bpr.copy()

        # Feature 2: deviation from simple moving average (tidal detrend)
        sma = _rolling_mean(bpr, _SMA_DETREND_WINDOW)
        feat_deviation = bpr - sma

        # Feature 3: absolute deviation (magnitude of departure)
        feat_abs_dev = np.abs(feat_deviation)

        # Feature 4: rate of sea-level change (first difference)
        feat_rate = np.diff(bpr, prepend=bpr[0])

        # Feature 5: rolling standard deviation (20-sample window)
        feat_rolling_std = _rolling_std(bpr, _ROLLING_STD_WINDOW)

        # Feature 6: short-window energy (5-sample rolling std)
        feat_short_energy = _rolling_std(bpr, 5)

        features = np.column_stack(
            [
                feat_bpr,
                feat_deviation,
                feat_abs_dev,
                feat_rate,
                feat_rolling_std,
                feat_short_energy,
            ]
        )

        # Final cleanup: replace any remaining inf/nan with 0.
        features = np.where(np.isfinite(features), features, 0.0)
        return features

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_station(self, station_id: str) -> pd.DataFrame:
        """
        Download and parse DART data for a single station.

        Args:
            station_id: NDBC station identifier (e.g. ``"21418"``).

        Returns:
            DataFrame with parsed columns including ``timestamp``,
            ``bpr``, and ``station_id``.
        """
        url = f"{self.SOURCE_URL}{station_id}.dart"
        raw_bytes = self._fetch_url(url)
        text = raw_bytes.decode("utf-8", errors="replace")

        # DART files have comment header lines starting with '#'.
        df = pd.read_csv(
            io.StringIO(text),
            comment="#",
            sep=r"\s+",
            header=None,
            na_values=["MM", "99.00", "999.0", "9999.0", "99999.0"],
        )

        if df.empty:
            logger.warning("tsunami: empty data for station %s", station_id)
            return df

        # The first five columns are typically: YY MM DD hh mm
        # followed by measurement columns.  Build a timestamp from
        # the date/time fields when they look plausible.
        df = self._parse_dart_columns(df, station_id)
        return df

    @staticmethod
    def _parse_dart_columns(df: pd.DataFrame, station_id: str) -> pd.DataFrame:
        """
        Assign column names and build a timestamp from date fields.

        Args:
            df: Raw dataframe read from NDBC DART file.
            station_id: Station identifier for tagging.

        Returns:
            DataFrame with ``timestamp``, ``bpr``, ``station_id`` and
            any additional measurement columns.
        """
        ncols = df.shape[1]

        # Minimal expected layout: YY MM DD hh mm T(TYPE) HEIGHT
        # Real DART files vary slightly, so we handle common formats.
        if ncols >= 7:
            base_names = ["year", "month", "day", "hour", "minute", "type", "bpr"]
            extra_names = [f"col_{i}" for i in range(7, ncols)]
            df.columns = base_names + extra_names

            # Build timestamp (handle 2-digit and 4-digit years).
            years = df["year"].astype(int)
            years = np.where(years < 100, years + 2000, years)

            timestamps: list[datetime] = []
            for _, row in df.iterrows():
                try:
                    ts = datetime(
                        int(years[_]) if isinstance(_, int) else int(row["year"]),
                        int(row["month"]),
                        int(row["day"]),
                        int(row["hour"]),
                        int(row["minute"]),
                        tzinfo=UTC,
                    )
                except (ValueError, OverflowError):
                    ts = datetime(2000, 1, 1, tzinfo=UTC)
                timestamps.append(ts)

            # Vectorised timestamp construction (faster for large files).
            try:
                ts_series = pd.to_datetime(
                    {
                        "year": years,
                        "month": df["month"].astype(int),
                        "day": df["day"].astype(int),
                        "hour": df["hour"].astype(int),
                        "minute": df["minute"].astype(int),
                    },
                    utc=True,
                )
            except Exception:
                ts_series = pd.Series(timestamps)

            df["timestamp"] = ts_series

        elif ncols >= 2:
            # Minimal fallback: treat first column as index, last as BPR.
            df.columns = [f"col_{i}" for i in range(ncols)]
            df = df.rename(columns={f"col_{ncols - 1}": "bpr"})
            df["timestamp"] = pd.Timestamp.now(tz="UTC")
        else:
            df["bpr"] = np.nan
            df["timestamp"] = pd.Timestamp.now(tz="UTC")

        df["station_id"] = station_id

        # Coerce BPR to numeric.
        df["bpr"] = pd.to_numeric(df["bpr"], errors="coerce")

        return df

    @staticmethod
    def _synthesize_event(event: dict[str, Any]) -> pd.DataFrame:
        """
        Generate synthetic BPR data mimicking a tsunami event.

        The synthetic time series contains a calm tidal background with
        superimposed tsunami-like oscillations during the event window.
        Statistical parameters are tuned per event to approximate
        published observations.

        Args:
            event: Event dict from :pydata:`_EVENT_CATALOG`.

        Returns:
            DataFrame with ``timestamp``, ``bpr``, and ``station_id``.
        """
        # Use a deterministic seed (Python's hash() is randomized per
        # process).  hashlib.sha256 produces the same bytes every time.
        import hashlib

        seed_bytes = hashlib.sha256(event["event_id"].encode()).digest()
        seed_int = int.from_bytes(seed_bytes[:4], "little") % (2**31)
        rng = np.random.default_rng(seed_int)

        window_start = datetime.fromisoformat(event["window_start"].replace("Z", "+00:00"))
        window_end = datetime.fromisoformat(event["window_end"].replace("Z", "+00:00"))

        # Scale observation window so the anomaly ratio stays below ~35%.
        # Long-duration events (>8 h) need proportionally more padding to
        # provide sufficient normal baseline for unsupervised detectors.
        event_duration_s = window_end.timestamp() - window_start.timestamp()
        padding_mult = 1.0 if event_duration_s > 8 * 3600 else 0.75
        pre_seconds = max(6 * 3600, int(event_duration_s * padding_mult))
        post_seconds = max(6 * 3600, int(event_duration_s * padding_mult))
        total_start = window_start.timestamp() - pre_seconds
        total_end = window_end.timestamp() + post_seconds
        step_seconds = 60  # 1-minute resolution

        time_array = np.arange(total_start, total_end, step_seconds)
        n_samples = len(time_array)

        # Background tide: slow sinusoidal (~12.42 hour period for M2 tidal constituent).
        tidal_period = 12.42 * 3600  # seconds
        baseline_pressure = 5000.0  # nominal deep-ocean BPR in dbar
        tidal_amplitude = 0.5  # typical deep-ocean tidal amplitude (dbar)

        tide = baseline_pressure + tidal_amplitude * np.sin(2 * np.pi * time_array / tidal_period)

        # Noise floor.
        noise = rng.normal(0, 0.005, size=n_samples)

        # Tsunami signal: applied only within the event window.
        tsunami_signal = np.zeros(n_samples)
        ws_epoch = window_start.timestamp()
        we_epoch = window_end.timestamp()
        in_window = (time_array >= ws_epoch) & (time_array <= we_epoch)

        if event["event_id"] == "tohoku_2011":
            amplitude = 0.15  # large signal (~15 cm water equiv.)
            period = 20 * 60  # ~20-minute dominant period
            decay_rate = 2.0
        elif event["event_id"] == "chile_2010":
            # Pacific-wide tsunami with sustained oscillations over ~7 h.
            amplitude = 0.08
            period = 25 * 60
            decay_rate = 1.5
        else:  # tonga_2022
            # Hunga Tonga eruption: VEI-5+ submarine volcanic event.
            # Pressure wave sustained oscillations over ~12 h across
            # the Pacific, with slower decay than seismic tsunamis.
            amplitude = 0.08
            period = 15 * 60
            decay_rate = 1.5

        # Decaying oscillation within the window.
        window_time = time_array[in_window] - ws_epoch
        window_duration = we_epoch - ws_epoch
        decay = np.exp(-decay_rate * window_time / window_duration)
        tsunami_signal[in_window] = amplitude * decay * np.sin(2 * np.pi * window_time / period)

        bpr = tide + tsunami_signal + noise

        timestamps = pd.to_datetime(time_array, unit="s", utc=True)

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "bpr": bpr,
                "station_id": event["station_id"],
            }
        )
        return df


# ---------------------------------------------------------------------------
# Pure-numpy helper functions (no sklearn dependency)
# ---------------------------------------------------------------------------


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Compute a rolling mean over *arr* with the given *window* size.

    Edge values are filled with the nearest valid mean to avoid NaNs.

    Args:
        arr: 1-D numpy array.
        window: Rolling window size (number of samples).

    Returns:
        1-D numpy array of the same length as *arr*.
    """
    if window < 1:
        return arr.copy()
    cumsum = np.cumsum(arr)
    cumsum = np.insert(cumsum, 0, 0.0)
    result = np.empty_like(arr)
    half = window // 2

    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + window - half)
        result[i] = (cumsum[hi] - cumsum[lo]) / (hi - lo)

    return result


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Compute a rolling standard deviation over *arr*.

    Uses the two-pass algorithm (mean then variance) for numerical
    stability.  Edge samples use a shrunk window.

    Args:
        arr: 1-D numpy array.
        window: Rolling window size (number of samples).

    Returns:
        1-D numpy array of the same length as *arr*.
    """
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    half = window // 2

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + window - half)
        segment = arr[lo:hi]
        result[i] = np.std(segment) if len(segment) > 1 else 0.0

    return result


def _fill_non_finite(arr: np.ndarray) -> np.ndarray:
    """
    Replace non-finite values with forward-fill then backward-fill.

    If the entire array is non-finite, fills with ``0.0``.

    Args:
        arr: 1-D numpy array (modified in place and returned).

    Returns:
        The same array with non-finite values replaced.
    """
    mask = ~np.isfinite(arr)
    if not mask.any():
        return arr

    # Forward fill.
    last_valid = np.nan
    for i in range(len(arr)):
        if mask[i]:
            if np.isfinite(last_valid):
                arr[i] = last_valid
        else:
            last_valid = arr[i]

    # Backward fill any remaining leading NaNs.
    first_valid = np.nan
    for i in range(len(arr) - 1, -1, -1):
        if np.isfinite(arr[i]):
            first_valid = arr[i]
            break

    if np.isfinite(first_valid):
        for i in range(len(arr)):
            if not np.isfinite(arr[i]):
                arr[i] = first_valid
            else:
                break
    else:
        # Entire array was non-finite.
        arr[:] = 0.0

    return arr
