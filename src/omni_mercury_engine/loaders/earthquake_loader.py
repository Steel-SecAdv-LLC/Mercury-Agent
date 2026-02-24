"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Domain loader for earthquake data from the USGS Earthquake Hazards Program API.

Connects to the USGS real-time GeoJSON feed and FDSN event web-service
to provide seismic data for Mercury anomaly detection.  Ground truth
events cover major earthquakes where mainshock + large aftershocks are
labeled as anomalies against a background of smaller seismicity.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from omni_mercury_engine.loaders.base import BaseDomainLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# USGS API endpoints
# ---------------------------------------------------------------------------
_REALTIME_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson"
_HISTORICAL_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# ---------------------------------------------------------------------------
# Ground truth event catalog
# ---------------------------------------------------------------------------
_EVENT_CATALOG: dict[str, dict[str, Any]] = {
    "turkey_syria_2023": {
        "name": "2023 Turkey-Syria Earthquake",
        "date": "2023-02-06",
        "description": "M7.8 earthquake in southeastern Turkey near the Syria border.",
        "magnitude": 7.8,
        "start": "2023-02-05",
        "end": "2023-02-08",
        "min_magnitude": 2.5,
    },
    "noto_2024": {
        "name": "2024 Noto Peninsula Earthquake",
        "date": "2024-01-01",
        "description": "M7.5 earthquake on the Noto Peninsula, Ishikawa Prefecture, Japan.",
        "magnitude": 7.5,
        "start": "2023-12-31",
        "end": "2024-01-04",
        "min_magnitude": 2.5,
    },
    "tohoku_2011": {
        "name": "2011 Tohoku Earthquake",
        "date": "2011-03-11",
        "description": "M9.1 megathrust earthquake off the Pacific coast of Tohoku, Japan.",
        "magnitude": 9.1,
        "start": "2011-03-10",
        "end": "2011-03-14",
        "min_magnitude": 4.0,
    },
    "haiti_2010": {
        "name": "2010 Haiti Earthquake",
        "date": "2010-01-12",
        "description": "M7.0 earthquake near Leogane, Haiti.",
        "magnitude": 7.0,
        "start": "2010-01-11",
        "end": "2010-01-15",
        "min_magnitude": 2.5,
    },
    "nepal_2015": {
        "name": "2015 Nepal Earthquake",
        "date": "2015-04-25",
        "description": "M7.8 earthquake in the Gorkha district of Nepal.",
        "magnitude": 7.8,
        "start": "2015-04-24",
        "end": "2015-04-28",
        "min_magnitude": 2.5,
    },
}


class EarthquakeLoader(BaseDomainLoader):
    """Loader for earthquake data from the USGS Earthquake Hazards Program.

    Uses two USGS endpoints:

    * **Real-time feed** -- GeoJSON summary of all earthquakes in the last
      hour, updated every minute.
    * **Historical query** -- FDSN event web-service for parameterized
      queries against the full USGS catalog.

    Feature engineering produces seismological observables suitable for
    anomaly detection: magnitude, depth, location, inter-event timing,
    seismicity rate, Gutenberg-Richter b-value, and magnitude deviation
    from the local mean.
    """

    DOMAIN: str = "earthquake"
    SOURCE_URL: str = _REALTIME_URL
    REQUIRES_API_KEY: bool = False

    FEATURE_COLUMNS: list[str] = [
        "magnitude",
        "depth",
        "latitude",
        "longitude",
        "time_delta_s",
        "seismicity_rate",
        "b_value",
        "mag_deviation",
    ]

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def fetch_realtime(self) -> pd.DataFrame:
        """Fetch the most recent hour of global earthquake data from USGS.

        Returns:
            DataFrame with columns: time, latitude, longitude, depth,
            magnitude, place, event_id.

        Raises:
            ConnectionError: If the USGS feed is unreachable after retries.
        """
        cache_key = "earthquake_realtime"
        cached = self._read_cache(cache_key)

        if cached is not None:
            logger.debug("Returning cached real-time earthquake data.")
            return pd.DataFrame(cached)

        geojson = self._fetch_json(_REALTIME_URL)
        df = self._geojson_to_dataframe(geojson)

        self._write_cache(cache_key, df.to_dict(orient="list"))
        logger.info("Fetched %d real-time earthquake records from USGS.", len(df))
        return df

    def fetch_historical(self, event_id: str) -> pd.DataFrame:
        """Fetch seismic catalog data surrounding a known historical event.

        Args:
            event_id: Key into the ground truth catalog (e.g.
                ``"turkey_syria_2023"``).

        Returns:
            DataFrame with the same schema as :meth:`fetch_realtime`.

        Raises:
            ValueError: If *event_id* is not in the catalog.
            ConnectionError: If the USGS query service is unreachable.
        """
        if event_id not in _EVENT_CATALOG:
            raise ValueError(
                f"Unknown event_id '{event_id}'. " f"Available: {list(_EVENT_CATALOG.keys())}"
            )

        cache_key = f"earthquake_historical_{event_id}"
        cached = self._read_cache(cache_key)

        if cached is not None:
            logger.debug("Returning cached historical data for '%s'.", event_id)
            return pd.DataFrame(cached)

        event = _EVENT_CATALOG[event_id]
        params: dict[str, str] = {
            "format": "geojson",
            "starttime": event["start"],
            "endtime": event["end"],
            "minmagnitude": str(event["min_magnitude"]),
        }

        geojson = self._fetch_json(_HISTORICAL_URL, params=params)
        df = self._geojson_to_dataframe(geojson)

        if df.empty:
            logger.warning("USGS returned no features for event '%s'.", event_id)
            return df

        # Sort chronologically so time-series features are meaningful
        df = df.sort_values("time").reset_index(drop=True)

        self._write_cache(cache_key, df.to_dict(orient="list"))
        logger.info("Fetched %d historical records for event '%s'.", len(df), event_id)
        return df

    def list_events(self) -> list[dict[str, Any]]:
        """Return the catalog of ground truth earthquake events.

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

    def get_ground_truth(self, event_id: str) -> np.ndarray:
        """Generate binary anomaly labels for a historical earthquake event.

        Labeling strategy: an earthquake is labeled *anomalous* (``1``) if
        its magnitude is at least ``mainshock_magnitude - 1.0``.  All
        smaller aftershocks and background events are labeled *normal*
        (``0``).  This captures the mainshock itself plus any unusually
        large aftershocks that deviate significantly from the background
        Gutenberg-Richter distribution.

        Args:
            event_id: Key into the ground truth catalog.

        Returns:
            1-D binary numpy array of shape ``(n_events,)``.

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

        mainshock_mag = _EVENT_CATALOG[event_id]["magnitude"]
        threshold = mainshock_mag - 1.0

        labels = (df["magnitude"].values >= threshold).astype(np.int64)
        logger.info(
            "Ground truth for '%s': %d anomalies / %d total (threshold M>=%.1f).",
            event_id,
            int(labels.sum()),
            len(labels),
            threshold,
        )
        return np.asarray(labels)

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def engineer_features(self, raw_data: pd.DataFrame) -> np.ndarray:
        """Transform raw earthquake catalog into a feature matrix.

        Engineered features (per event row):

        1. **magnitude** -- reported magnitude.
        2. **depth** -- hypocentral depth in km.
        3. **latitude** -- epicentral latitude.
        4. **longitude** -- epicentral longitude.
        5. **time_delta_s** -- seconds since the previous event (0 for
           the first row).
        6. **seismicity_rate** -- number of events in the preceding
           1-hour rolling window (proxy for local activity rate).
        7. **b_value** -- Gutenberg-Richter *b*-value estimated over a
           trailing window of 50 events (NaN-filled when fewer than 10
           events are available).
        8. **mag_deviation** -- deviation of the current magnitude from
           a 20-event rolling mean.

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
        df = df.sort_values("time").reset_index(drop=True)

        # ---- base observables ----
        magnitude = df["magnitude"].values.astype(np.float64)
        depth = df["depth"].values.astype(np.float64)
        latitude = df["latitude"].values.astype(np.float64)
        longitude = df["longitude"].values.astype(np.float64)

        # ---- inter-event time deltas (seconds) ----
        times_ms = df["time"].values.astype(np.float64)
        time_delta_s = np.zeros(len(df), dtype=np.float64)
        if len(df) > 1:
            time_delta_s[1:] = np.diff(times_ms) / 1000.0

        # ---- seismicity rate (events per hour, 1-hour rolling window) ----
        seismicity_rate = self._compute_seismicity_rate(times_ms)

        # ---- Gutenberg-Richter b-value (rolling window) ----
        b_value = self._compute_rolling_b_value(magnitude, window=50, min_events=10)

        # ---- magnitude deviation from rolling mean ----
        mag_deviation = self._compute_mag_deviation(magnitude, window=20)

        # Stack into feature matrix
        features = np.column_stack(
            [
                magnitude,
                depth,
                latitude,
                longitude,
                time_delta_s,
                seismicity_rate,
                b_value,
                mag_deviation,
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
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _geojson_to_dataframe(geojson: dict[str, Any]) -> pd.DataFrame:
        """Convert a USGS GeoJSON FeatureCollection to a flat DataFrame.

        Args:
            geojson: Parsed GeoJSON dict from the USGS API.

        Returns:
            DataFrame with columns: time, latitude, longitude, depth,
            magnitude, place, event_id.
        """
        features = geojson.get("features", [])
        if not features:
            return pd.DataFrame(
                columns=[
                    "time",
                    "latitude",
                    "longitude",
                    "depth",
                    "magnitude",
                    "place",
                    "event_id",
                ]
            )

        rows: list[dict[str, Any]] = []
        for feature in features:
            props = feature.get("properties", {})
            coords = feature.get("geometry", {}).get("coordinates", [None, None, None])
            rows.append(
                {
                    "time": props.get("time"),
                    "latitude": coords[1] if len(coords) > 1 else None,
                    "longitude": coords[0] if len(coords) > 0 else None,
                    "depth": coords[2] if len(coords) > 2 else None,
                    "magnitude": props.get("mag"),
                    "place": props.get("place", ""),
                    "event_id": feature.get("id", ""),
                }
            )

        df = pd.DataFrame(rows)
        # Coerce numeric columns
        for col in ("time", "latitude", "longitude", "depth", "magnitude"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    @staticmethod
    def _compute_seismicity_rate(times_ms: np.ndarray) -> np.ndarray:
        """Compute the number of events in the preceding 1-hour window.

        Args:
            times_ms: Array of event origin times in milliseconds since
                epoch (USGS convention).

        Returns:
            1-D array of event counts in the trailing 3600-second window.
        """
        one_hour_ms = 3_600_000.0
        n = len(times_ms)
        rate = np.zeros(n, dtype=np.float64)

        for i in range(n):
            window_start = times_ms[i] - one_hour_ms
            # Count events in (window_start, times_ms[i]]
            rate[i] = float(np.sum((times_ms[:i] > window_start) & (times_ms[:i] <= times_ms[i])))

        return rate

    @staticmethod
    def _compute_rolling_b_value(
        magnitudes: np.ndarray,
        window: int = 50,
        min_events: int = 10,
    ) -> np.ndarray:
        """Estimate Gutenberg-Richter b-value over a trailing window.

        Uses the Aki-Utsu maximum-likelihood estimator:

            b = log10(e) / (M_mean - M_min)

        where *M_mean* is the sample mean magnitude and *M_min* is the
        minimum magnitude in the window.

        Args:
            magnitudes: 1-D array of magnitudes.
            window: Number of preceding events to include.
            min_events: Minimum events required for a valid estimate.

        Returns:
            1-D array of b-values (NaN where insufficient data).
        """
        n = len(magnitudes)
        b_values = np.full(n, np.nan, dtype=np.float64)
        log10_e = np.log10(np.e)

        for i in range(n):
            start = max(0, i - window + 1)
            win = magnitudes[start : i + 1]
            if len(win) < min_events:
                continue
            m_mean = np.mean(win)
            m_min = np.min(win)
            denom = m_mean - m_min
            if denom > 0:
                b_values[i] = log10_e / denom

        return b_values

    @staticmethod
    def _compute_mag_deviation(
        magnitudes: np.ndarray,
        window: int = 20,
    ) -> np.ndarray:
        """Compute magnitude deviation from a trailing rolling mean.

        Args:
            magnitudes: 1-D array of magnitudes.
            window: Rolling window size.

        Returns:
            1-D array of deviations (current magnitude minus rolling mean).
        """
        n = len(magnitudes)
        deviation = np.zeros(n, dtype=np.float64)

        for i in range(n):
            start = max(0, i - window + 1)
            win_mean = np.mean(magnitudes[start : i + 1])
            deviation[i] = magnitudes[i] - win_mean

        return deviation
