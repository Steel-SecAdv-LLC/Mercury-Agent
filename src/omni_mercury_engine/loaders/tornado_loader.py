"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Domain loader for tornado data from the NOAA Storm Prediction Center (SPC).

Connects to the SPC tornado archive CSV and daily storm reports feed
to provide severe weather data for Mercury anomaly detection.  Ground truth
events cover major tornado outbreaks where EF3+ tornadoes are labeled as
anomalies against a background of weaker (EF0-EF2) events.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from omni_mercury_engine.loaders.base import BaseDomainLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SPC data endpoints
# ---------------------------------------------------------------------------
_ARCHIVE_URL = (
    "https://www.spc.noaa.gov/wcm/data/1950-2023_actual_tornadoes.csv"
)
_DAILY_REPORTS_URL = "https://www.spc.noaa.gov/climo/reports/today.csv"

# ---------------------------------------------------------------------------
# SPC CSV column names (canonical ordering)
# ---------------------------------------------------------------------------
_SPC_COLUMNS = [
    "om", "yr", "mo", "dy", "date", "time", "tz", "st", "stf", "stn",
    "mag", "inj", "fat", "loss", "closs", "slat", "slon", "elat", "elon",
    "len", "wid", "ns", "sn", "sg", "f1", "f2", "f3", "f4", "fc",
]

# ---------------------------------------------------------------------------
# Ground truth event catalog
# ---------------------------------------------------------------------------
_EVENT_CATALOG: dict[str, dict[str, Any]] = {
    "super_outbreak_2011": {
        "name": "2011 Super Outbreak",
        "date": "2011-04-27",
        "description": (
            "April 25-28, 2011 Super Outbreak producing 362 tornadoes "
            "across the southeastern United States, including multiple "
            "violent EF4-EF5 tornadoes."
        ),
        "start": "2011-04-25",
        "end": "2011-04-28",
        "tornado_count": 362,
    },
    "moore_2013": {
        "name": "2013 Moore OK EF5 Tornado",
        "date": "2013-05-20",
        "description": (
            "EF5 tornado that struck Moore, Oklahoma on May 20, 2013 "
            "with peak winds of 210 mph, killing 24 people and causing "
            "$2 billion in damage."
        ),
        "start": "2013-05-19",
        "end": "2013-05-21",
        "tornado_count": None,
    },
    "midwest_2024": {
        "name": "2024 Midwest/Southeast Outbreak",
        "date": "2024-04-27",
        "description": (
            "Late April 2024 tornado outbreak across the Midwest and "
            "Southeast United States with multiple significant tornadoes."
        ),
        "start": "2024-04-25",
        "end": "2024-04-30",
        "tornado_count": None,
    },
}

# ---------------------------------------------------------------------------
# Approximate geographic centroid of US tornado activity (Tornado Alley)
# Used for geographic anomaly feature.
# ---------------------------------------------------------------------------
_TORNADO_DENSITY_CENTROID_LAT = 35.5
_TORNADO_DENSITY_CENTROID_LON = -97.5


class TornadoLoader(BaseDomainLoader):
    """Loader for tornado data from the NOAA Storm Prediction Center.

    Uses two SPC data sources:

    * **Tornado archive CSV** -- comprehensive record of all US tornadoes
      from 1950 to 2023 with EF-scale ratings, path geometry, and
      casualty data.
    * **Daily storm reports** -- today's preliminary severe weather
      reports including tornado sightings.

    Feature engineering produces meteorological and spatial observables
    suitable for anomaly detection: EF-scale magnitude, path dimensions,
    casualty counts, temporal clustering rate, geographic anomaly
    distance, and time-of-year features.
    """

    DOMAIN: str = "tornado"
    SOURCE_URL: str = _ARCHIVE_URL
    REQUIRES_API_KEY: bool = False

    FEATURE_COLUMNS: list[str] = [
        "ef_scale", "path_length", "path_width", "fatalities", "injuries",
        "slat", "slon", "temporal_cluster", "geo_anomaly",
        "month", "day_of_year", "hour",
    ]

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def fetch_realtime(self) -> pd.DataFrame:
        """Fetch today's tornado reports from the SPC daily storm reports.

        Returns:
            DataFrame with tornado report columns parsed from the SPC
            daily CSV feed.

        Raises:
            ConnectionError: If the SPC feed is unreachable after retries.
        """
        cache_key = "tornado_realtime"
        cached = self._read_cache(cache_key)

        if cached is not None:
            logger.debug("Returning cached real-time tornado data.")
            return pd.DataFrame(cached)

        df = self._fetch_csv(_DAILY_REPORTS_URL)

        # SPC daily reports may have different column layouts; normalise
        df = self._normalise_daily_report(df)

        self._write_cache(cache_key, df.to_dict(orient="list"))
        logger.info(
            "Fetched %d real-time tornado reports from SPC.", len(df)
        )
        return df

    def fetch_historical(self, event_id: str) -> pd.DataFrame:
        """Fetch tornado records for a specific historical event.

        Loads the full SPC tornado archive and filters to the date range
        associated with the requested event.

        Args:
            event_id: Key into the ground truth catalog (e.g.
                ``"super_outbreak_2011"``).

        Returns:
            DataFrame with SPC tornado archive columns filtered to the
            event's date range.

        Raises:
            ValueError: If *event_id* is not in the catalog.
            ConnectionError: If the SPC archive is unreachable.
        """
        if event_id not in _EVENT_CATALOG:
            raise ValueError(
                f"Unknown event_id '{event_id}'. "
                f"Available: {list(_EVENT_CATALOG.keys())}"
            )

        cache_key = f"tornado_historical_{event_id}"
        cached = self._read_cache(cache_key)

        if cached is not None:
            logger.debug(
                "Returning cached historical data for '%s'.", event_id
            )
            return pd.DataFrame(cached)

        # Fetch the full archive CSV
        archive_df = self._load_archive()

        event = _EVENT_CATALOG[event_id]
        start_date = pd.Timestamp(event["start"])
        end_date = pd.Timestamp(event["end"])

        # Filter to event date range
        df = self._filter_by_date_range(archive_df, start_date, end_date)

        if df.empty:
            logger.warning(
                "SPC archive returned no records for event '%s'.", event_id
            )
            return df

        # Sort chronologically
        df = df.sort_values("date").reset_index(drop=True)

        self._write_cache(cache_key, df.to_dict(orient="list"))
        logger.info(
            "Fetched %d historical tornado records for event '%s'.",
            len(df),
            event_id,
        )
        return df

    def list_events(self) -> list[dict[str, Any]]:
        """Return the catalog of ground truth tornado events.

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
        """Generate binary anomaly labels for a historical tornado event.

        Labeling strategy: a tornado is labeled *anomalous* (``1``) if
        its EF-scale rating (``mag`` column) is 3 or higher (EF3+).
        All weaker tornadoes (EF0-EF2) are labeled *normal* (``0``).

        For the Super Outbreak, this captures the violent tornadoes that
        represent the most extreme and destructive portion of the event.

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
                f"Unknown event_id '{event_id}'. "
                f"Available: {list(_EVENT_CATALOG.keys())}"
            )

        df = self.fetch_historical(event_id)
        if df.empty:
            return np.array([], dtype=np.int64)

        mag = pd.to_numeric(df["mag"], errors="coerce").fillna(-1).values
        labels = (mag >= 3).astype(np.int64)

        logger.info(
            "Ground truth for '%s': %d anomalies / %d total (threshold EF3+).",
            event_id,
            int(labels.sum()),
            len(labels),
        )
        return labels

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def engineer_features(self, raw_data: pd.DataFrame) -> np.ndarray:
        """Transform raw tornado data into a feature matrix.

        Engineered features (per tornado row):

        1. **ef_scale** -- EF-scale rating (``mag`` column).
        2. **path_length** -- tornado path length in miles (``len``).
        3. **path_width** -- tornado path width in yards (``wid``).
        4. **fatalities** -- number of fatalities (``fat``).
        5. **injuries** -- number of injuries (``inj``).
        6. **slat** -- start-point latitude.
        7. **slon** -- start-point longitude.
        8. **temporal_cluster** -- count of tornadoes in the same hour
           within a 100 km radius (proxy for outbreak intensity).
        9. **geo_anomaly** -- great-circle distance in km from the
           historical tornado density centroid (Tornado Alley).
        10. **month** -- month of occurrence (1-12).
        11. **day_of_year** -- day of year (1-366).
        12. **hour** -- hour of occurrence (0-23).

        Args:
            raw_data: DataFrame from :meth:`fetch_realtime` or
                :meth:`fetch_historical`.

        Returns:
            2-D numpy array of shape ``(n_samples, 12)``.
        """
        n_features = 12

        if raw_data.empty:
            return np.empty((0, n_features), dtype=np.float64)

        df = raw_data.copy()

        # ---- base observables ----
        ef_scale = pd.to_numeric(df["mag"], errors="coerce").fillna(0).values.astype(np.float64)
        path_length = pd.to_numeric(df["len"], errors="coerce").fillna(0).values.astype(np.float64)
        path_width = pd.to_numeric(df["wid"], errors="coerce").fillna(0).values.astype(np.float64)
        fatalities = pd.to_numeric(df["fat"], errors="coerce").fillna(0).values.astype(np.float64)
        injuries = pd.to_numeric(df["inj"], errors="coerce").fillna(0).values.astype(np.float64)

        # ---- geographic coordinates ----
        slat = pd.to_numeric(df["slat"], errors="coerce").fillna(0).values.astype(np.float64)
        slon = pd.to_numeric(df["slon"], errors="coerce").fillna(0).values.astype(np.float64)

        # ---- temporal features ----
        timestamps = self._parse_timestamps(df)
        month = np.array([ts.month for ts in timestamps], dtype=np.float64)
        day_of_year = np.array(
            [ts.timetuple().tm_yday for ts in timestamps], dtype=np.float64
        )
        hour = np.array([ts.hour for ts in timestamps], dtype=np.float64)

        # ---- temporal clustering (tornadoes per hour in 100 km radius) ----
        temporal_cluster = self._compute_temporal_clustering(
            timestamps, slat, slon, radius_km=100.0
        )

        # ---- geographic anomaly (distance from density centroid) ----
        geo_anomaly = self._compute_geographic_anomaly(slat, slon)

        # Stack into feature matrix
        features = np.column_stack(
            [
                ef_scale,
                path_length,
                path_width,
                fatalities,
                injuries,
                slat,
                slon,
                temporal_cluster,
                geo_anomaly,
                month,
                day_of_year,
                hour,
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

    def _load_archive(self) -> pd.DataFrame:
        """Load the full SPC tornado archive CSV with caching.

        Returns:
            DataFrame containing the complete 1950-2023 tornado archive.

        Raises:
            ConnectionError: If the SPC archive URL is unreachable.
        """
        cache_key = "tornado_archive_full"
        cached = self._read_cache(cache_key)

        if cached is not None:
            logger.debug("Returning cached tornado archive.")
            return pd.DataFrame(cached)

        logger.info("Downloading full SPC tornado archive from %s", _ARCHIVE_URL)
        df = self._fetch_csv(_ARCHIVE_URL, low_memory=False)

        # Standardise column names to lowercase and strip whitespace
        df.columns = [c.strip().lower() for c in df.columns]

        # Coerce numeric columns
        numeric_cols = ["mag", "inj", "fat", "loss", "closs", "slat", "slon",
                        "elat", "elon", "len", "wid"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        self._write_cache(cache_key, df.to_dict(orient="list"))
        logger.info("Loaded %d records from SPC tornado archive.", len(df))
        return df

    @staticmethod
    def _filter_by_date_range(
        df: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """Filter archive DataFrame to a specific date range.

        Constructs a date from the ``yr``, ``mo``, ``dy`` columns (or
        the ``date`` column if available) and selects rows within the
        inclusive range ``[start_date, end_date]``.

        Args:
            df: Full tornado archive DataFrame.
            start_date: Start of the date range (inclusive).
            end_date: End of the date range (inclusive).

        Returns:
            Filtered DataFrame.
        """
        if "date" in df.columns:
            parsed_dates = pd.to_datetime(df["date"], errors="coerce")
        elif all(c in df.columns for c in ("yr", "mo", "dy")):
            parsed_dates = pd.to_datetime(
                df[["yr", "mo", "dy"]].rename(
                    columns={"yr": "year", "mo": "month", "dy": "day"}
                ),
                errors="coerce",
            )
        else:
            logger.warning("Cannot determine date columns for filtering.")
            return df

        mask = (parsed_dates >= start_date) & (parsed_dates <= end_date)
        return df.loc[mask].reset_index(drop=True)

    @staticmethod
    def _normalise_daily_report(df: pd.DataFrame) -> pd.DataFrame:
        """Normalise SPC daily storm report columns to match archive schema.

        The daily report CSV may have a different layout than the
        historical archive.  This method renames and aligns columns
        where possible.

        Args:
            df: Raw DataFrame from SPC daily reports CSV.

        Returns:
            DataFrame with normalised column names.
        """
        df.columns = [c.strip().lower() for c in df.columns]

        # Common SPC daily report columns: Time, F_Scale, Location, County,
        # State, Lat, Lon, Comments
        rename_map: dict[str, str] = {
            "f_scale": "mag",
            "lat": "slat",
            "lon": "slon",
        }
        for old_name, new_name in rename_map.items():
            if old_name in df.columns and new_name not in df.columns:
                df = df.rename(columns={old_name: new_name})

        return df

    @staticmethod
    def _parse_timestamps(df: pd.DataFrame) -> list[pd.Timestamp]:
        """Parse date and time information from the tornado DataFrame.

        Attempts to construct timestamps from the ``date`` and ``time``
        columns, falling back to ``yr``, ``mo``, ``dy`` columns if
        ``date`` is not available.

        Args:
            df: Tornado DataFrame with date/time columns.

        Returns:
            List of pandas Timestamps, one per row.
        """
        n = len(df)
        timestamps: list[pd.Timestamp] = []

        # Try combining date + time columns
        if "date" in df.columns and "time" in df.columns:
            for i in range(n):
                try:
                    date_str = str(df["date"].iloc[i])
                    time_str = str(df["time"].iloc[i])
                    ts = pd.Timestamp(f"{date_str} {time_str}")
                    timestamps.append(ts)
                except (ValueError, TypeError):
                    timestamps.append(pd.Timestamp("1970-01-01"))
        elif "date" in df.columns:
            for i in range(n):
                try:
                    ts = pd.Timestamp(str(df["date"].iloc[i]))
                    timestamps.append(ts)
                except (ValueError, TypeError):
                    timestamps.append(pd.Timestamp("1970-01-01"))
        elif all(c in df.columns for c in ("yr", "mo", "dy")):
            for i in range(n):
                try:
                    yr = int(df["yr"].iloc[i])
                    mo = int(df["mo"].iloc[i])
                    dy = int(df["dy"].iloc[i])
                    timestamps.append(pd.Timestamp(year=yr, month=mo, day=dy))
                except (ValueError, TypeError):
                    timestamps.append(pd.Timestamp("1970-01-01"))
        else:
            timestamps = [pd.Timestamp("1970-01-01")] * n

        return timestamps

    @staticmethod
    def _compute_temporal_clustering(
        timestamps: list[pd.Timestamp],
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        radius_km: float = 100.0,
    ) -> np.ndarray:
        """Compute temporal clustering: tornado count per hour in surrounding region.

        For each tornado, counts the number of other tornadoes occurring
        within the same hour and within the specified radius.

        Args:
            timestamps: List of event timestamps.
            latitudes: 1-D array of start-point latitudes.
            longitudes: 1-D array of start-point longitudes.
            radius_km: Radius in km for spatial proximity (default 100).

        Returns:
            1-D array of cluster counts (float64).
        """
        n = len(timestamps)
        cluster_counts = np.zeros(n, dtype=np.float64)

        if n <= 1:
            return cluster_counts

        # Convert timestamps to epoch seconds for efficient comparison
        epoch_seconds = np.array(
            [ts.timestamp() for ts in timestamps], dtype=np.float64
        )
        one_hour_s = 3600.0

        for i in range(n):
            count = 0.0
            for j in range(n):
                if i == j:
                    continue
                # Check temporal proximity (within same hour)
                if abs(epoch_seconds[i] - epoch_seconds[j]) > one_hour_s:
                    continue
                # Check spatial proximity using haversine approximation
                dist = _haversine_km(
                    latitudes[i], longitudes[i],
                    latitudes[j], longitudes[j],
                )
                if dist <= radius_km:
                    count += 1.0
            cluster_counts[i] = count

        return cluster_counts

    @staticmethod
    def _compute_geographic_anomaly(
        latitudes: np.ndarray,
        longitudes: np.ndarray,
    ) -> np.ndarray:
        """Compute distance from the historical tornado density centroid.

        Returns the great-circle distance in km from each tornado's
        start point to the approximate centroid of US tornado activity
        (Tornado Alley: ~35.5N, 97.5W).

        Args:
            latitudes: 1-D array of start-point latitudes.
            longitudes: 1-D array of start-point longitudes.

        Returns:
            1-D array of distances in km (float64).
        """
        n = len(latitudes)
        distances = np.zeros(n, dtype=np.float64)

        for i in range(n):
            distances[i] = _haversine_km(
                latitudes[i], longitudes[i],
                _TORNADO_DENSITY_CENTROID_LAT,
                _TORNADO_DENSITY_CENTROID_LON,
            )

        return distances


# ---------------------------------------------------------------------------
# Module-level utility functions
# ---------------------------------------------------------------------------

def _haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Compute great-circle distance between two points using the haversine formula.

    Args:
        lat1: Latitude of point 1 in decimal degrees.
        lon1: Longitude of point 1 in decimal degrees.
        lat2: Latitude of point 2 in decimal degrees.
        lon2: Longitude of point 2 in decimal degrees.

    Returns:
        Distance in kilometres.
    """
    earth_radius_km = 6371.0

    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    return float(earth_radius_km * c)
