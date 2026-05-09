"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

Domain loader for hurricane/cyclone data from NOAA IBTrACS (International Best Track Archive for
Climate Stewardship).

Connects to the IBTrACS v04r01 CSV archive to provide tropical cyclone track data for Mercury
anomaly detection.  Ground truth events cover major hurricanes where rapid intensification periods
(wind speed increase >= 30 kt in 24 h) are labeled as anomalies against a background of normal track
evolution.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
import pandas as pd

from omni_mercury_engine.loaders.base import BaseDomainLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IBTrACS data URLs
# ---------------------------------------------------------------------------
_BASE_CSV_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-"
    "stewardship-ibtracs/v04r01/access/csv/"
)

_LAST3YEARS_URL = f"{_BASE_CSV_URL}ibtracs.last3years.list.v04r01.csv"

_NA_BASIN_URL = f"{_BASE_CSV_URL}ibtracs.NA.list.v04r01.csv"

# ---------------------------------------------------------------------------
# IBTrACS columns of interest
# ---------------------------------------------------------------------------
_IBTRACS_COLUMNS = [
    "SID",
    "SEASON",
    "NAME",
    "ISO_TIME",
    "LAT",
    "LON",
    "WMO_WIND",
    "WMO_PRES",
    "USA_WIND",
    "USA_PRES",
    "NATURE",
]

# Number of header/comment rows to skip in IBTrACS CSV.
# Row 0 is the header, row 1 is the units row.
_IBTRACS_HEADER_ROWS = [1]

# Rapid intensification threshold: >= 30 kt wind speed increase in 24 h.
_RI_THRESHOLD_KT = 30

# Number of 6-hourly track observations in 24 hours.
_STEPS_24H = 4

# ---------------------------------------------------------------------------
# Ground truth event catalog
# ---------------------------------------------------------------------------
_EVENT_CATALOG: dict[str, dict[str, Any]] = {
    "katrina_2005": {
        "name": "Hurricane Katrina (2005)",
        "date": "2005-08-29",
        "description": (
            "Category 5 hurricane in the Gulf of Mexico with extreme "
            "rapid intensification.  One of the costliest and deadliest "
            "hurricanes in U.S. history."
        ),
        "sid": "2005236N23285",
        "basin": "NA",
        "source": "historical",
    },
    "harvey_2017": {
        "name": "Hurricane Harvey (2017)",
        "date": "2017-08-25",
        "description": (
            "Category 4 hurricane that caused catastrophic flooding in southeastern Texas."
        ),
        "sid": "2017232N14283",
        "basin": "NA",
        "source": "historical",
    },
    "maria_2017": {
        "name": "Hurricane Maria (2017)",
        "date": "2017-09-20",
        "description": (
            "Category 5 hurricane that devastated Dominica and Puerto Rico "
            "with rapid intensification in the Caribbean Sea."
        ),
        "sid": "2017253N12318",
        "basin": "NA",
        "source": "historical",
    },
    "ian_2022": {
        "name": "Hurricane Ian (2022)",
        "date": "2022-09-28",
        "description": (
            "Category 4 hurricane that caused extreme damage in Florida "
            "after rapid intensification in the Caribbean."
        ),
        "sid": "2022266N14279",
        "basin": "NA",
        "source": "historical",
    },
    "helene_2024": {
        "name": "Hurricane Helene (2024)",
        "date": "2024-09-26",
        "description": (
            "Major hurricane that impacted the southeastern United States "
            "causing significant inland flooding."
        ),
        "sid": None,
        "basin": "NA",
        "source": "last3years",
    },
    "milton_2024": {
        "name": "Hurricane Milton (2024)",
        "date": "2024-10-09",
        "description": (
            "Rapidly intensifying hurricane in the Gulf of Mexico that "
            "made landfall on Florida's west coast."
        ),
        "sid": None,
        "basin": "NA",
        "source": "last3years",
    },
    # --- Additional storms for larger sample size ---
    "irma_2017": {
        "name": "Hurricane Irma (2017)",
        "date": "2017-09-10",
        "description": (
            "Category 5 hurricane that caused widespread destruction "
            "across the Caribbean and Florida."
        ),
        "sid": "2017242N16333",
        "basin": "NA",
        "source": "historical",
    },
    "michael_2018": {
        "name": "Hurricane Michael (2018)",
        "date": "2018-10-10",
        "description": (
            "Category 5 hurricane that rapidly intensified before making "
            "landfall on the Florida Panhandle."
        ),
        "sid": "2018280N18082",
        "basin": "NA",
        "source": "historical",
    },
    "dorian_2019": {
        "name": "Hurricane Dorian (2019)",
        "date": "2019-09-01",
        "description": (
            "Category 5 hurricane that devastated the Bahamas after extreme rapid intensification."
        ),
        "sid": "2019236N10340",
        "basin": "NA",
        "source": "historical",
    },
    "ida_2021": {
        "name": "Hurricane Ida (2021)",
        "date": "2021-08-29",
        "description": (
            "Category 4 hurricane that rapidly intensified in the "
            "Gulf of Mexico before striking Louisiana."
        ),
        "sid": "2021238N17279",
        "basin": "NA",
        "source": "historical",
    },
    "sandy_2012": {
        "name": "Hurricane Sandy (2012)",
        "date": "2012-10-29",
        "description": (
            "Post-tropical cyclone that caused massive damage to the northeastern United States."
        ),
        "sid": "2012296N14283",
        "basin": "NA",
        "source": "historical",
    },
    "matthew_2016": {
        "name": "Hurricane Matthew (2016)",
        "date": "2016-10-08",
        "description": (
            "Category 5 hurricane in the Caribbean that caused "
            "devastating impacts in Haiti and the southeastern US."
        ),
        "sid": "2016272N13318",
        "basin": "NA",
        "source": "historical",
    },
}


class HurricaneLoader(BaseDomainLoader):
    """
    Loader for hurricane/cyclone data from NOAA IBTrACS.

    Uses the International Best Track Archive for Climate Stewardship
    (IBTrACS) v04r01 CSV data to provide tropical cyclone track
    observations for anomaly detection.

    Data sources:

    * **Last 3 years** -- A smaller subset covering recent storms,
      suitable for near-real-time analysis.
    * **North Atlantic basin** -- Full historical archive of North
      Atlantic tropical cyclones for historical event analysis.

    Feature engineering produces meteorological observables suitable
    for anomaly detection: wind speed, central pressure, track
    coordinates, wind speed change (rapid intensification proxy),
    pressure drop rate, track deviation, and translation speed.
    """

    DOMAIN: str = "hurricane"
    SOURCE_URL: str = _BASE_CSV_URL
    REQUIRES_API_KEY: bool = False
    FEATURE_COLUMNS: list[str] = [
        "delta_wind_6h",
        "delta_wind_12h",
        "delta_wind_24h",
        "delta_pressure_6h",
        "delta_pressure_12h",
        "wind_pressure_deficit",
        "ri_interaction_24h",
        "ri_interaction_6h",
    ]

    # Cache for 6 hours since IBTrACS updates less frequently
    CACHE_TTL: int = 21600

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def fetch_realtime(self) -> pd.DataFrame:
        """
        Fetch recent tropical cyclone track data from IBTrACS.

        Retrieves the "last 3 years" subset which includes all
        storms from the most recent three seasons.

        Returns:
            DataFrame with columns: sid, season, name, iso_time, lat,
            lon, wind_kt, pressure_mb.

        Raises:
            ConnectionError: If the IBTrACS server is unreachable
                after retries.
        """
        cache_key = "hurricane_realtime_last3years"
        cached = self._read_cache(cache_key)

        if cached is not None:
            logger.debug("Returning cached real-time hurricane data.")
            return pd.DataFrame(cached)

        df = self._download_ibtracs_csv(_LAST3YEARS_URL)

        self._write_cache(cache_key, df.to_dict(orient="list"))
        logger.info(
            "Fetched %d real-time hurricane track records from IBTrACS.",
            len(df),
        )
        return df

    def fetch_historical(self, event_id: str) -> pd.DataFrame:
        """
        Fetch track data for a specific historical hurricane event.

        For events with a known Storm ID (SID), the data is filtered
        from the appropriate basin CSV.  For recent events without a
        pre-defined SID, the storm is matched by name and season from
        the last-3-years subset.

        Args:
            event_id: Key into the ground truth catalog (e.g.
                ``"katrina_2005"``).

        Returns:
            DataFrame with the same schema as :meth:`fetch_realtime`,
            filtered to the single storm track.

        Raises:
            ValueError: If *event_id* is not in the catalog.
            ConnectionError: If the IBTrACS server is unreachable.
        """
        if event_id not in _EVENT_CATALOG:
            raise ValueError(
                f"Unknown event_id '{event_id}'. Available: {list(_EVENT_CATALOG.keys())}"
            )

        cache_key = f"hurricane_historical_{event_id}"
        cached = self._read_cache(cache_key)

        if cached is not None:
            logger.debug("Returning cached historical data for '%s'.", event_id)
            return pd.DataFrame(cached)

        event = _EVENT_CATALOG[event_id]
        df = self._fetch_storm_track(event)

        if df.empty:
            logger.warning("IBTrACS returned no track data for event '%s'.", event_id)
            return df

        # Sort chronologically so time-series features are meaningful
        df = df.sort_values("iso_time").reset_index(drop=True)

        self._write_cache(cache_key, df.to_dict(orient="list"))
        logger.info(
            "Fetched %d historical track records for event '%s'.",
            len(df),
            event_id,
        )
        return df

    def list_events(self) -> list[dict[str, Any]]:
        """
        Return the catalog of ground truth hurricane events.

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
        """
        Generate binary anomaly labels for a historical hurricane event.

        Labeling strategy: a track observation is labeled *anomalous*
        (``1``) if it falls within a rapid intensification period,
        defined as a 24-hour window where the maximum sustained wind
        speed increased by at least 30 kt.  All other track points
        are labeled *normal* (``0``).

        Args:
            event_id: Key into the ground truth catalog.

        Returns:
            1-D binary numpy array of shape ``(n_observations,)``.

        Raises:
            ValueError: If *event_id* is not recognized or no data is
                available.
        """
        if event_id not in _EVENT_CATALOG:
            raise ValueError(
                f"Unknown event_id '{event_id}'. Available: {list(_EVENT_CATALOG.keys())}"
            )

        df = self.fetch_historical(event_id)
        if df.empty:
            return np.array([], dtype=np.int64)

        wind = df["wind_kt"].values.astype(np.float64)
        labels = self._label_rapid_intensification(wind)

        logger.info(
            "Ground truth for '%s': %d anomalies / %d total (RI threshold >= %d kt / 24 h).",
            event_id,
            int(labels.sum()),
            len(labels),
            _RI_THRESHOLD_KT,
        )
        return labels

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def engineer_features(self, raw_data: pd.DataFrame) -> np.ndarray:
        """Transform raw hurricane track data into a feature matrix.

        Only rapid-intensification-relevant derived features are
        retained.  Diagnostic analysis showed that raw observables
        (wind, pressure, lat, lon, storm speed) and track deviation
        introduce noise that degrades unsupervised AUC, while the
        multi-scale delta, deficit, and interaction features capture
        RI signal.

        Engineered features (per track observation):

        1. **delta_wind_6h** -- wind change over prior 6h (1 step).
        2. **delta_wind_12h** -- wind change over prior 12h (2 steps).
        3. **delta_wind_24h** -- wind change over prior 24h (4 steps).
           Values >= 30 kt indicate rapid intensification.
        4. **delta_pressure_6h** -- pressure drop over prior 6h
           (positive = intensification).
        5. **delta_pressure_12h** -- pressure drop over prior 12h.
        6. **wind_pressure_deficit** -- deviation from expected
            wind-pressure relationship.  Negative means storm has
            untapped pressure gradient and may precede RI.
        7. **ri_interaction_24h** -- ``delta_wind_24h * delta_pressure_12h``.
            Joint wind-pressure intensification at sustained (24h)
            timescale.  Large positive values indicate concurrent
            rapid wind increase and pressure drop characteristic of RI.
        8. **ri_interaction_6h** -- ``delta_wind_6h * delta_pressure_6h``.
            Same interaction at short (6h) timescale, capturing rapid
            RI onset.

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
        df = df.sort_values("iso_time").reset_index(drop=True)

        # ---- base observables (needed for derived features) ----
        wind_kt = df["wind_kt"].values.astype(np.float64)
        pressure_mb = df["pressure_mb"].values.astype(np.float64)

        # ---- multi-scale wind deltas (6h, 12h, 24h) ----
        delta_wind_6h = self._compute_delta(wind_kt, steps=1)
        delta_wind_12h = self._compute_delta(wind_kt, steps=2)
        delta_wind_24h = self._compute_delta(wind_kt, steps=_STEPS_24H)

        # ---- multi-scale pressure drops (negate so drop=positive) ----
        delta_pressure_6h = -self._compute_delta(pressure_mb, steps=1)
        delta_pressure_12h = -self._compute_delta(pressure_mb, steps=2)

        # ---- wind-pressure deficit ----
        # Expected wind from pressure via simplified Atkinson-Holliday:
        #   V_expected ≈ 6.7 * (1015 - P)^0.644  (knots)
        # Deficit = actual_wind - expected_wind. Negative means storm
        # has untapped pressure gradient → precedes RI.
        pressure_diff = np.maximum(1015.0 - pressure_mb, 0.0)
        expected_wind = 6.7 * np.power(pressure_diff + 1e-8, 0.644)
        wind_pressure_deficit = wind_kt - expected_wind

        # ---- RI interaction features (multi-scale) ----
        # Joint wind-pressure product amplifies the RI signature:
        # large wind increase concurrent with large pressure drop
        # produces a large positive product, while either alone
        # contributes much less.
        ri_interaction_24h = delta_wind_24h * delta_pressure_12h
        ri_interaction_6h = delta_wind_6h * delta_pressure_6h

        # Stack into feature matrix — only RI-relevant features.
        features = np.column_stack(
            [
                delta_wind_6h,
                delta_wind_12h,
                delta_wind_24h,
                delta_pressure_6h,
                delta_pressure_12h,
                wind_pressure_deficit,
                ri_interaction_24h,
                ri_interaction_6h,
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

    def _download_ibtracs_csv(self, url: str) -> pd.DataFrame:
        """
        Download and parse an IBTrACS CSV file.

        IBTrACS CSVs have a header row (row 0) followed by a units
        row (row 1) that must be skipped.  Many fields use whitespace
        or empty strings for missing values.

        Args:
            url: Full URL to the IBTrACS CSV file.

        Returns:
            DataFrame with standardized column names and cleaned
            numeric values.
        """
        raw_bytes = self._fetch_url(url)
        df = pd.read_csv(
            io.BytesIO(raw_bytes),
            skiprows=_IBTRACS_HEADER_ROWS,
            usecols=lambda c: c in _IBTRACS_COLUMNS,
            na_values=[" ", "", "  ", "   "],
            keep_default_na=True,
            low_memory=False,
        )

        return self._standardize_ibtracs(df)

    def _fetch_storm_track(self, event: dict[str, Any]) -> pd.DataFrame:
        """
        Fetch the track for a single storm from IBTrACS.

        Selects the appropriate data source (historical basin CSV or
        last-3-years subset) and filters by storm identifier or name.

        Args:
            event: Event metadata dict from ``_EVENT_CATALOG``.

        Returns:
            DataFrame filtered to the target storm track.
        """
        source = event.get("source", "historical")

        if source == "last3years":
            url = _LAST3YEARS_URL
        else:
            # Use basin-specific URL (North Atlantic by default)
            basin = event.get("basin", "NA")
            url = f"{_BASE_CSV_URL}ibtracs.{basin}.list.v04r01.csv"

        cache_key = f"hurricane_basin_csv_{url}"
        cached = self._read_cache(cache_key)

        if cached is not None:
            basin_df = pd.DataFrame(cached)
        else:
            basin_df = self._download_ibtracs_csv(url)
            self._write_cache(cache_key, basin_df.to_dict(orient="list"))

        # Filter by SID if available
        sid = event.get("sid")
        if sid is not None:
            storm_df = basin_df[basin_df["sid"] == sid].copy()
        else:
            # Match by name and approximate date for storms without SID
            name = event["name"]
            # Extract the short storm name from "Hurricane Katrina (2005)"
            storm_name = name.split("(")[0].strip()
            if storm_name.startswith("Hurricane "):
                storm_name = storm_name[len("Hurricane ") :]
            storm_name = storm_name.strip().upper()

            date_str = event["date"]
            year = int(date_str[:4])

            storm_df = basin_df[
                (basin_df["name"].str.upper() == storm_name) & (basin_df["season"] == year)
            ].copy()

        return storm_df.reset_index(drop=True)

    @staticmethod
    def _standardize_ibtracs(df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize IBTrACS column names and data types.

        Maps the raw IBTrACS column names to a consistent schema and
        resolves wind/pressure values by preferring USA agency data
        when WMO values are missing.

        Args:
            df: Raw DataFrame from IBTrACS CSV.

        Returns:
            DataFrame with columns: sid, season, name, iso_time, lat,
            lon, wind_kt, pressure_mb.
        """
        result = pd.DataFrame()

        result["sid"] = df["SID"].astype(str) if "SID" in df.columns else ""
        result["season"] = (
            pd.to_numeric(df["SEASON"], errors="coerce").astype("Int64")
            if "SEASON" in df.columns
            else pd.NA
        )
        result["name"] = df["NAME"].astype(str).str.strip() if "NAME" in df.columns else ""
        result["iso_time"] = (
            df["ISO_TIME"].astype(str).str.strip() if "ISO_TIME" in df.columns else ""
        )

        # Latitude / longitude
        result["lat"] = pd.to_numeric(df["LAT"], errors="coerce") if "LAT" in df.columns else np.nan
        result["lon"] = pd.to_numeric(df["LON"], errors="coerce") if "LON" in df.columns else np.nan

        # Wind speed: prefer WMO, fall back to USA
        wmo_wind = (
            pd.to_numeric(df["WMO_WIND"], errors="coerce")
            if "WMO_WIND" in df.columns
            else pd.Series(np.nan, index=df.index)
        )
        usa_wind = (
            pd.to_numeric(df["USA_WIND"], errors="coerce")
            if "USA_WIND" in df.columns
            else pd.Series(np.nan, index=df.index)
        )
        result["wind_kt"] = wmo_wind.fillna(usa_wind)

        # Pressure: prefer WMO, fall back to USA
        wmo_pres = (
            pd.to_numeric(df["WMO_PRES"], errors="coerce")
            if "WMO_PRES" in df.columns
            else pd.Series(np.nan, index=df.index)
        )
        usa_pres = (
            pd.to_numeric(df["USA_PRES"], errors="coerce")
            if "USA_PRES" in df.columns
            else pd.Series(np.nan, index=df.index)
        )
        result["pressure_mb"] = wmo_pres.fillna(usa_pres)

        # Nature of the system (tropical, subtropical, etc.)
        if "NATURE" in df.columns:
            result["nature"] = df["NATURE"].astype(str).str.strip()

        return result

    # ------------------------------------------------------------------
    # Private helpers -- feature computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_delta(values: np.ndarray, steps: int) -> np.ndarray:
        """
        Compute the change in a variable over a fixed number of steps.

        Args:
            values: 1-D array of observations (chronologically ordered).
            steps: Number of steps back to compute the difference.

        Returns:
            1-D array of deltas.  The first *steps* entries are set to 0.
        """
        n = len(values)
        delta = np.zeros(n, dtype=np.float64)
        if n > steps:
            delta[steps:] = values[steps:] - values[:-steps]
        return delta

    @staticmethod
    def _compute_track_deviation(
        lat: np.ndarray,
        lon: np.ndarray,
        window: int = 5,
    ) -> np.ndarray:
        """
        Compute deviation of track position from a running mean.

        The deviation is the Euclidean distance (in degrees) between
        the actual position and the centroid of a trailing window of
        positions.  Large deviations may indicate erratic track motion.

        Args:
            lat: 1-D array of latitudes.
            lon: 1-D array of longitudes.
            window: Size of the trailing window for the running mean.

        Returns:
            1-D array of deviation distances in degrees.
        """
        n = len(lat)
        deviation = np.zeros(n, dtype=np.float64)

        for i in range(n):
            start = max(0, i - window + 1)
            mean_lat = np.nanmean(lat[start : i + 1])
            mean_lon = np.nanmean(lon[start : i + 1])
            dlat = lat[i] - mean_lat
            dlon = lon[i] - mean_lon
            deviation[i] = np.sqrt(dlat**2 + dlon**2)

        return deviation

    @staticmethod
    def _compute_translation_speed(
        lat: np.ndarray,
        lon: np.ndarray,
    ) -> np.ndarray:
        """
        Compute storm translation speed between consecutive track points.

        Uses a simplified great-circle distance approximation suitable
        for the typical scales of tropical cyclone tracks.  Speed is
        expressed in degrees per 6-hour interval.

        Args:
            lat: 1-D array of latitudes (degrees).
            lon: 1-D array of longitudes (degrees).

        Returns:
            1-D array of translation speeds.  The first entry is 0.
        """
        n = len(lat)
        speed = np.zeros(n, dtype=np.float64)

        if n < 2:
            return speed

        # Convert to radians for Haversine approximation
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        for i in range(1, n):
            dlat = lat_rad[i] - lat_rad[i - 1]
            dlon = lon_rad[i] - lon_rad[i - 1]
            # Haversine formula
            a = (
                np.sin(dlat / 2.0) ** 2
                + np.cos(lat_rad[i - 1]) * np.cos(lat_rad[i]) * np.sin(dlon / 2.0) ** 2
            )
            c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
            # Return in degrees for consistency with track coordinates
            speed[i] = np.degrees(c)

        return speed

    @staticmethod
    def _label_rapid_intensification(
        wind_kt: np.ndarray,
    ) -> np.ndarray:
        """
        Label track observations during rapid intensification periods.

        A rapid intensification (RI) event is defined as a wind speed
        increase of >= 30 kt over any 24-hour period (4 six-hourly
        observations).  All observations within such a 24-h window are
        labeled as anomalous.

        Args:
            wind_kt: 1-D array of maximum sustained wind speeds in knots.

        Returns:
            1-D binary numpy array (0=normal, 1=anomaly).
        """
        n = len(wind_kt)
        labels = np.zeros(n, dtype=np.int64)

        for i in range(_STEPS_24H, n):
            wind_change = wind_kt[i] - wind_kt[i - _STEPS_24H]
            if wind_change >= _RI_THRESHOLD_KT:
                # Label the entire 24-h window as anomalous
                start = i - _STEPS_24H
                labels[start : i + 1] = 1

        return labels
