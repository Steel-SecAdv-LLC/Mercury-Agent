"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

Domain loader for volcanic activity data from the USGS Volcano Hazards Program.

Connects to the USGS Volcano Hazards Program API to provide volcanic alert
data for Mercury anomaly detection.  Ground truth events cover major
eruptions where WARNING/RED alert levels are labeled as anomalies against
a background of NORMAL/GREEN activity.

Data sources:
- Real-time alerts: https://volcanoes.usgs.gov/vsc/api/volcanoApi/alerts
- Volcano list: https://volcanoes.usgs.gov/vsc/api/volcanoApi/volcanoList
- Smithsonian GVP (supplemental): eruption records
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from omni_mercury_engine.datasets.exceptions import DataSourceUnavailableError
from omni_mercury_engine.loaders.base import BaseDomainLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# USGS Volcano API endpoints
# ---------------------------------------------------------------------------
_API_BASE = "https://volcanoes.usgs.gov/vsc/api/volcanoApi/"
_ALERTS_URL = f"{_API_BASE}alerts"
_VOLCANO_LIST_URL = f"{_API_BASE}volcanoList"

# ---------------------------------------------------------------------------
# Alert level and color code encodings
# ---------------------------------------------------------------------------
_ALERT_LEVEL_MAP: dict[str, int] = {
    "NORMAL": 0,
    "ADVISORY": 1,
    "WATCH": 2,
    "WARNING": 3,
}

_COLOR_CODE_MAP: dict[str, int] = {
    "GREEN": 0,
    "YELLOW": 1,
    "ORANGE": 2,
    "RED": 3,
}

# ---------------------------------------------------------------------------
# Ground truth event catalog
# ---------------------------------------------------------------------------
_EVENT_CATALOG: dict[str, dict[str, Any]] = {
    "tonga_2022": {
        "name": "2022 Hunga Tonga Eruption",
        "date": "2022-01-15",
        "description": (
            "VEI 5-6 eruption of Hunga Tonga-Hunga Ha'apai submarine volcano. "
            "Generated atmospheric shockwaves and tsunami observed globally."
        ),
        "volcano_name": "Hunga Tonga-Hunga Ha'apai",
        "vei": 5,
        "start": "2022-01-14",
        "end": "2022-01-20",
        "latitude": -20.536,
        "longitude": -175.382,
        "elevation": -114,
    },
    "cumbre_vieja_2021": {
        "name": "2021 Cumbre Vieja Eruption",
        "date": "2021-09-19",
        "description": (
            "Eruption of Cumbre Vieja volcano on La Palma, Canary Islands. "
            "Lasted approximately 85 days, destroying thousands of structures."
        ),
        "volcano_name": "Cumbre Vieja",
        "vei": 3,
        "start": "2021-09-10",
        "end": "2021-12-25",
        "latitude": 28.57,
        "longitude": -17.84,
        "elevation": 2426,
    },
    "kilauea_2018": {
        "name": "2018 Kilauea Eruption",
        "date": "2018-05-03",
        "description": (
            "Lower East Rift Zone eruption of Kilauea, Hawaii. "
            "Produced extensive lava flows and summit caldera collapse."
        ),
        "volcano_name": "Kilauea",
        "vei": 4,
        "start": "2018-05-01",
        "end": "2018-08-10",
        "latitude": 19.421,
        "longitude": -155.287,
        "elevation": 1222,
    },
    "eyjafjallajokull_2010": {
        "name": "2010 Eyjafjallajokull Eruption",
        "date": "2010-04-14",
        "description": (
            "Eruption of Eyjafjallajokull in Iceland. "
            "Ash cloud caused unprecedented European airspace closures."
        ),
        "volcano_name": "Eyjafjallajokull",
        "vei": 4,
        "start": "2010-04-01",
        "end": "2010-05-30",
        "latitude": 63.633,
        "longitude": -19.633,
        "elevation": 1666,
    },
}


class VolcanicLoader(BaseDomainLoader):
    """
    Loader for volcanic activity data from the USGS Volcano Hazards Program.

    Uses the USGS Volcano API endpoints:

    * **Real-time alerts** -- Current volcanic alert levels and aviation
      color codes for monitored U.S. volcanoes.
    * **Volcano list** -- Catalog of monitored volcanoes with geographic
      coordinates and metadata.

    Feature engineering produces volcanological observables suitable for
    anomaly detection: alert level encoding, color code encoding,
    geographic features, and temporal features including days since last
    alert change.

    Ground truth labeling strategy:

    * WARNING / RED alert levels are labeled anomaly (1).
    * NORMAL / GREEN levels are labeled normal (0).
    * ADVISORY / WATCH levels are treated as borderline and labeled
      normal (0) under conservative labeling.

    Note:
        The USGS volcano API may return limited historical data.  For
        events where API data is insufficient, the loader raises
        :class:`DataSourceUnavailableError` rather than fabricating data.
    """

    DOMAIN: str = "volcanic"
    SOURCE_URL: str = _API_BASE
    REQUIRES_API_KEY: bool = False
    FEATURE_COLUMNS: list[str] = [
        "alert_level_numeric",
        "color_code_numeric",
        "alert_level_delta",
        "color_code_delta",
        "latitude",
        "longitude",
        "elevation",
        "days_since_last_change",
    ]

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def fetch_realtime(self) -> pd.DataFrame:
        """
        Fetch current volcanic alert data from the USGS Volcano API.

        Retrieves the latest alert levels and aviation color codes for
        all monitored volcanoes, then enriches with geographic metadata
        from the volcano list endpoint.

        Returns:
            DataFrame with columns: volcano_name, alert_level, color_code,
            alert_level_numeric, color_code_numeric, latitude, longitude,
            elevation, alert_date.

        Raises:
            ConnectionError: If the USGS API is unreachable after retries.
        """
        cache_key = "volcanic_realtime"
        cached = self._read_cache(cache_key)

        if cached is not None:
            logger.debug("Returning cached real-time volcanic alert data.")
            return pd.DataFrame(cached)

        try:
            alerts_raw = self._fetch_json(_ALERTS_URL)
        except ConnectionError:
            raise
        except Exception as exc:
            raise ConnectionError(
                f"volcanic: Failed to fetch alerts from {_ALERTS_URL}: {exc}"
            ) from exc

        df = self._alerts_to_dataframe(alerts_raw)

        # Enrich with geographic data from the volcano list
        df = self._enrich_with_geography(df)

        self._write_cache(cache_key, df.to_dict(orient="list"))
        logger.info("Fetched %d real-time volcanic alert records from USGS.", len(df))
        return df

    def fetch_historical(self, event_id: str) -> pd.DataFrame:
        """
        Fetch volcanic activity data for a specific historical eruption.

        Attempts to retrieve alert-level data from the USGS API for the
        specified event.  Because the USGS volcano API provides limited
        historical data, this method may raise
        :class:`DataSourceUnavailableError` when the API does not contain
        sufficient records for a given event.

        Args:
            event_id: Key into the ground truth catalog (e.g.
                ``"tonga_2022"``).

        Returns:
            DataFrame with the same schema as :meth:`fetch_realtime`.

        Raises:
            ValueError: If *event_id* is not in the catalog.
            DataSourceUnavailableError: If the USGS API does not provide
                sufficient historical data for the requested event.
        """
        if event_id not in _EVENT_CATALOG:
            raise ValueError(
                f"Unknown event_id '{event_id}'. " f"Available: {list(_EVENT_CATALOG.keys())}"
            )

        cache_key = f"volcanic_historical_{event_id}"
        cached = self._read_cache(cache_key)

        if cached is not None:
            logger.debug("Returning cached historical data for '%s'.", event_id)
            return pd.DataFrame(cached)

        event = _EVENT_CATALOG[event_id]

        # The USGS volcano API exposes current alerts but does not provide
        # a parameterized historical query endpoint.  We attempt to fetch
        # the current alerts and filter for the volcano of interest.  If
        # the event is not represented in the current data (which is the
        # common case for past eruptions), we raise an error rather than
        # fabricating data.
        try:
            alerts_raw = self._fetch_json(_ALERTS_URL)
        except Exception as exc:
            raise DataSourceUnavailableError(
                loader_name="VolcanicLoader",
                source_url=_ALERTS_URL,
                reason=(f"Cannot reach USGS volcano API for event " f"'{event_id}': {exc}"),
            ) from exc

        df = self._alerts_to_dataframe(alerts_raw)

        if df.empty:
            raise DataSourceUnavailableError(
                loader_name="VolcanicLoader",
                source_url=_ALERTS_URL,
                reason=(
                    f"USGS volcano API returned no alert data. "
                    f"Historical data for event '{event_id}' "
                    f"({event['name']}) is unavailable."
                ),
            )

        # Filter for the specific volcano name
        volcano_name = event["volcano_name"]
        mask = df["volcano_name"].str.contains(volcano_name, case=False, na=False)
        df_filtered = df[mask].copy()

        if df_filtered.empty:
            # API does not contain data for this eruption -- this is
            # expected for historical events outside the current alert
            # window.  Raise rather than fabricate.
            raise DataSourceUnavailableError(
                loader_name="VolcanicLoader",
                source_url=_ALERTS_URL,
                reason=(
                    f"No alert records found for volcano "
                    f"'{volcano_name}' in current USGS data. "
                    f"Historical data for event '{event_id}' "
                    f"({event['name']}, {event['date']}) is not "
                    f"available from the real-time API."
                ),
            )

        # Enrich with geographic data
        df_filtered = self._enrich_with_geography(df_filtered)

        # Sort chronologically if dates are available
        if "alert_date" in df_filtered.columns:
            df_filtered = df_filtered.sort_values("alert_date").reset_index(drop=True)

        self._write_cache(cache_key, df_filtered.to_dict(orient="list"))
        logger.info(
            "Fetched %d historical records for volcanic event '%s'.",
            len(df_filtered),
            event_id,
        )
        return df_filtered

    def list_events(self) -> list[dict[str, Any]]:
        """
        Return the catalog of ground truth volcanic eruption events.

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
        """
        Generate binary anomaly labels for a historical volcanic event.

        Labeling strategy (conservative):

        * **WARNING** alert level or **RED** aviation color code is
          labeled anomalous (``1``).
        * **NORMAL** alert level or **GREEN** color code is labeled
          normal (``0``).
        * **ADVISORY** and **WATCH** levels are borderline and labeled
          normal (``0``) under the conservative approach.

        Args:
            event_id: Key into the ground truth catalog.

        Returns:
            1-D binary numpy array of shape ``(n_records,)``.

        Raises:
            ValueError: If *event_id* is not recognized.
            DataSourceUnavailableError: If historical data is
                unavailable from the API.
        """
        if event_id not in _EVENT_CATALOG:
            raise ValueError(
                f"Unknown event_id '{event_id}'. " f"Available: {list(_EVENT_CATALOG.keys())}"
            )

        df = self.fetch_historical(event_id)

        if df.empty:
            return np.array([], dtype=np.int64)

        # Anomaly = WARNING (3) alert level OR RED (3) color code
        alert_numeric = df["alert_level_numeric"].values.astype(np.float64)
        color_numeric = df["color_code_numeric"].values.astype(np.float64)

        labels = (
            (alert_numeric >= _ALERT_LEVEL_MAP["WARNING"])
            | (color_numeric >= _COLOR_CODE_MAP["RED"])
        ).astype(np.int64)

        logger.info(
            "Ground truth for '%s': %d anomalies / %d total " "(WARNING/RED threshold).",
            event_id,
            int(labels.sum()),
            len(labels),
        )
        return np.asarray(labels)

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def engineer_features(self, raw_data: pd.DataFrame) -> np.ndarray[Any, Any]:
        """
        Transform raw volcanic alert data into a feature matrix.

        Engineered features (per record):

        1. **alert_level_numeric** -- Numeric encoding of alert level
           (NORMAL=0, ADVISORY=1, WATCH=2, WARNING=3).
        2. **color_code_numeric** -- Numeric encoding of aviation color
           code (GREEN=0, YELLOW=1, ORANGE=2, RED=3).
        3. **alert_level_delta** -- Change in alert level from the
           previous record (0 for the first row).
        4. **color_code_delta** -- Change in color code from the
           previous record (0 for the first row).
        5. **latitude** -- Volcano latitude.
        6. **longitude** -- Volcano longitude.
        7. **elevation** -- Volcano summit elevation in meters.
        8. **days_since_last_change** -- Days since the most recent
           alert level change (0 if no change detected or for the
           first row).

        Args:
            raw_data: DataFrame from :meth:`fetch_realtime` or
                :meth:`fetch_historical`.

        Returns:
            2-D numpy array of shape ``(n_samples, 8)``.
        """
        n_features = 8
        if raw_data.empty:
            return np.empty((0, n_features), dtype=np.float64)

        df = raw_data.copy()

        # Ensure numeric encodings exist
        if "alert_level_numeric" not in df.columns:
            df["alert_level_numeric"] = (
                df["alert_level"].str.upper().map(_ALERT_LEVEL_MAP).fillna(0).astype(np.float64)
            )
        if "color_code_numeric" not in df.columns:
            df["color_code_numeric"] = (
                df["color_code"].str.upper().map(_COLOR_CODE_MAP).fillna(0).astype(np.float64)
            )

        alert_level_numeric = df["alert_level_numeric"].values.astype(np.float64)
        color_code_numeric = df["color_code_numeric"].values.astype(np.float64)

        # ---- Alert level delta (change from previous record) ----
        alert_level_delta = self._compute_deltas(alert_level_numeric)

        # ---- Color code delta (change from previous record) ----
        color_code_delta = self._compute_deltas(color_code_numeric)

        # ---- Geographic features ----
        latitude = (
            df["latitude"].values.astype(np.float64)
            if "latitude" in df.columns
            else np.zeros(len(df), dtype=np.float64)
        )
        longitude = (
            df["longitude"].values.astype(np.float64)
            if "longitude" in df.columns
            else np.zeros(len(df), dtype=np.float64)
        )
        elevation = (
            df["elevation"].values.astype(np.float64)
            if "elevation" in df.columns
            else np.zeros(len(df), dtype=np.float64)
        )

        # ---- Days since last alert change ----
        days_since_last_change = self._compute_days_since_last_change(df, alert_level_numeric)

        # Stack into feature matrix
        features = np.column_stack(
            [
                alert_level_numeric,
                color_code_numeric,
                alert_level_delta,
                color_code_delta,
                latitude,
                longitude,
                elevation,
                days_since_last_change,
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
    def _alerts_to_dataframe(
        alerts_raw: Any,
    ) -> pd.DataFrame:
        """
        Convert raw USGS volcano alert API response to a flat DataFrame.

        The USGS volcano alert API returns a list of alert objects.  This
        method normalizes the response into a tabular format.

        Args:
            alerts_raw: Parsed JSON from the USGS volcano alerts endpoint.
                Expected to be a list of dicts or a dict with an alerts
                key containing a list.

        Returns:
            DataFrame with columns: volcano_name, alert_level,
            color_code, alert_level_numeric, color_code_numeric,
            alert_date.
        """
        columns = [
            "volcano_name",
            "alert_level",
            "color_code",
            "alert_level_numeric",
            "color_code_numeric",
            "alert_date",
        ]

        # Handle different response formats from the API
        if isinstance(alerts_raw, dict):
            alerts_list = alerts_raw.get("alerts", alerts_raw.get("features", []))
            if not isinstance(alerts_list, list):
                alerts_list = []
        elif isinstance(alerts_raw, list):
            alerts_list = alerts_raw
        else:
            return pd.DataFrame(columns=columns)

        if not alerts_list:
            return pd.DataFrame(columns=columns)

        rows: list[dict[str, Any]] = []
        for alert in alerts_list:
            if not isinstance(alert, dict):
                continue

            # Support both flat format and nested properties format
            props = alert.get("properties", alert)

            volcano_name = (
                props.get("volcanoName", "")
                or props.get("volcano_name", "")
                or props.get("name", "")
            )
            alert_level = props.get("alertLevel", "") or props.get("alert_level", "") or ""
            color_code = (
                props.get("colorCode", "")
                or props.get("color_code", "")
                or props.get("aviationColorCode", "")
                or ""
            )
            alert_date = (
                props.get("alertDate", "")
                or props.get("date", "")
                or props.get("updated", "")
                or ""
            )

            alert_upper = str(alert_level).upper().strip()
            color_upper = str(color_code).upper().strip()

            rows.append(
                {
                    "volcano_name": str(volcano_name).strip(),
                    "alert_level": alert_upper,
                    "color_code": color_upper,
                    "alert_level_numeric": _ALERT_LEVEL_MAP.get(alert_upper, 0),
                    "color_code_numeric": _COLOR_CODE_MAP.get(color_upper, 0),
                    "alert_date": alert_date,
                }
            )

        df = pd.DataFrame(rows, columns=columns)
        return df

    def _enrich_with_geography(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich alert DataFrame with geographic data from the volcano list.

        Fetches the USGS volcano list and merges latitude, longitude, and
        elevation onto the alert records by matching volcano name.

        Args:
            df: Alert DataFrame with a ``volcano_name`` column.

        Returns:
            DataFrame with added latitude, longitude, and elevation
            columns (or NaN where no match is found).
        """
        if df.empty:
            for col in ("latitude", "longitude", "elevation"):
                if col not in df.columns:
                    df[col] = pd.Series(dtype=np.float64)
            return df

        # Skip if geography is already present
        if all(col in df.columns for col in ("latitude", "longitude", "elevation")):
            return df

        cache_key = "volcanic_volcano_list"
        cached_list = self._read_cache(cache_key)

        if cached_list is not None:
            geo_df = pd.DataFrame(cached_list)
        else:
            try:
                volcanoes_raw = self._fetch_json(_VOLCANO_LIST_URL)
                geo_df = self._volcano_list_to_geodf(volcanoes_raw)
                if not geo_df.empty:
                    self._write_cache(cache_key, geo_df.to_dict(orient="list"))
            except Exception as exc:
                logger.warning(
                    "Could not fetch volcano list for geographic "
                    "enrichment: %s. Filling with NaN.",
                    exc,
                )
                for col in ("latitude", "longitude", "elevation"):
                    if col not in df.columns:
                        df[col] = np.nan
                return df

        if geo_df.empty:
            for col in ("latitude", "longitude", "elevation"):
                if col not in df.columns:
                    df[col] = np.nan
            return df

        # Merge on volcano name (case-insensitive)
        df_lower = df["volcano_name"].str.lower().str.strip()
        geo_lower = geo_df["volcano_name"].str.lower().str.strip()

        geo_lookup: dict[str, dict[str, float]] = {}
        for idx, name in enumerate(geo_lower):
            if name and name not in geo_lookup:
                geo_lookup[name] = {
                    "latitude": float(geo_df.iloc[idx].get("latitude", np.nan)),
                    "longitude": float(geo_df.iloc[idx].get("longitude", np.nan)),
                    "elevation": float(geo_df.iloc[idx].get("elevation", np.nan)),
                }

        latitudes: list[float] = []
        longitudes: list[float] = []
        elevations: list[float] = []

        for name in df_lower:
            match = geo_lookup.get(name, {})
            latitudes.append(match.get("latitude", np.nan))
            longitudes.append(match.get("longitude", np.nan))
            elevations.append(match.get("elevation", np.nan))

        df = df.copy()
        df["latitude"] = latitudes
        df["longitude"] = longitudes
        df["elevation"] = elevations

        return df

    @staticmethod
    def _volcano_list_to_geodf(
        volcanoes_raw: Any,
    ) -> pd.DataFrame:
        """
        Convert raw USGS volcano list response to a geographic DataFrame.

        Args:
            volcanoes_raw: Parsed JSON from the USGS volcano list endpoint.

        Returns:
            DataFrame with columns: volcano_name, latitude, longitude,
            elevation.
        """
        columns = ["volcano_name", "latitude", "longitude", "elevation"]

        if isinstance(volcanoes_raw, dict):
            volcanoes_list = volcanoes_raw.get("volcanoList", volcanoes_raw.get("features", []))
            if not isinstance(volcanoes_list, list):
                volcanoes_list = []
        elif isinstance(volcanoes_raw, list):
            volcanoes_list = volcanoes_raw
        else:
            return pd.DataFrame(columns=columns)

        if not volcanoes_list:
            return pd.DataFrame(columns=columns)

        rows: list[dict[str, Any]] = []
        for volcano in volcanoes_list:
            if not isinstance(volcano, dict):
                continue

            props = volcano.get("properties", volcano)
            geometry = volcano.get("geometry", {})
            coords = geometry.get("coordinates", []) if geometry else []

            name = (
                props.get("volcanoName", "")
                or props.get("name", "")
                or props.get("volcano_name", "")
                or ""
            )

            # Try coordinates from geometry first, then from properties
            if coords and len(coords) >= 2:
                lon = coords[0]
                lat = coords[1]
            else:
                lat = props.get("latitude", props.get("lat", None))
                lon = props.get("longitude", props.get("lon", None))

            elev = props.get(
                "elevation",
                props.get("elev", props.get("summit_elevation", None)),
            )

            rows.append(
                {
                    "volcano_name": str(name).strip(),
                    "latitude": lat,
                    "longitude": lon,
                    "elevation": elev,
                }
            )

        geo_df = pd.DataFrame(rows, columns=columns)
        for col in ("latitude", "longitude", "elevation"):
            geo_df[col] = pd.to_numeric(geo_df[col], errors="coerce")
        return geo_df

    @staticmethod
    def _compute_deltas(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        Compute the change from the previous value in a 1-D array.

        Args:
            values: 1-D numeric array.

        Returns:
            1-D array of deltas (0 for the first element).
        """
        deltas = np.zeros(len(values), dtype=np.float64)
        if len(values) > 1:
            deltas[1:] = np.diff(values)
        return deltas

    @staticmethod
    def _compute_days_since_last_change(
        df: pd.DataFrame,
        alert_level_numeric: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """
        Compute days since the most recent alert level change.

        Uses the ``alert_date`` column if available; otherwise falls
        back to row-index-based approximation (treating each row as
        one observation period).

        Args:
            df: DataFrame containing an optional ``alert_date`` column.
            alert_level_numeric: 1-D array of numeric alert levels.

        Returns:
            1-D array of days since the last alert level change.
        """
        n = len(alert_level_numeric)
        days_since = np.zeros(n, dtype=np.float64)

        # Try to parse alert_date into datetime for accurate calculation
        timestamps: np.ndarray[Any, Any] | None = None
        if "alert_date" in df.columns:
            try:
                parsed_dates = pd.to_datetime(df["alert_date"], errors="coerce", utc=True)
                if not parsed_dates.isna().all():
                    timestamps = parsed_dates.values
            except Exception:
                timestamps = None

        last_change_idx = 0
        for i in range(n):
            if i == 0:
                days_since[i] = 0.0
                last_change_idx = 0
                continue

            # Detect if alert level changed
            if alert_level_numeric[i] != alert_level_numeric[i - 1]:
                last_change_idx = i

            # Calculate days since last change
            if timestamps is not None:
                current_ts = timestamps[i]
                change_ts = timestamps[last_change_idx]
                if pd.notna(current_ts) and pd.notna(change_ts):
                    delta = current_ts - change_ts
                    # Convert numpy timedelta64 to days
                    days_since[i] = float(delta / np.timedelta64(1, "D"))
                else:
                    # Fallback: row distance as proxy
                    days_since[i] = float(i - last_change_idx)
            else:
                # Fallback: row distance as proxy
                days_since[i] = float(i - last_change_idx)

        return days_since
