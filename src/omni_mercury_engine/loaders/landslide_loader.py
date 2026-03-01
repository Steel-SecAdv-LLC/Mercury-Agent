"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Domain loader for landslide data from NASA Global Landslide Catalog (COOLR).

Connects to the NASA Cooperative Open Online Landslide Repository (COOLR)
ArcGIS REST endpoint to provide global landslide event data for Mercury
anomaly detection.  Ground truth events cover major catastrophic landslides
where fatal or large-scale events are labeled as anomalies against a
background of smaller, non-fatal slope failures.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from omni_mercury_engine.loaders.base import BaseDomainLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NASA COOLR ArcGIS REST API endpoint
# ---------------------------------------------------------------------------
_QUERY_URL = (
    "https://maps.nccs.nasa.gov/arcgis/rest/services/"
    "global_landslide_catalog/global_landslide_catalog_export/"
    "FeatureServer/0/query"
)

# ---------------------------------------------------------------------------
# Category and trigger encoding maps (Mercury-native)
# ---------------------------------------------------------------------------
_CATEGORY_ENCODING: dict[str, int] = {
    "landslide": 1,
    "mudslide": 2,
    "rockfall": 3,
    "rock_slide": 4,
    "debris_flow": 5,
    "earth_flow": 6,
    "lahar": 7,
    "avalanche": 8,
    "riverbank_collapse": 9,
    "creep": 10,
    "complex": 11,
    "other": 12,
    "unknown": 0,
}

_TRIGGER_ENCODING: dict[str, int] = {
    "rain": 1,
    "downpour": 2,
    "continuous_rain": 3,
    "monsoon": 4,
    "tropical_cyclone": 5,
    "earthquake": 6,
    "flooding": 7,
    "snowfall_snowmelt": 8,
    "construction": 9,
    "mining": 10,
    "vibration": 11,
    "volcano": 12,
    "dam_embankment_collapse": 13,
    "freeze_thaw": 14,
    "unknown": 0,
}

_SIZE_ENCODING: dict[str, int] = {
    "small": 1,
    "medium": 2,
    "large": 3,
    "very_large": 4,
    "unknown": 0,
}

# ---------------------------------------------------------------------------
# Ground truth event catalog
# ---------------------------------------------------------------------------
_EVENT_CATALOG: dict[str, dict[str, Any]] = {
    "oso_2014": {
        "name": "2014 Oso, Washington Landslide",
        "date": "2014-03-22",
        "description": (
            "Catastrophic mudslide in Oso, Washington that killed 43 people "
            "and destroyed the Steelhead Haven community."
        ),
        "latitude": 48.28,
        "longitude": -121.84,
        "search_radius_km": 200.0,
        "start": "2014-03-01",
        "end": "2014-04-15",
    },
    "sierra_leone_2017": {
        "name": "2017 Freetown, Sierra Leone Landslide",
        "date": "2017-08-14",
        "description": (
            "Massive mudslide near Freetown, Sierra Leone triggered by heavy "
            "rains during the monsoon season, killing over 1,000 people."
        ),
        "latitude": 8.45,
        "longitude": -13.25,
        "search_radius_km": 300.0,
        "start": "2017-07-01",
        "end": "2017-09-30",
    },
    "japan_2018": {
        "name": "2018 Japan Landslides from Heavy Rain",
        "date": "2018-07-06",
        "description": (
            "Widespread landslides and flooding across western Japan caused "
            "by unprecedented heavy rainfall, killing over 200 people."
        ),
        "latitude": 34.3,
        "longitude": 132.5,
        "search_radius_km": 500.0,
        "start": "2018-06-15",
        "end": "2018-08-15",
    },
}


class LandslideLoader(BaseDomainLoader):
    """Loader for landslide data from NASA COOLR (Global Landslide Catalog).

    Uses the NASA Cooperative Open Online Landslide Repository ArcGIS REST
    API to fetch global landslide event records.

    * **Real-time feed** -- Most recent 2000 landslide events from the
      global catalog, updated as reports are submitted.
    * **Historical query** -- Spatial and temporal filtering for events
      around known catastrophic landslides.

    Feature engineering produces landslide-specific observables suitable
    for anomaly detection: event category, trigger type, fatality/injury
    counts, landslide size, geographic location, and temporal features.

    All math is Mercury-native (numpy only).
    """

    DOMAIN: str = "landslide"
    SOURCE_URL: str = "https://maps.nccs.nasa.gov/arcgis/rest/services/" "global_landslide_catalog/"
    REQUIRES_API_KEY: bool = False
    FEATURE_COLUMNS: list[str] = [
        "category_code",
        "fatality_count",
        "injury_count",
        "trigger_code",
        "latitude",
        "longitude",
        "size_code",
        "country_code",
        "month",
        "day_of_year",
    ]

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def fetch_realtime(self) -> pd.DataFrame:
        """Fetch the most recent landslide events from NASA COOLR.

        Queries the ArcGIS REST endpoint for up to 2000 of the most recent
        landslide reports in the global catalog.

        Returns:
            DataFrame with columns derived from NASA COOLR attributes
            including event category, trigger, location, fatalities, etc.

        Raises:
            ConnectionError: If the NASA COOLR service is unreachable
                after retries.
        """
        cache_key = "landslide_realtime"
        cached = self._read_cache(cache_key)

        if cached is not None:
            logger.debug("Returning cached real-time landslide data.")
            return pd.DataFrame(cached)

        params: dict[str, str] = {
            "where": "1=1",
            "outFields": "*",
            "f": "json",
            "resultRecordCount": "2000",
        }

        response = self._fetch_json(_QUERY_URL, params=params)
        df = self._arcgis_to_dataframe(response)

        self._write_cache(cache_key, df.to_dict(orient="list"))
        logger.info(
            "Fetched %d real-time landslide records from NASA COOLR.",
            len(df),
        )
        return df

    def fetch_historical(self, event_id: str) -> pd.DataFrame:
        """Fetch landslide catalog data surrounding a known historical event.

        Constructs a spatial and temporal query around the event epicenter
        to retrieve all landslide reports in the area during the event
        window.

        Args:
            event_id: Key into the ground truth catalog (e.g.
                ``"oso_2014"``).

        Returns:
            DataFrame with the same schema as :meth:`fetch_realtime`.

        Raises:
            ValueError: If *event_id* is not in the catalog.
            ConnectionError: If the NASA COOLR service is unreachable.
        """
        if event_id not in _EVENT_CATALOG:
            raise ValueError(
                f"Unknown event_id '{event_id}'. " f"Available: {list(_EVENT_CATALOG.keys())}"
            )

        cache_key = f"landslide_historical_{event_id}"
        cached = self._read_cache(cache_key)

        if cached is not None:
            logger.debug("Returning cached historical data for '%s'.", event_id)
            return pd.DataFrame(cached)

        event = _EVENT_CATALOG[event_id]

        # Build spatial/temporal where clause for ArcGIS query
        where_clause = self._build_where_clause(
            lat=event["latitude"],
            lon=event["longitude"],
            radius_km=event["search_radius_km"],
            start_date=event["start"],
            end_date=event["end"],
        )

        params: dict[str, str] = {
            "where": where_clause,
            "outFields": "*",
            "f": "json",
            "resultRecordCount": "2000",
        }

        response = self._fetch_json(_QUERY_URL, params=params)
        df = self._arcgis_to_dataframe(response)

        if df.empty:
            logger.warning(
                "NASA COOLR returned no features for event '%s'. "
                "Falling back to full catalog query.",
                event_id,
            )
            # Fallback: try a broader temporal-only query
            fallback_where = (
                f"event_date >= '{event['start']}' " f"AND event_date <= '{event['end']}'"
            )
            fallback_params: dict[str, str] = {
                "where": fallback_where,
                "outFields": "*",
                "f": "json",
                "resultRecordCount": "2000",
            }
            response = self._fetch_json(_QUERY_URL, params=fallback_params)
            df = self._arcgis_to_dataframe(response)

        if not df.empty:
            df = df.sort_values("event_date").reset_index(drop=True)

        self._write_cache(cache_key, df.to_dict(orient="list"))
        logger.info(
            "Fetched %d historical records for event '%s'.",
            len(df),
            event_id,
        )
        return df

    def list_events(self) -> list[dict[str, Any]]:
        """Return the catalog of ground truth landslide events.

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
        """Generate binary anomaly labels for a historical landslide event.

        Labeling strategy: a landslide event is labeled *anomalous* (``1``)
        if it resulted in fatalities (``fatality_count > 0``) or if the
        landslide size is ``large`` or ``very_large``.  All smaller,
        non-fatal events are labeled *normal* (``0``).

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

        # Fatal events are anomalous
        fatality_mask = df["fatality_count"].fillna(0).values > 0

        # Large or very_large events are anomalous
        size_values = df["landslide_size"].fillna("unknown").str.lower()
        size_mask = size_values.isin(["large", "very_large"]).values

        labels = (fatality_mask | size_mask).astype(np.int64)

        logger.info(
            "Ground truth for '%s': %d anomalies / %d total " "(fatal or large/very_large).",
            event_id,
            int(labels.sum()),
            len(labels),
        )
        return np.asarray(labels)

    # ------------------------------------------------------------------
    # Feature engineering (Mercury-native)
    # ------------------------------------------------------------------

    def engineer_features(self, raw_data: pd.DataFrame) -> np.ndarray:
        """Transform raw landslide catalog data into a feature matrix.

        Engineered features (per event row):

        1. **category_code** -- numeric encoding of landslide category
           (landslide, mudslide, rockfall, etc.).
        2. **fatality_count** -- number of reported fatalities.
        3. **injury_count** -- number of reported injuries.
        4. **trigger_code** -- numeric encoding of landslide trigger
           (rain, earthquake, construction, etc.).
        5. **latitude** -- event latitude.
        6. **longitude** -- event longitude.
        7. **size_code** -- landslide size encoding
           (small=1, medium=2, large=3, very_large=4).
        8. **country_code** -- numeric hash of country/admin division.
        9. **month** -- month of year (1--12).
        10. **day_of_year** -- day of year (1--366).

        All encoding is done with Mercury-native math (Mercury-native).

        Args:
            raw_data: DataFrame from :meth:`fetch_realtime` or
                :meth:`fetch_historical`.

        Returns:
            2-D numpy array of shape ``(n_samples, 10)``.
        """
        n_features = 10

        if raw_data.empty:
            return np.empty((0, n_features), dtype=np.float64)

        df = raw_data.copy()
        n_samples = len(df)

        features = np.zeros((n_samples, n_features), dtype=np.float64)

        # 1. Category encoding
        features[:, 0] = self._encode_column(df, "event_category", _CATEGORY_ENCODING)

        # 2. Fatality count
        features[:, 1] = (
            pd.to_numeric(df.get("fatality_count"), errors="coerce")
            .fillna(0)
            .values.astype(np.float64)
        )

        # 3. Injury count
        features[:, 2] = (
            pd.to_numeric(df.get("injury_count"), errors="coerce")
            .fillna(0)
            .values.astype(np.float64)
        )

        # 4. Trigger encoding
        features[:, 3] = self._encode_column(df, "landslide_trigger", _TRIGGER_ENCODING)

        # 5. Latitude
        features[:, 4] = (
            pd.to_numeric(df.get("latitude"), errors="coerce").fillna(0).values.astype(np.float64)
        )

        # 6. Longitude
        features[:, 5] = (
            pd.to_numeric(df.get("longitude"), errors="coerce").fillna(0).values.astype(np.float64)
        )

        # 7. Size encoding
        features[:, 6] = self._encode_column(df, "landslide_size", _SIZE_ENCODING)

        # 8. Country/admin division encoding (stable numeric hash)
        features[:, 7] = self._encode_country(df)

        # 9. Month (1-12)
        features[:, 8] = self._extract_month(df)

        # 10. Day of year (1-366)
        features[:, 9] = self._extract_day_of_year(df)

        # Clean up non-finite values (Mercury-native)
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
    def _arcgis_to_dataframe(
        response: dict[str, Any],
    ) -> pd.DataFrame:
        """Convert an ArcGIS REST API JSON response to a flat DataFrame.

        The ArcGIS endpoint returns a ``features`` array where each feature
        has an ``attributes`` dict containing the record fields.

        Args:
            response: Parsed JSON dict from the ArcGIS REST API.

        Returns:
            DataFrame with columns from the attribute fields.
        """
        features = response.get("features", [])
        if not features:
            return pd.DataFrame(
                columns=[
                    "event_id",
                    "event_date",
                    "event_category",
                    "landslide_trigger",
                    "landslide_size",
                    "fatality_count",
                    "injury_count",
                    "latitude",
                    "longitude",
                    "country_name",
                    "admin_division_name",
                ]
            )

        rows: list[dict[str, Any]] = []
        for feature in features:
            attrs = feature.get("attributes", {})
            rows.append(attrs)

        df = pd.DataFrame(rows)

        # Normalize column names to lowercase
        df.columns = [str(c).lower() for c in df.columns]

        # Coerce numeric columns where expected
        for col in ("fatality_count", "injury_count", "latitude", "longitude"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Ensure event_date is string for consistent handling
        if "event_date" in df.columns:
            df["event_date"] = df["event_date"].astype(str)

        return df

    @staticmethod
    def _build_where_clause(
        lat: float,
        lon: float,
        radius_km: float,
        start_date: str,
        end_date: str,
    ) -> str:
        """Build an ArcGIS-compatible SQL where clause for spatial/temporal filtering.

        Uses a bounding box approximation for spatial filtering since the
        ArcGIS REST API supports SQL where clauses on attribute fields.
        One degree of latitude is approximately 111 km.

        Args:
            lat: Center latitude in decimal degrees.
            lon: Center longitude in decimal degrees.
            radius_km: Search radius in kilometers.
            start_date: Start date (ISO format YYYY-MM-DD).
            end_date: End date (ISO format YYYY-MM-DD).

        Returns:
            SQL where clause string.
        """
        # Approximate degrees from km (1 degree lat ~ 111 km)
        lat_delta = radius_km / 111.0
        # Longitude degrees vary with latitude
        lon_delta = radius_km / (111.0 * max(np.cos(np.radians(lat)), 0.01))

        lat_min = lat - lat_delta
        lat_max = lat + lat_delta
        lon_min = lon - lon_delta
        lon_max = lon + lon_delta

        where = (
            f"latitude >= {lat_min:.4f} AND latitude <= {lat_max:.4f} "
            f"AND longitude >= {lon_min:.4f} AND longitude <= {lon_max:.4f} "
            f"AND event_date >= '{start_date}' "
            f"AND event_date <= '{end_date}'"
        )
        return where

    @staticmethod
    def _encode_column(
        df: pd.DataFrame,
        column: str,
        encoding_map: dict[str, int],
    ) -> np.ndarray:
        """Encode a categorical column to numeric values using a fixed map.

        Mercury-native encoding: direct dictionary lookup.

        Args:
            df: DataFrame containing the column.
            column: Column name to encode.
            encoding_map: Mapping from category string to integer code.

        Returns:
            1-D numpy array of integer codes.
        """
        if column not in df.columns:
            return np.zeros(len(df), dtype=np.float64)

        default_code = encoding_map.get("unknown", 0)
        values = df[column].fillna("unknown").astype(str).str.lower().str.strip()
        encoded = values.map(lambda v: encoding_map.get(v, default_code)).values.astype(np.float64)
        return np.asarray(encoded)

    @staticmethod
    def _encode_country(df: pd.DataFrame) -> np.ndarray:
        """Encode country/admin division to a stable numeric code.

        Uses a deterministic hash to produce consistent numeric IDs for
        each unique country name.  Mercury-native (Mercury-native).

        Args:
            df: DataFrame with optional ``country_name`` column.

        Returns:
            1-D numpy array of numeric country codes.
        """
        if "country_name" not in df.columns:
            return np.zeros(len(df), dtype=np.float64)

        values = df["country_name"].fillna("unknown").astype(str).str.lower()
        # Use a stable hash modulo a prime for numeric encoding
        codes = values.map(lambda v: sum(ord(c) for c in v) % 997).values.astype(np.float64)
        return np.asarray(codes)

    @staticmethod
    def _extract_month(df: pd.DataFrame) -> np.ndarray:
        """Extract month from event_date column.

        Handles both epoch-millisecond timestamps (as used by ArcGIS) and
        ISO date strings.

        Args:
            df: DataFrame with ``event_date`` column.

        Returns:
            1-D numpy array of month values (1--12).
        """
        if "event_date" not in df.columns:
            return np.ones(len(df), dtype=np.float64)

        months = np.ones(len(df), dtype=np.float64)
        for i, val in enumerate(df["event_date"]):
            parsed = _parse_date(val)
            if parsed is not None:
                months[i] = float(parsed.month)
        return months

    @staticmethod
    def _extract_day_of_year(df: pd.DataFrame) -> np.ndarray:
        """Extract day of year from event_date column.

        Handles both epoch-millisecond timestamps and ISO date strings.

        Args:
            df: DataFrame with ``event_date`` column.

        Returns:
            1-D numpy array of day-of-year values (1--366).
        """
        if "event_date" not in df.columns:
            return np.ones(len(df), dtype=np.float64)

        days = np.ones(len(df), dtype=np.float64)
        for i, val in enumerate(df["event_date"]):
            parsed = _parse_date(val)
            if parsed is not None:
                days[i] = float(parsed.timetuple().tm_yday)
        return days


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _parse_date(value: Any) -> datetime | None:
    """Parse a date value from ArcGIS into a datetime object.

    Handles:
    - Epoch millisecond timestamps (int or numeric string).
    - ISO format date strings (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS).

    Args:
        value: Raw date value from the ArcGIS attributes.

    Returns:
        Parsed datetime or None if parsing fails.
    """
    if value is None or (isinstance(value, str) and value.strip() in ("", "None", "nan")):
        return None

    # Try epoch milliseconds (ArcGIS often returns dates as ms since epoch)
    try:
        numeric = float(value)
        if numeric > 1e12:
            # Likely milliseconds since epoch
            return datetime.utcfromtimestamp(numeric / 1000.0)
        elif numeric > 1e9:
            # Likely seconds since epoch
            return datetime.utcfromtimestamp(numeric)
    except (ValueError, TypeError, OSError):
        pass

    # Try ISO format parsing
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d", "%m/%d/%Y"):
            try:
                return datetime.strptime(value.strip(), fmt)
            except ValueError:
                continue

    return None
