# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Domain loader for wildfire data from NASA FIRMS (Fire Information for Resource Management System).

Connects to the NASA FIRMS active-fire API to provide thermal hotspot data for Mercury anomaly
detection.  Ground truth events cover major wildfires where high fire-radiative-power detections
during the event window are labeled as anomalies against a background of normal thermal activity.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
import pandas as pd

from omni_mercury_engine.loaders.base import BaseDomainLoader, FetchHTTPError
from omni_mercury_engine.utils.geo import (
    haversine_km,
    haversine_km_to_point,
    neighbor_counts_within_km,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NASA FIRMS API endpoints
# ---------------------------------------------------------------------------
_AREA_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
_COUNTRY_URL = "https://firms.modaps.eosdis.nasa.gov/api/country/csv"

_SENSOR = "VIIRS_NOAA20_NRT"

# Maximum number of days allowed for NRT data requests
_MAX_NRT_DAYS = 10

# ---------------------------------------------------------------------------
# Ground truth event catalog
# ---------------------------------------------------------------------------
_EVENT_CATALOG: dict[str, dict[str, Any]] = {
    "la_2025": {
        "name": "2025 Los Angeles Wildfires",
        "date": "2025-01-07",
        "description": (
            "Palisades and Eaton fires in the Los Angeles metropolitan area, " "January 2025."
        ),
        "area": "-119.0,33.5,-117.5,34.5",
        "country_code": None,
        "start": "2025-01-07",
        "end": "2025-01-14",
        "days": 8,
    },
    "maui_2023": {
        "name": "2023 Maui Wildfire",
        "date": "2023-08-08",
        "description": ("Devastating wildfire in Lahaina, Maui, Hawaii, " "August 2023."),
        "area": "-156.7,20.8,-156.4,21.0",
        "country_code": None,
        "start": "2023-08-08",
        "end": "2023-08-12",
        "days": 5,
    },
    "australia_2020": {
        "name": "2020 Australian Bushfires",
        "date": "2019-11-01",
        "description": (
            "Catastrophic bushfire season across southeastern Australia, "
            "November 2019 through February 2020."
        ),
        "area": None,
        "country_code": "AUS",
        "start": "2019-11-01",
        "end": "2020-02-29",
        "days": 10,
    },
    "west_coast_2020": {
        "name": "2020 US West Coast Fires",
        "date": "2020-09-01",
        "description": (
            "Record-breaking wildfire season across California, Oregon, "
            "and Washington, September 2020."
        ),
        "area": "-125,32,-116,49",
        "country_code": None,
        "start": "2020-09-01",
        "end": "2020-09-30",
        "days": 10,
    },
}


class WildfireLoader(BaseDomainLoader):
    """Loader for wildfire / active-fire data from NASA FIRMS.

    Uses the NASA FIRMS VIIRS NOAA-20 Near Real-Time (NRT) active-fire
    product via two endpoints:

    * **Area endpoint** -- CSV of fire detections within a geographic
      bounding box over a specified number of days.
    * **Country endpoint** -- CSV of fire detections within a country
      boundary over a specified number of days.

    Feature engineering produces fire-behaviour observables suitable for
    anomaly detection: brightness temperature, fire radiative power,
    confidence, pixel geometry, spatial clustering density, and rate of
    spread.
    """

    DOMAIN: str = "wildfire"
    SOURCE_URL: str = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
    # Labels = (FRP >= 90th percentile) and ``frp`` is a scored feature.
    # Per-dataset percentile threshold on a scored feature — circular.
    LABEL_SOURCE: str = "statistical"
    REQUIRES_API_KEY: bool = True
    API_KEY_ENV_VAR: str = "NASA_FIRMS_MAP_KEY"
    # The canonical name is ``NASA_FIRMS_MAP_KEY`` (see ``.env.example``); the
    # repository Actions secret is named ``FIRMS_MAP_KEY``. Accept both so the
    # loader stays wired regardless of which one the environment provides.
    API_KEY_ENV_FALLBACKS: tuple[str, ...] = ("FIRMS_MAP_KEY",)
    FEATURE_COLUMNS: list[str] = [
        "brightness",
        "frp",
        "confidence",
        "scan",
        "track",
        "cluster_count",
        "spread_rate",
    ]

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def fetch_realtime(self) -> pd.DataFrame:
        """Fetch the most recent day of global active-fire detections.

        Queries the NASA FIRMS area endpoint for a global bounding box
        over the last 1 day of NRT data.

        Returns:
            DataFrame with FIRMS CSV columns including latitude,
            longitude, brightness, frp, confidence, acq_date, acq_time,
            scan, track, satellite, instrument, and version.

        Raises:
            ConnectionError: If the FIRMS API is unreachable after retries.
            EnvironmentError: If no API key is configured.
        """
        self._require_api_key()

        cache_key = "wildfire_realtime"
        cached = self._read_cache(cache_key)

        if cached is not None:
            logger.debug("Returning cached real-time wildfire data.")
            return pd.DataFrame(cached)

        # Global bounding box, 1 day of NRT data
        url = self._build_area_url("-180,-90,180,90", days=1)
        df = self._fetch_firms_csv(url)

        if not df.empty:
            self._write_cache(cache_key, df.to_dict(orient="list"))
        logger.info("Fetched %d real-time fire detections from FIRMS.", len(df))
        return df

    def fetch_historical(self, event_id: str) -> pd.DataFrame:
        """Fetch fire-detection data for a known historical wildfire event.

        Args:
            event_id: Key into the ground truth catalog (e.g.
                ``"la_2025"``).

        Returns:
            DataFrame with FIRMS CSV columns sorted chronologically.

        Raises:
            ValueError: If *event_id* is not in the catalog.
            ConnectionError: If the FIRMS API is unreachable.
            EnvironmentError: If no API key is configured.
        """
        if event_id not in _EVENT_CATALOG:
            raise ValueError(
                f"Unknown event_id '{event_id}'. " f"Available: {list(_EVENT_CATALOG.keys())}"
            )

        self._require_api_key()

        cache_key = f"wildfire_historical_{event_id}"
        cached = self._read_cache(cache_key)

        if cached is not None:
            logger.debug("Returning cached historical data for '%s'.", event_id)
            return pd.DataFrame(cached)

        event = _EVENT_CATALOG[event_id]
        days = min(event["days"], _MAX_NRT_DAYS)

        if event["country_code"] is not None:
            url = self._build_country_url(event["country_code"], days=days)
        else:
            url = self._build_area_url(event["area"], days=days)

        df = self._fetch_firms_csv(url)

        if df.empty:
            logger.warning("FIRMS returned no detections for event '%s'.", event_id)
            return df

        # Sort chronologically by acquisition date and time
        df = self._sort_chronologically(df)

        self._write_cache(cache_key, df.to_dict(orient="list"))
        logger.info("Fetched %d fire detections for event '%s'.", len(df), event_id)
        return df

    def list_events(self) -> list[dict[str, Any]]:
        """Return the catalog of ground truth wildfire events.

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
        """Generate binary anomaly labels for a historical wildfire event.

        Labeling strategy: a fire detection is labeled *anomalous* (``1``)
        if its fire radiative power (FRP) is at or above the 90th
        percentile of the dataset for that event.  All lower-FRP
        detections (background thermal hotspots) are labeled *normal*
        (``0``).  This captures the most intense fire activity that
        deviates significantly from routine thermal signatures.

        Args:
            event_id: Key into the ground truth catalog.

        Returns:
            1-D binary numpy array of shape ``(n_detections,)``.

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

        frp = pd.to_numeric(df["frp"], errors="coerce").values.astype(np.float64)

        # Replace NaN with 0 for threshold computation
        frp_clean = np.where(np.isnan(frp), 0.0, frp)
        threshold = np.percentile(frp_clean, 90)

        labels = (frp_clean >= threshold).astype(np.int64)
        logger.info(
            "Ground truth for '%s': %d anomalies / %d total " "(FRP >= %.1f, 90th percentile).",
            event_id,
            int(labels.sum()),
            len(labels),
            threshold,
        )
        return np.asarray(labels)

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def engineer_features(self, raw_data: pd.DataFrame) -> np.ndarray[Any, Any]:
        """Transform raw FIRMS data into a feature matrix.

        Engineered features (per detection row):

        1. **brightness** -- brightness temperature (Kelvin).
        2. **frp** -- fire radiative power (MW).
        3. **confidence** -- detection confidence score (mapped to
           numeric: low=0, nominal=50, high=100 for categorical;
           kept as-is for numeric).
        4. **scan** -- along-scan pixel size (km).
        5. **track** -- along-track pixel size (km).
        6. **cluster_count** -- number of other fire detections within
           a 10 km radius (spatial clustering density).
        7. **spread_rate** -- estimated rate of spread computed as the
           distance to the nearest subsequent detection divided by the
           elapsed time (km/h).  Zero for the last detection.

        Args:
            raw_data: DataFrame from :meth:`fetch_realtime` or
                :meth:`fetch_historical`.

        Returns:
            2-D numpy array of shape ``(n_samples, 7)``.
        """
        if raw_data.empty:
            return np.empty((0, 7), dtype=np.float64)

        df = raw_data.copy()
        df = self._sort_chronologically(df)

        # ---- base observables ----
        brightness = (
            pd.to_numeric(
                df["bright_ti4"] if "bright_ti4" in df.columns else df.get("brightness", 0),
                errors="coerce",
            )
            .fillna(0.0)
            .values.astype(np.float64)
        )

        frp = pd.to_numeric(df["frp"], errors="coerce").fillna(0.0).values.astype(np.float64)

        confidence = self._parse_confidence(df)

        scan = pd.to_numeric(df["scan"], errors="coerce").fillna(0.0).values.astype(np.float64)

        track = pd.to_numeric(df["track"], errors="coerce").fillna(0.0).values.astype(np.float64)

        # ---- spatial coordinates ----
        lat = pd.to_numeric(df["latitude"], errors="coerce").fillna(0.0).values.astype(np.float64)

        lon = pd.to_numeric(df["longitude"], errors="coerce").fillna(0.0).values.astype(np.float64)

        # ---- timestamps in fractional hours since first detection ----
        hours = self._compute_hours_since_start(df)

        # ---- spatial clustering: count of hotspots within 10 km ----
        cluster_count = self._compute_spatial_clustering(lat, lon, radius_km=10.0)

        # ---- rate of spread (km/h) ----
        spread_rate = self._compute_spread_rate(lat, lon, hours)

        # Stack into feature matrix
        features = np.column_stack(
            [
                brightness,
                frp,
                confidence,
                scan,
                track,
                cluster_count,
                spread_rate,
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
    # Private helpers -- URL construction
    # ------------------------------------------------------------------

    def _build_area_url(self, area: str, days: int) -> str:
        """Build a FIRMS area-endpoint URL.

        Args:
            area: Bounding box as ``"west,south,east,north"``.
            days: Number of days of NRT data to request (max 10).

        Returns:
            Fully-qualified URL string including the API key.
        """
        days = min(days, _MAX_NRT_DAYS)
        return f"{_AREA_URL}/{self._api_key}/{_SENSOR}/{area}/{days}"

    def _build_country_url(self, country_code: str, days: int) -> str:
        """Build a FIRMS country-endpoint URL.

        Args:
            country_code: ISO 3166-1 alpha-3 country code (e.g. ``"AUS"``).
            days: Number of days of NRT data to request (max 10).

        Returns:
            Fully-qualified URL string including the API key.
        """
        days = min(days, _MAX_NRT_DAYS)
        return f"{_COUNTRY_URL}/{self._api_key}/{_SENSOR}/{country_code}/{days}"

    # ------------------------------------------------------------------
    # Private helpers -- data parsing
    # ------------------------------------------------------------------

    def _require_api_key(self) -> None:
        """Raise if no API key is configured.

        Raises:
            EnvironmentError: When the API key is empty.
        """
        if not self._api_key:
            raise OSError(
                "NASA FIRMS MAP key not set. The wildfire domain loader requires a "
                "free NASA FIRMS MAP key. Register at "
                "https://firms.modaps.eosdis.nasa.gov/api/map_key/ "
                "and set the NASA_FIRMS_MAP_KEY environment variable "
                "(the FIRMS_MAP_KEY name is also accepted)."
            )

    def _fetch_firms_csv(self, url: str) -> pd.DataFrame:
        """Fetch CSV data from a FIRMS endpoint and return a DataFrame.

        NASA FIRMS returns CSV directly as the response body.

        Args:
            url: Fully-qualified FIRMS API URL.

        Returns:
            DataFrame parsed from the CSV response.

        Raises:
            ConnectionError: FIRMS rate limit hit (HTTP 429), rewritten as a
                purpose-built quota message. Key redaction for EVERY status
                is anchored one layer down: ``_fetch_url`` raises
                ``FetchHTTPError`` with ``from None`` and a URL-free message,
                because the underlying ``requests.HTTPError`` message embeds
                the full URL, whose path segment IS the MAP key
                (``_build_area_url``). Both this 429 rewrite and the non-429
                re-raise therefore propagate exceptions that carry no URL,
                no cause, and no suppressed-context leak;
                ``scripts/live_data_smoke.py``'s redaction remains as
                defence in depth.
            ValueError: FIRMS returned a non-CSV body. FIRMS signals key and
                quota problems as HTTP-200 text bodies ("Invalid MAP_KEY.",
                transaction-limit messages) that would otherwise parse into a
                nonsense one-column frame and flow downstream.
        """
        try:
            raw_bytes = self._fetch_url(url)
        except FetchHTTPError as exc:
            if exc.status_code == 429:
                raise ConnectionError(
                    "wildfire: NASA FIRMS rate limit hit (HTTP 429). The MAP "
                    "key's transaction quota is exhausted; wait for the "
                    "10-minute window to reset (status at "
                    "https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey/) "
                    "instead of retrying."
                ) from None
            # Non-429: propagate unchanged. Key-safe because _fetch_url
            # severs the exception chain and its message never contains
            # the URL — pinned end-to-end (real _fetch_url, key in path)
            # by test_wildfire_firms_guards.py and at the base layer by
            # TestFetchCredentialRedaction.
            raise
        text = raw_bytes.decode("utf-8", errors="replace")

        if not text.strip():
            logger.warning("FIRMS returned empty response")
            return pd.DataFrame()

        # Every FIRMS CSV product's header starts with ``latitude,longitude``;
        # anything else is an error body served with HTTP 200. Fail closed
        # rather than parse it into a nonsense frame. The slice keeps the
        # diagnostic short. Observed FIRMS error bodies do not contain the
        # MAP key, but that is upstream behaviour, not a contract — the
        # value scrub converts the assumption into a guarantee.
        from omni_mercury_engine.security.redaction import redact_secrets

        first_line = text.lstrip().splitlines()[0]
        if "latitude" not in first_line:
            safe_snippet = redact_secrets(first_line, ("NASA_FIRMS_MAP_KEY",), (self._api_key,))[
                :80
            ]
            raise ValueError(
                f"wildfire: FIRMS returned a non-CSV body ({safe_snippet!r}); "
                "this usually means an invalid MAP key or an exhausted "
                "transaction quota."
            )

        df = pd.read_csv(io.StringIO(text))
        return df

    @staticmethod
    def _sort_chronologically(df: pd.DataFrame) -> pd.DataFrame:
        """Sort a FIRMS DataFrame by acquisition date and time.

        Args:
            df: DataFrame with ``acq_date`` and ``acq_time`` columns.

        Returns:
            Sorted DataFrame with reset index.
        """
        if "acq_date" in df.columns and "acq_time" in df.columns:
            df = df.sort_values(["acq_date", "acq_time"]).reset_index(drop=True)
        return df

    @staticmethod
    def _parse_confidence(df: pd.DataFrame) -> np.ndarray[Any, Any]:
        """Parse the confidence column into numeric values.

        VIIRS confidence may be categorical (``"low"``, ``"nominal"``,
        ``"high"``) or numeric.  This method maps categorical values to
        numeric scores: low=0, nominal=50, high=100.

        Args:
            df: DataFrame with a ``confidence`` column.

        Returns:
            1-D float64 array of confidence scores.
        """
        if "confidence" not in df.columns:
            return np.zeros(len(df), dtype=np.float64)

        conf = df["confidence"].copy()

        # Map categorical confidence labels to numeric
        label_map = {"low": 0.0, "nominal": 50.0, "high": 100.0, "l": 0.0, "n": 50.0, "h": 100.0}

        def _map_value(val: Any) -> float:
            """Map a single confidence value to float."""
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                lower = val.strip().lower()
                if lower in label_map:
                    return label_map[lower]
                try:
                    return float(lower)
                except ValueError:
                    return 0.0
            return 0.0

        return np.array([_map_value(v) for v in conf], dtype=np.float64)

    @staticmethod
    def _compute_hours_since_start(df: pd.DataFrame) -> np.ndarray[Any, Any]:
        """Compute hours since the first detection in the DataFrame.

        Uses ``acq_date`` (YYYY-MM-DD) and ``acq_time`` (HHMM integer)
        to construct datetime values.

        Args:
            df: DataFrame with ``acq_date`` and ``acq_time`` columns.

        Returns:
            1-D float64 array of elapsed hours from the earliest detection.
        """
        n = len(df)
        if n == 0:
            return np.array([], dtype=np.float64)

        if "acq_date" not in df.columns or "acq_time" not in df.columns:
            return np.zeros(n, dtype=np.float64)

        # Build datetime from date + HHMM integer time
        acq_time = pd.to_numeric(df["acq_time"], errors="coerce").fillna(0).astype(int)
        hours_col = acq_time // 100
        minutes_col = acq_time % 100

        timestamps = pd.to_datetime(df["acq_date"], errors="coerce")
        timestamps = (
            timestamps
            + pd.to_timedelta(hours_col, unit="h")
            + pd.to_timedelta(minutes_col, unit="m")
        )

        # Convert to fractional hours since the first detection
        t0 = timestamps.min()
        elapsed = (timestamps - t0).dt.total_seconds().fillna(0.0).values / 3600.0
        return np.asarray(elapsed.astype(np.float64))

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Compute the Haversine distance between two points in kilometres.

        Delegates to the canonical kernel in
        :mod:`omni_mercury_engine.utils.geo`; kept as a class-level helper
        for API stability.

        Args:
            lat1: Latitude of point 1 (degrees).
            lon1: Longitude of point 1 (degrees).
            lat2: Latitude of point 2 (degrees).
            lon2: Longitude of point 2 (degrees).

        Returns:
            Distance in kilometres.
        """
        return haversine_km(lat1, lon1, lat2, lon2)

    @staticmethod
    def _compute_spatial_clustering(
        lat: np.ndarray[Any, Any],
        lon: np.ndarray[Any, Any],
        radius_km: float = 10.0,
    ) -> np.ndarray[Any, Any]:
        """Count the number of other fire detections within a given radius.

        Delegates to :func:`~omni_mercury_engine.utils.geo.neighbor_counts_within_km`,
        which prunes candidates with an *exact* latitude band (sort +
        ``searchsorted``; on the sphere the central angle is at least |Δlat|,
        so the band can never drop a true neighbour) and measures exact
        Haversine distance only within each point's band — never materialising
        the full n² matrix.  The previous implementation pre-filtered
        candidates with a flat ``radius_km / 111 * 1.5`` degree box, which
        silently dropped true neighbours above ~48 deg latitude — at 62 deg N
        (boreal fire country) it missed ~20% of genuine within-radius pairs.

        Args:
            lat: 1-D array of latitudes (degrees).
            lon: 1-D array of longitudes (degrees).
            radius_km: Search radius in kilometres.

        Returns:
            1-D float64 array of neighbour counts.
        """
        return neighbor_counts_within_km(lat, lon, radius_km)

    @staticmethod
    def _compute_spread_rate(
        lat: np.ndarray[Any, Any],
        lon: np.ndarray[Any, Any],
        hours: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Estimate fire spread rate from sequential detections.

        For each detection, computes the distance to the nearest of the
        next 49 detections divided by the elapsed time between them.  Ties
        keep the earliest detection, matching the original scalar loop.

        Args:
            lat: 1-D array of latitudes (degrees), chronologically sorted.
            lon: 1-D array of longitudes (degrees), chronologically sorted.
            hours: 1-D array of hours since the first detection.

        Returns:
            1-D float64 array of spread rates in km/h.  Zero for the
            last detection or when the time delta is zero.
        """
        n = len(lat)
        rate = np.zeros(n, dtype=np.float64)

        if n <= 1:
            return rate

        for i in range(n - 1):
            # Vectorized distances to the lookahead window of detections
            window_end = min(i + 50, n)
            dists = haversine_km_to_point(
                lat[i + 1 : window_end], lon[i + 1 : window_end], lat[i], lon[i]
            )
            j = i + 1 + int(np.argmin(dists))
            min_dist = float(dists[j - i - 1])
            min_dt = hours[j] - hours[i]

            if min_dt > 0.0 and np.isfinite(min_dist):
                rate[i] = min_dist / min_dt
            else:
                rate[i] = 0.0

        return rate
