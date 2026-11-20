# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Domain loader for landslide data from NASA Global Landslide Catalog (COOLR).

Connects to the NASA Cooperative Open Online Landslide Repository (COOLR) ArcGIS REST endpoint to
provide global landslide event data for Mercury anomaly detection.  Ground truth events cover major
catastrophic landslides where fatal or large-scale events are labeled as anomalies against a
background of smaller, non-fatal slope failures.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from omni_mercury_engine.datasets.exceptions import (
    DataSourceUnavailableError,
    OfflineModeError,
)
from omni_mercury_engine.loaders.base import BaseDomainLoader, FetchHTTPError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Catalog sources, in the order they are asked
# ---------------------------------------------------------------------------
# NASA's own service comes first. Its host publishes an IPv6 address and no
# IPv4 one (no A record in public DNS, read 2026-11-17), so a network without
# IPv6 cannot reach it at all; such a network reads the public ArcGIS Online
# copy of the same catalog, the one the trained landslide model already reads
# (``ml/hazard_training/landslide_stability.py``). 14,750 rows, last edited
# 2025-08-10, when this was written.
_QUERY_URL = (
    "https://maps.nccs.nasa.gov/arcgis/rest/services/"
    "global_landslide_catalog/global_landslide_catalog_export/"
    "FeatureServer/0/query"
)
_MIRROR_QUERY_URL = (
    "https://services1.arcgis.com/yFGHRCyBneULM8ci/arcgis/rest/services/"
    "nasa_global_landslide_catalog_point/FeatureServer/0/query"
)

#: Values of the ``catalog_source`` column: which service a frame was read from.
SOURCE_NASA = "nasa_coolr"
SOURCE_MIRROR = "arcgis_online_mirror"

_SOURCES: tuple[tuple[str, str], ...] = (
    (SOURCE_NASA, _QUERY_URL),
    (SOURCE_MIRROR, _MIRROR_QUERY_URL),
)

#: Catalog field -> this loader's column. The catalog calls the kind of slide
#: ``landslide_category``; this loader has always called it ``event_category``,
#: and read a field of that name the catalog does not have, so the category
#: feature was zero on every real row. The ArcGIS Online copy was published
#: from a shapefile, which cuts every field name to ten characters.
_CATALOG_FIELDS: dict[str, str] = {
    "landslide_category": "event_category",
    "landslide_": "event_category",
    "landslide1": "landslide_trigger",
    "landslid_1": "landslide_size",
    "landslid_2": "landslide_setting",
    "fatality_c": "fatality_count",
    "injury_cou": "injury_count",
    "country_na": "country_name",
    "country_co": "country_code",
    "admin_divi": "admin_division_name",
    "source_nam": "source_name",
    "source_lin": "source_link",
    "event_titl": "event_title",
    "event_desc": "event_description",
    "location_d": "location_description",
    "location_a": "location_accuracy",
    "event_impo": "event_import_source",
    "event_im_1": "event_import_id",
    "gazetteer_": "gazeteer_closest_point",
    "gazetteer1": "gazeteer_distance",
    "submitted_": "submitted_date",
    "last_edite": "last_edited_date",
}

#: Sizes the labels call anomalous. ``catastrophic`` is the catalog's largest
#: class (26 of 14,750 rows); it was absent here, so the largest slides in the
#: catalog were encoded "unknown" and labelled normal unless someone died.
_ANOMALOUS_SIZES: frozenset[str] = frozenset({"large", "very_large", "catastrophic"})

# ---------------------------------------------------------------------------
# Category and trigger encoding maps (Mercury-native, no sklearn)
# ---------------------------------------------------------------------------
# Every value the catalog holds has a code here (the values and their counts,
# read 2026-11-17, are in ``tests/fixtures/landslide/glc_vocabulary.json``).
# The catalog spells rockfall ``rock_fall`` (942 rows) and avalanche
# ``snow_avalanche``; neither spelling was listed, nor were rotational and
# translational slides or topples, so 1,384 rows (9.4%) were encoded "unknown".
_CATEGORY_ENCODING: dict[str, int] = {
    "landslide": 1,
    "mudslide": 2,
    "rockfall": 3,
    "rock_fall": 3,
    "rock_slide": 4,
    "debris_flow": 5,
    "earth_flow": 6,
    "lahar": 7,
    "avalanche": 8,
    "snow_avalanche": 8,
    "riverbank_collapse": 9,
    "creep": 10,
    "complex": 11,
    "other": 12,
    "rotational_slide": 13,
    "translational_slide": 14,
    "topple": 15,
    "unknown": 0,
    "unkown": 0,  # the catalog's own misspelling, one row
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
    "no_apparent_trigger": 15,
    "leaking_pipe": 16,
    "other": 17,
    # Free-text spellings of rain in the catalog (8 rows). They say rain and
    # nothing about its intensity, so they take the plain ``rain`` code.
    "rainfall": 1,
    "heavy_rain": 1,
    "heavy_rainfall": 1,
    "unknown": 0,
}

_SIZE_ENCODING: dict[str, int] = {
    "small": 1,
    "medium": 2,
    "large": 3,
    "very_large": 4,
    "catastrophic": 5,
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
    * **Sources** -- NASA's service first, then the public ArcGIS Online
      copy of the same catalog when NASA's cannot be reached; every frame
      says which one it came from in ``catalog_source``.
    * **Historical query** -- Spatial and temporal filtering for events
      around known catastrophic landslides.

    Feature engineering produces landslide-specific observables suitable
    for anomaly detection: event category, trigger type, fatality/injury
    counts, landslide size, geographic location, and temporal features.

    All math is Mercury-native (numpy only, no sklearn).
    """

    DOMAIN: str = "landslide"
    SOURCE_URL: str = "https://maps.nccs.nasa.gov/arcgis/rest/services/" "global_landslide_catalog/"
    # Labels = (fatality_count > 0) OR (size_code >= large). Both
    # ``fatality_count`` and ``size_code`` are scored features. Direct
    # feature-equality / feature-threshold circularity.
    LABEL_SOURCE: str = "statistical"
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

    #: De-leak: labels are ``fatality_count > 0`` OR ``size_code >= large``.
    #: Still computed and returned by ``engineer_features``; excluded only
    #: from the scored matrix, so nothing is lost and the labels stop
    #: describing the thing being scored.
    LABEL_DERIVED_FEATURES: tuple[str, ...] = (
        "fatality_count",
        "size_code",
    )
    LABEL_FEATURE_DISJOINT: bool = True

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def fetch_realtime(self) -> pd.DataFrame:
        """Fetch the most recent landslide events from NASA COOLR.

        Queries the ArcGIS REST endpoint for the 2000 most recent landslide
        reports in the global catalog, newest first.

        Returns:
            DataFrame with columns derived from NASA COOLR attributes
            including event category, trigger, location, fatalities, etc.

        Raises:
            ConnectionError: If no catalog source can be read.
        """
        cache_key = "landslide_realtime"
        cached = self._read_cache(cache_key)

        if cached is not None:
            logger.debug("Returning cached real-time landslide data.")
            return pd.DataFrame(cached)

        # Newest first. Without an order the service returns its first 2000
        # rows by object id: read live, those ran from 1974 to 2025, not "the
        # most recent"; ordered, they run from 2017-03-29 to 2025-05-18.
        params: dict[str, str] = {
            "where": "1=1",
            "outFields": "*",
            "f": "json",
            "resultRecordCount": "2000",
            "orderByFields": "event_date DESC",
        }

        response, source = self._query(params)
        df = self._arcgis_to_dataframe(response, source)

        self._write_cache(cache_key, df.to_dict(orient="list"))
        logger.info("Fetched %d real-time landslide records from %s.", len(df), source)
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
            ConnectionError: If no catalog source can be read.
        """
        if event_id not in _EVENT_CATALOG:
            raise ValueError(
                f"Unknown event_id '{event_id}'. " f"Available: {list(_EVENT_CATALOG.keys())}"
            )

        # Cache schema v2: pre-fix entries could be a worldwide date-only
        # stand-in (unrelated events) or an empty result cached as data.
        # Versioning the key retires both so the corrected logic always runs.
        cache_key = f"landslide_historical_v2_{event_id}"
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

        features, source = self._query_all_pages(params)
        df = self._arcgis_to_dataframe({"features": features}, source)

        if df.empty:
            # The spatial/temporal query for THIS event returned nothing.
            # The previous behaviour fell back to a worldwide, date-only query
            # capped at 2,000 unordered rows, so unrelated landslides from
            # anywhere on Earth stood in for the target event -- a fabricated
            # benchmark. An empty area result is an honest "no catalogued
            # landslide for this event"; raise rather than substitute, and do
            # NOT cache the empty frame (a transient gap must not be frozen in
            # for the whole cache TTL).
            raise DataSourceUnavailableError(
                "LandslideLoader",
                source_url=dict(_SOURCES).get(source, _QUERY_URL),
                reason=(
                    f"NASA landslide catalog returned no event within "
                    f"{event['search_radius_km']} km of "
                    f"({event['latitude']}, {event['longitude']}) for "
                    f"{event['start']}..{event['end']} (event '{event_id}'). "
                    f"No worldwide stand-in is substituted."
                ),
            )

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

    def get_ground_truth(self, event_id: str) -> np.ndarray[Any, Any]:
        """Generate binary anomaly labels for a historical landslide event.

        Labeling strategy: a landslide event is labeled *anomalous* (``1``)
        if it resulted in fatalities (``fatality_count > 0``) or if the
        landslide size is ``large``, ``very_large`` or ``catastrophic``.
        All smaller, non-fatal events are labeled *normal* (``0``).

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
        size_mask = _terms(df["landslide_size"]).isin(_ANOMALOUS_SIZES).values

        labels = (fatality_mask | size_mask).astype(np.int64)

        logger.info(
            "Ground truth for '%s': %d anomalies / %d total (fatal, or large and above).",
            event_id,
            int(labels.sum()),
            len(labels),
        )
        return np.asarray(labels)

    # ------------------------------------------------------------------
    # Feature engineering (Mercury-native, no sklearn)
    # ------------------------------------------------------------------

    def engineer_features(self, raw_data: pd.DataFrame) -> np.ndarray[Any, Any]:
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
           (small=1, medium=2, large=3, very_large=4, catastrophic=5).
        8. **country_code** -- numeric hash of country/admin division.
        9. **month** -- month of year (1--12).
        10. **day_of_year** -- day of year (1--366).

        All encoding is done with Mercury-native math (no sklearn).

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

        # Clean up non-finite values (Mercury-native, no sklearn)
        features = np.asarray(np.where(np.isinf(features), np.nan, features))
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
        source: str,
    ) -> pd.DataFrame:
        """Convert an ArcGIS REST API JSON response to a flat DataFrame.

        The ArcGIS endpoint returns a ``features`` array where each feature
        has an ``attributes`` dict containing the record fields. Fields are
        renamed to this loader's columns (:data:`_CATALOG_FIELDS`), so a
        frame has the same columns whichever source it came from.

        Args:
            response: Parsed JSON dict from the ArcGIS REST API.
            source: Which catalog service answered; written to the
                ``catalog_source`` column.

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
                    "catalog_source",
                ]
            )

        rows: list[dict[str, Any]] = []
        for feature in features:
            attrs = feature.get("attributes", {})
            rows.append(attrs)

        df = pd.DataFrame(rows)

        # Lowercase names, then the catalog's names as this loader's columns.
        # A name is only taken when the frame does not already have it, so a
        # service that publishes both spellings keeps its own full one.
        df.columns = [str(c).lower() for c in df.columns]
        taken = set(df.columns)
        renames: dict[str, str] = {}
        for name, column in _CATALOG_FIELDS.items():
            if name in df.columns and column not in taken:
                renames[name] = column
                taken.add(column)
        df = df.rename(columns=renames)

        # Coerce numeric columns where expected
        for col in ("fatality_count", "injury_count", "latitude", "longitude"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # ArcGIS sends a date as milliseconds since 1970. As text, in ISO
        # form: digits of differing length do not sort by time, and a row with
        # no date used to become the string "None".
        if "event_date" in df.columns:
            df["event_date"] = [_iso_date(value) for value in df["event_date"]]

        df["catalog_source"] = source
        return df

    def _query(self, params: dict[str, str]) -> tuple[dict[str, Any], str]:
        """Ask each catalog source in turn; the first answer and which source gave it.

        A source is passed over when it cannot be reached, when its host does
        not resolve, and when it answers HTTP 200 with an ArcGIS ``error``
        body -- which has no ``features`` and used to be read as an empty
        catalog.

        Raises:
            OfflineModeError: Untouched, when ``MERCURY_OFFLINE`` is set.
            FetchHTTPError: When no source answered (a ``ConnectionError``).
        """
        failures: list[str] = []
        for source, url in _SOURCES:
            try:
                response = self._fetch_json(url, params=params)
            except OfflineModeError:
                raise
            except (ConnectionError, OSError, ValueError) as exc:
                failures.append(f"{source}: {type(exc).__name__}")
                logger.warning("landslide: %s could not be read (%s)", source, type(exc).__name__)
                continue
            if not isinstance(response, dict) or "error" in response:
                failures.append(f"{source}: the service answered with an error")
                logger.warning("landslide: %s answered with an error body", source)
                continue
            return response, source
        raise FetchHTTPError(
            "landslide: no catalog source could be read (" + "; ".join(failures) + ")"
        )

    def _query_all_pages(self, params: dict[str, str]) -> tuple[list[dict[str, Any]], str]:
        """Query a catalog source and follow ``exceededTransferLimit`` paging.

        ArcGIS caps a response at ``resultRecordCount`` rows and sets
        ``exceededTransferLimit: true`` when more match. The previous code
        read only the first page and silently dropped the rest, so a dense
        event was scored on an arbitrary 2,000-row slice. This pages the SAME
        source with ``resultOffset`` until the service stops flagging more
        rows, concatenating the features.

        Args:
            params: Base ArcGIS query parameters (``resultOffset`` is added).

        Returns:
            Tuple of (all feature dicts across pages, the source that answered).

        Raises:
            OfflineModeError: Propagated unchanged from the first query.
            FetchHTTPError: If no source could answer the first query.
        """
        first, source = self._query(params)
        features: list[dict[str, Any]] = list(first.get("features", []))

        if not first.get("exceededTransferLimit"):
            return features, source

        seen = {key for key in map(self._feature_key, features) if key is not None}

        # Stay on the source that answered page one.
        url = dict(_SOURCES).get(source)
        if url is None:
            return features, source
        try:
            page_size = int(params.get("resultRecordCount", "2000"))
        except (TypeError, ValueError):
            page_size = 2000

        offset = len(features)
        while True:
            page_params = dict(params)
            page_params["resultOffset"] = str(offset)
            try:
                resp = self._fetch_json(url, params=page_params)
            except OfflineModeError:
                raise
            except (ConnectionError, OSError, ValueError) as exc:
                logger.warning(
                    "landslide: pagination stopped at offset %d (%s); "
                    "returning %d rows gathered so far.",
                    offset,
                    type(exc).__name__,
                    len(features),
                )
                break
            if not isinstance(resp, dict) or "error" in resp:
                logger.warning(
                    "landslide: pagination error body at offset %d; "
                    "returning %d rows gathered so far.",
                    offset,
                    len(features),
                )
                break
            page_features = resp.get("features", [])
            if not page_features:
                break
            # A service that ignores ``resultOffset`` answers every page with
            # the same rows. Only rows not seen before are kept, and a page
            # that adds none ends the paging -- so such a service can neither
            # duplicate the catalog nor loop forever.
            fresh = []
            for feature in page_features:
                key = self._feature_key(feature)
                if key is not None and key in seen:
                    continue
                if key is not None:
                    seen.add(key)
                fresh.append(feature)
            if not fresh:
                break
            features.extend(fresh)
            offset += len(page_features)
            if not resp.get("exceededTransferLimit"):
                break
            if len(page_features) < page_size:
                break

        return features, source

    @staticmethod
    def _feature_key(feature: dict[str, Any]) -> Any | None:
        """Stable identity of a catalog row: the first id field it carries, else ``None``."""
        attributes = feature.get("attributes") if isinstance(feature, dict) else None
        if not isinstance(attributes, dict):
            return None
        for name in ("FID", "OBJECTID", "objectid", "event_id"):
            value = attributes.get(name)
            if value is not None:
                return (name, value)
        return None

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
    ) -> np.ndarray[Any, Any]:
        """Encode a categorical column to numeric values using a fixed map.

        Mercury-native encoding: direct dictionary lookup, no sklearn.

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
        encoded = (
            _terms(df[column]).map(lambda v: encoding_map.get(v, default_code)).values
        ).astype(np.float64)
        return np.asarray(encoded)

    @staticmethod
    def _encode_country(df: pd.DataFrame) -> np.ndarray[Any, Any]:
        """Encode country/admin division to a stable numeric code.

        Uses a deterministic hash to produce consistent numeric IDs for
        each unique country name.  Mercury-native (no sklearn).

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
    def _extract_month(df: pd.DataFrame) -> np.ndarray[Any, Any]:
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
    def _extract_day_of_year(df: pd.DataFrame) -> np.ndarray[Any, Any]:
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


_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

#: In :func:`_parse_date`, a bare number at least this large is milliseconds:
#: as seconds it would be the year 5138. The line used to be 1e12, which is
#: September 2001 in milliseconds, so every earlier date in the catalog (it
#: goes back to 1974) was read as seconds and landed thousands of years out,
#: in the wrong month.
_MILLISECONDS_FROM = 1e11
#: ...and one at least this large is seconds (September 2001 onward), as before.
_SECONDS_FROM = 1e9


def _terms(values: pd.Series) -> pd.Series:
    """Catalog terms in one spelling: lower case, trimmed, words joined by ``_``.

    The catalog mixes ``Riverbank collapse`` with ``riverbank_collapse`` and
    pads blanks with a space; a blank is ``unknown``.
    """
    terms = values.fillna("unknown").astype(str).str.strip().str.lower()
    terms = terms.str.replace(r"\s+", "_", regex=True)
    return terms.where(terms != "", "unknown")


def _from_milliseconds(milliseconds: float) -> datetime | None:
    """The UTC time a number of milliseconds since 1970 stands for (naive)."""
    try:
        return (_EPOCH + timedelta(milliseconds=milliseconds)).replace(tzinfo=None)
    except OverflowError:
        return None


def _iso_date(value: Any) -> str | None:
    """An ArcGIS date as ISO text; a missing one stays missing.

    ArcGIS sends every date field as milliseconds since 1970, so a number
    here is milliseconds whatever its size. Anything else is kept as text.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        return str(value)
    moment = _from_milliseconds(float(value))
    return moment.strftime("%Y-%m-%dT%H:%M:%S") if moment is not None else None


def _parse_date(value: Any) -> datetime | None:
    """Parse a date value from ArcGIS into a datetime object.

    Handles:
    - Epoch millisecond timestamps (int or numeric string), and epoch
      seconds from September 2001 onward.
    - ISO format date strings (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS).

    Args:
        value: Raw date value from the ArcGIS attributes.

    Returns:
        Parsed datetime or None if parsing fails.
    """
    if value is None or (isinstance(value, str) and value.strip() in ("", "None", "nan")):
        return None

    numeric: float | None
    try:
        numeric = float(value)
    except (ValueError, TypeError, OSError):
        numeric = None

    if numeric is not None and np.isfinite(numeric):
        if abs(numeric) >= _MILLISECONDS_FROM:
            return _from_milliseconds(numeric)
        if numeric >= _SECONDS_FROM:
            return _from_milliseconds(numeric * 1000.0)

    # Try ISO format parsing
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d", "%m/%d/%Y"):
            try:
                return datetime.strptime(value.strip(), fmt)
            except ValueError:
                continue

    return None
