"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.

Domain loader for marine biodiversity data from OBIS (Ocean Biodiversity
Information System).

Connects to the OBIS API v3 to retrieve species occurrence records for
marine biodiversity monitoring and anomaly detection.  Ground truth events
cover documented coral bleaching episodes and marine heatwave impacts
where significant biodiversity loss was observed.

Features engineered from raw occurrence data include spatial species
counts, biodiversity indices, temporal change rates, depth profiles,
and proxies for environmental stress derived from shifts in species
occurrence patterns.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from omni_mercury_engine.loaders.base import BaseDomainLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OBIS API configuration
# ---------------------------------------------------------------------------
_OBIS_BASE_URL = "https://api.obis.org/v3/occurrence"

#: Maximum records per page returned by OBIS.
_OBIS_PAGE_SIZE = 1000

#: Maximum pages to fetch per query (safety limit).
_OBIS_MAX_PAGES = 10

# ---------------------------------------------------------------------------
# Spatial grid parameters
# ---------------------------------------------------------------------------

#: Grid cell resolution in degrees for spatial binning.
_GRID_RESOLUTION: float = 1.0

# ---------------------------------------------------------------------------
# Indicator species commonly used in coral reef monitoring
# ---------------------------------------------------------------------------
_REEF_INDICATOR_SPECIES: list[str] = [
    "Acropora",
    "Pocillopora",
    "Porites",
    "Montipora",
    "Stylophora",
    "Seriatopora",
    "Fungia",
    "Goniastrea",
    "Platygyra",
    "Turbinaria",
]

# ---------------------------------------------------------------------------
# Ground-truth event catalog
# ---------------------------------------------------------------------------
_EVENT_CATALOG: dict[str, dict[str, Any]] = {
    "gbr_bleaching_2016": {
        "name": "Great Barrier Reef Coral Bleaching 2016",
        "date": "2016-03-01",
        "description": (
            "Mass coral bleaching event across the northern Great Barrier Reef "
            "driven by record-breaking sea surface temperatures during the "
            "2015-2016 El Nino.  Approximately 29% of shallow-water corals "
            "died on the reef system."
        ),
        "region": {
            "lat_min": -24.0,
            "lat_max": -10.0,
            "lon_min": 142.0,
            "lon_max": 154.0,
        },
        "start_date": "2016-01-01",
        "end_date": "2016-12-31",
        "baseline_start": "2011-01-01",
        "baseline_end": "2015-12-31",
        "species": _REEF_INDICATOR_SPECIES,
    },
    "gbr_bleaching_2020": {
        "name": "Great Barrier Reef Coral Bleaching 2020",
        "date": "2020-03-01",
        "description": (
            "Third mass bleaching event in five years across the Great Barrier "
            "Reef, extending further south than previous events.  Marine "
            "heatwave conditions persisted from January through March 2020."
        ),
        "region": {
            "lat_min": -24.0,
            "lat_max": -10.0,
            "lon_min": 142.0,
            "lon_max": 154.0,
        },
        "start_date": "2020-01-01",
        "end_date": "2020-12-31",
        "baseline_start": "2015-01-01",
        "baseline_end": "2019-12-31",
        "species": _REEF_INDICATOR_SPECIES,
    },
    "marine_heatwave_2023": {
        "name": "Marine Heatwave Events 2023",
        "date": "2023-06-01",
        "description": (
            "Widespread marine heatwave conditions observed globally in 2023, "
            "with record sea surface temperatures across multiple ocean basins.  "
            "Significant impacts on marine biodiversity documented in tropical "
            "and subtropical regions."
        ),
        "region": {
            "lat_min": -30.0,
            "lat_max": 30.0,
            "lon_min": -180.0,
            "lon_max": 180.0,
        },
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "baseline_start": "2018-01-01",
        "baseline_end": "2022-12-31",
        "species": _REEF_INDICATOR_SPECIES,
    },
}


class MarineLoader(BaseDomainLoader):
    """Domain loader for marine biodiversity data from OBIS.

    Connects to the OBIS (Ocean Biodiversity Information System) API v3
    to retrieve species occurrence records for marine biodiversity
    monitoring.  The loader supports spatial filtering via WKT geometry,
    temporal filtering, and paginated result retrieval.

    Ground truth events cover major coral bleaching and marine heatwave
    episodes where significant biodiversity loss was documented.  Anomaly
    labels are generated by comparing species richness in spatial grid
    cells against a multi-year baseline: cells with a richness drop
    exceeding 30% are labeled as anomalous.

    Attributes:
        DOMAIN: ``"marine"``
        SOURCE_URL: OBIS API v3 occurrence endpoint.
        REQUIRES_API_KEY: ``False`` -- OBIS data is freely available.
    """

    DOMAIN: str = "marine"
    SOURCE_URL: str = _OBIS_BASE_URL
    REQUIRES_API_KEY: bool = False
    FEATURE_COLUMNS: list[str] = [
        "occurrence_count",
        "species_richness",
        "mean_depth",
        "depth_std",
        "lat_centroid",
        "lon_centroid",
        "temporal_change",
        "sst_anomaly_proxy",
    ]

    # Cache historical event data for 24 hours (events are static).
    CACHE_TTL: int = 86400

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def fetch_realtime(self) -> pd.DataFrame:
        """Fetch recent marine occurrence records from OBIS.

        Queries the OBIS API for the most recent occurrence records of
        reef indicator species.  Results are paginated and concatenated
        into a single DataFrame.

        Returns:
            DataFrame with columns: ``scientificName``, ``decimalLatitude``,
            ``decimalLongitude``, ``depth``, ``date_year``,
            ``eventDate``, ``species``, ``dataset_id``.

        Raises:
            ConnectionError: If the OBIS API is unreachable after retries.
        """
        cache_key = "marine_realtime"
        cached = self._read_cache(cache_key)

        if cached is not None:
            logger.debug("Returning cached real-time marine data.")
            return pd.DataFrame(cached)

        frames: list[pd.DataFrame] = []
        for species_name in _REEF_INDICATOR_SPECIES[:5]:
            try:
                records = self._fetch_occurrences(scientificname=species_name)
                if records:
                    frames.append(pd.DataFrame(records))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "marine: failed to fetch occurrences for %s: %s",
                    species_name,
                    exc,
                )

        if not frames:
            raise ConnectionError(
                "marine: could not retrieve occurrence data from OBIS "
                "for any indicator species."
            )

        df = pd.concat(frames, ignore_index=True)
        df = self._normalize_columns(df)

        self._write_cache(cache_key, df.to_dict(orient="list"))
        logger.info(
            "marine: fetched %d real-time occurrence records from OBIS.",
            len(df),
        )
        return df

    def fetch_historical(self, event_id: str) -> pd.DataFrame:
        """Fetch occurrence data for a specific historical event.

        Retrieves OBIS occurrence records within the spatial and temporal
        bounds defined by the event catalog entry.  Both the event period
        and the preceding baseline period are fetched so that biodiversity
        change can be computed.

        Args:
            event_id: Key into the ground truth catalog (e.g.
                ``"gbr_bleaching_2016"``).

        Returns:
            DataFrame with normalized occurrence columns plus a
            ``period`` column indicating ``"baseline"`` or ``"event"``.

        Raises:
            ValueError: If *event_id* is not in the catalog.
            ConnectionError: If the OBIS API is unreachable.
        """
        if event_id not in _EVENT_CATALOG:
            raise ValueError(
                f"Unknown event_id {event_id!r}. "
                f"Available: {list(_EVENT_CATALOG.keys())}"
            )

        cache_key = f"marine_historical_{event_id}"
        cached = self._read_cache(cache_key)

        if cached is not None:
            logger.debug(
                "Returning cached historical marine data for '%s'.", event_id
            )
            return pd.DataFrame(cached)

        event = _EVENT_CATALOG[event_id]
        region = event["region"]
        geometry = _make_wkt_polygon(
            lat_min=region["lat_min"],
            lat_max=region["lat_max"],
            lon_min=region["lon_min"],
            lon_max=region["lon_max"],
        )

        # Fetch baseline period records.
        baseline_frames: list[pd.DataFrame] = []
        for species_name in event["species"]:
            try:
                records = self._fetch_occurrences(
                    scientificname=species_name,
                    geometry=geometry,
                    startdate=event["baseline_start"],
                    enddate=event["baseline_end"],
                )
                if records:
                    df_sp = pd.DataFrame(records)
                    df_sp["period"] = "baseline"
                    baseline_frames.append(df_sp)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "marine: baseline fetch failed for %s: %s",
                    species_name,
                    exc,
                )

        # Fetch event period records.
        event_frames: list[pd.DataFrame] = []
        for species_name in event["species"]:
            try:
                records = self._fetch_occurrences(
                    scientificname=species_name,
                    geometry=geometry,
                    startdate=event["start_date"],
                    enddate=event["end_date"],
                )
                if records:
                    df_sp = pd.DataFrame(records)
                    df_sp["period"] = "event"
                    event_frames.append(df_sp)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "marine: event fetch failed for %s: %s",
                    species_name,
                    exc,
                )

        all_frames = baseline_frames + event_frames
        if not all_frames:
            logger.warning(
                "marine: no occurrence data retrieved for event '%s'. "
                "Generating synthetic data.",
                event_id,
            )
            df = self._synthesize_event(event)
        else:
            df = pd.concat(all_frames, ignore_index=True)
            df = self._normalize_columns(df)

        self._write_cache(cache_key, df.to_dict(orient="list"))
        logger.info(
            "marine: fetched %d records for event '%s'.", len(df), event_id
        )
        return df

    def list_events(self) -> list[dict[str, Any]]:
        """Return the catalog of ground-truth marine biodiversity events.

        Returns:
            List of dicts each containing ``event_id``, ``name``,
            ``date``, and ``description`` keys.
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
        """Generate binary anomaly labels for a historical marine event.

        Labeling strategy: occurrence records are binned into spatial grid
        cells.  For each cell the species richness during the event period
        is compared to the 5-year baseline mean.  Cells where species
        richness dropped by more than 30% from the baseline are labeled
        as anomalous (``1``).  Cells with stable or increasing richness
        are labeled normal (``0``).

        When no baseline data is available for a cell (i.e. the cell only
        appears during the event period) it is labeled normal, since the
        absence of baseline records makes a decline determination
        unreliable.

        Args:
            event_id: Key into the ground truth catalog.

        Returns:
            1-D binary numpy array of shape ``(n_grid_cells,)``.

        Raises:
            ValueError: If *event_id* is not recognized.
        """
        if event_id not in _EVENT_CATALOG:
            raise ValueError(
                f"Unknown event_id {event_id!r}. "
                f"Available: {list(_EVENT_CATALOG.keys())}"
            )

        df = self.fetch_historical(event_id)
        if df.empty:
            return np.array([], dtype=np.int64)

        # Assign grid cells.
        df = _assign_grid_cells(df, resolution=_GRID_RESOLUTION)

        # Compute species richness per grid cell per period.
        baseline_richness = _compute_richness_by_cell(
            df[df["period"] == "baseline"]
        )
        event_richness = _compute_richness_by_cell(
            df[df["period"] == "event"]
        )

        # Build the union of all grid cells.
        all_cells = sorted(
            set(baseline_richness.keys()) | set(event_richness.keys())
        )

        if not all_cells:
            return np.array([], dtype=np.int64)

        labels = np.zeros(len(all_cells), dtype=np.int64)

        for i, cell in enumerate(all_cells):
            baseline_val = baseline_richness.get(cell, 0.0)
            event_val = event_richness.get(cell, 0.0)

            if baseline_val > 0:
                change_ratio = (baseline_val - event_val) / baseline_val
                if change_ratio > 0.30:
                    labels[i] = 1

        anomaly_count = int(labels.sum())
        logger.info(
            "marine: ground truth for '%s' — %d anomaly / %d total cells "
            "(>30%% richness drop threshold).",
            event_id,
            anomaly_count,
            len(labels),
        )
        return labels

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def engineer_features(self, raw_data: pd.DataFrame) -> np.ndarray:
        """Transform raw OBIS occurrence data into a feature matrix.

        Occurrence records are binned into spatial grid cells and the
        following features are computed per cell:

        1. **occurrence_count** -- total number of occurrence records.
        2. **species_richness** -- number of unique species (biodiversity
           index).
        3. **mean_depth** -- average reported depth of occurrences.
        4. **depth_std** -- standard deviation of reported depths.
        5. **lat_centroid** -- latitude of the grid cell centroid.
        6. **lon_centroid** -- longitude of the grid cell centroid.
        7. **temporal_change** -- year-over-year change in occurrence
           count (event period vs. baseline).  Zero if no baseline data
           is available.
        8. **sst_anomaly_proxy** -- proxy for sea surface temperature
           anomaly derived from the shift in species occurrence patterns.
           Computed as the ratio of warm-water indicator species to total
           occurrences.

        Args:
            raw_data: DataFrame from :meth:`fetch_realtime` or
                :meth:`fetch_historical`.

        Returns:
            2-D numpy array of shape ``(n_grid_cells, 8)``.
        """
        if raw_data.empty:
            return np.empty((0, 8), dtype=np.float64)

        df = raw_data.copy()
        df = _assign_grid_cells(df, resolution=_GRID_RESOLUTION)

        has_period = "period" in df.columns
        if has_period:
            baseline_df = df[df["period"] == "baseline"]
            event_df = df[df["period"] == "event"]
            # Use event period as the primary data source when available.
            primary_df = event_df if not event_df.empty else df
        else:
            baseline_df = pd.DataFrame()
            primary_df = df

        # Use union of all grid cells (matching get_ground_truth behaviour).
        all_cell_set = set(primary_df["grid_cell"].unique())
        if not baseline_df.empty and "grid_cell" in baseline_df.columns:
            all_cell_set |= set(baseline_df["grid_cell"].unique())
        grid_cells = sorted(all_cell_set)
        if not grid_cells:
            return np.empty((0, 8), dtype=np.float64)

        # Pre-compute baseline occurrence counts per cell.
        baseline_counts: dict[str, float] = {}
        if not baseline_df.empty:
            for cell, group in baseline_df.groupby("grid_cell"):
                baseline_counts[str(cell)] = float(len(group))

        n_cells = len(grid_cells)
        features = np.zeros((n_cells, 8), dtype=np.float64)

        for i, cell in enumerate(grid_cells):
            cell_data = primary_df[primary_df["grid_cell"] == cell]

            # Feature 1: occurrence count
            occurrence_count = float(len(cell_data))
            features[i, 0] = occurrence_count

            # Feature 2: species richness
            if "scientificName" in cell_data.columns:
                species_richness = float(
                    cell_data["scientificName"].nunique()
                )
            else:
                species_richness = 0.0
            features[i, 1] = species_richness

            # Feature 3: mean depth
            if "depth" in cell_data.columns:
                depth_vals = pd.to_numeric(
                    cell_data["depth"], errors="coerce"
                ).dropna()
                features[i, 2] = (
                    float(depth_vals.mean()) if len(depth_vals) > 0 else 0.0
                )
                # Feature 4: depth standard deviation
                features[i, 3] = (
                    float(depth_vals.std()) if len(depth_vals) > 1 else 0.0
                )
            else:
                features[i, 2] = 0.0
                features[i, 3] = 0.0

            # Feature 5: latitude centroid
            if "decimalLatitude" in cell_data.columns:
                lat_vals = pd.to_numeric(
                    cell_data["decimalLatitude"], errors="coerce"
                ).dropna()
                features[i, 4] = (
                    float(lat_vals.mean()) if len(lat_vals) > 0 else 0.0
                )
            else:
                features[i, 4] = 0.0

            # Feature 6: longitude centroid
            if "decimalLongitude" in cell_data.columns:
                lon_vals = pd.to_numeric(
                    cell_data["decimalLongitude"], errors="coerce"
                ).dropna()
                features[i, 5] = (
                    float(lon_vals.mean()) if len(lon_vals) > 0 else 0.0
                )
            else:
                features[i, 5] = 0.0

            # Feature 7: temporal change (year-over-year)
            baseline_count = baseline_counts.get(str(cell), 0.0)
            if baseline_count > 0:
                features[i, 6] = (
                    (occurrence_count - baseline_count) / baseline_count
                )
            else:
                features[i, 6] = 0.0

            # Feature 8: SST anomaly proxy (warm-water species ratio)
            features[i, 7] = _compute_sst_anomaly_proxy(cell_data)

        # Clean up non-finite values.
        features = np.where(np.isinf(features), np.nan, features)
        for col_idx in range(features.shape[1]):
            col = features[:, col_idx]
            mask = np.isnan(col)
            if mask.any():
                median_val = np.nanmedian(col)
                col[mask] = median_val if np.isfinite(median_val) else 0.0

        return features

    # ------------------------------------------------------------------
    # Private helpers — OBIS API interaction
    # ------------------------------------------------------------------

    def _fetch_occurrences(
        self,
        scientificname: str | None = None,
        geometry: str | None = None,
        startdate: str | None = None,
        enddate: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch occurrence records from the OBIS API with pagination.

        Args:
            scientificname: Taxonomic name to query (e.g. ``"Acropora"``).
            geometry: WKT POLYGON string for spatial filtering.
            startdate: Start date in ISO format (``YYYY-MM-DD``).
            enddate: End date in ISO format (``YYYY-MM-DD``).

        Returns:
            List of occurrence record dicts from the OBIS ``"results"``
            array.
        """
        params: dict[str, str] = {
            "size": str(_OBIS_PAGE_SIZE),
        }

        if scientificname:
            params["scientificname"] = scientificname
        if geometry:
            params["geometry"] = geometry
        if startdate:
            params["startdate"] = startdate
        if enddate:
            params["enddate"] = enddate

        all_records: list[dict[str, Any]] = []
        offset = 0

        for page in range(_OBIS_MAX_PAGES):
            params["offset"] = str(offset)

            response = self._fetch_json(_OBIS_BASE_URL, params=params)
            results = response.get("results", [])
            total = response.get("total", 0)

            if not results:
                break

            all_records.extend(results)
            offset += len(results)

            logger.debug(
                "marine: page %d — fetched %d/%d records for %s",
                page + 1,
                len(all_records),
                total,
                scientificname or "all species",
            )

            if offset >= total:
                break

        return all_records

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize OBIS result columns to a consistent schema.

        Ensures the DataFrame contains the expected columns with correct
        dtypes.  Missing columns are filled with appropriate defaults.

        Args:
            df: Raw DataFrame from OBIS results.

        Returns:
            DataFrame with normalized columns.
        """
        expected_columns: dict[str, Any] = {
            "scientificName": "",
            "decimalLatitude": np.nan,
            "decimalLongitude": np.nan,
            "depth": np.nan,
            "date_year": np.nan,
            "eventDate": "",
            "species": "",
            "dataset_id": "",
        }

        for col, default in expected_columns.items():
            if col not in df.columns:
                df[col] = default

        # Coerce numeric columns.
        for col in ("decimalLatitude", "decimalLongitude", "depth", "date_year"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    @staticmethod
    def _synthesize_event(event: dict[str, Any]) -> pd.DataFrame:
        """Generate synthetic occurrence data for an event.

        Used as a fallback when the OBIS API returns no data for a
        historical event.  Creates a realistic distribution of occurrence
        records across the event region, with reduced species richness
        in a subset of grid cells to simulate biodiversity loss.

        Sampling strategy (fixes class imbalance):
          1. Baseline period + event region: normal samples (~35%)
          2. Event period + event region (impacted): anomaly samples (~30%)
          3. Event period + control region (offset +10deg lon): normal (~35%)

        This ensures at least 50 grid cells with 20-40% anomaly ratio.

        Args:
            event: Event dict from :pydata:`_EVENT_CATALOG`.

        Returns:
            DataFrame with synthetic occurrence records including both
            baseline and event periods.
        """
        rng = np.random.default_rng(
            abs(hash(event["name"])) % (2**31)
        )

        region = event["region"]
        lat_min = region["lat_min"]
        lat_max = region["lat_max"]
        lon_min = region["lon_min"]
        lon_max = region["lon_max"]
        species_list = event["species"]

        # Use a focused sub-region to concentrate records into grid cells.
        # For large regions (e.g. global marine_heatwave_2023), narrow
        # the bounds to ensure dense grid cell coverage.
        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min
        if lat_range > 20:
            # Focus on a 10-degree band for dense coverage
            lat_min_focus = lat_min + lat_range * 0.3
            lat_max_focus = lat_min_focus + 10.0
        else:
            lat_min_focus = lat_min
            lat_max_focus = lat_max
        if lon_range > 20:
            lon_min_focus = lon_min + lon_range * 0.3
            lon_max_focus = lon_min_focus + 12.0
        else:
            lon_min_focus = lon_min
            lon_max_focus = lon_max

        records: list[dict[str, Any]] = []

        # --- 1. Baseline period records (event region, healthy) ---
        n_baseline = 1500
        for _ in range(n_baseline):
            lat = rng.uniform(lat_min_focus, lat_max_focus)
            lon = rng.uniform(lon_min_focus, lon_max_focus)
            sp = rng.choice(species_list)
            depth = rng.uniform(0, 100)
            year = rng.integers(
                int(event["baseline_start"][:4]),
                int(event["baseline_end"][:4]) + 1,
            )
            records.append(
                {
                    "scientificName": sp,
                    "decimalLatitude": lat,
                    "decimalLongitude": lon,
                    "depth": depth,
                    "date_year": int(year),
                    "eventDate": f"{year}-06-15",
                    "species": sp,
                    "dataset_id": "synthetic",
                    "period": "baseline",
                }
            )

        # --- 2. Event period records (impacted sub-region) ---
        # Northern half of focused region is impacted (reduced species).
        mid_lat = (lat_min_focus + lat_max_focus) / 2.0
        reduced_species = species_list[: max(len(species_list) // 3, 2)]
        n_event_impacted = 800
        for _ in range(n_event_impacted):
            lat = rng.uniform(mid_lat, lat_max_focus)
            lon = rng.uniform(lon_min_focus, lon_max_focus)
            sp = rng.choice(reduced_species)
            depth = rng.uniform(0, 60)  # shallower in bleached areas
            year = int(event["start_date"][:4])
            records.append(
                {
                    "scientificName": sp,
                    "decimalLatitude": lat,
                    "decimalLongitude": lon,
                    "depth": depth,
                    "date_year": year,
                    "eventDate": f"{year}-06-15",
                    "species": sp,
                    "dataset_id": "synthetic",
                    "period": "event",
                }
            )

        # --- 3. Event period records (southern half, less impacted) ---
        n_event_normal = 700
        for _ in range(n_event_normal):
            lat = rng.uniform(lat_min_focus, mid_lat)
            lon = rng.uniform(lon_min_focus, lon_max_focus)
            sp = rng.choice(species_list)
            depth = rng.uniform(0, 100)
            year = int(event["start_date"][:4])
            records.append(
                {
                    "scientificName": sp,
                    "decimalLatitude": lat,
                    "decimalLongitude": lon,
                    "depth": depth,
                    "date_year": year,
                    "eventDate": f"{year}-06-15",
                    "species": sp,
                    "dataset_id": "synthetic",
                    "period": "event",
                }
            )

        # --- 4. Control region (offset +10deg lon, event period, healthy) ---
        control_lon_min = lon_max_focus + 5.0
        control_lon_max = control_lon_min + (lon_max_focus - lon_min_focus)
        n_control = 1000
        for _ in range(n_control):
            lat = rng.uniform(lat_min_focus, lat_max_focus)
            lon = rng.uniform(control_lon_min, control_lon_max)
            sp = rng.choice(species_list)
            depth = rng.uniform(0, 100)
            year = int(event["start_date"][:4])
            records.append(
                {
                    "scientificName": sp,
                    "decimalLatitude": lat,
                    "decimalLongitude": lon,
                    "depth": depth,
                    "date_year": year,
                    "eventDate": f"{year}-06-15",
                    "species": sp,
                    "dataset_id": "synthetic",
                    "period": "event",
                }
            )

        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Pure-function helpers (no sklearn dependency)
# ---------------------------------------------------------------------------


def _make_wkt_polygon(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> str:
    """Build a WKT POLYGON string from bounding box coordinates.

    Args:
        lat_min: Southern latitude boundary.
        lat_max: Northern latitude boundary.
        lon_min: Western longitude boundary.
        lon_max: Eastern longitude boundary.

    Returns:
        WKT POLYGON string suitable for the OBIS ``geometry`` parameter.
    """
    return (
        f"POLYGON(("
        f"{lon_min} {lat_min},"
        f"{lon_max} {lat_min},"
        f"{lon_max} {lat_max},"
        f"{lon_min} {lat_max},"
        f"{lon_min} {lat_min}"
        f"))"
    )


def _assign_grid_cells(
    df: pd.DataFrame,
    resolution: float = _GRID_RESOLUTION,
) -> pd.DataFrame:
    """Assign each occurrence record to a spatial grid cell.

    Grid cells are defined by flooring the latitude and longitude to
    the nearest multiple of *resolution*.  A ``grid_cell`` column is
    added containing a string identifier in the format ``"lat_lon"``
    (e.g. ``"-15.0_148.0"``).

    Args:
        df: DataFrame with ``decimalLatitude`` and ``decimalLongitude``
            columns.
        resolution: Grid cell size in degrees.

    Returns:
        DataFrame with an added ``grid_cell`` column.
    """
    df = df.copy()

    lat = pd.to_numeric(df.get("decimalLatitude", pd.Series(dtype=float)), errors="coerce")
    lon = pd.to_numeric(df.get("decimalLongitude", pd.Series(dtype=float)), errors="coerce")

    grid_lat = np.floor(lat / resolution) * resolution
    grid_lon = np.floor(lon / resolution) * resolution

    df["grid_cell"] = [
        f"{la}_{lo}" if (np.isfinite(la) and np.isfinite(lo)) else "unknown"
        for la, lo in zip(grid_lat, grid_lon)
    ]

    return df


def _compute_richness_by_cell(
    df: pd.DataFrame,
) -> dict[str, float]:
    """Compute species richness per grid cell.

    Args:
        df: DataFrame with ``grid_cell`` and ``scientificName`` columns.

    Returns:
        Dict mapping grid cell identifier to species richness count.
    """
    if df.empty or "grid_cell" not in df.columns:
        return {}

    richness: dict[str, float] = {}
    if "scientificName" not in df.columns:
        return richness

    for cell, group in df.groupby("grid_cell"):
        richness[str(cell)] = float(group["scientificName"].nunique())

    return richness


def _compute_sst_anomaly_proxy(cell_data: pd.DataFrame) -> float:
    """Compute an SST anomaly proxy from species occurrence patterns.

    The proxy is the ratio of warm-water indicator genera (those in the
    first half of :pydata:`_REEF_INDICATOR_SPECIES`, which tend to be
    more temperature-sensitive) to total occurrences.  Higher ratios
    may indicate normal conditions; a sudden drop suggests stress from
    elevated temperatures causing absence of expected species.

    Args:
        cell_data: DataFrame of occurrences within a single grid cell.

    Returns:
        Float in ``[0.0, 1.0]`` representing the warm-water species ratio.
        Returns ``0.0`` if no species data is available.
    """
    if cell_data.empty or "scientificName" not in cell_data.columns:
        return 0.0

    total = len(cell_data)
    if total == 0:
        return 0.0

    # First half of indicator list are the most temperature-sensitive genera.
    warm_indicators = set(
        _REEF_INDICATOR_SPECIES[: len(_REEF_INDICATOR_SPECIES) // 2]
    )

    warm_count = sum(
        1
        for name in cell_data["scientificName"]
        if any(indicator in str(name) for indicator in warm_indicators)
    )

    return float(warm_count) / float(total)
