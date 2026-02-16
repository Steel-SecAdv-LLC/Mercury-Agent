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

Domain loader for cross-domain FEMA disaster data from OpenFEMA API.

Connects to the OpenFEMA Disaster Declarations Summaries endpoint to
provide federal disaster declaration data for Mercury anomaly detection.
Ground truth events cover major disasters where declarations requiring
full federal response (Major Disaster with both IA and PA programs) are
labeled as anomalies against a background of emergency-only declarations.

Use case: cross-domain validation -- correlate FEMA disaster declarations
with Mercury's domain-specific detectors (earthquake, tsunami, etc.).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from omni_mercury_engine.loaders.base import BaseDomainLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenFEMA API endpoints
# ---------------------------------------------------------------------------
_DECLARATIONS_URL = (
    "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"
)
_HAZARD_MITIGATION_URL = (
    "https://www.fema.gov/api/open/v2/HazardMitigationGrants"
)

# ---------------------------------------------------------------------------
# Pagination limits
# ---------------------------------------------------------------------------
_PAGE_SIZE: int = 1000
_MAX_RECORDS: int = 10000

# ---------------------------------------------------------------------------
# Disaster type encoding
# ---------------------------------------------------------------------------
_INCIDENT_TYPE_MAP: dict[str, int] = {
    "Hurricane": 1,
    "Flood": 2,
    "Fire": 3,
    "Tornado": 4,
    "Earthquake": 5,
    "Severe Storm(s)": 6,
    "Snow": 7,
    "Ice Storm": 8,
    "Typhoon": 9,
    "Mud/Landslide": 10,
    "Coastal Storm": 11,
    "Drought": 12,
    "Freezing": 13,
    "Severe Ice Storm": 14,
    "Dam/Levee Break": 15,
    "Volcanic Eruption": 16,
    "Tsunami": 17,
    "Toxic Substances": 18,
    "Other": 99,
}

# ---------------------------------------------------------------------------
# Declaration type encoding
# ---------------------------------------------------------------------------
_DECLARATION_TYPE_MAP: dict[str, int] = {
    "DR": 1,   # Major Disaster
    "EM": 2,   # Emergency
    "FM": 3,   # Fire Management
    "FS": 4,   # Fire Suppression
}

# ---------------------------------------------------------------------------
# US state FIPS codes
# ---------------------------------------------------------------------------
_STATE_FIPS: dict[str, int] = {
    "AL": 1, "AK": 2, "AZ": 4, "AR": 5, "CA": 6, "CO": 8, "CT": 9,
    "DE": 10, "DC": 11, "FL": 12, "GA": 13, "HI": 15, "ID": 16,
    "IL": 17, "IN": 18, "IA": 19, "KS": 20, "KY": 21, "LA": 22,
    "ME": 23, "MD": 24, "MA": 25, "MI": 26, "MN": 27, "MS": 28,
    "MO": 29, "MT": 30, "NE": 31, "NV": 32, "NH": 33, "NJ": 34,
    "NM": 35, "NY": 36, "NC": 37, "ND": 38, "OH": 39, "OK": 40,
    "OR": 41, "PA": 42, "RI": 44, "SC": 45, "SD": 46, "TN": 47,
    "TX": 48, "UT": 49, "VT": 50, "VA": 51, "WA": 53, "WV": 54,
    "WI": 55, "WY": 56, "AS": 60, "GU": 66, "MH": 68, "FM": 64,
    "MP": 69, "PW": 70, "PR": 72, "VI": 78,
}

# ---------------------------------------------------------------------------
# Ground-truth event catalog
# ---------------------------------------------------------------------------
_EVENT_CATALOG: dict[str, dict[str, Any]] = {
    "flood_2024": {
        "name": "2024 Flood Disaster Declarations",
        "date": "2024-01-01",
        "description": (
            "All flood disaster declarations for fiscal year 2024. "
            "Captures major flooding events requiring federal response."
        ),
        "filter": "incidentType eq 'Flood' and fyDeclared eq 2024",
    },
    "hurricane_2024": {
        "name": "2024 Hurricane Disaster Declarations",
        "date": "2024-01-01",
        "description": (
            "All hurricane disaster declarations for 2024. "
            "Includes major Atlantic and Gulf hurricane seasons."
        ),
        "filter": "incidentType eq 'Hurricane'",
    },
    "fire_2023": {
        "name": "2023 Fire Declarations",
        "date": "2023-01-01",
        "description": (
            "All fire-related disaster declarations for 2023. "
            "Covers wildfire and fire management assistance."
        ),
        "filter": "incidentType eq 'Fire'",
    },
    "all_2023": {
        "name": "All 2023 Disaster Declarations",
        "date": "2023-01-01",
        "description": (
            "Complete set of disaster declarations for 2023 across "
            "all incident types and declaration categories."
        ),
        "filter": "fyDeclared eq 2023",
    },
    "earthquake_all": {
        "name": "All Earthquake Declarations",
        "date": "1953-01-01",
        "description": (
            "All earthquake-related disaster declarations in the "
            "OpenFEMA database. Useful for cross-domain validation "
            "with the earthquake domain loader."
        ),
        "filter": "incidentType eq 'Earthquake'",
    },
}


class FEMALoader(BaseDomainLoader):
    """Domain loader for FEMA disaster declaration data from OpenFEMA.

    Connects to the OpenFEMA Disaster Declarations Summaries API to
    retrieve federal disaster declaration records.  Supports OData-style
    filtering and pagination for targeted queries.

    Data sources:

    * **Disaster Declarations Summaries** -- primary endpoint for all
      federally declared disasters since 1953.
    * **Hazard Mitigation Grants** -- supplementary grant data (future).
    * **IPAWS Archived Alerts** -- CAP alert integration (future).

    Feature engineering produces observables suitable for anomaly
    detection: disaster type encoding, declaration type, program
    designations, temporal features, and geographic clustering.

    Ground truth labeling: Major Disaster declarations (DR) with both
    Individual Assistance (IA) and Public Assistance (PA) programs
    designated are labeled anomaly (1) = severe events requiring full
    federal response.  Emergency-only (EM) declarations are labeled
    normal (0).

    Attributes:
        DOMAIN: ``"fema"``
        SOURCE_URL: OpenFEMA Disaster Declarations Summaries endpoint.
        REQUIRES_API_KEY: ``False`` -- OpenFEMA data is freely available.
    """

    DOMAIN: str = "fema"
    SOURCE_URL: str = _DECLARATIONS_URL
    REQUIRES_API_KEY: bool = False
    FEATURE_COLUMNS: list[str] = [
        "disaster_type",
        "state_fips",
        "declaration_month",
        "declaration_day_of_year",
        "days_since_last_same_state",
        "declarations_trailing_12mo_same_state",
        "declarations_trailing_12mo_national",
        "program_count",
        "is_major_disaster",
        "geographic_cluster",
        "time_between_declarations",
    ]

    # Cache historical event data for 24 hours (declarations are stable).
    CACHE_TTL: int = 86400

    def _fetch_fema_json(
        self,
        url: str,
        params: dict[str, str],
    ) -> Any:
        """Fetch JSON from the FEMA OpenFEMA API with OData $-parameters.

        The base ``_fetch_json`` method uses ``urllib.parse.urlencode`` which
        percent-encodes ``$`` as ``%24`` in query keys.  The FEMA OData API
        requires literal ``$`` characters, so we build the query string
        manually.

        Args:
            url: Base URL.
            params: OData query parameters (may contain ``$`` keys).

        Returns:
            Parsed JSON response.
        """
        import urllib.parse

        # FEMA OData API requires $inlinecount=allpages for $filter to work
        if "$inlinecount" not in params:
            params["$inlinecount"] = "allpages"

        parts: list[str] = []
        for k, v in params.items():
            # Encode value (spaces -> %20, quotes -> %27) but keep $ in keys
            encoded_val = urllib.parse.quote(str(v), safe="")
            parts.append(f"{k}={encoded_val}")
        query_string = "&".join(parts)
        full_url = f"{url}?{query_string}"
        data = self._fetch_url(full_url)
        import json as _json

        return _json.loads(data)

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def fetch_realtime(self) -> pd.DataFrame:
        """Fetch the most recent disaster declarations from OpenFEMA.

        Retrieves the latest page of disaster declaration summaries,
        ordered by declaration date descending, providing a snapshot
        of the most recent federal disaster activity.

        Returns:
            DataFrame with columns: disasterNumber, state,
            declarationType, declarationDate, incidentType,
            fyDeclared, incidentBeginDate, incidentEndDate,
            ihProgramDeclared, iaProgramDeclared, paProgramDeclared,
            hmProgramDeclared, declarationTitle.

        Raises:
            ConnectionError: If the OpenFEMA API is unreachable
                after retries.
        """
        cache_key = "fema_realtime"
        cached = self._read_cache(cache_key)

        if cached is not None:
            logger.debug("Returning cached real-time FEMA data.")
            return pd.DataFrame(cached)

        params: dict[str, str] = {
            "$orderby": "declarationDate desc",
            "$top": str(_PAGE_SIZE),
        }

        data = self._fetch_fema_json(_DECLARATIONS_URL, params)
        records = data.get("DisasterDeclarationsSummaries", [])
        df = self._records_to_dataframe(records)

        self._write_cache(cache_key, df.to_dict(orient="list"))
        logger.info(
            "Fetched %d real-time FEMA declaration records.", len(df)
        )
        return df

    def fetch_historical(self, event_id: str) -> pd.DataFrame:
        """Fetch disaster declaration data for a specific ground truth event.

        Uses OData-style filtering to retrieve declarations matching
        the event criteria.  Automatically paginates through results
        up to the API maximum of 10,000 records.

        Args:
            event_id: Key into the ground truth catalog (e.g.
                ``"flood_2024"``, ``"all_2023"``).

        Returns:
            DataFrame with the same schema as :meth:`fetch_realtime`.

        Raises:
            ValueError: If *event_id* is not in the catalog.
            ConnectionError: If the OpenFEMA API is unreachable.
        """
        if event_id not in _EVENT_CATALOG:
            raise ValueError(
                f"Unknown event_id '{event_id}'. "
                f"Available: {list(_EVENT_CATALOG.keys())}"
            )

        cache_key = f"fema_historical_{event_id}"
        cached = self._read_cache(cache_key)

        if cached is not None:
            logger.debug(
                "Returning cached historical data for '%s'.", event_id
            )
            return pd.DataFrame(cached)

        event = _EVENT_CATALOG[event_id]
        all_records: list[dict[str, Any]] = []
        skip = 0

        while skip < _MAX_RECORDS:
            params: dict[str, str] = {
                "$filter": event["filter"],
                "$skip": str(skip),
                "$top": str(_PAGE_SIZE),
                "$orderby": "declarationDate asc",
            }

            data = self._fetch_fema_json(_DECLARATIONS_URL, params)
            records = data.get("DisasterDeclarationsSummaries", [])

            if not records:
                break

            all_records.extend(records)
            logger.debug(
                "fema: fetched page at skip=%d, got %d records.",
                skip,
                len(records),
            )

            if len(records) < _PAGE_SIZE:
                break
            skip += _PAGE_SIZE

        df = self._records_to_dataframe(all_records)

        if df.empty:
            logger.warning(
                "OpenFEMA returned no records for event '%s'.", event_id
            )
            return df

        # Sort chronologically for temporal feature engineering.
        if "declarationDate" in df.columns:
            df = df.sort_values("declarationDate").reset_index(drop=True)

        self._write_cache(cache_key, df.to_dict(orient="list"))
        logger.info(
            "Fetched %d historical records for event '%s'.",
            len(df),
            event_id,
        )
        return df

    def list_events(self) -> list[dict[str, Any]]:
        """Return the catalog of ground truth FEMA events.

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
        """Generate binary anomaly labels for a historical FEMA event.

        Labeling strategy: a declaration is labeled *anomalous* (``1``)
        if it is a Major Disaster (declaration type ``DR``) with both
        Individual Assistance (IA) and Public Assistance (PA) programs
        designated.  These represent the most severe events requiring
        full federal response.  Emergency-only (``EM``) declarations
        and declarations without both IA and PA are labeled *normal*
        (``0``).

        Args:
            event_id: Key into the ground truth catalog.

        Returns:
            1-D binary numpy array of shape ``(n_records,)``.

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

        # Anomaly = Major Disaster (DR) with >= 3 assistance programs.
        # This separates catastrophic events requiring full federal
        # response from routine declarations with fewer programs.
        is_major_disaster = df["declarationType"].values == "DR"
        has_ia = df["iaProgramDeclared"].fillna(False).values.astype(np.float64)
        has_pa = df["paProgramDeclared"].fillna(False).values.astype(np.float64)
        has_hm = df["hmProgramDeclared"].fillna(False).values.astype(np.float64)
        has_ih = df["ihProgramDeclared"].fillna(False).values.astype(np.float64)
        program_count = has_ia + has_pa + has_hm + has_ih

        labels = (is_major_disaster & (program_count >= 3)).astype(np.int64)

        logger.info(
            "Ground truth for '%s': %d anomalies / %d total "
            "(DR with >=3 programs).",
            event_id,
            int(labels.sum()),
            len(labels),
        )
        return labels

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def engineer_features(self, raw_data: pd.DataFrame) -> np.ndarray:
        """Transform raw FEMA declaration data into a feature matrix.

        Engineered features (per declaration row):

        1. **disaster_type** -- encoded incident type (Hurricane=1,
           Flood=2, Fire=3, Tornado=4, Earthquake=5, etc.).
        2. **state_fips** -- numeric FIPS code for the declaring state.
        3. **declaration_month** -- month of the declaration date.
        4. **declaration_day_of_year** -- day of year (1-366),
           captures seasonal patterns.
        5. **days_since_last_same_state** -- days since the previous
           declaration in the same state (0 for the first).
        6. **declarations_trailing_12mo_same_state** -- count of
           declarations in the same state over the prior 12 months.
        7. **declarations_trailing_12mo_national** -- count of all
           declarations nationally over the prior 12 months.
        8. **program_count** -- number of assistance programs activated
           (IA + PA + HM + IH, range 0-4).
        9. **is_major_disaster** -- 1 if declaration type is DR (Major
           Disaster), 0 otherwise.
        10. **geographic_cluster** -- Census Bureau region cluster.
        11. **time_between_declarations** -- seconds since the
            previous declaration (0 for the first row).

        Args:
            raw_data: DataFrame from :meth:`fetch_realtime` or
                :meth:`fetch_historical`.

        Returns:
            2-D numpy array of shape ``(n_samples, 11)``.
        """
        if raw_data.empty:
            return np.empty((0, 11), dtype=np.float64)

        df = raw_data.copy()

        # Ensure chronological order.
        if "declarationDate" in df.columns:
            df["declarationDate"] = pd.to_datetime(
                df["declarationDate"], errors="coerce", utc=True
            )
            df = df.sort_values("declarationDate").reset_index(drop=True)

        n = len(df)

        # ---- Feature 1: disaster type encoding ----
        disaster_type = df["incidentType"].map(_INCIDENT_TYPE_MAP).fillna(99)
        disaster_type = disaster_type.values.astype(np.float64)

        # ---- Feature 2: state FIPS code ----
        state_fips = df["state"].map(_STATE_FIPS).fillna(0)
        state_fips = state_fips.values.astype(np.float64)

        # ---- Features 3-4: temporal components ----
        if "declarationDate" in df.columns:
            dates = pd.to_datetime(
                df["declarationDate"], errors="coerce", utc=True
            )
            month = dates.dt.month.fillna(0).values.astype(np.float64)
            day_of_year = dates.dt.dayofyear.fillna(0).values.astype(np.float64)
        else:
            month = np.zeros(n, dtype=np.float64)
            day_of_year = np.zeros(n, dtype=np.float64)

        # ---- Feature 5: days since last declaration in same state ----
        days_since_last_same_state = self._compute_days_since_last_same_state(df)

        # ---- Features 6-7: trailing 12-month declaration counts ----
        trailing_same_state, trailing_national = (
            self._compute_trailing_counts(df)
        )

        # ---- Feature 8: program count (0-4) ----
        ia = self._bool_column(df, "iaProgramDeclared")
        pa = self._bool_column(df, "paProgramDeclared")
        hm = self._bool_column(df, "hmProgramDeclared")
        ih = self._bool_column(df, "ihProgramDeclared")
        program_count = ia + pa + hm + ih

        # ---- Feature 9: is major disaster ----
        is_major = np.zeros(n, dtype=np.float64)
        if "declarationType" in df.columns:
            is_major = (df["declarationType"].values == "DR").astype(np.float64)

        # ---- Feature 10: geographic clustering ----
        geographic_cluster = self._compute_geographic_cluster(state_fips)

        # ---- Feature 11: time between declarations (seconds) ----
        time_between = self._compute_time_between_declarations(df)

        # Stack into feature matrix.
        features = np.column_stack(
            [
                disaster_type,
                state_fips,
                month,
                day_of_year,
                days_since_last_same_state,
                trailing_same_state,
                trailing_national,
                program_count,
                is_major,
                geographic_cluster,
                time_between,
            ]
        )

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
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _records_to_dataframe(
        records: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Convert OpenFEMA JSON records to a flat DataFrame.

        Extracts the key fields from each declaration record and
        returns a DataFrame with standardized column names.

        Args:
            records: List of declaration dicts from the OpenFEMA API.

        Returns:
            DataFrame with columns: disasterNumber, state,
            declarationType, declarationDate, incidentType,
            fyDeclared, incidentBeginDate, incidentEndDate,
            ihProgramDeclared, iaProgramDeclared, paProgramDeclared,
            hmProgramDeclared, declarationTitle.
        """
        _COLUMNS = [
            "disasterNumber",
            "state",
            "declarationType",
            "declarationDate",
            "incidentType",
            "fyDeclared",
            "incidentBeginDate",
            "incidentEndDate",
            "ihProgramDeclared",
            "iaProgramDeclared",
            "paProgramDeclared",
            "hmProgramDeclared",
            "declarationTitle",
        ]

        if not records:
            return pd.DataFrame(columns=_COLUMNS)

        rows: list[dict[str, Any]] = []
        for record in records:
            row: dict[str, Any] = {}
            for col in _COLUMNS:
                row[col] = record.get(col)
            rows.append(row)

        df = pd.DataFrame(rows, columns=_COLUMNS)

        # Coerce numeric columns.
        df["disasterNumber"] = pd.to_numeric(
            df["disasterNumber"], errors="coerce"
        )
        df["fyDeclared"] = pd.to_numeric(df["fyDeclared"], errors="coerce")

        # Coerce boolean program columns.
        for col in (
            "ihProgramDeclared",
            "iaProgramDeclared",
            "paProgramDeclared",
            "hmProgramDeclared",
        ):
            df[col] = df[col].fillna(False).astype(bool)

        return df

    @staticmethod
    def _bool_column(df: pd.DataFrame, col: str) -> np.ndarray:
        """Extract a boolean column as a float64 array (1.0 / 0.0).

        Args:
            df: Source DataFrame.
            col: Column name.

        Returns:
            1-D float64 numpy array.
        """
        if col in df.columns:
            return df[col].fillna(False).astype(np.float64).values
        return np.zeros(len(df), dtype=np.float64)

    @staticmethod
    def _compute_days_since_last_same_state(
        df: pd.DataFrame,
    ) -> np.ndarray:
        """Compute days since the last declaration in the same state.

        Args:
            df: DataFrame with ``state`` and ``declarationDate`` columns.

        Returns:
            1-D float64 array. 0 for the first occurrence in each state.
        """
        n = len(df)
        result = np.zeros(n, dtype=np.float64)

        if "state" not in df.columns or "declarationDate" not in df.columns:
            return result

        dates = pd.to_datetime(
            df["declarationDate"], errors="coerce", utc=True
        )
        states = df["state"].values

        last_date_by_state: dict[str, pd.Timestamp] = {}
        for i in range(n):
            state = str(states[i])
            current_date = dates.iloc[i]
            if pd.isna(current_date):
                continue
            if state in last_date_by_state:
                delta = current_date - last_date_by_state[state]
                result[i] = max(0.0, delta.total_seconds() / 86400.0)
            last_date_by_state[state] = current_date

        return result

    @staticmethod
    def _compute_trailing_counts(
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute trailing 12-month declaration counts.

        Returns two arrays:
        - declarations in the same state over prior 12 months
        - declarations nationally over prior 12 months

        Args:
            df: DataFrame with ``state`` and ``declarationDate`` columns.

        Returns:
            Tuple of (same_state_counts, national_counts) arrays.
        """
        n = len(df)
        same_state = np.zeros(n, dtype=np.float64)
        national = np.zeros(n, dtype=np.float64)

        if "state" not in df.columns or "declarationDate" not in df.columns:
            return same_state, national

        dates = pd.to_datetime(
            df["declarationDate"], errors="coerce", utc=True
        )
        states = df["state"].values
        one_year = pd.Timedelta(days=365)

        for i in range(n):
            current_date = dates.iloc[i]
            if pd.isna(current_date):
                continue
            cutoff = current_date - one_year
            current_state = str(states[i])

            # Count prior declarations within 12 months
            nat_count = 0
            state_count = 0
            for j in range(max(0, i - 5000), i):  # cap lookback
                prior_date = dates.iloc[j]
                if pd.isna(prior_date) or prior_date < cutoff:
                    continue
                nat_count += 1
                if str(states[j]) == current_state:
                    state_count += 1

            same_state[i] = float(state_count)
            national[i] = float(nat_count)

        return same_state, national

    @staticmethod
    def _compute_time_between_declarations(
        df: pd.DataFrame,
    ) -> np.ndarray:
        """Compute seconds between consecutive declarations.

        Used as a temporal feature for anomaly detection: clusters of
        rapid declarations may indicate unusually severe events.

        Args:
            df: DataFrame with a ``declarationDate`` column.

        Returns:
            1-D float64 array of inter-declaration time in seconds.
            The first element is always 0.
        """
        n = len(df)
        time_between = np.zeros(n, dtype=np.float64)

        if "declarationDate" not in df.columns or n < 2:
            return time_between

        dates = pd.to_datetime(
            df["declarationDate"], errors="coerce", utc=True
        )

        # Convert to epoch seconds for differencing.
        epoch_s = dates.astype(np.int64).values / 1e9

        if n > 1:
            time_between[1:] = np.diff(epoch_s)

        # Clamp negative values (possible from unsorted or bad data).
        time_between = np.maximum(time_between, 0.0)

        return time_between

    @staticmethod
    def _compute_geographic_cluster(
        state_fips: np.ndarray,
    ) -> np.ndarray:
        """Assign a geographic region cluster from state FIPS codes.

        Clusters are based on US Census Bureau regions:

        * 1 = Northeast (FIPS 9, 23, 25, 33, 34, 36, 42, 44, 50)
        * 2 = Midwest (FIPS 17-20, 26-29, 31, 38, 39, 46, 55)
        * 3 = South (FIPS 1, 5, 10-13, 21, 22, 24, 28, 37, 40, 45-48,
          51, 54)
        * 4 = West (FIPS 2, 4, 6, 8, 15, 16, 30, 32, 35, 41, 49, 53,
          56)
        * 5 = Territories (FIPS >= 60)
        * 0 = Unknown

        Args:
            state_fips: 1-D array of numeric FIPS codes.

        Returns:
            1-D float64 array of region cluster IDs.
        """
        _NORTHEAST = {9, 23, 25, 33, 34, 36, 42, 44, 50}
        _MIDWEST = {17, 18, 19, 20, 26, 27, 29, 31, 38, 39, 46, 55}
        _SOUTH = {
            1, 5, 10, 11, 12, 13, 21, 22, 24, 28, 37, 40, 45, 47, 48,
            51, 54,
        }
        _WEST = {2, 4, 6, 8, 15, 16, 30, 32, 35, 41, 49, 53, 56}

        n = len(state_fips)
        clusters = np.zeros(n, dtype=np.float64)

        for i in range(n):
            fips = int(state_fips[i])
            if fips in _NORTHEAST:
                clusters[i] = 1.0
            elif fips in _MIDWEST:
                clusters[i] = 2.0
            elif fips in _SOUTH:
                clusters[i] = 3.0
            elif fips in _WEST:
                clusters[i] = 4.0
            elif fips >= 60:
                clusters[i] = 5.0
            # else: 0.0 (unknown)

        return clusters
