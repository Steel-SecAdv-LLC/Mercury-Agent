# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Domain loader for flood data from NOAA AHPS and USGS Water Services.

Connects to the USGS National Water Information System (NWIS) Instantaneous Values web service to
retrieve streamflow gauge height and discharge time series.  Ground truth events cover major US
flood disasters where gauge readings exceeded documented NWS flood stages, plus supplemental FEMA
disaster declaration data for broader context.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from omni_mercury_engine.loaders.base import BaseDomainLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# USGS Water Services endpoints
# ---------------------------------------------------------------------------
# USGS 301-redirects ``waterservices.usgs.gov`` to the ``nwis.`` host. The
# SafeHTTPClient refuses to follow redirects by design, so every flood fetch
# failed with UnsafeURLError; request the final destination directly.
_USGS_IV_URL = "https://nwis.waterservices.usgs.gov/nwis/iv/"

# USGS parameter codes
_PARAM_GAUGE_HEIGHT = "00065"  # Gauge height, ft
_PARAM_DISCHARGE = "00060"  # Discharge, cubic feet per second

# NOAA National Water Prediction Service: every National Weather Service river
# gauge, with the flood category its forecast office assigns the latest reading.
_NWPS_GAUGES_URL = "https://api.water.noaa.gov/nwps/v1/gauges"
#: The service times out on a national request (HTTP 504 after 60 s, measured
#: 2026-11-17). These tiles (west, south, east, north, in degrees) cover the
#: fifty states, Puerto Rico and the U.S. Virgin Islands.
_NWPS_TILES: tuple[tuple[int, int, int, int], ...] = (
    (-125, 24, -105, 50),
    (-105, 24, -90, 50),
    (-90, 24, -66, 50),
    (-180, 51, -129, 72),
    (-161, 18, -154, 23),
    (-68, 17, -64, 19),
)
#: The service's "no value" for a stage or flow.
_NWPS_MISSING = -900.0
#: NWS flood categories, lowest first. "action" is below flood stage.
FLOOD_CATEGORIES: tuple[str, ...] = ("action", "minor", "moderate", "major")
_STATUS_COLUMNS = [
    "gauge_id",
    "name",
    "state",
    "latitude",
    "longitude",
    "stage",
    "unit",
    "category",
    "observed_at",
    "forecast_category",
]

# Supplemental FEMA endpoint
_FEMA_DECLARATIONS_URL = "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"

# ---------------------------------------------------------------------------
# Ground truth event catalog
# ---------------------------------------------------------------------------
_EVENT_CATALOG: dict[str, dict[str, Any]] = {
    "helene_2024": {
        "name": "2024 Hurricane Helene Flooding",
        "date": "2024-09-27",
        "description": (
            "Catastrophic flooding in Appalachia caused by Hurricane Helene. "
            "Record-breaking flood levels along the French Broad River in "
            "western North Carolina."
        ),
        "sites": ["03451500", "03439000"],
        "site_names": [
            "French Broad River at Marshall, NC",
            "French Broad River at Asheville, NC",
        ],
        "start": "2024-09-25",
        "end": "2024-10-02",
        # NWS flood stages (ft) per site
        "flood_stages": {"03451500": 11.0, "03439000": 8.0},
    },
    "vermont_2023": {
        "name": "2023 Vermont Flooding",
        "date": "2023-07-10",
        "description": (
            "Severe flooding across Vermont from heavy rainfall. "
            "The Winooski River and tributaries experienced major flooding "
            "in the Burlington and Montpelier areas."
        ),
        "sites": ["04288000"],
        "site_names": ["Winooski River at Montpelier, VT"],
        "start": "2023-07-08",
        "end": "2023-07-15",
        "flood_stages": {"04288000": 14.0},
    },
    "european_2021": {
        "name": "2021 European Floods",
        "date": "2021-07-14",
        "description": (
            "Devastating floods across western Europe, primarily Germany "
            "and Belgium. FEMA correlation only — no direct USGS gauge data."
        ),
        "sites": [],
        "site_names": [],
        "start": "2021-07-12",
        "end": "2021-07-20",
        "flood_stages": {},
        "fema_only": True,
    },
}


class FloodLoader(BaseDomainLoader):
    """Loader for flood data from USGS Water Services and FEMA.

    Uses the USGS NWIS Instantaneous Values service to retrieve gauge
    height and discharge time series at specific stream gauge sites.
    Ground truth labeling marks time periods when gauge height exceeds
    the documented NWS flood stage for each site.

    Feature engineering produces hydrological observables suitable for
    anomaly detection: gauge height, discharge rate, rate of rise,
    deviation from rolling median, and peak-to-baseline ratio.
    """

    DOMAIN: str = "flood"
    SOURCE_URL: str = _USGS_IV_URL
    # Labels = (gauge_height_ft >= NWS flood-stage). The NWS flood-stage is
    # an external per-site catalog constant, but the threshold is on
    # ``gauge_height_ft`` which is feature[0]. Feature-threshold circularity.
    LABEL_SOURCE: str = "statistical"
    REQUIRES_API_KEY: bool = False
    FEATURE_COLUMNS: list[str] = [
        "gauge_height_ft",
        "discharge_cfs",
        "rate_of_rise",
        "median_deviation",
        "peak_to_baseline",
    ]

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def fetch_realtime(self) -> pd.DataFrame:
        """Fetch the most recent 7 days of gauge data from USGS sites.

        Retrieves instantaneous values for gauge height and discharge
        from the default set of monitored sites (all sites in the event
        catalog).

        Returns:
            DataFrame with columns: datetime, site_id, site_name,
            gauge_height_ft, discharge_cfs.

        Raises:
            ConnectionError: If the USGS Water Services API is
                unreachable after retries.
        """
        cache_key = "flood_realtime"
        cached = self._read_frame_cache(cache_key)

        if cached is not None:
            logger.debug("Returning cached real-time flood data.")
            return cached

        # Collect all unique site IDs across the catalog
        all_sites: list[str] = []
        for event in _EVENT_CATALOG.values():
            for site in event["sites"]:
                if site not in all_sites:
                    all_sites.append(site)

        if not all_sites:
            logger.warning("No USGS sites configured for real-time fetch.")
            return self._empty_dataframe()

        df = self._fetch_sites(
            site_ids=all_sites,
            period="P7D",
        )

        if not df.empty:
            self._write_frame_cache(cache_key, df)
        logger.info(
            "Fetched %d real-time flood records from USGS Water Services.",
            len(df),
        )
        return df

    def fetch_historical(self, event_id: str) -> pd.DataFrame:
        """Fetch gauge data for a specific historical flood event.

        Args:
            event_id: Key into the ground truth catalog (e.g.
                ``"helene_2024"``).

        Returns:
            DataFrame with columns: datetime, site_id, site_name,
            gauge_height_ft, discharge_cfs.

        Raises:
            ValueError: If *event_id* is not in the catalog.
            ConnectionError: If the USGS Water Services API is
                unreachable after retries.
        """
        if event_id not in _EVENT_CATALOG:
            raise ValueError(
                f"Unknown event_id '{event_id}'. " f"Available: {list(_EVENT_CATALOG.keys())}"
            )

        cache_key = f"flood_historical_{event_id}"
        cached = self._read_frame_cache(cache_key)

        if cached is not None:
            logger.debug("Returning cached historical data for '%s'.", event_id)
            return cached

        event = _EVENT_CATALOG[event_id]

        # European 2021 event has no USGS sites — return FEMA data only
        if event.get("fema_only", False):
            df = self._fetch_fema_declarations(
                start_date=event["start"],
                end_date=event["end"],
            )
            if not df.empty:
                self._write_frame_cache(cache_key, df)
            logger.info(
                "Fetched %d FEMA declaration records for event '%s'.",
                len(df),
                event_id,
            )
            return df

        if not event["sites"]:
            logger.warning("No USGS sites configured for event '%s'.", event_id)
            return self._empty_dataframe()

        df = self._fetch_sites(
            site_ids=event["sites"],
            start_date=event["start"],
            end_date=event["end"],
        )

        if df.empty:
            logger.warning("USGS returned no data for event '%s'.", event_id)
            return df

        # Sort chronologically
        df = df.sort_values("datetime").reset_index(drop=True)

        self._write_frame_cache(cache_key, df)
        logger.info(
            "Fetched %d historical records for event '%s'.",
            len(df),
            event_id,
        )
        return df

    def list_events(self) -> list[dict[str, Any]]:
        """Return the catalog of ground truth flood events.

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
        """Generate binary anomaly labels for a historical flood event.

        Labeling strategy: each time step where gauge height equals or
        exceeds the NWS-documented flood stage for that site is labeled
        *anomalous* (``1``).  All readings below flood stage are labeled
        *normal* (``0``).

        For events with multiple sites, readings from all sites are
        concatenated chronologically before labeling.

        For FEMA-only events (no USGS gauge data), all records are
        labeled as anomalous since they represent confirmed flood
        disaster declarations.

        Args:
            event_id: Key into the ground truth catalog.

        Returns:
            1-D binary numpy array of shape ``(n_samples,)``.

        Raises:
            ValueError: If *event_id* is not recognized.
        """
        if event_id not in _EVENT_CATALOG:
            raise ValueError(
                f"Unknown event_id '{event_id}'. " f"Available: {list(_EVENT_CATALOG.keys())}"
            )

        df = self.fetch_historical(event_id)
        if df.empty:
            return np.array([], dtype=np.int64)

        event = _EVENT_CATALOG[event_id]

        # FEMA-only events: all declarations are anomalies
        if event.get("fema_only", False):
            return np.ones(len(df), dtype=np.int64)

        flood_stages = event["flood_stages"]
        labels = np.zeros(len(df), dtype=np.int64)

        for idx in range(len(df)):
            site_id = str(df.iloc[idx].get("site_id", ""))
            gauge_height = df.iloc[idx].get("gauge_height_ft", np.nan)

            if site_id in flood_stages and not np.isnan(gauge_height):
                if gauge_height >= flood_stages[site_id]:
                    labels[idx] = 1

        anomaly_count = int(labels.sum())
        logger.info(
            "Ground truth for '%s': %d anomalies / %d total.",
            event_id,
            anomaly_count,
            len(labels),
        )
        return labels

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def engineer_features(self, raw_data: pd.DataFrame) -> np.ndarray[Any, Any]:
        """Transform raw flood gauge data into a feature matrix.

        Engineered features (per time step):

        1. **gauge_height_ft** -- Gauge height in feet.
        2. **discharge_cfs** -- Discharge rate in cubic feet per second.
        3. **rate_of_rise** -- First derivative of gauge height (ft/hr),
           approximated by finite differences.
        4. **median_deviation** -- Deviation of gauge height from a
           rolling median (24-hour window), normalized by the median.
        5. **peak_to_baseline** -- Ratio of current gauge height to
           the rolling baseline (minimum over trailing 72-hour window).

        Args:
            raw_data: DataFrame from :meth:`fetch_realtime` or
                :meth:`fetch_historical`.

        Returns:
            2-D numpy array of shape ``(n_samples, 5)``.
        """
        if raw_data.empty:
            return np.empty((0, 5), dtype=np.float64)

        df = raw_data.copy()

        # Ensure chronological order
        if "datetime" in df.columns:
            df = df.sort_values("datetime").reset_index(drop=True)

        # ---- base observables ----
        gauge_height = df.get("gauge_height_ft")
        if gauge_height is None:
            gauge_height = pd.Series(np.zeros(len(df)), dtype=np.float64)
        gauge_height = gauge_height.astype(np.float64).values

        discharge = df.get("discharge_cfs")
        if discharge is None:
            discharge = pd.Series(np.zeros(len(df)), dtype=np.float64)
        discharge = discharge.astype(np.float64).values

        # ---- rate of rise: dH/dt in ft/hr ----
        rate_of_rise = self._compute_rate_of_rise(df, gauge_height)

        # ---- deviation from rolling median (normalized) ----
        median_deviation = self._compute_median_deviation(
            gauge_height, window=96  # ~24 hours at 15-min intervals
        )

        # ---- peak-to-baseline ratio ----
        peak_to_baseline = self._compute_peak_to_baseline(
            gauge_height, window=288  # ~72 hours at 15-min intervals
        )

        # Stack into feature matrix
        features = np.column_stack(
            [
                gauge_height,
                discharge,
                rate_of_rise,
                median_deviation,
                peak_to_baseline,
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
    # National flood status (NOAA National Water Prediction Service)
    # ------------------------------------------------------------------

    def fetch_flood_status(self) -> pd.DataFrame:
        """Every NWS river gauge's latest reading and its flood category.

        One row per gauge of the National Water Prediction Service: about
        12,000 across the fifty states and Puerto Rico. ``category`` is the
        service's own word for the latest observation: ``no_flooding``,
        ``action``, ``minor``, ``moderate`` or ``major`` where the forecast
        office has set flood stages, and ``not_defined``,
        ``obs_not_current``, ``out_of_service`` or ``low_threshold``
        otherwise.

        Returns:
            DataFrame with columns ``gauge_id``, ``name``, ``state``,
            ``latitude``, ``longitude``, ``stage``, ``unit``, ``category``,
            ``observed_at`` (UTC; ``NaT`` when the gauge has no reading) and
            ``forecast_category``.

        Raises:
            ConnectionError: If any tile cannot be fetched. A partial read
                would report the country with a region missing.
            ValueError: If a tile is not the documented shape.
        """
        cache_key = "flood_status"
        cached = self._read_frame_cache(cache_key)
        if cached is not None:
            return cached
        rows: dict[str, dict[str, Any]] = {}
        for west, south, east, north in _NWPS_TILES:
            payload = self._fetch_json(
                _NWPS_GAUGES_URL,
                {
                    "bbox.xmin": str(west),
                    "bbox.ymin": str(south),
                    "bbox.xmax": str(east),
                    "bbox.ymax": str(north),
                    "srid": "EPSG_4326",
                },
            )
            gauges = payload.get("gauges") if isinstance(payload, dict) else None
            if not isinstance(gauges, list):
                raise ValueError("NWPS gauge listing is not the documented shape: no 'gauges' list")
            for gauge in gauges:
                row = self._status_row(gauge)
                if row is not None:
                    rows[row["gauge_id"]] = row  # a gauge on a tile edge is listed twice
        frame = pd.DataFrame(list(rows.values()), columns=_STATUS_COLUMNS)
        frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True, errors="coerce")
        if not frame.empty:
            self._write_frame_cache(cache_key, frame)
        logger.info("Fetched flood status for %d NWS gauges.", len(frame))
        return frame

    @staticmethod
    def _status_row(gauge: Any) -> dict[str, Any] | None:
        """One gauge of the NWPS listing as a row, or ``None`` when it has no identifier."""
        if not isinstance(gauge, dict) or not gauge.get("lid"):
            return None

        def part(holder: dict[str, Any], key: str) -> dict[str, Any]:
            value = holder.get(key)
            return value if isinstance(value, dict) else {}

        status = part(gauge, "status")
        observed, forecast, state = (
            part(status, "observed"),
            part(status, "forecast"),
            part(gauge, "state"),
        )
        stage = pd.to_numeric(observed.get("primary"), errors="coerce")
        when = pd.to_datetime(observed.get("validTime"), utc=True, errors="coerce")
        return {
            "gauge_id": str(gauge["lid"]),
            "name": str(gauge.get("name") or gauge["lid"]).strip(),
            "state": str(state.get("name") or state.get("abbreviation") or "").strip(),
            "latitude": pd.to_numeric(gauge.get("latitude"), errors="coerce"),
            "longitude": pd.to_numeric(gauge.get("longitude"), errors="coerce"),
            "stage": float(stage) if np.isfinite(stage) and stage > _NWPS_MISSING else np.nan,
            "unit": str(observed.get("primaryUnit") or ""),
            "category": str(observed.get("floodCategory") or "not_defined"),
            # The service writes year 1 for "no reading".
            "observed_at": when if pd.notna(when) and when.year > 1900 else pd.NaT,
            "forecast_category": str(forecast.get("floodCategory") or ""),
        }

    def fetch_gauge_record(self, gauge_id: str) -> tuple[dict[str, float], pd.DataFrame]:
        """A gauge's flood stages and its observed record of the past 30 days.

        Args:
            gauge_id: The gauge's NWS location identifier (``COCF1``).

        Returns:
            ``(stages, record)``: the stage at which each flood category
            begins, in the record's unit, for the categories the forecast
            office has set (empty when it has set none in that unit); and a
            DataFrame with columns ``observed_at`` (UTC) and ``stage``,
            oldest first.

        Raises:
            ConnectionError: If the service cannot be reached.
            ValueError: If ``gauge_id`` is not an NWS identifier.
        """
        if not gauge_id.isalnum():
            raise ValueError(f"not an NWS gauge identifier: {gauge_id!r}")
        gauge = self._fetch_json(f"{_NWPS_GAUGES_URL}/{gauge_id}")
        series = self._fetch_json(f"{_NWPS_GAUGES_URL}/{gauge_id}/stageflow/observed")
        points = series.get("data") if isinstance(series, dict) else None
        record = pd.DataFrame(points if isinstance(points, list) else [])
        if record.empty or not {"validTime", "primary"} <= set(record.columns):
            record = pd.DataFrame({"observed_at": pd.to_datetime([], utc=True), "stage": []})
        else:
            record = pd.DataFrame(
                {
                    "observed_at": pd.to_datetime(record["validTime"], utc=True, errors="coerce"),
                    "stage": pd.to_numeric(record["primary"], errors="coerce"),
                }
            )
            record = record[record["observed_at"].notna() & (record["stage"] > _NWPS_MISSING)]
            record = record.sort_values("observed_at").reset_index(drop=True)
        described = gauge.get("flood") if isinstance(gauge, dict) else None
        flood: dict[str, Any] = described if isinstance(described, dict) else {}
        unit = str(series.get("primaryUnits") or "") if isinstance(series, dict) else ""
        # Stages are given as a stage (ft) and as a flow (cfs); the record is in one of them.
        if unit == str(flood.get("stageUnits") or "ft"):
            field, scale = "stage", 1.0
        elif unit == "kcfs" and str(flood.get("flowUnits") or "cfs") == "cfs":
            field, scale = "flow", 1e-3
        else:
            field, scale = "", 1.0
        stages: dict[str, float] = {}
        listed = flood.get("categories")
        categories: dict[str, Any] = listed if isinstance(listed, dict) else {}
        for category in FLOOD_CATEGORIES:
            level = pd.to_numeric((categories.get(category) or {}).get(field), errors="coerce")
            if field and np.isfinite(level) and level > _NWPS_MISSING:
                stages[category] = float(level) * scale
        return stages, record

    # ------------------------------------------------------------------
    # Private helpers — data fetching
    # ------------------------------------------------------------------

    def _fetch_sites(
        self,
        site_ids: list[str],
        period: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Fetch gauge height and discharge from USGS for given sites.

        Args:
            site_ids: List of USGS site ID strings.
            period: ISO 8601 duration (e.g. ``"P7D"``). Mutually
                exclusive with *start_date*/*end_date*.
            start_date: Start date in ISO format (``YYYY-MM-DD``).
            end_date: End date in ISO format (``YYYY-MM-DD``).

        Returns:
            DataFrame with columns: datetime, site_id, site_name,
            gauge_height_ft, discharge_cfs.
        """
        sites_str = ",".join(site_ids)
        param_codes = f"{_PARAM_GAUGE_HEIGHT},{_PARAM_DISCHARGE}"

        params: dict[str, str] = {
            "format": "json",
            "sites": sites_str,
            "parameterCd": param_codes,
        }

        if period is not None:
            params["period"] = period
        elif start_date is not None and end_date is not None:
            params["startDT"] = start_date
            params["endDT"] = end_date

        response = self._fetch_json(_USGS_IV_URL, params=params)
        return self._parse_usgs_response(response)

    def _parse_usgs_response(self, response: dict[str, Any]) -> pd.DataFrame:
        """Parse the nested USGS Water Services JSON response.

        The USGS instantaneous values API returns a deeply nested JSON
        structure::

            response.value.timeSeries[].values[].value[]

        Each value entry contains ``dateTime`` and ``value`` fields.

        Args:
            response: Parsed JSON dict from the USGS API.

        Returns:
            DataFrame with columns: datetime, site_id, site_name,
            gauge_height_ft, discharge_cfs.
        """
        try:
            time_series_list = response["value"]["timeSeries"]
        except (KeyError, TypeError):
            logger.warning("Unexpected USGS response structure.")
            return self._empty_dataframe()

        if not time_series_list:
            return self._empty_dataframe()

        # Build a dict keyed by (site_id, datetime) to merge gauge
        # height and discharge into the same row.
        merged: dict[tuple[str, str], dict[str, Any]] = {}

        for ts in time_series_list:
            try:
                source_info = ts["sourceInfo"]
                site_id = source_info["siteCode"][0]["value"]
                site_name = source_info.get("siteName", "")

                variable = ts["variable"]
                param_code = variable["variableCode"][0]["value"]

                values_list = ts["values"]
            except (KeyError, IndexError, TypeError):
                logger.debug("Skipping malformed time series entry.")
                continue

            for values_group in values_list:
                for val_entry in values_group.get("value", []):
                    dt_str = val_entry.get("dateTime", "")
                    raw_value = val_entry.get("value")

                    if not dt_str or raw_value is None:
                        continue

                    try:
                        numeric_value = float(raw_value)
                    except (ValueError, TypeError):
                        continue

                    # USGS uses -999999 as a sentinel for missing data
                    if numeric_value <= -999999:
                        continue

                    key = (site_id, dt_str)
                    if key not in merged:
                        merged[key] = {
                            "datetime": dt_str,
                            "site_id": site_id,
                            "site_name": site_name,
                            "gauge_height_ft": np.nan,
                            "discharge_cfs": np.nan,
                        }

                    if param_code == _PARAM_GAUGE_HEIGHT:
                        merged[key]["gauge_height_ft"] = numeric_value
                    elif param_code == _PARAM_DISCHARGE:
                        merged[key]["discharge_cfs"] = numeric_value

        if not merged:
            return self._empty_dataframe()

        df = pd.DataFrame(list(merged.values()))

        # Parse datetime strings and coerce numeric columns
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        for col in ("gauge_height_ft", "discharge_cfs"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.sort_values("datetime").reset_index(drop=True)
        return df

    def _fetch_fema_declarations(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch FEMA flood disaster declarations for a date range.

        Args:
            start_date: Start date in ISO format (``YYYY-MM-DD``).
            end_date: End date in ISO format (``YYYY-MM-DD``).

        Returns:
            DataFrame with columns: datetime, declaration_id,
            state, title, incident_type.
        """
        params: dict[str, str] = {
            "$filter": (
                f"incidentType eq 'Flood' and "
                f"declarationDate ge '{start_date}T00:00:00.000Z' and "
                f"declarationDate le '{end_date}T23:59:59.999Z'"
            ),
            "$orderby": "declarationDate asc",
        }

        try:
            response = self._fetch_json(_FEMA_DECLARATIONS_URL, params=params)
        except ConnectionError:
            logger.warning("FEMA API unavailable; returning empty DataFrame.")
            return pd.DataFrame(
                columns=[
                    "datetime",
                    "declaration_id",
                    "state",
                    "title",
                    "incident_type",
                ]
            )

        records = response.get("DisasterDeclarationsSummaries", [])
        if not records:
            return pd.DataFrame(
                columns=[
                    "datetime",
                    "declaration_id",
                    "state",
                    "title",
                    "incident_type",
                ]
            )

        rows: list[dict[str, Any]] = []
        for rec in records:
            rows.append(
                {
                    "datetime": rec.get("declarationDate", ""),
                    "declaration_id": rec.get("disasterNumber", rec.get("id", "")),
                    "state": rec.get("state", ""),
                    "title": rec.get("declarationTitle", ""),
                    "incident_type": rec.get("incidentType", "Flood"),
                }
            )

        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        return df

    @staticmethod
    def _empty_dataframe() -> pd.DataFrame:
        """Return an empty DataFrame with the standard flood schema.

        Returns:
            Empty DataFrame with columns: datetime, site_id,
            site_name, gauge_height_ft, discharge_cfs.
        """
        return pd.DataFrame(
            columns=[
                "datetime",
                "site_id",
                "site_name",
                "gauge_height_ft",
                "discharge_cfs",
            ]
        )

    # ------------------------------------------------------------------
    # Private helpers — feature engineering
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_rate_of_rise(
        df: pd.DataFrame,
        gauge_height: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Compute first derivative of gauge height in ft/hr.

        Uses finite differences between consecutive timestamps. When
        ``datetime`` column is available, the actual time delta is used;
        otherwise a default 15-minute interval is assumed (typical USGS
        instantaneous value reporting frequency).

        Args:
            df: Source DataFrame (used for datetime column).
            gauge_height: 1-D array of gauge height values in feet.

        Returns:
            1-D array of rate-of-rise values in ft/hr.
        """
        n = len(gauge_height)
        rate = np.zeros(n, dtype=np.float64)

        if n < 2:
            return rate

        if "datetime" in df.columns:
            times = pd.to_datetime(df["datetime"], errors="coerce")
            for i in range(1, n):
                dt_diff = times.iloc[i] - times.iloc[i - 1]
                hours = dt_diff.total_seconds() / 3600.0
                if hours > 0 and np.isfinite(gauge_height[i]) and np.isfinite(gauge_height[i - 1]):
                    rate[i] = (gauge_height[i] - gauge_height[i - 1]) / hours
        else:
            # Assume 15-minute intervals (0.25 hours)
            interval_hr = 0.25
            for i in range(1, n):
                if np.isfinite(gauge_height[i]) and np.isfinite(gauge_height[i - 1]):
                    rate[i] = (gauge_height[i] - gauge_height[i - 1]) / interval_hr

        return rate

    @staticmethod
    def _compute_median_deviation(
        gauge_height: np.ndarray[Any, Any],
        window: int = 96,
    ) -> np.ndarray[Any, Any]:
        """Compute deviation from rolling median, normalized by median.

        The normalized deviation is defined as::

            (gauge_height - rolling_median) / rolling_median

        This highlights how far the current reading deviates from the
        recent typical water level, making it scale-invariant across
        gauges with different baseline levels.

        Args:
            gauge_height: 1-D array of gauge height values.
            window: Number of trailing observations for the rolling
                median (default 96 = ~24 hours at 15-min intervals).

        Returns:
            1-D array of normalized deviations.
        """
        n = len(gauge_height)
        deviation = np.zeros(n, dtype=np.float64)

        for i in range(n):
            start = max(0, i - window + 1)
            win = gauge_height[start : i + 1]
            valid = win[np.isfinite(win)]

            if len(valid) == 0:
                continue

            median_val = np.median(valid)
            if median_val > 0 and np.isfinite(gauge_height[i]):
                deviation[i] = (gauge_height[i] - median_val) / median_val

        return deviation

    @staticmethod
    def _compute_peak_to_baseline(
        gauge_height: np.ndarray[Any, Any],
        window: int = 288,
    ) -> np.ndarray[Any, Any]:
        """Compute ratio of current gauge height to trailing baseline.

        The baseline is defined as the minimum gauge height observed
        in the trailing window.  A ratio of 1.0 means the current
        level equals the recent minimum; higher values indicate a
        rising water level relative to the recent floor.

        Args:
            gauge_height: 1-D array of gauge height values.
            window: Number of trailing observations for baseline
                calculation (default 288 = ~72 hours at 15-min
                intervals).

        Returns:
            1-D array of peak-to-baseline ratios.
        """
        n = len(gauge_height)
        ratio = np.ones(n, dtype=np.float64)

        for i in range(n):
            start = max(0, i - window + 1)
            win = gauge_height[start : i + 1]
            valid = win[np.isfinite(win)]

            if len(valid) == 0:
                continue

            baseline = np.min(valid)
            if baseline > 0 and np.isfinite(gauge_height[i]):
                ratio[i] = gauge_height[i] / baseline

        return ratio
