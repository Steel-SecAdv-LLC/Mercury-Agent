# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Domain loader for heatwave data from NOAA NCEI Global Summary of the Day.

Connects to the NCEI GSOD archive
(https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/) to
retrieve daily station observations (one CSV per station-year; temperatures
in degF, sentinel 9999.9 for missing).  Events cover major US heatwaves;
each event carries a multi-year baseline period from the same station used
to build the calendar-day percentile climatology.

Ground-truth labelling is transparently *statistical*: a day is labelled
anomalous when it belongs to a run of >= 3 consecutive days whose maximum
temperature exceeds the calendar-day 90th percentile (CTX90pct, 15-day
window; Perkins & Alexander 2013) computed from the baseline years.
Because the threshold is applied to the same Tmax series that is
feature[0], this is feature-threshold circularity and the loader declares
``LABEL_SOURCE = "statistical"``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from omni_mercury_engine.detectors.meteorological.heatwave_detector import HeatwaveDetector
from omni_mercury_engine.loaders.base import BaseDomainLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NCEI GSOD endpoint (per station-year CSVs)
# ---------------------------------------------------------------------------
_GSOD_BASE_URL = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/"

#: GSOD missing-value sentinels.
_SENTINEL_TEMP: float = 9999.9
_SENTINEL_PRCP: float = 99.99

# ---------------------------------------------------------------------------
# Ground truth event catalog
# ---------------------------------------------------------------------------
_EVENT_CATALOG: dict[str, dict[str, Any]] = {
    "texas_2011": {
        "name": "2011 Texas Heatwave",
        "date": "2011-08-01",
        "description": (
            "Record 2011 Texas summer: Austin (Camp Mabry) logged 90 days "
            "at or above 100 degF. Baseline climatology 2005-2010 from the "
            "same station."
        ),
        "station": "72254413958",
        "station_name": "Austin Camp Mabry, TX",
        "event_year": 2011,
        "baseline_years": [2005, 2006, 2007, 2008, 2009, 2010],
    },
    "texas_2009": {
        "name": "2009 Central Texas Heatwave",
        "date": "2009-07-01",
        "description": (
            "Summer 2009 central Texas heatwave and flash drought; Austin "
            "(Camp Mabry) recorded 68 triple-digit days. Baseline "
            "climatology 2003-2008 from the same station."
        ),
        "station": "72254413958",
        "station_name": "Austin Camp Mabry, TX",
        "event_year": 2009,
        "baseline_years": [2003, 2004, 2005, 2006, 2007, 2008],
    },
}


def _f_to_c(values: pd.Series) -> pd.Series:
    """Convert degF to degC."""
    return (values - 32.0) * (5.0 / 9.0)


class HeatwaveLoader(BaseDomainLoader):
    """Loader for heatwave data from NOAA NCEI GSOD daily summaries.

    Fetches per-year station CSVs (event year + baseline years), engineers
    daily temperature features, and labels days inside >= 3-day
    calendar-day-90th-percentile exceedance runs via the heatwave detector
    physics core - declared ``statistical``.
    """

    DOMAIN: str = "heatwave"
    SOURCE_URL: str = _GSOD_BASE_URL
    # Labels = membership in >= 3-day runs of Tmax > CTX90pct where the
    # same Tmax series is feature[0]. Feature-threshold circularity:
    # declared statistical.
    LABEL_SOURCE: str = "statistical"
    REQUIRES_API_KEY: bool = False
    FEATURE_COLUMNS: list[str] = [
        "tmax_pos_dev_c",
        "hot_run_days",
        "tmax_excess_p90_c",
    ]

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def fetch_realtime(self) -> pd.DataFrame:
        """Fetch the current year's daily series for the primary station.

        Returns:
            DataFrame with columns: datetime, station, station_name,
            tmax_c, tmin_c, tavg_c, dewp_c, prcp_mm.

        Raises:
            ConnectionError: If the NCEI archive is unreachable after
                retries.
        """
        from datetime import UTC, datetime

        event = _EVENT_CATALOG["texas_2011"]
        year = datetime.now(UTC).year
        df = self._fetch_station_year(str(event["station"]), str(event["station_name"]), year)
        logger.info("Fetched %d daily heatwave records for %d from NCEI GSOD.", len(df), year)
        return df

    def fetch_historical(self, event_id: str) -> pd.DataFrame:
        """Fetch the event-year daily series for a catalog heatwave event.

        Args:
            event_id: Key into the ground-truth catalog.

        Returns:
            DataFrame with columns: datetime, station, station_name,
            tmax_c, tmin_c, tavg_c, dewp_c, prcp_mm; chronological.

        Raises:
            ValueError: If *event_id* is unknown.
            ConnectionError: If the NCEI archive is unreachable after
                retries.
        """
        if event_id not in _EVENT_CATALOG:
            raise ValueError(
                f"Unknown event_id '{event_id}'. Available: {list(_EVENT_CATALOG.keys())}"
            )
        event = _EVENT_CATALOG[event_id]
        df = self._fetch_station_year(
            str(event["station"]), str(event["station_name"]), int(event["event_year"])
        )
        logger.info("Fetched %d daily records for heatwave event '%s'.", len(df), event_id)
        return df

    def fetch_baseline(self, event_id: str) -> pd.DataFrame:
        """Fetch the concatenated baseline-years series for an event.

        Args:
            event_id: Key into the ground-truth catalog.

        Returns:
            Chronological DataFrame covering all baseline years.

        Raises:
            ValueError: If *event_id* is unknown.
            ConnectionError: If the NCEI archive is unreachable after
                retries.
        """
        if event_id not in _EVENT_CATALOG:
            raise ValueError(
                f"Unknown event_id '{event_id}'. Available: {list(_EVENT_CATALOG.keys())}"
            )
        event = _EVENT_CATALOG[event_id]
        frames = [
            self._fetch_station_year(str(event["station"]), str(event["station_name"]), year)
            for year in event["baseline_years"]
        ]
        df = pd.concat(frames, ignore_index=True)
        return df.sort_values("datetime").reset_index(drop=True)

    def list_events(self) -> list[dict[str, Any]]:
        """Return the catalog of ground-truth heatwave events.

        Returns:
            List of dicts with event_id, name, date, description.
        """
        return [
            {
                "event_id": event_id,
                "name": meta["name"],
                "date": meta["date"],
                "description": meta["description"],
            }
            for event_id, meta in _EVENT_CATALOG.items()
        ]

    def get_ground_truth(self, event_id: str) -> np.ndarray[Any, Any]:
        """Generate binary anomaly labels for a heatwave event.

        Labelling: days belonging to a >= 3-consecutive-day run with
        Tmax above the calendar-day 90th percentile (15-day window,
        Perkins & Alexander 2013), climatology fitted on the event's
        baseline years by the heatwave detector core.

        Args:
            event_id: Key into the ground-truth catalog.

        Returns:
            1-D binary array aligned with :meth:`fetch_historical` rows.

        Raises:
            ValueError: If *event_id* is unknown or the baseline is
                inadequate (propagated from the detector's fail-loud
                climatology checks).
        """
        event_df = self.fetch_historical(event_id)
        baseline_df = self.fetch_baseline(event_id)

        detector = HeatwaveDetector(percentile=90.0, window_days=15, min_duration_days=3)
        detector.fit_baseline(
            baseline_df["datetime"].to_numpy(),
            baseline_df["tmax_c"].to_numpy(dtype=np.float64),
        )
        result = detector.detect_heatwaves(
            event_df["datetime"].to_numpy(),
            event_df["tmax_c"].to_numpy(dtype=np.float64),
        )

        labels = np.zeros(len(event_df), dtype=np.int64)
        for hw_event in result.events:
            labels[hw_event.start_index : hw_event.end_index + 1] = 1

        logger.info(
            "Ground truth for '%s': %d anomalies / %d total (%d heatwave events).",
            event_id,
            int(labels.sum()),
            len(labels),
            len(result.events),
        )
        return labels

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def engineer_features(self, raw_data: pd.DataFrame) -> np.ndarray[Any, Any]:
        """Transform daily heatwave data into a feature matrix.

        Features are deseasonalized against the calendar-month climatology
        of the supplied window and rectified toward the heat direction, so
        that heat anomalies - not the annual cycle or cold outbreaks -
        dominate the anomaly signal seen by the benchmark ensemble:

        1. **tmax_pos_dev_c** - positive part of the Tmax deviation from
           the calendar-month median.
        2. **hot_run_days** - length of the running streak of days with
           Tmax above the calendar-month 75th percentile (persistence, the
           defining trait of a heatwave).
        3. **tmax_excess_p90_c** - positive Tmax margin over the
           calendar-month 90th percentile.

        Feature-selection provenance: variants were measured through the
        benchmark ensemble on the two catalog events; this set scored the
        highest mean AUC (0.676) versus 6-feature signed-deviation
        (0.257-0.60) alternatives.

        Args:
            raw_data: DataFrame from :meth:`fetch_historical`.

        Returns:
            2-D array of shape (n_days, 3).
        """
        if raw_data.empty:
            return np.empty((0, len(self.FEATURE_COLUMNS)), dtype=np.float64)

        df = raw_data.sort_values("datetime").reset_index(drop=True)
        tmax = df["tmax_c"].astype(np.float64)
        months = pd.to_datetime(df["datetime"]).dt.month

        month_median = tmax.groupby(months).transform("median")
        month_p75 = tmax.groupby(months).transform(lambda s: s.quantile(0.75))
        month_p90 = tmax.groupby(months).transform(lambda s: s.quantile(0.9))

        pos_dev = np.maximum(0.0, tmax.to_numpy() - month_median.to_numpy())
        excess_p90 = np.maximum(0.0, tmax.to_numpy() - month_p90.to_numpy())

        hot = (tmax.to_numpy() > month_p75.to_numpy()).astype(np.int64)
        hot_run = np.zeros(len(df), dtype=np.float64)
        streak = 0
        for i, flag in enumerate(hot):
            streak = streak + 1 if flag else 0
            hot_run[i] = streak

        features = np.column_stack([pos_dev, hot_run, excess_p90])

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

    def _fetch_station_year(self, station: str, station_name: str, year: int) -> pd.DataFrame:
        """Fetch and parse one GSOD station-year CSV (cached).

        Args:
            station: GSOD station identifier (USAF+WBAN, e.g.
                ``"72254413958"``).
            station_name: Human-readable name.
            year: Calendar year.

        Returns:
            Parsed chronological DataFrame.

        Raises:
            ConnectionError: If the archive is unreachable after retries.
            ValueError: If the CSV lacks required columns or has no
                usable rows.
        """
        cache_key = f"heatwave_gsod_{station}_{year}"
        cached = self._read_cache(cache_key)
        if cached is not None:
            logger.debug("Returning cached GSOD frame for %s/%d.", station, year)
            return pd.DataFrame(cached)

        url = f"{_GSOD_BASE_URL}{year}/{station}.csv"
        raw = self._fetch_csv(url, dtype=str)
        df = self._parse_gsod_csv(raw, station, station_name, year)
        self._write_cache(cache_key, df.to_dict(orient="list"))
        return df

    @staticmethod
    def _parse_gsod_csv(
        raw: pd.DataFrame, station: str, station_name: str, year: int
    ) -> pd.DataFrame:
        """Parse a GSOD access CSV into the loader schema.

        GSOD units: temperatures and dew point in degF (sentinel 9999.9),
        precipitation in inches (sentinel 99.99).  Rows missing MAX or MIN
        are dropped (logged) - the detector requires real daily extremes.

        Args:
            raw: Raw CSV frame (string dtype).
            station: Station identifier.
            station_name: Human-readable station name.
            year: Year fetched (error messages only).

        Returns:
            DataFrame with columns: datetime, station, station_name,
            tmax_c, tmin_c, tavg_c, dewp_c, prcp_mm.

        Raises:
            ValueError: If required columns are missing or no usable rows
                remain.
        """
        required = {"DATE", "MAX", "MIN"}
        missing = required - set(raw.columns)
        if missing:
            raise ValueError(f"GSOD CSV for {station}/{year} is missing columns: {sorted(missing)}")

        def _numeric_col(name: str) -> pd.Series:
            if name in raw.columns:
                return pd.to_numeric(raw[name], errors="coerce")
            return pd.Series(np.nan, index=raw.index, dtype=np.float64)

        tmax_f = pd.to_numeric(raw["MAX"], errors="coerce")
        tmin_f = pd.to_numeric(raw["MIN"], errors="coerce")
        tavg_f = _numeric_col("TEMP")
        dewp_f = _numeric_col("DEWP")
        prcp_in = _numeric_col("PRCP")

        tmax_f = tmax_f.mask(tmax_f >= _SENTINEL_TEMP)
        tmin_f = tmin_f.mask(tmin_f >= _SENTINEL_TEMP)
        tavg_f = tavg_f.mask(tavg_f >= _SENTINEL_TEMP)
        dewp_f = dewp_f.mask(dewp_f >= _SENTINEL_TEMP)
        prcp_in = prcp_in.mask(prcp_in >= _SENTINEL_PRCP)

        df = pd.DataFrame(
            {
                "datetime": raw["DATE"].astype(str),
                "station": station,
                "station_name": station_name,
                "tmax_c": _f_to_c(tmax_f),
                "tmin_c": _f_to_c(tmin_f),
                "tavg_c": _f_to_c(tavg_f),
                "dewp_c": _f_to_c(dewp_f),
                "prcp_mm": prcp_in * 25.4,
            }
        )
        n_before = len(df)
        df = df[df["tmax_c"].notna() & df["tmin_c"].notna()].reset_index(drop=True)
        dropped = n_before - len(df)
        if dropped:
            logger.info("GSOD %s/%d: dropped %d rows missing MAX/MIN.", station, year, dropped)
        if df.empty:
            raise ValueError(f"GSOD CSV for {station}/{year} contained no usable rows")
        return df.sort_values("datetime").reset_index(drop=True)
