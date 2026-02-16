"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Domain loader for pandemic/outbreak data from WHO and Our World in Data.

Connects to the Our World in Data COVID-19 dataset and the WHO Global
Health Observatory (GHO) OData API to provide epidemiological time-series
data for Mercury anomaly detection.  Ground truth events cover major
pandemic waves where rapid acceleration of case counts is labeled
as anomalous against a background of normal transmission dynamics.
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
# Data source URLs
# ---------------------------------------------------------------------------
_OWID_CSV_URL = (
    "https://raw.githubusercontent.com/owid/covid-19-data/"
    "master/public/data/owid-covid-data.csv"
)

_WHO_GHO_API_URL = "https://ghoapi.azureedge.net/api/"

_JHU_CSSE_BASE_URL = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/"
    "master/csse_covid_19_data/csse_covid_19_time_series/"
)

# ---------------------------------------------------------------------------
# OWID columns of interest
# ---------------------------------------------------------------------------
_OWID_COLUMNS = [
    "date",
    "location",
    "new_cases",
    "new_deaths",
    "total_cases",
    "total_deaths",
    "new_cases_per_million",
    "new_deaths_per_million",
    "new_cases_smoothed",
    "reproduction_rate",
    "new_tests_per_thousand",
    "positive_rate",
    "stringency_index",
]

# ---------------------------------------------------------------------------
# Ground truth event catalog
# ---------------------------------------------------------------------------
_EVENT_CATALOG: dict[str, dict[str, Any]] = {
    "covid_usa_wave1": {
        "name": "COVID-19 Multi-Wave USA",
        "date": "2020-03-15",
        "description": (
            "COVID-19 in the United States (Jan 2020 - Dec 2022). "
            "Covers multiple waves: initial outbreak, winter 2020-21, "
            "Delta summer 2021, and Omicron winter 2021-22. Daily "
            "granularity provides ~1000 observations."
        ),
        "location": "United States",
        "start": "2020-01-22",
        "end": "2022-12-31",
        "source": "owid",
    },
    "covid_italy_wave1": {
        "name": "COVID-19 Multi-Wave Italy",
        "date": "2020-02-21",
        "description": (
            "COVID-19 in Italy (Feb 2020 - Dec 2021). Covers the "
            "devastating first wave, second wave, and Alpha/Delta "
            "variant waves."
        ),
        "location": "Italy",
        "start": "2020-02-15",
        "end": "2021-12-31",
        "source": "owid",
    },
    "covid_india_delta": {
        "name": "COVID-19 Multi-Wave India",
        "date": "2021-04-15",
        "description": (
            "COVID-19 in India (Mar 2020 - Dec 2022). Covers the "
            "initial wave, devastating Delta variant wave (Apr-Jun "
            "2021), and subsequent Omicron waves."
        ),
        "location": "India",
        "start": "2020-03-01",
        "end": "2022-12-31",
        "source": "owid",
    },
    "ebola_2014": {
        "name": "2014 West Africa Ebola Outbreak",
        "date": "2014-03-23",
        "description": (
            "2014 West Africa Ebola outbreak (limited data via WHO GHO). "
            "The largest Ebola epidemic in history, primarily affecting "
            "Guinea, Liberia, and Sierra Leone."
        ),
        "location": "West Africa",
        "start": "2014-03-01",
        "end": "2014-12-31",
        "source": "who_gho",
    },
    "mpox_2022": {
        "name": "2022 Mpox Outbreak",
        "date": "2022-05-06",
        "description": (
            "2022 global Mpox (monkeypox) outbreak. Multi-country outbreak "
            "outside endemic regions with sustained community transmission."
        ),
        "location": "World",
        "start": "2022-05-01",
        "end": "2022-12-31",
        "source": "owid",
    },
}

# Number of days used as the reference baseline for anomaly labeling
_BASELINE_WINDOW_DAYS = 30


class PandemicLoader(BaseDomainLoader):
    """Loader for pandemic and outbreak data from OWID and WHO.

    Uses two primary data sources:

    * **Our World in Data (OWID)** -- comprehensive COVID-19 and Mpox
      datasets updated daily, providing case counts, death counts,
      testing rates, positivity rates, and policy stringency indices.
    * **WHO Global Health Observatory (GHO)** -- OData API for non-COVID
      disease data including historical Ebola surveillance.

    Feature engineering produces epidemiological indicators suitable
    for anomaly detection: daily case/death counts, rolling averages,
    growth rates, reproduction numbers, testing metrics, and policy
    stringency indices.

    The OWID CSV (~60MB) is cached locally after first download to
    avoid repeated large transfers.
    """

    DOMAIN: str = "pandemic"
    SOURCE_URL: str = "https://github.com/owid/covid-19-data"
    REQUIRES_API_KEY: bool = False
    FEATURE_COLUMNS: list[str] = [
        "new_cases_smoothed",
        "new_deaths_smoothed",
        "new_cases_per_million",
        "new_deaths_per_million",
        "rolling_avg_7d",
        "case_growth_rate",
        "reproduction_rate",
        "testing_rate",
        "positivity_rate",
        "stringency_index",
        "case_acceleration",
        "death_acceleration",
    ]

    #: Cache the large OWID dataset for 6 hours
    CACHE_TTL: int = 21600

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def fetch_realtime(self) -> pd.DataFrame:
        """Fetch the most recent pandemic data from Our World in Data.

        Downloads the full OWID COVID-19 CSV, caches it locally, and
        returns the last 30 days of global data.

        Returns:
            DataFrame with epidemiological columns including date,
            location, new_cases, new_deaths, and derived metrics.

        Raises:
            ConnectionError: If the OWID data source is unreachable
                after retries.
        """
        df = self._load_owid_data()

        # Return last 30 days of data across all locations
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        max_date = df["date"].max()
        cutoff = max_date - pd.Timedelta(days=30)
        recent = df[df["date"] >= cutoff].copy()
        recent = recent.sort_values(["location", "date"]).reset_index(drop=True)

        logger.info(
            "Fetched %d recent pandemic records (%s to %s).",
            len(recent),
            cutoff.strftime("%Y-%m-%d") if pd.notna(cutoff) else "N/A",
            max_date.strftime("%Y-%m-%d") if pd.notna(max_date) else "N/A",
        )
        return recent

    def fetch_historical(self, event_id: str) -> pd.DataFrame:
        """Fetch data for a specific historical pandemic event.

        Args:
            event_id: Key into the ground truth catalog (e.g.
                ``"covid_usa_wave1"``).

        Returns:
            DataFrame with epidemiological time-series data for the
            specified event, filtered by location and date range.

        Raises:
            ValueError: If *event_id* is not in the catalog.
            ConnectionError: If the data source is unreachable.
        """
        if event_id not in _EVENT_CATALOG:
            raise ValueError(
                f"Unknown event_id '{event_id}'. "
                f"Available: {list(_EVENT_CATALOG.keys())}"
            )

        cache_key = f"pandemic_historical_{event_id}"
        cached = self._read_cache(cache_key)

        if cached is not None:
            logger.debug("Returning cached historical data for '%s'.", event_id)
            return pd.DataFrame(cached)

        event = _EVENT_CATALOG[event_id]
        source = event["source"]

        if source == "owid":
            df = self._fetch_owid_event(event)
        elif source == "who_gho":
            df = self._fetch_who_gho_event(event)
        else:
            raise ValueError(
                f"Unknown data source '{source}' for event '{event_id}'."
            )

        if df.empty:
            logger.warning(
                "No data returned for event '%s'.", event_id
            )
            return df

        # Sort chronologically
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date").reset_index(drop=True)

        self._write_cache(cache_key, df.to_dict(orient="list"))
        logger.info(
            "Fetched %d historical records for event '%s'.", len(df), event_id
        )
        return df

    def list_events(self) -> list[dict[str, Any]]:
        """Return the catalog of ground truth pandemic events.

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
        """Generate binary anomaly labels for a historical pandemic event.

        Labeling strategy: a day is labeled *anomalous* (``1``) when the
        7-day rolling average of new cases exceeds twice the mean of
        new cases over the preceding 30 days (the prior month).  This
        captures periods of **wave acceleration** -- rapid growth that
        deviates significantly from baseline transmission dynamics.
        Normal growth periods are labeled ``0``.

        Args:
            event_id: Key into the ground truth catalog.

        Returns:
            1-D binary numpy array of shape ``(n_days,)``.

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

        # Compute 7-day rolling average of new cases
        new_cases = df["new_cases"].fillna(0).values.astype(np.float64)
        rolling_7d = self._rolling_mean(new_cases, window=7)

        # Compute rolling baseline: mean of prior 30 days
        n = len(new_cases)
        labels = np.zeros(n, dtype=np.int64)

        for i in range(n):
            # Build baseline from the 30 days preceding day i
            baseline_start = max(0, i - _BASELINE_WINDOW_DAYS)
            baseline_window = new_cases[baseline_start:i]

            if len(baseline_window) == 0:
                # Not enough history; cannot label as anomalous
                continue

            baseline_mean = np.mean(baseline_window)

            # Avoid division-by-zero: if baseline is near zero, any
            # non-trivial spike is anomalous
            if baseline_mean < 1.0:
                if rolling_7d[i] > 10.0:
                    labels[i] = 1
            elif rolling_7d[i] > 2.0 * baseline_mean:
                labels[i] = 1

        logger.info(
            "Ground truth for '%s': %d anomalies / %d total "
            "(wave acceleration threshold: 2x prior-month mean).",
            event_id,
            int(labels.sum()),
            len(labels),
        )
        return labels

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def engineer_features(self, raw_data: pd.DataFrame) -> np.ndarray:
        """Transform raw pandemic data into a feature matrix.

        Engineered features (per day):

        1. **new_cases_smoothed** -- 7-day smoothed new cases.
        2. **new_deaths_smoothed** -- 7-day smoothed new deaths.
        3. **new_cases_per_million** -- new cases normalized by
           population.
        4. **new_deaths_per_million** -- new deaths normalized by
           population.
        5. **rolling_avg_7d** -- 7-day rolling average of new cases.
        6. **case_growth_rate** -- ratio of today's new cases to
           new cases 7 days ago.
        7. **reproduction_rate** -- effective reproduction number (Rt)
           from OWID.
        8. **testing_rate** -- new tests per thousand population.
        9. **positivity_rate** -- test positivity rate.
        10. **stringency_index** -- government response stringency
            index (0-100).
        11. **case_acceleration** -- 7d avg / 30d trailing avg of
            new cases (captures wave onsets).
        12. **death_acceleration** -- same ratio for deaths.

        Args:
            raw_data: DataFrame from :meth:`fetch_historical` or
                :meth:`fetch_realtime`.

        Returns:
            2-D numpy array of shape ``(n_samples, 12)``.
        """
        if raw_data.empty:
            return np.empty((0, 12), dtype=np.float64)

        df = raw_data.copy()

        # Ensure chronological order
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date").reset_index(drop=True)

        # ---- base columns (fill missing with 0) ----
        new_cases = df["new_cases"].fillna(0).values.astype(np.float64)
        new_deaths = df["new_deaths"].fillna(0).values.astype(np.float64)

        # ---- smoothed columns (7-day rolling) ----
        new_cases_smoothed = self._rolling_mean(new_cases, window=7)
        new_deaths_smoothed = self._rolling_mean(new_deaths, window=7)

        # ---- per-million rates ----
        new_cases_per_million = (
            df["new_cases_per_million"].fillna(0).values.astype(np.float64)
            if "new_cases_per_million" in df.columns
            else np.zeros(len(df), dtype=np.float64)
        )
        new_deaths_per_million = (
            df["new_deaths_per_million"].fillna(0).values.astype(np.float64)
            if "new_deaths_per_million" in df.columns
            else np.zeros(len(df), dtype=np.float64)
        )

        # ---- 7-day rolling average of new cases ----
        rolling_avg_7d = new_cases_smoothed  # reuse

        # ---- case growth rate (new_cases / new_cases_7_days_ago) ----
        case_growth_rate = self._compute_growth_rate(new_cases, lag=7)

        # ---- reproduction rate (Rt) from OWID ----
        reproduction_rate = (
            df["reproduction_rate"].fillna(0).values.astype(np.float64)
            if "reproduction_rate" in df.columns
            else np.zeros(len(df), dtype=np.float64)
        )

        # ---- testing rate ----
        testing_rate = (
            df["new_tests_per_thousand"].fillna(0).values.astype(np.float64)
            if "new_tests_per_thousand" in df.columns
            else np.zeros(len(df), dtype=np.float64)
        )

        # ---- positivity rate ----
        positivity_rate = (
            df["positive_rate"].fillna(0).values.astype(np.float64)
            if "positive_rate" in df.columns
            else np.zeros(len(df), dtype=np.float64)
        )

        # ---- stringency index ----
        stringency_index = (
            df["stringency_index"].fillna(0).values.astype(np.float64)
            if "stringency_index" in df.columns
            else np.zeros(len(df), dtype=np.float64)
        )

        # ---- case acceleration: 7d avg / 30d trailing avg ----
        rolling_30d = self._rolling_mean(new_cases, window=30)
        case_acceleration = np.where(
            rolling_30d > 1.0,
            new_cases_smoothed / rolling_30d,
            0.0,
        )

        # ---- death acceleration ----
        deaths_30d = self._rolling_mean(new_deaths, window=30)
        death_acceleration = np.where(
            deaths_30d > 0.1,
            new_deaths_smoothed / deaths_30d,
            0.0,
        )

        # Stack into feature matrix
        features = np.column_stack(
            [
                new_cases_smoothed,
                new_deaths_smoothed,
                new_cases_per_million,
                new_deaths_per_million,
                rolling_avg_7d,
                case_growth_rate,
                reproduction_rate,
                testing_rate,
                positivity_rate,
                stringency_index,
                case_acceleration,
                death_acceleration,
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
    # Private helpers -- data loading
    # ------------------------------------------------------------------

    def _load_owid_data(self) -> pd.DataFrame:
        """Load the full OWID COVID-19 dataset, using cache when available.

        The OWID CSV is ~60MB.  This method caches the raw CSV as a
        Parquet file in the loader's cache directory to avoid repeated
        large downloads.

        Returns:
            DataFrame with all OWID COVID-19 columns.

        Raises:
            ConnectionError: If the OWID CSV cannot be downloaded.
        """
        parquet_path = self.cache_dir / "owid_covid_data.parquet"

        # Check if a fresh cached parquet exists
        if parquet_path.exists():
            import time as _time

            age_seconds = _time.time() - parquet_path.stat().st_mtime
            if age_seconds < self.CACHE_TTL:
                logger.debug("Loading cached OWID data from %s.", parquet_path)
                return pd.read_parquet(parquet_path)

        logger.info("Downloading OWID COVID-19 CSV (~60MB). This may take a moment.")
        raw_bytes = self._fetch_url(_OWID_CSV_URL)
        df = pd.read_csv(io.BytesIO(raw_bytes), low_memory=False)

        # Cache as parquet for faster subsequent loads
        try:
            df.to_parquet(parquet_path, index=False)
            logger.debug("Cached OWID data to %s.", parquet_path)
        except Exception as exc:
            logger.debug("Failed to cache OWID parquet: %s", exc)

        return df

    def _fetch_owid_event(self, event: dict[str, Any]) -> pd.DataFrame:
        """Fetch OWID data filtered by location and date range.

        Args:
            event: Event metadata dict from ``_EVENT_CATALOG``.

        Returns:
            DataFrame filtered to the specified location and date range.
        """
        df = self._load_owid_data()

        # Filter by location
        location = event["location"]
        if location != "World":
            df = df[df["location"] == location].copy()

        # Filter by date range
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        start = pd.Timestamp(event["start"])
        end = pd.Timestamp(event["end"])
        df = df[(df["date"] >= start) & (df["date"] <= end)].copy()

        # Select columns of interest (keep only those that exist)
        available_cols = [c for c in _OWID_COLUMNS if c in df.columns]
        df = df[available_cols].reset_index(drop=True)

        return df

    def _fetch_who_gho_event(self, event: dict[str, Any]) -> pd.DataFrame:
        """Fetch disease data from the WHO GHO OData API.

        The WHO GHO API provides aggregated surveillance data for
        various diseases.  For the 2014 Ebola event, we query the
        relevant indicator and construct a time-series.

        Args:
            event: Event metadata dict from ``_EVENT_CATALOG``.

        Returns:
            DataFrame with date and case count columns.  Falls back
            to a synthetic time-series derived from WHO situation
            reports if the API does not return granular data.
        """
        # Attempt to fetch Ebola case data from WHO GHO API
        try:
            indicator = "EBOLA_CASESDEATHS"
            url = f"{_WHO_GHO_API_URL}{indicator}"
            data = self._fetch_json(url)

            records = data.get("value", [])
            if not records:
                logger.warning(
                    "WHO GHO returned no records for indicator '%s'. "
                    "Falling back to synthetic Ebola data.",
                    indicator,
                )
                return self._synthetic_ebola_2014()

            # Parse WHO GHO records into a DataFrame
            rows: list[dict[str, Any]] = []
            for rec in records:
                rows.append(
                    {
                        "date": rec.get("TimeDim", rec.get("YEAR", "")),
                        "location": rec.get(
                            "SpatialDim",
                            rec.get("COUNTRY", ""),
                        ),
                        "new_cases": rec.get("NumericValue", 0),
                        "new_deaths": 0,
                        "total_cases": 0,
                        "total_deaths": 0,
                    }
                )

            df = pd.DataFrame(rows)

            # Filter to relevant time period
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            start = pd.Timestamp(event["start"])
            end = pd.Timestamp(event["end"])
            df = df[(df["date"] >= start) & (df["date"] <= end)].copy()

            if df.empty:
                logger.warning(
                    "WHO GHO data empty after date filtering. "
                    "Falling back to synthetic Ebola data."
                )
                return self._synthetic_ebola_2014()

            df = df.sort_values("date").reset_index(drop=True)
            return df

        except (ConnectionError, KeyError, TypeError) as exc:
            logger.warning(
                "Failed to fetch WHO GHO data: %s. "
                "Falling back to synthetic Ebola data.",
                exc,
            )
            return self._synthetic_ebola_2014()

    @staticmethod
    def _synthetic_ebola_2014() -> pd.DataFrame:
        """Generate a synthetic Ebola 2014 time-series from WHO reports.

        The WHO GHO API does not always provide daily-granularity Ebola
        data.  This method produces a synthetic daily time-series based
        on published WHO situation report totals for the 2014 West Africa
        Ebola outbreak.

        Returns:
            DataFrame with date, location, new_cases, new_deaths,
            total_cases, and total_deaths columns.
        """
        # Monthly cumulative case estimates from WHO situation reports
        monthly_totals = [
            ("2014-03-31", 130),
            ("2014-04-30", 242),
            ("2014-05-31", 309),
            ("2014-06-30", 759),
            ("2014-07-31", 1440),
            ("2014-08-31", 3707),
            ("2014-09-30", 7178),
            ("2014-10-31", 13567),
            ("2014-11-30", 17145),
            ("2014-12-31", 20206),
        ]

        rows: list[dict[str, Any]] = []
        prev_total = 0

        for date_str, cumulative in monthly_totals:
            end_date = pd.Timestamp(date_str)
            # Distribute new cases evenly across the month's days
            month_new = cumulative - prev_total
            start_date = end_date.replace(day=1)
            days_in_month = (end_date - start_date).days + 1
            daily_new = month_new / days_in_month

            for day_offset in range(days_in_month):
                current_date = start_date + pd.Timedelta(days=day_offset)
                running_total = prev_total + daily_new * (day_offset + 1)
                rows.append(
                    {
                        "date": current_date,
                        "location": "West Africa",
                        "new_cases": round(daily_new, 1),
                        "new_deaths": round(daily_new * 0.5, 1),
                        "total_cases": round(running_total),
                        "total_deaths": round(running_total * 0.5),
                        "new_cases_per_million": 0.0,
                        "new_deaths_per_million": 0.0,
                    }
                )
            prev_total = cumulative

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Private helpers -- computation
    # ------------------------------------------------------------------

    @staticmethod
    def _rolling_mean(values: np.ndarray, window: int = 7) -> np.ndarray:
        """Compute a trailing rolling mean over a 1-D array.

        Args:
            values: 1-D array of numeric values.
            window: Number of preceding observations to average.

        Returns:
            1-D array of the same length with rolling mean values.
            The first ``window - 1`` entries use a shorter window
            (expanding mean).
        """
        n = len(values)
        result = np.zeros(n, dtype=np.float64)

        for i in range(n):
            start = max(0, i - window + 1)
            result[i] = np.mean(values[start : i + 1])

        return result

    @staticmethod
    def _compute_growth_rate(
        new_cases: np.ndarray,
        lag: int = 7,
    ) -> np.ndarray:
        """Compute case growth rate as ratio of current to lagged values.

        The growth rate is defined as:
            ``new_cases[i] / new_cases[i - lag]``

        A value > 1 indicates accelerating case growth; < 1 indicates
        deceleration.

        Args:
            new_cases: 1-D array of daily new case counts.
            lag: Number of days to look back for the denominator.

        Returns:
            1-D array of growth rates.  Entries where the lagged value
            is zero or where insufficient history exists are set to 0.
        """
        n = len(new_cases)
        growth_rate = np.zeros(n, dtype=np.float64)

        for i in range(lag, n):
            denominator = new_cases[i - lag]
            if denominator > 0:
                growth_rate[i] = new_cases[i] / denominator
            else:
                # Cannot compute meaningful ratio; leave as 0
                growth_rate[i] = 0.0

        return growth_rate
