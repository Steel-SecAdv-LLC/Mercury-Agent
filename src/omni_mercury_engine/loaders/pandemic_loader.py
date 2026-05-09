"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

Domain loader for pandemic/outbreak data from WHO and Our World in Data.

Connects to the Our World in Data COVID-19 dataset and the WHO Global Health Observatory (GHO) OData
API to provide epidemiological time-series data for Mercury anomaly detection.  Supports six
pathogen classes: virus, bacteria, fungus, parasite, prion, and biosurveillance.

Ground truth events cover major pandemic waves where rapid acceleration of case counts is labeled as
anomalous against a background of normal transmission dynamics.
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
    "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
)

_WHO_GHO_BASE_URL = "https://ghoapi.azureedge.net/api"

_WHO_GHO_API_URL = "https://ghoapi.azureedge.net/api/"

_WHO_EMERGENCIES_URL = "https://www.who.int/api/hubs/emergencies"

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
    # --- VIRUS (pathogen_class=virus) ---
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
        "pathogen_class": "virus",
        "pathogen": "SARS-CoV-2",
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
        "pathogen_class": "virus",
        "pathogen": "SARS-CoV-2",
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
        "pathogen_class": "virus",
        "pathogen": "SARS-CoV-2",
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
        "pathogen_class": "virus",
        "pathogen": "Ebolavirus",
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
        "pathogen_class": "virus",
        "pathogen": "Monkeypox virus (MPXV)",
    },
    # --- BACTERIA (pathogen_class=bacteria) ---
    "cholera_yemen_2017": {
        "name": "2017 Yemen Cholera Outbreak",
        "date": "2017-04-27",
        "description": (
            "2017 Yemen cholera — largest outbreak in modern history. "
            "Over 1 million suspected cases reported. Annual data from "
            "WHO GHO indicator WHS3_41."
        ),
        "location": "Yemen",
        "country": "YEM",
        "start": "2010-01-01",
        "end": "2023-12-31",
        "source": "who_gho",
        "gho_indicator": "WHS3_41",
        "pathogen_class": "bacteria",
        "pathogen": "Vibrio cholerae",
        "_verified": "2026-02-16",
    },
    # --- FUNGUS (pathogen_class=fungus) ---
    "candida_auris_us_tracking": {
        "name": "Candida auris US Emergence",
        "date": "2016-06-01",
        "description": (
            "C. auris emergence and spread in US healthcare facilities. "
            "CDC C. auris data requires manual download; no free "
            "auth-free API confirmed."
        ),
        "location": "United States",
        "start": "2016-01-01",
        "end": "2024-12-31",
        "source": "stub",
        "pathogen_class": "fungus",
        "pathogen": "Candida auris",
        "_stub_reason": (
            "CDC C. auris data requires manual download. Placeholder for future integration."
        ),
    },
    # --- PARASITE (pathogen_class=parasite) ---
    "malaria_subsaharan_2019_2022": {
        "name": "Sub-Saharan Africa Malaria Surveillance",
        "date": "2019-01-01",
        "description": (
            "Sub-Saharan Africa malaria — seasonal surge tracking. "
            "WHO GHO indicator MALARIA001 returns empty; marked as stub "
            "pending data source confirmation."
        ),
        "location": "Nigeria",
        "country": "NGA",
        "start": "2019-01-01",
        "end": "2022-12-31",
        "source": "stub",
        "gho_indicator": "MALARIA001",
        "pathogen_class": "parasite",
        "pathogen": "Plasmodium falciparum",
        "_stub_reason": (
            "WHO GHO MALARIA001 indicator returns empty for NGA. "
            "Placeholder pending alternative data source."
        ),
    },
    # --- PRION (pathogen_class=prion) ---
    "cjd_us_surveillance": {
        "name": "US CJD Surveillance",
        "date": "2000-01-01",
        "description": (
            "US CJD surveillance — annual data only. CDC CJD data is "
            "published as annual PDF reports, not API. Insufficient "
            "density for real-time anomaly detection."
        ),
        "location": "United States",
        "start": "2000-01-01",
        "end": "2023-12-31",
        "source": "stub",
        "pathogen_class": "prion",
        "pathogen": "Prion (CJD)",
        "_warning": "Annual data only. Insufficient density for real-time detection.",
        "_stub_reason": ("CDC CJD data is published as annual PDF reports, not API."),
    },
    # --- BIOSURVEILLANCE (pathogen_class=biosurveillance) ---
    "who_emergencies_2020_2024": {
        "name": "WHO Emergencies Alert Clustering 2020-2024",
        "date": "2020-01-01",
        "description": (
            "WHO health emergencies alert frequency — novel pathogen "
            "emergence tracking. Uses the WHO Emergencies Hub API to "
            "construct an alert frequency time series for syndromic "
            "surveillance."
        ),
        "location": "Global",
        "start": "2020-01-01",
        "end": "2024-12-31",
        "source": "who_emergencies",
        "pathogen_class": "biosurveillance",
        "pathogen": "Multi-pathogen (syndromic)",
        "_verified": "2026-02-16",
    },
}

# Number of days used as the reference baseline for anomaly labeling
_BASELINE_WINDOW_DAYS = 30


class PandemicLoader(BaseDomainLoader):
    """
    Loader for pandemic and outbreak data from OWID and WHO.

    Supports six pathogen classes: virus, bacteria, fungus, parasite,
    prion, and biosurveillance.

    Uses three primary data sources:

    * **Our World in Data (OWID)** -- comprehensive COVID-19 and Mpox
      datasets updated daily, providing case counts, death counts,
      testing rates, positivity rates, and policy stringency indices.
    * **WHO Global Health Observatory (GHO)** -- OData API for non-COVID
      disease data including cholera, Ebola, and malaria surveillance.
    * **WHO Emergencies Hub** -- API for health emergency alerts,
      used for biosurveillance alert frequency analysis.

    Some pathogen classes (fungus, parasite, prion) are marked as stubs
    pending free, auth-free API availability.  Stub events return
    ``None`` from :meth:`fetch_historical` and are skipped by the
    benchmark harness.

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
        """
        Fetch the most recent pandemic data from Our World in Data.

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
        """
        Fetch data for a specific historical pandemic event.

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
                f"Unknown event_id '{event_id}'. Available: {list(_EVENT_CATALOG.keys())}"
            )

        cache_key = f"pandemic_historical_{event_id}"
        cached = self._read_cache(cache_key)

        if cached is not None:
            logger.debug("Returning cached historical data for '%s'.", event_id)
            return pd.DataFrame(cached)

        event = _EVENT_CATALOG[event_id]
        source = event["source"]

        if source == "stub":
            logger.warning(
                "Data source for '%s' is a stub — no live API available. Reason: %s",
                event_id,
                event.get("_stub_reason", "unknown"),
            )
            return pd.DataFrame()

        if source == "owid":
            df = self._fetch_owid_event(event)
        elif source == "who_gho":
            if "gho_indicator" in event:
                df = self._fetch_who_gho(
                    indicator=event["gho_indicator"],
                    country=event.get("country", ""),
                    start_date=event["start"],
                    end_date=event["end"],
                )
            else:
                df = self._fetch_who_gho_event(event)
        elif source == "who_emergencies":
            df = self._fetch_who_emergencies(
                start_date=event["start"],
                end_date=event["end"],
            )
        else:
            raise ValueError(f"Unknown data source '{source}' for event '{event_id}'.")

        if df.empty:
            logger.warning("No data returned for event '%s'.", event_id)
            return df

        # Sort chronologically
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date").reset_index(drop=True)

        self._write_cache(cache_key, df.to_dict(orient="list"))
        logger.info("Fetched %d historical records for event '%s'.", len(df), event_id)
        return df

    def list_events(self) -> list[dict[str, Any]]:
        """
        Return the catalog of ground truth pandemic events.

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

    def list_pathogen_classes(self) -> list[str]:
        """
        Return distinct pathogen classes available in the event catalog.

        Returns:
            Sorted list of unique pathogen class strings.
        """
        return sorted({e.get("pathogen_class", "unknown") for e in _EVENT_CATALOG.values()})

    def get_ground_truth(self, event_id: str) -> np.ndarray:
        """
        Generate binary anomaly labels for a historical pandemic event.

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
                f"Unknown event_id '{event_id}'. Available: {list(_EVENT_CATALOG.keys())}"
            )

        event = _EVENT_CATALOG[event_id]

        # Stub events have no data
        if event.get("source") == "stub":
            logger.warning("Ground truth unavailable for stub event '%s'.", event_id)
            return np.array([], dtype=np.int64)

        df = self.fetch_historical(event_id)
        if df.empty:
            return np.array([], dtype=np.int64)

        n_samples = len(df)

        # Density warning for sparse data
        if n_samples < 100:
            logger.warning(
                "%s | N=%d | Insufficient density for reliable anomaly detection",
                event_id,
                n_samples,
            )

        # Prion-specific warning
        if event.get("_warning"):
            logger.warning("PRION | N=%d | %s", n_samples, event["_warning"])

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
        """
        Transform raw pandemic data into a feature matrix.

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
    # Private helpers -- new data sources
    # ------------------------------------------------------------------

    def _fetch_who_gho(
        self,
        indicator: str,
        country: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetch from WHO Global Health Observatory OData API.

        Endpoint: ``{_WHO_GHO_BASE_URL}/{indicator}``
        Filter: ``$filter=SpatialDim eq '{country}'``
        No API key required.

        The GHO API returns annual data.  Each year's value is
        distributed evenly across 12 monthly records to provide
        enough temporal density for anomaly detection.

        Args:
            indicator: WHO GHO indicator code (e.g., ``'WHS3_41'``
                for cholera).
            country: ISO 3166-1 alpha-3 country code.
            start_date: Start date string (``YYYY-MM-DD``).
            end_date: End date string (``YYYY-MM-DD``).

        Returns:
            DataFrame with columns: date, location, new_cases,
            new_deaths, total_cases, total_deaths.
            Empty DataFrame on failure.
        """
        url = f"{_WHO_GHO_BASE_URL}/{indicator}"

        params: dict[str, str] = {}
        if country:
            params["$filter"] = f"SpatialDim eq '{country}'"

        try:
            data = self._fetch_json(url, params=params)
        except (ConnectionError, ValueError) as exc:
            logger.warning(
                "Failed to fetch WHO GHO indicator %s: %s",
                indicator,
                exc,
            )
            return pd.DataFrame()

        records = data.get("value", [])
        if not records:
            logger.warning(
                "WHO GHO returned no records for indicator '%s' country='%s'.",
                indicator,
                country,
            )
            return pd.DataFrame()

        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        start_year = start_ts.year
        end_year = end_ts.year

        # Parse annual records and distribute across months
        rows: list[dict[str, Any]] = []
        for rec in records:
            year = rec.get("TimeDim")
            if year is None:
                continue
            year = int(year)
            if year < start_year or year > end_year:
                continue

            annual_value = float(rec.get("NumericValue", 0) or 0)
            monthly_value = annual_value / 12.0

            for month in range(1, 13):
                row_date = pd.Timestamp(year=year, month=month, day=15)
                if row_date < start_ts or row_date > end_ts:
                    continue
                rows.append(
                    {
                        "date": row_date,
                        "location": rec.get("SpatialDim", country),
                        "new_cases": monthly_value,
                        "new_deaths": 0.0,
                        "total_cases": 0.0,
                        "total_deaths": 0.0,
                        "new_cases_per_million": 0.0,
                        "new_deaths_per_million": 0.0,
                    }
                )

        if not rows:
            logger.warning(
                "WHO GHO: no data in date range %s to %s for %s/%s.",
                start_date,
                end_date,
                indicator,
                country,
            )
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.sort_values("date").reset_index(drop=True)

        # Compute running total_cases
        df["total_cases"] = df["new_cases"].cumsum()

        logger.info(
            "WHO GHO: fetched %d records for %s/%s (%s to %s).",
            len(df),
            indicator,
            country,
            start_date,
            end_date,
        )
        return df

    def _fetch_who_emergencies(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetch WHO health emergency data for biosurveillance.

        Uses the WHO Emergencies Hub API to retrieve health emergency
        declarations.  Constructs a monthly time series of alert
        frequency metrics: alert_count, unique_diseases, and
        alert_acceleration.

        Args:
            start_date: Start date string (``YYYY-MM-DD``).
            end_date: End date string (``YYYY-MM-DD``).

        Returns:
            DataFrame with biosurveillance features.
            Empty DataFrame on failure.
        """
        try:
            data = self._fetch_json(_WHO_EMERGENCIES_URL)
        except (ConnectionError, ValueError) as exc:
            logger.warning("Failed to fetch WHO Emergencies: %s", exc)
            return pd.DataFrame()

        emergencies = data.get("value", [])
        if not emergencies:
            logger.warning("WHO Emergencies API returned no data.")
            return pd.DataFrame()

        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        # Parse emergency start dates
        alert_dates: list[pd.Timestamp] = []
        alert_titles: list[str] = []
        for em in emergencies:
            date_str = em.get("EmergencyStartDate")
            if not date_str:
                continue
            try:
                ts = pd.Timestamp(date_str)
                # Normalize to tz-naive for comparison
                if ts.tzinfo is not None:
                    ts = ts.tz_localize(None)
            except (ValueError, TypeError):
                continue
            if start_ts <= ts <= end_ts:
                alert_dates.append(ts)
                alert_titles.append(em.get("Title", ""))

        if not alert_dates:
            logger.warning(
                "WHO Emergencies: no alerts in range %s to %s.",
                start_date,
                end_date,
            )
            return pd.DataFrame()

        # Build a monthly time series of alert counts
        date_range = pd.date_range(start=start_ts, end=end_ts, freq="MS")
        rows: list[dict[str, Any]] = []

        for month_start in date_range:
            month_end = month_start + pd.offsets.MonthEnd(1)
            count = sum(1 for d in alert_dates if month_start <= d <= month_end)
            unique_diseases = len(
                {t for t, d in zip(alert_titles, alert_dates) if month_start <= d <= month_end}
            )
            rows.append(
                {
                    "date": month_start,
                    "location": "Global",
                    "new_cases": float(count),
                    "new_deaths": 0.0,
                    "total_cases": 0.0,
                    "total_deaths": 0.0,
                    "new_cases_per_million": 0.0,
                    "new_deaths_per_million": 0.0,
                    "alert_count": float(count),
                    "unique_diseases": float(unique_diseases),
                }
            )

        df = pd.DataFrame(rows)
        df["total_cases"] = df["new_cases"].cumsum()

        logger.info(
            "WHO Emergencies: constructed %d-month biosurveillance time series (%d alerts total).",
            len(df),
            len(alert_dates),
        )
        return df

    # ------------------------------------------------------------------
    # Private helpers -- data loading
    # ------------------------------------------------------------------

    def _load_owid_data(self) -> pd.DataFrame:
        """
        Load the full OWID COVID-19 dataset, using cache when available.

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
        """
        Fetch OWID data filtered by location and date range.

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
        """
        Fetch disease data from the WHO GHO OData API.

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
                    "WHO GHO data empty after date filtering. Falling back to synthetic Ebola data."
                )
                return self._synthetic_ebola_2014()

            df = df.sort_values("date").reset_index(drop=True)
            return df

        except (ConnectionError, KeyError, TypeError) as exc:
            logger.warning(
                "Failed to fetch WHO GHO data: %s. Falling back to synthetic Ebola data.",
                exc,
            )
            return self._synthetic_ebola_2014()

    @staticmethod
    def _synthetic_ebola_2014() -> pd.DataFrame:
        """
        Generate a synthetic Ebola 2014 time-series from WHO reports.

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
        """
        Compute a trailing rolling mean over a 1-D array.

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
        """
        Compute case growth rate as ratio of current to lagged values.

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
