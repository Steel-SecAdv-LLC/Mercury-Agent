# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Domain loader for financial crisis data from FRED and market data.

Connects to the Federal Reserve Economic Data (FRED) API maintained by the Federal Reserve Bank of
St. Louis to retrieve key financial stress indicators: VIX volatility index, Treasury yield curve
spread, high-yield credit spreads, the federal funds rate, and the TED spread.  Ground truth events
cover major financial crises from the 1997 Asian Financial Crisis through the 2023 SVB regional bank
crisis.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from omni_mercury_engine.datasets.exceptions import DataSourceUnavailableError
from omni_mercury_engine.loaders.base import BaseDomainLoader

logger = logging.getLogger(__name__)

#: Frame attribute naming the columns whose FRED request failed at the
#: transport. Such a frame is a partial outage, not the record: the lost
#: columns are NaN because the request never completed, not because FRED
#: published nothing. It is returned (the series that did arrive are real) but
#: never cached, and :meth:`FinancialLoader.get_ground_truth` refuses to label
#: from it when a label-bearing column is among the lost ones.
INCOMPLETE_SERIES_ATTR = "mercury_incomplete_series"

#: The columns the crisis label is computed from (see ``get_ground_truth``).
_LABEL_COLUMNS: frozenset[str] = frozenset({"vix", "yield_curve_10y2y"})


def _renamed(columns: list[str]) -> list[str]:
    """Column names as the returned frame spells them (``VIX`` becomes ``vix``)."""
    return ["vix" if name == "VIX" else name for name in columns]


# ---------------------------------------------------------------------------
# FRED API configuration
# ---------------------------------------------------------------------------
_FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# Key FRED series IDs used for financial stress detection
_SERIES_IDS: dict[str, str] = {
    "VIXCLS": "VIX",
    "T10Y2Y": "yield_curve_10y2y",
    "BAMLH0A0HYM2": "high_yield_spread",
    "DFF": "fed_funds_rate",
    "SOFR": "funding_rate",
}

# ---------------------------------------------------------------------------
# Ground truth event catalog
# ---------------------------------------------------------------------------
_EVENT_CATALOG: dict[str, dict[str, Any]] = {
    "gfc_2008": {
        "name": "2008 Global Financial Crisis",
        "date": "2008-09-15",
        "description": (
            "Global financial crisis triggered by the collapse of Lehman Brothers "
            "on September 15, 2008.  Widespread credit freeze, bank failures, and "
            "equity market crash."
        ),
        "start": "2008-01-01",
        "end": "2009-06-30",
        "peak": "2008-09-15",
    },
    "covid_crash_2020": {
        "name": "2020 COVID Market Crash",
        "date": "2020-03-16",
        "description": (
            "Rapid market sell-off driven by the global COVID-19 pandemic.  "
            "VIX reached record highs above 80 in March 2020."
        ),
        "start": "2020-01-01",
        "end": "2020-06-30",
        "peak": "2020-03-16",
    },
    "svb_2023": {
        "name": "2023 SVB / Regional Bank Crisis",
        "date": "2023-03-10",
        "description": (
            "Silicon Valley Bank collapse and subsequent regional banking stress "
            "in March 2023.  Contagion fears led to elevated volatility and "
            "credit spread widening."
        ),
        "start": "2023-01-01",
        "end": "2023-06-30",
        "peak": "2023-03-10",
    },
    "asian_crisis_1997": {
        "name": "1997 Asian Financial Crisis",
        "date": "1997-07-02",
        "description": (
            "Currency and financial crisis originating in Thailand (baht devaluation) "
            "that spread across East Asia and into global markets."
        ),
        "start": "1997-01-01",
        "end": "1998-06-30",
        "peak": "1997-10-27",
    },
    "flash_crash_2010": {
        "name": "2010 Flash Crash",
        "date": "2010-05-06",
        "description": (
            "Sudden intraday market crash on May 6, 2010 where the DJIA briefly "
            "lost nearly 1,000 points before recovering within minutes."
        ),
        "start": "2010-04-01",
        "end": "2010-06-30",
        "peak": "2010-05-06",
    },
}


class FinancialLoader(BaseDomainLoader):
    """Loader for financial crisis data from the FRED API.

    Retrieves multiple FRED time-series that serve as financial stress
    indicators and aligns them into a single DataFrame indexed by date.
    Feature engineering produces derived signals suitable for anomaly
    detection: levels, rates of change, z-scores, cross-series
    correlations, and yield-curve inversion flags.

    Requires a FRED API key, available free at https://fred.stlouisfed.org/docs/api/api_key.html.
    Set the ``FRED_API_KEY`` environment variable or pass ``api_key`` to the constructor.
    """

    DOMAIN: str = "financial"
    SOURCE_URL: str = _FRED_BASE_URL
    # Labels = (VIX > 30 AND yield_curve < 0) OR (VIX > 45). VIX and yield-
    # curve are both scored features (and ``vix_zscore`` / ``yc_zscore`` too).
    # Feature-threshold circularity.
    LABEL_SOURCE: str = "statistical"
    REQUIRES_API_KEY: bool = True
    API_KEY_ENV_VAR: str = "FRED_API_KEY"
    FEATURE_COLUMNS: list[str] = [
        "vix",
        "vix_roc",
        "yield_curve",
        "yield_curve_inverted",
        "credit_spread",
        "credit_spread_roc",
        "funding_rate",
        "vix_yc_corr",
        "vix_zscore",
        "yc_zscore",
        "cs_zscore",
        "funding_rate_zscore",
    ]

    #: De-leak: labels threshold VIX and the yield curve; these are those two series.
    #: Still computed and returned by ``engineer_features``; excluded only
    #: from the scored matrix, so nothing is lost and the labels stop
    #: describing the thing being scored.
    LABEL_DERIVED_FEATURES: tuple[str, ...] = (
        "vix",
        "vix_roc",
        "vix_zscore",
        "yield_curve",
        "yield_curve_inverted",
        "yc_zscore",
        "vix_yc_corr",
    )
    LABEL_FEATURE_DISJOINT: bool = True

    def _require_api_key(self) -> None:
        """Raise EnvironmentError if the FRED API key is not configured."""
        if not self._api_key:
            raise OSError(
                "FRED_API_KEY not set. The financial domain loader requires a free "
                "FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html "
                "and set the FRED_API_KEY environment variable."
            )

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def fetch_realtime(self) -> pd.DataFrame:
        """Fetch the most recent financial stress data from FRED.

        Pulls the latest 90 days of observations for each FRED series
        and merges them on date.

        Returns:
            DataFrame indexed by date with columns for each financial
            indicator: ``vix``, ``yield_curve_10y2y``, ``high_yield_spread``,
            ``fed_funds_rate``, ``funding_rate``.

        Raises:
            EnvironmentError: If FRED_API_KEY is not set.
            DataSourceUnavailableError: If the FRED API is unreachable after
                retries (every series request failed).
        """
        self._require_api_key()
        # Cache schema v2: the merged series is now business-day only,
        # forward-filled (no bfill leakage), with funding_rate sourced from
        # DFF pre-2018 — so stored payloads differ from pre-fix entries.
        cache_key = "financial_realtime_v2"
        cached = self._read_frame_cache(cache_key)

        if cached is not None:
            logger.debug("Returning cached real-time financial data.")
            return cached

        # Use a 90-day lookback for real-time context
        end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=90)).strftime("%Y-%m-%d")

        df = self._fetch_merged_series(start_date, end_date)

        if df.empty:
            # Not cached, as in fetch_historical: an empty frame parked under
            # this key was served as "no financial data" for a full CACHE_TTL.
            logger.warning("FRED returned no observations for the last 90 days.")
            return df

        if self._cache_if_complete(cache_key, df):
            logger.info("Fetched %d real-time financial records from FRED.", len(df))
        return df

    def fetch_historical(self, event_id: str) -> pd.DataFrame:
        """Fetch financial data surrounding a known crisis event.

        Args:
            event_id: Key into the ground truth catalog (e.g.
                ``"gfc_2008"``).

        Returns:
            DataFrame with the same schema as :meth:`fetch_realtime`,
            covering the event's date range.

        Raises:
            EnvironmentError: If FRED_API_KEY is not set.
            ValueError: If *event_id* is not in the catalog.
            DataSourceUnavailableError: If the FRED API is unreachable after
                retries (every series request failed).
        """
        self._require_api_key()
        if event_id not in _EVENT_CATALOG:
            raise ValueError(
                f"Unknown event_id '{event_id}'. " f"Available: {list(_EVENT_CATALOG.keys())}"
            )

        # Cache schema v2: see fetch_realtime — merged-series definition
        # changed (business-day only, ffill-only, DFF funding-rate fallback).
        cache_key = f"financial_historical_v2_{event_id}"
        cached = self._read_frame_cache(cache_key)

        if cached is not None:
            logger.debug("Returning cached historical data for '%s'.", event_id)
            return cached

        event = _EVENT_CATALOG[event_id]
        df = self._fetch_merged_series(event["start"], event["end"])

        if df.empty:
            logger.warning("FRED returned no observations for event '%s'.", event_id)
            return df

        if self._cache_if_complete(cache_key, df):
            logger.info("Fetched %d historical records for event '%s'.", len(df), event_id)
        return df

    def list_events(self) -> list[dict[str, Any]]:
        """Return the catalog of ground truth financial crisis events.

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
        """Generate binary anomaly labels for a historical financial crisis.

        Labeling strategy: a trading day is labeled *anomalous* (``1``) if:

        - VIX > 30 **AND** yield curve is inverted (T10Y2Y < 0), **OR**
        - VIX > 45 (extreme volatility regardless of yield curve).

        All other days are labeled *normal* (``0``).

        Args:
            event_id: Key into the ground truth catalog.

        Returns:
            1-D binary numpy array of shape ``(n_trading_days,)``.

        Raises:
            ValueError: If *event_id* is not recognized or no data is
                available.
            DataSourceUnavailableError: If the VIX or yield-curve request
                failed at the transport. Their NaN days would otherwise be
                labelled normal, turning a crisis into an all-zero label
                vector because of a network fault.
        """
        if event_id not in _EVENT_CATALOG:
            raise ValueError(
                f"Unknown event_id '{event_id}'. " f"Available: {list(_EVENT_CATALOG.keys())}"
            )

        df = self.fetch_historical(event_id)
        lost_label_columns = sorted(_LABEL_COLUMNS & set(df.attrs.get(INCOMPLETE_SERIES_ATTR, ())))
        if lost_label_columns:
            raise DataSourceUnavailableError(
                "FinancialLoader",
                _FRED_BASE_URL,
                f"cannot label event '{event_id}': the FRED request for "
                f"{', '.join(lost_label_columns)} failed, and the label is computed from it",
            )
        if df.empty:
            return np.array([], dtype=np.int64)

        vix = df["vix"].values.astype(np.float64)
        yield_curve = df["yield_curve_10y2y"].values.astype(np.float64)

        # Anomaly: (VIX > 30 AND inverted yield curve) OR VIX > 45
        stress_and_inverted = (vix > 30.0) & (yield_curve < 0.0)
        extreme_vol = vix > 45.0
        labels = (stress_and_inverted | extreme_vol).astype(np.int64)

        # Handle NaN values conservatively: NaN days are labeled normal
        nan_mask = np.isnan(vix) | np.isnan(yield_curve)
        labels[nan_mask] = 0

        logger.info(
            "Ground truth for '%s': %d anomalies / %d total trading days.",
            event_id,
            int(labels.sum()),
            len(labels),
        )
        return np.asarray(labels)

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def engineer_features(self, raw_data: pd.DataFrame) -> np.ndarray[Any, Any]:
        """Transform raw financial data into a feature matrix.

        Engineered features (per trading day):

        1.  **vix** -- VIX level.
        2.  **vix_roc** -- VIX rate of change (first difference).
        3.  **yield_curve** -- T10Y2Y level.
        4.  **yield_curve_inverted** -- Binary flag: 1 if T10Y2Y < 0.
        5.  **credit_spread** -- High-yield spread (BAMLH0A0HYM2) level.
        6.  **credit_spread_roc** -- Credit spread rate of widening.
        7.  **funding_rate** -- Overnight secured funding rate (SOFR).
        8.  **vix_yc_corr** -- Rolling 20-day correlation between VIX
            and yield curve.
        9.  **vix_zscore** -- Z-score of VIX vs. 252-day lookback.
        10. **yc_zscore** -- Z-score of yield curve vs. 252-day lookback.
        11. **cs_zscore** -- Z-score of credit spread vs. 252-day lookback.
        12. **funding_rate_zscore** -- Z-score of the funding rate vs. a
            252-day lookback.

        Args:
            raw_data: DataFrame from :meth:`fetch_realtime` or
                :meth:`fetch_historical`.

        Returns:
            2-D numpy array of shape ``(n_samples, 12)``.
        """
        if raw_data.empty:
            return np.empty((0, 12), dtype=np.float64)

        df = raw_data.copy()

        # Ensure chronological order
        if "date" in df.columns:
            df = df.sort_values("date").reset_index(drop=True)

        # ---- Base levels ----
        vix = df["vix"].values.astype(np.float64)
        yield_curve = df["yield_curve_10y2y"].values.astype(np.float64)
        credit_spread = df["high_yield_spread"].values.astype(np.float64)
        funding_rate = df["funding_rate"].values.astype(np.float64)

        # ---- Rate of change (first difference) ----
        vix_roc = self._compute_rate_of_change(vix)
        credit_spread_roc = self._compute_rate_of_change(credit_spread)

        # ---- Yield curve inversion indicator ----
        yield_curve_inverted = np.where(
            np.isnan(yield_curve), np.nan, (yield_curve < 0.0).astype(np.float64)
        )

        # ---- Rolling 20-day correlation between VIX and yield curve ----
        vix_yc_corr = self._compute_rolling_correlation(vix, yield_curve, window=20)

        # ---- Z-scores (252-day / 1-year lookback) ----
        vix_zscore = self._compute_rolling_zscore(vix, window=252)
        yc_zscore = self._compute_rolling_zscore(yield_curve, window=252)
        cs_zscore = self._compute_rolling_zscore(credit_spread, window=252)
        funding_rate_zscore = self._compute_rolling_zscore(funding_rate, window=252)

        # Stack into feature matrix
        features = np.column_stack(
            [
                vix,
                vix_roc,
                yield_curve,
                yield_curve_inverted,
                credit_spread,
                credit_spread_roc,
                funding_rate,
                vix_yc_corr,
                vix_zscore,
                yc_zscore,
                cs_zscore,
                funding_rate_zscore,
            ]
        )

        return self._impute_gaps(features)

    @staticmethod
    def _impute_gaps(features: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Fill non-finite cells without reaching further into the future than needed.

        A gap takes the last value observed BEFORE it. Only a gap at the very
        start of a column has no past to draw on; it takes the first observed
        value, the nearest reading in time. That value is later than the rows
        it fills -- rows cannot be dropped here without breaking alignment
        with the labels -- so it is the smallest look-ahead available, and it
        is stated rather than hidden: the whole-window median this replaced
        drew on every later observation in the event. A column with no finite
        value at all becomes 0.0.
        """
        features = np.where(np.isinf(features), np.nan, features).astype(float, copy=True)
        for col_idx in range(features.shape[1]):
            col = features[:, col_idx]
            finite = np.flatnonzero(np.isfinite(col))
            if finite.size == col.size:
                continue
            if finite.size == 0:
                col[:] = 0.0
                continue
            # Index of the most recent finite value at or before each row.
            last_seen = np.maximum.accumulate(np.where(np.isfinite(col), np.arange(col.size), -1))
            col[:] = np.where(last_seen >= 0, col[np.maximum(last_seen, 0)], col[finite[0]])
        return features

    # ------------------------------------------------------------------
    # Private helpers — FRED API
    # ------------------------------------------------------------------

    def _fetch_fred_series(
        self,
        series_id: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch a single FRED series as a DataFrame.

        Args:
            series_id: FRED series identifier (e.g. ``"VIXCLS"``).
            start_date: Start date in ``YYYY-MM-DD`` format.
            end_date: End date in ``YYYY-MM-DD`` format.

        Returns:
            DataFrame with columns ``date`` and the series column name
            from :data:`_SERIES_IDS`.

        Raises:
            ConnectionError: If the FRED API is unreachable after retries.
        """
        col_name = _SERIES_IDS.get(series_id, series_id.lower())

        params: dict[str, str] = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date,
        }

        data = self._fetch_json(_FRED_BASE_URL, params=params)

        observations = data.get("observations", [])
        if not observations:
            logger.warning(
                "FRED returned no observations for series '%s' (%s to %s).",
                series_id,
                start_date,
                end_date,
            )
            return pd.DataFrame(columns=["date", col_name])

        rows: list[dict[str, Any]] = []
        for obs in observations:
            date_str = obs.get("date", "")
            value_str = obs.get("value", ".")

            # FRED uses "." for missing data
            if value_str == "." or value_str == "":
                value = np.nan
            else:
                try:
                    value = float(value_str)
                except (ValueError, TypeError):
                    value = np.nan

            rows.append({"date": date_str, col_name: value})

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        return df

    def _cache_if_complete(self, cache_key: str, frame: pd.DataFrame) -> bool:
        """Cache ``frame`` unless a series was lost at the transport; True when cached.

        A partial frame parked under the event's key was served as the record
        for a full ``CACHE_TTL`` with the lost columns all NaN, long after
        FRED had recovered.
        """
        lost = frame.attrs.get(INCOMPLETE_SERIES_ATTR)
        if lost:
            logger.warning(
                "FRED partial outage: %s not retrieved; returning %d rows uncached.",
                ", ".join(lost),
                len(frame),
            )
            return False
        # The frame cache, not ``_write_cache(frame.to_dict(...))``: the
        # ``date`` column holds Timestamps, which the plain JSON writer cannot
        # serialise, so every write here failed (logged at DEBUG only) and
        # each call re-fetched all five series from FRED.
        self._write_frame_cache(cache_key, frame)
        return True

    def _fetch_merged_series(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch all FRED series and merge them on date.

        Args:
            start_date: Start date in ``YYYY-MM-DD`` format.
            end_date: End date in ``YYYY-MM-DD`` format.

        Returns:
            DataFrame indexed by date with one column per series.
            Missing values are forward-filled then back-filled to handle
            series with different publication schedules.

        Raises:
            DataSourceUnavailableError: If every series request failed at
                the transport, i.e. FRED itself is unreachable. A single
                failed series stays a logged gap, and the returned frame
                names it in ``attrs[INCOMPLETE_SERIES_ATTR]`` so that the
                callers neither cache the frame nor label from it.
        """
        merged: pd.DataFrame | None = None
        failures: list[ConnectionError] = []
        failed_columns: list[str] = []
        # Dates on which the equity-market series actually traded. DFF (the
        # fed-funds rate) publishes 7 days a week, so an outer merge otherwise
        # introduces weekend rows carrying no market data; we use these dates
        # to drop them below.
        market_dates: pd.Series | None = None

        for series_id, column in _SERIES_IDS.items():
            try:
                series_df = self._fetch_fred_series(series_id, start_date, end_date)
            except ConnectionError as exc:
                logger.error("Failed to fetch FRED series '%s'. Skipping.", series_id)
                failures.append(exc)
                failed_columns.append(column)
                continue

            if series_df.empty:
                continue

            # VIX is the equity-market series; its observation dates are the
            # trading days the merged frame should be restricted to.
            if series_id == "VIXCLS":
                market_dates = series_df["date"].copy()

            if merged is None:
                merged = series_df
            else:
                merged = merged.merge(series_df, on="date", how="outer")

        if len(failures) == len(_SERIES_IDS):
            # All five requests failing is an outage, not an empty window:
            # with the transport down this used to fall through to the empty
            # frame below, which fetch_realtime then cached for an hour. The
            # transport error's text is credential-free by construction (see
            # FetchHTTPError), and the constructor redacts regardless.
            last_error = failures[-1]
            raise DataSourceUnavailableError(
                "FinancialLoader",
                _FRED_BASE_URL,
                f"FRED unreachable: all {len(failures)} series requests failed "
                f"for {start_date} to {end_date}; last error: {last_error}",
                status_code=getattr(last_error, "status_code", None),
            )

        if merged is None or merged.empty:
            logger.warning("No FRED data available for range %s to %s.", start_date, end_date)
            columns = ["date"] + list(_SERIES_IDS.values())
            empty = pd.DataFrame(columns=columns)
            if failed_columns:
                empty.attrs[INCOMPLETE_SERIES_ATTR] = _renamed(failed_columns)
            return empty

        # Sort chronologically and fill gaps
        merged = merged.sort_values("date").reset_index(drop=True)

        # Ensure all expected columns exist
        for col_name in _SERIES_IDS.values():
            if col_name not in merged.columns:
                merged[col_name] = np.nan

        # Restrict to the trading days present in the market (VIX) series so
        # DFF's 7-day-a-week publication does not inject weekend rows that
        # merely copy Friday's market values (duplicated labels and inflated
        # "trading-day" counts). If VIX was unavailable, fall back to a
        # Monday-Friday business-day mask.
        if market_dates is not None:
            merged = merged[merged["date"].isin(set(market_dates))].reset_index(drop=True)
        else:
            merged = merged[merged["date"].dt.day_of_week < 5].reset_index(drop=True)

        value_cols = list(_SERIES_IDS.values())

        # Fill staggered publication gaps FORWARD only. Back-filling (the old
        # ``.bfill()``) copied later observations backward over the start of
        # the window, leaking future values into the leading rows. Leading
        # rows with no prior observation are left NaN here; ``_impute_gaps``
        # gives them the first observed value, the smallest look-ahead that
        # keeps the rows aligned with their labels.
        merged[value_cols] = merged[value_cols].ffill()

        # Overnight funding rate: SOFR (``funding_rate``) only begins in April
        # 2018, so for pre-2018 events (GFC 2008, Asian crisis 1997, flash
        # crash 2010) it is empty and this column would be all-NaN -> a
        # constant after imputation. DFF (``fed_funds_rate``) covers those
        # years, so use it as the funding rate wherever SOFR is missing; the
        # fed-funds rate is the relevant policy funding rate for that era and
        # keeps the column carrying real signal instead of a flat 0.
        if "funding_rate" in merged.columns and "fed_funds_rate" in merged.columns:
            merged["funding_rate"] = merged["funding_rate"].fillna(merged["fed_funds_rate"])

        # Rename VIX column for cleaner downstream usage
        if "VIX" in merged.columns:
            merged = merged.rename(columns={"VIX": "vix"})

        # Any value column still constant across the whole event (a series
        # FRED does not serve for this period, even after the DFF fallback) is
        # uninformative. The feature matrix has a fixed, de-leak-mapped shape
        # so the column cannot be physically dropped, but it must not be fed
        # silently: name it so the gap is visible in the logs.
        constant_cols: list[str] = []
        for raw_col in value_cols:
            col = "vix" if raw_col == "VIX" else raw_col
            if col not in merged.columns:
                continue
            series = merged[col]
            if series.notna().any() and int(series.nunique(dropna=True)) <= 1:
                constant_cols.append(col)
        if constant_cols:
            logger.warning(
                "FRED series constant across event (%s to %s): %s — "
                "uninformative for anomaly detection.",
                start_date,
                end_date,
                ", ".join(sorted(set(constant_cols))),
            )

        if failed_columns:
            # Set last: ``merge`` and ``rename`` build new frames, and only
            # the final one is what the callers inspect.
            merged.attrs[INCOMPLETE_SERIES_ATTR] = _renamed(failed_columns)
        return merged

    # ------------------------------------------------------------------
    # Private helpers — Feature engineering
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_rate_of_change(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute first-difference rate of change.

        Args:
            values: 1-D array of numeric values.

        Returns:
            1-D array of first differences. The first element is 0.0.
        """
        roc = np.zeros(len(values), dtype=np.float64)
        if len(values) > 1:
            roc[1:] = np.diff(values)
        return roc

    @staticmethod
    def _compute_rolling_zscore(
        values: np.ndarray[Any, Any],
        window: int = 252,
    ) -> np.ndarray[Any, Any]:
        """Compute rolling z-score relative to a trailing lookback window.

        Args:
            values: 1-D array of numeric values.
            window: Lookback period (252 trading days ~ 1 year).

        Returns:
            1-D array of z-scores. NaN where insufficient data or
            zero standard deviation.
        """
        n = len(values)
        zscore = np.full(n, np.nan, dtype=np.float64)

        for i in range(n):
            start = max(0, i - window + 1)
            win = values[start : i + 1]
            # Need at least 2 observations for a meaningful std
            valid = win[~np.isnan(win)]
            if len(valid) < 2:
                continue
            mean = np.mean(valid)
            std = np.std(valid, ddof=1)
            if std > 0:
                zscore[i] = (values[i] - mean) / std

        return zscore

    @staticmethod
    def _compute_rolling_correlation(
        series_a: np.ndarray[Any, Any],
        series_b: np.ndarray[Any, Any],
        window: int = 20,
    ) -> np.ndarray[Any, Any]:
        """Compute rolling Pearson correlation between two series.

        Args:
            series_a: First 1-D array.
            series_b: Second 1-D array.
            window: Rolling window size in observations.

        Returns:
            1-D array of correlation coefficients. NaN where insufficient
            data or constant values within the window.
        """
        n = len(series_a)
        corr = np.full(n, np.nan, dtype=np.float64)

        for i in range(n):
            start = max(0, i - window + 1)
            win_a = series_a[start : i + 1]
            win_b = series_b[start : i + 1]

            # Need at least 3 paired non-NaN observations
            valid_mask = ~(np.isnan(win_a) | np.isnan(win_b))
            if valid_mask.sum() < 3:
                continue

            a_valid = win_a[valid_mask]
            b_valid = win_b[valid_mask]

            std_a = np.std(a_valid, ddof=1)
            std_b = np.std(b_valid, ddof=1)
            if std_a == 0.0 or std_b == 0.0:
                continue

            mean_a = np.mean(a_valid)
            mean_b = np.mean(b_valid)
            cov = np.mean((a_valid - mean_a) * (b_valid - mean_b))
            corr[i] = cov / (std_a * std_b)

        return corr
