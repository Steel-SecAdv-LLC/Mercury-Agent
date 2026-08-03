# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
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

from omni_mercury_engine.loaders.base import BaseDomainLoader

logger = logging.getLogger(__name__)

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
            ConnectionError: If the FRED API is unreachable after retries.
        """
        self._require_api_key()
        cache_key = "financial_realtime"
        cached = self._read_cache(cache_key)

        if cached is not None:
            logger.debug("Returning cached real-time financial data.")
            df = pd.DataFrame(cached)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            return df

        # Use a 90-day lookback for real-time context
        end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=90)).strftime("%Y-%m-%d")

        df = self._fetch_merged_series(start_date, end_date)

        self._write_cache(cache_key, df.to_dict(orient="list"))
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
            ConnectionError: If the FRED API is unreachable after retries.
        """
        self._require_api_key()
        if event_id not in _EVENT_CATALOG:
            raise ValueError(
                f"Unknown event_id '{event_id}'. " f"Available: {list(_EVENT_CATALOG.keys())}"
            )

        cache_key = f"financial_historical_{event_id}"
        cached = self._read_cache(cache_key)

        if cached is not None:
            logger.debug("Returning cached historical data for '%s'.", event_id)
            df = pd.DataFrame(cached)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            return df

        event = _EVENT_CATALOG[event_id]
        df = self._fetch_merged_series(event["start"], event["end"])

        if df.empty:
            logger.warning("FRED returned no observations for event '%s'.", event_id)
            return df

        self._write_cache(cache_key, df.to_dict(orient="list"))
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
        """
        if event_id not in _EVENT_CATALOG:
            raise ValueError(
                f"Unknown event_id '{event_id}'. " f"Available: {list(_EVENT_CATALOG.keys())}"
            )

        df = self.fetch_historical(event_id)
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
        """
        merged: pd.DataFrame | None = None

        for series_id in _SERIES_IDS:
            try:
                series_df = self._fetch_fred_series(series_id, start_date, end_date)
            except ConnectionError:
                logger.error("Failed to fetch FRED series '%s'. Skipping.", series_id)
                continue

            if series_df.empty:
                continue

            if merged is None:
                merged = series_df
            else:
                merged = merged.merge(series_df, on="date", how="outer")

        if merged is None or merged.empty:
            logger.warning("No FRED data available for range %s to %s.", start_date, end_date)
            columns = ["date"] + list(_SERIES_IDS.values())
            return pd.DataFrame(columns=columns)

        # Sort chronologically and fill gaps
        merged = merged.sort_values("date").reset_index(drop=True)

        # Ensure all expected columns exist
        for col_name in _SERIES_IDS.values():
            if col_name not in merged.columns:
                merged[col_name] = np.nan

        # Forward-fill then back-fill to handle staggered publication dates
        value_cols = list(_SERIES_IDS.values())
        merged[value_cols] = merged[value_cols].ffill().bfill()

        # Rename VIX column for cleaner downstream usage
        if "VIX" in merged.columns:
            merged = merged.rename(columns={"VIX": "vix"})

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
