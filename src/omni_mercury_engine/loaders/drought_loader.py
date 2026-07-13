# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Domain loader for drought data from NOAA NCEI Global Summary of the Month.

Connects to the NCEI GSOM station archive
(https://www.ncei.noaa.gov/data/gsom/access/) to retrieve full-period
monthly precipitation and temperature series for stations covering major US
droughts (2011 Texas drought, 2012-2016 California drought).  GSOM ``access``
CSVs report precipitation in millimetres and temperature in degrees Celsius.

Ground-truth labelling is transparently *statistical*: a month is labelled
anomalous when its 6-month Standardized Precipitation Index (McKee et al.
1993, computed with per-calendar-month gamma fits by the drought detector
physics core) is at or below the US Drought Monitor D2 severe-drought
threshold of -1.3 (Svoboda et al. 2002).  Because the SPI is derived from
the same monthly precipitation series that feeds the precipitation
features, this is feature-threshold circularity and the loader declares
``LABEL_SOURCE = "statistical"``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from omni_mercury_engine.detectors.meteorological.drought_detector import compute_spi
from omni_mercury_engine.loaders.base import BaseDomainLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NCEI GSOM endpoint (station period-of-record CSVs)
# ---------------------------------------------------------------------------
_GSOM_BASE_URL = "https://www.ncei.noaa.gov/data/gsom/access/"

#: SPI aggregation window (months) used for labelling.
_LABEL_SPI_WINDOW: int = 6

#: USDM D2 (severe drought) SPI threshold (Svoboda et al. 2002).
_LABEL_SPI_THRESHOLD: float = -1.3

# ---------------------------------------------------------------------------
# Ground truth event catalog
# ---------------------------------------------------------------------------
_EVENT_CATALOG: dict[str, dict[str, Any]] = {
    "texas_2011": {
        "name": "2010-2011 Texas Drought",
        "date": "2011-09-01",
        "description": (
            "Most intense one-year drought on record for Texas; Austin "
            "(Camp Mabry) recorded its driest 12 months. Series 1950-2015 "
            "provides the SPI climatology."
        ),
        "station": "USW00013958",
        "station_name": "Austin Camp Mabry, TX",
        "start": "1950-01",
        "end": "2015-12",
    },
    "california_2012_2016": {
        "name": "2012-2016 California Drought",
        "date": "2014-01-01",
        "description": (
            "Multi-year California drought with record-low Sierra snowpack "
            "and Central Valley precipitation deficits; Fresno Yosemite "
            "International station. Series 1950-2016 provides the SPI "
            "climatology."
        ),
        "station": "USW00093193",
        "station_name": "Fresno Yosemite International, CA",
        "start": "1950-01",
        "end": "2016-12",
    },
}


class DroughtLoader(BaseDomainLoader):
    """Loader for drought data from NOAA NCEI GSOM monthly summaries.

    Fetches one period-of-record CSV per station, slices the catalog
    window, and engineers precipitation-deficit features.  Labels are
    SPI-6 <= -1.3 (USDM D2), computed by the drought detector's SPI core
    from the same precipitation series - declared ``statistical``.
    """

    DOMAIN: str = "drought"
    SOURCE_URL: str = _GSOM_BASE_URL
    # Labels = SPI-6 <= -1.3 where the SPI is computed from the same
    # monthly PRCP series that is feature[0] (and its rolling sums).
    # Feature-threshold circularity: declared statistical.
    LABEL_SOURCE: str = "statistical"
    REQUIRES_API_KEY: bool = False
    FEATURE_COLUMNS: list[str] = [
        "prcp_z_1mo",
        "prcp_z_3mo",
        "prcp_z_6mo",
        "prcp_z_12mo",
        "prcp_frac_of_normal",
        "tavg_dev_c",
    ]

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def fetch_realtime(self) -> pd.DataFrame:
        """Fetch the most recent 24 months for the primary catalog station.

        Returns:
            DataFrame with columns: datetime, station, station_name,
            prcp_mm, tavg_c, tmax_c, month.

        Raises:
            ConnectionError: If the NCEI archive is unreachable after
                retries.
        """
        event = _EVENT_CATALOG["texas_2011"]
        df = self._fetch_station_frame(str(event["station"]), str(event["station_name"]))
        df = df.tail(24).reset_index(drop=True)
        logger.info("Fetched %d recent monthly drought records from NCEI GSOM.", len(df))
        return df

    def fetch_historical(self, event_id: str) -> pd.DataFrame:
        """Fetch the monthly series window for a catalog drought event.

        Args:
            event_id: Key into the ground-truth catalog
                (e.g. ``"texas_2011"``).

        Returns:
            DataFrame with columns: datetime (YYYY-MM), station,
            station_name, prcp_mm, tavg_c, tmax_c, month; chronological.

        Raises:
            ValueError: If *event_id* is unknown, or the station series
                has gaps inside the catalog window (rolling SPI windows
                must not silently span missing months).
            ConnectionError: If the NCEI archive is unreachable after
                retries.
        """
        if event_id not in _EVENT_CATALOG:
            raise ValueError(
                f"Unknown event_id '{event_id}'. Available: {list(_EVENT_CATALOG.keys())}"
            )
        event = _EVENT_CATALOG[event_id]
        df = self._fetch_station_frame(str(event["station"]), str(event["station_name"]))

        window = df[(df["datetime"] >= event["start"]) & (df["datetime"] <= event["end"])]
        window = window.sort_values("datetime").reset_index(drop=True)
        if window.empty:
            raise ValueError(
                f"Station {event['station']} returned no rows inside "
                f"{event['start']}..{event['end']}"
            )
        self._assert_contiguous_months(window, event_id)
        logger.info("Fetched %d monthly records for drought event '%s'.", len(window), event_id)
        return window

    def list_events(self) -> list[dict[str, Any]]:
        """Return the catalog of ground-truth drought events.

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
        """Generate binary anomaly labels for a drought event.

        Labelling: SPI-6 (per-calendar-month gamma fits, McKee et al.
        1993) at or below -1.3 - the USDM D2 severe-drought threshold
        (Svoboda et al. 2002).  The first five months of the window carry
        no complete 6-month aggregation and are labelled 0.

        Args:
            event_id: Key into the ground-truth catalog.

        Returns:
            1-D binary array aligned with :meth:`fetch_historical` rows.

        Raises:
            ValueError: If *event_id* is unknown or the series is too
                short for the SPI climatology.
        """
        df = self.fetch_historical(event_id)
        precip = df["prcp_mm"].to_numpy(dtype=np.float64)
        months = df["month"].to_numpy(dtype=np.int64)

        spi6 = compute_spi(precip, _LABEL_SPI_WINDOW, months)
        labels = np.zeros(len(df), dtype=np.int64)
        labels[_LABEL_SPI_WINDOW - 1 :] = (spi6 <= _LABEL_SPI_THRESHOLD).astype(np.int64)

        logger.info(
            "Ground truth for '%s': %d anomalies / %d total (SPI-%d <= %.1f).",
            event_id,
            int(labels.sum()),
            len(labels),
            _LABEL_SPI_WINDOW,
            _LABEL_SPI_THRESHOLD,
        )
        return labels

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    @staticmethod
    def _month_z(series: pd.Series, months: pd.Series) -> np.ndarray[Any, Any]:
        """Standardize a series against its calendar-month climatology.

        Args:
            series: Values to standardize.
            months: Calendar month (1-12) per entry.

        Returns:
            Per-month z-scores (0 where the stratum is degenerate).
        """
        mean = series.groupby(months).transform("mean")
        std = series.groupby(months).transform("std")
        std_arr = std.to_numpy(dtype=np.float64)
        z = np.where(
            std_arr > 0.0,
            (series.to_numpy(dtype=np.float64) - mean.to_numpy(dtype=np.float64)) / std_arr,
            0.0,
        )
        return z

    def engineer_features(self, raw_data: pd.DataFrame) -> np.ndarray[Any, Any]:
        """Transform monthly drought data into a feature matrix.

        Features are deseasonalized (standardized against the calendar-
        month climatology of the supplied window) so drought structure -
        not the annual cycle - dominates the anomaly signal:

        1. **prcp_z_1mo** - monthly precipitation z-score.
        2. **prcp_z_3mo** - z-score of the trailing 3-month sum.
        3. **prcp_z_6mo** - z-score of the trailing 6-month sum.
        4. **prcp_z_12mo** - z-score of the trailing 12-month sum.
        5. **prcp_frac_of_normal** - precipitation as a fraction of the
           calendar-month mean.
        6. **tavg_dev_c** - temperature deviation from the calendar-month
           mean.

        Trailing sums are truncated at the series start (documented:
        ``min_periods=1``).

        Args:
            raw_data: DataFrame from :meth:`fetch_historical`.

        Returns:
            2-D array of shape (n_months, 6).
        """
        if raw_data.empty:
            return np.empty((0, len(self.FEATURE_COLUMNS)), dtype=np.float64)

        df = raw_data.sort_values("datetime").reset_index(drop=True)
        prcp = df["prcp_mm"].astype(np.float64)
        tavg = df["tavg_c"].astype(np.float64)
        months = df["month"].astype(np.int64)

        prcp_3 = prcp.rolling(3, min_periods=1).sum()
        prcp_6 = prcp.rolling(6, min_periods=1).sum()
        prcp_12 = prcp.rolling(12, min_periods=1).sum()

        monthly_normals = prcp.groupby(months).transform("mean")
        with np.errstate(divide="ignore", invalid="ignore"):
            frac_normal = np.where(
                monthly_normals.to_numpy() > 0.0,
                prcp.to_numpy() / monthly_normals.to_numpy(),
                1.0,
            )
        tavg_mean = tavg.groupby(months).transform("mean")
        tavg_dev = tavg.to_numpy() - tavg_mean.to_numpy()

        features = np.column_stack(
            [
                self._month_z(prcp, months),
                self._month_z(prcp_3, months),
                self._month_z(prcp_6, months),
                self._month_z(prcp_12, months),
                frac_normal,
                tavg_dev,
            ]
        )

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

    def _fetch_station_frame(self, station: str, station_name: str) -> pd.DataFrame:
        """Fetch and parse one GSOM period-of-record station CSV (cached).

        Args:
            station: GHCN station identifier (e.g. ``"USW00013958"``).
            station_name: Human-readable name for the frame.

        Returns:
            Parsed DataFrame (all months with a PRCP value), chronological.

        Raises:
            ConnectionError: If the archive is unreachable after retries.
            ValueError: If the CSV lacks the required columns.
        """
        cache_key = f"drought_gsom_{station}"
        cached = self._read_cache(cache_key)
        if cached is not None:
            logger.debug("Returning cached GSOM frame for %s.", station)
            return pd.DataFrame(cached)

        url = f"{_GSOM_BASE_URL}{station}.csv"
        raw = self._fetch_csv(url, dtype=str)
        df = self._parse_gsom_csv(raw, station, station_name)
        self._write_cache(cache_key, df.to_dict(orient="list"))
        return df

    @staticmethod
    def _parse_gsom_csv(raw: pd.DataFrame, station: str, station_name: str) -> pd.DataFrame:
        """Parse a GSOM access CSV into the loader schema.

        GSOM ``access`` CSVs are metric: PRCP in millimetres, TAVG/TMAX in
        degrees Celsius.  Rows without a PRCP value are dropped (logged).

        Args:
            raw: Raw CSV frame (string dtype).
            station: Station identifier.
            station_name: Human-readable station name.

        Returns:
            DataFrame with columns: datetime, station, station_name,
            prcp_mm, tavg_c, tmax_c, month.

        Raises:
            ValueError: If required columns are missing or no usable rows
                remain.
        """
        required = {"DATE", "PRCP"}
        missing = required - set(raw.columns)
        if missing:
            raise ValueError(f"GSOM CSV for {station} is missing columns: {sorted(missing)}")

        def _numeric_col(name: str) -> pd.Series:
            if name in raw.columns:
                return pd.to_numeric(raw[name], errors="coerce")
            return pd.Series(np.nan, index=raw.index, dtype=np.float64)

        prcp = pd.to_numeric(raw["PRCP"], errors="coerce")
        tavg = _numeric_col("TAVG")
        tmax = _numeric_col("TMAX")

        df = pd.DataFrame(
            {
                "datetime": raw["DATE"].astype(str),
                "station": station,
                "station_name": station_name,
                "prcp_mm": prcp,
                "tavg_c": tavg,
                "tmax_c": tmax,
            }
        )
        n_before = len(df)
        df = df[df["prcp_mm"].notna()].reset_index(drop=True)
        dropped = n_before - len(df)
        if dropped:
            logger.info("GSOM %s: dropped %d rows without PRCP.", station, dropped)
        if df.empty:
            raise ValueError(f"GSOM CSV for {station} contained no rows with PRCP values")

        df["month"] = df["datetime"].str.slice(5, 7).astype(int)
        return df.sort_values("datetime").reset_index(drop=True)

    @staticmethod
    def _assert_contiguous_months(df: pd.DataFrame, event_id: str) -> None:
        """Fail loudly when the monthly series has gaps inside the window.

        Rolling SPI aggregations assume month-over-month contiguity;
        silently spanning a gap would blend non-adjacent months into one
        "window" and quietly corrupt the index.

        Args:
            df: Window frame (sorted, ``datetime`` as YYYY-MM strings).
            event_id: Event id for the error message.

        Raises:
            ValueError: If any consecutive rows are not adjacent months.
        """
        periods = pd.PeriodIndex(df["datetime"], freq="M")
        steps = np.diff(periods.asi8)
        if np.any(steps != 1):
            n_gaps = int(np.sum(steps != 1))
            raise ValueError(
                f"drought event '{event_id}': station series has {n_gaps} "
                "month gap(s) inside the catalog window; SPI windows must "
                "not silently span missing months - adjust the catalog "
                "window or choose a complete station"
            )
