# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Domain loader for severe-hail data from the NOAA Storm Prediction Center (SPC).

Connects to the SPC severe-hail report archive (1955-2023, distributed as a
zipped CSV at ``spc.noaa.gov/wcm/data``) and the SPC filtered daily storm
reports feed.  Ground-truth events cover well-documented major hail episodes
(the 2010 Vivian SD record-hailstone supercell, the April 2016 Texas hail
outbreak, and the May 2017 Colorado Front Range hailstorm), with
*significant hail* reports (diameter >= 2.00 in, the SPC significant-hail
size) labelled as anomalies against the background of smaller severe-hail
reports.

Label provenance is declared ``"statistical"``: the label thresholds the
``mag`` (hail diameter) column, which is also engineered as feature[0] --
a direct feature-threshold circularity, declared honestly so the
governed-fusion headline excludes it (see
:mod:`omni_mercury_engine.loaders.label_provenance`).
"""

from __future__ import annotations

import csv
import logging
from typing import Any

import numpy as np
import pandas as pd

from omni_mercury_engine.loaders.base import BaseDomainLoader
from omni_mercury_engine.utils.geo import haversine_km_to_point, neighbor_counts_within_km

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SPC data endpoints
# ---------------------------------------------------------------------------
_ARCHIVE_URL = "https://www.spc.noaa.gov/wcm/data/1955-2023_hail.csv.zip"
_DAILY_REPORTS_URL = "https://www.spc.noaa.gov/climo/reports/today_filtered.csv"

#: SPC "significant hail" diameter threshold in inches (>= 2.00 in).
_SIG_HAIL_IN = 2.0

#: SPC daily-report hail sizes are reported in hundredths of an inch.
_DAILY_SIZE_SCALE = 0.01

# ---------------------------------------------------------------------------
# Ground truth event catalog (validated against the live archive: report
# counts and maximum sizes below were measured from the 1955-2023 archive).
# ---------------------------------------------------------------------------
_EVENT_CATALOG: dict[str, dict[str, Any]] = {
    "vivian_2010": {
        "name": "2010 Vivian SD Record Hailstone Supercell",
        "date": "2010-07-23",
        "description": (
            "Supercell of July 23, 2010 that dropped the largest hailstone "
            "on record in the United States (8.0 in diameter, 1.94 lb) at "
            "Vivian, South Dakota; 69 archive reports in the July 22-24 "
            "window, 4 of them significant (>= 2 in)."
        ),
        "start": "2010-07-22",
        "end": "2010-07-24",
    },
    "texas_2016": {
        "name": "April 2016 Texas Hail Outbreak",
        "date": "2016-04-12",
        "description": (
            "April 10-13, 2016 Texas hail outbreak, including the Wylie TX "
            "5.25 in hail of April 11 and the San Antonio hailstorm of "
            "April 12 (up to 4.5 in), at the time the costliest hail event "
            "in Texas history; 238 archive reports, 67 significant."
        ),
        "start": "2016-04-10",
        "end": "2016-04-13",
    },
    "colorado_2017": {
        "name": "May 2017 Colorado Front Range Hailstorm",
        "date": "2017-05-08",
        "description": (
            "May 8, 2017 Denver-metro hailstorm (up to 2.75 in), the "
            "costliest insured catastrophe in Colorado history at the "
            "time; 144 archive reports in the May 7-9 window, 15 "
            "significant."
        ),
        "start": "2017-05-07",
        "end": "2017-05-09",
    },
}

# ---------------------------------------------------------------------------
# Approximate centroid of maximum US hail frequency ("hail alley", the
# Colorado / Nebraska / Wyoming tri-state region, which averages the most
# hail days per year in the US per the NOAA/NSSL severe-weather
# climatology).  Used for the geographic-anomaly feature.
# ---------------------------------------------------------------------------
_HAIL_ALLEY_CENTROID_LAT = 41.0
_HAIL_ALLEY_CENTROID_LON = -104.0


class HailLoader(BaseDomainLoader):
    """Loader for severe-hail data from the NOAA Storm Prediction Center.

    Uses two SPC data sources:

    * **Severe-hail archive CSV** (zipped) -- the comprehensive record of
      US severe-hail reports 1955-2023 with hail diameter (``mag``, in
      inches), report location, and casualty/loss columns; same column
      schema as the SPC tornado archive.
    * **Filtered daily storm reports** -- today's preliminary reports;
      the hail section carries sizes in hundredths of an inch, which are
      normalised to inches here.

    Feature engineering produces hail-report observables suitable for
    anomaly detection: hail diameter, casualties, loss, location, temporal
    clustering rate, distance from the hail-alley climatological centroid,
    and time-of-year features.

    Unlike :class:`~omni_mercury_engine.loaders.tornado_loader.TornadoLoader`,
    the full ~396k-row archive is *not* written to the JSON file cache (it
    would serialize to tens of MB); it is memoized in-process and only the
    per-event slices are cached on disk.
    """

    DOMAIN: str = "hail"
    SOURCE_URL: str = _ARCHIVE_URL
    # Labels = ``mag >= 2.0`` (SPC significant-hail diameter) and feature[0]
    # is the same ``mag`` column.  Direct feature-threshold circularity,
    # declared honestly (see loaders/label_provenance.py).
    LABEL_SOURCE: str = "statistical"
    REQUIRES_API_KEY: bool = False

    FEATURE_COLUMNS: list[str] = [
        "hail_size_in",
        "injuries",
        "fatalities",
        "property_loss",
        "slat",
        "slon",
        "temporal_cluster",
        "geo_anomaly",
        "month",
        "day_of_year",
        "hour",
    ]

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the hail loader.

        Args:
            **kwargs: Forwarded to :class:`BaseDomainLoader`.
        """
        super().__init__(**kwargs)
        self._archive_df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def fetch_realtime(self) -> pd.DataFrame:
        """Fetch today's hail reports from the SPC filtered daily reports.

        The daily CSV concatenates tornado / wind / hail sections, each
        introduced by its own header row; only the hail section
        (``Time,Size,...``) is parsed, and ``Size`` (hundredths of an inch)
        is converted to inches in a ``mag`` column to match the archive.

        Returns:
            DataFrame with columns ``time``, ``mag`` (inches), ``location``,
            ``county``, ``st``, ``slat``, ``slon``, ``comments``.

        Raises:
            ConnectionError: If the SPC feed is unreachable after retries.
            ValueError: If the feed has no recognizable hail section.
        """
        cache_key = "hail_realtime"
        cached = self._read_cache(cache_key)
        if cached is not None:
            logger.debug("Returning cached real-time hail data.")
            return pd.DataFrame(cached)

        raw = self._fetch_url(_DAILY_REPORTS_URL)
        df = self._parse_daily_hail_section(raw)

        self._write_cache(cache_key, df.to_dict(orient="list"))
        logger.info("Fetched %d real-time hail reports from SPC.", len(df))
        return df

    def fetch_historical(self, event_id: str) -> pd.DataFrame:
        """Fetch hail records for a specific historical event.

        Loads the SPC severe-hail archive and filters to the date range
        associated with the requested event.

        Args:
            event_id: Key into the ground-truth catalog (e.g.
                ``"vivian_2010"``).

        Returns:
            DataFrame with SPC hail-archive columns filtered to the event's
            date range, sorted chronologically.

        Raises:
            ValueError: If *event_id* is not in the catalog.
            ConnectionError: If the SPC archive is unreachable.
        """
        if event_id not in _EVENT_CATALOG:
            raise ValueError(
                f"Unknown event_id '{event_id}'. Available: {list(_EVENT_CATALOG.keys())}"
            )

        cache_key = f"hail_historical_{event_id}"
        cached = self._read_cache(cache_key)
        if cached is not None:
            logger.debug("Returning cached historical data for '%s'.", event_id)
            return pd.DataFrame(cached)

        archive_df = self._load_archive()

        event = _EVENT_CATALOG[event_id]
        start_date = pd.Timestamp(event["start"])
        end_date = pd.Timestamp(event["end"])
        df = self._filter_by_date_range(archive_df, start_date, end_date)

        if df.empty:
            logger.warning("SPC hail archive returned no records for event '%s'.", event_id)
            return df

        df = df.sort_values(["date", "time"]).reset_index(drop=True)

        self._write_cache(cache_key, df.to_dict(orient="list"))
        logger.info("Fetched %d historical hail records for event '%s'.", len(df), event_id)
        return df

    def list_events(self) -> list[dict[str, Any]]:
        """Return the catalog of ground-truth hail events.

        Returns:
            List of dicts each containing *event_id*, *name*, *date*, and
            *description* keys.
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
        """Generate binary anomaly labels for a historical hail event.

        Labeling strategy: a report is labelled *anomalous* (``1``) when
        its hail diameter (``mag``) is at least 2.00 in -- the SPC
        *significant hail* size.  Smaller severe-hail reports are ``0``.

        This is a feature-threshold label (``mag`` is also feature[0]);
        the loader declares ``LABEL_SOURCE = "statistical"`` accordingly.

        Args:
            event_id: Key into the ground-truth catalog.

        Returns:
            1-D binary numpy array of shape ``(n_reports,)``.

        Raises:
            ValueError: If *event_id* is not recognized.
        """
        if event_id not in _EVENT_CATALOG:
            raise ValueError(
                f"Unknown event_id '{event_id}'. Available: {list(_EVENT_CATALOG.keys())}"
            )

        df = self.fetch_historical(event_id)
        if df.empty:
            return np.array([], dtype=np.int64)

        mag = pd.to_numeric(df["mag"], errors="coerce").fillna(-1).values
        labels = (mag >= _SIG_HAIL_IN).astype(np.int64)

        logger.info(
            "Ground truth for '%s': %d anomalies / %d total (threshold >= %.2f in).",
            event_id,
            int(labels.sum()),
            len(labels),
            _SIG_HAIL_IN,
        )
        return np.asarray(labels)

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def engineer_features(self, raw_data: pd.DataFrame) -> np.ndarray[Any, Any]:
        """Transform raw hail data into a feature matrix.

        Engineered features (per hail report row):

        1.  **hail_size_in** -- hail diameter in inches (``mag``).
        2.  **injuries** -- reported injuries (``inj``).
        3.  **fatalities** -- reported fatalities (``fat``).
        4.  **property_loss** -- property-loss figure (``loss``).
        5.  **slat** -- report latitude.
        6.  **slon** -- report longitude.
        7.  **temporal_cluster** -- count of hail reports in the same hour
            within a 100 km radius (outbreak-intensity proxy).
        8.  **geo_anomaly** -- great-circle distance (km) from the
            hail-alley climatological centroid (~41 N, 104 W).
        9.  **month** -- month of occurrence (1-12).
        10. **day_of_year** -- day of year (1-366).
        11. **hour** -- hour of occurrence (0-23).

        Args:
            raw_data: DataFrame from :meth:`fetch_realtime` or
                :meth:`fetch_historical`.

        Returns:
            2-D numpy array of shape ``(n_samples, 11)``.
        """
        n_features = len(self.FEATURE_COLUMNS)
        if raw_data.empty:
            return np.empty((0, n_features), dtype=np.float64)

        df = raw_data.copy()

        def _col(name: str) -> np.ndarray[Any, Any]:
            if name in df.columns:
                return np.asarray(
                    pd.to_numeric(df[name], errors="coerce").fillna(0).values, dtype=np.float64
                )
            return np.zeros(len(df), dtype=np.float64)

        hail_size = _col("mag")
        injuries = _col("inj")
        fatalities = _col("fat")
        loss = _col("loss")
        slat = _col("slat")
        slon = _col("slon")

        timestamps = self._parse_timestamps(df)
        month = np.array([ts.month for ts in timestamps], dtype=np.float64)
        day_of_year = np.array([ts.timetuple().tm_yday for ts in timestamps], dtype=np.float64)
        hour = np.array([ts.hour for ts in timestamps], dtype=np.float64)

        epoch_seconds = np.array([ts.timestamp() for ts in timestamps], dtype=np.float64)
        if len(timestamps) > 1:
            temporal_cluster = neighbor_counts_within_km(
                slat, slon, 100.0, times_s=epoch_seconds, time_window_s=3600.0
            )
        else:
            temporal_cluster = np.zeros(len(df), dtype=np.float64)

        geo_anomaly = haversine_km_to_point(
            slat, slon, _HAIL_ALLEY_CENTROID_LAT, _HAIL_ALLEY_CENTROID_LON
        )

        features = np.column_stack(
            [
                hail_size,
                injuries,
                fatalities,
                loss,
                slat,
                slon,
                temporal_cluster,
                geo_anomaly,
                month,
                day_of_year,
                hour,
            ]
        )

        # Replace non-finite values with the column median (matching the
        # BaseDomainLoader hygiene convention).
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

    def _load_archive(self) -> pd.DataFrame:
        """Load the SPC severe-hail archive (zipped CSV) with memoization.

        Returns:
            DataFrame with the complete 1955-2023 hail-report archive.

        Raises:
            ConnectionError: If the SPC archive URL is unreachable.
        """
        if self._archive_df is not None:
            logger.debug("Returning memoized hail archive.")
            return self._archive_df

        logger.info("Downloading SPC severe-hail archive from %s", _ARCHIVE_URL)
        df = self._fetch_csv(_ARCHIVE_URL, compression="zip", low_memory=False)

        df.columns = [c.strip().lower() for c in df.columns]
        numeric_cols = ["mag", "inj", "fat", "loss", "closs", "slat", "slon"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        self._archive_df = df
        logger.info("Loaded %d records from SPC hail archive.", len(df))
        return df

    @staticmethod
    def _filter_by_date_range(
        df: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """Filter archive DataFrame to an inclusive date range.

        Args:
            df: Full hail-archive DataFrame.
            start_date: Start of the date range (inclusive).
            end_date: End of the date range (inclusive).

        Returns:
            Filtered DataFrame.

        Raises:
            ValueError: If no date columns are present (schema drift must
                fail loud rather than silently return everything).
        """
        if "date" in df.columns:
            parsed_dates = pd.to_datetime(df["date"], errors="coerce")
        elif all(c in df.columns for c in ("yr", "mo", "dy")):
            parsed_dates = pd.to_datetime(
                df[["yr", "mo", "dy"]].rename(columns={"yr": "year", "mo": "month", "dy": "day"}),
                errors="coerce",
            )
        else:
            raise ValueError(
                "SPC hail archive has neither a 'date' column nor 'yr'/'mo'/'dy' "
                f"columns (got {list(df.columns)[:8]}...); cannot filter by event window."
            )

        mask = (parsed_dates >= start_date) & (parsed_dates <= end_date)
        return df.loc[mask].reset_index(drop=True)

    @staticmethod
    def _parse_daily_hail_section(raw: bytes) -> pd.DataFrame:
        """Parse the hail section out of the SPC filtered daily reports CSV.

        The daily file concatenates three sections, each with its own
        header (``Time,F_Scale,...`` tornado, ``Time,Speed,...`` wind,
        ``Time,Size,...`` hail).  Sizes are hundredths of an inch and are
        converted to inches (``mag``).

        Args:
            raw: Raw CSV bytes from the SPC daily feed.

        Returns:
            DataFrame with normalised columns (``mag`` in inches, ``slat``,
            ``slon``, ...).  Empty (0 rows) when the hail section exists
            but has no reports yet today.

        Raises:
            ValueError: If no hail section header is present (format drift).
        """
        text = raw.decode("utf-8", errors="replace")
        lines = text.strip().split("\n")

        hail_header_idx: int | None = None
        section_bounds: list[int] = []
        for i, line in enumerate(lines):
            if line.startswith("Time,"):
                section_bounds.append(i)
                if ",Size," in line:
                    hail_header_idx = i
        if hail_header_idx is None:
            raise ValueError(
                "SPC daily reports feed has no hail section header ('Time,Size,...'); "
                "the feed format may have changed."
            )

        next_bounds = [b for b in section_bounds if b > hail_header_idx]
        section_end = next_bounds[0] if next_bounds else len(lines)
        section_lines = lines[hail_header_idx:section_end]

        # SPC daily rows occasionally carry unquoted commas inside the
        # trailing Comments field; fold any surplus fields back into it so a
        # real report is never dropped or misaligned.
        header = next(csv.reader([section_lines[0]]))
        n_cols = len(header)
        rows: list[list[str]] = []
        for line in section_lines[1:]:
            if not line.strip():
                continue
            fields = next(csv.reader([line]))
            if len(fields) > n_cols:
                fields = fields[: n_cols - 1] + [",".join(fields[n_cols - 1 :])]
            elif len(fields) < n_cols:
                fields = fields + [""] * (n_cols - len(fields))
            rows.append(fields)

        df = pd.DataFrame(rows, columns=header)
        df.columns = [c.strip().lower() for c in df.columns]
        rename_map = {"size": "mag", "lat": "slat", "lon": "slon"}
        for old_name, new_name in rename_map.items():
            if old_name in df.columns and new_name not in df.columns:
                df = df.rename(columns={old_name: new_name})

        if "mag" in df.columns:
            df["mag"] = pd.to_numeric(df["mag"], errors="coerce") * _DAILY_SIZE_SCALE
        for col in ("slat", "slon"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    @staticmethod
    def _parse_timestamps(df: pd.DataFrame) -> list[pd.Timestamp]:
        """Parse per-row timestamps from archive or daily-report columns.

        Prefers the archive's ``date`` + ``time`` columns, then ``date``
        alone, then ``yr``/``mo``/``dy``; daily-report rows (``time`` only,
        HHMM) get today's date with the report hour.  Unparseable rows fall
        back to the epoch (1970-01-01), matching the tornado loader's
        convention.

        Args:
            df: Hail DataFrame.

        Returns:
            List of pandas Timestamps, one per row.
        """
        n = len(df)
        timestamps: list[pd.Timestamp] = []

        if "date" in df.columns and "time" in df.columns:
            for i in range(n):
                try:
                    timestamps.append(pd.Timestamp(f"{df['date'].iloc[i]} {df['time'].iloc[i]}"))
                except (ValueError, TypeError):
                    timestamps.append(pd.Timestamp("1970-01-01"))
        elif "date" in df.columns:
            for i in range(n):
                try:
                    timestamps.append(pd.Timestamp(str(df["date"].iloc[i])))
                except (ValueError, TypeError):
                    timestamps.append(pd.Timestamp("1970-01-01"))
        elif all(c in df.columns for c in ("yr", "mo", "dy")):
            for i in range(n):
                try:
                    timestamps.append(
                        pd.Timestamp(
                            year=int(df["yr"].iloc[i]),
                            month=int(df["mo"].iloc[i]),
                            day=int(df["dy"].iloc[i]),
                        )
                    )
                except (ValueError, TypeError):
                    timestamps.append(pd.Timestamp("1970-01-01"))
        elif "time" in df.columns:
            today = pd.Timestamp.utcnow().tz_localize(None).normalize()
            for i in range(n):
                try:
                    hhmm = f"{int(df['time'].iloc[i]):04d}"
                    timestamps.append(
                        today + pd.Timedelta(hours=int(hhmm[:2]), minutes=int(hhmm[2:]))
                    )
                except (ValueError, TypeError):
                    timestamps.append(pd.Timestamp("1970-01-01"))
        else:
            timestamps = [pd.Timestamp("1970-01-01")] * n

        return timestamps
