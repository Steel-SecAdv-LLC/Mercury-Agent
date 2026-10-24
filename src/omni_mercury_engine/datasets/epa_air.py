# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""EPA Air Quality System (AQS) — Daily PM2.5 Loader.

Downloads daily PM2.5 monitoring data from EPA's pre-generated files. Anomaly flags are computed
against EPA AQI thresholds.

Source:
https://aqs.epa.gov/aqsweb/airdata/download_files.html
"""

from __future__ import annotations

import io
import logging
import zipfile
from typing import Any

import numpy as np
import requests

from omni_mercury_engine.security.input_validation import TrustedEndpoints

from .base import DatasetConfig, DatasetLoader, DatasetRegistry, http_get_with_retry
from .exceptions import DataSourceUnavailableError

logger = logging.getLogger(__name__)


class EPAAirQualityLoader(DatasetLoader):
    """EPA Air Quality System daily PM2.5 loader.

    Downloads daily PM2.5 summary data from EPA's pre-generated annual
    ZIP files. Each ZIP contains a CSV with site-level daily measurements.

    One row per monitor-day: the file also carries each continuous
    monitor's day as a "1 HOUR" summary beside its "24-HR BLK AVG" standard
    value, and exceptional-event days as both an "Included" and an
    "Excluded" row, so only rows with a ``Pollutant Standard`` and not
    ``Excluded`` are kept (in the 2024 file: 398,031 of 740,924 rows, one per
    monitor-day; formerly every row was kept, so most days counted twice).
    Days without a standard value (fewer hourly samples than a 24-hour mean
    needs) are not in the standard rows.

    Anomaly labeling: the EPA AQI of the 24-hour mean (``pm25_aqi``: 40 CFR
    Part 58 Appendix G) is 101 or more, "Unhealthy for Sensitive Groups" or
    worse.

    Args:
        config: DatasetConfig. Preprocessing options:
            - year (int): Year to download (default: 2023)
    """

    DATASET_NAME = "epa_air_quality"
    LABEL_SOURCE = "statistical"  # labels = PM2.5 > AQI threshold (domain cut on the feature)
    DATASET_URL = "https://aqs.epa.gov/aqsweb/airdata/download_files.html"
    LICENSE = "Public Domain (US Government)"
    CITATION = "U.S. Environmental Protection Agency. Air Quality System (AQS)."
    REQUIRES_CREDENTIALS = False

    EPA_BASE_URL = TrustedEndpoints.EPA_AQS_BASE

    # Top of the PM2.5 "Moderate" band (µg/m³): a mean that truncates to
    # more than this is AQI 101+, "Unhealthy for Sensitive Groups" (40 CFR
    # Part 58 Appendix G, Table 2). Labels use ``pm25_aqi`` itself.
    AQI_UNHEALTHY_SENSITIVE = 35.4
    #: Columns every AirData daily file carries and this parser reads.
    REQUIRED_COLUMNS = frozenset(
        {
            "State Code",
            "County Code",
            "Latitude",
            "Longitude",
            "Date Local",
            "Arithmetic Mean",
            "Observation Count",
            "Pollutant Standard",
            "Event Type",
        }
    )

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize the instance."""
        super().__init__(config)
        self.year = config.preprocessing.get("year", 2023)

    # EPA AQS publishes annual prebuilt files with a ~6 month lag from year-end.
    # Requesting the current calendar year almost always 404s; fall through to
    # progressively older years so the loader still returns a real dataset.
    _YEAR_FALLBACK_RANGE = 3

    def download(self) -> bool:
        """Download EPA daily PM2.5 ZIP for the configured year.

        If the requested year is not yet published (HTTP 404), automatically
        fall back to the most recent prior year that is available, up to
        ``_YEAR_FALLBACK_RANGE`` years back.

        Raises:
            DataSourceUnavailableError: If no available year can be reached.
        """
        candidate_years = [self.year - offset for offset in range(self._YEAR_FALLBACK_RANGE + 1)]

        last_err: Exception | None = None
        last_url = ""
        for candidate in candidate_years:
            cache_file = self.data_path / f"epa_pm25_{candidate}.npz"
            if cache_file.exists():
                logger.info("EPA PM2.5 %d already cached", candidate)
                if candidate != self.year:
                    self.year = candidate
                return True

            url = f"{self.EPA_BASE_URL}daily_88101_{candidate}.zip"
            last_url = url
            logger.info("Downloading EPA PM2.5 data for %d from %s", candidate, url)

            try:
                content = http_get_with_retry(url, timeout=120)

                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                    if not csv_names:
                        raise ValueError("No CSV found in EPA ZIP archive")
                    csv_text = zf.read(csv_names[0]).decode("utf-8", errors="replace")

                features, labels = self._parse_epa_csv(csv_text)

                self.data_path.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache_file, features=features, labels=labels)

                if candidate != self.year:
                    logger.info(
                        "EPA PM2.5 %d not yet published; using %d instead",
                        self.year,
                        candidate,
                    )
                    self.year = candidate

                logger.info(
                    "EPA PM2.5 %d loaded: %d records, %.1f%% above AQI threshold",
                    candidate,
                    len(features),
                    100.0 * labels.mean(),
                )
                return True

            except requests.HTTPError as e:
                last_err = e
                status = e.response.status_code if e.response is not None else None
                if status == 404:
                    logger.info(
                        "EPA PM2.5 %d not published (HTTP 404); trying older year", candidate
                    )
                    continue
                logger.warning(
                    "EPA download for %d failed: HTTP %s", candidate, status if status else "?"
                )
                continue
            except Exception as e:
                last_err = e
                logger.warning("EPA download for %d failed: %s", candidate, e)
                continue

        raise DataSourceUnavailableError(
            loader_name="EPA-AirQuality",
            source_url=last_url,
            reason=(
                f"No EPA PM2.5 file available for {self.year} or "
                f"the prior {self._YEAR_FALLBACK_RANGE} years: {last_err}"
            ),
        ) from last_err

    def _parse_epa_csv(self, text: str) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Parse an EPA AirData daily PM2.5 CSV: one row per monitor-day.

        A row without a measured mean or coordinates is skipped and counted
        (never read as 0 ug/m3 at 0 N 0 E).

        Raises:
            ValueError: If the header lacks an AirData column this parser
                needs, or no row is usable.
        """
        import csv

        from omni_mercury_engine.detectors.meteorological.air_quality import pm25_aqi

        rows: list[list[float]] = []
        labels: list[int] = []
        reader = csv.DictReader(io.StringIO(text))
        missing = self.REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"EPA CSV lacks the AirData column(s) {sorted(missing)}")

        skipped = 0
        for record in reader:
            # The 24-hour standard value only: the "1 HOUR" summary of the same
            # day carries no Pollutant Standard, and an "Excluded" row repeats
            # an exceptional-event day without its event hours.
            if not record["Pollutant Standard"] or record["Event Type"] == "Excluded":
                continue
            try:
                pm25 = float(record["Arithmetic Mean"])
                lat = float(record["Latitude"])
                lon = float(record["Longitude"])
                obs_count = float(record["Observation Count"])
                _, month_text, day_text = record["Date Local"].split("-")
                month, day = int(month_text), int(day_text)
                state_code = int(record["State Code"])
                county_code = int(record["County Code"])
                aqi, _ = pm25_aqi(pm25)
            except (ValueError, TypeError):
                skipped += 1
                continue

            rows.append([pm25, lat, lon, obs_count, month, day, state_code, county_code])
            labels.append(int(aqi >= 101))

        if skipped:
            logger.warning("EPA CSV: skipped %d unparseable standard row(s)", skipped)
        if not rows:
            raise ValueError("No parseable records in EPA CSV")

        features = np.array(rows, dtype=np.float64)
        return features, np.array(labels, dtype=np.int32)

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        cache_file = self.data_path / f"epa_pm25_{self.year}.npz"
        if not cache_file.exists():
            raise FileNotFoundError(f"EPA PM2.5 {self.year} not found. Run with download=True.")
        data = np.load(cache_file)
        return data["features"], data["labels"]

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess."""
        data = np.nan_to_num(data, nan=0.0)
        return ((data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)).astype(np.float32)


DatasetRegistry.register("epa_air_quality", EPAAirQualityLoader)
DatasetRegistry.register("epa_pm25", EPAAirQualityLoader)
