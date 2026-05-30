"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

EPA Air Quality System (AQS) — Daily PM2.5 Loader

Downloads daily PM2.5 monitoring data from EPA's pre-generated files. Anomaly flags are computed
against EPA AQI thresholds.

Source:
https://aqs.epa.gov/aqsweb/airdata/download_files.html
License: Public Domain (US Government)
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
    """
    EPA Air Quality System daily PM2.5 loader.

    Downloads daily PM2.5 summary data from EPA's pre-generated annual
    ZIP files. Each ZIP contains a CSV with site-level daily measurements.

    Anomaly labeling: Daily mean PM2.5 > 35.4 µg/m³ (EPA "Unhealthy for
    Sensitive Groups" AQI threshold) is labeled as anomaly.

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

    # PM2.5 AQI breakpoints (µg/m³)
    AQI_UNHEALTHY_SENSITIVE = 35.4

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)
        self.year = config.preprocessing.get("year", 2023)

    # EPA AQS publishes annual prebuilt files with a ~6 month lag from year-end.
    # Requesting the current calendar year almost always 404s; fall through to
    # progressively older years so the loader still returns a real dataset.
    _YEAR_FALLBACK_RANGE = 3

    def download(self) -> bool:
        """
        Download EPA daily PM2.5 ZIP for the configured year.

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
        """Parse EPA daily PM2.5 CSV."""
        import csv

        rows: list[list[float]] = []
        reader = csv.DictReader(io.StringIO(text))

        for record in reader:
            try:
                pm25 = float(record.get("Arithmetic Mean", "0") or "0")
                lat = float(record.get("Latitude", "0") or "0")
                lon = float(record.get("Longitude", "0") or "0")
                obs_count = float(record.get("Observation Count", "1") or "1")

                # Parse date
                date_str = record.get("Date Local", "2023-01-01")
                parts = date_str.split("-")
                month = int(parts[1]) if len(parts) > 1 else 1
                day = int(parts[2]) if len(parts) > 2 else 1

                state_code = int(record.get("State Code", "0") or "0")
                county_code = int(record.get("County Code", "0") or "0")

                rows.append([pm25, lat, lon, obs_count, month, day, state_code, county_code])
            except (ValueError, TypeError):
                continue

        if not rows:
            raise ValueError("No parseable records in EPA CSV")

        features = np.array(rows, dtype=np.float64)
        labels = (features[:, 0] > self.AQI_UNHEALTHY_SENSITIVE).astype(np.int32)
        return features, labels

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
