"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

EPA Air Quality System (AQS) — Daily PM2.5 Loader

Downloads daily PM2.5 monitoring data from EPA's pre-generated files.
Anomaly flags are computed against EPA AQI thresholds.

Source: https://aqs.epa.gov/aqsweb/airdata/download_files.html
License: Public Domain (US Government)
"""

from __future__ import annotations

import io
import logging
import urllib.error
import urllib.request
import zipfile
from typing import Any

import numpy as np

from omni_mercury_engine.security.input_validation import TrustedEndpoints

from .base import DatasetConfig, DatasetLoader, DatasetRegistry
from .exceptions import DataSourceUnavailableError

logger = logging.getLogger(__name__)


class EPAAirQualityLoader(DatasetLoader):
    """EPA Air Quality System daily PM2.5 loader.

    Downloads daily PM2.5 summary data from EPA's pre-generated annual
    ZIP files. Each ZIP contains a CSV with site-level daily measurements.

    Anomaly labeling: Daily mean PM2.5 > 35.4 µg/m³ (EPA "Unhealthy for
    Sensitive Groups" AQI threshold) is labeled as anomaly.

    Args:
        config: DatasetConfig. Preprocessing options:
            - year (int): Year to download (default: 2023)
    """

    DATASET_NAME = "epa_air_quality"
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

    def download(self) -> bool:
        """Download EPA daily PM2.5 ZIP for the configured year.

        Raises:
            DataSourceUnavailableError: If EPA data is unreachable.
        """
        cache_file = self.data_path / f"epa_pm25_{self.year}.npz"
        if cache_file.exists():
            logger.info("EPA PM2.5 %d already cached", self.year)
            return True

        url = f"{self.EPA_BASE_URL}daily_88101_{self.year}.zip"
        logger.info("Downloading EPA PM2.5 data for %d from %s", self.year, url)

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mercury-Agent/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:  # nosec B310
                content = resp.read()

            # Extract CSV from ZIP
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_names:
                    raise ValueError("No CSV found in EPA ZIP archive")
                csv_text = zf.read(csv_names[0]).decode("utf-8", errors="replace")

            features, labels = self._parse_epa_csv(csv_text)

            self.data_path.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_file, features=features, labels=labels)

            logger.info(
                "EPA PM2.5 %d loaded: %d records, %.1f%% above AQI threshold",
                self.year,
                len(features),
                100.0 * labels.mean(),
            )
            return True

        except Exception as e:
            logger.error("EPA download failed: %s", e)
            raise DataSourceUnavailableError(
                loader_name="EPA-AirQuality",
                source_url=url,
                reason=str(e),
            ) from e

    def _parse_epa_csv(self, text: str) -> tuple[np.ndarray, np.ndarray]:
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
        data = np.nan_to_num(data, nan=0.0)
        return ((data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)).astype(np.float32)


DatasetRegistry.register("epa_air_quality", EPAAirQualityLoader)
DatasetRegistry.register("epa_pm25", EPAAirQualityLoader)
