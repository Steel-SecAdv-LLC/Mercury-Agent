"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

NOAA GSOD — Global Summary of the Day Loader

Daily weather summaries from 9000+ stations worldwide. Free access.

Source: https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/{YEAR}/
License: Public Domain (US Government)

Anomaly labeling:
  Readings exceeding 3σ from the station's historical mean are flagged as anomalies.
"""

from __future__ import annotations

import csv
import io
import logging
from typing import Any

import numpy as np
import requests

from omni_mercury_engine.security.input_validation import TrustedEndpoints

from .base import DatasetConfig, DatasetLoader, DatasetRegistry, http_get_with_retry
from .exceptions import DataSourceUnavailableError

logger = logging.getLogger(__name__)

# Representative US stations (USAF-WBAN IDs)
DEFAULT_STATIONS = [
    "72503014732",  # JFK Airport, NY
    "72219013874",  # Dallas/Fort Worth, TX
    "72295023174",  # LAX Airport, CA
    "72534014733",  # O'Hare Airport, IL
    "72278003017",  # Denver, CO
    "72253012842",  # Phoenix, AZ
    "72290023234",  # San Francisco, CA
    "72209013889",  # Houston, TX
    "72528014739",  # Minneapolis, MN
    "72258024233",  # Atlanta, GA
]


class NOAAGSODLoader(DatasetLoader):
    """
    NOAA Global Summary of the Day (GSOD) weather data loader.

    Downloads daily weather summary CSVs from NCEI for selected stations.
    Each record contains temperature, dew point, pressure, wind speed,
    precipitation, and other meteorological measurements.

    Anomaly labeling: Observations where any measured value exceeds 3σ
    from the station's mean for that year are labeled as anomalies.

    Args:
        config: DatasetConfig. Preprocessing options:
            - year (int): Year to load (default: 2023)
            - stations (list[str]): Station IDs (default: 10 US stations)
    """

    DATASET_NAME = "noaa_gsod"
    LABEL_SOURCE = "statistical"  # labels = per-station statistical anomaly mask (threshold)
    DATASET_URL = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/"
    LICENSE = "Public Domain (US Government)"
    CITATION = "NOAA National Centers for Environmental Information. Global Summary of the Day."
    REQUIRES_CREDENTIALS = False

    GSOD_BASE = TrustedEndpoints.NOAA_GSOD_BASE

    FEATURE_NAMES = [
        "temp_mean",
        "temp_max",
        "temp_min",
        "dewp",
        "slp",
        "stp",
        "visib",
        "wdsp",
        "mxspd",
        "gust",
        "prcp",
        "sndp",
        "station_idx",
        "month",
        "day",
    ]

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)
        self.year = config.preprocessing.get("year", 2023)
        self.stations = config.preprocessing.get("stations", DEFAULT_STATIONS)

    # GSOD year-end roll-up lags real-time by months. Fall through to older
    # years if the requested year directory is empty / 404 for every station.
    _YEAR_FALLBACK_RANGE = 2

    def download(self) -> bool:
        """
        Download GSOD CSVs for configured stations and year.

        Falls back to ``self.year - 1`` and ``self.year - 2`` if the
        requested year has no published station files yet.

        Raises:
            DataSourceUnavailableError: If no station data is retrievable
            from any candidate year.
        """
        candidate_years = [self.year - offset for offset in range(self._YEAR_FALLBACK_RANGE + 1)]

        for candidate in candidate_years:
            cache_file = self.data_path / f"gsod_{candidate}.npz"
            if cache_file.exists():
                logger.info("GSOD %d already cached", candidate)
                if candidate != self.year:
                    self.year = candidate
                return True

            all_rows: list[list[float]] = []
            for idx, station_id in enumerate(self.stations):
                url = f"{self.GSOD_BASE}{candidate}/{station_id}.csv"
                try:
                    body = http_get_with_retry(url, timeout=30)
                    text = body.decode("utf-8", errors="replace")
                    rows = self._parse_gsod_csv(text, idx)
                    all_rows.extend(rows)
                    logger.info(
                        "  GSOD %d station %s: %d daily records",
                        candidate,
                        station_id,
                        len(rows),
                    )
                except requests.HTTPError as e:
                    status = e.response.status_code if e.response is not None else "?"
                    logger.warning("  GSOD %d station %s: HTTP %s", candidate, station_id, status)
                except Exception as e:
                    logger.warning("  GSOD %d station %s failed: %s", candidate, station_id, e)

            if all_rows:
                if candidate != self.year:
                    logger.info("GSOD %d returned no data; using %d instead", self.year, candidate)
                    self.year = candidate
                return self._finalize(all_rows, cache_file)

            logger.info("GSOD %d empty for all stations; trying older year", candidate)

        raise DataSourceUnavailableError(
            loader_name="NOAA-GSOD",
            source_url=f"{self.GSOD_BASE}{self.year}/",
            reason=(
                f"No GSOD station data retrieved for {self.year} or the prior "
                f"{self._YEAR_FALLBACK_RANGE} years"
            ),
        )

    def _finalize(self, all_rows: list[list[float]], cache_file: Any) -> bool:
        """Build features/labels and persist the npz cache."""
        features = np.array(all_rows, dtype=np.float64)

        # 3-sigma anomaly labeling per station
        labels = np.zeros(len(features), dtype=np.int32)
        for s_idx in range(len(self.stations)):
            mask = features[:, 12] == s_idx
            if mask.sum() < 10:
                continue
            station_data = features[mask, :12]
            mean = station_data.mean(axis=0)
            std = station_data.std(axis=0) + 1e-8
            z = np.abs((station_data - mean) / std)
            anomalous = z.max(axis=1) > 3.0
            labels[mask] = anomalous.astype(np.int32)

        self.data_path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, features=features, labels=labels)

        logger.info(
            "GSOD %d loaded: %d records from %d stations, %.1f%% anomalies",
            self.year,
            len(features),
            len(self.stations),
            100.0 * labels.mean(),
        )
        return True

    def _parse_gsod_csv(self, text: str, station_idx: int) -> list[list[float]]:
        """Parse a single station's GSOD CSV."""
        rows: list[list[float]] = []
        reader = csv.DictReader(io.StringIO(text))

        missing_vals = {"9999.9", "999.9", "99.99"}

        for record in reader:
            try:

                def safe_float(key: str, default: float = 0.0) -> float:
                    val = record.get(key, "").strip()
                    if val in missing_vals or not val:
                        return default
                    return float(val)

                date_str = record.get("DATE", "")
                parts = date_str.split("-")
                month = int(parts[1]) if len(parts) > 1 else 1
                day = int(parts[2]) if len(parts) > 2 else 1

                rows.append(
                    [
                        safe_float("TEMP"),
                        safe_float("MAX"),
                        safe_float("MIN"),
                        safe_float("DEWP"),
                        safe_float("SLP"),
                        safe_float("STP"),
                        safe_float("VISIB"),
                        safe_float("WDSP"),
                        safe_float("MXSPD"),
                        safe_float("GUST"),
                        safe_float("PRCP"),
                        safe_float("SNDP"),
                        float(station_idx),
                        float(month),
                        float(day),
                    ]
                )
            except (ValueError, TypeError):
                continue
        return rows

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        cache_file = self.data_path / f"gsod_{self.year}.npz"
        if not cache_file.exists():
            raise FileNotFoundError(f"GSOD {self.year} not found. Run with download=True.")
        data = np.load(cache_file)
        return data["features"], data["labels"]

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess."""
        data = np.nan_to_num(data, nan=0.0)
        return ((data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)).astype(np.float32)


DatasetRegistry.register("noaa_gsod", NOAAGSODLoader)
DatasetRegistry.register("gsod", NOAAGSODLoader)
