# Copyright (C) 2025 Steel Security Advisors LLC
"""NOAA Storm Events Database Loader."""

from __future__ import annotations

import gzip
import io
import logging
import re
from typing import Any

import numpy as np
import requests

from omni_mercury_engine.security.input_validation import TrustedEndpoints

from .base import DatasetConfig, DatasetLoader, DatasetRegistry, http_get_with_retry
from .exceptions import DataSourceUnavailableError

logger = logging.getLogger(__name__)


class NOAAStormEventsLoader(DatasetLoader):
    """NOAA Storm Events Database loader.

    Downloads bulk CSV files from NCEI containing severe weather events
    across the US. Each record includes event type, property/crop damage,
    injuries, fatalities, and geographic information.

    Anomaly labeling: Events with fatalities > 0 OR property damage > $1M
    are labeled as anomalies (extreme weather events).

    Args:
        config: DatasetConfig. Preprocessing options:
            - year_start (int): Start year (default: current - 5)
            - year_end (int): End year (default: current year)
    """

    DATASET_NAME = "noaa_storm_events"
    LABEL_SOURCE = "statistical"  # labels = threshold on damage/casualty feature columns
    DATASET_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
    LICENSE = "Public Domain (US Government)"
    CITATION = (
        "NOAA National Centers for Environmental Information. "
        "Storm Events Database. https://www.ncdc.noaa.gov/stormevents/"
    )
    REQUIRES_CREDENTIALS = False

    STORM_BASE_URL = TrustedEndpoints.NOAA_STORM_EVENTS

    EVENT_TYPES = [
        "Tornado",
        "Hail",
        "Thunderstorm Wind",
        "Flash Flood",
        "Flood",
        "Heavy Rain",
        "Heavy Snow",
        "Ice Storm",
        "Winter Storm",
        "Blizzard",
        "High Wind",
        "Hurricane",
        "Tropical Storm",
        "Wildfire",
        "Drought",
        "Heat",
        "Cold/Wind Chill",
    ]

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize the instance."""
        super().__init__(config)
        import datetime

        current_year = datetime.datetime.now(datetime.UTC).year
        self.year_start = config.preprocessing.get("year_start", current_year - 5)
        self.year_end = config.preprocessing.get("year_end", current_year)

    # NCEI publishes per-year files of the form:
    #   StormEvents_details-ftp_v1.0_d{YEAR}_c{COMPILE_DATE}.csv.gz
    # The COMPILE_DATE component is required and varies per release. The
    # previous loader hard-coded the URL without it and got HTTP 404 for
    # every year. We discover the actual filename by parsing the directory
    # listing and then download the most recent compile for the year.
    _DETAIL_FILE_RE = re.compile(
        r'href="(StormEvents_details-ftp_v1\.0_d(?P<year>\d{4})_c(?P<compile>\d{8})\.csv\.gz)"'
    )

    def _resolve_detail_filenames(self) -> dict[int, str]:
        """Fetch the NCEI directory index and map year -> latest compile filename."""
        try:
            content = http_get_with_retry(self.STORM_BASE_URL, timeout=60)
        except Exception as e:
            raise DataSourceUnavailableError(
                loader_name="NOAAStormEvents",
                source_url=self.STORM_BASE_URL,
                reason=f"Failed to fetch NCEI directory index: {e}",
            ) from e

        index_html = content.decode("utf-8", errors="replace")
        latest: dict[int, tuple[str, str]] = {}
        for match in self._DETAIL_FILE_RE.finditer(index_html):
            year = int(match.group("year"))
            compile_date = match.group("compile")
            filename = match.group(1)
            existing = latest.get(year)
            if existing is None or compile_date > existing[0]:
                latest[year] = (compile_date, filename)
        return {year: name for year, (_, name) in latest.items()}

    def download(self) -> bool:
        """Download storm event detail CSVs from NCEI.

        Raises:
            DataSourceUnavailableError: If NCEI is unreachable.
        """
        cache_file = self.data_path / "storm_events.npz"
        if cache_file.exists():
            logger.info("Storm events already cached")
            return True

        filename_by_year = self._resolve_detail_filenames()
        all_rows: list[list[float]] = []

        for year in range(self.year_start, self.year_end + 1):
            filename = filename_by_year.get(year)
            if filename is None:
                logger.warning(
                    "  Storm events %d: no published file in NCEI directory listing",
                    year,
                )
                continue
            url = f"{self.STORM_BASE_URL}{filename}"

            try:
                logger.info("  Downloading storm events for %d (%s)...", year, filename)
                content = http_get_with_retry(url, timeout=60)

                try:
                    text = gzip.decompress(content).decode("utf-8", errors="replace")
                except gzip.BadGzipFile:
                    text = content.decode("utf-8", errors="replace")

                rows = self._parse_storm_csv(text, year)
                all_rows.extend(rows)
                logger.info("    %d events for %d", len(rows), year)

            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else "?"
                logger.warning("  Storm events %d: HTTP %s", year, status)
            except Exception as e:
                logger.warning("  Storm events %d failed: %s", year, e)

        if not all_rows:
            raise DataSourceUnavailableError(
                loader_name="NOAAStormEvents",
                source_url=self.STORM_BASE_URL,
                reason=f"No storm event data downloaded for years {self.year_start}-{self.year_end}",
            )

        features = np.array(all_rows, dtype=np.float64)
        # Anomaly: fatalities > 0 OR property damage > $1M
        labels = ((features[:, 5] > 0) | (features[:, 3] > 1_000_000)).astype(np.int32)

        self.data_path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, features=features, labels=labels)
        logger.info(
            "Storm events loaded: %d records, %.1f%% anomalies",
            len(features),
            100.0 * labels.mean(),
        )
        return True

    def _parse_storm_csv(self, text: str, year: int) -> list[list[float]]:
        """Parse a storm events detail CSV into numeric rows."""
        import csv

        rows: list[list[float]] = []
        reader = csv.DictReader(io.StringIO(text))

        event_lookup = {e.lower(): i for i, e in enumerate(self.EVENT_TYPES)}

        for record in reader:
            try:
                event_type = record.get("EVENT_TYPE", "").strip().lower()
                event_code = event_lookup.get(event_type, len(self.EVENT_TYPES))

                state_fips = int(record.get("STATE_FIPS", "0") or "0")
                month = int(record.get("BEGIN_YEARMONTH", str(year * 100 + 1))[-2:])

                # Parse damage amounts (can have K/M suffixes)
                prop_damage = self._parse_damage(record.get("DAMAGE_PROPERTY", "0"))
                crop_damage = self._parse_damage(record.get("DAMAGE_CROPS", "0"))

                injuries = int(float(record.get("INJURIES_DIRECT", "0") or "0"))
                fatalities = int(float(record.get("DEATHS_DIRECT", "0") or "0"))

                lat = float(record.get("BEGIN_LAT", "0") or "0")
                lon = float(record.get("BEGIN_LON", "0") or "0")

                rows.append(
                    [
                        event_code,
                        state_fips,
                        year,
                        month,
                        prop_damage,
                        crop_damage,
                        injuries,
                        fatalities,
                        lat,
                        lon,
                    ]
                )
            except (ValueError, TypeError):
                continue

        return rows

    @staticmethod
    def _parse_damage(value: str) -> float:
        """Parse NOAA damage string (e.g., '25K', '1.5M') to float."""
        value = value.strip().upper()
        if not value or value == "0":
            return 0.0
        multiplier = 1.0
        if value.endswith("K"):
            multiplier = 1_000
            value = value[:-1]
        elif value.endswith("M"):
            multiplier = 1_000_000
            value = value[:-1]
        elif value.endswith("B"):
            multiplier = 1_000_000_000
            value = value[:-1]
        try:
            return float(value) * multiplier
        except ValueError:
            return 0.0

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        cache_file = self.data_path / "storm_events.npz"
        if not cache_file.exists():
            raise FileNotFoundError("Storm events not found. Run with download=True.")
        data = np.load(cache_file)
        return data["features"], data["labels"]

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess."""
        data = np.nan_to_num(data, nan=0.0)
        # Log-transform damage columns
        data[:, 4] = np.log1p(data[:, 4])
        data[:, 5] = np.log1p(data[:, 5])
        return ((data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)).astype(np.float32)


DatasetRegistry.register("noaa_storm_events", NOAAStormEventsLoader)
DatasetRegistry.register("storm_events", NOAAStormEventsLoader)
