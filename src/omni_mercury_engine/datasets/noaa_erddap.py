"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

NOAA ERDDAP — Oceanographic and Climate Dataset Loader

Replaces broken Copernicus SLA, SimonsCMAP, and WorldOcean loaders with
NOAA's free, auth-free ERDDAP REST API.

Data sources:
  - Sea Level Anomaly: coastwatch.pfeg.noaa.gov/erddap/griddap/nesdisSSH1day.csv
  - Sea Surface Temp (Chlorophyll): coastwatch.pfeg.noaa.gov/erddap/griddap/nesdisVHNSQchlaDaily.csv
  - NDBC Buoy Real-Time: www.ndbc.noaa.gov/data/realtime2/{STATION}.txt

Anomaly detection methodology:
  Statistical deviation from historical mean — readings exceeding 3σ from
  the station/grid-cell historical mean are flagged as anomalies.

License: Public Domain (US Government)
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import requests

from omni_mercury_engine.security.input_validation import TrustedEndpoints

from .base import DatasetConfig, DatasetLoader, DatasetRegistry, http_get_with_retry
from .exceptions import DataSourceUnavailableError

logger = logging.getLogger(__name__)


class NOAAERDDAPLoader(DatasetLoader):
    """
    NOAA ERDDAP oceanographic data loader.

    Fetches gridded oceanographic data (sea level anomaly, chlorophyll-a)
    from NOAA's ERDDAP REST API. No authentication required.

    Anomaly labeling: Observations exceeding 3 standard deviations from
    the column mean are labeled as anomalies (statistical method).

    Args:
        config: DatasetConfig. Preprocessing options:
            - dataset (str): "ssh" (sea level) or "chlorophyll" (default "ssh")
            - lat_range (tuple): (min_lat, max_lat), default (20.0, 50.0)
            - lon_range (tuple): (min_lon, max_lon), default (-130.0, -60.0)
    """

    DATASET_NAME = "noaa_erddap"
    LABEL_SOURCE = "statistical"  # labels = max z-score > 3.0 (threshold on features)
    DATASET_URL = "https://coastwatch.pfeg.noaa.gov/erddap/"
    LICENSE = "Public Domain (US Government)"
    CITATION = "NOAA CoastWatch / OceanWatch. ERDDAP Data Server."
    REQUIRES_CREDENTIALS = False

    SSH_URL = TrustedEndpoints.NOAA_ERDDAP_SSH
    CHL_URL = TrustedEndpoints.NOAA_ERDDAP_CHL

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)
        self.dataset_type = config.preprocessing.get("dataset", "ssh")
        self.lat_range = config.preprocessing.get("lat_range", (20.0, 50.0))
        self.lon_range = config.preprocessing.get("lon_range", (-130.0, -60.0))
        self._is_real_data = False

    # SSH (sea level anomaly) is sliced by an explicit time constraint, so
    # the ingest-lag fallback steps SSH through progressively earlier dates
    # until ERDDAP returns a populated grid. CHL (chlorophyll-a) is exposed
    # as a single time-series endpoint with no date constraint in the URL,
    # so retrying multiple offsets there would just re-issue an identical
    # request and log misleading "trying earlier date" messages — we make
    # a single attempt for that path instead.
    _SSH_DATE_OFFSET_DAYS_FALLBACK = (7, 14, 21, 30)

    def _build_ssh_url(self, base_url: str, date_str: str) -> str:
        """Construct the SSH ERDDAP URL with a time/lat/lon constraint."""
        lat_min, lat_max = self.lat_range
        lon_min, lon_max = self.lon_range
        return f"{base_url}?sla[({date_str})]" f"[({lat_min}):({lat_max})][({lon_min}):({lon_max})]"

    def _persist(self, content: bytes, cache_file: Any) -> tuple[Any, Any]:
        """Parse, label, and persist an ERDDAP CSV response."""
        text = content.decode("utf-8", errors="replace")
        features, labels = self._parse_erddap_csv(text)
        sha = hashlib.sha256(content).hexdigest()
        self.data_path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, features=features, labels=labels, sha256=sha)
        self._is_real_data = True
        return features, labels

    def download(self) -> bool:
        """
        Download ERDDAP data via REST CSV API.

        For ``dataset_type == "ssh"`` the loader steps through
        ``_SSH_DATE_OFFSET_DAYS_FALLBACK`` (7, 14, 21, 30 days back) to
        absorb the dataset's ingest lag. For ``dataset_type != "ssh"`` the
        URL has no date constraint, so a single request is issued.

        Returns:
            True on success.

        Raises:
            DataSourceUnavailableError: If ERDDAP is unreachable.
        """
        cache_file = self.data_path / f"erddap_{self.dataset_type}.npz"
        if cache_file.exists():
            logger.info("ERDDAP %s already cached", self.dataset_type)
            self._is_real_data = True
            return True

        if self.dataset_type == "ssh":
            base_url = self.SSH_URL
            now = datetime.now(UTC)
            last_err: Exception | None = None
            last_url = ""
            for offset_days in self._SSH_DATE_OFFSET_DAYS_FALLBACK:
                target = now - timedelta(days=offset_days)
                date_str = target.strftime("%Y-%m-%dT00:00:00Z")
                url = self._build_ssh_url(base_url, date_str)
                last_url = url
                logger.info(
                    "Downloading ERDDAP ssh (date %s, offset -%dd) from %s",
                    date_str,
                    offset_days,
                    url[:100],
                )
                try:
                    content = http_get_with_retry(url, timeout=60)
                    features, labels = self._persist(content, cache_file)
                    logger.info(
                        "ERDDAP ssh loaded: %d samples, %.1f%% anomalies (offset -%dd)",
                        len(features),
                        100.0 * labels.mean(),
                        offset_days,
                    )
                    return True
                except requests.HTTPError as e:
                    last_err = e
                    status = e.response.status_code if e.response is not None else "?"
                    logger.info(
                        "ERDDAP ssh offset -%dd: HTTP %s; trying earlier date",
                        offset_days,
                        status,
                    )
                    continue
                except Exception as e:
                    last_err = e
                    logger.info(
                        "ERDDAP ssh offset -%dd failed: %s; trying earlier date",
                        offset_days,
                        e,
                    )
                    continue

            raise DataSourceUnavailableError(
                loader_name="NOAA-ERDDAP-ssh",
                source_url=last_url,
                reason=f"All ERDDAP date offsets failed: {last_err}",
            ) from last_err

        # Non-SSH path: single request, no date constraint to vary.
        base_url = self.CHL_URL
        url = f"{base_url}?chlor_a"
        logger.info("Downloading ERDDAP %s from %s", self.dataset_type, url[:100])
        try:
            content = http_get_with_retry(url, timeout=60)
        except Exception as e:
            raise DataSourceUnavailableError(
                loader_name=f"NOAA-ERDDAP-{self.dataset_type}",
                source_url=url,
                reason=str(e),
            ) from e

        features, labels = self._persist(content, cache_file)
        logger.info(
            "ERDDAP %s loaded: %d samples, %.1f%% anomalies",
            self.dataset_type,
            len(features),
            100.0 * labels.mean(),
        )
        return True

    def _parse_erddap_csv(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """Parse ERDDAP CSV response into features and labels."""
        lines = text.strip().split("\n")

        # Skip header rows (first line is column names, second is units)
        data_lines = []
        for line in lines[2:]:
            parts = line.split(",")
            row = []
            for p in parts:
                p = p.strip()
                try:
                    row.append(float(p))
                except ValueError:
                    row.append(0.0)
            if row:
                data_lines.append(row)

        if not data_lines:
            raise ValueError("ERDDAP returned no parseable data rows")

        features = np.array(data_lines, dtype=np.float64)
        features = np.nan_to_num(features, nan=0.0)

        # Statistical anomaly labeling: 3-sigma deviation on each column
        mean = features.mean(axis=0)
        std = features.std(axis=0) + 1e-8
        z_scores = np.abs((features - mean) / std)
        labels = (z_scores.max(axis=1) > 3.0).astype(np.int32)

        return features, labels

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load cached ERDDAP data."""
        cache_file = self.data_path / f"erddap_{self.dataset_type}.npz"
        if not cache_file.exists():
            raise FileNotFoundError(
                f"ERDDAP {self.dataset_type} data not found. Run with download=True."
            )
        data = np.load(cache_file)
        self._is_real_data = True
        return data["features"], data["labels"]

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Z-score normalize."""
        data = np.nan_to_num(data, nan=0.0)
        return ((data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)).astype(np.float32)


DatasetRegistry.register("noaa_erddap", NOAAERDDAPLoader)
DatasetRegistry.register("erddap", NOAAERDDAPLoader)
DatasetRegistry.register("erddap_ssh", NOAAERDDAPLoader)
