# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""MIT-BIH Arrhythmia Database Loader.

48 half-hour ECG recordings from PhysioNet with cardiologist-annotated beat labels.
Open access — no PhysioNet credentials required for the MIT-BIH database.

Source: https://physionet.org/content/mitdb/1.0.0/

Anomaly labeling:
  Non-normal beats (any annotation != 'N') are labeled as anomalies.
  This is the standard approach in ECG anomaly detection literature.

Requires: wfdb library (pip install wfdb)
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import DatasetConfig, DatasetLoader, DatasetRegistry
from .exceptions import DataSourceUnavailableError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterator

# All 48 MIT-BIH records
MITBIH_RECORDS = [
    "100",
    "101",
    "102",
    "103",
    "104",
    "105",
    "106",
    "107",
    "108",
    "109",
    "111",
    "112",
    "113",
    "114",
    "115",
    "116",
    "117",
    "118",
    "119",
    "121",
    "122",
    "123",
    "124",
    "200",
    "201",
    "202",
    "203",
    "205",
    "207",
    "208",
    "209",
    "210",
    "212",
    "213",
    "214",
    "215",
    "217",
    "219",
    "220",
    "221",
    "222",
    "223",
    "228",
    "230",
    "231",
    "232",
    "233",
    "234",
]

# Normal beat annotation symbols
NORMAL_SYMBOLS = {"N", "L", "R", "e", "j"}


@contextmanager
def _wfdb_request_timeout(timeout: float) -> Iterator[None]:
    """Apply a default timeout to WFDB's unbounded requests calls."""
    import requests

    original_request = requests.Session.request

    def request_with_timeout(
        session: requests.Session,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> requests.Response:
        kwargs.setdefault("timeout", timeout)
        return original_request(session, method, url, **kwargs)

    requests.Session.request = request_with_timeout  # type: ignore[assignment]
    try:
        yield
    finally:
        requests.Session.request = original_request  # type: ignore[method-assign]


class MITBIHLoader(DatasetLoader):
    """MIT-BIH Arrhythmia Database loader.

    Downloads 48 half-hour ECG recordings from PhysioNet. Each recording
    has cardiologist-annotated beat labels. Non-normal beats are labeled
    as anomalies.

    This dataset is open access and requires no PhysioNet credentialing.
    Requires the ``wfdb`` library: ``pip install wfdb``.

    Args:
        config: DatasetConfig. Preprocessing options:
            - records (list[str]): Record IDs to load (default: all 48)
            - segment_length (int): Samples per segment (default: 360, ~1 sec at 360Hz)
    """

    DATASET_NAME = "mitbih"
    LABEL_SOURCE = "expert_annotated"  # cardiologist-annotated heartbeat classes
    DATASET_URL = "https://physionet.org/content/mitdb/1.0.0/"
    LICENSE = "Open Data Commons Attribution License v1.0"
    CITATION = (
        "Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. "
        "IEEE Eng in Med and Biol 20(3):45-50, 2001."
    )
    REQUIRES_CREDENTIALS = False

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize the instance."""
        super().__init__(config)
        self.records = config.preprocessing.get("records", MITBIH_RECORDS)
        self.segment_length = config.preprocessing.get("segment_length", 360)
        self.request_timeout = float(config.preprocessing.get("request_timeout", 20.0))
        self.max_record_failures = int(config.preprocessing.get("max_record_failures", 3))

    def download(self) -> bool:
        """Download MIT-BIH records via wfdb.

        Raises:
            DataSourceUnavailableError: If wfdb is not installed or download fails.
        """
        try:
            import wfdb
        except ImportError as e:
            raise DataSourceUnavailableError(
                loader_name="MIT-BIH",
                reason=(
                    "wfdb library required: pip install mercury-agent[medical] "
                    "(or: pip install wfdb>=4.3.1). "
                    "MIT-BIH data source: https://physionet.org/content/mitdb/1.0.0/"
                ),
            ) from e

        cache_file = self.data_path / "mitbih_segments.npz"
        if cache_file.exists():
            logger.info("MIT-BIH already cached")
            return True

        self.data_path.mkdir(parents=True, exist_ok=True)

        all_segments: list[np.ndarray[Any, Any]] = []
        all_labels: list[int] = []
        loaded_records = 0
        consecutive_failures = 0

        for rec_id in self.records:
            try:
                with _wfdb_request_timeout(self.request_timeout):
                    record = wfdb.rdrecord(rec_id, pn_dir="mitdb")
                    annotation = wfdb.rdann(rec_id, "atr", pn_dir="mitdb")

                signal = record.p_signal  # (n_samples, n_channels)
                if signal is None:
                    continue

                # Use first channel (MLII)
                channel = signal[:, 0]

                # Segment around each annotated beat
                for i, sample_idx in enumerate(annotation.sample):
                    start = sample_idx - self.segment_length // 2
                    end = start + self.segment_length

                    if start < 0 or end > len(channel):
                        continue

                    segment = channel[start:end]
                    if len(segment) != self.segment_length:
                        continue

                    sym = annotation.symbol[i]
                    is_anomaly = 0 if sym in NORMAL_SYMBOLS else 1

                    all_segments.append(segment)
                    all_labels.append(is_anomaly)

                loaded_records += 1
                consecutive_failures = 0
                logger.info("  Record %s: %d beats", rec_id, len(annotation.sample))
                if self.config.max_samples and len(all_segments) >= self.config.max_samples:
                    break

            except Exception as e:
                consecutive_failures += 1
                logger.warning("  Record %s failed: %s", rec_id, e)
                if consecutive_failures >= self.max_record_failures:
                    logger.warning(
                        "Stopping MIT-BIH download after %d consecutive record failures",
                        consecutive_failures,
                    )
                    break

        if loaded_records == 0 or not all_segments:
            raise DataSourceUnavailableError(
                loader_name="MIT-BIH",
                source_url=self.DATASET_URL,
                reason=(
                    "No MIT-BIH records could be downloaded before the bounded "
                    f"failure threshold ({self.max_record_failures}) was reached"
                ),
            )

        X = np.array(all_segments, dtype=np.float64)
        y = np.array(all_labels, dtype=np.int32)
        if self.config.max_samples and len(X) > self.config.max_samples:
            X = X[: self.config.max_samples]
            y = y[: self.config.max_samples]

        np.savez_compressed(cache_file, X=X, y=y)

        logger.info(
            "MIT-BIH loaded: %d records, %d beats, %.1f%% anomalous",
            loaded_records,
            len(y),
            100.0 * y.mean(),
        )
        return True

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        cache_file = self.data_path / "mitbih_segments.npz"
        if not cache_file.exists():
            raise FileNotFoundError("MIT-BIH not found. Run with download=True.")
        data = np.load(cache_file)
        return data["X"], data["y"]

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Per-segment Z-score normalization."""
        data = np.nan_to_num(data, nan=0.0)
        mean = data.mean(axis=1, keepdims=True)
        std = data.std(axis=1, keepdims=True) + 1e-8
        return ((data - mean) / std).astype(np.float32)

    def get_metadata(self) -> dict[str, Any]:
        """Get metadata."""
        features, labels = self.load()
        return {
            "name": "MIT-BIH Arrhythmia Database",
            "source": self.DATASET_URL,
            "n_samples": len(features),
            "n_features": features.shape[1],
            "anomaly_ratio": float(labels.mean()),
            "is_real_data": True,
            "data_source": "live",
            "label_source": "expert_annotated",
            "n_records": len(self.records),
            "citation": self.CITATION,
        }


DatasetRegistry.register("mitbih", MITBIHLoader)
DatasetRegistry.register("mit-bih", MITBIHLoader)
DatasetRegistry.register("arrhythmia", MITBIHLoader)
