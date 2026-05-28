"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

ADBench — Tabular Anomaly Detection Benchmark Datasets

Loads real-world anomaly detection datasets from the ADBench repository
(Minqi et al., NeurIPS 2022 Datasets and Benchmarks Track).

Each dataset is a single NPZ file on GitHub containing:
  - X: feature matrix (n_samples, n_features)
  - y: binary labels (0=normal, 1=anomaly)

Source: https://github.com/Minqi824/ADBench
License: MIT

Citation:
    Han S, Hu X, Huang H, Jiang M, Zhao Y.
    ADBench: Anomaly Detection Benchmark. NeurIPS 2022.
"""

from __future__ import annotations

import hashlib
import io
import logging
from typing import Any

import numpy as np

from omni_mercury_engine.security.input_validation import TrustedEndpoints

from .base import DatasetConfig, DatasetLoader, DatasetRegistry, http_get_with_retry
from .exceptions import DataSourceUnavailableError

logger = logging.getLogger(__name__)


# ADBench Classical dataset catalog: index -> name
ADBENCH_CATALOG: dict[int, str] = {
    1: "ALOI",
    2: "annthyroid",
    3: "backdoor",
    4: "breastw",
    5: "campaign",
    6: "cardio",
    7: "Cardiotocography",
    8: "celeba",
    9: "census",
    10: "cover",
    11: "donors",
    12: "fault",
    13: "fraud",
    14: "glass",
    15: "Hepatitis",
    16: "http",
    17: "InternetAds",
    18: "Ionosphere",
    19: "landsat",
    20: "letter",
    21: "Lymphography",
    22: "magic.gamma",
    23: "mammography",
    24: "mnist",
    25: "musk",
    26: "optdigits",
    27: "PageBlocks",
    28: "pendigits",
    29: "Pima",
    30: "satellite",
    31: "satimage-2",
    32: "shuttle",
    33: "skin",
    34: "smtp",
    35: "SpamBase",
    36: "speech",
    37: "Stamps",
    38: "thyroid",
    39: "vertebral",
    40: "vowels",
    41: "Waveform",
    42: "WBC",
    43: "WDBC",
    44: "Wilt",
    45: "wine",
    46: "WPBC",
    47: "yeast",
}

# Reverse lookup: name (lowercase) -> (index, canonical_name)
_NAME_TO_INDEX: dict[str, tuple[int, str]] = {
    name.lower(): (idx, name) for idx, name in ADBENCH_CATALOG.items()
}


class ADBenchLoader(DatasetLoader):
    """
    ADBench tabular anomaly detection dataset loader.

    Downloads real NPZ files from the ADBench GitHub repository. Each file
    contains feature matrix X and label vector y with ground-truth anomaly
    labels.

    Data source:
        https://github.com/Minqi824/ADBench/tree/main/adbench/datasets/Classical

    Supported datasets include: backdoor, campaign, cardio, fraud, smtp, http,
    thyroid, mammography, breastw, Cardiotocography, Hepatitis, WBC, WDBC, WPBC,
    and 33 others (47 total).

    Args:
        config: DatasetConfig. Use ``preprocessing={"dataset": "fraud"}`` to
            select a specific dataset by name.
    """

    DATASET_NAME = "adbench"
    DATASET_URL = "https://github.com/Minqi824/ADBench"
    LICENSE = "MIT"
    CITATION = (
        "Han S, Hu X, Huang H, Jiang M, Zhao Y. ADBench: Anomaly Detection Benchmark. NeurIPS 2022."
    )
    REQUIRES_CREDENTIALS = False

    BASE_URL = TrustedEndpoints.ADBENCH_BASE

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)
        # Dataset selection precedence: an explicit ``preprocessing['dataset']``
        # wins (the benchmark harness sets it per-index); otherwise derive the
        # dataset from ``config.name`` so the registry entries registered as
        # ``adbench-<name>`` actually load ``<name>`` instead of silently
        # collapsing to the historical ``fraud`` default.
        dataset_key = config.preprocessing.get("dataset")
        if dataset_key is None:
            dataset_key = self._dataset_key_from_name(config.name)
        self._resolve_dataset(dataset_key)

    @staticmethod
    def _dataset_key_from_name(name: str | None) -> str:
        """Derive a catalog key from a (possibly registry-prefixed) config name.

        Strips an ``adbench-``/``adbench_``/``adbench:`` prefix so the
        ``DatasetRegistry.create("adbench-cardio", ...)`` path resolves to
        ``cardio``. Falls back to the historical ``"fraud"`` default when the
        name is empty, the bare ``"adbench"`` alias, or does not identify a
        catalog entry — preserving behaviour for callers that set neither
        ``preprocessing['dataset']`` nor a recognisable name.
        """
        raw = (name or "").strip()
        low = raw.lower()
        for prefix in ("adbench-", "adbench_", "adbench:"):
            if low.startswith(prefix):
                raw, low = raw[len(prefix) :], low[len(prefix) :]
                break
        if not low or low == "adbench":
            return "fraud"
        if low in _NAME_TO_INDEX:
            return raw
        try:
            if int(raw) in ADBENCH_CATALOG:
                return raw
        except (ValueError, TypeError):
            pass
        logger.warning(
            "ADBench config name %r does not identify a catalog dataset; "
            "defaulting to 'fraud'. Pass preprocessing={'dataset': <name>} or "
            "use a registered name like 'adbench-cardio'.",
            name,
        )
        return "fraud"

    def _resolve_dataset(self, key: str) -> None:
        """Resolve a dataset name or index to canonical form."""
        # Try as integer index
        try:
            idx = int(key)
            if idx in ADBENCH_CATALOG:
                self._dataset_index = idx
                self._dataset_name = ADBENCH_CATALOG[idx]
                return
        except (ValueError, TypeError):
            pass

        # Try as name (case-insensitive)
        lookup = _NAME_TO_INDEX.get(key.lower())
        if lookup is not None:
            self._dataset_index, self._dataset_name = lookup
            return

        raise ValueError(
            f"Unknown ADBench dataset: '{key}'. Available: {list(ADBENCH_CATALOG.values())}"
        )

    @property
    def npz_filename(self) -> str:
        """Filename of the NPZ on GitHub."""
        return f"{self._dataset_index}_{self._dataset_name}.npz"

    @property
    def npz_url(self) -> str:
        """Full URL to the NPZ file."""
        return f"{self.BASE_URL}{self.npz_filename}"

    def download(self) -> bool:
        """
        Download the selected ADBench dataset NPZ from GitHub.

        Returns:
            True on success.

        Raises:
            DataSourceUnavailableError: If download fails.
        """
        cache_file = self.data_path / self.npz_filename
        if cache_file.exists():
            logger.info("ADBench %s already cached at %s", self._dataset_name, cache_file)
            return True

        url = self.npz_url
        logger.info("Downloading ADBench %s from %s", self._dataset_name, url)

        try:
            # ``ADBENCH_BASE`` is pinned to ``raw.githubusercontent.com``
            # so no redirect is involved. Benchmark runs walk all 47
            # catalog entries back-to-back, which routinely trips
            # GitHub's anonymous rate limit; retry on 429/5xx.
            content = http_get_with_retry(url, timeout=120)

            # Verify we got a valid NPZ
            buf = io.BytesIO(content)
            data = np.load(buf, allow_pickle=False)
            if "X" not in data or "y" not in data:
                raise ValueError(f"NPZ missing X/y keys, found: {list(data.keys())}")

            # Save locally
            self.data_path.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "wb") as f:
                f.write(content)

            sha = hashlib.sha256(content).hexdigest()
            logger.info(
                "ADBench %s downloaded: %d samples, %d features, sha256=%s",
                self._dataset_name,
                data["X"].shape[0],
                data["X"].shape[1],
                sha[:16],
            )
            return True

        except Exception as e:
            logger.error("ADBench %s download failed: %s", self._dataset_name, e)
            raise DataSourceUnavailableError(
                loader_name=f"ADBench-{self._dataset_name}",
                source_url=url,
                reason=str(e),
            ) from e

    def _check_data_exists(self) -> bool:
        """Whether *this* dataset's NPZ is cached.

        Overrides the base directory-level check: all ADBench datasets share
        one ``adbench/`` directory, so the default ``any(dir.iterdir())`` would
        report data present for every dataset as soon as a single one is
        cached, causing ``load()`` to skip the download and then fail on (or
        mis-serve) the missing file. Gate on the specific NPZ instead.
        """
        return (self.data_path / self.npz_filename).exists()

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load the NPZ from local cache."""
        cache_file = self.data_path / self.npz_filename
        if not cache_file.exists():
            raise FileNotFoundError(
                f"ADBench {self._dataset_name} not found at {cache_file}. Run with download=True."
            )

        data = np.load(cache_file, allow_pickle=False)
        X = data["X"].astype(np.float64)
        y = data["y"].astype(np.int32).ravel()

        # Ensure binary labels
        y = (y > 0).astype(np.int32)

        logger.info(
            "Loaded ADBench %s: %d samples, %d features, %.1f%% anomalies",
            self._dataset_name,
            X.shape[0],
            X.shape[1],
            100.0 * y.mean(),
        )
        return X, y

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Z-score normalize features."""
        data = np.nan_to_num(data, nan=0.0, posinf=1e10, neginf=-1e10)
        mean = data.mean(axis=0)
        std = data.std(axis=0) + 1e-8
        return ((data - mean) / std).astype(np.float32)

    def get_metadata(self) -> dict[str, Any]:
        """Return dataset metadata."""
        features, labels = self.load()
        return {
            "name": f"ADBench-{self._dataset_name}",
            "source": self.npz_url,
            "n_samples": len(features),
            "n_features": features.shape[1],
            "anomaly_ratio": float(labels.mean()),
            "is_real_data": True,
            "data_source": "live",
            "url": self.DATASET_URL,
            "citation": self.CITATION,
        }

    @classmethod
    def list_datasets(cls) -> list[str]:
        """List all available ADBench dataset names."""
        return list(ADBENCH_CATALOG.values())


# Register key ADBench datasets individually
for _idx, _name in ADBENCH_CATALOG.items():
    DatasetRegistry.register(f"adbench-{_name.lower()}", ADBenchLoader)

# Also register the generic name
DatasetRegistry.register("adbench", ADBenchLoader)
