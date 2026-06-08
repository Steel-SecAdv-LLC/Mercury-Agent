# Copyright (C) 2025 Steel Security Advisors LLC
"""Base classes for real-world dataset loading and management."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

try:
    import torch
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


def safe_urlretrieve(url: str, filename: str | Path) -> None:
    """Safely download a file from a URL with scheme and domain validation.

    Delegates to :func:`http_get_with_retry`, which is HTTPS-only by
    default and validates the host against ``TrustedEndpoints``.
    Plain ``http://`` is refused at the gate; ``file://`` and other
    schemes never reach this helper.

    Internally inherits User-Agent, exponential backoff, and
    4xx-vs-5xx-aware retry semantics — important for the bulk loaders
    (BATADAL, NAB, SMD, SMAP/MSL, UCR, ADRepository) that hit GitHub
    raw under rate-limit pressure during a benchmark run.

    Args:
        url: The URL to download from (must be ``https://`` and on
            ``TrustedEndpoints.TRUSTED_DOMAINS``).
        filename: The local path to save the file to.

    Raises:
        UnsafeURLError / ValueError: URL failed the SafeHTTPClient
            scheme or trusted-domain gate.
    """
    body = http_get_with_retry(url, timeout=120)
    target = Path(filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as f:
        f.write(body)


# Default User-Agent. Many public dataset CDNs (raw.githubusercontent.com,
# www.fema.gov, www.ncei.noaa.gov) silently rate-limit or return 403 to
# requests without an identifying UA. Identifying as a real browser-class
# UA suffix avoids those penalties without misrepresenting the source.
_DEFAULT_DATASET_UA = "Mozilla/5.0 (compatible; Mercury-Agent/1.0; +https://github.com/Steel-SecAdv-LLC/Mercury-Agent)"


def http_get_with_retry(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
    retries: int = 3,
    backoff: float = 2.0,
    retry_on_status: tuple[int, ...] = (408, 425, 429, 500, 502, 503, 504),
    allow_http: bool = False,
    timeout_per_attempt: float | None = None,
) -> bytes:
    """HTTP GET with scheme/domain validation, default UA, and exponential backoff.

    Centralizes the retry behavior previously duplicated across loaders.
    Retries on transient HTTP status codes and on socket-level errors
    (timeout, connection reset). Permanent errors (404, 401, 403) raise
    immediately so callers can fail over to a different mirror.

    The transport is :class:`SafeHTTPClient`, which gates every URL:

    * **Trusted-allowlist** -- the host must be in
      ``TrustedEndpoints.TRUSTED_DOMAINS`` regardless of scheme.
      Operator-configured hosts must use ``SafeHTTPClient`` directly
      with ``user_configured=True`` so SSRF checks stay explicit.
    * **HTTPS-only by default** -- ``http://`` is rejected unless
      ``allow_http=True``.  When that opt-in is granted, the URL is
      additionally checked against the private-network / IMDS gate so
      a plain-HTTP mirror cannot pivot through internal infrastructure.

    Args:
        url: HTTP/HTTPS URL to fetch.
        headers: Extra request headers. User-Agent is injected if not provided.
        timeout: Per-attempt socket timeout in seconds.
        retries: Total attempts (initial + retries). Must be >= 1.
        backoff: Exponential factor; sleep = backoff ** attempt seconds.
        retry_on_status: HTTP status codes that trigger a retry rather than
            raising. 4xx not in this set are treated as permanent.
        allow_http: Permit ``http://`` URLs. Default False (HTTPS-only).
            When True the host still has to clear both the trusted-domain
            allowlist and the private-network / IMDS gate.
        timeout_per_attempt: Optional per-attempt timeout. When set, each retry
            uses ``min(timeout, timeout_per_attempt)`` so a dead upstream cannot
            consume the full caller budget on its first socket attempt.

    Returns:
        Response body as bytes.

    Raises:
        ValueError / UnsafeURLError: URL scheme/domain not permitted under the
            security defaults (or current opt-in flags).
        requests.HTTPError: Final attempt returned a non-retried status.
        requests.RequestException / TimeoutError: All attempts exhausted on
            transient socket errors.
    """
    import time

    import requests

    from omni_mercury_engine.security.safe_http import SafeHTTPClient

    request_headers = {"User-Agent": _DEFAULT_DATASET_UA}
    if headers:
        request_headers.update(headers)
    effective_timeout = (
        min(timeout, timeout_per_attempt) if timeout_per_attempt is not None else timeout
    )
    deadline = time.monotonic() + timeout

    last_exc: Exception | None = None
    attempts = max(1, retries)
    for attempt in range(attempts):
        try:
            return SafeHTTPClient.get_bytes(
                url,
                headers=request_headers,
                timeout=effective_timeout,
                allow_http=allow_http,
                deadline=deadline,
            )
        except requests.HTTPError as e:
            last_exc = e
            status = e.response.status_code if e.response is not None else None
            if status is None or status not in retry_on_status:
                # Permanent error -- let caller fail over to next mirror.
                raise
            logger.info(
                "HTTP %d on %s (attempt %d/%d); will retry.",
                status,
                url,
                attempt + 1,
                attempts,
            )
        except (requests.RequestException, TimeoutError, ConnectionError) as e:
            last_exc = e
            logger.info(
                "Transient %s on %s (attempt %d/%d); will retry.",
                type(e).__name__,
                url,
                attempt + 1,
                attempts,
            )

        if attempt < attempts - 1:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(backoff ** (attempt + 1), remaining))

    if last_exc is None:
        # The loop runs at least once (attempts >= 1) and every code
        # path either returns or assigns last_exc, so this branch is
        # only reachable on a logic regression in the loop above.
        raise RuntimeError(
            f"http_get_with_retry: exhausted {attempts} attempts on {url} "
            "without recording an exception (loop invariant violated)."
        )
    raise last_exc


class DatasetSplit(Enum):
    """Standard dataset splits."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    ALL = "all"


@dataclass
class DatasetConfig:
    """Configuration for dataset loading.

    Attributes:
        name: Dataset identifier
        version: Dataset version (e.g., "1.4" for MIMIC-III)
        data_dir: Local directory for downloaded data
        cache_dir: Directory for processed cache files
        download: Whether to download if not present
        preprocessing: Preprocessing configuration
        split_ratios: Train/val/test split ratios
        max_samples: Maximum samples to load (None = all)
        random_seed: Seed for reproducibility
        credentials_path: Path to credentials file (for PhysioNet)
    """

    name: str
    version: str = "latest"
    data_dir: str = "./data"
    cache_dir: str = "./cache"
    download: bool = True
    preprocessing: dict[str, Any] = field(default_factory=dict)
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
    max_samples: int | None = None
    random_seed: int = 42
    credentials_path: str | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if abs(sum(self.split_ratios) - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")

        # Create directories if possible (may fail for system paths like /custom)
        for dir_path in [self.data_dir, self.cache_dir]:
            try:
                os.makedirs(dir_path, exist_ok=True)
            except PermissionError:
                # Directory creation will be attempted again when actually needed
                logger.debug(f"Cannot create directory {dir_path} (permission denied)")
            except OSError as e:
                logger.debug(f"Cannot create directory {dir_path}: {e}")

    def get_cache_key(self) -> str:
        """Generate unique cache key for this configuration."""
        key_data = f"{self.name}_{self.version}_{self.max_samples}_{self.random_seed}"
        key_data += json.dumps(self.preprocessing, sort_keys=True)
        # Using SHA3-256 for Ava-Guardian alignment
        # Note: This is non-cryptographic use for cache key generation
        return hashlib.sha3_256(key_data.encode()).hexdigest()[:16]


@dataclass
class DatasetMetadata:
    """Metadata about a loaded dataset."""

    name: str
    version: str
    num_samples: int
    num_features: int
    feature_names: list[str]
    target_names: list[str]
    class_distribution: dict[str, int]
    source_url: str
    license: str
    citation: str
    preprocessing_applied: list[str]

    def to_dict(self) -> dict[str, Any]:
        """To dict."""
        return {
            "name": self.name,
            "version": self.version,
            "num_samples": self.num_samples,
            "num_features": self.num_features,
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "class_distribution": self.class_distribution,
            "source_url": self.source_url,
            "license": self.license,
            "citation": self.citation,
            "preprocessing": self.preprocessing_applied,
        }


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders.

    All real-world dataset loaders inherit from this class.
    Provides standardized interface for:
    - Loading and caching data
    - Preprocessing pipelines
    - Train/val/test splitting
    - Iterator interfaces
    """

    # Dataset-specific constants (override in subclasses)
    DATASET_NAME: str = "base"
    DATASET_URL: str = ""
    LICENSE: str = "Unknown"
    CITATION: str = ""
    REQUIRES_CREDENTIALS: bool = False

    # Label provenance, declared at source. One of the values in
    # ``datasets.metadata.VALID_LABEL_SOURCES``. Loaders that manufacture
    # anomaly labels by thresholding a score/feature (z-score, percentile,
    # domain cut) or by synthetic generation MUST override this to
    # ``"statistical"`` so headline supervised AUC excludes them as circular.
    # Defaults to ``"ground_truth"``; override to be honest.
    LABEL_SOURCE: str = "ground_truth"

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize dataset loader.

        Args:
            config: Dataset configuration
        """
        self.config = config
        self.data_path = Path(config.data_dir) / self.DATASET_NAME
        self.cache_path = Path(config.cache_dir) / self.DATASET_NAME

        self._data: dict[DatasetSplit, np.ndarray[Any, Any]] | None = None
        self._labels: dict[DatasetSplit, np.ndarray[Any, Any]] | None = None
        self._metadata: DatasetMetadata | None = None
        self._is_loaded = False

        # Create directories
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def download(self) -> bool:
        """Download dataset from source.

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load raw data from files.

        Returns:
            Tuple of (features, labels)
        """
        pass

    @abstractmethod
    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Apply dataset-specific preprocessing.

        Args:
            data: Raw feature data

        Returns:
            Preprocessed features
        """
        pass

    def load(
        self, split: DatasetSplit = DatasetSplit.ALL
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load dataset with specified split.

        Args:
            split: Which split to return

        Returns:
            Tuple of (features, labels) for requested split
        """
        if not self._is_loaded:
            self._load_and_cache()

        # Validate data is loaded
        if self._data is None or self._labels is None:
            raise RuntimeError("Dataset not loaded. Call _load_and_cache() first.")

        if split == DatasetSplit.ALL:
            # Concatenate all splits
            features = np.concatenate(
                [
                    self._data[DatasetSplit.TRAIN],
                    self._data[DatasetSplit.VALIDATION],
                    self._data[DatasetSplit.TEST],
                ]
            )
            labels = np.concatenate(
                [
                    self._labels[DatasetSplit.TRAIN],
                    self._labels[DatasetSplit.VALIDATION],
                    self._labels[DatasetSplit.TEST],
                ]
            )
            return features, labels

        return self._data[split], self._labels[split]

    def _load_and_cache(self) -> None:
        """Load data, apply preprocessing, split, and cache."""
        cache_key = self.config.get_cache_key()
        cache_file = self.cache_path / f"{cache_key}.npz"

        # Check cache
        if cache_file.exists():
            logger.info(f"Loading {self.DATASET_NAME} from cache")
            # Cache files are self-generated; they must round-trip
            # through allow_pickle=False. A ValueError here means the
            # cache was written by an older code path that allowed
            # pickled objects; refuse to deserialise it. The operator
            # can rebuild the cache (delete the file and re-run) or
            # convert the legacy artefact via
            # ``python -m omni_mercury_engine.tools.migrate_pkl``.
            #
            # IMPORTANT: ``np.load`` on a ``.npz`` archive returns a
            # lazy ``NpzFile`` and only raises ``ValueError`` when an
            # object-dtype member is *accessed*. Wrapping only the
            # ``np.load`` call therefore lets a pickled cache slip past
            # this gate and surface as a raw NumPy traceback further
            # down. We materialise every member we will read inside the
            # same try block so the operator-actionable RuntimeError
            # fires at the boundary, not deep in numpy.
            try:
                cached = np.load(cache_file, allow_pickle=False)
                train_features = cached["train_features"]
                val_features = cached["val_features"]
                test_features = cached["test_features"]
                train_labels = cached["train_labels"]
                val_labels = cached["val_labels"]
                test_labels = cached["test_labels"]
            except ValueError as exc:
                raise RuntimeError(
                    f"Refusing to load legacy cache '{cache_file}' that requires "
                    "allow_pickle=True. Delete the file and let it rebuild, or use "
                    "tools/migrate_pkl.py to convert it offline."
                ) from exc
            self._data = {
                DatasetSplit.TRAIN: train_features,
                DatasetSplit.VALIDATION: val_features,
                DatasetSplit.TEST: test_features,
            }
            self._labels = {
                DatasetSplit.TRAIN: train_labels,
                DatasetSplit.VALIDATION: val_labels,
                DatasetSplit.TEST: test_labels,
            }
            self._is_loaded = True
            return

        # Check if data exists, download if needed
        if not self._check_data_exists():
            if self.config.download:
                logger.info(f"Downloading {self.DATASET_NAME}...")
                if not self.download():
                    raise RuntimeError(f"Failed to download {self.DATASET_NAME}")
            else:
                raise FileNotFoundError(f"Dataset {self.DATASET_NAME} not found and download=False")

        # Load raw data
        logger.info(f"Loading {self.DATASET_NAME} from disk...")
        features, labels = self._load_raw()

        # Apply max_samples limit
        if self.config.max_samples is not None:
            features = features[: self.config.max_samples]
            labels = labels[: self.config.max_samples]

        # Preprocess
        logger.info(f"Preprocessing {self.DATASET_NAME}...")
        features = self.preprocess(features)

        # Split data
        n = len(features)
        rng = np.random.default_rng(self.config.random_seed)
        indices = rng.permutation(n)

        train_end = int(n * self.config.split_ratios[0])
        val_end = train_end + int(n * self.config.split_ratios[1])

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        self._data = {
            DatasetSplit.TRAIN: features[train_idx],
            DatasetSplit.VALIDATION: features[val_idx],
            DatasetSplit.TEST: features[test_idx],
        }
        self._labels = {
            DatasetSplit.TRAIN: labels[train_idx],
            DatasetSplit.VALIDATION: labels[val_idx],
            DatasetSplit.TEST: labels[test_idx],
        }

        # Cache processed data
        np.savez_compressed(
            cache_file,
            train_features=self._data[DatasetSplit.TRAIN],
            train_labels=self._labels[DatasetSplit.TRAIN],
            val_features=self._data[DatasetSplit.VALIDATION],
            val_labels=self._labels[DatasetSplit.VALIDATION],
            test_features=self._data[DatasetSplit.TEST],
            test_labels=self._labels[DatasetSplit.TEST],
        )

        self._is_loaded = True
        logger.info(f"Loaded {self.DATASET_NAME}: {n} samples")

    def _check_data_exists(self) -> bool:
        """Check if raw data files exist."""
        # Default implementation - override for specific datasets
        return self.data_path.exists() and any(self.data_path.iterdir())

    def get_metadata(self) -> DatasetMetadata | dict[str, Any]:
        """Get dataset metadata.

        Returns:
            DatasetMetadata object or dict with metadata.
            Subclasses may return dict[str, Any] with additional domain-specific fields.
        """
        if not self._is_loaded:
            self._load_and_cache()

        if self._metadata is None:
            features, labels = self.load(DatasetSplit.ALL)
            unique, counts = np.unique(labels, return_counts=True)

            self._metadata = DatasetMetadata(
                name=self.DATASET_NAME,
                version=self.config.version,
                num_samples=len(features),
                num_features=features.shape[1] if len(features.shape) > 1 else 1,
                feature_names=self._get_feature_names(),
                target_names=self._get_target_names(),
                class_distribution={str(k): int(v) for k, v in zip(unique, counts, strict=False)},
                source_url=self.DATASET_URL,
                license=self.LICENSE,
                citation=self.CITATION,
                preprocessing_applied=list(self.config.preprocessing.keys()),
            )

        return self._metadata

    def _get_feature_names(self) -> list[str]:
        """Get feature names - override in subclasses."""
        if self._data is not None:
            n_features = self._data[DatasetSplit.TRAIN].shape[1]
            return [f"feature_{i}" for i in range(n_features)]
        return []

    def _get_target_names(self) -> list[str]:
        """Get target names - override in subclasses."""
        return ["normal", "anomaly"]

    def __len__(self) -> int:
        """Get total number of samples."""
        if not self._is_loaded:
            self._load_and_cache()
        if self._data is None:
            raise RuntimeError("Dataset not loaded. Call _load_and_cache() first.")
        return sum(len(d) for d in self._data.values())

    def __iter__(self) -> Iterator[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]]:
        """Iterate over all samples."""
        features, labels = self.load(DatasetSplit.ALL)
        for i in range(len(features)):
            yield features[i], labels[i]

    def to_pytorch_dataset(self, split: DatasetSplit = DatasetSplit.TRAIN) -> Any:
        """Convert to PyTorch Dataset.

        Args:
            split: Which split to convert

        Returns:
            PyTorch Dataset object
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for to_pytorch_dataset")

        features, labels = self.load(split)

        class TorchDataset(Dataset):  # type: ignore[type-arg, unused-ignore]
            def __init__(self, X: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> None:
                """Initialize the instance."""
                self.X = torch.FloatTensor(X)
                self.y = torch.LongTensor(y)

            def __len__(self) -> int:
                """Return the length."""
                return len(self.X)

            def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
                """Implement the Python data model method."""
                return self.X[idx], self.y[idx]

        return TorchDataset(features, labels)

    def get_dataloader(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> Any:
        """Get PyTorch DataLoader.

        Args:
            split: Which split to use
            batch_size: Batch size
            shuffle: Whether to shuffle
            num_workers: Number of worker processes

        Returns:
            PyTorch DataLoader
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for get_dataloader")

        dataset = self.to_pytorch_dataset(split)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )


class DatasetRegistry:
    """Registry of available dataset loaders."""

    _loaders: dict[str, type[DatasetLoader]] = {}

    @classmethod
    def register(cls, name: str, loader_class: type[DatasetLoader]) -> None:
        """Register a dataset loader."""
        cls._loaders[name] = loader_class

    @classmethod
    def get(cls, name: str) -> type[DatasetLoader] | None:
        """Get a registered loader class."""
        return cls._loaders.get(name)

    @classmethod
    def list_datasets(cls) -> list[str]:
        """List all registered datasets."""
        return list(cls._loaders.keys())

    @classmethod
    def create(cls, name: str, config: DatasetConfig) -> DatasetLoader:
        """Create a dataset loader instance."""
        loader_class = cls.get(name)
        if loader_class is None:
            raise ValueError(f"Unknown dataset: {name}")
        return loader_class(config)
