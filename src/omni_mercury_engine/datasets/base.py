"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

Base classes for real-world dataset loading and management.
"""

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
    """Safely download a file from a URL with scheme validation.

    Only allows https:// and http:// schemes to prevent file:// or other
    potentially dangerous URL schemes.

    Args:
        url: The URL to download from (must be http:// or https://)
        filename: The local path to save the file to

    Raises:
        ValueError: If the URL scheme is not http or https
    """
    import urllib.request
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"URL scheme must be http or https, got: {parsed.scheme}")

    urllib.request.urlretrieve(url, filename)  # noqa: S310  # nosec B310


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
        # Using SHA-256 instead of MD5 to satisfy security scanners (CodeQL/ruff S324)
        # Note: This is non-cryptographic use for cache key generation
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]


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
            # Security: Cache files are self-generated, should be pure numpy arrays
            # Use allow_pickle=False for safety; fall back only if legacy cache exists
            try:
                cached = np.load(cache_file, allow_pickle=False)
            except ValueError:
                logger.warning(f"Legacy cache format detected, loading with pickle")
                cached = np.load(cache_file, allow_pickle=True)  # nosec B301
            self._data = {
                DatasetSplit.TRAIN: cached["train_features"],
                DatasetSplit.VALIDATION: cached["val_features"],
                DatasetSplit.TEST: cached["test_features"],
            }
            self._labels = {
                DatasetSplit.TRAIN: cached["train_labels"],
                DatasetSplit.VALIDATION: cached["val_labels"],
                DatasetSplit.TEST: cached["test_labels"],
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
        np.random.seed(self.config.random_seed)
        indices = np.random.permutation(n)

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

    def get_metadata(self) -> DatasetMetadata:
        """Get dataset metadata."""
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

        class TorchDataset(Dataset):  # type: ignore[type-arg]
            def __init__(self, X: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> None:
                self.X = torch.FloatTensor(X)
                self.y = torch.LongTensor(y)

            def __len__(self) -> int:
                return len(self.X)

            def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
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
