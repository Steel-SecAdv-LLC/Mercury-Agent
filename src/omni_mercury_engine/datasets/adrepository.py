"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ADRepository Dataset Loaders - REAL Anomaly Detection Datasets

This module provides loaders for the ADRepository collection of real-world
anomaly detection datasets. These are REAL datasets with REAL anomalies,
used in academic benchmarks.

Repository: https://github.com/GuansongPang/ADRepository-Anomaly-detection-datasets
Paper: Pang et al., "Deep Learning for Anomaly Detection: A Review",
       ACM Computing Surveys, 2021.

Datasets include:
- Tabular: fraud, backdoor, campaign, thyroid, donors, census, celeba
- Time Series: SMD, SWAT, DSADS, Epilepsy
- Graph: Multiple graph-level anomaly detection datasets
"""

from __future__ import annotations

import logging
import zipfile
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


class DatasetMetadata(TypedDict):
    """Type definition for dataset metadata."""

    samples: int
    features: int
    anomaly_ratio: float
    domain: str
    description: str
    url: str
    file: str


class ODDSDatasetInfo(TypedDict, total=False):
    """Type definition for ODDS dataset info."""

    url: str
    format: str
    requires_auth: bool
    instructions: str


from .base import DatasetConfig, DatasetLoader, DatasetRegistry, safe_urlretrieve

logger = logging.getLogger(__name__)


# =============================================================================
# ADRepository Dataset Metadata
# =============================================================================

ADREPOSITORY_DATASETS: dict[str, DatasetMetadata] = {
    # Tabular datasets (DevNet collection)
    "fraud": {
        "samples": 284807,
        "features": 29,
        "anomaly_ratio": 0.00173,
        "domain": "finance",
        "description": "Credit card fraud detection dataset",
        "url": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
        "file": "creditcard.csv",
    },
    "backdoor": {
        "samples": 95329,
        "features": 196,
        "anomaly_ratio": 0.0244,
        "domain": "security",
        "description": "Network intrusion backdoor detection",
        "url": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
        "file": "UNSW_NB15_training-set.csv",
    },
    "campaign": {
        "samples": 41188,
        "features": 62,
        "anomaly_ratio": 0.1127,
        "domain": "marketing",
        "description": "Bank marketing campaign success prediction",
        "url": "https://archive.ics.uci.edu/ml/datasets/bank+marketing",
        "file": "bank-additional-full.csv",
    },
    "thyroid": {
        "samples": 7200,
        "features": 21,
        "anomaly_ratio": 0.0244,
        "domain": "medical",
        "description": "Thyroid disease detection",
        "url": "https://archive.ics.uci.edu/ml/datasets/thyroid+disease",
        "file": "thyroid.npz",
    },
    "donors": {
        "samples": 619326,
        "features": 10,
        "anomaly_ratio": 0.059,
        "domain": "nonprofit",
        "description": "KDD Cup 2014 donor prediction",
        "url": "https://www.kaggle.com/c/kdd-cup-2014-predicting-excitement-at-donors-choose",
        "file": "donors.npz",
    },
    "census": {
        "samples": 299285,
        "features": 500,
        "anomaly_ratio": 0.06,
        "domain": "demographics",
        "description": "US Census income prediction (high-income as anomaly)",
        "url": "https://archive.ics.uci.edu/ml/datasets/census+income",
        "file": "census.npz",
    },
    "celeba": {
        "samples": 202599,
        "features": 39,
        "anomaly_ratio": 0.0227,
        "domain": "vision",
        "description": "Celebrity face attributes (bald as anomaly)",
        "url": "http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html",
        "file": "celeba.npz",
    },
    # Time series datasets
    "smd": {
        "samples": 708405,
        "features": 38,
        "anomaly_ratio": 0.042,
        "domain": "infrastructure",
        "description": "Server Machine Dataset - real server metrics",
        "url": "https://github.com/NetManAIOps/OmniAnomaly",
        "file": "SMD.zip",
    },
    "swat": {
        "samples": 496800,
        "features": 51,
        "anomaly_ratio": 0.112,
        "domain": "industrial",
        "description": "Secure Water Treatment testbed",
        "url": "https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/",
        "file": "SWaT.zip",
    },
    "dsads": {
        "samples": 9120,
        "features": 405,
        "anomaly_ratio": 0.066,
        "domain": "activity",
        "description": "Daily and Sports Activities Dataset",
        "url": "https://archive.ics.uci.edu/ml/datasets/daily+and+sports+activities",
        "file": "DSADS.npz",
    },
    "epilepsy": {
        "samples": 11500,
        "features": 178,
        "anomaly_ratio": 0.2,
        "domain": "medical",
        "description": "Epileptic seizure detection from EEG",
        "url": "https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition",
        "file": "epilepsy.npz",
    },
}


class ADRepositoryLoader(DatasetLoader):
    """
    Loader for ADRepository real-world anomaly detection datasets.

    This loader fetches REAL datasets from the ADRepository collection,
    which are standard benchmarks used in academic anomaly detection research.

    Reference:
        Pang, G., Shen, C., Cao, L., & Hengel, A. V. D. (2021).
        Deep learning for anomaly detection: A review.
        ACM Computing Surveys (CSUR), 54(2), 1-38.

    Repository:
        https://github.com/GuansongPang/ADRepository-Anomaly-detection-datasets
    """

    DATASET_NAME = "adrepository"
    DATASET_URL = "https://github.com/GuansongPang/ADRepository-Anomaly-detection-datasets"
    LICENSE = "Various (see individual datasets)"
    CITATION = """Pang G, Shen C, Cao L, Hengel AVD. Deep learning for anomaly detection:
    A review. ACM Computing Surveys. 2021;54(2):1-38."""
    REQUIRES_CREDENTIALS = False

    # Mirror URLs for preprocessed datasets
    MIRROR_URLS = {
        "primary": "https://github.com/GuansongPang/ADRepository-Anomaly-detection-datasets/raw/main/",
        "fallback": "https://raw.githubusercontent.com/mala-lab/ADRepository-Anomaly-detection-datasets/main/",
    }

    # ODDS (Outlier Detection DataSets) - Stony Brook University
    # These are VERIFIED working URLs for real anomaly detection datasets
    # Reference: https://odds.cs.stonybrook.edu/
    ODDS_URLS: dict[str, ODDSDatasetInfo] = {
        "thyroid": {
            "url": "https://odds.cs.stonybrook.edu/wp-content/uploads/2016/04/thyroid.mat",
            "format": "mat",
        },
        "smtp": {
            "url": "https://odds.cs.stonybrook.edu/wp-content/uploads/2016/04/smtp.mat",
            "format": "mat",
        },
        "satellite": {
            "url": "https://odds.cs.stonybrook.edu/wp-content/uploads/2016/04/satellite.mat",
            "format": "mat",
        },
        "pendigits": {
            "url": "https://odds.cs.stonybrook.edu/wp-content/uploads/2016/04/pendigits.mat",
            "format": "mat",
        },
        "mammography": {
            "url": "https://odds.cs.stonybrook.edu/wp-content/uploads/2016/04/mammography.mat",
            "format": "mat",
        },
        "shuttle": {
            "url": "https://odds.cs.stonybrook.edu/wp-content/uploads/2016/04/shuttle.mat",
            "format": "mat",
        },
        "fraud": {
            "url": "kaggle://mlg-ulb/creditcardfraud/creditcard.csv",
            "format": "csv",
            "requires_auth": True,
            "instructions": "Run: kaggle datasets download -d mlg-ulb/creditcardfraud",
        },
    }

    def __init__(self, config: DatasetConfig, dataset_name: str = "thyroid") -> None:
        """
        Initialize ADRepository loader.

        Args:
            config: Dataset configuration
            dataset_name: Name of dataset to load (e.g., 'fraud', 'thyroid', 'smd')
        """
        super().__init__(config)
        self.dataset_name = dataset_name.lower()

        if self.dataset_name not in ADREPOSITORY_DATASETS:
            available = ", ".join(ADREPOSITORY_DATASETS.keys())
            raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")

        self.dataset_info = ADREPOSITORY_DATASETS[self.dataset_name]
        self._features: np.ndarray | None = None
        self._labels: np.ndarray | None = None  # type: ignore[assignment, unused-ignore]
        self._is_real_data = False

        logger.info(
            f"ADRepositoryLoader initialized for '{dataset_name}' "
            f"({self.dataset_info['samples']} samples, "
            f"{self.dataset_info['features']} features, "
            f"{self.dataset_info['anomaly_ratio']*100:.2f}% anomalies)"
        )

    @property
    def is_real_data(self) -> bool:
        """Return True if real data was loaded (not synthetic fallback)."""
        return self._is_real_data

    def download(self) -> bool:
        """
        Download dataset from ADRepository or original source.

        Returns:
            True if download successful, False otherwise.
        """
        try:
            return self._download_from_repository()
        except Exception as e:
            logger.warning(f"Failed to download real data: {e}")
            logger.info("Falling back to synthetic approximation")
            return self._create_synthetic_fallback()

    def _load_raw(self) -> tuple[np.ndarray, np.ndarray]:
        """Load raw data from files (implements abstract method)."""
        dataset_dir = self.data_path / self.dataset_name
        filename = self.dataset_info["file"]
        local_path = dataset_dir / filename

        # Check for ODDS-downloaded .mat file first
        if self.dataset_name in self.ODDS_URLS:
            odds_info = self.ODDS_URLS[self.dataset_name]
            odds_path = dataset_dir / f"{self.dataset_name}.{odds_info['format']}"
            if odds_path.exists():
                self._load_from_file(odds_path)
                if self._features is not None and self._labels is not None:
                    return self._features, self._labels

        if not local_path.exists():
            self.download()

        # Check again for ODDS file after download
        if self.dataset_name in self.ODDS_URLS:
            odds_info = self.ODDS_URLS[self.dataset_name]
            odds_path = dataset_dir / f"{self.dataset_name}.{odds_info['format']}"
            if odds_path.exists():
                self._load_from_file(odds_path)
                if self._features is not None and self._labels is not None:
                    return self._features, self._labels

        if local_path.exists():
            self._load_from_file(local_path)

        if self._features is None:
            # Use synthetic fallback
            self._create_synthetic_fallback()

        # Type guard for mypy - at this point both should be set
        if self._features is None or self._labels is None:
            raise RuntimeError("Failed to load dataset features and labels")

        return self._features, self._labels

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """Apply preprocessing (implements abstract method)."""
        # Basic normalization - zero mean, unit variance
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True) + 1e-8
        return np.asarray((data - mean) / std)  # type: ignore[no-any-return, unused-ignore]

    def _download_from_repository(self) -> bool:
        """Download from ODDS or ADRepository GitHub."""
        dataset_dir = self.data_path / self.dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Try ODDS URLs first (verified working)
        if self.dataset_name in self.ODDS_URLS:
            odds_info = self.ODDS_URLS[self.dataset_name]

            # Check if requires authentication (e.g., Kaggle)
            if odds_info.get("requires_auth"):
                instructions = odds_info.get("instructions", "Authentication required")
                logger.warning(
                    f"Dataset '{self.dataset_name}' requires authentication. " f"{instructions}"
                )
                raise ValueError(
                    f"Dataset '{self.dataset_name}' requires authentication. " f"{instructions}"
                )

            url = odds_info["url"]
            file_ext = odds_info["format"]
            local_path = dataset_dir / f"{self.dataset_name}.{file_ext}"

            if local_path.exists():
                logger.info(f"Dataset already exists at {local_path}")
                self._is_real_data = True
                return True

            try:
                logger.info(f"Downloading {self.dataset_name} from ODDS: {url}")
                safe_urlretrieve(url, str(local_path))
                self._is_real_data = True
                logger.info(f"Successfully downloaded ODDS dataset to {local_path}")
                return True
            except Exception as e:
                logger.warning(f"ODDS download failed: {e}")

        # Fallback to ADRepository GitHub mirrors
        # Determine folder based on dataset type
        if self.dataset_name in ["smd", "swat", "dsads", "epilepsy"]:
            folder = "Time%20Series"
        elif self.dataset_name in [
            "fraud",
            "backdoor",
            "campaign",
            "thyroid",
            "donors",
            "census",
            "celeba",
        ]:
            folder = "Numerical%20Data%20(DevNet)"
        else:
            folder = "Numerical%20Data%20(DevNet)"

        filename = self.dataset_info["file"]
        local_path = dataset_dir / filename

        if local_path.exists():
            logger.info(f"Dataset already exists at {local_path}")
            self._is_real_data = True
            return True

        # Try primary mirror
        url = f"{self.MIRROR_URLS['primary']}{folder}/{filename}"
        try:
            logger.info(f"Downloading {self.dataset_name} from {url}")
            safe_urlretrieve(url, str(local_path))
            self._is_real_data = True
            logger.info(f"Successfully downloaded to {local_path}")
            return True
        except Exception as e:
            logger.warning(f"Primary mirror failed: {e}")

        # Try fallback mirror
        url = f"{self.MIRROR_URLS['fallback']}{folder}/{filename}"
        try:
            logger.info(f"Trying fallback mirror: {url}")
            safe_urlretrieve(url, str(local_path))
            self._is_real_data = True
            logger.info("Successfully downloaded from fallback")
            return True
        except Exception as e:
            logger.warning(f"Fallback mirror failed: {e}")
            raise

    def _create_synthetic_fallback(self) -> bool:
        """
        Create synthetic approximation when real data unavailable.

        This is a FALLBACK only. Real data should be preferred.
        """
        logger.warning(
            f"Creating SYNTHETIC approximation for {self.dataset_name}. "
            "Results may not reflect real-world performance."
        )

        np.random.seed(self.config.random_seed)

        info = self.dataset_info
        n_samples = min(info["samples"], self.config.max_samples or info["samples"])
        n_features = info["features"]
        anomaly_ratio = info["anomaly_ratio"]

        n_anomalies = int(n_samples * anomaly_ratio)
        n_normal = n_samples - n_anomalies

        # Generate normal samples
        normal_data = np.random.randn(n_normal, n_features)

        # Generate anomalies (shifted distribution)
        anomaly_data = np.random.randn(n_anomalies, n_features) * 2 + 3

        # Combine
        self._features = np.vstack([normal_data, anomaly_data]).astype(np.float32)
        self._labels = np.array([0] * n_normal + [1] * n_anomalies, dtype=np.int64)

        # Shuffle
        perm = np.random.permutation(n_samples)
        self._features = self._features[perm]
        self._labels = self._labels[perm]

        self._is_real_data = False
        return True

    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load dataset features and labels.

        This is the main entry point for loading ADRepository datasets.
        Use this instead of the base class load() for simpler access.

        Returns:
            Tuple of (features, labels) numpy arrays.
        """
        if self._features is not None and self._labels is not None:
            return self._features, self._labels

        return self._load_raw()

    def _load_mat_file(self, path: Path) -> None:
        """Load MATLAB .mat file from ODDS repository."""
        from scipy.io import loadmat

        data = loadmat(str(path))
        self._features = data["X"].astype(np.float32)
        self._labels = data["y"].ravel().astype(np.int64)
        self._is_real_data = True
        logger.info(f"Loaded .mat file from {path.name} (real_data=True)")

    def _load_from_file(self, path: Path) -> None:
        """Load data from downloaded file."""
        suffix = path.suffix.lower()

        try:
            if suffix == ".mat":
                self._load_mat_file(path)

            elif suffix == ".npz":
                # Security: External dataset files - try safe load first
                try:
                    data = np.load(path, allow_pickle=False)
                except ValueError:
                    logger.warning(f"Dataset {path} requires pickle - verify source is trusted")
                    data = np.load(path, allow_pickle=True)  # nosec B301
                self._features = data["X"].astype(np.float32)
                self._labels = data["y"].astype(np.int64)
                self._is_real_data = True

            elif suffix == ".csv":
                import pandas as pd

                df = pd.read_csv(path)

                # Assume last column is label
                self._features = df.iloc[:, :-1].values.astype(np.float32)
                self._labels = df.iloc[:, -1].values.astype(np.int64)
                self._is_real_data = True

            elif suffix == ".zip":
                # Extract and load
                extract_dir = path.parent / path.stem
                with zipfile.ZipFile(path, "r") as zf:
                    zf.extractall(extract_dir)

                # Find npz or csv files
                for f in extract_dir.rglob("*.npz"):
                    # Security: External dataset files - try safe load first
                    try:
                        data = np.load(f, allow_pickle=False)
                    except ValueError:
                        logger.warning(f"Dataset {f} requires pickle - verify source")
                        data = np.load(f, allow_pickle=True)  # nosec B301
                    if "X" in data and "y" in data:
                        self._features = data["X"].astype(np.float32)
                        self._labels = data["y"].astype(np.int64)
                        self._is_real_data = True
                        break

            # Apply max_samples limit
            if self._features is not None and self._labels is not None and self.config.max_samples:
                n = min(len(self._features), self.config.max_samples)
                self._features = self._features[:n]
                self._labels = self._labels[:n]

            if self._features is not None:
                logger.info(
                    f"Loaded {len(self._features)} samples from {path.name} "
                    f"(real_data={self._is_real_data})"
                )

        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            self._create_synthetic_fallback()

    def get_metadata(self) -> dict[str, Any]:
        """Get dataset metadata."""
        info = self.dataset_info
        return {
            "name": self.dataset_name,
            "source": "ADRepository",
            "samples": info["samples"],
            "features": info["features"],
            "anomaly_ratio": info["anomaly_ratio"],
            "domain": info["domain"],
            "description": info["description"],
            "original_url": info["url"],
            "is_real_data": self._is_real_data,
            "citation": self.CITATION,
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about loaded data."""
        if self._features is None:
            self.load_data()

        # Type guards for mypy - load_data() ensures these are not None
        if self._features is None or self._labels is None:
            raise RuntimeError("Failed to load data")

        return {
            "n_samples": len(self._features),
            "n_features": self._features.shape[1],
            "n_anomalies": int(self._labels.sum()),
            "anomaly_ratio": float(self._labels.mean()),
            "feature_mean": float(self._features.mean()),
            "feature_std": float(self._features.std()),
            "is_real_data": self._is_real_data,
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def list_available_datasets() -> dict[str, DatasetMetadata]:
    """List all available ADRepository datasets with metadata."""
    return ADREPOSITORY_DATASETS.copy()


def load_dataset(
    name: str,
    data_dir: str = "./data/adrepository",
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Convenience function to load an ADRepository dataset.

    Args:
        name: Dataset name (e.g., 'fraud', 'thyroid', 'smd')
        data_dir: Directory to store downloaded data
        max_samples: Maximum samples to load

    Returns:
        Tuple of (features, labels, metadata)

    Example:
        >>> X, y, meta = load_dataset('thyroid')
        >>> print(f"Loaded {len(X)} samples, {meta['anomaly_ratio']*100:.1f}% anomalies")
    """
    config = DatasetConfig(
        name=name,
        data_dir=data_dir,
        max_samples=max_samples,
    )

    loader = ADRepositoryLoader(config, dataset_name=name)
    X, y = loader.load_data()
    metadata = loader.get_metadata()

    return X, y, metadata


# Register datasets
for dataset_name in ADREPOSITORY_DATASETS:

    def _make_loader(dn: str) -> DatasetLoader:
        def _factory(cfg: DatasetConfig) -> DatasetLoader:
            return ADRepositoryLoader(cfg, dataset_name=dn)

        return _factory  # type: ignore[return-value]

    DatasetRegistry.register(f"adrepository-{dataset_name}", _make_loader(dataset_name))  # type: ignore[arg-type]
