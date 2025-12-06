"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

UCR Time Series Archive and Additional Benchmark Loaders

Loaders for academic time-series benchmarks:
- UCR Time Series Archive: 128+ univariate datasets
- MBA (Machine Bearing Anomaly): Industrial bearing data
- MSDS (Multi-Source Data Stream): Multi-domain streaming data

References:
    - Dau et al., "The UCR Time Series Archive", IEEE/CAA JAS 2019
    - CWRU Bearing Dataset, Case Western Reserve University
"""

from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from .base import DatasetConfig, DatasetLoader, DatasetMetadata, DatasetSplit

logger = logging.getLogger(__name__)

__all__ = [
    "UCRLoader",
    "MBALoader",
    "MSDSLoader",
    "CWRUBearingLoader",
]


class UCRLoader(DatasetLoader):
    """
    UCR Time Series Archive Loader.

    The UCR Time Series Archive is the largest collection of benchmark
    time-series datasets for classification. While primarily for classification,
    it's widely used for anomaly detection benchmarking.

    Data Source: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
    Paper: Dau et al., "The UCR Time Series Archive", IEEE/CAA JAS 2019

    Contains 128+ datasets across domains:
    - ECG/Medical
    - Motion capture
    - Sensor readings
    - Spectrograms
    - Simulations
    """

    DATASET_NAME = "ucr"
    DATASET_URL = "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/"
    LICENSE = "Academic Use"
    CITATION = """Dau HA, et al. The UCR Time Series Archive.
    IEEE/CAA Journal of Automatica Sinica, 2019."""
    REQUIRES_CREDENTIALS = False

    # Popular UCR datasets for anomaly detection
    POPULAR_DATASETS = [
        "ECG5000",
        "ECGFiveDays",
        "Wafer",
        "FordA",
        "FordB",
        "Earthquakes",
        "Strawberry",
        "Coffee",
        "SonyAIBORobotSurface1",
        "SonyAIBORobotSurface2",
    ]

    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        self.dataset_name = config.preprocessing.get("dataset_name", "ECG5000")

    def _load_raw(self) -> tuple[np.ndarray, np.ndarray]:
        """Load raw UCR data - redirects to load()."""
        return self.load()

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """Apply UCR-specific preprocessing (z-normalization)."""
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True) + 1e-8
        return (data - mean) / std

    def download(self) -> bool:
        """Download UCR archive (or specific dataset)."""
        import urllib.error
        import urllib.request

        logger.info(f"Downloading UCR dataset: {self.dataset_name}")

        # UCR archive URL
        archive_url = "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip"

        self.data_path.mkdir(parents=True, exist_ok=True)

        # Try to download specific dataset first (smaller)
        specific_url = f"https://www.timeseriesclassification.com/Downloads/{self.dataset_name}.zip"

        try:
            zip_path = self.data_path / f"{self.dataset_name}.zip"
            logger.info(f"  Trying dataset-specific download...")
            urllib.request.urlretrieve(specific_url, zip_path)

            # Extract
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(self.data_path)

            os.remove(zip_path)
            logger.info(f"  Downloaded {self.dataset_name}")
            return True

        except urllib.error.URLError:
            logger.warning(f"  Dataset-specific download failed")

        # Provide instructions for full archive
        logger.info("")
        logger.info("For full UCR archive, download from:")
        logger.info(f"  {archive_url}")
        logger.info(f"  Extract to: {self.data_path}")

        instructions_path = self.data_path / "DOWNLOAD_INSTRUCTIONS.txt"
        with open(instructions_path, "w") as f:
            f.write("UCR Time Series Archive Download\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Download from: {archive_url}\n")
            f.write(f"Extract to: {self.data_path}\n\n")
            f.write("Or use timeseriesclassification.com for individual datasets\n")

        return False

    def load(self, split: DatasetSplit = DatasetSplit.ALL) -> tuple[np.ndarray, np.ndarray]:
        """Load UCR dataset."""
        dataset_path = self.data_path / self.dataset_name

        # Check for extracted files
        train_file = dataset_path / f"{self.dataset_name}_TRAIN.tsv"
        test_file = dataset_path / f"{self.dataset_name}_TEST.tsv"

        if not train_file.exists():
            # Try alternate location
            train_file = self.data_path / f"{self.dataset_name}_TRAIN.tsv"
            test_file = self.data_path / f"{self.dataset_name}_TEST.tsv"

        if not train_file.exists():
            self.download()
            if not train_file.exists():
                raise FileNotFoundError(
                    f"UCR dataset {self.dataset_name} not found. "
                    "Please download from UCR archive."
                )

        # Load train and test
        train_data = np.loadtxt(train_file, delimiter="\t")
        test_data = np.loadtxt(test_file, delimiter="\t")

        # First column is label
        train_labels = train_data[:, 0].astype(int)
        train_features = train_data[:, 1:]

        test_labels = test_data[:, 0].astype(int)
        test_features = test_data[:, 1:]

        if split == DatasetSplit.TRAIN:
            return train_features, train_labels
        elif split == DatasetSplit.TEST:
            return test_features, test_labels
        else:
            features = np.vstack([train_features, test_features])
            labels = np.concatenate([train_labels, test_labels])
            return features, labels

    def convert_to_anomaly_labels(
        self, labels: np.ndarray, anomaly_class: int | None = None
    ) -> np.ndarray:
        """
        Convert classification labels to binary anomaly labels.

        Args:
            labels: Original class labels
            anomaly_class: Which class to treat as anomaly (default: minority)

        Returns:
            Binary anomaly labels (0=normal, 1=anomaly)
        """
        unique, counts = np.unique(labels, return_counts=True)

        if anomaly_class is None:
            # Treat minority class as anomaly
            anomaly_class = unique[np.argmin(counts)]

        return (labels == anomaly_class).astype(int)

    def get_metadata(self) -> DatasetMetadata:
        """Get dataset metadata."""
        return DatasetMetadata(
            name=f"UCR-{self.dataset_name}",
            version="2018",
            num_samples=0,  # Varies by dataset
            num_features=0,  # Varies by dataset
            feature_names=[],
            target_names=["Normal", "Anomaly"],
            class_distribution={},
            source_url=self.DATASET_URL,
            license=self.LICENSE,
            citation=self.CITATION,
            preprocessing_applied=[],
        )

    def list_available_datasets(self) -> list[str]:
        """List popular UCR datasets."""
        return self.POPULAR_DATASETS


class MBALoader(DatasetLoader):
    """
    Machine Bearing Anomaly (MBA) Dataset Loader.

    Based on the CWRU Bearing Dataset - the most widely used benchmark
    for mechanical fault detection.

    Contains vibration data from bearings with:
    - Normal operation
    - Inner race faults
    - Outer race faults
    - Ball faults
    - Various fault severities (0.007, 0.014, 0.021 inch)

    Data Source: https://engineering.case.edu/bearingdatacenter/download-data-file
    """

    DATASET_NAME = "mba"
    DATASET_URL = "https://engineering.case.edu/bearingdatacenter"
    LICENSE = "Public Domain"
    CITATION = """CWRU Bearing Data Center. Case Western Reserve University."""
    REQUIRES_CREDENTIALS = False

    # Sampling rate
    SAMPLE_RATE = 12000  # Hz for drive end

    # Fault types
    FAULT_TYPES = ["Normal", "Inner_Race", "Outer_Race", "Ball"]

    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        self.fault_type = config.preprocessing.get("fault_type", "all")
        self.load_rpm = config.preprocessing.get("load_rpm", 1797)

    def _load_raw(self) -> tuple[np.ndarray, np.ndarray]:
        """Load raw MBA/CWRU data - redirects to load()."""
        return self.load()

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """Apply MBA-specific preprocessing (z-normalization per window)."""
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True) + 1e-8
        return (data - mean) / std

    def download(self) -> bool:
        """Download CWRU bearing data."""
        import urllib.error
        import urllib.request

        from scipy.io import loadmat

        logger.info("Downloading CWRU Bearing Dataset (MBA)...")

        # CWRU data URLs for 12k drive end
        base_url = "https://engineering.case.edu/sites/default/files/"

        # Normal baseline data
        urls = {
            "normal": f"{base_url}97.mat",
            "inner_007": f"{base_url}105.mat",
            "ball_007": f"{base_url}118.mat",
            "outer_007": f"{base_url}130.mat",
        }

        self.data_path.mkdir(parents=True, exist_ok=True)

        for name, url in urls.items():
            output_path = self.data_path / f"{name}.mat"
            if not output_path.exists():
                try:
                    logger.info(f"  Downloading {name}...")
                    urllib.request.urlretrieve(url, output_path)
                except urllib.error.URLError as e:
                    logger.warning(f"  Failed: {e}")

        logger.info("CWRU download complete (partial)")
        logger.info("For full dataset, visit: " + self.DATASET_URL)

        return True

    def load(self, split: DatasetSplit = DatasetSplit.ALL) -> tuple[np.ndarray, np.ndarray]:
        """Load MBA/CWRU bearing data."""
        try:
            from scipy.io import loadmat
        except ImportError:
            raise ImportError("scipy required for MBA loading: pip install scipy")

        # Check for data files
        normal_file = self.data_path / "normal.mat"
        if not normal_file.exists():
            self.download()

        if not normal_file.exists():
            raise FileNotFoundError(
                f"MBA data not found in {self.data_path}. "
                "Download from CWRU Bearing Data Center."
            )

        features_list = []
        labels_list = []

        # Load and process each file
        fault_files = {
            0: "normal.mat",
            1: "inner_007.mat",
            2: "ball_007.mat",
            3: "outer_007.mat",
        }

        for label, filename in fault_files.items():
            file_path = self.data_path / filename
            if file_path.exists():
                mat_data = loadmat(str(file_path))

                # Find the vibration data key
                data_key = None
                for key in mat_data.keys():
                    if not key.startswith("_") and isinstance(mat_data[key], np.ndarray):
                        if mat_data[key].size > 1000:
                            data_key = key
                            break

                if data_key:
                    signal = mat_data[data_key].flatten()

                    # Segment into windows
                    window_size = 1024
                    n_windows = len(signal) // window_size

                    for i in range(n_windows):
                        window = signal[i * window_size : (i + 1) * window_size]
                        features_list.append(window)
                        labels_list.append(label)

        features = np.array(features_list, dtype=np.float32)
        labels = np.array(labels_list, dtype=np.int32)

        # Convert to binary (normal vs fault)
        binary_labels = (labels > 0).astype(int)

        logger.info(f"Loaded MBA: {features.shape[0]} samples, {features.shape[1]} features")
        logger.info(f"Anomaly ratio: {binary_labels.mean():.2%}")

        return features, binary_labels

    def get_metadata(self) -> DatasetMetadata:
        """Get dataset metadata."""
        return DatasetMetadata(
            name="MBA (CWRU Bearing)",
            version="1.0",
            num_samples=0,  # Varies
            num_features=1024,
            feature_names=[f"vibration_{i}" for i in range(1024)],
            target_names=["Normal", "Fault"],
            class_distribution={},
            source_url=self.DATASET_URL,
            license=self.LICENSE,
            citation=self.CITATION,
            preprocessing_applied=["windowing"],
        )


class CWRUBearingLoader(MBALoader):
    """Alias for MBA loader (CWRU Bearing Data)."""

    DATASET_NAME = "cwru_bearing"


class MSDSLoader(DatasetLoader):
    """
    Multi-Source Data Stream (MSDS) Dataset Loader.

    Synthetic benchmark for multi-domain anomaly detection,
    combining multiple data sources with correlated anomalies.

    Used for testing fusion methods that combine information
    from heterogeneous sources.

    Note: This provides a synthetic generation mechanism for
    multi-source testing when real multi-source data is unavailable.
    """

    DATASET_NAME = "msds"
    DATASET_URL = "https://github.com/imperial-qore/TranAD"  # Included in TranAD repo
    LICENSE = "Apache-2.0"
    CITATION = """Multi-Source Data Stream synthetic benchmark."""
    REQUIRES_CREDENTIALS = False

    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        self.n_sources = config.preprocessing.get("n_sources", 3)
        self.n_samples = config.preprocessing.get("n_samples", 10000)
        self.anomaly_ratio = config.preprocessing.get("anomaly_ratio", 0.05)

    def _load_raw(self) -> tuple[np.ndarray, np.ndarray]:
        """Load raw MSDS data - redirects to load()."""
        return self.load()

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """Apply MSDS-specific preprocessing (z-normalization per feature)."""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-8
        return (data - mean) / std

    def download(self) -> bool:
        """Generate synthetic MSDS data."""
        logger.info("MSDS: Generating synthetic multi-source data...")

        self.data_path.mkdir(parents=True, exist_ok=True)

        # Generate correlated multi-source data
        np.random.seed(42)

        n_features_per_source = 10
        total_features = n_features_per_source * self.n_sources

        # Base signal (shared latent)
        t = np.linspace(0, 100, self.n_samples)
        base = np.sin(0.1 * t) + 0.5 * np.sin(0.3 * t)

        # Generate sources with shared component
        features = []
        for s in range(self.n_sources):
            source_data = np.zeros((self.n_samples, n_features_per_source))
            for f in range(n_features_per_source):
                # Mix of shared and unique patterns
                shared = 0.5 * base
                unique = np.sin(0.2 * (s + 1) * (f + 1) * t / 10)
                noise = 0.1 * np.random.randn(self.n_samples)
                source_data[:, f] = shared + unique + noise
            features.append(source_data)

        features = np.hstack(features)

        # Generate correlated anomalies
        n_anomalies = int(self.n_samples * self.anomaly_ratio)
        anomaly_indices = np.random.choice(self.n_samples, n_anomalies, replace=False)

        labels = np.zeros(self.n_samples)
        labels[anomaly_indices] = 1

        # Inject anomalies (affect multiple sources)
        for idx in anomaly_indices:
            # Random anomaly type
            anomaly_type = np.random.choice(["spike", "drift", "noise"])
            affected_sources = np.random.choice(
                self.n_sources, np.random.randint(1, self.n_sources + 1), replace=False
            )

            for source in affected_sources:
                start = source * n_features_per_source
                end = start + n_features_per_source

                if anomaly_type == "spike":
                    features[idx, start:end] += np.random.randn(n_features_per_source) * 5
                elif anomaly_type == "drift":
                    features[idx, start:end] += 3
                else:
                    features[idx, start:end] += np.random.randn(n_features_per_source) * 2

        # Save
        np.savez(
            self.data_path / "msds_data.npz",
            features=features.astype(np.float32),
            labels=labels.astype(np.int32),
        )

        logger.info(f"Generated MSDS: {self.n_samples} samples, {total_features} features")
        return True

    def load(self, split: DatasetSplit = DatasetSplit.ALL) -> tuple[np.ndarray, np.ndarray]:
        """Load MSDS data."""
        data_file = self.data_path / "msds_data.npz"

        if not data_file.exists():
            self.download()

        data = np.load(data_file)
        features = data["features"]
        labels = data["labels"]

        logger.info(f"Loaded MSDS: {features.shape[0]} samples, {features.shape[1]} features")
        logger.info(f"Anomaly ratio: {labels.mean():.2%}")

        return features, labels

    def get_metadata(self) -> DatasetMetadata:
        """Get dataset metadata."""
        return DatasetMetadata(
            name="MSDS",
            version="synthetic",
            num_samples=self.n_samples,
            num_features=self.n_sources * 10,
            feature_names=[f"source{s}_feat{f}" for s in range(self.n_sources) for f in range(10)],
            target_names=["Normal", "Anomaly"],
            class_distribution={"normal": 1 - self.anomaly_ratio, "anomaly": self.anomaly_ratio},
            source_url=self.DATASET_URL,
            license=self.LICENSE,
            citation=self.CITATION,
            preprocessing_applied=["synthetic_generation"],
        )
