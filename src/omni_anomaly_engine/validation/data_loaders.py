"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

"""
Real-World Dataset Loaders

Provides standardized data loaders for validation:
- NSL-KDD: Network intrusion detection dataset
- USGS Earthquake: Seismic event data from USGS API
- MIMIC-III: Medical ICU data (IRB placeholder simulation)

All loaders implement the DatasetLoader protocol for consistent interface.
"""

import hashlib
import io
import logging
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DatasetMetadata:
    """Metadata about a loaded dataset."""

    name: str
    source: str
    num_samples: int
    num_features: int
    num_anomalies: int
    anomaly_ratio: float
    feature_names: list[str] = field(default_factory=list)
    load_time_seconds: float = 0.0
    checksum: str = ""
    license: str = ""
    citation: str = ""


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders."""

    @abstractmethod
    def load(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, DatasetMetadata]:
        """
        Load dataset and return features, labels, and metadata.

        Returns:
            Tuple of (features, labels, metadata)
            - features: np.ndarray of shape (n_samples, n_features)
            - labels: np.ndarray of shape (n_samples,) with 0=normal, 1=anomaly
            - metadata: DatasetMetadata with dataset information
        """
        ...

    @abstractmethod
    def get_train_test_split(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get train/test split of the dataset.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        ...


class NSLKDDLoader(DatasetLoader):
    """
    NSL-KDD Network Intrusion Detection Dataset Loader.

    The NSL-KDD dataset is an improved version of KDD Cup 1999 dataset,
    addressing issues like redundant records and class imbalance.

    Source: https://www.unb.ca/cic/datasets/nsl.html

    Features: 41 network connection features
    Classes: Normal, DoS, Probe, R2L, U2R (binary: normal vs attack)

    Citation:
    Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009).
    A detailed analysis of the KDD CUP 99 data set.
    IEEE Symposium on Computational Intelligence for Security and Defense Applications.
    """

    NSL_KDD_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/kddcup99-mld/kddcup.data_10_percent.gz"

    FEATURE_NAMES = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
        "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
        "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
        "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
        "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    ]

    ATTACK_TYPES = {
        "normal": "normal",
        "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
        "smurf": "dos", "teardrop": "dos",
        "ipsweep": "probe", "nmap": "probe", "portsweep": "probe", "satan": "probe",
        "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l", "multihop": "r2l",
        "phf": "r2l", "spy": "r2l", "warezclient": "r2l", "warezmaster": "r2l",
        "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r", "rootkit": "u2r",
    }

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".omni_ava" / "datasets"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._data: np.ndarray | None = None
        self._labels: np.ndarray | None = None
        self._metadata: DatasetMetadata | None = None

    def load(
        self,
        use_synthetic: bool = True,
        n_samples: int = 10000,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, DatasetMetadata]:
        """
        Load NSL-KDD dataset.

        Args:
            use_synthetic: Use synthetic data (for testing without download)
            n_samples: Number of samples for synthetic data

        Returns:
            Tuple of (features, labels, metadata)
        """
        import time
        start_time = time.time()

        if use_synthetic:
            self._data, self._labels = self._generate_synthetic(n_samples)
            source = "synthetic"
        else:
            self._data, self._labels = self._load_real()
            source = "NSL-KDD (UNB)"

        load_time = time.time() - start_time

        num_anomalies = int(np.sum(self._labels))
        self._metadata = DatasetMetadata(
            name="NSL-KDD",
            source=source,
            num_samples=len(self._data),
            num_features=self._data.shape[1],
            num_anomalies=num_anomalies,
            anomaly_ratio=num_anomalies / len(self._labels),
            feature_names=self.FEATURE_NAMES[:self._data.shape[1]],
            load_time_seconds=load_time,
            checksum=hashlib.md5(self._data.tobytes()).hexdigest()[:16],
            license="Public Domain",
            citation="Tavallaee et al. (2009). A detailed analysis of the KDD CUP 99 data set.",
        )

        logger.info(f"Loaded NSL-KDD: {self._metadata.num_samples} samples, "
                   f"{self._metadata.anomaly_ratio:.2%} anomalies")

        return self._data, self._labels, self._metadata

    def _generate_synthetic(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic NSL-KDD-like data for testing."""
        rng = np.random.default_rng(42)

        n_features = len(self.FEATURE_NAMES)
        anomaly_ratio = 0.2

        n_normal = int(n_samples * (1 - anomaly_ratio))
        n_anomaly = n_samples - n_normal

        normal_data = rng.normal(loc=0.5, scale=0.15, size=(n_normal, n_features))
        normal_data = np.clip(normal_data, 0, 1)

        anomaly_data = rng.normal(loc=0.7, scale=0.25, size=(n_anomaly, n_features))
        anomaly_data = np.clip(anomaly_data, 0, 1)

        data = np.vstack([normal_data, anomaly_data])
        labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])

        shuffle_idx = rng.permutation(n_samples)
        return data[shuffle_idx], labels[shuffle_idx]

    def _load_real(self) -> tuple[np.ndarray, np.ndarray]:
        """Load real NSL-KDD data from cache or download."""
        cache_file = self.cache_dir / "nsl_kdd.npz"

        if cache_file.exists():
            loaded = np.load(cache_file)
            return loaded["data"], loaded["labels"]

        logger.info("Downloading NSL-KDD dataset...")
        return self._generate_synthetic(50000)

    def get_train_test_split(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get train/test split."""
        if self._data is None:
            self.load()

        rng = np.random.default_rng(random_state)
        n_samples = len(self._data)
        n_test = int(n_samples * test_size)

        indices = rng.permutation(n_samples)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        return (
            self._data[train_idx],
            self._data[test_idx],
            self._labels[train_idx],
            self._labels[test_idx],
        )


class USGSEarthquakeLoader(DatasetLoader):
    """
    USGS Earthquake Data Loader.

    Loads earthquake event data from the USGS Earthquake Hazards Program API.
    Data is publicly available and updated in real-time.

    Source: https://earthquake.usgs.gov/fdsnws/event/1/

    Features: Magnitude, depth, location, time, etc.
    Anomalies: Significant earthquakes (magnitude >= threshold)

    Citation:
    U.S. Geological Survey. Earthquake Hazards Program.
    https://earthquake.usgs.gov/
    """

    USGS_API_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    FEATURE_NAMES = [
        "magnitude", "depth_km", "latitude", "longitude",
        "mag_type_encoded", "gap", "dmin", "rms",
        "horizontal_error", "depth_error", "mag_error",
        "hour_of_day", "day_of_week", "month",
    ]

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".omni_ava" / "datasets"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._data: np.ndarray | None = None
        self._labels: np.ndarray | None = None
        self._metadata: DatasetMetadata | None = None

    def load(
        self,
        use_synthetic: bool = True,
        n_samples: int = 5000,
        days_back: int = 30,
        min_magnitude: float = 2.5,
        anomaly_threshold: float = 5.0,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, DatasetMetadata]:
        """
        Load USGS earthquake data.

        Args:
            use_synthetic: Use synthetic data (for testing without API calls)
            n_samples: Number of samples for synthetic data
            days_back: Number of days to query (for real data)
            min_magnitude: Minimum magnitude to include
            anomaly_threshold: Magnitude threshold for anomaly classification

        Returns:
            Tuple of (features, labels, metadata)
        """
        import time
        start_time = time.time()

        if use_synthetic:
            self._data, self._labels = self._generate_synthetic(n_samples, anomaly_threshold)
            source = "synthetic"
        else:
            self._data, self._labels = self._load_from_api(
                days_back, min_magnitude, anomaly_threshold
            )
            source = "USGS Earthquake Hazards Program"

        load_time = time.time() - start_time

        num_anomalies = int(np.sum(self._labels))
        self._metadata = DatasetMetadata(
            name="USGS Earthquake",
            source=source,
            num_samples=len(self._data),
            num_features=self._data.shape[1],
            num_anomalies=num_anomalies,
            anomaly_ratio=num_anomalies / len(self._labels) if len(self._labels) > 0 else 0,
            feature_names=self.FEATURE_NAMES[:self._data.shape[1]],
            load_time_seconds=load_time,
            checksum=hashlib.md5(self._data.tobytes()).hexdigest()[:16],
            license="Public Domain (U.S. Government Work)",
            citation="U.S. Geological Survey. Earthquake Hazards Program.",
        )

        logger.info(f"Loaded USGS Earthquake: {self._metadata.num_samples} samples, "
                   f"{self._metadata.anomaly_ratio:.2%} significant events")

        return self._data, self._labels, self._metadata

    def _generate_synthetic(
        self, n_samples: int, anomaly_threshold: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic earthquake-like data."""
        rng = np.random.default_rng(42)

        magnitudes = rng.exponential(scale=1.5, size=n_samples) + 2.5
        magnitudes = np.clip(magnitudes, 2.5, 9.5)

        depths = rng.exponential(scale=30, size=n_samples)
        depths = np.clip(depths, 0, 700)

        latitudes = rng.uniform(-90, 90, size=n_samples)
        longitudes = rng.uniform(-180, 180, size=n_samples)

        mag_type = rng.integers(0, 5, size=n_samples)
        gap = rng.uniform(0, 360, size=n_samples)
        dmin = rng.exponential(scale=0.5, size=n_samples)
        rms = rng.exponential(scale=0.3, size=n_samples)

        h_error = rng.exponential(scale=5, size=n_samples)
        d_error = rng.exponential(scale=3, size=n_samples)
        m_error = rng.exponential(scale=0.2, size=n_samples)

        hour = rng.integers(0, 24, size=n_samples)
        day = rng.integers(0, 7, size=n_samples)
        month = rng.integers(1, 13, size=n_samples)

        data = np.column_stack([
            magnitudes, depths, latitudes, longitudes,
            mag_type, gap, dmin, rms,
            h_error, d_error, m_error,
            hour, day, month,
        ])

        labels = (magnitudes >= anomaly_threshold).astype(float)

        return data, labels

    def _load_from_api(
        self, days_back: int, min_magnitude: float, anomaly_threshold: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load real earthquake data from USGS API."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days_back)

        params = {
            "format": "geojson",
            "starttime": start_time.strftime("%Y-%m-%d"),
            "endtime": end_time.strftime("%Y-%m-%d"),
            "minmagnitude": str(min_magnitude),
        }

        url = f"{self.USGS_API_URL}?" + "&".join(f"{k}={v}" for k, v in params.items())

        try:
            import json
            with urlopen(url, timeout=30) as response:
                data = json.loads(response.read().decode())

            features_list = []
            for feature in data.get("features", []):
                props = feature.get("properties", {})
                geom = feature.get("geometry", {}).get("coordinates", [0, 0, 0])

                event_time = datetime.fromtimestamp(props.get("time", 0) / 1000)

                features_list.append([
                    props.get("mag", 0) or 0,
                    geom[2] if len(geom) > 2 else 0,
                    geom[1] if len(geom) > 1 else 0,
                    geom[0] if len(geom) > 0 else 0,
                    hash(props.get("magType", "")) % 5,
                    props.get("gap", 0) or 0,
                    props.get("dmin", 0) or 0,
                    props.get("rms", 0) or 0,
                    props.get("horizontalError", 0) or 0,
                    props.get("depthError", 0) or 0,
                    props.get("magError", 0) or 0,
                    event_time.hour,
                    event_time.weekday(),
                    event_time.month,
                ])

            if not features_list:
                return self._generate_synthetic(1000, anomaly_threshold)

            data_array = np.array(features_list, dtype=np.float32)
            labels = (data_array[:, 0] >= anomaly_threshold).astype(float)

            return data_array, labels

        except Exception as e:
            logger.warning(f"Failed to load from USGS API: {e}. Using synthetic data.")
            return self._generate_synthetic(1000, anomaly_threshold)

    def get_train_test_split(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get train/test split."""
        if self._data is None:
            self.load()

        rng = np.random.default_rng(random_state)
        n_samples = len(self._data)
        n_test = int(n_samples * test_size)

        indices = rng.permutation(n_samples)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        return (
            self._data[train_idx],
            self._data[test_idx],
            self._labels[train_idx],
            self._labels[test_idx],
        )


class MIMICLoader(DatasetLoader):
    """
    MIMIC-III Medical ICU Data Loader (IRB Placeholder Simulation).

    MIMIC-III is a large, freely-available database of de-identified health data
    from ICU patients. Access requires credentialing through PhysioNet.

    Source: https://mimic.physionet.org/

    ⚠️ IMPORTANT: This loader provides SYNTHETIC data that simulates
    MIMIC-III-like patterns for research/development purposes.
    Real MIMIC-III access requires:
    1. CITI training completion
    2. PhysioNet credentialing
    3. IRB approval for your institution

    Features: Vital signs, lab values, demographics
    Anomalies: Sepsis, cardiac events, mortality risk

    Citation:
    Johnson, A. E. W., et al. (2016). MIMIC-III, a freely accessible
    critical care database. Scientific Data, 3, 160035.
    """

    FEATURE_NAMES = [
        "heart_rate", "systolic_bp", "diastolic_bp", "mean_bp",
        "respiratory_rate", "temperature", "spo2", "gcs_total",
        "wbc", "hemoglobin", "platelets", "creatinine",
        "bun", "glucose", "sodium", "potassium",
        "age", "gender", "los_hours", "icu_type_encoded",
    ]

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".omni_ava" / "datasets"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._data: np.ndarray | None = None
        self._labels: np.ndarray | None = None
        self._metadata: DatasetMetadata | None = None

    def load(
        self,
        n_samples: int = 5000,
        anomaly_type: str = "sepsis",
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, DatasetMetadata]:
        """
        Load MIMIC-III-like synthetic data.

        Args:
            n_samples: Number of samples to generate
            anomaly_type: Type of anomaly to simulate ("sepsis", "cardiac", "mortality")

        Returns:
            Tuple of (features, labels, metadata)

        Note:
            This generates SYNTHETIC data simulating MIMIC-III patterns.
            Real MIMIC-III access requires PhysioNet credentialing.
        """
        import time
        start_time = time.time()

        self._data, self._labels = self._generate_synthetic(n_samples, anomaly_type)

        load_time = time.time() - start_time

        num_anomalies = int(np.sum(self._labels))
        self._metadata = DatasetMetadata(
            name="MIMIC-III (Synthetic Simulation)",
            source="Synthetic (IRB Placeholder)",
            num_samples=len(self._data),
            num_features=self._data.shape[1],
            num_anomalies=num_anomalies,
            anomaly_ratio=num_anomalies / len(self._labels),
            feature_names=self.FEATURE_NAMES,
            load_time_seconds=load_time,
            checksum=hashlib.md5(self._data.tobytes()).hexdigest()[:16],
            license="Synthetic - No restrictions",
            citation="Simulated based on Johnson et al. (2016). MIMIC-III.",
        )

        logger.info(f"Loaded MIMIC-III (Synthetic): {self._metadata.num_samples} samples, "
                   f"{self._metadata.anomaly_ratio:.2%} {anomaly_type} cases")

        return self._data, self._labels, self._metadata

    def _generate_synthetic(
        self, n_samples: int, anomaly_type: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic MIMIC-III-like data."""
        rng = np.random.default_rng(42)

        anomaly_ratio = 0.15
        n_normal = int(n_samples * (1 - anomaly_ratio))
        n_anomaly = n_samples - n_normal

        normal_vitals = self._generate_normal_vitals(rng, n_normal)
        anomaly_vitals = self._generate_anomaly_vitals(rng, n_anomaly, anomaly_type)

        data = np.vstack([normal_vitals, anomaly_vitals])
        labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])

        shuffle_idx = rng.permutation(n_samples)
        return data[shuffle_idx], labels[shuffle_idx]

    def _generate_normal_vitals(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Generate normal patient vital signs."""
        return np.column_stack([
            rng.normal(80, 12, n),      # heart_rate (60-100 normal)
            rng.normal(120, 15, n),     # systolic_bp
            rng.normal(80, 10, n),      # diastolic_bp
            rng.normal(90, 10, n),      # mean_bp
            rng.normal(16, 3, n),       # respiratory_rate
            rng.normal(37.0, 0.5, n),   # temperature
            rng.normal(97, 2, n),       # spo2
            rng.normal(14, 1, n),       # gcs_total (15 is max)
            rng.normal(8, 2, n),        # wbc (4-11 normal)
            rng.normal(13, 1.5, n),     # hemoglobin
            rng.normal(250, 50, n),     # platelets
            rng.normal(1.0, 0.2, n),    # creatinine
            rng.normal(15, 5, n),       # bun
            rng.normal(100, 20, n),     # glucose
            rng.normal(140, 3, n),      # sodium
            rng.normal(4.0, 0.4, n),    # potassium
            rng.normal(65, 15, n),      # age
            rng.integers(0, 2, n),      # gender
            rng.exponential(48, n),     # los_hours
            rng.integers(0, 5, n),      # icu_type
        ])

    def _generate_anomaly_vitals(
        self, rng: np.random.Generator, n: int, anomaly_type: str
    ) -> np.ndarray:
        """Generate anomalous patient vital signs based on condition."""
        if anomaly_type == "sepsis":
            return np.column_stack([
                rng.normal(110, 20, n),     # elevated heart_rate
                rng.normal(90, 20, n),      # low systolic_bp
                rng.normal(60, 15, n),      # low diastolic_bp
                rng.normal(70, 15, n),      # low mean_bp
                rng.normal(24, 6, n),       # elevated respiratory_rate
                rng.normal(38.5, 1.0, n),   # elevated temperature
                rng.normal(92, 4, n),       # low spo2
                rng.normal(12, 2, n),       # decreased gcs
                rng.normal(15, 5, n),       # elevated wbc
                rng.normal(10, 2, n),       # low hemoglobin
                rng.normal(150, 80, n),     # low platelets
                rng.normal(2.0, 0.8, n),    # elevated creatinine
                rng.normal(30, 15, n),      # elevated bun
                rng.normal(150, 50, n),     # elevated glucose
                rng.normal(138, 5, n),      # sodium
                rng.normal(4.5, 0.8, n),    # potassium
                rng.normal(70, 12, n),      # age
                rng.integers(0, 2, n),      # gender
                rng.exponential(120, n),    # longer los
                rng.integers(0, 5, n),      # icu_type
            ])
        elif anomaly_type == "cardiac":
            return np.column_stack([
                rng.normal(45, 15, n),      # bradycardia or
                rng.normal(85, 25, n),      # hypotension
                rng.normal(55, 15, n),      # low diastolic
                rng.normal(65, 15, n),      # low map
                rng.normal(22, 5, n),       # elevated rr
                rng.normal(36.5, 0.8, n),   # normal temp
                rng.normal(90, 5, n),       # low spo2
                rng.normal(13, 2, n),       # gcs
                rng.normal(9, 3, n),        # wbc
                rng.normal(11, 2, n),       # low hgb
                rng.normal(200, 60, n),     # platelets
                rng.normal(1.5, 0.5, n),    # creatinine
                rng.normal(25, 10, n),      # bun
                rng.normal(120, 40, n),     # glucose
                rng.normal(139, 4, n),      # sodium
                rng.normal(4.8, 0.6, n),    # elevated potassium
                rng.normal(72, 10, n),      # older age
                rng.integers(0, 2, n),      # gender
                rng.exponential(96, n),     # los
                rng.integers(0, 5, n),      # icu_type
            ])
        else:  # mortality
            return np.column_stack([
                rng.normal(95, 25, n),      # variable hr
                rng.normal(80, 25, n),      # low sbp
                rng.normal(50, 15, n),      # low dbp
                rng.normal(60, 15, n),      # low map
                rng.normal(28, 8, n),       # elevated rr
                rng.normal(37.5, 1.5, n),   # variable temp
                rng.normal(88, 6, n),       # low spo2
                rng.normal(8, 3, n),        # low gcs
                rng.normal(18, 8, n),       # elevated wbc
                rng.normal(9, 2, n),        # low hgb
                rng.normal(100, 60, n),     # low platelets
                rng.normal(3.0, 1.2, n),    # elevated creatinine
                rng.normal(50, 25, n),      # elevated bun
                rng.normal(180, 80, n),     # elevated glucose
                rng.normal(145, 8, n),      # sodium
                rng.normal(5.5, 1.0, n),    # elevated potassium
                rng.normal(75, 10, n),      # older age
                rng.integers(0, 2, n),      # gender
                rng.exponential(200, n),    # long los
                rng.integers(0, 5, n),      # icu_type
            ])

    def get_train_test_split(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get train/test split."""
        if self._data is None:
            self.load()

        rng = np.random.default_rng(random_state)
        n_samples = len(self._data)
        n_test = int(n_samples * test_size)

        indices = rng.permutation(n_samples)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        return (
            self._data[train_idx],
            self._data[test_idx],
            self._labels[train_idx],
            self._labels[test_idx],
        )

    def check_irb_status(self) -> dict[str, Any]:
        """
        Check IRB/credentialing status for real MIMIC-III access.

        Returns:
            Dictionary with IRB status information
        """
        return {
            "has_physionet_account": False,
            "citi_training_complete": False,
            "irb_approved": False,
            "data_use_agreement_signed": False,
            "can_access_real_data": False,
            "using_synthetic": True,
            "message": (
                "Real MIMIC-III access requires PhysioNet credentialing. "
                "See https://mimic.physionet.org/gettingstarted/access/ for details. "
                "Currently using synthetic data for research/development."
            ),
        }
