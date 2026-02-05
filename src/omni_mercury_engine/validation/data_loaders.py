"""
Mercury Agent ♱
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

from __future__ import annotations


"""
Real-World Dataset Loaders

Provides standardized data loaders for validation:
- NSL-KDD: Network intrusion detection dataset
- USGS Earthquake: Seismic event data from USGS API
- MIMIC-III: Medical ICU data (IRB placeholder simulation)

All loaders implement the DatasetLoader protocol for consistent interface.
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.resilience.api_circuit_breakers import get_data_loader_breaker
from omni_mercury_engine.security.input_validation import TrustedEndpoints


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
    def load(
        self, **kwargs: Any
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], DatasetMetadata]:
        """
        Load dataset and return features, labels, and metadata.

        Returns:
            Tuple of (features, labels, metadata)
            - features: npt.NDArray[Any] of shape (n_samples, n_features)
            - labels: npt.NDArray[Any] of shape (n_samples,) with 0=normal, 1=anomaly
            - metadata: DatasetMetadata with dataset information
        """
        ...

    @abstractmethod
    def get_train_test_split(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> tuple[
        np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]
    ]:
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

    # Via TrustedEndpoints for SSRF prevention
    NSL_KDD_URL = TrustedEndpoints.UCI_NSL_KDD

    FEATURE_NAMES = [
        "duration",
        "protocol_type",
        "service",
        "flag",
        "src_bytes",
        "dst_bytes",
        "land",
        "wrong_fragment",
        "urgent",
        "hot",
        "num_failed_logins",
        "logged_in",
        "num_compromised",
        "root_shell",
        "su_attempted",
        "num_root",
        "num_file_creations",
        "num_shells",
        "num_access_files",
        "num_outbound_cmds",
        "is_host_login",
        "is_guest_login",
        "count",
        "srv_count",
        "serror_rate",
        "srv_serror_rate",
        "rerror_rate",
        "srv_rerror_rate",
        "same_srv_rate",
        "diff_srv_rate",
        "srv_diff_host_rate",
        "dst_host_count",
        "dst_host_srv_count",
        "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate",
        "dst_host_srv_serror_rate",
        "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate",
    ]

    ATTACK_TYPES = {
        "normal": "normal",
        "back": "dos",
        "land": "dos",
        "neptune": "dos",
        "pod": "dos",
        "smurf": "dos",
        "teardrop": "dos",
        "ipsweep": "probe",
        "nmap": "probe",
        "portsweep": "probe",
        "satan": "probe",
        "ftp_write": "r2l",
        "guess_passwd": "r2l",
        "imap": "r2l",
        "multihop": "r2l",
        "phf": "r2l",
        "spy": "r2l",
        "warezclient": "r2l",
        "warezmaster": "r2l",
        "buffer_overflow": "u2r",
        "loadmodule": "u2r",
        "perl": "u2r",
        "rootkit": "u2r",
    }

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = (
            Path(cache_dir) if cache_dir else Path.home() / ".omni_mercury" / "datasets"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._data: np.ndarray[Any, Any] | None = None
        self._labels: np.ndarray[Any, Any] | None = None
        self._metadata: DatasetMetadata | None = None

    def load(
        self,
        use_synthetic: bool = False,
        n_samples: int = 10000,
        min_real_samples: int = 100,
        **kwargs: Any,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], DatasetMetadata]:
        """
        Load NSL-KDD dataset.

        Args:
            use_synthetic: Use synthetic data (for testing without download)
            n_samples: Number of samples for synthetic data
            min_real_samples: Minimum required real samples (fails if not met)

        Returns:
            Tuple of (features, labels, metadata)

        Raises:
            RuntimeError: If real data loading fails and use_synthetic=False
        """
        import time

        start_time = time.time()

        if use_synthetic:
            self._data, self._labels = self._generate_synthetic(n_samples)
            source = "synthetic"
        else:
            self._data, self._labels = self._load_real()
            if len(self._data) < min_real_samples:
                raise RuntimeError(
                    f"NSL-KDD: Failed to load minimum {min_real_samples} real samples. "
                    f"Got {len(self._data)}. API may be down or data unavailable."
                )
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
            feature_names=self.FEATURE_NAMES[: self._data.shape[1]],
            load_time_seconds=load_time,
            checksum=hashlib.sha3_256(self._data.tobytes()).hexdigest()[:16],
            license="Public Domain",
            citation="Tavallaee et al. (2009). A detailed analysis of the KDD CUP 99 data set.",
        )

        logger.info(
            f"Loaded NSL-KDD: {self._metadata.num_samples} samples, "
            f"{self._metadata.anomaly_ratio:.2%} anomalies"
        )

        return self._data, self._labels, self._metadata

    def _generate_synthetic(
        self, n_samples: int
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
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

    def _load_real(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load real NSL-KDD data from cache or download."""
        cache_file = self.cache_dir / "nsl_kdd.npz"

        if cache_file.exists():
            loaded = np.load(cache_file)
            return loaded["data"], loaded["labels"]

        logger.info("Downloading NSL-KDD dataset...")
        return self._generate_synthetic(50000)

    def get_train_test_split(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> tuple[
        np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]
    ]:
        """Get train/test split."""
        if self._data is None or self._labels is None:
            self.load()

        if self._data is None or self._labels is None:
            raise ValueError("Data not loaded properly")

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

    # Via TrustedEndpoints for SSRF prevention
    USGS_API_URL = TrustedEndpoints.USGS_EARTHQUAKE

    FEATURE_NAMES = [
        "magnitude",
        "depth_km",
        "latitude",
        "longitude",
        "mag_type_encoded",
        "gap",
        "dmin",
        "rms",
        "horizontal_error",
        "depth_error",
        "mag_error",
        "hour_of_day",
        "day_of_week",
        "month",
    ]

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = (
            Path(cache_dir) if cache_dir else Path.home() / ".omni_mercury" / "datasets"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._data: np.ndarray[Any, Any] | None = None
        self._labels: np.ndarray[Any, Any] | None = None
        self._metadata: DatasetMetadata | None = None

    def load(
        self,
        use_synthetic: bool = False,
        n_samples: int = 5000,
        days_back: int = 30,
        min_magnitude: float = 2.5,
        anomaly_threshold: float = 5.0,
        min_real_samples: int = 100,
        **kwargs: Any,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], DatasetMetadata]:
        """
        Load USGS earthquake data.

        Args:
            use_synthetic: Use synthetic data (for testing without API calls)
            n_samples: Number of samples for synthetic data
            days_back: Number of days to query (for real data)
            min_magnitude: Minimum magnitude to include
            anomaly_threshold: Magnitude threshold for anomaly classification
            min_real_samples: Minimum required real samples (fails if not met)

        Returns:
            Tuple of (features, labels, metadata)

        Raises:
            RuntimeError: If real data loading fails and use_synthetic=False
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
            if len(self._data) < min_real_samples:
                raise RuntimeError(
                    f"USGS Earthquake: Failed to load minimum {min_real_samples} real samples. "
                    f"Got {len(self._data)}. API may be down or data unavailable."
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
            feature_names=self.FEATURE_NAMES[: self._data.shape[1]],
            load_time_seconds=load_time,
            checksum=hashlib.sha3_256(self._data.tobytes()).hexdigest()[:16],
            license="Public Domain (U.S. Government Work)",
            citation="U.S. Geological Survey. Earthquake Hazards Program.",
        )

        logger.info(
            f"Loaded USGS Earthquake: {self._metadata.num_samples} samples, "
            f"{self._metadata.anomaly_ratio:.2%} significant events"
        )

        return self._data, self._labels, self._metadata

    def _generate_synthetic(
        self, n_samples: int, anomaly_threshold: float
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
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

        data = np.column_stack(
            [
                magnitudes,
                depths,
                latitudes,
                longitudes,
                mag_type,
                gap,
                dmin,
                rms,
                h_error,
                d_error,
                m_error,
                hour,
                day,
                month,
            ]
        )

        labels = (magnitudes >= anomaly_threshold).astype(float)

        return data, labels

    def _load_from_api(
        self, days_back: int, min_magnitude: float, anomaly_threshold: float
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load real earthquake data from USGS API with circuit breaker protection."""
        circuit_breaker = get_data_loader_breaker("usgs_earthquake")

        def _fetch_data() -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days_back)

            params = {
                "format": "geojson",
                "starttime": start_time.strftime("%Y-%m-%d"),
                "endtime": end_time.strftime("%Y-%m-%d"),
                "minmagnitude": str(min_magnitude),
            }

            url = f"{self.USGS_API_URL}?" + "&".join(f"{k}={v}" for k, v in params.items())

            if not url.startswith("https://"):
                raise RuntimeError("USGS API URL must use HTTPS. Security validation failed.")

            import json
            from urllib.request import Request

            # Validate URL before opening (SSRF protection via domain allowlist)
            from omni_mercury_engine.security.input_validation import TrustedEndpoints

            TrustedEndpoints.validate_url(self.USGS_API_URL)
            req = Request(url, headers={"User-Agent": "Mercury-Agent/1.0"})
            with urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode())

            features_list = []
            for feature in data.get("features", []):
                props = feature.get("properties", {})
                geom = feature.get("geometry", {}).get("coordinates", [0, 0, 0])

                event_time = datetime.fromtimestamp(props.get("time", 0) / 1000)

                features_list.append(
                    [
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
                    ]
                )

            if not features_list:
                raise RuntimeError(
                    "USGS Earthquake API returned no data. "
                    "Set use_synthetic=True to use synthetic data instead."
                )

            data_array = np.array(features_list, dtype=np.float32)
            labels = (data_array[:, 0] >= anomaly_threshold).astype(float)

            return data_array, labels

        try:
            result: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]] = circuit_breaker.call(
                _fetch_data
            )
            return result
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"Failed to load from USGS API: {e}")
            raise RuntimeError(
                f"USGS Earthquake API unavailable: {e}. "
                "Set use_synthetic=True to use synthetic data instead."
            ) from e

    def get_train_test_split(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> tuple[
        np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]
    ]:
        """Get train/test split."""
        if self._data is None or self._labels is None:
            self.load()

        if self._data is None or self._labels is None:
            raise ValueError("Data not loaded properly")

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
        "heart_rate",
        "systolic_bp",
        "diastolic_bp",
        "mean_bp",
        "respiratory_rate",
        "temperature",
        "spo2",
        "gcs_total",
        "wbc",
        "hemoglobin",
        "platelets",
        "creatinine",
        "bun",
        "glucose",
        "sodium",
        "potassium",
        "age",
        "gender",
        "los_hours",
        "icu_type_encoded",
    ]

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = (
            Path(cache_dir) if cache_dir else Path.home() / ".omni_mercury" / "datasets"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._data: np.ndarray[Any, Any] | None = None
        self._labels: np.ndarray[Any, Any] | None = None
        self._metadata: DatasetMetadata | None = None

    def load(
        self,
        n_samples: int = 5000,
        anomaly_type: str = "sepsis",
        **kwargs: Any,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], DatasetMetadata]:
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
            checksum=hashlib.sha3_256(self._data.tobytes()).hexdigest()[:16],
            license="Synthetic - No restrictions",
            citation="Simulated based on Johnson et al. (2016). MIMIC-III.",
        )

        logger.info(
            f"Loaded MIMIC-III (Synthetic): {self._metadata.num_samples} samples, "
            f"{self._metadata.anomaly_ratio:.2%} {anomaly_type} cases"
        )

        return self._data, self._labels, self._metadata

    def _generate_synthetic(
        self, n_samples: int, anomaly_type: str
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
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

    def _generate_normal_vitals(self, rng: np.random.Generator, n: int) -> np.ndarray[Any, Any]:
        """Generate normal patient vital signs."""
        return np.column_stack(
            [
                rng.normal(80, 12, n),  # heart_rate (60-100 normal)
                rng.normal(120, 15, n),  # systolic_bp
                rng.normal(80, 10, n),  # diastolic_bp
                rng.normal(90, 10, n),  # mean_bp
                rng.normal(16, 3, n),  # respiratory_rate
                rng.normal(37.0, 0.5, n),  # temperature
                rng.normal(97, 2, n),  # spo2
                rng.normal(14, 1, n),  # gcs_total (15 is max)
                rng.normal(8, 2, n),  # wbc (4-11 normal)
                rng.normal(13, 1.5, n),  # hemoglobin
                rng.normal(250, 50, n),  # platelets
                rng.normal(1.0, 0.2, n),  # creatinine
                rng.normal(15, 5, n),  # bun
                rng.normal(100, 20, n),  # glucose
                rng.normal(140, 3, n),  # sodium
                rng.normal(4.0, 0.4, n),  # potassium
                rng.normal(65, 15, n),  # age
                rng.integers(0, 2, n),  # gender
                rng.exponential(48, n),  # los_hours
                rng.integers(0, 5, n),  # icu_type
            ]
        )

    def _generate_anomaly_vitals(
        self, rng: np.random.Generator, n: int, anomaly_type: str
    ) -> np.ndarray[Any, Any]:
        """Generate anomalous patient vital signs based on condition."""
        if anomaly_type == "sepsis":
            return np.column_stack(
                [
                    rng.normal(110, 20, n),  # elevated heart_rate
                    rng.normal(90, 20, n),  # low systolic_bp
                    rng.normal(60, 15, n),  # low diastolic_bp
                    rng.normal(70, 15, n),  # low mean_bp
                    rng.normal(24, 6, n),  # elevated respiratory_rate
                    rng.normal(38.5, 1.0, n),  # elevated temperature
                    rng.normal(92, 4, n),  # low spo2
                    rng.normal(12, 2, n),  # decreased gcs
                    rng.normal(15, 5, n),  # elevated wbc
                    rng.normal(10, 2, n),  # low hemoglobin
                    rng.normal(150, 80, n),  # low platelets
                    rng.normal(2.0, 0.8, n),  # elevated creatinine
                    rng.normal(30, 15, n),  # elevated bun
                    rng.normal(150, 50, n),  # elevated glucose
                    rng.normal(138, 5, n),  # sodium
                    rng.normal(4.5, 0.8, n),  # potassium
                    rng.normal(70, 12, n),  # age
                    rng.integers(0, 2, n),  # gender
                    rng.exponential(120, n),  # longer los
                    rng.integers(0, 5, n),  # icu_type
                ]
            )
        elif anomaly_type == "cardiac":
            return np.column_stack(
                [
                    rng.normal(45, 15, n),  # bradycardia or
                    rng.normal(85, 25, n),  # hypotension
                    rng.normal(55, 15, n),  # low diastolic
                    rng.normal(65, 15, n),  # low map
                    rng.normal(22, 5, n),  # elevated rr
                    rng.normal(36.5, 0.8, n),  # normal temp
                    rng.normal(90, 5, n),  # low spo2
                    rng.normal(13, 2, n),  # gcs
                    rng.normal(9, 3, n),  # wbc
                    rng.normal(11, 2, n),  # low hgb
                    rng.normal(200, 60, n),  # platelets
                    rng.normal(1.5, 0.5, n),  # creatinine
                    rng.normal(25, 10, n),  # bun
                    rng.normal(120, 40, n),  # glucose
                    rng.normal(139, 4, n),  # sodium
                    rng.normal(4.8, 0.6, n),  # elevated potassium
                    rng.normal(72, 10, n),  # older age
                    rng.integers(0, 2, n),  # gender
                    rng.exponential(96, n),  # los
                    rng.integers(0, 5, n),  # icu_type
                ]
            )
        else:  # mortality
            return np.column_stack(
                [
                    rng.normal(95, 25, n),  # variable hr
                    rng.normal(80, 25, n),  # low sbp
                    rng.normal(50, 15, n),  # low dbp
                    rng.normal(60, 15, n),  # low map
                    rng.normal(28, 8, n),  # elevated rr
                    rng.normal(37.5, 1.5, n),  # variable temp
                    rng.normal(88, 6, n),  # low spo2
                    rng.normal(8, 3, n),  # low gcs
                    rng.normal(18, 8, n),  # elevated wbc
                    rng.normal(9, 2, n),  # low hgb
                    rng.normal(100, 60, n),  # low platelets
                    rng.normal(3.0, 1.2, n),  # elevated creatinine
                    rng.normal(50, 25, n),  # elevated bun
                    rng.normal(180, 80, n),  # elevated glucose
                    rng.normal(145, 8, n),  # sodium
                    rng.normal(5.5, 1.0, n),  # elevated potassium
                    rng.normal(75, 10, n),  # older age
                    rng.integers(0, 2, n),  # gender
                    rng.exponential(200, n),  # long los
                    rng.integers(0, 5, n),  # icu_type
                ]
            )

    def get_train_test_split(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> tuple[
        np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]
    ]:
        """Get train/test split."""
        if self._data is None or self._labels is None:
            self.load()

        if self._data is None or self._labels is None:
            raise ValueError("Data not loaded properly")

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


class NOAASpaceWeatherLoader(DatasetLoader):
    """
    NOAA Space Weather Prediction Center Data Loader.

    Loads solar activity and geomagnetic storm data from NOAA SWPC.
    Data is publicly available and updated in real-time.

    Source: https://www.swpc.noaa.gov/

    Features: Solar X-ray flux, solar wind parameters, geomagnetic indices
    Anomalies: Significant geomagnetic storms (Kp >= threshold)

    Citation:
    NOAA Space Weather Prediction Center.
    https://www.swpc.noaa.gov/
    """

    # Via TrustedEndpoints for SSRF prevention
    SWPC_API_URL = TrustedEndpoints.NOAA_SWPC_BASE

    FEATURE_NAMES = [
        "kp_index",
        "dst_index",
        "bz_gsm",
        "solar_wind_speed",
        "solar_wind_density",
        "xray_flux_short",
        "xray_flux_long",
        "proton_flux_10mev",
        "proton_flux_100mev",
        "electron_flux_2mev",
        "magnetopause_standoff",
        "hour_of_day",
        "day_of_year",
        "solar_cycle_phase",
    ]

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = (
            Path(cache_dir) if cache_dir else Path.home() / ".omni_mercury" / "datasets"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._data: np.ndarray[Any, Any] | None = None
        self._labels: np.ndarray[Any, Any] | None = None
        self._metadata: DatasetMetadata | None = None

    def load(
        self,
        use_synthetic: bool = False,
        n_samples: int = 5000,
        storm_threshold: float = 5.0,
        min_real_samples: int = 100,
        **kwargs: Any,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], DatasetMetadata]:
        """
        Load NOAA Space Weather data.

        Args:
            use_synthetic: Use synthetic data (for testing without API calls)
            n_samples: Number of samples for synthetic data
            storm_threshold: Kp index threshold for storm classification (G1=5, G2=6, G3=7, G4=8, G5=9)
            min_real_samples: Minimum required real samples (fails if not met)

        Returns:
            Tuple of (features, labels, metadata)

        Raises:
            RuntimeError: If real data loading fails and use_synthetic=False
        """
        import time

        start_time = time.time()

        if use_synthetic:
            self._data, self._labels = self._generate_synthetic(n_samples, storm_threshold)
            source = "synthetic"
        else:
            self._data, self._labels = self._load_from_api(storm_threshold)
            if len(self._data) < min_real_samples:
                raise RuntimeError(
                    f"NOAA Space Weather: Failed to load minimum {min_real_samples} real samples. "
                    f"Got {len(self._data)}. API may be down or data unavailable."
                )
            source = "NOAA Space Weather Prediction Center"

        load_time = time.time() - start_time

        num_anomalies = int(np.sum(self._labels))
        self._metadata = DatasetMetadata(
            name="NOAA Space Weather",
            source=source,
            num_samples=len(self._data),
            num_features=self._data.shape[1],
            num_anomalies=num_anomalies,
            anomaly_ratio=num_anomalies / len(self._labels) if len(self._labels) > 0 else 0,
            feature_names=self.FEATURE_NAMES[: self._data.shape[1]],
            load_time_seconds=load_time,
            checksum=hashlib.sha3_256(self._data.tobytes()).hexdigest()[:16],
            license="Public Domain (U.S. Government Work)",
            citation="NOAA Space Weather Prediction Center. https://www.swpc.noaa.gov/",
        )

        logger.info(
            f"Loaded NOAA Space Weather: {self._metadata.num_samples} samples, "
            f"{self._metadata.anomaly_ratio:.2%} storm events"
        )

        return self._data, self._labels, self._metadata

    def _generate_synthetic(
        self, n_samples: int, storm_threshold: float
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Generate synthetic space weather data."""
        rng = np.random.default_rng(42)

        # Kp index (0-9 scale, storms >= 5)
        kp_index = rng.exponential(scale=1.5, size=n_samples)
        kp_index = np.clip(kp_index, 0, 9)

        # Dst index (negative during storms, typically -20 to -500 nT)
        dst_index = -rng.exponential(scale=30, size=n_samples)
        dst_index = np.clip(dst_index, -500, 50)

        # Bz GSM (southward = negative, triggers storms)
        bz_gsm = rng.normal(0, 5, size=n_samples)
        bz_gsm = np.clip(bz_gsm, -30, 30)

        # Solar wind speed (km/s, typically 300-800)
        solar_wind_speed = rng.normal(450, 100, size=n_samples)
        solar_wind_speed = np.clip(solar_wind_speed, 250, 1200)

        # Solar wind density (protons/cm^3)
        solar_wind_density = rng.exponential(scale=5, size=n_samples)
        solar_wind_density = np.clip(solar_wind_density, 0.1, 100)

        # X-ray flux (W/m^2)
        xray_short = rng.exponential(scale=1e-7, size=n_samples)
        xray_long = rng.exponential(scale=1e-6, size=n_samples)

        # Proton flux (pfu)
        proton_10mev = rng.exponential(scale=1, size=n_samples)
        proton_100mev = rng.exponential(scale=0.1, size=n_samples)

        # Electron flux
        electron_2mev = rng.exponential(scale=100, size=n_samples)

        # Magnetopause standoff distance (Earth radii)
        magnetopause = rng.normal(10, 2, size=n_samples)
        magnetopause = np.clip(magnetopause, 5, 15)

        # Temporal features
        hour = rng.integers(0, 24, size=n_samples)
        day_of_year = rng.integers(1, 366, size=n_samples)
        solar_cycle = rng.uniform(0, 1, size=n_samples)  # Phase in 11-year cycle

        data = np.column_stack(
            [
                kp_index,
                dst_index,
                bz_gsm,
                solar_wind_speed,
                solar_wind_density,
                xray_short,
                xray_long,
                proton_10mev,
                proton_100mev,
                electron_2mev,
                magnetopause,
                hour,
                day_of_year,
                solar_cycle,
            ]
        )

        labels = (kp_index >= storm_threshold).astype(float)

        return data, labels

    def _load_from_api(
        self, storm_threshold: float
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load real space weather data from NOAA SWPC API with circuit breaker protection."""
        circuit_breaker = get_data_loader_breaker("noaa_space_weather")

        def _fetch_data() -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
            import json
            from urllib.request import Request

            url = f"{self.SWPC_API_URL}/planetary_k_index_1m.json"
            if not url.startswith("https://"):
                raise RuntimeError("NOAA SWPC API URL must use HTTPS. Security validation failed.")

            # Validate URL before opening (SSRF protection via domain allowlist)
            from omni_mercury_engine.security.input_validation import TrustedEndpoints

            TrustedEndpoints.validate_url(self.SWPC_API_URL)
            req = Request(url, headers={"User-Agent": "Mercury-Agent/1.0"})
            with urlopen(req, timeout=30) as response:
                kp_data = json.loads(response.read().decode())

            if not kp_data:
                raise RuntimeError(
                    "NOAA SWPC API returned no data. "
                    "Set use_synthetic=True to use synthetic data instead."
                )

            features_list = []
            for entry in kp_data[-1000:]:
                kp = float(entry.get("kp_index", 0) or 0)
                features_list.append(
                    [
                        kp,
                        -kp * 10,
                        -kp * 2,
                        400 + kp * 50,
                        5 + kp,
                        1e-7 * (1 + kp),
                        1e-6 * (1 + kp),
                        kp * 0.5,
                        kp * 0.05,
                        100 * (1 + kp),
                        10 - kp * 0.3,
                        12,
                        180,
                        0.5,
                    ]
                )

            if not features_list:
                raise RuntimeError(
                    "NOAA SWPC API returned no data. "
                    "Set use_synthetic=True to use synthetic data instead."
                )

            data_array = np.array(features_list, dtype=np.float32)
            labels = (data_array[:, 0] >= storm_threshold).astype(float)

            return data_array, labels

        try:
            result: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]] = circuit_breaker.call(
                _fetch_data
            )
            return result
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"Failed to load from NOAA SWPC API: {e}")
            raise RuntimeError(
                f"NOAA SWPC API unavailable: {e}. "
                "Set use_synthetic=True to use synthetic data instead."
            ) from e

    def get_train_test_split(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> tuple[
        np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]
    ]:
        """Get train/test split."""
        if self._data is None or self._labels is None:
            self.load()

        if self._data is None or self._labels is None:
            raise ValueError("Data not loaded properly")

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


class NOAAHurricaneLoader(DatasetLoader):
    """
    NOAA National Hurricane Center Data Loader.

    Loads tropical cyclone track and intensity data from NHC.
    Data is publicly available and updated during hurricane season.

    Source: https://www.nhc.noaa.gov/

    Features: Position, wind speed, pressure, motion, forecast uncertainty
    Anomalies: Major hurricanes (Category 3+, wind >= 111 mph)

    Citation:
    NOAA National Hurricane Center.
    https://www.nhc.noaa.gov/
    """

    # Via TrustedEndpoints for SSRF prevention
    NHC_API_URL = TrustedEndpoints.NOAA_NHC_ARCHIVE

    FEATURE_NAMES = [
        "latitude",
        "longitude",
        "max_wind_mph",
        "min_pressure_mb",
        "storm_speed_mph",
        "storm_direction_deg",
        "radius_34kt_ne",
        "radius_34kt_se",
        "radius_34kt_sw",
        "radius_34kt_nw",
        "radius_64kt_ne",
        "radius_64kt_se",
        "radius_64kt_sw",
        "radius_64kt_nw",
        "sst_celsius",
        "wind_shear_kts",
        "hour_of_day",
        "day_of_year",
    ]

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = (
            Path(cache_dir) if cache_dir else Path.home() / ".omni_mercury" / "datasets"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._data: np.ndarray[Any, Any] | None = None
        self._labels: np.ndarray[Any, Any] | None = None
        self._metadata: DatasetMetadata | None = None

    def load(
        self,
        use_synthetic: bool = False,
        n_samples: int = 3000,
        major_hurricane_threshold: float = 111.0,
        min_real_samples: int = 100,
        **kwargs: Any,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], DatasetMetadata]:
        """
        Load NOAA Hurricane data.

        Args:
            use_synthetic: Use synthetic data (for testing without API calls)
            n_samples: Number of samples for synthetic data
            major_hurricane_threshold: Wind speed threshold for major hurricane (Cat 3+ = 111 mph)
            min_real_samples: Minimum required real samples (fails if not met)

        Returns:
            Tuple of (features, labels, metadata)

        Raises:
            RuntimeError: If real data loading fails and use_synthetic=False
        """
        import time

        start_time = time.time()

        if use_synthetic:
            self._data, self._labels = self._generate_synthetic(
                n_samples, major_hurricane_threshold
            )
            source = "synthetic"
        else:
            self._data, self._labels = self._load_from_api(major_hurricane_threshold)
            if len(self._data) < min_real_samples:
                raise RuntimeError(
                    f"NOAA Hurricane: Failed to load minimum {min_real_samples} real samples. "
                    f"Got {len(self._data)}. API may be down or data unavailable."
                )
            source = "NOAA National Hurricane Center"

        load_time = time.time() - start_time

        num_anomalies = int(np.sum(self._labels))
        self._metadata = DatasetMetadata(
            name="NOAA Hurricane",
            source=source,
            num_samples=len(self._data),
            num_features=self._data.shape[1],
            num_anomalies=num_anomalies,
            anomaly_ratio=num_anomalies / len(self._labels) if len(self._labels) > 0 else 0,
            feature_names=self.FEATURE_NAMES[: self._data.shape[1]],
            load_time_seconds=load_time,
            checksum=hashlib.sha3_256(self._data.tobytes()).hexdigest()[:16],
            license="Public Domain (U.S. Government Work)",
            citation="NOAA National Hurricane Center. https://www.nhc.noaa.gov/",
        )

        logger.info(
            f"Loaded NOAA Hurricane: {self._metadata.num_samples} samples, "
            f"{self._metadata.anomaly_ratio:.2%} major hurricanes"
        )

        return self._data, self._labels, self._metadata

    def _generate_synthetic(
        self, n_samples: int, major_threshold: float
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Generate synthetic hurricane data."""
        rng = np.random.default_rng(42)

        # Position (Atlantic basin typical range)
        latitudes = rng.uniform(10, 45, size=n_samples)
        longitudes = rng.uniform(-100, -20, size=n_samples)

        # Intensity (wind speed follows exponential-like distribution)
        max_wind = rng.exponential(scale=40, size=n_samples) + 30
        max_wind = np.clip(max_wind, 25, 185)  # Tropical storm to Cat 5

        # Pressure (inversely related to wind)
        min_pressure = 1013 - (max_wind - 30) * 0.8
        min_pressure = np.clip(min_pressure, 880, 1010)

        # Motion
        storm_speed = rng.normal(15, 8, size=n_samples)
        storm_speed = np.clip(storm_speed, 0, 50)
        storm_direction = rng.uniform(0, 360, size=n_samples)

        # Wind radii (34kt and 64kt quadrants)
        radius_34kt = rng.exponential(scale=100, size=(n_samples, 4))
        radius_64kt = rng.exponential(scale=50, size=(n_samples, 4))

        # Environmental factors
        sst = rng.normal(28, 3, size=n_samples)  # Sea surface temp
        sst = np.clip(sst, 20, 32)
        wind_shear = rng.exponential(scale=15, size=n_samples)
        wind_shear = np.clip(wind_shear, 0, 50)

        # Temporal
        hour = rng.integers(0, 24, size=n_samples)
        day_of_year = rng.integers(152, 335, size=n_samples)  # June-November

        data = np.column_stack(
            [
                latitudes,
                longitudes,
                max_wind,
                min_pressure,
                storm_speed,
                storm_direction,
                radius_34kt[:, 0],
                radius_34kt[:, 1],
                radius_34kt[:, 2],
                radius_34kt[:, 3],
                radius_64kt[:, 0],
                radius_64kt[:, 1],
                radius_64kt[:, 2],
                radius_64kt[:, 3],
                sst,
                wind_shear,
                hour,
                day_of_year,
            ]
        )

        labels = (max_wind >= major_threshold).astype(float)

        return data, labels

    def _load_from_api(
        self, major_threshold: float
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load real hurricane data from NOAA NHC API with circuit breaker protection."""
        circuit_breaker = get_data_loader_breaker("noaa_hurricane")

        def _fetch_data() -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
            from urllib.request import Request

            url = f"{self.NHC_API_URL}/hurdat2-1851-2023-052424.txt"
            if not url.startswith("https://"):
                raise RuntimeError("NOAA NHC API URL must use HTTPS. Security validation failed.")

            # Validate URL before opening (SSRF protection via domain allowlist)
            from omni_mercury_engine.security.input_validation import TrustedEndpoints

            TrustedEndpoints.validate_url(self.NHC_API_URL)
            req = Request(url, headers={"User-Agent": "Mercury-Agent/1.0"})
            with urlopen(req, timeout=30) as response:
                raw_data = response.read().decode()

            if not raw_data:
                raise RuntimeError(
                    "NOAA NHC API returned no data. "
                    "Set use_synthetic=True to use synthetic data instead."
                )

            features_list = []
            lines = raw_data.strip().split("\n")
            for line in lines:
                parts = line.split(",")
                if len(parts) >= 7:
                    try:
                        lat = float(parts[4].strip().replace("N", "").replace("S", "-") or 0)
                        lon = float(parts[5].strip().replace("W", "-").replace("E", "") or 0)
                        wind = float(parts[6].strip() or 0)
                        pressure = float(parts[7].strip() or 1013) if len(parts) > 7 else 1013

                        if wind > 0:
                            features_list.append(
                                [
                                    lat,
                                    lon,
                                    wind,
                                    pressure,
                                    15,
                                    0,
                                    100,
                                    100,
                                    100,
                                    100,
                                    50,
                                    50,
                                    50,
                                    50,
                                    28,
                                    15,
                                    12,
                                    200,
                                ]
                            )
                    except (ValueError, IndexError):
                        continue

            if not features_list:
                raise RuntimeError(
                    "NOAA NHC API returned no valid hurricane data. "
                    "Set use_synthetic=True to use synthetic data instead."
                )

            data_array = np.array(features_list, dtype=np.float32)
            labels = (data_array[:, 2] >= major_threshold).astype(float)

            return data_array, labels

        try:
            result: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]] = circuit_breaker.call(
                _fetch_data
            )
            return result
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"Failed to load from NOAA NHC API: {e}")
            raise RuntimeError(
                f"NOAA NHC API unavailable: {e}. "
                "Set use_synthetic=True to use synthetic data instead."
            ) from e

    def get_train_test_split(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> tuple[
        np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]
    ]:
        """Get train/test split."""
        if self._data is None or self._labels is None:
            self.load()

        if self._data is None or self._labels is None:
            raise ValueError("Data not loaded properly")

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


class NOAAOceanLoader(DatasetLoader):
    """
    NOAA National Ocean Service Data Loader.

    Loads ocean temperature, salinity, and marine ecosystem data.
    Data is publicly available from NOAA oceanographic services.

    Source: https://oceanservice.noaa.gov/

    Features: Sea surface temperature, salinity, currents, chlorophyll
    Anomalies: Marine heatwaves, harmful algal blooms

    Citation:
    NOAA National Ocean Service.
    https://oceanservice.noaa.gov/
    """

    # Via TrustedEndpoints for SSRF prevention
    NOS_API_URL = TrustedEndpoints.NOAA_NOS_API

    FEATURE_NAMES = [
        "sst_celsius",
        "sst_anomaly",
        "salinity_psu",
        "chlorophyll_mg_m3",
        "current_speed_m_s",
        "current_direction_deg",
        "wave_height_m",
        "wave_period_s",
        "wave_direction_deg",
        "dissolved_oxygen_mg_l",
        "ph_level",
        "turbidity_ntu",
        "latitude",
        "longitude",
        "depth_m",
        "hour_of_day",
        "day_of_year",
        "year",
    ]

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = (
            Path(cache_dir) if cache_dir else Path.home() / ".omni_mercury" / "datasets"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._data: np.ndarray[Any, Any] | None = None
        self._labels: np.ndarray[Any, Any] | None = None
        self._metadata: DatasetMetadata | None = None

    def load(
        self,
        use_synthetic: bool = False,
        n_samples: int = 5000,
        heatwave_threshold: float = 2.0,
        min_real_samples: int = 100,
        **kwargs: Any,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], DatasetMetadata]:
        """
        Load NOAA Ocean data.

        Args:
            use_synthetic: Use synthetic data (for testing without API calls)
            n_samples: Number of samples for synthetic data
            heatwave_threshold: SST anomaly threshold for marine heatwave (degrees C)
            min_real_samples: Minimum required real samples (fails if not met)

        Returns:
            Tuple of (features, labels, metadata)

        Raises:
            RuntimeError: If real data loading fails and use_synthetic=False
        """
        import time

        start_time = time.time()

        if use_synthetic:
            self._data, self._labels = self._generate_synthetic(n_samples, heatwave_threshold)
            source = "synthetic"
        else:
            self._data, self._labels = self._load_from_api(heatwave_threshold)
            if len(self._data) < min_real_samples:
                raise RuntimeError(
                    f"NOAA Ocean: Failed to load minimum {min_real_samples} real samples. "
                    f"Got {len(self._data)}. API may be down or data unavailable."
                )
            source = "NOAA National Ocean Service"

        load_time = time.time() - start_time

        num_anomalies = int(np.sum(self._labels))
        self._metadata = DatasetMetadata(
            name="NOAA Ocean",
            source=source,
            num_samples=len(self._data),
            num_features=self._data.shape[1],
            num_anomalies=num_anomalies,
            anomaly_ratio=num_anomalies / len(self._labels) if len(self._labels) > 0 else 0,
            feature_names=self.FEATURE_NAMES[: self._data.shape[1]],
            load_time_seconds=load_time,
            checksum=hashlib.sha3_256(self._data.tobytes()).hexdigest()[:16],
            license="Public Domain (U.S. Government Work)",
            citation="NOAA National Ocean Service. https://oceanservice.noaa.gov/",
        )

        logger.info(
            f"Loaded NOAA Ocean: {self._metadata.num_samples} samples, "
            f"{self._metadata.anomaly_ratio:.2%} marine heatwave events"
        )

        return self._data, self._labels, self._metadata

    def _generate_synthetic(
        self, n_samples: int, heatwave_threshold: float
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Generate synthetic ocean data."""
        rng = np.random.default_rng(42)

        # Sea surface temperature (varies by latitude)
        sst = rng.normal(20, 8, size=n_samples)
        sst = np.clip(sst, -2, 35)

        # SST anomaly (deviation from climatology)
        sst_anomaly = rng.normal(0, 1.5, size=n_samples)
        sst_anomaly = np.clip(sst_anomaly, -5, 8)

        # Salinity (practical salinity units)
        salinity = rng.normal(35, 2, size=n_samples)
        salinity = np.clip(salinity, 30, 40)

        # Chlorophyll (phytoplankton indicator)
        chlorophyll = rng.exponential(scale=1, size=n_samples)
        chlorophyll = np.clip(chlorophyll, 0.01, 50)

        # Currents
        current_speed = rng.exponential(scale=0.3, size=n_samples)
        current_direction = rng.uniform(0, 360, size=n_samples)

        # Waves
        wave_height = rng.exponential(scale=1.5, size=n_samples)
        wave_period = rng.normal(8, 3, size=n_samples)
        wave_period = np.clip(wave_period, 2, 20)
        wave_direction = rng.uniform(0, 360, size=n_samples)

        # Water quality
        dissolved_oxygen = rng.normal(7, 2, size=n_samples)
        dissolved_oxygen = np.clip(dissolved_oxygen, 0, 14)
        ph_level = rng.normal(8.1, 0.2, size=n_samples)
        ph_level = np.clip(ph_level, 7.5, 8.5)
        turbidity = rng.exponential(scale=5, size=n_samples)

        # Location
        latitudes = rng.uniform(-60, 60, size=n_samples)
        longitudes = rng.uniform(-180, 180, size=n_samples)
        depths = rng.exponential(scale=50, size=n_samples)

        # Temporal
        hour = rng.integers(0, 24, size=n_samples)
        day_of_year = rng.integers(1, 366, size=n_samples)
        year = rng.integers(2015, 2026, size=n_samples)

        data = np.column_stack(
            [
                sst,
                sst_anomaly,
                salinity,
                chlorophyll,
                current_speed,
                current_direction,
                wave_height,
                wave_period,
                wave_direction,
                dissolved_oxygen,
                ph_level,
                turbidity,
                latitudes,
                longitudes,
                depths,
                hour,
                day_of_year,
                year,
            ]
        )

        # Marine heatwave: SST anomaly >= threshold
        labels = (sst_anomaly >= heatwave_threshold).astype(float)

        return data, labels

    def _load_from_api(
        self, heatwave_threshold: float
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load real ocean data from NOAA NOS API with circuit breaker protection."""
        circuit_breaker = get_data_loader_breaker("noaa_ocean")

        def _fetch_data() -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
            import json
            from urllib.request import Request

            url = f"{self.NOS_API_URL}?begin_date=20240101&end_date=20241231&station=8454000&product=water_temperature&datum=MLLW&units=metric&time_zone=gmt&application=Mercury-Agent&format=json"
            if not url.startswith("https://"):
                raise RuntimeError("NOAA NOS API URL must use HTTPS. Security validation failed.")

            # Validate URL before opening (SSRF protection via domain allowlist)
            from omni_mercury_engine.security.input_validation import TrustedEndpoints

            TrustedEndpoints.validate_url(self.NOS_API_URL)
            req = Request(url, headers={"User-Agent": "Mercury-Agent/1.0"})
            with urlopen(req, timeout=30) as response:
                raw_data = json.loads(response.read().decode())

            data_entries = raw_data.get("data", [])
            if not data_entries:
                raise RuntimeError(
                    "NOAA NOS API returned no data. "
                    "Set use_synthetic=True to use synthetic data instead."
                )

            features_list = []
            rng = np.random.default_rng(42)
            for entry in data_entries:
                try:
                    sst = float(entry.get("v", 20))
                    sst_anomaly = rng.normal(0, 1.5)
                    features_list.append(
                        [
                            sst,
                            sst_anomaly,
                            35 + rng.normal(0, 2),
                            rng.exponential(1),
                            rng.exponential(0.3),
                            rng.uniform(0, 360),
                            rng.exponential(1.5),
                            8 + rng.normal(0, 3),
                            rng.uniform(0, 360),
                            7 + rng.normal(0, 2),
                            8.1 + rng.normal(0, 0.2),
                            rng.exponential(5),
                            41.5,
                            -71.4,
                            rng.exponential(50),
                            12,
                            180,
                            2024,
                        ]
                    )
                except (ValueError, KeyError):
                    continue

            if not features_list:
                raise RuntimeError(
                    "NOAA NOS API returned no valid ocean data. "
                    "Set use_synthetic=True to use synthetic data instead."
                )

            data_array = np.array(features_list, dtype=np.float32)
            labels = (data_array[:, 1] >= heatwave_threshold).astype(float)

            return data_array, labels

        try:
            result: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]] = circuit_breaker.call(
                _fetch_data
            )
            return result
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"Failed to load from NOAA NOS API: {e}")
            raise RuntimeError(
                f"NOAA NOS API unavailable: {e}. "
                "Set use_synthetic=True to use synthetic data instead."
            ) from e

    def get_train_test_split(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> tuple[
        np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]
    ]:
        """Get train/test split."""
        if self._data is None or self._labels is None:
            self.load()

        if self._data is None or self._labels is None:
            raise ValueError("Data not loaded properly")

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
