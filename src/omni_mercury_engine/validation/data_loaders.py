# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Real-World Dataset Loaders (DEPRECATED).

.. deprecated:: 1.5.0
    Use :mod:`omni_mercury_engine.datasets` module directly for the loaders
    that exist there.

Provides standardized data loaders for validation:
- NSL-KDD: Network intrusion detection dataset
- USGS Earthquake: Seismic event data from USGS API
- MIMIC-III: Medical ICU data (IRB placeholder simulation)

Actual state of the consolidation into ``datasets/`` (this header
previously claimed the overlapping classes were re-exports and that the
module would be removed in v2.0; neither was true, and v2.0 shipped with
the module intact):

- NOAASpaceWeatherLoader, NOAAHurricaneLoader and NOAAOceanLoader are now
  thin wrappers over their canonical ``datasets/`` ports
  (``space_weather.py``, ``hurricane.py``, ``ocean.py``), keeping this
  module's ``load()``/``get_train_test_split()`` interface. The port also
  retired the fabrication the old local implementations carried: their
  "real" paths fetched one-to-four real channels and synthesised the rest
  (Kp-derived columns, constant hurricane radii, RNG oceanography). The
  canonical loaders read every feature from the instrument that measures
  it.
- NSLKDDLoader, USGSEarthquakeLoader and MIMICLoader remain full LOCAL
  definitions here, duplicating same-named canonical loaders in
  ``datasets/`` (``security.py``, ``environmental.py``, ``medical.py``).
  The two implementations must be reconciled before either is removed.

Removal: with the NOAA trio ported, this module is announced for removal
in v4.0. New code imports from ``omni_mercury_engine.datasets``.
"""

from __future__ import annotations

import hashlib
import warnings

warnings.warn(
    "validation.data_loaders is deprecated and announced for removal in v4.0: "
    "import from omni_mercury_engine.datasets instead. The NOAA space-weather, "
    "hurricane and ocean classes here are thin wrappers over their canonical "
    "datasets/ ports (see the module docstring).",
    DeprecationWarning,
    stacklevel=2,
)
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from omni_mercury_engine.datasets.exceptions import (
    check_synthetic_allowed,
)
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
        """Load dataset and return features, labels, and metadata.

        Returns:
            Tuple of (features, labels, metadata)
            - features: np.ndarray[Any, Any] of shape (n_samples, n_features)
            - labels: np.ndarray[Any, Any] of shape (n_samples,) with 0=normal, 1=anomaly
            - metadata: DatasetMetadata with dataset information
        """
        ...

    @abstractmethod
    def get_train_test_split(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> tuple[
        np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]
    ]:
        """Get train/test split of the dataset.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        ...


class NSLKDDLoader(DatasetLoader):
    """NSL-KDD Network Intrusion Detection Dataset Loader.

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
        """Initialize the instance."""
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
        """Load NSL-KDD dataset.

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
            # Route the caller's request through the policy gate so a
            # deployment-level ``MERCURY_ALLOW_SYNTHETIC=0`` cannot be
            # bypassed by a caller flag.  Without the env var,
            # ``check_synthetic_allowed`` raises ``DataSourceUnavailableError``.
            check_synthetic_allowed("NSL-KDD", "Caller passed use_synthetic=True")
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
        """Load real NSL-KDD data through the canonical ``datasets/`` loader.

        A pre-reconciliation local cache (``nsl_kdd.npz``) stays readable;
        otherwise the canonical :class:`omni_mercury_engine.datasets.security.
        NSLKDDLoader` downloads and parses the real corpus (this module's old
        path could only read a cache it had no way to create).
        """
        cache_file = self.cache_dir / "nsl_kdd.npz"
        if cache_file.exists():
            loaded = np.load(cache_file)
            return loaded["data"], loaded["labels"]

        from omni_mercury_engine.datasets.base import DatasetConfig
        from omni_mercury_engine.datasets.security import NSLKDDLoader as _Canonical

        loader = _Canonical(
            DatasetConfig(
                name="nsl-kdd",
                data_dir=str(self.cache_dir),
                cache_dir=str(self.cache_dir),
            )
        )
        loader.download()
        data, labels = loader._load_raw()
        return np.asarray(data, dtype=np.float64), np.asarray(labels, dtype=np.float64)

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
    """USGS Earthquake Data Loader.

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
        """Initialize the instance."""
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
        """Load USGS earthquake data.

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
            # Caller-flag bypass closure: route through policy gate so the
            # deployment-level ``MERCURY_ALLOW_SYNTHETIC=0`` floor cannot be
            # circumvented by a caller passing ``use_synthetic=True``.
            check_synthetic_allowed("USGS Earthquake", "Caller passed use_synthetic=True")
            self._data, self._labels = self._generate_synthetic(n_samples, anomaly_threshold)
            source = "synthetic"
        else:
            # Delegate to the canonical datasets/ loader: one USGS fetch and
            # label rule for the whole tree. The local ``_load_from_api``
            # this replaces also encoded ``magType`` via Python's per-process
            # salted ``hash()``, so that feature changed value between runs.
            feature_names, (self._data, self._labels) = self._load_real_via_canonical(
                days_back, min_magnitude, anomaly_threshold
            )
            if len(self._data) < min_real_samples:
                raise RuntimeError(
                    f"USGS Earthquake: Failed to load minimum {min_real_samples} real samples. "
                    f"Got {len(self._data)}. API may be down or data unavailable."
                )
            source = "USGS Earthquake Hazards Program"

        load_time = time.time() - start_time

        feature_names_out = (
            feature_names if not use_synthetic else self.FEATURE_NAMES[: self._data.shape[1]]
        )
        num_anomalies = int(np.sum(self._labels))
        self._metadata = DatasetMetadata(
            name="USGS Earthquake",
            source=source,
            num_samples=len(self._data),
            num_features=self._data.shape[1],
            num_anomalies=num_anomalies,
            anomaly_ratio=num_anomalies / len(self._labels) if len(self._labels) > 0 else 0,
            feature_names=feature_names_out,
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

    def _load_real_via_canonical(
        self, days_back: int, min_magnitude: float, anomaly_threshold: float
    ) -> tuple[list[str], tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]]:
        """Fetch real USGS data through the canonical ``datasets/`` loader.

        Returns:
            ``(feature_names, (features, labels))`` from
            :class:`omni_mercury_engine.datasets.environmental.USGSEarthquakeLoader`,
            with ``anomaly_threshold`` honoured as the canonical
            ``significant_magnitude`` label rule.
        """
        from omni_mercury_engine.datasets.base import DatasetConfig
        from omni_mercury_engine.datasets.environmental import (
            USGSEarthquakeLoader as _Canonical,
        )

        loader = _Canonical(
            DatasetConfig(
                name="earthquake",
                data_dir=str(self.cache_dir),
                cache_dir=str(self.cache_dir),
                preprocessing={
                    "days_back": days_back,
                    "min_magnitude": min_magnitude,
                    "significant_magnitude": anomaly_threshold,
                },
            )
        )
        loader.download()
        data, labels = loader._load_raw()
        return list(_Canonical.FEATURE_NAMES), (
            np.asarray(data, dtype=np.float64),
            np.asarray(labels, dtype=np.float64),
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


class MIMICLoader(DatasetLoader):
    """MIMIC-III Medical ICU Data Loader (IRB Placeholder Simulation).

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
        """Initialize the instance."""
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
        use_synthetic: bool = False,
        **kwargs: Any,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], DatasetMetadata]:
        """Load MIMIC-III-like synthetic data.

        Args:
            n_samples: Number of samples to generate
            anomaly_type: Type of anomaly to simulate ("sepsis", "cardiac", "mortality")
            use_synthetic: Use synthetic data (for testing without PhysioNet credentials)

        Returns:
            Tuple of (features, labels, metadata)

        Note:
            This generates SYNTHETIC data simulating MIMIC-III patterns.
            Real MIMIC-III access requires PhysioNet credentialing.
        """
        import time

        start_time = time.time()

        # Real MIMIC-III requires PhysioNet credentialing not supported by
        # this loader -- every code path here yields the synthetic simulation.
        # Both the explicit caller request (``use_synthetic=True``) and the
        # implicit fallback (``use_synthetic=False``) therefore go through
        # the single policy gate.  Without ``MERCURY_ALLOW_SYNTHETIC=1`` set
        # at the deployment level, the gate raises
        # ``DataSourceUnavailableError`` -- closing the prior bypass where
        # ``use_synthetic=True`` skipped the policy entirely.
        check_synthetic_allowed(
            "MIMIC-III",
            (
                "MIMIC-III requires PhysioNet credentialing not supported "
                "by this loader; only synthetic simulation is available. "
                "Real data: https://physionet.org/content/mimiciii/1.4/"
            ),
        )

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
        """Check IRB/credentialing status for real MIMIC-III access.

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


class _CanonicalNOAALoader(DatasetLoader):
    """Validation-interface wrapper over a canonical ``datasets/`` loader.

    The NOAA trio used to carry full local implementations whose "real"
    paths fabricated most features (Kp-derived columns, constant radii, RNG
    oceanography — see the module docstring's history note). Each wrapper
    now delegates to its canonical loader, which reads every channel from
    the instrument that measures it, and keeps this module's historical
    ``load()``/``get_train_test_split()`` interface.
    """

    #: Name used by the synthetic-policy gate and the metadata record.
    _GATE_NAME = "override"
    _SOURCE_NAME = "override"
    _LICENSE = "Public Domain (U.S. Government Work)"
    _CITATION = "override"

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        """Initialize the instance."""
        self.cache_dir = (
            Path(cache_dir) if cache_dir else Path.home() / ".omni_mercury" / "datasets"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._data: np.ndarray[Any, Any] | None = None
        self._labels: np.ndarray[Any, Any] | None = None
        self._metadata: DatasetMetadata | None = None

    def _canonical(self, n_samples: int, **preprocessing: Any) -> Any:
        """Build the canonical loader this wrapper delegates to."""
        raise NotImplementedError

    def _delegate_load(
        self,
        use_synthetic: bool,
        n_samples: int,
        min_real_samples: int,
        **preprocessing: Any,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], DatasetMetadata]:
        import time

        start_time = time.time()
        loader = self._canonical(n_samples, **preprocessing)
        if use_synthetic:
            # Caller-flag bypass closure: route through the policy gate so the
            # deployment-level ``MERCURY_ALLOW_SYNTHETIC=0`` floor cannot be
            # circumvented by a caller passing ``use_synthetic=True``.
            check_synthetic_allowed(self._GATE_NAME, "Caller passed use_synthetic=True")
            loader._create_synthetic()
            source = "synthetic"
        else:
            loader.download()
            source = self._SOURCE_NAME if loader.is_real_data else "synthetic"
        data, labels = loader._load_raw()
        if not use_synthetic and loader.is_real_data and len(data) < min_real_samples:
            raise RuntimeError(
                f"{self._GATE_NAME}: Failed to load minimum {min_real_samples} real "
                f"samples. Got {len(data)}. API may be down or data unavailable."
            )
        self._data = np.asarray(data, dtype=np.float64)
        self._labels = np.asarray(labels, dtype=np.float64)
        num_anomalies = int(self._labels.sum())
        self._metadata = DatasetMetadata(
            name=self._GATE_NAME,
            source=source,
            num_samples=len(self._data),
            num_features=int(self._data.shape[1]),
            num_anomalies=num_anomalies,
            anomaly_ratio=num_anomalies / len(self._labels) if len(self._labels) > 0 else 0,
            feature_names=list(loader.FEATURE_NAMES),
            load_time_seconds=time.time() - start_time,
            checksum=hashlib.sha3_256(self._data.tobytes()).hexdigest()[:16],
            license=self._LICENSE,
            citation=self._CITATION,
        )
        logger.info(
            f"Loaded {self._GATE_NAME}: {self._metadata.num_samples} samples, "
            f"{self._metadata.anomaly_ratio:.2%} anomalous"
        )
        return self._data, self._labels, self._metadata

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


class NOAASpaceWeatherLoader(_CanonicalNOAALoader):
    """NOAA SWPC space-weather loader (validation interface).

    Delegates to :class:`omni_mercury_engine.datasets.space_weather.
    NOAASpaceWeatherLoader`, which reads Kp, solar-wind plasma/IMF, GOES
    X-ray and integral proton flux from their own SWPC products. The
    pre-port implementation fetched only Kp and synthesised the other
    thirteen features as deterministic functions of it.
    """

    _GATE_NAME = "NOAA Space Weather"
    _SOURCE_NAME = "NOAA Space Weather Prediction Center"
    _CITATION = "NOAA Space Weather Prediction Center. https://www.swpc.noaa.gov/"

    def _canonical(self, n_samples: int, **preprocessing: Any) -> Any:
        from omni_mercury_engine.datasets.base import DatasetConfig
        from omni_mercury_engine.datasets.space_weather import (
            NOAASpaceWeatherLoader as _Canonical,
        )

        return _Canonical(
            DatasetConfig(
                name="noaa_space_weather",
                data_dir=str(self.cache_dir),
                cache_dir=str(self.cache_dir),
                max_samples=n_samples,
                preprocessing=preprocessing,
            )
        )

    def load(
        self,
        use_synthetic: bool = False,
        n_samples: int = 5000,
        storm_threshold: float = 5.0,
        min_real_samples: int = 100,
        **kwargs: Any,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], DatasetMetadata]:
        """Load NOAA space-weather data (see the canonical loader for features).

        Args:
            use_synthetic: Use synthetic data (policy-gated).
            n_samples: Sample count for the synthetic path.
            storm_threshold: Kp threshold for storm labels (G1 = 5).
            min_real_samples: Minimum required real samples.

        Returns:
            Tuple of (features, labels, metadata).
        """
        return self._delegate_load(
            use_synthetic, n_samples, min_real_samples, storm_threshold=storm_threshold
        )


class NOAAHurricaneLoader(_CanonicalNOAALoader):
    """NOAA NHC HURDAT2 loader (validation interface).

    Delegates to :class:`omni_mercury_engine.datasets.hurricane.
    NOAAHurricaneLoader`, which parses the full best-track layout (real
    wind radii, kinematics derived from consecutive fixes). The pre-port
    implementation padded fourteen constant columns and compared HURDAT2's
    knots against a 111 *mph* threshold; the threshold is now converted
    (mph -> kt) before it meets the data.
    """

    _GATE_NAME = "NOAA Hurricane"
    _SOURCE_NAME = "NOAA National Hurricane Center"
    _CITATION = "NOAA National Hurricane Center. https://www.nhc.noaa.gov/"

    #: Statute miles per hour per knot.
    _MPH_PER_KT = 1.15078

    def _canonical(self, n_samples: int, **preprocessing: Any) -> Any:
        from omni_mercury_engine.datasets.base import DatasetConfig
        from omni_mercury_engine.datasets.hurricane import NOAAHurricaneLoader as _Canonical

        return _Canonical(
            DatasetConfig(
                name="noaa_hurricane",
                data_dir=str(self.cache_dir),
                cache_dir=str(self.cache_dir),
                max_samples=n_samples,
                preprocessing=preprocessing,
            )
        )

    def load(
        self,
        use_synthetic: bool = False,
        n_samples: int = 3000,
        major_hurricane_threshold: float = 111.0,
        min_real_samples: int = 100,
        **kwargs: Any,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], DatasetMetadata]:
        """Load HURDAT2 best-track data (see the canonical loader for features).

        Args:
            use_synthetic: Use synthetic data (policy-gated).
            n_samples: Sample count for the synthetic path.
            major_hurricane_threshold: Major-hurricane wind threshold in
                **mph** (kept for interface compatibility; Category 3 is
                111 mph). Converted to knots before it meets HURDAT2's
                knot-valued winds — the pre-port code compared the mph
                number against knots directly and under-labelled.
            min_real_samples: Minimum required real samples.

        Returns:
            Tuple of (features, labels, metadata).
        """
        return self._delegate_load(
            use_synthetic,
            n_samples,
            min_real_samples,
            major_threshold_kt=major_hurricane_threshold / self._MPH_PER_KT,
        )


class NOAAOceanLoader(_CanonicalNOAALoader):
    """NOAA CO-OPS station loader (validation interface).

    Delegates to :class:`omni_mercury_engine.datasets.ocean.
    NOAACoOpsStationLoader`, which fetches each real station product
    (water/air temperature, pressure, hourly water level, wind) and labels
    heatwave-like hours at the in-window monthly 90th percentile of water
    temperature. The pre-port implementation fetched one real channel and
    generated seventeen RNG columns under ``source="NOAA National Ocean
    Service"``; its absolute ``heatwave_threshold`` applied to that
    fabricated matrix and is accepted-but-ignored for compatibility.
    """

    _GATE_NAME = "NOAA Ocean"
    _SOURCE_NAME = "NOAA CO-OPS (tidesandcurrents.noaa.gov)"
    _CITATION = (
        "NOAA Center for Operational Oceanographic Products and Services (CO-OPS). "
        "https://tidesandcurrents.noaa.gov/"
    )

    def _canonical(self, n_samples: int, **preprocessing: Any) -> Any:
        from omni_mercury_engine.datasets.base import DatasetConfig
        from omni_mercury_engine.datasets.ocean import NOAACoOpsStationLoader as _Canonical

        return _Canonical(
            DatasetConfig(
                name="noaa_coops_station",
                data_dir=str(self.cache_dir),
                cache_dir=str(self.cache_dir),
                max_samples=n_samples,
                preprocessing=preprocessing,
            )
        )

    def load(
        self,
        use_synthetic: bool = False,
        n_samples: int = 2000,
        heatwave_threshold: float = 2.0,
        min_real_samples: int = 100,
        station: str = "8454000",
        days_back: int = 60,
        **kwargs: Any,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], DatasetMetadata]:
        """Load CO-OPS station data (see the canonical loader for features).

        Args:
            use_synthetic: Use synthetic data (policy-gated).
            n_samples: Sample count for the synthetic path.
            heatwave_threshold: Accepted for interface compatibility and
                ignored — it thresholded a fabricated ``sst_anomaly`` RNG
                column; labels now come from the canonical in-window
                monthly 90th percentile of the real water temperature.
            min_real_samples: Minimum required real samples.
            station: CO-OPS station id.
            days_back: Days of history to fetch.

        Returns:
            Tuple of (features, labels, metadata).
        """
        if heatwave_threshold != 2.0:
            logger.warning(
                "NOAAOceanLoader: heatwave_threshold is accepted for interface "
                "compatibility and ignored; labels use the canonical in-window "
                "monthly p90 of the real water temperature."
            )
        return self._delegate_load(
            use_synthetic, n_samples, min_real_samples, station=station, days_back=days_back
        )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

# Map of canonical dataset names → concrete ``DatasetLoader`` class.
# The keys are deliberately lowercase / hyphenated for ergonomic
# operator use; ``get_loader`` lower-cases the input so callers may
# pass any case variant.
_LOADER_REGISTRY: dict[str, type[DatasetLoader]] = {
    "nslkdd": NSLKDDLoader,
    "nsl-kdd": NSLKDDLoader,
    "usgs": USGSEarthquakeLoader,
    "usgs-earthquake": USGSEarthquakeLoader,
    "earthquake": USGSEarthquakeLoader,
    "mimic": MIMICLoader,
    "noaa-space-weather": NOAASpaceWeatherLoader,
    "space-weather": NOAASpaceWeatherLoader,
    "noaa-hurricane": NOAAHurricaneLoader,
    "hurricane": NOAAHurricaneLoader,
    "noaa-ocean": NOAAOceanLoader,
    "ocean": NOAAOceanLoader,
}


def get_loader(name: str, **kwargs: Any) -> DatasetLoader:
    """Resolve a dataset name to a concrete :class:`DatasetLoader`.

    The validation-pipeline previously shipped a factory of this name
    but it was lost during a refactor; the operator tooling and the
    property-based test in ``tests/test_property_based.py`` both
    expect it.  Re-introducing it here keeps the factory side-by-side
    with the concrete loaders so the next refactor cannot quietly
    delete one but not the other.

    Args:
        name: Canonical loader name (case-insensitive). See
            ``_LOADER_REGISTRY`` for the full list.
        **kwargs: Forwarded to the loader constructor.

    Returns:
        A fresh, un-loaded :class:`DatasetLoader` instance.

    Raises:
        ValueError: If ``name`` is not a known loader.
    """
    key = name.strip().lower() if isinstance(name, str) else ""
    cls = _LOADER_REGISTRY.get(key)
    if cls is None:
        known = ", ".join(sorted(set(_LOADER_REGISTRY)))
        raise ValueError(f"Unknown dataset loader {name!r}; known loaders: {known}")
    return cls(**kwargs)
