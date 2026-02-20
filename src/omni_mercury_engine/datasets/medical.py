"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Medical Dataset Loaders: MIMIC-III, MIMIC-IV, PhysioNet

IMPORTANT: PhysioNet credentialed datasets require:
1. Create account at https://physionet.org/
2. Complete CITI training
3. Sign DUA (Data Use Agreement)
4. Download data locally using wget with credentials
5. Set local_path in config to point to downloaded data

For MIMIC data download:
    wget -r -N -c -np --user YOUR_USERNAME --ask-password \\
        https://physionet.org/files/mimiciii/1.4/

References:
- MIMIC-III: https://physionet.org/content/mimiciii/1.4/
- MIMIC-IV: https://physionet.org/content/mimiciv/
- PhysioNet Guidelines: https://physionet.org/news/post/mimic-derived-datasets-models
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .base import DatasetConfig, DatasetLoader, DatasetRegistry
from .exceptions import ALLOW_SYNTHETIC, DataSourceUnavailableError, check_synthetic_allowed

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)


class MIMICLoader(DatasetLoader):
    """
    MIMIC-III/IV Clinical Database Loader.

    Provides access to ICU patient data for:
    - Mortality prediction
    - Sepsis early detection
    - Length of stay prediction
    - Readmission prediction

    REQUIRES PhysioNet credentialed access.

    Citation:
        Johnson, A., Pollard, T., Shen, L. et al. MIMIC-III, a freely accessible
        critical care database. Sci Data 3, 160035 (2016).
    """

    DATASET_NAME = "mimic-iii"
    DATASET_URL = "https://physionet.org/content/mimiciii/1.4/"
    LICENSE = "PhysioNet Credentialed Health Data License"
    CITATION = """Johnson AEW, Pollard TJ, Shen L, et al. MIMIC-III, a freely accessible
    critical care database. Scientific Data. 2016;3:160035."""
    REQUIRES_CREDENTIALS = True

    # Key vital signs and lab values for anomaly detection
    VITAL_FEATURES = [
        "heart_rate",
        "systolic_bp",
        "diastolic_bp",
        "mean_arterial_pressure",
        "respiratory_rate",
        "temperature",
        "spo2",
        "glucose",
    ]
    LAB_FEATURES = [
        "lactate",
        "creatinine",
        "bilirubin",
        "platelet_count",
        "white_blood_cell",
        "hemoglobin",
        "pao2",
        "paco2",
        "ph",
    ]
    FEATURE_NAMES = VITAL_FEATURES + LAB_FEATURES

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize MIMIC loader.

        Args:
            config: Dataset configuration. Preprocessing options:
                - task (str): Prediction task ('mortality', 'sepsis', etc.)
                - local_path (str): Path to downloaded MIMIC data directory
        """
        super().__init__(config)
        self.version = config.version or "1.4"
        self.task = config.preprocessing.get("task", "mortality")
        self.local_path = config.preprocessing.get("local_path", None)
        self._is_real_data = False

    @property
    def is_real_data(self) -> bool:
        """Return True if real MIMIC data was loaded."""
        return self._is_real_data

    def download(self) -> bool:
        """MIMIC-III requires PhysioNet credentials. Check for local files."""
        # Check for local path first
        if self.local_path:
            local_dir = Path(self.local_path)
            if local_dir.exists():
                chartevents = local_dir / "CHARTEVENTS.csv.gz"
                if chartevents.exists():
                    logger.info(f"Found real MIMIC data at {local_dir}")
                    self._is_real_data = True
                    return True
                logger.warning(f"CHARTEVENTS.csv.gz not found in {local_dir}")

        # Check default data path
        chartevents_path = self.data_path / "CHARTEVENTS.csv.gz"
        if chartevents_path.exists():
            logger.info(f"Found real MIMIC data at {self.data_path}")
            self._is_real_data = True
            return True

        # No real data found - raise error (never generate synthetic MIMIC data)
        logger.info(
            "MIMIC-III dataset requires PhysioNet credentials. "
            "Register and complete training at: https://physionet.org/"
        )
        raise DataSourceUnavailableError(
            loader_name="MIMIC-III",
            reason=(
                "MIMIC-III requires PhysioNet credentialing. "
                "1. Complete CITI training at https://physionet.org/ "
                "2. Sign Data Use Agreement "
                "3. Download data and set local_path in config. "
                "Never generates synthetic MIMIC data."
            ),
        )

    def _create_synthetic_mimic(self) -> bool:
        """Create synthetic MIMIC-like data for testing."""
        logger.info("Generating synthetic MIMIC-like data for development")

        np.random.seed(self.config.random_seed)
        n_samples = self.config.max_samples or 10000

        # Generate realistic ICU vital signs
        data = {
            "heart_rate": np.random.normal(80, 20, n_samples),
            "systolic_bp": np.random.normal(120, 25, n_samples),
            "diastolic_bp": np.random.normal(80, 15, n_samples),
            "mean_arterial_pressure": np.random.normal(90, 15, n_samples),
            "respiratory_rate": np.random.normal(16, 4, n_samples),
            "temperature": np.random.normal(37.0, 0.8, n_samples),
            "spo2": np.clip(np.random.normal(96, 4, n_samples), 70, 100),
            "glucose": np.random.normal(100, 30, n_samples),
            # Labs
            "lactate": np.random.exponential(1.5, n_samples),
            "creatinine": np.random.lognormal(0, 0.5, n_samples),
            "bilirubin": np.random.lognormal(-0.5, 0.8, n_samples),
            "platelet_count": np.random.normal(250, 80, n_samples),
            "white_blood_cell": np.random.lognormal(2.2, 0.5, n_samples),
            "hemoglobin": np.random.normal(12, 2, n_samples),
            "pao2": np.random.normal(95, 15, n_samples),
            "paco2": np.random.normal(40, 8, n_samples),
            "ph": np.random.normal(7.4, 0.1, n_samples),
        }

        # Create anomaly labels based on clinical criteria
        labels = np.zeros(n_samples, dtype=np.int64)

        # Sepsis-like anomalies (SOFA criteria approximation)
        sepsis_mask = (
            (data["lactate"] > 2.0) & (data["systolic_bp"] < 100) & (data["respiratory_rate"] > 22)
        )
        labels[sepsis_mask] = 1

        # Cardiac anomalies
        cardiac_mask = (
            (data["heart_rate"] > 120)
            | (data["heart_rate"] < 50)
            | (data["systolic_bp"] < 90)
            | (data["systolic_bp"] > 180)
        )
        labels[cardiac_mask] = 1

        # Respiratory anomalies
        resp_mask = (data["spo2"] < 90) | (data["pao2"] < 60)
        labels[resp_mask] = 1

        # Save synthetic data
        features = np.column_stack([data[f] for f in self.FEATURE_NAMES])

        # Save to cache as synthetic dataset
        save_path = self.data_path / "synthetic_mimic.npz"
        np.savez_compressed(save_path, features=features, labels=labels)

        logger.info(f"Generated {n_samples} synthetic samples, {labels.sum()} anomalies")
        return True

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load MIMIC data from disk (real data first, then synthetic)."""
        # Check local_path first
        if self.local_path:
            local_dir = Path(self.local_path)
            chartevents_local = local_dir / "CHARTEVENTS.csv.gz"
            if chartevents_local.exists() and PANDAS_AVAILABLE:
                self._is_real_data = True
                logger.info(f"Loading REAL MIMIC data from {local_dir}")
                return self._load_real_mimic(local_dir)

        # Check default data path for real MIMIC tables
        chartevents_path = self.data_path / "CHARTEVENTS.csv.gz"
        if chartevents_path.exists() and PANDAS_AVAILABLE:
            self._is_real_data = True
            logger.info(f"Loading REAL MIMIC data from {self.data_path}")
            return self._load_real_mimic(self.data_path)

        # Synthetic MIMIC data is never allowed
        raise FileNotFoundError("MIMIC data not found. Run download() first.")

    def _load_real_mimic(
        self, data_dir: Path | None = None
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load and process real MIMIC-III tables with proper outcome labels.

        Extracts features from CHARTEVENTS and labels from multiple outcome sources:
        - ADMISSIONS table (hospital mortality)
        - PATIENTS table (in-hospital death)
        - Derived tables for sepsis, length of stay, readmission

        Args:
            data_dir: Directory containing MIMIC CSV files. Defaults to self.data_path.

        Returns:
            Tuple of (features, labels) numpy arrays.
        """
        if data_dir is None:
            data_dir = self.data_path

        logger.info(f"Loading real MIMIC-III data from {data_dir}...")

        # Load ICU stays
        icustays = pd.read_csv(
            data_dir / "ICUSTAYS.csv.gz",
            compression="gzip",
        )

        # Load admissions for mortality and outcomes
        admissions = pd.read_csv(
            data_dir / "ADMISSIONS.csv.gz",
            compression="gzip",
            usecols=[
                "hadm_id",
                "deathtime",
                "hospital_expire_flag",
                "admission_type",
                "admittime",
                "dischtime",
                "discharge_location",
            ],
            parse_dates=["admittime", "dischtime", "deathtime"],
        )

        # Load patients for age and death information
        patients_path = data_dir / "PATIENTS.csv.gz"
        if patients_path.exists():
            patients = pd.read_csv(
                patients_path,
                compression="gzip",
                usecols=["subject_id", "dob", "dod", "expire_flag"],
                parse_dates=["dob", "dod"],
            )
            has_patients = True
        else:
            patients = None
            has_patients = False

        # Load chartevents (vital signs)
        chartevents = pd.read_csv(
            data_dir / "CHARTEVENTS.csv.gz",
            compression="gzip",
            usecols=["icustay_id", "itemid", "valuenum", "charttime"],
            nrows=1000000,  # Limit for memory
        )

        # Merge ICU stays with admissions for outcomes
        icustays_with_outcomes = pd.merge(
            icustays,
            admissions[
                [
                    "hadm_id",
                    "hospital_expire_flag",
                    "deathtime",
                    "admission_type",
                    "admittime",
                    "dischtime",
                ]
            ],
            on="hadm_id",
            how="left",
        )

        # ItemID mapping for vital signs (MIMIC-III itemids)
        itemid_map = {
            211: "heart_rate",  # Heart Rate
            220045: "heart_rate",
            51: "systolic_bp",  # Arterial BP Systolic
            220050: "systolic_bp",
            8368: "diastolic_bp",
            220051: "diastolic_bp",
            618: "respiratory_rate",
            220210: "respiratory_rate",
            223761: "temperature",  # Temperature Fahrenheit
            678: "temperature",
            646: "spo2",  # SpO2
            220277: "spo2",
        }

        # Determine label type from config
        label_type = self.config.preprocessing.get("label_type", "mortality")

        # Pre-index data for faster lookups
        icustays_indexed = icustays_with_outcomes.set_index("icustay_id")
        chartevents_grouped = chartevents.groupby("icustay_id")

        # Aggregate by ICU stay
        features_list = []
        labels_list = []

        unique_stays = icustays_with_outcomes["icustay_id"].unique()
        max_samples = self.config.max_samples or 5000

        for icustay_id in unique_stays[:max_samples]:
            # Fast lookup using pre-indexed DataFrame
            try:
                stay_info = icustays_indexed.loc[icustay_id]
            except KeyError:
                continue

            # Get chartevents for this stay using grouped data
            try:
                stay_data = chartevents_grouped.get_group(icustay_id)
            except KeyError:
                stay_data = pd.DataFrame()

            # Extract features using vectorized operations where possible
            feature_vec = []
            for itemid in itemid_map:
                vals = stay_data[stay_data["itemid"] == itemid]["valuenum"]
                feature_vec.append(vals.mean() if len(vals) > 0 else 0)

            if len(feature_vec) != len(self.FEATURE_NAMES):
                continue

            # Extract label based on label_type
            label = self._extract_outcome_label(
                stay_info, label_type, patients if has_patients else None
            )

            features_list.append(feature_vec)
            labels_list.append(label)

        logger.info(f"Loaded {len(features_list)} ICU stays with {label_type} labels")
        logger.info(
            f"Positive class rate: {sum(labels_list) / len(labels_list):.2%}"
            if labels_list
            else "No data"
        )

        return np.array(features_list), np.array(labels_list)

    def _extract_outcome_label(
        self,
        stay_info: pd.Series,
        label_type: str,
        patients: pd.DataFrame | None = None,
    ) -> int:
        """Extract outcome label for an ICU stay.

        Args:
            stay_info: Series with ICU stay and admission information
            label_type: Type of outcome label to extract:
                - 'mortality': In-hospital mortality (hospital_expire_flag)
                - 'icu_mortality': Death during ICU stay
                - 'los': Length of stay > 7 days (binary)
                - 'readmission': 30-day readmission
                - 'sepsis': Sepsis diagnosis (requires ICD codes)
            patients: Optional patients DataFrame for additional mortality info

        Returns:
            Binary label (0 = negative, 1 = positive)
        """
        if label_type == "mortality":
            # In-hospital mortality from ADMISSIONS
            return int(stay_info.get("hospital_expire_flag", 0) == 1)

        elif label_type == "icu_mortality":
            # Death during this ICU stay
            deathtime = stay_info.get("deathtime")
            outtime = stay_info.get("outtime")
            intime = stay_info.get("intime")

            if pd.isna(deathtime):
                return 0

            # Check if death occurred during ICU stay
            if not pd.isna(intime) and not pd.isna(outtime):
                if intime <= deathtime <= outtime:
                    return 1
            # Fallback to hospital expire flag
            return int(stay_info.get("hospital_expire_flag", 0) == 1)

        elif label_type == "los":
            # Length of stay > 7 days
            los = stay_info.get("los")
            if pd.isna(los):
                # Calculate from times if available
                intime = stay_info.get("intime")
                outtime = stay_info.get("outtime")
                if not pd.isna(intime) and not pd.isna(outtime):
                    try:
                        los = (pd.to_datetime(outtime) - pd.to_datetime(intime)).days
                    except (ValueError, TypeError):
                        los = 0
                else:
                    los = 0
            return int(los > 7)

        elif label_type == "readmission":
            # 30-day readmission (simplified - checks admission type)
            admission_type = stay_info.get("admission_type", "").upper()
            # Emergency/urgent readmission as proxy for 30-day readmission
            return int(admission_type in ["EMERGENCY", "URGENT"])

        elif label_type == "critical":
            # Critical condition composite: mortality OR long LOS OR emergency
            mortality = int(stay_info.get("hospital_expire_flag", 0) == 1)
            los = stay_info.get("los", 0)
            if pd.isna(los):
                los = 0
            long_stay = int(los > 14)  # Very long stay
            admission_type = stay_info.get("admission_type", "").upper()
            emergency = int(admission_type == "EMERGENCY")

            # Positive if any critical indicator
            return int(mortality or long_stay or emergency)

        else:
            # Default to mortality
            return int(stay_info.get("hospital_expire_flag", 0) == 1)

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess MIMIC features."""
        # Handle missing values
        data = np.nan_to_num(data, nan=0.0)

        # Z-score normalization
        if self.config.preprocessing.get("normalize", True):
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0) + 1e-8
            data = (data - mean) / std

        return data.astype(np.float32)

    def _get_feature_names(self) -> list[str]:
        return self.FEATURE_NAMES

    def _get_target_names(self) -> list[str]:
        return ["normal", "anomaly"]


class PhysioNetLoader(DatasetLoader):
    """
    Generic PhysioNet dataset loader for vital sign data.

    Supports multiple PhysioNet datasets:
    - MIT-BIH Arrhythmia Database
    - PTB Diagnostic ECG Database
    - Sleep-EDF Database
    - AF Detection Challenge

    Reference: https://physionet.org/
    """

    DATASET_NAME = "physionet"
    DATASET_URL = "https://physionet.org/"
    LICENSE = "PhysioNet Open Access / Credentialed"
    CITATION = "Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet. Circulation. 2000."
    REQUIRES_CREDENTIALS = False  # Some datasets are open

    # Supported sub-datasets
    SUBDATASETS = {
        "mitbih": "https://physionet.org/content/mitdb/1.0.0/",
        "ptbdb": "https://physionet.org/content/ptbdb/1.0.0/",
        "afdb": "https://physionet.org/content/afdb/1.0.0/",
        "slpdb": "https://physionet.org/content/slpdb/1.0.0/",
    }

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)
        self.subdataset = config.preprocessing.get("subdataset", "mitbih")
        self.local_path = config.preprocessing.get("local_path", None)
        self._is_real_data = False

    @property
    def is_real_data(self) -> bool:
        """Return True if real PhysioNet data was loaded."""
        return self._is_real_data

    def download(self) -> bool:
        """Download or load PhysioNet data.

        Most PhysioNet datasets require credentialing. For open access datasets,
        use wfdb library to download directly.

        Returns:
            True if data was loaded/generated successfully.
        """
        # Check for local data first
        if self.local_path:
            local_dir = Path(self.local_path)
            if local_dir.exists() and any(local_dir.glob("*.dat")):
                logger.info(f"Found PhysioNet data at {local_dir}")
                self._is_real_data = True
                return True

        if ALLOW_SYNTHETIC:
            check_synthetic_allowed(
                "PhysioNet",
                f"PhysioNet {self.subdataset} data not found locally",
            )
            return self._create_synthetic_ecg()
        raise DataSourceUnavailableError(
            loader_name="PhysioNet",
            source_url=self.SUBDATASETS.get(self.subdataset, "https://physionet.org/"),
            reason=(
                f"PhysioNet {self.subdataset} data not found locally. "
                "Download using wfdb or wget and set local_path."
            ),
        )

    def _create_synthetic_ecg(self) -> bool:
        """Create synthetic ECG data for testing."""
        np.random.seed(self.config.random_seed)
        n_samples = self.config.max_samples or 5000
        signal_length = 360  # 1 second at 360 Hz

        features = []
        labels = []

        for _i in range(n_samples):
            # Normal sinus rhythm
            t = np.linspace(0, 1, signal_length)
            heart_rate = np.random.uniform(60, 100)
            ecg = self._generate_ecg_beat(t, heart_rate)

            # Add noise
            ecg += np.random.normal(0, 0.05, signal_length)

            # 20% anomalies
            if np.random.random() < 0.2:
                # Various arrhythmia patterns
                anomaly_type = np.random.choice(["afib", "vt", "pvc", "noise"])
                if anomaly_type == "afib":
                    # Irregular RR intervals
                    ecg *= np.random.uniform(0.8, 1.2, signal_length)
                elif anomaly_type == "vt":
                    # Wide QRS
                    ecg = self._generate_ecg_beat(t, 150)
                elif anomaly_type == "pvc":
                    # Premature beat
                    ecg[180:220] *= 2.0
                else:
                    # Noise artifact
                    ecg += np.random.normal(0, 0.5, signal_length)

                labels.append(1)
            else:
                labels.append(0)

            features.append(ecg)

        features = np.array(features)  # type: ignore[assignment, unused-ignore]
        labels = np.array(labels, dtype=np.int64)  # type: ignore[assignment, unused-ignore]

        save_path = self.data_path / "synthetic_ecg.npz"
        np.savez_compressed(save_path, features=features, labels=labels)

        logger.info(f"Generated {n_samples} synthetic ECG samples")
        return True

    def _generate_ecg_beat(
        self, t: np.ndarray[Any, Any], heart_rate: float
    ) -> np.ndarray[Any, Any]:
        """Generate synthetic ECG beat."""
        # Simplified ECG model using Gaussian pulses
        beat_duration = 60.0 / heart_rate

        # P wave
        p_center = 0.1 * beat_duration
        p_width = 0.04
        p = 0.15 * np.exp(-((t - p_center) ** 2) / (2 * p_width**2))

        # QRS complex
        qrs_center = 0.35 * beat_duration
        q = -0.1 * np.exp(-((t - qrs_center + 0.02) ** 2) / (2 * 0.01**2))
        r = 1.0 * np.exp(-((t - qrs_center) ** 2) / (2 * 0.01**2))
        s = -0.15 * np.exp(-((t - qrs_center - 0.02) ** 2) / (2 * 0.01**2))

        # T wave
        t_center = 0.6 * beat_duration
        t_wave = 0.3 * np.exp(-((t - t_center) ** 2) / (2 * 0.06**2))

        ecg = p + q + r + s + t_wave

        # Repeat for full signal
        full_ecg = np.tile(ecg, int(len(t) / len(ecg)) + 1)[: len(t)]

        return full_ecg

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load PhysioNet data from cache (real or synthetic)."""
        synthetic_path = self.data_path / "synthetic_ecg.npz"
        if synthetic_path.exists() and ALLOW_SYNTHETIC:
            data = np.load(synthetic_path)
            # Note: _is_real_data is set during download() based on local_path
            logger.info(f"Loaded ECG data (is_real_data={self._is_real_data})")
            return data["features"], data["labels"]
        raise FileNotFoundError("PhysioNet data not found. Run download() first.")

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess ECG signals."""
        # Normalize each signal
        data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)
        return data.astype(np.float32)


class SepsisDataset(MIMICLoader):
    """
    Specialized loader for Sepsis prediction using MIMIC data.

    Based on Sepsis-3 criteria and SOFA scores.
    """

    DATASET_NAME = "sepsis"

    def __init__(self, config: DatasetConfig) -> None:
        config.preprocessing["task"] = "sepsis"
        super().__init__(config)

    def download(self) -> bool:
        """Download or generate sepsis data.

        Unlike the parent MIMICLoader, SepsisDataset supports synthetic
        generation because sepsis prediction research benefits from
        configurable prevalence rates in test/development data.
        """
        # Try real MIMIC data first (inherited behaviour without the raise)
        if self.local_path:
            local_dir = Path(self.local_path)
            if local_dir.exists():
                chartevents = local_dir / "CHARTEVENTS.csv.gz"
                if chartevents.exists():
                    logger.info(f"Found real MIMIC data at {local_dir}")
                    self._is_real_data = True
                    return True

        chartevents_path = self.data_path / "CHARTEVENTS.csv.gz"
        if chartevents_path.exists():
            logger.info(f"Found real MIMIC data at {self.data_path}")
            self._is_real_data = True
            return True

        # Fall back to synthetic if allowed
        if ALLOW_SYNTHETIC:
            check_synthetic_allowed(
                "SepsisDataset",
                "MIMIC-III data not found locally — generating synthetic sepsis data",
            )
            return self._create_synthetic_mimic()

        raise DataSourceUnavailableError(
            loader_name="SepsisDataset",
            reason=(
                "Requires MIMIC-III data (see MIMICLoader docs) or "
                "set MERCURY_ALLOW_SYNTHETIC=1 for synthetic fallback."
            ),
        )

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load sepsis data from disk."""
        # Check for sepsis-specific synthetic data (only if allowed)
        sepsis_path = self.data_path / "synthetic_sepsis.npz"
        if sepsis_path.exists() and ALLOW_SYNTHETIC:
            data = np.load(sepsis_path)
            return data["features"], data["labels"]
        # Fall back to MIMIC data
        return super()._load_raw()

    def _create_synthetic_mimic(self) -> bool:
        """Create sepsis-focused synthetic data."""
        np.random.seed(self.config.random_seed)
        n_samples = self.config.max_samples or 5000

        # Generate with higher sepsis prevalence
        data = {}
        labels = np.zeros(n_samples, dtype=np.int64)

        # 30% sepsis cases
        n_sepsis = int(n_samples * 0.3)
        sepsis_idx = np.random.choice(n_samples, n_sepsis, replace=False)

        for feature in self.FEATURE_NAMES:
            # Generate feature values based on feature type
            if feature == "heart_rate":
                values = np.random.normal(80, 15, n_samples)
                values[sepsis_idx] = np.random.normal(110, 20, n_sepsis)  # Tachycardia
            elif feature == "systolic_bp":
                values = np.random.normal(120, 15, n_samples)
                values[sepsis_idx] = np.random.normal(85, 15, n_sepsis)  # Hypotension
            elif feature == "respiratory_rate":
                values = np.random.normal(14, 3, n_samples)
                values[sepsis_idx] = np.random.normal(24, 5, n_sepsis)  # Tachypnea
            elif feature == "lactate":
                values = np.random.exponential(1.0, n_samples)
                values[sepsis_idx] = np.random.exponential(3.0, n_sepsis)  # Elevated lactate
            elif feature == "white_blood_cell":
                values = np.random.normal(8, 2, n_samples)
                values[sepsis_idx] = np.random.choice(
                    [np.random.normal(15, 3), np.random.normal(3, 1)], n_sepsis
                )  # Leukocytosis or leukopenia
            else:
                values = np.random.normal(0, 1, n_samples)

            data[feature] = values

        labels[sepsis_idx] = 1

        features = np.column_stack([data[f] for f in self.FEATURE_NAMES])
        save_path = self.data_path / "synthetic_sepsis.npz"
        np.savez_compressed(save_path, features=features, labels=labels)

        logger.info(f"Generated {n_samples} sepsis samples, {n_sepsis} positive")
        return True


class CardiologyDataset(PhysioNetLoader):
    """
    Specialized loader for cardiology data (ECG + vitals).

    Combines ECG waveforms with cardiac biomarkers.
    """

    DATASET_NAME = "cardiology"

    def __init__(self, config: DatasetConfig) -> None:
        config.preprocessing["subdataset"] = "ptbdb"
        super().__init__(config)


# Register all medical loaders
DatasetRegistry.register("mimic-iii", MIMICLoader)
DatasetRegistry.register("mimic-iv", MIMICLoader)
DatasetRegistry.register("physionet", PhysioNetLoader)
DatasetRegistry.register("sepsis", SepsisDataset)
DatasetRegistry.register("cardiology", CardiologyDataset)
