"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisors LLC

Industrial Control System (ICS) Dataset Loaders

Loaders for real-world ICS/SCADA anomaly detection benchmarks:
- SWaT (Secure Water Treatment): Singapore water treatment testbed
- WADI (Water Distribution): Water distribution system
- BATADAL: Water network attack detection

These datasets are critical for validating anomaly detection in:
- Critical infrastructure protection
- Industrial cybersecurity
- Process control systems

Ethical Note:
    These datasets are used for DEFENSIVE security research only.
    Mercury-Agent is designed to PROTECT infrastructure, never attack it.

References:
    - Goh et al., "A Dataset to Support Research in the Design of Secure
      Water Treatment Systems", CRITIS 2016
    - Ahmed et al., "WADI: A Water Distribution Testbed for Research
      in the Design of Secure Cyber Physical Systems", ICS-CSR 2017
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .base import DatasetConfig, DatasetLoader, DatasetMetadata, DatasetSplit, safe_urlretrieve
from .exceptions import DataSourceUnavailableError

logger = logging.getLogger(__name__)

__all__ = [
    "BATADALLoader",
    "SWaTLoader",
    "WADILoader",
]


class SWaTLoader(DatasetLoader):
    """
    Secure Water Treatment (SWaT) Dataset Loader.

    SWaT is a water treatment testbed at Singapore University of Technology
    and Design (SUTD). The dataset contains normal operation and 36 attack
    scenarios targeting PLCs and sensors.

    Data Source: iTrust Labs (https://itrust.sutd.edu.sg/itrust-labs-datasets/)
    Paper: Goh et al., CRITIS 2016

    Features:
    - 51 sensors and actuators
    - 11 days of data (7 normal + 4 under attack)
    - 36 unique attack scenarios
    - ~950K samples

    Requires registration at iTrust to download.
    """

    DATASET_NAME = "swat"
    DATASET_URL = "https://itrust.sutd.edu.sg/itrust-labs-datasets/dataset_info/"
    LICENSE = "Research Only (iTrust Agreement)"
    CITATION = """Goh J, Adepu S, Junejo KN, Mathur A.
    A Dataset to Support Research in the Design of Secure Water Treatment Systems.
    CRITIS 2016."""
    REQUIRES_CREDENTIALS = True

    # Feature names for SWaT
    FEATURE_NAMES = [
        # Process 1: Raw Water (P1)
        "FIT101",
        "LIT101",
        "MV101",
        "P101",
        "P102",
        # Process 2: Chemical Dosing (P2)
        "AIT201",
        "AIT202",
        "AIT203",
        "FIT201",
        "MV201",
        "P201",
        "P202",
        "P203",
        "P204",
        "P205",
        "P206",
        # Process 3: Ultrafiltration (P3)
        "DPIT301",
        "FIT301",
        "LIT301",
        "MV301",
        "MV302",
        "MV303",
        "MV304",
        "P301",
        "P302",
        # Process 4: Dechlorination (P4)
        "AIT401",
        "AIT402",
        "FIT401",
        "LIT401",
        "P401",
        "P402",
        "P403",
        "P404",
        "UV401",
        # Process 5: Reverse Osmosis (P5)
        "AIT501",
        "AIT502",
        "AIT503",
        "AIT504",
        "FIT501",
        "FIT502",
        "FIT503",
        "FIT504",
        "P501",
        "P502",
        "PIT501",
        "PIT502",
        "PIT503",
        # Process 6: Backwash (P6)
        "FIT601",
        "P601",
        "P602",
        "P603",
    ]

    NUM_FEATURES = 51
    ATTACK_COUNT = 36

    # Feature groups by process
    FEATURE_GROUPS = {
        "P1": ["FIT101", "LIT101", "MV101", "P101", "P102"],
        "P2": [
            "AIT201",
            "AIT202",
            "AIT203",
            "FIT201",
            "MV201",
            "P201",
            "P202",
            "P203",
            "P204",
            "P205",
            "P206",
        ],
        "P3": ["DPIT301", "FIT301", "LIT301", "MV301", "MV302", "MV303", "MV304", "P301", "P302"],
        "P4": ["AIT401", "AIT402", "FIT401", "LIT401", "P401", "P402", "P403", "P404", "UV401"],
        "P5": [
            "AIT501",
            "AIT502",
            "AIT503",
            "AIT504",
            "FIT501",
            "FIT502",
            "FIT503",
            "FIT504",
            "P501",
            "P502",
            "PIT501",
            "PIT502",
            "PIT503",
        ],
        "P6": ["FIT601", "P601", "P602", "P603"],
    }

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)
        self.attack_labels_map: dict[int, str] = {}

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load raw SWaT data - redirects to load()."""
        return self.load()

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Apply SWaT-specific preprocessing (normalization)."""
        # Z-score normalization
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-8
        return np.asarray((data - mean) / std)  # type: ignore[no-any-return, unused-ignore]

    def download(self) -> bool:
        """SWaT requires iTrust institutional registration."""
        logger.info(
            "SWaT dataset requires credentials. "
            "Register at: https://itrust.sutd.edu.sg/itrust-labs-datasets/"
        )
        raise DataSourceUnavailableError(
            loader_name="SWaT",
            source_url="https://itrust.sutd.edu.sg/itrust-labs-datasets/",
            reason=(
                "SWaT/WADI requires iTrust registration: "
                "https://itrust.sutd.edu.sg/itrust-labs-datasets/ "
                "Download data and place in " + str(self.data_path)
            ),
        )

    def load(
        self, split: DatasetSplit = DatasetSplit.ALL
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """
        Load SWaT dataset.

        Args:
            split: Dataset split to load

        Returns:
            Tuple of (features, labels)
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("pandas required for SWaT loading: pip install pandas") from e

        # Check for data files
        possible_files = [
            self.data_path / "SWaT_Dataset_Normal_v1.xlsx",
            self.data_path / "SWaT_Dataset_Attack_v0.xlsx",
            self.data_path / "Physical" / "SWaT.A1 & A2_Dec 2015" / "SWaT_Dataset_Normal_v0.csv",
        ]

        data_file = None
        for f in possible_files:
            if f.exists():
                data_file = f
                break

        if data_file is None:
            self.download()
            raise FileNotFoundError(
                f"SWaT data not found in {self.data_path}. Please download from iTrust Labs."
            )

        logger.info(f"Loading SWaT data from {data_file}")

        # Load based on file type
        if data_file.suffix == ".xlsx":
            df = pd.read_excel(data_file)
        else:
            df = pd.read_csv(data_file)

        # Extract features and labels
        feature_cols = [c for c in df.columns if c not in ["Timestamp", "Attack", "Normal/Attack"]]
        features = df[feature_cols].values.astype(np.float32)

        # Get labels
        if "Attack" in df.columns:
            labels = df["Attack"].values
        elif "Normal/Attack" in df.columns:
            labels = (df["Normal/Attack"] != "Normal").astype(int).values
        else:
            labels = np.zeros(len(features))

        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0)

        logger.info(f"Loaded SWaT: {features.shape[0]} samples, {features.shape[1]} features")
        logger.info(f"Anomaly ratio: {labels.mean():.2%}")

        return features, labels

    def get_metadata(self) -> DatasetMetadata:
        """Get dataset metadata."""
        num_samples = 946722  # Approximate
        return DatasetMetadata(
            name="SWaT",
            version="v1",
            num_samples=num_samples,
            num_features=self.NUM_FEATURES,
            feature_names=self.FEATURE_NAMES,
            target_names=["Normal", "Attack"],
            class_distribution={
                "normal": int(0.88 * num_samples),
                "anomaly": int(0.12 * num_samples),
            },
            source_url=self.DATASET_URL,
            license=self.LICENSE,
            citation=self.CITATION,
            preprocessing_applied=[],
        )

    def get_attack_descriptions(self) -> dict[int, str]:
        """Get descriptions of attack scenarios."""
        return {
            1: "MV-101 spoofed open when tank full",
            2: "P-102 spoofed on continuously",
            3: "LIT-101 spoofed low when high",
            # ... (36 total attack scenarios)
            36: "Combined multi-point attack",
        }


class WADILoader(DatasetLoader):
    """
    Water Distribution (WADI) Dataset Loader.

    WADI is a water distribution testbed at SUTD. Larger and more complex
    than SWaT, it simulates a complete water distribution network.

    Data Source: iTrust Labs (https://itrust.sutd.edu.sg/itrust-labs-datasets/)
    Paper: Ahmed et al., ICS-CSR 2017

    Features:
    - 123 sensors and actuators
    - 16 days of data (14 normal + 2 under attack)
    - 15 unique attack scenarios
    - ~1.2M samples

    Note: WADI is known to be more challenging than SWaT due to:
    - More complex system dynamics
    - Subtler attacks
    - Lower anomaly ratio
    """

    DATASET_NAME = "wadi"
    DATASET_URL = "https://itrust.sutd.edu.sg/itrust-labs-datasets/dataset_info/"
    LICENSE = "Research Only (iTrust Agreement)"
    CITATION = """Ahmed CM, Palleti VR, Mathur AP.
    WADI: A Water Distribution Testbed for Research in the Design of
    Secure Cyber Physical Systems. ICS-CSR 2017."""
    REQUIRES_CREDENTIALS = True

    NUM_FEATURES = 123
    ATTACK_COUNT = 15
    STAGES = ["Stage_1", "Stage_2"]  # WADI pipeline stages

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load raw WADI data - redirects to load()."""
        return self.load()

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Apply WADI-specific preprocessing (normalization)."""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-8
        return np.asarray((data - mean) / std)  # type: ignore[no-any-return, unused-ignore]

    def download(self) -> bool:
        """WADI requires iTrust institutional registration."""
        logger.info(
            "WADI dataset requires credentials. "
            "Register at: https://itrust.sutd.edu.sg/itrust-labs-datasets/"
        )
        raise DataSourceUnavailableError(
            loader_name="WADI",
            source_url="https://itrust.sutd.edu.sg/itrust-labs-datasets/",
            reason=(
                "SWaT/WADI requires iTrust registration: "
                "https://itrust.sutd.edu.sg/itrust-labs-datasets/ "
                "Download data and place in " + str(self.data_path)
            ),
        )

    def load(
        self, split: DatasetSplit = DatasetSplit.ALL
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load WADI dataset."""
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("pandas required for WADI loading: pip install pandas") from e

        possible_files = [
            self.data_path / "WADI_14days_new.csv",
            self.data_path / "WADI_attackdataLABLE.csv",
            self.data_path / "WADI.A1_9 Oct 2017" / "WADI_attackdata.csv",
        ]

        data_file = None
        for f in possible_files:
            if f.exists():
                data_file = f
                break

        if data_file is None:
            self.download()
            raise FileNotFoundError(
                f"WADI data not found in {self.data_path}. Please download from iTrust Labs."
            )

        logger.info(f"Loading WADI data from {data_file}")

        df = pd.read_csv(data_file)

        # Extract features
        feature_cols = [
            c
            for c in df.columns
            if c not in ["Row", "Date", "Time", "Attack LABLE (1:No Attack, -1:Attack)"]
        ]
        features = df[feature_cols].values.astype(np.float32)

        # Get labels
        if "Attack LABLE (1:No Attack, -1:Attack)" in df.columns:
            labels = (df["Attack LABLE (1:No Attack, -1:Attack)"] == -1).astype(int).values
        else:
            labels = np.zeros(len(features))

        features = np.nan_to_num(features, nan=0.0)

        logger.info(f"Loaded WADI: {features.shape[0]} samples, {features.shape[1]} features")
        logger.info(f"Anomaly ratio: {labels.mean():.2%}")

        return features, labels

    def get_metadata(self) -> DatasetMetadata:
        """Get dataset metadata."""
        num_samples = 1209601  # Approximate
        return DatasetMetadata(
            name="WADI",
            version="A1",
            num_samples=num_samples,
            num_features=self.NUM_FEATURES,
            feature_names=[f"feature_{i}" for i in range(self.NUM_FEATURES)],
            target_names=["Normal", "Attack"],
            class_distribution={
                "normal": int(0.94 * num_samples),
                "anomaly": int(0.06 * num_samples),
            },
            source_url=self.DATASET_URL,
            license=self.LICENSE,
            citation=self.CITATION,
            preprocessing_applied=[],
        )


class BATADALLoader(DatasetLoader):
    """
    BATADAL (Battle of Attack Detection Algorithms) Dataset Loader.

    Dataset from the water network attack detection competition.
    Contains both training (with labels) and test data.

    Data Source: https://www.batadal.net/
    Paper: Taormina et al., "Battle of the Attack Detection Algorithms",
           Journal of Water Resources Planning and Management, 2018
    """

    DATASET_NAME = "batadal"
    DATASET_URL = "https://www.batadal.net/data.html"
    LICENSE = "CC BY 4.0"
    CITATION = """Taormina R, et al. Battle of the Attack Detection Algorithms:
    Disclosing Cyber Attacks on Water Distribution Networks.
    Journal of Water Resources Planning and Management, 2018."""
    REQUIRES_CREDENTIALS = False

    NUM_FEATURES = 43

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load raw BATADAL data - redirects to load()."""
        return self.load()

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Apply BATADAL-specific preprocessing (normalization)."""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-8
        return np.asarray((data - mean) / std)  # type: ignore[no-any-return, unused-ignore]

    def download(self) -> bool:
        """Download BATADAL dataset from official source."""
        import urllib.error

        logger.info("Downloading BATADAL dataset...")

        urls = {
            "train": "https://www.batadal.net/data/BATADAL_dataset03.csv",
            "test": "https://www.batadal.net/data/BATADAL_dataset04.csv",
        }

        self.data_path.mkdir(parents=True, exist_ok=True)

        for name, url in urls.items():
            output_path = self.data_path / f"BATADAL_{name}.csv"
            try:
                logger.info(f"  Downloading {name}...")
                safe_urlretrieve(url, output_path)
            except (urllib.error.URLError, ValueError) as e:
                logger.error(f"  Failed to download {name}: {e}")
                raise DataSourceUnavailableError(
                    loader_name="BATADAL",
                    source_url=url,
                    reason=f"Failed to download {name}: {e}",
                ) from e

        logger.info("BATADAL download complete")
        return True

    def load(
        self, split: DatasetSplit = DatasetSplit.ALL
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load BATADAL dataset.

        Loads both the training CSV (no attacks) and the test CSV (with
        ATT_FLAG column containing attack labels).  For anomaly detection
        benchmarks the test set is the one that matters — it contains the
        labeled attack periods.

        Returns:
            Tuple of (features, labels) where labels are binary (1 = attack).
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("pandas required for BATADAL loading") from e

        test_file = self.data_path / "BATADAL_test.csv"
        train_file = self.data_path / "BATADAL_train.csv"

        if not test_file.exists() and not train_file.exists():
            self.download()

        dfs = []
        for fpath in [train_file, test_file]:
            if fpath.exists():
                dfs.append(pd.read_csv(fpath))

        if not dfs:
            raise FileNotFoundError(f"BATADAL data not found in {self.data_path}")

        df = pd.concat(dfs, ignore_index=True)

        # Strip whitespace from column names (BATADAL CSVs have trailing spaces)
        df.columns = df.columns.str.strip()

        feature_cols = [c for c in df.columns if c not in ["DATETIME", "ATT_FLAG"]]
        features = df[feature_cols].values.astype(np.float32)

        if "ATT_FLAG" in df.columns:
            labels = df["ATT_FLAG"].values.astype(int)
        else:
            labels = np.zeros(len(features), dtype=int)

        features = np.nan_to_num(features, nan=0.0)

        logger.info(
            f"Loaded BATADAL: {features.shape[0]} samples, {features.shape[1]} features, "
            f"anomalies: {labels.sum()}"
        )

        return features, labels

    def get_metadata(self) -> DatasetMetadata:
        """Get dataset metadata."""
        num_samples = 8761
        return DatasetMetadata(
            name="BATADAL",
            version="2018",
            num_samples=num_samples,
            num_features=self.NUM_FEATURES,
            feature_names=[f"feature_{i}" for i in range(self.NUM_FEATURES)],
            target_names=["Normal", "Attack"],
            class_distribution={
                "normal": int(0.93 * num_samples),
                "anomaly": int(0.07 * num_samples),
            },
            source_url=self.DATASET_URL,
            license=self.LICENSE,
            citation=self.CITATION,
            preprocessing_applied=[],
        )
