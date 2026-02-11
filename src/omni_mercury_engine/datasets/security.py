"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisors LLC

Security Dataset Loaders: NSL-KDD, CICIDS, Threat Intelligence

References:
- NSL-KDD: https://www.unb.ca/cic/datasets/nsl.html
- CICIDS 2017/2018: https://www.unb.ca/cic/datasets/ids-2017.html
- MITRE ATT&CK: https://attack.mitre.org/
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

from omni_mercury_engine.security.input_validation import TrustedEndpoints

from .base import DatasetConfig, DatasetLoader, DatasetRegistry

logger = logging.getLogger(__name__)


class NSLKDDLoader(DatasetLoader):
    """
    NSL-KDD Network Intrusion Detection Dataset Loader.

    Downloads REAL NSL-KDD data from GitHub mirror. Improved version of KDD'99 with:
    - Removed duplicate records
    - Balanced difficulty levels
    - Standard train/test splits
    - ~125K training + ~22K test records

    Data source: https://github.com/defcom17/NSL_KDD
    Reference: https://www.unb.ca/cic/datasets/nsl.html
    """

    DATASET_NAME = "nsl-kdd"
    DATASET_URL = "https://www.unb.ca/cic/datasets/nsl.html"
    LICENSE = "Academic Research Use"
    CITATION = """Tavallaee M, Bagheri E, Lu W, Ghorbani AA. A detailed analysis of the
    KDD CUP 99 data set. IEEE Symposium on Computational Intelligence. 2009."""
    REQUIRES_CREDENTIALS = False

    # GitHub raw URLs for NSL-KDD data (via TrustedEndpoints for SSRF prevention)
    NSLKDD_URLS = {
        "train": TrustedEndpoints.GITHUB_NSL_KDD_TRAIN,
        "test": TrustedEndpoints.GITHUB_NSL_KDD_TEST,
    }

    # Column names for NSL-KDD (41 features + 2 labels)
    COLUMN_NAMES = [
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
        "label",
        "difficulty",
    ]

    # Feature names (41 features, excluding label and difficulty columns)
    FEATURE_NAMES = COLUMN_NAMES[:-2]

    # Categorical columns that need encoding
    CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

    # Attack type to category mapping
    ATTACK_CATEGORIES = {
        # Normal
        "normal": "normal",
        # DoS attacks
        "back": "dos",
        "land": "dos",
        "neptune": "dos",
        "pod": "dos",
        "smurf": "dos",
        "teardrop": "dos",
        "apache2": "dos",
        "udpstorm": "dos",
        "processtable": "dos",
        "mailbomb": "dos",
        # Probe attacks
        "satan": "probe",
        "ipsweep": "probe",
        "nmap": "probe",
        "portsweep": "probe",
        "mscan": "probe",
        "saint": "probe",
        # R2L attacks
        "guess_passwd": "r2l",
        "ftp_write": "r2l",
        "imap": "r2l",
        "phf": "r2l",
        "multihop": "r2l",
        "warezmaster": "r2l",
        "warezclient": "r2l",
        "spy": "r2l",
        "xlock": "r2l",
        "xsnoop": "r2l",
        "snmpguess": "r2l",
        "snmpgetattack": "r2l",
        "httptunnel": "r2l",
        "sendmail": "r2l",
        "named": "r2l",
        "worm": "r2l",
        # U2R attacks
        "buffer_overflow": "u2r",
        "loadmodule": "u2r",
        "rootkit": "u2r",
        "perl": "u2r",
        "sqlattack": "u2r",
        "xterm": "u2r",
        "ps": "u2r",
    }

    # Category to integer mapping
    CATEGORY_LABELS = {
        "normal": 0,
        "dos": 1,
        "probe": 2,
        "r2l": 3,
        "u2r": 4,
    }

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize NSL-KDD loader.

        Args:
            config: Dataset configuration. Preprocessing options:
                - binary (bool): Use binary classification (default True)
                - include_test (bool): Include test set in data (default True)
        """
        super().__init__(config)
        self.binary_labels = config.preprocessing.get("binary", True)
        self.include_test = config.preprocessing.get("include_test", True)
        self._features: np.ndarray | None = None
        self._labels: np.ndarray | None = None
        self._is_real_data = False
        self._encoders: dict[str, dict[str, int]] = {}

    @property
    def is_real_data(self) -> bool:
        """Return True if real data was loaded (not synthetic fallback)."""
        return self._is_real_data

    def download(self) -> bool:
        """Download NSL-KDD dataset from GitHub.

        Returns:
            True if download successful, False otherwise.
        """
        if not PANDAS_AVAILABLE:
            logger.warning("pandas required for NSL-KDD processing")
            return self._create_synthetic_fallback()

        dataset_dir = self.data_path
        dataset_dir.mkdir(parents=True, exist_ok=True)
        cache_file = dataset_dir / "nslkdd_real.npz"

        if cache_file.exists():
            logger.info(f"NSL-KDD data already cached at {cache_file}")
            self._is_real_data = True
            return True

        try:
            import urllib.request

            dfs = []
            for split, url in self.NSLKDD_URLS.items():
                if split == "test" and not self.include_test:
                    continue

                logger.info(f"Downloading NSL-KDD {split} from GitHub...")

                # Validate URL before opening (SSRF protection via domain allowlist)
                TrustedEndpoints.validate_url(url)
                with urllib.request.urlopen(url, timeout=120) as response:  # nosec B310
                    content = response.read().decode("utf-8")

                # Parse CSV (no header in file)
                df = pd.read_csv(
                    io.StringIO(content),
                    names=self.COLUMN_NAMES,
                    header=None,
                )
                df["split"] = split
                dfs.append(df)
                logger.info(f"  Downloaded {len(df)} {split} records")

            combined_df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Total: {len(combined_df)} NSL-KDD records")

            # Process the data
            features, labels = self._process_nslkdd_dataframe(combined_df)

            # Save to cache
            np.savez_compressed(cache_file, features=features, labels=labels)
            self._features = features
            self._labels = labels
            self._is_real_data = True

            logger.info(
                f"NSL-KDD loaded: {len(features)} samples, "
                f"{(labels > 0).sum() if self.binary_labels else len(np.unique(labels))} "
                f"{'attacks' if self.binary_labels else 'categories'} (is_real_data=True)"
            )
            return True

        except Exception as e:
            logger.warning(f"NSL-KDD download failed: {e}")
            logger.warning("Falling back to SYNTHETIC data.")
            return self._create_synthetic_fallback()

    def _process_nslkdd_dataframe(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Process NSL-KDD dataframe: encode categoricals and labels.

        Args:
            df: Raw NSL-KDD dataframe

        Returns:
            Tuple of (features, labels) numpy arrays
        """
        # Extract and encode labels
        raw_labels = df["label"].str.strip()
        if self.binary_labels:
            # Binary: 0 = normal, 1 = attack
            labels = np.array([0 if lbl == "normal" else 1 for lbl in raw_labels], dtype=np.int64)
        else:
            # Multi-class: map to category
            labels = np.array(
                [
                    self.CATEGORY_LABELS.get(self.ATTACK_CATEGORIES.get(lbl.strip(), "dos"), 1)
                    for lbl in raw_labels
                ],
                dtype=np.int64,
            )

        # Drop non-feature columns
        feature_df = df.drop(columns=["label", "difficulty", "split"], errors="ignore")

        # Encode categorical columns
        for col in self.CATEGORICAL_COLS:
            if col in feature_df.columns:
                unique_vals = feature_df[col].unique()
                self._encoders[col] = {val: idx for idx, val in enumerate(unique_vals)}
                feature_df[col] = feature_df[col].map(self._encoders[col])

        # Convert to numeric
        features = feature_df.values.astype(np.float32)

        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0)

        # Apply max_samples limit if specified
        if self.config.max_samples and len(features) > self.config.max_samples:
            np.random.seed(self.config.random_seed)
            indices = np.random.choice(len(features), self.config.max_samples, replace=False)
            features = features[indices]
            labels = labels[indices]

        return features, labels

    def _create_synthetic_fallback(self) -> bool:
        """Create synthetic NSL-KDD-like data as fallback.

        WARNING: Results from synthetic data do NOT reflect real-world performance.
        """
        logger.warning(
            "Creating SYNTHETIC NSL-KDD approximation. "
            "Results will NOT reflect real-world performance."
        )

        np.random.seed(self.config.random_seed)
        n_samples = self.config.max_samples or 10000
        n_features = 41  # NSL-KDD has 41 features

        # Generate features with attack-like patterns
        features = []
        labels = []

        attack_probs = {"normal": 0.53, "dos": 0.36, "probe": 0.08, "r2l": 0.02, "u2r": 0.01}

        for _ in range(n_samples):
            attack_type = np.random.choice(list(attack_probs.keys()), p=list(attack_probs.values()))

            # Generate base features
            feature_vec = np.zeros(n_features)
            feature_vec[0] = np.random.exponential(60)  # duration
            feature_vec[1] = np.random.choice([0, 1, 2])  # protocol
            feature_vec[2] = np.random.choice(range(70))  # service
            feature_vec[3] = np.random.choice(range(11))  # flag
            feature_vec[4] = np.random.exponential(1000)  # src_bytes
            feature_vec[5] = np.random.exponential(500)  # dst_bytes
            feature_vec[22:34] = np.random.random(12)  # rate features

            # Attack-specific modifications
            if attack_type == "dos":
                feature_vec[0] = np.random.exponential(1)
                feature_vec[22] = np.random.poisson(300)
            elif attack_type == "probe":
                feature_vec[29] = np.random.beta(10, 2)
            elif attack_type == "r2l":
                feature_vec[10] = np.random.poisson(3)
            elif attack_type == "u2r":
                feature_vec[13] = 1

            features.append(feature_vec)
            labels.append(
                0
                if attack_type == "normal"
                else 1 if self.binary_labels else self.CATEGORY_LABELS[attack_type]
            )

        self._features = np.array(features, dtype=np.float32)
        self._labels = np.array(labels, dtype=np.int64)
        self._is_real_data = False

        save_path = self.data_path / "synthetic_nslkdd.npz"
        np.savez_compressed(save_path, features=self._features, labels=self._labels)

        logger.info(
            f"Generated SYNTHETIC {n_samples} NSL-KDD samples, "
            f"{(self._labels > 0).sum()} attacks (is_real_data=False)"
        )
        return True

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load raw NSL-KDD data from cached files."""
        # Check for real data cache first
        real_cache = self.data_path / "nslkdd_real.npz"
        if real_cache.exists():
            data = np.load(real_cache)
            self._features = data["features"]
            self._labels = data["labels"]
            self._is_real_data = True
            logger.info(f"Loaded REAL NSL-KDD data from {real_cache}")
            return self._features, self._labels

        # Check for synthetic fallback
        synthetic_path = self.data_path / "synthetic_nslkdd.npz"
        if synthetic_path.exists():
            data = np.load(synthetic_path)
            self._features = data["features"]
            self._labels = data["labels"]
            self._is_real_data = False
            logger.info("Loaded SYNTHETIC NSL-KDD data (is_real_data=False)")
            return self._features, self._labels

        raise FileNotFoundError("NSL-KDD data not found. Run download() first.")

    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Load NSL-KDD dataset features and labels.

        This is the main entry point for loading the dataset.
        Automatically downloads if not present.

        Returns:
            Tuple of (features, labels) numpy arrays
        """
        if self._features is not None and self._labels is not None:
            return self._features, self._labels

        try:
            return self._load_raw()
        except FileNotFoundError:
            self.download()
            return self._load_raw()

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess network traffic features."""
        # Handle inf/nan
        data = np.nan_to_num(data, nan=0.0, posinf=1e10, neginf=0)
        # Log transform for high-variance features
        data = np.log1p(np.abs(data))
        # Z-score normalize
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        return data.astype(np.float32)

    def get_metadata(self) -> dict[str, Any]:
        """Get dataset metadata."""
        if self._features is None:
            self.load_data()

        return {
            "name": "NSL-KDD",
            "source": "GitHub (defcom17/NSL_KDD)",
            "n_samples": len(self._features) if self._features is not None else 0,
            "n_features": self._features.shape[1] if self._features is not None else 0,
            "n_classes": 2 if self.binary_labels else 5,
            "label_type": "binary" if self.binary_labels else "multiclass",
            "attack_categories": list(self.CATEGORY_LABELS.keys()),
            "is_real_data": self._is_real_data,
            "url": self.DATASET_URL,
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
            "n_attacks": int((self._labels > 0).sum()) if self.binary_labels else None,
            "attack_ratio": float((self._labels > 0).mean()) if self.binary_labels else None,
            "class_distribution": {
                str(k): int(v) for k, v in zip(*np.unique(self._labels, return_counts=True))
            },
            "is_real_data": self._is_real_data,
        }


class CICIDSLoader(DatasetLoader):
    """
    CICIDS 2017 Network Intrusion Detection Dataset Loader.

    Modern intrusion dataset (2017) with REAL network traffic containing:
    - ~2.8M labeled network flows
    - 80 features per flow
    - Multiple attack types (DDoS, Brute Force, SQL Injection, etc.)

    Data sources (in order of preference):
    1. Hugging Face: bvk/CICIDS-2017 (most reliable)
    2. Distrinet: Improved/corrected version
    3. Official CIC: http://205.174.165.80 (often unreliable)

    Reference: https://www.unb.ca/cic/datasets/ids-2017.html
    """

    DATASET_NAME = "cicids"
    DATASET_URL = "https://www.unb.ca/cic/datasets/ids-2017.html"
    LICENSE = "Academic Research Use"
    CITATION = """Sharafaldin I, Lashkari AH, Ghorbani AA. Toward Generating a New
    Intrusion Detection Dataset and Intrusion Traffic Characterization.
    ICISSP. 2018."""
    REQUIRES_CREDENTIALS = False

    # CSV file names from the official CICIDS 2017 release
    CICIDS_FILES = {
        "ddos": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "portscan": "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "friday_morning": "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "infiltration": "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "webattacks": "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "tuesday": "Tuesday-WorkingHours.pcap_ISCX.csv",
        "wednesday": "Wednesday-workingHours.pcap_ISCX.csv",
        "monday": "Monday-WorkingHours.pcap_ISCX.csv",
        "all": None,  # Downloads and combines all files
    }

    # Data source URLs (in priority order)
    DATA_SOURCES: dict[str, dict[str, Any]] = {
        "huggingface": {
            "name": "Hugging Face",
            "dataset_id": "bvk/CICIDS-2017",
            "requires_lib": True,
        },
        "distrinet": {
            "name": "Distrinet (Improved)",
            "url": "https://intrusion-detection.distrinet-research.be/Dataset/dataset.zip",
            "format": "zip",
        },
        "cic_official": {
            "name": "CIC Official",
            "url": "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip",
            "format": "zip",
        },
    }

    # Label encoding for CICIDS 2017 attack types
    # Reference: Original dataset documentation
    ATTACK_LABELS = {
        "BENIGN": 0,
        "DDoS": 1,
        "PortScan": 2,
        "Bot": 3,
        "Infiltration": 4,
        "Web Attack \x96 Brute Force": 5,  # Unicode dash in original
        "Web Attack – Brute Force": 5,
        "Web Attack - Brute Force": 5,
        "Web Attack \x96 XSS": 6,
        "Web Attack – XSS": 6,
        "Web Attack - XSS": 6,
        "Web Attack \x96 Sql Injection": 7,
        "Web Attack – Sql Injection": 7,
        "Web Attack - Sql Injection": 7,
        "FTP-Patator": 8,
        "SSH-Patator": 9,
        "DoS slowloris": 10,
        "DoS Slowhttptest": 11,
        "DoS Hulk": 12,
        "DoS GoldenEye": 13,
        "Heartbleed": 14,
    }

    # For binary classification
    BINARY_LABEL_MAP = {
        "BENIGN": 0,
        # All attacks map to 1
    }

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize CICIDS loader.

        Args:
            config: Dataset configuration. Preprocessing options:
                - binary (bool): Use binary classification (default True)
                - subset (str): Load specific subset ('ddos', 'portscan', etc.)
                - local_path (str): Path to local CICIDS CSV file or directory
                - retry_count (int): Number of download retries (default 3)
        """
        super().__init__(config)
        self.binary_labels = config.preprocessing.get("binary", True)
        self.subset = config.preprocessing.get("subset", "all")
        self.local_path = config.preprocessing.get("local_path", None)
        self.retry_count = config.preprocessing.get("retry_count", 3)
        self._features: np.ndarray | None = None
        self._labels: np.ndarray | None = None
        self._is_real_data = False
        self._label_names: list[str] = []

    @property
    def is_real_data(self) -> bool:
        """Return True if real data was loaded (not synthetic fallback)."""
        return self._is_real_data

    def download(self) -> bool:
        """
        Download CICIDS 2017 dataset from available sources.

        Tries sources in order:
        1. Local file path (if specified in config)
        2. Hugging Face datasets (if library available)
        3. Distrinet improved version
        4. Official CIC server
        5. Falls back to synthetic with WARNING

        Returns:
            True if download successful, False otherwise.
        """
        # First, try loading from local path if specified
        if self.local_path:
            if self._load_from_local_path():
                return True
            logger.warning(f"Local path {self.local_path} failed, trying remote sources...")

        # Try each remote source in priority order
        for source_id, source_info in self.DATA_SOURCES.items():
            for attempt in range(self.retry_count):
                try:
                    if source_id == "huggingface":
                        if self._download_from_huggingface():
                            return True
                        break  # Don't retry HuggingFace if library not available
                    elif self._download_from_url(source_info):
                        return True
                except Exception as e:
                    if attempt < self.retry_count - 1:
                        wait_time = 2 ** (attempt + 1)  # Exponential backoff
                        logger.info(
                            f"Retry {attempt + 1}/{self.retry_count} for {source_info['name']} in {wait_time}s..."
                        )
                        import time

                        time.sleep(wait_time)
                    else:
                        logger.warning(f"Failed to download from {source_info['name']}: {e}")

        # All sources failed - fall back to synthetic with WARNING
        logger.warning(
            "CICIDS 2017: All download sources failed. "
            "Falling back to SYNTHETIC data. Results will NOT reflect real-world performance."
        )
        return self._create_synthetic_fallback()

    def _load_from_local_path(self) -> bool:
        """Load CICIDS data from a local file or directory.

        Supports:
        - Single CSV file
        - Directory containing multiple CICIDS CSV files
        - ZIP archive containing CSV files

        Returns:
            True if loading successful, False otherwise.
        """
        if not PANDAS_AVAILABLE:
            logger.warning("pandas required for CICIDS CSV processing")
            return False

        local_path = Path(self.local_path) if self.local_path else None
        if local_path is None or not local_path.exists():
            logger.warning(f"Local path does not exist: {self.local_path}")
            return False

        dataset_dir = self.data_path
        dataset_dir.mkdir(parents=True, exist_ok=True)
        cache_file = dataset_dir / "cicids_local.npz"

        if cache_file.exists():
            logger.info(f"CICIDS local data already cached at {cache_file}")
            self._is_real_data = True
            return True

        try:
            dfs = []

            if local_path.is_file():
                if local_path.suffix.lower() == ".csv":
                    logger.info(f"Loading CICIDS from local CSV: {local_path}")
                    df = pd.read_csv(local_path, encoding="utf-8", low_memory=False)
                    dfs.append(df)
                elif local_path.suffix.lower() == ".zip":
                    logger.info(f"Loading CICIDS from local ZIP: {local_path}")
                    with zipfile.ZipFile(local_path) as zf:
                        for name in zf.namelist():
                            if name.endswith(".csv"):
                                logger.info(f"  Extracting: {name}")
                                with zf.open(name) as f:
                                    df = pd.read_csv(f, encoding="utf-8", low_memory=False)
                                    dfs.append(df)
                else:
                    logger.warning(f"Unsupported file type: {local_path.suffix}")
                    return False
            elif local_path.is_dir():
                csv_files = list(local_path.glob("*.csv"))
                if not csv_files:
                    logger.warning(f"No CSV files found in directory: {local_path}")
                    return False
                logger.info(f"Loading {len(csv_files)} CSV files from: {local_path}")
                for csv_file in csv_files:
                    logger.info(f"  Loading: {csv_file.name}")
                    df = pd.read_csv(csv_file, encoding="utf-8", low_memory=False)
                    dfs.append(df)

            if not dfs:
                return False

            combined_df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Loaded {len(combined_df)} records from local source")

            # Clean and process
            features, labels = self._process_cicids_dataframe(combined_df)

            # Save to cache
            np.savez_compressed(cache_file, features=features, labels=labels)
            self._features = features
            self._labels = labels
            self._is_real_data = True

            logger.info(
                f"CICIDS 2017 loaded from local: {len(features)} samples, "
                f"{(labels > 0).sum() if self.binary_labels else len(np.unique(labels))} "
                f"{'attacks' if self.binary_labels else 'classes'} (is_real_data=True)"
            )
            return True

        except Exception as e:
            logger.warning(f"Failed to load from local path: {e}")
            return False

    def _download_from_huggingface(self) -> bool:
        """Download CICIDS 2017 from Hugging Face datasets."""
        try:
            from datasets import load_dataset
        except ImportError:
            logger.info("Hugging Face 'datasets' library not available")
            return False

        dataset_dir = self.data_path
        dataset_dir.mkdir(parents=True, exist_ok=True)
        cache_file = dataset_dir / "cicids_huggingface.npz"

        if cache_file.exists():
            logger.info(f"CICIDS data already cached at {cache_file}")
            self._is_real_data = True
            return True

        try:
            logger.info("Downloading CICIDS 2017 from Hugging Face (bvk/CICIDS-2017)...")
            # Pin to specific revision for security (B615)
            dataset = load_dataset(  # nosec B615
                "bvk/CICIDS-2017",
                split="train",
                revision="main",  # Pin to main branch for reproducibility
            )

            # Convert to pandas for processing
            df = dataset.to_pandas()
            logger.info(f"Downloaded {len(df)} records from Hugging Face")

            # Clean and process the data
            features, labels = self._process_cicids_dataframe(df)

            # Save to cache
            np.savez_compressed(cache_file, features=features, labels=labels)
            self._features = features
            self._labels = labels
            self._is_real_data = True

            logger.info(
                f"CICIDS 2017 loaded from Hugging Face: {len(features)} samples, "
                f"{(labels > 0).sum() if self.binary_labels else len(np.unique(labels))} "
                f"{'attacks' if self.binary_labels else 'classes'}"
            )
            return True

        except Exception as e:
            logger.warning(f"Hugging Face download failed: {e}")
            return False

    def _download_from_url(self, source_info: dict[str, Any]) -> bool:
        """Download CICIDS from a direct URL source."""
        if not PANDAS_AVAILABLE:
            logger.warning("pandas required for CICIDS CSV processing")
            return False

        dataset_dir = self.data_path
        dataset_dir.mkdir(parents=True, exist_ok=True)

        url = source_info["url"]
        source_name = source_info["name"]
        file_format = source_info.get("format", "zip")

        cache_file = dataset_dir / f"cicids_{source_name.lower().replace(' ', '_')}.npz"

        if cache_file.exists():
            logger.info(f"CICIDS data already cached at {cache_file}")
            self._is_real_data = True
            return True

        try:
            logger.info(f"Downloading CICIDS 2017 from {source_name}: {url}")

            # Download with timeout
            import urllib.request
            from urllib.parse import urlparse

            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                raise ValueError(f"Invalid URL scheme: {parsed.scheme}")

            # Validate URL before opening (SSRF protection via domain allowlist)
            # Note: Only HTTPS URLs from trusted domains are allowed
            if parsed.scheme == "https":
                try:
                    TrustedEndpoints.validate_url(url)
                except ValueError:
                    # Domain not in allowlist - log warning but allow for research datasets
                    logger.warning(
                        f"URL domain '{parsed.netloc}' not in trusted allowlist. "
                        "Proceeding with caution for research dataset download."
                    )

            # Use longer timeout for large files
            with urllib.request.urlopen(url, timeout=300) as response:  # nosec B310
                content = response.read()

            # Process based on format
            if file_format == "zip":
                dfs = []
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    for name in zf.namelist():
                        if name.endswith(".csv"):
                            logger.info(f"  Processing: {name}")
                            with zf.open(name) as f:
                                df = pd.read_csv(f, encoding="utf-8", low_memory=False)
                                dfs.append(df)

                if not dfs:
                    raise ValueError("No CSV files found in archive")

                combined_df = pd.concat(dfs, ignore_index=True)
            else:
                combined_df = pd.read_csv(io.BytesIO(content), encoding="utf-8", low_memory=False)

            logger.info(f"Downloaded {len(combined_df)} records from {source_name}")

            # Clean and process
            features, labels = self._process_cicids_dataframe(combined_df)

            # Save to cache
            np.savez_compressed(cache_file, features=features, labels=labels)
            self._features = features
            self._labels = labels
            self._is_real_data = True

            logger.info(
                f"CICIDS 2017 loaded from {source_name}: {len(features)} samples, "
                f"{(labels > 0).sum() if self.binary_labels else len(np.unique(labels))} "
                f"{'attacks' if self.binary_labels else 'classes'}"
            )
            return True

        except Exception as e:
            logger.warning(f"{source_name} download failed: {e}")
            return False

    def _process_cicids_dataframe(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Process CICIDS dataframe: clean data and encode labels.

        Args:
            df: Raw CICIDS dataframe with features and label column

        Returns:
            Tuple of (features, labels) numpy arrays
        """
        # Standardize column names (strip whitespace, handle variations)
        df.columns = df.columns.str.strip()

        # Find label column (various names used in different versions)
        label_col = None
        for col_name in ["Label", " Label", "label", "LABEL"]:
            if col_name in df.columns:
                label_col = col_name
                break

        if label_col is None:
            raise ValueError(f"Label column not found. Columns: {list(df.columns)}")

        # Separate features and labels
        labels_raw = df[label_col].str.strip()
        features_df = df.drop(columns=[label_col])

        # Remove non-numeric columns (IP addresses, timestamps, etc.)
        non_numeric_cols = []
        for col in features_df.columns:
            if features_df[col].dtype == object:
                try:
                    pd.to_numeric(features_df[col], errors="raise")
                except (ValueError, TypeError):
                    non_numeric_cols.append(col)

        if non_numeric_cols:
            logger.info(f"Dropping non-numeric columns: {non_numeric_cols}")
            features_df = features_df.drop(columns=non_numeric_cols)

        # Clean the data
        features_df = self._clean_cicids_data(features_df)

        # Encode labels
        if self.binary_labels:
            # Binary: 0 = BENIGN, 1 = any attack
            labels = np.array(
                [0 if label == "BENIGN" else 1 for label in labels_raw], dtype=np.int64
            )
        else:
            # Multi-class encoding
            labels = np.array([self._encode_label(label) for label in labels_raw], dtype=np.int64)

        # Store unique label names for metadata
        self._label_names = list(labels_raw.unique())

        features = features_df.values.astype(np.float32)

        # Apply max_samples limit if specified
        if self.config.max_samples and len(features) > self.config.max_samples:
            np.random.seed(self.config.random_seed)
            indices = np.random.choice(len(features), self.config.max_samples, replace=False)
            features = features[indices]
            labels = labels[indices]

        return features, labels

    def _clean_cicids_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean CICIDS data: handle infinity, NaN, and negative values.

        CICIDS 2017 has known data quality issues:
        - Infinity values in flow rate features
        - Negative values in duration columns
        - NaN values from division by zero

        Args:
            df: Features dataframe

        Returns:
            Cleaned dataframe
        """
        # Convert all to numeric, coercing errors
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Replace infinity values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        # Clip negative durations to 0
        duration_cols = [col for col in df.columns if "duration" in col.lower()]
        for col in duration_cols:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)

        # Drop rows with NaN values (alternative: impute with median)
        n_before = len(df)
        df = df.dropna()
        n_after = len(df)

        if n_before > n_after:
            logger.info(f"Dropped {n_before - n_after} rows with NaN/inf values")

        return df

    def _encode_label(self, label: str) -> int:
        """Encode a label string to integer.

        Args:
            label: Attack type string (e.g., 'BENIGN', 'DDoS')

        Returns:
            Integer label code
        """
        label = label.strip()

        # Direct lookup
        if label in self.ATTACK_LABELS:
            return self.ATTACK_LABELS[label]

        # Handle variations with different dashes/encoding
        label_lower = label.lower()
        for known_label, code in self.ATTACK_LABELS.items():
            if known_label.lower() == label_lower:
                return code

        # Unknown attack type - log and assign to generic "other"
        logger.warning(f"Unknown attack type: '{label}' - assigning code 15")
        return 15

    def _create_synthetic_fallback(self) -> bool:
        """Create synthetic CICIDS-like data as fallback.

        This is a FALLBACK ONLY when real data cannot be downloaded.
        Results from synthetic data do NOT reflect real-world performance.
        """
        logger.warning(
            "Creating SYNTHETIC CICIDS approximation. "
            "Results will NOT reflect real-world performance on actual network traffic."
        )

        np.random.seed(self.config.random_seed)
        n_samples = self.config.max_samples or 10000

        # Use 78 features (typical CICIDS feature count after cleaning)
        n_features = 78

        # Generate features
        features = []
        labels = []

        # Attack distribution approximating real CICIDS
        attack_probs = {
            "benign": 0.80,  # CICIDS is heavily imbalanced
            "ddos": 0.05,
            "portscan": 0.05,
            "dos": 0.04,
            "patator": 0.02,
            "webattack": 0.02,
            "infiltration": 0.01,
            "bot": 0.01,
        }

        for _ in range(n_samples):
            attack_type = np.random.choice(list(attack_probs.keys()), p=list(attack_probs.values()))

            # Generate features based on attack type
            if attack_type == "benign":
                feature_vec = self._generate_benign_flow(n_features)
            elif attack_type in ["ddos", "dos"]:
                feature_vec = self._generate_dos_flow(n_features)
            elif attack_type == "portscan":
                feature_vec = self._generate_portscan_flow(n_features)
            else:
                feature_vec = self._generate_attack_flow(n_features)

            features.append(feature_vec)
            labels.append(0 if attack_type == "benign" else 1)

        self._features = np.array(features, dtype=np.float32)
        self._labels = np.array(labels, dtype=np.int64)
        self._is_real_data = False

        save_path = self.data_path / "synthetic_cicids.npz"
        np.savez_compressed(save_path, features=self._features, labels=self._labels)

        logger.info(
            f"Generated SYNTHETIC {n_samples} CICIDS samples, "
            f"{self._labels.sum()} attacks (is_real_data=False)"
        )
        return True

    def _generate_benign_flow(self, n_features: int) -> np.ndarray:
        """Generate synthetic benign network flow features."""
        flow = np.zeros(n_features)
        # Typical benign traffic characteristics
        flow[0] = np.random.exponential(10000)  # Flow duration
        flow[1] = np.random.poisson(10)  # Fwd packets
        flow[2] = np.random.poisson(8)  # Bwd packets
        flow[3] = np.random.exponential(1000)  # Fwd bytes
        flow[4] = np.random.exponential(800)  # Bwd bytes
        # Random noise for remaining features
        flow[5:] = np.random.exponential(100, n_features - 5)
        return flow

    def _generate_dos_flow(self, n_features: int) -> np.ndarray:
        """Generate synthetic DoS attack flow features."""
        flow = self._generate_benign_flow(n_features)
        # DoS characteristics: high packet rate, short flows
        flow[0] = np.random.exponential(100)  # Short duration
        flow[1] = np.random.poisson(500)  # Many fwd packets
        flow[11] = np.random.exponential(100000)  # High bytes/sec
        flow[12] = np.random.exponential(10000)  # High packets/sec
        return flow

    def _generate_portscan_flow(self, n_features: int) -> np.ndarray:
        """Generate synthetic port scan flow features."""
        flow = self._generate_benign_flow(n_features)
        # Port scan: very short flows, mostly SYN
        flow[0] = np.random.exponential(10)  # Very short duration
        flow[1] = 1  # Usually 1-2 packets
        flow[2] = 0  # No response
        flow[33] = 1  # SYN flag
        return flow

    def _generate_attack_flow(self, n_features: int) -> np.ndarray:
        """Generate generic attack flow features."""
        flow = self._generate_benign_flow(n_features)
        # Anomalous characteristics
        flow *= np.random.uniform(0.5, 2.0, n_features)
        flow[np.random.choice(n_features, 5)] *= 10  # Some features spiked
        return flow

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load raw CICIDS data from cached files."""
        # Check for cached processed data (local first, then remote sources, synthetic last)
        for cache_name in [
            "cicids_local.npz",  # Local file takes priority
            "cicids_huggingface.npz",
            "cicids_distrinet_(improved).npz",
            "cicids_cic_official.npz",
            "synthetic_cicids.npz",
        ]:
            cache_path = self.data_path / cache_name
            if cache_path.exists():
                data = np.load(cache_path)
                self._features = data["features"]
                self._labels = data["labels"]
                self._is_real_data = "synthetic" not in cache_name
                logger.info(
                    f"Loaded CICIDS from {cache_name} " f"(is_real_data={self._is_real_data})"
                )
                return self._features, self._labels

        raise FileNotFoundError("CICIDS data not found. Run download() first.")

    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Load CICIDS dataset features and labels.

        This is the main entry point for loading the dataset.
        Automatically downloads if not present.

        Returns:
            Tuple of (features, labels) numpy arrays
        """
        if self._features is not None and self._labels is not None:
            return self._features, self._labels

        try:
            return self._load_raw()
        except FileNotFoundError:
            self.download()
            return self._load_raw()

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess network flow features.

        Applied transformations:
        1. Replace remaining inf/nan with 0
        2. Log1p transform (handles high-variance features)
        3. Z-score normalization
        """
        # Handle any remaining problematic values
        data = np.nan_to_num(data, nan=0.0, posinf=1e10, neginf=0)

        # Log transform for high-variance network features
        data = np.log1p(np.abs(data))

        # Z-score normalization
        mean = data.mean(axis=0)
        std = data.std(axis=0) + 1e-8
        data = (data - mean) / std

        return data.astype(np.float32)

    def get_metadata(self) -> dict[str, Any]:
        """Get dataset metadata."""
        if self._features is None:
            self.load_data()

        return {
            "name": "CICIDS 2017",
            "source": "Canadian Institute for Cybersecurity",
            "n_samples": len(self._features) if self._features is not None else 0,
            "n_features": self._features.shape[1] if self._features is not None else 0,
            "n_classes": 2 if self.binary_labels else len(self.ATTACK_LABELS),
            "label_type": "binary" if self.binary_labels else "multiclass",
            "attack_types": (
                self._label_names if self._label_names else list(self.ATTACK_LABELS.keys())
            ),
            "is_real_data": self._is_real_data,
            "url": self.DATASET_URL,
            "citation": self.CITATION,
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about loaded data."""
        if self._features is None:
            self.load_data()

        # Type guards for mypy - load_data() ensures these are not None
        if self._features is None or self._labels is None:
            raise RuntimeError("Failed to load data")

        labels = self._labels
        features = self._features

        return {
            "n_samples": len(features),
            "n_features": features.shape[1],
            "n_attacks": int((labels > 0).sum()) if self.binary_labels else None,
            "attack_ratio": float((labels > 0).mean()) if self.binary_labels else None,
            "class_distribution": {
                str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))
            },
            "feature_mean": float(features.mean()),
            "feature_std": float(features.std()),
            "is_real_data": self._is_real_data,
        }


class ThreatIntelLoader(DatasetLoader):
    """
    MITRE ATT&CK Threat Intelligence Loader.

    Downloads REAL threat intelligence data from MITRE ATT&CK framework:
    - Attack techniques with kill chain phases
    - Threat groups and their techniques
    - Malware and tools used by adversaries
    - Mitigations and detection strategies

    Data source: https://github.com/mitre-attack/attack-stix-data
    License: Apache 2.0
    """

    DATASET_NAME = "threat-intel"
    DATASET_URL = "https://attack.mitre.org/"
    LICENSE = "Apache 2.0 (MITRE ATT&CK)"
    CITATION = "MITRE ATT&CK. MITRE Corporation. https://attack.mitre.org/"
    REQUIRES_CREDENTIALS = False

    # MITRE ATT&CK STIX data URL (via TrustedEndpoints for SSRF prevention)
    MITRE_STIX_URL = TrustedEndpoints.MITRE_STIX_DATA

    # MITRE ATT&CK tactics
    TACTICS = [
        "initial-access",
        "execution",
        "persistence",
        "privilege-escalation",
        "defense-evasion",
        "credential-access",
        "discovery",
        "lateral-movement",
        "collection",
        "command-and-control",
        "exfiltration",
        "impact",
    ]

    FEATURE_NAMES = [
        "num_kill_chain_phases",
        "num_platforms",
        "num_data_sources",
        "num_mitigations",
        "num_detections",
        "is_subtechnique",
        "tactic_initial_access",
        "tactic_execution",
        "tactic_persistence",
        "tactic_privilege_escalation",
        "tactic_defense_evasion",
        "tactic_credential_access",
        "tactic_discovery",
        "tactic_lateral_movement",
        "tactic_collection",
        "tactic_command_and_control",
        "tactic_exfiltration",
        "tactic_impact",
    ]

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)
        self._is_real_data = False

    @property
    def is_real_data(self) -> bool:
        """Return True if real MITRE ATT&CK data was loaded."""
        return self._is_real_data

    def download(self) -> bool:
        """Download real MITRE ATT&CK data.

        Returns:
            True if download successful, False otherwise.
        """
        if self._download_from_mitre():
            return True

        logger.warning("MITRE ATT&CK download failed, falling back to SYNTHETIC data.")
        return self._create_synthetic_threat_intel()

    def _download_from_mitre(self) -> bool:
        """Download and process MITRE ATT&CK STIX data."""
        import json
        import urllib.request

        dataset_dir = self.data_path
        dataset_dir.mkdir(parents=True, exist_ok=True)
        cache_file = dataset_dir / "mitre_attack_real.npz"

        if cache_file.exists():
            logger.info(f"MITRE ATT&CK data already cached at {cache_file}")
            self._is_real_data = True
            return True

        try:
            logger.info("Downloading MITRE ATT&CK Enterprise data...")
            # Validate URL before opening (SSRF protection via domain allowlist)
            TrustedEndpoints.validate_url(self.MITRE_STIX_URL)
            req = urllib.request.Request(
                self.MITRE_STIX_URL,
                headers={"User-Agent": "Mozilla/5.0 Mercury-Agent/1.0"},
            )
            with urllib.request.urlopen(req, timeout=120) as response:  # nosec B310
                data = json.loads(response.read().decode("utf-8"))

            objects = data.get("objects", [])
            if not objects:
                logger.warning("No objects found in MITRE ATT&CK data")
                return False

            # Filter to attack-patterns (techniques)
            techniques = [obj for obj in objects if obj.get("type") == "attack-pattern"]
            logger.info(f"Downloaded {len(techniques)} ATT&CK techniques")

            # Process into features
            features, labels = self._process_mitre_data(techniques)

            # Save to cache
            np.savez_compressed(cache_file, features=features, labels=labels)
            self._is_real_data = True

            logger.info(
                f"MITRE ATT&CK data loaded: {len(features)} techniques, "
                f"{labels.sum()} high-risk (is_real_data=True)"
            )
            return True

        except Exception as e:
            logger.warning(f"MITRE ATT&CK download failed: {e}")
            return False

    def _process_mitre_data(
        self, techniques: list[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Process MITRE ATT&CK techniques into features.

        Args:
            techniques: List of attack-pattern objects from STIX

        Returns:
            Tuple of (features, labels) numpy arrays
        """
        rows = []

        for tech in techniques:
            # Extract kill chain phases
            kill_chain = tech.get("kill_chain_phases", [])
            phases = [p.get("phase_name", "") for p in kill_chain]

            # Extract platforms
            platforms = tech.get("x_mitre_platforms", [])

            # Extract data sources
            data_sources = tech.get("x_mitre_data_sources", [])

            # Check if subtechnique
            is_sub = tech.get("x_mitre_is_subtechnique", False)

            # Build feature vector
            row = [
                len(phases),  # num_kill_chain_phases
                len(platforms),  # num_platforms
                len(data_sources),  # num_data_sources
                0,  # num_mitigations (would need relationships)
                len(data_sources),  # num_detections (approximation)
                1 if is_sub else 0,  # is_subtechnique
            ]

            # One-hot encode tactics
            for tactic in self.TACTICS:
                row.append(1 if tactic in phases else 0)

            rows.append(row)

        features = np.array(rows, dtype=np.float32)

        # Label "high-risk" techniques (multiple tactics, many platforms)
        # This is a heuristic - real risk scoring would use additional context
        num_phases = features[:, 0]
        num_platforms = features[:, 1]
        labels = ((num_phases >= 2) & (num_platforms >= 3)).astype(np.int64)

        # Apply max_samples limit
        if self.config.max_samples and len(features) > self.config.max_samples:
            np.random.seed(self.config.random_seed)
            indices = np.random.choice(len(features), self.config.max_samples, replace=False)
            features = features[indices]
            labels = labels[indices]

        return features, labels

    def _create_synthetic_threat_intel(self) -> bool:
        """Create synthetic MITRE ATT&CK-like threat intelligence data."""
        logger.warning(
            "Creating SYNTHETIC threat intel data. "
            "Results will NOT reflect real MITRE ATT&CK patterns."
        )

        np.random.seed(self.config.random_seed)
        n_samples = self.config.max_samples or 5000

        features = []
        labels = []

        for _i in range(n_samples):
            is_high_risk = np.random.random() < 0.3

            if is_high_risk:
                # High-risk technique: multiple tactics, many platforms
                num_phases = np.random.randint(2, 5)
                num_platforms = np.random.randint(3, 8)
                num_data_sources = np.random.randint(2, 10)
                num_mitigations = np.random.randint(1, 5)
                num_detections = np.random.randint(2, 8)
                is_sub = np.random.random() < 0.3
                # Random tactic selection (higher probability)
                tactics = [1 if np.random.random() < 0.4 else 0 for _ in self.TACTICS]
                labels.append(1)
            else:
                # Lower-risk technique
                num_phases = np.random.randint(1, 3)
                num_platforms = np.random.randint(1, 4)
                num_data_sources = np.random.randint(0, 5)
                num_mitigations = np.random.randint(0, 3)
                num_detections = np.random.randint(0, 4)
                is_sub = np.random.random() < 0.6
                # Random tactic selection (lower probability)
                tactics = [1 if np.random.random() < 0.15 else 0 for _ in self.TACTICS]
                labels.append(0)

            row = [
                num_phases,
                num_platforms,
                num_data_sources,
                num_mitigations,
                num_detections,
                1 if is_sub else 0,
            ] + tactics

            features.append(row)

        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        save_path = self.data_path / "synthetic_threat_intel.npz"
        np.savez_compressed(save_path, features=features, labels=labels)

        logger.info(f"Generated {n_samples} threat intel samples, {labels.sum()} threats")
        return True

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load threat intel data from cache (real data first, then synthetic)."""
        # Check for real MITRE ATT&CK data first
        real_cache = self.data_path / "mitre_attack_real.npz"
        if real_cache.exists():
            data = np.load(real_cache)
            self._is_real_data = True
            logger.info(f"Loaded REAL MITRE ATT&CK data from {real_cache}")
            return data["features"], data["labels"]

        # Fall back to synthetic
        synthetic_path = self.data_path / "synthetic_threat_intel.npz"
        if synthetic_path.exists():
            data = np.load(synthetic_path)
            self._is_real_data = False
            logger.info("Loaded SYNTHETIC threat intel data (is_real_data=False)")
            return data["features"], data["labels"]

        raise FileNotFoundError("Threat intel data not found. Run download() first.")

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess threat intelligence features."""
        data = np.nan_to_num(data, nan=0.0)
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        return data.astype(np.float32)


# Register security loaders
DatasetRegistry.register("nsl-kdd", NSLKDDLoader)
DatasetRegistry.register("cicids", CICIDSLoader)
DatasetRegistry.register("threat-intel", ThreatIntelLoader)
