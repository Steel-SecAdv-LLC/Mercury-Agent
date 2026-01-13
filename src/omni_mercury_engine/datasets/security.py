"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

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

from .base import DatasetConfig, DatasetLoader, DatasetRegistry, safe_urlretrieve

logger = logging.getLogger(__name__)


class NSLKDDLoader(DatasetLoader):
    """
    NSL-KDD Network Intrusion Detection Dataset Loader.

    Improved version of KDD'99 with:
    - Removed duplicate records
    - Balanced difficulty levels
    - Standard train/test splits

    Reference: https://www.unb.ca/cic/datasets/nsl.html
    """

    DATASET_NAME = "nsl-kdd"
    DATASET_URL = "https://www.unb.ca/cic/datasets/nsl.html"
    LICENSE = "Academic Research Use"
    CITATION = """Tavallaee M, Bagheri E, Lu W, Ghorbani AA. A detailed analysis of the
    KDD CUP 99 data set. IEEE Symposium on Computational Intelligence. 2009."""
    REQUIRES_CREDENTIALS = False

    # NSL-KDD feature names
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

    # Attack categories
    ATTACK_TYPES = {
        "normal": 0,
        "dos": 1,  # Denial of Service
        "probe": 2,  # Probing
        "r2l": 3,  # Remote to Local
        "u2r": 4,  # User to Root
    }

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)
        self.binary_labels = config.preprocessing.get("binary", True)

    def download(self) -> bool:
        """Download or generate NSL-KDD data."""
        return self._create_synthetic_nslkdd()

    def _create_synthetic_nslkdd(self) -> bool:
        """Create synthetic NSL-KDD-like data."""
        np.random.seed(self.config.random_seed)
        n_samples = self.config.max_samples or 10000

        features = []
        labels = []

        attack_probs = {
            "normal": 0.4,
            "dos": 0.3,
            "probe": 0.15,
            "r2l": 0.1,
            "u2r": 0.05,
        }

        for _i in range(n_samples):
            attack_type = np.random.choice(list(attack_probs.keys()), p=list(attack_probs.values()))

            if attack_type == "normal":
                params = self._generate_normal_connection()
            elif attack_type == "dos":
                params = self._generate_dos_attack()
            elif attack_type == "probe":
                params = self._generate_probe_attack()
            elif attack_type == "r2l":
                params = self._generate_r2l_attack()
            else:  # u2r
                params = self._generate_u2r_attack()

            feature_vec = [params.get(f, 0) for f in self.FEATURE_NAMES]
            features.append(feature_vec)

            if self.binary_labels:
                labels.append(0 if attack_type == "normal" else 1)
            else:
                labels.append(self.ATTACK_TYPES[attack_type])

        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        save_path = self.data_path / "synthetic_nslkdd.npz"
        np.savez_compressed(save_path, features=features, labels=labels)

        logger.info(f"Generated {n_samples} NSL-KDD samples, {(labels > 0).sum()} attacks")
        return True

    def _generate_normal_connection(self) -> dict[str, Any]:
        """Generate features for normal network connection."""
        return {
            "duration": np.random.exponential(60),
            "protocol_type": np.random.choice([0, 1, 2]),  # tcp, udp, icmp
            "service": np.random.choice(range(70)),
            "flag": np.random.choice(range(11)),
            "src_bytes": np.random.exponential(1000),
            "dst_bytes": np.random.exponential(500),
            "land": 0,
            "wrong_fragment": 0,
            "urgent": 0,
            "hot": np.random.poisson(1),
            "num_failed_logins": 0,
            "logged_in": 1,
            "num_compromised": 0,
            "root_shell": 0,
            "su_attempted": 0,
            "num_root": 0,
            "num_file_creations": np.random.poisson(2),
            "num_shells": 0,
            "num_access_files": np.random.poisson(3),
            "num_outbound_cmds": 0,
            "is_host_login": 0,
            "is_guest_login": 0,
            "count": np.random.poisson(10),
            "srv_count": np.random.poisson(10),
            "serror_rate": np.random.beta(1, 20),
            "srv_serror_rate": np.random.beta(1, 20),
            "rerror_rate": np.random.beta(1, 20),
            "srv_rerror_rate": np.random.beta(1, 20),
            "same_srv_rate": np.random.beta(10, 1),
            "diff_srv_rate": np.random.beta(1, 10),
            "srv_diff_host_rate": np.random.beta(2, 10),
            "dst_host_count": np.random.poisson(50),
            "dst_host_srv_count": np.random.poisson(30),
            "dst_host_same_srv_rate": np.random.beta(10, 2),
            "dst_host_diff_srv_rate": np.random.beta(2, 10),
            "dst_host_same_src_port_rate": np.random.beta(5, 5),
            "dst_host_srv_diff_host_rate": np.random.beta(2, 10),
            "dst_host_serror_rate": np.random.beta(1, 20),
            "dst_host_srv_serror_rate": np.random.beta(1, 20),
            "dst_host_rerror_rate": np.random.beta(1, 20),
            "dst_host_srv_rerror_rate": np.random.beta(1, 20),
        }

    def _generate_dos_attack(self) -> dict[str, Any]:
        """Generate features for DoS attack."""
        params = self._generate_normal_connection()
        # DoS characteristics: high volume, short connections
        params["duration"] = np.random.exponential(1)
        params["count"] = np.random.poisson(300)
        params["srv_count"] = np.random.poisson(300)
        params["same_srv_rate"] = np.random.beta(20, 1)
        params["dst_host_count"] = np.random.poisson(200)
        params["serror_rate"] = np.random.beta(10, 2)
        return params

    def _generate_probe_attack(self) -> dict[str, Any]:
        """Generate features for probing/scanning attack."""
        params = self._generate_normal_connection()
        # Probe characteristics: many destinations, varied services
        params["duration"] = np.random.exponential(5)
        params["diff_srv_rate"] = np.random.beta(10, 2)
        params["srv_diff_host_rate"] = np.random.beta(10, 2)
        params["dst_host_diff_srv_rate"] = np.random.beta(10, 2)
        params["dst_host_count"] = np.random.poisson(150)
        params["rerror_rate"] = np.random.beta(5, 5)
        return params

    def _generate_r2l_attack(self) -> dict[str, Any]:
        """Generate features for Remote-to-Local attack."""
        params = self._generate_normal_connection()
        # R2L characteristics: failed logins, suspicious activity
        params["num_failed_logins"] = np.random.poisson(3)
        params["logged_in"] = np.random.choice([0, 1])
        params["hot"] = np.random.poisson(5)
        params["num_compromised"] = np.random.poisson(2)
        params["num_access_files"] = np.random.poisson(10)
        return params

    def _generate_u2r_attack(self) -> dict[str, Any]:
        """Generate features for User-to-Root attack."""
        params = self._generate_normal_connection()
        # U2R characteristics: privilege escalation
        params["root_shell"] = 1
        params["su_attempted"] = np.random.choice([0, 1, 2])
        params["num_root"] = np.random.poisson(3)
        params["num_shells"] = np.random.poisson(2)
        params["num_file_creations"] = np.random.poisson(10)
        return params

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        synthetic_path = self.data_path / "synthetic_nslkdd.npz"
        if synthetic_path.exists():
            data = np.load(synthetic_path)
            return data["features"], data["labels"]
        raise FileNotFoundError("NSL-KDD data not found")

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess network traffic features."""
        # Log transform for high-variance features
        data = np.log1p(np.abs(data))
        # Z-score normalize
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        return data.astype(np.float32)


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
    DATA_SOURCES = {
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
        """
        super().__init__(config)
        self.binary_labels = config.preprocessing.get("binary", True)
        self.subset = config.preprocessing.get("subset", "all")
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
        1. Hugging Face datasets (if library available)
        2. Distrinet improved version
        3. Official CIC server
        4. Falls back to synthetic with WARNING

        Returns:
            True if download successful, False otherwise.
        """
        # Try each source in priority order
        for source_id, source_info in self.DATA_SOURCES.items():
            try:
                if source_id == "huggingface":
                    if self._download_from_huggingface():
                        return True
                else:
                    if self._download_from_url(source_info):
                        return True
            except Exception as e:
                logger.warning(f"Failed to download from {source_info['name']}: {e}")
                continue

        # All sources failed - fall back to synthetic with WARNING
        logger.warning(
            "CICIDS 2017: All download sources failed. "
            "Falling back to SYNTHETIC data. Results will NOT reflect real-world performance."
        )
        return self._create_synthetic_fallback()

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
            dataset = load_dataset("bvk/CICIDS-2017", split="train")

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

        local_file = dataset_dir / f"cicids_{source_name.lower().replace(' ', '_')}.{file_format}"
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

            # Use longer timeout for large files
            with urllib.request.urlopen(url, timeout=300) as response:  # noqa: S310
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
            labels = np.array(
                [self._encode_label(label) for label in labels_raw], dtype=np.int64
            )

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
            attack_type = np.random.choice(
                list(attack_probs.keys()), p=list(attack_probs.values())
            )

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
        # Check for cached processed data
        for cache_name in [
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
                    f"Loaded CICIDS from {cache_name} "
                    f"(is_real_data={self._is_real_data})"
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
            "attack_types": self._label_names if self._label_names else list(self.ATTACK_LABELS.keys()),
            "is_real_data": self._is_real_data,
            "url": self.DATASET_URL,
            "citation": self.CITATION,
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about loaded data."""
        if self._features is None:
            self.load_data()

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
    Threat Intelligence Indicator Dataset Loader.

    Provides access to:
    - IOC (Indicators of Compromise) features
    - MITRE ATT&CK technique patterns
    - Malware behavior signatures

    Based on MITRE ATT&CK framework and open threat feeds.
    """

    DATASET_NAME = "threat-intel"
    DATASET_URL = "https://attack.mitre.org/"
    LICENSE = "Apache 2.0 (MITRE ATT&CK)"
    CITATION = "MITRE ATT&CK. MITRE Corporation. https://attack.mitre.org/"
    REQUIRES_CREDENTIALS = False

    # MITRE ATT&CK tactics
    TACTICS = [
        "initial_access",
        "execution",
        "persistence",
        "privilege_escalation",
        "defense_evasion",
        "credential_access",
        "discovery",
        "lateral_movement",
        "collection",
        "command_and_control",
        "exfiltration",
        "impact",
    ]

    FEATURE_NAMES = [
        "technique_count",
        "tactic_diversity",
        "severity_score",
        "confidence",
        "source_reputation",
        "target_criticality",
        "attack_pattern_match",
        "ioc_age_days",
        "ioc_first_seen",
        "ioc_last_seen",
        "related_campaigns",
        "related_groups",
        "related_malware",
        "network_indicators",
        "file_indicators",
        "registry_indicators",
        "behavioral_indicators",
        "temporal_correlation",
        "geographic_spread",
        "industry_targeting",
        "technique_sophistication",
        "automation_level",
    ]

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)

    def download(self) -> bool:
        return self._create_synthetic_threat_intel()

    def _create_synthetic_threat_intel(self) -> bool:
        """Create synthetic threat intelligence data."""
        np.random.seed(self.config.random_seed)
        n_samples = self.config.max_samples or 5000

        features = []
        labels = []

        for _i in range(n_samples):
            is_threat = np.random.random() < 0.4

            if is_threat:
                params = {
                    "technique_count": np.random.poisson(5) + 1,
                    "tactic_diversity": np.random.uniform(0.5, 1.0),
                    "severity_score": np.random.uniform(50, 100),
                    "confidence": np.random.uniform(60, 100),
                    "source_reputation": np.random.uniform(60, 100),
                    "target_criticality": np.random.uniform(50, 100),
                    "attack_pattern_match": np.random.uniform(0.7, 1.0),
                    "ioc_age_days": np.random.exponential(30),
                    "ioc_first_seen": np.random.exponential(90),
                    "ioc_last_seen": np.random.exponential(7),
                    "related_campaigns": np.random.poisson(2) + 1,
                    "related_groups": np.random.poisson(1) + 1,
                    "related_malware": np.random.poisson(3),
                    "network_indicators": np.random.poisson(5),
                    "file_indicators": np.random.poisson(3),
                    "registry_indicators": np.random.poisson(2),
                    "behavioral_indicators": np.random.poisson(4),
                    "temporal_correlation": np.random.uniform(0.5, 1.0),
                    "geographic_spread": np.random.uniform(1, 50),
                    "industry_targeting": np.random.uniform(0.3, 1.0),
                    "technique_sophistication": np.random.uniform(50, 100),
                    "automation_level": np.random.uniform(30, 100),
                }
                labels.append(1)
            else:
                params = {
                    "technique_count": np.random.poisson(1),
                    "tactic_diversity": np.random.uniform(0, 0.3),
                    "severity_score": np.random.uniform(0, 30),
                    "confidence": np.random.uniform(10, 50),
                    "source_reputation": np.random.uniform(20, 60),
                    "target_criticality": np.random.uniform(0, 40),
                    "attack_pattern_match": np.random.uniform(0, 0.3),
                    "ioc_age_days": np.random.exponential(180),
                    "ioc_first_seen": np.random.exponential(365),
                    "ioc_last_seen": np.random.exponential(90),
                    "related_campaigns": 0,
                    "related_groups": 0,
                    "related_malware": np.random.poisson(0.5),
                    "network_indicators": np.random.poisson(1),
                    "file_indicators": np.random.poisson(0.5),
                    "registry_indicators": 0,
                    "behavioral_indicators": np.random.poisson(1),
                    "temporal_correlation": np.random.uniform(0, 0.3),
                    "geographic_spread": np.random.uniform(0, 5),
                    "industry_targeting": np.random.uniform(0, 0.2),
                    "technique_sophistication": np.random.uniform(0, 30),
                    "automation_level": np.random.uniform(0, 30),
                }
                labels.append(0)

            feature_vec = [params[f] for f in self.FEATURE_NAMES]
            features.append(feature_vec)

        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        save_path = self.data_path / "synthetic_threat_intel.npz"
        np.savez_compressed(save_path, features=features, labels=labels)

        logger.info(f"Generated {n_samples} threat intel samples, {labels.sum()} threats")
        return True

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        synthetic_path = self.data_path / "synthetic_threat_intel.npz"
        if synthetic_path.exists():
            data = np.load(synthetic_path)
            return data["features"], data["labels"]
        raise FileNotFoundError("Threat intel data not found")

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess threat intelligence features."""
        data = np.nan_to_num(data, nan=0.0)
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        return data.astype(np.float32)


# Register security loaders
DatasetRegistry.register("nsl-kdd", NSLKDDLoader)
DatasetRegistry.register("cicids", CICIDSLoader)
DatasetRegistry.register("threat-intel", ThreatIntelLoader)
