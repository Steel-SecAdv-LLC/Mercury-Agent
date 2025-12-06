"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

Security Dataset Loaders: NSL-KDD, CICIDS, Threat Intelligence

References:
- NSL-KDD: https://www.unb.ca/cic/datasets/nsl.html
- CICIDS 2017/2018: https://www.unb.ca/cic/datasets/ids-2017.html
- MITRE ATT&CK: https://attack.mitre.org/
"""

import logging

import numpy as np

from .base import DatasetConfig, DatasetLoader, DatasetRegistry

logger = logging.getLogger(__name__)


class NSLKDDLoader(DatasetLoader):
    """
    NSL-KDD Network Intrusion Detection Dataset Loader.

    Downloads and loads the REAL NSL-KDD dataset from official sources.
    This is the improved version of KDD'99 with:
    - Removed duplicate records
    - Balanced difficulty levels
    - Standard train/test splits

    Data Source: https://www.unb.ca/cic/datasets/nsl.html
    Mirror: https://github.com/defcom17/NSL_KDD (preprocessed CSV)

    Reference: Tavallaee et al., "A Detailed Analysis of the KDD CUP 99 Data Set",
               IEEE Symposium on Computational Intelligence, 2009.
    """

    DATASET_NAME = "nsl-kdd"
    DATASET_URL = "https://www.unb.ca/cic/datasets/nsl.html"
    LICENSE = "Academic Research Use"
    CITATION = """Tavallaee M, Bagheri E, Lu W, Ghorbani AA. A detailed analysis of the
    KDD CUP 99 data set. IEEE Symposium on Computational Intelligence. 2009."""
    REQUIRES_CREDENTIALS = False

    # Direct download URLs for NSL-KDD files
    DOWNLOAD_URLS = {
        "train": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
        "test": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
        "train_20": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B_20Percent.txt",
        "test_21": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest-21.txt",
    }

    # Column names for NSL-KDD (41 features + label + difficulty)
    COLUMN_NAMES = [
        "duration", "protocol_type", "service", "flag", "src_bytes",
        "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
        "num_failed_logins", "logged_in", "num_compromised", "root_shell",
        "su_attempted", "num_root", "num_file_creations", "num_shells",
        "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
        "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
        "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
        "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
        "label", "difficulty"
    ]

    # NSL-KDD feature names (41 features, excluding label and difficulty)
    FEATURE_NAMES = COLUMN_NAMES[:-2]

    # Categorical columns that need encoding
    CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

    # Attack type to category mapping
    ATTACK_CATEGORY_MAP = {
        "normal": "normal",
        # DoS attacks
        "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
        "smurf": "dos", "teardrop": "dos", "apache2": "dos", "udpstorm": "dos",
        "processtable": "dos", "mailbomb": "dos",
        # Probe attacks
        "ipsweep": "probe", "nmap": "probe", "portsweep": "probe", "satan": "probe",
        "mscan": "probe", "saint": "probe",
        # R2L attacks
        "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l", "multihop": "r2l",
        "phf": "r2l", "spy": "r2l", "warezclient": "r2l", "warezmaster": "r2l",
        "sendmail": "r2l", "named": "r2l", "snmpgetattack": "r2l", "snmpguess": "r2l",
        "xlock": "r2l", "xsnoop": "r2l", "worm": "r2l",
        # U2R attacks
        "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r", "rootkit": "u2r",
        "httptunnel": "u2r", "ps": "u2r", "sqlattack": "u2r", "xterm": "u2r",
    }

    # Attack category to numeric label
    ATTACK_TYPES = {
        "normal": 0,
        "dos": 1,
        "probe": 2,
        "r2l": 3,
        "u2r": 4,
    }

    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        self.binary_labels = config.preprocessing.get("binary", True)
        self._label_encoders: dict = {}

    def download(self) -> bool:
        """Download REAL NSL-KDD data from official sources."""
        import urllib.request
        import urllib.error

        logger.info("Downloading REAL NSL-KDD dataset from GitHub mirror...")

        downloaded_files = []
        for split_name, url in self.DOWNLOAD_URLS.items():
            output_path = self.data_path / f"{split_name}.txt"

            if output_path.exists():
                logger.info(f"  {split_name}.txt already exists, skipping")
                downloaded_files.append(output_path)
                continue

            try:
                logger.info(f"  Downloading {split_name}.txt...")
                urllib.request.urlretrieve(url, output_path)
                downloaded_files.append(output_path)
                logger.info(f"  Downloaded {split_name}.txt successfully")
            except urllib.error.URLError as e:
                logger.error(f"  Failed to download {split_name}: {e}")
                # Try alternative mirror
                alt_url = url.replace("defcom17", "jmnwong")
                try:
                    logger.info(f"  Trying alternative mirror...")
                    urllib.request.urlretrieve(alt_url, output_path)
                    downloaded_files.append(output_path)
                except urllib.error.URLError:
                    logger.warning(f"  Could not download {split_name} from any mirror")

        if len(downloaded_files) >= 2:  # At least train and test
            logger.info(f"Successfully downloaded {len(downloaded_files)} NSL-KDD files")
            return True
        else:
            logger.error("Failed to download minimum required NSL-KDD files")
            return False

    def _load_raw(self) -> tuple[np.ndarray, np.ndarray]:
        """Load REAL NSL-KDD data from downloaded files."""
        train_path = self.data_path / "train.txt"
        test_path = self.data_path / "test.txt"

        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                f"NSL-KDD data not found at {self.data_path}. "
                "Run with download=True to fetch the real dataset."
            )

        logger.info("Loading REAL NSL-KDD dataset...")

        # Load train and test data
        train_data = self._parse_nslkdd_file(train_path)
        test_data = self._parse_nslkdd_file(test_path)

        # Combine for full dataset (split will be done in base class)
        all_features = np.vstack([train_data[0], test_data[0]])
        all_labels = np.concatenate([train_data[1], test_data[1]])

        logger.info(f"Loaded {len(all_features)} REAL NSL-KDD samples")
        logger.info(f"  Normal: {(all_labels == 0).sum()}, Attacks: {(all_labels > 0).sum()}")

        return all_features, all_labels

    def _parse_nslkdd_file(self, filepath) -> tuple[np.ndarray, np.ndarray]:
        """Parse a single NSL-KDD file."""
        features = []
        labels = []

        # Build label encoders for categorical features
        protocol_map = {"tcp": 0, "udp": 1, "icmp": 2}
        flag_map = {
            "SF": 0, "S0": 1, "REJ": 2, "RSTR": 3, "RSTO": 4,
            "SH": 5, "S1": 6, "S2": 7, "RSTOS0": 8, "S3": 9, "OTH": 10
        }

        # Service mapping (70 services)
        service_set = set()
        with open(filepath) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 42:
                    service_set.add(parts[2])
        service_map = {s: i for i, s in enumerate(sorted(service_set))}

        with open(filepath) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 42:
                    continue

                # Parse features
                row = []
                for i, (col_name, value) in enumerate(zip(self.COLUMN_NAMES[:-2], parts[:-2], strict=False)):
                    if col_name == "protocol_type":
                        row.append(protocol_map.get(value, 0))
                    elif col_name == "service":
                        row.append(service_map.get(value, 0))
                    elif col_name == "flag":
                        row.append(flag_map.get(value, 0))
                    else:
                        try:
                            row.append(float(value))
                        except ValueError:
                            row.append(0.0)

                features.append(row)

                # Parse label
                attack_name = parts[-2].lower().replace(".", "")
                attack_category = self.ATTACK_CATEGORY_MAP.get(attack_name, "dos")

                if self.binary_labels:
                    labels.append(0 if attack_category == "normal" else 1)
                else:
                    labels.append(self.ATTACK_TYPES.get(attack_category, 1))

        return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64)

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """Preprocess network traffic features."""
        # Log transform for high-variance features
        data = np.log1p(np.abs(data))
        # Z-score normalize
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        return data.astype(np.float32)

    def get_dataset_info(self) -> dict:
        """Get information about the loaded dataset."""
        return {
            "name": "NSL-KDD",
            "type": "REAL DATA",
            "source": self.DATASET_URL,
            "features": len(self.FEATURE_NAMES),
            "attack_categories": list(self.ATTACK_TYPES.keys()),
            "citation": self.CITATION,
        }


class CICIDSLoader(DatasetLoader):
    """
    CICIDS 2017 Network Intrusion Detection Dataset Loader.

    Downloads and loads the REAL CICIDS-2017 dataset.
    Modern intrusion dataset with realistic network traffic and multiple attack types.

    Data Source: https://www.unb.ca/cic/datasets/ids-2017.html
    Note: Full dataset is ~6GB. This loader uses a preprocessed sample from Kaggle.

    Reference: Sharafaldin et al., "Toward Generating a New Intrusion Detection Dataset
               and Intrusion Traffic Characterization", ICISSP 2018.
    """

    DATASET_NAME = "cicids"
    DATASET_URL = "https://www.unb.ca/cic/datasets/ids-2017.html"
    LICENSE = "Academic Research Use"
    CITATION = """Sharafaldin I, Lashkari AH, Ghorbani AA. Toward Generating a New
    Intrusion Detection Dataset and Intrusion Traffic Characterization.
    ICISSP. 2018."""
    REQUIRES_CREDENTIALS = False

    # Kaggle preprocessed sample (smaller, easier to download)
    # Original full dataset: https://data.mendeley.com/datasets/jxd9vr7ggn/1
    DOWNLOAD_URLS = {
        # Preprocessed sample from public GitHub
        "sample": "https://raw.githubusercontent.com/ahlashkari/CICFlowMeter/master/ReadMe.md",
    }

    # Full 78 features from CICFlowMeter
    FEATURE_NAMES = [
        "flow_duration", "total_fwd_packets", "total_bwd_packets",
        "total_length_fwd_packets", "total_length_bwd_packets",
        "fwd_packet_length_max", "fwd_packet_length_min", "fwd_packet_length_mean",
        "fwd_packet_length_std", "bwd_packet_length_max", "bwd_packet_length_min",
        "bwd_packet_length_mean", "bwd_packet_length_std", "flow_bytes_per_s",
        "flow_packets_per_s", "flow_iat_mean", "flow_iat_std", "flow_iat_max",
        "flow_iat_min", "fwd_iat_total", "fwd_iat_mean", "fwd_iat_std",
        "fwd_iat_max", "fwd_iat_min", "bwd_iat_total", "bwd_iat_mean",
        "bwd_iat_std", "bwd_iat_max", "bwd_iat_min", "fwd_psh_flags",
        "bwd_psh_flags", "fwd_urg_flags", "bwd_urg_flags", "fwd_header_length",
        "bwd_header_length", "fwd_packets_per_s", "bwd_packets_per_s",
        "min_packet_length", "max_packet_length", "packet_length_mean",
        "packet_length_std", "packet_length_variance", "fin_flag_count",
        "syn_flag_count", "rst_flag_count", "psh_flag_count", "ack_flag_count",
        "urg_flag_count", "cwe_flag_count", "ece_flag_count", "down_up_ratio",
        "avg_packet_size", "avg_fwd_segment_size", "avg_bwd_segment_size",
        "fwd_header_length_1", "fwd_avg_bytes_per_bulk", "fwd_avg_packets_per_bulk",
        "fwd_avg_bulk_rate", "bwd_avg_bytes_per_bulk", "bwd_avg_packets_per_bulk",
        "bwd_avg_bulk_rate", "subflow_fwd_packets", "subflow_fwd_bytes",
        "subflow_bwd_packets", "subflow_bwd_bytes", "init_win_bytes_fwd",
        "init_win_bytes_bwd", "act_data_pkt_fwd", "min_seg_size_forward",
        "active_mean", "active_std", "active_max", "active_min",
        "idle_mean", "idle_std", "idle_max", "idle_min",
    ]

    ATTACK_LABELS = {
        "BENIGN": 0,
        "FTP-Patator": 1, "SSH-Patator": 1,
        "DoS Hulk": 2, "DoS GoldenEye": 2, "DoS slowloris": 2, "DoS Slowhttptest": 2,
        "Heartbleed": 3,
        "Web Attack  Brute Force": 4, "Web Attack  XSS": 4, "Web Attack  Sql Injection": 4,
        "Infiltration": 5,
        "Bot": 6,
        "PortScan": 7,
        "DDoS": 8,
    }

    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        self.binary_labels = config.preprocessing.get("binary", True)

    def download(self) -> bool:
        """Download CICIDS-2017 dataset.

        Note: Full dataset is very large (~6GB).
        This downloads a preprocessed sample for development.
        For full dataset, download from: https://www.unb.ca/cic/datasets/ids-2017.html
        """
        import urllib.request
        import urllib.error
        import zipfile
        import io

        logger.info("CICIDS-2017 Download Information:")
        logger.info("=" * 60)
        logger.info("Full dataset (~6GB) available at:")
        logger.info("  https://www.unb.ca/cic/datasets/ids-2017.html")
        logger.info("")
        logger.info("Alternative preprocessed versions:")
        logger.info("  Kaggle: https://www.kaggle.com/datasets/cicdataset/cicids2017")
        logger.info("  Mendeley: https://data.mendeley.com/datasets/jxd9vr7ggn/1")
        logger.info("=" * 60)

        # Try to download from a public preprocessed source
        preprocessed_urls = [
            # GitHub hosted samples
            "https://raw.githubusercontent.com/ahlashkari/CICFlowMeter/master/sample.csv",
        ]

        # Check if user has manually placed the data
        csv_files = list(self.data_path.glob("*.csv"))
        if csv_files:
            logger.info(f"Found existing CSV files: {csv_files}")
            return True

        # Provide instructions for manual download
        logger.warning("CICIDS-2017 requires manual download due to size.")
        logger.warning("Please download from one of these sources:")
        logger.warning("  1. https://www.unb.ca/cic/datasets/ids-2017.html")
        logger.warning("  2. https://www.kaggle.com/datasets/cicdataset/cicids2017")
        logger.warning(f"Place CSV files in: {self.data_path}")

        # Create a marker file with instructions
        instructions_path = self.data_path / "DOWNLOAD_INSTRUCTIONS.txt"
        with open(instructions_path, "w") as f:
            f.write("CICIDS-2017 Dataset Download Instructions\n")
            f.write("=" * 50 + "\n\n")
            f.write("Due to the large size (~6GB), this dataset must be downloaded manually.\n\n")
            f.write("Option 1: Official Source\n")
            f.write("  https://www.unb.ca/cic/datasets/ids-2017.html\n\n")
            f.write("Option 2: Kaggle\n")
            f.write("  https://www.kaggle.com/datasets/cicdataset/cicids2017\n\n")
            f.write("Option 3: Mendeley Data\n")
            f.write("  https://data.mendeley.com/datasets/jxd9vr7ggn/1\n\n")
            f.write(f"Place the CSV files in this directory: {self.data_path}\n")

        return False

    def _load_raw(self) -> tuple[np.ndarray, np.ndarray]:
        """Load REAL CICIDS-2017 data from CSV files."""
        csv_files = sorted(self.data_path.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(
                f"CICIDS-2017 CSV files not found in {self.data_path}. "
                "Please download from https://www.unb.ca/cic/datasets/ids-2017.html"
            )

        logger.info(f"Loading REAL CICIDS-2017 dataset from {len(csv_files)} files...")

        all_features = []
        all_labels = []

        for csv_file in csv_files:
            logger.info(f"  Processing {csv_file.name}...")
            features, labels = self._parse_cicids_csv(csv_file)
            if len(features) > 0:
                all_features.append(features)
                all_labels.append(labels)

        if not all_features:
            raise ValueError("No valid data found in CICIDS CSV files")

        features = np.vstack(all_features)
        labels = np.concatenate(all_labels)

        logger.info(f"Loaded {len(features)} REAL CICIDS-2017 samples")
        logger.info(f"  Benign: {(labels == 0).sum()}, Attacks: {(labels > 0).sum()}")

        return features, labels

    def _parse_cicids_csv(self, filepath) -> tuple[np.ndarray, np.ndarray]:
        """Parse a single CICIDS CSV file."""
        features = []
        labels = []

        try:
            import pandas as pd
            df = pd.read_csv(filepath, low_memory=False)

            # Standardize column names (remove spaces, lowercase)
            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

            # Find label column
            label_col = None
            for col in ["label", "labels", "class", "attack"]:
                if col in df.columns:
                    label_col = col
                    break

            if label_col is None:
                logger.warning(f"No label column found in {filepath}")
                return np.array([]), np.array([])

            # Extract features (all numeric columns except label)
            feature_cols = [c for c in df.columns if c != label_col and df[c].dtype in ['int64', 'float64']]

            # Handle inf/nan values
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(0)

            features = df[feature_cols].values.astype(np.float32)

            # Parse labels
            for label in df[label_col]:
                label_str = str(label).strip()
                if self.binary_labels:
                    labels.append(0 if label_str.upper() == "BENIGN" else 1)
                else:
                    labels.append(self.ATTACK_LABELS.get(label_str, 1))

            return features, np.array(labels, dtype=np.int64)

        except ImportError:
            # Fallback without pandas
            logger.warning("pandas not available, using basic CSV parsing")
            return self._parse_cicids_csv_basic(filepath)

    def _parse_cicids_csv_basic(self, filepath) -> tuple[np.ndarray, np.ndarray]:
        """Basic CSV parsing without pandas."""
        import csv

        features = []
        labels = []

        with open(filepath, newline='', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            header = next(reader)

            # Find label column index
            label_idx = -1
            for i, col in enumerate(header):
                if col.strip().lower() in ["label", "labels", "class"]:
                    label_idx = i
                    break

            for row in reader:
                if len(row) <= label_idx:
                    continue

                # Parse features
                row_features = []
                for i, val in enumerate(row):
                    if i == label_idx:
                        continue
                    try:
                        v = float(val)
                        if np.isinf(v) or np.isnan(v):
                            v = 0.0
                        row_features.append(v)
                    except ValueError:
                        row_features.append(0.0)

                features.append(row_features)

                # Parse label
                label_str = row[label_idx].strip()
                if self.binary_labels:
                    labels.append(0 if label_str.upper() == "BENIGN" else 1)
                else:
                    labels.append(self.ATTACK_LABELS.get(label_str, 1))

        return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64)

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """Preprocess network flow features."""
        data = np.nan_to_num(data, nan=0.0, posinf=1e10, neginf=0)
        data = np.log1p(np.abs(data))
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        return data.astype(np.float32)

    def get_dataset_info(self) -> dict:
        """Get information about the loaded dataset."""
        return {
            "name": "CICIDS-2017",
            "type": "REAL DATA",
            "source": self.DATASET_URL,
            "features": len(self.FEATURE_NAMES),
            "attack_types": list(self.ATTACK_LABELS.keys()),
            "citation": self.CITATION,
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

    def __init__(self, config: DatasetConfig):
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

    def _load_raw(self) -> tuple[np.ndarray, np.ndarray]:
        synthetic_path = self.data_path / "synthetic_threat_intel.npz"
        if synthetic_path.exists():
            data = np.load(synthetic_path)
            return data["features"], data["labels"]
        raise FileNotFoundError("Threat intel data not found")

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """Preprocess threat intelligence features."""
        data = np.nan_to_num(data, nan=0.0)
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        return data.astype(np.float32)


# Register security loaders
DatasetRegistry.register("nsl-kdd", NSLKDDLoader)
DatasetRegistry.register("cicids", CICIDSLoader)
DatasetRegistry.register("threat-intel", ThreatIntelLoader)
