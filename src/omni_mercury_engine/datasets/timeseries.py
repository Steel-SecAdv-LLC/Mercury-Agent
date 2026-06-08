# Copyright (C) 2025 Steel Security Advisors LLC
"""Time-Series Anomaly Detection Benchmark Loaders."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import DatasetConfig, DatasetLoader, DatasetRegistry, safe_urlretrieve
from .exceptions import DataSourceUnavailableError

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class NABLoader(DatasetLoader):
    """Numenta Anomaly Benchmark (NAB) Dataset Loader.

    Downloads REAL time-series data from the official NAB repository.
    NAB is the standard benchmark for streaming anomaly detection.

    Data Source: https://github.com/numenta/NAB
    Paper: Lavin & Ahmad, "Evaluating Real-time Anomaly Detection Algorithms",
           IEEE ICMLA 2015.

    Categories:
    - artificialNoAnomaly: Synthetic baseline (no anomalies)
    - artificialWithAnomaly: Synthetic with known anomalies
    - realAWSCloudwatch: AWS CloudWatch metrics
    - realAdExchange: Online advertising metrics
    - realKnownCause: Real data with documented anomaly causes
    - realTraffic: Traffic data
    - realTweets: Twitter volume data
    """

    DATASET_NAME = "nab"
    DATASET_URL = "https://github.com/numenta/NAB"
    LICENSE = "AGPL-3.0"
    CITATION = """Lavin A, Ahmad S. Evaluating Real-time Anomaly Detection Algorithms -
    the Numenta Anomaly Benchmark. IEEE ICMLA 2015."""
    REQUIRES_CREDENTIALS = False

    # GitHub raw URLs for NAB data
    NAB_DATA_URL = "https://raw.githubusercontent.com/numenta/NAB/master/data/"
    NAB_LABELS_URL = (
        "https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_windows.json"
    )

    # NAB data categories and files
    NAB_FILES = {
        "realAWSCloudwatch": [
            "ec2_cpu_utilization_24ae8d.csv",
            "ec2_cpu_utilization_53ea38.csv",
            "ec2_cpu_utilization_5f5533.csv",
            "ec2_cpu_utilization_77c1ca.csv",
            "ec2_cpu_utilization_825cc2.csv",
            "ec2_cpu_utilization_ac20cd.csv",
            "ec2_cpu_utilization_c6585a.csv",
            "ec2_cpu_utilization_fe7f93.csv",
            "ec2_disk_write_bytes_1ef3de.csv",
            "ec2_disk_write_bytes_c0d644.csv",
            "ec2_network_in_257a54.csv",
            "ec2_network_in_5abac7.csv",
            "elb_request_count_8c0756.csv",
            "grok_asg_anomaly.csv",
            "iio_us-east-1_i-a2eb1cd9_NetworkIn.csv",
            "rds_cpu_utilization_cc0c53.csv",
            "rds_cpu_utilization_e47b3b.csv",
        ],
        "realKnownCause": [
            "ambient_temperature_system_failure.csv",
            "cpu_utilization_asg_misconfiguration.csv",
            "ec2_request_latency_system_failure.csv",
            "machine_temperature_system_failure.csv",
            "nyc_taxi.csv",
            "rogue_agent_key_hold.csv",
            "rogue_agent_key_updown.csv",
        ],
        "realTraffic": [
            "TravelTime_387.csv",
            "TravelTime_451.csv",
            "occupancy_6005.csv",
            "occupancy_t4013.csv",
            "speed_6005.csv",
            "speed_7578.csv",
            "speed_t4013.csv",
        ],
    }

    FEATURE_NAMES = ["value", "timestamp_hour", "timestamp_day", "timestamp_month"]

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize the instance."""
        super().__init__(config)
        self.categories = config.preprocessing.get(
            "categories", ["realAWSCloudwatch", "realKnownCause"]
        )

    def download(self) -> bool:
        """Download REAL NAB data from GitHub."""
        import requests

        logger.info("Downloading REAL NAB (Numenta Anomaly Benchmark) data...")

        # Download labels first
        labels_path = self.data_path / "labels.json"
        try:
            logger.info("  Downloading anomaly labels...")
            safe_urlretrieve(self.NAB_LABELS_URL, labels_path)
        except (requests.RequestException, ValueError) as e:
            logger.error(f"Failed to download NAB labels: {e}")
            return False

        # Download data files
        downloaded_count = 0
        for category in self.categories:
            if category not in self.NAB_FILES:
                continue

            category_path = self.data_path / category
            category_path.mkdir(exist_ok=True)

            for filename in self.NAB_FILES[category]:
                file_path = category_path / filename
                if file_path.exists():
                    downloaded_count += 1
                    continue

                url = f"{self.NAB_DATA_URL}{category}/{filename}"
                try:
                    safe_urlretrieve(url, file_path)
                    downloaded_count += 1
                except (requests.RequestException, ValueError) as e:
                    logger.warning(f"  Failed to download {filename}: {e}")

        logger.info(f"Downloaded {downloaded_count} NAB data files")
        return downloaded_count > 0

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load REAL NAB data from downloaded files."""
        # Load labels
        labels_path = self.data_path / "labels.json"
        if not labels_path.exists():
            raise FileNotFoundError(
                "NAB labels not found. Run with download=True to fetch real data."
            )

        with open(labels_path) as f:
            anomaly_windows = json.load(f)

        logger.info("Loading REAL NAB benchmark data...")

        all_features = []
        all_labels = []

        for category in self.categories:
            category_path = self.data_path / category
            if not category_path.exists():
                continue

            for csv_file in category_path.glob("*.csv"):
                features, labels = self._parse_nab_file(csv_file, anomaly_windows)
                all_features.extend(features)
                all_labels.extend(labels)

        features = np.array(all_features, dtype=np.float32)  # type: ignore[assignment, unused-ignore]
        labels = np.array(all_labels, dtype=np.int64)  # type: ignore[assignment, unused-ignore]

        logger.info(f"Loaded {len(features)} NAB samples")
        logger.info(f"  Anomalies: {labels.sum()}")  # type: ignore[attr-defined, unused-ignore]

        return features, labels  # type: ignore[return-value, unused-ignore]

    def _parse_nab_file(
        self, filepath: Path, anomaly_windows: dict[str, Any]
    ) -> tuple[list[list[float]], list[int]]:
        """Parse a single NAB CSV file."""
        import csv
        from datetime import datetime

        features = []
        labels = []

        # Get anomaly windows for this file.
        # NAB combined_windows.json keys use forward-slash paths like
        # "realKnownCause/ambient_temperature_system_failure.csv"
        rel_path = f"{filepath.parent.name}/{filepath.name}"
        windows = anomaly_windows.get(rel_path, [])

        if not windows:
            logger.debug(f"  No anomaly windows for {rel_path} (key not in labels JSON)")

        # Parse windows into datetime ranges
        anomaly_ranges = []
        for window in windows:
            try:
                start = datetime.fromisoformat(window[0].replace("Z", "+00:00"))
                end = datetime.fromisoformat(window[1].replace("Z", "+00:00"))
                anomaly_ranges.append((start, end))
            except (ValueError, IndexError):
                # Invalid window format; skip this anomaly window
                pass

        with open(filepath, newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    # Parse timestamp
                    timestamp_str = row.get("timestamp", "")
                    try:
                        ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                        hour = ts.hour
                        day = ts.day
                        month = ts.month
                    except ValueError:
                        hour, day, month = 0, 1, 1

                    # Parse value
                    value = float(row.get("value", 0))

                    feature_row = [value, hour, day, month]
                    features.append(feature_row)

                    # Check if in anomaly window
                    is_anomaly = False
                    try:
                        ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                        for start, end in anomaly_ranges:
                            if start <= ts <= end:
                                is_anomaly = True
                                break
                    except ValueError:
                        # Invalid timestamp format; cannot check anomaly window
                        pass

                    labels.append(1 if is_anomaly else 0)

                except (ValueError, KeyError):
                    continue

        return features, labels

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess NAB features."""
        data = np.nan_to_num(data, nan=0.0)
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        return data.astype(np.float32)

    def get_dataset_info(self) -> dict[str, Any]:
        """Get information about the loaded dataset."""
        return {
            "name": "Numenta Anomaly Benchmark (NAB)",
            "type": "REAL DATA",
            "source": self.DATASET_URL,
            "features": len(self.FEATURE_NAMES),
            "categories": self.categories,
            "citation": self.CITATION,
        }


class SMDLoader(DatasetLoader):
    """Server Machine Dataset (SMD) Loader.

    Downloads REAL server monitoring metrics from a large internet company.
    28 server machines, each with 38 metrics, collected over 5 weeks.

    Data Source: https://github.com/NetManAIOps/OmniAnomaly
    Paper: Su et al., "Robust Anomaly Detection for Multivariate Time Series
           through Stochastic Recurrent Neural Network", KDD 2019.

    This is the benchmark dataset for OmniAnomaly, MSCRED, and many other methods.
    """

    DATASET_NAME = "smd"
    DATASET_URL = "https://github.com/NetManAIOps/OmniAnomaly"
    LICENSE = "MIT"
    CITATION = """Su Y, Zhao Y, Niu C, et al. Robust Anomaly Detection for Multivariate
    Time Series through Stochastic Recurrent Neural Network. KDD 2019."""
    REQUIRES_CREDENTIALS = False

    # GitHub raw URLs for SMD data
    SMD_BASE_URL = (
        "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset/"
    )

    # SMD has 28 machines, each with train/test splits
    MACHINES = (
        [f"machine-1-{i}" for i in range(1, 9)]
        + [f"machine-2-{i}" for i in range(1, 10)]
        + [f"machine-3-{i}" for i in range(1, 12)]
    )

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize the instance."""
        super().__init__(config)
        self.machines = config.preprocessing.get("machines", self.MACHINES[:5])

    def download(self) -> bool:
        """Download REAL SMD data from GitHub."""
        import requests

        logger.info("Downloading REAL SMD (Server Machine Dataset)...")

        downloaded_count = 0
        for machine in self.machines:
            machine_dir = self.data_path / machine
            machine_dir.mkdir(exist_ok=True)

            for split in ["train", "test", "test_label"]:
                file_path = machine_dir / f"{split}.npy"
                if file_path.exists():
                    downloaded_count += 1
                    continue

                # SMD stores data as .txt files on GitHub
                url = f"{self.SMD_BASE_URL}{split}/{machine}.txt"
                txt_path = machine_dir / f"{split}.txt"

                try:
                    safe_urlretrieve(url, txt_path)
                    # Convert txt to npy
                    data = np.loadtxt(txt_path, delimiter=",")
                    np.save(file_path, data)
                    downloaded_count += 1
                    logger.info(f"  Downloaded {machine}/{split}")
                except (requests.RequestException, ValueError) as e:
                    logger.warning(f"  Failed to download {machine}/{split}: {e}")
                except Exception as e:
                    logger.warning(f"  Failed to parse {machine}/{split}: {e}")

        logger.info(f"Downloaded {downloaded_count} SMD files")
        return downloaded_count > 0

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load REAL SMD data from downloaded files."""
        logger.info("Loading REAL SMD benchmark data...")

        all_features = []
        all_labels = []

        for machine in self.machines:
            machine_dir = self.data_path / machine

            # Try loading test data and labels
            test_path = machine_dir / "test.npy"
            label_path = machine_dir / "test_label.npy"

            if not test_path.exists():
                # Try loading from txt
                txt_path = machine_dir / "test.txt"
                if txt_path.exists():
                    data = np.loadtxt(txt_path, delimiter=",")
                    np.save(test_path, data)
                else:
                    logger.warning(f"  No data for {machine}")
                    continue

            features = np.load(test_path)
            all_features.append(features)

            if label_path.exists():
                labels = np.load(label_path).ravel()
            else:
                # Try loading from txt (test_label files are one value per line)
                txt_path = machine_dir / "test_label.txt"
                if txt_path.exists():
                    labels = np.loadtxt(txt_path).ravel()
                    # Cache as npy for next load
                    np.save(label_path, labels)
                else:
                    logger.warning(f"  No test_label found for {machine} — labels will be all-zero")
                    labels = np.zeros(len(features))

            # Ensure binary labels
            labels = (labels > 0).astype(np.int64)
            all_labels.append(labels)

        if not all_features:
            raise FileNotFoundError(
                "SMD data not found. Run with download=True to fetch real data."
            )

        features = np.vstack(all_features)
        labels = np.concatenate(all_labels).astype(np.int64)

        logger.info(f"Loaded {len(features)} SMD samples from {len(self.machines)} machines")
        logger.info(f"  Features: {features.shape[1]}, Anomalies: {labels.sum()}")

        return features, labels

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess SMD features."""
        data = np.nan_to_num(data, nan=0.0)
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        return data.astype(np.float32)

    def get_dataset_info(self) -> dict[str, Any]:
        """Get information about the loaded dataset."""
        return {
            "name": "Server Machine Dataset (SMD)",
            "type": "REAL DATA",
            "source": self.DATASET_URL,
            "machines": self.machines,
            "citation": self.CITATION,
        }


class SMAPMSLLoader(DatasetLoader):
    """NASA SMAP and MSL Spacecraft Telemetry Dataset Loader.

    Downloads REAL spacecraft telemetry data from NASA missions:
    - SMAP: Soil Moisture Active Passive satellite
    - MSL: Mars Science Laboratory (Curiosity rover)

    Data Source: https://github.com/khundman/telemanom
    Paper: Hundman et al., "Detecting Spacecraft Anomalies Using LSTMs and
           Nonparametric Dynamic Thresholding", KDD 2018.

    This dataset is used to benchmark anomaly detection on real spacecraft data.
    """

    DATASET_NAME = "smap_msl"
    DATASET_URL = "https://github.com/khundman/telemanom"
    LICENSE = "Apache-2.0"
    CITATION = """Hundman K, Constantinou V, Laporte C, Colwell I, Soderstrom T.
    Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding.
    KDD 2018."""
    REQUIRES_CREDENTIALS = False

    # Data URLs — multiple mirrors tried in order.
    # S3 (s3-us-west-2.amazonaws.com/telemanom/data.zip) returns 403.
    # OmniAnomaly GitHub mirror is the primary; khundman repo is the canonical fallback.
    OMNIANOMALY_BASE_URL = "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/data/"
    TELEMANOM_BASE_URL = "https://raw.githubusercontent.com/khundman/telemanom/master/data/"
    LABELED_ANOMALIES_URL = (
        "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv"
    )
    # Zenodo TimeEval archive — full SMAP/MSL preprocessed dataset (DOI: 10.5281/zenodo.5899270).
    # Operator can download and extract train/ test/ subdirs manually; see error message below.
    ZENODO_INSTRUCTIONS_URL = "https://zenodo.org/record/5899270"

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize the instance."""
        super().__init__(config)
        self.dataset = config.preprocessing.get("dataset", "SMAP")  # SMAP or MSL

    def download(self) -> bool:
        """Download REAL SMAP/MSL data.

        The original S3 URL (s3-us-west-2.amazonaws.com/telemanom/data.zip) returns 403.
        Labels are fetched from the GitHub repo. For the actual telemetry data, users must
        download from TimeEval HiDrive mirror or the GitHub repo's preprocessed files.

        Returns:
            True if data is available, False otherwise.

        Raises:
            DataSourceUnavailableError: If data cannot be obtained.
        """
        import requests

        logger.info(f"Preparing NASA {self.dataset} spacecraft telemetry...")

        # Download labeled anomalies from GitHub
        labels_path = self.data_path / "labeled_anomalies.csv"
        if not labels_path.exists():
            try:
                safe_urlretrieve(self.LABELED_ANOMALIES_URL, labels_path)
                logger.info("  Downloaded anomaly labels from GitHub")
            except (requests.RequestException, ValueError) as e:
                logger.error(f"  Failed to download labels: {e}")
                raise DataSourceUnavailableError(
                    loader_name=f"SMAP/MSL ({self.dataset})",
                    source_url=self.LABELED_ANOMALIES_URL,
                    reason=f"Failed to download anomaly labels: {e}",
                ) from e

        # Check if preprocessed data already exists
        train_dir = self.data_path / "train"
        test_dir = self.data_path / "test"

        if train_dir.exists() and test_dir.exists():
            n_train = len(list(train_dir.glob("*.npy")))
            n_test = len(list(test_dir.glob("*.npy")))
            if n_test > 0:
                logger.info(f"  Found {n_train} train, {n_test} test files")
                return True

        # Attempt download from OmniAnomaly GitHub mirror
        import csv

        labels_path = self.data_path / "labeled_anomalies.csv"
        if labels_path.exists():
            with open(labels_path) as f:
                reader = csv.DictReader(f)
                channels = [
                    row["chan_id"] for row in reader if row.get("spacecraft") == self.dataset
                ]

            if channels:
                train_dir.mkdir(parents=True, exist_ok=True)
                test_dir.mkdir(parents=True, exist_ok=True)
                downloaded = 0
                for chan in channels:
                    for split, sdir in [("train", train_dir), ("test", test_dir)]:
                        npy_path = sdir / f"{chan}.npy"
                        if npy_path.exists():
                            downloaded += 1
                            continue
                        # Try OmniAnomaly mirror first, then canonical telemanom repo.
                        mirrors = [
                            f"{self.OMNIANOMALY_BASE_URL}{split}/{chan}.npy",
                            f"{self.TELEMANOM_BASE_URL}{split}/{chan}.npy",
                        ]
                        for mirror_url in mirrors:
                            try:
                                safe_urlretrieve(mirror_url, npy_path)
                                downloaded += 1
                                break
                            except Exception:
                                logger.debug("Channel %s mirror %s failed", chan, mirror_url)
                if downloaded > 0:
                    logger.info(f"  Downloaded {downloaded} channel files from GitHub mirrors")
                    return True

        raise DataSourceUnavailableError(
            loader_name=f"SMAP/MSL ({self.dataset})",
            source_url=self.OMNIANOMALY_BASE_URL,
            reason=(
                "Could not download telemetry data from GitHub mirrors "
                "(OmniAnomaly + khundman/telemanom). "
                f"Manual install: download from Zenodo TimeEval archive "
                f"({self.ZENODO_INSTRUCTIONS_URL}) "
                f"and extract train/ and test/ directories to: {self.data_path}"
            ),
        )

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load REAL SMAP/MSL data from downloaded files."""
        test_dir = self.data_path / "test"
        labels_path = self.data_path / "labeled_anomalies.csv"

        if not test_dir.exists():
            raise FileNotFoundError(
                f"SMAP/MSL data not found in {self.data_path}. "
                "Download from HiDrive mirror: https://my.hidrive.com/share/ma4p8w4qqb "
                "and extract train/ and test/ directories."
            )

        logger.info(f"Loading REAL NASA {self.dataset} telemetry data...")

        # Load labeled anomalies
        anomaly_info = {}
        if labels_path.exists():
            import csv

            with open(labels_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("spacecraft") == self.dataset:
                        chan_id = row.get("chan_id", "")
                        anomaly_seqs = row.get("anomaly_sequences", "[]")
                        try:
                            import json as _json

                            parsed = _json.loads(anomaly_seqs)
                            # Ensure the parsed value is a list of sequences
                            if not isinstance(parsed, list):
                                parsed = []
                            anomaly_info[chan_id] = parsed
                        except (ValueError, TypeError):
                            anomaly_info[chan_id] = []

        all_features = []
        all_labels = []

        # Load test data files
        for npy_file in sorted(test_dir.glob("*.npy")):
            chan_id = npy_file.stem

            # Filter by dataset type
            if self.dataset == "SMAP" and not chan_id.startswith(
                ("A-", "B-", "C-", "D-", "E-", "F-", "G-", "P-", "R-", "S-", "T-")
            ):
                continue
            if self.dataset == "MSL" and not chan_id.startswith(
                ("M-", "C-", "D-", "F-", "P-", "T-")
            ):
                # MSL channels have different prefix pattern
                pass

            try:
                data = np.load(npy_file)
                all_features.append(data)

                # Create labels based on anomaly info
                labels = np.zeros(len(data), dtype=np.int64)
                for start, end in anomaly_info.get(chan_id, []):
                    labels[start:end] = 1
                all_labels.append(labels)

            except Exception as e:
                logger.warning(f"  Failed to load {npy_file}: {e}")

        if not all_features:
            raise FileNotFoundError(
                f"No {self.dataset} data files found. Check download instructions."
            )

        # Pad to same length and stack
        padded_features = []
        padded_labels = []

        for feat, label in zip(all_features, all_labels, strict=False):
            if len(feat.shape) == 1:
                feat = feat.reshape(-1, 1)
            # Use sliding window approach instead of padding
            padded_features.append(feat)
            padded_labels.append(label)

        # Concatenate all channels
        features = np.vstack(padded_features)
        labels = np.empty(sum(len(label) for label in padded_labels), dtype=np.int64)
        labels = np.concatenate(padded_labels, out=labels)

        logger.info(f"Loaded {len(features)} {self.dataset} telemetry samples")
        logger.info(f"  Features: {features.shape[1]}, Anomalies: {labels.sum()}")

        return features, labels

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess spacecraft telemetry."""
        data = np.nan_to_num(data, nan=0.0)
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        return data.astype(np.float32)

    def get_dataset_info(self) -> dict[str, Any]:
        """Get information about the loaded dataset."""
        return {
            "name": f"NASA {self.dataset} Spacecraft Telemetry",
            "type": "REAL DATA (Spacecraft)",
            "source": self.DATASET_URL,
            "dataset": self.dataset,
            "citation": self.CITATION,
        }


# Register time-series loaders
DatasetRegistry.register("nab", NABLoader)
DatasetRegistry.register("smd", SMDLoader)
DatasetRegistry.register("smap", SMAPMSLLoader)
DatasetRegistry.register("msl", SMAPMSLLoader)
