# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Time-Series Anomaly Detection Benchmark Loaders.

These loaders fetch REAL benchmark datasets used in academic research:
- NAB (Numenta Anomaly Benchmark): Standard time-series anomaly benchmark
- SMD (Server Machine Dataset): Real server metrics from large internet company
- SMAP/MSL: NASA spacecraft telemetry from Mars missions

All datasets download from official sources or mirrors.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .base import DatasetConfig, DatasetLoader, DatasetRegistry, safe_urlretrieve
from .exceptions import DataSourceUnavailableError

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
        # The remaining two *real* NAB categories (file inventory matches the
        # upstream labels/combined_windows.json keys). Additive only: neither
        # is in the default ``categories`` selection, so existing callers are
        # byte-for-byte unaffected; benchmarks/nab_competitive.py opts in to
        # cover all five real categories.
        "realAdExchange": [
            "exchange-2_cpc_results.csv",
            "exchange-2_cpm_results.csv",
            "exchange-3_cpc_results.csv",
            "exchange-3_cpm_results.csv",
            "exchange-4_cpc_results.csv",
            "exchange-4_cpm_results.csv",
        ],
        "realTweets": [
            "Twitter_volume_AAPL.csv",
            "Twitter_volume_AMZN.csv",
            "Twitter_volume_CRM.csv",
            "Twitter_volume_CVS.csv",
            "Twitter_volume_FB.csv",
            "Twitter_volume_GOOG.csv",
            "Twitter_volume_IBM.csv",
            "Twitter_volume_KO.csv",
            "Twitter_volume_PFE.csv",
            "Twitter_volume_UPS.csv",
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

    def iter_series(self) -> list[tuple[str, np.ndarray[Any, Any], np.ndarray[Any, Any]]]:
        """Yield per-file 1-D value series with per-point labels, in temporal order.

        Unlike :meth:`load` / :meth:`_load_raw` (which pool every file and then
        shuffle for tabular anomaly detection), this preserves each file's native
        temporal ordering and returns only the scalar ``value`` channel -- the
        per-point streaming ``(series, labels)`` a 1-D anomaly detector consumes.
        Downloads on first use (honours the configured categories, which exclude
        NAB's synthetic ``artificial*`` sets by default).

        Returns:
            A list of ``(relative_name, values, labels)`` triples, one per real
            NAB CSV in the configured categories: ``values`` a float64 1-D array
            in temporal order and ``labels`` an int64 0/1 array of the same
            length (1 where a point's timestamp falls inside a documented anomaly
            window). Empty files are skipped.

        Raises:
            DataSourceUnavailableError: NAB labels/data could not be obtained and
                downloading is disabled or failed.
        """
        labels_path = self.data_path / "labels.json"
        if not labels_path.exists() and not (self.config.download and self.download()):
            raise DataSourceUnavailableError(
                loader_name="NAB",
                source_url=self.NAB_LABELS_URL,
                reason="NAB labels/data unavailable and download disabled or failed.",
            )

        with open(labels_path) as handle:
            anomaly_windows = json.load(handle)

        series_list: list[tuple[str, np.ndarray[Any, Any], np.ndarray[Any, Any]]] = []
        for category in self.categories:
            category_path = self.data_path / category
            if not category_path.exists():
                continue
            for csv_file in sorted(category_path.glob("*.csv")):
                features, labels = self._parse_nab_file(csv_file, anomaly_windows)
                if not features:
                    continue
                values = np.asarray([row[0] for row in features], dtype=np.float64)
                targets = np.asarray(labels, dtype=np.int64)
                series_list.append((f"{category}/{csv_file.name}", values, targets))
        return series_list


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


class DSADSLoader(DatasetLoader):
    """Daily and Sports Activities Dataset (DSADS) — UCI 256.

    Downloads the REAL multivariate inertial-sensor recordings from the UCI
    Machine Learning Repository (dataset 256) and turns them into a tabular
    anomaly-detection task. Real data only — fails loud if the archive is
    unreachable; never substitutes synthetic.

    Data source:
        https://archive.ics.uci.edu/ml/machine-learning-databases/00256/data.zip
    Paper:
        Barshan B, Yüksek M C. "Recognizing Daily and Sports Activities in Two
        Open Source Machine Learning Environments Using Body-Worn Sensor Units."
        The Computer Journal 57(11), 2014.

    Archive layout: ``data/a{01..19}/p{1..8}/s{01..60}.txt`` — 19 activities ×
    8 subjects × 60 five-second segments = 9120 segments, each a 125 × 45 matrix
    (5 s @ 25 Hz across 45 channels: 9 axes × 5 body units).

    Representation: each 125 × 45 segment is reduced to a fixed **405-dim**
    feature vector — 9 per-channel statistics (mean, std, min, max, median,
    25th/75th percentile, peak-to-peak, RMS) across the 45 channels. Standard
    statistical HAR featurisation; fully deterministic, no randomness.

    Anomaly labels: DSADS has **no native anomaly labels** (it is a 19-class
    activity-recognition set), so this loader constructs a transparent,
    DOCUMENTED convention: segments of a designated minority of activities are
    labelled anomalous, the rest normal. The default singles out activity 19
    ("playing basketball") — the only ball sport and the most irregular,
    non-cyclic activity in the set — giving 480 / 9120 = 5.26% anomalies.
    Override with ``preprocessing={"anomaly_activities": [..1-based..]}``. The
    manufactured nature is surfaced in ``get_dataset_info`` (``label_source =
    "statistical"`` — the repo taxonomy's value for constructed/heuristic labels);
    only the *labels* are constructed — the sensor features are real and never
    fabricated.
    """

    DATASET_NAME = "dsads"
    DATASET_URL = "https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities"
    LICENSE = "CC BY 4.0 (UCI)"
    CITATION = (
        "Barshan B, Yüksek M C. Recognizing Daily and Sports Activities in Two Open "
        "Source Machine Learning Environments Using Body-Worn Sensor Units. "
        "The Computer Journal 57(11), 2014."
    )
    REQUIRES_CREDENTIALS = False
    # Manufactured anomaly labels (activity designation) — excluded from the
    # comparable headline; registered in datasets.label_provenance.
    LABEL_SOURCE = "statistical"

    DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00256/data.zip"
    N_ACTIVITIES = 19
    N_SUBJECTS = 8
    N_SEGMENTS = 60
    SEGMENT_ROWS = 125
    SEGMENT_COLS = 45
    _DEFAULT_ANOMALY_ACTIVITIES = (19,)

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize the instance, validating the anomaly-activity convention."""
        super().__init__(config)
        raw = config.preprocessing.get("anomaly_activities", self._DEFAULT_ANOMALY_ACTIVITIES)
        self.anomaly_activities = frozenset(int(a) for a in raw)
        if not self.anomaly_activities or not all(
            1 <= a <= self.N_ACTIVITIES for a in self.anomaly_activities
        ):
            raise ValueError(
                f"anomaly_activities must be a non-empty subset of 1..{self.N_ACTIVITIES}; "
                f"got {sorted(self.anomaly_activities)}"
            )

    @property
    def _zip_path(self) -> Path:
        return self.data_path / "dsads_data.zip"

    def download(self) -> bool:
        """Download the REAL DSADS archive from UCI (~170 MB), cached on disk."""
        if self._zip_path.exists():
            logger.info("DSADS archive already cached at %s", self._zip_path)
            return True
        logger.info("Downloading REAL DSADS (UCI 256, ~170 MB)...")
        safe_urlretrieve(self.DATA_URL, self._zip_path)
        return self._zip_path.exists()

    @staticmethod
    def _segment_features(seg: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Reduce a (125, 45) segment to 405 deterministic per-channel statistics."""
        seg = seg.astype(np.float64)
        return np.concatenate(
            [
                seg.mean(axis=0),
                seg.std(axis=0),
                seg.min(axis=0),
                seg.max(axis=0),
                np.median(seg, axis=0),
                np.percentile(seg, 25, axis=0),
                np.percentile(seg, 75, axis=0),
                np.ptp(seg, axis=0),
                np.sqrt((seg**2).mean(axis=0)),
            ]
        )

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Parse the cached archive into (X[9120, 405], y[9120]) — real data only."""
        import re
        import zipfile

        if not self._zip_path.exists():
            raise FileNotFoundError(
                f"DSADS archive not found at {self._zip_path}. Run with download=True."
            )

        seg_re = re.compile(r"(?:^|/)a(\d{2})/p(\d)/s(\d{2})\.txt$")
        feats: list[np.ndarray[Any, Any]] = []
        labels: list[int] = []
        expected = self.N_ACTIVITIES * self.N_SUBJECTS * self.N_SEGMENTS
        with zipfile.ZipFile(self._zip_path) as zf:
            members = sorted(n for n in zf.namelist() if seg_re.search(n))
            if len(members) != expected:
                raise ValueError(
                    f"DSADS archive layout unexpected: found {len(members)} segment "
                    f"files, expected {expected} (19 activities × 8 subjects × 60)."
                )
            for name in members:
                match = seg_re.search(name)
                assert match is not None  # guaranteed by the filter above
                activity = int(match.group(1))
                with zf.open(name) as handle:
                    seg = np.loadtxt(handle, delimiter=",")
                if seg.shape != (self.SEGMENT_ROWS, self.SEGMENT_COLS):
                    raise ValueError(
                        f"DSADS segment {name} has shape {seg.shape}, expected "
                        f"({self.SEGMENT_ROWS}, {self.SEGMENT_COLS})."
                    )
                feats.append(self._segment_features(seg))
                labels.append(1 if activity in self.anomaly_activities else 0)

        features = np.asarray(feats, dtype=np.float64)
        targets = np.asarray(labels, dtype=np.int64)
        logger.info(
            "Loaded REAL DSADS: %d segments × %d features, %d anomalies (%.2f%%); "
            "anomaly activities=%s",
            features.shape[0],
            features.shape[1],
            int(targets.sum()),
            100.0 * float(targets.mean()),
            sorted(self.anomaly_activities),
        )
        return features, targets

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Z-score normalise the per-channel statistic features."""
        data = np.nan_to_num(data, nan=0.0)
        return ((data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)).astype(np.float32)

    def get_dataset_info(self) -> dict[str, Any]:
        """Return dataset metadata, flagging the constructed anomaly labels."""
        return {
            "name": "Daily and Sports Activities (DSADS, UCI 256)",
            "type": "REAL DATA",
            "source": self.DATASET_URL,
            "label_source": "statistical",
            "anomaly_activities": sorted(self.anomaly_activities),
            "citation": self.CITATION,
        }


class EpilepsyLoader(DatasetLoader):
    """Bonn single-electrode EEG (Andrzejak et al. 2001) — Epileptic Seizure set.

    Reconstructs the canonical **11500 × 178** tabular anomaly-detection task from
    the OFFICIAL raw Bonn EEG time series (five sets A–E), hosted by the NTSA
    group at Universitat Pompeu Fabra: https://www.upf.edu/web/ntsa/downloads/.

    Official source only — no third-party mirrors, no simulated "mimic" data. The
    UCI tabular version (11500 × 178) was removed, and the UPF page sits behind a
    Cloudflare JS challenge that blocks automated download from the SSRF-safe
    client. So the data is supplied via a **local path** to the manually-
    downloaded official sets; absent that, the loader fails loud (it never
    fabricates and never pulls an unvetted mirror).

    Provide ``preprocessing={"bonn_dir": "<dir>"}`` where ``<dir>`` holds the five
    official set archives (``Z.zip O.zip N.zip F.zip S.zip``) or five extracted
    set directories (``Z/ O/ N/ F/ S/``), each containing 100 text files of 4097
    single-channel EEG samples (one value per line) — the published Bonn format.

    Reconstruction (the standard tabular derivation): every 4097-sample recording
    is chunked into 23 non-overlapping 178-sample segments (4094 used; the final
    3 dropped), so 100 × 23 = 2300 rows per set × 5 sets = **11500 × 178**.
    Labels: set **S** (set E in the paper; ictal/seizure activity) = anomaly (1),
    the four non-ictal sets (Z, O, N, F) = normal (0) → 2300/11500 = **0.20**.

    Citation (required by the data authors): Andrzejak RG, Lehnertz K, Mormann F,
    Rieke C, David P, Elger CE. "Indications of nonlinear deterministic and
    finite-dimensional structures in time series of brain electrical activity:
    Dependence on recording region and brain state." Phys. Rev. E 64, 061907
    (2001).
    """

    DATASET_NAME = "epilepsy"
    DATASET_URL = "https://www.upf.edu/web/ntsa/downloads/"
    LICENSE = "Free for research use with citation (Andrzejak et al. 2001)"
    CITATION = (
        "Andrzejak RG, Lehnertz K, Mormann F, Rieke C, David P, Elger CE. "
        "Indications of nonlinear deterministic and finite-dimensional structures "
        "in time series of brain electrical activity. Phys. Rev. E 64, 061907 (2001)."
    )
    REQUIRES_CREDENTIALS = False
    # Genuine ictal/seizure brain-state labels (set E) — registered ground_truth.
    LABEL_SOURCE = "ground_truth"

    SETS = ("Z", "O", "N", "F", "S")
    _SEIZURE_SET = "S"  # set E in the paper (ictal/seizure)
    FILES_PER_SET = 100
    SAMPLES_PER_FILE = 4097
    SEGMENT_LEN = 178
    SEGMENTS_PER_FILE = 23  # 23 * 178 = 4094 (final 3 samples dropped)

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize the instance; ``preprocessing['bonn_dir']`` supplies the data."""
        super().__init__(config)
        bonn_dir = config.preprocessing.get("bonn_dir")
        self.bonn_dir: Path | None = Path(bonn_dir) if bonn_dir else None

    def download(self) -> bool:
        """Validate the user-provided official data; fail loud if absent.

        The official UPF source is Cloudflare-gated, so there is no automated
        fetch — the supported path is a manually-downloaded local copy.
        """
        if self.bonn_dir is not None and self.bonn_dir.exists():
            return True
        raise DataSourceUnavailableError(
            loader_name="Epilepsy (Bonn EEG / UPF)",
            source_url=self.DATASET_URL,
            reason=(
                "Official Bonn EEG data (Andrzejak et al. 2001) is hosted at "
                f"{self.DATASET_URL} behind a Cloudflare challenge that blocks automated "
                "download. Download the five sets (A–E) from there and pass "
                "preprocessing={'bonn_dir': '<dir with Z/O/N/F/S .zip or folders>'}. "
                "Third-party mirrors and 'mimic' datasets are deliberately not used."
            ),
        )

    def _read_set(self, set_name: str) -> list[np.ndarray[Any, Any]]:
        """Read one set's 100 recordings (each a 1-D array) from a .zip or a dir."""
        import zipfile

        assert self.bonn_dir is not None  # download() guarantees this
        zip_path = self.bonn_dir / f"{set_name}.zip"
        dir_path = self.bonn_dir / set_name
        recordings: list[np.ndarray[Any, Any]] = []
        if zip_path.exists():
            with zipfile.ZipFile(zip_path) as zf:
                for name in sorted(n for n in zf.namelist() if not n.endswith("/")):
                    with zf.open(name) as handle:
                        recordings.append(np.loadtxt(handle))
        elif dir_path.is_dir():
            for path in sorted(p for p in dir_path.iterdir() if p.is_file()):
                recordings.append(np.loadtxt(path))
        else:
            raise FileNotFoundError(
                f"Epilepsy set '{set_name}': expected {zip_path} or {dir_path}/ "
                f"under bonn_dir={self.bonn_dir}."
            )
        return recordings

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Reconstruct (X[11500, 178], y[11500]) from the official sets."""
        if self.bonn_dir is None or not self.bonn_dir.exists():
            raise FileNotFoundError(
                "Epilepsy: no bonn_dir provided. See EpilepsyLoader docstring — "
                "supply the official Bonn EEG sets via preprocessing['bonn_dir']."
            )

        seg_per = self.SEGMENTS_PER_FILE
        usable = seg_per * self.SEGMENT_LEN
        feats: list[np.ndarray[Any, Any]] = []
        labels: list[int] = []
        for set_name in self.SETS:
            recordings = self._read_set(set_name)
            if len(recordings) != self.FILES_PER_SET:
                raise ValueError(
                    f"Epilepsy set '{set_name}': expected {self.FILES_PER_SET} recordings, "
                    f"got {len(recordings)} — not the official Bonn layout."
                )
            label = 1 if set_name == self._SEIZURE_SET else 0
            for rec in recordings:
                flat = np.asarray(rec, dtype=np.float64).ravel()
                if flat.shape[0] < usable:
                    raise ValueError(
                        f"Epilepsy set '{set_name}': a recording has {flat.shape[0]} samples, "
                        f"need >= {usable} (official files are {self.SAMPLES_PER_FILE})."
                    )
                feats.append(flat[:usable].reshape(seg_per, self.SEGMENT_LEN))
                labels.extend([label] * seg_per)

        features = np.vstack(feats)
        targets = np.asarray(labels, dtype=np.int64)
        expected_rows = len(self.SETS) * self.FILES_PER_SET * seg_per
        if features.shape != (expected_rows, self.SEGMENT_LEN):
            raise ValueError(
                f"Epilepsy reconstruction produced {features.shape}, expected "
                f"({expected_rows}, {self.SEGMENT_LEN})."
            )
        logger.info(
            "Reconstructed REAL Epilepsy (Bonn): %d × %d, %d seizure rows (%.1f%%)",
            features.shape[0],
            features.shape[1],
            int(targets.sum()),
            100.0 * float(targets.mean()),
        )
        return features, targets

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Z-score normalise the per-segment EEG amplitudes."""
        data = np.nan_to_num(data, nan=0.0)
        return ((data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)).astype(np.float32)

    def get_dataset_info(self) -> dict[str, Any]:
        """Return dataset metadata."""
        return {
            "name": "Epileptic Seizure (Bonn EEG, Andrzejak et al. 2001)",
            "type": "REAL DATA",
            "source": self.DATASET_URL,
            "label_source": "ground_truth",
            "citation": self.CITATION,
        }


# Register time-series loaders
DatasetRegistry.register("nab", NABLoader)
DatasetRegistry.register("smd", SMDLoader)
DatasetRegistry.register("smap", SMAPMSLLoader)
DatasetRegistry.register("msl", SMAPMSLLoader)
DatasetRegistry.register("dsads", DSADSLoader)
DatasetRegistry.register("epilepsy", EpilepsyLoader)
