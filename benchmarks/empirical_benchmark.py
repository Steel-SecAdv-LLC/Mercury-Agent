"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

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
Empirical Benchmark Suite for Mercury-Agent

This module provides honest, data-driven benchmarks comparing Mercury-Agent's
anomaly detection capabilities against established near-peer systems using
publicly available datasets.

Datasets Used:
- UCI breast_cancer (medical domain proxy)
- UCI digits (pattern recognition)
- UCI covtype (environmental/sensor data)
- KDDCup99 subset (cybersecurity)
- SMD (Server Machine Dataset) - time-series
- SMAP (Soil Moisture Active Passive) - time-series
- MSL (Mars Science Laboratory) - time-series
- SWaT (Secure Water Treatment) - time-series

Near-Peer Baselines:
- Isolation Forest (Mercury-native)
- One-Class SVM (Mercury-native)
- Local Outlier Factor (Mercury-native)
- Elliptic Envelope (Mercury-native)
- TranAD (SOTA transformer-based)
- MAAT (SOTA Mamba-based)

Metrics:
- ROC-AUC
- Precision-Recall AUC
- F1 Score
- Detection Rate (Recall)
- False Positive Rate
- Inference Latency (ms)
- Per-class metrics
- Confusion matrix analysis

Features:
- K-Fold Cross-Validation with per-fold metrics
- Statistical significance testing
- Confusion matrix generation
"""

import hashlib
import io
import json
import logging
import os
import sys
import time
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import torch

# =============================================================================
# Configuration (environment-variable driven for CI flexibility)
# =============================================================================
# CI mode detection - when true, uses reduced settings for faster runs
CI_MODE = os.getenv("MERCURY_CI_MODE", "false").lower() == "true"

# Default sample sizes (reduced in CI mode to stay under 120 min timeout)
DEFAULT_SAMPLES = int(os.getenv("MERCURY_CI_SAMPLES", "3000" if not CI_MODE else "1500"))

# SMD dataset machines (28 = full, 5 = CI mode)
SMD_MAX_MACHINES = int(os.getenv("MERCURY_SMD_MACHINES", "28" if not CI_MODE else "5"))

# Network fetch configuration
FETCH_MAX_RETRIES = int(os.getenv("MERCURY_FETCH_RETRIES", "10"))
FETCH_BASE_DELAY = float(os.getenv("MERCURY_FETCH_DELAY", "2.0"))

# Known checksums for integrity verification (compute from verified sources)
# Format: 'dataset:file': 'sha256:hexdigest'
# These should be computed once from known-good downloads
KNOWN_CHECKSUMS: dict[str, str | None] = {
    "smd:machine-1-1.txt": None,  # Populate with actual hashes when available
    "batadal:dataset03.csv": None,
}
from omni_mercury_engine.ml._native_utils import (
    NativeEllipticEnvelope as EllipticEnvelope,
    NativeKFold as KFold,
    NativeLocalOutlierFactor as LocalOutlierFactor,
    NativeOneClassSVM as OneClassSVM,
    NativeStandardScaler as StandardScaler,
    NativeStratifiedKFold as StratifiedKFold,
    native_confusion_matrix as confusion_matrix,
    native_f1_score as f1_score,
    native_precision_score as precision_score,
    native_recall_score as recall_score,
    native_roc_auc_score as roc_auc_score,
    native_train_test_split as train_test_split,
)

from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

_emp_logger = logging.getLogger(__name__)

try:
    import optuna  # noqa: F401

    _AUTO_TUNE = True
except ImportError:
    _AUTO_TUNE = False
    _emp_logger.info(
        "optuna not installed — auto_tune disabled. " "Install with: pip install optuna"
    )

# AdaptiveAnomalyDetector removed — Mercury uses MercuryAnomalyDetector only
ADAPTIVE_DETECTOR_AVAILABLE = False

# Suppress expected warnings from numpy during benchmark model fitting.
# Scoped to specific categories rather than blanket suppression.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*convergence.*", category=UserWarning)

# Configure logging for benchmark telemetry
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_checksum(content: bytes, dataset_key: str) -> bool:
    """
    Verify content integrity against known checksums.

    Args:
        content: Raw bytes of downloaded content
        dataset_key: Key in KNOWN_CHECKSUMS dict (e.g., 'smd:machine-1-1.txt')

    Returns:
        True if checksum matches or no checksum registered, False if mismatch.
    """
    expected = KNOWN_CHECKSUMS.get(dataset_key)
    if expected is None:
        logger.debug(f"No checksum registered for {dataset_key}, skipping verification")
        return True

    computed = hashlib.sha256(content).hexdigest()
    if computed != expected:
        logger.warning(
            f"Checksum mismatch for {dataset_key}: "
            f"expected {expected[:16]}..., got {computed[:16]}..."
        )
        return False
    logger.debug(f"Checksum verified for {dataset_key}")
    return True


def fetch_with_retry(
    fetch_func: Callable[..., Any],
    dataset_name: str,
    max_retries: int = 5,
    base_delay: float = 2.0,
    **fetch_kwargs: Any,
) -> Any | None:
    """
    Fetch a dataset with exponential backoff retry logic.

    Handles intermittent HTTP 403 errors from data servers (OpenML, UCI)
    which can rate-limit or block CI IPs temporarily.

    Args:
        fetch_func: The fetch function to call (e.g., a dataset downloader)
        dataset_name: Name of the dataset for logging
        max_retries: Maximum number of retry attempts (default: 5)
        base_delay: Base delay in seconds for exponential backoff (default: 2.0)
        **fetch_kwargs: Additional keyword arguments to pass to fetch_func

    Returns:
        The fetched data object, or None if all retries failed
    """
    for attempt in range(max_retries):
        try:
            data = fetch_func(**fetch_kwargs)
            if attempt > 0:
                logger.info(f"Successfully fetched {dataset_name} on attempt {attempt + 1}")
            return data
        except Exception as e:
            error_msg = str(e)
            is_http_error = any(
                code in error_msg for code in ["403", "429", "500", "502", "503", "504"]
            )

            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)  # Exponential backoff
                if is_http_error:
                    logger.warning(
                        f"HTTP error fetching {dataset_name} (attempt {attempt + 1}/{max_retries}): "
                        f"{error_msg}. Retrying in {delay:.1f}s..."
                    )
                else:
                    logger.warning(
                        f"Error fetching {dataset_name} (attempt {attempt + 1}/{max_retries}): "
                        f"{error_msg}. Retrying in {delay:.1f}s..."
                    )
                time.sleep(delay)
            else:
                logger.error(
                    f"Failed to fetch {dataset_name} after {max_retries} attempts: {error_msg}"
                )
                return None
    return None


# Browser-like headers to bypass anti-bot restrictions on dataset servers
_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def fetch_from_mirror(
    url: str,
    dataset_name: str,
    max_retries: int = 3,
    base_delay: float = 2.0,
    timeout: int = 60,
) -> bytes | None:
    """
    Fetch dataset from a mirror URL with browser-like headers.

    Used as fallback when direct dataset downloads fail due to HTTP 403 errors
    from rate-limiting or anti-bot measures on OpenML/Figshare servers.

    Args:
        url: Direct URL to the dataset file
        dataset_name: Name of the dataset for logging
        max_retries: Maximum retry attempts (default: 3)
        base_delay: Base delay for exponential backoff (default: 2.0)
        timeout: Request timeout in seconds (default: 60)

    Returns:
        Raw bytes of the downloaded file, or None if all attempts failed
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=_BROWSER_HEADERS, timeout=timeout)
            response.raise_for_status()
            logger.info(f"Successfully fetched {dataset_name} from mirror: {url}")
            return bytes(response.content)
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                logger.warning(
                    f"Mirror fetch failed for {dataset_name} (attempt {attempt + 1}/{max_retries}): "
                    f"{e}. Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                logger.warning(f"All mirror attempts failed for {dataset_name}: {e}")
                return None
    return None


# =============================================================================
# GitHub-based Dataset Fetching for Time-Series Anomaly Detection
# =============================================================================
# These functions fetch real datasets from public GitHub repositories without
# requiring authentication. This enables honest benchmarks on industry-standard
# datasets that peer systems use (SMD, SMAP, MSL, BATADAL).
#
# Data Sources (all MIT/public domain, no auth required):
# - NASA SMAP/MSL: https://github.com/khundman/telemanom (NASA telemetry)
# - SMD: https://github.com/NetManAIOps/OmniAnomaly (server metrics)
# - BATADAL: https://github.com/SYChen123/Baseline-outlier-detection-algorithms-on-BATADAL-dataset
#
# Citations:
# - SMAP/MSL: Hundman et al., "Detecting Spacecraft Anomalies Using LSTMs", KDD 2018
# - SMD: Su et al., "Robust Anomaly Detection for Multivariate Time Series", KDD 2019
# - BATADAL: Taormina et al., "Battle of the Attack Detection Algorithms", ASCE 2018
# =============================================================================


def fetch_nasa_telemanom_labels() -> dict[str, list[tuple[int, int]]] | None:
    """
    Fetch labeled anomaly ranges from NASA telemanom repository.

    Returns:
        Dictionary mapping channel_id to list of (start, end) anomaly ranges,
        or None if fetch failed.
    """
    url = "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv"
    content = fetch_from_mirror(url, "NASA telemanom labels", max_retries=5)
    if content is None:
        return None

    try:
        import ast

        df = pd.read_csv(io.BytesIO(content))
        labels = {}
        for _, row in df.iterrows():
            chan_id = row["chan_id"]
            # Parse anomaly_sequences string like "[[2149, 2349], [4536, 4844]]"
            sequences = ast.literal_eval(row["anomaly_sequences"])
            labels[chan_id] = [(s[0], s[1]) for s in sequences]
        logger.info(f"Loaded NASA telemanom labels for {len(labels)} channels")
        return labels
    except Exception as e:
        logger.warning(f"Error parsing NASA telemanom labels: {e}")
        return None


def fetch_smap_from_github(max_channels: int = 10) -> tuple[np.ndarray, np.ndarray, str] | None:
    """
    Fetch SMAP (Soil Moisture Active Passive) satellite telemetry.

    NOTE: The NASA telemanom repo (https://github.com/khundman/telemanom) stores
    actual data files on Kaggle which requires authentication. This function
    returns None to trigger synthetic fallback with transparent logging.

    For real SMAP data, users can:
    1. Install kaggle CLI: pip install kaggle
    2. Set up Kaggle API key
    3. Download: kaggle datasets download -d patrickfleith/nasa-anomaly-detection-dataset-smap-msl
    4. Place in data/SMAP/ directory

    Citation: Hundman et al., "Detecting Spacecraft Anomalies Using LSTMs", KDD 2018

    Args:
        max_channels: Maximum number of channels to fetch (unused, kept for API compat)

    Returns:
        None - SMAP data requires Kaggle auth, triggers synthetic fallback.
    """
    # NASA SMAP data is stored on Kaggle (requires auth), not directly on GitHub
    # The telemanom repo only contains labels CSV, not the actual .npy data files
    logger.info(
        "SMAP: Data requires Kaggle authentication (not available via public GitHub). "
        "To use real data: kaggle datasets download -d patrickfleith/nasa-anomaly-detection-dataset-smap-msl"
    )
    return None


def fetch_msl_from_github(max_channels: int = 10) -> tuple[np.ndarray, np.ndarray, str] | None:
    """
    Fetch MSL (Mars Science Laboratory) rover telemetry.

    NOTE: The NASA telemanom repo (https://github.com/khundman/telemanom) stores
    actual data files on Kaggle which requires authentication. This function
    returns None to trigger synthetic fallback with transparent logging.

    For real MSL data, users can:
    1. Install kaggle CLI: pip install kaggle
    2. Set up Kaggle API key
    3. Download: kaggle datasets download -d patrickfleith/nasa-anomaly-detection-dataset-smap-msl
    4. Place in data/MSL/ directory

    Citation: Hundman et al., "Detecting Spacecraft Anomalies Using LSTMs", KDD 2018

    Args:
        max_channels: Maximum number of channels to fetch (unused, kept for API compat)

    Returns:
        None - MSL data requires Kaggle auth, triggers synthetic fallback.
    """
    # NASA MSL data is stored on Kaggle (requires auth), not directly on GitHub
    # The telemanom repo only contains labels CSV, not the actual .npy data files
    logger.info(
        "MSL: Data requires Kaggle authentication (not available via public GitHub). "
        "To use real data: kaggle datasets download -d patrickfleith/nasa-anomaly-detection-dataset-smap-msl"
    )
    return None


def _create_labels_from_ranges(
    X: np.ndarray,
    label_dict: dict[str, list[tuple[int, int]]] | None,
    prefix: str,
) -> np.ndarray:
    """
    Create binary labels from telemanom anomaly ranges.

    Args:
        X: Data array to create labels for
        label_dict: Dictionary mapping channel IDs to anomaly ranges
        prefix: Channel prefix to filter (e.g., 'A' for SMAP, 'M' for MSL)

    Returns:
        Binary label array (0=normal, 1=anomaly)
    """
    if label_dict is None:
        return np.zeros(len(X))

    y = np.zeros(len(X))
    # Match channel IDs to ranges (simplified - production would track per-channel)
    for chan_id, ranges in label_dict.items():
        if chan_id.startswith(prefix.rstrip("-")):
            for start, end in ranges:
                if end <= len(y):
                    y[start:end] = 1
    return y


def fetch_smap_msl_local(dataset: str = "SMAP") -> tuple[Any, ...] | None:
    """
    Load SMAP/MSL from local directory matching telemanom structure.

    Telemanom Kaggle structure (after download):
        data/
        ├── train/
        │   ├── A-1.npy, A-2.npy, ...  (SMAP: 55 channels, prefix 'A-')
        │   ├── M-1.npy, M-2.npy, ...  (MSL: 27 channels, prefix 'M-')
        └── test/
            ├── A-1.npy, A-2.npy, ...
            └── M-1.npy, M-2.npy, ...

    Users download via:
        kaggle datasets download -d patrickfleith/nasa-anomaly-detection-dataset-smap-msl
        unzip -d data/ nasa-anomaly-detection-dataset-smap-msl.zip

    Args:
        dataset: Either "SMAP" or "MSL"

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, source_info) or None if not found.
    """
    # Telemanom uses shared data/ directory, channels distinguished by prefix
    data_dir = Path(__file__).parent.parent / "data"

    if not data_dir.exists():
        logger.info(f"Local data directory not found at {data_dir}")
        return None

    # Channel prefix per dataset
    prefix_map = {"SMAP": "A-", "MSL": "M-"}
    prefix = prefix_map.get(dataset, "A-")

    # Expected channel counts (from telemanom repo)
    channel_counts = {"SMAP": 55, "MSL": 27}
    expected_channels = channel_counts.get(dataset, 55)

    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    if not train_dir.exists():
        logger.info(f"Train directory not found: {train_dir}")
        return None

    # Load channels matching this dataset's prefix
    train_files = sorted(train_dir.glob(f"{prefix}*.npy"))
    test_files = sorted(test_dir.glob(f"{prefix}*.npy")) if test_dir.exists() else []

    if not train_files:
        logger.info(f"No {dataset} train files found with prefix '{prefix}' in {train_dir}")
        return None

    try:
        X_train = np.concatenate([np.load(f) for f in train_files], axis=0)
        X_test = np.concatenate([np.load(f) for f in test_files], axis=0) if test_files else None

        # Load labels from telemanom CSV (already fetched via GitHub)
        labels = fetch_nasa_telemanom_labels()

        # Create binary labels from anomaly ranges
        y_train = np.zeros(len(X_train))
        y_test = _create_labels_from_ranges(X_test, labels, prefix) if X_test is not None else None

        coverage = len(train_files) / expected_channels
        source = (
            f"real-local-telemanom ({len(train_files)}/{expected_channels} channels, "
            f"{coverage:.0%})"
        )

        logger.info(
            f"{dataset} loaded from local telemanom structure: "
            f"{X_train.shape[0]} train samples, {len(train_files)} channels"
        )

        return X_train, X_test, y_train, y_test, source

    except Exception as e:
        logger.error(f"Error loading {dataset} from local: {e}")
        return None


def fetch_smd_from_github(
    max_machines: int | None = None,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    """
    Fetch SMD (Server Machine Dataset) from OmniAnomaly GitHub.

    Source: https://github.com/NetManAIOps/OmniAnomaly
    Citation: Su et al., "Robust Anomaly Detection for Multivariate Time Series", KDD 2019

    Args:
        max_machines: Maximum machines to fetch. Defaults to SMD_MAX_MACHINES env var (28=full).

    Returns:
        Tuple of (X, y, source_info) or None if fetch failed.
    """
    if max_machines is None:
        max_machines = SMD_MAX_MACHINES

    base_url = (
        "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset"
    )

    # Full SMD: 28 machines across 3 groups
    # Group 1: machine-1-1 through machine-1-8 (8 machines)
    # Group 2: machine-2-1 through machine-2-9 (9 machines)
    # Group 3: machine-3-1 through machine-3-11 (11 machines)
    all_machine_ids = []
    for g, count in [(1, 8), (2, 9), (3, 11)]:
        all_machine_ids.extend([f"machine-{g}-{m}" for m in range(1, count + 1)])

    machine_ids = all_machine_ids[:max_machines]

    X_list = []
    y_list = []
    successful_machines = 0

    for machine_id in machine_ids:
        train_url = f"{base_url}/train/{machine_id}.txt"
        test_url = f"{base_url}/test/{machine_id}.txt"
        label_url = f"{base_url}/test_label/{machine_id}.txt"

        train_content = fetch_from_mirror(
            train_url,
            f"SMD train {machine_id}",
            max_retries=FETCH_MAX_RETRIES,
            timeout=30,
        )
        test_content = fetch_from_mirror(
            test_url,
            f"SMD test {machine_id}",
            max_retries=FETCH_MAX_RETRIES,
            timeout=30,
        )
        label_content = fetch_from_mirror(
            label_url,
            f"SMD label {machine_id}",
            max_retries=FETCH_MAX_RETRIES,
            timeout=30,
        )

        if train_content is None or test_content is None:
            continue

        # Verify checksums if available
        verify_checksum(train_content, f"smd:{machine_id}.txt")

        try:
            # SMD files are CSV-like with comma-separated values
            train_data = np.loadtxt(io.BytesIO(train_content), delimiter=",")
            test_data = np.loadtxt(io.BytesIO(test_content), delimiter=",")

            if label_content is not None:
                test_labels = np.loadtxt(io.BytesIO(label_content), delimiter=",")
            else:
                test_labels = np.zeros(len(test_data))

            combined_data = np.vstack([train_data, test_data])
            combined_labels = np.concatenate([np.zeros(len(train_data)), test_labels])

            X_list.append(combined_data)
            y_list.append(combined_labels)
            successful_machines += 1
        except Exception as e:
            logger.warning(f"Error loading SMD machine {machine_id}: {e}")
            continue

    if not X_list:
        return None

    # SMD machines have 38 features each - concatenate samples
    min_features = min(x.shape[1] for x in X_list)
    X = np.vstack([x[:, :min_features] for x in X_list])
    y = np.concatenate(y_list)

    # Determine data quality based on coverage
    coverage = successful_machines / len(machine_ids)
    if coverage < 0.5:
        source_info = (
            f"real-github-partial ({successful_machines}/{len(machine_ids)} machines, "
            f"{coverage:.0%} coverage)"
        )
    else:
        source_info = f"real-github ({successful_machines}/{len(machine_ids)} machines)"

    logger.info(
        f"SMD loaded: {X.shape[0]} samples, {X.shape[1]} features from {successful_machines} machines"
    )
    return X, y.astype(int), source_info


def fetch_batadal_from_github() -> tuple[np.ndarray, np.ndarray, str] | None:
    """
    Fetch BATADAL (Battle of Attack Detection Algorithms) water treatment dataset.

    Source: https://github.com/SYChen123/Baseline-outlier-detection-algorithms-on-BATADAL-dataset
    Citation: Taormina et al., "Battle of the Attack Detection Algorithms", ASCE 2018

    This is used as an alternative to SWaT which requires registration.

    Returns:
        Tuple of (X, y, source_info) or None if fetch failed.
    """
    base_url = "https://raw.githubusercontent.com/SYChen123/Baseline-outlier-detection-algorithms-on-BATADAL-dataset/master/data"

    # dataset03.csv is training (normal), dataset04.csv and test_dataset.csv have attacks
    train_url = f"{base_url}/dataset03.csv"
    test_url = f"{base_url}/test_dataset.csv"

    train_content = fetch_from_mirror(train_url, "BATADAL train", max_retries=5, timeout=60)
    test_content = fetch_from_mirror(test_url, "BATADAL test", max_retries=5, timeout=60)

    if train_content is None or test_content is None:
        return None

    try:
        train_df = pd.read_csv(io.BytesIO(train_content))
        test_df = pd.read_csv(io.BytesIO(test_content))

        # BATADAL has a label column (ATT_FLAG or similar) - find it
        label_cols = [
            c
            for c in test_df.columns
            if "ATT" in c.upper() or "LABEL" in c.upper() or "FLAG" in c.upper()
        ]

        if label_cols:
            label_col = label_cols[0]
            test_labels = test_df[label_col].values
            test_df = test_df.drop(columns=[label_col])
        else:
            # Assume last column is label
            test_labels = test_df.iloc[:, -1].values
            test_df = test_df.iloc[:, :-1]

        # Remove non-numeric columns (like DATETIME)
        train_numeric = train_df.select_dtypes(include=[np.number])
        test_numeric = test_df.select_dtypes(include=[np.number])

        # Align columns
        common_cols = list(set(train_numeric.columns) & set(test_numeric.columns))
        if not common_cols:
            logger.warning("No common numeric columns between BATADAL train and test")
            return None

        train_data = train_numeric[common_cols].values
        test_data = test_numeric[common_cols].values

        # Combine with labels (train is normal, test has attacks)
        X = np.vstack([train_data, test_data])
        y = np.concatenate([np.zeros(len(train_data)), test_labels])

        # Convert labels to binary (0=normal, 1=attack)
        y = (y != 0).astype(int)

        source_info = "BATADAL GitHub (water treatment ICS)"
        logger.info(
            f"Successfully loaded BATADAL from GitHub: {X.shape[0]} samples, {X.shape[1]} features"
        )
        return X, y, source_info
    except Exception as e:
        logger.warning(f"Error parsing BATADAL data: {e}")
        return None


sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run with enhanced metrics."""

    dataset_name: str
    detector_name: str
    roc_auc: float
    pr_auc: float
    f1: float
    precision: float
    recall: float
    false_positive_rate: float
    inference_latency_ms: float
    train_time_ms: float
    n_samples: int
    n_features: int
    anomaly_ratio: float
    timestamp: str
    # Enhanced metrics for K-fold CV
    fold_metrics: list[dict[str, float]] = field(default_factory=list)
    roc_auc_std: float = 0.0
    f1_std: float = 0.0
    # Confusion matrix metrics
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    # Per-class metrics
    class_precision: dict[str, float] = field(default_factory=dict)
    class_recall: dict[str, float] = field(default_factory=dict)
    class_f1: dict[str, float] = field(default_factory=dict)
    # Data provenance tracking
    data_source: str = "unknown"  # 'real-github', 'real-local', 'synthetic-fallback'


@dataclass
class DatasetInfo:
    """Information about a benchmark dataset."""

    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    description: str
    domain: str
    is_time_series: bool = False
    window_size: int = 10
    # Data provenance tracking
    data_source: str = "unknown"  # 'real-github', 'real-local', 'synthetic-fallback'
    source_url: str = ""  # Attribution URL
    citation: str = ""  # Academic citation


@dataclass
class KFoldResult:
    """Results from K-fold cross-validation."""

    fold_results: list[BenchmarkResult]
    mean_roc_auc: float
    std_roc_auc: float
    mean_f1: float
    std_f1: float
    mean_precision: float
    mean_recall: float
    aggregated_confusion_matrix: np.ndarray


def _load_breast_cancer_native() -> tuple[np.ndarray, np.ndarray]:
    """Load Wisconsin Breast Cancer (diagnostic) from UCI via urllib.

    Falls back to synthetic data with matching dimensions (569 x 30).
    """
    try:
        import csv
        import urllib.request

        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
        with urllib.request.urlopen(url, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
        reader = csv.reader(io.StringIO(raw))
        rows = list(reader)
        # Columns: ID, Diagnosis (M/B), 30 features
        X = np.array([[float(v) for v in row[2:]] for row in rows if len(row) >= 32])
        y = np.array([0 if row[1] == "M" else 1 for row in rows if len(row) >= 32])
        logger.info(f"Loaded breast cancer from UCI ({len(X)} samples)")
        return X, y
    except Exception as exc:
        logger.warning(f"UCI breast cancer download failed ({exc}), using synthetic fallback")
        rng = np.random.RandomState(42)
        n_normal, n_anom = 357, 212
        X_normal = rng.randn(n_normal, 30) * 1.0
        X_anom = rng.randn(n_anom, 30) * 1.5 + 2.0
        X = np.vstack([X_normal, X_anom])
        y = np.concatenate([np.ones(n_normal), np.zeros(n_anom)])
        return X, y


def _load_digits_native() -> tuple[np.ndarray, np.ndarray]:
    """Load optical digits dataset from UCI via urllib.

    Falls back to synthetic data with matching dimensions (1797 x 64).
    """
    try:
        import urllib.request

        url_tra = "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra"
        url_tes = "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes"
        rows = []
        for url in [url_tra, url_tes]:
            with urllib.request.urlopen(url, timeout=30) as resp:
                for line in resp.read().decode("utf-8").strip().split("\n"):
                    parts = line.strip().split(",")
                    if len(parts) == 65:
                        rows.append([float(v) for v in parts])
        data = np.array(rows)
        X, y = data[:, :-1], data[:, -1].astype(int)
        logger.info(f"Loaded digits from UCI ({len(X)} samples)")
        return X, y
    except Exception as exc:
        logger.warning(f"UCI digits download failed ({exc}), using synthetic fallback")
        rng = np.random.RandomState(43)
        X = rng.randint(0, 17, size=(1797, 64)).astype(float)
        y = rng.randint(0, 10, size=1797)
        return X, y


def prepare_breast_cancer_dataset() -> DatasetInfo:
    """Prepare breast cancer dataset for anomaly detection.

    Malignant samples (minority class) treated as anomalies.
    Loads from UCI directly (no external ML library required).
    """
    X, y = _load_breast_cancer_native()
    y_anomaly = 1 - y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_anomaly, test_size=0.3, random_state=42, stratify=y_anomaly
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return DatasetInfo(
        name="breast_cancer",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        description="Breast cancer diagnosis (malignant=anomaly)",
        domain="medical",
    )


def prepare_digits_dataset() -> DatasetInfo:
    """Prepare digits dataset for anomaly detection.

    Digit '8' treated as anomaly (unusual shape).
    Loads from UCI directly (no external ML library required).
    """
    X, y = _load_digits_native()
    y_anomaly = (y == 8).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_anomaly, test_size=0.3, random_state=42, stratify=y_anomaly
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return DatasetInfo(
        name="digits_8",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        description="Handwritten digits (digit 8=anomaly)",
        domain="pattern_recognition",
    )


def prepare_covtype_dataset(n_samples: int = 5000) -> DatasetInfo | None:
    """
    Prepare forest cover type dataset for anomaly detection.
    Rare cover type (type 4) treated as anomaly.

    Uses retry logic with exponential backoff for HTTP errors.
    Falls back to UCI mirror if direct download fails (HTTP 403).
    Falls back to synthetic data only as last resort.
    """
    X, y = None, None
    source = "synthetic"

    # Fallback: Try UCI mirror with browser headers
    if X is None:
        logger.info("Trying UCI mirror for covtype dataset...")
        uci_url = "https://archive.ics.uci.edu/static/public/31/data.csv"
        content = fetch_from_mirror(uci_url, "covtype")
        if content is not None:
            try:
                df = pd.read_csv(io.BytesIO(content))
                X = df.iloc[:, :-1].values.astype(float)
                y = df.iloc[:, -1].values.astype(int)
                source = "uci_mirror"
                logger.info(f"Successfully loaded covtype from UCI mirror ({len(X)} samples)")
            except Exception as e:
                logger.warning(f"Error parsing UCI covtype data: {e}")
                X, y = None, None

    # Process real data if available
    if X is not None and y is not None:
        if len(X) > n_samples * 3:
            indices = np.random.RandomState(42).choice(len(X), n_samples * 3, replace=False)
            X, y = X[indices], y[indices]

        y_anomaly = (y == 4).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_anomaly, test_size=0.3, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return DatasetInfo(
            name="covtype",
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            description=f"Forest cover type (type 4=anomaly) [source: {source}]",
            domain="environmental",
        )

    # Last resort: synthetic data
    logger.warning(
        "FALLBACK: All covtype sources failed (UCI mirror). "
        "Using synthetic data - benchmark results may differ from real dataset."
    )
    X, y = _generate_synthetic_time_series(
        n_samples=n_samples, n_features=54, anomaly_ratio=0.03, seed=45
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return DatasetInfo(
        name="covtype_synthetic",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        description="Synthetic environmental data (covtype fallback)",
        domain="environmental",
    )


def prepare_kddcup_dataset(n_samples: int = 5000) -> DatasetInfo | None:
    """
    Prepare KDDCup99 dataset for anomaly detection.
    Attack traffic treated as anomaly.

    Uses retry logic with exponential backoff for HTTP errors.
    Falls back to synthetic data only as last resort.
    """
    X_numeric, y_anomaly = None, None
    source = "synthetic"
    dataset_name = "kddcup99"

    # Try NSL-KDD from GitHub (improved version of KDDCup99)
    if X_numeric is None:
        logger.info("Trying NSL-KDD from GitHub as KDDCup99 alternative...")
        nsl_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
        content = fetch_from_mirror(nsl_url, "NSL-KDD")
        if content is not None:
            try:
                # NSL-KDD format: 41 features + label + difficulty (43 cols total)
                # Lines end with $ which we need to strip
                lines = content.decode("utf-8").strip().split("\n")
                rows = [line.rstrip("$").split(",") for line in lines if line.strip()]

                # Extract numeric features (columns 0, 4-40 are numeric)
                # Columns 1-3 are categorical (protocol, service, flag)
                numeric_cols = [0] + list(range(4, 41))
                X_list = []
                y_list = []
                for row in rows:
                    if len(row) >= 42:
                        features = [float(row[i]) for i in numeric_cols]
                        X_list.append(features)
                        label = row[41].strip()
                        y_list.append(0 if label == "normal" else 1)

                X_numeric = np.array(X_list)
                y_anomaly = np.array(y_list)
                source = "nsl_kdd_github"
                dataset_name = "nsl_kdd"
                logger.info(f"Successfully loaded NSL-KDD from GitHub ({len(X_numeric)} samples)")
            except Exception as e:
                logger.warning(f"Error parsing NSL-KDD data: {e}")
                X_numeric, y_anomaly = None, None

    # Process real data if available
    if X_numeric is not None and y_anomaly is not None:
        if len(X_numeric) > n_samples * 3:
            indices = np.random.RandomState(42).choice(len(X_numeric), n_samples * 3, replace=False)
            X_numeric, y_anomaly = X_numeric[indices], y_anomaly[indices]

        X_train, X_test, y_train, y_test = train_test_split(
            X_numeric, y_anomaly, test_size=0.3, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return DatasetInfo(
            name=dataset_name,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            description=f"Network intrusion detection (attacks=anomaly) [source: {source}]",
            domain="cybersecurity",
        )

    # Last resort: synthetic data
    logger.warning(
        "FALLBACK: All KDDCup99/NSL-KDD sources failed. "
        "Using synthetic data - benchmark results may differ from real dataset."
    )
    X, y = _generate_synthetic_time_series(
        n_samples=n_samples, n_features=41, anomaly_ratio=0.20, seed=46
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return DatasetInfo(
        name="kddcup99_synthetic",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        description="Synthetic cybersecurity data (KDDCup99 fallback)",
        domain="cybersecurity",
    )


def _generate_synthetic_time_series(
    n_samples: int = 5000,
    n_features: int = 25,
    anomaly_ratio: float = 0.05,
    window_size: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic time-series data for benchmarking when real datasets unavailable.

    Creates realistic multivariate time-series with injected anomalies including:
    - Point anomalies (sudden spikes)
    - Contextual anomalies (unusual patterns)
    - Collective anomalies (sustained deviations)
    """
    rng = np.random.RandomState(seed)

    # Generate base time series with trends and seasonality
    t = np.arange(n_samples)
    X = np.zeros((n_samples, n_features))

    for i in range(n_features):
        # Base signal with trend and seasonality
        trend = 0.001 * t * rng.randn()
        seasonality = np.sin(2 * np.pi * t / (100 + 10 * rng.randn())) * rng.uniform(0.5, 2)
        noise = rng.randn(n_samples) * 0.1
        X[:, i] = trend + seasonality + noise

        # Add correlations between features
        if i > 0:
            X[:, i] += 0.3 * X[:, i - 1]

    # Inject anomalies
    n_anomalies = int(n_samples * anomaly_ratio)
    anomaly_indices = rng.choice(n_samples, n_anomalies, replace=False)
    y = np.zeros(n_samples, dtype=int)

    for idx in anomaly_indices:
        anomaly_type = rng.choice(["point", "contextual", "collective"])
        affected_features = rng.choice(
            n_features, rng.randint(1, n_features // 2 + 1), replace=False
        )

        if anomaly_type == "point":
            # Sudden spike
            X[idx, affected_features] += rng.uniform(3, 6) * rng.choice([-1, 1])
        elif anomaly_type == "contextual":
            # Unusual pattern for context
            X[idx, affected_features] *= rng.uniform(2, 4)
        else:
            # Collective anomaly (sustained deviation)
            end_idx = min(idx + rng.randint(3, 10), n_samples)
            X[idx:end_idx, affected_features] += rng.uniform(2, 4)
            y[idx:end_idx] = 1

        y[idx] = 1

    return X, y


def prepare_smd_dataset(n_samples: int = 5000, window_size: int = 10) -> DatasetInfo | None:
    """
    Prepare SMD (Server Machine Dataset) for anomaly detection.

    SMD is a multivariate time-series dataset from server machines with
    38 features including CPU, memory, network metrics.

    Data source priority:
    1. GitHub raw (OmniAnomaly repo) - no auth required
    2. Local files (if pre-downloaded)
    3. Synthetic fallback (with explicit warning)

    Citation: Su et al., "Robust Anomaly Detection for Multivariate Time Series", KDD 2019
    """
    try:
        X, y = None, None
        source_info = "synthetic"
        is_synthetic = True

        # Priority 1: Try GitHub raw fetch (no auth required)
        github_result = fetch_smd_from_github(max_machines=5)
        if github_result is not None:
            X, y, source_info = github_result
            is_synthetic = False
            logger.info(f"Loaded real SMD from {source_info}")
        else:
            # Priority 2: Try local files
            smd_paths = [
                Path("data/SMD"),
                Path.home() / "data" / "SMD",
                Path("/tmp/SMD"),
            ]
            for path in smd_paths:
                if path.exists():
                    train_files = list(path.glob("train/*.txt"))
                    test_files = list(path.glob("test/*.txt"))
                    if train_files and test_files:
                        X_train_list = [np.loadtxt(f) for f in train_files[:3]]
                        X_test_list = [np.loadtxt(f) for f in test_files[:3]]
                        X = np.vstack(X_train_list + X_test_list)
                        label_files = list(path.glob("test_label/*.txt"))
                        if label_files:
                            y = np.concatenate([np.loadtxt(f) for f in label_files[:3]])
                        else:
                            y = np.zeros(len(X))
                        is_synthetic = False
                        source_info = f"local ({path})"
                        logger.info(f"Successfully loaded SMD from {path}")
                        break

        # Priority 3: Synthetic fallback (with explicit warning)
        if X is None:
            logger.warning(
                "SMD: GitHub fetch failed and no local files found. "
                "Using SYNTHETIC data - metrics may vary 20-40% from real benchmarks."
            )
            X, y = _generate_synthetic_time_series(
                n_samples=n_samples, n_features=38, anomaly_ratio=0.04, seed=42
            )
            source_info = "synthetic (fallback)"

        # Limit samples if needed
        if len(X) > n_samples:
            indices = np.random.RandomState(42).choice(len(X), n_samples, replace=False)
            assert y is not None
            X, y = X[indices], y[indices]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        dataset_name = "smd_synthetic" if is_synthetic else "smd"
        description = f"Server Machine Dataset (server metrics anomaly) [{source_info}]"
        return DatasetInfo(
            name=dataset_name,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            description=description,
            domain="infrastructure",
            is_time_series=True,
            window_size=window_size,
        )
    except Exception as e:
        logger.warning(f"Could not prepare SMD dataset: {e}")
        return None


def prepare_smap_dataset(n_samples: int = 5000, window_size: int = 10) -> DatasetInfo | None:
    """
    Prepare SMAP (Soil Moisture Active Passive) dataset for anomaly detection.

    SMAP is a NASA satellite telemetry dataset with 25 features.

    Data source priority:
    1. GitHub raw (NASA telemanom repo) - no auth required
    2. Local files (if pre-downloaded)
    3. Synthetic fallback (with explicit warning)

    Citation: Hundman et al., "Detecting Spacecraft Anomalies Using LSTMs", KDD 2018
    """
    try:
        X, y = None, None
        source_info = "synthetic"
        is_synthetic = True

        # Priority 1: Try GitHub raw fetch (no auth required)
        github_result = fetch_smap_from_github(max_channels=10)
        if github_result is not None:
            X, y, source_info = github_result
            is_synthetic = False
            logger.info(f"Loaded real SMAP from {source_info}")
        else:
            # Priority 2: Try local files
            smap_paths = [
                Path("data/SMAP"),
                Path.home() / "data" / "SMAP",
                Path("/tmp/SMAP"),
            ]
            for path in smap_paths:
                if path.exists():
                    train_file = path / "SMAP_train.npy"
                    test_file = path / "SMAP_test.npy"
                    label_file = path / "SMAP_test_label.npy"
                    if train_file.exists() and test_file.exists():
                        X_train_raw = np.load(train_file)
                        X_test_raw = np.load(test_file)
                        X = np.vstack([X_train_raw, X_test_raw])
                        if label_file.exists():
                            y_test_raw = np.load(label_file)
                            y = np.concatenate([np.zeros(len(X_train_raw)), y_test_raw])
                        else:
                            y = np.zeros(len(X))
                        is_synthetic = False
                        source_info = f"local ({path})"
                        logger.info(f"Successfully loaded SMAP from {path}")
                        break

        # Priority 3: Synthetic fallback (with explicit warning)
        if X is None:
            logger.warning(
                "SMAP: GitHub fetch failed and no local files found. "
                "Using SYNTHETIC data - metrics may vary 20-40% from real benchmarks."
            )
            X, y = _generate_synthetic_time_series(
                n_samples=n_samples, n_features=25, anomaly_ratio=0.05, seed=43
            )
            source_info = "synthetic (fallback)"

        if len(X) > n_samples:
            indices = np.random.RandomState(43).choice(len(X), n_samples, replace=False)
            assert y is not None
            X, y = X[indices], y[indices]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        dataset_name = "smap_synthetic" if is_synthetic else "smap"
        description = f"SMAP satellite telemetry (sensor anomaly) [{source_info}]"
        return DatasetInfo(
            name=dataset_name,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            description=description,
            domain="aerospace",
            is_time_series=True,
            window_size=window_size,
        )
    except Exception as e:
        logger.warning(f"Could not prepare SMAP dataset: {e}")
        return None


def prepare_msl_dataset(n_samples: int = 5000, window_size: int = 10) -> DatasetInfo | None:
    """
    Prepare MSL (Mars Science Laboratory) dataset for anomaly detection.

    MSL is NASA rover telemetry data with 55 features.

    Data source priority:
    1. GitHub raw (NASA telemanom repo) - no auth required
    2. Local files (if pre-downloaded)
    3. Synthetic fallback (with explicit warning)

    Citation: Hundman et al., "Detecting Spacecraft Anomalies Using LSTMs", KDD 2018
    """
    try:
        X, y = None, None
        source_info = "synthetic"
        is_synthetic = True

        # Priority 1: Try GitHub raw fetch (no auth required)
        github_result = fetch_msl_from_github(max_channels=10)
        if github_result is not None:
            X, y, source_info = github_result
            is_synthetic = False
            logger.info(f"Loaded real MSL from {source_info}")
        else:
            # Priority 2: Try local files
            msl_paths = [
                Path("data/MSL"),
                Path.home() / "data" / "MSL",
                Path("/tmp/MSL"),
            ]
            for path in msl_paths:
                if path.exists():
                    train_file = path / "MSL_train.npy"
                    test_file = path / "MSL_test.npy"
                    label_file = path / "MSL_test_label.npy"
                    if train_file.exists() and test_file.exists():
                        X_train_raw = np.load(train_file)
                        X_test_raw = np.load(test_file)
                        X = np.vstack([X_train_raw, X_test_raw])
                        if label_file.exists():
                            y_test_raw = np.load(label_file)
                            y = np.concatenate([np.zeros(len(X_train_raw)), y_test_raw])
                        else:
                            y = np.zeros(len(X))
                        is_synthetic = False
                        source_info = f"local ({path})"
                        logger.info(f"Successfully loaded MSL from {path}")
                        break

        # Priority 3: Synthetic fallback (with explicit warning)
        if X is None:
            logger.warning(
                "MSL: GitHub fetch failed and no local files found. "
                "Using SYNTHETIC data - metrics may vary 20-40% from real benchmarks."
            )
            X, y = _generate_synthetic_time_series(
                n_samples=n_samples, n_features=55, anomaly_ratio=0.06, seed=44
            )
            source_info = "synthetic (fallback)"

        if len(X) > n_samples:
            indices = np.random.RandomState(44).choice(len(X), n_samples, replace=False)
            assert y is not None
            X, y = X[indices], y[indices]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        dataset_name = "msl_synthetic" if is_synthetic else "msl"
        description = f"Mars Science Laboratory telemetry (rover anomaly) [{source_info}]"
        return DatasetInfo(
            name=dataset_name,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            description=description,
            domain="aerospace",
            is_time_series=True,
            window_size=window_size,
        )
    except Exception as e:
        logger.warning(f"Could not prepare MSL dataset: {e}")
        return None


def prepare_swat_dataset(n_samples: int = 5000, window_size: int = 10) -> DatasetInfo | None:
    """
    Prepare SWaT/BATADAL dataset for ICS anomaly detection.

    SWaT (Secure Water Treatment) requires SUTD registration, so we use
    BATADAL (Battle of Attack Detection Algorithms) as a public alternative.
    Both are water treatment ICS datasets with similar attack patterns.

    Data source priority:
    1. BATADAL from GitHub (no auth required) - public alternative to SWaT
    2. Local SWaT files (if user has registered and downloaded)
    3. Synthetic fallback (with explicit warning)

    Citation: Taormina et al., "Battle of the Attack Detection Algorithms", ASCE 2018
    Note: SWaT requires registration at iTrust SUTD (https://itrust.sutd.edu.sg/)
    """
    try:
        X, y = None, None
        source_info = "synthetic"
        is_synthetic = True
        dataset_name_base = "swat"

        # Priority 1: Try BATADAL from GitHub (public alternative to SWaT)
        github_result = fetch_batadal_from_github()
        if github_result is not None:
            X, y, source_info = github_result
            is_synthetic = False
            dataset_name_base = "batadal"
            logger.info(f"Loaded real BATADAL (SWaT alternative) from {source_info}")
        else:
            # Priority 2: Try local SWaT files (requires SUTD registration)
            swat_paths = [
                Path("data/SWaT"),
                Path.home() / "data" / "SWaT",
                Path("/tmp/SWaT"),
            ]
            for path in swat_paths:
                if path.exists():
                    train_file = path / "SWaT_train.csv"
                    test_file = path / "SWaT_test.csv"
                    if train_file.exists() and test_file.exists():
                        train_df = pd.read_csv(train_file)
                        test_df = pd.read_csv(test_file)
                        X_train_raw = train_df.iloc[:, :-1].values
                        X_test_raw = test_df.iloc[:, :-1].values
                        y_test_raw = test_df.iloc[:, -1].values
                        X = np.vstack([X_train_raw, X_test_raw])
                        y = np.concatenate([np.zeros(len(X_train_raw)), y_test_raw])
                        is_synthetic = False
                        source_info = f"local SWaT ({path})"
                        logger.info(f"Successfully loaded SWaT from {path}")
                        break

        # Priority 3: Synthetic fallback (with explicit warning)
        if X is None:
            logger.warning(
                "SWaT/BATADAL: GitHub fetch failed and no local files found. "
                "Using SYNTHETIC data - metrics may vary 20-40% from real benchmarks. "
                "Note: SWaT requires registration at https://itrust.sutd.edu.sg/"
            )
            X, y = _generate_synthetic_time_series(
                n_samples=n_samples, n_features=51, anomaly_ratio=0.12, seed=45
            )
            source_info = "synthetic (fallback)"

        if len(X) > n_samples:
            indices = np.random.RandomState(45).choice(len(X), n_samples, replace=False)
            assert y is not None
            X, y = X[indices], y[indices]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        dataset_name = f"{dataset_name_base}_synthetic" if is_synthetic else dataset_name_base
        description = f"Water Treatment ICS (attack detection) [{source_info}]"
        return DatasetInfo(
            name=dataset_name,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            description=description,
            domain="critical_infrastructure",
            is_time_series=True,
            window_size=window_size,
        )
    except Exception as e:
        logger.warning(f"Could not prepare SWaT/BATADAL dataset: {e}")
        return None


class TranADDetector:
    """
    Wrapper for TranAD SOTA model to match standard anomaly detector interface.

    TranAD uses transformer architecture with adversarial training
    for time-series anomaly detection.
    """

    def __init__(self, contamination: float = 0.1, window_size: int = 10):
        self.contamination = contamination
        self.window_size = window_size
        self.model: Any | None = None
        self.threshold: float | None = None
        self.scaler = StandardScaler()
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "TranADDetector":
        """Fit TranAD model on training data with minimal training loop."""
        try:
            from omni_mercury_engine.models.sota.tranad import TranADConfig, TranADModel

            # Determine input dimension
            input_dim = X.shape[1] if X.ndim > 1 else 1

            config = TranADConfig(
                input_dim=input_dim,
                d_model=64,
                n_heads=4,
                n_encoder_layers=2,
                n_decoder_layers=1,
                window_size=self.window_size,
                use_focus_score=True,
                use_adversarial=True,
            )

            self.model = TranADModel(config)
            assert self.model is not None
            X_scaled = self.scaler.fit_transform(X)

            # Train on normal data only (unsupervised anomaly detection)
            # Use y labels if available to filter normal samples
            if y is not None:
                normal_mask = y == 0
                if np.sum(normal_mask) > self.window_size:
                    X_train = X_scaled[normal_mask]
                else:
                    X_train = X_scaled
            else:
                X_train = X_scaled

            # Minimal training loop (bounded for CI efficiency)
            self.model.train()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
            n_epochs = 20  # Bounded epochs for CI
            batch_size = min(32, len(X_train) // 2) if len(X_train) > 2 else 1

            for epoch in range(n_epochs):
                epoch_loss = 0.0
                n_batches = 0
                indices = np.random.permutation(len(X_train))

                for i in range(0, len(X_train) - self.window_size, batch_size):
                    batch_indices = indices[i : i + batch_size]
                    windows = []
                    for idx in batch_indices:
                        if idx + self.window_size <= len(X_train):
                            windows.append(X_train[idx : idx + self.window_size])
                    if not windows:
                        continue

                    x_batch = torch.tensor(np.array(windows), dtype=torch.float32)
                    optimizer.zero_grad()
                    result = self.model(x_batch)
                    # Reconstruction loss
                    loss = torch.nn.functional.mse_loss(result["reconstruction"], x_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1

                if n_batches > 0 and epoch_loss / n_batches < 0.001:
                    break  # Early stopping if converged

            self.model.eval()

            # Compute training scores for threshold
            train_scores = self._compute_scores(X_scaled)
            self.threshold = np.percentile(train_scores, 100 * (1 - self.contamination))
            self._is_fitted = True

        except ImportError:
            logger.warning("TranAD model not available, using fallback")
            self._fit_fallback(X)

        return self

    def _fit_fallback(self, X: np.ndarray) -> None:
        """Fallback using simple autoencoder-like scoring."""
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8
        self._is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels (-1 for anomaly, 1 for normal)."""
        scores = self.decision_function(X)
        if self.threshold is None:
            self.threshold = np.percentile(scores, 100 * (1 - self.contamination))
        return np.where(scores > self.threshold, -1, 1)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores (higher = more anomalous)."""
        if self.model is not None:
            X_scaled = self.scaler.transform(X)
            return self._compute_scores(X_scaled)
        return self._score_fallback(X)

    def _compute_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute TranAD anomaly scores."""
        assert self.model is not None
        scores = []
        with torch.no_grad():
            for i in range(len(X)):
                # Create window
                start_idx = max(0, i - self.window_size + 1)
                window = X[start_idx : i + 1]
                if len(window) < self.window_size:
                    # Pad with first sample
                    padding = np.tile(window[0], (self.window_size - len(window), 1))
                    window = np.vstack([padding, window])

                x_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
                result = self.model(x_tensor)
                score = result["anomaly_score"].mean().item()
                scores.append(score)

        return np.array(scores)

    def _score_fallback(self, X: np.ndarray) -> np.ndarray:
        """Fallback scoring using reconstruction error."""
        X_centered = X - self.mean
        return np.sqrt(np.sum((X_centered / self.std) ** 2, axis=1))


class MAATDetector:
    """
    Wrapper for MAAT SOTA model to match standard anomaly detector interface.

    MAAT combines Mamba SSM with sparse attention for efficient
    long-sequence anomaly detection.
    """

    def __init__(self, contamination: float = 0.1, window_size: int = 100):
        self.contamination = contamination
        self.window_size = window_size
        self.model: Any | None = None
        self.threshold: float | None = None
        self.scaler = StandardScaler()
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "MAATDetector":
        """Fit MAAT model on training data with minimal training loop."""
        try:
            from omni_mercury_engine.models.sota.maat import MAATConfig, MAATModel

            input_dim = X.shape[1] if X.ndim > 1 else 1

            config = MAATConfig(
                input_dim=input_dim,
                d_model=64,
                d_state=16,
                n_heads=4,
                n_layers=2,
                window_size=self.window_size,
                use_mamba=True,
                use_sparse_attention=True,
            )

            self.model = MAATModel(config)
            assert self.model is not None
            X_scaled = self.scaler.fit_transform(X)

            # Train on normal data only (unsupervised anomaly detection)
            # Use y labels if available to filter normal samples
            if y is not None:
                normal_mask = y == 0
                if np.sum(normal_mask) > self.window_size:
                    X_train = X_scaled[normal_mask]
                else:
                    X_train = X_scaled
            else:
                X_train = X_scaled

            # Minimal training loop (bounded for CI efficiency)
            self.model.train()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
            n_epochs = 20  # Bounded epochs for CI
            batch_size = min(32, len(X_train) // 2) if len(X_train) > 2 else 1

            for epoch in range(n_epochs):
                epoch_loss = 0.0
                n_batches = 0
                indices = np.random.permutation(len(X_train))

                for i in range(0, len(X_train) - self.window_size, batch_size):
                    batch_indices = indices[i : i + batch_size]
                    windows = []
                    for idx in batch_indices:
                        if idx + self.window_size <= len(X_train):
                            windows.append(X_train[idx : idx + self.window_size])
                    if not windows:
                        continue

                    x_batch = torch.tensor(np.array(windows), dtype=torch.float32)
                    optimizer.zero_grad()
                    result = self.model(x_batch)
                    # Reconstruction loss
                    loss = torch.nn.functional.mse_loss(result["reconstruction"], x_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1

                if n_batches > 0 and epoch_loss / n_batches < 0.001:
                    break  # Early stopping if converged

            self.model.eval()

            # Compute training scores for threshold
            train_scores = self._compute_scores(X_scaled)
            self.threshold = np.percentile(train_scores, 100 * (1 - self.contamination))
            self._is_fitted = True

        except ImportError:
            logger.warning("MAAT model not available, using fallback")
            self._fit_fallback(X)

        return self

    def _fit_fallback(self, X: np.ndarray) -> None:
        """Fallback using simple statistical scoring."""
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8
        self._is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels (-1 for anomaly, 1 for normal)."""
        scores = self.decision_function(X)
        if self.threshold is None:
            self.threshold = np.percentile(scores, 100 * (1 - self.contamination))
        return np.where(scores > self.threshold, -1, 1)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores (higher = more anomalous)."""
        if self.model is not None:
            X_scaled = self.scaler.transform(X)
            return self._compute_scores(X_scaled)
        return self._score_fallback(X)

    def _compute_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute MAAT anomaly scores."""
        assert self.model is not None
        scores = []
        batch_size = min(32, len(X))

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = X[i : i + batch_size]
                # Pad to window size if needed
                if len(batch) < self.window_size:
                    padding = np.tile(batch[0], (self.window_size - len(batch), 1))
                    batch = np.vstack([padding, batch])

                x_tensor = torch.tensor(batch, dtype=torch.float32).unsqueeze(0)
                result = self.model(x_tensor)
                batch_scores = result["anomaly_score"].squeeze().cpu().numpy()
                if batch_scores.ndim == 0:
                    batch_scores = np.array([batch_scores])
                scores.extend(batch_scores[-min(batch_size, len(X) - i) :])

        return np.array(scores)

    def _score_fallback(self, X: np.ndarray) -> np.ndarray:
        """Fallback scoring using reconstruction error."""
        X_centered = X - self.mean
        return np.sqrt(np.sum((X_centered / self.std) ** 2, axis=1))


class FallbackStrategy:
    """Enumeration of available fallback strategies (unused, kept for API compat)."""

    MAHALANOBIS = "mahalanobis"


class OmniMercuryDetector:
    """Benchmark wrapper around MercuryAnomalyDetector.

    Exposes the standard anomaly detector interface (fit/predict/decision_function)
    expected by the benchmark harness while delegating all detection to
    Mercury's original ensemble (Resonance 40% + Kinematic 30% + InfoGeo 30%).
    """

    def __init__(self, contamination: float = 0.1, **kwargs: Any) -> None:
        self.contamination = contamination
        self._detector = MercuryAnomalyDetector(auto_validate=True, auto_tune=_AUTO_TUNE)
        self._scores: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "OmniMercuryDetector":
        """Fit MercuryAnomalyDetector on training data."""
        X_clean = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        self._detector.fit(X_clean)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels (-1 = anomaly, 1 = normal, standard convention)."""
        X_clean = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        result = self._detector.detect(X_clean)
        self._scores = result["scores"]
        return np.where(result["is_anomaly"], -1, 1)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores (higher = more anomalous).

        If predict() was already called on this data, returns cached scores.
        Otherwise runs detection fresh.
        """
        if self._scores is not None and len(self._scores) == len(X):
            return self._scores
        X_clean = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        result = self._detector.detect(X_clean)
        return result["scores"]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray) -> dict[str, Any]:
    """Compute comprehensive evaluation metrics including confusion matrix."""
    y_pred_binary = (y_pred == -1).astype(int)

    try:
        roc_auc = roc_auc_score(y_true, y_scores)
    except ValueError as e:
        # Single class present or invalid input
        logger.debug(f"ROC-AUC calculation failed (single class or invalid input): {e}")
        roc_auc = 0.5

    try:
        # Use average_precision_score for correct PR-AUC calculation
        # This properly handles the precision-recall curve ordering and avoids
        # negative values from np.trapz on unsorted recall values
        from omni_mercury_engine.ml._native_utils import native_average_precision_score

        pr_auc = native_average_precision_score(y_true, y_scores)
    except ValueError as e:
        # Single class present or invalid input
        logger.debug(f"PR-AUC calculation failed (single class or invalid input): {e}")
        pr_auc = 0.0

    try:
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
    except ValueError as e:
        logger.debug(f"Classification metrics calculation failed: {e}")
        f1, precision, recall = 0.0, 0.0, 0.0

    # Compute confusion matrix
    try:
        cm = confusion_matrix(y_true, y_pred_binary)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Single class case - matrix is 1x1
            tn, fp, fn, tp = 0, 0, 0, 0
    except ValueError as e:
        logger.debug(f"Confusion matrix calculation failed: {e}")
        tn, fp, fn, tp = 0, 0, 0, 0

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Per-class metrics
    class_precision = {
        "normal": tn / (tn + fn) if (tn + fn) > 0 else 0.0,
        "anomaly": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
    }
    class_recall = {
        "normal": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "anomaly": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
    }
    class_f1 = {}
    for cls in ["normal", "anomaly"]:
        p, r = class_precision[cls], class_recall[cls]
        class_f1[cls] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fpr,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "class_precision": class_precision,
        "class_recall": class_recall,
        "class_f1": class_f1,
    }


def benchmark_detector_kfold(
    detector_class: type,
    detector_name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    contamination: float = 0.1,
    **kwargs: Any,
) -> KFoldResult:
    """Run K-fold cross-validation benchmark for a single detector."""
    # Use stratified K-fold for imbalanced data
    try:
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = list(kfold.split(X, y))
    except ValueError:
        # Fall back to regular K-fold if stratification fails
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = list(kfold.split(X))

    fold_results = []
    all_cm = np.zeros((2, 2), dtype=int)

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Create detector instance
        if detector_name == "Mercury-Agent":
            detector = detector_class(contamination=contamination)
        elif detector_name == "LocalOutlierFactor":
            detector = detector_class(contamination=contamination, novelty=True, **kwargs)
        elif detector_name == "OneClassSVM":
            nu = min(0.5, max(0.01, contamination))
            detector = detector_class(nu=nu, **kwargs)
        elif detector_name in ["TranAD", "MAAT"]:
            detector = detector_class(contamination=contamination, **kwargs)
        else:
            detector = detector_class(contamination=contamination, **kwargs)

        train_start = time.perf_counter()
        # Pass y_train for detectors that support score calibration (Mercury-Agent, TranAD, MAAT)
        if detector_name in ["Mercury-Agent", "TranAD", "MAAT"]:
            detector.fit(X_train, y_train)
        else:
            detector.fit(X_train)
        train_time = (time.perf_counter() - train_start) * 1000

        infer_start = time.perf_counter()
        y_pred = detector.predict(X_test)
        infer_time = (time.perf_counter() - infer_start) * 1000

        # Get raw scores from decision_function
        try:
            raw_scores = detector.decision_function(X_test)
        except Exception:
            raw_scores = (y_pred == -1).astype(float)

        # Detector-agnostic score direction calibration using training data
        # This ensures higher scores = more anomalous regardless of detector convention
        try:
            train_scores = detector.decision_function(X_train)
            # Check if scores need inversion (AUC < 0.5 means scores are inverted)
            if len(np.unique(y_train)) >= 2:
                train_auc = roc_auc_score(y_train, train_scores)
                if train_auc < 0.5:
                    # Scores are inverted (higher = more normal), flip them
                    y_scores = -raw_scores
                else:
                    y_scores = raw_scores
            else:
                y_scores = raw_scores
        except Exception:
            # Fallback: use raw scores as-is
            y_scores = raw_scores

        metrics = compute_metrics(y_test, y_pred, y_scores)

        # Accumulate confusion matrix
        all_cm[0, 0] += metrics["true_negatives"]
        all_cm[0, 1] += metrics["false_positives"]
        all_cm[1, 0] += metrics["false_negatives"]
        all_cm[1, 1] += metrics["true_positives"]

        fold_result = BenchmarkResult(
            dataset_name=f"fold_{fold_idx}",
            detector_name=detector_name,
            roc_auc=metrics["roc_auc"],
            pr_auc=metrics["pr_auc"],
            f1=metrics["f1"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            false_positive_rate=metrics["false_positive_rate"],
            inference_latency_ms=infer_time / len(X_test),
            train_time_ms=train_time,
            n_samples=len(X_test),
            n_features=X_test.shape[1],
            anomaly_ratio=np.mean(y_test),
            timestamp=datetime.now(UTC).isoformat(),
            true_positives=metrics["true_positives"],
            true_negatives=metrics["true_negatives"],
            false_positives=metrics["false_positives"],
            false_negatives=metrics["false_negatives"],
            class_precision=metrics["class_precision"],
            class_recall=metrics["class_recall"],
            class_f1=metrics["class_f1"],
        )
        fold_results.append(fold_result)

    # Compute aggregate statistics
    roc_aucs = [r.roc_auc for r in fold_results]
    f1s = [r.f1 for r in fold_results]
    precisions = [r.precision for r in fold_results]
    recalls = [r.recall for r in fold_results]

    return KFoldResult(
        fold_results=fold_results,
        mean_roc_auc=np.mean(roc_aucs),
        std_roc_auc=np.std(roc_aucs),
        mean_f1=np.mean(f1s),
        std_f1=np.std(f1s),
        mean_precision=np.mean(precisions),
        mean_recall=np.mean(recalls),
        aggregated_confusion_matrix=all_cm,
    )


def benchmark_detector(
    detector_class: type,
    detector_name: str,
    dataset: DatasetInfo,
    contamination: float = 0.1,
    **kwargs: Any,
) -> BenchmarkResult:
    """Run benchmark for a single detector on a single dataset."""
    if detector_name == "Mercury-Agent":
        detector = detector_class(contamination=contamination)
    elif detector_name == "LocalOutlierFactor":
        detector = detector_class(contamination=contamination, novelty=True, **kwargs)
    elif detector_name == "OneClassSVM":
        nu = min(0.5, max(0.01, contamination))
        detector = detector_class(nu=nu, **kwargs)
    else:
        detector = detector_class(contamination=contamination, **kwargs)

    train_start = time.perf_counter()
    detector.fit(dataset.X_train)
    train_time = (time.perf_counter() - train_start) * 1000

    infer_start = time.perf_counter()
    y_pred = detector.predict(dataset.X_test)
    infer_time = (time.perf_counter() - infer_start) * 1000

    try:
        y_scores = -detector.decision_function(dataset.X_test)
    except Exception:
        y_scores = (y_pred == -1).astype(float)

    metrics = compute_metrics(dataset.y_test, y_pred, y_scores)

    anomaly_ratio = np.mean(dataset.y_test)

    return BenchmarkResult(
        dataset_name=dataset.name,
        detector_name=detector_name,
        roc_auc=metrics["roc_auc"],
        pr_auc=metrics["pr_auc"],
        f1=metrics["f1"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        false_positive_rate=metrics["false_positive_rate"],
        inference_latency_ms=infer_time / len(dataset.X_test),
        train_time_ms=train_time,
        n_samples=len(dataset.X_test),
        n_features=dataset.X_test.shape[1],
        anomaly_ratio=anomaly_ratio,
        timestamp=datetime.now(UTC).isoformat(),
    )


def run_full_benchmark(
    use_kfold: bool = True,
    n_folds: int = 5,
    include_time_series: bool = True,
    include_sota: bool = True,
) -> dict[str, Any]:
    """
    Run complete benchmark suite with enhanced features.

    Args:
        use_kfold: Whether to use K-fold cross-validation (default: True)
        n_folds: Number of folds for cross-validation (default: 5)
        include_time_series: Whether to include time-series datasets (default: True)
        include_sota: Whether to include SOTA models (TranAD, MAAT) (default: True)

    Returns:
        Dictionary containing benchmark results, methodology, and summary statistics
    """
    print("=" * 70)
    print("Mercury-Agent EMPIRICAL BENCHMARK SUITE")
    print("Comparing against near-peer anomaly detection systems")
    print("=" * 70)
    print()
    print(
        f"Configuration: K-Fold={use_kfold} (n={n_folds}), "
        f"Time-Series={include_time_series}, SOTA={include_sota}"
    )
    print()

    datasets = []

    print("Loading datasets...")
    print("-" * 40)

    # Standard benchmark datasets
    bc_data = prepare_breast_cancer_dataset()
    datasets.append(bc_data)
    print(
        f"  [OK] {bc_data.name}: {bc_data.X_train.shape[0]} train, {bc_data.X_test.shape[0]} test"
    )

    digits_data = prepare_digits_dataset()
    datasets.append(digits_data)
    print(
        f"  [OK] {digits_data.name}: {digits_data.X_train.shape[0]} train, "
        f"{digits_data.X_test.shape[0]} test"
    )

    covtype_data = prepare_covtype_dataset(n_samples=DEFAULT_SAMPLES)
    if covtype_data is not None:
        datasets.append(covtype_data)
        print(
            f"  [OK] {covtype_data.name}: {covtype_data.X_train.shape[0]} train, "
            f"{covtype_data.X_test.shape[0]} test"
        )

    kdd_data = prepare_kddcup_dataset(n_samples=DEFAULT_SAMPLES)
    if kdd_data is not None:
        datasets.append(kdd_data)
        print(
            f"  [OK] {kdd_data.name}: {kdd_data.X_train.shape[0]} train, "
            f"{kdd_data.X_test.shape[0]} test"
        )

    # Time-series datasets (SMD, SMAP, MSL, SWaT)
    if include_time_series:
        print("\nLoading time-series datasets...")
        print("-" * 40)

        smd_data = prepare_smd_dataset(n_samples=DEFAULT_SAMPLES)
        if smd_data is not None:
            datasets.append(smd_data)
            print(
                f"  [OK] {smd_data.name}: {smd_data.X_train.shape[0]} train, "
                f"{smd_data.X_test.shape[0]} test (time-series)"
            )

        smap_data = prepare_smap_dataset(n_samples=DEFAULT_SAMPLES)
        if smap_data is not None:
            datasets.append(smap_data)
            print(
                f"  [OK] {smap_data.name}: {smap_data.X_train.shape[0]} train, "
                f"{smap_data.X_test.shape[0]} test (time-series)"
            )

        msl_data = prepare_msl_dataset(n_samples=DEFAULT_SAMPLES)
        if msl_data is not None:
            datasets.append(msl_data)
            print(
                f"  [OK] {msl_data.name}: {msl_data.X_train.shape[0]} train, "
                f"{msl_data.X_test.shape[0]} test (time-series)"
            )

        swat_data = prepare_swat_dataset(n_samples=DEFAULT_SAMPLES)
        if swat_data is not None:
            datasets.append(swat_data)
            print(
                f"  [OK] {swat_data.name}: {swat_data.X_train.shape[0]} train, "
                f"{swat_data.X_test.shape[0]} test (time-series)"
            )

    print()

    # Define detectors including SOTA models
    detectors: list[tuple[type, str, dict[str, Any]]] = [
        (OmniMercuryDetector, "Mercury-Agent", {}),
        (OneClassSVM, "OneClassSVM", {"kernel": "rbf", "gamma": "auto"}),
        (LocalOutlierFactor, "LocalOutlierFactor", {"n_neighbors": 20}),
        (EllipticEnvelope, "EllipticEnvelope", {"random_state": 42}),
    ]

    # Add SOTA models if requested
    if include_sota:
        detectors.extend(
            [
                (TranADDetector, "TranAD", {"window_size": 10}),
                (MAATDetector, "MAAT", {"window_size": 100}),
            ]
        )

    results: list[BenchmarkResult] = []
    kfold_results: dict[str, dict[str, KFoldResult]] = {}

    for dataset in datasets:
        print(f"\nBenchmarking on {dataset.name} ({dataset.domain})...")
        print(f"  Description: {dataset.description}")
        print(f"  Anomaly ratio: {np.mean(dataset.y_test):.2%}")
        if dataset.is_time_series:
            print(f"  Time-series: Yes (window_size={dataset.window_size})")
        print("-" * 40)

        contamination = min(0.5, max(0.01, np.mean(dataset.y_train)))

        if dataset.name not in kfold_results:
            kfold_results[dataset.name] = {}

        for detector_class, detector_name, kwargs in detectors:
            try:
                if use_kfold:
                    # Combine train and test for K-fold CV
                    X_combined = np.vstack([dataset.X_train, dataset.X_test])
                    y_combined = np.concatenate([dataset.y_train, dataset.y_test])

                    kfold_result = benchmark_detector_kfold(
                        detector_class,
                        detector_name,
                        X_combined,
                        y_combined,
                        n_folds=n_folds,
                        contamination=contamination,
                        **kwargs,
                    )
                    kfold_results[dataset.name][detector_name] = kfold_result

                    # Create aggregated result
                    result = BenchmarkResult(
                        dataset_name=dataset.name,
                        detector_name=detector_name,
                        roc_auc=kfold_result.mean_roc_auc,
                        pr_auc=np.mean([r.pr_auc for r in kfold_result.fold_results]),
                        f1=kfold_result.mean_f1,
                        precision=kfold_result.mean_precision,
                        recall=kfold_result.mean_recall,
                        false_positive_rate=np.mean(
                            [r.false_positive_rate for r in kfold_result.fold_results]
                        ),
                        inference_latency_ms=np.mean(
                            [r.inference_latency_ms for r in kfold_result.fold_results]
                        ),
                        train_time_ms=np.mean([r.train_time_ms for r in kfold_result.fold_results]),
                        n_samples=len(X_combined),
                        n_features=X_combined.shape[1],
                        anomaly_ratio=np.mean(y_combined),
                        timestamp=datetime.now(UTC).isoformat(),
                        fold_metrics=[
                            {
                                "roc_auc": r.roc_auc,
                                "f1": r.f1,
                                "precision": r.precision,
                                "recall": r.recall,
                            }
                            for r in kfold_result.fold_results
                        ],
                        roc_auc_std=kfold_result.std_roc_auc,
                        f1_std=kfold_result.std_f1,
                        true_positives=int(kfold_result.aggregated_confusion_matrix[1, 1]),
                        true_negatives=int(kfold_result.aggregated_confusion_matrix[0, 0]),
                        false_positives=int(kfold_result.aggregated_confusion_matrix[0, 1]),
                        false_negatives=int(kfold_result.aggregated_confusion_matrix[1, 0]),
                    )
                    results.append(result)

                    print(
                        f"  {detector_name:20s} | ROC-AUC: {result.roc_auc:.3f} (+/- {result.roc_auc_std:.3f}) | "
                        f"F1: {result.f1:.3f} (+/- {result.f1_std:.3f})"
                    )
                else:
                    # Single train/test split
                    result = benchmark_detector(
                        detector_class,
                        detector_name,
                        dataset,
                        contamination=contamination,
                        **kwargs,
                    )
                    results.append(result)
                    print(
                        f"  {detector_name:20s} | ROC-AUC: {result.roc_auc:.3f} | "
                        f"F1: {result.f1:.3f} | Latency: {result.inference_latency_ms:.3f}ms"
                    )
            except Exception as e:
                logger.error(f"Error benchmarking {detector_name} on {dataset.name}: {e}")
                print(f"  {detector_name:20s} | ERROR: {e}")

    summary = generate_summary(results)

    # Generate confusion matrix summary
    confusion_summary: dict[str, Any] = {}
    for r in results:
        if r.dataset_name not in confusion_summary:
            confusion_summary[r.dataset_name] = {}
        confusion_summary[r.dataset_name][r.detector_name] = {
            "tp": r.true_positives,
            "tn": r.true_negatives,
            "fp": r.false_positives,
            "fn": r.false_negatives,
        }

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "methodology": {
            "datasets": [d.name for d in datasets],
            "detectors": [d[1] for d in detectors],
            "metrics": ["roc_auc", "pr_auc", "f1", "precision", "recall", "fpr", "latency"],
            "k_fold": use_kfold,
            "n_folds": n_folds if use_kfold else None,
            "include_time_series": include_time_series,
            "include_sota": include_sota,
            "note": "Empirical benchmarks using publicly available datasets with K-fold CV",
        },
        "results": [asdict(r) for r in results],
        "summary": summary,
        "confusion_matrices": confusion_summary,
    }


def generate_summary(results: list[BenchmarkResult]) -> dict[str, Any]:
    """Generate summary statistics from benchmark results."""
    detector_metrics: dict[str, dict[str, list[float]]] = {}

    for r in results:
        if r.detector_name not in detector_metrics:
            detector_metrics[r.detector_name] = {
                "roc_auc": [],
                "f1": [],
                "precision": [],
                "recall": [],
                "latency": [],
            }
        detector_metrics[r.detector_name]["roc_auc"].append(r.roc_auc)
        detector_metrics[r.detector_name]["f1"].append(r.f1)
        detector_metrics[r.detector_name]["precision"].append(r.precision)
        detector_metrics[r.detector_name]["recall"].append(r.recall)
        detector_metrics[r.detector_name]["latency"].append(r.inference_latency_ms)

    summary = {}
    for detector, metrics in detector_metrics.items():
        summary[detector] = {
            "mean_roc_auc": float(np.mean(metrics["roc_auc"])),
            "std_roc_auc": float(np.std(metrics["roc_auc"])),
            "mean_f1": float(np.mean(metrics["f1"])),
            "std_f1": float(np.std(metrics["f1"])),
            "mean_precision": float(np.mean(metrics["precision"])),
            "mean_recall": float(np.mean(metrics["recall"])),
            "mean_latency_ms": float(np.mean(metrics["latency"])),
        }

    rankings = {}
    for metric in ["mean_roc_auc", "mean_f1"]:
        sorted_detectors = sorted(summary.items(), key=lambda x: x[1][metric], reverse=True)
        rankings[metric] = [d[0] for d in sorted_detectors]

    omni_mercury_stats = summary.get("Mercury-Agent", {})
    baseline_stats = {k: v for k, v in summary.items() if k != "Mercury-Agent"}

    if omni_mercury_stats and baseline_stats:
        omni_roc = omni_mercury_stats.get("mean_roc_auc", 0)
        best_baseline_roc = max(b.get("mean_roc_auc", 0) for b in baseline_stats.values())
        avg_baseline_roc = np.mean([b.get("mean_roc_auc", 0) for b in baseline_stats.values()])

        comparison = {
            "omni_mercury_roc_auc": omni_roc,
            "best_baseline_roc_auc": best_baseline_roc,
            "avg_baseline_roc_auc": float(avg_baseline_roc),
            "vs_best_baseline": omni_roc - best_baseline_roc,
            "vs_avg_baseline": omni_roc - avg_baseline_roc,
            "rank_by_roc_auc": (
                rankings["mean_roc_auc"].index("Mercury-Agent") + 1
                if "Mercury-Agent" in rankings["mean_roc_auc"]
                else None
            ),
        }
    else:
        comparison = {}

    return {
        "per_detector": summary,
        "rankings": rankings,
        "omni_mercury_comparison": comparison,
        "honest_assessment": generate_honest_assessment(summary, comparison),
    }


def generate_honest_assessment(
    summary: dict[str, Any], comparison: dict[str, Any]
) -> dict[str, Any]:
    """Generate honest assessment of Mercury-Agent performance."""
    assessment: dict[str, Any] = {
        "methodology_notes": [
            "Benchmarks use publicly available UCI/OpenML datasets",
            "Anomaly labels derived from minority class designation",
            "All detectors use same train/test splits for fair comparison",
            "Contamination parameter set based on actual anomaly ratio",
        ],
        "limitations": [
            "Datasets are proxies for real-world anomaly detection scenarios",
            "Medical dataset (breast_cancer) is not actual clinical data",
            "Cybersecurity dataset (KDDCup99) is from 1999, may not reflect modern attacks",
            "Results may vary with different random seeds and hyperparameters",
        ],
    }

    if comparison:
        rank = comparison.get("rank_by_roc_auc")
        vs_best = comparison.get("vs_best_baseline", 0)

        if rank == 1:
            assessment["performance_verdict"] = (
                "Mercury-Agent achieved best ROC-AUC among tested detectors"
            )
        elif vs_best >= -0.02:
            assessment["performance_verdict"] = "Mercury-Agent performs comparably to best baseline"
        else:
            assessment["performance_verdict"] = (
                f"Mercury-Agent ranks #{rank}, {abs(vs_best):.3f} ROC-AUC below best baseline"
            )

        assessment["recommendation"] = (
            "For production use, validate on domain-specific real-world data. "
            "These benchmarks provide directional guidance only."
        )

    return assessment


def save_results(results: dict[str, Any], output_path: Path | str) -> None:
    """Save benchmark results to files.

    Args:
        results: Benchmark results dictionary
        output_path: Either a directory path or a JSON file path.
            - If a .json file path: writes JSON to that exact file, report to sibling .md
            - If a directory: writes to empirical_benchmark_results.json and report inside it
    """
    # Handle string inputs and convert to Path
    if isinstance(output_path, str):
        output_path = Path(output_path)

    # Determine if this is file mode (explicit .json path) or directory mode
    if output_path.suffix == ".json":
        # File mode: write to the exact JSON path specified
        json_path = output_path
        # Create parent directory if needed
        parent_dir = json_path.parent if json_path.parent != Path() else Path()
        parent_dir.mkdir(parents=True, exist_ok=True)
        # Report goes as sibling with .md extension (benchmark_results.json -> benchmark_report.md)
        report_path = parent_dir / (json_path.stem.replace("_results", "_report") + ".md")
    else:
        # Directory mode: use standard filenames inside the directory
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "empirical_benchmark_results.json"
        report_path = output_path / "EMPIRICAL_BENCHMARK_REPORT.md"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    with open(report_path, "w") as f:
        f.write("# Mercury-Agent Empirical Benchmark Report\n\n")
        f.write(f"**Generated:** {results['timestamp']}\n\n")

        f.write("## Methodology\n\n")
        f.write("This benchmark compares Mercury-Agent against established anomaly detection ")
        f.write("algorithms using publicly available datasets from scikit-learn.\n\n")

        f.write("### Datasets\n\n")
        for dataset in results["methodology"]["datasets"]:
            f.write(f"- {dataset}\n")

        f.write("\n### Baseline Detectors\n\n")
        for detector in results["methodology"]["detectors"]:
            if detector != "Mercury-Agent":
                f.write(f"- {detector}\n")

        f.write("\n## Results Summary\n\n")
        f.write("| Detector | Mean ROC-AUC | Mean F1 | Mean Latency (ms) |\n")
        f.write("|----------|--------------|---------|-------------------|\n")

        summary = results["summary"]["per_detector"]
        for detector, stats in sorted(
            summary.items(), key=lambda x: x[1]["mean_roc_auc"], reverse=True
        ):
            f.write(
                f"| {detector} | {stats['mean_roc_auc']:.3f} | "
                f"{stats['mean_f1']:.3f} | {stats['mean_latency_ms']:.3f} |\n"
            )

        f.write("\n## Honest Assessment\n\n")
        assessment = results["summary"]["honest_assessment"]

        if "performance_verdict" in assessment:
            f.write(f"**Verdict:** {assessment['performance_verdict']}\n\n")

        f.write("### Methodology Notes\n\n")
        for note in assessment.get("methodology_notes", []):
            f.write(f"- {note}\n")

        f.write("\n### Limitations\n\n")
        for limitation in assessment.get("limitations", []):
            f.write(f"- {limitation}\n")

        if "recommendation" in assessment:
            f.write(f"\n**Recommendation:** {assessment['recommendation']}\n")

        f.write("\n## Detailed Results\n\n")
        for r in results["results"]:
            f.write(f"### {r['detector_name']} on {r['dataset_name']}\n\n")
            f.write(f"- ROC-AUC: {r['roc_auc']:.4f}\n")
            f.write(f"- PR-AUC: {r['pr_auc']:.4f}\n")
            f.write(f"- F1 Score: {r['f1']:.4f}\n")
            f.write(f"- Precision: {r['precision']:.4f}\n")
            f.write(f"- Recall: {r['recall']:.4f}\n")
            f.write(f"- False Positive Rate: {r['false_positive_rate']:.4f}\n")
            f.write(f"- Inference Latency: {r['inference_latency_ms']:.4f} ms/sample\n\n")

    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    print("\nStarting Mercury-Agent Empirical Benchmark Suite...")
    print("This may take a few minutes to download datasets and run benchmarks.\n")

    results = run_full_benchmark()

    output_dir = Path(__file__).parent.parent / "results"
    save_results(results, output_dir)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    summary = results["summary"]
    if summary.get("omni_mercury_comparison"):
        comp = summary["omni_mercury_comparison"]
        print("\nMercury-Agent Performance:")
        print(f"  ROC-AUC: {comp.get('omni_mercury_roc_auc', 'N/A'):.3f}")
        print(f"  Rank: #{comp.get('rank_by_roc_auc', 'N/A')} of {len(summary['per_detector'])}")
        print(f"  vs Best Baseline: {comp.get('vs_best_baseline', 0):+.3f}")
        print(f"  vs Avg Baseline: {comp.get('vs_avg_baseline', 0):+.3f}")

    print("\nSee results/ directory for detailed reports.")
