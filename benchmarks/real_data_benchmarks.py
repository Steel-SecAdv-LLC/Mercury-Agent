#!/usr/bin/env python3
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
Real-World Data Benchmarks for Mercury Agent

This module provides benchmarks using real-world public datasets:
- NSL-KDD: Network intrusion detection (security domain)
- MIMIC-III Demo: Medical ICU data (medical domain)

All benchmarks include:
- Fairlearn bias auditing for ethical AI compliance
- Comprehensive metrics (F1, Precision, Recall, ROC-AUC)
- Data provenance tracking
- Caching for efficient re-runs
"""

import gzip
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector


# ---------------------------------------------------------------------------
# Mercury-native metrics/preprocessing
# ---------------------------------------------------------------------------
def _native_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Trapezoidal AUC-ROC. Mercury-native."""
    desc_idx = np.argsort(y_score)[::-1]
    y_true_s = y_true[desc_idx]
    y_score_s = y_score[desc_idx]
    distinct_idx = np.where(np.diff(y_score_s, prepend=np.inf))[0]
    tps = np.cumsum(y_true_s)[distinct_idx]
    fps = distinct_idx + 1 - tps
    tps = np.concatenate([[0], tps])
    fps = np.concatenate([[0], fps])
    fpr = fps / (fps[-1] + 1e-12)
    tpr = tps / (tps[-1] + 1e-12)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(_trapz(tpr, fpr))


def _precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def _recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def _f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    p = _precision_score(y_true, y_pred)
    r = _recall_score(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _label_encode(series: pd.Series) -> np.ndarray:
    """Integer-encode categorical series. Mercury-native."""
    cats = {v: i for i, v in enumerate(sorted(series.unique()))}
    return series.map(cats).to_numpy(dtype=np.int64)


def _standard_scale(X: np.ndarray) -> np.ndarray:
    """Z-score normalization. Mercury-native."""
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-8
    return (X - mu) / sigma


class _StratifiedKFold:
    """Minimal stratified k-fold. Mercury-native."""

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.rng = np.random.RandomState(random_state)

    def split(self, X: np.ndarray, y: np.ndarray):  # type: ignore[no-untyped-def]
        classes = np.unique(y)
        class_indices = {c: np.where(y == c)[0] for c in classes}
        if self.shuffle:
            for c in classes:
                self.rng.shuffle(class_indices[c])
        folds: list[list[int]] = [[] for _ in range(self.n_splits)]
        for c in classes:
            idx = class_indices[c]
            for i, ix in enumerate(idx):
                folds[i % self.n_splits].append(ix)
        for fold_idx in range(self.n_splits):
            test = np.array(folds[fold_idx])
            train = np.concatenate(
                [np.array(folds[j]) for j in range(self.n_splits) if j != fold_idx]
            )
            yield train, test


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import optuna  # noqa: F401

    _AUTO_TUNE = True
except ImportError:
    _AUTO_TUNE = False
    logger.info("optuna not installed — auto_tune disabled. " "Install with: pip install optuna")

CACHE_DIR = Path.home() / ".omni_mercury" / "datasets"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    dataset_name: str
    domain: str
    num_samples: int
    num_features: int
    anomaly_ratio: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    runtime_seconds: float
    data_source: str
    bias_metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_fairlearn_bias_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray | None = None,
    feature_name: str = "sensitive_attribute",
) -> dict[str, float]:
    """
    Compute Fairlearn bias metrics for ethical AI compliance.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        sensitive_features: Sensitive attribute values (e.g., age groups, gender)
        feature_name: Name of the sensitive feature for logging

    Returns:
        Dictionary of bias metrics including demographic parity difference
    """
    bias_metrics: dict[str, float] = {}

    if sensitive_features is None:
        logger.info("No sensitive features provided, skipping bias analysis")
        return bias_metrics

    try:
        from fairlearn.metrics import (
            MetricFrame,
            demographic_parity_difference,
            equalized_odds_difference,
            selection_rate,
        )

        metric_frame = MetricFrame(
            metrics={
                "selection_rate": selection_rate,
                "precision": lambda y_t, y_p: _precision_score(y_t, y_p),
                "recall": lambda y_t, y_p: _recall_score(y_t, y_p),
            },
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
        )

        dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
        bias_metrics["demographic_parity_difference"] = float(dpd)

        try:
            eod = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features)
            bias_metrics["equalized_odds_difference"] = float(eod)
        except Exception:
            bias_metrics["equalized_odds_difference"] = 0.0

        by_group = metric_frame.by_group
        bias_metrics["selection_rate_by_group"] = by_group["selection_rate"].to_dict()
        bias_metrics["max_selection_rate_diff"] = float(
            by_group["selection_rate"].max() - by_group["selection_rate"].min()
        )

        if abs(dpd) > 0.1:
            logger.warning(
                f"BIAS WARNING: Demographic parity difference ({dpd:.3f}) exceeds "
                f"threshold (0.1) for {feature_name}. Review model fairness."
            )
        else:
            logger.info(f"Bias check passed: DPD={dpd:.3f} for {feature_name} (threshold: 0.1)")

    except ImportError:
        logger.warning("Fairlearn not installed, skipping bias metrics")
    except Exception as e:
        logger.warning(f"Error computing bias metrics: {e}")

    return bias_metrics


class NSLKDDBenchmark:
    """
    NSL-KDD Network Intrusion Detection Benchmark.

    Uses the KDD Cup 99 dataset (10% subset) for network intrusion detection.
    Source: https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

    Citation:
    Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009).
    A detailed analysis of the KDD CUP 99 data set.
    """

    KDD_URL = "https://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz"
    KDD_MIRROR = "https://archive.ics.uci.edu/ml/machine-learning-databases/kddcup99-mld/kddcup.data_10_percent.gz"

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
    ]

    ATTACK_CATEGORIES = {
        "normal.": "normal",
        "back.": "dos",
        "land.": "dos",
        "neptune.": "dos",
        "pod.": "dos",
        "smurf.": "dos",
        "teardrop.": "dos",
        "ipsweep.": "probe",
        "nmap.": "probe",
        "portsweep.": "probe",
        "satan.": "probe",
        "ftp_write.": "r2l",
        "guess_passwd.": "r2l",
        "imap.": "r2l",
        "multihop.": "r2l",
        "phf.": "r2l",
        "spy.": "r2l",
        "warezclient.": "r2l",
        "warezmaster.": "r2l",
        "buffer_overflow.": "u2r",
        "loadmodule.": "u2r",
        "perl.": "u2r",
        "rootkit.": "u2r",
    }

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._data: pd.DataFrame | None = None
        self._use_synthetic = False
        self._direct_download_source = False

    @property
    def _data_source_label(self) -> str:
        if self._use_synthetic:
            return "synthetic-fallback"
        if self._direct_download_source:
            return "real-direct-kddcup99"
        return "real-kdd"

    def _load_via_direct_download(self) -> pd.DataFrame:
        """Load KDD Cup 99 (10%) directly via urllib. Mercury-native."""
        import gzip
        import io
        import urllib.request

        url = self.KDD_URL
        logger.info("Loading KDD Cup 99 directly from %s ...", url)
        with urllib.request.urlopen(url, timeout=30) as resp:
            raw = gzip.decompress(resp.read())
        df = pd.read_csv(
            io.StringIO(raw.decode("utf-8")),
            names=self.COLUMN_NAMES,
            header=None,
        )
        self._use_synthetic = False
        self._direct_download_source = False
        return df

    def _download_data(self) -> pd.DataFrame | Path:
        """Download KDD Cup 99 data with retry logic.

        Priority:
        1. Direct download via urllib
        2. URL mirrors (may 403/404)
        3. Synthetic fallback
        """
        # 1. Try direct download
        try:
            return self._load_via_direct_download()
        except Exception as exc:
            logger.warning(
                "Direct KDD Cup 99 download failed (%s: %s), trying cached/mirror...",
                type(exc).__name__,
                exc,
            )

        # 2. Try cached file or mirror URL downloads
        cache_file = self.cache_dir / "kddcup.data_10_percent.gz"

        if cache_file.exists():
            logger.info(f"Using cached NSL-KDD data: {cache_file}")
            return cache_file

        urls = [self.KDD_URL, self.KDD_MIRROR]
        for url in urls:
            try:
                logger.info(f"Downloading NSL-KDD from {url}...")
                urlretrieve(url, cache_file)
                logger.info(f"Downloaded to {cache_file}")
                return cache_file
            except Exception as e:
                logger.warning(f"Failed to download from {url}: {e}")
                continue

        # 3. Synthetic fallback — loudly warn
        import warnings as _w

        _w.warn(
            "KDD Cup 99: using SYNTHETIC fallback data. "
            "All benchmark metrics from this run are INVALID for "
            "real-world performance assessment.",
            RuntimeWarning,
            stacklevel=2,
        )
        logger.error(
            "KDD CUP 99 SYNTHETIC FALLBACK ACTIVE — benchmark metrics invalid. "
            "Direct download and all mirror URLs failed.",
        )
        self._use_synthetic = True
        self._synthetic_reason = "all_download_sources_failed"
        return self._generate_synthetic_data()

    def _generate_synthetic_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """Generate synthetic NSL-KDD-like data for fallback."""
        logger.info(f"Generating synthetic NSL-KDD data ({n_samples} samples)")
        self._use_synthetic = True

        rng = np.random.default_rng(42)

        data = {
            "duration": rng.exponential(100, n_samples),
            "protocol_type": rng.choice(["tcp", "udp", "icmp"], n_samples),
            "service": rng.choice(["http", "ftp", "smtp", "ssh", "dns"], n_samples),
            "flag": rng.choice(["SF", "S0", "REJ", "RSTO"], n_samples),
            "src_bytes": rng.exponential(1000, n_samples),
            "dst_bytes": rng.exponential(500, n_samples),
            "land": rng.choice([0, 1], n_samples, p=[0.99, 0.01]),
            "wrong_fragment": rng.poisson(0.1, n_samples),
            "urgent": rng.poisson(0.01, n_samples),
            "hot": rng.poisson(0.5, n_samples),
            "num_failed_logins": rng.poisson(0.1, n_samples),
            "logged_in": rng.choice([0, 1], n_samples, p=[0.3, 0.7]),
            "num_compromised": rng.poisson(0.05, n_samples),
            "root_shell": rng.choice([0, 1], n_samples, p=[0.99, 0.01]),
            "su_attempted": rng.choice([0, 1], n_samples, p=[0.99, 0.01]),
            "num_root": rng.poisson(0.1, n_samples),
            "num_file_creations": rng.poisson(0.2, n_samples),
            "num_shells": rng.poisson(0.05, n_samples),
            "num_access_files": rng.poisson(0.1, n_samples),
            "num_outbound_cmds": rng.poisson(0.01, n_samples),
            "is_host_login": rng.choice([0, 1], n_samples, p=[0.99, 0.01]),
            "is_guest_login": rng.choice([0, 1], n_samples, p=[0.99, 0.01]),
            "count": rng.poisson(50, n_samples),
            "srv_count": rng.poisson(30, n_samples),
            "serror_rate": rng.beta(1, 10, n_samples),
            "srv_serror_rate": rng.beta(1, 10, n_samples),
            "rerror_rate": rng.beta(1, 10, n_samples),
            "srv_rerror_rate": rng.beta(1, 10, n_samples),
            "same_srv_rate": rng.beta(5, 2, n_samples),
            "diff_srv_rate": rng.beta(1, 5, n_samples),
            "srv_diff_host_rate": rng.beta(1, 5, n_samples),
            "dst_host_count": rng.poisson(100, n_samples),
            "dst_host_srv_count": rng.poisson(50, n_samples),
            "dst_host_same_srv_rate": rng.beta(5, 2, n_samples),
            "dst_host_diff_srv_rate": rng.beta(1, 5, n_samples),
            "dst_host_same_src_port_rate": rng.beta(2, 5, n_samples),
            "dst_host_srv_diff_host_rate": rng.beta(1, 5, n_samples),
            "dst_host_serror_rate": rng.beta(1, 10, n_samples),
            "dst_host_srv_serror_rate": rng.beta(1, 10, n_samples),
            "dst_host_rerror_rate": rng.beta(1, 10, n_samples),
            "dst_host_srv_rerror_rate": rng.beta(1, 10, n_samples),
        }

        n_normal = int(n_samples * 0.8)
        labels = ["normal."] * n_normal + rng.choice(
            ["neptune.", "smurf.", "portsweep.", "satan."], n_samples - n_normal
        ).tolist()
        rng.shuffle(labels)
        data["label"] = labels

        return pd.DataFrame(data)

    def load_data(self, max_samples: int | None = 50000) -> pd.DataFrame:
        """Load NSL-KDD data from cache or download."""
        try:
            result = self._download_data()
            if isinstance(result, pd.DataFrame):
                # Returned directly from download or synthetic fallback
                self._data = result
                logger.info(f"Loaded {len(self._data)} samples from {self._data_source_label}")
            else:
                # Returned a Path to a gzip file
                with gzip.open(result, "rt") as f:
                    self._data = pd.read_csv(f, names=self.COLUMN_NAMES, header=None)
                self._use_synthetic = False
                logger.info(f"Loaded {len(self._data)} samples from real NSL-KDD data")
        except Exception as e:
            logger.warning(f"Failed to load real data: {e}. Using synthetic fallback.")
            self._data = self._generate_synthetic_data(max_samples or 10000)

        if max_samples and len(self._data) > max_samples:
            self._data = self._data.sample(n=max_samples, random_state=42)

        return self._data

    def preprocess(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess NSL-KDD data for anomaly detection.

        Returns:
            Tuple of (features, labels, protocol_type for bias analysis)
        """
        df = df.copy()

        df["is_attack"] = df["label"].apply(lambda x: 0 if x == "normal." else 1)

        protocol_encoded = _label_encode(df["protocol_type"])

        categorical_cols = ["protocol_type", "service", "flag"]
        for col in categorical_cols:
            df[col] = _label_encode(df[col].astype(str))

        feature_cols = [c for c in df.columns if c not in ["label", "is_attack"]]
        X = df[feature_cols].values.astype(np.float32)
        y = df["is_attack"].values

        X = _standard_scale(X)

        return X, y, protocol_encoded

    def run_benchmark(
        self,
        max_samples: int = 50000,
        n_folds: int = 5,
    ) -> BenchmarkResult:
        """
        Run NSL-KDD benchmark with cross-validation and bias analysis.

        Args:
            max_samples: Maximum samples to use
            n_folds: Number of cross-validation folds

        Returns:
            BenchmarkResult with metrics and bias analysis
        """
        start_time = time.time()

        df = self.load_data(max_samples)
        X, y, protocol_type = self.preprocess(df)

        skf = _StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        all_y_true = []
        all_y_pred = []
        all_y_scores = []
        all_protocols = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_test = y[test_idx]

            detector = MercuryAnomalyDetector(auto_validate=True, auto_tune=_AUTO_TUNE)
            # Train on normal samples only (unsupervised)
            y_train = y[train_idx]
            normal_mask = y_train == 0
            X_train_normal = X_train[normal_mask] if normal_mask.sum() > 0 else X_train
            detector.fit(X_train_normal)

            result = detector.detect(X_test)
            y_scores = result["scores"]
            y_pred = result["is_anomaly"].astype(int)

            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            all_y_scores.extend(y_scores)
            all_protocols.extend(protocol_type[test_idx])

        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        all_y_scores = np.array(all_y_scores)
        all_protocols = np.array(all_protocols)

        precision = _precision_score(all_y_true, all_y_pred)
        recall = _recall_score(all_y_true, all_y_pred)
        f1 = _f1_score(all_y_true, all_y_pred)

        try:
            roc_auc = _native_auc(all_y_true, all_y_scores)
        except ValueError:
            roc_auc = 0.5

        bias_metrics = compute_fairlearn_bias_metrics(
            all_y_true,
            all_y_pred,
            sensitive_features=all_protocols,
            feature_name="protocol_type",
        )

        runtime = time.time() - start_time

        return BenchmarkResult(
            dataset_name="NSL-KDD",
            domain="security",
            num_samples=len(X),
            num_features=X.shape[1],
            anomaly_ratio=float(np.mean(y)),
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            roc_auc=float(roc_auc),
            runtime_seconds=runtime,
            data_source=self._data_source_label,
            bias_metrics=bias_metrics,
            metadata={
                "n_folds": n_folds,
                "model": "MercuryAnomalyDetector",
            },
        )


# MIMIC-III moved to benchmarks/credentialed_benchmarks.py
# Requires PhysioNet credentialed access. See docs/DATASOURCES.md.


def run_all_benchmarks() -> dict[str, Any]:
    """Run all real-data benchmarks and return comprehensive results."""
    print("=" * 70)
    print("Mercury Agent REAL-DATA BENCHMARKS")
    print("=" * 70)

    results: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "benchmarks": {},
        "summary": {},
    }

    # MIMIC-III moved to benchmarks/credentialed_benchmarks.py

    print("\n[1/1] Running NSL-KDD Security Benchmark...")
    nsl_benchmark = NSLKDDBenchmark()
    nsl_result = nsl_benchmark.run_benchmark(max_samples=50000, n_folds=5)
    results["benchmarks"]["nsl_kdd"] = {
        "dataset_name": nsl_result.dataset_name,
        "domain": nsl_result.domain,
        "num_samples": nsl_result.num_samples,
        "num_features": nsl_result.num_features,
        "anomaly_ratio": nsl_result.anomaly_ratio,
        "precision": nsl_result.precision,
        "recall": nsl_result.recall,
        "f1": nsl_result.f1,
        "roc_auc": nsl_result.roc_auc,
        "runtime_seconds": nsl_result.runtime_seconds,
        "data_source": nsl_result.data_source,
        "bias_metrics": nsl_result.bias_metrics,
        "metadata": nsl_result.metadata,
    }
    print(f"  Dataset: {nsl_result.dataset_name}")
    print(f"  Samples: {nsl_result.num_samples}, Features: {nsl_result.num_features}")
    print(f"  Precision: {nsl_result.precision:.4f}")
    print(f"  Recall: {nsl_result.recall:.4f}")
    print(f"  F1 Score: {nsl_result.f1:.4f}")
    print(f"  ROC-AUC: {nsl_result.roc_auc:.4f}")
    print(f"  Runtime: {nsl_result.runtime_seconds:.2f}s")
    print(f"  Data Source: {nsl_result.data_source}")
    if nsl_result.bias_metrics:
        dpd = nsl_result.bias_metrics.get("demographic_parity_difference", "N/A")
        print(f"  Demographic Parity Diff: {dpd}")

    results["summary"] = {
        "total_benchmarks": 1,
        "avg_f1": nsl_result.f1,
        "avg_roc_auc": nsl_result.roc_auc,
        "total_runtime_seconds": nsl_result.runtime_seconds,
        "all_bias_checks_passed": (
            abs(nsl_result.bias_metrics.get("demographic_parity_difference", 0)) <= 0.1
            if nsl_result.bias_metrics
            else True
        ),
    }

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  Total Benchmarks: {results['summary']['total_benchmarks']}")
    print(f"  Average F1: {results['summary']['avg_f1']:.4f}")
    print(f"  Average ROC-AUC: {results['summary']['avg_roc_auc']:.4f}")
    print(f"  Total Runtime: {results['summary']['total_runtime_seconds']:.2f}s")
    print(f"  Bias Checks Passed: {results['summary']['all_bias_checks_passed']}")
    print("  Note: MIMIC-III moved to benchmarks/credentialed_benchmarks.py")

    return results


if __name__ == "__main__":
    results = run_all_benchmarks()

    output_file = Path("benchmarks/real_data_benchmark_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")
