"""
Mercury Agent - Real-World Benchmark Runner
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Runs benchmarks on real-world datasets:
- SMD (Server Machine Dataset) - time-series anomaly detection
- NSL-KDD - network intrusion detection
- BATADAL - water infrastructure cyber-physical attacks

Provides:
- Quantified before/after metrics
- Statistical significance testing
- Reproducible evaluation with fixed seeds
- Domain-specific metrics (time-to-detection, event F1, etc.)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


logger = logging.getLogger(__name__)

# Fixed seed for reproducibility
GLOBAL_SEED = 42


@dataclass
class DatasetInfo:
    """Information about a benchmark dataset."""

    name: str
    domain: str  # cyber, timeseries, infrastructure
    n_samples: int = 0
    n_features: int = 0
    anomaly_ratio: float = 0.0
    description: str = ""
    # Provenance tracking for real data validation
    source: str = "unknown"  # "real-local", "real-download", "synthetic"
    checksum: str = ""
    used_synthetic: bool = False


@dataclass
class BenchmarkMetrics:
    """Metrics from a benchmark run."""

    # Core metrics
    roc_auc: float = 0.0
    pr_auc: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    brier_score: float = 0.0

    # Event-based (for time-series)
    event_f1: float = 0.0
    time_to_detection: float = 0.0

    # Ethical metrics
    benevolence_score: float = 1.0
    sigma_immutable: float = 0.96

    # Timing
    fit_time_ms: float = 0.0
    predict_time_ms: float = 0.0

    # Confidence intervals
    roc_auc_ci: tuple[float, float] = (0.0, 0.0)
    f1_ci: tuple[float, float] = (0.0, 0.0)

    def improvement_over(self, baseline: BenchmarkMetrics) -> dict[str, float]:
        """Compute improvement percentages over baseline."""
        improvements = {}
        for metric in ["roc_auc", "pr_auc", "f1", "precision", "recall", "event_f1"]:
            base = getattr(baseline, metric)
            current = getattr(self, metric)
            if base > 0:
                improvements[metric] = (current - base) / base * 100
            else:
                improvements[metric] = 0.0

        # Brier is lower=better
        if baseline.brier_score > 0:
            improvements["brier_score"] = (
                -(self.brier_score - baseline.brier_score) / baseline.brier_score * 100
            )

        # Time-to-detection is lower=better
        if baseline.time_to_detection > 0:
            improvements["time_to_detection"] = (
                -(self.time_to_detection - baseline.time_to_detection)
                / baseline.time_to_detection
                * 100
            )

        return improvements


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""

    dataset: DatasetInfo
    detector_name: str
    metrics: BenchmarkMetrics
    n_folds: int
    seed: int
    fold_results: list[dict[str, float]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with full provenance tracking."""
        return {
            "dataset": self.dataset.name,
            "detector": self.detector_name,
            "metrics": {
                "roc_auc": self.metrics.roc_auc,
                "pr_auc": self.metrics.pr_auc,
                "f1": self.metrics.f1,
                "precision": self.metrics.precision,
                "recall": self.metrics.recall,
                "brier_score": self.metrics.brier_score,
                "event_f1": self.metrics.event_f1,
                "time_to_detection": self.metrics.time_to_detection,
                "benevolence_score": self.metrics.benevolence_score,
                "sigma_immutable": self.metrics.sigma_immutable,
            },
            "n_folds": self.n_folds,
            "seed": self.seed,
            "provenance": {
                "source": self.dataset.source,
                "checksum": self.dataset.checksum,
                "used_synthetic": self.dataset.used_synthetic,
                "n_samples": self.dataset.n_samples,
                "n_features": self.dataset.n_features,
                "anomaly_ratio": self.dataset.anomaly_ratio,
            },
            "timestamp": self.timestamp,
        }


class SyntheticDataGenerator:
    """
    Generate synthetic benchmark data mimicking real-world datasets.

    Used when real data files are not available.
    """

    def __init__(self, seed: int = GLOBAL_SEED):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_smd_like(
        self, n_samples: int = 5000, n_features: int = 38
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate data mimicking SMD (Server Machine Dataset).

        SMD characteristics:
        - Multi-dimensional time-series
        - Point anomalies and contextual anomalies
        - ~5% anomaly ratio
        """
        X = self.rng.standard_normal((n_samples, n_features))

        # Add temporal correlations
        for i in range(1, n_samples):
            X[i] = 0.7 * X[i - 1] + 0.3 * X[i]

        # Add periodic patterns
        t = np.linspace(0, 10 * np.pi, n_samples)
        for j in range(min(5, n_features)):
            X[:, j] += np.sin(t * (j + 1)) * 0.5

        # Generate anomalies (~5%)
        y = np.zeros(n_samples, dtype=int)
        n_anomalies = int(n_samples * 0.05)

        # Point anomalies
        anomaly_idx = self.rng.choice(n_samples, size=n_anomalies // 2, replace=False)
        for idx in anomaly_idx:
            X[idx] += self.rng.standard_normal(n_features) * 3
            y[idx] = 1

        # Contextual anomalies (contiguous segments)
        n_segments = n_anomalies // 20
        for _ in range(n_segments):
            start = self.rng.integers(100, n_samples - 100)
            length = self.rng.integers(5, 20)
            X[start : start + length] *= 2
            y[start : start + length] = 1

        return X, y

    def generate_nslkdd_like(
        self, n_samples: int = 5000, n_features: int = 41
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate data mimicking NSL-KDD.

        NSL-KDD characteristics:
        - Network traffic features
        - Multiple attack types (we simplify to binary)
        - ~20% attack ratio
        """
        # Normal traffic
        n_normal = int(n_samples * 0.8)
        X_normal = self.rng.standard_normal((n_normal, n_features))
        X_normal[:, :10] = np.abs(X_normal[:, :10])  # Counts are positive

        # Attack traffic
        n_attacks = n_samples - n_normal
        X_attack = self.rng.standard_normal((n_attacks, n_features))
        X_attack[:, :10] = np.abs(X_attack[:, :10]) * 2  # Higher counts
        X_attack[:, 10:20] += 1.5  # Different feature distribution

        X = np.vstack([X_normal, X_attack])
        y = np.concatenate([np.zeros(n_normal), np.ones(n_attacks)]).astype(int)

        # Shuffle
        shuffle_idx = self.rng.permutation(n_samples)
        X = X[shuffle_idx]
        y = y[shuffle_idx]

        return X, y

    def generate_batadal_like(
        self, n_samples: int = 5000, n_features: int = 43
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate data mimicking BATADAL.

        BATADAL characteristics:
        - Water infrastructure sensors
        - Cyber-physical attacks
        - ~10% attack ratio with long segments
        """
        # Base sensor readings
        X = self.rng.standard_normal((n_samples, n_features)) * 0.5

        # Add realistic sensor patterns
        t = np.linspace(0, 20 * np.pi, n_samples)

        # Tank levels (oscillating)
        for j in range(7):
            X[:, j] = 50 + 10 * np.sin(t * 0.5 + j) + self.rng.standard_normal(n_samples) * 2

        # Pump status (mostly stable)
        for j in range(7, 14):
            X[:, j] = self.rng.choice([0, 1], size=n_samples, p=[0.3, 0.7]).astype(float)

        # Flow rates (correlated with pumps)
        for j in range(14, 21):
            X[:, j] = X[:, j - 7] * 10 + self.rng.standard_normal(n_samples) * 2

        # Pressure readings
        for j in range(21, n_features):
            X[:, j] = 30 + self.rng.standard_normal(n_samples) * 5

        # Generate attack segments (~10%)
        y = np.zeros(n_samples, dtype=int)
        n_attack_segments = 5
        attack_length = int(n_samples * 0.02)

        for _ in range(n_attack_segments):
            start = self.rng.integers(100, n_samples - attack_length - 100)

            # Attack: manipulate sensor readings
            X[start : start + attack_length, :7] += 15  # Tank levels
            X[start : start + attack_length, 14:21] *= 0.5  # Flow rates
            y[start : start + attack_length] = 1

        return X, y


class RealWorldBenchmarkRunner:
    """
    Benchmark runner for real-world datasets.

    Supports:
    - SMD (Server Machine Dataset)
    - NSL-KDD (Network Intrusion)
    - BATADAL (Water Infrastructure)

    Uses synthetic data when real files unavailable.
    """

    def __init__(
        self,
        n_folds: int = 10,
        seed: int = GLOBAL_SEED,
        data_dir: str | Path | None = None,
        use_synthetic: bool = False,
        min_real_samples: int = 100,
    ):
        """
        Initialize the benchmark runner.

        Args:
            n_folds: Number of cross-validation folds
            seed: Random seed for reproducibility
            data_dir: Directory containing real dataset files
            use_synthetic: If True, allow synthetic data fallback (default: False)
                          When False, raises RuntimeError if real data unavailable
            min_real_samples: Minimum required real samples (fails if not met)
        """
        self.n_folds = n_folds
        self.seed = seed
        self.data_dir = Path(data_dir) if data_dir else None
        self.use_synthetic = use_synthetic
        self.min_real_samples = min_real_samples

        self.synthetic_generator = SyntheticDataGenerator(seed)

        # Available datasets
        self.datasets: dict[str, DatasetInfo] = {
            "SMD": DatasetInfo(
                name="SMD",
                domain="timeseries",
                description="Server Machine Dataset - multi-dimensional time-series",
            ),
            "NSL-KDD": DatasetInfo(
                name="NSL-KDD", domain="cyber", description="Network Intrusion Detection"
            ),
            "BATADAL": DatasetInfo(
                name="BATADAL",
                domain="infrastructure",
                description="Water Infrastructure Cyber-Physical Attacks",
            ),
        }

        logger.info(
            f"RealWorldBenchmarkRunner initialized: n_folds={n_folds}, seed={seed}, "
            f"use_synthetic={use_synthetic}, min_real_samples={min_real_samples}"
        )

    def load_dataset(self, name: str) -> tuple[np.ndarray, np.ndarray, DatasetInfo]:
        """
        Load a benchmark dataset.

        Args:
            name: Dataset name (SMD, NSL-KDD, BATADAL)

        Returns:
            Tuple of (X, y, info)

        Raises:
            RuntimeError: If real data unavailable and use_synthetic=False
        """
        import hashlib

        name = name.upper().replace("-", "_").replace(" ", "_")

        # Try to load real data
        real_data = None
        if self.data_dir:
            real_data = self._try_load_real(name)

        if real_data is not None:
            X, y = real_data
            info = self.datasets.get(
                name.replace("_", "-"), DatasetInfo(name=name, domain="unknown")
            )
            info.n_samples = len(X)
            info.n_features = X.shape[1]
            info.anomaly_ratio = float(np.mean(y))
            info.source = "real-local"
            info.checksum = hashlib.sha256(X.tobytes()).hexdigest()[:16]
            info.used_synthetic = False

            # Validate minimum samples
            if len(X) < self.min_real_samples:
                raise RuntimeError(
                    f"{name}: Loaded {len(X)} samples but minimum is {self.min_real_samples}. "
                    f"Dataset may be incomplete or corrupted."
                )

            logger.info(
                f"Loaded REAL data for {name}: {info.n_samples} samples, "
                f"{info.n_features} features, checksum={info.checksum}"
            )
            return X, y, info

        # Real data not available - check if synthetic fallback is allowed
        if not self.use_synthetic:
            raise RuntimeError(
                f"REAL DATA REQUIRED: {name} dataset not found and use_synthetic=False. "
                f"To run benchmarks on real data, either:\n"
                f"  1. Set data_dir to a directory containing {name.lower()}.npz or {name.lower()}_train.csv\n"
                f"  2. Download the dataset from its official source\n"
                f"  3. Set use_synthetic=True to allow synthetic fallback (NOT RECOMMENDED for validation)\n"
                f"\nDataset sources:\n"
                f"  - SMD: https://github.com/NetManAIOps/OmniAnomaly\n"
                f"  - NSL-KDD: https://www.unb.ca/cic/datasets/nsl.html\n"
                f"  - BATADAL: https://www.batadal.net/\n"
                f"\nSynthetic data is NOT acceptable for production validation per Civilization-First principles."
            )

        # Fall back to synthetic (only if explicitly allowed)
        logger.warning(
            f"SYNTHETIC FALLBACK: Using synthetic data for {name}. "
            f"Results should NOT be used for production validation claims."
        )
        X, y = self._generate_synthetic(name)

        info = self.datasets.get(name.replace("_", "-"), DatasetInfo(name=name, domain="unknown"))
        info.n_samples = len(X)
        info.n_features = X.shape[1]
        info.anomaly_ratio = float(np.mean(y))
        info.source = "synthetic"
        info.checksum = hashlib.sha256(X.tobytes()).hexdigest()[:16]
        info.used_synthetic = True

        return X, y, info

    def _try_load_real(self, name: str) -> tuple[np.ndarray, np.ndarray] | None:
        """Attempt to load real data files."""
        if not self.data_dir or not self.data_dir.exists():
            return None

        # Check for common file patterns
        patterns = [
            f"{name.lower()}.npz",
            f"{name.lower()}_train.csv",
            f"{name.lower()}/data.npy",
        ]

        for pattern in patterns:
            file_path = self.data_dir / pattern
            if file_path.exists():
                try:
                    if pattern.endswith(".npz"):
                        data = np.load(file_path)
                        return data["X"], data["y"]
                    elif pattern.endswith(".csv"):
                        import pandas as pd

                        df = pd.read_csv(file_path)
                        X = df.drop("label", axis=1, errors="ignore").values
                        y = df.get("label", np.zeros(len(df))).values
                        return X, y.astype(int)
                    elif pattern.endswith(".npy"):
                        X = np.load(file_path)
                        labels_path = file_path.parent / "labels.npy"
                        y = np.load(labels_path) if labels_path.exists() else np.zeros(len(X))
                        return X, y.astype(int)
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")

        return None

    def _generate_synthetic(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic data for dataset."""
        name_key = name.upper().replace("-", "").replace("_", "")

        if name_key == "SMD":
            return self.synthetic_generator.generate_smd_like()
        elif name_key in ["NSLKDD", "NSL_KDD"]:
            return self.synthetic_generator.generate_nslkdd_like()
        elif name_key == "BATADAL":
            return self.synthetic_generator.generate_batadal_like()
        else:
            # Generic synthetic data
            n_samples = 5000
            n_features = 40
            X = self.synthetic_generator.rng.standard_normal((n_samples, n_features))
            y = (self.synthetic_generator.rng.random(n_samples) > 0.95).astype(int)
            return X, y

    def run_benchmark(
        self,
        detector: Any,
        dataset_name: str,
        detector_name: str = "Unknown",
    ) -> BenchmarkResult:
        """
        Run benchmark on a dataset.

        Args:
            detector: Detector with fit/predict/predict_proba methods
            dataset_name: Name of dataset
            detector_name: Name for reporting

        Returns:
            BenchmarkResult with all metrics
        """
        np.random.seed(self.seed)

        X, y, info = self.load_dataset(dataset_name)

        logger.info(
            f"Running benchmark: {detector_name} on {dataset_name} "
            f"(n={info.n_samples}, d={info.n_features}, anomaly_ratio={info.anomaly_ratio:.2%})"
        )

        # Stratified K-fold
        from sklearn.metrics import (
            average_precision_score,
            brier_score_loss,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Fit
            fit_start = time.perf_counter()
            try:
                detector.fit(X_train, y_train)
            except TypeError:
                detector.fit(X_train)
            fit_time = (time.perf_counter() - fit_start) * 1000

            # Predict
            predict_start = time.perf_counter()

            try:
                # Try predict_proba first
                y_proba = detector.predict_proba(X_test)
                if y_proba.ndim == 2:
                    y_proba = y_proba[:, 1]
                y_pred = (y_proba > 0.5).astype(int)
            except (AttributeError, NotImplementedError):
                # Fall back to predict
                result = detector.predict(X_test)
                if isinstance(result, list):
                    y_proba = np.array([r.anomaly_score for r in result])
                    y_pred = np.array([r.is_anomaly for r in result]).astype(int)
                else:
                    y_pred = result
                    y_proba = result.astype(float)

            predict_time = (time.perf_counter() - predict_start) * 1000

            # Handle sklearn's -1/1 convention
            if set(np.unique(y_pred)) == {-1, 1}:
                y_pred = (y_pred == -1).astype(int)

            # Compute metrics
            try:
                roc_auc = roc_auc_score(y_test, y_proba)
            except ValueError:
                roc_auc = 0.5

            try:
                pr_auc = average_precision_score(y_test, y_proba)
            except ValueError:
                pr_auc = 0.0

            f1 = f1_score(y_test, y_pred, zero_division=0.0)
            precision = precision_score(y_test, y_pred, zero_division=0.0)
            recall = recall_score(y_test, y_pred, zero_division=0.0)

            try:
                brier = brier_score_loss(y_test, y_proba)
            except ValueError:
                brier = 0.25

            # Event-based metrics for time-series
            event_f1, ttd = self._compute_event_metrics(y_test, y_pred)

            fold_results.append(
                {
                    "roc_auc": roc_auc,
                    "pr_auc": pr_auc,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                    "brier_score": brier,
                    "event_f1": event_f1,
                    "time_to_detection": ttd,
                    "fit_time_ms": fit_time,
                    "predict_time_ms": predict_time,
                }
            )

            logger.debug(f"Fold {fold_idx + 1}/{self.n_folds}: " f"AUC={roc_auc:.3f}, F1={f1:.3f}")

        # Aggregate results
        metrics = BenchmarkMetrics(
            roc_auc=np.mean([f["roc_auc"] for f in fold_results]),
            pr_auc=np.mean([f["pr_auc"] for f in fold_results]),
            f1=np.mean([f["f1"] for f in fold_results]),
            precision=np.mean([f["precision"] for f in fold_results]),
            recall=np.mean([f["recall"] for f in fold_results]),
            brier_score=np.mean([f["brier_score"] for f in fold_results]),
            event_f1=np.mean([f["event_f1"] for f in fold_results]),
            time_to_detection=np.mean([f["time_to_detection"] for f in fold_results]),
            fit_time_ms=np.mean([f["fit_time_ms"] for f in fold_results]),
            predict_time_ms=np.mean([f["predict_time_ms"] for f in fold_results]),
        )

        # Compute confidence intervals
        auc_values = [f["roc_auc"] for f in fold_results]
        f1_values = [f["f1"] for f in fold_results]

        if len(auc_values) > 1:
            ci = stats.t.interval(
                0.95, len(auc_values) - 1, loc=np.mean(auc_values), scale=stats.sem(auc_values)
            )
            metrics.roc_auc_ci = (ci[0], ci[1])

            ci = stats.t.interval(
                0.95, len(f1_values) - 1, loc=np.mean(f1_values), scale=stats.sem(f1_values)
            )
            metrics.f1_ci = (ci[0], ci[1])

        # Get benevolence if detector supports it
        try:
            if hasattr(detector, "get_gosnn_scalars"):
                scalars = detector.get_gosnn_scalars()
                metrics.benevolence_score = scalars.get("benevolence", 1.0)
        except Exception as e:
            logger.debug("Benevolence scoring unavailable: %s", e)

        result = BenchmarkResult(
            dataset=info,
            detector_name=detector_name,
            metrics=metrics,
            n_folds=self.n_folds,
            seed=self.seed,
            fold_results=fold_results,
        )

        logger.info(
            f"{detector_name} on {dataset_name}: "
            f"AUC={metrics.roc_auc:.3f}±{np.std(auc_values):.3f}, "
            f"F1={metrics.f1:.3f}±{np.std(f1_values):.3f}"
        )

        return result

    def _compute_event_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> tuple[float, float]:
        """Compute event-based F1 and time-to-detection."""

        # Extract events
        def get_events(arr: np.ndarray) -> list[tuple[int, int]]:
            events: list[tuple[int, int]] = []
            in_event = False
            start = 0
            for i, v in enumerate(arr):
                if v == 1 and not in_event:
                    start = i
                    in_event = True
                elif v == 0 and in_event:
                    events.append((start, i - 1))
                    in_event = False
            if in_event:
                events.append((start, len(arr) - 1))
            return events

        true_events = get_events(y_true)
        pred_events = get_events(y_pred)

        if not true_events:
            return 1.0 if not pred_events else 0.0, 0.0

        if not pred_events:
            return 0.0, float(np.mean([e[1] - e[0] + 1 for e in true_events]))

        # Event F1
        detected = sum(
            1
            for te in true_events
            if any(not (te[1] < pe[0] or pe[1] < te[0]) for pe in pred_events)
        )
        event_recall = detected / len(true_events)

        matched = sum(
            1
            for pe in pred_events
            if any(not (pe[1] < te[0] or te[1] < pe[0]) for te in true_events)
        )
        event_precision = matched / len(pred_events)

        if event_precision + event_recall > 0:
            event_f1 = 2 * event_precision * event_recall / (event_precision + event_recall)
        else:
            event_f1 = 0.0

        # Time-to-detection
        ttd_values = []
        for te_start, te_end in true_events:
            event_preds = y_pred[te_start : te_end + 1]
            detected_idx = np.where(event_preds == 1)[0]
            if len(detected_idx) > 0:
                ttd_values.append(detected_idx[0])
            else:
                ttd_values.append(te_end - te_start + 1)

        ttd = float(np.mean(ttd_values)) if ttd_values else 0.0

        return event_f1, ttd

    def compare_detectors(
        self,
        result_a: BenchmarkResult,
        result_b: BenchmarkResult,
        metric: str = "f1",
    ) -> dict[str, Any]:
        """
        Statistical comparison between two detectors.

        Args:
            result_a: First result
            result_b: Second result
            metric: Metric to compare

        Returns:
            Comparison statistics
        """
        values_a = [f[metric] for f in result_a.fold_results]
        values_b = [f[metric] for f in result_b.fold_results]

        # Paired t-test
        t_stat, t_pvalue = stats.ttest_rel(values_a, values_b)

        # Wilcoxon
        try:
            w_pvalue = stats.wilcoxon(values_a, values_b).pvalue
        except ValueError:
            w_pvalue = 1.0

        # Effect size
        diff = np.array(values_a) - np.array(values_b)
        cohens_d = np.mean(diff) / (np.std(diff) + 1e-10)

        improvement = (np.mean(values_a) - np.mean(values_b)) / max(np.mean(values_b), 1e-10) * 100

        return {
            "detector_a": result_a.detector_name,
            "detector_b": result_b.detector_name,
            "metric": metric,
            "mean_a": np.mean(values_a),
            "mean_b": np.mean(values_b),
            "improvement_percent": improvement,
            "t_test_pvalue": t_pvalue,
            "wilcoxon_pvalue": w_pvalue,
            "cohens_d": cohens_d,
            "significant": t_pvalue < 0.05 or w_pvalue < 0.05,
        }


def run_all_benchmarks(
    detector: Any,
    detector_name: str = "Mercury",
    n_folds: int = 10,
    seed: int = GLOBAL_SEED,
) -> dict[str, BenchmarkResult]:
    """
    Run benchmarks on all available datasets.

    Args:
        detector: Detector to evaluate
        detector_name: Name for reporting
        n_folds: Number of CV folds
        seed: Random seed

    Returns:
        Dictionary of dataset name to result
    """
    runner = RealWorldBenchmarkRunner(n_folds=n_folds, seed=seed)

    results = {}
    for dataset_name in ["SMD", "NSL-KDD", "BATADAL"]:
        try:
            results[dataset_name] = runner.run_benchmark(detector, dataset_name, detector_name)
        except Exception as e:
            logger.error(f"Benchmark failed for {dataset_name}: {e}")

    return results
