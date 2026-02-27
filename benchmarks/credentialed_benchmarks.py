"""
Benchmarks requiring external credentials or manual dataset download.

Run ONLY when datasets are locally available.
See docs/DATASOURCES.md for access instructions.

Available benchmarks:
- MIMICDemoBenchmark: Requires PhysioNet credentialed access.
  See https://physionet.org/content/mimiciii/

Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC
Licensed under GPL-3.0-or-later
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

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
    bias_metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_fairlearn_bias_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray | None = None,
    feature_name: str = "unknown",
) -> dict[str, float]:
    """Compute Fairlearn bias metrics for ethical AI compliance."""
    bias_metrics: dict[str, float] = {}
    if sensitive_features is None:
        return bias_metrics
    try:
        from fairlearn.metrics import (
            MetricFrame,
            demographic_parity_difference,
            selection_rate,
        )

        metric_frame = MetricFrame(
            metrics={"selection_rate": selection_rate},
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
        )

        dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
        bias_metrics["demographic_parity_difference"] = float(dpd)

        by_group = metric_frame.by_group
        bias_metrics["selection_rate_by_group"] = by_group["selection_rate"].to_dict()
    except ImportError:
        logger.warning("Fairlearn not installed, skipping bias metrics")
    except Exception as e:
        logger.warning(f"Error computing bias metrics: {e}")
    return bias_metrics


class MIMICDemoBenchmark:
    """
    MIMIC-III Demo Benchmark for Medical Anomaly Detection.

    Uses simulated vital signs data based on MIMIC-III patterns.
    Note: Full MIMIC-III requires credentialed access from PhysioNet.

    For production use, obtain credentials at:
    https://physionet.org/content/mimiciii-demo/

    Citation:
    Johnson, A. E. W., et al. (2016). MIMIC-III, a freely accessible
    critical care database. Scientific Data, 3, 160035.
    """

    VITAL_SIGNS = ["heart_rate", "sbp", "dbp", "resp_rate", "spo2", "temperature"]

    NORMAL_RANGES = {
        "heart_rate": (60, 100),
        "sbp": (90, 140),
        "dbp": (60, 90),
        "resp_rate": (12, 20),
        "spo2": (95, 100),
        "temperature": (36.1, 37.2),
    }

    SEPSIS_INDICATORS = {
        "heart_rate": (100, 140),
        "sbp": (70, 90),
        "dbp": (40, 60),
        "resp_rate": (22, 35),
        "spo2": (88, 94),
        "temperature": (38.0, 40.0),
    }

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _generate_patient_data(
        self,
        n_patients: int = 1000,
        sepsis_ratio: float = 0.15,
        time_steps: int = 24,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic patient vital signs data.

        Simulates MIMIC-III-like ICU monitoring data with:
        - Normal patients: vital signs within normal ranges
        - Sepsis patients: vital signs showing sepsis indicators

        Args:
            n_patients: Number of patients to simulate
            sepsis_ratio: Proportion of sepsis cases
            time_steps: Number of hourly measurements per patient

        Returns:
            Tuple of (features, labels, age_groups for bias analysis)
        """
        rng = np.random.default_rng(42)

        n_sepsis = int(n_patients * sepsis_ratio)
        n_normal = n_patients - n_sepsis

        all_features = []
        all_labels = []
        all_ages = []

        for _ in range(n_normal):
            patient_data = []
            for vital in self.VITAL_SIGNS:
                low, high = self.NORMAL_RANGES[vital]
                mean = (low + high) / 2
                std = (high - low) / 6
                values = rng.normal(mean, std, time_steps)
                values = np.clip(values, low * 0.9, high * 1.1)
                patient_data.extend(
                    [
                        np.mean(values),
                        np.std(values),
                        np.min(values),
                        np.max(values),
                        values[-1] - values[0],
                    ]
                )

            all_features.append(patient_data)
            all_labels.append(0)
            all_ages.append(rng.choice(["young", "middle", "elderly"], p=[0.2, 0.5, 0.3]))

        for _ in range(n_sepsis):
            patient_data = []
            for vital in self.VITAL_SIGNS:
                low, high = self.SEPSIS_INDICATORS[vital]
                mean = (low + high) / 2
                std = (high - low) / 4
                values = rng.normal(mean, std, time_steps)

                trend = np.linspace(0, rng.uniform(0.1, 0.3) * mean, time_steps)
                if vital in ["heart_rate", "resp_rate", "temperature"]:
                    values += trend
                else:
                    values -= trend

                patient_data.extend(
                    [
                        np.mean(values),
                        np.std(values),
                        np.min(values),
                        np.max(values),
                        values[-1] - values[0],
                    ]
                )

            all_features.append(patient_data)
            all_labels.append(1)
            all_ages.append(rng.choice(["young", "middle", "elderly"], p=[0.1, 0.3, 0.6]))

        X = np.array(all_features, dtype=np.float32)
        y = np.array(all_labels)
        ages = np.array(all_ages)

        shuffle_idx = rng.permutation(len(X))
        return X[shuffle_idx], y[shuffle_idx], ages[shuffle_idx]

    def run_benchmark(
        self,
        n_patients: int = 2000,
        sepsis_ratio: float = 0.15,
        n_folds: int = 5,
    ) -> BenchmarkResult:
        """
        Run MIMIC-III demo benchmark with cross-validation and bias analysis.

        Args:
            n_patients: Number of patients to simulate
            sepsis_ratio: Proportion of sepsis cases
            n_folds: Number of cross-validation folds

        Returns:
            BenchmarkResult with metrics and bias analysis
        """
        start_time = time.time()

        X, y, age_groups = self._generate_patient_data(n_patients, sepsis_ratio)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        all_y_true = []
        all_y_pred = []
        all_y_scores = []
        all_ages = []

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
            all_ages.extend(age_groups[test_idx])

        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        all_y_scores = np.array(all_y_scores)
        all_ages = np.array(all_ages)

        precision = precision_score(all_y_true, all_y_pred, zero_division=0)
        recall = recall_score(all_y_true, all_y_pred, zero_division=0)
        f1 = f1_score(all_y_true, all_y_pred, zero_division=0)

        try:
            roc_auc = roc_auc_score(all_y_true, all_y_scores)
        except ValueError:
            roc_auc = 0.5

        bias_metrics = compute_fairlearn_bias_metrics(
            all_y_true,
            all_y_pred,
            sensitive_features=all_ages,
            feature_name="age_group",
        )

        runtime = time.time() - start_time

        return BenchmarkResult(
            dataset_name="MIMIC-III Demo",
            domain="medical",
            num_samples=len(X),
            num_features=X.shape[1],
            anomaly_ratio=float(np.mean(y)),
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            roc_auc=float(roc_auc),
            runtime_seconds=runtime,
            data_source="synthetic-mimic-demo",
            bias_metrics=bias_metrics,
            metadata={
                "n_folds": n_folds,
                "model": "MercuryAnomalyDetector",
                "sepsis_ratio": sepsis_ratio,
                "vital_signs": self.VITAL_SIGNS,
                "note": "Simulated data based on MIMIC-III patterns. "
                "Full dataset requires PhysioNet credentials.",
            },
        )


if __name__ == "__main__":
    print("=" * 70)
    print("Mercury Agent CREDENTIALED BENCHMARKS")
    print("=" * 70)

    print("\n[1/1] Running MIMIC-III Demo Benchmark...")
    mimic = MIMICDemoBenchmark()
    result = mimic.run_benchmark()
    print(
        f"  MIMIC-III Demo: F1={result.f1:.4f} AUC={result.roc_auc:.4f} "
        f"Source={result.data_source}"
    )
