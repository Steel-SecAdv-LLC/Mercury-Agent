#!/usr/bin/env python3
"""
Mercury Agent v1.4 - Comprehensive live-data benchmark suite.

Measures detection performance across:
- 16 ADBench datasets (tabular anomalies)
- NSL-KDD (network security)
- CICIDS-2017 (network intrusion)

Detectors tested:
- StatisticalAnomalyDetector (baseline)
- TemporalAnomalyDetector (temporal patterns)

Output: benchmarks/v1.4_comprehensive_results.json with full metadata.

Copyright (C) 2025 Steel Security Advisors LLC
License: GPL-3.0+
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from omni_mercury_engine import __version__
from omni_mercury_engine.datasets.adbench import ADBenchLoader
from omni_mercury_engine.datasets.base import DatasetConfig
from omni_mercury_engine.datasets.security import CICIDSLoader, NSLKDDLoader
from omni_mercury_engine.detectors.statistical import StatisticalAnomalyDetector
from omni_mercury_engine.detectors.temporal import TemporalAnomalyDetector
from omni_mercury_engine.detectors.threshold_calibrator import find_optimal_threshold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class BenchmarkSuite:
    """Comprehensive benchmark harness for Mercury-Agent detectors."""

    def __init__(self) -> None:
        self.results: dict = {
            "generated_at": datetime.now(UTC).isoformat(),
            "version": __version__,
            "environment": self._get_environment(),
            "datasets": {},
            "summary": {},
        }

    def _get_environment(self) -> dict:
        """Capture system/environment info for reproducibility."""
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()[:7]

        env: dict = {
            "python_version": (
                f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            ),
            "git_commit": git_commit,
        }

        try:
            env["numpy_version"] = np.__version__
        except Exception:
            pass

        try:
            import torch

            env["pytorch_version"] = torch.__version__
            env["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                env["gpu_name"] = torch.cuda.get_device_name(0)
        except ImportError:
            env["cuda_available"] = False

        return env

    def _run_detector(
        self,
        detector_cls: type,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        """Run a single detector and compute metrics."""
        # Fit
        fit_start = time.time()
        detector = detector_cls()
        detector.fit(X_train)
        fit_time = time.time() - fit_start

        # Score via detect()
        score_start = time.time()
        result = detector.detect(X_test)
        scores = result["scores"]
        score_time = time.time() - score_start

        # Metrics
        try:
            auc = float(roc_auc_score(y_test, scores))
        except ValueError:
            auc = 0.0

        threshold = find_optimal_threshold(scores, y_test)
        predictions = (scores >= threshold).astype(int)
        f1 = float(f1_score(y_test, predictions, zero_division=0))
        accuracy = float(accuracy_score(y_test, predictions))

        return {
            "auc": auc,
            "f1": f1,
            "accuracy": accuracy,
            "threshold": threshold,
            "fit_time_seconds": round(fit_time, 3),
            "score_time_seconds": round(score_time, 3),
        }

    def benchmark_adbench(self, dataset_name: str) -> None:
        """Benchmark a single ADBench dataset with all detectors."""
        logger.info("Benchmarking: ADBench %s", dataset_name)

        try:
            config = DatasetConfig(
                name=f"adbench-{dataset_name}",
                preprocessing={"dataset": dataset_name},
            )
            loader = ADBenchLoader(config)
            loader.download()
            X, y = loader._load_raw()
            X = loader.preprocess(X)

            # 70/30 split with fixed seed
            n = len(X)
            n_train = int(n * 0.7)
            rng = np.random.RandomState(42)
            indices = rng.permutation(n)
            X_train, X_test = X[indices[:n_train]], X[indices[n_train:]]
            y_test = y[indices[n_train:]]

            dataset_results: dict = {
                "n_total": n,
                "n_train": n_train,
                "n_test": n - n_train,
                "n_features": int(X.shape[1]),
                "anomaly_ratio_test": round(float(y_test.mean()), 4),
                "detectors": {},
            }

            # Statistical detector
            stat = self._run_detector(StatisticalAnomalyDetector, X_train, X_test, y_test)
            dataset_results["detectors"]["statistical"] = stat

            # Temporal detector
            temp = self._run_detector(TemporalAnomalyDetector, X_train, X_test, y_test)
            dataset_results["detectors"]["temporal"] = temp

            self.results["datasets"][f"adbench_{dataset_name}"] = dataset_results

            logger.info(
                "  ADBench %s: Stat AUC=%.3f F1=%.3f | Temp AUC=%.3f F1=%.3f",
                dataset_name,
                stat["auc"],
                stat["f1"],
                temp["auc"],
                temp["f1"],
            )

        except Exception as e:
            logger.error("  ADBench %s FAILED: %s", dataset_name, e)
            self.results["datasets"][f"adbench_{dataset_name}"] = {"error": str(e)}

    def benchmark_nslkdd(self) -> None:
        """Benchmark NSL-KDD network intrusion dataset."""
        logger.info("Benchmarking: NSL-KDD")

        try:
            config = DatasetConfig(
                name="nsl-kdd",
                preprocessing={"binary": True, "include_test": True},
            )
            loader = NSLKDDLoader(config)
            features, labels = loader.load_data()
            features = loader.preprocess(features)

            n = len(features)
            n_train = int(n * 0.7)
            rng = np.random.RandomState(42)
            indices = rng.permutation(n)
            X_train = features[indices[:n_train]]
            X_test = features[indices[n_train:]]
            y_test = labels[indices[n_train:]]

            dataset_results: dict = {
                "n_total": n,
                "n_train": n_train,
                "n_test": n - n_train,
                "n_features": int(features.shape[1]),
                "anomaly_ratio_test": round(float(y_test.mean()), 4),
                "detectors": {},
            }

            stat = self._run_detector(StatisticalAnomalyDetector, X_train, X_test, y_test)
            dataset_results["detectors"]["statistical"] = stat

            temp = self._run_detector(TemporalAnomalyDetector, X_train, X_test, y_test)
            dataset_results["detectors"]["temporal"] = temp

            self.results["datasets"]["nslkdd"] = dataset_results

            logger.info(
                "  NSL-KDD: Stat AUC=%.3f F1=%.3f | Temp AUC=%.3f F1=%.3f",
                stat["auc"],
                stat["f1"],
                temp["auc"],
                temp["f1"],
            )

        except Exception as e:
            logger.error("  NSL-KDD FAILED: %s", e)
            self.results["datasets"]["nslkdd"] = {"error": str(e)}

    def benchmark_cicids(self) -> None:
        """Benchmark CICIDS-2017 network intrusion dataset."""
        logger.info("Benchmarking: CICIDS-2017")

        try:
            config = DatasetConfig(
                name="cicids",
                preprocessing={"binary": True, "source": "huggingface"},
            )
            loader = CICIDSLoader(config)
            features, labels = loader.load_data()
            features = loader.preprocess(features)

            # Subsample for efficiency
            n = min(len(features), 50000)
            rng = np.random.RandomState(42)
            indices = rng.permutation(len(features))[:n]
            features, labels = features[indices], labels[indices]

            n_train = int(n * 0.7)
            X_train = features[:n_train]
            X_test = features[n_train:]
            y_test = labels[n_train:]

            dataset_results: dict = {
                "n_total": n,
                "n_train": n_train,
                "n_test": n - n_train,
                "n_features": int(features.shape[1]),
                "anomaly_ratio_test": round(float(y_test.mean()), 4),
                "detectors": {},
            }

            stat = self._run_detector(StatisticalAnomalyDetector, X_train, X_test, y_test)
            dataset_results["detectors"]["statistical"] = stat

            self.results["datasets"]["cicids_2017"] = dataset_results

            logger.info("  CICIDS-2017: Stat AUC=%.3f F1=%.3f", stat["auc"], stat["f1"])

        except Exception as e:
            logger.error("  CICIDS-2017 FAILED: %s", e)
            self.results["datasets"]["cicids_2017"] = {"error": str(e)}

    def compute_summary(self) -> None:
        """Compute aggregate statistics."""
        stat_aucs: list[float] = []
        stat_f1s: list[float] = []
        temp_aucs: list[float] = []
        temp_f1s: list[float] = []
        failed = 0

        for data in self.results["datasets"].values():
            if "error" in data:
                failed += 1
                continue
            detectors = data.get("detectors", {})
            if "statistical" in detectors:
                stat_aucs.append(detectors["statistical"]["auc"])
                stat_f1s.append(detectors["statistical"]["f1"])
            if "temporal" in detectors:
                temp_aucs.append(detectors["temporal"]["auc"])
                temp_f1s.append(detectors["temporal"]["f1"])

        self.results["summary"] = {
            "total_datasets_tested": len(self.results["datasets"]) - failed,
            "total_datasets_failed": failed,
            "statistical_detector": {
                "mean_auc": round(float(np.mean(stat_aucs)), 3) if stat_aucs else None,
                "median_auc": round(float(np.median(stat_aucs)), 3) if stat_aucs else None,
                "mean_f1": round(float(np.mean(stat_f1s)), 3) if stat_f1s else None,
            },
            "temporal_detector": {
                "mean_auc": round(float(np.mean(temp_aucs)), 3) if temp_aucs else None,
                "median_auc": round(float(np.median(temp_aucs)), 3) if temp_aucs else None,
                "mean_f1": round(float(np.mean(temp_f1s)), 3) if temp_f1s else None,
            },
        }

    def save(self, path: str = "benchmarks/v1.4_comprehensive_results.json") -> None:
        """Save benchmark results to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info("Results saved: %s", path)

    def run(self) -> None:
        """Execute full benchmark suite."""
        logger.info("Mercury-Agent v%s Comprehensive Benchmark Suite", __version__)

        adbench_datasets = [
            "cardio", "thyroid", "mammography", "breastw",
            "Ionosphere", "Pima", "satellite", "shuttle",
            "wine", "glass", "musk", "arrhythmia",
            "optdigits", "pendigits", "vertebral", "WBC",
        ]

        for name in adbench_datasets:
            self.benchmark_adbench(name)

        self.benchmark_nslkdd()
        self.benchmark_cicids()

        self.compute_summary()
        self.save()

        summary = self.results["summary"]
        logger.info("Suite complete: %d datasets tested, %d failed",
                     summary["total_datasets_tested"], summary["total_datasets_failed"])
        if summary["statistical_detector"]["mean_auc"]:
            logger.info("Statistical: mean AUC=%.3f, mean F1=%.3f",
                         summary["statistical_detector"]["mean_auc"],
                         summary["statistical_detector"]["mean_f1"])


if __name__ == "__main__":
    suite = BenchmarkSuite()
    suite.run()
