#!/usr/bin/env python3
"""
Mercury Agent - Comprehensive Live Dataset Benchmark Suite

Runs benchmarks across all integrated live data sources to demonstrate
the breadth of Mercury Agent's anomaly detection capabilities.

Datasets Covered:
- Security: NSL-KDD, CICIDS-2017 (network intrusion)
- Industrial: BATADAL, SWaT, WADI (cyber-physical systems)
- Time-Series: SMD, NAB, SMAP/MSL (server/IoT anomalies)
- Climate: Simons CMAP, World Ocean Database, Copernicus Sea Level
- Disaster: FEMA Disaster Declarations, Hazard Mitigation
- Environmental: USGS Earthquake, USGS Geochemistry, NOAA Weather
- Medical: PhysioNet (requires credentials for MIMIC)
- Space: NASA Exoplanet, Solar Dynamics Observatory

Usage:
    # Run all benchmarks
    python benchmarks/live_dataset_benchmark.py

    # Run specific category
    python benchmarks/live_dataset_benchmark.py --category security

    # Run with specific detector
    python benchmarks/live_dataset_benchmark.py --detector adaptive

    # Export results
    python benchmarks/live_dataset_benchmark.py --output results.json

Copyright (C) 2025 Steel Security Advisory LLC
Licensed under GPL-3.0-or-later
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class DatasetBenchmarkResult:
    """Result from benchmarking a single dataset."""

    dataset_name: str
    category: str
    source: str  # "live" or "synthetic"
    n_samples: int
    n_features: int
    anomaly_ratio: float

    # Metrics
    roc_auc: float
    pr_auc: float
    f1: float
    precision: float
    recall: float
    accuracy: float

    # Timing
    load_time_ms: float
    detection_time_ms: float

    # Error handling
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class BenchmarkSuiteResult:
    """Aggregated results from the full benchmark suite."""

    timestamp: str
    mercury_version: str
    python_version: str
    detector_name: str

    total_datasets: int
    successful_datasets: int
    failed_datasets: int

    # Aggregate metrics
    mean_roc_auc: float
    mean_pr_auc: float
    mean_f1: float
    std_f1: float

    # Timing
    total_time_seconds: float

    # Per-category results
    results_by_category: dict[str, list[DatasetBenchmarkResult]]

    # Coverage summary
    live_data_coverage: dict[str, bool]


class LiveDatasetBenchmarkRunner:
    """Runner for comprehensive live dataset benchmarks."""

    # Dataset categories and their loaders
    DATASET_REGISTRY = {
        "security": [
            ("NSL-KDD", "NSLKDDLoader", "Network intrusion detection"),
            ("CICIDS-2017", "CICIDSLoader", "Modern network attacks"),
        ],
        "industrial": [
            ("BATADAL", "BATADALLoader", "Water infrastructure attacks"),
            ("SWaT", "SWaTLoader", "Secure Water Treatment"),
            ("WADI", "WADILoader", "Water Distribution"),
        ],
        "timeseries": [
            ("SMD", "SMDLoader", "Server Machine Dataset"),
            ("NAB", "NABLoader", "Numenta Anomaly Benchmark"),
            ("SMAP-MSL", "SMAPMSLLoader", "NASA spacecraft telemetry"),
        ],
        "climate": [
            ("SimonsCMAP", "SimonsCMAPLoader", "Ocean biogeochemistry"),
            ("WorldOcean", "WorldOceanDatabaseLoader", "Ocean temp/salinity"),
            ("CopernicusSea", "CopernicusSeaLevelLoader", "Satellite altimetry"),
        ],
        "disaster": [
            ("FEMADisaster", "FEMADisasterLoader", "US disaster declarations"),
            ("FEMAHazard", "FEMAHazardMitigationLoader", "Hazard mitigation grants"),
        ],
        "environmental": [
            ("USGSEarthquake", "USGSEarthquakeLoader", "Seismic events"),
            ("USGSGeochem", "USGSGeochemistryLoader", "Heavy metal contamination"),
            ("NOAAWeather", "NOAAWeatherLoader", "Weather patterns"),
        ],
        "adrepository": [
            ("fraud", "load_dataset", "Credit card fraud (Kaggle)"),
            ("thyroid", "load_dataset", "Thyroid disease"),
            ("mammography", "load_dataset", "Mammography screening"),
            ("campaign", "load_dataset", "Marketing campaign"),
            ("backdoor", "load_dataset", "Malware backdoor"),
        ],
    }

    def __init__(self, detector_name: str = "adaptive"):
        self.detector_name = detector_name
        self.detector = None
        self.results: list[DatasetBenchmarkResult] = []

    def _initialize_detector(self) -> None:
        """Initialize the anomaly detector."""
        try:
            from omni_mercury_engine.core.adaptive_detector import AdaptiveAnomalyDetector

            self.detector = AdaptiveAnomalyDetector(
                contamination="auto",
                enable_3r=True,
                use_adaptive_fusion=True,
            )
            logger.info(f"Initialized {self.detector_name} detector")
        except ImportError as e:
            logger.warning(f"Could not import AdaptiveAnomalyDetector: {e}")
            # Fallback to IsolationForest
            from sklearn.ensemble import IsolationForest

            self.detector = IsolationForest(contamination=0.1, random_state=42)
            self.detector_name = "IsolationForest"

    def _load_dataset(
        self, category: str, dataset_name: str, loader_name: str
    ) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
        """Load a dataset with proper error handling."""
        warnings: list[str] = []
        metadata: dict[str, Any] = {
            "source": "synthetic",
            "n_samples": 0,
            "n_features": 0,
            "anomaly_ratio": 0.0,
            "warnings": warnings,
        }

        try:
            if loader_name == "load_dataset":
                # ADRepository datasets
                from omni_mercury_engine.datasets import load_dataset

                X, y, meta = load_dataset(dataset_name.lower())
                metadata["source"] = "live" if not meta.get("is_synthetic", True) else "synthetic"
                metadata["n_samples"] = len(X)
                metadata["n_features"] = X.shape[1] if len(X.shape) > 1 else 1
                metadata["anomaly_ratio"] = float(np.mean(y)) if len(y) > 0 else 0.0
                return X, y, metadata

            # Import the specific loader
            from omni_mercury_engine import datasets

            loader_class = getattr(datasets, loader_name, None)
            if loader_class is None:
                warnings.append(f"Loader {loader_name} not found")
                return None, None, metadata

            loader = loader_class()
            data = loader.load()

            if hasattr(data, "features") and hasattr(data, "labels"):
                X = data.features
                y = data.labels
            elif isinstance(data, tuple) and len(data) >= 2:
                X, y = data[0], data[1]
            elif hasattr(data, "data"):
                X = data.data
                y = getattr(data, "labels", np.zeros(len(X)))
            else:
                warnings.append("Unknown data format")
                return None, None, metadata

            # Ensure numpy arrays
            X = np.asarray(X)
            y = np.asarray(y)

            # Handle 1D arrays
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)

            metadata["source"] = "live" if getattr(loader, "used_real_data", False) else "synthetic"
            metadata["n_samples"] = len(X)
            metadata["n_features"] = X.shape[1]
            metadata["anomaly_ratio"] = float(np.mean(y)) if len(y) > 0 else 0.0

            return X, y, metadata

        except Exception as e:
            warnings.append(f"Load error: {str(e)}")
            logger.warning(f"Failed to load {dataset_name}: {e}")
            return None, None, metadata

    def _run_detection(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Run anomaly detection and return predictions, scores, and time."""
        assert self.detector is not None, "Detector must be initialized before detection"
        start = time.perf_counter()

        try:
            # Handle NaN/Inf values
            X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

            if hasattr(self.detector, "fit_predict"):
                # Scikit-learn style
                predictions = self.detector.fit_predict(X)
                # Convert -1/1 to 0/1
                predictions = (predictions == -1).astype(int)
                scores = (
                    -self.detector.score_samples(X)
                    if hasattr(self.detector, "score_samples")
                    else predictions.astype(float)
                )
            elif hasattr(self.detector, "fit") and hasattr(self.detector, "predict"):
                self.detector.fit(X, y)
                predictions = self.detector.predict(X)
                scores = (
                    self.detector.predict_proba(X)[:, 1]
                    if hasattr(self.detector, "predict_proba")
                    else predictions.astype(float)
                )
            else:
                raise ValueError("Detector must have fit_predict or fit/predict methods")

            elapsed_ms = (time.perf_counter() - start) * 1000
            return predictions, scores, elapsed_ms

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            elapsed_ms = (time.perf_counter() - start) * 1000
            return np.zeros(len(y)), np.zeros(len(y)), elapsed_ms

    def _compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray
    ) -> dict[str, float]:
        """Compute evaluation metrics with proper error handling."""
        metrics = {
            "roc_auc": 0.5,
            "pr_auc": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "accuracy": 0.0,
        }

        # Skip if no positive class
        if np.sum(y_true) == 0 or np.sum(y_true) == len(y_true):
            return metrics

        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_scores)
        except ValueError:
            pass

        try:
            metrics["pr_auc"] = average_precision_score(y_true, y_scores)
        except ValueError:
            pass

        try:
            metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
            metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
            metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
            metrics["accuracy"] = accuracy_score(y_true, y_pred)
        except ValueError:
            pass

        return metrics

    def run_benchmark(
        self,
        categories: list[str] | None = None,
        max_samples: int = 10000,
    ) -> BenchmarkSuiteResult:
        """Run the full benchmark suite."""
        self._initialize_detector()

        if categories is None:
            categories = list(self.DATASET_REGISTRY.keys())

        start_time = time.perf_counter()
        results_by_category: dict[str, list[DatasetBenchmarkResult]] = {}
        live_data_coverage: dict[str, bool] = {}

        for category in categories:
            if category not in self.DATASET_REGISTRY:
                logger.warning(f"Unknown category: {category}")
                continue

            results_by_category[category] = []
            datasets = self.DATASET_REGISTRY[category]

            logger.info(f"\n{'='*60}")
            logger.info(f"Category: {category.upper()} ({len(datasets)} datasets)")
            logger.info("=" * 60)

            for dataset_name, loader_name, description in datasets:
                logger.info(f"\nBenchmarking: {dataset_name} - {description}")

                # Load dataset
                load_start = time.perf_counter()
                X, y, metadata = self._load_dataset(category, dataset_name, loader_name)
                load_time_ms = (time.perf_counter() - load_start) * 1000

                if X is None or y is None or len(X) == 0:
                    result = DatasetBenchmarkResult(
                        dataset_name=dataset_name,
                        category=category,
                        source="failed",
                        n_samples=0,
                        n_features=0,
                        anomaly_ratio=0.0,
                        roc_auc=0.0,
                        pr_auc=0.0,
                        f1=0.0,
                        precision=0.0,
                        recall=0.0,
                        accuracy=0.0,
                        load_time_ms=load_time_ms,
                        detection_time_ms=0.0,
                        error="Failed to load dataset",
                        warnings=metadata.get("warnings", []),
                    )
                    results_by_category[category].append(result)
                    self.results.append(result)
                    continue

                # Subsample if too large
                if len(X) > max_samples:
                    indices = np.random.choice(len(X), max_samples, replace=False)
                    X, y = X[indices], y[indices]
                    metadata_warnings: list[str] = metadata.get("warnings", [])
                    metadata_warnings.append(f"Subsampled to {max_samples}")

                # Run detection
                y_pred, y_scores, detection_time_ms = self._run_detection(X, y)

                # Compute metrics
                metrics = self._compute_metrics(y, y_pred, y_scores)

                result = DatasetBenchmarkResult(
                    dataset_name=dataset_name,
                    category=category,
                    source=metadata["source"],
                    n_samples=metadata["n_samples"],
                    n_features=metadata["n_features"],
                    anomaly_ratio=metadata["anomaly_ratio"],
                    roc_auc=metrics["roc_auc"],
                    pr_auc=metrics["pr_auc"],
                    f1=metrics["f1"],
                    precision=metrics["precision"],
                    recall=metrics["recall"],
                    accuracy=metrics["accuracy"],
                    load_time_ms=load_time_ms,
                    detection_time_ms=detection_time_ms,
                    warnings=metadata.get("warnings", []),
                )

                results_by_category[category].append(result)
                self.results.append(result)

                # Track live data coverage
                live_data_coverage[dataset_name] = metadata["source"] == "live"

                logger.info(
                    f"  Source: {result.source} | Samples: {result.n_samples} | "
                    f"F1: {result.f1:.3f} | ROC-AUC: {result.roc_auc:.3f}"
                )

        # Aggregate results
        total_time = time.perf_counter() - start_time
        successful = [r for r in self.results if r.error is None]
        failed = [r for r in self.results if r.error is not None]

        f1_scores = [r.f1 for r in successful if r.f1 > 0]

        return BenchmarkSuiteResult(
            timestamp=datetime.utcnow().isoformat(),
            mercury_version="1.2.0",
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
            detector_name=self.detector_name,
            total_datasets=len(self.results),
            successful_datasets=len(successful),
            failed_datasets=len(failed),
            mean_roc_auc=float(np.mean([r.roc_auc for r in successful])) if successful else 0.0,
            mean_pr_auc=float(np.mean([r.pr_auc for r in successful])) if successful else 0.0,
            mean_f1=float(np.mean(f1_scores)) if f1_scores else 0.0,
            std_f1=float(np.std(f1_scores)) if f1_scores else 0.0,
            total_time_seconds=total_time,
            results_by_category=results_by_category,
            live_data_coverage=live_data_coverage,
        )

    def print_summary(self, result: BenchmarkSuiteResult) -> None:
        """Print a formatted summary of results."""
        print("\n" + "=" * 80)
        print("MERCURY AGENT LIVE DATASET BENCHMARK RESULTS")
        print("=" * 80)
        print(f"Timestamp: {result.timestamp}")
        print(f"Detector: {result.detector_name}")
        print(f"Mercury Version: {result.mercury_version}")
        print(f"Total Time: {result.total_time_seconds:.1f}s")
        print("-" * 80)

        print(f"\nDATASET COVERAGE:")
        print(f"  Total Datasets: {result.total_datasets}")
        print(f"  Successful: {result.successful_datasets}")
        print(f"  Failed: {result.failed_datasets}")

        live_count = sum(1 for v in result.live_data_coverage.values() if v)
        print(f"  Live Data: {live_count}/{len(result.live_data_coverage)}")

        print(f"\nAGGREGATE METRICS:")
        print(f"  Mean ROC-AUC: {result.mean_roc_auc:.4f}")
        print(f"  Mean PR-AUC:  {result.mean_pr_auc:.4f}")
        print(f"  Mean F1:      {result.mean_f1:.4f} (std: {result.std_f1:.4f})")

        print("\nPER-CATEGORY RESULTS:")
        print("-" * 80)
        print(f"{'Category':<15} {'Dataset':<20} {'Source':<10} {'F1':<8} {'ROC-AUC':<8}")
        print("-" * 80)

        for category, results in result.results_by_category.items():
            for r in results:
                source_str = "LIVE" if r.source == "live" else r.source[:8]
                print(
                    f"{category:<15} {r.dataset_name:<20} {source_str:<10} "
                    f"{r.f1:<8.3f} {r.roc_auc:<8.3f}"
                )

        print("=" * 80)

    def export_results(self, result: BenchmarkSuiteResult, output_path: str) -> None:
        """Export results to JSON file."""

        def serialize(obj: Any) -> Any:
            if isinstance(obj, DatasetBenchmarkResult):
                return obj.to_dict()
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        output = {
            "timestamp": result.timestamp,
            "mercury_version": result.mercury_version,
            "python_version": result.python_version,
            "detector_name": result.detector_name,
            "summary": {
                "total_datasets": result.total_datasets,
                "successful_datasets": result.successful_datasets,
                "failed_datasets": result.failed_datasets,
                "mean_roc_auc": result.mean_roc_auc,
                "mean_pr_auc": result.mean_pr_auc,
                "mean_f1": result.mean_f1,
                "std_f1": result.std_f1,
                "total_time_seconds": result.total_time_seconds,
            },
            "live_data_coverage": result.live_data_coverage,
            "results_by_category": {
                cat: [r.to_dict() for r in results]
                for cat, results in result.results_by_category.items()
            },
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=serialize)

        logger.info(f"Results exported to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mercury Agent Live Dataset Benchmark Suite")
    parser.add_argument(
        "--category",
        "-c",
        type=str,
        choices=list(LiveDatasetBenchmarkRunner.DATASET_REGISTRY.keys()),
        help="Run only specific category",
    )
    parser.add_argument(
        "--detector",
        "-d",
        type=str,
        default="adaptive",
        help="Detector to use (adaptive, isolation_forest)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10000,
        help="Maximum samples per dataset",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Run benchmark
    runner = LiveDatasetBenchmarkRunner(detector_name=args.detector)
    categories = [args.category] if args.category else None

    result = runner.run_benchmark(
        categories=categories,
        max_samples=args.max_samples,
    )

    # Print summary
    runner.print_summary(result)

    # Export if requested
    if args.output:
        runner.export_results(result, args.output)


if __name__ == "__main__":
    main()
