"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

Real-World Benchmark Suite for Mercury Agent ♱

Comprehensive benchmarking across all real-world datasets with:
- Per-sample precision, recall, F1
- Cross-domain evaluation
- Statistical significance testing
- Baseline comparisons
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import DatasetConfig, DatasetLoader, DatasetRegistry, DatasetSplit


if TYPE_CHECKING:
    from collections.abc import Callable

try:
    from scipy import stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run.

    Attributes:
        dataset_name: Name of dataset tested
        detector_name: Name of detector evaluated
        split: Which data split was used
        accuracy: Overall accuracy
        precision: Precision score
        recall: Recall score (sensitivity)
        f1_score: F1 score
        specificity: True negative rate
        auc_roc: Area under ROC curve
        auc_pr: Area under Precision-Recall curve
        confusion_matrix: 2x2 confusion matrix
        inference_time_ms: Average inference time per sample
        total_samples: Number of samples evaluated
        predictions: Raw predictions (optional)
        ground_truth: Ground truth labels (optional)
        per_sample_metrics: Per-sample F1/precision/recall (optional)
    """

    dataset_name: str
    detector_name: str
    split: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    specificity: float = 0.0
    auc_roc: float = 0.0
    auc_pr: float = 0.0
    confusion_matrix: list[list[int]] = field(default_factory=lambda: [[0, 0], [0, 0]])
    inference_time_ms: float = 0.0
    total_samples: int = 0
    predictions: np.ndarray[Any, Any] | None = None
    ground_truth: np.ndarray[Any, Any] | None = None
    per_sample_metrics: dict[str, np.ndarray[Any, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (without large arrays)."""
        return {
            "dataset_name": self.dataset_name,
            "detector_name": self.detector_name,
            "split": self.split,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "specificity": round(self.specificity, 4),
            "auc_roc": round(self.auc_roc, 4),
            "auc_pr": round(self.auc_pr, 4),
            "confusion_matrix": self.confusion_matrix,
            "inference_time_ms": round(self.inference_time_ms, 3),
            "total_samples": self.total_samples,
        }


@dataclass
class BenchmarkComparison:
    """Comparison between multiple benchmark results."""

    results: list[BenchmarkResult]
    baseline_name: str
    improvement_vs_baseline: dict[str, float] = field(default_factory=dict)
    statistical_tests: dict[str, Any] = field(default_factory=dict)
    overall_f1: float = 0.0
    overall_improvement: float = 0.0


class RealWorldBenchmarkSuite:
    """
    Comprehensive benchmark suite for real-world datasets.

    Supports:
    - Medical: MIMIC-III, PhysioNet (sepsis, cardiology)
    - Space: SETI, Exoplanet, Solar dynamics
    - Environmental: Earthquake, Weather, Wildfire
    - Security: NSL-KDD, CICIDS, Threat Intelligence

    Example:
        >>> from omni_mercury_engine.datasets.benchmarks import RealWorldBenchmarkSuite
        >>> from omni_mercury_engine import OmniMercuryEngine

        >>> suite = RealWorldBenchmarkSuite()
        >>> engine = OmniMercuryEngine()

        >>> results = suite.run_all_benchmarks(
        ...     detector=lambda x: engine.detect_with_fusion(x)["anomaly_scores"],
        ...     detector_name="OmniMercury"
        ... )
        >>> suite.print_summary(results)
    """

    # All available benchmark datasets
    BENCHMARK_DATASETS = {
        # Medical
        "mimic-iii": {"category": "medical", "description": "ICU patient data"},
        "sepsis": {"category": "medical", "description": "Sepsis prediction"},
        "cardiology": {"category": "medical", "description": "Cardiac anomalies"},
        "physionet": {"category": "medical", "description": "ECG signals"},
        # Space
        "seti": {"category": "space", "description": "SETI signal detection"},
        "exoplanet": {"category": "space", "description": "Exoplanet detection"},
        "solar": {"category": "space", "description": "Solar storm prediction"},
        # Environmental
        "earthquake": {"category": "environmental", "description": "Seismic events"},
        "weather": {"category": "environmental", "description": "Extreme weather"},
        "wildfire": {"category": "environmental", "description": "Fire detection"},
        # Security
        "nsl-kdd": {"category": "security", "description": "Network intrusion"},
        "cicids": {"category": "security", "description": "Modern IDS traffic"},
        "threat-intel": {"category": "security", "description": "Threat indicators"},
    }

    def __init__(
        self,
        data_dir: str = "./data",
        cache_dir: str = "./cache",
        random_seed: int = 42,
        max_samples_per_dataset: int | None = None,
    ):
        """Initialize benchmark suite.

        Args:
            data_dir: Directory for dataset storage
            cache_dir: Directory for cached processed data
            random_seed: Seed for reproducibility
            max_samples_per_dataset: Limit samples per dataset (None = all)
        """
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.random_seed = random_seed
        self.max_samples = max_samples_per_dataset

        self.results: list[BenchmarkResult] = []

    def get_dataset_loader(self, dataset_name: str) -> DatasetLoader:
        """Get loader for a specific dataset."""
        config = DatasetConfig(
            name=dataset_name,
            data_dir=self.data_dir,
            cache_dir=self.cache_dir,
            random_seed=self.random_seed,
            max_samples=self.max_samples,
        )
        return DatasetRegistry.create(dataset_name, config)

    def run_benchmark(
        self,
        dataset_name: str,
        detector: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]],
        detector_name: str,
        threshold: float = 0.5,
        split: DatasetSplit = DatasetSplit.TEST,
    ) -> BenchmarkResult:
        """Run benchmark on a single dataset.

        Args:
            dataset_name: Name of dataset to benchmark
            detector: Function that takes features and returns anomaly scores
            detector_name: Name for the detector
            threshold: Classification threshold
            split: Which data split to evaluate

        Returns:
            BenchmarkResult with all metrics
        """
        logger.info(f"Running benchmark: {detector_name} on {dataset_name}")

        # Load dataset
        loader = self.get_dataset_loader(dataset_name)
        features, labels = loader.load(split)

        # Run inference
        start_time = time.perf_counter()
        scores = detector(features)
        inference_time = (time.perf_counter() - start_time) * 1000  # ms

        # Ensure scores are numpy array
        if TORCH_AVAILABLE and isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()

        scores = np.atleast_1d(scores).flatten()

        # Handle score shape mismatch
        if len(scores) != len(labels):
            if len(scores) == 1:
                scores = np.full(len(labels), scores[0])
            else:
                logger.warning(f"Score length mismatch: {len(scores)} vs {len(labels)}")
                scores = (
                    scores[: len(labels)]
                    if len(scores) > len(labels)
                    else np.pad(scores, (0, len(labels) - len(scores)))
                )

        # Convert scores to predictions
        predictions = (scores >= threshold).astype(int)

        # Calculate metrics
        metrics = self._calculate_metrics(labels, predictions, scores)

        result = BenchmarkResult(
            dataset_name=dataset_name,
            detector_name=detector_name,
            split=split.value,
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1_score=metrics["f1"],
            specificity=metrics["specificity"],
            auc_roc=metrics.get("auc_roc", 0.0),
            auc_pr=metrics.get("auc_pr", 0.0),
            confusion_matrix=metrics["confusion_matrix"],
            inference_time_ms=inference_time / len(features),
            total_samples=len(features),
            predictions=predictions,
            ground_truth=labels,
            per_sample_metrics=metrics.get("per_sample"),
        )

        self.results.append(result)
        return result

    def run_all_benchmarks(
        self,
        detector: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]],
        detector_name: str,
        categories: list[str] | None = None,
        datasets: list[str] | None = None,
        threshold: float = 0.5,
    ) -> list[BenchmarkResult]:
        """Run benchmarks across all datasets.

        Args:
            detector: Detection function
            detector_name: Detector name
            categories: Filter by categories (None = all)
            datasets: Filter by specific datasets (None = all)
            threshold: Classification threshold

        Returns:
            List of BenchmarkResult for each dataset
        """
        results = []

        target_datasets = datasets or list(self.BENCHMARK_DATASETS.keys())

        if categories:
            target_datasets = [
                d
                for d, info in self.BENCHMARK_DATASETS.items()
                if info["category"] in categories and d in target_datasets
            ]

        for dataset_name in target_datasets:
            try:
                result = self.run_benchmark(
                    dataset_name=dataset_name,
                    detector=detector,
                    detector_name=detector_name,
                    threshold=threshold,
                )
                results.append(result)
                logger.info(f"  {dataset_name}: F1={result.f1_score:.4f}")
            except Exception as e:
                logger.error(f"Benchmark failed for {dataset_name}: {e}")

        return results

    def compare_with_baseline(
        self,
        results: list[BenchmarkResult],
        baseline_detector: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]],
        baseline_name: str = "RandomBaseline",
    ) -> BenchmarkComparison:
        """Compare results against a baseline detector.

        Args:
            results: Results from main detector
            baseline_detector: Baseline detection function
            baseline_name: Name for baseline

        Returns:
            BenchmarkComparison with statistical tests
        """
        baseline_results = []
        improvements = {}
        stat_tests = {}

        for result in results:
            # Run baseline on same dataset
            baseline_result = self.run_benchmark(
                dataset_name=result.dataset_name,
                detector=baseline_detector,
                detector_name=baseline_name,
            )
            baseline_results.append(baseline_result)

            # Calculate improvement
            improvement = result.f1_score - baseline_result.f1_score
            improvements[result.dataset_name] = improvement

            # Statistical significance test (if predictions available)
            if (
                SCIPY_AVAILABLE
                and result.predictions is not None
                and baseline_result.predictions is not None
            ):
                # McNemar's test for paired predictions
                try:
                    n01 = np.sum((result.predictions == 1) & (baseline_result.predictions == 0))
                    n10 = np.sum((result.predictions == 0) & (baseline_result.predictions == 1))
                    if n01 + n10 > 0:
                        mcnemar_stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
                        p_value = 1 - stats.chi2.cdf(mcnemar_stat, 1)
                        stat_tests[result.dataset_name] = {
                            "mcnemar_stat": mcnemar_stat,
                            "p_value": p_value,
                            "significant": p_value < 0.05,
                        }
                except Exception as e:
                    logger.debug(f"Statistical test failed: {e}")

        # Overall metrics
        overall_f1 = np.mean([r.f1_score for r in results])
        baseline_f1 = np.mean([r.f1_score for r in baseline_results])
        overall_improvement = overall_f1 - baseline_f1

        return BenchmarkComparison(
            results=results,
            baseline_name=baseline_name,
            improvement_vs_baseline=improvements,
            statistical_tests=stat_tests,
            overall_f1=overall_f1,
            overall_improvement=overall_improvement,
        )

    def _calculate_metrics(
        self,
        labels: np.ndarray[Any, Any],
        predictions: np.ndarray[Any, Any],
        scores: np.ndarray[Any, Any],
    ) -> dict[str, Any]:
        """Calculate comprehensive metrics."""
        labels = labels.flatten()
        predictions = predictions.flatten()
        scores = scores.flatten()

        # Confusion matrix
        tp = np.sum((predictions == 1) & (labels == 1))
        tn = np.sum((predictions == 0) & (labels == 0))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))

        # Core metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        specificity = tn / (tn + fp + 1e-10)

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "specificity": float(specificity),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        }

        # AUC-ROC and AUC-PR (if scipy available)
        if SCIPY_AVAILABLE:
            try:
                # Sort by scores for ROC calculation
                sorted_idx = np.argsort(-scores)
                sorted_labels = labels[sorted_idx]

                # Calculate TPR and FPR at each threshold
                tpr_list = []
                fpr_list = []
                precision_list = []
                recall_list = []

                total_pos = np.sum(labels == 1)
                total_neg = np.sum(labels == 0)

                for i in range(len(sorted_labels) + 1):
                    tp_i = np.sum(sorted_labels[:i] == 1)
                    fp_i = np.sum(sorted_labels[:i] == 0)

                    tpr = tp_i / (total_pos + 1e-10)
                    fpr = fp_i / (total_neg + 1e-10)
                    prec_i = tp_i / (tp_i + fp_i + 1e-10)
                    rec_i = tp_i / (total_pos + 1e-10)

                    tpr_list.append(tpr)
                    fpr_list.append(fpr)
                    precision_list.append(prec_i)
                    recall_list.append(rec_i)

                # AUC-ROC (trapezoidal)
                auc_roc = np.trapezoid(tpr_list, fpr_list)
                metrics["auc_roc"] = float(abs(auc_roc))

                # AUC-PR
                auc_pr = np.trapezoid(precision_list, recall_list)
                metrics["auc_pr"] = float(abs(auc_pr))

            except Exception as e:
                logger.debug(f"AUC calculation failed: {e}")

        # Per-sample metrics
        per_sample = {
            "correct": (predictions == labels).astype(float),
            "score": scores,
        }
        metrics["per_sample"] = per_sample

        return metrics

    def print_summary(
        self,
        results: list[BenchmarkResult] | None = None,
        comparison: BenchmarkComparison | None = None,
    ) -> str:
        """Print formatted benchmark summary.

        Args:
            results: List of results (default: use stored results)
            comparison: Optional comparison with baseline

        Returns:
            Formatted summary string
        """
        results = results or self.results

        lines = []
        lines.append("=" * 80)
        lines.append("Mercury Agent ♱ REAL-WORLD BENCHMARK RESULTS")
        lines.append("=" * 80)
        lines.append("")

        # Group by category
        by_category: dict[str, list[BenchmarkResult]] = {}
        for r in results:
            cat = self.BENCHMARK_DATASETS.get(r.dataset_name, {}).get("category", "other")
            by_category.setdefault(cat, []).append(r)

        for category, cat_results in by_category.items():
            lines.append(f"\n{category.upper()} DATASETS")
            lines.append("-" * 60)
            lines.append(f"{'Dataset':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
            lines.append("-" * 60)

            for r in cat_results:
                lines.append(
                    f"{r.dataset_name:<20} "
                    f"{r.precision:>10.4f} "
                    f"{r.recall:>10.4f} "
                    f"{r.f1_score:>10.4f} "
                    f"{r.auc_roc:>10.4f}"
                )

            cat_f1 = np.mean([r.f1_score for r in cat_results])
            lines.append(f"{'Average':<20} {'':<10} {'':<10} {cat_f1:>10.4f}")

        # Overall summary
        lines.append("\n" + "=" * 80)
        overall_f1 = np.mean([r.f1_score for r in results])
        overall_precision = np.mean([r.precision for r in results])
        overall_recall = np.mean([r.recall for r in results])
        lines.append(
            f"OVERALL: Precision={overall_precision:.4f}, "
            f"Recall={overall_recall:.4f}, F1={overall_f1:.4f}"
        )

        # Comparison summary
        if comparison:
            lines.append(
                f"\nVs {comparison.baseline_name}: "
                f"+{comparison.overall_improvement:.4f} F1 improvement"
            )
            sig_count = sum(
                1 for t in comparison.statistical_tests.values() if t.get("significant")
            )
            n_tests = len(comparison.statistical_tests)
            lines.append(f"Statistically significant improvements: {sig_count}/{n_tests}")

        lines.append("=" * 80)

        summary = "\n".join(lines)
        print(summary)
        return summary

    def save_results(self, filepath: str | Path) -> None:
        """Save benchmark results to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": [r.to_dict() for r in self.results],
            "overall_f1": np.mean([r.f1_score for r in self.results]) if self.results else 0,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved benchmark results to {filepath}")

    def load_results(self, filepath: str | Path) -> list[BenchmarkResult]:
        """Load benchmark results from JSON file."""
        with open(filepath) as f:
            data = json.load(f)

        results = []
        for r in data["results"]:
            result = BenchmarkResult(
                dataset_name=r["dataset_name"],
                detector_name=r["detector_name"],
                split=r["split"],
                accuracy=r["accuracy"],
                precision=r["precision"],
                recall=r["recall"],
                f1_score=r["f1_score"],
                specificity=r.get("specificity", 0),
                auc_roc=r.get("auc_roc", 0),
                auc_pr=r.get("auc_pr", 0),
                confusion_matrix=r.get("confusion_matrix", [[0, 0], [0, 0]]),
                inference_time_ms=r.get("inference_time_ms", 0),
                total_samples=r.get("total_samples", 0),
            )
            results.append(result)

        self.results = results
        return results


# Baseline detectors for comparison
def random_baseline(features: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Random baseline detector."""
    return np.random.rand(len(features))


def isolation_forest_baseline(features: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Isolation Forest baseline detector."""
    try:
        from sklearn.ensemble import IsolationForest

        clf = IsolationForest(random_state=42, contamination=0.1)
        clf.fit(features)
        scores = -clf.score_samples(features)  # Higher = more anomalous
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
    except ImportError:
        return random_baseline(features)


def one_class_svm_baseline(features: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """One-Class SVM baseline detector."""
    try:
        from sklearn.svm import OneClassSVM

        clf = OneClassSVM(nu=0.1, kernel="rbf", gamma="auto")
        clf.fit(features[: min(1000, len(features))])  # Limit for speed
        scores = -clf.score_samples(features)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
    except ImportError:
        return random_baseline(features)
