"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

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
Validation Pipeline

Provides comprehensive validation utilities:
- Data quality checks
- A/B testing framework
- Cross-validation with multiple metrics
- Statistical significance testing
- Benchmark comparison

Implements the validation framework described in VALIDATION_FRAMEWORK.md.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class QualityCheckResult:
    """Result of a data quality check."""

    check_name: str
    passed: bool
    score: float
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ABTestResult:
    """Result of an A/B test comparison."""

    model_a_name: str
    model_b_name: str
    metric_name: str
    model_a_score: float
    model_b_score: float
    improvement: float
    p_value: float
    statistically_significant: bool
    confidence_level: float
    winner: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Comprehensive validation result."""

    dataset_name: str
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    auc_pr: float
    confusion_matrix: np.ndarray
    quality_checks: list[QualityCheckResult]
    validation_time_seconds: float
    num_samples: int
    num_features: int
    cross_val_scores: list[float] = field(default_factory=list)
    additional_metrics: dict[str, float] = field(default_factory=dict)


class DataQualityChecker:
    """
    Data quality validation checks.

    Implements comprehensive data quality checks:
    - Missing value detection
    - Outlier detection
    - Distribution analysis
    - Feature correlation
    - Class balance
    """

    def __init__(
        self,
        missing_threshold: float = 0.05,
        outlier_threshold: float = 3.0,
        correlation_threshold: float = 0.95,
        imbalance_threshold: float = 0.1,
    ):
        self.missing_threshold = missing_threshold
        self.outlier_threshold = outlier_threshold
        self.correlation_threshold = correlation_threshold
        self.imbalance_threshold = imbalance_threshold

    def run_all_checks(self, data: np.ndarray, labels: np.ndarray) -> list[QualityCheckResult]:
        """Run all quality checks on the data."""
        checks = [
            self.check_missing_values(data),
            self.check_outliers(data),
            self.check_feature_variance(data),
            self.check_class_balance(labels),
            self.check_feature_correlation(data),
            self.check_data_range(data),
        ]
        return checks

    def check_missing_values(self, data: np.ndarray) -> QualityCheckResult:
        """Check for missing values in the data."""
        missing_count = np.sum(np.isnan(data))
        total_values = data.size
        missing_ratio = missing_count / total_values if total_values > 0 else 0

        passed = missing_ratio <= self.missing_threshold
        score = 1.0 - missing_ratio

        return QualityCheckResult(
            check_name="missing_values",
            passed=passed,
            score=score,
            message=f"Missing value ratio: {missing_ratio:.4f} (threshold: {self.missing_threshold})",
            details={
                "missing_count": int(missing_count),
                "total_values": int(total_values),
                "missing_ratio": float(missing_ratio),
            },
        )

    def check_outliers(self, data: np.ndarray) -> QualityCheckResult:
        """Check for outliers using z-score method."""
        if data.size == 0:
            return QualityCheckResult(
                check_name="outliers",
                passed=True,
                score=1.0,
                message="No data to check",
            )

        z_scores = np.abs(stats.zscore(data, nan_policy="omit"))
        outlier_mask = z_scores > self.outlier_threshold
        outlier_ratio = np.sum(outlier_mask) / data.size

        passed = outlier_ratio <= 0.05
        score = 1.0 - min(outlier_ratio * 10, 1.0)

        return QualityCheckResult(
            check_name="outliers",
            passed=passed,
            score=score,
            message=f"Outlier ratio: {outlier_ratio:.4f} (z-score > {self.outlier_threshold})",
            details={
                "outlier_count": int(np.sum(outlier_mask)),
                "outlier_ratio": float(outlier_ratio),
                "threshold": self.outlier_threshold,
            },
        )

    def check_feature_variance(self, data: np.ndarray) -> QualityCheckResult:
        """Check for low-variance features."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        variances = np.var(data, axis=0)
        low_variance_count = np.sum(variances < 1e-6)
        low_variance_ratio = low_variance_count / len(variances) if len(variances) > 0 else 0

        passed = low_variance_ratio <= 0.1
        score = 1.0 - low_variance_ratio

        return QualityCheckResult(
            check_name="feature_variance",
            passed=passed,
            score=score,
            message=f"Low-variance features: {low_variance_count}/{len(variances)}",
            details={
                "low_variance_count": int(low_variance_count),
                "total_features": len(variances),
                "min_variance": float(np.min(variances)) if len(variances) > 0 else 0,
                "max_variance": float(np.max(variances)) if len(variances) > 0 else 0,
            },
        )

    def check_class_balance(self, labels: np.ndarray) -> QualityCheckResult:
        """Check class balance in labels."""
        unique, counts = np.unique(labels, return_counts=True)
        if len(counts) < 2:
            return QualityCheckResult(
                check_name="class_balance",
                passed=False,
                score=0.0,
                message="Only one class present in labels",
            )

        min_ratio = np.min(counts) / np.sum(counts)
        passed = min_ratio >= self.imbalance_threshold
        score = min(min_ratio / 0.5, 1.0)

        return QualityCheckResult(
            check_name="class_balance",
            passed=passed,
            score=score,
            message=f"Minority class ratio: {min_ratio:.4f} (threshold: {self.imbalance_threshold})",
            details={
                "class_counts": dict(zip(unique.tolist(), counts.tolist())),
                "minority_ratio": float(min_ratio),
            },
        )

    def check_feature_correlation(self, data: np.ndarray) -> QualityCheckResult:
        """Check for highly correlated features."""
        if data.ndim == 1 or data.shape[1] < 2:
            return QualityCheckResult(
                check_name="feature_correlation",
                passed=True,
                score=1.0,
                message="Not enough features for correlation check",
            )

        corr_matrix = np.corrcoef(data.T)
        np.fill_diagonal(corr_matrix, 0)

        high_corr_count = np.sum(np.abs(corr_matrix) > self.correlation_threshold) // 2
        total_pairs = data.shape[1] * (data.shape[1] - 1) // 2
        high_corr_ratio = high_corr_count / total_pairs if total_pairs > 0 else 0

        passed = high_corr_ratio <= 0.1
        score = 1.0 - high_corr_ratio

        return QualityCheckResult(
            check_name="feature_correlation",
            passed=passed,
            score=score,
            message=f"Highly correlated feature pairs: {high_corr_count}/{total_pairs}",
            details={
                "high_correlation_pairs": int(high_corr_count),
                "total_pairs": int(total_pairs),
                "threshold": self.correlation_threshold,
            },
        )

    def check_data_range(self, data: np.ndarray) -> QualityCheckResult:
        """Check data range and detect potential scaling issues."""
        data_min = np.nanmin(data)
        data_max = np.nanmax(data)
        data_range = data_max - data_min

        needs_scaling = data_range > 1000 or data_min < -1000 or data_max > 1000
        passed = not needs_scaling
        score = 1.0 if passed else 0.5

        return QualityCheckResult(
            check_name="data_range",
            passed=passed,
            score=score,
            message=f"Data range: [{data_min:.2f}, {data_max:.2f}]",
            details={
                "min": float(data_min),
                "max": float(data_max),
                "range": float(data_range),
                "needs_scaling": needs_scaling,
            },
        )


class ABTester:
    """
    A/B Testing framework for model comparison.

    Implements statistical testing for comparing model performance:
    - Paired t-test for cross-validation scores
    - Bootstrap confidence intervals
    - Effect size calculation (Cohen's d)
    """

    def __init__(self, confidence_level: float = 0.95, n_bootstrap: int = 1000):
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.alpha = 1.0 - confidence_level

    def compare_models(
        self,
        model_a_scores: np.ndarray,
        model_b_scores: np.ndarray,
        model_a_name: str = "Model A",
        model_b_name: str = "Model B",
        metric_name: str = "F1 Score",
    ) -> ABTestResult:
        """
        Compare two models using statistical testing.

        Args:
            model_a_scores: Array of scores from model A (e.g., cross-val scores)
            model_b_scores: Array of scores from model B
            model_a_name: Name of model A
            model_b_name: Name of model B
            metric_name: Name of the metric being compared

        Returns:
            ABTestResult with comparison statistics
        """
        mean_a = np.mean(model_a_scores)
        mean_b = np.mean(model_b_scores)
        improvement = (mean_b - mean_a) / mean_a if mean_a != 0 else 0

        t_stat, p_value = stats.ttest_rel(model_a_scores, model_b_scores)

        statistically_significant = p_value < self.alpha

        if mean_b > mean_a and statistically_significant:
            winner = model_b_name
        elif mean_a > mean_b and statistically_significant:
            winner = model_a_name
        else:
            winner = "No significant difference"

        effect_size = self._cohens_d(model_a_scores, model_b_scores)

        ci_lower, ci_upper = self._bootstrap_ci(model_b_scores - model_a_scores)

        return ABTestResult(
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            metric_name=metric_name,
            model_a_score=float(mean_a),
            model_b_score=float(mean_b),
            improvement=float(improvement),
            p_value=float(p_value),
            statistically_significant=statistically_significant,
            confidence_level=self.confidence_level,
            winner=winner,
            details={
                "t_statistic": float(t_stat),
                "effect_size_cohens_d": float(effect_size),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "model_a_std": float(np.std(model_a_scores)),
                "model_b_std": float(np.std(model_b_scores)),
            },
        )

    def _cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        return (np.mean(group2) - np.mean(group1)) / pooled_std

    def _bootstrap_ci(self, differences: np.ndarray) -> tuple[float, float]:
        """Calculate bootstrap confidence interval for differences."""
        rng = np.random.default_rng(42)
        bootstrap_means = []

        for _ in range(self.n_bootstrap):
            sample = rng.choice(differences, size=len(differences), replace=True)
            bootstrap_means.append(np.mean(sample))

        lower = np.percentile(bootstrap_means, (1 - self.confidence_level) / 2 * 100)
        upper = np.percentile(bootstrap_means, (1 + self.confidence_level) / 2 * 100)

        return lower, upper


class ValidationPipeline:
    """
    Comprehensive validation pipeline for anomaly detection models.

    Provides:
    - Data quality validation
    - Cross-validation with multiple metrics
    - A/B testing for model comparison
    - Benchmark tracking
    """

    def __init__(
        self,
        n_folds: int = 5,
        random_state: int = 42,
        quality_checker: DataQualityChecker | None = None,
        ab_tester: ABTester | None = None,
    ):
        self.n_folds = n_folds
        self.random_state = random_state
        self.quality_checker = quality_checker or DataQualityChecker()
        self.ab_tester = ab_tester or ABTester()
        self._benchmarks: dict[str, list[ValidationResult]] = {}

    def validate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        dataset_name: str = "unknown",
        model_name: str = "unknown",
        run_quality_checks: bool = True,
    ) -> ValidationResult:
        """
        Run full validation pipeline on a model.

        Args:
            model: Model with fit/predict methods
            X: Feature matrix
            y: Labels
            dataset_name: Name of the dataset
            model_name: Name of the model
            run_quality_checks: Whether to run data quality checks

        Returns:
            ValidationResult with all metrics and checks
        """
        start_time = time.time()

        quality_checks = []
        if run_quality_checks:
            quality_checks = self.quality_checker.run_all_checks(X, y)

        cv_results = self._cross_validate(model, X, y)

        final_metrics = self._compute_final_metrics(model, X, y)

        validation_time = time.time() - start_time

        result = ValidationResult(
            dataset_name=dataset_name,
            model_name=model_name,
            accuracy=final_metrics["accuracy"],
            precision=final_metrics["precision"],
            recall=final_metrics["recall"],
            f1_score=final_metrics["f1_score"],
            auc_roc=final_metrics["auc_roc"],
            auc_pr=final_metrics["auc_pr"],
            confusion_matrix=final_metrics["confusion_matrix"],
            quality_checks=quality_checks,
            validation_time_seconds=validation_time,
            num_samples=len(X),
            num_features=X.shape[1] if X.ndim > 1 else 1,
            cross_val_scores=cv_results["f1_scores"],
            additional_metrics=cv_results,
        )

        self._store_benchmark(result)

        logger.info(
            f"Validation complete: {model_name} on {dataset_name} - "
            f"F1={result.f1_score:.4f}, AUC-ROC={result.auc_roc:.4f}"
        )

        return result

    def _cross_validate(self, model: Any, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Perform cross-validation."""
        rng = np.random.default_rng(self.random_state)
        indices = rng.permutation(len(X))

        fold_size = len(X) // self.n_folds
        f1_scores = []
        accuracy_scores = []

        for i in range(self.n_folds):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_folds - 1 else len(X)

            test_idx = indices[test_start:test_end]
            train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if hasattr(model, "fit"):
                model.fit(X_train, y_train)

            if hasattr(model, "predict"):
                y_pred = model.predict(X_test)
            else:
                y_pred = np.zeros_like(y_test)

            f1 = self._compute_f1(y_test, y_pred)
            acc = np.mean(y_test == y_pred)

            f1_scores.append(f1)
            accuracy_scores.append(acc)

        return {
            "f1_scores": f1_scores,
            "accuracy_scores": accuracy_scores,
            "mean_f1": float(np.mean(f1_scores)),
            "std_f1": float(np.std(f1_scores)),
            "mean_accuracy": float(np.mean(accuracy_scores)),
            "std_accuracy": float(np.std(accuracy_scores)),
        }

    def _compute_final_metrics(self, model: Any, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Compute final metrics on full dataset."""
        if hasattr(model, "fit"):
            model.fit(X, y)

        if hasattr(model, "predict"):
            y_pred = model.predict(X)
        else:
            y_pred = np.zeros_like(y)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)
            if y_proba.ndim > 1:
                y_proba = y_proba[:, 1]
        else:
            y_proba = y_pred.astype(float)

        tp = np.sum((y == 1) & (y_pred == 1))
        tn = np.sum((y == 0) & (y_pred == 0))
        fp = np.sum((y == 0) & (y_pred == 1))
        fn = np.sum((y == 1) & (y_pred == 0))

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        auc_roc = self._compute_auc_roc(y, y_proba)
        auc_pr = self._compute_auc_pr(y, y_proba)

        confusion_matrix = np.array([[tn, fp], [fn, tp]])

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "auc_roc": float(auc_roc),
            "auc_pr": float(auc_pr),
            "confusion_matrix": confusion_matrix,
        }

    def _compute_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute F1 score."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    def _compute_auc_roc(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Compute AUC-ROC using trapezoidal rule."""
        sorted_indices = np.argsort(y_scores)[::-1]
        y_true_sorted = y_true[sorted_indices]

        tpr_list = []
        fpr_list = []

        total_pos = np.sum(y_true == 1)
        total_neg = np.sum(y_true == 0)

        if total_pos == 0 or total_neg == 0:
            return 0.5

        tp = 0
        fp = 0

        for label in y_true_sorted:
            if label == 1:
                tp += 1
            else:
                fp += 1
            tpr_list.append(tp / total_pos)
            fpr_list.append(fp / total_neg)

        auc = np.trapz(tpr_list, fpr_list)
        return float(auc)

    def _compute_auc_pr(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Compute AUC-PR (Area Under Precision-Recall Curve)."""
        sorted_indices = np.argsort(y_scores)[::-1]
        y_true_sorted = y_true[sorted_indices]

        precision_list = []
        recall_list = []

        total_pos = np.sum(y_true == 1)
        if total_pos == 0:
            return 0.0

        tp = 0
        fp = 0

        for label in y_true_sorted:
            if label == 1:
                tp += 1
            else:
                fp += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / total_pos

            precision_list.append(precision)
            recall_list.append(recall)

        auc = np.trapz(precision_list, recall_list)
        return float(abs(auc))

    def _store_benchmark(self, result: ValidationResult) -> None:
        """Store validation result as benchmark."""
        key = f"{result.dataset_name}_{result.model_name}"
        if key not in self._benchmarks:
            self._benchmarks[key] = []
        self._benchmarks[key].append(result)

    def compare_to_baseline(
        self,
        current_result: ValidationResult,
        baseline_name: str = "baseline",
    ) -> ABTestResult | None:
        """Compare current result to baseline."""
        baseline_key = f"{current_result.dataset_name}_{baseline_name}"
        if baseline_key not in self._benchmarks:
            return None

        baseline_results = self._benchmarks[baseline_key]
        if not baseline_results:
            return None

        baseline_scores = np.array([r.f1_score for r in baseline_results])
        current_scores = np.array(current_result.cross_val_scores)

        if len(baseline_scores) != len(current_scores):
            min_len = min(len(baseline_scores), len(current_scores))
            baseline_scores = baseline_scores[:min_len]
            current_scores = current_scores[:min_len]

        return self.ab_tester.compare_models(
            baseline_scores,
            current_scores,
            baseline_name,
            current_result.model_name,
            "F1 Score",
        )

    def get_benchmarks(self) -> dict[str, list[ValidationResult]]:
        """Get all stored benchmarks."""
        return self._benchmarks.copy()

    def generate_report(self, result: ValidationResult) -> str:
        """Generate a human-readable validation report."""
        lines = [
            "=" * 60,
            f"VALIDATION REPORT: {result.model_name}",
            f"Dataset: {result.dataset_name}",
            "=" * 60,
            "",
            "PERFORMANCE METRICS:",
            f"  Accuracy:  {result.accuracy:.4f}",
            f"  Precision: {result.precision:.4f}",
            f"  Recall:    {result.recall:.4f}",
            f"  F1 Score:  {result.f1_score:.4f}",
            f"  AUC-ROC:   {result.auc_roc:.4f}",
            f"  AUC-PR:    {result.auc_pr:.4f}",
            "",
            "CROSS-VALIDATION:",
            f"  Mean F1:   {np.mean(result.cross_val_scores):.4f} (+/- {np.std(result.cross_val_scores):.4f})",
            "",
            "DATA QUALITY CHECKS:",
        ]

        for check in result.quality_checks:
            status = "PASS" if check.passed else "FAIL"
            lines.append(f"  [{status}] {check.check_name}: {check.message}")

        lines.extend(
            [
                "",
                "CONFUSION MATRIX:",
                f"  TN={result.confusion_matrix[0, 0]}, FP={result.confusion_matrix[0, 1]}",
                f"  FN={result.confusion_matrix[1, 0]}, TP={result.confusion_matrix[1, 1]}",
                "",
                f"Validation Time: {result.validation_time_seconds:.2f}s",
                f"Samples: {result.num_samples}, Features: {result.num_features}",
                "=" * 60,
            ]
        )

        return "\n".join(lines)
