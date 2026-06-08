# Copyright (C) 2025 Steel Security Advisors LLC
"""Evaluation Module for Anomaly Detection."""

from __future__ import annotations

from .baselines import (
    BASELINE_RESULTS,
    BaselineComparison,
    compare_to_baselines,
    get_baseline_citations,
    get_sota_for_dataset,
    list_available_datasets,
    print_baseline_table,
)
from .benchmark_diagnostics import (
    BenchmarkDiagnostics,
    DiagnosticResult,
    MetricDiscrepancy,
    run_diagnostic_benchmark,
)
from .metrics import (
    AnomalyMetrics,
    compute_auc_pr,
    compute_auc_roc,
    compute_best_f1,
    compute_f1,
    compute_point_adjusted_f1,
    compute_precision_at_k,
    compute_range_based_f1,
    evaluate_anomaly_detection,
    print_metrics_report,
)

__all__ = [
    # Original exports
    "BASELINE_RESULTS",
    "AnomalyMetrics",
    "BaselineComparison",
    # Benchmark diagnostics (for F1=0 debugging)
    "BenchmarkDiagnostics",
    "DiagnosticResult",
    "MetricDiscrepancy",
    "compare_to_baselines",
    "compute_auc_pr",
    "compute_auc_roc",
    "compute_best_f1",
    "compute_f1",
    "compute_point_adjusted_f1",
    "compute_precision_at_k",
    "compute_range_based_f1",
    "evaluate_anomaly_detection",
    "get_baseline_citations",
    "get_sota_for_dataset",
    "list_available_datasets",
    "print_baseline_table",
    "print_metrics_report",
    "run_diagnostic_benchmark",
]
