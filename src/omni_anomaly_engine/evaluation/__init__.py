"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

Evaluation Module for Anomaly Detection

Provides standard evaluation metrics and baseline comparisons used in
academic anomaly detection research.

Metrics implemented:
- AUC-ROC: Area Under ROC Curve
- AUC-PR: Area Under Precision-Recall Curve
- F1-Score: With optimal threshold search
- Precision@K: Precision at top-K predictions
- Point-Adjusted F1: Time-series segment-aware evaluation
- Range-Based F1: Overlap-based evaluation (Tatbul et al., NeurIPS 2018)

These metrics match those used in benchmark papers:
- OmniAnomaly (KDD 2019)
- MSCRED (AAAI 2019)
- DAGMM (ICLR 2018)
- TranAD (VLDB 2022)
"""

from .baselines import (
    BASELINE_RESULTS,
    BaselineComparison,
    compare_to_baselines,
    get_baseline_citations,
    get_sota_for_dataset,
    list_available_datasets,
    print_baseline_table,
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
    "BASELINE_RESULTS",
    "AnomalyMetrics",
    "BaselineComparison",
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
]
