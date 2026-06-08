# Copyright (C) 2025 Steel Security Advisors LLC
"""Standardized Metrics for Anomaly Detection.

Provides evaluation metrics matching literature standards:
- Image-level metrics (AUROC, AUPRC, F1-max)
- Pixel-level metrics (pixel-AUROC, PRO)
- Video metrics (frame-level AUC)
- Novel metrics (PGn, PBn from CVPR 2025)

All metrics are compatible with both numpy and torch tensors.
"""

from __future__ import annotations

from omni_mercury_engine.metrics.anomaly_metrics import (
    AnomalyMetrics,
    compute_auprc,
    compute_auroc,
    compute_f1_max,
    compute_optimal_threshold,
    compute_pixel_auroc,
    compute_pro,
)
from omni_mercury_engine.metrics.benchmark_evaluator import BenchmarkEvaluator, EvaluationResult

__all__ = [
    # Core metrics
    "AnomalyMetrics",
    # Evaluation
    "BenchmarkEvaluator",
    "EvaluationResult",
    "compute_auprc",
    "compute_auroc",
    "compute_f1_max",
    "compute_optimal_threshold",
    "compute_pixel_auroc",
    "compute_pro",
]
