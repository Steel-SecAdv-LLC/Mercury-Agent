#!/usr/bin/env python3
"""
Mercury Agent - Live-data metrics validation.

Validates live-data metrics against baseline thresholds.
Prevents regressions in real-world performance.
Exit code 0 = all metrics pass, 1 = regression detected.

Reads thresholds from the baseline file if available, falling back
to environment variables for CI override.

Copyright (C) 2025 Steel Security Advisors LLC
License: GPL-3.0+
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def load_baseline() -> dict | None:
    """Load baseline metrics from last successful run."""
    baseline_path = Path("benchmarks/live_data_baseline.json")
    if baseline_path.exists():
        with open(baseline_path) as f:
            return json.load(f)
    return None


def load_current_metrics() -> dict[str, float]:
    """Load current test results."""
    metrics_path = Path("live-data-report.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}


def _get_threshold(baseline: dict | None, key: str, env_var: str, default: str) -> float:
    """Get threshold from baseline file or environment variable."""
    # Environment variable takes priority (CI override)
    env_val = os.getenv(env_var)
    if env_val is not None:
        return float(env_val)

    # Try baseline thresholds section
    if baseline and "thresholds" in baseline:
        thresholds = baseline["thresholds"]
        if key in thresholds:
            return float(thresholds[key])

    return float(default)


def check_thresholds() -> bool:
    """Verify metrics meet minimum thresholds."""
    baseline = load_baseline()

    if baseline is None:
        print("No baseline found. Run: python scripts/generate_baseline_report.py")
        return False

    min_adbench_auc = _get_threshold(baseline, "min_adbench_auc", "MERCURY_MIN_ADBENCH_AUC", "0.85")
    min_nslkdd_f1 = _get_threshold(baseline, "min_nslkdd_f1", "MERCURY_MIN_NSLKDD_F1", "0.50")
    min_nslkdd_auc = _get_threshold(baseline, "min_nslkdd_auc", "MERCURY_MIN_NSLKDD_AUC", "0.55")
    allow_regression = _get_threshold(
        baseline, "allow_regression_percent", "MERCURY_ALLOW_REGRESSION", "0.02"
    )
    # Normalize: baseline stores as percent (2.0), env var as fraction (0.02)
    if allow_regression > 1.0:
        allow_regression = allow_regression / 100.0

    current = load_current_metrics()

    errors: list[str] = []

    # Check ADBench datasets (all 16 validated datasets)
    adbench_datasets = [
        "cardio", "thyroid", "mammography", "breastw",
        "Ionosphere", "Pima", "satellite", "shuttle",
        "wine", "glass", "musk", "arrhythmia",
        "optdigits", "pendigits", "vertebral", "WBC",
    ]
    for dataset in adbench_datasets:
        current_auc = current.get(f"adbench_{dataset}_auc", 0.0)
        if current_auc < min_adbench_auc:
            errors.append(
                f"ADBench {dataset}: AUC {current_auc:.3f} < threshold {min_adbench_auc:.3f}"
            )

        # Check for regression vs baseline
        baseline_auc = baseline.get(f"adbench_{dataset}_auc", current_auc)
        regression = baseline_auc - current_auc
        if regression > allow_regression:
            errors.append(
                f"ADBench {dataset}: Regression {regression:.1%} > allowed {allow_regression:.1%}"
            )

    # Check NSL-KDD
    nslkdd_f1 = current.get("nslkdd_f1", 0.0)
    if nslkdd_f1 < min_nslkdd_f1:
        errors.append(f"NSL-KDD: F1 {nslkdd_f1:.3f} < threshold {min_nslkdd_f1:.3f}")

    nslkdd_auc = current.get("nslkdd_auc", 0.0)
    if nslkdd_auc < min_nslkdd_auc:
        errors.append(f"NSL-KDD: AUC {nslkdd_auc:.3f} < threshold {min_nslkdd_auc:.3f}")

    if errors:
        print("Live-data validation FAILED:")
        for error in errors:
            print(f"   - {error}")
        return False

    print("Live-data validation PASSED")
    return True


if __name__ == "__main__":
    success = check_thresholds()
    sys.exit(0 if success else 1)
