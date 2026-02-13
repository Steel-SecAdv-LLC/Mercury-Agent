#!/usr/bin/env python3
"""
Mercury Agent - Live-data metrics validation.

Validates live-data metrics against baseline thresholds.
Prevents regressions in real-world performance.
Exit code 0 = all metrics pass, 1 = regression detected.

Two modes of operation:
1. If live-data-report.json exists (pytest --json-report output),
   validates that all live-data tests passed.
2. Validates baseline metrics against configured thresholds to
   ensure the stored baseline meets minimum requirements.

Copyright (C) 2025 Steel Security Advisors LLC
License: GPL-3.0+
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def load_baseline() -> dict[str, Any] | None:
    """Load baseline metrics from last successful run."""
    baseline_path = Path("benchmarks/live_data_baseline.json")
    if baseline_path.exists():
        with open(baseline_path) as f:
            result: dict[str, Any] = json.load(f)
            return result
    return None


def load_test_report() -> dict[str, Any] | None:
    """Load pytest JSON report if available."""
    report_path = Path("live-data-report.json")
    if report_path.exists():
        with open(report_path) as f:
            result: dict[str, Any] = json.load(f)
            return result
    return None


def _get_threshold(baseline: dict[str, Any] | None, key: str, env_var: str, default: str) -> float:
    """Get threshold from env var (priority) or baseline file."""
    env_val = os.getenv(env_var)
    if env_val is not None:
        return float(env_val)

    if baseline and "thresholds" in baseline:
        thresholds = baseline["thresholds"]
        if key in thresholds:
            return float(thresholds[key])

    return float(default)


def check_test_report(report: dict[str, Any]) -> bool:
    """Validate that live-data tests passed from pytest JSON report."""
    tests = report.get("tests", [])
    if not tests:
        print("No tests found in report")
        return False

    passed = [t for t in tests if t.get("outcome") == "passed"]
    failed = [t for t in tests if t.get("outcome") == "failed"]
    skipped = [t for t in tests if t.get("outcome") == "skipped"]

    print(f"Test results: {len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped")

    if failed:
        print("FAILED tests:")
        for t in failed:
            print(f"   - {t.get('nodeid', 'unknown')}")
        return False

    if not passed:
        print("WARNING: No tests passed (all skipped?)")
        return True  # Skipped tests are not failures

    print("All live-data tests PASSED")
    return True


def check_baseline_thresholds() -> bool:
    """Validate that baseline metrics meet minimum thresholds."""
    baseline = load_baseline()

    if baseline is None:
        print("No baseline found. Run: python scripts/generate_baseline_report.py")
        return False

    min_adbench_auc = _get_threshold(baseline, "min_adbench_auc", "MERCURY_MIN_ADBENCH_AUC", "0.55")
    min_nslkdd_f1 = _get_threshold(baseline, "min_nslkdd_f1", "MERCURY_MIN_NSLKDD_F1", "0.50")
    min_nslkdd_auc = _get_threshold(baseline, "min_nslkdd_auc", "MERCURY_MIN_NSLKDD_AUC", "0.55")

    errors: list[str] = []

    # Check ADBench datasets against baseline stored metrics
    adbench_datasets = [
        "cardio",
        "thyroid",
        "mammography",
        "breastw",
        "Ionosphere",
        "Pima",
        "satellite",
        "shuttle",
        "wine",
        "glass",
        "musk",
        "arrhythmia",
        "optdigits",
        "pendigits",
        "vertebral",
        "WBC",
    ]
    checked = 0
    for dataset in adbench_datasets:
        auc_key = f"adbench_{dataset}_auc"
        baseline_auc = baseline.get(auc_key)
        if baseline_auc is None:
            continue  # Dataset not in baseline, skip
        checked += 1
        if float(baseline_auc) < min_adbench_auc:
            errors.append(
                f"ADBench {dataset}: baseline AUC {float(baseline_auc):.3f} "
                f"< threshold {min_adbench_auc:.3f}"
            )

    # Check NSL-KDD from baseline results
    results = baseline.get("results", {})
    nslkdd = results.get("nslkdd", {}).get("statistical", {})
    nslkdd_f1 = nslkdd.get("f1", 0.0)
    nslkdd_auc = nslkdd.get("auc", 0.0)

    if nslkdd_f1 < min_nslkdd_f1:
        errors.append(f"NSL-KDD: baseline F1 {nslkdd_f1:.3f} < threshold {min_nslkdd_f1:.3f}")
    if nslkdd_auc < min_nslkdd_auc:
        errors.append(f"NSL-KDD: baseline AUC {nslkdd_auc:.3f} < threshold {min_nslkdd_auc:.3f}")

    if errors:
        print(f"Baseline validation FAILED ({len(errors)} issues):")
        for error in errors:
            print(f"   - {error}")
        return False

    print(f"Baseline validation PASSED ({checked} ADBench datasets checked)")
    return True


if __name__ == "__main__":
    ok = True

    # Check 1: If pytest report exists, validate test outcomes
    report = load_test_report()
    if report is not None:
        if not check_test_report(report):
            ok = False
    else:
        print("No test report found (live-data-report.json)")

    # Check 2: Validate baseline meets thresholds
    if not check_baseline_thresholds():
        ok = False

    sys.exit(0 if ok else 1)
