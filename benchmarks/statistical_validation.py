# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Statistical validation of benchmark results using t-tests and confidence intervals."""

from typing import Any

import numpy as np
from scipy import stats


def statistical_analysis(
    baseline_results: np.ndarray, improved_results: np.ndarray
) -> dict[str, Any]:
    """
    Perform statistical validation comparing baseline vs improved performance.

    Args:
        baseline_results: Array of baseline measurements
        improved_results: Array of improved measurements

    Returns:
        Dict with statistical test results
    """
    t_statistic, p_value = stats.ttest_rel(baseline_results, improved_results)

    improvement = (
        (baseline_results.mean() - improved_results.mean()) / baseline_results.mean()
    ) * 100

    diff = baseline_results - improved_results
    ci = stats.t.interval(0.95, len(diff) - 1, loc=np.mean(diff), scale=stats.sem(diff))

    pooled_std = np.sqrt((baseline_results.std() ** 2 + improved_results.std() ** 2) / 2)
    cohens_d = (
        (baseline_results.mean() - improved_results.mean()) / pooled_std if pooled_std > 0 else 0
    )

    return {
        "t_statistic": float(t_statistic),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "improvement_percent": float(improvement),
        "confidence_interval_95": (float(ci[0]), float(ci[1])),
        "cohens_d": float(cohens_d),
        "effect_size": (
            "large" if abs(cohens_d) > 0.8 else "medium" if abs(cohens_d) > 0.5 else "small"
        ),
    }


if __name__ == "__main__":
    baseline_times = np.random.normal(10.0, 2.0, 100)
    improved_times = np.random.normal(8.0, 1.5, 100)

    results = statistical_analysis(baseline_times, improved_times)

    print("=== STATISTICAL VALIDATION ===")
    print(f"Improvement: {results['improvement_percent']:.2f}%")
    print(f"T-statistic: {results['t_statistic']:.3f}")
    print(f"P-value: {results['p_value']:.6f}")
    print(f"Statistically significant: {results['significant']}")
    print(
        "95% CI: ({:.2f}, {:.2f})".format(
            results["confidence_interval_95"][0], results["confidence_interval_95"][1]
        )
    )
    print(f"Cohen's d: {results['cohens_d']:.3f} ({results['effect_size']})")
