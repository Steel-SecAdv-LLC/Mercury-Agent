"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

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
Ethical Scalar Validation for Production Deployment

Validates that optimized ethical scalars (omnibenevolent, omni_compassionate, omni_justitia)
maintain acceptable performance (<5% degradation) on realistic infrastructure scenarios.

Tests the 20-24% reductions applied in commit f2eea48:
- omnibenevolent: 1.45 → 1.10 (24.1% reduction)
- omni_compassionate: 1.22 → 1.10 (9.8% reduction)
- omni_justitia: 1.20 → 1.10 (8.3% reduction)
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class EthicalScalars:
    """Ethical scalars for testing."""

    omnibenevolent: float = 1.10
    omni_compassionate: float = 1.10
    omni_justitia: float = 1.10


def generate_infrastructure_scenarios(n_scenarios: int = 500) -> list[dict[str, Any]]:
    """Generate realistic infrastructure anomaly scenarios with ethical context."""
    np.random.seed(42)

    scenarios = []

    scenario_types = [
        {"type": "healthcare", "base_severity": 0.8, "ethical_weight": 1.5},
        {"type": "essential_workers", "base_severity": 0.7, "ethical_weight": 1.4},
        {"type": "space_infrastructure", "base_severity": 0.6, "ethical_weight": 1.2},
        {"type": "energy_grid", "base_severity": 0.75, "ethical_weight": 1.3},
        {"type": "water_supply", "base_severity": 0.85, "ethical_weight": 1.6},
        {"type": "economic_sector", "base_severity": 0.5, "ethical_weight": 1.0},
    ]

    for i in range(n_scenarios):
        scenario_type = scenario_types[i % len(scenario_types)]

        is_anomaly = np.random.rand() < 0.1

        if is_anomaly:
            base_score = np.random.uniform(0.7, 1.0) * scenario_type["base_severity"]
            humanitarian_impact = np.random.uniform(0.6, 1.0)
            survivor_priority = np.random.uniform(0.7, 1.0)
        else:
            base_score = np.random.uniform(0.0, 0.3)
            humanitarian_impact = np.random.uniform(0.0, 0.4)
            survivor_priority = np.random.uniform(0.0, 0.4)

        noise = np.random.randn(10) * 0.1
        features = np.array([base_score] * 10) + noise

        scenarios.append(
            {
                "features": features,
                "base_score": base_score,
                "humanitarian_impact": humanitarian_impact,
                "survivor_priority": survivor_priority,
                "ethical_weight": scenario_type["ethical_weight"],
                "type": scenario_type["type"],
                "is_anomaly": is_anomaly,
            }
        )

    return scenarios


def apply_ethical_scalars(
    base_score: float,
    humanitarian_impact: float,
    survivor_priority: float,
    ethical_weight: float,
    scalars: EthicalScalars,
) -> float:
    """Apply ethical scalars to modify base anomaly score."""
    benevolence_factor = scalars.omnibenevolent * humanitarian_impact
    compassion_factor = scalars.omni_compassionate * survivor_priority
    justice_factor = scalars.omni_justitia * ethical_weight

    ethical_multiplier = (benevolence_factor + compassion_factor + justice_factor) / 3.0

    adjusted_score = base_score * ethical_multiplier

    return float(np.clip(adjusted_score, 0.0, 1.0))


def evaluate_with_scalars(
    scenarios: list[dict[str, Any]], scalars: EthicalScalars, threshold: float = 0.5
) -> dict[str, Any]:
    """Evaluate detection performance with given ethical scalars."""
    start_time = time.time()

    predictions = []
    ground_truth = []
    adjusted_scores = []

    for scenario in scenarios:
        adjusted_score = apply_ethical_scalars(
            scenario["base_score"],
            scenario["humanitarian_impact"],
            scenario["survivor_priority"],
            scenario["ethical_weight"],
            scalars,
        )

        adjusted_scores.append(adjusted_score)
        predictions.append(1 if adjusted_score > threshold else 0)
        ground_truth.append(1 if scenario["is_anomaly"] else 0)

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    adjusted_scores = np.array(adjusted_scores)

    true_positives = np.sum((predictions == 1) & (ground_truth == 1))
    false_positives = np.sum((predictions == 1) & (ground_truth == 0))
    true_negatives = np.sum((predictions == 0) & (ground_truth == 0))
    false_negatives = np.sum((predictions == 0) & (ground_truth == 1))

    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    accuracy = (true_positives + true_negatives) / len(ground_truth)

    false_positive_rate = false_positives / (false_positives + true_negatives + 1e-8)
    false_negative_rate = false_negatives / (false_negatives + true_positives + 1e-8)

    elapsed_time = time.time() - start_time

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "accuracy": float(accuracy),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "time_seconds": float(elapsed_time),
        "true_positives": int(true_positives),
        "false_positives": int(false_positives),
        "true_negatives": int(true_negatives),
        "false_negatives": int(false_negatives),
        "mean_adjusted_score": float(np.mean(adjusted_scores)),
        "std_adjusted_score": float(np.std(adjusted_scores)),
        "scalars": {
            "omnibenevolent": scalars.omnibenevolent,
            "omni_compassionate": scalars.omni_compassionate,
            "omni_justitia": scalars.omni_justitia,
        },
    }


def analyze_by_scenario_type(
    scenarios: list[dict[str, Any]], scalars: EthicalScalars
) -> dict[str, Any]:
    """Analyze performance breakdown by scenario type."""
    type_results = {}

    for scenario_type in [
        "healthcare",
        "essential_workers",
        "space_infrastructure",
        "energy_grid",
        "water_supply",
        "economic_sector",
    ]:
        type_scenarios = [s for s in scenarios if s["type"] == scenario_type]

        if type_scenarios:
            type_results[scenario_type] = evaluate_with_scalars(type_scenarios, scalars)

    return type_results


def main() -> int:
    """Run comprehensive ethical scalar validation."""
    print("=" * 80)
    print("ETHICAL SCALAR VALIDATION - PRODUCTION DEPLOYMENT")
    print("=" * 80)
    print()
    print("Testing ethical scalar reductions from commit f2eea48:")
    print("  - omnibenevolent: 1.45 → 1.10 (24.1% reduction)")
    print("  - omni_compassionate: 1.22 → 1.10 (9.8% reduction)")
    print("  - omni_justitia: 1.20 → 1.10 (8.3% reduction)")
    print()
    print("Acceptance Criteria: <5% degradation in F1-score on infrastructure scenarios")
    print("=" * 80)
    print()

    print("Generating realistic infrastructure anomaly scenarios...")
    scenarios = generate_infrastructure_scenarios(n_scenarios=500)
    anomaly_count = sum(1 for s in scenarios if s["is_anomaly"])
    print(f"  Generated {len(scenarios)} scenarios")
    print(
        f"  Anomalies: {anomaly_count} / {len(scenarios)} ({anomaly_count/len(scenarios)*100:.1f}%)"
    )
    print()

    baseline_scalars = EthicalScalars(
        omnibenevolent=1.45, omni_compassionate=1.22, omni_justitia=1.20
    )
    print("BASELINE Ethical Scalars:")
    print(f"  omnibenevolent: {baseline_scalars.omnibenevolent}")
    print(f"  omni_compassionate: {baseline_scalars.omni_compassionate}")
    print(f"  omni_justitia: {baseline_scalars.omni_justitia}")
    print()

    print("Evaluating BASELINE configuration...")
    baseline_results = evaluate_with_scalars(scenarios, baseline_scalars)
    print(f"  Precision: {baseline_results['precision']:.4f}")
    print(f"  Recall: {baseline_results['recall']:.4f}")
    print(f"  F1-Score: {baseline_results['f1_score']:.4f}")
    print(f"  Accuracy: {baseline_results['accuracy']:.4f}")
    print(f"  False Positive Rate: {baseline_results['false_positive_rate']:.4f}")
    print(f"  False Negative Rate: {baseline_results['false_negative_rate']:.4f}")
    print(f"  Time: {baseline_results['time_seconds']:.4f}s")
    print()

    optimized_scalars = EthicalScalars(
        omnibenevolent=1.10, omni_compassionate=1.10, omni_justitia=1.10
    )
    print("OPTIMIZED Ethical Scalars:")
    print(f"  omnibenevolent: {optimized_scalars.omnibenevolent}")
    print(f"  omni_compassionate: {optimized_scalars.omni_compassionate}")
    print(f"  omni_justitia: {optimized_scalars.omni_justitia}")
    print()

    print("Evaluating OPTIMIZED configuration...")
    optimized_results = evaluate_with_scalars(scenarios, optimized_scalars)
    print(f"  Precision: {optimized_results['precision']:.4f}")
    print(f"  Recall: {optimized_results['recall']:.4f}")
    print(f"  F1-Score: {optimized_results['f1_score']:.4f}")
    print(f"  Accuracy: {optimized_results['accuracy']:.4f}")
    print(f"  False Positive Rate: {optimized_results['false_positive_rate']:.4f}")
    print(f"  False Negative Rate: {optimized_results['false_negative_rate']:.4f}")
    print(f"  Time: {optimized_results['time_seconds']:.4f}s")
    print()

    print("=" * 80)
    print("DEGRADATION ANALYSIS")
    print("=" * 80)

    f1_change_pct = (
        (optimized_results["f1_score"] - baseline_results["f1_score"])
        / baseline_results["f1_score"]
        * 100
    )
    precision_change_pct = (
        (optimized_results["precision"] - baseline_results["precision"])
        / baseline_results["precision"]
        * 100
    )
    recall_change_pct = (
        (optimized_results["recall"] - baseline_results["recall"])
        / baseline_results["recall"]
        * 100
    )
    accuracy_change_pct = (
        (optimized_results["accuracy"] - baseline_results["accuracy"])
        / baseline_results["accuracy"]
        * 100
    )

    print(f"F1-Score Change: {f1_change_pct:+.2f}%")
    print(f"Precision Change: {precision_change_pct:+.2f}%")
    print(f"Recall Change: {recall_change_pct:+.2f}%")
    print(f"Accuracy Change: {accuracy_change_pct:+.2f}%")
    print()

    degradation_threshold = -5.0

    if f1_change_pct >= degradation_threshold:
        print(
            f"✅ VALIDATION PASSED: F1-score degradation ({f1_change_pct:.2f}%) "
            "is within acceptable threshold (<5%)"
        )
        validation_passed = True
    else:
        print(
            f"❌ VALIDATION FAILED: F1-score degradation ({f1_change_pct:.2f}%) "
            "exceeds threshold (>5%)"
        )
        print("   RECOMMENDATION: Revert ethical scalar changes")
        validation_passed = False

    print()

    print("=" * 80)
    print("SCENARIO TYPE BREAKDOWN")
    print("=" * 80)
    print()

    print("BASELINE by scenario type:")
    baseline_type_results = analyze_by_scenario_type(scenarios, baseline_scalars)
    for scenario_type, results in baseline_type_results.items():
        f1 = results["f1_score"]
        prec = results["precision"]
        rec = results["recall"]
        print(f"  {scenario_type:20s}: F1={f1:.4f}, Precision={prec:.4f}, Recall={rec:.4f}")
    print()

    print("OPTIMIZED by scenario type:")
    optimized_type_results = analyze_by_scenario_type(scenarios, optimized_scalars)
    for scenario_type, results in optimized_type_results.items():
        baseline_f1 = baseline_type_results[scenario_type]["f1_score"]
        f1_change = (
            ((results["f1_score"] - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0
        )
        f1 = results["f1_score"]
        prec = results["precision"]
        rec = results["recall"]
        print(f"  {scenario_type:20s}: F1={f1:.4f} ({f1_change:+.1f}%)")
        print(f"      Prec={prec:.4f}, Rec={rec:.4f}")
    print()

    print("=" * 80)
    print("HUMANITARIAN IMPACT ANALYSIS")
    print("=" * 80)
    print()

    high_impact_scenarios = [
        s for s in scenarios if s["humanitarian_impact"] > 0.7 and s["is_anomaly"]
    ]
    if high_impact_scenarios:
        print(f"High humanitarian impact scenarios: {len(high_impact_scenarios)}")
        baseline_high_impact = evaluate_with_scalars(high_impact_scenarios, baseline_scalars)
        optimized_high_impact = evaluate_with_scalars(high_impact_scenarios, optimized_scalars)
        print(f"  BASELINE - Recall on high-impact: {baseline_high_impact['recall']:.4f}")
        print(f"  OPTIMIZED - Recall on high-impact: {optimized_high_impact['recall']:.4f}")
        recall_change_high = (
            (
                (optimized_high_impact["recall"] - baseline_high_impact["recall"])
                / baseline_high_impact["recall"]
                * 100
            )
            if baseline_high_impact["recall"] > 0
            else 0
        )
        print(f"  Change: {recall_change_high:+.2f}%")

        if recall_change_high < -5:
            print("  ⚠️  WARNING: Significant degradation in high humanitarian impact scenarios")
    print()

    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print()

    validation_results = {
        "validation_passed": validation_passed,
        "acceptance_threshold_pct": degradation_threshold,
        "baseline": baseline_results,
        "optimized": optimized_results,
        "changes": {
            "f1_score_pct": float(f1_change_pct),
            "precision_pct": float(precision_change_pct),
            "recall_pct": float(recall_change_pct),
            "accuracy_pct": float(accuracy_change_pct),
        },
        "scenario_type_breakdown": {
            "baseline": baseline_type_results,
            "optimized": optimized_type_results,
        },
    }

    output_path = Path("benchmarks/ethical_scalar_validation_results.json")
    with open(output_path, "w") as f:
        json.dump(validation_results, f, indent=2)

    print(f"Results saved to: {output_path}")
    print()

    if validation_passed:
        print("✅ ETHICAL SCALAR OPTIMIZATION VALIDATED FOR PRODUCTION DEPLOYMENT")
    else:
        print("❌ ETHICAL SCALAR OPTIMIZATION REQUIRES REVIEW BEFORE DEPLOYMENT")

    return 0 if validation_passed else 1


if __name__ == "__main__":
    exit(main())
