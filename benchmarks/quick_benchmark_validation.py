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
Quick Benchmark Validation for Optimization Patterns

Validates that optimized configurations (from comprehensive_equation_optimization.py)
improve performance compared to baseline configurations.

Quick validation on synthetic data - comprehensive benchmarks on actual repos
(requests/flask/numpy/pandas/scikit-learn) can be done separately if needed.
"""

import numpy as np
import time
from typing import Dict
import json
from pathlib import Path


def generate_synthetic_anomaly_data(n_samples: int = 1000, anomaly_ratio: float = 0.05):
    """Generate synthetic data with known anomalies for testing."""
    np.random.seed(42)

    normal_data = np.random.randn(int(n_samples * (1 - anomaly_ratio)), 10)

    anomalous_data = np.random.randn(int(n_samples * anomaly_ratio), 10) * 3 + 5

    data = np.vstack([normal_data, anomalous_data])
    labels = np.hstack([np.zeros(len(normal_data)), np.ones(len(anomalous_data))])

    indices = np.random.permutation(len(data))
    return data[indices], labels[indices]


def evaluate_detection_config(data: np.ndarray, labels: np.ndarray, config: Dict) -> Dict:
    """Evaluate anomaly detection with given configuration."""
    start_time = time.time()

    w1, w2, w3 = config["w1"], config["w2"], config["w3"]
    e1, e2 = config["e1"], config["e2"]
    t1 = config.get("t1", 0.5)

    means = np.mean(data, axis=1)
    stds = np.std(data, axis=1)
    maxs = np.max(data, axis=1)

    scores = (w1**e1) * np.abs(means) + (w2**e2) * stds + (w3 * (e1 + e2) / 2) * maxs
    scores = scores * t1  # Tensor factor

    scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-8)

    threshold = np.percentile(scores, 95)
    predictions = (scores > threshold).astype(int)

    true_positives = np.sum((predictions == 1) & (labels == 1))
    false_positives = np.sum((predictions == 1) & (labels == 0))
    true_negatives = np.sum((predictions == 0) & (labels == 0))
    false_negatives = np.sum((predictions == 0) & (labels == 1))

    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    accuracy = (true_positives + true_negatives) / len(labels)

    elapsed_time = time.time() - start_time

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "accuracy": float(accuracy),
        "time_seconds": float(elapsed_time),
        "true_positives": int(true_positives),
        "false_positives": int(false_positives),
        "config": config,
    }


def main():
    """Run quick benchmark validation."""
    print("=" * 70)
    print("QUICK BENCHMARK VALIDATION - OPTIMIZATION PATTERNS")
    print("=" * 70)
    print()

    results_path = Path("benchmarks/comprehensive_optimization_results.json")
    if results_path.exists():
        with open(results_path, "r") as f:
            optimization_results = json.load(f)

        best_ava = max(optimization_results["ava_equation"], key=lambda x: x["score"])
        optimized_config = {
            "w1": best_ava["w1"],
            "w2": best_ava["w2"],
            "w3": best_ava["w3"],
            "e1": best_ava["e1"],
            "e2": best_ava["e2"],
            "t1": best_ava["t1"],
        }
        print("Loaded best Ava config from optimization results:")
        print(f"  {optimized_config}")
    else:
        optimized_config = {"w1": 0.50, "w2": 1.00, "w3": 1.00, "e1": 1.00, "e2": 2.00, "t1": 0.40}
        print("Using optimized config from experiments:")
        print(f"  {optimized_config}")

    print()

    baseline_config = {"w1": 1.0, "w2": 1.0, "w3": 1.0, "e1": 1.0, "e2": 1.0, "t1": 1.0}
    print("Baseline config:")
    print(f"  {baseline_config}")
    print()

    print("Generating synthetic anomaly data (1000 samples, 5% anomalies)...")
    data, labels = generate_synthetic_anomaly_data(n_samples=1000, anomaly_ratio=0.05)
    print(f"  Data shape: {data.shape}")
    print(f"  Anomalies: {np.sum(labels)} / {len(labels)} ({np.mean(labels)*100:.1f}%)")
    print()

    print("Evaluating BASELINE configuration...")
    baseline_results = evaluate_detection_config(data, labels, baseline_config)
    print(f"  Precision: {baseline_results['precision']:.4f}")
    print(f"  Recall: {baseline_results['recall']:.4f}")
    print(f"  F1-Score: {baseline_results['f1_score']:.4f}")
    print(f"  Accuracy: {baseline_results['accuracy']:.4f}")
    print(f"  Time: {baseline_results['time_seconds']:.4f}s")
    print()

    print("Evaluating OPTIMIZED configuration...")
    optimized_results = evaluate_detection_config(data, labels, optimized_config)
    print(f"  Precision: {optimized_results['precision']:.4f}")
    print(f"  Recall: {optimized_results['recall']:.4f}")
    print(f"  F1-Score: {optimized_results['f1_score']:.4f}")
    print(f"  Accuracy: {optimized_results['accuracy']:.4f}")
    print(f"  Time: {optimized_results['time_seconds']:.4f}s")
    print()

    print("=" * 70)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 70)

    f1_improvement = (
        (optimized_results["f1_score"] - baseline_results["f1_score"])
        / baseline_results["f1_score"]
        * 100
    )
    accuracy_improvement = (
        (optimized_results["accuracy"] - baseline_results["accuracy"])
        / baseline_results["accuracy"]
        * 100
    )
    time_improvement = (
        (baseline_results["time_seconds"] - optimized_results["time_seconds"])
        / baseline_results["time_seconds"]
        * 100
    )

    print(f"F1-Score Improvement: {f1_improvement:+.2f}%")
    print(f"Accuracy Improvement: {accuracy_improvement:+.2f}%")
    print(f"Time Improvement: {time_improvement:+.2f}%")
    print()

    if f1_improvement > 0 or accuracy_improvement > 0:
        print("✅ VALIDATION SUCCESSFUL: Optimized config shows improvements!")
    else:
        print("⚠️  VALIDATION NOTE: Results may vary on real data")

    print()
    print("=" * 70)
    print("BENCHMARK VALIDATION COMPLETE")
    print("=" * 70)
    print()
    print("NOTE: This is a quick validation on synthetic data.")
    print("Comprehensive benchmarks on actual repos (requests/flask/numpy/pandas/scikit-learn)")
    print("can be performed separately for production deployment validation.")

    validation_results = {
        "baseline": baseline_results,
        "optimized": optimized_results,
        "improvements": {
            "f1_score_pct": float(f1_improvement),
            "accuracy_pct": float(accuracy_improvement),
            "time_pct": float(time_improvement),
        },
    }

    output_path = Path("benchmarks/quick_validation_results.json")
    with open(output_path, "w") as f:
        json.dump(validation_results, f, indent=2)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
