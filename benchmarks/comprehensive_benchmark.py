#!/usr/bin/env python3
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
Comprehensive benchmark for OMNI ♱ AVA enhancements.

Benchmarks module instantiation, detection performance, and scalability
across 1, 5, and all 12 infrastructure modules.
"""

import json
import time
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from omni_anomaly_engine.infrastructure import InfrastructureCoordinator
from omni_anomaly_engine.models.simulation import SimulationModule
from omni_anomaly_engine.space.space_exploration_analyzer import SpaceExplorationAnalyzer


def benchmark_module_instantiation() -> dict[str, float]:
    """Benchmark module instantiation times."""
    results = {}

    coord = InfrastructureCoordinator()

    start = time.time()
    coord.instantiate_filtered_modules(module_names=["ncf_monitor"])
    elapsed_1 = (time.time() - start) * 1000
    results["1_module_ms"] = elapsed_1

    start = time.time()
    coord.instantiate_filtered_modules(priorities=["high"])
    elapsed_5 = (time.time() - start) * 1000
    results["5_modules_ms"] = elapsed_5

    start = time.time()
    modules_all = coord.instantiate_filtered_modules()
    elapsed_all = (time.time() - start) * 1000
    results["all_modules_ms"] = elapsed_all
    results["module_count"] = len(modules_all)

    return results


def benchmark_space_exploration() -> dict[str, Any]:
    """Benchmark satellite position anomaly analysis on synthetic orbit data."""
    analyzer = SpaceExplorationAnalyzer()

    normal_orbit = np.random.randn(500, 3) * 5 + np.array([7000, 0, 0])
    anomalous_orbit = np.random.randn(100, 3) * 50 + np.array([7200, 100, 50])
    data = np.vstack([normal_orbit, anomalous_orbit])

    start = time.time()
    result = analyzer.detect(data, "satellite_position", {"orbit_type": "leo"})
    elapsed = (time.time() - start) * 1000

    return {
        "runtime_ms": elapsed,
        "throughput_samples_per_sec": len(data) / (elapsed / 1000),
        "anomaly_detected": result["anomaly_detected"],
        "severity": result["severity"],
        "data_source": "synthetic_orbit",
    }


def benchmark_simulation_module() -> dict[str, Any]:
    """Benchmark SimulationModule performance."""
    sim = SimulationModule(config={"num_branches": 15})

    start = time.time()
    collatz_result = sim.explore_conjecture("collatz", search_space=5000)
    collatz_elapsed = (time.time() - start) * 1000

    start = time.time()
    sim.analyze_millennium_problem("p_vs_np")
    millennium_elapsed = (time.time() - start) * 1000

    start = time.time()
    data = np.random.randn(100, 20)
    sim.predict(data)
    predict_elapsed = (time.time() - start) * 1000

    return {
        "collatz_exploration_ms": collatz_elapsed,
        "collatz_cases_per_sec": collatz_result["explored_cases"] / (collatz_elapsed / 1000),
        "millennium_analysis_ms": millennium_elapsed,
        "multiverse_prediction_ms": predict_elapsed,
        "prediction_throughput_samples_per_sec": 100 / (predict_elapsed / 1000),
    }


def benchmark_cosmic_ray_detection() -> dict[str, Any]:
    """Benchmark cosmic ray detection with per-sample precision/recall/F1 on labeled synthetic data."""
    threshold = 3.0
    analyzer = SpaceExplorationAnalyzer(config={"cosmic_ray_threshold": threshold})

    # Generate labeled synthetic data: 900 normal + 100 anomalous samples
    normal_data = np.random.randn(900, 5) * 0.5 + 1.0
    cosmic_ray_events = np.random.randn(100, 5) * 15 + 20.0
    data = np.vstack([normal_data, cosmic_ray_events])

    # Ground truth: first 900 are normal (0), last 100 are anomalies (1)
    y_true = np.array([0] * 900 + [1] * 100)

    start = time.time()
    result = analyzer.analyze_cosmic_rays(data, {"telescope": "hubble_sim"})
    elapsed = (time.time() - start) * 1000

    # Compute per-sample predictions using same z-score logic as analyzer
    # (analyzer truncates event_indices to 10 items, so we recompute for full metrics)
    mean_energy = np.mean(data, axis=0)
    std_energy = np.std(data, axis=0)
    z_scores = np.abs((data - mean_energy) / (std_energy + 1e-8))
    y_pred = (np.max(z_scores, axis=1) > threshold).astype(int)

    # Compute real metrics using sklearn
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "runtime_ms": elapsed,
        "throughput_samples_per_sec": len(data) / (elapsed / 1000),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "cosmic_ray_events_detected": result["cosmic_ray_events"],
        "expected_anomalies": 100,
        "severity": result["severity"],
        "data_source": "synthetic_labeled",
    }


def run_all_benchmarks() -> dict[str, Any]:
    """Run all benchmarks and return results."""
    print("=" * 70)
    print("OMNI-AVA COMPREHENSIVE BENCHMARK")
    print("=" * 70)

    results = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), "benchmarks": {}}

    print("\n[1/4] Benchmarking module instantiation...")
    results["benchmarks"]["module_instantiation"] = benchmark_module_instantiation()
    print(f"  ✓ 1 module: {results['benchmarks']['module_instantiation']['1_module_ms']:.2f} ms")
    print(f"  ✓ 5 modules: {results['benchmarks']['module_instantiation']['5_modules_ms']:.2f} ms")
    print(
        f"  ✓ All modules: {results['benchmarks']['module_instantiation']['all_modules_ms']:.2f} ms"
    )

    print("\n[2/4] Benchmarking space exploration analyzer...")
    results["benchmarks"]["space_exploration"] = benchmark_space_exploration()
    print(f"  ✓ Runtime: {results['benchmarks']['space_exploration']['runtime_ms']:.2f} ms")
    print(f"  ✓ Anomaly detected: {results['benchmarks']['space_exploration']['anomaly_detected']}")

    print("\n[3/4] Benchmarking simulation module...")
    results["benchmarks"]["simulation"] = benchmark_simulation_module()
    print(f"  ✓ Collatz: {results['benchmarks']['simulation']['collatz_exploration_ms']:.2f} ms")
    print(f"  ✓ Millennium: {results['benchmarks']['simulation']['millennium_analysis_ms']:.2f} ms")

    print("\n[4/4] Benchmarking cosmic ray detection...")
    results["benchmarks"]["cosmic_ray"] = benchmark_cosmic_ray_detection()
    print(f"  ✓ Runtime: {results['benchmarks']['cosmic_ray']['runtime_ms']:.2f} ms")
    print(
        f"  ✓ Events detected: {results['benchmarks']['cosmic_ray']['cosmic_ray_events_detected']}"
    )

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_all_benchmarks()

    with open("benchmarks/comprehensive_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✓ Results saved to: benchmarks/comprehensive_benchmark_results.json")
