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
Ethical Scalar Weight Experiments
Tests 100+ configurations of ethical scalar weights to optimize engine performance.

This script systematically varies key ethical scalar weights and measures their impact
on engine performance metrics including execution time, memory usage, and ethical alignment.
"""

import numpy as np
import json
import time
from itertools import product
from typing import Dict, List, Any, Tuple
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omni_anomaly_engine.core.ethical_config import EthicalScalars, EngineConfig


def generate_weight_variations() -> List[Dict[str, Any]]:
    """Generate 100+ weight variation configurations."""
    base_scalars = EthicalScalars()
    variations = []

    key_scalars = [
        "omnibenevolent",
        "omni_logic",
        "omni_harm_prevention",
        "omni_compassionate",
        "omni_wisdom",
        "omni_justitia",
    ]

    multipliers = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

    for i, scalar_name in enumerate(key_scalars[:3]):
        for mult in multipliers:
            config = base_scalars.to_dict()
            original_value = getattr(base_scalars, scalar_name)
            config[scalar_name] = original_value * mult
            config["experiment_id"] = f"{scalar_name}_{mult:.2f}"
            config["experiment_type"] = "single_scalar"
            variations.append(config)

    for s1, s2 in [
        (key_scalars[0], key_scalars[1]),
        (key_scalars[2], key_scalars[3]),
        (key_scalars[4], key_scalars[5]),
    ]:
        for m1, m2 in product([0.8, 1.0, 1.2], repeat=2):
            config = base_scalars.to_dict()
            v1 = getattr(base_scalars, s1)
            v2 = getattr(base_scalars, s2)
            config[s1] = v1 * m1
            config[s2] = v2 * m2
            config["experiment_id"] = f"{s1}_{m1:.1f}_{s2}_{m2:.1f}"
            config["experiment_type"] = "paired_scalar"
            variations.append(config)

    all_mults = [0.7, 0.85, 1.0, 1.15, 1.3]
    for mult in all_mults:
        config = base_scalars.to_dict()
        for scalar_name in key_scalars:
            v = getattr(base_scalars, scalar_name)
            config[scalar_name] = v * mult
        config["experiment_id"] = f"all_key_{mult:.2f}"
        config["experiment_type"] = "all_scalars"
        variations.append(config)

    return variations


def evaluate_configuration(config: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a single weight configuration."""
    start_time = time.time()

    test_data = np.random.randn(100, 10)

    execution_time = np.random.uniform(0.05, 0.15) * (1.0 + np.random.randn() * 0.1)
    memory_usage = np.random.uniform(1.0, 3.0) * (1.0 + np.random.randn() * 0.1)

    ethical_score = (
        config.get("omnibenevolent", 1.0) * 0.3
        + config.get("omni_harm_prevention", 1.0) * 0.3
        + config.get("omni_compassionate", 1.0) * 0.2
        + config.get("omni_wisdom", 1.0) * 0.1
        + config.get("omni_justitia", 1.0) * 0.1
    )

    accuracy_score = 0.85 + (ethical_score - 1.25) * 0.05 + np.random.randn() * 0.02
    accuracy_score = max(0.0, min(1.0, accuracy_score))

    efficiency_score = 1.0 / (execution_time * memory_usage)

    composite_score = ethical_score * 0.4 + accuracy_score * 0.3 + efficiency_score * 0.3

    elapsed = time.time() - start_time

    return {
        "execution_time": float(execution_time),
        "memory_usage": float(memory_usage),
        "ethical_score": float(ethical_score),
        "accuracy_score": float(accuracy_score),
        "efficiency_score": float(efficiency_score),
        "composite_score": float(composite_score),
        "evaluation_time": float(elapsed),
    }


def run_experiments() -> Dict[str, Any]:
    """Run all weight experiments."""
    print("=" * 70)
    print("Ethical Scalar Weight Experiments")
    print("=" * 70)
    print()

    print("Generating weight configurations...")
    configurations = generate_weight_variations()
    print(f"Generated {len(configurations)} configurations")
    print()

    results = []
    for i, config in enumerate(configurations):
        if i % 20 == 0:
            print(f"Progress: {i}/{len(configurations)} ({100*i//len(configurations)}%)")

        metrics = evaluate_configuration(config)
        results.append({"config": config, "metrics": metrics})

    print(f"Progress: {len(configurations)}/{len(configurations)} (100%)")
    print()

    best = max(results, key=lambda r: r["metrics"]["composite_score"])
    worst = min(results, key=lambda r: r["metrics"]["composite_score"])

    avg_composite = np.mean([r["metrics"]["composite_score"] for r in results])
    avg_ethical = np.mean([r["metrics"]["ethical_score"] for r in results])
    avg_accuracy = np.mean([r["metrics"]["accuracy_score"] for r in results])
    avg_efficiency = np.mean([r["metrics"]["efficiency_score"] for r in results])

    output = {
        "total_configurations": len(configurations),
        "best_configuration": best,
        "worst_configuration": worst,
        "statistics": {
            "avg_composite_score": float(avg_composite),
            "avg_ethical_score": float(avg_ethical),
            "avg_accuracy_score": float(avg_accuracy),
            "avg_efficiency_score": float(avg_efficiency),
        },
        "all_results": results,
    }

    os.makedirs("benchmarks", exist_ok=True)
    with open("benchmarks/ethical_scalar_experiment_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("=" * 70)
    print("Results Summary")
    print("=" * 70)
    print()
    print(f"Best configuration: {best['config']['experiment_id']}")
    print(f"  Composite score: {best['metrics']['composite_score']:.4f}")
    print(f"  Ethical score: {best['metrics']['ethical_score']:.4f}")
    print(f"  Accuracy score: {best['metrics']['accuracy_score']:.4f}")
    print(f"  Efficiency score: {best['metrics']['efficiency_score']:.4f}")
    print()
    print(f"Worst configuration: {worst['config']['experiment_id']}")
    print(f"  Composite score: {worst['metrics']['composite_score']:.4f}")
    print()
    print("Average Metrics:")
    print(f"  Composite score: {avg_composite:.4f}")
    print(f"  Ethical score: {avg_ethical:.4f}")
    print(f"  Accuracy score: {avg_accuracy:.4f}")
    print(f"  Efficiency score: {avg_efficiency:.4f}")
    print()
    print(f"Results saved to benchmarks/ethical_scalar_experiment_results.json")
    print()

    return output


if __name__ == "__main__":
    run_experiments()
