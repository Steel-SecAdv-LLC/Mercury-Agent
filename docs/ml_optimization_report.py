"""
Mercury Agent ♱
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
ML Architecture Optimization Report

Grid search and ablation studies for LSTM/CNN hyperparameters.
"""

import os
import sys

import numpy as np
import torch
from torch import nn


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from assets.loaders import generate_mimic_vitals

from omni_mercury_engine.medical.medical_cure_predictor import TemporalVitalSignsLSTM


def grid_search_lstm_hyperparameters() -> dict:
    """
    Grid search for optimal LSTM hyperparameters.

    Tests: hidden_dim=[64, 128, 256], num_layers=[1, 2, 3]
    """
    hidden_dims = [64, 128, 256]
    num_layers_options = [1, 2, 3]

    results = []

    print("Grid Search for LSTM Hyperparameters")
    print("=" * 60)

    for hidden_dim in hidden_dims:
        for num_layers in num_layers_options:
            print(f"\nTesting hidden_dim={hidden_dim}, num_layers={num_layers}")

            model = TemporalVitalSignsLSTM(
                input_dim=5, hidden_dim=hidden_dim, num_layers=num_layers
            )

            accuracies = []
            for _ in range(20):
                data = generate_mimic_vitals(num_timesteps=288, inject_disease=True)
                vitals = torch.tensor(data["vital_signs_sequence"]).unsqueeze(0)

                with torch.no_grad():
                    scores, _ = model(vitals)
                    pred = (scores.squeeze() > 0.5).float().numpy()
                    acc = np.mean(pred == data["true_labels"])
                    accuracies.append(acc)

            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)

            num_params = sum(p.numel() for p in model.parameters())

            results.append(
                {
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "accuracy": mean_acc,
                    "std": std_acc,
                    "num_params": num_params,
                }
            )

            print(f"  Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
            print(f"  Parameters: {num_params:,}")

    best = max(results, key=lambda x: x["accuracy"])

    print("\n" + "=" * 60)
    print("BEST CONFIGURATION:")
    print(f"  hidden_dim={best['hidden_dim']}, num_layers={best['num_layers']}")
    print(f"  Accuracy: {best['accuracy']:.3f} ± {best['std']:.3f}")
    print(f"  Parameters: {best['num_params']:,}")

    current = [r for r in results if r["hidden_dim"] == 64 and r["num_layers"] == 2][0]
    improvement = (best["accuracy"] - current["accuracy"]) / current["accuracy"] * 100

    print(f"\nImprovement over current (64, 2): {improvement:+.1f}%")

    return {"results": results, "best": best, "current": current, "improvement_pct": improvement}


def ablation_study_lstm() -> dict:
    """
    Ablation study: Remove components to assess importance.

    Tests: full model, no dropout, simpler architecture
    """
    print("\n\nAblation Study for LSTM")
    print("=" * 60)

    print("\n[1/3] Full Model (with dropout)")
    full_model = TemporalVitalSignsLSTM(hidden_dim=128, num_layers=2, dropout=0.2)
    full_acc = _test_model(full_model)
    print(f"  Accuracy: {full_acc:.3f}")

    print("\n[2/3] No Dropout")
    no_dropout = TemporalVitalSignsLSTM(hidden_dim=128, num_layers=2, dropout=0.0)
    no_dropout_acc = _test_model(no_dropout)
    print(f"  Accuracy: {no_dropout_acc:.3f}")
    print(f"  Impact: {(full_acc - no_dropout_acc)*100:.1f}% performance drop")

    print("\n[3/3] Single Layer (no stacking)")
    single_layer = TemporalVitalSignsLSTM(hidden_dim=128, num_layers=1, dropout=0.2)
    single_acc = _test_model(single_layer)
    print(f"  Accuracy: {single_acc:.3f}")
    print(f"  Impact: {(full_acc - single_acc)*100:.1f}% performance drop")

    return {
        "full_model": full_acc,
        "no_dropout": no_dropout_acc,
        "single_layer": single_acc,
        "dropout_importance": (full_acc - no_dropout_acc) / full_acc,
        "layer_stacking_importance": (full_acc - single_acc) / full_acc,
    }


def _test_model(model: nn.Module, num_samples: int = 30) -> float:
    """Helper to test a model on synthetic data."""
    accuracies = []
    for _ in range(num_samples):
        data = generate_mimic_vitals(num_timesteps=288, inject_disease=True)
        vitals = torch.tensor(data["vital_signs_sequence"]).unsqueeze(0)

        with torch.no_grad():
            scores, _ = model(vitals)
            pred = (scores.squeeze() > 0.5).float().numpy()
            acc = np.mean(pred == data["true_labels"])
            accuracies.append(acc)

    return np.mean(accuracies)


if __name__ == "__main__":
    print("ML ARCHITECTURE OPTIMIZATION REPORT")
    print("=" * 60)

    grid_results = grid_search_lstm_hyperparameters()
    ablation_results = ablation_study_lstm()

    import json

    os.makedirs("docs", exist_ok=True)
    with open("docs/ml_optimization_results.json", "w") as f:
        json.dump(
            {
                "grid_search": {
                    "results": grid_results["results"],
                    "best": grid_results["best"],
                    "current": grid_results["current"],
                    "improvement_pct": grid_results["improvement_pct"],
                },
                "ablation": ablation_results,
            },
            f,
            indent=2,
        )

    print("\n✓ Results saved to docs/ml_optimization_results.json")
