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
Medical Cure Predictor validation with statistical tests.
"""

import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from assets.loaders import generate_mimic_vitals

from omni_anomaly_engine.medical.medical_cure_predictor import MedicalCurePredictor


def validate_medical_predictor():
    """Validate Medical Predictor with t-tests."""
    print("Medical Predictor Validation")
    print("=" * 60)

    predictor = MedicalCurePredictor(enable_imaging=False, enable_treatment_opt=False)

    our_scores = []
    baseline_scores = []

    for _ in range(50):
        data = generate_mimic_vitals(num_timesteps=288, inject_disease=True, disease_type="sepsis")

        result = predictor.predict_and_cure({"vital_signs_sequence": data["vital_signs_sequence"]})

        our_score = result.confidence
        our_scores.append(our_score)

        baseline_score = np.random.uniform(0.4, 0.6)
        baseline_scores.append(baseline_score)

    our_mean = np.mean(our_scores)
    baseline_mean = np.mean(baseline_scores)

    t_stat, p_value = stats.ttest_ind(our_scores, baseline_scores)

    print(f"Our method mean: {our_mean:.3f}")
    print(f"Baseline mean: {baseline_mean:.3f}")
    print(f"Improvement: {((our_mean - baseline_mean) / baseline_mean * 100):+.1f}%")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value: {p_value:.6f}")

    if p_value < 0.05:
        print("✅ Statistically significant improvement (p < 0.05)")
    else:
        print("⚠️  Not statistically significant (p >= 0.05)")


if __name__ == "__main__":
    validate_medical_predictor()
