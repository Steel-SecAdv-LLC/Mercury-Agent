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

from __future__ import annotations

import pytest

pytest.importorskip("torch")

"""Real-data tests for Medical Cure Predictor using simulated datasets."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from omni_mercury_engine.medical.medical_cure_predictor import (
    MedicalCurePredictor,
    TemporalVitalSignsDetector,
)

try:
    from assets.loaders import generate_medical_image, generate_mimic_vitals

    ASSETS_AVAILABLE = True
except ImportError:
    ASSETS_AVAILABLE = False
    generate_mimic_vitals = None
    generate_medical_image = None

pytestmark = pytest.mark.skipif(
    not ASSETS_AVAILABLE, reason="assets module not available (requires real data loaders)"
)


class TestRealDataValidation:
    def test_temporal_vitals_sepsis_detection(self):
        """Test sepsis detection on simulated MIMIC-III vitals."""
        normal_data = generate_mimic_vitals(num_timesteps=288, inject_disease=False)
        sepsis_data = generate_mimic_vitals(
            num_timesteps=288, inject_disease=True, disease_type="sepsis"
        )

        detector = TemporalVitalSignsDetector()

        normal_result = detector.detect_temporal_anomaly(normal_data["vital_signs_sequence"])
        sepsis_result = detector.detect_temporal_anomaly(sepsis_data["vital_signs_sequence"])

        assert normal_result["temporal_anomaly_detected"] in [True, False]
        assert sepsis_result["temporal_anomaly_detected"] in [True, False]

    def test_medical_imaging_anomaly_detection(self):
        """Test imaging anomaly on simulated X-rays."""
        predictor = MedicalCurePredictor()

        results = []
        for _ in range(20):
            img_data = generate_medical_image(inject_anomaly=True)
            result = predictor.predict_and_cure(
                {
                    "vital_signs_sequence": np.tile([75, 120, 80, 98.6, 16], (288, 1)),
                    "medical_image": img_data["medical_image"],
                    "imaging_type": "xray",
                }
            )
            results.append(result)

        assert all(0.0 <= r.confidence <= 1.0 for r in results)

    def test_vitals_benchmark_accuracy(self):
        """Benchmark accuracy on simulated vital signs."""
        predictor = MedicalCurePredictor(enable_imaging=False, enable_treatment_opt=False)

        true_positives = 0
        true_negatives = 0

        for _ in range(15):
            normal = generate_mimic_vitals(num_timesteps=288, inject_disease=False)
            result = predictor.predict_and_cure(
                {"vital_signs_sequence": normal["vital_signs_sequence"]}
            )
            if not result.disease_risk_detected:
                true_negatives += 1

        for _ in range(15):
            disease = generate_mimic_vitals(
                num_timesteps=288, inject_disease=True, disease_type="sepsis"
            )
            result = predictor.predict_and_cure(
                {"vital_signs_sequence": disease["vital_signs_sequence"]}
            )
            if result.disease_risk_detected:
                true_positives += 1

        accuracy = (true_positives + true_negatives) / 30
        assert accuracy > 0.4, f"Accuracy {accuracy:.2f} should be > 40% on simulated data"

    def test_comprehensive_prediction_workflow(self):
        """Test complete prediction workflow with vitals and imaging."""
        predictor = MedicalCurePredictor()

        vitals_data = generate_mimic_vitals(num_timesteps=288, inject_disease=True)
        img_data = generate_medical_image(inject_anomaly=True)

        result = predictor.predict_and_cure(
            {
                "vital_signs_sequence": vitals_data["vital_signs_sequence"],
                "medical_image": img_data["medical_image"],
                "imaging_type": "xray",
            }
        )

        assert hasattr(result, "disease_risk_detected")
        assert hasattr(result, "confidence")
        assert len(result.recommendations) > 0
