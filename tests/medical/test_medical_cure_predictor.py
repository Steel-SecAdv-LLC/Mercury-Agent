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

"""Comprehensive tests for Medical Cure Predictor module."""

import numpy as np
from omni_anomaly_engine.medical.medical_cure_predictor import (
    MedicalCurePredictor,
    TemporalVitalSignsDetector,
    MedicalPredictionResult,
)


class TestTemporalVitalSigns:
    def test_normal_vitals(self):
        detector = TemporalVitalSignsDetector()
        vitals = np.tile([75, 120, 98, 98.6, 16], (100, 1))
        result = detector.detect_temporal_anomaly(vitals)
        assert "temporal_anomaly_detected" in result
        assert 0.0 <= result["anomaly_score"] <= 1.0


class TestMedicalCurePredictor:
    def test_comprehensive_prediction(self):
        predictor = MedicalCurePredictor()
        patient_data = {
            "vital_signs_sequence": np.tile([75, 120, 98, 98.6, 16], (100, 1)),
            "medical_image": np.random.randn(256, 256) * 50 + 128,
            "imaging_type": "xray",
        }
        result = predictor.predict_and_cure(patient_data)
        assert isinstance(result, MedicalPredictionResult)
        assert 0.0 <= result.confidence <= 1.0
