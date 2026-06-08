# Copyright (C) 2025 Steel Security Advisors LLC
"""Comprehensive tests for Medical Cure Predictor module."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import numpy as np

from omni_mercury_engine.medical.medical_cure_predictor import (
    MedicalCurePredictor,
    MedicalPredictionResult,
    TemporalVitalSignsDetector,
)


class TestTemporalVitalSigns:
    def test_normal_vitals(self) -> None:
        detector = TemporalVitalSignsDetector()
        vitals = np.tile([75, 120, 98, 98.6, 16], (100, 1))
        result = detector.detect_temporal_anomaly(vitals)
        assert "temporal_anomaly_detected" in result
        assert 0.0 <= result["anomaly_score"] <= 1.0


class TestMedicalCurePredictor:
    def test_comprehensive_prediction(self) -> None:
        predictor = MedicalCurePredictor()
        patient_data = {
            "vital_signs_sequence": np.tile([75, 120, 98, 98.6, 16], (100, 1)),
            "medical_image": np.random.randn(256, 256) * 50 + 128,
            "imaging_type": "xray",
        }
        result = predictor.predict_and_cure(patient_data)
        assert isinstance(result, MedicalPredictionResult)
        assert 0.0 <= result.confidence <= 1.0
