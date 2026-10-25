# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Comprehensive tests for Medical Cure Predictor module."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import numpy as np
import torch

from omni_mercury_engine.medical.medical_cure_predictor import (
    MedicalCurePredictor,
    MedicalImagingAnomalyDetector,
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


NORMAL_VITALS = np.tile([75.0, 120.0, 98.0, 98.6, 16.0], (24, 1))


class TestUntrainedNetworksAbstain:
    """Untrained networks once decided 'anomalous' or 'normal' from their initial weights."""

    def test_temporal_result_does_not_depend_on_the_network_initialisation(self) -> None:
        results = []
        for seed in range(5):
            torch.manual_seed(seed)
            out = TemporalVitalSignsDetector().detect_temporal_anomaly(NORMAL_VITALS)
            assert out["ml_available"] is False
            results.append((out["temporal_anomaly_detected"], out["anomaly_score"]))
            assert out["attention_weights"] == []
            assert out["critical_time_indices"] == []
        assert set(results) == {(False, 0.0)}

    def test_the_rule_based_trend_risk_still_fires_without_the_network(self) -> None:
        rising_heart_rate = NORMAL_VITALS.copy()
        rising_heart_rate[-3:, 0] = [75.0, 90.0, 110.0]
        out = TemporalVitalSignsDetector().detect_temporal_anomaly(rising_heart_rate)
        assert out["disease_risk"]["cardiac"] > 0.0

    def test_temporal_loaded_weights_are_scored(self) -> None:
        detector = TemporalVitalSignsDetector()
        detector.lstm_model.load_trained_weights(detector.lstm_model.state_dict())

        out = detector.detect_temporal_anomaly(NORMAL_VITALS)

        assert out["ml_available"] is True
        normalized = detector._normalize_vitals(NORMAL_VITALS)
        with torch.no_grad():
            expected, _ = detector.lstm_model.eval()(
                torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)
            )
        assert out["anomaly_score"] == pytest.approx(float(expected.item()))
        assert len(out["attention_weights"]) == len(NORMAL_VITALS)

    def test_a_blank_image_is_not_called_anomalous_by_random_weights(self) -> None:
        blank = np.zeros((224, 224))
        outcomes = set()
        for seed in range(3):
            torch.manual_seed(seed)
            out = MedicalImagingAnomalyDetector().detect_imaging_anomaly(blank)
            assert out["ml_available"] is False
            outcomes.add(
                (out["imaging_anomaly_detected"], out["anomaly_score"], tuple(out["findings"]))
            )
        assert outcomes == {(False, 0.0, ())}

    def test_imaging_loaded_weights_are_scored(self) -> None:
        detector = MedicalImagingAnomalyDetector()
        detector.load_trained_weights(detector.model.state_dict())
        assert detector.is_fitted is True

        out = detector.detect_imaging_anomaly(np.zeros((224, 224)))

        assert out["ml_available"] is True
        with torch.no_grad():
            expected = detector.model.eval()(
                torch.tensor(detector._preprocess_image(np.zeros((224, 224))), dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
            )
        assert out["anomaly_score"] == pytest.approx(float(expected.item()))

    def test_the_orchestrator_reports_which_networks_scored(self) -> None:
        result = MedicalCurePredictor().predict_and_cure(
            {"vital_signs_sequence": NORMAL_VITALS, "medical_image": np.zeros((224, 224))}
        )
        assert result.ml_temporal_available is False
        assert result.ml_imaging_available is False
        assert result.imaging_anomaly is False
        assert result.vital_signs_anomaly is False
        assert result.confidence == 0.0
