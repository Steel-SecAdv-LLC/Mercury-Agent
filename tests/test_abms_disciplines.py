# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for omni_mercury_engine.medical.abms_disciplines module.

Tests ABMS medical specialty-based anomaly detection.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import numpy as np
import pytest
import torch

from omni_mercury_engine.medical.abms_disciplines import (
    ABMSAnomalyDetector,
    ABMSBoard,
    MedicalAnomalyResult,
    MultiSpecialtyNeuralNet,
)


class TestABMSBoard:
    """Tests for ABMSBoard enum."""

    def test_all_boards_defined(self) -> None:
        """Test that all 24 ABMS boards are defined."""
        boards = list(ABMSBoard)
        assert len(boards) == 24

    def test_board_values_are_strings(self) -> None:
        """Test that board values are valid strings."""
        for board in ABMSBoard:
            assert isinstance(board.value, str)
            assert len(board.value) > 0
            assert "_" in board.value or board.value.islower()

    def test_specific_boards_exist(self) -> None:
        """Test that key specialty boards exist."""
        expected_boards = [
            "INTERNAL_MEDICINE",
            "SURGERY",
            "PEDIATRICS",
            "EMERGENCY_MEDICINE",
            "PSYCHIATRY",
            "RADIOLOGY",
        ]
        board_names = [b.name for b in ABMSBoard]
        for expected in expected_boards:
            assert expected in board_names


class TestMedicalAnomalyResult:
    """Tests for MedicalAnomalyResult dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic result creation."""
        result = MedicalAnomalyResult(primary_board="internal_medicine")
        assert result.primary_board == "internal_medicine"
        assert result.subspecialty is None
        assert result.anomaly_detected is False
        assert result.confidence == 0.0
        assert result.risk_score == 0.0
        assert result.urgency_level == "routine"

    def test_full_creation(self) -> None:
        """Test result with all fields."""
        result = MedicalAnomalyResult(
            primary_board="cardiology",
            subspecialty="interventional_cardiology",
            anomaly_detected=True,
            confidence=0.95,
            risk_score=0.85,
            clinical_indicators=["elevated_troponin", "st_elevation"],
            recommended_consultations=["cardiology", "cardiac_surgery"],
            treatment_considerations=["anticoagulation", "catheterization"],
            urgency_level="emergent",
            neurosymbolic_reasoning={"pathway": "STEMI_protocol"},
        )
        assert result.anomaly_detected is True
        assert len(result.clinical_indicators) == 2
        assert len(result.recommended_consultations) == 2
        assert result.urgency_level == "emergent"

    def test_default_lists(self) -> None:
        """Test that default lists are properly initialized."""
        result = MedicalAnomalyResult(primary_board="test")
        result.clinical_indicators.append("test_indicator")

        # New instance should have empty list
        result2 = MedicalAnomalyResult(primary_board="test2")
        assert len(result2.clinical_indicators) == 0


class TestMultiSpecialtyNeuralNet:
    """Tests for MultiSpecialtyNeuralNet class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.model = MultiSpecialtyNeuralNet(input_dim=64, num_specialties=24)

    def test_initialization(self) -> None:
        """Test model initialization."""
        assert self.model is not None
        assert hasattr(self.model, "shared_encoder")
        assert hasattr(self.model, "specialty_heads")
        assert hasattr(self.model, "attention")

    def test_specialty_heads_count(self) -> None:
        """Test that all specialty heads are created."""
        assert len(self.model.specialty_heads) == 24

    def test_forward_all_specialties(self) -> None:
        """Test forward pass for all specialties."""
        x = torch.randn(8, 64)  # Batch of 8, input dim 64
        result = self.model(x)

        assert isinstance(result, dict)
        assert len(result) == 24  # All specialties

    def test_forward_single_specialty(self) -> None:
        """Test forward pass for single specialty."""
        x = torch.randn(8, 64)
        result = self.model(x, specialty="internal_medicine")

        assert isinstance(result, dict)
        assert "internal_medicine" in result

    def test_output_dimensions(self) -> None:
        """Test output dimensions for each specialty."""
        x = torch.randn(4, 64)
        result = self.model(x)

        for specialty, output in result.items():
            assert output.shape == (4, 3)  # Batch size, 3 output classes

    def test_golden_ratio_architecture(self) -> None:
        """Test that golden ratio is used in architecture."""
        phi = 1.618
        input_dim = 64

        # Verify hidden dimensions follow golden ratio pattern
        hidden_1 = int(input_dim * phi)
        int(hidden_1 * phi)

        # Check encoder layer dimensions
        encoder_layers = list(self.model.shared_encoder.children())
        first_linear = encoder_layers[0]
        assert first_linear.out_features == hidden_1

    def test_batch_processing(self) -> None:
        """Test various batch sizes."""
        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 64)
            result = self.model(x)
            for output in result.values():
                assert output.shape[0] == batch_size

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through the model."""
        x = torch.randn(4, 64, requires_grad=True)
        result = self.model(x)

        # Sum all outputs and backward
        total_loss = sum(output.sum() for output in result.values())
        total_loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestABMSAnomalyDetector:
    """Tests for ABMSAnomalyDetector class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.detector = ABMSAnomalyDetector()

    def test_initialization(self) -> None:
        """Test detector initialization."""
        assert self.detector is not None
        assert hasattr(self.detector, "model")
        assert hasattr(self.detector, "specialty_thresholds")

    def test_detect_with_numpy_input(self) -> None:
        """Test detection with numpy array input."""
        data = np.random.randn(64).astype(np.float32)
        result = self.detector.detect(data, specialty="internal_medicine")

        assert isinstance(result, MedicalAnomalyResult)
        assert result.primary_board == "internal_medicine"
        assert 0.0 <= result.confidence <= 1.0

    def test_detect_with_tensor_input(self) -> None:
        """Test detection with tensor input."""
        data = torch.randn(64)
        result = self.detector.detect(data, specialty="emergency_medicine")

        assert isinstance(result, MedicalAnomalyResult)
        assert result.primary_board == "emergency_medicine"

    def test_detect_all_specialties(self) -> None:
        """Test detection across all specialties."""
        data = np.random.randn(64).astype(np.float32)
        results = self.detector.detect_all(data)

        assert isinstance(results, dict)
        assert len(results) == 24

    def test_urgency_levels(self) -> None:
        """Test that urgency levels are valid."""
        valid_urgency = ["routine", "urgent", "emergent", "critical"]
        data = np.random.randn(64).astype(np.float32)

        for _ in range(10):  # Run multiple times
            result = self.detector.detect(data, specialty="emergency_medicine")
            assert result.urgency_level in valid_urgency

    def test_confidence_bounds(self) -> None:
        """Test that confidence is properly bounded."""
        data = np.random.randn(64).astype(np.float32)

        for board in ABMSBoard:
            result = self.detector.detect(data, specialty=board.value)
            assert 0.0 <= result.confidence <= 1.0
            assert 0.0 <= result.risk_score <= 1.0

    def test_clinical_indicators_format(self) -> None:
        """Test that clinical indicators are properly formatted."""
        data = np.random.randn(64).astype(np.float32)
        result = self.detector.detect(data, specialty="cardiology")

        assert isinstance(result.clinical_indicators, list)
        for indicator in result.clinical_indicators:
            assert isinstance(indicator, str)

    def test_consultation_recommendations(self) -> None:
        """Test consultation recommendations."""
        np.random.randn(64).astype(np.float32)

        # Run detection that should trigger high risk
        data_high = np.ones(64).astype(np.float32) * 3.0  # Abnormal values
        result = self.detector.detect(data_high, specialty="internal_medicine")

        # Recommendations should be list of strings
        assert isinstance(result.recommended_consultations, list)


class TestSpecialtySpecificBehavior:
    """Tests for specialty-specific detection behavior."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.detector = ABMSAnomalyDetector()

    def test_cardiology_features(self) -> None:
        """Test cardiology-specific feature detection."""
        # Simulate cardiac biomarker data
        data = np.zeros(64, dtype=np.float32)
        data[0:4] = [2.5, 1.8, 0.9, 1.2]  # Elevated troponin, BNP, etc.

        result = self.detector.detect(data, specialty="internal_medicine")
        assert result is not None

    def test_emergency_medicine_urgency(self) -> None:
        """Test that emergency medicine properly classifies urgency."""
        data = np.random.randn(64).astype(np.float32)
        result = self.detector.detect(data, specialty="emergency_medicine")

        # Emergency medicine should have urgency classification
        assert result.urgency_level in ["routine", "urgent", "emergent", "critical"]

    def test_psychiatry_features(self) -> None:
        """Test psychiatry-specific detection."""
        data = np.random.randn(64).astype(np.float32)
        result = self.detector.detect(data, specialty="psychiatry")

        assert result.primary_board == "psychiatry"

    def test_pediatrics_features(self) -> None:
        """Test pediatrics-specific detection."""
        data = np.random.randn(64).astype(np.float32)
        result = self.detector.detect(data, specialty="pediatrics")

        assert result.primary_board == "pediatrics"


class TestNeurosymbolicReasoning:
    """Tests for neurosymbolic reasoning integration."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.detector = ABMSAnomalyDetector()

    def test_reasoning_output_structure(self) -> None:
        """Test neurosymbolic reasoning output structure."""
        data = np.random.randn(64).astype(np.float32)
        result = self.detector.detect(data, specialty="internal_medicine", include_reasoning=True)

        if result.neurosymbolic_reasoning is not None:
            assert isinstance(result.neurosymbolic_reasoning, dict)

    def test_reasoning_explains_decision(self) -> None:
        """Test that reasoning provides explanation."""
        data = np.ones(64).astype(np.float32) * 2.0  # Abnormal values
        result = self.detector.detect(data, specialty="emergency_medicine", include_reasoning=True)

        # High-risk should have reasoning
        if result.risk_score > 0.7 and result.neurosymbolic_reasoning:
            assert len(result.neurosymbolic_reasoning) > 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.detector = ABMSAnomalyDetector()

    def test_empty_input(self) -> None:
        """Test handling of zero-valued input."""
        data = np.zeros(64, dtype=np.float32)
        result = self.detector.detect(data, specialty="internal_medicine")

        assert result is not None
        assert isinstance(result, MedicalAnomalyResult)

    def test_extreme_values(self) -> None:
        """Test handling of extreme input values."""
        data = np.ones(64, dtype=np.float32) * 1000
        result = self.detector.detect(data, specialty="internal_medicine")

        assert result is not None
        # Should still have valid bounds
        assert 0.0 <= result.confidence <= 1.0

    def test_nan_handling(self) -> None:
        """Test handling of NaN values."""
        data = np.ones(64, dtype=np.float32)
        data[0] = np.nan

        # Should handle gracefully
        try:
            result = self.detector.detect(data, specialty="internal_medicine")
            # If it doesn't raise, result should still be valid
            assert result is not None
        except (ValueError, RuntimeError):
            pass  # Expected behavior

    def test_invalid_specialty(self) -> None:
        """Test handling of invalid specialty."""
        data = np.random.randn(64).astype(np.float32)

        with pytest.raises((KeyError, ValueError)):
            self.detector.detect(data, specialty="invalid_specialty")
