# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for omni_mercury_engine.security.intelligence_fusion module.

Tests all-source intelligence fusion, the multi-INT report modality contract,
and the fused-encoder network dimensions.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import numpy as np

from omni_mercury_engine.security.intelligence_fusion import (
    IntelligenceFusionEngine,
    IntelligenceFusionResult,
)


def _sample_reports() -> dict[str, dict[str, object]]:
    """Build a representative multi-INT report mapping."""
    return {
        "open_source": {
            "confidence": 0.72,
            "timeliness": 0.6,
            "relevance": 0.8,
            "completeness": 0.5,
            "indicators": ["recruitment", "finance"],
            "threat_score": 0.65,
        },
        "communications": {
            "confidence": 0.81,
            "timeliness": 0.9,
            "relevance": 0.85,
            "completeness": 0.7,
            "indicators": ["covert_comms"],
            "threat_score": 0.78,
            "encryption_detected": True,
        },
        "human": {
            "confidence": 0.88,
            "timeliness": 0.5,
            "relevance": 0.9,
            "completeness": 0.6,
            "indicators": ["insider_threat", "training"],
            "threat_score": 0.82,
        },
        "cyber": {
            "confidence": 0.79,
            "timeliness": 0.85,
            "relevance": 0.88,
            "completeness": 0.72,
            "indicators": ["reconnaissance", "c2"],
            "threat_score": 0.9,
            "encryption_algorithm": "AES-256",
        },
    }


class TestModalityContract:
    """Tests for the multi-INT report modality contract."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.engine = IntelligenceFusionEngine()

    def test_off_modality_array_extract_features_rejected(self) -> None:
        """A generic float array must yield a clean, actionable ValueError."""
        off_modality = np.random.rand(200, 8).astype(np.float32)

        with pytest.raises(ValueError, match="multi-INT report mapping"):
            self.engine.extract_features(off_modality)  # type: ignore[arg-type]

    def test_off_modality_array_predict_rejected(self) -> None:
        """The predict path must reject a generic float array cleanly."""
        off_modality = np.random.rand(200, 8).astype(np.float32)

        with pytest.raises(ValueError, match=r"shape \(200, 8\)"):
            self.engine.predict(off_modality)  # type: ignore[arg-type]


class TestOnModalityFusion:
    """Tests driving the fusion network on correct-modality intel reports."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.engine = IntelligenceFusionEngine()

    def test_extract_features_shape(self) -> None:
        """Feature extraction returns a concatenated per-discipline tensor."""
        features = self.engine.extract_features(_sample_reports())
        assert features.shape[0] == 1
        assert features.shape[1] == 128 // 13 * 13

    def test_fuse_intelligence_end_to_end(self) -> None:
        """Multi-INT reports flow through the full fusion network."""
        result = self.engine.fuse_intelligence(_sample_reports())

        assert isinstance(result, IntelligenceFusionResult)
        assert result.threat_level in {
            "LOW",
            "MODERATE",
            "SUBSTANTIAL",
            "SEVERE",
            "CRITICAL",
        }
        assert 0.0 <= result.confidence <= 1.0
        assert result.primary_intel_sources
        assert result.threat_indicators

    def test_predict_returns_scores(self) -> None:
        """The predict adapter surfaces anomaly scores and threat level."""
        prediction = self.engine.predict(_sample_reports())

        assert prediction["anomaly_scores"].shape == (1,)
        assert prediction["threat_level"]
        assert 0.0 <= prediction["confidence"] <= 1.0

    def test_fuse_intelligence_with_temporal_context(self) -> None:
        """The optional temporal LSTM path fuses without dimension errors."""
        temporal = [
            {"threat_level": 2, "confidence": 0.5, "num_sources": 2},
            {"threat_level": 3, "confidence": 0.6, "num_sources": 3},
            {"threat_level": 4, "confidence": 0.75, "num_sources": 5},
        ]

        result = self.engine.fuse_intelligence(_sample_reports(), temporal_context=temporal)

        assert isinstance(result, IntelligenceFusionResult)
        assert result.temporal_patterns["trend"] == "escalating"
