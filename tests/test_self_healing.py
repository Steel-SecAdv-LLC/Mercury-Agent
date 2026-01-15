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

from __future__ import annotations


"""Tests for CRISPR-inspired self-healing module"""

import numpy as np

from omni_mercury_engine.core.self_healing import AnomalySignature, CRISPRInspiredSelfHealing


def test_self_healing_initialization():
    """Test self-healing system initialization"""
    system = CRISPRInspiredSelfHealing(max_signatures=100, similarity_threshold=0.85)
    assert system.max_signatures == 100
    assert system.similarity_threshold == 0.85
    assert len(system.signature_library) == 0


def test_stage_1_acquisition():
    """Test Stage 1: Acquisition of anomaly signature"""
    system = CRISPRInspiredSelfHealing()
    anomaly_data = np.random.randn(10, 5)

    signature = system.stage_1_acquisition(anomaly_data)

    assert isinstance(signature, AnomalySignature)
    assert signature.signature_id in system.signature_library
    assert len(system.acquisition_history) == 1
    assert signature.detection_count == 1


def test_stage_2_expression():
    """Test Stage 2: Expression of signature into detection pattern"""
    system = CRISPRInspiredSelfHealing()
    anomaly_data = np.random.randn(10, 5)

    signature = system.stage_1_acquisition(anomaly_data)
    detection_pattern = system.stage_2_expression(signature)

    assert isinstance(detection_pattern, np.ndarray)
    assert len(detection_pattern) == len(signature.feature_vector)
    assert np.abs(np.linalg.norm(detection_pattern) - 1.0) < 0.01


def test_stage_3_interference():
    """Test Stage 3: Interference - detecting known anomalies"""
    system = CRISPRInspiredSelfHealing(similarity_threshold=0.7)

    anomaly_data = np.random.randn(10, 5)
    system.stage_1_acquisition(anomaly_data)

    similar_data = anomaly_data + np.random.randn(10, 5) * 0.1
    is_anomaly, confidence, sig_id = system.stage_3_interference(similar_data)

    assert isinstance(is_anomaly, bool)
    assert 0.0 <= confidence <= 1.0
    assert sig_id is None or isinstance(sig_id, str)


def test_heritable_immunity():
    """Test heritable immunity via save/load"""
    import tempfile
    from pathlib import Path

    system = CRISPRInspiredSelfHealing()

    for i in range(3):
        anomaly_data = np.random.randn(10, 5) * (i + 1)
        system.stage_1_acquisition(anomaly_data)

    assert len(system.signature_library) == 3

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        temp_path = f.name

    try:
        system.save_signature_library(temp_path)

        new_system = CRISPRInspiredSelfHealing()
        new_system.load_signature_library(temp_path)

        assert len(new_system.signature_library) == 3
        assert len(new_system.acquisition_history) == 3
    finally:
        Path(temp_path).unlink()


def test_max_signatures_pruning():
    """Test that old signatures are pruned when max is reached"""
    system = CRISPRInspiredSelfHealing(max_signatures=5)

    for i in range(10):
        anomaly_data = np.random.randn(10, 5) * (i + 1)
        system.stage_1_acquisition(anomaly_data)

    assert len(system.signature_library) <= 5
