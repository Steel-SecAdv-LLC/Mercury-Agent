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

from __future__ import annotations

"""Basic integration tests for omnidisciplinary modules"""

import numpy as np
import pytest


def test_medical_abms_import():
    """Test ABMS medical module imports correctly"""
    from omni_anomaly_engine.medical.abms_disciplines import ABMSDisciplineDetector

    detector = ABMSDisciplineDetector()
    assert detector is not None


def test_intelligence_fusion_import():
    """Test intelligence fusion module imports correctly"""
    from omni_anomaly_engine.security.intelligence_fusion import IntelligenceFusionEngine

    engine = IntelligenceFusionEngine()
    assert engine is not None


def test_schumann_resonance_import():
    """Test Schumann resonance module imports correctly"""
    from omni_anomaly_engine.space.schumann_resonance import SchumannResonanceDetector

    detector = SchumannResonanceDetector()
    assert detector is not None


def test_chemistry_import():
    """Test chemistry module imports correctly"""
    from omni_anomaly_engine.models.chemistry import ChemistryAnomalyDetector

    detector = ChemistryAnomalyDetector()
    assert detector is not None


def test_parapsychology_import():
    """Test parapsychology module imports correctly"""
    from omni_anomaly_engine.models.parapsychology import ParapsychologyDetector

    detector = ParapsychologyDetector()
    assert detector is not None


def test_medical_abms_basic_detection():
    """Test ABMS medical detection with simulated data"""
    from omni_anomaly_engine.medical.abms_disciplines import ABMSDisciplineDetector

    detector = ABMSDisciplineDetector()
    patient_data = {
        "vitals": {
            "heart_rate_bpm": 85,
            "blood_pressure_systolic": 130,
            "oxygen_saturation_pct": 95,
        },
        "labs": {},
        "symptoms": ["fatigue"],
        "history": {},
    }

    result = detector.detect_medical_anomaly(patient_data)
    assert result.primary_board is not None
    assert result.confidence >= 0.0


def test_schumann_resonance_detection():
    """Test Schumann resonance detection with synthetic signal"""
    from omni_anomaly_engine.space.schumann_resonance import SchumannResonanceDetector

    detector = SchumannResonanceDetector(sampling_rate=100.0)

    t = np.linspace(0, 10, 1000)
    elf_signal = np.sin(2 * np.pi * 7.83 * t) + 0.1 * np.random.randn(1000)

    result = detector.detect_resonance_anomaly(elf_signal)
    assert result.fundamental_freq > 0.0
    assert abs(result.fundamental_freq - 7.83) < 2.0


def test_engine_with_new_models():
    """Test that engine initializes with all new models"""
    from omni_anomaly_engine import OmniAnomalyEngine

    engine = OmniAnomalyEngine(mode="fusion")

    assert "medical_abms" in engine.models
    assert "intelligence_fusion" in engine.models
    assert "schumann_resonance" in engine.models
    assert "chemistry" in engine.models
    assert "parapsychology" in engine.models


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
