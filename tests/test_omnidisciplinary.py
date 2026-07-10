# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Basic integration tests for omnidisciplinary modules."""

from __future__ import annotations

import numpy as np
import pytest

# Every detector exercised in this module (ABMS medical, intelligence
# fusion, Schumann resonance, chemistry, parapsychology) lives on the
# torch backbone — the module-level imports inside each test pull in
# torch transitively.  Skip cleanly at collection time when torch is
# absent so the rest of the suite stays discoverable.
pytest.importorskip("torch")


def test_medical_abms_import() -> None:
    """Test ABMS medical module imports correctly"""
    from omni_mercury_engine.medical.abms_disciplines import ABMSDisciplineDetector

    detector = ABMSDisciplineDetector()
    assert detector is not None


def test_intelligence_fusion_import() -> None:
    """Test intelligence fusion module imports correctly"""
    from omni_mercury_engine.security.intelligence_fusion import IntelligenceFusionEngine

    engine = IntelligenceFusionEngine()
    assert engine is not None


def test_schumann_resonance_import() -> None:
    """Test Schumann resonance module imports correctly"""
    from omni_mercury_engine.space.schumann_resonance import SchumannResonanceDetector

    detector = SchumannResonanceDetector()
    assert detector is not None


def test_chemistry_import() -> None:
    """Test chemistry module imports correctly"""
    from omni_mercury_engine.models.chemistry import ChemistryAnomalyDetector

    detector = ChemistryAnomalyDetector()
    assert detector is not None


def test_chemistry_classical_metals_gold_entry() -> None:
    """Every classical-metal KB entry must carry the 'property' key.

    Regression: the gold entry used to spell its key "perfection", so any
    composition containing Au raised KeyError inside
    ``_correlate_classical_metals``.
    """
    from omni_mercury_engine.models.chemistry import ChemistryAnomalyDetector

    detector = ChemistryAnomalyDetector()
    for name, info in detector.classical_metals_kb["classical_metals"].items():
        assert "property" in info, f"{name} entry missing 'property'"
    correlation = detector._correlate_classical_metals(
        {"elemental_composition": {"Au": 0.5, "Fe": 0.3, "Pb": 0.2}}, []
    )
    metals = {entry["metal"] for entry in correlation["classical_metals_present"]}
    assert {"gold", "iron", "lead"} <= metals


def test_parapsychology_import() -> None:
    """Test parapsychology module imports correctly"""
    from omni_mercury_engine.models.parapsychology import ParapsychologyDetector

    detector = ParapsychologyDetector()
    assert detector is not None


def test_medical_abms_basic_detection() -> None:
    """Test ABMS medical detection with simulated data"""
    from omni_mercury_engine.medical.abms_disciplines import ABMSDisciplineDetector

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


def test_schumann_resonance_detection() -> None:
    """Test Schumann resonance detection with synthetic signal"""
    from omni_mercury_engine.space.schumann_resonance import SchumannResonanceDetector

    detector = SchumannResonanceDetector(sampling_rate=100.0)

    t = np.linspace(0, 10, 1000)
    elf_signal = np.sin(2 * np.pi * 7.83 * t) + 0.1 * np.random.randn(1000)

    result = detector.detect_resonance_anomaly(elf_signal)
    assert result.fundamental_freq > 0.0
    assert abs(result.fundamental_freq - 7.83) < 2.0


def test_engine_with_new_models() -> None:
    """Test that engine initializes with all new models"""
    from omni_mercury_engine import OmniMercuryEngine

    engine = OmniMercuryEngine(mode="fusion")

    assert "medical_abms" in engine.models
    assert "intelligence_fusion" in engine.models
    assert "schumann_resonance" in engine.models
    assert "chemistry" in engine.models
    assert "parapsychology" in engine.models


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
