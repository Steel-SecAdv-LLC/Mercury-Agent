# Copyright (C) 2025 Steel Security Advisors LLC
"""(at your option) any later version."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")


def test_isotope_predictor_importable() -> None:
    from omni_mercury_engine.models.isotope_predictor import IsotopePredictor

    assert IsotopePredictor is not None


def test_nuclear_forensics_analyzer_importable() -> None:
    from omni_mercury_engine.models.isotope_predictor import NuclearForensicsAnalyzer

    assert NuclearForensicsAnalyzer is not None


def test_radiological_threat_assessor_importable() -> None:
    from omni_mercury_engine.models.isotope_predictor import RadiologicalThreatAssessor

    assert RadiologicalThreatAssessor is not None
