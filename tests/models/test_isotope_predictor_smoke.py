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


def test_isotope_predictor_importable() -> None:
    from omni_mercury_engine.models.isotope_predictor import IsotopePredictor

    assert IsotopePredictor is not None


def test_nuclear_forensics_analyzer_importable() -> None:
    from omni_mercury_engine.models.isotope_predictor import NuclearForensicsAnalyzer

    assert NuclearForensicsAnalyzer is not None


def test_radiological_threat_assessor_importable() -> None:
    from omni_mercury_engine.models.isotope_predictor import RadiologicalThreatAssessor

    assert RadiologicalThreatAssessor is not None
