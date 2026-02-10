"""
Mercury Agent ♱
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

"""
Test main engine functionality
"""

from omni_mercury_engine.engine import OmniMercuryEngine


def test_engine_initialization():
    """Test engine can be initialized"""
    engine = OmniMercuryEngine()
    assert engine is not None
    assert hasattr(engine, "fusion_model")


def test_detect_basic(sample_data):
    """Test basic anomaly detection"""
    engine = OmniMercuryEngine()
    result = engine.detect(sample_data)

    assert "detectors" in result
    assert "is_anomaly" in result
    assert isinstance(result["is_anomaly"], bool)


def test_detect_with_models(sample_data):
    """Test detection with specific detectors"""
    engine = OmniMercuryEngine()
    result = engine.detect(sample_data, detector_types=["statistical", "temporal"])

    assert "detectors" in result
    assert "is_anomaly" in result
    assert "statistical" in result["detectors"]
    assert "temporal" in result["detectors"]


def test_engine_save_load(tmp_path, sample_data):
    """Test saving and loading engine state"""
    engine = OmniMercuryEngine()
    engine.detect(sample_data)

    save_path = tmp_path / "engine_state.pt"
    engine.save_model(str(save_path))

    new_engine = OmniMercuryEngine()
    new_engine.load_model(str(save_path))

    result1 = engine.detect(sample_data)
    result2 = new_engine.detect(sample_data)

    assert result1["is_anomaly"] == result2["is_anomaly"]
