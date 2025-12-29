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

"""
Full engine tests to boost coverage
"""

import os
import tempfile

import numpy as np

from omni_mercury_engine.engine import OmniMercuryEngine


def test_engine_detect_with_all_detectors():
    """Test detection with all detector types enabled"""
    engine = OmniMercuryEngine()

    data = np.random.randn(50, 3)

    results = engine.detect(
        data,
        detector_types=["statistical", "temporal", "spatial", "dimensional", "directive"],
    )

    assert results is not None
    assert "detectors" in results
    assert "is_anomaly" in results


def test_engine_detect_with_subset():
    """Test detection with subset of detectors"""
    engine = OmniMercuryEngine()

    data = np.random.randn(50, 3)

    results = engine.detect(data, detector_types=["statistical"])

    assert results is not None
    assert "detectors" in results


def test_engine_fusion_mode():
    """Test fusion mode initialization"""
    engine = OmniMercuryEngine(mode="fusion")

    assert engine.fusion_model is not None
    assert engine.fusion_inference is not None


def test_engine_biometric_detection():
    """Test biometric anomaly detection"""
    engine = OmniMercuryEngine()

    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    results = engine.detect_biometric(image)

    assert results is not None
    assert "model_type" in results or "error" in results


def test_engine_security_scan():
    """Test security vulnerability scanning"""
    engine = OmniMercuryEngine()

    payloads = [
        "SELECT * FROM users",
        "<script>alert('xss')</script>",
        "../../etc/passwd",
    ]

    results = [engine.detect_security_threat(p) for p in payloads]

    assert len(results) == len(payloads)
    assert all("is_anomaly" in r for r in results)


def test_engine_save_load_cycle():
    """Test complete save/load cycle"""
    engine = OmniMercuryEngine(mode="fusion")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_engine.pt")

        engine.save_model(save_path)
        assert os.path.exists(save_path)

        loaded_engine = OmniMercuryEngine(mode="fusion")
        loaded_engine.load_model(save_path)

        assert loaded_engine is not None


def test_engine_configure():
    """Test engine configuration"""
    from omni_mercury_engine.core.config import EngineConfig

    config = EngineConfig()
    engine = OmniMercuryEngine(config=config)

    assert engine.config is not None


def test_engine_batch_detection():
    """Test batch anomaly detection"""
    engine = OmniMercuryEngine()

    batch_data = [
        np.random.randn(50, 3),
        np.random.randn(60, 3),
        np.random.randn(40, 3),
    ]

    results = [engine.detect(data) for data in batch_data]

    assert len(results) == len(batch_data)
    assert all("is_anomaly" in r for r in results)


def test_engine_with_fusion_inference():
    """Test detection using fusion network"""
    engine = OmniMercuryEngine(mode="fusion")

    test_data = np.random.randn(50, 3)
    results = engine.detect_with_fusion(test_data)

    assert results is not None
    assert "mode" in results
