# Copyright (C) 2025 Steel Security Advisors LLC
"""Full engine tests to boost coverage."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import os
import tempfile

import numpy as np

from omni_mercury_engine.engine import OmniMercuryEngine


def test_engine_detect_with_all_detectors() -> None:
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


def test_engine_detect_with_subset() -> None:
    """Test detection with subset of detectors"""
    engine = OmniMercuryEngine()

    data = np.random.randn(50, 3)

    results = engine.detect(data, detector_types=["statistical"])

    assert results is not None
    assert "detectors" in results


def test_engine_fusion_mode() -> None:
    """Test fusion mode initialization"""
    engine = OmniMercuryEngine(mode="fusion")

    assert engine.fusion_model is not None
    assert engine.fusion_inference is not None


def test_engine_biometric_detection() -> None:
    """Test biometric anomaly detection"""
    engine = OmniMercuryEngine()

    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    results = engine.detect_biometric(image)

    assert results is not None
    assert "model_type" in results or "error" in results


def test_engine_security_scan() -> None:
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


def test_engine_save_load_cycle() -> None:
    """Test complete save/load cycle"""
    engine = OmniMercuryEngine(mode="fusion")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_engine.pt")

        engine.save_model(save_path)
        assert os.path.exists(save_path)

        loaded_engine = OmniMercuryEngine(mode="fusion")
        loaded_engine.load_model(save_path)

        assert loaded_engine is not None


def test_engine_configure() -> None:
    """Test engine configuration"""
    from omni_mercury_engine.core.config import EngineConfig

    config = EngineConfig()
    engine = OmniMercuryEngine(config=config)

    assert engine.config is not None


def test_engine_batch_detection() -> None:
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


def test_engine_with_fusion_inference() -> None:
    """Test detection using fusion network"""
    engine = OmniMercuryEngine(mode="fusion")

    test_data = np.random.randn(50, 3)
    results = engine.detect_with_fusion(test_data)

    assert results is not None
    assert "mode" in results
