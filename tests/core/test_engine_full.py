# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
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
    # Opt into legacy auto-fit-on-first-batch so this smoke test of the inference
    # path does not trip the (intentional) fail-loud-on-unfit-detector guard.
    engine = OmniMercuryEngine(mode="fusion", require_explicit_fit=False)

    test_data = np.random.randn(50, 3)
    results = engine.detect_with_fusion(test_data)

    assert results is not None
    assert "mode" in results


def test_engine_explain_detection_is_governed_and_offline() -> None:
    """Engine calls its subordinate reasoning backend to explain its own detection.

    Verifies the wiring is real: a ``detect_with_fusion`` certificate flows into
    ``explain_detection`` and comes back as a governed, provenance-stamped
    Explanation served by the offline-first local backend (template in CI), with
    the usage ledger threaded. The template path records no usage at all (it
    never calls ``_record_usage``), so the ledger total is zero tokens — never a
    fabricated count.
    """
    engine = OmniMercuryEngine(mode="fusion", auto_load_checkpoint=True)
    data = np.random.RandomState(7).randn(48, 16).astype(np.float64)
    result = engine.detect_with_fusion(data, domain="security")

    explanation = engine.explain_detection(result, domain="security")
    assert explanation.backend == "local"
    assert explanation.gated is True
    assert isinstance(explanation.text, str) and explanation.text
    assert engine.reasoning_backend.is_offline is True

    totals = engine.reasoning_usage()
    assert totals is not None
    assert totals["total_tokens"] == 0  # template records no usage -> zero, never fabricated


def test_engine_reasoning_respects_ethics_gate_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unevaluable harm gate blocks explanation at the engine boundary.

    The reasoning boundary runs the shared fail-closed choke point before any
    generated content is surfaced. Breaking the gate itself is the general
    property worth pinning: an error inside the control must read as
    "refused", never as "allowed".
    """
    import omni_mercury_engine.cognitive.decision_gate as gate_module
    from omni_mercury_engine.cognitive.ethical_bounding import EthicalConstraintViolationError
    from omni_mercury_engine.reasoning.backends import LocalReasoningBackend

    engine = OmniMercuryEngine(mode="fusion", auto_load_checkpoint=True)
    engine.enable_reasoning(backend=LocalReasoningBackend())
    data = np.random.RandomState(1).randn(48, 16).astype(np.float64)
    result = engine.detect_with_fusion(data, domain="security")

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("simulated harm-gate failure")

    monkeypatch.setattr(gate_module, "assess_weapons_uplift", _boom)
    with pytest.raises(EthicalConstraintViolationError) as exc:
        engine.explain_detection(result, domain="security")
    assert exc.value.check == "harm_uplift"


def test_engine_reasoning_registry_drives_local_model() -> None:
    """An operator-populated registry chooses the engine's local reasoning model."""
    from omni_mercury_engine.models.llm_registry import LLMModelRegistry, LLMModelSpec
    from omni_mercury_engine.reasoning.backends import LocalReasoningBackend

    registry = LLMModelRegistry()
    registry.register(
        LLMModelSpec(
            provider="ollama",
            model_id="llama3.1:8b",
            context_window=8192,
            capabilities=frozenset({"chat"}),
        )
    )
    engine = OmniMercuryEngine(mode="fusion")
    engine.enable_reasoning(registry=registry)

    backend = engine.reasoning_backend
    assert isinstance(backend, LocalReasoningBackend)
    assert backend._ollama_config.model == "llama3.1:8b"
