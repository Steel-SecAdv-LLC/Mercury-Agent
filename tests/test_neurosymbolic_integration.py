# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for neurosymbolic engine integration."""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.models.neurosymbolic import NeurosymbolicEngine


def test_neurosymbolic_initialization() -> None:
    """Test neurosymbolic engine initialization"""
    engine = NeurosymbolicEngine(input_dim=64)
    assert engine.input_dim == 64
    assert len(engine.knowledge_base) > 0


def test_symbolic_inference() -> None:
    """Test symbolic reasoning"""
    engine = NeurosymbolicEngine()
    engine.add_fact("missing_person")
    engine.add_fact("child")

    result = engine.symbolic_inference("priority_high")
    assert result["result"] is True


def test_neural_inference() -> None:
    """Test neural inference when PyTorch available"""
    engine = NeurosymbolicEngine(input_dim=32)
    features = np.random.randn(32)

    confidence = engine.neural_inference(features)
    assert 0.0 <= confidence <= 1.0


def test_extract_features() -> None:
    """Test feature extraction"""
    engine = NeurosymbolicEngine(input_dim=48)
    data = np.random.randn(5, 20)

    features = engine.extract_features(data)
    assert features.shape[0] == 5


def test_predict() -> None:
    """Test anomaly prediction"""
    engine = NeurosymbolicEngine()
    data = np.random.randn(3, 15)

    result = engine.predict(data)
    assert "anomaly_scores" in result
    assert len(result["anomaly_scores"]) == 3


# ---------------------------------------------------------------------------
# LogicTensorNetwork re-wired to the canonical co-trained module (2026-06-02).
#
# The original ``LogicTensorNetwork`` was a never-trained nn.Module whose
# random-init noise was fed into fusion as if a neural confidence. Rather than
# leave it retired, it is re-wired: ``LogicTensorNetwork.predict`` now routes
# detector scores through the canonical co-trained
# ``ml/symbolic_constraint.py::SymbolicConstraintModule`` consensus predicate
# (deterministic untrained, learned-reliability when co-trained). The raw-
# feature ``neural_inference`` remains an honestly-labelled heuristic.
# ---------------------------------------------------------------------------


def test_ltn_is_wired_to_canonical_symbolic_module() -> None:
    """The LTN surface exists and is backed by SymbolicConstraintModule."""
    import omni_mercury_engine.models.neurosymbolic as nsm

    assert hasattr(nsm, "LogicTensorNetwork")
    engine = NeurosymbolicEngine(input_dim=64)
    assert hasattr(engine, "ltn")
    assert engine.ltn is not None  # torch is present in the [ml] env
    # It wraps the canonical module, not a bespoke random network.
    from omni_mercury_engine.ml.symbolic_constraint import SymbolicConstraintModule

    assert isinstance(engine.ltn.module, SymbolicConstraintModule)
    stats = engine.get_statistics()
    assert stats["ltn_available"] is True
    assert stats["ltn_backend"] == "symbolic_constraint_module"


def test_ltn_predict_delegates_to_symbolic_consensus() -> None:
    """LTN.predict returns the canonical module's per-sample consensus.

    Real signal, deterministic, and monotone in the detector scores (a batch of
    high scores yields higher consensus than a batch of low scores) — and it
    matches ``SymbolicConstraintModule.predict`` exactly (genuine delegation,
    not a reimplementation).
    """
    import torch

    from omni_mercury_engine.models.neurosymbolic import LogicTensorNetwork

    ltn = LogicTensorNetwork(num_detectors=4)
    high = np.full((3, 4), 0.9)
    low = np.full((3, 4), 0.1)
    p_high = ltn.predict(high)
    p_low = ltn.predict(low)
    assert p_high.shape == (3,)
    assert np.all((p_high >= 0.0) & (p_high <= 1.0))
    assert float(p_high.mean()) > float(p_low.mean())  # consensus is monotone
    # Deterministic: same input -> same output (no random network).
    assert np.allclose(ltn.predict(high), p_high)
    # Genuine delegation to the wrapped module's predict.
    direct = ltn.module.predict(torch.tensor(high, dtype=torch.float32)).numpy()
    assert np.allclose(p_high, direct)


def test_ltn_predict_relocates_scores_to_wrapped_module_device() -> None:
    """LTN.predict moves its CPU-built scores onto the wrapped module's device.

    A co-trained ``SymbolicConstraintModule`` that was ``.to(<accelerator>)``'d
    must not raise a device mismatch: the wrapper builds ``scores`` on CPU, so it
    has to relocate them onto the module's parameter device before delegating.
    The always-available ``meta`` device stands in for a non-CPU accelerator
    (pre-fix this raised "Tensor on device meta is not on the expected device
    cpu!").
    """
    from unittest.mock import patch

    import torch

    from omni_mercury_engine.models.neurosymbolic import LogicTensorNetwork

    ltn = LogicTensorNetwork(num_detectors=4)
    ltn.module.to("meta")  # relocate the wrapped module off the default (CPU) device

    # Hand back a real CPU tensor so the wrapper's ``.cpu().numpy()`` succeeds
    # (a meta tensor carries no data); the contract under test is the *device*
    # of the tensor the wrapper passes into ``module.predict``.
    with patch.object(ltn.module, "predict", return_value=torch.full((3,), 0.5)) as mock_predict:
        out = ltn.predict(np.full((3, 4), 0.9))

    handed = mock_predict.call_args.args[0]
    assert handed.device.type == "meta"  # scores were relocated onto the module's device
    assert out.shape == (3,)


def test_neural_inference_is_deterministic_and_feature_responsive() -> None:
    """The retired-network replacement is deterministic and bounded."""
    engine = NeurosymbolicEngine(input_dim=64)
    spiky = np.array([0.1, 6.0, 0.1, 0.2, 0.1, 0.1])
    flat = np.ones(6)
    a1 = engine.neural_inference(spiky)
    a2 = engine.neural_inference(spiky.copy())
    assert a1 == a2  # deterministic (no random network)
    assert 0.0 <= a1 <= 1.0
    # Higher relative dispersion => higher anomaly confidence.
    assert engine.neural_inference(flat) < a1


def test_migration_pointer_to_canonical_cotrained_module() -> None:
    """The canonical trained neuro-symbolic surface is importable and named in
    the legacy module's migration note."""
    from omni_mercury_engine.ml.symbolic_constraint import SymbolicConstraintModule

    assert SymbolicConstraintModule is not None
    # The retired-network method documents the canonical replacement.
    assert "SymbolicConstraintModule" in (NeurosymbolicEngine.neural_inference.__doc__ or "")
