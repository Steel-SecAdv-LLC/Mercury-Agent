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

"""Tests for neurosymbolic engine integration"""

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
# Legacy LTN retirement (2026-06-02).
#
# The untrained ``LogicTensorNetwork`` (a never-trained nn.Module whose
# random-init forward pass was fed into the fusion consensus as if it were a
# neural confidence) was retired.  ``neural_inference`` is now a deterministic
# statistical heuristic.  The canonical *trained* neuro-symbolic surface is
# ``ml/symbolic_constraint.py::SymbolicConstraintModule``.
# ---------------------------------------------------------------------------


def test_untrained_ltn_class_is_retired() -> None:
    """No second 'LTN' remains exported/constructed while doing nothing."""
    import omni_mercury_engine.models.neurosymbolic as nsm

    assert not hasattr(nsm, "LogicTensorNetwork")
    engine = NeurosymbolicEngine(input_dim=64)
    assert not hasattr(engine, "ltn")
    stats = engine.get_statistics()
    assert stats["ltn_available"] is False
    assert stats["neural_inference_mode"] == "deterministic_heuristic"


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
