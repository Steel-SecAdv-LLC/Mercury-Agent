# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fail-closed behaviour of the σ_Immutable boundary gate when its network is absent.

When the trained GOSNN network cannot be loaded (no torch / missing weights), the
gate must refuse rather than pass: ``evaluate`` reports ``passes=False`` and
``enforce`` raises ``check="gosnn_unavailable"``. Constructed directly with the
network forced unavailable, so the check does not depend on torch being absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.cognitive.ethical_bounding import EthicalConstraintViolationError
from omni_mercury_engine.security.sigma_immutable_gate import SigmaImmutableGate


def _gate_with_no_network() -> SigmaImmutableGate:
    gate = SigmaImmutableGate(verify_corpus=False)
    # Force the trained-network-unavailable posture regardless of torch/weights.
    gate._gate = None
    gate._corpus_error = None
    gate._gate_load_error = "test: σ_Immutable network forced unavailable"
    return gate


class TestSigmaImmutableFailsClosed:
    def test_evaluate_reports_unavailable_and_fails(self) -> None:
        evaluation = _gate_with_no_network().evaluate(np.zeros(180, dtype=float))
        assert evaluation.passes is False
        assert evaluation.backend == "unavailable"

    def test_enforce_raises_gosnn_unavailable(self) -> None:
        gate = _gate_with_no_network()
        with pytest.raises(EthicalConstraintViolationError) as exc_info:
            gate.enforce("gated action", np.zeros(180, dtype=float))
        assert getattr(exc_info.value, "check", "") == "gosnn_unavailable"
