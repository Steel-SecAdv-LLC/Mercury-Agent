# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fail-closed guard for the orchestrator boundary when the σ gate is absent.

The gate- and engine-level ``gosnn_unavailable`` refusals are covered in
``tests/security`` / ``tests/ethical``; this pins the same contract at the
**agentic** boundary: a :class:`MultiAgentOrchestrator` whose σ_Immutable
network is unavailable must refuse every episode with
``check="gosnn_unavailable"`` — never degrade to ungated decisions. The gate
is forced unavailable directly (the established pattern from
``tests/security/test_sigma_immutable_fail_closed.py``), so the assertion
holds regardless of torch being installed; the matching ML-lane pass path
(a fitted episode clearing both gates) is
``tests/cognitive/test_orchestration_behavioral.py::TestEthicalGating::
test_benign_episode_passes_both_gates``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.agentic.orchestration import MultiAgentOrchestrator
from omni_mercury_engine.cognitive.ethical_bounding import EthicalConstraintViolationError
from omni_mercury_engine.governance.self_improvement import MeasurementGovernance
from omni_mercury_engine.security.sigma_immutable_gate import SigmaImmutableGate


def _gate_with_no_network() -> SigmaImmutableGate:
    """A σ gate in the trained-network-unavailable posture (torch-independent)."""
    gate = SigmaImmutableGate(verify_corpus=False)
    gate._gate = None
    gate._corpus_error = None
    gate._gate_load_error = "test: σ_Immutable network forced unavailable"
    return gate


def test_orchestrated_episode_refuses_with_gosnn_unavailable() -> None:
    """An episode against an unavailable σ network raises the exact check."""
    rng = np.random.default_rng(0)
    X_train: np.ndarray[Any, Any] = rng.normal(0.0, 1.0, size=(200, 5))
    orchestrator = MultiAgentOrchestrator(seed=0, threshold_governance=MeasurementGovernance()).fit(
        X_train
    )
    orchestrator._sigma_immutable_gate = _gate_with_no_network()

    with pytest.raises(EthicalConstraintViolationError) as exc_info:
        orchestrator.run_episode(rng.normal(0.0, 1.0, size=(40, 5)))
    assert getattr(exc_info.value, "check", "") == "gosnn_unavailable"
    assert getattr(exc_info.value, "score", None) == 0.0
