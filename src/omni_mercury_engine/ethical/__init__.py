# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Ethical AI Module - Ethical Constraint Engine.

This module provides ethical constraint systems for bias detection,
fairness verification, and multi-dimensional ethical scoring.

Hard-enforcement decision boundary
----------------------------------

The following public surfaces are the *decision boundary* for ethics
enforcement.  Each one MUST raise
:class:`EthicalViolation` (re-exported from
:class:`~omni_mercury_engine.cognitive.ethical_bounding.EthicalConstraintViolationError`)
when its check fails — there is no advisory or observe-only mode at the
boundary, and there is no flag the caller can set to disable it.

Two **independent** hard gates run at every boundary, in order:

1. The **harm-uplift gate** — the single fail-closed choke point in
   :mod:`omni_mercury_engine.cognitive.decision_gate`, which scores the two-axis
   (hazard-domain × operational-intent) assessment of
   :func:`~omni_mercury_engine.cognitive.ethical_bounding.assess_weapons_uplift`
   over the **real decision** (surface, domain, request, payload). Raised as
   :class:`EthicalViolation` with ``check="harm_uplift"``. See
   ``docs/HARM_POLICY.md`` for the policy and
   ``tests/pillars/test_non_maleficence.py`` for the enforcement tests.
2. :class:`~omni_mercury_engine.security.sigma_immutable_gate.SigmaImmutableGate`
   — trained 256-D scalar network (process-wide singleton, weights at
   ``src/omni_mercury_engine/security/sigma_immutable_weights.pt``)
   raised as :class:`EthicalViolation` with ``check="sigma_immutable"``.
   When GOSNN itself cannot be evaluated (corpus signature failure,
   missing torch, …) the boundary raises ``check="gosnn_unavailable"``.

What changed, and why
---------------------

The first gate used to be a **benevolence pass-bar**: every action had to score
``>= 0.99`` on a keyword/context heuristic. Two things were wrong with it.

* Each boundary handed the scorer a **fixed string it wrote about itself**
  (``"anomaly_detection:{domain}:audit verify protect research evidence fair
  oversight monitor data care help support"``). The caller's request never
  reached the scorer, so the check could not discriminate anything.
* A high bar on a positivity lexicon is not a harm control. It refused benign
  work whose vocabulary was plain, and admitted anything phrased warmly.

Benevolence is retained as an **advisory** score: computed, logged and attached
to the :class:`EthicalScore`, deciding nothing. ``check="benevolence"`` is
retired; the enforced code is ``check="harm_uplift"``.

Boundary surfaces:

- :meth:`omni_mercury_engine.engine.OmniMercuryEngine.detect`,
  :meth:`~omni_mercury_engine.engine.OmniMercuryEngine.detect_batch`,
  :meth:`~omni_mercury_engine.engine.OmniMercuryEngine.detect_biometric`,
  :meth:`~omni_mercury_engine.engine.OmniMercuryEngine.detect_security_threat`,
  :meth:`~omni_mercury_engine.engine.OmniMercuryEngine.detect_with_fusion` and
  its calibrated variant each carry the ``GATED_BOUNDARY`` capability contract
  (:mod:`omni_mercury_engine.agentic.capabilities.contract`), which runs the
  choke point before the method body. The annotation is registered, so deleting
  it fails ``tests/pillars/test_non_maleficence.py`` in CI.
- :meth:`omni_mercury_engine.cognitive.orchestrator.CognitiveOrchestrator.analyze`
  and :meth:`omni_mercury_engine.core.neurosymbolic_hub.NeuroSymbolicHub.predict`
  call the choke point directly, then run σ_Immutable.
- :meth:`omni_mercury_engine.agentic.orchestration.MultiAgentOrchestrator.detect`,
  ``SubAgentFleet.commit``, ``MercuryAgent._execute_task``, the narrative voice,
  the federation aggregator, the FL server and the reasoning backend all route
  through the same function — the last four via
  :func:`~omni_mercury_engine.security.sigma_immutable_gate.enforce_dual_ethical_gate`.

Production callers MUST NOT toggle the private ``_enable_gosnn``
parameter on the engine's detect_with_fusion variants.  Unit tests
that need to run without GOSNN must additionally set the auditable
module-level flag
:data:`omni_mercury_engine.engine._GOSNN_TESTING_BYPASS`.

Surfaces *outside* the boundary (e.g., training-time scorers, audit
helpers, narrative engines) still expose the underlying scores for
inspection and may log warnings, but they are *not* the gate.  All
production inference paths funnel through the dual-gate boundary above.

``tests/ethical/test_hard_enforcement.py`` exercises every boundary surface and
``tests/pillars/test_non_maleficence.py`` pins the routing, the fail-closed
behaviour and the surface-independence of the verdict.  Both run in the
``Neuro-Symbolic Tests`` CI job, so a gate regression cannot merge silently.
"""

from __future__ import annotations

from omni_mercury_engine.cognitive.ethical_bounding import (
    EthicalConstraintViolationError as EthicalViolation,
)
from omni_mercury_engine.ethical.ethical_alignment_engine import (
    AlignmentArchetype,
    GeometricPatternProcessor,
    IndivisibleEngine,
    PercipienceEngine,
    StrategicEngine,
)
from omni_mercury_engine.ethical.ethical_constraint_engine import (
    AthenaWisdomEngine,
    GeometricPattern,
    ImmutableGeometryProcessor,
    ImmutableWisdomEngine,
    MaatBalanceEngine,
    TwelveFoldVerificationSystem,
    VerificationDimension,
)

__all__ = [
    "AlignmentArchetype",
    "AthenaWisdomEngine",
    "EthicalViolation",
    "GeometricPattern",
    "GeometricPatternProcessor",
    "ImmutableGeometryProcessor",
    "ImmutableWisdomEngine",
    "IndivisibleEngine",
    "MaatBalanceEngine",
    "PercipienceEngine",
    "StrategicEngine",
    "TwelveFoldVerificationSystem",
    "VerificationDimension",
]
