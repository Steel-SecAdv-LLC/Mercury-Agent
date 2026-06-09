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

Two **independent** hard ethical gates run at every boundary, in order:

1. :class:`~omni_mercury_engine.cognitive.ethical_bounding.BenevolenceScorer`
   — keyword/context primitive raised as
   :class:`EthicalViolation` with ``check="benevolence"``.
2. :class:`~omni_mercury_engine.security.sigma_immutable_gate.SigmaImmutableGate`
   — trained 256-D scalar network (process-wide singleton, weights at
   ``src/omni_mercury_engine/security/sigma_immutable_weights.pt``)
   raised as :class:`EthicalViolation` with ``check="sigma_immutable"``.
   When GOSNN itself cannot be evaluated (corpus signature failure,
   missing torch, …) the boundary raises ``check="gosnn_unavailable"``.

Boundary surfaces:

- :meth:`omni_mercury_engine.cognitive.orchestrator.CognitiveOrchestrator.analyze`
  raises ``check="benevolence"`` when the per-analysis benevolence
  score falls below the scorer's threshold, then projects the score
  through ``security.sigma_immutable_gate.project_benevolence_to_sigma_band``
  and calls :meth:`SigmaImmutableGate.enforce` (Wave A).
- :meth:`omni_mercury_engine.core.neurosymbolic_hub.NeuroSymbolicHub.predict`
  raises ``check="benevolence"`` for any sample whose computed
  benevolence is below ``benevolence_threshold``, then runs the
  per-sample σ_Immutable check via :meth:`SigmaImmutableGate.enforce`
  (Wave A).
- :meth:`omni_mercury_engine.engine.OmniMercuryEngine.detect_with_fusion`
  (and ``detect_with_fusion_calibrated``) runs the dual gate inside
  ``_enforce_ethics_at_boundary`` (Wave B): BenevolenceScorer raises
  ``check="benevolence"`` first, the σ_Immutable gate raises
  ``check="sigma_immutable"`` for sub-threshold scalar vectors, and a
  ``check="gosnn_unavailable"`` is raised when GOSNN cannot run.  The
  previous ``gosnn_metadata.fallback_mode=True`` path is gone — the
  σ_Immutable gate is no longer optional.
- :meth:`omni_mercury_engine.cognitive.ethical_bounding.BenevolenceScorer.enforce`
  is the primitive enforcement hook used by the boundary methods above.
- :meth:`omni_mercury_engine.security.sigma_immutable_gate.SigmaImmutableGate.enforce`
  is the σ_Immutable primitive enforcement hook.

Production callers MUST NOT toggle the private ``_enable_gosnn``
parameter on the engine's detect_with_fusion variants.  Unit tests
that need to run without GOSNN must additionally set the auditable
module-level flag
:data:`omni_mercury_engine.engine._GOSNN_TESTING_BYPASS`.

Surfaces *outside* the boundary (e.g., training-time scorers, audit
helpers, narrative engines) still expose the underlying scores for
inspection and may log warnings, but they are *not* the gate.  All
production inference paths funnel through the dual-gate boundary above.

A regression-style test in ``tests/ethical/test_hard_enforcement.py``
exercises every boundary surface — both BenevolenceScorer and
σ_Immutable — and is included in the ``Neuro-Symbolic Tests`` CI job,
so a benevolence- or σ_Immutable-threshold regression cannot merge
silently.
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
    ImmutableGeometryProcessor,
    ImmutableWisdomEngine,
    MaatBalanceEngine,
    TwelveFoldVerificationSystem,
    VerificationDimension,
    WisdomArchetype,
)

__all__ = [
    "AlignmentArchetype",
    "AthenaWisdomEngine",
    "EthicalViolation",
    "GeometricPatternProcessor",
    "ImmutableGeometryProcessor",
    "ImmutableWisdomEngine",
    "IndivisibleEngine",
    "MaatBalanceEngine",
    "PercipienceEngine",
    "StrategicEngine",
    "TwelveFoldVerificationSystem",
    "VerificationDimension",
    "WisdomArchetype",
]
