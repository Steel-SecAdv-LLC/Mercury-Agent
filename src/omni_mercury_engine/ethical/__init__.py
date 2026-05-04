"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Ethical AI Module - Ethical Constraint Engine

This module provides ethical constraint systems for bias detection,
fairness verification, and multi-dimensional ethical scoring.

Hard-enforcement decision boundary
----------------------------------

The following public surfaces are the *decision boundary* for ethics
enforcement.  Each one MUST raise
:class:`EthicalViolation` (re-exported from
:class:`~omni_mercury_engine.cognitive.ethical_bounding.EthicalConstraintViolationError`)
when its check fails — there is no advisory or observe-only mode at the
boundary, and there is no flag the caller can set to disable it:

- :meth:`omni_mercury_engine.cognitive.orchestrator.CognitiveOrchestrator.analyze`
  raises with ``check="benevolence"`` when the per-analysis benevolence
  score falls below the scorer's threshold.
- :meth:`omni_mercury_engine.core.neurosymbolic_hub.NeuroSymbolicHub.predict`
  raises with ``check="benevolence"`` for any sample whose computed
  benevolence is below ``benevolence_threshold``.
- :meth:`omni_mercury_engine.engine.OmniMercuryEngine.detect_with_fusion`
  (and ``detect_with_fusion_calibrated``) raises with
  ``check="benevolence"`` via :meth:`BenevolenceScorer.enforce` — the
  same primitive used by the orchestrator.  Additionally, the GOSNN
  σ_Immutable neural gate (trained by
  ``scripts/train_sigma_immutable.py``) provides a second independent
  ethical check — its score is recorded in ``gosnn_metadata``.  GOSNN
  failures do not block detection; they populate ``gosnn_metadata``
  with ``fallback_mode=True`` and the error.
- :meth:`omni_mercury_engine.cognitive.ethical_bounding.BenevolenceScorer.enforce`
  is the primitive enforcement hook used by the boundary methods above.

Surfaces *outside* the boundary (e.g., training-time scorers, audit
helpers, narrative engines) still expose the underlying score for
inspection and may log warnings, but they are *not* the gate.  All
production inference paths funnel through the boundary above.

A regression-style test in ``tests/ethical/test_hard_enforcement.py``
exercises every boundary surface and is included in the
``Neuro-Symbolic Tests`` CI job — a benevolence-threshold regression
cannot merge silently.
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
