# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Mercury Agent: Neuro-Symbolic AI Framework.

Mercury Agent is a neuro-symbolic AI platform built for a **civilization-first**
mission: detecting and reasoning about hazards across security, medical,
environmental, infrastructure and humanitarian domains, and refusing to act when
it cannot ground a decision. FINDΩYOU — locating the lost, missing and abducted
— is **one deployment** of that mission, not its ceiling.

It hybridises a deep-learning core (173 ``torch.nn.Module`` subclasses spanning
visual, behavioural, physics-based, fusion and differentiable-logic
theorem-proving subsystems, imported across 175 source files; both counts
measured by ``scripts/measure_codebase_scale.py`` and CI-gated in the README
Codebase Scale block) with an explicit symbolic layer (knowledge graphs, rule
bases, formal verification, AST-based code analysis and case-based reasoning),
wired together through ``core.neurosymbolic_hub.NeuroSymbolicHub`` and
``cognitive.neurosymbolic_fusion.NeurosymbolicFusionEngine`` to produce
explainable, ethically-bounded decisions.

The framework ships:

* A cognitive layer wired at runtime by
  ``cognitive.orchestrator.CognitiveOrchestrator`` over ten components
  (knowledge graph, multi-hop reasoner and uncertainty quantifier always on,
  plus plasticity, causal discovery, IPB, case-based reasoning, indicator
  development, curiosity and enhanced anomaly detection optional). Its
  historical build spine is a 7-phase evolution (neural memory → symbolic
  logic → hybrid fusion → enhanced anomaly detection → autonomous OODA agent →
  ethical bounding → curiosity-driven exploration); that is history, not a
  runtime pipeline.
* 30 specialised detection engines, 21 live data-loader classes under
  ``loaders/`` (CI-gated count; 20 are concrete loaders and one is the shared
  base class the count's regex also matches), and the ``datasets/`` benchmark
  corpus (USGS, NOAA, NASA, FEMA, EPA, financial, energy, network security, …).
* One fail-closed harm control at every public decision boundary: the two-axis
  (hazard-domain × operational-intent) **harm-uplift gate**
  (``cognitive.decision_gate``, ``docs/HARM_POLICY.md``), scored on the real
  request, followed by the σ_Immutable configuration-integrity gate. Benevolence
  is an advisory score, not a pass-bar — see ``cognitive.decision_gate`` for why
  the previous ``0.99`` bar was removed.
* Post-quantum cryptography (Kyber-1024 / ML-DSA-65 / SPHINCS+),
  federated learning, conformal prediction, FastAPI server and CLI.

Anomaly detection is one of the capabilities this AI exposes — not the
limit of what it is.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

# Quiet TensorFlow's C++/oneDNN startup banners before anything in the package
# can trigger the deepface -> retinaface -> tensorflow import chain (TF reads
# these only at import time). Set here, at the earliest package entry point, so
# every TF importer is covered regardless of import order (e.g. both
# ``models.biometric`` and ``models.biometric_advanced``). ``setdefault`` keeps
# an operator's explicit choice.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# Type-only imports for static analysis (CodeQL, mypy, etc.)
# These are not imported at runtime to support lazy loading
if TYPE_CHECKING:
    from omni_mercury_engine.core.config import EngineConfig as EngineConfig
    from omni_mercury_engine.core.exceptions import (
        DetectorException as DetectorException,
        FusionException as FusionException,
        ModelException as ModelException,
        OmniAnomalyException as OmniAnomalyException,
    )
    from omni_mercury_engine.detectors.math_arrest.arrest import (
        AnomalyMathArrest as AnomalyMathArrest,
    )
    from omni_mercury_engine.engine import OmniMercuryEngine as OmniMercuryEngine

# ---------------------------------------------------------------------------
# PQC gate.
#
# Mercury package import refuses to proceed unless the AMA Cryptography
# native C backend is fully loadable.  There is no AMA-less development,
# CI, or production mode.
#
# Implementation lives in ``omni_mercury_engine._pqc_gate`` so it has a
# stable importable location for unit tests; the function is invoked once
# here at package-load time and then the local re-binding is deleted to
# keep the public package surface clean.
# ---------------------------------------------------------------------------
from omni_mercury_engine._pqc_gate import _enforce_pqc_production_gate

_enforce_pqc_production_gate()
del _enforce_pqc_production_gate

# Lazy imports to support running without ML dependencies (torch)
# The OmniMercuryEngine requires torch, but we defer the import to allow
# CLI help commands and other lightweight operations to work without it.


def __getattr__(name: str) -> type:
    """Lazy import for OmniMercuryEngine to defer torch dependency."""
    if name == "OmniMercuryEngine":
        from omni_mercury_engine.engine import OmniMercuryEngine

        return OmniMercuryEngine
    elif name == "EngineConfig":
        from omni_mercury_engine.core.config import EngineConfig

        return EngineConfig
    elif name in ("OmniAnomalyException", "DetectorException", "ModelException", "FusionException"):
        from omni_mercury_engine.core.exceptions import (
            DetectorException,
            FusionException,
            ModelException,
            OmniAnomalyException,
        )

        return {
            "OmniAnomalyException": OmniAnomalyException,
            "DetectorException": DetectorException,
            "ModelException": ModelException,
            "FusionException": FusionException,
        }[name]
    elif name == "AnomalyMathArrest":
        from omni_mercury_engine.detectors.math_arrest.arrest import (
            AnomalyMathArrest,
        )

        return AnomalyMathArrest
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from omni_mercury_engine._version import get_version as _get_version

__version__ = _get_version()
__author__ = "Steel Security Advisors LLC"
__license__ = "GPL-3.0-or-later"

__all__ = [
    "AnomalyMathArrest",
    "DetectorException",
    "EngineConfig",
    "FusionException",
    "ModelException",
    "OmniAnomalyException",
    "OmniMercuryEngine",
]
