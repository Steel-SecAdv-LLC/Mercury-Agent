"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see
https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
Mercury Agent: Neuro-Symbolic AI Framework

Mercury Agent is a comprehensive neuro-symbolic AI platform.  It hybridises a
deep-learning core (163 ``torch.nn.Module`` subclasses spanning visual,
behavioural, physics-based, fusion and differentiable-logic theorem-proving
subsystems, imported across 120 source files) with an explicit symbolic
layer (knowledge graphs, rule bases, formal verification, AST-based code
analysis and case-based reasoning), wired together through
``core.neurosymbolic_hub.NeuroSymbolicHub`` and
``cognitive.neurosymbolic_fusion.NeurosymbolicFusionEngine`` to produce
explainable, ethically-bounded decisions across security, medical,
environmental, humanitarian and infrastructure domains.

The framework ships:

* A 7-phase cognitive evolution stack (neural memory → symbolic logic →
  hybrid fusion → enhanced anomaly detection → autonomous OODA agent →
  ethical bounding → cognitive evolution / self-improvement).
* 22+ specialised detection engines and 14 live real-world data loaders
  (USGS, NOAA, NASA, FEMA, EPA, financial, energy, network security, …).
* A ``NeuroSymbolicHub`` enforcing a hard benevolence floor of 0.70 at
  every decision boundary.
* Post-quantum cryptography (Kyber-1024 / ML-DSA-65 / SPHINCS+),
  federated learning, conformal prediction, FastAPI server and CLI.

Anomaly detection is one of the capabilities this AI exposes — not the
limit of what it is.
"""

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


__version__ = "1.7.0"
__author__ = "Steel Security Advisors LLC"
__license__ = "GPL-3.0"

__all__ = [
    "AnomalyMathArrest",
    "DetectorException",
    "EngineConfig",
    "FusionException",
    "ModelException",
    "OmniAnomalyException",
    "OmniMercuryEngine",
]
