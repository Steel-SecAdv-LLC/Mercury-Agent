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

from typing import TYPE_CHECKING

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
# Production PQC gate.
#
# When ``AMA_REQUIRE_REAL_PQC=true`` (or the legacy ``AVA_REQUIRE_REAL_PQC``)
# is set in the environment, package import refuses to proceed unless the
# AMA Cryptography native C backend is actually loadable.  Without the env
# var, Mercury continues to import against the stub PQC functions in
# ``security/pqc_backends.py`` for development convenience — there is no
# automatic fail in dev mode by design.
#
# This is the gate referenced by ``docs/index.md`` and ``docs/INSTALLATION.md``
# as "the production startup gate".  Deployments that set the env var get
# automatic fail-closed behaviour at import time; deployments that do not
# get a soft import.
# ---------------------------------------------------------------------------
_PQC_BUILD_RECOVERY_HINT = (
    "Build the AMA-Cryptography native library from a clone of the upstream\n"
    "repo (Mercury-Agent has no CMakeLists.txt of its own):\n"
    "  git clone --depth 1 --branch v3.1.0 \\\n"
    "      https://github.com/Steel-SecAdv-LLC/AMA-Cryptography.git /tmp/ama-cryptography\n"
    "  cd /tmp/ama-cryptography\n"
    "  cmake -B build -DAMA_USE_NATIVE_PQC=ON && cmake --build build\n"
    "  AMA_NO_CYTHON=1 pip install --no-build-isolation .\n"
    "  export LD_LIBRARY_PATH=/tmp/ama-cryptography/build/lib:"
    "/tmp/ama-cryptography/build:${LD_LIBRARY_PATH:-}\n"
    "See docs/INSTALLATION.md 'Post-Quantum Cryptography backend' for the\n"
    "verified procedure (mirrors .github/workflows/pqc-production-check.yml)."
)


def _enforce_pqc_production_gate() -> None:
    """Fail-closed PQC startup gate.

    Inlined rather than dispatched through
    ``security.pqc_guards.check_pqc_production_readiness`` so the no-op
    path (env var unset, the dev-mode default) imports nothing from
    ``security/`` and stays free of ``cryptography``-stack side effects.
    The fail-closed path (env var set, native lib missing) raises before
    any heavier package work.

    Algorithm coverage matches ``check_pqc_production_readiness``: the
    gate fails closed unless **all three** AMA algorithms are loadable
    (ML-DSA-65 via ``dilithium``, Kyber-1024 via ``kyber``, SPHINCS+ via
    ``sphincs``).  Any partial build is rejected because Mercury still
    exposes the Kyber and SPHINCS surfaces elsewhere, and a Dilithium-
    only install would let the process start in a cryptographically
    incomplete state.
    """
    import os

    require_real = os.environ.get(
        "AMA_REQUIRE_REAL_PQC", os.environ.get("AVA_REQUIRE_REAL_PQC", "")
    ).lower() in ("true", "1", "yes")
    if not require_real:
        return

    missing: list[str] = []
    submodules = (
        ("dilithium", "DILITHIUM_AVAILABLE", "ML-DSA-65 (Dilithium)"),
        ("kyber", "KYBER_AVAILABLE", "Kyber-1024"),
        ("sphincs", "SPHINCS_AVAILABLE", "SPHINCS+"),
    )
    import importlib

    for module_name, flag_name, friendly in submodules:
        try:
            mod = importlib.import_module(f"ama_cryptography.{module_name}")
        except ImportError:
            missing.append(friendly)
            continue
        if not getattr(mod, flag_name, False):
            missing.append(friendly)

    if missing:
        raise RuntimeError(
            "AMA_REQUIRE_REAL_PQC=true but the AMA Cryptography native C "
            f"backend is incomplete; missing or unavailable: {', '.join(missing)}.\n"
            f"{_PQC_BUILD_RECOVERY_HINT}"
        )


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


__version__ = "1.6.0"
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
