"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

Import-time PQC production gate.

When ``AMA_REQUIRE_REAL_PQC=true`` (or the legacy ``AVA_REQUIRE_REAL_PQC``)
is set, ``omni_mercury_engine``'s package import refuses to proceed unless
the AMA Cryptography native C backend is fully loadable.  Without the env
var, ``_enforce_pqc_production_gate`` is a no-op and Mercury imports
against the soft PQC stubs in ``security.pqc_backends`` for development
convenience.

Algorithm coverage matches ``security.pqc_guards.check_pqc_production_readiness``:
the gate fails closed unless **all three** AMA algorithms are loadable
(ML-DSA-65 via ``dilithium``, Kyber-1024 via ``kyber``, SPHINCS+ via
``sphincs``).  Any partial build is rejected because Mercury still exposes
the Kyber and SPHINCS surfaces elsewhere, and a Dilithium-only install
would let the process start in a cryptographically incomplete state.

Lives in its own module rather than inline in ``__init__.py`` so the gate
function has a stable, importable location for unit tests; ``__init__.py``
imports and calls it once at package-load time, then deletes the local
re-binding to keep the public package surface clean.
"""

from __future__ import annotations

import os

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

_PQC_FLAGS: tuple[tuple[str, str], ...] = (
    ("DILITHIUM_AVAILABLE", "ML-DSA-65 (Dilithium)"),
    ("KYBER_AVAILABLE", "Kyber-1024"),
    ("SPHINCS_AVAILABLE", "SPHINCS+"),
)


def _enforce_pqc_production_gate() -> None:
    """Fail-closed PQC startup gate.  See module docstring for the contract.

    Reads the three ``*_AVAILABLE`` flags from the **top-level**
    ``ama_cryptography`` package, matching how
    ``security/pqc_backends.py`` already consumes them
    (``from ama_cryptography import DILITHIUM_AVAILABLE, ...``).  Earlier
    iterations of this gate read the flags from per-algorithm submodules
    (``ama_cryptography.dilithium.DILITHIUM_AVAILABLE``), which the
    actual AMA v3.1.0 surface does not always populate identically to
    the top-level constants — so a real verified-real-pqc CI run with
    a successful build could trigger a false-positive partial-install
    rejection.  Reading from the top level keeps this gate consistent
    with the rest of the codebase's view of AMA availability.
    """
    require_real = os.environ.get(
        "AMA_REQUIRE_REAL_PQC", os.environ.get("AVA_REQUIRE_REAL_PQC", "")
    ).lower() in ("true", "1", "yes")
    if not require_real:
        return

    try:
        import ama_cryptography
    except ImportError as exc:
        raise RuntimeError(
            "AMA_REQUIRE_REAL_PQC=true but the AMA Cryptography Python "
            "package is not importable (import ama_cryptography failed).\n"
            f"{_PQC_BUILD_RECOVERY_HINT}"
        ) from exc

    missing: list[str] = []
    for flag_name, friendly in _PQC_FLAGS:
        if not getattr(ama_cryptography, flag_name, False):
            missing.append(friendly)

    if missing:
        raise RuntimeError(
            "AMA_REQUIRE_REAL_PQC=true but the AMA Cryptography native C "
            f"backend is incomplete; missing or unavailable: {', '.join(missing)}.\n"
            f"{_PQC_BUILD_RECOVERY_HINT}"
        )
