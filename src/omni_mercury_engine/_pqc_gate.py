"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

Import-time PQC production gate.

When ``AMA_REQUIRE_REAL_PQC=true`` (or the legacy ``AVA_REQUIRE_REAL_PQC``)
is set, ``omni_mercury_engine``'s package import refuses to proceed unless
the AMA Cryptography native C backend is fully loadable.  Without the env
var, ``_enforce_pqc_production_gate`` is a no-op and Mercury imports
against the soft PQC stubs in ``security.pqc_backends`` for development
convenience.

Algorithm coverage matches
``security.pqc_guards.check_pqc_production_readiness``: the gate
fails closed unless **all three** AMA algorithms are loadable —
ML-DSA-65 (``DILITHIUM_AVAILABLE``), Kyber-1024
(``KYBER_AVAILABLE``), and SPHINCS+ (``SPHINCS_AVAILABLE``).  Any
partial build is rejected because Mercury exposes all three
surfaces and a Dilithium-only install would let the process start
in a cryptographically incomplete state.

The flags are read from ``ama_cryptography.pqc_backends`` — the
canonical location matching what ``security/pqc_backends.py`` reads
(``from ama_cryptography.pqc_backends import DILITHIUM_AVAILABLE,
KYBER_AVAILABLE, SPHINCS_AVAILABLE``).  The top-level
``ama_cryptography`` package and per-algorithm submodules
(``ama_cryptography.dilithium``, etc.) are NOT reliable sources
for these flags on a real install — earlier iterations of this
gate read from those locations and produced false-positive
partial-install rejections.

The two raise paths (this gate and
``check_pqc_production_readiness``) are independent
implementations of the same contract — neither delegates to the
other.  They share the ``_PQC_BUILD_RECOVERY_HINT`` constant so
operators see identical remediation steps regardless of which
path raises.

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

_PQC_HARD_REQUIRED: tuple[tuple[str, str], ...] = (
    ("DILITHIUM_AVAILABLE", "ML-DSA-65 (Dilithium)"),
    ("KYBER_AVAILABLE", "Kyber-1024"),
)
_PQC_SOFT_REQUIRED: tuple[tuple[str, str], ...] = (("SPHINCS_AVAILABLE", "SPHINCS+"),)


def _enforce_pqc_production_gate() -> None:
    """Fail-closed PQC startup gate.  See module docstring for the contract.

    Reads the ``*_AVAILABLE`` flags from ``ama_cryptography.pqc_backends``
    (the canonical location), matching how
    ``security/pqc_backends.py`` already consumes them
    (``from ama_cryptography.pqc_backends import DILITHIUM_AVAILABLE,
    KYBER_AVAILABLE, SPHINCS_AVAILABLE``).  Earlier iterations of this
    gate read the flags from per-algorithm submodules
    (``ama_cryptography.dilithium.DILITHIUM_AVAILABLE``) and from the
    top-level package (``ama_cryptography.DILITHIUM_AVAILABLE``).
    Neither location is reliably populated by the real AMA v3.1.0
    install — the canonical location is the ``pqc_backends`` submodule,
    which is where Mercury's own ``security/pqc_backends.py`` reads
    them.  Aligning the gate with that reader keeps both views of AMA
    availability consistent and stops false-positive partial-install
    rejections in the verify-real-pqc CI lane.
    """
    require_real = os.environ.get(
        "AMA_REQUIRE_REAL_PQC", os.environ.get("AVA_REQUIRE_REAL_PQC", "")
    ).lower() in ("true", "1", "yes")
    if not require_real:
        return

    try:
        import ama_cryptography.pqc_backends as ama_pqc_backends
    except ImportError as exc:
        raise RuntimeError(
            "AMA_REQUIRE_REAL_PQC=true but the AMA Cryptography Python "
            "package is not importable "
            "(import ama_cryptography.pqc_backends failed).\n"
            f"{_PQC_BUILD_RECOVERY_HINT}"
        ) from exc

    missing: list[str] = [
        friendly
        for flag_name, friendly in _PQC_HARD_REQUIRED
        if not getattr(ama_pqc_backends, flag_name, False)
    ]

    if missing_hard:
        raise RuntimeError(
            "AMA_REQUIRE_REAL_PQC=true but the AMA Cryptography native C "
            f"backend is incomplete; missing or unavailable: {', '.join(missing_hard)}.\n"
            f"{_PQC_BUILD_RECOVERY_HINT}"
        )

    missing_soft: list[str] = [
        friendly
        for flag_name, friendly in _PQC_SOFT_REQUIRED
        if not getattr(ama_pqc_backends, flag_name, False)
    ]

    if missing_soft:
        import warnings

        warnings.warn(
            "AMA_REQUIRE_REAL_PQC=true and the AMA Cryptography native C "
            "backend is loadable, but the following soft-required surface "
            f"is unavailable: {', '.join(missing_soft)}.  Mercury will "
            "import but any code path that uses this surface will degrade "
            "to the soft PQC stub at call time.  See "
            "docs/INSTALLATION.md 'Post-Quantum Cryptography backend' for "
            "the full build-with-all-algorithms procedure.",
            UserWarning,
            stacklevel=2,
        )
