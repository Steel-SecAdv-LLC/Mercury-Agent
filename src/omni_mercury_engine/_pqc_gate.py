"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

Import-time PQC production gate.

When ``AMA_REQUIRE_REAL_PQC=true`` (or the legacy ``AVA_REQUIRE_REAL_PQC``)
is set, ``omni_mercury_engine``'s package import refuses to proceed unless
the AMA Cryptography native C backend is fully loadable.  Without the env
var, ``_enforce_pqc_production_gate`` is a no-op and Mercury imports
against the soft PQC stubs in ``security.pqc_backends`` for development
convenience.

Algorithm coverage:

- **Hard-required** (gate raises ``RuntimeError`` when missing):
  ``DILITHIUM_AVAILABLE`` (ML-DSA-65) and ``KYBER_AVAILABLE`` (Kyber-1024).
  These match the assertions in
  ``.github/workflows/pqc-production-check.yml`` for a real v3.1.0 build.
- **Soft-required** (gate emits ``UserWarning`` but does not raise):
  ``SPHINCS_AVAILABLE``.  SPHINCS+ is documented as part of v3.1.0 but
  the upstream flag is not consistently populated even on a successful
  native build, so making it a hard-required flag would produce
  false-positive rejections in the real-pqc CI lane.  Mercury still
  imports without it; any code path that needs SPHINCS+ will fall
  through to the soft stub at call time.

This matches ``security.pqc_guards.check_pqc_production_readiness`` after
that helper was strengthened in the same branch.  Both raise paths share
the ``_PQC_BUILD_RECOVERY_HINT`` constant so operators see identical
remediation steps regardless of which gate they hit.

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

# Hard-required flags: any one missing fails the gate closed.
#
# Why these two and not SPHINCS+: the upstream
# ``.github/workflows/pqc-production-check.yml`` job that exercises a
# real AMA v3.1.0 native build only asserts ``DILITHIUM_AVAILABLE`` and
# ``KYBER_AVAILABLE`` (see lines 138-144 of that workflow); SPHINCS+ /
# SLH-DSA support is documented as part of v3.1.0 but the
# ``SPHINCS_AVAILABLE`` flag is not consistently set by the upstream
# package even on a successful build.  Including it here as a
# hard-required flag would produce false-positive partial-install
# rejections in the verify-real-pqc CI lane and in any production
# deployment that mirrors that build path.  SPHINCS remains a
# soft-required flag below: a missing SPHINCS surface emits a
# ``UserWarning`` but does not block startup.
_PQC_HARD_REQUIRED: tuple[tuple[str, str], ...] = (
    ("DILITHIUM_AVAILABLE", "ML-DSA-65 (Dilithium)"),
    ("KYBER_AVAILABLE", "Kyber-1024"),
)
_PQC_SOFT_REQUIRED: tuple[tuple[str, str], ...] = (("SPHINCS_AVAILABLE", "SPHINCS+"),)


def _enforce_pqc_production_gate() -> None:
    """Fail-closed PQC startup gate.  See module docstring for the contract.

    Reads the ``*_AVAILABLE`` flags from the **top-level**
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

    missing_hard: list[str] = [
        friendly
        for flag_name, friendly in _PQC_HARD_REQUIRED
        if not getattr(ama_cryptography, flag_name, False)
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
        if not getattr(ama_cryptography, flag_name, False)
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
