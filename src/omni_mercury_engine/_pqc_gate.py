"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

Import-time PQC production gate.

AMA is mandatory at runtime in production.  The gate's effective default is:

* ``MERCURY_ENV=production`` (and ``AMA_REQUIRE_REAL_PQC`` unset) →
  ``omni_mercury_engine`` package import refuses to proceed unless the AMA
  Cryptography native C backend is fully loadable.  A production process can
  no longer start against the soft PQC stubs.
* ``MERCURY_ENV`` unset/``development`` (the default mode, used by CI and
  local dev) → the gate is a silent no-op, so importing the package without
  AMA's native build keeps working.

An explicit ``AMA_REQUIRE_REAL_PQC`` (or the legacy ``AVA_REQUIRE_REAL_PQC``)
overrides the mode-derived default in either direction: ``=true`` forces the
gate on anywhere, ``=false`` is the documented opt-out for an AMA-less lane
(set it in any CI job that does not build AMA).  When the gate is on it
imports the soft PQC stubs in ``security.pqc_backends`` only as a last
resort; a complete AMA native build is required to pass.

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
    "  git clone --depth 1 --branch v3.2.0 \\\n"
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
    ("SPHINCS_AVAILABLE", "SPHINCS+"),
)


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
    Neither location is reliably populated by the real AMA v3.2.0
    install — the canonical location is the ``pqc_backends`` submodule,
    which is where Mercury's own ``security/pqc_backends.py`` reads
    them.  Aligning the gate with that reader keeps both views of AMA
    availability consistent and stops false-positive partial-install
    rejections in the verify-real-pqc CI lane.
    """
    require_flag = os.environ.get("AMA_REQUIRE_REAL_PQC", os.environ.get("AVA_REQUIRE_REAL_PQC"))
    if require_flag is not None:
        # Explicit operator override wins in either direction and in any
        # mode: ``=true`` forces the gate on, ``=false`` (the documented
        # opt-out for an AMA-less CI/dev lane) forces it off.
        require_real = require_flag.strip().lower() in ("true", "1", "yes", "on")
    else:
        # Default-on in production, opt-in elsewhere.  AMA is mandatory at
        # runtime when ``MERCURY_ENV=production`` so a production process
        # cannot start against the soft PQC stubs; development/CI (the
        # default mode) stays a silent no-op so importing the package
        # without AMA's native build keeps working unless real PQC is
        # explicitly requested.  This is the "strict in production, do not
        # surprise-break unrelated CI" contract — flip it per-lane with
        # ``AMA_REQUIRE_REAL_PQC``.
        from omni_mercury_engine._env import is_production

        require_real = is_production()
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

    if missing:
        raise RuntimeError(
            "AMA_REQUIRE_REAL_PQC=true but the AMA Cryptography native C "
            f"backend is incomplete; missing or unavailable: {', '.join(missing)}.\n"
            f"{_PQC_BUILD_RECOVERY_HINT}"
        )
