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
Post-Quantum Cryptography Production Guards

Ensures PQC environment is properly configured for production use.
AMA Cryptography is the sole PQC backend — there are no fallbacks.
The validated upstream surface is pinned to ``v3.2.0`` (matching the
CI ``AMA_REF`` and the ``[pqc]`` extra in ``pyproject.toml``).

Mercury Agent hard-requires AMA Cryptography. If the package is not
installed, ``pqc_backends`` will raise ``ImportError`` at module load.
These guards verify that the *native C library* inside AMA is built so
that real PQC algorithms (ML-DSA-65, ML-KEM-1024 / Kyber-1024,
SLH-DSA-SHAKE-128s / SPHINCS+) are available at runtime.
"""

import logging
import os

from omni_mercury_engine.security.pqc_backends import (
    AMA_CRYPTOGRAPHY_AVAILABLE,
    AVA_GUARDIAN_AVAILABLE,
    DILITHIUM_AVAILABLE,
    KYBER_AVAILABLE,
    SPHINCS_AVAILABLE,
    get_active_backend,
)

logger = logging.getLogger(__name__)


class PQCProductionWarning(UserWarning):
    """Warning raised when native PQC algorithms are unavailable."""

    pass


# Backward compatibility alias
PQCSimulationWarning = PQCProductionWarning


def check_pqc_production_readiness() -> dict[str, bool | str]:
    """
    Check if PQC algorithms are available via AMA Cryptography's native C library.

    AMA Cryptography is always installed (enforced by pqc_backends.py).
    This checks whether the native C backend is built, which provides
    ML-DSA-65, Kyber-1024, and SPHINCS+ implementations.

    Returns:
        Dictionary with availability status for each algorithm.

    Raises:
        RuntimeError: If AMA_REQUIRE_REAL_PQC=true and native PQC not built.
    """
    backend = get_active_backend()

    results: dict[str, bool | str] = {
        "dilithium": DILITHIUM_AVAILABLE,
        "kyber": KYBER_AVAILABLE,
        "sphincs": SPHINCS_AVAILABLE,
        "backend": backend.value,
        "ama_cryptography": AMA_CRYPTOGRAPHY_AVAILABLE,
        "ava_guardian": AVA_GUARDIAN_AVAILABLE,  # backward compat alias
    }

    if DILITHIUM_AVAILABLE and KYBER_AVAILABLE and SPHINCS_AVAILABLE:
        logger.info("AMA Cryptography native C backend: all PQC algorithms available")
    elif DILITHIUM_AVAILABLE or KYBER_AVAILABLE or SPHINCS_AVAILABLE:
        missing = []
        if not DILITHIUM_AVAILABLE:
            missing.append("ML-DSA-65")
        if not KYBER_AVAILABLE:
            missing.append("Kyber-1024")
        if not SPHINCS_AVAILABLE:
            missing.append("SPHINCS+")
        logger.warning(
            "AMA Cryptography native C backend: partial — missing %s. "
            "Build with: cmake -B build -DAMA_USE_NATIVE_PQC=ON && cmake --build build",
            ", ".join(missing),
        )
    else:
        logger.warning(
            "AMA Cryptography native C backend not built — no PQC algorithms available. "
            "Build with: cmake -B build -DAMA_USE_NATIVE_PQC=ON && cmake --build build"
        )

    # Enforce production requirement if set (support both env var names for compat)
    require_real = os.environ.get(
        "AMA_REQUIRE_REAL_PQC", os.environ.get("AVA_REQUIRE_REAL_PQC", "")
    ).lower() in (
        "true",
        "1",
        "yes",
    )

    if require_real:
        # Reject any partial install: Mercury exposes Dilithium, Kyber,
        # AND SPHINCS+ surfaces, and a Dilithium-only build would let
        # the process start in a cryptographically incomplete state.
        #
        # ``omni_mercury_engine._pqc_gate._enforce_pqc_production_gate``
        # is an *independent* implementation of the same contract — it
        # does NOT call into this helper, nor does this helper call into
        # it.  The two are kept in sync by sharing the
        # ``_PQC_BUILD_RECOVERY_HINT`` constant.
        missing = []
        if not DILITHIUM_AVAILABLE:
            missing.append("ML-DSA-65 (Dilithium)")
        if not KYBER_AVAILABLE:
            missing.append("Kyber-1024")
        if not SPHINCS_AVAILABLE:
            missing.append("SPHINCS+")
        if missing:
            # Reuse the canonical recovery hint defined alongside the
            # import-time gate so the two raise paths give operators
            # identical remediation steps.
            from omni_mercury_engine._pqc_gate import _PQC_BUILD_RECOVERY_HINT

            raise RuntimeError(
                "AMA_REQUIRE_REAL_PQC=true but the AMA Cryptography native C "
                f"backend is incomplete; missing or unavailable: {', '.join(missing)}.\n"
                f"{_PQC_BUILD_RECOVERY_HINT}"
            )

    if not DILITHIUM_AVAILABLE:
        import warnings

        warnings.warn(
            "PQC native algorithms not available. "
            "Build AMA Cryptography's native C library for production security.",
            PQCProductionWarning,
            stacklevel=2,
        )

    return results


def assert_no_simulation_in_production() -> None:
    """
    BLOCKS application startup if native PQC algorithms are unavailable in production.

    Mercury Agent refuses to run without real PQC cryptography in production
    environments.

    Usage:
        if os.environ.get("ENVIRONMENT") == "production":
            assert_no_simulation_in_production()

    Raises:
        RuntimeError: If native PQC algorithms are not available.
    """
    if not DILITHIUM_AVAILABLE:
        raise RuntimeError(
            "PRODUCTION BLOCKED: Native PQC algorithms not available.\n"
            "AMA Cryptography is installed but its native C library is not built.\n"
            "Build with: cmake -B build -DAMA_USE_NATIVE_PQC=ON && cmake --build build\n"
            "\n"
            "Mercury Agent refuses to run without real PQC cryptography in production."
        )


__all__ = [
    "PQCProductionWarning",
    "PQCSimulationWarning",
    "assert_no_simulation_in_production",
    "check_pqc_production_readiness",
]
