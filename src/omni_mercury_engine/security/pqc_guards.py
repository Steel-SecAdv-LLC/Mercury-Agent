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
AMA Cryptography v2.0 is the sole PQC backend — there are no fallbacks.

Mercury Agent hard-requires AMA Cryptography.  If the package is not
installed, ``pqc_backends`` will raise ``ImportError`` at module load.
These guards verify that the *native C library* inside AMA is built so
that real PQC algorithms (ML-DSA-65, Kyber-1024, SPHINCS+) are available
at runtime.
"""

import logging

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
    """Public exception type for PQC-availability warnings.

    Retained as a stable, importable/catchable symbol for downstream
    integrators. NOTE: Mercury's import-time PQC gate
    (``_pqc_gate._enforce_pqc_production_gate``) is now **fail-closed and
    unconditional** — a missing or partial AMA build raises ``RuntimeError``
    at import rather than degrading with a warning. As a result this warning
    is *not* emitted by the package's own startup path; there is no degraded
    "dev mode" to warn about. It remains for third-party code that wants to
    raise/catch a typed PQC warning in its own flows.
    """

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
        RuntimeError: If any mandatory native PQC algorithm is unavailable.
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

    missing = []
    if not DILITHIUM_AVAILABLE:
        missing.append("ML-DSA-65 (Dilithium)")
    if not KYBER_AVAILABLE:
        missing.append("Kyber-1024")
    if not SPHINCS_AVAILABLE:
        missing.append("SPHINCS+")
    if missing:
        from omni_mercury_engine._pqc_gate import _PQC_BUILD_RECOVERY_HINT

        raise RuntimeError(
            "AMA/PQC is mandatory for Mercury, but the AMA Cryptography native C "
            f"backend is incomplete; missing or unavailable: {', '.join(missing)}.\n"
            f"{_PQC_BUILD_RECOVERY_HINT}"
        )

    logger.info("AMA Cryptography native C backend: all PQC algorithms available")
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
    if not (DILITHIUM_AVAILABLE and KYBER_AVAILABLE and SPHINCS_AVAILABLE):
        raise RuntimeError(
            "PRODUCTION BLOCKED: Native PQC algorithms not available.\n"
            "AMA Cryptography is installed but its native C library is incomplete.\n"
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
