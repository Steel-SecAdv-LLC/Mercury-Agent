"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
Post-Quantum Cryptography Production Guards

Ensures simulation mode is never silently used in production contexts.
Implements fail-fast principles to force installation of real cryptographic
libraries (ama-cryptography, liboqs-python, pqcrypto) for production security.

PQC Backend Priority:
1. AMA Cryptography (ama_cryptography) - Primary, full-featured
2. liboqs-python (oqs) - Secondary fallback
3. pqcrypto - Tertiary fallback
4. SIMULATION - Fail-fast (blocked in production)
"""

import logging
import os
import warnings

from omni_mercury_engine.security.pqc_backends import (
    AMA_CRYPTOGRAPHY_AVAILABLE,
    AVA_GUARDIAN_AVAILABLE,
    LIBOQS_AVAILABLE,
    PQCRYPTO_AVAILABLE,
    PQCBackend,
    get_active_backend,
)

logger = logging.getLogger(__name__)


class PQCSimulationWarning(UserWarning):
    """Warning raised when PQC operates in simulation mode."""

    pass


def check_pqc_production_readiness() -> dict[str, bool | str]:
    """
    Check if real PQC libraries are available.

    Uses the backend detection from pqc_backends.py which follows the priority:
    1. AMA Cryptography (primary)
    2. liboqs-python (secondary)
    3. pqcrypto (tertiary)
    4. SIMULATION (blocked in production)

    Returns:
        Dictionary with availability status for each algorithm and active backend.

    Raises:
        RuntimeError: If AMA_REQUIRE_REAL_PQC=true and libraries missing.
    """
    # Use centralized backend detection from pqc_backends
    backend = get_active_backend()
    has_real_backend = backend != PQCBackend.SIMULATION

    results: dict[str, bool | str] = {
        "dilithium": has_real_backend,
        "kyber": has_real_backend,
        "sphincs": AMA_CRYPTOGRAPHY_AVAILABLE or LIBOQS_AVAILABLE,
        "backend": backend.value,
        "ama_cryptography": AMA_CRYPTOGRAPHY_AVAILABLE,
        "ava_guardian": AVA_GUARDIAN_AVAILABLE,  # backward compat alias
        "liboqs": LIBOQS_AVAILABLE,
        "pqcrypto": PQCRYPTO_AVAILABLE,
    }

    # Log backend status
    if AMA_CRYPTOGRAPHY_AVAILABLE:
        logger.info("AMA Cryptography PQC backend available (PRIMARY)")
    elif LIBOQS_AVAILABLE:
        logger.info("liboqs-python PQC backend available (SECONDARY)")
    elif PQCRYPTO_AVAILABLE:
        logger.warning("pqcrypto PQC backend available (TERTIARY - timing variations)")
    else:
        logger.warning("No real PQC backend available - SIMULATION mode (NOT SECURE)")

    # Enforce production requirement if set (support both env var names for compat)
    require_real = os.environ.get(
        "AMA_REQUIRE_REAL_PQC", os.environ.get("AVA_REQUIRE_REAL_PQC", "")
    ).lower() in (
        "true",
        "1",
        "yes",
    )

    if require_real and not has_real_backend:
        raise RuntimeError(
            "AMA_REQUIRE_REAL_PQC=true but no real PQC backend available.\n"
            "Install one of:\n"
            "  pip install ama-cryptography    # Primary (recommended)\n"
            "  pip install liboqs-python       # Secondary fallback\n"
            "  pip install pqcrypto            # Tertiary fallback"
        )

    if not has_real_backend:
        warnings.warn(
            "PQC operating in SIMULATION mode. "
            "Install ama-cryptography or liboqs-python for production security.",
            PQCSimulationWarning,
            stacklevel=2,
        )

    return results


def assert_no_simulation_in_production() -> None:
    """
    BLOCKS application startup if running with simulated PQC in production.

    This function implements the fail-fast philosophy: Mercury Agent refuses
    to run with simulated cryptography in production environments.

    Usage:
        if os.environ.get("ENVIRONMENT") == "production":
            assert_no_simulation_in_production()

    Raises:
        RuntimeError: If no real PQC backend is available.
    """
    backend = get_active_backend()
    if backend == PQCBackend.SIMULATION:
        raise RuntimeError(
            "PRODUCTION BLOCKED: No real PQC backend available.\n"
            "Install one of:\n"
            "  pip install ama-cryptography    # Primary (recommended)\n"
            "  pip install liboqs-python       # Secondary fallback\n"
            "\n"
            "Mercury Agent refuses to run with simulated cryptography in production."
        )


__all__ = [
    "PQCSimulationWarning",
    "assert_no_simulation_in_production",
    "check_pqc_production_readiness",
]
