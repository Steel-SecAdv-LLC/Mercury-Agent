"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

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
libraries (liboqs-python, pqcrypto) for production security.
"""

import logging
import os
import warnings


logger = logging.getLogger(__name__)


class PQCSimulationWarning(UserWarning):
    """Warning raised when PQC operates in simulation mode."""

    pass


def check_pqc_production_readiness() -> dict[str, bool]:
    """
    Check if real PQC libraries are available.

    Returns:
        Dictionary with availability status for each algorithm.

    Raises:
        RuntimeError: If AVA_REQUIRE_REAL_PQC=true and libraries missing.
    """
    results = {
        "dilithium": False,
        "kyber": False,
        "sphincs": False,
    }

    # Check liboqs (primary PQC backend)
    try:
        import oqs  # noqa: F401

        results["dilithium"] = True
        results["kyber"] = True
        results["sphincs"] = True
        logger.debug("liboqs-python available - all PQC algorithms supported")
    except ImportError:
        # liboqs not installed - will try fallback backends
        logger.debug("liboqs-python not available, checking fallback backends")

    # Check pqcrypto fallback
    if not all(results.values()):
        try:
            import pqcrypto.sign.dilithium2 as _

            results["dilithium"] = True
            logger.debug("pqcrypto dilithium available")
        except ImportError:
            # pqcrypto dilithium not available - algorithm will use simulation
            logger.debug("pqcrypto dilithium not available")

        try:
            import pqcrypto.kem.kyber512 as _  # noqa: F401

            results["kyber"] = True
            logger.debug("pqcrypto kyber available")
        except ImportError:
            # pqcrypto kyber not available - algorithm will use simulation
            logger.debug("pqcrypto kyber not available")

    # Enforce production requirement if set
    require_real = os.environ.get("AVA_REQUIRE_REAL_PQC", "").lower() in (
        "true",
        "1",
        "yes",
    )

    if require_real and not all(results.values()):
        missing = [k for k, v in results.items() if not v]
        raise RuntimeError(
            f"AVA_REQUIRE_REAL_PQC=true but missing libraries for: {missing}. "
            f"Install liboqs-python: pip install liboqs-python"
        )

    if not all(results.values()):
        warnings.warn(
            "PQC operating in SIMULATION mode. Install liboqs-python for production security.",
            PQCSimulationWarning,
            stacklevel=2,
        )

    return results


def assert_no_simulation_in_production() -> None:
    """
    Call this at application startup to ensure real PQC in production.

    Usage:
        if os.environ.get("ENVIRONMENT") == "production":
            assert_no_simulation_in_production()

    Raises:
        RuntimeError: If any PQC algorithm is using simulation mode.
    """
    results = check_pqc_production_readiness()
    if not all(results.values()):
        missing = [k for k, v in results.items() if not v]
        raise RuntimeError(
            f"PRODUCTION BLOCK: Cannot start with simulated PQC. "
            f"Missing real implementations for: {missing}"
        )


__all__ = [
    "PQCSimulationWarning",
    "assert_no_simulation_in_production",
    "check_pqc_production_readiness",
]
