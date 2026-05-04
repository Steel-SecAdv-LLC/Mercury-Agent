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
Backward Compatibility Shim — mercury_guardian → mercury_amacrypto

This module re-exports all public names from mercury_amacrypto.
New code should import directly from
``omni_mercury_engine.integrations.mercury_amacrypto``.
"""

from omni_mercury_engine.integrations.mercury_amacrypto import (
    AMA_CRYPTOGRAPHY_AVAILABLE,
    AVA_GUARDIAN_AVAILABLE,
    DILITHIUM_AVAILABLE,
    KYBER_AVAILABLE,
    CryptoAnomaly,
    CryptoAnomalyType,
    DilithiumKeyPair,
    EWMATimingMonitor,
    KyberEncapsulation,
    KyberKeyPair,
    MercuryGuardianAdapter,
    TimingStats,
    create_ama_cryptography_adapter,
    create_mercury_guardian_adapter,
)

__all__ = [
    "AMA_CRYPTOGRAPHY_AVAILABLE",
    "AVA_GUARDIAN_AVAILABLE",
    "DILITHIUM_AVAILABLE",
    "KYBER_AVAILABLE",
    "CryptoAnomaly",
    "CryptoAnomalyType",
    "DilithiumKeyPair",
    "EWMATimingMonitor",
    "KyberEncapsulation",
    "KyberKeyPair",
    "MercuryGuardianAdapter",
    "TimingStats",
    "create_ama_cryptography_adapter",
    "create_mercury_guardian_adapter",
]
