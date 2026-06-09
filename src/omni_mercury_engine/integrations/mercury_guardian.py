# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Backward Compatibility Shim — mercury_guardian → mercury_amacrypto.

This module re-exports all public names from mercury_amacrypto.
New code should import directly from
``omni_mercury_engine.integrations.mercury_amacrypto``.
"""

from __future__ import annotations

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
