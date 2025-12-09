"""
OMNI ♱ AVA (O♱A)
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
Security module for OMNI ♱ AVA

Provides threat detection, rate limiting, encryption utilities,
and post-quantum cryptographic protection.
"""

from omni_anomaly_engine.security.crypto_api import (
    AlgorithmType,
    AvaGuardianCrypto,
    CryptoBackend,
    CryptoPackageConfig,
    CryptoPackageResult,
    EncapsulatedSecret,
    HybridSignature,
    HybridSignatureProvider,
    KeyPair,
    KyberProvider,
    MLDSAProvider,
    SecurityLevel,
    Signature,
    SphincsProvider,
)
from omni_anomaly_engine.security.encryption import SecureDataHandler
from omni_anomaly_engine.security.intelligence_fusion import IntelligenceFusionEngine
from omni_anomaly_engine.security.pqc_backends import (
    DILITHIUM_AVAILABLE,
    KYBER_AVAILABLE,
    LIBOQS_AVAILABLE,
    SPHINCS_AVAILABLE,
    DilithiumKeyPair,
    KyberEncapsulation,
    KyberKeyPair,
    PQCBackend,
    SphincsKeyPair,
    dilithium_sign,
    dilithium_verify,
    generate_dilithium_keypair,
    generate_kyber_keypair,
    generate_sphincs_keypair,
    get_active_backend,
    get_pqc_capabilities,
    kyber_decapsulate,
    kyber_encapsulate,
    sphincs_sign,
    sphincs_verify,
)
from omni_anomaly_engine.security.rate_limiting import RateLimiter
from omni_anomaly_engine.security.threat_detection import ThreatDetector

__all__ = [
    "DILITHIUM_AVAILABLE",
    "KYBER_AVAILABLE",
    "LIBOQS_AVAILABLE",
    "SPHINCS_AVAILABLE",
    "AlgorithmType",
    "AvaGuardianCrypto",
    "CryptoBackend",
    "CryptoPackageConfig",
    "CryptoPackageResult",
    "DilithiumKeyPair",
    "EncapsulatedSecret",
    "HybridSignature",
    "HybridSignatureProvider",
    "IntelligenceFusionEngine",
    "KeyPair",
    "KyberEncapsulation",
    "KyberKeyPair",
    "KyberProvider",
    "MLDSAProvider",
    "PQCBackend",
    "RateLimiter",
    "SecureDataHandler",
    "SecurityLevel",
    "Signature",
    "SphincsKeyPair",
    "SphincsProvider",
    "ThreatDetector",
    "dilithium_sign",
    "dilithium_verify",
    "generate_dilithium_keypair",
    "generate_kyber_keypair",
    "generate_sphincs_keypair",
    "get_active_backend",
    "get_pqc_capabilities",
    "kyber_decapsulate",
    "kyber_encapsulate",
    "sphincs_sign",
    "sphincs_verify",
]
