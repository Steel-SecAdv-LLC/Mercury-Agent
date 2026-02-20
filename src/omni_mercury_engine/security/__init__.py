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
Security module for Mercury Agent

Provides threat detection, rate limiting, encryption utilities,
and post-quantum cryptographic protection.
"""

from omni_mercury_engine.security.crypto_api import (
    AlgorithmType,
    CryptoBackend,
    CryptoPackageConfig,
    CryptoPackageResult,
    EncapsulatedSecret,
    HybridSignature,
    HybridSignatureProvider,
    KeyPair,
    KyberProvider,
    MercuryCrypto,
    MLDSAProvider,
    SecurityLevel,
    Signature,
    SphincsProvider,
)
from omni_mercury_engine.security.encryption import SecureDataHandler

try:
    from omni_mercury_engine.security.intelligence_fusion import IntelligenceFusionEngine
except ImportError:  # torch not installed
    IntelligenceFusionEngine = None  # type: ignore[assignment, misc]

from omni_mercury_engine.security.pqc_backends import (
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
from omni_mercury_engine.security.rate_limiting import RateLimiter
from omni_mercury_engine.security.secure_audit_logging import (
    AuditEvent,
    AuditEventCategory,
    AuditEventSeverity,
    PIIMasker,
    SecureAuditLogger,
    SecureHashChain,
    configure_audit_logger,
    get_audit_logger,
)
from omni_mercury_engine.security.threat_detection import ThreatDetector

__all__ = [
    "DILITHIUM_AVAILABLE",
    "KYBER_AVAILABLE",
    "LIBOQS_AVAILABLE",
    "SPHINCS_AVAILABLE",
    "AlgorithmType",
    # Secure Audit Logging
    "AuditEvent",
    "AuditEventCategory",
    "AuditEventSeverity",
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
    "MercuryCrypto",
    "PIIMasker",
    "PQCBackend",
    "RateLimiter",
    "SecureAuditLogger",
    "SecureDataHandler",
    "SecureHashChain",
    "SecurityLevel",
    "Signature",
    "SphincsKeyPair",
    "SphincsProvider",
    "ThreatDetector",
    "configure_audit_logger",
    # PQC functions
    "dilithium_sign",
    "dilithium_verify",
    "generate_dilithium_keypair",
    "generate_kyber_keypair",
    "generate_sphincs_keypair",
    "get_active_backend",
    "get_audit_logger",
    "get_pqc_capabilities",
    "kyber_decapsulate",
    "kyber_encapsulate",
    "sphincs_sign",
    "sphincs_verify",
]
