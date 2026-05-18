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
from omni_mercury_engine.security.model_policy import (
    HFModelPolicy,
    SafeHFLoader,
    UnsafeModelError,
)
from omni_mercury_engine.security.safe_exec import (
    UnsafeSubprocessError,
    safe_exec,
)
from omni_mercury_engine.security.safe_http import (
    SafeHTTPClient,
    UnsafeURLError,
)

try:
    from omni_mercury_engine.security.intelligence_fusion import IntelligenceFusionEngine
except ImportError:  # torch not installed
    IntelligenceFusionEngine = None  # type: ignore[assignment, misc]

from omni_mercury_engine.security.pqc_backends import (
    AMA_CRYPTOGRAPHY_AVAILABLE,
    DILITHIUM_AVAILABLE,
    KYBER_AVAILABLE,
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
from omni_mercury_engine.security.pqc_guards import (
    PQCProductionWarning,
    PQCSimulationWarning,
    assert_no_simulation_in_production,
    check_pqc_production_readiness,
)
from omni_mercury_engine.security.rate_limiting import RateLimiter
from omni_mercury_engine.security.safe_load import (
    DEFAULT_MAX_BYTES,
    NPZ_MAGIC,
    SIG_SUFFIX,
    UnsafePayloadError,
    safe_load_training_data,
    sign_npz,
    verify_npz_signature,
)
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
    "AMA_CRYPTOGRAPHY_AVAILABLE",
    "DEFAULT_MAX_BYTES",
    "DILITHIUM_AVAILABLE",
    "KYBER_AVAILABLE",
    "NPZ_MAGIC",
    "SIG_SUFFIX",
    "SPHINCS_AVAILABLE",
    "AlgorithmType",
    "AuditEvent",
    "AuditEventCategory",
    "AuditEventSeverity",
    "CryptoBackend",
    "CryptoPackageConfig",
    "CryptoPackageResult",
    "DilithiumKeyPair",
    "EncapsulatedSecret",
    "HFModelPolicy",
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
    "PQCProductionWarning",
    "PQCSimulationWarning",
    "RateLimiter",
    "SafeHFLoader",
    "SafeHTTPClient",
    "SecureAuditLogger",
    "SecureDataHandler",
    "SecureHashChain",
    "SecurityLevel",
    "Signature",
    "SphincsKeyPair",
    "SphincsProvider",
    "ThreatDetector",
    "UnsafeModelError",
    "UnsafePayloadError",
    "UnsafeSubprocessError",
    "UnsafeURLError",
    "assert_no_simulation_in_production",
    "check_pqc_production_readiness",
    "configure_audit_logger",
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
    "safe_exec",
    "safe_load_training_data",
    "sign_npz",
    "sphincs_sign",
    "sphincs_verify",
    "verify_npz_signature",
]
