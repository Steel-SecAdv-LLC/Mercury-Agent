# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Security primitives for Mercury Agent.

This package hosts *implementation primitives* (crypto backends, PQC
key material, audit logging, safe I/O guards, rate limiting, threat
detection). Governance and policy frameworks (NIST CSF, OSHA / eCFR,
TLP) live in :mod:`omni_mercury_engine.compliance`; downstream code
should import those names from ``compliance`` rather than from
``security``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni_mercury_engine.security.intelligence_fusion import (
        IntelligenceFusionEngine as IntelligenceFusionEngine,
    )

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
from omni_mercury_engine.security.safe_exec import (
    UnsafeSubprocessError,
    safe_exec,
)
from omni_mercury_engine.security.safe_http import (
    SafeHTTPClient,
    UnsafeURLError,
)
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


def __getattr__(name: str) -> object:
    """Lazy (PEP 562) resolution for the torch-backed intelligence engine.

    ``IntelligenceFusionEngine`` is the security package's sole torch
    dependency.  Importing it eagerly dragged the ~2 GB ML stack into
    every consumer of the lightweight primitives (``SafeHTTPClient``,
    ``SafeHFLoader``, audit logging, PQC) — including the pure-``requests``
    cloud LLM adapters, whose only security need is ``model_policy``.
    Resolution preserves the historical contract exactly: the class when
    torch is importable, ``None`` when it is not.
    """
    if name == "IntelligenceFusionEngine":
        try:
            from omni_mercury_engine.security.intelligence_fusion import (
                IntelligenceFusionEngine as engine_cls,
            )
        except ImportError:  # torch not installed
            return None
        return engine_cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
