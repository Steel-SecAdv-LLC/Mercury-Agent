# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Security primitives for Mercury Agent.

This package hosts *implementation primitives* (crypto backends, PQC
key material, audit logging, safe I/O guards, rate limiting, threat
detection). Governance and policy frameworks (NIST CSF, OSHA / eCFR,
TLP) live in :mod:`omni_mercury_engine.compliance`; downstream code
should import those names from ``compliance`` rather than from
``security``.

Import contract (scoped PQC gate)
---------------------------------
Importing this package is lightweight and does **not** require AMA
Cryptography: analytic consumers (dataset loaders, detectors,
validators) reach AMA-free primitives such as ``SafeHTTPClient``,
``RateLimiter`` or the audit logger without a native PQC build.  Every
re-export resolves lazily (PEP 562), so the AMA-backed names
(``MercuryCrypto`` and the rest of ``crypto_api``, the ``pqc_backends``
key material, the ``pqc_guards`` readiness checks) fail closed at first
*access* when AMA is missing — raising the scoped production gate's
refusal (see :mod:`omni_mercury_engine._pqc_gate`) rather than a cryptic
``ModuleNotFoundError``.  The contract is pinned by
``tests/test_pqc_startup_gate.py::TestScopedGateContract``.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
    from omni_mercury_engine.security.intelligence_fusion import (
        IntelligenceFusionEngine as IntelligenceFusionEngine,
    )
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

#: Which submodule defines each lazily resolved re-export.  AMA-backed
#: submodules (``crypto_api``, ``pqc_backends``, ``pqc_guards``) refuse at
#: import with the scoped PQC gate's message when AMA is missing; the rest
#: import AMA-free (``encryption``'s AES/HMAC AMA calls are function-level).
_EXPORTS_BY_MODULE: dict[str, tuple[str, ...]] = {
    "crypto_api": (
        "AlgorithmType",
        "CryptoBackend",
        "CryptoPackageConfig",
        "CryptoPackageResult",
        "EncapsulatedSecret",
        "HybridSignature",
        "HybridSignatureProvider",
        "KeyPair",
        "KyberProvider",
        "MLDSAProvider",
        "MercuryCrypto",
        "SecurityLevel",
        "Signature",
        "SphincsProvider",
    ),
    "encryption": ("SecureDataHandler",),
    "model_policy": ("HFModelPolicy", "SafeHFLoader", "UnsafeModelError"),
    "pqc_backends": (
        "AMA_CRYPTOGRAPHY_AVAILABLE",
        "DILITHIUM_AVAILABLE",
        "KYBER_AVAILABLE",
        "SPHINCS_AVAILABLE",
        "DilithiumKeyPair",
        "KyberEncapsulation",
        "KyberKeyPair",
        "PQCBackend",
        "SphincsKeyPair",
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
    ),
    "pqc_guards": (
        "PQCProductionWarning",
        "PQCSimulationWarning",
        "assert_no_simulation_in_production",
        "check_pqc_production_readiness",
    ),
    "rate_limiting": ("RateLimiter",),
    "safe_exec": ("UnsafeSubprocessError", "safe_exec"),
    "safe_http": ("SafeHTTPClient", "UnsafeURLError"),
    "safe_load": (
        "DEFAULT_MAX_BYTES",
        "NPZ_MAGIC",
        "SIG_SUFFIX",
        "UnsafePayloadError",
        "safe_load_training_data",
        "sign_npz",
        "verify_npz_signature",
    ),
    "secure_audit_logging": (
        "AuditEvent",
        "AuditEventCategory",
        "AuditEventSeverity",
        "PIIMasker",
        "SecureAuditLogger",
        "SecureHashChain",
        "configure_audit_logger",
        "get_audit_logger",
    ),
    "threat_detection": ("ThreatDetector",),
}

_LAZY_EXPORTS: dict[str, str] = {
    name: module for module, names in _EXPORTS_BY_MODULE.items() for name in names
}


def __getattr__(name: str) -> object:
    """Lazy (PEP 562) resolution of every re-export in ``__all__``.

    Two independent reasons the package resolves its surface lazily:

    * ``IntelligenceFusionEngine`` is the package's sole torch dependency.
      Importing it eagerly dragged the ~2 GB ML stack into every consumer
      of the lightweight primitives.  Resolution preserves the historical
      contract exactly: the class when torch is importable, ``None`` when
      it is not.
    * The AMA-backed names (``crypto_api`` / ``pqc_backends`` /
      ``pqc_guards``) would otherwise make *package import* require a
      native AMA build, blocking the AMA-free analytic quickstart path.
      Deferring them to first access keeps the scoped PQC gate's
      guarantee — crypto surfaces fail closed, analytics import freely.

    Resolved names are cached in ``globals()`` so later look-ups bypass
    this hook; a submodule name in :data:`_EXPORTS_BY_MODULE` resolves to
    the submodule itself, matching the old eager-import behaviour of
    ``security.crypto_api``-style attribute access.
    """
    if name == "IntelligenceFusionEngine":
        try:
            from omni_mercury_engine.security.intelligence_fusion import (
                IntelligenceFusionEngine as engine_cls,
            )
        except ModuleNotFoundError as exc:
            # Only an absent torch means "fusion unavailable" (the historical
            # ``None``). Any other import failure -- scipy missing, a broken
            # submodule -- is a real fault and must not masquerade as it.
            if (exc.name or "").split(".")[0] != "torch":
                raise
            return None
        return engine_cls
    if name in _EXPORTS_BY_MODULE:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value: object = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Advertise the lazy surface alongside the eagerly defined names."""
    return sorted(set(globals()) | set(__all__))
