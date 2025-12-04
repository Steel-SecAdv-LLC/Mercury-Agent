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

"""
Security module for OMNI ♱ AVA

Provides comprehensive security capabilities:
- Threat detection and rate limiting
- Post-quantum cryptographic protection (Kyber, MLDSA, Sphincs)
- Counterintelligence and ethical CI operations
- Cyber fortress defense (hash integrity, zero-day simulation, traffic analysis)
- Quantum risk assessment and migration planning
- Anti-terrorism pattern recognition

Includes SecurityCoordinator for flexible module selection and filtering.
"""

from typing import Any

# Cryptographic API
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

# Counterintelligence
from omni_anomaly_engine.security.counterintelligence import (
    OverwatchNexus,
    OverwatchNexusResult,
)

# Cyber Fortress
from omni_anomaly_engine.security.cyber_fortress import (
    CyberFortress,
    EncryptedTrafficAnomalyDetector,
    FortressResult,
    MultiverseZeroDaySimulator,
    ResonanceHashIntegrityChecker,
)

# Quantum Risk
from omni_anomaly_engine.security.quantum_risk_cyber import (
    CryptoSystem,
    PostQuantumMigrationPlanner,
    QuantumRiskCyber,
    QuantumThreat,
    ThreatLevel,
)

# Anti-Terrorism
from omni_anomaly_engine.security.anti_terrorism import (
    TerrorismPatternDetector,
    TerrorismThreatResult,
)


class SecurityCoordinator:
    """Coordinator for security modules with flexible selection.

    Enables filtering/selection to run 1, 2, 5, or 10+ modules simultaneously
    based on priorities, categories, or explicit names. Handles all security
    domains (crypto, threat detection, CI, cyber defense, etc.) individually
    or as a coordinated whole.

    Example:
        coordinator = SecurityCoordinator()

        # Get only high-priority modules
        high_priority = coordinator.instantiate_filtered_modules(priorities=['high'])

        # Get crypto-related modules
        crypto_modules = coordinator.instantiate_filtered_modules(categories=['crypto'])

        # Get specific modules
        specific_modules = coordinator.instantiate_filtered_modules(
            module_names=['threat_detector', 'cyber_fortress']
        )
    """

    def __init__(self):
        """Initialize module registry with all available modules."""
        self.modules = {
            # Threat Detection
            "threat_detector": {
                "class": ThreatDetector,
                "category": "threat_detection",
                "priority": "high",
                "description": "Core threat detection and analysis",
            },
            "intelligence_fusion": {
                "class": IntelligenceFusionEngine,
                "category": "intelligence",
                "priority": "high",
                "description": "Multi-source intelligence fusion (13 INT disciplines)",
            },
            # Counterintelligence
            "overwatch_nexus": {
                "class": OverwatchNexus,
                "category": "counterintelligence",
                "priority": "high",
                "description": "Proactive CI with medical interdiction and ethical safeguards",
            },
            "terrorism_detector": {
                "class": TerrorismPatternDetector,
                "category": "counterintelligence",
                "priority": "high",
                "description": "Radicalization pattern detection via OSINT/COMINT fusion",
            },
            # Cyber Defense
            "cyber_fortress": {
                "class": CyberFortress,
                "category": "cyber_defense",
                "priority": "high",
                "description": "Unified cyber defense (hash integrity, zero-day sim, traffic analysis)",
            },
            "quantum_risk": {
                "class": QuantumRiskCyber,
                "category": "cyber_defense",
                "priority": "high",
                "description": "Quantum computing threat assessment and migration planning",
            },
            # Cryptography
            "ava_guardian_crypto": {
                "class": AvaGuardianCrypto,
                "category": "crypto",
                "priority": "high",
                "description": "Post-quantum cryptographic protection (Kyber, MLDSA, Sphincs)",
            },
            "secure_data_handler": {
                "class": SecureDataHandler,
                "category": "crypto",
                "priority": "medium",
                "description": "Secure data encryption and handling utilities",
            },
            # Rate Limiting
            "rate_limiter": {
                "class": RateLimiter,
                "category": "protection",
                "priority": "medium",
                "description": "API rate limiting and abuse prevention",
            },
        }

    def get_module(self, module_name: str, **kwargs) -> Any:
        """Instantiate a specific module by name.

        Args:
            module_name: Name of module from registry
            **kwargs: Initialization arguments for the module

        Returns:
            Instantiated module
        """
        if module_name not in self.modules:
            raise ValueError(
                f"Unknown module: {module_name}. Available: {list(self.modules.keys())}"
            )

        module_class = self.modules[module_name]["class"]
        return module_class(**kwargs)

    def get_modules_by_category(self, category: str) -> list[str]:
        """Get all module names in a category.

        Args:
            category: 'threat_detection', 'intelligence', 'counterintelligence',
                      'cyber_defense', 'crypto', 'protection'

        Returns:
            List of module names in the category
        """
        return [name for name, info in self.modules.items() if info["category"] == category]

    def get_modules_by_priority(self, priority: str) -> list[str]:
        """Get all module names with a priority level.

        Args:
            priority: 'high', 'medium', 'low'

        Returns:
            List of module names with the priority
        """
        return [name for name, info in self.modules.items() if info["priority"] == priority]

    def filter_modules(
        self,
        categories: list[str] | None = None,
        priorities: list[str] | None = None,
        module_names: list[str] | None = None,
    ) -> list[str]:
        """Filter modules based on multiple criteria.

        Args:
            categories: Filter by categories
            priorities: Filter by priorities (e.g., ['high'])
            module_names: Explicit list of module names to include

        Returns:
            List of module names matching all filters
        """
        if module_names:
            return [name for name in module_names if name in self.modules]

        filtered = set(self.modules.keys())

        if categories:
            category_modules = set()
            for cat in categories:
                category_modules.update(self.get_modules_by_category(cat))
            filtered &= category_modules

        if priorities:
            priority_modules = set()
            for pri in priorities:
                priority_modules.update(self.get_modules_by_priority(pri))
            filtered &= priority_modules

        return list(filtered)

    def instantiate_filtered_modules(
        self,
        categories: list[str] | None = None,
        priorities: list[str] | None = None,
        module_names: list[str] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Instantiate all modules matching filters.

        Args:
            categories: Filter by categories
            priorities: Filter by priorities
            module_names: Explicit list of module names
            **kwargs: Initialization arguments passed to all modules

        Returns:
            Dictionary mapping module names to instantiated modules
        """
        filtered_names = self.filter_modules(categories, priorities, module_names)
        instances = {}

        for name in filtered_names:
            instances[name] = self.get_module(name, **kwargs)

        return instances

    def list_all_modules(self) -> dict[str, dict[str, str]]:
        """List all available modules with metadata.

        Returns:
            Dictionary of module metadata
        """
        return {
            name: {
                "category": info["category"],
                "priority": info["priority"],
                "description": info["description"],
            }
            for name, info in self.modules.items()
        }


__all__ = [
    # Coordinator
    "SecurityCoordinator",
    # Cryptographic API
    "AlgorithmType",
    "AvaGuardianCrypto",
    "CryptoBackend",
    "CryptoPackageConfig",
    "CryptoPackageResult",
    "DILITHIUM_AVAILABLE",
    "DilithiumKeyPair",
    "EncapsulatedSecret",
    "HybridSignature",
    "HybridSignatureProvider",
    "IntelligenceFusionEngine",
    "KYBER_AVAILABLE",
    "KeyPair",
    "KyberEncapsulation",
    "KyberKeyPair",
    "KyberProvider",
    "LIBOQS_AVAILABLE",
    "MLDSAProvider",
    "PQCBackend",
    "RateLimiter",
    "SecureDataHandler",
    "SecurityLevel",
    "SPHINCS_AVAILABLE",
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
    # Counterintelligence
    "OverwatchNexus",
    "OverwatchNexusResult",
    # Cyber Fortress
    "CyberFortress",
    "FortressResult",
    "ResonanceHashIntegrityChecker",
    "MultiverseZeroDaySimulator",
    "EncryptedTrafficAnomalyDetector",
    # Quantum Risk
    "QuantumRiskCyber",
    "QuantumThreat",
    "ThreatLevel",
    "CryptoSystem",
    "PostQuantumMigrationPlanner",
    # Anti-Terrorism
    "TerrorismPatternDetector",
    "TerrorismThreatResult",
]
