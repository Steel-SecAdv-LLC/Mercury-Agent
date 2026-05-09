"""
Backward-compatibility shim for the renamed spectral_domain_frequency module.

The Spectral Domain Frequency detector (formerly Spectral Domain Sound Oracle) was renamed to better
reflect its purpose. This module re-exports all public symbols so that existing imports from
``spectral_domain_oracle`` continue to work.

New code should import from ``spectral_domain_frequency`` directly.
"""

from omni_mercury_engine.detectors.spectral_domain_frequency import (
    DEFAULT_ALPHA,
    DOMAIN_ANOMALY_SPECTRAL_HINTS,
    DOMAIN_FREQUENCY_BANDS,
    EPSILON,
    PHI,
    FrequencyBandResult,
    FrequencyDomainOracle,
    FrequencyDomainOracleConfig,
    FrequencyInfluenceVector,
    FrequencyWeighting,
    SpectralDomainFrequency,
    SpectralDomainFrequencyConfig,
    SpectralDomainOracle,
    SpectralDomainOracleConfig,
    create_frequency_oracle,
    create_spectral_frequency,
    create_spectral_oracle,
    get_domain_frequency_bands,
)

__all__ = [
    "DEFAULT_ALPHA",
    "DOMAIN_ANOMALY_SPECTRAL_HINTS",
    "DOMAIN_FREQUENCY_BANDS",
    "EPSILON",
    "PHI",
    "FrequencyBandResult",
    "FrequencyDomainOracle",
    "FrequencyDomainOracleConfig",
    "FrequencyInfluenceVector",
    "FrequencyWeighting",
    "SpectralDomainFrequency",
    "SpectralDomainFrequencyConfig",
    "SpectralDomainOracle",
    "SpectralDomainOracleConfig",
    "create_frequency_oracle",
    "create_spectral_frequency",
    "create_spectral_oracle",
    "get_domain_frequency_bands",
]
