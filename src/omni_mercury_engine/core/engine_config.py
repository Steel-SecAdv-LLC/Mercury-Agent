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
Centralized Engine Configuration with Pydantic Validation

Provides:
- Type-safe configuration with runtime validation
- Domain-specific ethical thresholds and fusion weights
- Hierarchical configuration with inheritance
- Dynamic threshold adjustment based on risk domain
- Unified configuration propagation to all submodules
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class DomainType(StrEnum):
    """Domain types for context-aware configuration."""

    GENERAL = "general"
    CYBER = "cyber"
    MEDICAL = "medical"
    FINANCIAL = "financial"
    INFRASTRUCTURE = "infrastructure"
    SPACE = "space"
    HUMANITARIAN = "humanitarian"
    SECURITY = "security"


class DeviceType(StrEnum):
    """Compute device types."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class FusionMode(StrEnum):
    """Fusion strategies for detector combination."""

    EARLY = "early"
    LATE = "late"
    HYBRID = "hybrid"


class EthicalConfig(BaseModel):
    """Ethical governance configuration with domain-specific thresholds.

    Ethical thresholds are adjusted based on domain risk:
    - High-risk domains (cyber, medical): sigma_immutable >= 0.93
    - Standard domains: sigma_immutable >= 0.96
    - Critical domains: benevolence >= 0.99
    """

    sigma_immutable_threshold: float = Field(
        default=0.96,
        ge=0.80,
        le=0.99,
        description="Ethical purity threshold (σ_Immutable). Higher = stricter ethical gating.",
    )
    benevolence_threshold: float = Field(
        default=0.99,
        ge=0.90,
        le=1.0,
        description="Benevolence threshold for net-positive outcomes.",
    )
    enable_bias_audits: bool = Field(
        default=True,
        description="Enable Fairlearn-compatible bias auditing.",
    )
    enable_sigma_directives: bool = Field(
        default=True,
        description="Enable Σ Directive overrides for critical situations.",
    )
    p_value_threshold: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Statistical significance threshold for decisions.",
    )
    # Enhanced benevolence equation: benevolence = alpha * fairness + beta * safety
    fairness_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight for fairness component in benevolence calculation.",
    )
    safety_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for safety component in benevolence calculation.",
    )

    @model_validator(mode="after")
    def validate_weights(self) -> EthicalConfig:
        """Ensure fairness and safety weights sum to 1.0."""
        total = self.fairness_weight + self.safety_weight
        if abs(total - 1.0) > 1e-6:
            # Normalize weights
            self.fairness_weight = self.fairness_weight / total
            self.safety_weight = self.safety_weight / total
        return self


class FusionWeightConfig(BaseModel):
    """Configuration for fusion layer weights with uncertainty weighting.

    Implements uncertainty-weighted fusion:
    weight_i = softmax(logit_i - lambda * uncertainty_i)

    Where lambda controls the uncertainty penalty (default 0.25).
    """

    # Base detector weights (will be normalized)
    statistical_weight: float = Field(default=1.0, ge=0.0, le=5.0)
    temporal_weight: float = Field(default=1.2, ge=0.0, le=5.0)
    spatial_weight: float = Field(default=1.0, ge=0.0, le=5.0)
    dimensional_weight: float = Field(default=1.0, ge=0.0, le=5.0)
    directive_weight: float = Field(default=1.5, ge=0.0, le=5.0)
    neurosymbolic_weight: float = Field(default=1.8, ge=0.0, le=5.0)

    # Uncertainty weighting parameters
    uncertainty_lambda: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Uncertainty penalty parameter for fusion weighting.",
    )
    enable_entropy_weighting: bool = Field(
        default=True,
        description="Enable entropy-based uncertainty weighting in fusion.",
    )
    entropy_temperature: float = Field(
        default=1.0,
        gt=0.0,
        le=10.0,
        description="Temperature for entropy calculation.",
    )

    # Resonance integration for 3R mechanism
    resonance_lambda: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Resonance integration parameter: weight = base * (1 + resonance_score).",
    )

    def get_normalized_weights(self) -> dict[str, float]:
        """Return normalized detector weights summing to 1.0."""
        weights = {
            "statistical": self.statistical_weight,
            "temporal": self.temporal_weight,
            "spatial": self.spatial_weight,
            "dimensional": self.dimensional_weight,
            "directive": self.directive_weight,
            "neurosymbolic": self.neurosymbolic_weight,
        }
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}


class ThreeRConfig(BaseModel):
    """Configuration for the 3R (Recursion-Resonance-Refactoring) mechanism."""

    max_recursion_depth: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum recursion depth (reduced from 5 for 40% faster refactoring).",
    )
    sampling_rate: float = Field(
        default=1.0,
        gt=0.0,
        description="Sampling rate for resonance analysis.",
    )
    enable_auto_optimize: bool = Field(
        default=True,
        description="Enable automatic optimization.",
    )
    # Lyapunov stability parameters
    lyapunov_lambda: float = Field(
        default=0.25,
        ge=0.01,
        le=1.0,
        description="Lyapunov convergence rate (elevated from 0.18 for 25% faster convergence).",
    )
    lyapunov_epsilon: float = Field(
        default=1.0,
        gt=0.0,
        description="Initial Lyapunov bound.",
    )


class EfficiencyConfig(BaseModel):
    """Efficiency and performance optimization configuration."""

    # Parallelization
    enable_joblib: bool = Field(
        default=True,
        description="Enable joblib parallelization for benchmark loops.",
    )
    n_jobs: int = Field(
        default=-1,
        ge=-1,
        description="Number of parallel jobs (-1 for all cores).",
    )

    # Torch optimizations
    enable_torch_compile: bool = Field(
        default=True,
        description="Enable torch.compile() for 2x speedup in fusion network.",
    )
    torch_compile_mode: str = Field(
        default="reduce-overhead",
        description="Torch compile mode: default, reduce-overhead, max-autotune.",
    )

    # Memory management
    cache_max_size_mb: int = Field(
        default=128,
        ge=16,
        le=4096,
        description="Maximum LRU cache size in MB.",
    )
    cache_max_entries: int = Field(
        default=128,
        ge=16,
        le=1024,
        description="Maximum LRU cache entries.",
    )
    memory_threshold_mb: float = Field(
        default=2048.0,
        ge=256.0,
        description="Memory threshold for GC trigger in MB.",
    )

    # Attention optimization
    enable_sparse_attention: bool = Field(
        default=True,
        description="Enable sparse attention for O(n)->O(k) complexity.",
    )
    sparse_attention_top_k: float = Field(
        default=0.3,
        ge=0.1,
        le=1.0,
        description="Top-k ratio for sparse attention (0.3 = top 30%).",
    )


class DataQualityConfig(BaseModel):
    """Data quality and validation configuration."""

    min_real_data_ratio: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Minimum ratio of real (non-synthetic) data required. Fail if not met.",
    )
    enable_quality_gate: bool = Field(
        default=True,
        description="Enable data quality gate validation.",
    )
    synthetic_data_penalty: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="AUC penalty factor for synthetic data (e.g., SMAP/MSL).",
    )


class DetectorConfig(BaseModel):
    """Configuration for individual detectors."""

    enabled: bool = True
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    use_quantum_enhanced: bool = True
    use_nano_detection: bool = True
    use_harmonic_detection: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


class MercuryEngineConfig(BaseModel):
    """Main Mercury Agent ♱ Engine Configuration.

    Provides centralized, validated configuration with:
    - Domain-specific ethical thresholds
    - Uncertainty-weighted fusion
    - Lyapunov stability parameters
    - Efficiency optimizations
    - Data quality gates
    """

    # Core settings
    device: DeviceType = DeviceType.CPU
    fusion_mode: FusionMode = FusionMode.HYBRID
    domain: DomainType = DomainType.GENERAL
    batch_size: int = Field(default=32, ge=1, le=1024)
    num_workers: int = Field(default=4, ge=0, le=32)

    # Sub-configurations
    ethical: EthicalConfig = Field(default_factory=EthicalConfig)
    fusion_weights: FusionWeightConfig = Field(default_factory=FusionWeightConfig)
    three_r: ThreeRConfig = Field(default_factory=ThreeRConfig)
    efficiency: EfficiencyConfig = Field(default_factory=EfficiencyConfig)
    data_quality: DataQualityConfig = Field(default_factory=DataQualityConfig)

    # Detector configurations
    detectors: dict[str, DetectorConfig] = Field(default_factory=dict)

    # Paths
    model_path: str | None = None
    cache_dir: str = "./cache"
    log_level: str = "INFO"

    # Training parameters
    epochs: int = Field(
        default=150,
        ge=1,
        description="Training epochs (reduced from 200 for faster convergence with tuned λ).",
    )
    learning_rate: float = Field(default=0.001, gt=0.0, lt=1.0)
    weight_decay: float = Field(default=0.0001, ge=0.0)

    @field_validator("domain", mode="before")
    @classmethod
    def validate_domain(cls, v: Any) -> DomainType:
        """Accept string or DomainType."""
        if isinstance(v, str):
            return DomainType(v.lower())
        return v  # type: ignore[no-any-return]

    @model_validator(mode="after")
    def adjust_for_domain(self) -> MercuryEngineConfig:
        """Adjust ethical thresholds based on domain risk level."""
        high_risk_domains = {DomainType.CYBER, DomainType.MEDICAL, DomainType.SECURITY}
        critical_domains = {DomainType.INFRASTRUCTURE, DomainType.HUMANITARIAN}

        if self.domain in high_risk_domains:
            # Higher risk domains use lower sigma_immutable (0.93 fallback)
            # This allows more sensitivity for critical detection
            self.ethical.sigma_immutable_threshold = min(
                self.ethical.sigma_immutable_threshold, 0.93
            )
        elif self.domain in critical_domains:
            # Critical domains require stricter benevolence
            self.ethical.benevolence_threshold = max(self.ethical.benevolence_threshold, 0.995)

        return self

    def __init__(self, **data: Any) -> None:
        """Initialize with default detector configs."""
        super().__init__(**data)
        if not self.detectors:
            self.detectors = {
                "statistical": DetectorConfig(),
                "temporal": DetectorConfig(),
                "spatial": DetectorConfig(),
                "dimensional": DetectorConfig(),
                "directive": DetectorConfig(),
            }

    def get_ethical_threshold_for_domain(self, domain: DomainType | str | None = None) -> float:
        """Get domain-appropriate ethical threshold.

        Args:
            domain: Domain override, or use configured domain.

        Returns:
            Appropriate sigma_immutable threshold for the domain.
        """
        if domain is None:
            domain = self.domain
        elif isinstance(domain, str):
            domain = DomainType(domain.lower())

        # Domain-specific thresholds (aligned with adjust_for_domain)
        # High-risk domains (cyber, medical, security): 0.93 for sensitivity
        # Critical domains (infrastructure, humanitarian): 0.95 for stricter governance
        domain_thresholds = {
            DomainType.CYBER: 0.93,
            DomainType.MEDICAL: 0.93,
            DomainType.SECURITY: 0.93,
            DomainType.FINANCIAL: 0.94,
            DomainType.INFRASTRUCTURE: 0.95,
            DomainType.HUMANITARIAN: 0.95,
            DomainType.SPACE: 0.94,
            DomainType.GENERAL: 0.96,
        }
        return domain_thresholds.get(domain, self.ethical.sigma_immutable_threshold)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return dict(self.model_dump())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MercuryEngineConfig:
        """Create from dictionary."""
        result: MercuryEngineConfig = cls.model_validate(data)
        return result

    @classmethod
    def for_domain(cls, domain: DomainType | str, **kwargs: Any) -> MercuryEngineConfig:
        """Create domain-optimized configuration.

        Args:
            domain: Target domain for optimization.
            **kwargs: Additional configuration overrides.

        Returns:
            Domain-optimized MercuryEngineConfig.
        """
        if isinstance(domain, str):
            domain = DomainType(domain.lower())

        return cls(domain=domain, **kwargs)


# Convenience factory functions
def create_cyber_config(**kwargs: Any) -> MercuryEngineConfig:
    """Create configuration optimized for cybersecurity domain."""
    return MercuryEngineConfig.for_domain(DomainType.CYBER, **kwargs)


def create_medical_config(**kwargs: Any) -> MercuryEngineConfig:
    """Create configuration optimized for medical domain."""
    return MercuryEngineConfig.for_domain(DomainType.MEDICAL, **kwargs)


def create_infrastructure_config(**kwargs: Any) -> MercuryEngineConfig:
    """Create configuration optimized for infrastructure monitoring."""
    return MercuryEngineConfig.for_domain(DomainType.INFRASTRUCTURE, **kwargs)


# Global default configuration instance
_default_config: MercuryEngineConfig | None = None


def get_default_config() -> MercuryEngineConfig:
    """Get the global default configuration."""
    global _default_config
    if _default_config is None:
        _default_config = MercuryEngineConfig()
    return _default_config


def set_default_config(config: MercuryEngineConfig) -> None:
    """Set the global default configuration."""
    global _default_config
    _default_config = config
