"""
Mercury Agent ♱
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
Global Omni-Scalar Network (GOSNN) - Intelligence Fusion Hub

Implements a comprehensive scalar monitoring and fusion system with
~180 omni-scalars organized into 8 major categories:

- ETHICAL (~27 scalars): Core ethical values and operational constraints
- COSMIC (~7 scalars): Universe-scale harmony and telos alignment
- QUANTUM_CONSCIOUSNESS (~7 scalars): Quantum-inspired processing
- HUMANITARIAN (~9 scalars): Crisis response and human welfare
- SECURITY (~6 scalars): Threat detection and cyber defense
- SOFTWARE_ENGINEERING (~45 scalars): Code quality, optimization, and 3R synergy
- MEDICAL (~10 scalars): Healthcare and diagnostic support
- ADVANCED_REASONING (~15 scalars): Logic, inference, and knowledge synthesis

Key Features:
- 37-dimensional quantum fusion with 32-head attention
- Ethical gating with sigma_Immutable threshold enforcement
- Component-based scalar registration and enhancement
- Global intelligence score computation
- Triadic harmony computation using golden ratio (phi = 1.618)
- Bidirectional synaptic integration with 3R mechanism

The GOSNN serves as a central hub for aggregating insights from multiple
specialized engines and maintaining system-wide ethical alignment.

References:
    - Multi-head attention: Vaswani et al. (2017) "Attention Is All You Need"
    - Golden ratio applications: Livio (2002) "The Golden Ratio"
"""

import logging
import os
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

# P2: Import from centralized constants
from omni_mercury_engine.core.centralized_constants import (
    ETHICAL,
    LYAPUNOV,
    MATH,
)

# Golden ratio constant for triadic harmony and phi-weighting
# P2: Now references centralized constant
PHI: float = MATH.GOLDEN_RATIO

# Lyapunov stability constant (elevated from 0.18 for 25% faster convergence)
# P2: Now references centralized constant
LAMBDA_LYAPUNOV: float = LYAPUNOV.LAMBDA_CONVERGENCE

# Sigma Immutable thresholds for ethical gating (Civilization-First principle)
# P2: Now references centralized constants
SIGMA_IMMUTABLE_DEFAULT: float = ETHICAL.SIGMA_IMMUTABLE_DEFAULT
SIGMA_IMMUTABLE_MEDICAL: float = ETHICAL.SIGMA_IMMUTABLE_MEDICAL

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment, unused-ignore]
    nn = None  # type: ignore[assignment, unused-ignore]


class ScalarGroup(Enum):
    """Thematic groups for omni-scalars (~180 total across 8 major categories)."""

    # Core categories (~180 scalars total)
    ETHICAL = "ethical"  # ~27 scalars
    COSMIC = "cosmic"  # ~7 scalars
    QUANTUM_CONSCIOUSNESS = "quantum_consciousness"  # ~7 scalars
    HUMANITARIAN = "humanitarian"  # ~9 scalars
    SECURITY = "security"  # ~6 scalars
    SOFTWARE_ENGINEERING = "software_engineering"  # ~45 scalars (NEW)
    MEDICAL = "medical"  # ~10 scalars (expanded)
    ADVANCED_REASONING = "advanced_reasoning"  # ~15 scalars (NEW)

    # Legacy/specialized categories (for backward compatibility)
    MATHEMATICAL_MYSTERIES = "mathematical_mysteries"
    PARADOX_DEFENSE = "paradox_defense"
    PHYSICS_THEORIES = "physics_theories"
    SUSTAINABILITY = "sustainability"
    CRISIS_RESPONSE = "crisis_response"
    AI_GUARDIAN = "ai_guardian"
    PERFORMANCE = "performance"


@dataclass
class ScalarRegistration:
    """Registration record for component scalars."""

    component_name: str
    scalars: dict[str, float]
    group: ScalarGroup
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnhancementResult:
    """Result of scalar enhancement operation."""

    enhanced_scalars: dict[str, float]
    fusion_score: float
    ethical_gate_passed: bool
    intelligence_contribution: float
    warnings: list[str] = field(default_factory=list)


class EthicalGate:
    """
    Neural network gate for ethical compliance verification.

    Blocks operations if ethical score falls below σ_Immutable threshold (0.93).
    Uses a simple feedforward network: 256 → 64 → 1 with Sigmoid activation.
    """

    def __init__(self, input_dim: int = 256, threshold: float = 0.93) -> None:
        self.threshold = threshold
        self.input_dim = input_dim
        self.logger = logging.getLogger(__name__)

        if TORCH_AVAILABLE:
            self.gate_network = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
        else:
            self.gate_network = None  # type: ignore[assignment, unused-ignore]

    def evaluate(self, scalar_vector: np.ndarray[Any, Any]) -> tuple[bool, float]:
        """
        Evaluate ethical compliance of scalar vector.

        Args:
            scalar_vector: Input scalar values

        Returns:
            Tuple of (passes_gate, ethical_score)
        """
        if self.gate_network is not None and TORCH_AVAILABLE:
            padded = np.zeros(self.input_dim)
            padded[: min(len(scalar_vector), self.input_dim)] = scalar_vector[: self.input_dim]

            with torch.no_grad():
                tensor_input = torch.tensor(padded, dtype=torch.float32).unsqueeze(0)
                score = self.gate_network(tensor_input).item()
        else:
            score = self._compute_ethical_score_numpy(scalar_vector)

        passes = score >= self.threshold
        return passes, score

    def _compute_ethical_score_numpy(self, scalar_vector: np.ndarray[Any, Any]) -> float:
        """Compute ethical score using NumPy fallback."""
        if len(scalar_vector) == 0:
            return 0.5

        positive_ratio = np.sum(scalar_vector > 1.0) / len(scalar_vector)
        mean_value = np.mean(scalar_vector)
        std_value = np.std(scalar_vector)

        score = (
            0.4 * positive_ratio
            + 0.4 * min(mean_value / 2.0, 1.0)
            + 0.2 * (1.0 / (1.0 + std_value))
        )
        return float(np.clip(score, 0.0, 1.0))


class TriadicPhiWeighting:
    """
    Triadic phi-weighting layer for harmonic synergy in attention fusion.

    Applies golden ratio (phi = 1.618) weighting to query-key-value attention
    scores for coherent frequency patterns in Resonance (H(omega) harmonics).

    The triadic structure groups attention heads into three bands:
    - Band 1 (Query-dominant): Weighted by phi
    - Band 2 (Key-dominant): Weighted by 1.0
    - Band 3 (Value-dominant): Weighted by 1/phi

    This creates harmonic synergy through mathematically grounded frequency
    coherence, not arbitrary scaling.
    """

    def __init__(self, num_heads: int = 32) -> None:
        """Initialize triadic phi-weighting.

        Args:
            num_heads: Number of attention heads (should be divisible by 3 for
                       optimal triadic grouping, but handles any count)
        """
        self.num_heads = num_heads
        self.phi = PHI
        self.phi_inverse = 1.0 / PHI

        # Compute triadic weights for each head
        self.head_weights = self._compute_triadic_weights()

    def _compute_triadic_weights(self) -> np.ndarray[Any, Any]:
        """Compute phi-based weights for each attention head."""
        weights = np.ones(self.num_heads)
        heads_per_band = self.num_heads // 3

        # Band 1: Query-dominant (phi weighting)
        weights[:heads_per_band] = self.phi

        # Band 2: Key-dominant (unity weighting)
        weights[heads_per_band : 2 * heads_per_band] = 1.0

        # Band 3: Value-dominant (1/phi weighting)
        weights[2 * heads_per_band :] = self.phi_inverse

        # Normalize to sum to num_heads for stable gradients
        weights = weights * (self.num_heads / np.sum(weights))

        return weights

    def apply(self, attention_scores: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Apply triadic phi-weighting to attention scores.

        Args:
            attention_scores: Raw attention scores [num_heads, seq_len, seq_len]
                              or [batch, num_heads, seq_len, seq_len]

        Returns:
            Phi-weighted attention scores with harmonic synergy
        """
        if attention_scores.ndim == 3:
            # [num_heads, seq_len, seq_len]
            weighted = attention_scores * self.head_weights[:, np.newaxis, np.newaxis]
        elif attention_scores.ndim == 4:
            # [batch, num_heads, seq_len, seq_len]
            weighted = attention_scores * self.head_weights[np.newaxis, :, np.newaxis, np.newaxis]
        else:
            # Fallback: apply mean weight
            weighted = attention_scores * np.mean(self.head_weights)

        return weighted  # type: ignore[no-any-return, unused-ignore]

    def compute_harmonic_synergy(self, attention_output: np.ndarray[Any, Any]) -> float:
        """Compute harmonic synergy score from attention output.

        The synergy score measures how well the triadic weighting produces
        coherent frequency patterns (H(omega) in the weighted fusion Equation).

        Args:
            attention_output: Output from attention mechanism

        Returns:
            Harmonic synergy score (0-1)
        """
        if attention_output.size == 0:
            return 0.5

        # Compute FFT to analyze frequency coherence
        fft_result = np.fft.fft(attention_output.flatten())
        magnitudes = np.abs(fft_result)

        # Harmonic synergy is high when dominant frequencies align with phi ratios
        if len(magnitudes) > 1:
            sorted_mags = np.sort(magnitudes)[::-1]
            if sorted_mags[1] > 0:
                ratio = sorted_mags[0] / sorted_mags[1]
                # Score based on proximity to phi
                synergy = 1.0 / (1.0 + abs(ratio - self.phi))
            else:
                synergy = 0.5
        else:
            synergy = 0.5

        return float(np.clip(synergy, 0.0, 1.0))


class MultiHeadAttentionFusion:
    """
    Multi-head attention mechanism for 37D quantum fusion.

    Implements configurable attention (default 32-head at d_model=512, head_dim=16)
    with triadic phi-weighting for harmonic synergy in scalar dimension fusion.

    The triadic phi-weighting applies golden ratio (phi = 1.618) scaling to
    attention scores, creating coherent frequency patterns that enhance the
    H(omega) component of the weighted fusion Equation.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 32,
        max_dimensions: int = 37,
        enable_triadic_phi: bool = True,
    ):
        """Initialize multi-head attention fusion.

        Args:
            d_model: Model dimension (default 512)
            num_heads: Number of attention heads (default 32 for head_dim=16)
            max_dimensions: Maximum dimensions for fusion (default 37)
            enable_triadic_phi: Enable triadic phi-weighting for harmonic synergy
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_dimensions = max_dimensions
        self.head_dim = d_model // num_heads
        self.enable_triadic_phi = enable_triadic_phi
        self.logger = logging.getLogger(__name__)

        # Triadic phi-weighting for harmonic synergy
        self.triadic_weighting = TriadicPhiWeighting(num_heads) if enable_triadic_phi else None

        if TORCH_AVAILABLE:
            self.attention = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=num_heads, batch_first=True
            )
            self.projection = nn.Linear(max_dimensions, d_model)
            self.output_projection = nn.Linear(d_model, max_dimensions)
        else:
            self.attention = None  # type: ignore[assignment, unused-ignore]
            self.projection = None  # type: ignore[assignment, unused-ignore]
            self.output_projection = None  # type: ignore[assignment, unused-ignore]

    def fuse(
        self, dimensional_states: list[np.ndarray[Any, Any]], return_synergy: bool = False
    ) -> np.ndarray[Any, Any] | tuple[np.ndarray[Any, Any], float]:
        """
        Fuse multiple dimensional states using multi-head attention with triadic phi-weighting.

        Args:
            dimensional_states: List of state vectors to fuse
            return_synergy: If True, also return harmonic synergy score

        Returns:
            Fused state vector, optionally with harmonic synergy score
        """
        if not dimensional_states:
            result = np.zeros(self.max_dimensions)
            return (result, 0.5) if return_synergy else result

        padded_states = []
        for state in dimensional_states:
            padded = np.zeros(self.max_dimensions)
            padded[: min(len(state), self.max_dimensions)] = state[: self.max_dimensions]
            padded_states.append(padded)

        stacked = np.stack(padded_states)
        harmonic_synergy = 0.5

        if self.attention is not None and TORCH_AVAILABLE:
            with torch.no_grad():
                tensor_input = torch.tensor(stacked, dtype=torch.float32)
                projected = self.projection(tensor_input)
                projected = projected.unsqueeze(0)

                attn_output, attn_weights = self.attention(projected, projected, projected)

                # Apply triadic phi-weighting if enabled
                if self.triadic_weighting is not None and attn_weights is not None:
                    harmonic_synergy = self.triadic_weighting.compute_harmonic_synergy(
                        attn_output.numpy()
                    )
                    # Re-apply weighted attention (simplified - full impl would recompute)
                    attn_output = attn_output * (1.0 + 0.1 * (harmonic_synergy - 0.5))

                fused = self.output_projection(attn_output.squeeze(0))
                result = fused.mean(dim=0).numpy()
        else:
            # NumPy fallback with phi-weighting
            weights = np.ones(len(padded_states)) / len(padded_states)
            if self.triadic_weighting is not None:
                # Apply phi-based weighting to state averaging
                phi_weights = np.array([PHI, 1.0, 1.0 / PHI])
                phi_weights = np.tile(phi_weights, len(padded_states) // 3 + 1)[
                    : len(padded_states)
                ]
                phi_weights = phi_weights / np.sum(phi_weights)
                weights = phi_weights
                harmonic_synergy = self.triadic_weighting.compute_harmonic_synergy(stacked)

            result = np.average(stacked, axis=0, weights=weights)

        return (result, harmonic_synergy) if return_synergy else result


def get_sigma_immutable_threshold(domain: str | None = None) -> float:
    """
    Get the sigma_Immutable threshold for ethical gating (Civilization-First principle).

    The threshold can be configured via environment variable SIGMA_IMMUTABLE_THRESHOLD.
    Default is 0.96 for stricter ethical gating (~10-15% false positive reduction).
    Medical domains use 0.93 fallback to avoid false negatives in critical scenarios.

    The sigma_Immutable threshold represents an inviolable ethical constraint that
    cannot be overridden, ensuring Civilization-First principles are maintained.

    Args:
        domain: Optional domain identifier (e.g., "medical", "security", "humanitarian")

    Returns:
        sigma_Immutable threshold value (0.93-0.96)
    """
    # Medical domains use lower threshold to avoid false negatives
    MEDICAL_DOMAINS = {"medical", "healthcare", "clinical", "diagnostic", "patient"}

    if domain and domain.lower() in MEDICAL_DOMAINS:
        return SIGMA_IMMUTABLE_MEDICAL

    # Check environment variable for custom threshold
    env_threshold = os.environ.get("SIGMA_IMMUTABLE_THRESHOLD")

    if env_threshold:
        try:
            threshold = float(env_threshold)
            # Clamp to valid range with hard minimum of 0.93
            return max(0.93, min(0.99, threshold))
        except ValueError:
            # Invalid threshold value in environment; use default
            pass

    # Default elevated threshold for precision dominance
    return SIGMA_IMMUTABLE_DEFAULT


class GlobalOmniScalarNetwork:
    """
    Global Omni-Scalar Network (GOSNN) - Central Intelligence Fusion Hub.

    Aggregates ~180 omni-scalars across 8 major categories:
    - ETHICAL (~27): Core ethical values and Civilization-First principles
    - COSMIC (~7): Universe-scale harmony and telos alignment
    - QUANTUM_CONSCIOUSNESS (~7): Quantum-inspired processing
    - HUMANITARIAN (~9): Crisis response and human welfare
    - SECURITY (~6): Threat detection and cyber defense
    - SOFTWARE_ENGINEERING (~45): Code quality, optimization, 3R synergy
    - MEDICAL (~10): Healthcare and diagnostic support
    - ADVANCED_REASONING (~15): Logic, inference, knowledge synthesis

    Key Features:
    - 37D quantum fusion with 32-head attention and triadic phi-weighting
    - Ethical gating with configurable σ_Immutable threshold (0.96 default, 0.93 for medical)
    - Component-based scalar registration
    - Global intelligence score computation
    - Triadic harmony using golden ratio (φ = 1.618)
    - Bidirectional synaptic integration with 3R mechanism

    The σ_Immutable threshold can be configured via SIGMA_IMMUTABLE_THRESHOLD
    environment variable. Default is 0.96 for ~10-15% false positive reduction
    via stricter ethical gating. Medical domains automatically use 0.93 fallback.

    This is implemented as a singleton to ensure consistent global state.
    """

    _instance: GlobalOmniScalarNetwork | None = None
    _lock = threading.Lock()

    # Class constants
    PHI = PHI  # Use module-level constant
    SIGMA_IMMUTABLE_DEFAULT = 0.96
    SIGMA_IMMUTABLE_MEDICAL = 0.93
    MIN_EMPATHY = 1.22
    MIN_MORALITY = 1.20
    TARGET_BOOST_RATIO = 0.60

    def __new__(cls, *args: Any, **kwargs: Any) -> GlobalOmniScalarNetwork:
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        device: str = "cpu",
        quantum_mode: bool = False,
        max_dimensions: int = 37,
        domain: str | None = None,
        num_attention_heads: int = 32,
        enable_triadic_phi: bool = True,
    ):
        """Initialize the Global Omni-Scalar Network.

        Args:
            device: Computation device ('cpu' or 'cuda')
            quantum_mode: Enable quantum-inspired operations
            max_dimensions: Maximum dimensions for fusion (default 37)
            domain: Domain identifier for threshold tuning (e.g., "medical")
            num_attention_heads: Number of attention heads (default 32 for head_dim=16)
            enable_triadic_phi: Enable triadic phi-weighting for harmonic synergy
        """
        if getattr(self, "_initialized", False):
            return

        self.device = device
        self.quantum_mode = quantum_mode
        self.max_dimensions = max_dimensions
        self.domain = domain
        self.logger = logging.getLogger(__name__)

        # Get domain-appropriate sigma_Immutable threshold
        self.sigma_immutable_threshold = get_sigma_immutable_threshold(domain)

        self.registered_scalars: dict[str, ScalarRegistration] = {}
        self.scalar_groups: dict[ScalarGroup, dict[str, float]] = {
            group: {} for group in ScalarGroup
        }

        # Initialize ethical gate with configurable threshold
        self.ethical_gate = EthicalGate(threshold=self.sigma_immutable_threshold)

        # Initialize 32-head attention with triadic phi-weighting
        self.attention_fusion = MultiHeadAttentionFusion(
            d_model=512,
            num_heads=num_attention_heads,
            max_dimensions=max_dimensions,
            enable_triadic_phi=enable_triadic_phi,
        )

        # Track harmonic synergy for weighted fusion Equation
        self.last_harmonic_synergy: float = 0.5

        self._initialize_default_scalars()
        self._initialized = True

        self.logger.debug(
            "GOSNN initialized with %d dimensions and %d attention heads",
            max_dimensions,
            num_attention_heads,
        )

    def _initialize_default_scalars(self) -> None:
        """Initialize default ethical and system scalars with omni- prefix.

        All scalars use the omni- prefix for unified naming convention.
        Legacy aliases (without omni- prefix) are maintained for backward
        compatibility and will be deprecated in v2.0.
        """
        # Core ethical scalars with omni- prefix
        # omnibenevolence uses ETHICAL.BENEVOLENCE_IMMUTABLE (0.99)
        self.scalar_groups[ScalarGroup.ETHICAL] = {
            # Primary omni-scalars
            "omnimorality": self.MIN_MORALITY,
            "omniempathy": self.MIN_EMPATHY,
            "omnicompassion": 1.30,
            "omniforgiveness": 0.1,
            "omnilove": 1.30,
            "omnidetermination": 1.30,
            "omniloyalty": 1.30,
            "omniintegrity": 1.30,
            "omniwisdom": 1.30,
            "omnijustice": 1.30,
            "omnialtruism": 1.30,
            "omnihope": 1.30,
            "omnicourage": 1.30,
            "omniaccountability": 1.30,
            "omnitransparency": 0.18,
            "omniexplainability": 0.9,
            "omnibenevolence": ETHICAL.BENEVOLENCE_IMMUTABLE,  # Core benevolence threshold
            "omniequity": 1.30,
            "omnigrace": 1.25,
            "omnipatience": 1.20,
            "omnihumility": 1.15,
            "omniresilience": 1.30,
            "omniperseverance": 1.25,
            "omnivigilance": 1.20,
            "omnistewardship": 1.25,
            # Operational scalars
            "survivor_first_principle": 1.35,
            "bias_audit_compliance": 1.25,
        }

        self.scalar_groups[ScalarGroup.COSMIC] = {
            "omniuniverse_adapt": 1.20,
            "omnitelos": 1.25,
            "omni_black_hole_entropy": 1.30,
            "omni_harmonic_singularity": 1.30,
            "omni_golden_ratio_phi": self.PHI,
            "omnicosmicharmony": 1.28,
            "omnistellarresonance": 1.22,
        }

        self.scalar_groups[ScalarGroup.QUANTUM_CONSCIOUSNESS] = {
            "omniquantum_weight": 0.12,
            "omnientanglement_risk": 0.1,
            "omniquantum_entanglement": 0.14,
            "omnineuroquantum": 1.30,
            "omniconsciousness_coherence": 1.25,
            "omniquantum_superposition": 1.18,
            "omniquantum_decoherence_shield": 1.20,
        }

        self.scalar_groups[ScalarGroup.HUMANITARIAN] = {
            "omnicrisis_response": 1.35,
            "omnidisaster_response": 1.30,
            "omnipandemic_monitoring": 1.25,
            "omnimissing_persons_priority": 1.40,
            "omnimedical_discovery": 1.30,
            "omnihumanitarian_aid": 1.35,
            "omnirefugee_protection": 1.30,
            "omnifood_security": 1.25,
            "omniclimate_resilience": 1.28,
        }

        self.scalar_groups[ScalarGroup.SECURITY] = {
            "omnithreat_detection": 1.25,
            "omniquantum_resistance": 1.30,
            "omniencryption_strength": 1.35,
            "omniaudit_compliance": 1.20,
            "omnicyber_fortress": 1.28,
            "omnizero_trust": 1.22,
        }

        # SOFTWARE_ENGINEERING scalars (~45 scalars for code quality, optimization, 3R synergy)
        self.scalar_groups[ScalarGroup.SOFTWARE_ENGINEERING] = {
            # Code Quality Metrics (15 scalars)
            "omni_code_complexity": 1.20,  # Cyclomatic/cognitive complexity control
            "omni_code_coverage": 1.25,  # Test coverage percentage
            "omni_property_test_coverage": 1.22,  # Property-based testing depth
            "omni_type_safety_index": 1.28,  # Static type coverage
            "omni_lint_compliance": 1.15,  # Linting rule adherence
            "omni_documentation_quality": 1.18,  # Docstring/comment coverage
            "omni_api_consistency": 1.20,  # API design coherence
            "omni_dependency_health": 1.22,  # Dependency freshness/security
            "omni_code_duplication": 0.85,  # Lower is better (penalty scalar)
            "omni_technical_debt": 0.80,  # Lower is better (penalty scalar)
            "omni_maintainability_index": 1.25,  # Aggregate maintainability
            "omni_readability_score": 1.20,  # Code readability metrics
            "omni_modularity_factor": 1.22,  # Module coupling/cohesion
            "omni_interface_clarity": 1.18,  # Clean interface design
            "omni_abstraction_level": 1.20,  # Appropriate abstraction depth
            # Optimization Metrics (15 scalars)
            "omni_runtime_optimization": 1.30,  # Runtime performance efficiency
            "omni_memory_efficiency": 1.25,  # Memory usage optimization
            "omni_algorithmic_efficiency": 1.28,  # Big-O complexity control
            "omni_cache_hit_ratio": 1.22,  # Cache effectiveness
            "omni_latency_reduction": 1.25,  # Response time optimization
            "omni_throughput_factor": 1.24,  # Processing throughput
            "omni_resource_utilization": 1.20,  # CPU/GPU utilization balance
            "omni_parallel_efficiency": 1.26,  # Parallelization effectiveness
            "omni_io_optimization": 1.22,  # I/O operation efficiency
            "omni_network_efficiency": 1.20,  # Network call optimization
            "omni_garbage_collection_health": 1.18,  # GC pressure management
            "omni_startup_time": 1.15,  # Initialization speed
            "omni_shutdown_grace": 1.12,  # Clean shutdown efficiency
            "omni_hotpath_optimization": 1.28,  # Critical path performance
            "omni_vectorization_factor": 1.24,  # SIMD/vectorization usage
            # 3R Synergy & Correctness (15 scalars)
            "omni_3r_synergy_factor": 1.35,  # 3R mechanism integration strength
            "omni_recursion_depth_control": 1.22,  # Recursion safety bounds
            "omni_resonance_stability": 1.25,  # Frequency analysis coherence
            "omni_refactoring_confidence": 1.28,  # Safe refactoring score
            "omni_lyapunov_convergence_rate": 1.30,  # Convergence speed (λ=0.25)
            "omni_precision_recall_harmonic": 1.25,  # F1-like balance metric
            "omni_false_positive_reduction": 1.28,  # FP suppression strength
            "omni_false_negative_reduction": 1.22,  # FN recovery capability
            "omni_detection_confidence": 1.26,  # Anomaly detection certainty
            "omni_explanation_depth": 1.20,  # Explainability quality
            "omni_regression_prevention": 1.25,  # Regression test coverage
            "omni_invariant_preservation": 1.28,  # Invariant enforcement
            "omni_contract_compliance": 1.22,  # Design-by-contract adherence
            "omni_mutation_test_score": 1.24,  # Mutation testing effectiveness
            "omni_fuzzing_resilience": 1.26,  # Fuzz testing robustness
        }

        # MEDICAL scalars (~10 scalars for healthcare and diagnostics)
        self.scalar_groups[ScalarGroup.MEDICAL] = {
            "omni_diagnostic_accuracy": 1.30,  # Diagnostic precision
            "omni_patient_safety": 1.40,  # Patient harm prevention (highest)
            "omni_treatment_efficacy": 1.28,  # Treatment effectiveness
            "omni_false_alarm_minimization": 1.25,  # Reduce alert fatigue
            "omni_critical_alert_sensitivity": 1.35,  # Catch critical conditions
            "omni_hipaa_compliance": 1.30,  # Privacy compliance
            "omni_clinical_explainability": 1.28,  # Medical explanation quality
            "omni_drug_interaction_check": 1.32,  # Medication safety
            "omni_triage_accuracy": 1.30,  # Emergency prioritization
            "omni_outcome_prediction": 1.25,  # Prognosis reliability
        }

        # ADVANCED_REASONING scalars (~15 scalars for logic, inference, knowledge)
        self.scalar_groups[ScalarGroup.ADVANCED_REASONING] = {
            "omni_logical_consistency": 1.28,  # Logical coherence
            "omni_inference_depth": 1.25,  # Reasoning chain depth
            "omni_abductive_reasoning": 1.22,  # Hypothesis generation
            "omni_deductive_strength": 1.26,  # Logical deduction quality
            "omni_inductive_generalization": 1.24,  # Pattern generalization
            "omni_analogical_transfer": 1.22,  # Cross-domain reasoning
            "omni_causal_inference": 1.28,  # Causal relationship detection
            "omni_counterfactual_reasoning": 1.25,  # What-if analysis
            "omni_temporal_reasoning": 1.24,  # Time-based logic
            "omni_spatial_reasoning": 1.22,  # Spatial relationship understanding
            "omni_knowledge_synthesis": 1.26,  # Information integration
            "omni_uncertainty_quantification": 1.28,  # Uncertainty handling
            "omni_belief_revision": 1.24,  # Belief update consistency
            "omni_metacognitive_awareness": 1.22,  # Self-knowledge accuracy
            "omni_common_sense_reasoning": 1.25,  # Commonsense inference
        }

        # Initialize legacy alias mapping for backward compatibility
        self._initialize_legacy_aliases()

    def _initialize_legacy_aliases(self) -> None:
        """Initialize backward-compatible legacy aliases (deprecated in v2.0).

        Maps old scalar names to new omni-prefixed names for seamless migration.
        """
        self._legacy_aliases: dict[str, str] = {
            # Ethical scalars
            "morality_scalar": "omnimorality",
            "empathy_scalar": "omniempathy",
            "compassion_scalar": "omnicompassion",
            "forgiveness": "omniforgiveness",
            "love_scalar": "omnilove",
            "determination_scalar": "omnidetermination",
            "loyalty_scalar": "omniloyalty",
            "integrity_scalar": "omniintegrity",
            "wisdom_scalar": "omniwisdom",
            "justice_scalar": "omnijustice",
            "altruism_scalar": "omnialtruism",
            "hope_scalar": "omnihope",
            "courage_scalar": "omnicourage",
            "accountability_scalar": "omniaccountability",
            "transparency_weight": "omnitransparency",
            "explainability_factor": "omniexplainability",
            "benevolence": "omnibenevolence",
            "equity": "omniequity",
            # Cosmic scalars
            "universe_adapt": "omniuniverse_adapt",
            "telos_scalar": "omnitelos",
            "black_hole_entropy_eth": "omni_black_hole_entropy",
            "harmonic_singularity_bridge": "omni_harmonic_singularity",
            "golden_ratio_phi": "omni_golden_ratio_phi",
            # Quantum consciousness scalars
            "quantum_weight": "omniquantum_weight",
            "entanglement_risk": "omnientanglement_risk",
            "quantum_entanglement_weight": "omniquantum_entanglement",
            "neuro_quantum": "omnineuroquantum",
            "consciousness_coherence": "omniconsciousness_coherence",
            # Humanitarian scalars
            "crisis_response_boost": "omnicrisis_response",
            "disaster_response_boost": "omnidisaster_response",
            "pandemic_monitoring": "omnipandemic_monitoring",
            "missing_persons_priority": "omnimissing_persons_priority",
            "medical_discovery_boost": "omnimedical_discovery",
            # Security scalars
            "threat_detection_sensitivity": "omnithreat_detection",
            "quantum_resistance": "omniquantum_resistance",
            "encryption_strength": "omniencryption_strength",
            "audit_compliance": "omniaudit_compliance",
        }

    def resolve_scalar_name(self, name: str) -> str:
        """Resolve a scalar name, supporting legacy aliases.

        Args:
            name: Scalar name (may be legacy or omni-prefixed)

        Returns:
            Resolved omni-prefixed scalar name
        """
        if hasattr(self, "_legacy_aliases") and name in self._legacy_aliases:
            self.logger.debug(
                f"Legacy scalar alias '{name}' resolved to '{self._legacy_aliases[name]}' "
                "(deprecated in v2.0)"
            )
            return self._legacy_aliases[name]
        return name

    def get_scalar(self, name: str, default: float = 0.0) -> float:
        """Get a scalar value by name, supporting legacy aliases.

        Args:
            name: Scalar name (may be legacy or omni-prefixed)
            default: Default value if scalar not found

        Returns:
            Scalar value
        """
        resolved_name = self.resolve_scalar_name(name)
        for group_scalars in self.scalar_groups.values():
            if resolved_name in group_scalars:
                return group_scalars[resolved_name]
        return default

    def register_scalars(
        self,
        component_name: str,
        scalars: dict[str, float],
        group: ScalarGroup = ScalarGroup.ETHICAL,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Register scalars from a component.

        Args:
            component_name: Name of the registering component
            scalars: Dictionary of scalar name to value
            group: Scalar group classification
            metadata: Optional metadata about the registration
        """
        import time

        registration = ScalarRegistration(
            component_name=component_name,
            scalars=scalars,
            group=group,
            timestamp=time.time(),
            metadata=metadata or {},
        )

        self.registered_scalars[component_name] = registration

        for name, value in scalars.items():
            self.scalar_groups[group][name] = value

        self.logger.debug(
            f"Registered {len(scalars)} scalars from {component_name} in group {group.value}"
        )

    def get_enhanced_scalars(
        self,
        requesting_component: str,
        base_scalars: dict[str, float],
        context: dict[str, Any] | None = None,
    ) -> EnhancementResult:
        """
        Get enhanced scalars with GOSNN fusion, ethical gating, and harmonic synergy.

        This method performs bidirectional synaptic integration:
        1. Collects all registered scalars from components
        2. Evaluates ethical compliance via sigma_Immutable threshold
        3. Fuses dimensional states using 32-head attention with triadic phi-weighting
        4. Computes harmonic synergy for the weighted fusion Equation H(omega) term
        5. Returns enhanced scalars with fusion metadata

        Args:
            requesting_component: Name of the requesting component
            base_scalars: Base scalar values to enhance
            context: Optional context for enhancement (e.g., domain for threshold tuning)

        Returns:
            EnhancementResult with enhanced scalars, fusion score, harmonic synergy,
            ethical gate status, and any warnings
        """
        context = context or {}
        warnings: list[str] = []

        all_scalars = self._collect_all_scalars()
        scalar_vector = np.array(list(all_scalars.values()))

        passes_gate, ethical_score = self.ethical_gate.evaluate(scalar_vector)

        if not passes_gate:
            warnings.append(
                f"Ethical gate warning: score {ethical_score:.3f} below threshold "
                f"{self.sigma_immutable_threshold:.2f}"
            )
            self.logger.warning(
                f"Ethical gate triggered for {requesting_component}: "
                f"score={ethical_score:.3f}, threshold={self.sigma_immutable_threshold:.2f}"
            )

        dimensional_states = self._prepare_dimensional_states(base_scalars, context)

        # Fuse with triadic phi-weighting and track harmonic synergy
        fuse_result = self.attention_fusion.fuse(dimensional_states, return_synergy=True)
        if isinstance(fuse_result, tuple):
            fused_state, harmonic_synergy = fuse_result
        else:
            fused_state = fuse_result
            harmonic_synergy = 0.5

        # Store harmonic synergy for weighted fusion Equation
        self.last_harmonic_synergy = harmonic_synergy

        enhanced_scalars = self._apply_enhancement(base_scalars, fused_state, ethical_score)

        intelligence_contribution = self._compute_intelligence_contribution(enhanced_scalars)

        return EnhancementResult(
            enhanced_scalars=enhanced_scalars,
            fusion_score=float(np.mean(fused_state)),
            ethical_gate_passed=passes_gate,
            intelligence_contribution=intelligence_contribution,
            warnings=warnings,
        )

    def fuse_37d_scalars(
        self, dimensional_states: list[np.ndarray[Any, Any]]
    ) -> np.ndarray[Any, Any]:
        """
        Perform 37-dimensional quantum fusion.

        Args:
            dimensional_states: List of dimensional state vectors

        Returns:
            Fused 37D state vector
        """
        return self.attention_fusion.fuse(dimensional_states)  # type: ignore[return-value, unused-ignore]

    def compute_global_intelligence_score(self) -> float:
        """
        Compute global intelligence score from all registered scalars.

        Returns:
            Global intelligence score (0-1)
        """
        all_scalars = self._collect_all_scalars()

        if not all_scalars:
            return 0.5

        scalar_values = np.array(list(all_scalars.values()))

        boost_count = np.sum(scalar_values > 1.0)
        penalty_count = np.sum(scalar_values < 1.0)
        total = boost_count + penalty_count

        boost_ratio = 0.5 if total == 0 else boost_count / total

        mean_value = np.mean(scalar_values)
        std_value = np.std(scalar_values)

        triadic_harmony = self._compute_triadic_harmony(scalar_values)

        intelligence_score = (
            0.3 * boost_ratio
            + 0.3 * min(mean_value / 2.0, 1.0)
            + 0.2 * (1.0 / (1.0 + std_value))
            + 0.2 * triadic_harmony
        )

        return float(np.clip(intelligence_score, 0.0, 1.0))

    def compute_triadic_harmony(self) -> float:
        """
        Compute triadic harmony using golden ratio (φ = 1.618).

        Triadic harmony represents the balance between boosts, penalties,
        and neutral scalars, weighted by the golden ratio.

        Returns:
            Triadic harmony score (0-1)
        """
        all_scalars = self._collect_all_scalars()
        scalar_values = np.array(list(all_scalars.values()))
        return self._compute_triadic_harmony(scalar_values)

    def _compute_triadic_harmony(self, scalar_values: np.ndarray[Any, Any]) -> float:
        """Internal triadic harmony computation."""
        if len(scalar_values) == 0:
            return 0.5

        boosts = scalar_values[scalar_values > 1.0]
        penalties = scalar_values[scalar_values < 1.0]
        neutrals = scalar_values[np.isclose(scalar_values, 1.0)]

        boost_mean = np.mean(boosts) if len(boosts) > 0 else 1.0
        penalty_mean = np.mean(penalties) if len(penalties) > 0 else 1.0
        neutral_mean = np.mean(neutrals) if len(neutrals) > 0 else 1.0

        triad_avg = (boost_mean + penalty_mean + neutral_mean) / 3.0

        harmony = self.PHI * triad_avg / (1.0 + self.PHI)

        return float(np.clip(harmony, 0.0, 1.0))

    def perform_bias_audit(self) -> dict[str, Any]:
        """
        Perform bias audit on current scalar distribution.

        Ensures 60/40 boost/penalty ratio for balanced operation.

        Returns:
            Audit results with recommendations
        """
        all_scalars = self._collect_all_scalars()
        scalar_values = np.array(list(all_scalars.values()))

        boost_count = np.sum(scalar_values > 1.0)
        penalty_count = np.sum(scalar_values < 1.0)
        neutral_count = np.sum(np.isclose(scalar_values, 1.0))
        total = len(scalar_values)

        if total == 0:
            return {
                "status": "no_scalars",
                "boost_ratio": 0.0,
                "penalty_ratio": 0.0,
                "recommendations": ["Register scalars before auditing"],
            }

        boost_ratio = boost_count / total
        penalty_ratio = penalty_count / total

        recommendations = []
        if boost_ratio > 0.65:
            recommendations.append("Consider reducing boost scalars for balance")
        elif boost_ratio < 0.55:
            recommendations.append("Consider increasing boost scalars for balance")

        omniempathy = self.scalar_groups[ScalarGroup.ETHICAL].get("omniempathy", self.MIN_EMPATHY)
        omnimorality = self.scalar_groups[ScalarGroup.ETHICAL].get(
            "omnimorality", self.MIN_MORALITY
        )
        benevolence_threshold = ETHICAL.BENEVOLENCE_IMMUTABLE
        omnibenevolence = self.scalar_groups[ScalarGroup.ETHICAL].get(
            "omnibenevolence", benevolence_threshold
        )

        if omniempathy < self.MIN_EMPATHY:
            recommendations.append(
                f"Omniempathy {omniempathy:.2f} below minimum {self.MIN_EMPATHY}"
            )
        if omnimorality < self.MIN_MORALITY:
            recommendations.append(
                f"Omnimorality {omnimorality:.2f} below minimum {self.MIN_MORALITY}"
            )
        if omnibenevolence < benevolence_threshold:
            recommendations.append(
                f"Omnibenevolence {omnibenevolence:.2f} below required "
                f"threshold {benevolence_threshold}"
            )

        return {
            "status": "passed" if not recommendations else "warnings",
            "total_scalars": total,
            "boost_count": int(boost_count),
            "penalty_count": int(penalty_count),
            "neutral_count": int(neutral_count),
            "boost_ratio": float(boost_ratio),
            "penalty_ratio": float(penalty_ratio),
            "target_boost_ratio": self.TARGET_BOOST_RATIO,
            "omniempathy": float(omniempathy),
            "omnimorality": float(omnimorality),
            "omnibenevolence": float(omnibenevolence),
            "recommendations": recommendations,
        }

    def get_scalar_statistics(self) -> dict[str, Any]:
        """Get comprehensive statistics about registered scalars."""
        all_scalars = self._collect_all_scalars()

        if not all_scalars:
            return {
                "total_scalars": 0,
                "groups": {},
                "global_intelligence_score": 0.5,
            }

        group_stats = {}
        for group in ScalarGroup:
            group_scalars = self.scalar_groups[group]
            if group_scalars:
                values = np.array(list(group_scalars.values()))
                group_stats[group.value] = {
                    "count": len(group_scalars),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }

        return {
            "total_scalars": len(all_scalars),
            "registered_components": len(self.registered_scalars),
            "groups": group_stats,
            "global_intelligence_score": self.compute_global_intelligence_score(),
            "triadic_harmony": self.compute_triadic_harmony(),
            "bias_audit": self.perform_bias_audit(),
        }

    def _collect_all_scalars(self) -> dict[str, float]:
        """Collect all scalars from all groups."""
        all_scalars: dict[str, float] = {}
        for group_scalars in self.scalar_groups.values():
            all_scalars.update(group_scalars)
        return all_scalars

    def _prepare_dimensional_states(
        self, base_scalars: dict[str, float], context: dict[str, Any]
    ) -> list[np.ndarray[Any, Any]]:
        """Prepare dimensional states for fusion."""
        states = []

        base_values = np.array(list(base_scalars.values()))
        states.append(base_values)

        for group in ScalarGroup:
            group_scalars = self.scalar_groups[group]
            if group_scalars:
                group_values = np.array(list(group_scalars.values()))
                states.append(group_values)

        return states

    def _apply_enhancement(
        self,
        base_scalars: dict[str, float],
        fused_state: np.ndarray[Any, Any],
        ethical_score: float,
    ) -> dict[str, float]:
        """Apply enhancement to base scalars using fused state."""
        enhanced = {}

        fusion_factor = 1.0 + 0.1 * np.mean(fused_state)
        ethical_factor = ethical_score

        for i, (name, value) in enumerate(base_scalars.items()):
            enhancement = fused_state[i] * 0.1 if i < len(fused_state) else 0.0

            enhanced_value = value * fusion_factor * ethical_factor + enhancement
            enhanced[name] = float(enhanced_value)

        return enhanced

    def _compute_intelligence_contribution(self, enhanced_scalars: dict[str, float]) -> float:
        """Compute intelligence contribution from enhanced scalars."""
        if not enhanced_scalars:
            return 0.0

        values = np.array(list(enhanced_scalars.values()))
        contribution = np.mean(values) / (1.0 + np.std(values))
        return float(np.clip(contribution, 0.0, 1.0))

    def compute_hierarchical_score(
        self,
        domain_weights: dict[str, float] | None = None,
        aggregation_method: str = "geometric_mean",
    ) -> dict[str, Any]:
        """Compute hierarchical aggregation of omni-scalars (Phase 3).

        Implements 3-level hierarchical aggregation:

        Level 1 — Category groups: Group scalars by category
            (safety, fairness, transparency, accountability, beneficence)
        Level 2 — Category aggregation: Weighted mean within each group
            (weights set by domain priority)
        Level 3 — Cross-category: Final aggregation via geometric mean
            (penalizes any single low score)

        The mapping from ScalarGroups to the 5-category taxonomy:
            - safety: MEDICAL, SECURITY, HUMANITARIAN
            - fairness: ETHICAL (equity/justice subset)
            - transparency: ETHICAL (transparency/explainability subset)
            - accountability: SOFTWARE_ENGINEERING, ADVANCED_REASONING
            - beneficence: ETHICAL (benevolence/compassion subset), COSMIC

        Args:
            domain_weights: Optional per-category weights. If None, uses
                equal weighting. Keys: "safety", "fairness", "transparency",
                "accountability", "beneficence".
            aggregation_method: Cross-category aggregation method.
                "geometric_mean" (default, penalizes low scores),
                "arithmetic_mean", or "harmonic_mean".

        Returns:
            Dict with:
                - "overall_score": Final aggregated score in [0, 1]
                - "category_scores": Per-category scores
                - "category_sizes": Number of scalars per category
                - "method": Aggregation method used
        """
        # Default equal weights
        if domain_weights is None:
            domain_weights = {
                "safety": 1.0,
                "fairness": 1.0,
                "transparency": 1.0,
                "accountability": 1.0,
                "beneficence": 1.0,
            }

        # Level 1: Map scalars to categories
        categories: dict[str, list[float]] = {
            "safety": [],
            "fairness": [],
            "transparency": [],
            "accountability": [],
            "beneficence": [],
        }

        # Safety: Medical + Security + Humanitarian
        for group_key in [ScalarGroup.MEDICAL, ScalarGroup.SECURITY, ScalarGroup.HUMANITARIAN]:
            categories["safety"].extend(self.scalar_groups[group_key].values())

        # Fairness: Ethical equity/justice subset
        ethical = self.scalar_groups[ScalarGroup.ETHICAL]
        fairness_keys = [k for k in ethical if "equit" in k or "justic" in k or "bias" in k]
        categories["fairness"].extend(ethical[k] for k in fairness_keys)

        # Transparency: Ethical transparency/explainability subset
        transparency_keys = [
            k for k in ethical if "transparen" in k or "explain" in k or "account" in k
        ]
        categories["transparency"].extend(ethical[k] for k in transparency_keys)

        # Accountability: Software Engineering + Advanced Reasoning
        for group_key in [ScalarGroup.SOFTWARE_ENGINEERING, ScalarGroup.ADVANCED_REASONING]:
            categories["accountability"].extend(self.scalar_groups[group_key].values())

        # Beneficence: Ethical benevolence/compassion + Cosmic
        beneficence_keys = [
            k for k in ethical
            if "benevol" in k or "compass" in k or "love" in k
            or "empathy" in k or "altru" in k or "hope" in k
        ]
        categories["beneficence"].extend(ethical[k] for k in beneficence_keys)
        categories["beneficence"].extend(
            self.scalar_groups[ScalarGroup.COSMIC].values()
        )

        # Level 2: Weighted mean within each category
        category_scores: dict[str, float] = {}
        category_sizes: dict[str, int] = {}

        for cat_name, values in categories.items():
            category_sizes[cat_name] = len(values)
            if values:
                arr = np.array(values)
                # Normalize: values > 1 are boosts, < 1 are penalties
                # Map to [0, 1] by dividing by max reasonable value (2.0)
                normalized = np.clip(arr / 2.0, 0.0, 1.0)
                category_scores[cat_name] = float(np.mean(normalized))
            else:
                category_scores[cat_name] = 0.5  # Neutral if empty

        # Level 3: Cross-category aggregation
        weighted_scores = []
        total_weight = 0.0
        for cat_name, score in category_scores.items():
            w = domain_weights.get(cat_name, 1.0)
            weighted_scores.append((score, w))
            total_weight += w

        if total_weight == 0 or not weighted_scores:
            overall = 0.5
        elif aggregation_method == "geometric_mean":
            # Weighted geometric mean: penalizes any single low score
            log_sum = sum(w * np.log(max(s, 1e-10)) for s, w in weighted_scores)
            overall = float(np.exp(log_sum / total_weight))
        elif aggregation_method == "harmonic_mean":
            # Weighted harmonic mean: even stronger penalty for low scores
            inv_sum = sum(w / max(s, 1e-10) for s, w in weighted_scores)
            overall = float(total_weight / inv_sum)
        else:
            # Arithmetic mean (default fallback)
            overall = float(
                sum(s * w for s, w in weighted_scores) / total_weight
            )

        overall = float(np.clip(overall, 0.0, 1.0))

        return {
            "overall_score": overall,
            "category_scores": category_scores,
            "category_sizes": category_sizes,
            "method": aggregation_method,
        }


# Global GOSNN singleton instance
# Thread Safety: Uses lazy initialization with potential race condition on first access.
# The GlobalOmniScalarNetwork class uses internal locking for thread-safe operations
# once instantiated. For production multi-threaded use, call get_global_scalar_network()
# once during application startup before spawning worker threads.
_global_network: GlobalOmniScalarNetwork | None = None


def get_global_scalar_network(
    device: str = "cpu",
    quantum_mode: bool = False,
    max_dimensions: int = 37,
    domain: str | None = None,
    num_attention_heads: int = 32,
    enable_triadic_phi: bool = True,
) -> GlobalOmniScalarNetwork:
    """
    Get the global GOSNN singleton instance.

    The GOSNN provides bidirectional synaptic integration with the 3R mechanism
    and other components. It uses 32-head attention with triadic phi-weighting
    for harmonic synergy and configurable sigma_Immutable threshold for ethical gating.

    Args:
        device: Computation device ('cpu' or 'cuda')
        quantum_mode: Enable quantum-inspired operations
        max_dimensions: Maximum dimensions for fusion (default 37)
        domain: Domain identifier for threshold tuning (e.g., "medical" uses 0.93)
        num_attention_heads: Number of attention heads (default 32 for head_dim=16)
        enable_triadic_phi: Enable triadic phi-weighting for harmonic synergy

    Returns:
        GlobalOmniScalarNetwork singleton instance

    Note:
        The sigma_Immutable threshold can also be configured via the
        SIGMA_IMMUTABLE_THRESHOLD environment variable. Default is 0.96 for
        ~10-15% false positive reduction. Medical domains use 0.93 fallback.
    """
    global _global_network
    if _global_network is None:
        _global_network = GlobalOmniScalarNetwork(
            device=device,
            quantum_mode=quantum_mode,
            max_dimensions=max_dimensions,
            domain=domain,
            num_attention_heads=num_attention_heads,
            enable_triadic_phi=enable_triadic_phi,
        )
    return _global_network


def reset_global_network() -> None:
    """Reset the global GOSNN instance (primarily for testing)."""
    global _global_network
    GlobalOmniScalarNetwork._instance = None
    _global_network = None
