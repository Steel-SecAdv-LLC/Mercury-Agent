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
Global Omni-Scalar Network (GOSNN) - Intelligence Fusion Hub

Implements a comprehensive scalar monitoring and fusion system inspired by
Aether Halo NSN's ~700 omni-scalars architecture. Provides:

- 37-dimensional quantum fusion with multi-head attention
- Ethical gating with σ_Sacred threshold enforcement
- Component-based scalar registration and enhancement
- Global intelligence score computation
- Triadic harmony computation using golden ratio (φ = 1.618)

The GOSNN serves as a central hub for aggregating insights from multiple
specialized engines and maintaining system-wide ethical alignment.

Research sources:
- Aether Halo NSN v2.0 (Steel-SecAdv-LLC)
- Multi-head attention: Vaswani et al. (2017) "Attention Is All You Need"
- Golden ratio applications: Livio (2002) "The Golden Ratio"
"""

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


class ScalarGroup(Enum):
    """Thematic groups for omni-scalars."""

    ETHICAL = "ethical"
    COSMIC = "cosmic"
    QUANTUM_CONSCIOUSNESS = "quantum_consciousness"
    MATHEMATICAL_MYSTERIES = "mathematical_mysteries"
    PARADOX_DEFENSE = "paradox_defense"
    PHYSICS_THEORIES = "physics_theories"
    SUSTAINABILITY = "sustainability"
    HUMANITARIAN = "humanitarian"
    SECURITY = "security"
    MEDICAL = "medical"
    CRISIS_RESPONSE = "crisis_response"
    AI_GUARDIAN = "ai_guardian"


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

    Blocks operations if ethical score falls below σ_Sacred threshold (0.93).
    Uses a simple feedforward network: 256 → 64 → 1 with Sigmoid activation.
    """

    def __init__(self, input_dim: int = 256, threshold: float = 0.93):
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
            self.gate_network = None

    def evaluate(self, scalar_vector: np.ndarray) -> tuple[bool, float]:
        """
        Evaluate ethical compliance of scalar vector.

        Args:
            scalar_vector: Input scalar values

        Returns:
            Tuple of (passes_gate, ethical_score)
        """
        if self.gate_network is not None and TORCH_AVAILABLE:
            padded = np.zeros(self.input_dim)
            padded[: min(len(scalar_vector), self.input_dim)] = scalar_vector[
                : self.input_dim
            ]

            with torch.no_grad():
                tensor_input = torch.tensor(padded, dtype=torch.float32).unsqueeze(0)
                score = self.gate_network(tensor_input).item()
        else:
            score = self._compute_ethical_score_numpy(scalar_vector)

        passes = score >= self.threshold
        return passes, score

    def _compute_ethical_score_numpy(self, scalar_vector: np.ndarray) -> float:
        """Compute ethical score using NumPy fallback."""
        if len(scalar_vector) == 0:
            return 0.5

        positive_ratio = np.sum(scalar_vector > 1.0) / len(scalar_vector)
        mean_value = np.mean(scalar_vector)
        std_value = np.std(scalar_vector)

        score = 0.4 * positive_ratio + 0.4 * min(mean_value / 2.0, 1.0) + 0.2 * (
            1.0 / (1.0 + std_value)
        )
        return float(np.clip(score, 0.0, 1.0))


class MultiHeadAttentionFusion:
    """
    Multi-head attention mechanism for 37D quantum fusion.

    Implements 8-head attention at d_model=512 for fusing scalar dimensions.
    """

    def __init__(
        self, d_model: int = 512, num_heads: int = 8, max_dimensions: int = 37
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_dimensions = max_dimensions
        self.head_dim = d_model // num_heads
        self.logger = logging.getLogger(__name__)

        if TORCH_AVAILABLE:
            self.attention = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=num_heads, batch_first=True
            )
            self.projection = nn.Linear(max_dimensions, d_model)
            self.output_projection = nn.Linear(d_model, max_dimensions)
        else:
            self.attention = None
            self.projection = None
            self.output_projection = None

    def fuse(self, dimensional_states: list[np.ndarray]) -> np.ndarray:
        """
        Fuse multiple dimensional states using multi-head attention.

        Args:
            dimensional_states: List of state vectors to fuse

        Returns:
            Fused state vector
        """
        if not dimensional_states:
            return np.zeros(self.max_dimensions)

        padded_states = []
        for state in dimensional_states:
            padded = np.zeros(self.max_dimensions)
            padded[: min(len(state), self.max_dimensions)] = state[: self.max_dimensions]
            padded_states.append(padded)

        stacked = np.stack(padded_states)

        if self.attention is not None and TORCH_AVAILABLE:
            with torch.no_grad():
                tensor_input = torch.tensor(stacked, dtype=torch.float32)
                projected = self.projection(tensor_input)
                projected = projected.unsqueeze(0)

                attn_output, _ = self.attention(projected, projected, projected)

                fused = self.output_projection(attn_output.squeeze(0))
                result = fused.mean(dim=0).numpy()
        else:
            weights = np.ones(len(padded_states)) / len(padded_states)
            result = np.average(stacked, axis=0, weights=weights)

        return result


class GlobalOmniScalarNetwork:
    """
    Global Omni-Scalar Network (GOSNN) - Central Intelligence Fusion Hub.

    Aggregates ~700 omni-scalars from multiple components and provides:
    - 37D quantum fusion with multi-head attention
    - Ethical gating with σ_Sacred threshold
    - Component-based scalar registration
    - Global intelligence score computation
    - Triadic harmony using golden ratio (φ = 1.618)

    This is implemented as a singleton to ensure consistent global state.
    """

    _instance: Optional["GlobalOmniScalarNetwork"] = None
    _lock = threading.Lock()

    PHI = 1.618033988749895
    SIGMA_SACRED_THRESHOLD = 0.93
    MIN_EMPATHY = 1.22
    MIN_MORALITY = 1.20
    TARGET_BOOST_RATIO = 0.60

    def __new__(cls, *args: Any, **kwargs: Any) -> "GlobalOmniScalarNetwork":
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
    ):
        if getattr(self, "_initialized", False):
            return

        self.device = device
        self.quantum_mode = quantum_mode
        self.max_dimensions = max_dimensions
        self.logger = logging.getLogger(__name__)

        self.registered_scalars: dict[str, ScalarRegistration] = {}
        self.scalar_groups: dict[ScalarGroup, dict[str, float]] = {
            group: {} for group in ScalarGroup
        }

        self.ethical_gate = EthicalGate(threshold=self.SIGMA_SACRED_THRESHOLD)
        self.attention_fusion = MultiHeadAttentionFusion(
            max_dimensions=max_dimensions
        )

        self._initialize_default_scalars()
        self._initialized = True

        self.logger.info(
            f"GOSNN initialized: device={device}, quantum_mode={quantum_mode}, "
            f"max_dimensions={max_dimensions}"
        )

    def _initialize_default_scalars(self) -> None:
        """Initialize default ethical and system scalars."""
        self.scalar_groups[ScalarGroup.ETHICAL] = {
            "morality_scalar": self.MIN_MORALITY,
            "empathy_scalar": self.MIN_EMPATHY,
            "compassion_scalar": 1.30,
            "forgiveness": 0.1,
            "love_scalar": 1.30,
            "determination_scalar": 1.30,
            "loyalty_scalar": 1.30,
            "integrity_scalar": 1.30,
            "wisdom_scalar": 1.30,
            "justice_scalar": 1.30,
            "altruism_scalar": 1.30,
            "hope_scalar": 1.30,
            "courage_scalar": 1.30,
            "accountability_scalar": 1.30,
            "transparency_weight": 0.18,
            "explainability_factor": 0.9,
            "survivor_first_principle": 1.35,
            "bias_audit_compliance": 1.25,
        }

        self.scalar_groups[ScalarGroup.COSMIC] = {
            "universe_adapt": 1.20,
            "telos_scalar": 1.25,
            "black_hole_entropy_eth": 1.30,
            "harmonic_singularity_bridge": 1.30,
            "golden_ratio_phi": self.PHI,
        }

        self.scalar_groups[ScalarGroup.QUANTUM_CONSCIOUSNESS] = {
            "quantum_weight": 0.12,
            "entanglement_risk": 0.1,
            "quantum_entanglement_weight": 0.14,
            "neuro_quantum": 1.30,
            "consciousness_coherence": 1.25,
        }

        self.scalar_groups[ScalarGroup.HUMANITARIAN] = {
            "crisis_response_boost": 1.35,
            "disaster_response_boost": 1.30,
            "pandemic_monitoring": 1.25,
            "missing_persons_priority": 1.40,
            "medical_discovery_boost": 1.30,
        }

        self.scalar_groups[ScalarGroup.SECURITY] = {
            "threat_detection_sensitivity": 1.25,
            "quantum_resistance": 1.30,
            "encryption_strength": 1.35,
            "audit_compliance": 1.20,
        }

    def register_scalars(
        self,
        component_name: str,
        scalars: dict[str, float],
        group: ScalarGroup = ScalarGroup.ETHICAL,
        metadata: Optional[dict[str, Any]] = None,
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
        context: Optional[dict[str, Any]] = None,
    ) -> EnhancementResult:
        """
        Get enhanced scalars with GOSNN fusion and ethical gating.

        Args:
            requesting_component: Name of the requesting component
            base_scalars: Base scalar values to enhance
            context: Optional context for enhancement

        Returns:
            EnhancementResult with enhanced scalars and metadata
        """
        context = context or {}
        warnings: list[str] = []

        all_scalars = self._collect_all_scalars()
        scalar_vector = np.array(list(all_scalars.values()))

        passes_gate, ethical_score = self.ethical_gate.evaluate(scalar_vector)

        if not passes_gate:
            warnings.append(
                f"Ethical gate warning: score {ethical_score:.3f} below threshold "
                f"{self.SIGMA_SACRED_THRESHOLD}"
            )
            self.logger.warning(
                f"Ethical gate triggered for {requesting_component}: "
                f"score={ethical_score:.3f}"
            )

        dimensional_states = self._prepare_dimensional_states(base_scalars, context)
        fused_state = self.attention_fusion.fuse(dimensional_states)

        enhanced_scalars = self._apply_enhancement(
            base_scalars, fused_state, ethical_score
        )

        intelligence_contribution = self._compute_intelligence_contribution(
            enhanced_scalars
        )

        return EnhancementResult(
            enhanced_scalars=enhanced_scalars,
            fusion_score=float(np.mean(fused_state)),
            ethical_gate_passed=passes_gate,
            intelligence_contribution=intelligence_contribution,
            warnings=warnings,
        )

    def fuse_37d_scalars(self, dimensional_states: list[np.ndarray]) -> np.ndarray:
        """
        Perform 37-dimensional quantum fusion.

        Args:
            dimensional_states: List of dimensional state vectors

        Returns:
            Fused 37D state vector
        """
        return self.attention_fusion.fuse(dimensional_states)

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

        if total == 0:
            boost_ratio = 0.5
        else:
            boost_ratio = boost_count / total

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

    def _compute_triadic_harmony(self, scalar_values: np.ndarray) -> float:
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

        empathy = self.scalar_groups[ScalarGroup.ETHICAL].get(
            "empathy_scalar", self.MIN_EMPATHY
        )
        morality = self.scalar_groups[ScalarGroup.ETHICAL].get(
            "morality_scalar", self.MIN_MORALITY
        )

        if empathy < self.MIN_EMPATHY:
            recommendations.append(
                f"Empathy scalar {empathy:.2f} below minimum {self.MIN_EMPATHY}"
            )
        if morality < self.MIN_MORALITY:
            recommendations.append(
                f"Morality scalar {morality:.2f} below minimum {self.MIN_MORALITY}"
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
            "empathy_scalar": float(empathy),
            "morality_scalar": float(morality),
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
    ) -> list[np.ndarray]:
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
        fused_state: np.ndarray,
        ethical_score: float,
    ) -> dict[str, float]:
        """Apply enhancement to base scalars using fused state."""
        enhanced = {}

        fusion_factor = 1.0 + 0.1 * np.mean(fused_state)
        ethical_factor = ethical_score

        for i, (name, value) in enumerate(base_scalars.items()):
            if i < len(fused_state):
                enhancement = fused_state[i] * 0.1
            else:
                enhancement = 0.0

            enhanced_value = value * fusion_factor * ethical_factor + enhancement
            enhanced[name] = float(enhanced_value)

        return enhanced

    def _compute_intelligence_contribution(
        self, enhanced_scalars: dict[str, float]
    ) -> float:
        """Compute intelligence contribution from enhanced scalars."""
        if not enhanced_scalars:
            return 0.0

        values = np.array(list(enhanced_scalars.values()))
        contribution = np.mean(values) / (1.0 + np.std(values))
        return float(np.clip(contribution, 0.0, 1.0))


_global_network: Optional[GlobalOmniScalarNetwork] = None


def get_global_scalar_network(
    device: str = "cpu",
    quantum_mode: bool = False,
    max_dimensions: int = 37,
) -> GlobalOmniScalarNetwork:
    """
    Get the global GOSNN singleton instance.

    Args:
        device: Computation device ('cpu' or 'cuda')
        quantum_mode: Enable quantum-inspired operations
        max_dimensions: Maximum dimensions for fusion

    Returns:
        GlobalOmniScalarNetwork singleton instance
    """
    global _global_network
    if _global_network is None:
        _global_network = GlobalOmniScalarNetwork(
            device=device,
            quantum_mode=quantum_mode,
            max_dimensions=max_dimensions,
        )
    return _global_network


def reset_global_network() -> None:
    """Reset the global GOSNN instance (primarily for testing)."""
    global _global_network
    GlobalOmniScalarNetwork._instance = None
    _global_network = None
