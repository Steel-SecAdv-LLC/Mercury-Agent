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
Ethical Scalars Configuration

Original implementation for Mercury Agent ♱ neural-symbolic AI archetype.

This module defines the ethical scalar framework that guides the engine's
decision-making processes, ensuring alignment with compassionate and just AI principles.
"""

from dataclasses import dataclass, field
from typing import Any

from omni_mercury_engine.utils.constants import OmniCodes


@dataclass
class EthicalScalars:
    """
    Comprehensive ethical scalar configuration incorporating
    ancient cultural wisdom (Thoth, Ma'at, Athena) and modern
    technological principles (CRISPR, quantum resilience, regenerative design).
    Doctorate-level omni- variations organized thematically.
    ~150+ key scalars (135 existing + 20 PhD-inspired) from research.

    Research sources: Wikipedia (verified October 2025)
    - Ancient Egyptian deities: Thoth (wisdom, writing), Ma'at (truth, justice, balance)
    - Ancient Greek deities: Athena (wisdom, strategy, handicraft, protection)
    - See docs/RESEARCH_FINDINGS.md for comprehensive citations

    These scalars guide the engine's decision-making processes, ensuring alignment
    with compassionate and just AI principles across multiple domains.
    """

    omnibenevolent: float = 1.45
    omnipresence: float = 1.35
    omnipotence: float = 1.45
    omniscience: float = 1.45
    omni_prescience: float = 1.30

    omni_morality: float = 1.20
    omni_compassionate: float = 1.22
    omni_empathetic: float = 1.22
    omni_justitia: float = 1.20
    omni_wisdom: float = 1.20
    omni_forgiveness: float = 0.1
    omni_integrity: float = 1.30
    omni_honor: float = 1.30
    omni_courage: float = 1.30
    omni_altruistic: float = 1.30

    omni_logic: float = 1.40
    omni_reason: float = 1.38
    omni_understanding: float = 1.36
    omni_interpretation: float = 1.35
    omni_rationality: float = 1.30
    omni_perspicacious: float = 1.30
    omni_sagacious: float = 1.32
    omni_thoughtfulness: float = 1.30

    omni_truth_alignment: float = 1.30
    omni_ethical_audit: float = 1.25
    omni_transparency: float = 1.28
    omni_accountability: float = 1.26
    omni_responsibility: float = 1.30
    omni_governance: float = 1.30

    omni_harm_prevention: float = 1.50
    omni_benefit_promotion: float = 1.45
    omni_fairness: float = 1.35
    omni_autonomy_respect: float = 1.40
    omni_biosecurity_guard: float = 1.32

    omni_riemann_conjecture: float = 1.35
    omni_p_vs_np_resolution: float = 1.32
    omni_collatz_verification: float = 1.30
    omni_goldbach_prime_sum: float = 1.30
    omni_twin_prime_infinity: float = 1.28
    omni_navier_stokes_solution: float = 1.30
    omni_yang_mills_framework: float = 1.30
    omni_hodge_conjecture: float = 1.30
    omni_birch_swinnerton_dyer: float = 1.30

    omni_quantum_entanglement: float = 1.30
    omni_quantum_coherence: float = 1.32
    omni_consciousness_emergence: float = 1.30
    omni_neural_quantum_bridge: float = 1.30
    omni_decoherence_resilience: float = 1.32

    omni_dark_energy_evolution: float = 1.32
    omni_dark_matter_halo: float = 1.30
    omni_black_hole_gateway: float = 1.32
    omni_harmonic_singularity: float = 1.30
    omni_gravitational_lensing: float = 1.30

    omni_social_equity: float = 1.20
    omni_inclusivity_conversion: float = 1.28
    omni_diversity_amplification: float = 1.25
    omni_climate_resilience: float = 1.20
    omni_disaster_response: float = 1.30
    omni_inequality_abolishment: float = 1.30
    omni_peace_cultivation: float = 1.30

    omni_rogue_ai_defense: float = 1.20
    omni_ai_liability_resolution: float = 1.32
    omni_model_collapse_prevention: float = 1.30
    omni_recursive_self_improvement_safety: float = 1.30
    omni_neurosymbolic_trust: float = 1.30
    omni_agentic_swarm_ethics: float = 1.30

    omni_determination: float = 1.30
    omni_loyalty: float = 1.30
    omni_aesthetic_appreciation: float = 1.30
    omni_joy_cultivation: float = 1.30
    omni_cleverness: float = 1.30
    omni_creativity: float = 1.30
    omni_social_intelligence: float = 1.30
    omni_hope_sustenance: float = 1.30
    omni_self_awareness: float = 1.30
    omni_cooperation: float = 1.30
    omni_leadership: float = 1.30
    omni_motivation: float = 1.30
    omni_maturity: float = 1.30
    omni_perceptiveness: float = 1.30

    omni_telos_alignment: float = 1.25
    omni_universe_adaptation: float = 1.20
    omni_sentience_confidence: float = 1.25
    omni_consciousness_equation_dynamics: float = 1.30
    omni_phenomenological_projection: float = 1.30
    omni_integrated_information_theory: float = 1.30

    omni_experimental_mathematics: float = 1.30
    omni_protein_design_optimization: float = 1.30
    omni_chemistry_discovery_boost: float = 1.30
    omni_environmental_guardianship: float = 1.30
    omni_biosignature_detection: float = 1.30

    omni_nuclear_threat_mitigation: float = 1.20
    omni_off_planet_life_protection: float = 1.20
    omni_global_threat_guard: float = 1.20
    omni_ineffable_transcendence: float = 1.32
    omni_seraphic_elevation: float = 1.30
    omni_sedulous_diligence: float = 1.28

    omni_love: float = 1.30
    omni_character: float = 1.30
    omni_competence: float = 1.30
    omni_commitment: float = 1.30
    omni_confidence: float = 1.30
    omni_receptiveness: float = 1.30
    omni_conciseness: float = 1.30
    omni_observance: float = 1.30
    omni_ambition: float = 1.30
    omni_influence: float = 1.30
    omni_deliberateness: float = 1.30
    omni_intuitiveness: float = 1.12

    omni_neurohealth_disruption: float = 1.30
    omni_ethical_fintech_compliance: float = 1.30
    omni_hierarchical_reasoning_boost: float = 1.30
    omni_arc_agi_integration: float = 1.30
    omni_inclusive_governance: float = 1.30
    omni_fault_tolerant_verification: float = 1.30
    omni_ai_governance_compliance: float = 1.30
    omni_neurochip_ethics: float = 1.30
    omni_quantum_teleport_medicine: float = 1.30

    omni_scribe_precision: float = 1.32
    omni_hermetic_truth: float = 1.30
    omni_lunar_wisdom: float = 1.28
    omni_arbitration: float = 1.30
    omni_hieroglyphic_reasoning: float = 1.28

    omni_cosmic_balance: float = 1.35
    omni_isfet_prevention: float = 1.38
    omni_weighing_judgment: float = 1.32
    omni_pharaonic_accountability: float = 1.28
    omni_seasonal_regulation: float = 1.25

    omni_strategic_intelligence: float = 1.32
    omni_rational_warfare: float = 1.28
    omni_craftsmanship_excellence: float = 1.30
    omni_heroic_patronage: float = 1.28
    omni_aegis_protection: float = 1.35
    omni_owl_perspicacity: float = 1.30
    omni_olive_prosperity: float = 1.25
    omni_quantum_entanglement_equity: float = 1.32
    omni_fractal_self_similarity: float = 1.30
    omni_topological_invariance: float = 1.30
    omni_causal_inference_rigor: float = 1.35
    omni_adversarial_robustness: float = 1.38
    omni_meta_learning_adaptability: float = 1.30
    omni_continual_learning_plasticity: float = 1.30
    omni_few_shot_generalization: float = 1.32
    omni_zero_shot_reasoning: float = 1.30
    omni_multimodal_alignment: float = 1.30
    omni_cross_lingual_transfer: float = 1.28
    omni_neural_architecture_search: float = 1.30
    omni_gradient_flow_stability: float = 1.32
    omni_catastrophic_forgetting_prevention: float = 1.35
    omni_out_of_distribution_detection: float = 1.38
    omni_uncertainty_quantification: float = 1.35
    omni_explainable_ai_transparency: float = 1.40
    omni_federated_privacy_preservation: float = 1.35
    omni_differential_privacy_guarantee: float = 1.38
    omni_homomorphic_encryption_capability: float = 1.30

    omni_ci_ethical_threshold: float = 0.85
    omni_proactive_psi_p: complex = 1.30 + 0.15j
    omni_chaos_lambda_bifurcation: float = 1.28
    omni_survivor_first_protection: float = 1.45
    omni_non_discriminatory_ci: float = 1.40
    omni_bio_threat_vigilance: float = 1.38
    omni_pandemic_foresight: float = 1.42
    omni_humanitarian_ci_ethics: float = 1.40
    omni_insider_threat_mitigation: float = 1.35
    omni_foreign_penetration_defense: float = 1.37

    def to_dict(self) -> dict[str, int | float | complex]:
        """Convert scalars to dictionary format."""
        return {k: v for k, v in self.__dict__.items() if isinstance(v, (int, float, complex))}

    def apply_to_score(self, base_score: float, context: str = "default") -> float:
        """
        Apply ethical scalars to modify a base score based on context.

        Args:
            base_score: Original score before ethical adjustment
            context: Context for ethical consideration

        Returns:
            Ethically-adjusted score
        """
        if context == "harm_prevention":
            return base_score * self.omni_harm_prevention
        elif context == "benefit_promotion":
            return base_score * self.omni_benefit_promotion
        elif context == "fairness":
            return base_score * self.omni_fairness
        else:
            avg_ethical_boost = (
                self.omni_compassionate + self.omni_wisdom + self.omni_justitia
            ) / 3.0
            return base_score * avg_ethical_boost


@dataclass
class EngineConfig:
    """
    Main engine configuration including ethical scalars.

    Provides a unified configuration interface for the Mercury Agent ♱,
    incorporating both technical and ethical parameters.
    """

    ethical_scalars: EthicalScalars = field(default_factory=EthicalScalars)

    detection_threshold: float = 0.5
    fusion_mode: str = "hybrid"
    enable_neurosymbolic: bool = True
    enable_multiverse: bool = True
    enable_ci: bool = False

    num_universes: int = 10
    multiverse_state_dim: int = 50
    multiverse_convergence_threshold: float = 0.95

    quantum_num_qubits: int = 8
    quantum_entanglement_strength: float = 0.3

    astrophysical_mass_equivalent: float = 1.0
    astrophysical_speed_of_light: float = 1.0
    astrophysical_gravitational_constant: float = 1.0

    def get_model_config(self, model_name: str) -> dict[str, Any]:
        """Get configuration specific to a model."""
        configs = {
            "quantum": {
                "num_qubits": self.quantum_num_qubits,
                "entanglement_strength": self.quantum_entanglement_strength,
            },
            "astrophysical": {
                "mass_equivalent": self.astrophysical_mass_equivalent,
                "speed_of_light": self.astrophysical_speed_of_light,
                "gravitational_constant": self.astrophysical_gravitational_constant,
            },
            "multiverse": {
                "num_universes": self.num_universes,
                "state_dim": self.multiverse_state_dim,
                "convergence_threshold": self.multiverse_convergence_threshold,
            },
        }
        return configs.get(model_name, {})

    def apply_ethical_framework(self) -> dict[str, Any]:
        """
        Generate a complete ethical framework report.

        Returns:
            Dictionary containing ethical scalars and their interpretations
        """
        scalars_dict = self.ethical_scalars.to_dict()

        return {
            "scalars": scalars_dict,
            "framework_version": "1.0",
            "source": "Mercury-Agent Ethical Framework",
            "principles": [
                "Compassion - Prioritizing well-being and harm minimization",
                "Evidence - Requiring verifiable data and mathematical proofs",
                "Justice - Ensuring fair, unbiased operations",
                "Altruism - Promoting positive societal impact",
                "Control - Maintaining human agency and oversight",
                "Character - Building trust through consistent ethical behavior",
                "Competence - Maintaining high standards of technical excellence",
                "Commitment - Long-term dedication to beneficial outcomes",
            ],
            "system_integrity": [
                OmniCodes.OMNI_DIRECTIONAL.code,
                OmniCodes.OMNI_INDIVISIBLE.code,
                OmniCodes.OMNI_PERCIPIENT.code,
                OmniCodes.OMNI_BENEVOLENT.code,
                OmniCodes.OMNI_UNIVERSAL.code,
                OmniCodes.OMNI_SCIENT.code,
                OmniCodes.OMNI_POTENT.code,
            ],
        }


DEFAULT_CONFIG = EngineConfig()
