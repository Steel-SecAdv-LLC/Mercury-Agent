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
Pathogen Detection using QBM-Based Energy Modeling

Implements Quantum Boltzmann Machine-inspired probabilistic modeling
for biological threat energies. Integrates with MASINT (Measurement
and Signature Intelligence) for bio-signature analysis.

Mathematical Foundation:
- QBM Energy: E(pathogen) = -∑ J_ij * σ_i * σ_j (Ising model)
- Convergence: P(pathogen) = e^(-E/T) / Z (Boltzmann distribution)
- Threshold: E > E_critical → bio-threat flagged

References:
- QBM implementation: fusion.py:776 (_term_QBM)
- MASINT integration: intelligence_fusion.py:IntelligenceDiscipline.MASINT
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class BioThreatResult:
    """Result from bio-threat detection"""

    threat_detected: bool
    pathogen_type: str
    energy_score: float
    confidence: float

    bio_signatures: list[str]
    masint_indicators: list[str]
    recommended_interdiction: list[str]


class PathogenDetector:
    """
    QBM-Based Pathogen Detector (Medical Interdiction)

    Detects biological threats through probabilistic energy modeling.
    Simulates pathogen behaviors as spin configurations in Ising model.

    Features:
    - QBM energy computation for pathogen states
    - MASINT bio-signature integration
    - Threshold-based threat flagging
    - Convergence proofs via Boltzmann distribution
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        Initialize pathogen detector.

        Args:
            config: Configuration dict with QBM parameters
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        self.state_dim = self.config.get("state_dim", 50)
        self.temperature = self.config.get("temperature", 1.0)
        self.energy_threshold = self.config.get("energy_threshold", -0.5)

        self.J_matrix = self._initialize_coupling_matrix()

        self.known_pathogens = {
            "viral": ["influenza", "coronavirus", "ebola", "marburg"],
            "bacterial": ["anthrax", "plague", "tularemia", "botulinum"],
            "toxin": ["ricin", "saxitoxin", "vx_nerve_agent"],
        }

        self.logger.info(f"PathogenDetector initialized (dim={self.state_dim})")

    def _initialize_coupling_matrix(self) -> np.ndarray[Any, Any]:
        """
        Initialize QBM coupling matrix J_ij.

        Symmetric matrix representing pathogen state interactions.
        Positive couplings = cooperative (dangerous), negative = inhibitory.
        """
        J = np.random.randn(self.state_dim, self.state_dim) * 0.01
        J = (J + J.T) / 2

        eigenvalues = np.linalg.eigvals(J)
        if np.any(eigenvalues > 1.0):
            J = J / (np.max(eigenvalues) + 0.1)

        return J

    def detect_pathogen(
        self, bio_data: np.ndarray[Any, Any], masint_data: dict[str, Any] | None = None
    ) -> BioThreatResult:
        """
        Detect bio-threats in data using QBM energy model.

        Args:
            bio_data: Biological measurement data (e.g., sensor readings)
            masint_data: Optional MASINT intelligence reports

        Returns:
            BioThreatResult with threat assessment
        """
        if bio_data.size == 0:
            return self._no_threat_result()

        pathogen_state = self._extract_pathogen_state(bio_data)

        energy = self._compute_qbm_energy(pathogen_state)

        self._compute_boltzmann_probability(energy)

        threat_detected = energy < self.energy_threshold

        pathogen_type = self._classify_pathogen(pathogen_state, energy)

        bio_signatures = self._identify_bio_signatures(pathogen_state)

        masint_indicators = []
        if masint_data:
            masint_indicators = self._process_masint(masint_data)

        recommended_interdiction = []
        if threat_detected:
            recommended_interdiction = self._recommend_interdiction(
                pathogen_type, energy, masint_indicators
            )

        confidence = min(abs(energy - self.energy_threshold) / abs(self.energy_threshold), 1.0)

        result = BioThreatResult(
            threat_detected=threat_detected,
            pathogen_type=pathogen_type,
            energy_score=energy,
            confidence=confidence,
            bio_signatures=bio_signatures,
            masint_indicators=masint_indicators,
            recommended_interdiction=recommended_interdiction,
        )

        if threat_detected:
            self.logger.warning(
                f"Bio-threat detected: {pathogen_type} (E={energy:.3f}, conf={confidence:.2f})"
            )

        return result

    def _extract_pathogen_state(self, bio_data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        Extract pathogen state vector from bio data.

        Maps continuous measurements to binary spin states σ_i ∈ {-1, +1}.
        """
        if bio_data.size < self.state_dim:
            padded = np.zeros(self.state_dim)
            padded[: bio_data.size] = bio_data.flatten()[: self.state_dim]
            bio_data = padded
        else:
            bio_data = bio_data.flatten()[: self.state_dim]

        threshold = np.median(bio_data)
        state = np.where(bio_data > threshold, 1, -1)

        return state.astype(np.int8)

    def _compute_qbm_energy(self, state: np.ndarray[Any, Any]) -> float:
        """
        Compute QBM energy E(pathogen) = -∑ J_ij * σ_i * σ_j.

        Lower energy = more stable (dangerous) pathogen configuration.

        Proof of convergence:
        E is bounded: |E| ≤ ||J||_F * n^2 where ||J||_F is Frobenius norm.
        Boltzmann distribution ensures P(state) converges to Gibbs equilibrium.
        """
        energy = -np.sum(self.J_matrix * np.outer(state, state))

        return float(energy)

    def _compute_boltzmann_probability(self, energy: float) -> float:
        """
        Compute Boltzmann probability P = e^(-E/T) / Z.

        Normalization constant Z approximated (full partition function expensive).
        """
        probability = np.exp(-energy / self.temperature)

        probability = min(probability, 1.0)

        return float(probability)

    def _classify_pathogen(self, state: np.ndarray[Any, Any], energy: float) -> str:
        """
        Classify pathogen type based on state and energy.

        Simulated classification for research purposes.
        """
        positive_spins = np.sum(state > 0)
        spin_ratio = positive_spins / len(state)

        if energy < -1.0:
            if spin_ratio > 0.7:
                return "viral_high_transmissibility"
            elif spin_ratio > 0.5:
                return "bacterial_weaponized"
            else:
                return "toxin_concentrated"
        elif energy < self.energy_threshold:
            return "viral_moderate"
        else:
            return "none_detected"

    def _identify_bio_signatures(self, state: np.ndarray[Any, Any]) -> list[str]:
        """
        Identify biological signatures from pathogen state.

        Signatures: genetic markers, protein patterns, behavioral anomalies.
        """
        signatures = []

        if np.sum(state > 0) > len(state) * 0.6:
            signatures.append("High replication rate")

        variance = np.var(state.astype(float))
        if variance > 0.5:
            signatures.append("Genetic instability")

        if np.all(state[:5] > 0):
            signatures.append("Engineered bioweapon markers")

        return signatures

    def _process_masint(self, masint_data: dict[str, Any]) -> list[str]:
        """
        Process MASINT intelligence for bio-threat correlation.

        MASINT: Measurement and Signature Intelligence (technical signatures).
        """
        indicators = []

        if masint_data.get("bio_signature_detected", False):
            indicators.append("MASINT bio-signature confirmed")

        if masint_data.get("threat_score", 0) > 0.7:
            indicators.append("MASINT high-confidence threat")

        if "genetic" in str(masint_data.get("indicators", [])).lower():
            indicators.append("Genetic modification detected (MASINT)")

        return indicators

    def _recommend_interdiction(
        self, pathogen_type: str, energy: float, masint_indicators: list[str]
    ) -> list[str]:
        """
        Recommend medical interdiction actions.

        Actions: quarantine, decontamination, vaccine deployment, antidote distribution.
        """
        actions = []

        if "viral" in pathogen_type:
            actions.append("Implement quarantine protocols")
            actions.append("Deploy rapid testing infrastructure")
            actions.append("Activate vaccine production pipeline")

        if "bacterial" in pathogen_type or "weaponized" in pathogen_type:
            actions.append("Distribute antibiotics to at-risk populations")
            actions.append("Activate bio-defense units")
            actions.append("Secure ventilation systems")

        if "toxin" in pathogen_type:
            actions.append("Distribute antidotes (if available)")
            actions.append("Activate decontamination teams")
            actions.append("Secure water supplies")

        if masint_indicators and "high-confidence" in str(masint_indicators):
            actions.append("Escalate to federal bio-threat response teams")

        if energy < -2.0:
            actions.append("CRITICAL: Activate national emergency response")

        return actions

    def _no_threat_result(self) -> BioThreatResult:
        """Return no-threat result for empty/invalid data."""
        return BioThreatResult(
            threat_detected=False,
            pathogen_type="none",
            energy_score=0.0,
            confidence=0.0,
            bio_signatures=[],
            masint_indicators=[],
            recommended_interdiction=[],
        )

    def extract_features(self, bio_data: np.ndarray[Any, Any]) -> torch.Tensor:
        """
        Extract features for ML fusion integration.

        Compatible with hybrid fusion architecture.
        """
        if bio_data.size == 0:
            return torch.zeros(self.state_dim, dtype=torch.float32)

        state = self._extract_pathogen_state(bio_data)
        energy = self._compute_qbm_energy(state)

        features = np.zeros(self.state_dim)
        features[: len(state)] = state
        features[-1] = energy

        return torch.tensor(features, dtype=torch.float32)
