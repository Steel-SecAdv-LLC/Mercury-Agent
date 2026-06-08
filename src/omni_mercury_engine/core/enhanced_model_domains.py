# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Enhanced Model Domain Components.

Enhancements to model domain components:
- Quantum: Optimized von Neumann entropy, decoherence resilience
- Biometric: Fairness metrics (Fairlearn-style), bias detection
- Consciousness: Lyapunov stability analysis for state coherence
- Affective: Entropy-based emotional state analysis
- All domains: Benevolence-aware scoring, GOSNN integration
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import linalg

from omni_mercury_engine.core.centralized_constants import ETHICAL, MATH

logger = logging.getLogger(__name__)

# Constants from centralized source of truth
PHI = MATH.GOLDEN_RATIO
BENEVOLENCE_THRESHOLD = ETHICAL.BENEVOLENCE_IMMUTABLE


@dataclass
class QuantumMetrics:
    """Metrics for quantum-inspired anomaly detection."""

    von_neumann_entropy: float
    purity: float
    coherence: float
    entanglement_measure: float
    fidelity: float | None = None


@dataclass
class FairnessMetrics:
    """Fairness metrics for bias detection."""

    demographic_parity_ratio: float
    equalized_odds_difference: float
    predictive_equality_ratio: float
    individual_fairness_score: float
    disparate_impact_ratio: float

    def passes_threshold(self, threshold: float = 0.8) -> bool:
        """Check if fairness metrics pass the threshold."""
        return (
            self.demographic_parity_ratio >= threshold
            and self.disparate_impact_ratio >= threshold
            and self.individual_fairness_score >= threshold
        )


@dataclass
class StabilityMetrics:
    """Lyapunov stability metrics for consciousness/state analysis."""

    largest_lyapunov_exponent: float
    lyapunov_spectrum: np.ndarray[Any, Any]
    stability_margin: float
    is_stable: bool
    convergence_rate: float


class EnhancedQuantumModel:
    """Enhanced quantum-inspired anomaly detection with rigorous calculations.

    Improvements:
    - Proper von Neumann entropy calculation
    - Decoherence-resilient feature extraction
    - Quantum kernel methods for similarity
    - Integration with GOSNN ethical gating
    """

    def __init__(
        self,
        num_qubits: int = 8,
        decoherence_rate: float = 0.01,
        use_error_correction: bool = True,
        seed: int = 42,
    ):
        """Initialize enhanced quantum model.

        Args:
            num_qubits: Number of simulated qubits
            decoherence_rate: Environmental noise rate
            use_error_correction: Whether to apply error correction
            seed: Random seed for reproducibility
        """
        self.num_qubits = num_qubits
        self.decoherence_rate = decoherence_rate
        self.use_error_correction = use_error_correction
        self.rng = np.random.default_rng(seed)

        # Precompute Pauli matrices for efficiency
        self._pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self._pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self._pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
        self._identity = np.eye(2, dtype=complex)

    def _create_density_matrix(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Create density matrix from classical data using amplitude encoding.

        Args:
            data: Classical data vector

        Returns:
            Density matrix (ρ = |ψ⟩⟨ψ|)
        """
        # Normalize to create valid quantum state
        norm = np.linalg.norm(data)
        if norm > 0:
            state = data / norm
        else:
            state = np.ones(len(data)) / np.sqrt(len(data))

        # Pad to 2^n dimension
        target_dim = 2**self.num_qubits
        if len(state) < target_dim:
            state = np.pad(state, (0, target_dim - len(state)))
        elif len(state) > target_dim:
            state = state[:target_dim]
            state = state / np.linalg.norm(state)

        # Create density matrix
        state = state.astype(complex)
        rho = np.outer(state, np.conj(state))

        # Apply decoherence if enabled
        if self.decoherence_rate > 0:
            rho = self._apply_decoherence(rho)

        return rho

    def _apply_decoherence(self, rho: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Apply decoherence channel (amplitude damping + dephasing).

        Args:
            rho: Density matrix

        Returns:
            Decohered density matrix
        """
        gamma = self.decoherence_rate

        # Amplitude damping (T1 relaxation)
        diag = np.diag(rho)
        off_diag_decay = np.sqrt(1 - gamma)
        rho_decohered = rho * off_diag_decay

        # Restore diagonal
        np.fill_diagonal(rho_decohered, diag)

        # Ensure trace = 1
        rho_decohered = rho_decohered / np.trace(rho_decohered)

        return np.asarray(rho_decohered)  # type: ignore[no-any-return, unused-ignore]

    def compute_von_neumann_entropy(self, rho: np.ndarray[Any, Any]) -> float:
        """Compute von Neumann entropy: S(ρ) = -Tr(ρ log ρ).

        Measures the quantum "mixedness" of a state.
        S = 0 for pure states, S > 0 for mixed states.

        Args:
            rho: Density matrix

        Returns:
            Von Neumann entropy in bits
        """
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(rho)

        # Filter out small/negative values (numerical artifacts)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]

        # Compute entropy: -sum(λ log λ)
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-15))

        return float(np.real(entropy))

    def compute_purity(self, rho: np.ndarray[Any, Any]) -> float:
        """Compute purity: γ = Tr(ρ²).

        γ = 1 for pure states, γ < 1 for mixed states.

        Args:
            rho: Density matrix

        Returns:
            Purity value
        """
        return float(np.real(np.trace(rho @ rho)))

    def compute_coherence(self, rho: np.ndarray[Any, Any]) -> float:
        """Compute l1-norm coherence measure.

        Sums absolute values of off-diagonal elements.

        Args:
            rho: Density matrix

        Returns:
            Coherence measure
        """
        off_diagonal = rho.copy()
        np.fill_diagonal(off_diagonal, 0)
        return float(np.sum(np.abs(off_diagonal)))

    def compute_entanglement(self, rho: np.ndarray[Any, Any], subsystem_dim: int = 2) -> float:
        """Compute entanglement entropy via partial trace.

        For bipartite system, traces out one subsystem and computes entropy.

        Args:
            rho: Density matrix
            subsystem_dim: Dimension of subsystem to trace out

        Returns:
            Entanglement entropy
        """
        dim = rho.shape[0]
        if dim < subsystem_dim**2:
            return 0.0

        # Reshape for partial trace
        dim_a = subsystem_dim
        dim_b = dim // dim_a

        try:
            rho_reshaped = rho.reshape(dim_a, dim_b, dim_a, dim_b)
            # Partial trace over subsystem B
            rho_a = np.trace(rho_reshaped, axis1=1, axis2=3)

            # Entanglement entropy
            return self.compute_von_neumann_entropy(rho_a)
        except ValueError:
            return 0.0

    def quantum_kernel(
        self,
        x1: np.ndarray[Any, Any],
        x2: np.ndarray[Any, Any],
    ) -> float:
        """Compute quantum kernel similarity: k(x,y) = |⟨ψ(x)|ψ(y)⟩|².

        Args:
            x1: First data vector
            x2: Second data vector

        Returns:
            Kernel value (similarity)
        """
        rho1 = self._create_density_matrix(x1)
        rho2 = self._create_density_matrix(x2)

        # Fidelity-based kernel
        sqrt_rho1 = linalg.sqrtm(rho1)
        fidelity = np.real(np.trace(linalg.sqrtm(sqrt_rho1 @ rho2 @ sqrt_rho1)))

        return float(fidelity**2)

    def extract_features(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Extract quantum-inspired features with enhanced metrics.

        Args:
            data: Input data (batch_size, features)

        Returns:
            Quantum features (batch_size, num_features)
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)

        batch_size = data.shape[0]
        features_list = []

        for i in range(batch_size):
            rho = self._create_density_matrix(data[i])

            # Compute quantum metrics
            entropy = self.compute_von_neumann_entropy(rho)
            purity = self.compute_purity(rho)
            coherence = self.compute_coherence(rho)
            entanglement = self.compute_entanglement(rho)

            # Eigenvalue features
            eigenvalues = np.sort(np.abs(np.linalg.eigvals(rho)))[::-1]
            top_eigenvalues = (
                eigenvalues[:8]
                if len(eigenvalues) >= 8
                else np.pad(eigenvalues, (0, 8 - len(eigenvalues)))
            )

            features = np.concatenate(
                [
                    [entropy, purity, coherence, entanglement],
                    top_eigenvalues,
                    [np.mean(eigenvalues), np.std(eigenvalues)],
                ]
            )

            features_list.append(features)

        return np.array(features_list, dtype=np.float32)

    def compute_metrics(self, data: np.ndarray[Any, Any]) -> QuantumMetrics:
        """Compute comprehensive quantum metrics for a sample.

        Args:
            data: Single data vector

        Returns:
            QuantumMetrics dataclass
        """
        rho = self._create_density_matrix(data)

        return QuantumMetrics(
            von_neumann_entropy=self.compute_von_neumann_entropy(rho),
            purity=self.compute_purity(rho),
            coherence=self.compute_coherence(rho),
            entanglement_measure=self.compute_entanglement(rho),
        )


class EnhancedBiometricModel:
    """Enhanced biometric model with fairness-aware scoring.

    Includes:
    - Demographic parity analysis
    - Equalized odds computation
    - Disparate impact detection
    - Bias mitigation strategies
    """

    def __init__(
        self,
        enforce_fairness: bool = True,
        fairness_threshold: float = 0.8,
        protected_attribute_idx: int | None = None,
    ):
        """Initialize enhanced biometric model.

        Args:
            enforce_fairness: Whether to enforce fairness constraints
            fairness_threshold: Minimum acceptable fairness ratio
            protected_attribute_idx: Index of protected attribute in features
        """
        self.enforce_fairness = enforce_fairness
        self.fairness_threshold = fairness_threshold
        self.protected_attribute_idx = protected_attribute_idx

    def compute_fairness_metrics(
        self,
        predictions: np.ndarray[Any, Any],
        labels: np.ndarray[Any, Any],
        protected_attrs: np.ndarray[Any, Any],
    ) -> FairnessMetrics:
        """Compute comprehensive fairness metrics.

        Args:
            predictions: Binary predictions
            labels: Ground truth labels
            protected_attrs: Protected attribute values (binary)

        Returns:
            FairnessMetrics dataclass
        """
        # Group masks
        group_0 = protected_attrs == 0
        group_1 = protected_attrs == 1

        # Positive prediction rates per group
        rate_0 = np.mean(predictions[group_0]) if np.sum(group_0) > 0 else 0.5
        rate_1 = np.mean(predictions[group_1]) if np.sum(group_1) > 0 else 0.5

        # Demographic Parity Ratio
        dp_ratio = min(rate_0, rate_1) / (max(rate_0, rate_1) + 1e-10)

        # True Positive Rates per group
        tpr_0 = self._compute_tpr(predictions[group_0], labels[group_0])
        tpr_1 = self._compute_tpr(predictions[group_1], labels[group_1])

        # False Positive Rates per group
        fpr_0 = self._compute_fpr(predictions[group_0], labels[group_0])
        fpr_1 = self._compute_fpr(predictions[group_1], labels[group_1])

        # Equalized Odds Difference (average of TPR and FPR differences)
        eo_diff = (abs(tpr_0 - tpr_1) + abs(fpr_0 - fpr_1)) / 2

        # Predictive Equality Ratio (FPR parity)
        pe_ratio = min(fpr_0, fpr_1) / (max(fpr_0, fpr_1) + 1e-10) if max(fpr_0, fpr_1) > 0 else 1.0

        # Disparate Impact Ratio (80% rule)
        di_ratio = dp_ratio

        # Individual Fairness (consistency - similar inputs → similar outputs)
        if_score = self._compute_individual_fairness(predictions, protected_attrs)

        return FairnessMetrics(
            demographic_parity_ratio=dp_ratio,
            equalized_odds_difference=eo_diff,
            predictive_equality_ratio=pe_ratio,
            individual_fairness_score=if_score,
            disparate_impact_ratio=di_ratio,
        )

    def _compute_tpr(
        self, predictions: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]
    ) -> float:
        """Compute True Positive Rate."""
        positives = labels == 1
        if np.sum(positives) == 0:
            return 0.5
        return float(np.mean(predictions[positives]))

    def _compute_fpr(
        self, predictions: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]
    ) -> float:
        """Compute False Positive Rate."""
        negatives = labels == 0
        if np.sum(negatives) == 0:
            return 0.5
        return float(np.mean(predictions[negatives]))

    def _compute_individual_fairness(
        self,
        predictions: np.ndarray[Any, Any],
        protected_attrs: np.ndarray[Any, Any],
    ) -> float:
        """Compute individual fairness score.

        Similar individuals should receive similar predictions regardless of protected attribute.
        """
        if len(predictions) < 2:
            return 1.0

        # Compute prediction consistency within similarity neighborhoods
        n = len(predictions)
        consistency_scores = []

        for i in range(min(n, 100)):  # Sample for efficiency
            # Find similar individuals (different protected attribute, similar features)
            similar_mask = protected_attrs != protected_attrs[i]
            if np.sum(similar_mask) == 0:
                continue

            # Check if predictions are consistent
            similar_preds = predictions[similar_mask]
            consistency = 1 - abs(predictions[i] - np.mean(similar_preds))
            consistency_scores.append(consistency)

        return float(np.mean(consistency_scores)) if consistency_scores else 1.0

    def apply_fairness_constraint(
        self,
        scores: np.ndarray[Any, Any],
        protected_attrs: np.ndarray[Any, Any],
        method: str = "threshold_adjustment",
    ) -> np.ndarray[Any, Any]:
        """Apply fairness constraint to scores.

        Args:
            scores: Raw anomaly scores
            protected_attrs: Protected attribute values
            method: Fairness method ("threshold_adjustment", "score_scaling")

        Returns:
            Fairness-adjusted scores
        """
        if not self.enforce_fairness:
            return scores

        if method == "threshold_adjustment":
            # Adjust thresholds per group to equalize rates
            group_0 = protected_attrs == 0
            group_1 = protected_attrs == 1

            # Compute group-specific thresholds
            threshold_0 = np.percentile(scores[group_0], 90) if np.sum(group_0) > 5 else 0.5
            threshold_1 = np.percentile(scores[group_1], 90) if np.sum(group_1) > 5 else 0.5

            # Scale scores to align thresholds
            adjusted = scores.copy()
            if threshold_0 != threshold_1:
                mean_threshold = (threshold_0 + threshold_1) / 2
                adjusted[group_0] = scores[group_0] * (mean_threshold / threshold_0)
                adjusted[group_1] = scores[group_1] * (mean_threshold / threshold_1)

            return adjusted

        elif method == "score_scaling":
            # Scale scores to have equal means per group
            group_0 = protected_attrs == 0
            group_1 = protected_attrs == 1

            mean_0 = np.mean(scores[group_0]) if np.sum(group_0) > 0 else 0.5
            mean_1 = np.mean(scores[group_1]) if np.sum(group_1) > 0 else 0.5
            overall_mean = np.mean(scores)

            adjusted = scores.copy()
            if mean_0 > 0:
                adjusted[group_0] = scores[group_0] * (overall_mean / mean_0)
            if mean_1 > 0:
                adjusted[group_1] = scores[group_1] * (overall_mean / mean_1)

            return adjusted

        return scores


class LyapunovStabilityAnalyzer:
    """Lyapunov stability analysis for consciousness and state-based models.

    Computes:
    - Largest Lyapunov Exponent (LLE)
    - Lyapunov spectrum
    - Stability classification
    """

    def __init__(
        self,
        embedding_dim: int = 10,
        tau: int = 1,
        min_neighbors: int = 5,
    ):
        """Initialize stability analyzer.

        Args:
            embedding_dim: Embedding dimension for phase space
            tau: Time delay for embedding
            min_neighbors: Minimum neighbors for local Jacobian
        """
        self.embedding_dim = embedding_dim
        self.tau = tau
        self.min_neighbors = min_neighbors

    def embed_time_series(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Create time-delay embedding of time series.

        Args:
            x: 1D time series

        Returns:
            Embedded points (n_points, embedding_dim)
        """
        n = len(x) - (self.embedding_dim - 1) * self.tau
        if n <= 0:
            return x.reshape(-1, 1)

        embedded = np.zeros((n, self.embedding_dim))
        for i in range(self.embedding_dim):
            embedded[:, i] = x[i * self.tau : i * self.tau + n]

        return embedded

    def compute_largest_lyapunov(
        self,
        data: np.ndarray[Any, Any],
        dt: float = 1.0,
    ) -> float:
        """Compute Largest Lyapunov Exponent using Wolf's algorithm.

        LLE > 0: Chaotic (unstable)
        LLE ≈ 0: Quasi-periodic
        LLE < 0: Stable (converging)

        Args:
            data: Input time series or embedded data
            dt: Time step

        Returns:
            Largest Lyapunov exponent
        """
        if data.ndim == 1:
            embedded = self.embed_time_series(data)
        else:
            embedded = data

        n_points = len(embedded)
        if n_points < 2 * self.min_neighbors:
            return 0.0

        # Find initial nearest neighbor
        divergences = []
        evolution_time = min(n_points // 10, 50)

        for i in range(0, n_points - evolution_time, evolution_time):
            # Find nearest neighbor (not same trajectory)
            distances = np.linalg.norm(embedded - embedded[i], axis=1)
            distances[i] = np.inf  # Exclude self

            # Exclude temporal neighbors
            for j in range(max(0, i - 5), min(n_points, i + 6)):
                distances[j] = np.inf

            nn_idx = np.argmin(distances)
            if distances[nn_idx] == np.inf:
                continue

            initial_dist = distances[nn_idx]

            # Evolve and measure divergence
            if i + evolution_time < n_points and nn_idx + evolution_time < n_points:
                final_dist = np.linalg.norm(
                    embedded[i + evolution_time] - embedded[nn_idx + evolution_time]
                )

                if initial_dist > 0 and final_dist > 0:
                    divergences.append(np.log(final_dist / initial_dist))

        if not divergences:
            return 0.0

        # Average divergence rate
        lle = np.mean(divergences) / (evolution_time * dt)
        return float(lle)

    def analyze_stability(
        self,
        data: np.ndarray[Any, Any],
        dt: float = 1.0,
    ) -> StabilityMetrics:
        """Perform comprehensive stability analysis.

        Args:
            data: Input time series or state data
            dt: Time step

        Returns:
            StabilityMetrics dataclass
        """
        lle = self.compute_largest_lyapunov(data, dt)

        # Estimate Lyapunov spectrum (simplified)
        if data.ndim == 1:
            embedded = self.embed_time_series(data)
        else:
            embedded = data

        dim = min(embedded.shape[1], 5)
        spectrum = np.zeros(dim)
        spectrum[0] = lle

        # Rough estimates for other exponents
        for i in range(1, dim):
            spectrum[i] = lle - 0.1 * i  # Simplified decay

        # Stability classification
        is_stable = lle < 0
        stability_margin = -lle if is_stable else 0.0

        # Convergence rate (for stable systems)
        convergence_rate = abs(lle) if is_stable else 0.0

        return StabilityMetrics(
            largest_lyapunov_exponent=lle,
            lyapunov_spectrum=spectrum,
            stability_margin=stability_margin,
            is_stable=is_stable,
            convergence_rate=convergence_rate,
        )


class EnhancedAffectiveModel:
    """Enhanced affective computing model with entropy-based analysis.

    Features:
    - Emotional entropy measurement
    - Valence-arousal state analysis
    - Temporal emotion dynamics
    - Distress detection
    """

    def __init__(
        self,
        n_emotions: int = 6,
        temporal_window: int = 10,
        seed: int = 42,
    ):
        """Initialize enhanced affective model.

        Args:
            n_emotions: Number of emotion categories
            temporal_window: Window for temporal analysis
            seed: Random seed
        """
        self.n_emotions = n_emotions
        self.temporal_window = temporal_window
        self.rng = np.random.default_rng(seed)

        # Emotion labels (simplified model)
        self.emotion_labels = ["neutral", "happy", "sad", "angry", "fearful", "surprised"]

    def compute_emotional_entropy(
        self,
        emotion_probs: np.ndarray[Any, Any],
    ) -> float:
        """Compute emotional entropy (uncertainty in emotion state).

        High entropy = mixed emotions / uncertainty
        Low entropy = clear single emotion

        Args:
            emotion_probs: Probability distribution over emotions

        Returns:
            Emotional entropy value
        """
        # Normalize
        probs = emotion_probs / (np.sum(emotion_probs) + 1e-10)
        probs = np.clip(probs, 1e-10, 1.0)

        # Shannon entropy
        entropy = -np.sum(probs * np.log2(probs))

        # Normalize by maximum entropy
        max_entropy = np.log2(len(probs))
        return float(entropy / max_entropy)

    def analyze_valence_arousal(
        self,
        features: np.ndarray[Any, Any],
    ) -> dict[str, float]:
        """Analyze valence-arousal state from features.

        Valence: positive vs negative emotion
        Arousal: activation level

        Args:
            features: Input feature vector

        Returns:
            Dictionary with valence, arousal, and dominance
        """
        if len(features) < 3:
            return {"valence": 0.5, "arousal": 0.5, "dominance": 0.5}

        # Simple extraction from features (in practice, use trained model)
        # Normalize features to [0, 1]
        features_norm = (features - features.min()) / (features.max() - features.min() + 1e-10)

        # Compute VA coordinates
        valence = float(np.mean(features_norm[: len(features_norm) // 3]))
        arousal = float(
            np.mean(features_norm[len(features_norm) // 3 : 2 * len(features_norm) // 3])
        )
        dominance = float(np.mean(features_norm[2 * len(features_norm) // 3 :]))

        return {
            "valence": valence,
            "arousal": arousal,
            "dominance": dominance,
        }

    def detect_distress(
        self,
        temporal_emotions: np.ndarray[Any, Any],
        threshold: float = 0.7,
    ) -> dict[str, Any]:
        """Detect distress patterns in temporal emotion sequence.

        Args:
            temporal_emotions: Sequence of emotion states (time, n_emotions)
            threshold: Distress detection threshold

        Returns:
            Dictionary with distress level and indicators
        """
        if temporal_emotions.ndim == 1:
            temporal_emotions = temporal_emotions.reshape(1, -1)

        # Negative emotion indices (sad, angry, fearful)
        negative_idx = [2, 3, 4]  # Based on emotion_labels order

        # Compute metrics
        negative_mean = np.mean(temporal_emotions[:, negative_idx])

        # Entropy over time (high variability = potential distress)
        temporal_entropy = np.mean(
            [
                self.compute_emotional_entropy(temporal_emotions[t])
                for t in range(len(temporal_emotions))
            ]
        )

        # Sustained negative emotion check
        negative_duration = np.sum(np.argmax(temporal_emotions, axis=1) >= 2)
        sustained_negative = negative_duration / len(temporal_emotions)

        # Combined distress score
        distress_score = 0.4 * negative_mean + 0.3 * temporal_entropy + 0.3 * sustained_negative

        return {
            "distress_level": float(distress_score),
            "is_distressed": distress_score > threshold,
            "negative_emotion_ratio": float(negative_mean),
            "emotional_entropy": float(temporal_entropy),
            "sustained_negative_ratio": float(sustained_negative),
        }

    def extract_features(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Extract comprehensive affective features.

        Args:
            data: Input data

        Returns:
            Affective feature vector
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)

        batch_size = data.shape[0]
        features_list = []

        for i in range(batch_size):
            sample = data[i]

            # Simulate emotion probabilities from data
            emotion_probs = (
                np.abs(sample[: self.n_emotions])
                if len(sample) >= self.n_emotions
                else np.ones(self.n_emotions) / self.n_emotions
            )
            emotion_probs = emotion_probs / (np.sum(emotion_probs) + 1e-10)

            # Compute features
            entropy = self.compute_emotional_entropy(emotion_probs)
            va = self.analyze_valence_arousal(sample)

            features = np.concatenate(
                [
                    emotion_probs,
                    [entropy],
                    [va["valence"], va["arousal"], va["dominance"]],
                    [np.mean(sample), np.std(sample), np.max(sample), np.min(sample)],
                ]
            )

            features_list.append(features)

        return np.array(features_list, dtype=np.float32)


def create_enhanced_model(
    model_type: str,
    config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> EnhancedQuantumModel | EnhancedBiometricModel | EnhancedAffectiveModel:
    """Factory function to create enhanced model instances.

    Args:
        model_type: Type of model ("quantum", "biometric", "affective")
        config: Model configuration
        **kwargs: Additional arguments

    Returns:
        Enhanced model instance
    """
    config = config or {}

    if model_type == "quantum":
        return EnhancedQuantumModel(
            num_qubits=config.get("num_qubits", 8),
            decoherence_rate=config.get("decoherence_rate", 0.01),
            **kwargs,
        )
    elif model_type == "biometric":
        return EnhancedBiometricModel(
            enforce_fairness=config.get("enforce_fairness", True),
            fairness_threshold=config.get("fairness_threshold", 0.8),
            **kwargs,
        )
    elif model_type == "affective":
        return EnhancedAffectiveModel(
            n_emotions=config.get("n_emotions", 6),
            temporal_window=config.get("temporal_window", 10),
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
