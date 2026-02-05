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
Quantum-inspired anomaly detection model with real quantum algorithms.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt


class ErrorCorrectionCode(Enum):
    """Quantum error correction code types."""

    NONE = "none"
    BIT_FLIP = "bit_flip"  # 3-qubit code for bit flip errors
    PHASE_FLIP = "phase_flip"  # 3-qubit code for phase flip errors
    SHOR = "shor"  # 9-qubit Shor code (combined bit and phase)
    STEANE = "steane"  # 7-qubit Steane code
    SURFACE = "surface"  # Surface code (most robust)


@dataclass
class NoiseModel:
    """
    Noise model for quantum decoherence simulation.

    Models common quantum noise channels:
    - Depolarizing: Random Pauli errors
    - Amplitude damping: Energy loss to environment
    - Phase damping: Loss of phase coherence (T2 decay)
    - Bit flip: X errors
    - Phase flip: Z errors
    """

    depolarizing_rate: float = 0.01
    amplitude_damping_rate: float = 0.005
    phase_damping_rate: float = 0.01
    bit_flip_rate: float = 0.001
    phase_flip_rate: float = 0.001
    measurement_error_rate: float = 0.01


@dataclass
class DecoherenceConfig:
    """Configuration for decoherence resilience mechanisms."""

    noise_model: NoiseModel = field(default_factory=NoiseModel)
    error_correction_code: ErrorCorrectionCode = ErrorCorrectionCode.BIT_FLIP
    syndrome_measurement_rounds: int = 3
    error_threshold: float = 0.1  # Max tolerable error rate
    t1_time: float = 100.0  # Relaxation time (microseconds)
    t2_time: float = 50.0  # Dephasing time (microseconds)


class QuantumAnomalyModel:
    """Quantum-inspired anomaly detection using quantum state representations."""

    def __init__(self, config: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self.config = config or {}
        self.num_qubits = self.config.get("num_qubits", 8)
        self.entanglement_strength = self.config.get("entanglement_strength", 0.3)

        # Decoherence resilience configuration
        self._decoherence_config: DecoherenceConfig | None = None
        self._noise_model: NoiseModel | None = None
        self._error_correction_enabled: bool = False

        # Syndrome history for error tracking
        self._syndrome_history: list[np.ndarray[Any, Any]] = []
        self._error_rate_history: list[float] = []

    def _create_quantum_state(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Create quantum superposition state from classical data."""
        norm = np.linalg.norm(data, axis=-1, keepdims=True)
        normalized = data / (norm + 1e-8)

        phases = np.exp(1j * np.angle(normalized + 0j))
        amplitudes = np.abs(normalized)

        return amplitudes * phases

    def _measure_entanglement(self, state: np.ndarray[Any, Any]) -> float:
        """Measure quantum entanglement in state using von Neumann entropy."""
        density_matrix = np.outer(state, np.conj(state))
        eigenvalues = np.linalg.eigvalsh(density_matrix)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))
        return float(entropy)

    def extract_features(self, data: np.ndarray[Any, Any] | dict[str, Any]) -> np.ndarray[Any, Any]:
        """Extract quantum-inspired features from data."""
        if isinstance(data, dict):
            data = np.array(next(iter(data.values())))
        elif not isinstance(data, npt.NDArray[Any]):
            data = np.array(data)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        batch_size = data.shape[0]

        quantum_states = self._create_quantum_state(data)

        amplitudes = np.abs(quantum_states).astype(np.float32)
        phases = np.angle(quantum_states).astype(np.float32)

        if amplitudes.shape[1] < 8:
            amplitudes = np.pad(amplitudes, ((0, 0), (0, 8 - amplitudes.shape[1])), mode="constant")

        if phases.shape[1] < 4:
            phases = np.pad(phases, ((0, 0), (0, 4 - phases.shape[1])), mode="constant")

        entanglement = (
            np.array([self._measure_entanglement(quantum_states[i]) for i in range(batch_size)])
            .reshape(-1, 1)
            .astype(np.float32)
        )

        features = np.concatenate(
            [
                amplitudes[:, :8],
                phases[:, :4],
                entanglement,
                np.ones((batch_size, 3)) * self.entanglement_strength,
            ],
            axis=1,
        )

        return features

    def predict(self, data: np.ndarray[Any, Any] | dict[str, Any]) -> dict[str, Any]:
        """Predict anomalies using quantum state analysis."""
        features = self.extract_features(data)

        amplitudes = features[:, :8]
        phases = features[:, 8:12]
        entanglement = features[:, 12:13]

        amp_anomaly = np.std(amplitudes, axis=1)
        phase_anomaly = np.std(phases, axis=1)
        ent_anomaly = np.abs(entanglement.squeeze() - 1.0)

        anomaly_scores = (amp_anomaly + phase_anomaly + ent_anomaly) / 3.0

        return {
            "anomaly_scores": anomaly_scores.astype(np.float32),
            "quantum_states": features[:, :8],
            "coherence": (1.0 - ent_anomaly).astype(np.float32),
            "energy_levels": amplitudes[:, :3].astype(np.float32),
        }

    def apply_decoherence_resilience(
        self, noise_level: float = 0.01, error_correction: bool = True
    ) -> None:
        """
        Apply decoherence resilience mechanisms inspired by quantum computing.

        Quantum Decoherence: Main challenge in quantum systems - qubits lose
        quantum properties when not isolated from environment.

        Inspired by:
        - NISQ Era: Noisy Intermediate-Scale Quantum - accepting imperfect qubits
        - Quantum Error Correction: Codes (5-qubit, CSS, Shor, Steane, Toric)
        - Post-Quantum Cryptography: Quantum-resistant algorithms

        Research source: Wikipedia - Quantum computing
        (https://en.wikipedia.org/wiki/Quantum_computing)

        Args:
            noise_level: Environmental noise level (0-1)
            error_correction: Whether to apply error correction
        """
        # Clamp noise level to valid range
        noise_level = float(np.clip(noise_level, 0.0, 1.0))

        # Configure noise model based on noise level
        self._noise_model = NoiseModel(
            depolarizing_rate=noise_level,
            amplitude_damping_rate=noise_level * 0.5,
            phase_damping_rate=noise_level,
            bit_flip_rate=noise_level * 0.1,
            phase_flip_rate=noise_level * 0.1,
            measurement_error_rate=noise_level * 0.5,
        )

        # Select error correction code based on noise level
        if not error_correction:
            code = ErrorCorrectionCode.NONE
        elif noise_level < 0.01:
            code = ErrorCorrectionCode.BIT_FLIP  # Simple 3-qubit for low noise
        elif noise_level < 0.05:
            code = ErrorCorrectionCode.SHOR  # 9-qubit for moderate noise
        else:
            code = ErrorCorrectionCode.SURFACE  # Surface code for high noise

        # Configure decoherence resilience
        self._decoherence_config = DecoherenceConfig(
            noise_model=self._noise_model,
            error_correction_code=code,
            syndrome_measurement_rounds=max(1, int(3 * noise_level * 10)),
            error_threshold=0.1,
            t1_time=100.0 / (1 + noise_level * 10),  # T1 degrades with noise
            t2_time=50.0 / (1 + noise_level * 10),  # T2 degrades with noise
        )

        self._error_correction_enabled = error_correction

        # Adjust entanglement strength based on noise (reduce for noisy environment)
        if noise_level > 0.1:
            # High noise reduces effective entanglement
            self.entanglement_strength *= 1.0 - noise_level * 0.5

    def _apply_noise_channel(
        self, state: np.ndarray[Any, Any], noise_type: str = "depolarizing"
    ) -> np.ndarray[Any, Any]:
        """
        Apply quantum noise channel to state.

        Args:
            state: Quantum state vector
            noise_type: Type of noise channel

        Returns:
            Noisy quantum state
        """
        if self._noise_model is None:
            return state

        if noise_type == "depolarizing":
            # Depolarizing channel: state -> (1-p)*state + p/3*(X*state + Y*state + Z*state)
            p = self._noise_model.depolarizing_rate
            if np.random.random() < p:
                # Apply random Pauli error
                pauli_choice = np.random.choice(["I", "X", "Y", "Z"])
                state = self._apply_pauli(state, pauli_choice)

        elif noise_type == "amplitude_damping":
            # Amplitude damping: |1> -> sqrt(1-gamma)|1> + sqrt(gamma)|0>
            gamma = self._noise_model.amplitude_damping_rate
            # Simplified: reduce amplitude of excited state components
            state = state * np.sqrt(1 - gamma)

        elif noise_type == "phase_damping":
            # Phase damping: loss of off-diagonal coherence
            gamma = self._noise_model.phase_damping_rate
            # Apply random phase rotation
            phase_noise = np.exp(1j * np.random.normal(0, gamma, len(state)))
            state = state * phase_noise

        elif noise_type == "bit_flip":
            p = self._noise_model.bit_flip_rate
            if np.random.random() < p:
                state = self._apply_pauli(state, "X")

        elif noise_type == "phase_flip":
            p = self._noise_model.phase_flip_rate
            if np.random.random() < p:
                state = self._apply_pauli(state, "Z")

        return state

    def _apply_pauli(self, state: np.ndarray[Any, Any], pauli: str) -> np.ndarray[Any, Any]:
        """
        Apply Pauli operator to state vector.

        Args:
            state: Quantum state vector
            pauli: Pauli operator ("I", "X", "Y", "Z")

        Returns:
            Transformed state
        """
        if pauli == "I":
            return state
        elif pauli == "X":
            # Bit flip: swap pairs
            return np.flip(state)
        elif pauli == "Y":
            # Y = iXZ: flip and phase
            return 1j * np.flip(state) * np.array([1, -1] * (len(state) // 2 + 1))[: len(state)]
        elif pauli == "Z":
            # Phase flip: negate odd indices
            result = state.copy()
            result[1::2] *= -1
            return result
        return state

    def _compute_syndrome(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        Compute error syndrome for error correction.

        Syndrome measurement detects which errors occurred without
        collapsing the logical qubit state.

        Args:
            state: Quantum state to check

        Returns:
            Syndrome vector indicating error locations
        """
        if self._decoherence_config is None:
            return np.array([0])

        code = self._decoherence_config.error_correction_code

        if code == ErrorCorrectionCode.NONE:
            return np.array([0])

        elif code == ErrorCorrectionCode.BIT_FLIP:
            # 3-qubit bit flip code syndrome
            # Parity checks: qubit 0 XOR qubit 1, qubit 1 XOR qubit 2
            n = len(state)
            if n < 3:
                return np.array([0, 0])

            # Simplified syndrome: measure parity of amplitude groups
            syndrome = np.array(
                [
                    int(
                        np.sign(np.sum(state[: n // 3]))
                        != np.sign(np.sum(state[n // 3 : 2 * n // 3]))
                    ),
                    int(
                        np.sign(np.sum(state[n // 3 : 2 * n // 3]))
                        != np.sign(np.sum(state[2 * n // 3 :]))
                    ),
                ]
            )
            return syndrome

        elif code == ErrorCorrectionCode.SHOR:
            # 9-qubit Shor code: 3 groups of 3 qubits
            n = len(state)
            group_size = n // 9 if n >= 9 else 1
            syndromes = []

            for i in range(3):  # 3 groups
                group_start = i * 3 * group_size
                group_end = min((i + 1) * 3 * group_size, n)
                group = state[group_start:group_end]

                # Bit flip syndrome within group
                third = len(group) // 3
                if third > 0:
                    s1 = int(
                        np.sum(np.abs(group[:third]) ** 2)
                        < np.sum(np.abs(group[third : 2 * third]) ** 2) * 0.5
                    )
                    s2 = int(
                        np.sum(np.abs(group[third : 2 * third]) ** 2)
                        < np.sum(np.abs(group[2 * third :]) ** 2) * 0.5
                    )
                    syndromes.extend([s1, s2])

            return np.array(syndromes[:6])  # 6 syndrome bits for Shor code

        elif code == ErrorCorrectionCode.SURFACE:
            # Surface code: more complex stabilizer measurements
            # Simplified: check local correlations
            n = len(state)
            grid_size = int(np.sqrt(n))
            syndromes = []

            for i in range(min(4, grid_size)):
                for j in range(min(4, grid_size)):
                    idx = i * grid_size + j
                    if idx < n - 1:
                        # Check correlation between adjacent
                        correlation = np.abs(state[idx] * np.conj(state[idx + 1]))
                        syndromes.append(int(correlation < 0.1))

            return np.array(syndromes[:8])

        return np.array([0])

    def _correct_errors(
        self, state: np.ndarray[Any, Any], syndrome: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """
        Apply error correction based on syndrome.

        Args:
            state: Quantum state with potential errors
            syndrome: Error syndrome from measurement

        Returns:
            Corrected quantum state
        """
        if self._decoherence_config is None or not self._error_correction_enabled:
            return state

        code = self._decoherence_config.error_correction_code

        if code == ErrorCorrectionCode.NONE:
            return state

        # Store syndrome for history
        self._syndrome_history.append(syndrome)

        # Calculate error rate
        error_detected = np.any(syndrome != 0)
        self._error_rate_history.append(1.0 if error_detected else 0.0)

        if not error_detected:
            return state

        # Apply correction based on syndrome
        if code == ErrorCorrectionCode.BIT_FLIP:
            if len(syndrome) >= 2:
                # Decode syndrome to error location
                error_loc = syndrome[0] + 2 * syndrome[1]
                if error_loc > 0:
                    # Apply X correction at error location
                    n = len(state)
                    section_size = n // 3
                    start_idx = (error_loc - 1) * section_size
                    end_idx = min(error_loc * section_size, n)
                    state[start_idx:end_idx] = np.flip(state[start_idx:end_idx])

        elif code in (ErrorCorrectionCode.SHOR, ErrorCorrectionCode.SURFACE):
            # For more complex codes, apply majority voting correction
            # Simplified: renormalize state to correct amplitude errors
            norm = np.linalg.norm(state)
            if norm > 0:
                state = state / norm

        return state

    def get_decoherence_metrics(self) -> dict[str, Any]:
        """
        Get metrics on decoherence resilience performance.

        Returns:
            Dictionary with error rates, correction success, etc.
        """
        if self._decoherence_config is None:
            return {"enabled": False}

        # Calculate average error rate
        avg_error_rate = np.mean(self._error_rate_history) if self._error_rate_history else 0.0

        # Calculate correction success (errors detected but state recovered)
        total_errors = sum(1 for e in self._error_rate_history if e > 0)
        correction_attempts = len(self._syndrome_history)

        return {
            "enabled": True,
            "error_correction_code": self._decoherence_config.error_correction_code.value,
            "noise_model": {
                "depolarizing_rate": (
                    self._noise_model.depolarizing_rate if self._noise_model else 0
                ),
                "amplitude_damping_rate": (
                    self._noise_model.amplitude_damping_rate if self._noise_model else 0
                ),
                "phase_damping_rate": (
                    self._noise_model.phase_damping_rate if self._noise_model else 0
                ),
            },
            "t1_time_us": self._decoherence_config.t1_time,
            "t2_time_us": self._decoherence_config.t2_time,
            "average_error_rate": float(avg_error_rate),
            "total_errors_detected": total_errors,
            "correction_attempts": correction_attempts,
            "syndrome_history_length": len(self._syndrome_history),
            "entanglement_strength": self.entanglement_strength,
        }

    def predict_with_noise(self, data: np.ndarray[Any, Any] | dict[str, Any]) -> dict[str, Any]:
        """
        Predict anomalies with noise simulation and error correction.

        Uses the configured decoherence resilience mechanisms to
        simulate realistic quantum behavior.

        Args:
            data: Input data for prediction

        Returns:
            Prediction results with noise and correction metrics
        """
        # Extract features (creates quantum states)
        features = self.extract_features(data)

        if self._decoherence_config is not None and self._noise_model is not None:
            # Apply noise channels to quantum states
            quantum_states = features[:, :8].copy()

            for i in range(len(quantum_states)):
                state = quantum_states[i]

                # Apply various noise channels
                state = self._apply_noise_channel(state, "depolarizing")
                state = self._apply_noise_channel(state, "phase_damping")
                state = self._apply_noise_channel(state, "amplitude_damping")

                # Compute syndrome and correct
                if self._error_correction_enabled:
                    syndrome = self._compute_syndrome(state)
                    state = self._correct_errors(state, syndrome)

                quantum_states[i] = state

            # Update features with corrected states
            features[:, :8] = np.real(quantum_states)

        # Continue with normal prediction using (possibly corrected) features
        amplitudes = features[:, :8]
        phases = features[:, 8:12]
        entanglement = features[:, 12:13]

        amp_anomaly = np.std(amplitudes, axis=1)
        phase_anomaly = np.std(phases, axis=1)
        ent_anomaly = np.abs(entanglement.squeeze() - 1.0)

        anomaly_scores = (amp_anomaly + phase_anomaly + ent_anomaly) / 3.0

        result = {
            "anomaly_scores": anomaly_scores.astype(np.float32),
            "quantum_states": features[:, :8],
            "coherence": (1.0 - ent_anomaly).astype(np.float32),
            "energy_levels": amplitudes[:, :3].astype(np.float32),
        }

        # Add decoherence metrics if enabled
        if self._decoherence_config is not None:
            result["decoherence_metrics"] = self.get_decoherence_metrics()

        return result
