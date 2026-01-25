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
Sigma Directive detector implementing PCP, GSIS, RMD, and EOA protocols
Enhanced with quantum pattern containment and nano-scale detection
"""

import hashlib
from typing import Any

import numpy as np
import torch
from scipy.fft import fft

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.core.exceptions import DetectorException


class SigmaDirectiveDetector(BaseDetector):
    """
    Sigma Directive protocols for anomaly detection:
    - PCP (Pattern Convergence Protocol)
    - GSIS (Gravitational Stability Integrity System)
    - RMD (Recursive Memory Dynamics)
    - EOA (Ethical Oversight Amplifier)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.convergence_threshold = self.config.get("convergence_threshold", 0.01)
        self.stability_factor = self.config.get("stability_factor", 1.0)
        self.memory_depth = self.config.get("memory_depth", 5)
        self.use_quantum_enhanced = self.config.get("use_quantum_enhanced", True)
        self.use_nano_detection = self.config.get("use_nano_detection", True)
        self.use_harmonic_detection = self.config.get("use_harmonic_detection", True)

        self.baseline_pattern: np.ndarray[Any, Any] | None = None
        self.memory_buffer: list[Any] = []

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> SigmaDirectiveDetector:
        """Fit Sigma protocols to normal patterns"""
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        self.baseline_pattern = np.mean(data, axis=0)

        self._is_fitted = True
        return self

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Detect anomalies using Sigma protocols with optional auto-calibration.

        Implements multiple Sigma Directive protocols:
        - PCP (Pattern Convergence Protocol)
        - GSIS (Gravitational Stability Integrity System)
        - RMD (Recursive Memory Dynamics)
        - EOA (Ethical Oversight Amplifier)
        - Optional: Quantum Pattern Containment, Nano-Scale Detection, Harmonics

        Auto-Calibration:
            When auto_calibrate=True (via enable_auto_calibration()), the
            threshold is automatically calibrated based on the score
            distribution, solving the F1=0 problem.

        Returns:
            Dictionary containing:
                - is_anomaly: Boolean array of anomaly predictions
                - scores: Combined anomaly scores [0, 1]
                - pcp_scores, gsis_scores, rmd_scores, eoa_scores: Component scores
                - quantum_scores, nano_scores, harmonic_score: Enhanced scores
                - detector_type: "directive"
                - threshold: Effective threshold (may be calibrated)
                - calibration_diagnostics: Diagnostics if auto-calibrated
        """
        if not self._is_fitted:
            raise DetectorException("Detector must be fitted before detection")

        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        pcp_scores = self._pattern_convergence_protocol(data)
        gsis_scores = self._gravitational_stability_check(data)
        rmd_scores = self._recursive_memory_dynamics(data)
        eoa_scores = self._ethical_oversight_amplifier(data)

        quantum_scores = {}
        if self.use_quantum_enhanced:
            quantum_scores = self._quantum_pattern_containment(data)

        nano_scores = {}
        if self.use_nano_detection:
            nano_scores = self._nano_scale_detection(data)

        harmonic_score = 0.0
        if self.use_harmonic_detection:
            harmonic_score = self._harmonic_anomaly_detection(data)

        combined_scores = pcp_scores * 0.3 + gsis_scores * 0.3 + rmd_scores * 0.2 + eoa_scores * 0.2

        if quantum_scores:
            quantum_avg = np.mean(list(quantum_scores.values()))
            combined_scores = combined_scores * 0.8 + quantum_avg * 0.2

        if nano_scores:
            nano_avg = np.mean(list(nano_scores.values()))
            combined_scores = combined_scores * 0.85 + nano_avg * 0.15

        if harmonic_score > 0:
            combined_scores = combined_scores * 0.9 + harmonic_score * 0.1

        # Auto-calibration: compute optimal threshold from score distribution
        effective_threshold = self.threshold
        calibration_diagnostics = None

        if self._auto_calibrate:
            effective_threshold = self.calibrate_threshold(combined_scores)
            calibration_diagnostics = self._last_diagnostics

        is_anomaly = combined_scores > effective_threshold

        return {
            "is_anomaly": is_anomaly,
            "scores": combined_scores,
            "pcp_scores": pcp_scores,
            "gsis_scores": gsis_scores,
            "rmd_scores": rmd_scores,
            "eoa_scores": eoa_scores,
            "quantum_scores": quantum_scores,
            "nano_scores": nano_scores,
            "harmonic_score": harmonic_score,
            "detector_type": "directive",
            "threshold": effective_threshold,
            "calibration_diagnostics": calibration_diagnostics,
        }

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> torch.Tensor:
        """Extract Sigma protocol features for ML fusion"""
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        if not self._is_fitted:
            self.fit(data)

        pcp_scores = self._pattern_convergence_protocol(data)
        gsis_scores = self._gravitational_stability_check(data)
        rmd_scores = self._recursive_memory_dynamics(data)
        eoa_scores = self._ethical_oversight_amplifier(data)

        features = np.column_stack(
            [
                pcp_scores,
                gsis_scores,
                rmd_scores,
                eoa_scores,
                np.mean(data, axis=1),
                np.std(data, axis=1),
            ]
        )

        if features.shape[1] < 20:
            padding = np.zeros((features.shape[0], 20 - features.shape[1]))
            features = np.column_stack([features, padding])

        return torch.tensor(features, dtype=torch.float32)

    def _pattern_convergence_protocol(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """PCP: Detect pattern convergence anomalies.

        Returns continuous scores without hard clipping to preserve
        ranking information for downstream fusion models.

        Fix for Issue #7: No Score Continuity. Previously used
        np.minimum(..., 1.0) which capped scores, losing differentiation
        between extreme anomalies. Now uses soft normalization.
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        convergence_diffs = np.linalg.norm(data - self.baseline_pattern, axis=1)

        normalized_diffs = convergence_diffs / (np.linalg.norm(self.baseline_pattern) + 1e-6)

        # Soft normalization: x / (threshold + x) approaches 1 asymptotically
        # Preserves ordering while keeping scores in [0, 1) range
        scores = normalized_diffs / (self.convergence_threshold + normalized_diffs)

        return scores

    def _gravitational_stability_check(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """GSIS: Check gravitational stability (data distribution stability)"""
        if len(data) < 2:
            return np.zeros(len(data))

        scores = np.zeros(len(data))

        for i in range(len(data)):
            distances = np.linalg.norm(data - data[i], axis=1)
            local_density = np.sum(distances < np.percentile(distances, 20))

            stability = local_density / len(data)
            scores[i] = 1.0 - stability

        return scores * self.stability_factor

    def _recursive_memory_dynamics(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """RMD: Detect anomalies using recursive memory.

        Returns continuous scores without hard clipping.

        Fix for Issue #7: Uses soft normalization instead of min(deviation, 1.0).
        """
        scores = np.zeros(len(data))

        for i, sample in enumerate(data):
            self.memory_buffer.append(sample)
            if len(self.memory_buffer) > self.memory_depth:
                self.memory_buffer.pop(0)

            if len(self.memory_buffer) > 1:
                memory_mean = np.mean(self.memory_buffer, axis=0)
                deviation = np.linalg.norm(sample - memory_mean)
                # Soft normalization: deviation / (1 + deviation) for [0, 1) range
                scores[i] = deviation / (1.0 + deviation)

        return scores

    def _ethical_oversight_amplifier(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """EOA: Amplify detection of ethically significant anomalies"""
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        magnitude = np.linalg.norm(data, axis=1)
        magnitude_norm = magnitude / (np.max(magnitude) + 1e-6)

        return magnitude_norm

    def _quantum_pattern_containment(self, data: np.ndarray[Any, Any]) -> dict[str, float]:
        """
        Quantum Pattern Containment Protocol (QPCP)
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        normalized = data / (np.linalg.norm(data, axis=0, keepdims=True) + 1e-10)

        superposition_state = np.sum(normalized, axis=0) / len(normalized)

        coherence = np.abs(superposition_state)
        entanglement = np.std(normalized, axis=0)

        pattern_scores = {
            "coherence": float(np.mean(coherence)),
            "entanglement": float(np.mean(entanglement)),
            "superposition_strength": float(np.linalg.norm(superposition_state)),
        }

        return pattern_scores

    def _nano_scale_detection(self, data: np.ndarray[Any, Any]) -> dict[str, float]:
        """
        Nano-Scale Detection & Response System (NDRS)
        Enhanced N term with dimensional downsampling for micro-anomaly detection
        """
        if data.ndim == 1:
            data = data.reshape(-1)

        data_bytes = data.tobytes()

        molecular_hash = self._molecular_hash_function(data_bytes)

        checksum = self._quantum_dot_checksum(data_bytes)

        bit_anomalies = self._detect_bit_anomalies(data_bytes)

        micro_anomalies = self._detect_micro_anomalies(data)

        dimensional_micro = self._dimensional_downsampling_detection(data)

        return {
            "molecular_hash_entropy": float(molecular_hash),
            "quantum_checksum": float(checksum),
            "bit_anomaly_rate": float(bit_anomalies),
            "micro_anomaly_score": float(micro_anomalies),
            "dimensional_micro_score": float(dimensional_micro),
        }

    def _harmonic_anomaly_detection(self, data: np.ndarray[Any, Any]) -> float:
        """
        Harmonic anomaly detection using FFT
        """
        signal = data if data.ndim == 1 else data.flatten()

        if len(signal) < 8:
            return 0.0

        fft_result = fft(signal)
        power_spectrum = np.abs(fft_result) ** 2

        frequencies = np.fft.fftfreq(len(signal))

        fundamental_freq = frequencies[1] if len(frequencies) > 1 else 0.0

        harmonic_powers = []
        for n in range(1, min(8, len(signal) // 2)):
            harmonic_idx = int(n * fundamental_freq * len(signal))
            if 0 <= harmonic_idx < len(power_spectrum):
                harmonic_powers.append(power_spectrum[harmonic_idx])

        if not harmonic_powers:
            return 0.0

        total_power = np.sum(power_spectrum)
        harmonic_power = np.sum(harmonic_powers)

        harmonic_ratio = harmonic_power / (total_power + 1e-10)

        anomaly_score = 1.0 - min(harmonic_ratio * 2.0, 1.0)

        return float(anomaly_score)

    @staticmethod
    def _molecular_hash_function(data: bytes) -> float:
        """
        Molecular-level hash function for nano-scale integrity
        """
        hash_obj = hashlib.sha256(data)
        hash_bytes = hash_obj.digest()

        byte_values = np.frombuffer(hash_bytes, dtype=np.uint8)

        _, counts = np.unique(byte_values, return_counts=True)
        probabilities = counts / len(byte_values)

        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))

        normalized_entropy = entropy / 8.0

        return normalized_entropy

    @staticmethod
    def _quantum_dot_checksum(data: bytes) -> float:
        """
        Quantum dot-inspired checksum
        """
        current = data
        checksum = 0.0

        for _i in range(4):
            hash_obj = hashlib.sha256(current)
            current = hash_obj.digest()

            byte_sum = sum(current)
            checksum += byte_sum / (256.0 * len(current))

        return checksum / 4.0

    @staticmethod
    def _detect_bit_anomalies(data: bytes) -> float:
        """
        Detect bit-level anomalies
        """
        if len(data) < 2:
            return 0.0

        byte_values = np.frombuffer(data, dtype=np.uint8)

        transitions = np.abs(np.diff(byte_values.astype(int)))

        anomalous_transitions = np.sum(transitions > 200)

        anomaly_rate = anomalous_transitions / len(transitions)

        return anomaly_rate

    def _detect_micro_anomalies(self, data: np.ndarray[Any, Any]) -> float:
        """
        N Term Enhancement: Detect micro-anomalies at sub-feature level
        """
        if data.size < 4:
            return 0.0

        data_flat = data.flatten()

        local_variances = []
        window_size = min(4, len(data_flat) // 2)

        for i in range(len(data_flat) - window_size + 1):
            window = data_flat[i : i + window_size]
            variance = np.var(window)
            local_variances.append(variance)

        if not local_variances:
            return 0.0

        variance_array = np.array(local_variances)
        variance_changes = np.abs(np.diff(variance_array))

        micro_score = np.mean(variance_changes) / (np.std(variance_array) + 1e-10)

        return min(micro_score, 1.0)

    def _dimensional_downsampling_detection(self, data: np.ndarray[Any, Any]) -> float:
        """
        N Term Enhancement: Dimensional downsampling for micro-anomaly detection
        Downsample to low dimensions to detect subtle micro-patterns
        """
        data_2d = data.reshape(-1, 1) if data.ndim == 1 else data

        if data_2d.shape[1] < 2:
            return 0.0

        target_dim = max(1, min(3, data_2d.shape[1] // 2))

        try:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=target_dim)
            downsampled = pca.fit_transform(data_2d)

            reconstructed = pca.inverse_transform(downsampled)

            micro_residuals = np.abs(data_2d - reconstructed)

            residual_threshold = np.percentile(micro_residuals.flatten(), 95)

            micro_anomaly_pixels = np.sum(micro_residuals > residual_threshold)
            total_pixels = micro_residuals.size

            micro_anomaly_rate = micro_anomaly_pixels / total_pixels

            local_concentrations = []
            for i in range(min(5, data_2d.shape[0])):
                row_residuals = micro_residuals[i, :]
                concentration = np.max(row_residuals) / (np.mean(row_residuals) + 1e-10)
                local_concentrations.append(concentration)

            concentration_score = np.mean(local_concentrations) if local_concentrations else 0.0

            final_score = micro_anomaly_rate * 0.6 + min(concentration_score / 10.0, 1.0) * 0.4

            return min(final_score, 1.0)

        except Exception:
            return 0.0
