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

"""Quantum Kernel Machines for Anomaly Detection.

Based on: Quantum anomaly detection in the latent space of proton collision events at the LHC
(Nature Communications Physics, 2024: https://www.nature.com/articles/s42005-024-01811-6)

Implements quantum-inspired kernel machines for unsupervised anomaly detection.
"""

from typing import Any, Callable, Dict, Optional

import numpy as np


class QuantumKernelMachine:
    """Quantum-inspired kernel machine for anomaly detection."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize quantum kernel machine.

        Args:
            config: Configuration including:
                - kernel_type: 'rbf', 'quantum_inspired', or 'polynomial'
                    (default: 'quantum_inspired')
                - num_qubits: Number of qubits in quantum circuit (default: 4)
                - entanglement_depth: Depth of entangling layers (default: 2)
                - gamma: RBF kernel parameter (default: 1.0)
        """
        self.config = config or {}
        self.kernel_type = self.config.get("kernel_type", "quantum_inspired")
        self.num_qubits = self.config.get("num_qubits", 4)
        self.entanglement_depth = self.config.get("entanglement_depth", 2)
        self.gamma = self.config.get("gamma", 1.0)
        self.training_data: Optional[np.ndarray] = None
        self.anomaly_threshold: Optional[float] = None

    def quantum_inspired_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute quantum-inspired kernel between two samples.

        Inspired by quantum circuits with entanglement, but implemented classically.
        Based on the data encoding circuit from the Nature paper.

        Args:
            x1: First sample vector
            x2: Second sample vector

        Returns:
            Kernel value (similarity score)
        """
        x1_norm = x1 / (np.linalg.norm(x1) + 1e-10)
        x2_norm = x2 / (np.linalg.norm(x2) + 1e-10)

        phi_x1 = self._quantum_feature_map(x1_norm, self.entanglement_depth)
        phi_x2 = self._quantum_feature_map(x2_norm, self.entanglement_depth)

        kernel_value = np.dot(phi_x1, phi_x2)

        return float(kernel_value)

    def _quantum_feature_map(self, x: np.ndarray, depth: int) -> np.ndarray:
        """Apply quantum-inspired feature map with entanglement.

        Simulates the effect of quantum data encoding circuits with multiple
        layers of rotation and entanglement gates.

        Args:
            x: Input vector
            depth: Number of entangling layers

        Returns:
            Transformed feature vector
        """
        feature_dim = 2**self.num_qubits
        if len(x) < feature_dim:
            x_padded = np.pad(x, (0, feature_dim - len(x)))
        else:
            x_padded = x[:feature_dim]

        phi = x_padded.copy()
        for layer in range(depth):
            phi = np.cos(phi * np.pi) + 1j * np.sin(phi * np.pi)
            phi = np.abs(phi)

            for i in range(0, len(phi) - 1, 2):
                phi[i], phi[i + 1] = phi[i] * phi[i + 1], phi[i] + phi[i + 1]
                phi[i], phi[i + 1] = phi[i] / (np.abs(phi[i]) + 1e-10), phi[i + 1] / (
                    np.abs(phi[i + 1]) + 1e-10
                )

        return phi

    def rbf_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute RBF (Gaussian) kernel."""
        return float(np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2))

    def compute_kernel_matrix(
        self, X: np.ndarray, kernel_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
    ) -> np.ndarray:
        """Compute kernel matrix for dataset.

        Args:
            X: Data matrix (n_samples, n_features)
            kernel_func: Optional custom kernel function

        Returns:
            Kernel matrix (n_samples, n_samples)
        """
        if kernel_func is None:
            if self.kernel_type == "quantum_inspired":
                kernel_func = self.quantum_inspired_kernel
            else:
                kernel_func = self.rbf_kernel

        n_samples = len(X)
        K = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(i, n_samples):
                K[i, j] = kernel_func(X[i], X[j])
                K[j, i] = K[i, j]

        return K

    def fit(self, training_data: np.ndarray) -> None:
        """Fit quantum kernel machine on training data.

        Args:
            training_data: In-distribution training samples
        """
        self.training_data = training_data

        K_train = self.compute_kernel_matrix(training_data)

        train_scores = np.mean(K_train, axis=1)

        self.anomaly_threshold = float(np.mean(train_scores) - 3 * np.std(train_scores))

    def predict(self, test_data: np.ndarray) -> Dict[str, Any]:
        """Predict anomalies using quantum kernel machine.

        Args:
            test_data: Test samples to evaluate

        Returns:
            Prediction results with anomaly scores and labels
        """
        if self.training_data is None:
            raise ValueError("Must fit model first using fit()")

        n_test = len(test_data)
        anomaly_scores = np.zeros(n_test)

        kernel_func = (
            self.quantum_inspired_kernel
            if self.kernel_type == "quantum_inspired"
            else self.rbf_kernel
        )

        for i in range(n_test):
            similarities = [
                kernel_func(test_data[i], train_sample) for train_sample in self.training_data
            ]
            anomaly_scores[i] = np.mean(similarities)

        threshold = self.anomaly_threshold if self.anomaly_threshold is not None else 0.0
        predictions = anomaly_scores < threshold

        return {
            "anomaly_scores": anomaly_scores,
            "predictions": predictions,
            "threshold": self.anomaly_threshold,
            "kernel_type": self.kernel_type,
        }
