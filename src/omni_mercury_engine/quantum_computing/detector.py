# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Quantum Anomaly Detection for Mercury Agent.

High-level interface for quantum-enhanced anomaly detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from omni_mercury_engine.quantum_computing.circuits import (
    AnomalyEncodingCircuit,
    EncodingType,
    ErrorMitigationCircuit,
    VariationalAnsatz,
)
from omni_mercury_engine.quantum_computing.executor import (
    QuantumExecutor,
)
from omni_mercury_engine.quantum_computing.hybrid import (
    QAOAAnomalyDetector,
    QuantumKernel,
    VQEAnomalyDetector,
)

logger = logging.getLogger(__name__)


@dataclass
class QuantumDetectionResult:
    """Result of quantum anomaly detection."""

    anomaly_scores: np.ndarray[Any, Any]
    predictions: np.ndarray[Any, Any]
    threshold: float
    method: str
    quantum_metrics: dict[str, float] = field(default_factory=dict)
    classical_fallback_used: bool = False
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuantumResourceEstimate:
    """Estimate of quantum resources required."""

    num_qubits: int
    circuit_depth: int
    num_gates: int
    estimated_runtime_ms: float
    estimated_shots: int
    hardware_compatible: bool
    recommendations: list[str] = field(default_factory=list)


class QuantumAnomalyDetector:
    """Quantum-enhanced anomaly detection.

    Provides multiple quantum methods for anomaly detection with
    automatic fallback to classical methods when quantum hardware
    is unavailable or unsuitable.

    Example:
        detector = QuantumAnomalyDetector(
            backend="aer_simulator",
            shots=1024,
            error_mitigation="zne",
        )

        # Quantum-classical hybrid detection
        result = detector.detect(
            data=features,
            method="vqe_anomaly",
            classical_fallback=True,
        )
    """

    SUPPORTED_METHODS = [
        "quantum_kernel",
        "vqe_anomaly",
        "qaoa_anomaly",
        "amplitude_estimation",
    ]

    def __init__(
        self,
        backend: str = "aer_simulator",
        shots: int = 1024,
        error_mitigation: str | None = "zne",
        optimization_level: int = 3,
        api_token: str | None = None,
    ) -> None:
        """Initialize the quantum anomaly detector.

        Args:
            backend: Quantum backend name
            shots: Default number of measurement shots
            error_mitigation: Error mitigation method (None, "zne", "pec")
            optimization_level: Circuit optimization level
            api_token: IBM Quantum API token
        """
        self._executor = QuantumExecutor(
            backend=backend,
            shots=shots,
            optimization_level=optimization_level,
            api_token=api_token,
        )
        self._shots = shots
        self._error_mitigation = error_mitigation

        self._error_mitigator: ErrorMitigationCircuit | None = None
        if error_mitigation:
            self._error_mitigator = ErrorMitigationCircuit(error_mitigation)

        self._trained_model: Any = None
        self._method: str | None = None
        self._num_qubits: int = 4
        self._threshold: float = 0.5

    def fit(
        self,
        X_train: np.ndarray[Any, Any],
        y_train: np.ndarray[Any, Any] | None = None,
        method: str = "quantum_kernel",
        num_qubits: int | None = None,
        **kwargs: Any,
    ) -> QuantumAnomalyDetector:
        """Train the quantum anomaly detector.

        Args:
            X_train: Training data (for unsupervised: normal samples)
            y_train: Training labels (optional, for supervised methods)
            method: Detection method
            num_qubits: Number of qubits (default: min(4, n_features))
            **kwargs: Method-specific parameters

        Returns:
            self for method chaining
        """
        n_features = X_train.shape[1] if X_train.ndim > 1 else 1
        self._num_qubits = num_qubits or min(4, n_features)
        self._method = method

        X_normalized = self._normalize_data(X_train)

        if method == "quantum_kernel":
            self._trained_model = self._fit_quantum_kernel(X_normalized, y_train, **kwargs)
        elif method == "vqe_anomaly":
            self._trained_model = self._fit_vqe(X_normalized, **kwargs)
        elif method == "qaoa_anomaly":
            self._trained_model = self._fit_qaoa(X_normalized, **kwargs)
        else:
            self._trained_model = self._fit_vqe(X_normalized, **kwargs)

        return self

    def detect(
        self,
        data: np.ndarray[Any, Any],
        method: str | None = None,
        classical_fallback: bool = True,
        threshold: float | None = None,
    ) -> QuantumDetectionResult:
        """Detect anomalies in data.

        Args:
            data: Data to analyze
            method: Detection method (uses fitted method if None)
            classical_fallback: Use classical method if quantum fails
            threshold: Anomaly threshold (default: 0.5)

        Returns:
            QuantumDetectionResult with anomaly scores and predictions
        """
        method = method or self._method or "vqe_anomaly"
        threshold = threshold or self._threshold

        if self._trained_model is None and method in [
            "quantum_kernel",
            "vqe_anomaly",
            "qaoa_anomaly",
        ]:
            logger.warning("Model not fitted, using default scoring")
            return self._classical_detection(data, threshold)

        try:
            X_normalized = self._normalize_data(data)

            if method == "quantum_kernel":
                scores = self._detect_quantum_kernel(X_normalized)
            elif method == "vqe_anomaly":
                scores = self._detect_vqe(X_normalized)
            elif method == "qaoa_anomaly":
                scores = self._detect_qaoa(X_normalized)
            elif method == "amplitude_estimation":
                scores = self._detect_amplitude_estimation(X_normalized)
            else:
                scores = self._detect_vqe(X_normalized)

            predictions = (scores > threshold).astype(int)

            return QuantumDetectionResult(
                anomaly_scores=scores,
                predictions=predictions,
                threshold=threshold,
                method=method,
                quantum_metrics={
                    "num_qubits": self._num_qubits,
                    "shots": self._shots,
                },
                classical_fallback_used=False,
            )

        except Exception as e:
            logger.warning("Quantum detection failed: %s", e)
            if classical_fallback:
                return self._classical_detection(data, threshold)
            raise

    def estimate_resources(
        self,
        data_shape: tuple[int, ...],
        method: str = "vqe_anomaly",
    ) -> QuantumResourceEstimate:
        """Estimate quantum resources required for detection.

        Args:
            data_shape: Shape of input data
            method: Detection method

        Returns:
            QuantumResourceEstimate with resource requirements
        """
        n_samples = data_shape[0]
        n_features = data_shape[1] if len(data_shape) > 1 else 1

        num_qubits = min(8, n_features)
        recommendations = []

        if method == "quantum_kernel":
            circuit_depth = 2 * num_qubits * 2
            num_gates = num_qubits * 6 * 2
            estimated_shots = self._shots * n_samples * (n_samples + 1) // 2

        elif method == "vqe_anomaly":
            circuit_depth = num_qubits * 3 * 2
            num_gates = num_qubits * 4 * 2 + (num_qubits - 1) * 2
            estimated_shots = self._shots * n_samples

        elif method == "qaoa_anomaly":
            p = 2
            circuit_depth = num_qubits + 2 * p * (num_qubits + num_qubits)
            num_gates = num_qubits + p * (3 * num_qubits + num_qubits)
            estimated_shots = self._shots * n_samples

        else:
            circuit_depth = num_qubits * 4
            num_gates = num_qubits * 6
            estimated_shots = self._shots * n_samples

        gate_time_us = 0.2
        shot_time_us = 100
        estimated_runtime_ms = (num_gates * gate_time_us + estimated_shots * shot_time_us) / 1000

        hardware_compatible = num_qubits <= 127 and circuit_depth <= 100

        if num_qubits > 5:
            recommendations.append("Consider feature selection to reduce qubit count")
        if circuit_depth > 50:
            recommendations.append("High circuit depth may benefit from error mitigation")
        if estimated_runtime_ms > 60000:
            recommendations.append("Long runtime - consider batching or simulation")
        if not hardware_compatible:
            recommendations.append("Circuit exceeds typical hardware constraints")

        return QuantumResourceEstimate(
            num_qubits=num_qubits,
            circuit_depth=circuit_depth,
            num_gates=num_gates,
            estimated_runtime_ms=estimated_runtime_ms,
            estimated_shots=estimated_shots,
            hardware_compatible=hardware_compatible,
            recommendations=recommendations,
        )

    def _normalize_data(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Normalize data to [0, 1] range."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        X_range = X_max - X_min + 1e-10

        return (X - X_min) / X_range

    def _fit_quantum_kernel(
        self,
        X_train: np.ndarray[Any, Any],
        y_train: np.ndarray[Any, Any] | None,
        **kwargs: Any,
    ) -> Any:
        """Fit quantum kernel model."""
        kernel = QuantumKernel(
            self._num_qubits,
            feature_map_reps=kwargs.get("reps", 2),
            executor=self._executor,
        )

        if y_train is None:
            y_train = np.zeros(X_train.shape[0])

        predict_fn = kernel.fit_svm(X_train, y_train)
        return {"kernel": kernel, "predict": predict_fn, "X_train": X_train}

    def _fit_vqe(self, X_train: np.ndarray[Any, Any], **kwargs: Any) -> VQEAnomalyDetector:
        """Fit VQE anomaly detector."""
        detector = VQEAnomalyDetector(
            self._num_qubits,
            ansatz=kwargs.get("ansatz", VariationalAnsatz.REAL_AMPLITUDES),
            reps=kwargs.get("reps", 2),
            executor=self._executor,
        )
        detector.fit(X_train, maxiter=kwargs.get("maxiter", 50))
        return detector

    def _fit_qaoa(self, X_train: np.ndarray[Any, Any], **kwargs: Any) -> QAOAAnomalyDetector:
        """Fit QAOA anomaly detector."""
        adjacency = self._compute_similarity_matrix(X_train)

        detector = QAOAAnomalyDetector(
            self._num_qubits,
            p=kwargs.get("p", 2),
            executor=self._executor,
        )
        detector.fit(adjacency, maxiter=kwargs.get("maxiter", 50))
        return detector

    def _compute_similarity_matrix(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute similarity matrix for QAOA."""
        n = min(X.shape[0], self._num_qubits)
        X_subset = X[:n]

        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distances[i, j] = np.linalg.norm(X_subset[i] - X_subset[j])

        max_dist = distances.max() + 1e-10
        similarity = 1 - distances / max_dist

        return similarity

    def _detect_quantum_kernel(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Detect using quantum kernel."""
        model = self._trained_model
        kernel = model["kernel"]
        X_train = model["X_train"]

        K = kernel.compute_kernel_matrix(X, X_train, self._shots)

        scores = 1 - np.max(K, axis=1)
        return scores

    def _detect_vqe(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Detect using VQE."""
        return self._trained_model.score(X)

    def _detect_qaoa(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Detect using QAOA."""
        return self._trained_model.score(X)

    def _detect_amplitude_estimation(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Detect using amplitude estimation.

        Estimates the amplitude of anomalous states.
        """
        encoding = AnomalyEncodingCircuit(
            self._num_qubits,
            EncodingType.AMPLITUDE,
        )

        scores = []
        for sample in X:
            circuit = encoding.encode(sample[: 2**self._num_qubits])
            circuit.measure_all()

            result = self._executor.run(circuit)
            if isinstance(result, list):
                result = result[0]

            probs = result.probabilities
            uniform_prob = 1.0 / 2**self._num_qubits

            kl_divergence = 0.0
            for bitstring, prob in probs.items():
                if prob > 0:
                    kl_divergence += prob * np.log(prob / uniform_prob + 1e-10)

            scores.append(abs(kl_divergence))

        return np.array(scores)

    def _classical_detection(
        self,
        X: np.ndarray[Any, Any],
        threshold: float,
    ) -> QuantumDetectionResult:
        """Classical fallback detection using statistical methods."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0) + 1e-10

        z_scores = np.abs((X - mean) / std)
        scores = np.mean(z_scores, axis=1)

        scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        predictions = (scores_normalized > threshold).astype(int)

        return QuantumDetectionResult(
            anomaly_scores=scores_normalized,
            predictions=predictions,
            threshold=threshold,
            method="classical_zscore",
            classical_fallback_used=True,
            details={"reason": "Classical fallback used"},
        )
